import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from evalplat import store, traces  # noqa: E402
from evalplat.server import MAX_EVENTS_PER_REQUEST, create_app  # noqa: E402
from tests.factories import make_event  # noqa: E402


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "live.db")


def client(db, token=None):
    return TestClient(create_app(db, token=token))


def test_health(db):
    assert client(db).get("/health").json()["status"] == "ok"


def test_events_are_stored_as_live_traces(db):
    events = [
        make_event("e1", model="deepseek-flash", ok=True, latency_ms=900.0),
        make_event("e2", "tool_call", name="update_fields", ok=True, duration_ms=1.0),
    ]
    response = client(db).post("/v1/events", json=events)

    assert response.status_code == 200
    assert response.json() == {"added": 2, "duplicates": 0, "invalid": 0, "problems": []}
    [source] = traces.list_trace_sources(store.connect(db))
    assert (source["source"], source["events"]) == ("live", 2)


def test_retrying_a_batch_does_not_duplicate_events(db):
    app = client(db)
    app.post("/v1/events", json=[make_event("e1")])
    assert app.post("/v1/events", json=[make_event("e1")]).json()["duplicates"] == 1


def test_invalid_events_are_reported_and_valid_ones_kept(db):
    response = client(db).post("/v1/events", json=[make_event("e1"), {"type": "llm_call"}])
    body = response.json()
    assert (body["added"], body["invalid"]) == (1, 1)
    assert body["problems"] == ["unsupported schema_version None"]


def test_body_must_be_a_list_of_events(db):
    assert client(db).post("/v1/events", json={"event_id": "e1"}).status_code == 422


def test_too_many_events_in_one_request(db):
    events = [make_event(f"e{i}") for i in range(MAX_EVENTS_PER_REQUEST + 1)]
    assert client(db).post("/v1/events", json=events).status_code == 413


def test_token_is_required_when_configured(db):
    app = client(db, token="s3cret")
    assert app.post("/v1/events", json=[make_event()]).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert app.post("/v1/events", json=[make_event()], headers=wrong).status_code == 401
    right = {"Authorization": "Bearer s3cret"}
    assert app.post("/v1/events", json=[make_event()], headers=right).status_code == 200
