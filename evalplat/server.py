"""HTTP ingestion: an agent posts its trace events here while it runs.

    evalplat serve                       # http://127.0.0.1:8700
    POST /v1/events   [event, event, ...] -> {"added": n, "duplicates": n, "invalid": n}
    GET  /health

Events go through the same validation and storage as `evalplat ingest`, with
source 'live', so the dashboard shows them as they arrive. Posting the same event
twice is harmless (events are keyed by event_id), so clients can safely retry.

Auth: if EVALPLAT_INGEST_TOKEN is set, requests need `Authorization: Bearer <token>`.
The server binds to 127.0.0.1 by default; it is meant for local use and is not
deployed with the public demo.
"""

from __future__ import annotations

import hmac
import os
from typing import Annotated, Any

from fastapi import Body, FastAPI, Header, HTTPException, status

from evalplat import __version__, store
from evalplat.ingest import LIVE_SOURCE, FileReport, ingest_events

TOKEN_ENV = "EVALPLAT_INGEST_TOKEN"
MAX_EVENTS_PER_REQUEST = 1000
MAX_BODY_BYTES = 5 * 1024 * 1024


def create_app(db_path: str | None = None, token: str | None = None) -> FastAPI:
    """Build the app. Arguments override the EVALPLAT_DB / EVALPLAT_INGEST_TOKEN settings."""
    db = str(store.resolve_db_path(db_path))
    token = token if token is not None else os.environ.get(TOKEN_ENV) or None
    store.connect(db).close()  # create / migrate once at startup

    app = FastAPI(title="evalplat ingest", version=__version__)

    def check_token(authorization: str | None) -> None:
        if token is None:
            return
        expected = f"Bearer {token}"
        # Constant-time comparison: timing must not reveal how much of the token matched.
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or wrong bearer token")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    # A plain (non-async) handler: FastAPI runs it in a worker thread, so the
    # blocking SQLite write does not stall the event loop. Each request opens its
    # own connection because SQLite connections cannot be shared across threads.
    @app.post("/v1/events", max_body_size=MAX_BODY_BYTES)
    def post_events(
        events: Annotated[list[Any], Body()],
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        check_token(authorization)
        if len(events) > MAX_EVENTS_PER_REQUEST:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"at most {MAX_EVENTS_PER_REQUEST} events per request",
            )
        conn = store.connect(db)
        try:
            report = ingest_events(
                conn, events, source=LIVE_SOURCE, report=FileReport("http", "traces")
            )
        finally:
            conn.close()
        return {
            "added": report.added,
            "duplicates": report.duplicates,
            "invalid": report.invalid,
            "problems": report.notes,
        }

    return app
