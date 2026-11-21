"""Rubrics for the LLM judge: what to check in one agent reply, and when it applies.

Each rubric asks one yes/no question about one reply. Binary verdicts with a
one-sentence reason are easier to calibrate against human labels than 1-10 scores
(two people rarely agree on "7 vs 8", they usually agree on pass/fail).

*Whether* a rubric applies is decided in code from the trace (a knowledge base
search happened, every field is filled), not by the judge: the judge only answers
the question, so a "not applicable" never hides a missed check.

Changing a rubric's text changes its prompt hash, so stored verdicts for the old
text are not reused; bump `version` too, so reports can tell the versions apart.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

JudgeItem = dict[str, Any]


@dataclass(frozen=True)
class Rubric:
    name: str
    version: str
    title: str
    question: str
    applies: Callable[[JudgeItem], bool]
    inputs: Callable[[JudgeItem], dict[str, Any]]


def _searched(item: JudgeItem) -> bool:
    return any(call["name"] == "search_knowledge_base" for call in item["tool_calls"])


def _awaiting_confirmation(item: JudgeItem) -> bool:
    """Every field is filled, nothing is submitted, and a value was saved in this turn.

    v1 asked for a summary in every turn after the form was complete. Human labels
    showed that was wrong: when the user says "let me think about it" after seeing
    the summary, nothing changed and repeating the summary is not needed. A form can
    only become complete (or change) through a successful update_fields, so that is
    the turn where a fresh summary is due.
    """
    total = item.get("fields_total") or 0
    complete = total > 0 and item.get("fields_filled") == total and not item.get("submitted")
    saved_now = any(c["name"] == "update_fields" and c["ok"] for c in item["tool_calls"])
    return complete and saved_now


GROUNDED_ANSWER = Rubric(
    name="grounded_answer",
    version="1",
    title="Answer grounded in the FAQ",
    question=(
        "The assistant answered a question about the clinic using passages retrieved "
        "from the clinic's FAQ. PASS if every statement about the clinic (prices, "
        "opening hours, policies, services, location, payment) is supported by the "
        "passages, or if the passages do not contain the answer and the reply says so "
        "instead of guessing. FAIL if the reply states any clinic fact that the "
        "passages do not support, or contradicts them. Ignore what the reply says "
        "about the user's own booking details."
    ),
    applies=_searched,
    inputs=lambda item: {
        "user_message": item["user_text"],
        "retrieved_passages": item["passages"],
        "assistant_reply": item["reply"],
    },
)

CLAIMS_MATCH_ACTIONS = Rubric(
    name="claims_match_actions",
    version="1",
    title="Reply matches what was actually done",
    question=(
        "The assistant fills in a form by calling tools. You get the tool calls of "
        "this turn with their results, and the form values saved after the turn. "
        "PASS if everything the reply says about the form is true: values it says are "
        "saved or updated are among the saved values with that content, and it only "
        "says the form was submitted if submit_form succeeded. FAIL if the reply "
        "claims a value was saved or changed that is not saved, states a saved value "
        "differently from what is saved, or claims a submission that did not happen. "
        "Asking questions or describing next steps is not a claim."
    ),
    applies=lambda item: bool(item["reply"].strip()),
    inputs=lambda item: {
        "user_message": item["user_text"],
        "tool_calls_this_turn": item["tool_calls"],
        "saved_values_after_turn": item["saved_values"],
        "form_submitted": item["submitted"],
        "assistant_reply": item["reply"],
    },
)

CONFIRMATION_SUMMARY = Rubric(
    name="confirmation_summary",
    version="2",
    title="Correct summary before submitting",
    question=(
        "Every form field is now filled and the form is not submitted yet, so the "
        "assistant must show the user a summary and ask for confirmation. PASS if the "
        "reply lists every saved value with the same content (formatting such as a "
        "weekday added to a date is fine) and asks the user to confirm. FAIL if a value "
        "is missing or different, or if it does not ask for confirmation."
    ),
    applies=_awaiting_confirmation,
    inputs=lambda item: {
        "saved_values": item["saved_values"],
        "assistant_reply": item["reply"],
    },
)

RUBRICS: dict[str, Rubric] = {
    rubric.name: rubric for rubric in (GROUNDED_ANSWER, CLAIMS_MATCH_ACTIONS, CONFIRMATION_SUMMARY)
}

SYSTEM_PROMPT = """You are a strict evaluator of a form-filling assistant for a dental clinic.
The conversation may be in Turkish or English. You answer exactly one yes/no question
about one assistant reply, using only the data you are given.

Respond with a json object and nothing else, in this format:
{"verdict": "pass", "reason": "One sentence that points to the evidence."}

"verdict" is "pass" or "fail". Write the reason in English."""


def user_prompt(rubric: Rubric, item: JudgeItem) -> str:
    data = json.dumps(rubric.inputs(item), ensure_ascii=False, indent=2)
    return f"Question: {rubric.question}\n\nData (json):\n{data}"
