"""Client for the local SemIf decision server (semif_server.py).

A decision model answers a multiple-choice question about some state with a
probability for every option, instead of generating text. The app uses it for
small control decisions (where to route, whether a check passes) and keeps the
LLM for writing.

If the server is not running, or a request fails, decide() returns None and the
caller falls back to its LLM path, so the app works with or without SemIf.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

SEMIF_URL = os.getenv("SEMIF_URL", "http://127.0.0.1:8765").rstrip("/")
# CPU scoring of a long state can take several seconds on a laptop.
SEMIF_TIMEOUT = float(os.getenv("SEMIF_TIMEOUT", "60"))
CONNECT_TIMEOUT = 3  # fail fast when the server isn't running

# The last error, so the UI can explain why it fell back to the LLM.
last_error: str = ""


@dataclass
class Decision:
    """SemIf's answer to one question."""

    id: str
    choice: str          # the option with the highest probability
    confidence: float    # that option's probability
    probabilities: dict[str, float]
    seconds: float       # time for the whole request (all questions)

    def p(self, option: str) -> float:
        return self.probabilities.get(option, 0.0)


# The two questions the review step asks about the traveler's reply. They live
# here so eval_router.py can test exactly what the app asks.
ACTION_QUESTION = "What does the traveler want to do next?"
ACTION_OPTIONS = {
    "approve": "The traveler is happy with the itinerary and wants no more changes.",
    "change": "The traveler wants to adjust this itinerary: different stops, pace, "
              "food, or a longer or shorter trip to the same places.",
    "new_trip": "The traveler wants a different trip: other destinations, adding or "
                "dropping a destination, or a whole new trip request (places, length, "
                "interests) instead of feedback on this plan.",
    "question": "The traveler asks a question (about the places, weather, costs, "
                "visas, ...) and neither approves the plan nor asks for a change.",
}
LENGTH_QUESTION = "Does the traveler's reply ask for a different number of days?"
LENGTH_OPTIONS = {
    "yes": "The reply asks to make the trip longer or shorter.",
    "no": "The reply does not ask to change the number of days.",
}


def review_questions() -> list[dict]:
    return [question("action", ACTION_QUESTION, ACTION_OPTIONS),
            question("length", LENGTH_QUESTION, LENGTH_OPTIONS)]


def review_state(destinations: list[str], days: int, earlier: list[str], reply: str) -> dict:
    """What SemIf reads when it classifies the traveler's reply."""
    return {
        "current_trip": f"{', '.join(destinations)}, {days} days",
        "earlier_change_requests": earlier,
        "traveler_reply": reply,
    }


def question(qid: str, text: str, options: dict[str, str]) -> dict:
    """Build one question. `options` maps option id -> description (2-16 options)."""
    return {
        "id": qid,
        "question": text,
        "options": [{"id": oid, "description": desc} for oid, desc in options.items()],
    }


def decide(state, questions: list[dict]) -> dict[str, Decision] | None:
    """Ask SemIf one or more questions about the same state.

    Returns {question id: Decision}, or None if SemIf is unavailable.
    """
    global last_error
    try:
        resp = requests.post(
            f"{SEMIF_URL}/decide",
            json={"state": state, "questions": questions},
            timeout=(CONNECT_TIMEOUT, SEMIF_TIMEOUT),
        )
        data = resp.json()
        if resp.status_code != 200:
            raise RuntimeError(data.get("error") or f"HTTP {resp.status_code}")
        decisions = {
            r["id"]: Decision(
                id=r["id"],
                choice=r["choice"],
                confidence=float(r["confidence"]),
                probabilities={k: float(v) for k, v in r["probabilities"].items()},
                seconds=float(data.get("seconds", 0.0)),
            )
            for r in data["results"]
        }
    except (requests.RequestException, ValueError, KeyError, RuntimeError) as e:
        last_error = describe_error(e)
        return None
    last_error = ""
    return decisions


def health() -> dict | None:
    """Server info if SemIf is running, else None."""
    global last_error
    try:
        resp = requests.get(f"{SEMIF_URL}/health", timeout=(CONNECT_TIMEOUT, 5))
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        last_error = describe_error(e)
        return None


def describe_error(e: Exception) -> str:
    """A short, readable reason for the UI."""
    if isinstance(e, requests.ConnectionError):
        return f"can't connect to {SEMIF_URL}"
    if isinstance(e, requests.Timeout):
        return f"no answer within {SEMIF_TIMEOUT:.0f} s"
    return f"{type(e).__name__}: {e}"
