"""LLM access for the Trip Planner Agent: OpenRouter's Auto Router, or OpenAI.

With OPENROUTER_API_KEY set (and USE_OPENROUTER not "0"), every LLM call goes to
OpenRouter's Auto Router (model "openrouter/auto"), which picks a model for each
request: a cheap one for parsing a sentence, a stronger one for writing a plan.
Without it, the app calls gpt-4.1-mini on OpenAI directly, as in Step 6.

call_llm() runs one call and returns (result, log entry). The entry records
which model answered, the router's tier and effort for it, the candidates it
considered, tokens, cost and time; the app shows these in its Models panel.
If a routed call fails (an API error, or a structured output that doesn't
parse), it is retried once on gpt-4.1-mini, when an OpenAI key is set.

Router settings follow the course demo (autoroute_demo.py): the plugin id must
match the router slug, X-OpenRouter-Metadata asks for the router's reasoning,
and usage.include asks for the cost of each request.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any, Optional

import openai
from langchain_openai import ChatOpenAI

OPENAI_MODEL = "gpt-4.1-mini"
ROUTER_MODEL = "openrouter/auto"
ROUTER_PLUGIN_ID = "auto-router"   # must match ROUTER_MODEL, or settings are ignored
OPENROUTER_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = 0.4                  # OpenAI only; routed models use their defaults


def _env_list(name: str) -> list[str]:
    return [s.strip() for s in os.getenv(name, "").split(",") if s.strip()]


OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENAI_KEY = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN") or "").strip()
USE_ROUTER = bool(OPENROUTER_KEY) and os.getenv("USE_OPENROUTER", "1").strip() != "0"
COST_TIER = os.getenv("OPENROUTER_COST_TIER", "").strip() or None   # low | medium | high
ALLOWED_MODELS = _env_list("OPENROUTER_ALLOWED_MODELS")    # e.g. anthropic/*,openai/*
EXCLUDED_MODELS = _env_list("OPENROUTER_EXCLUDED_MODELS")


def describe_setup() -> str:
    """One line for the sidebar."""
    if not USE_ROUTER:
        return f"OpenAI `{OPENAI_MODEL}` (fixed model)"
    parts = [f"cost tier: {COST_TIER or 'router default'}"]
    if ALLOWED_MODELS:
        parts.append("allowed: " + ", ".join(ALLOWED_MODELS))
    if EXCLUDED_MODELS:
        parts.append("excluded: " + ", ".join(EXCLUDED_MODELS))
    return f"OpenRouter Auto Router (`{ROUTER_MODEL}`; {'; '.join(parts)})"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class RoutedChatOpenAI(ChatOpenAI):
    """ChatOpenAI that keeps OpenRouter's extra response fields.

    LangChain copies the served model name and token usage into
    response_metadata, but not OpenRouter's `openrouter_metadata` (the router's
    candidates and reasoning) or the request cost. This adds both.
    """

    def _create_chat_result(self, response, *args, **kwargs):
        result = super()._create_chat_result(response, *args, **kwargs)
        raw = response if isinstance(response, dict) else response.model_dump()
        router_meta = raw.get("openrouter_metadata")
        cost = find_key(raw.get("usage") or {}, "cost")
        for generation in result.generations:
            metadata = generation.message.response_metadata
            if router_meta is not None:
                metadata["openrouter_metadata"] = router_meta
            if cost is not None:
                metadata["cost"] = cost
        return result


def router_llm(session_id: str) -> ChatOpenAI:
    plugin: dict[str, Any] = {"id": ROUTER_PLUGIN_ID}
    if COST_TIER:
        plugin["cost_tier"] = COST_TIER
    if ALLOWED_MODELS:
        plugin["allowed_models"] = ALLOWED_MODELS
    if EXCLUDED_MODELS:
        plugin["excluded_models"] = EXCLUDED_MODELS
    return RoutedChatOpenAI(
        model=ROUTER_MODEL,
        base_url=OPENROUTER_URL,
        api_key=OPENROUTER_KEY,
        timeout=180,
        max_retries=1,
        default_headers={
            "X-OpenRouter-Metadata": "enabled",  # the router's candidates and reason
            "X-Title": "Trip Planner Agent (MPCS 57200)",
        },
        extra_body={
            "plugins": [plugin],
            "usage": {"include": True},  # report the cost of each request
            "session_id": session_id,
        },
    )


def openai_llm() -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_MODEL, temperature=TEMPERATURE,
                      api_key=OPENAI_KEY or None)


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------
def call_llm(purpose: str, prompt, *, schema=None, tools=None,
             session_id: Optional[str] = None) -> tuple[Any, dict]:
    """Run one LLM call. Returns (result, log entry).

    schema -> the parsed Pydantic object; tools -> the AIMessage (with any tool
    calls); otherwise -> the AIMessage. session_id keeps related calls on one
    model (see session_for in trip_planner.py); None means a fresh session.
    """
    if not USE_ROUTER:
        return _run(openai_llm(), purpose, prompt, schema, tools, routed=False)
    try:
        llm = router_llm(session_id or str(uuid.uuid4()))
        return _run(llm, purpose, prompt, schema, tools, routed=True)
    except (openai.APIError, ValueError) as error:
        if not OPENAI_KEY:
            raise
        result, entry = _run(openai_llm(), purpose, prompt, schema, tools, routed=False)
        entry["note"] = f"router failed ({short_error(error)}); used {OPENAI_MODEL}"
        return result, entry


def _run(llm: ChatOpenAI, purpose: str, prompt, schema, tools, routed: bool):
    started = time.perf_counter()
    if schema is not None:
        # Function calling, not json_schema: OpenRouter documents tool calling
        # for every routed model, but not structured outputs.
        out = llm.with_structured_output(
            schema, method="function_calling", include_raw=True
        ).invoke(prompt)
        if out.get("parsing_error") is not None or out.get("parsed") is None:
            raise ValueError(f"{schema.__name__} didn't parse: {out.get('parsing_error')}")
        result, raw = out["parsed"], out["raw"]
    elif tools:
        raw = llm.bind_tools(tools).invoke(prompt)
        result = raw
    else:
        raw = llm.invoke(prompt)
        result = raw
    return result, describe_call(purpose, raw, time.perf_counter() - started, routed)


def describe_call(purpose: str, raw, seconds: float, routed: bool) -> dict:
    """The log entry for one call, from the response's metadata."""
    metadata = getattr(raw, "response_metadata", None) or {}
    usage = getattr(raw, "usage_metadata", None) or {}
    served = metadata.get("model_name") or (ROUTER_MODEL if routed else OPENAI_MODEL)
    router_meta = metadata.get("openrouter_metadata") or {}
    candidates = find_key(router_meta, "candidates") or []
    # The router's own judgement of this request: the served model's candidate
    # entry if present, else its first choice (as in autoroute_demo.py).
    mine = next((c for c in candidates if short(c.get("model")) == short(served)),
                candidates[0] if candidates else {})
    reason = find_key(router_meta, "reason")
    # Reasoning tokens are hidden "thinking" that a reasoning model writes before
    # its answer; they are billed as output. Cached tokens are input the
    # provider had already seen (cheaper on most models).
    output_details = usage.get("output_token_details") or {}
    input_details = usage.get("input_token_details") or {}
    return {
        "purpose": purpose,
        "model": served,
        "routed": routed,
        "tier": mine.get("tier"),
        "effort": mine.get("effort"),
        "candidates": [short(c.get("model")) for c in candidates][:6],
        "reason": str(reason)[:300] if reason else "",
        "tokens": usage.get("total_tokens"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": output_details.get("reasoning") or 0,
        "cached_tokens": input_details.get("cache_read") or 0,
        "cost": metadata.get("cost"),
        "seconds": round(seconds, 1),
        "note": "",
    }


# ---------------------------------------------------------------------------
# Helpers (find_key and short are from the course demo, autoroute_demo.py)
# ---------------------------------------------------------------------------
def find_key(obj, key):
    """Recursively find the first value for `key` in nested dicts/lists."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = list(obj.values())
    if isinstance(obj, list):
        for item in obj:
            found = find_key(item, key)
            if found is not None:
                return found
    return None


def short(slug) -> str:
    """Drop the revision suffix (-YYYYMMDD, -YYYY-MM-DD or -MMDD) for display and matching."""
    return re.sub(r"-(\d{8}|\d{4}-\d{2}-\d{2}|\d{4})$", "", slug or "")


def short_error(error: Exception) -> str:
    text = str(error).splitlines()[0] if str(error) else ""
    return f"{type(error).__name__}: {text}"[:160]
