"""Trip Planner Agent - Step 5
LangGraph + Streamlit

Graph:
    START -> parse_request --(missing info)--> ask_for_details  [interrupt]
                           --(complete)------> research_agent
    ask_for_details -> parse_request   (with the user's answer added)
    research_agent --(tool calls)--> tools -> research_agent   (loop)
                   --(notes ready)--> draft_itinerary
    draft_itinerary -> geocode_stops -> validate_itinerary
    validate_itinerary --(issues found, < MAX_REVISIONS)--> draft_itinerary (revise)
                       --(passed or out of revisions)----> review  [interrupt]
    review --(change request)--> draft_itinerary (revise with the user's feedback)
           --(approved / new trip)--> END

interrupt() pauses the graph and waits for the user; the state is kept by a
checkpointer (InMemorySaver) under a thread_id, and the UI resumes the graph
with Command(resume=<user's reply>).

Run with:
    uv run streamlit run trip_planner.py
"""

import json
import math
import operator
import os
import time
import uuid
from typing import Annotated, Literal, Optional, TypedDict

import pydeck as pdk
import requests
import streamlit as st
import wikipedia
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

load_dotenv()

MAX_DAYS = 14             # keep the drafted itinerary a manageable size
MAX_RESEARCH_ROUNDS = 4   # max rounds of tool calls before the agent must stop
MAX_STOP_DISTANCE_KM = 150  # a geocoded stop farther than this from its city is a wrong match

# Plan-check rules (validate_itinerary)
MAX_REVISIONS = 2            # how many times the planner may revise after a failed check
MAX_STOPS_PER_DAY = 4
MAX_DAY_ROUTE_KM = 100       # total distance between one day's stops, in visiting order
TRANSFER_KM = 150            # moving this far between cities makes it a travel day...
MAX_STOPS_ON_TRANSFER_DAY = 2  # ...which should be light

# Wikipedia and OpenStreetMap both ask for a descriptive User-Agent with a way
# to contact you. Put CONTACT_EMAIL=you@example.com in your .env file.
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "").strip()
USER_AGENT = "TripPlannerAgent/0.5 (MPCS 57200 coursework" + (
    f"; {CONTACT_EMAIL})" if CONTACT_EMAIL else ")"
)
wikipedia.set_user_agent(USER_AGENT)


# ---------------------------------------------------------------------------
# Structured-output schemas (what we ask the LLM to return)
# ---------------------------------------------------------------------------
class TripRequest(BaseModel):
    destinations: list[str] = Field(
        description="Cities, regions or countries the user wants to visit, in visiting order"
    )
    days: Optional[int] = Field(
        description="Total trip length in days, or null if the user did not say"
    )
    season: Optional[str] = Field(
        description="When the trip happens (month or season), or null if not stated"
    )
    budget: Optional[str] = Field(
        description="Budget level: 'low', 'medium' or 'high', or null if not stated"
    )
    interests: list[str] = Field(
        description="Activities or themes the user cares about (empty list if none)"
    )


class Stop(BaseModel):
    name: str = Field(description="Specific place name, e.g. 'Fushimi Inari Taisha'")
    note: str = Field(description="One short sentence on what to do there")


class DayPlan(BaseModel):
    day: int = Field(description="Day number, starting at 1")
    city: str = Field(
        description="City or base for that day, with its country, e.g. 'Antigua, Guatemala'"
    )
    stops: list[Stop] = Field(description="2-4 stops for the day, in visiting order")


class Itinerary(BaseModel):
    day_plans: list[DayPlan] = Field(description="One entry per day of the trip")
    tips: list[str] = Field(description="3-5 practical tips for this trip")


class FeedbackDecision(BaseModel):
    action: Literal["approve", "change", "new_trip"] = Field(
        description="'approve' if the traveler is happy and wants no more changes; "
        "'change' if they want to adjust this itinerary, including making the trip "
        "longer or shorter; 'new_trip' if they want different destinations "
        "(a different place, or adding or dropping a destination)"
    )
    new_days: Optional[int] = Field(
        description="If the traveler asked for a different trip length, the new total "
        "number of days; otherwise null"
    )
    instructions: str = Field(
        description="For 'change': the requested changes, restated as clear instructions "
        "for the planner. For 'new_trip': the complete new trip request. "
        "Empty for 'approve'."
    )


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class TripState(TypedDict, total=False):
    request: str          # the user's request text
    trip: dict            # parsed TripRequest
    missing: list[str]    # details we still need from the user
    # Research agent's conversation (LLM messages + tool results).
    # add_messages appends new messages instead of overwriting the list.
    messages: Annotated[list[AnyMessage], add_messages]
    research_rounds: int  # how many rounds of tool calls so far
    research_notes: str   # the agent's final notes
    # Pages the agent read; operator.add concatenates lists from each round.
    sources: Annotated[list[dict], operator.add]
    itinerary: dict       # drafted Itinerary (geocode_stops adds lat/lon)
    unlocated: list[str]  # stops we could not find on the map
    geocode_errors: list[str]  # errors from the map service, shown to the user
    issues: list[str]     # problems found by validate_itinerary (empty = passed)
    revision_count: int   # how many times draft_itinerary has revised the plan
    check_log: list[dict]  # plan checks in the current round: {"round": n, "issues": [...]}
    user_feedback: list[str]  # change requests from the user, oldest first
    feedback_round: int   # how many times the user has asked for changes
    approved: bool        # the user accepted the plan
    new_request: str      # the user asked for a completely different trip
    message: str          # final message for the user


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# show_spinner=False: LangGraph may run nodes in worker threads, where
# Streamlit can't draw a spinner (it would raise NoSessionContext).
@st.cache_resource(show_spinner=False)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.4,
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN"),
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool(response_format="content_and_artifact")
def wikipedia_search(query: str) -> tuple[str, list[dict]]:
    """Search English Wikipedia and return summaries of the top matching pages.

    Use it for destinations, attractions, local culture and typical weather.
    Queries work best as short topic names, e.g. 'Antigua Guatemala' or
    'Climate of Honduras'.
    """
    summaries, sources = [], []
    try:
        titles = wikipedia.search(query, results=2)
    except Exception as e:  # network error, rate limit, etc.
        return f"Wikipedia search failed: {e}", []
    for title in titles:
        try:
            page = wikipedia.page(title, auto_suggest=False)
        except Exception:  # PageError, DisambiguationError, ...
            continue
        summaries.append(f"Page: {page.title}\nSummary: {page.summary[:1500]}")
        sources.append({"title": page.title, "url": page.url})
    content = "\n\n".join(summaries) or f"No Wikipedia pages found for '{query}'."
    # content goes back to the LLM; the artifact (sources) stays in our app
    return content, sources


TOOLS = [wikipedia_search]
tool_node = ToolNode(TOOLS)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
PARSE_PROMPT = """You extract trip details from a travel request.
Only fill in what the user actually said. Use null for anything not stated.
Keep destination names in English."""


def parse_request(state: TripState) -> TripState:
    """Turn the free-text request into structured trip details."""
    llm = get_llm().with_structured_output(TripRequest)
    trip = llm.invoke([("system", PARSE_PROMPT), ("human", state["request"])])

    missing = []
    if not trip.destinations:
        missing.append("where you want to go")
    if not trip.days:
        missing.append("how many days the trip is")
    return {"trip": trip.model_dump(), "missing": missing}


def route_after_parse(state: TripState) -> str:
    """Conditional edge: ask for more details, or go ahead and research."""
    return "ask_for_details" if state["missing"] else "research_agent"


def ask_for_details(state: TripState) -> TripState:
    """Pause and ask the user for the missing details, then add their answer."""
    needed = " and ".join(state["missing"])
    # interrupt() stops the graph here. The question goes to the UI, and when
    # the UI resumes with Command(resume=answer), interrupt() returns the answer.
    # (On resume the node runs again from the top, so nothing before this
    # line should have side effects.)
    answer = interrupt(f"Happy to plan this! Could you also tell me {needed}?")
    return {"request": f"{state['request']}\n{answer}"}


RESEARCH_PROMPT = """You are a travel researcher preparing notes for an itinerary planner.

Trip:
- Destinations: {destinations}
- Length: {days} days
- When: {season}
- Interests: {interests}

Use the wikipedia_search tool to find, for each destination:
- the main cities/areas worth visiting and specific sights that match the interests,
- what the weather is typically like at that time of year,
- anything practical a traveler should know (getting around, currency).
Search every destination at least once. You may call the tool several times
in parallel. You have at most {max_rounds} rounds of searching.

When you have enough, reply WITHOUT calling a tool: write concise research
notes as bullet points grouped by destination. Only include facts supported
by the search results; label anything else as general knowledge."""


def research_agent(state: TripState) -> TripState:
    """The LLM decides what to look up next, or writes its final notes."""
    new_messages = []
    if not state.get("messages"):  # first visit: give the agent its task
        trip = state["trip"]
        new_messages = [
            SystemMessage(RESEARCH_PROMPT.format(
                destinations=", ".join(trip["destinations"]),
                days=trip["days"],
                season=trip["season"] or "not specified",
                interests=", ".join(trip["interests"]) or "general sightseeing",
                max_rounds=MAX_RESEARCH_ROUNDS,
            )),
            HumanMessage("Start researching."),
        ]
    history = state.get("messages", []) + new_messages
    rounds = state.get("research_rounds", 0)

    if rounds >= MAX_RESEARCH_ROUNDS:
        # Out of budget: call the model WITHOUT tools so it must write notes.
        stop = HumanMessage("Stop searching now and write your research notes.")
        new_messages.append(stop)
        response = get_llm().invoke(history + [stop])
    else:
        response = get_llm().bind_tools(TOOLS).invoke(history)
    new_messages.append(response)

    if response.tool_calls:
        return {"messages": new_messages, "research_rounds": rounds + 1}
    return {"messages": new_messages, "research_notes": response.content}


def route_after_research(state: TripState) -> str:
    """Conditional edge: run the requested tools, or move on to planning."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "draft_itinerary"


def run_tools(state: TripState) -> TripState:
    """Execute the tool calls and also collect the pages we read as sources."""
    result = tool_node.invoke(state)
    sources = [s for m in result["messages"] for s in (m.artifact or [])]
    return {"messages": result["messages"], "sources": sources}


DRAFT_PROMPT = """You are an experienced travel planner.
Write a realistic day-by-day itinerary for the trip below.

Rules:
- Exactly one entry per day ({days} days).
- 2-4 stops per day, geographically close to each other.
- If the trip covers several cities or countries, include travel days and
  keep the order of destinations sensible.
- Prefer real places that appear in the research notes.
- Match the user's interests and budget.
- Tips should be practical (getting around, money, seasonal weather, etiquette).
  Do not state visa or entry requirements as facts; tell the traveler to check
  official government sources instead.

Trip details:
- Destinations: {destinations}
- Length: {days} days
- When: {season}
- Budget: {budget}
- Interests: {interests}
- Original request: {request}

Research notes:
{notes}"""


REVISION_PROMPT = """

You already drafted the itinerary below. Revise it as described, and keep
everything else that works.

Previous itinerary:
{previous}
"""

FEEDBACK_SECTION = """
Changes the traveler asked for (oldest first; these take priority, and if two
of them conflict, the most recent one wins):
{feedback}
"""

ISSUES_SECTION = """
Problems found by the automatic plan check (fix every one):
{issues}
"""


def itinerary_for_prompt(itinerary: dict) -> str:
    """The itinerary as compact JSON, without the coordinates."""
    slim = [
        {"day": d["day"], "city": d["city"],
         "stops": [{"name": s["name"], "note": s["note"]} for s in d["stops"]]}
        for d in itinerary["day_plans"]
    ]
    return json.dumps(slim, ensure_ascii=False, indent=1)


def draft_itinerary(state: TripState) -> TripState:
    """Draft a day-by-day itinerary, or revise it if the plan check found issues."""
    trip = state["trip"]
    days = min(trip["days"], MAX_DAYS)
    prompt = DRAFT_PROMPT.format(
        destinations=", ".join(trip["destinations"]),
        days=days,
        season=trip["season"] or "not specified",
        budget=trip["budget"] or "not specified",
        interests=", ".join(trip["interests"]) or "general sightseeing",
        request=state["request"],
        notes=state.get("research_notes") or "(no research notes)",
    )
    issues = state.get("issues") or []
    feedback = state.get("user_feedback") or []
    if state.get("itinerary") and (issues or feedback):
        prompt += REVISION_PROMPT.format(previous=itinerary_for_prompt(state["itinerary"]))
        if feedback:
            prompt += FEEDBACK_SECTION.format(feedback="\n".join(f"- {f}" for f in feedback))
        if issues:
            prompt += ISSUES_SECTION.format(issues="\n".join(f"- {i}" for i in issues))
    llm = get_llm().with_structured_output(Itinerary)
    itinerary = llm.invoke(prompt)

    update: TripState = {"itinerary": itinerary.model_dump()}
    if issues:  # only plan-check revisions count toward MAX_REVISIONS
        update["revision_count"] = state.get("revision_count", 0) + 1
    return update


# ---------------------------------------------------------------------------
# Geocoding (plain Python, no LLM) - OpenStreetMap Nominatim
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth, in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def nominatim_lookup(query: str, country_code: str = "") -> Optional[dict]:
    """Look up a place name. Returns {lat, lon, country_code} or None.

    Cached so the same place is never requested twice. Network errors are
    raised (and therefore not cached) so a later run can retry.
    """
    params = {"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1}
    if country_code:
        params["countrycodes"] = country_code
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    time.sleep(1)  # Nominatim's usage policy: at most 1 request per second
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None
    hit = results[0]
    return {
        "lat": float(hit["lat"]),
        "lon": float(hit["lon"]),
        "country_code": hit.get("address", {}).get("country_code", ""),
    }


def safe_lookup(query: str, country_code: str, errors: list[str]) -> Optional[dict]:
    """nominatim_lookup that records failures in `errors` instead of raising."""
    try:
        return nominatim_lookup(query, country_code)
    except Exception as e:  # network error, HTTP 403/429, bad JSON...
        errors.append(f"{type(e).__name__}: {e}")
        return None


def geocode_stops(state: TripState) -> TripState:
    """Add lat/lon to every day (its city) and every stop of the itinerary."""
    write = get_stream_writer()  # sends custom progress events to the UI
    itinerary = state["itinerary"]
    days = itinerary["day_plans"]
    total = len(days) + sum(len(d["stops"]) for d in days)
    done = 0
    unlocated: list[str] = []
    errors: list[str] = []
    found_any = False

    for day in days:
        # If the first few lookups all fail, the service is unreachable or
        # blocking us; skip the rest instead of waiting on every request.
        if len(errors) >= 3 and not found_any:
            break
        city = safe_lookup(day["city"], "", errors)
        found_any = found_any or city is not None
        day["lat"], day["lon"] = (city["lat"], city["lon"]) if city else (None, None)
        done += 1
        write({"geocode": (done, total)})

        for stop in day["stops"]:
            # 1st try: "Stop, City"   2nd try: "Stop" restricted to the city's country
            hit = safe_lookup(f"{stop['name']}, {day['city']}", "", errors)
            if hit is None and city and city["country_code"]:
                hit = safe_lookup(stop["name"], city["country_code"], errors)
            found_any = found_any or hit is not None
            # Sanity check: reject matches that are implausibly far from the city
            if hit and city and haversine_km(
                hit["lat"], hit["lon"], city["lat"], city["lon"]
            ) > MAX_STOP_DISTANCE_KM:
                hit = None
            stop["lat"], stop["lon"] = (hit["lat"], hit["lon"]) if hit else (None, None)
            if hit is None:
                unlocated.append(f"Day {day['day']}: {stop['name']}")
            done += 1
            write({"geocode": (done, total)})

    # Days we skipped after repeated errors have no coordinates yet
    for day in days:
        day.setdefault("lat", None)
        day.setdefault("lon", None)
        for stop in day["stops"]:
            if "lat" not in stop:
                stop["lat"] = stop["lon"] = None
                unlocated.append(f"Day {day['day']}: {stop['name']}")

    # keep unique error messages, in order
    return {"itinerary": itinerary, "unlocated": unlocated,
            "geocode_errors": list(dict.fromkeys(errors))}


# ---------------------------------------------------------------------------
# Plan check (plain Python, no LLM)
# ---------------------------------------------------------------------------
def check_itinerary(trip: dict, itinerary: dict) -> list[str]:
    """Return a list of problems with the itinerary (empty list = looks fine)."""
    issues = []
    days = itinerary["day_plans"]

    # 1. Right number of days
    expected = min(trip["days"], MAX_DAYS)
    if len(days) != expected:
        issues.append(f"The plan has {len(days)} days but the trip is {expected} days.")

    # 2. Every destination appears somewhere (city or stop name)
    text = " ".join(
        [d["city"] for d in days] + [s["name"] for d in days for s in d["stops"]]
    ).lower()
    for dest in trip["destinations"]:
        if dest.lower() not in text:
            issues.append(f"'{dest}' is one of the requested destinations but no day visits it.")

    seen: dict[str, int] = {}
    prev = None
    for d in days:
        n = d["day"]
        stops = d["stops"]

        # 3. Not too many / too few stops
        if len(stops) == 0:
            issues.append(f"Day {n} has no stops.")
        if len(stops) > MAX_STOPS_PER_DAY:
            issues.append(f"Day {n} has {len(stops)} stops; keep it to {MAX_STOPS_PER_DAY} or fewer.")

        # 4. Same place twice
        for s in stops:
            key = s["name"].strip().lower()
            if key in seen and seen[key] != n:
                issues.append(f"'{s['name']}' appears on both Day {seen[key]} and Day {n}.")
            seen.setdefault(key, n)

        # 5. Too much ground to cover within one day (uses geocoded coordinates)
        located = [s for s in stops if s.get("lat") is not None]
        route_km = sum(
            haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
            for a, b in zip(located, located[1:])
        )
        if route_km > MAX_DAY_ROUTE_KM:
            issues.append(
                f"Day {n}'s stops are spread over about {route_km:.0f} km "
                f"(limit {MAX_DAY_ROUTE_KM} km); group nearby sights on the same day."
            )

        # 6. A long move between cities should be a light travel day
        if prev and prev.get("lat") is not None and d.get("lat") is not None:
            move_km = haversine_km(prev["lat"], prev["lon"], d["lat"], d["lon"])
            if move_km > TRANSFER_KM and len(stops) > MAX_STOPS_ON_TRANSFER_DAY:
                issues.append(
                    f"Day {n} moves about {move_km:.0f} km from {prev['city']} to "
                    f"{d['city']} but still has {len(stops)} stops; make it a travel "
                    f"day with at most {MAX_STOPS_ON_TRANSFER_DAY} stops."
                )
        prev = d

    return issues


def validate_itinerary(state: TripState) -> TripState:
    """Run the rule checks and record the result."""
    issues = check_itinerary(state["trip"], state["itinerary"])
    entry = {"round": state.get("revision_count", 0), "issues": issues}
    return {"issues": issues, "check_log": (state.get("check_log") or []) + [entry]}


def route_after_validate(state: TripState) -> str:
    """Conditional edge: revise the plan, or finish."""
    if state["issues"] and state.get("revision_count", 0) < MAX_REVISIONS:
        return "draft_itinerary"
    return "review"


# ---------------------------------------------------------------------------
# Review with the user (human in the loop)
# ---------------------------------------------------------------------------
REVIEW_PROMPT = """A traveler is reviewing a trip itinerary you made.
Current trip: {destinations}, {days} days.
Changes they asked for earlier: {earlier}

Their reply: "{answer}"

Decide what they want."""


def review_message(state: TripState) -> str:
    """What we tell the user when we show them the (updated) plan."""
    trip = state["trip"]
    if state.get("feedback_round"):
        message = "I've updated your itinerary."
    else:
        message = "Here's your itinerary, based on what I found on Wikipedia."
    revisions = state.get("revision_count", 0)
    if revisions:
        times = "once" if revisions == 1 else f"{revisions} times"
        message += f" The automatic plan check made me revise it {times}."
    if state.get("issues"):
        message += " A few problems are still left; see the plan check below."
    if trip["days"] > MAX_DAYS:
        message += f" (I planned the first {MAX_DAYS} days to keep it manageable.)"
    message += (
        "\n\nWhat would you like to change? For example *\"make day 2 more relaxed\"* "
        "or *\"add more food spots\"*. If it looks good, just tell me."
    )
    return message


def review(state: TripState) -> TripState:
    """Show the plan, wait for the user's reply, and decide what to do next."""
    answer = interrupt(review_message(state))

    trip = state["trip"]
    decision = get_llm().with_structured_output(FeedbackDecision).invoke(
        REVIEW_PROMPT.format(
            destinations=", ".join(trip["destinations"]),
            days=trip["days"],
            earlier="; ".join(state.get("user_feedback") or []) or "none",
            answer=answer,
        )
    )
    if decision.action == "approve":
        return {
            "approved": True,
            "message": "Great, your itinerary is final. Have a wonderful trip! ✈️  "
                       "Send a new request any time to plan another one.",
        }
    if decision.action == "new_trip":
        return {"new_request": decision.instructions or answer}
    # A change request: start a fresh round of drafting and plan checks
    update: TripState = {}
    if decision.new_days and decision.new_days != trip["days"]:
        # Update the trip itself, so the planner and the plan check both use the
        # new length (otherwise the check would flag the shorter plan as wrong).
        update["trip"] = {**trip, "days": decision.new_days}
    return update | {
        "user_feedback": (state.get("user_feedback") or []) + [decision.instructions or answer],
        "feedback_round": state.get("feedback_round", 0) + 1,
        "issues": [],
        "revision_count": 0,
        "check_log": [],
    }


def route_after_review(state: TripState) -> str:
    if state.get("approved") or state.get("new_request"):
        return END
    return "draft_itinerary"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
@st.cache_resource
def build_graph():
    graph = StateGraph(TripState)
    graph.add_node("parse_request", parse_request)
    graph.add_node("ask_for_details", ask_for_details)
    graph.add_node("research_agent", research_agent)
    graph.add_node("tools", run_tools)
    graph.add_node("draft_itinerary", draft_itinerary)
    graph.add_node("geocode_stops", geocode_stops)
    graph.add_node("validate_itinerary", validate_itinerary)
    graph.add_node("review", review)

    graph.add_edge(START, "parse_request")
    graph.add_conditional_edges(
        "parse_request", route_after_parse, ["ask_for_details", "research_agent"]
    )
    graph.add_edge("ask_for_details", "parse_request")  # re-parse with the answer
    graph.add_conditional_edges(
        "research_agent", route_after_research, ["tools", "draft_itinerary"]
    )
    graph.add_edge("tools", "research_agent")  # loop back after each tool round
    graph.add_edge("draft_itinerary", "geocode_stops")
    graph.add_edge("geocode_stops", "validate_itinerary")
    graph.add_conditional_edges(
        "validate_itinerary", route_after_validate, ["draft_itinerary", "review"]
    )
    graph.add_conditional_edges("review", route_after_review, ["draft_itinerary", END])
    # The checkpointer saves the state after every step, per thread_id.
    # That is what lets interrupt() pause the graph and resume it later.
    return graph.compile(checkpointer=InMemorySaver())


def describe_step(node: str, output: dict) -> list[str]:
    """Human-readable progress lines for one node update."""
    if node == "parse_request":
        return ["✅ Understood your request"]
    if node == "ask_for_details":
        return ["📝 Added your answer to the request"]
    if node == "research_agent":
        last = output["messages"][-1]
        if last.tool_calls:
            return [f"🔎 Searching Wikipedia: *{c['args'].get('query', '')}*"
                    for c in last.tool_calls]
        return ["📝 Wrote research notes"]
    if node == "tools":
        return [f"📖 Read {len(output.get('sources', []))} Wikipedia page(s)"]
    if node == "draft_itinerary":
        if output.get("revision_count"):
            return [f"🔁 Revised the itinerary (revision {output['revision_count']})"]
        return ["🗺️ Drafted the itinerary"]
    if node == "validate_itinerary":
        issues = output.get("issues", [])
        if not issues:
            return ["🧪 Plan check passed"]
        return [f"🧪 Plan check found {len(issues)} issue(s)"] + [f"  - {i}" for i in issues]
    if node == "review":
        if output.get("approved"):
            return ["👍 Plan approved"]
        if output.get("new_request"):
            return ["🆕 Starting a new trip"]
        lines = [f"✏️ Applying your changes: *{output['user_feedback'][-1]}*"]
        if "trip" in output:
            lines.insert(0, f"📅 Changed the trip length to {output['trip']['days']} days")
        return lines
    if node == "__interrupt__":
        return []
    if node == "geocode_stops":
        stops = sum(len(d["stops"]) for d in output["itinerary"]["day_plans"])
        found = stops - len(output.get("unlocated", []))
        return [f"📍 Located {found} of {stops} places on the map"]
    return [f"✅ {node}"]


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
# One color per day (cycles after 7); the emoji is used as the legend.
DAY_COLORS = [
    ("🔴", [220, 50, 50]), ("🟠", [245, 140, 30]), ("🟡", [230, 190, 30]),
    ("🟢", [60, 170, 80]), ("🔵", [40, 120, 220]), ("🟣", [150, 70, 200]),
    ("🟤", [140, 90, 50]),
]


def day_color(day: int):
    return DAY_COLORS[(day - 1) % len(DAY_COLORS)]


def render_simple_map(points: list[dict]) -> None:
    """Fallback: plain st.map (dots only, no routes or tooltips)."""
    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]
    span_km = haversine_km(min(lats), min(lons), max(lats), max(lons))
    st.map(
        {
            "lat": lats,
            "lon": lons,
            "color": ["#%02x%02x%02x" % tuple(p["color"]) for p in points],
        },
        latitude="lat",
        longitude="lon",
        color="color",
        size=max(150, span_km * 15),  # dot radius in meters, scaled to the area shown
    )


def render_map(day_plans: list[dict], simple: bool = False) -> None:
    """Draw the stops as colored dots, connected in visiting order per day."""
    points, paths = [], []
    for d in day_plans:
        _, rgb = day_color(d["day"])
        located = [s for s in d["stops"] if s.get("lat") is not None]
        for s in located:
            points.append({"name": s["name"], "day": d["day"], "city": d["city"],
                           "lat": s["lat"], "lon": s["lon"], "color": rgb})
        if len(located) >= 2:
            paths.append({"path": [[s["lon"], s["lat"]] for s in located], "color": rgb})

    if not points:
        st.info("None of these places could be located on the map.")
        return
    if simple:
        render_simple_map(points)
        return

    view = pdk.data_utils.compute_view([[p["lon"], p["lat"]] for p in points])
    view.zoom = min(view.zoom, 12)  # a single point would otherwise zoom in fully
    layers = [
        pdk.Layer("PathLayer", data=paths, get_path="path", get_color="color",
                  width_min_pixels=3, opacity=0.6),
        pdk.Layer("ScatterplotLayer", data=points, get_position="[lon, lat]",
                  get_fill_color="color", get_line_color=[255, 255, 255],
                  stroked=True, line_width_min_pixels=1, get_radius=300,
                  radius_min_pixels=6, radius_max_pixels=12, pickable=True),
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view,
                             tooltip={"text": "Day {day} · {name}\n{city}"}))


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Trip Planner Agent", page_icon="✈️")
st.title("✈️ Trip Planner Agent")
st.caption("Tell me where you want to go, for how long, and what you enjoy.")

graph = build_graph()

# Each conversation (one trip) is a LangGraph thread. The checkpointer keeps
# the graph's state per thread_id, so the UI only needs to remember the id.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history: {"role", "content"}


def new_thread() -> None:
    st.session_state.thread_id = str(uuid.uuid4())


def thread_config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def run_graph(graph_input) -> str:
    """Stream the graph, showing progress, and return the reply for the chat."""
    reply = ""
    with st.status("Working on it...", expanded=True) as status:
        progress = None
        # "updates" -> {node_name: output} after each node
        # "custom"  -> whatever a node sends with get_stream_writer()
        for mode, chunk in graph.stream(
            graph_input, thread_config(), stream_mode=["updates", "custom"]
        ):
            if mode == "custom":
                if "geocode" in chunk:
                    done, total = chunk["geocode"]
                    if progress is None:
                        progress = st.progress(0.0)
                    progress.progress(
                        done / total,
                        text=f"📍 Locating places on the map... {done}/{total}",
                    )
                continue
            if progress is not None:  # geocoding finished
                progress.empty()
                progress = None
            for node, output in chunk.items():
                if node == "__interrupt__":
                    # The graph paused at interrupt(); its value is our question.
                    reply = output[0].value
                    continue
                for line in describe_step(node, output or {}):
                    st.write(line)
        status.update(label="Done", state="complete", expanded=False)
    if not reply:  # the graph finished instead of pausing
        reply = graph.get_state(thread_config()).values.get("message", "")
    return reply


with st.sidebar:
    st.header("About")
    st.write(
        "A LangGraph agent that reads your travel request, asks for anything "
        "missing, researches the destinations on Wikipedia, drafts a "
        "day-by-day itinerary, puts it on a map, checks the plan with simple "
        "rules, and then revises it with you until you're happy."
    )
    if not CONTACT_EMAIL:
        st.warning(
            "Add `CONTACT_EMAIL=you@example.com` to your `.env` file. Wikipedia and "
            "OpenStreetMap may block requests that don't include a contact address."
        )
    with st.expander("Graph structure (Mermaid)"):
        st.code(graph.get_graph().draw_mermaid(), language="text")
    if st.button("Start over"):
        new_thread()
        st.session_state.messages = []
        st.rerun()

# Replay the chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

placeholder = "e.g. 7 days in Tokyo and Kyoto in December, food and temples, mid budget"
if user_input := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    snapshot = graph.get_state(thread_config())
    if snapshot.interrupts:
        # The graph is paused at a question: resume it with the user's reply.
        graph_input = Command(resume=user_input)
    else:
        if snapshot.values:  # the previous trip is finished: start a new one
            new_thread()
        graph_input = {"request": user_input}

    with st.chat_message("assistant"):
        try:
            reply = run_graph(graph_input)
            new_request = graph.get_state(thread_config()).values.get("new_request")
            if new_request:  # the user asked for a different trip
                new_thread()
                reply = run_graph({"request": new_request})
        except Exception as e:  # e.g. missing API key or network error
            reply = f"Sorry, something went wrong: `{e}`"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.map_day = "All days"  # reset the map filter

# Show the current itinerary, read straight from the graph's saved state
snapshot = graph.get_state(thread_config())
values = snapshot.values
if values.get("itinerary") and not values.get("new_request"):
    trip = values["trip"]
    itinerary = values["itinerary"]

    st.divider()
    st.subheader("🗺️ Your itinerary")
    if snapshot.interrupts:
        st.info("💬 Reply in the chat to change something, or say it looks good.")
    elif values.get("approved"):
        st.success("✅ Final plan")

    summary = [
        f"**Destinations:** {', '.join(trip['destinations'])}",
        f"**Length:** {trip['days']} days",
    ]
    if trip["season"]:
        summary.append(f"**When:** {trip['season']}")
    if trip["budget"]:
        summary.append(f"**Budget:** {trip['budget']}")
    if trip["interests"]:
        summary.append(f"**Interests:** {', '.join(trip['interests'])}")
    if values.get("user_feedback"):
        summary.append(f"**Your changes:** {'; '.join(values['user_feedback'])}")
    st.markdown("  \n".join(summary))

    day_plans = itinerary["day_plans"]

    # Map (changing the filter only reruns the UI; the graph is not called again)
    options = ["All days"] + [f"Day {d['day']}" for d in day_plans]
    col_day, col_style = st.columns([3, 2], vertical_alignment="bottom")
    with col_day:
        choice = st.selectbox("Show on map", options, key="map_day")
    with col_style:
        simple_map = st.toggle("Simple map", key="simple_map",
                               help="Use this if the detailed map doesn't appear.")
    shown = day_plans if choice == "All days" else [
        d for d in day_plans if f"Day {d['day']}" == choice
    ]
    if values.get("geocode_errors"):
        st.warning(
            "The map service returned errors, so some places are missing:\n\n"
            + "\n".join(f"- `{e}`" for e in values["geocode_errors"][:3])
        )
    render_map(shown, simple=simple_map)
    st.caption("   ".join(f"{day_color(d['day'])[0]} Day {d['day']}" for d in shown))

    tabs = st.tabs([f"{day_color(d['day'])[0]} Day {d['day']}" for d in day_plans])
    for tab, day in zip(tabs, day_plans):
        with tab:
            st.markdown(f"#### {day['city']}")
            for i, stop in enumerate(day["stops"], start=1):
                not_found = " *(not on map)*" if stop.get("lat") is None else ""
                st.markdown(f"{i}. **{stop['name']}** — {stop['note']}{not_found}")

    with st.expander("💡 Travel tips"):
        for tip in itinerary["tips"]:
            st.markdown(f"- {tip}")

    if values.get("check_log"):
        final_ok = not values.get("issues")
        label = "🧪 Plan check: passed" if final_ok else "🧪 Plan check: issues remain"
        with st.expander(label, expanded=not final_ok):
            for entry in values["check_log"]:
                if entry["round"] == 0:
                    title = "Updated draft" if values.get("feedback_round") else "First draft"
                else:
                    title = f"Revision {entry['round']}"
                if entry["issues"]:
                    st.markdown(f"**{title}** — {len(entry['issues'])} issue(s)")
                    for i in entry["issues"]:
                        st.markdown(f"- {i}")
                else:
                    st.markdown(f"**{title}** — ✅ passed")
            st.caption(
                f"Rules: ≤{MAX_STOPS_PER_DAY} stops/day, ≤{MAX_DAY_ROUTE_KM} km between a "
                f"day's stops, ≤{MAX_STOPS_ON_TRANSFER_DAY} stops on days that move "
                f">{TRANSFER_KM} km, no repeated places, every destination visited."
            )

    if values.get("unlocated"):
        with st.expander(f"❔ Not found on the map ({len(values['unlocated'])})"):
            for item in values["unlocated"]:
                st.markdown(f"- {item}")

    if values.get("research_notes"):
        with st.expander("🔎 Research notes"):
            st.markdown(values["research_notes"])

    sources = list({s["url"]: s for s in values.get("sources", [])}.values())  # de-dupe
    if sources:
        with st.expander(f"📚 Sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"- [{s['title']}]({s['url']})")