"""Measure how well SemIf classifies traveler replies, and pick a threshold.

The review step trusts SemIf only when its top probability is at least
ROUTE_CONFIDENCE (trip_planner.py); below that, the LLM decides. SemIf's
probabilities are not calibrated out of the box, so this script checks, on
labeled replies, how often SemIf is right at each confidence level.

Three groups of replies:
  easy      - the intent is clear
  hard      - mixed or vague replies ("Looks good, but..."), and change or
              new_trip requests phrased as questions ("Could we add Osaka?")
  question  - questions about the trip; the plan should stay as it is.
              Before the 'question' option existed, SemIf had to call these
              approve / change / new_trip and often said new_trip (0.75-0.83).

Start semif_server.py first, then run:
    uv run python eval_router.py
"""

from __future__ import annotations

import statistics

import decisions as semif

DESTINATIONS, DAYS = ["Tokyo", "Kyoto"], 7

# (traveler reply, expected action, expected "changes the length?", group)
# The length label only matters for 'change' replies (None = not scored).
EXAMPLES = [
    ("Looks good!", "approve", None, "easy"),
    ("Perfect, thanks", "approve", None, "easy"),
    ("That's great, let's go with it", "approve", None, "easy"),
    ("No changes, I'm happy with this", "approve", None, "easy"),
    ("ok this works for me", "approve", None, "easy"),
    ("Love it, book it", "approve", None, "easy"),
    ("Make day 2 more relaxed", "change", "no", "easy"),
    ("Add more food spots", "change", "no", "easy"),
    ("Can we swap the museum for a market?", "change", "no", "easy"),
    ("Too many temples, add some shopping", "change", "no", "easy"),
    ("Start day 3 later in the morning", "change", "no", "easy"),
    ("I don't like day 4, change it", "change", "no", "easy"),
    ("Make it a 6 day trip", "change", "yes", "easy"),
    ("Can we add two more days?", "change", "yes", "easy"),
    ("Shorten it to 5 days", "change", "yes", "easy"),
    ("Add one more day in Kyoto", "change", "yes", "easy"),
    ("Actually, plan 5 days in Seoul instead", "new_trip", None, "easy"),
    ("Let's add Osaka too", "new_trip", None, "easy"),
    ("Drop Kyoto, just Tokyo please", "new_trip", None, "easy"),
    ("Forget Japan, I want to go to Peru", "new_trip", None, "easy"),
    # Mixed or vague replies
    ("Looks good, but can day 3 be a bit shorter?", "change", "no", "hard"),
    ("I think it's fine, though day 1 feels rushed", "change", "no", "hard"),
    ("Great! Could you also add a day?", "change", "yes", "hard"),
    ("Cut the last two days", "change", "yes", "hard"),
    ("Can we do Kyoto first and then Tokyo?", "change", "no", "hard"),
    ("Maybe Osaka instead of Kyoto?", "new_trip", None, "hard"),
    ("5 days in Kyoto in December, temples, tea and hiking", "new_trip", None, "hard"),
    ("Just Tokyo for 4 days, focused on anime and food", "new_trip", None, "hard"),
    ("ok", "approve", None, "hard"),
    ("sure, whatever", "approve", None, "hard"),
    ("Can you make day 2 more relaxed?", "change", "no", "hard"),
    ("Could we add Osaka?", "new_trip", None, "hard"),
    # Questions: answer them, keep the plan
    ("What's the weather like in December?", "question", None, "question"),
    ("How much will this trip cost?", "question", None, "question"),
    ("Do I need a visa for Japan?", "question", None, "question"),
    ("Is Kyoto crowded in December?", "question", None, "question"),
    ("Which day is best for shopping?", "question", None, "question"),
]

THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
GROUPS = ["easy", "hard", "question"]


def main() -> None:
    if semif.health() is None:
        raise SystemExit(f"SemIf server not reachable at {semif.SEMIF_URL} "
                         f"({semif.last_error}). Start semif_server.py first.")

    rows = []
    print(f"{'reply':45} {'expected':9} {'SemIf':9} {'p':>5}    {'length':12} {'time':>6}")
    for reply, gold, gold_len, group in EXAMPLES:
        answers = semif.decide(semif.review_state(DESTINATIONS, DAYS, [], reply),
                               semif.review_questions())
        if answers is None:
            raise SystemExit(f"Request failed: {semif.last_error}")
        action, length = answers["action"], answers["length"]
        row = {
            "reply": reply, "group": group, "gold": gold, "pred": action.choice,
            "p": action.confidence, "gold_len": gold_len, "pred_len": length.choice,
            "seconds": action.seconds,  # server time for both questions
        }
        rows.append(row)
        mark = "✓" if action.choice == gold else "✗"
        length_note = ""
        if gold_len:
            ok = "✓" if length.choice == gold_len else "✗"
            length_note = f"{length.choice} ({length.confidence:.2f}) {ok}"
        print(f"{reply[:45]:45} {gold:9} {action.choice:9} "
              f"{action.confidence:5.2f} {mark}  {length_note:12} {row['seconds']:5.1f}s")

    print("\nAction accuracy")
    for group in GROUPS:
        part = [r for r in rows if r["group"] == group]
        ok = sum(r["gold"] == r["pred"] for r in part)
        print(f"  {group:8} {ok}/{len(part)}")
    ok = sum(r["gold"] == r["pred"] for r in rows)
    print(f"  {'all':8} {ok}/{len(rows)} = {ok / len(rows):.0%}")
    scored = [r for r in rows if r["gold_len"]]
    length_ok = sum(r["gold_len"] == r["pred_len"] for r in scored)
    print(f"Length accuracy (change replies): {length_ok}/{len(scored)}")

    wrong_rows = [r for r in rows if r["gold"] != r["pred"]]
    if wrong_rows:
        print("\nMistakes (a mistake at or above the threshold is acted on):")
        for r in sorted(wrong_rows, key=lambda r: -r["p"]):
            print(f"  {r['reply'][:45]:45} expected {r['gold']}, got {r['pred']} ({r['p']:.2f})")

    print("\nIf the router trusts SemIf at confidence >= threshold:")
    print(f"{'threshold':>9}  {'SemIf decides':>13}  {'accuracy there':>14}  "
          f"{'wrong but trusted':>17}")
    for t in THRESHOLDS:
        kept = [r for r in rows if r["p"] >= t]
        wrong = sum(r["gold"] != r["pred"] for r in kept)
        acc = f"{(len(kept) - wrong) / len(kept):.0%}" if kept else "-"
        print(f"{t:>9.2f}  {len(kept):>6}/{len(rows):<6}  {acc:>14}  {wrong:>17}")

    times = [r["seconds"] for r in rows]
    print(f"\nTime per reply (2 questions, server side): median "
          f"{statistics.median(times):.1f} s, mean {statistics.mean(times):.1f} s, "
          f"max {max(times):.1f} s")
    print("\nPick the threshold that keeps 'wrong but trusted' acceptable while SemIf "
          "still decides most replies, and set ROUTE_CONFIDENCE in trip_planner.py.")


if __name__ == "__main__":
    main()
