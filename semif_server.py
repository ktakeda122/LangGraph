"""Local decision server for the Trip Planner Agent (MPCS 57200 HW 7-2).

SemIf is a small open "decision model": it is given some state, a question and
a list of typed options, and it returns a probability for each option. It never
generates text. This script keeps one SemIf model loaded (Qwen3.5-4B GGUF on the
CPU, through SemIf's llama.cpp backend) and serves it over a tiny HTTP API, so
the Streamlit app can ask it questions without loading a 3 GB model itself.

Run it inside the SemIf-OpenJev virtual environment (on Windows: inside WSL2):

    cd ~/SemIf-OpenJev && . .venv/bin/activate
    python /path/to/Trip/semif_server.py --gguf models/Qwen_Qwen3.5-4B-Q4_K_M.gguf

API (JSON over HTTP, default http://127.0.0.1:8765)

    GET  /health  -> {"ok": true, "model": "...", "gguf": "...", "threads": 8}

    POST /decide
        {"state": "<text>" or {...},
         "questions": [
            {"id": "action",
             "question": "What does the traveler want to do next?",
             "options": [{"id": "approve", "description": "..."},
                         {"id": "change",  "description": "..."}]}]}
     -> {"results": [{"id": "action", "choice": "approve", "confidence": 0.93,
                      "probabilities": {"approve": 0.93, "change": 0.07},
                      "input_tokens": 412}],
         "mode": "direct", "seconds": 1.42}

Several questions about the SAME state are scored in SemIf's "shared" mode:
the state is read once and every question branches off it, which is much
faster than asking them one by one.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The model and tokenizer revision used in class (Lecture 7).
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_GGUF = "models/Qwen_Qwen3.5-4B-Q4_K_M.gguf"

MAX_BODY_BYTES = 1_000_000
MAX_QUESTIONS = 32


class DecisionEngine:
    """One loaded SemIf model. llama.cpp keeps a single scoring context, so
    requests are scored one at a time (the lock); HTTP handling stays threaded."""

    def __init__(self, model: str, revision: str, gguf: str, threads: int | None,
                 max_tokens: int):
        try:
            from semif_phase1 import llamacpp_backend
            from semif_phase1.core import validate_row
        except ImportError as error:
            raise SystemExit(
                "Can't import semif_phase1. Activate the SemIf-OpenJev virtual "
                "environment first (pip install -e '.[test,llamacpp]')."
            ) from error
        self.backend = llamacpp_backend
        self.validate_row = validate_row
        self.max_tokens = max_tokens
        self.lock = threading.Lock()
        print(f"Loading {gguf} (tokenizer {model}@{revision[:8]})...", flush=True)
        started = time.perf_counter()
        self.model, self.tokenizer, self.metadata = llamacpp_backend.load_model(
            model, revision, gguf, threads=threads, context_tokens=max_tokens
        )
        print(f"Model ready in {time.perf_counter() - started:.1f} s", flush=True)

    def info(self) -> dict:
        gguf = self.metadata.get("gguf", {})
        return {
            "ok": True,
            "model": self.metadata.get("source"),
            "revision": self.metadata.get("revision"),
            "gguf": gguf.get("file"),
            "threads": self.metadata.get("threads"),
            "max_prompt_tokens": self.max_tokens,
        }

    def decide(self, state, questions: list[dict]) -> dict:
        rows = [
            {"id": q["id"], "state": state, "question": q["question"], "options": q["options"]}
            for q in questions
        ]
        for row in rows:
            self.validate_row(row)  # raises ValueError with a readable message
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Question ids must be unique")

        started = time.perf_counter()
        with self.lock:
            mode = "direct"
            if len(rows) > 1:
                try:
                    scored, _ = self.backend.score_shared(
                        self.model, self.tokenizer, rows, self.metadata, self.max_tokens
                    )
                    mode = "shared"
                except ValueError:
                    # Shared mode needs the state to tokenize as a clean prefix;
                    # if it doesn't, score the questions one by one instead.
                    scored = None
            if mode == "direct":
                scored = [
                    self.backend.score(self.model, self.tokenizer, row, self.metadata,
                                       self.max_tokens)
                    for row in rows
                ]

        results = []
        for item in scored:
            probabilities = dict(zip(item["option_ids"], item["probabilities"]))
            choice = max(probabilities, key=probabilities.get)
            results.append({
                "id": item["id"],
                "choice": choice,
                "confidence": probabilities[choice],
                "probabilities": probabilities,
                "input_tokens": item.get("input_tokens"),
            })
        return {"results": results, "mode": mode,
                "seconds": round(time.perf_counter() - started, 3)}


def parse_request(body: dict) -> tuple[object, list[dict]]:
    """Check the shape of a /decide request before it reaches the model."""
    if not isinstance(body, dict):
        raise ValueError("Body must be a JSON object")
    state = body.get("state")
    if not state or not isinstance(state, (str, dict, list)):
        raise ValueError("'state' must be a nonempty string, object or array")
    questions = body.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("'questions' must be a nonempty list")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"At most {MAX_QUESTIONS} questions per request")
    for q in questions:
        if not isinstance(q, dict) or not {"id", "question", "options"} <= q.keys():
            raise ValueError("Each question needs 'id', 'question' and 'options'")
    return state, questions


def make_handler(engine: DecisionEngine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 (http.server naming)
            if self.path.rstrip("/") == "/health":
                self._send(200, engine.info())
            else:
                self._send(404, {"error": "Use GET /health or POST /decide"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/decide":
                self._send(404, {"error": "Use POST /decide"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY_BYTES:
                self._send(413, {"error": "Missing or oversized body"})
                return
            try:
                state, questions = parse_request(json.loads(self.rfile.read(length)))
                result = engine.decide(state, questions)
            except (ValueError, json.JSONDecodeError) as error:
                self._send(400, {"error": str(error)})
                return
            except Exception as error:  # model/runtime failure
                self._send(500, {"error": f"{type(error).__name__}: {error}"})
                return
            choices = ", ".join(
                f"{r['id']}={r['choice']} ({r['confidence']:.2f})" for r in result["results"]
            )
            print(f"[{result['mode']}] {result['seconds']:.2f} s  {choices}", flush=True)
            self._send(200, result)

        def log_message(self, format, *args):  # keep the console readable
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve SemIf decisions over HTTP.")
    parser.add_argument("--gguf", default=DEFAULT_GGUF, help="Local GGUF checkpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Reference tokenizer (Hugging Face id or local folder)")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--threads", type=int, help="CPU threads (default: all cores)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Longest prompt accepted, in tokens")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Use 0.0.0.0 if the app can't reach the server from Windows")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    engine = DecisionEngine(args.model, args.revision, args.gguf, args.threads,
                            args.max_tokens)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"SemIf decision server on http://{args.host}:{args.port}  (Ctrl+C to stop)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
