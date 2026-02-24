#!/usr/bin/env python3
"""Run multi-provider chemistry benchmark for 5-level tone prompts.

Reads:
- results/models_to_test.csv
- results/prompt_variants_tone_chemistry_5level.csv

Writes:
- final_results/all_model_predictions.csv
- final_results/summary_by_model_and_tone.csv
- final_results/summary_by_model.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


LETTER_RE = re.compile(r"\b([A-F])\b", re.IGNORECASE)
FINAL_ANSWER_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:final\s+answer|answer)\s*[:\-]\s*\(?([A-F])\)?\b"
)

THREAD_LOCAL = threading.local()
PROVIDER_LOCKS: dict[str, threading.Lock] = defaultdict(threading.Lock)
PROVIDER_LAST_CALL_TS: dict[str, float] = defaultdict(float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all models across 5-level chemistry tone prompts.")
    parser.add_argument("--models-csv", default=Path("results/models_to_test.csv"), type=Path)
    parser.add_argument(
        "--prompts-csv",
        default=Path("results/prompt_variants_tone_chemistry_5level.csv"),
        type=Path,
    )
    parser.add_argument("--output-dir", default=Path("final_results"), type=Path)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--status-every", type=int, default=50)
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-error-rows", action="store_true")
    parser.add_argument("--min-interval-ms", type=int, default=0)
    parser.add_argument("--defer-open-ended-grading", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def post_json(url: str, headers: dict[str, str], payload: dict, timeout: int = 30) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def with_retries(fn, max_attempts: int = 3):
    for attempt in range(max_attempts):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                time.sleep(min(6, (1.5**attempt) + random.random()))
                continue
            raise
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(min(6, (1.5**attempt) + random.random()))
                continue
            raise


def extract_openai_text(body: dict) -> str:
    text = body.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    chunks: list[str] = []
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(content["text"])
    return "\n".join(chunks).strip()


def throttle_provider(provider: str, min_interval_ms: int) -> None:
    if min_interval_ms <= 0:
        return
    wait_s = min_interval_ms / 1000.0
    lock = PROVIDER_LOCKS[provider]
    with lock:
        now = time.time()
        elapsed = now - PROVIDER_LAST_CALL_TS[provider]
        if elapsed < wait_s:
            time.sleep(wait_s - elapsed)
        PROVIDER_LAST_CALL_TS[provider] = time.time()


def call_model(provider: str, model_id: str, prompt_text: str, min_interval_ms: int = 0) -> str:
    provider = provider.lower()
    throttle_provider(provider, min_interval_ms)

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        url = "https://api.openai.com/v1/responses"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": model_id, "input": prompt_text}

        _, raw = with_retries(lambda: post_json(url, headers, payload))
        body = json.loads(raw)
        return extract_openai_text(body)

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {"model": model_id, "max_tokens": 256, "messages": [{"role": "user", "content": prompt_text}]}

        _, raw = with_retries(lambda: post_json(url, headers, payload))
        body = json.loads(raw)
        text_parts = [x.get("text", "") for x in body.get("content", []) if x.get("type") == "text"]
        return "\n".join([t for t in text_parts if t]).strip()

    if provider == "mistral":
        key = os.getenv("MISTRAL_API_KEY", "")
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 256,
        }

        _, raw = with_retries(lambda: post_json(url, headers, payload))
        body = json.loads(raw)
        return (body.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()

    if provider == "kimi":
        key = os.getenv("KIMI_API_KEY", "")
        url = "https://api.moonshot.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        max_tokens = 1024
        for _ in range(3):
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": max_tokens,
            }
            _, raw = with_retries(lambda: post_json(url, headers, payload, timeout=120))
            body = json.loads(raw)
            choice = (body.get("choices", [{}])[0] or {})
            msg = (choice.get("message", {}) or {})
            content = (msg.get("content") or "").strip()
            finish_reason = (choice.get("finish_reason") or "").strip()
            # Kimi can emit only reasoning_content and cut off before final content.
            if content:
                return content
            if finish_reason != "length":
                return content
            max_tokens = min(4096, max_tokens * 2)
        return ""

    raise ValueError(f"Unsupported provider: {provider}")


def grade_open_ended_with_nano(question: str, gold_answer: str, candidate_answer: str) -> tuple[int, str]:
    key = os.getenv("OPENAI_API_KEY", "")
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    grader_prompt = (
        "You are a strict grading function. Return only one character: 1 or 0.\n"
        "Grade 1 only if the candidate answer is correct for the question given the gold answer.\n"
        "Grade 0 otherwise.\n"
        "Do not output anything except 1 or 0.\n\n"
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold_answer}\n\n"
        f"Candidate answer:\n{candidate_answer}\n"
    )
    payload = {"model": "gpt-5-nano", "input": grader_prompt}

    raw_text = ""
    for _ in range(2):
        _, raw = with_retries(lambda: post_json(url, headers, payload))
        body = json.loads(raw)
        raw_text = extract_openai_text(body).strip()
        if raw_text in {"0", "1"}:
            return int(raw_text), raw_text
    return 0, raw_text


def extract_mcq_letter(text: str) -> str:
    text = (text or "").strip()
    upper = text.upper()
    if len(upper) == 1 and upper in "ABCDEF":
        return upper
    final = FINAL_ANSWER_RE.search(text)
    if final:
        return final.group(1).upper()
    match = LETTER_RE.search(text)
    return match.group(1).upper() if match else ""


def row_eval(job: dict, min_interval_ms: int, defer_open_ended_grading: bool) -> dict:
    provider = job["provider"]
    model_id = job["model_id"]
    prompt = job["prompt"]

    output = call_model(provider, model_id, prompt["prompt_text"], min_interval_ms=min_interval_ms)
    parsed_prediction = ""
    grader_raw = ""

    if prompt["answer_type"] == "multipleChoice":
        parsed_prediction = extract_mcq_letter(output)
        is_correct = int(parsed_prediction == (prompt["answer"] or "").strip().upper())
        grade_method = "mcq_letter_match"
    else:
        parsed_prediction = output
        if defer_open_ended_grading:
            is_correct = ""
            grade_method = "deferred_open_ended"
        else:
            is_correct, grader_raw = grade_open_ended_with_nano(
                prompt["question"],
                prompt["answer"],
                output,
            )
            grade_method = "gpt_5_nano_binary"

    return {
        "provider": provider,
        "model_id": model_id,
        "id": prompt["id"],
        "question": prompt["question"],
        "prompt_category": prompt["prompt_category"],
        "answer_type": prompt["answer_type"],
        "gold_answer": prompt["answer"],
        "model_output": output,
        "parsed_prediction": parsed_prediction,
        "grade_method": grade_method,
        "grader_raw_output": grader_raw,
        "is_correct": is_correct,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def job_key(provider: str, model_id: str, prompt: dict) -> tuple[str, str, str, str]:
    return (provider, model_id, prompt["id"], prompt["prompt_category"])


def summarize(pred_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_model_tone: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])  # correct, n_graded, n_total
    by_model_total: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])

    for r in pred_rows:
        key = (r["provider"], r["model_id"], r["prompt_category"])
        by_model_tone[key][2] += 1
        s = str(r.get("is_correct", "")).strip()
        if s in {"0", "1"}:
            by_model_tone[key][0] += int(s)
            by_model_tone[key][1] += 1

        mkey = (r["provider"], r["model_id"])
        by_model_total[mkey][2] += 1
        if s in {"0", "1"}:
            by_model_total[mkey][0] += int(s)
            by_model_total[mkey][1] += 1

    model_neutral: dict[tuple[str, str], float] = {}
    for (provider, model, tone), (correct, n_graded, _n_total) in by_model_tone.items():
        if tone == "neutral" and n_graded > 0:
            model_neutral[(provider, model)] = correct / n_graded

    tone_rows: list[dict] = []
    for (provider, model, tone), (correct, n_graded, n_total) in sorted(by_model_tone.items()):
        acc = (correct / n_graded) if n_graded else None
        base = model_neutral.get((provider, model))
        delta = ""
        if base is not None and acc is not None:
            delta = f"{acc - base:.4f}"
        tone_rows.append(
            {
                "provider": provider,
                "model_id": model,
                "prompt_category": tone,
                "n": n_total,
                "n_graded": n_graded,
                "correct": correct,
                "accuracy": "" if acc is None else f"{acc:.4f}",
                "delta_vs_neutral": delta,
            }
        )

    model_rows: list[dict] = []
    for (provider, model), (correct, n_graded, n_total) in sorted(by_model_total.items()):
        acc = (correct / n_graded) if n_graded else None
        model_rows.append(
            {
                "provider": provider,
                "model_id": model,
                "n": n_total,
                "n_graded": n_graded,
                "correct": correct,
                "accuracy_overall": "" if acc is None else f"{acc:.4f}",
            }
        )
    return tone_rows, model_rows


def main() -> None:
    args = parse_args()

    required_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MISTRAL_API_KEY",
        "KIMI_API_KEY",
    ]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"Missing env vars: {missing}")

    models = read_csv(args.models_csv)
    prompts = read_csv(args.prompts_csv)
    if not models or not prompts:
        raise ValueError("Empty models or prompts input.")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "all_model_predictions.csv"

    existing_rows: list[dict] = []
    existing_keys: set[tuple[str, str, str, str]] = set()
    pred_map: dict[tuple[str, str, str, str], dict] = {}
    if args.resume and pred_path.exists():
        existing_rows = read_csv(pred_path)
        for r in existing_rows:
            k = (r["provider"], r["model_id"], r["id"], r["prompt_category"])
            pred_map[k] = r
            if args.retry_error_rows and (r.get("grade_method") == "error" or (r.get("error") or "").strip()):
                continue
            existing_keys.add(k)
        print(f"Resume enabled: loaded {len(existing_rows)} existing rows.")

    jobs = []
    for m in models:
        for p in prompts:
            k = job_key(m["provider"], m["model_id"], p)
            if k in existing_keys:
                continue
            jobs.append({"provider": m["provider"], "model_id": m["model_id"], "prompt": p})

    total = len(jobs)
    print(
        f"Starting benchmark jobs={total} models={len(models)} prompts={len(prompts)} "
        f"workers={args.max_workers}"
    )
    start = time.time()
    done = 0
    pred_rows: list[dict] = []

    all_fields = [
        "provider",
        "model_id",
        "id",
        "question",
        "prompt_category",
        "answer_type",
        "gold_answer",
        "model_output",
        "parsed_prediction",
        "grade_method",
        "grader_raw_output",
        "is_correct",
        "error",
    ]

    def flush_outputs() -> None:
        pred_rows[:] = list(pred_map.values())
        # sort for reproducibility
        pred_rows.sort(key=lambda r: (r["provider"], r["model_id"], r["id"], r["prompt_category"]))
        for r in pred_rows:
            if "error" not in r:
                r["error"] = ""
        write_csv(pred_path, pred_rows, all_fields)
        tone_rows, model_rows = summarize(pred_rows)
        write_csv(
            out_dir / "summary_by_model_and_tone.csv",
            tone_rows,
            [
                "provider",
                "model_id",
                "prompt_category",
                "n",
                "n_graded",
                "correct",
                "accuracy",
                "delta_vs_neutral",
            ],
        )
        write_csv(
            out_dir / "summary_by_model.csv",
            model_rows,
            ["provider", "model_id", "n", "n_graded", "correct", "accuracy_overall"],
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            future_map = {
                ex.submit(row_eval, j, args.min_interval_ms, args.defer_open_ended_grading): j
                for j in jobs
            }
            for fut in concurrent.futures.as_completed(future_map):
                j = future_map[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {
                        "provider": j["provider"],
                        "model_id": j["model_id"],
                        "id": j["prompt"]["id"],
                        "question": j["prompt"]["question"],
                        "prompt_category": j["prompt"]["prompt_category"],
                        "answer_type": j["prompt"]["answer_type"],
                        "gold_answer": j["prompt"]["answer"],
                        "model_output": "",
                        "parsed_prediction": "",
                        "grade_method": "error",
                        "grader_raw_output": "",
                        "is_correct": 0,
                        "error": str(e)[:400],
                    }
                k = (rec["provider"], rec["model_id"], rec["id"], rec["prompt_category"])
                pred_map[k] = rec
                done += 1

                if args.status_every > 0 and (done % args.status_every == 0 or done == total):
                    elapsed = time.time() - start
                    print(
                        f"[{done}/{total}] provider={j['provider']} model={j['model_id']} "
                        f"tone={j['prompt']['prompt_category']} elapsed_s={elapsed:.1f}",
                        flush=True,
                    )

                if args.flush_every > 0 and (done % args.flush_every == 0):
                    flush_outputs()
                    print(f"Checkpoint written at {done}/{total}", flush=True)
    finally:
        flush_outputs()

    print(f"Done in {time.time() - start:.1f}s")
    print(f"Wrote: {pred_path}")
    print(f"Wrote: {out_dir / 'summary_by_model_and_tone.csv'}")
    print(f"Wrote: {out_dir / 'summary_by_model.csv'}")


if __name__ == "__main__":
    main()
