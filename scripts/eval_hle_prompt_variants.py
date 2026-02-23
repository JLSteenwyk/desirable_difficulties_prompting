#!/usr/bin/env python3
"""Run GPT-5 mini on prompt variants and grade outputs.

Input CSV must include columns:
- id
- prompt_category
- prompt_text
- answer_type (multipleChoice or exactMatch)
- answer (gold)
- question (used by open-ended grader)

Outputs:
- per-example CSV with predictions and correctness
- summary CSV with accuracy by category and delta vs baseline
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI, RateLimitError


LETTER_RE = re.compile(r"\b([A-F])\b", re.IGNORECASE)
FINAL_ANSWER_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:final\s+answer|answer)\s*[:\-]\s*\(?([A-F])\)?\b"
)
THREAD_LOCAL = threading.local()


@dataclass
class EvalRow:
    row_index: int
    id: str
    prompt_category: str
    prompt_text: str
    answer_type: str
    gold_answer: str
    question: str


def read_rows(path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "prompt_category", "prompt_text", "answer_type", "answer", "question"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        for i, raw in enumerate(reader):
            rows.append(
                EvalRow(
                    row_index=i,
                    id=(raw.get("id") or "").strip(),
                    prompt_category=(raw.get("prompt_category") or "").strip(),
                    prompt_text=(raw.get("prompt_text") or "").strip(),
                    answer_type=(raw.get("answer_type") or "").strip(),
                    gold_answer=(raw.get("answer") or "").strip(),
                    question=(raw.get("question") or "").strip(),
                )
            )
    return rows


def extract_mcq_letter(text: str) -> str:
    text = (text or "").strip().upper()
    if len(text) == 1 and text in "ABCDEF":
        return text
    final = FINAL_ANSWER_RE.search(text)
    if final:
        return final.group(1).upper()
    match = LETTER_RE.search(text)
    return match.group(1).upper() if match else ""


def get_client() -> OpenAI:
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAI()
        THREAD_LOCAL.client = client
    return client


def call_model_answer(client: OpenAI, model: str, prompt_text: str) -> str:
    req = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt_text}],
            }
        ],
        "temperature": 0,
    }
    for attempt in range(5):
        try:
            resp = client.responses.create(**req)
            break
        except BadRequestError as e:
            # Some models do not support temperature; retry deterministically with default settings.
            msg = str(e)
            if "Unsupported parameter: 'temperature'" in msg and "temperature" in req:
                req.pop("temperature", None)
                continue
            raise
        except (APIConnectionError, APITimeoutError, RateLimitError):
            if attempt == 4:
                raise
            time.sleep(min(8, 1.5**attempt))
    return (resp.output_text or "").strip()


def grade_open_ended_with_nano(
    client: OpenAI,
    grader_model: str,
    *,
    question: str,
    gold_answer: str,
    candidate_answer: str,
    max_retries: int = 1,
) -> tuple[int, str]:
    grader_prompt = (
        "You are a strict grading function. Return only one character: 1 or 0.\n"
        "Grade 1 only if the candidate answer is correct for the question given the gold answer.\n"
        "Grade 0 otherwise.\n"
        "Do not output anything except 1 or 0.\n\n"
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold_answer}\n\n"
        f"Candidate answer:\n{candidate_answer}\n"
    )

    last = ""
    for _ in range(max_retries + 1):
        req = {
            "model": grader_model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": grader_prompt}],
                }
            ],
            "temperature": 0,
        }
        for attempt in range(5):
            try:
                resp = client.responses.create(**req)
                break
            except BadRequestError as e:
                msg = str(e)
                if "Unsupported parameter: 'temperature'" in msg and "temperature" in req:
                    req.pop("temperature", None)
                    continue
                raise
            except (APIConnectionError, APITimeoutError, RateLimitError):
                if attempt == 4:
                    raise
                time.sleep(min(8, 1.5**attempt))
        last = (resp.output_text or "").strip()
        if last in {"0", "1"}:
            return int(last), last
    return 0, last


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_results(path: Path, records: list[dict]) -> None:
    ensure_parent_dir(path)
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def summarize(records: list[dict], baseline_name: str) -> list[dict]:
    by_cat: dict[str, dict[str, float]] = {}

    for r in records:
        cat = r["prompt_category"]
        bucket = by_cat.setdefault(cat, {"n": 0.0, "correct": 0.0})
        bucket["n"] += 1
        bucket["correct"] += int(r["is_correct"]) 

    summary: list[dict] = []
    baseline_acc: Optional[float] = None
    if baseline_name in by_cat and by_cat[baseline_name]["n"] > 0:
        baseline_acc = by_cat[baseline_name]["correct"] / by_cat[baseline_name]["n"]

    for cat in sorted(by_cat):
        n = int(by_cat[cat]["n"])
        correct = int(by_cat[cat]["correct"])
        acc = correct / n if n else 0.0
        delta = ""
        if baseline_acc is not None:
            delta = f"{acc - baseline_acc:.4f}"
        summary.append(
            {
                "prompt_category": cat,
                "n": n,
                "correct": correct,
                "accuracy": f"{acc:.4f}",
                "delta_vs_baseline": delta,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT-5 mini and grade answers.")
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", default=Path("results/eval_results.csv"), type=Path)
    parser.add_argument("--summary-csv", default=Path("results/eval_summary.csv"), type=Path)
    parser.add_argument("--answer-model", default="gpt-5-mini")
    parser.add_argument("--grader-model", default="gpt-5-nano")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--status-every",
        type=int,
        default=1,
        help="Print a status update every N rows (default: 1). Set 0 to disable.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers for model calls (default: 1).",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")

    rows = read_rows(args.input_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.input_csv}")

    records: list[dict] = []
    total = len(rows)
    start = time.time()
    completed = 0

    def evaluate_row(row: EvalRow) -> dict:
        client = get_client()
        model_output = call_model_answer(client, args.answer_model, row.prompt_text)

        parsed_prediction = ""
        grader_raw = ""
        if row.answer_type == "multipleChoice":
            parsed_prediction = extract_mcq_letter(model_output)
            is_correct = int(parsed_prediction == row.gold_answer.strip().upper())
            grade_method = "mcq_letter_match"
        else:
            parsed_prediction = model_output
            is_correct, grader_raw = grade_open_ended_with_nano(
                client,
                args.grader_model,
                question=row.question,
                gold_answer=row.gold_answer,
                candidate_answer=model_output,
            )
            grade_method = "gpt_5_nano_binary"

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

        return {
            "row_index": row.row_index,
            "id": row.id,
            "prompt_category": row.prompt_category,
            "answer_type": row.answer_type,
            "gold_answer": row.gold_answer,
            "model_output": model_output,
            "parsed_prediction": parsed_prediction,
            "grade_method": grade_method,
            "grader_raw_output": grader_raw,
            "is_correct": is_correct,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {ex.submit(evaluate_row, row): row for row in rows}
        for fut in concurrent.futures.as_completed(futures):
            row = futures[fut]
            rec = fut.result()
            records.append(rec)
            completed += 1

            if args.status_every > 0 and (completed % args.status_every == 0 or completed == total):
                correct_so_far = sum(int(r["is_correct"]) for r in records)
                running_acc = correct_so_far / completed
                elapsed = time.time() - start
                print(
                    f"[{completed}/{total}] id={row.id} category={row.prompt_category} "
                    f"correct={rec['is_correct']} running_acc={running_acc:.3f} elapsed_s={elapsed:.1f}",
                    flush=True,
                )

    records.sort(key=lambda r: int(r["row_index"]))

    write_results(args.output_csv, records)
    summary = summarize(records, baseline_name=args.baseline_name)
    write_results(args.summary_csv, summary)

    print(f"Wrote detailed results: {args.output_csv}")
    print(f"Wrote summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
