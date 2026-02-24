#!/usr/bin/env python3
"""Grade deferred open-ended rows in a predictions CSV using gpt-5-nano.

Updates rows where:
- answer_type == exactMatch
- grade_method == deferred_open_ended OR is_correct is blank/non-binary

Writes:
- updated predictions CSV (in-place or output path)
- refreshed summary_by_model_and_tone.csv
- refreshed summary_by_model.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc grade deferred open-ended rows.")
    p.add_argument("--predictions-csv", required=True, type=Path)
    p.add_argument("--output-csv", default=None, type=Path)
    p.add_argument("--summary-dir", default=None, type=Path)
    p.add_argument("--status-every", type=int, default=25)
    p.add_argument("--flush-every", type=int, default=25)
    p.add_argument("--min-interval-ms", type=int, default=1200)
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


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


def grade_with_nano(question: str, gold: str, candidate: str, min_interval_ms: int) -> tuple[str, str]:
    if min_interval_ms > 0:
        time.sleep(min_interval_ms / 1000.0)

    key = os.getenv("OPENAI_API_KEY", "")
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    prompt = (
        "You are a strict grading function. Return only one character: 1 or 0.\n"
        "Grade 1 only if the candidate answer is correct for the question given the gold answer.\n"
        "Grade 0 otherwise.\n"
        "Do not output anything except 1 or 0.\n\n"
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Candidate answer:\n{candidate}\n"
    )
    payload = {"model": "gpt-5-nano", "input": prompt}
    raw_text = ""
    for _ in range(2):
        _, raw = with_retries(lambda: post_json(url, headers, payload))
        body = json.loads(raw)
        raw_text = extract_openai_text(body).strip()
        if raw_text in {"0", "1"}:
            return raw_text, raw_text
    return "0", raw_text


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_model_tone: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])  # c, n_graded, n
    by_model_total: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])

    for r in rows:
        k = (r["provider"], r["model_id"], r["prompt_category"])
        by_model_tone[k][2] += 1
        s = str(r.get("is_correct", "")).strip()
        if s in {"0", "1"}:
            by_model_tone[k][0] += int(s)
            by_model_tone[k][1] += 1

        mk = (r["provider"], r["model_id"])
        by_model_total[mk][2] += 1
        if s in {"0", "1"}:
            by_model_total[mk][0] += int(s)
            by_model_total[mk][1] += 1

    neutral = {}
    for (p, m, tone), (c, ng, _n) in by_model_tone.items():
        if tone == "neutral" and ng > 0:
            neutral[(p, m)] = c / ng

    tone_rows = []
    for (p, m, tone), (c, ng, n) in sorted(by_model_tone.items()):
        acc = (c / ng) if ng else None
        base = neutral.get((p, m))
        delta = f"{acc - base:.4f}" if (acc is not None and base is not None) else ""
        tone_rows.append(
            {
                "provider": p,
                "model_id": m,
                "prompt_category": tone,
                "n": n,
                "n_graded": ng,
                "correct": c,
                "accuracy": "" if acc is None else f"{acc:.4f}",
                "delta_vs_neutral": delta,
            }
        )

    model_rows = []
    for (p, m), (c, ng, n) in sorted(by_model_total.items()):
        acc = (c / ng) if ng else None
        model_rows.append(
            {
                "provider": p,
                "model_id": m,
                "n": n,
                "n_graded": ng,
                "correct": c,
                "accuracy_overall": "" if acc is None else f"{acc:.4f}",
            }
        )
    return tone_rows, model_rows


def needs_grading(r: dict) -> bool:
    if (r.get("answer_type") or "").strip() != "exactMatch":
        return False
    s = str(r.get("is_correct", "")).strip()
    if s in {"0", "1"}:
        return False
    return True


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set")

    rows = read_csv(args.predictions_csv)
    if not rows:
        raise ValueError("Empty predictions file")

    output_csv = args.output_csv or args.predictions_csv
    summary_dir = args.summary_dir or output_csv.parent

    targets = [i for i, r in enumerate(rows) if needs_grading(r)]
    print(f"Deferred open-ended rows to grade: {len(targets)}")

    done = 0
    for idx in targets:
        r = rows[idx]
        try:
            # question may be missing in older files; fallback to id-only prompt.
            question = (r.get("question") or "").strip()
            if not question:
                question = f"Question ID: {r.get('id','')}"
            score, raw = grade_with_nano(
                question,
                r.get("gold_answer", ""),
                r.get("model_output", ""),
                args.min_interval_ms,
            )
            r["is_correct"] = score
            r["grader_raw_output"] = raw
            r["grade_method"] = "gpt_5_nano_binary"
            r["error"] = ""
        except Exception as e:
            r["grade_method"] = "error"
            r["error"] = str(e)[:400]

        done += 1
        if args.status_every > 0 and (done % args.status_every == 0 or done == len(targets)):
            print(f"[{done}/{len(targets)}] graded", flush=True)
        if args.flush_every > 0 and (done % args.flush_every == 0):
            fields = list(rows[0].keys())
            write_csv(output_csv, rows, fields)
            tone_rows, model_rows = summarize(rows)
            write_csv(
                summary_dir / "summary_by_model_and_tone.csv",
                tone_rows,
                ["provider", "model_id", "prompt_category", "n", "n_graded", "correct", "accuracy", "delta_vs_neutral"],
            )
            write_csv(
                summary_dir / "summary_by_model.csv",
                model_rows,
                ["provider", "model_id", "n", "n_graded", "correct", "accuracy_overall"],
            )

    fields = list(rows[0].keys())
    write_csv(output_csv, rows, fields)
    tone_rows, model_rows = summarize(rows)
    write_csv(
        summary_dir / "summary_by_model_and_tone.csv",
        tone_rows,
        ["provider", "model_id", "prompt_category", "n", "n_graded", "correct", "accuracy", "delta_vs_neutral"],
    )
    write_csv(
        summary_dir / "summary_by_model.csv",
        model_rows,
        ["provider", "model_id", "n", "n_graded", "correct", "accuracy_overall"],
    )
    print("Done")
    print(f"Wrote: {output_csv}")
    print(f"Wrote: {summary_dir / 'summary_by_model_and_tone.csv'}")
    print(f"Wrote: {summary_dir / 'summary_by_model.csv'}")


if __name__ == "__main__":
    main()
