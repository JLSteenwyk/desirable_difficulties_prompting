#!/usr/bin/env python3
"""Build prompt variants from questions CSV + categories.md.

Output schema is compatible with eval_hle_prompt_variants.py.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "question", "answer", "answer_type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        return list(reader)


def parse_categories(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    # In this repository, each non-empty line is one category description.
    blocks = [line.strip() for line in text.splitlines() if line.strip()]

    categories: list[tuple[str, str]] = []
    for block in blocks:
        first_line = block.splitlines()[0].strip()
        name = re.sub(r"\s*\(.*$", "", first_line).strip()
        ascii_name = (
            unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        )
        slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
        if not slug:
            continue
        categories.append((slug, block))
    return categories


def build_prompt(question: str, mode: str, category_text: str) -> str:
    if mode == "baseline":
        return (
            "Answer the following question. Return only your final answer.\n\n"
            f"{question}"
        )

    return (
        "Use the following prompting strategy while solving the question.\n"
        f"Strategy: {category_text}\n\n"
        "Then answer the question. Return only your final answer.\n\n"
        f"{question}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prompt variants CSV.")
    parser.add_argument("--questions-csv", required=True, type=Path)
    parser.add_argument("--categories-md", required=True, type=Path)
    parser.add_argument("--output-csv", default=Path("results/prompt_variants.csv"), type=Path)
    args = parser.parse_args()

    questions = load_questions(args.questions_csv)
    categories = parse_categories(args.categories_md)

    rows: list[dict] = []
    for q in questions:
        rows.append(
            {
                "id": (q.get("id") or "").strip(),
                "question": (q.get("question") or "").strip(),
                "answer": (q.get("answer") or "").strip(),
                "answer_type": (q.get("answer_type") or "").strip(),
                "prompt_category": "baseline",
                "prompt_text": build_prompt((q.get("question") or "").strip(), "baseline", ""),
            }
        )

        for slug, category_text in categories:
            rows.append(
                {
                    "id": (q.get("id") or "").strip(),
                    "question": (q.get("question") or "").strip(),
                    "answer": (q.get("answer") or "").strip(),
                    "answer_type": (q.get("answer_type") or "").strip(),
                    "prompt_category": slug,
                    "prompt_text": build_prompt((q.get("question") or "").strip(), slug, category_text),
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "question", "answer", "answer_type", "prompt_category", "prompt_text"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} prompt variants to {args.output_csv}")


if __name__ == "__main__":
    main()
