#!/usr/bin/env python3
"""Run full HLE-gold tone benchmark in per-model chunks.

This script executes `scripts/run_all_models_tone5_chemistry.py` once per model,
writing outputs into:
  final_results/hlegold_full/<provider>__<model_id>/

It is designed for long-running benchmark reliability:
- one-model-at-a-time isolation
- resume/retry support
- checkpointed outputs per chunk
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HLE-gold benchmark in model chunks.")
    p.add_argument("--models-csv", default=Path("results/models_to_test.csv"), type=Path)
    p.add_argument(
        "--prompts-csv",
        default=Path("results/prompt_variants_tone_hlegold_5level.csv"),
        type=Path,
    )
    p.add_argument("--output-root", default=Path("final_results/hlegold_full"), type=Path)
    p.add_argument("--runner-script", default=Path("scripts/run_all_models_tone5_chemistry.py"), type=Path)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--status-every", type=int, default=10)
    p.add_argument("--flush-every", type=int, default=10)
    p.add_argument("--min-interval-ms", type=int, default=1200)
    p.add_argument("--start-index", type=int, default=1, help="1-based model start index")
    p.add_argument("--end-index", type=int, default=0, help="1-based model end index; 0 means all")
    p.add_argument("--provider", default="", help="Optional provider filter (e.g., openai)")
    p.add_argument("--model-id", default="", help="Optional exact model filter")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--defer-open-ended-grading", action="store_true")
    return p.parse_args()


def load_models(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        provider = (r.get("provider") or "").strip()
        model_id = (r.get("model_id") or "").strip()
        if provider and model_id:
            out.append({"provider": provider, "model_id": model_id})
    return out


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def write_single_model_csv(path: Path, provider: str, model_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["provider", "model_id"])
        w.writeheader()
        w.writerow({"provider": provider, "model_id": model_id})


def main() -> int:
    args = parse_args()
    models = load_models(args.models_csv)
    if not models:
        print(f"No models found in {args.models_csv}", file=sys.stderr)
        return 2

    # Optional provider/model filters
    if args.provider:
        models = [m for m in models if m["provider"].lower() == args.provider.lower()]
    if args.model_id:
        models = [m for m in models if m["model_id"] == args.model_id]
    if not models:
        print("No models remain after filtering.", file=sys.stderr)
        return 2

    # 1-based index slicing for chunk control
    start_idx = max(1, args.start_index)
    end_idx = args.end_index if args.end_index > 0 else len(models)
    selected = models[start_idx - 1 : end_idx]
    if not selected:
        print("No models selected for requested index range.", file=sys.stderr)
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.output_root / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected {len(selected)} model chunk(s).")
    failed = 0

    for i, m in enumerate(selected, start=1):
        provider = m["provider"]
        model_id = m["model_id"]
        model_slug = f"{slug(provider)}__{slug(model_id)}"
        out_dir = args.output_root / model_slug
        one_model_csv = tmp_dir / f"{model_slug}.csv"
        write_single_model_csv(one_model_csv, provider, model_id)

        cmd = [
            sys.executable,
            str(args.runner_script),
            "--models-csv",
            str(one_model_csv),
            "--prompts-csv",
            str(args.prompts_csv),
            "--output-dir",
            str(out_dir),
            "--max-workers",
            str(args.max_workers),
            "--status-every",
            str(args.status_every),
            "--flush-every",
            str(args.flush_every),
            "--resume",
            "--retry-error-rows",
            "--min-interval-ms",
            str(args.min_interval_ms),
        ]
        if args.defer_open_ended_grading:
            cmd.append("--defer-open-ended-grading")

        print(
            f"\n[{i}/{len(selected)}] Running chunk provider={provider} model={model_id}\n"
            f"Output: {out_dir}"
        )
        rc = subprocess.call(cmd)
        if rc != 0:
            failed += 1
            print(f"Chunk FAILED (exit={rc}) provider={provider} model={model_id}", file=sys.stderr)
            if not args.continue_on_error:
                return rc
        else:
            print(f"Chunk completed provider={provider} model={model_id}")

    if failed:
        print(f"Finished with {failed} failed chunk(s).", file=sys.stderr)
        return 1

    print("All selected chunks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
