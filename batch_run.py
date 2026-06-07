"""
Batch pipeline runner — 18 runs across all models × toggle configs × capacity variants.

Usage:
    python batch_run.py [--output-dir output/batch_<timestamp>]

Exports one detailed .xlsx per run into the output directory.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

from app.config import (
    AVAILABLE_EMBEDDING_MODELS,
    CAPACITY_PRIORITY_CODES,
    EMBEDDING_TASK,
    TARGET_MAX_CAPACITY,
    TARGET_MIN_CAPACITY,
)
from app.database import SessionLocal
from app.recommender import RunOverrides
from app.services import export_recommendations_excel_detailed, generate_and_store_recommendations

TOGGLE_CONFIGS = [
    {"label": "no_group_bonus", "enable_group_bonus": False, "enable_extra_docs": True},
    {"label": "no_extra_docs",  "enable_group_bonus": True,  "enable_extra_docs": False},
    {"label": "both_off",       "enable_group_bonus": False, "enable_extra_docs": False},
]

CAPACITY_CONFIGS = [
    {"label": "no_priority",      "capacity_priority_codes": []},
    {"label": "default_priority", "capacity_priority_codes": list(CAPACITY_PRIORITY_CODES)},
]


def build_matrix() -> list[dict]:
    runs = []
    for model, toggle, capacity in product(AVAILABLE_EMBEDDING_MODELS, TOGGLE_CONFIGS, CAPACITY_CONFIGS):
        short_model = model.split("/")[-1]
        label = f"{short_model}__{toggle['label']}__{capacity['label']}"
        runs.append({
            "label": label,
            "overrides": RunOverrides(
                embedding_model=model,
                embedding_task=EMBEDDING_TASK,
                enable_group_bonus=toggle["enable_group_bonus"],
                enable_extra_docs=toggle["enable_extra_docs"],
                capacity_priority_codes=capacity["capacity_priority_codes"],
                target_min_capacity=TARGET_MIN_CAPACITY,
                target_max_capacity=TARGET_MAX_CAPACITY,
            ),
        })
    return runs


def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()

    print(f"\nBatch run — {len(matrix)} runs → {output_dir}\n")
    print(f"{'#':<4} {'Label':<60} {'Status':<10} {'Run ID':<8} {'Duration':>10}")
    print("─" * 96)

    results = []
    for idx, entry in enumerate(matrix, start=1):
        label = entry["label"]
        overrides = entry["overrides"]
        t0 = time.monotonic()
        try:
            with SessionLocal() as session:
                run = generate_and_store_recommendations(
                    session=session,
                    overrides=overrides,
                    input_source="batch-test",
                )
                run_id = run.id

            with SessionLocal() as session:
                xlsx_bytes, filename = export_recommendations_excel_detailed(
                    session=session,
                    run_id=run_id,
                )

            out_path = output_dir / filename
            out_path.write_bytes(xlsx_bytes)
            duration = time.monotonic() - t0
            status = "OK"
            error = None
        except Exception as exc:
            duration = time.monotonic() - t0
            status = "ERROR"
            run_id = "-"
            error = str(exc)

        print(f"{idx:<4} {label:<60} {status:<10} {str(run_id):<8} {duration:>9.1f}s")
        if error:
            print(f"     ↳ {error}")

        results.append({"idx": idx, "label": label, "status": status, "run_id": run_id, "duration": duration, "error": error})

    ok = sum(1 for r in results if r["status"] == "OK")
    fail = len(results) - ok
    print("─" * 96)
    print(f"\nDone — {ok} succeeded, {fail} failed. Exports saved to: {output_dir}\n")

    if fail:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / f"batch_{timestamp}",
    )
    args = parser.parse_args()
    main(args.output_dir)
