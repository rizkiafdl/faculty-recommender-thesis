#!/usr/bin/env python
"""
CLI for seeding static domain data into the database.

Usage:
    python seed.py --all
    python seed.py --supervisors --labels
    python seed.py --affinity
"""
from __future__ import annotations

import argparse
import sys

from app.database import SessionLocal
from app.services import init_db
from datasets.seed_dataset.seeder import (
    seed_affinity_matrix,
    seed_label_descriptions,
    seed_supervisor_categories,
    seed_supervisors,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed static domain data into the database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--supervisors",  action="store_true", help="Seed supervisor profiles")
    parser.add_argument("--categories",   action="store_true", help="Seed supervisor category assignments from static labels")
    parser.add_argument("--labels",       action="store_true", help="Seed label descriptions")
    parser.add_argument("--affinity",     action="store_true", help="Seed supervisor-label affinity matrix")
    parser.add_argument("--all",          dest="all_", action="store_true", help="Run all seeders")
    args = parser.parse_args()

    if not any([args.supervisors, args.categories, args.labels, args.affinity, args.all_]):
        parser.print_help()
        sys.exit(1)

    init_db()

    with SessionLocal() as session:
        if args.all_ or args.supervisors:
            n = seed_supervisors(session)
            print(f"[supervisors]   {n} change(s)")

        if args.all_ or args.categories:
            n = seed_supervisor_categories(session)
            print(f"[categories]    {n} assignment(s) inserted (0 = already seeded)")

        if args.all_ or args.labels:
            n = seed_label_descriptions(session)
            print(f"[labels]        {n} inserted")

        if args.all_ or args.affinity:
            n = seed_affinity_matrix(session)
            print(f"[affinity]      {n} rows inserted (0 = already seeded, skipped)")


if __name__ == "__main__":
    main()
