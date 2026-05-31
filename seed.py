#!/usr/bin/env python
"""
CLI for seeding static domain data into the database.

Usage:
    python seed.py --all
    python seed.py --supervisors
"""
from __future__ import annotations

import argparse
import sys

from app.database import SessionLocal
from app.services import init_db
from datasets.seed_dataset.seeder import seed_supervisors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed static domain data into the database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--supervisors", action="store_true", help="Seed supervisor profiles")
    parser.add_argument("--all", dest="all_", action="store_true", help="Run all seeders")
    args = parser.parse_args()

    if not any([args.supervisors, args.all_]):
        parser.print_help()
        sys.exit(1)

    init_db()

    with SessionLocal() as session:
        if args.all_ or args.supervisors:
            n = seed_supervisors(session)
            print(f"[supervisors]   {n} change(s)")


if __name__ == "__main__":
    main()
