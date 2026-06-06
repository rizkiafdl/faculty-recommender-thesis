#!/usr/bin/env python
"""
Migration: Remove rule_boost engine and affinity/label-description tables.

Run once against an existing database that still has the old schema:
    python migrations/drop_rule_boost_and_affinity.py

Safe to re-run — each step checks existence before acting.
Requires SQLite >= 3.35.0 for DROP COLUMN support.
"""
from __future__ import annotations

from sqlalchemy import inspect, text

from app.database import engine


def run() -> None:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        # Drop rule_boost column from recommendations
        if "recommendations" in existing_tables:
            existing_cols = {c["name"] for c in inspector.get_columns("recommendations")}
            if "rule_boost" in existing_cols:
                conn.execute(text("ALTER TABLE recommendations DROP COLUMN rule_boost"))
                print("[recommendations] Dropped column: rule_boost")
            else:
                print("[recommendations] Column rule_boost already absent — skipped")

        # Drop supervisor_label_affinities table
        if "supervisor_label_affinities" in existing_tables:
            conn.execute(text("DROP TABLE supervisor_label_affinities"))
            print("[supervisor_label_affinities] Table dropped")
        else:
            print("[supervisor_label_affinities] Already absent — skipped")

        # Drop label_descriptions table
        if "label_descriptions" in existing_tables:
            conn.execute(text("DROP TABLE label_descriptions"))
            print("[label_descriptions] Table dropped")
        else:
            print("[label_descriptions] Already absent — skipped")

    print("Migration complete.")


if __name__ == "__main__":
    run()
