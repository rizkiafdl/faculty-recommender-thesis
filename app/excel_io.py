from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


CANONICAL_COLUMNS = {
    "BINUSIAN ID": "binusian_id",
    "STUDENT ID": "student_id",
    "STUDENT NAME": "name",
    "STUDENT EMAIL": "email",
    "STUDENT PHONE": "phone",
    "TRACK": "track",
    "GPA": "gpa",
    "TOTAL SKS": "total_sks",
    "PARTNER/LECTURER": "partner_lecturer",
    "POSITION/TOPIC": "position_topic",
    "DURATION": "duration",
    "JOB START DATE": "job_start_date",
    "JOB END DATE": "job_end_date",
    "WORK SCHEMA": "work_schema",
    "ENROLLMENT STATUS": "enrollment_status",
    "kode dosen": "current_supervisor_code",
    "FS": "current_supervisor_name",
}


def _clean_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text or None


def _clean_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _clean_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _clean_date(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    return text or None


def _read_dataframe_from_path(path: str | Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name)


def _read_dataframe_from_bytes(content: bytes, sheet_name: str) -> pd.DataFrame:
    buffer = io.BytesIO(content)
    return pd.read_excel(buffer, sheet_name=sheet_name)


def _normalize_dataframe(df: pd.DataFrame) -> list[dict]:
    missing = [column for column in CANONICAL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df = df[df["STUDENT ID"].notna()].copy()
    rows: list[dict] = []
    for row in df.to_dict(orient="records"):
        student_id = _clean_id(row.get("STUDENT ID"))
        if not student_id:
            continue
        rows.append(
            {
                "student_id": student_id,
                "binusian_id": _clean_id(row.get("BINUSIAN ID")),
                "name": _clean_text(row.get("STUDENT NAME")) or "Unknown",
                "email": _clean_text(row.get("STUDENT EMAIL")),
                "phone": _clean_text(row.get("STUDENT PHONE")),
                "track": _clean_text(row.get("TRACK")),
                "gpa": _clean_float(row.get("GPA")),
                "total_sks": _clean_int(row.get("TOTAL SKS")),
                "partner_lecturer": _clean_text(row.get("PARTNER/LECTURER")),
                "position_topic": _clean_text(row.get("POSITION/TOPIC")),
                "duration": _clean_text(row.get("DURATION")),
                "job_start_date": _clean_date(row.get("JOB START DATE")),
                "job_end_date": _clean_date(row.get("JOB END DATE")),
                "work_schema": _clean_text(row.get("WORK SCHEMA")),
                "enrollment_status": _clean_text(row.get("ENROLLMENT STATUS")),
                "current_supervisor_code": _clean_id(row.get("kode dosen")),
                "current_supervisor_name": _clean_text(row.get("FS")),
            }
        )
    return rows


def read_students_from_excel_path(path: str | Path, sheet_name: str) -> list[dict]:
    df = _read_dataframe_from_path(path=path, sheet_name=sheet_name)
    return _normalize_dataframe(df)


def read_students_from_excel_bytes(content: bytes, sheet_name: str) -> list[dict]:
    df = _read_dataframe_from_bytes(content=content, sheet_name=sheet_name)
    return _normalize_dataframe(df)

