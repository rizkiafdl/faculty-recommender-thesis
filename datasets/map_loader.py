from __future__ import annotations

from pathlib import Path

import pandas as pd

# Ordered by signal quality — POSITION/TOPIC is richest, TRACK is least informative
_TEXT_COLUMNS = ["POSITION/TOPIC", "PARTNER/LECTURER", "TRACK"]

# Values that add no semantic signal
_NOISE_VALUES = {"-", "nan", "none", "n/a", ""}

_DEFAULT_PATH = Path(__file__).parent.parent / "map_2026.xlsx"
_SHEET = "Sheet1"
_CODE_COL = "kode dosen"


def load_supervisor_extra_docs(
    path: str | Path = _DEFAULT_PATH,
    sheet: str = _SHEET,
) -> dict[str, str]:
    """
    Read map_2026.xlsx and return a dict mapping supervisor code → concatenated
    topic text built from every student row assigned to that supervisor.

    The result is intended to be passed as `extra_supervisor_docs` into
    `generate_recommendations()` to enrich the supervisor embedding document.
    """
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = df.columns.str.strip()

    if _CODE_COL not in df.columns:
        raise ValueError(
            f"Column '{_CODE_COL}' not found in {path!s} sheet '{sheet}'. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[df[_CODE_COL].notna()].copy()
    df[_CODE_COL] = df[_CODE_COL].astype(str).str.strip()
    df = df[df[_CODE_COL] != ""]

    extra_docs: dict[str, str] = {}
    for code, group in df.groupby(_CODE_COL):
        seen: set[str] = set()
        parts: list[str] = []
        for col in _TEXT_COLUMNS:
            if col not in group.columns:
                continue
            for raw in group[col].dropna().astype(str).str.strip():
                token = raw.lower()
                if token in _NOISE_VALUES or token in seen:
                    continue
                seen.add(token)
                parts.append(raw)
        extra_docs[str(code)] = " ".join(parts)

    return extra_docs
