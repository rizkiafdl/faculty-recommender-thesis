from __future__ import annotations

import re
from typing import Iterable

from app.schemas import SupervisorProfile
from datasets.seed_dataset.label_terms import BINUS_INTERNAL_TERMS, LABEL_TERMS


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)

def detect_labels(text: str) -> set[str]:
    labels: set[str] = set()
    for label, terms in LABEL_TERMS.items():
        if contains_any(text, terms):
            labels.add(label)
    # Composite label: internship internal BINUS.
    if "internship" in labels and contains_any(text, BINUS_INTERNAL_TERMS):
        labels.add("binus_internal_internship")
        labels.add("binus_bandung")
    return labels

def profile_document(profile: SupervisorProfile) -> str:
    parts = [
        # profile.name,
        # profile.code,
        *profile.keywords,
        *profile.labels,
    ]
    return normalize_text(" ".join(parts))

def student_document(student: dict) -> str:
    parts = [
        student.get("track") or "",
        # student.get("partner_lecturer") or "",
        student.get("position_topic") or "",
        student.get("work_schema") or "",
    ]
    return normalize_text(" ".join(parts))

def student_labels(student: dict) -> set[str]:
    return detect_labels(student_document(student))
