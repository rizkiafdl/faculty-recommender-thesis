from __future__ import annotations

import re
from typing import Any, Iterable

from app.config import (
    CAPACITY_PRIORITY_CODES,
    HIGH_GPA_THRESHOLD,
)
from app.schemas import SupervisorProfile
from datasets.seed_dataset.label_terms import AGIT_TERMS, BINUS_INTERNAL_TERMS, LABEL_TERMS
from datasets.seed_dataset.supervisor_profiles import (
    BINUS_INTERNAL_ELIGIBLE_CODES,
    BINUS_INTERNAL_PRIORITY_CODES,
)


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

def detect_labels_semantic(
    student_vec: Any,
    label_descriptions: list[dict],
    label_cache: Any,
    provider: Any,
) -> set[str]:
    """
    Detects active labels for a student using cosine similarity between
    the student embedding and each label description embedding.
    Returns set of label_name strings whose cosine score >= threshold.
    Returns empty set if any label embedding cannot be computed
    (caller should then use detect_labels() string match fallback).
    """
    import numpy as np
    active: set[str] = set()
    for ld in label_descriptions:
        label_vec = label_cache.get_or_compute(ld["label_name"], ld["description"], provider)
        if label_vec is None:
            return set()
        score = float(np.dot(student_vec, label_vec))
        if score >= ld["threshold"]:
            active.add(ld["label_name"])
    return active

def profile_document(profile: SupervisorProfile) -> str:
    parts = [
        profile.name,
        profile.code,
        *profile.keywords,
        *profile.labels,
    ]
    return normalize_text(" ".join(parts))

def student_document(student: dict) -> str:
    parts = [
        student.get("track") or "",
        student.get("partner_lecturer") or "",
        student.get("position_topic") or "",
        student.get("work_schema") or "",
    ]
    return normalize_text(" ".join(parts))

def student_labels(student: dict) -> set[str]:
    return detect_labels(student_document(student))

def evaluate_rule_boost(
    student: dict,
    profile: SupervisorProfile,
    active_labels: set[str] | None = None,
    affinity_index: dict[tuple[str, str], float] | None = None,
    niche_defaults: dict[str, float] | None = None,
) -> tuple[float, list[str]]:
    """
    active_labels:   set of detected label strings from detect_labels_semantic().
                     If None → falls back to detect_labels() string matching.
    affinity_index:  dict[(supervisor_code, label_name)] -> boost_value from DB.
                     If empty or None → falls back to hardcoded boost logic.
    niche_defaults:  dict[label_name] -> penalty for all non-specialist supervisors.
                     If empty or None → falls back to hardcoded niche constraints.
    """
    text = normalize_text(" ".join([
        str(student.get("track") or ""),
        str(student.get("partner_lecturer") or ""),
        str(student.get("position_topic") or ""),
        str(student.get("work_schema") or ""),
    ]))
    gpa = student.get("gpa")
    gpa_value = float(gpa) if gpa is not None else 0.0
    high_gpa = gpa_value >= HIGH_GPA_THRESHOLD
    current_supervisor_code = str(student.get("current_supervisor_code") or "").strip()

    if active_labels is not None:
        student_label_set = active_labels
    else:
        student_label_set = detect_labels(text)

    profile_labels = set(profile.labels)
    boost = 0.0
    reasons: list[str] = []

    use_affinity = bool(affinity_index)

    if use_affinity:
        for label in student_label_set:
            key = (profile.code, label)
            if key in affinity_index:
                val = affinity_index[key]
                if val != 0.0:
                    boost += val
                    reasons.append(f"Affinity: {label} ({val:+.1f})")
            elif niche_defaults and label in niche_defaults:
                val = niche_defaults[label]
                boost += val
                reasons.append(f"Niche penalty: {label} ({val:+.1f})")

    # --- Always-on rules (not moved to DB) ---
    binus_internal = "binus_internal_internship" in student_label_set
    if binus_internal and profile.code in BINUS_INTERNAL_ELIGIBLE_CODES:
        internal_sub = 0.0
        if not use_affinity:
            internal_sub = 1.6
        if "software_engineering" in profile_labels:
            internal_sub += 0.3
        if "web_fullstack" in student_label_set and "web_fullstack" in profile_labels:
            internal_sub += 0.45
        if "data_ai" in student_label_set and "data_ai" in profile_labels:
            internal_sub += 0.35
        if profile.code in BINUS_INTERNAL_PRIORITY_CODES:
            internal_sub += 0.65; reasons.append("BINUS internal internship priority")
        else:
            reasons.append("BINUS internal internship eligible")
        if "internship" in profile_labels:
            internal_sub += 0.35
        boost += internal_sub

    agit = contains_any(text, AGIT_TERMS)
    if agit and profile.code == "D1749":
        boost += 1.1; reasons.append("AGIT company affinity")

    if not current_supervisor_code and profile.code in CAPACITY_PRIORITY_CODES:
        boost += 0.9; reasons.append("Unlabeled overflow routing")

    if high_gpa and {"research", "entrepreneurship"} & profile_labels:
        gpa_boost = 1.0
        if profile.code in {"D2211", "D1749"}:
            gpa_boost += 1.0
        boost += gpa_boost; reasons.append("High GPA guidance")


    return boost, list(dict.fromkeys(reasons))
