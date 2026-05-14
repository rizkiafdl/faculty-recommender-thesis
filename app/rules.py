from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.config import (
    CAPACITY_PRIORITY_CODES,
    HIGH_GPA_THRESHOLD,
)


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


@dataclass(frozen=True)
class SupervisorProfile:
    code: str
    name: str
    keywords: tuple[str, ...]
    labels: tuple[str, ...] = ()
    flexibility: float = 0.0


SUPERVISOR_PROFILES: tuple[SupervisorProfile, ...] = (
    SupervisorProfile(
        code="D6670",
        name="Budi Juarto",
        keywords=(
            "internship",
            "binus bandung",
            "studi independent",
            "specific independent study",
            "industry project",
            "software engineering",
            "project implementation",
            "web application",
        ),
        labels=(
            "internship",
            "independent_study",
            "binus_bandung",
            "software_engineering",
            "web_fullstack",
            "general_flexible",
        ),
        flexibility=0.35,
    ),
    SupervisorProfile(
        code="D7187",
        name="Cutifa Safitri, B.CS., M. IT., Ph.D.",
        keywords=(
            "internship",
            "binus bandung",
            "studi independent",
            "specific independent study",
            "company project",
            "software engineering",
            "research methodology",
            "digital innovation",
        ),
        labels=(
            "internship",
            "independent_study",
            "binus_bandung",
            "software_engineering",
            "research",
            "general_flexible",
        ),
        flexibility=0.4,
    ),
    SupervisorProfile(
        code="D5918",
        name="Dr. Boby Siswanto, S.T., M.T.",
        keywords=(
            "internship",
            "binus bandung",
            "studi independent",
            "research",
            "methodology",
            "data analytics",
            "software engineering",
            "project supervision",
        ),
        labels=(
            "independent_study",
            "research",
            "software_engineering",
            "data_ai",
            "internship",
        ),
        flexibility=0.3,
    ),
    SupervisorProfile(
        code="D6532",
        name="Dr. Dani Suandi, S.Si., M.Si.",
        keywords=(
            "internship",
            "binus bandung",
            "studi independent",
            "research",
            "data science",
            "analytics",
            "machine learning",
            "decision support",
        ),
        labels=(
            "independent_study",
            "research",
            "data_ai",
            "software_engineering",
            "internship",
        ),
        flexibility=0.3,
    ),
    SupervisorProfile(
        code="D2211",
        name="Dr. Abdul Haris Rangkuti, S.Kom., M.M., M.Si.",
        keywords=(
            "high gpa",
            "flexible",
            "entrepreneurship",
            "network",
            "general guidance",
            "company collaboration",
            "industry project",
            "management information system",
        ),
        labels=(
            "entrepreneurship",
            "network_cloud",
            "internship",
            "software_engineering",
            "general_flexible",
        ),
        flexibility=0.8,
    ),
    SupervisorProfile(
        code="D1749",
        name="Dr. Johan Muliadi Kerta, S.Kom., M.M.",
        keywords=(
            "high gpa",
            "network",
            "entrepreneurship",
            "startup",
            "industry",
            "business technology",
            "agit",
            "strategic project",
        ),
        labels=(
            "entrepreneurship",
            "network_cloud",
            "internship",
            "software_engineering",
            "general_flexible",
        ),
        flexibility=0.65,
    ),
    SupervisorProfile(
        code="D6407",
        name="Dr. Dany Eka Saputra, S.T., M.T.",
        keywords=(
            "drone",
            "uav",
            "government",
            "public sector",
            "smart city",
            "iot",
            "embedded system",
            "sensor network",
        ),
        labels=(
            "iot_embedded",
            "government_public",
            "network_cloud",
            "research",
        ),
        flexibility=0.2,
    ),
    SupervisorProfile(
        code="D6274",
        name="Dr. Husni Iskandar Pohan, S.Kom, M.T.",
        keywords=(
            "health",
            "kesehatan",
            "medical system",
            "hospital",
            "health informatics",
            "clinical data",
            "decision support",
        ),
        labels=(
            "health_medical",
            "hospital_niche",
            "research",
            "data_ai",
            "software_engineering",
        ),
        flexibility=0.15,
    ),
    SupervisorProfile(
        code="D6184",
        name="Dr. Mochammad Haldi Widianto, S.T., M.T.",
        keywords=(
            "research",
            "priority research",
            "iot",
            "free topic",
            "methodology",
            "data analytics",
            "machine learning",
            "industry research",
        ),
        labels=(
            "research",
            "iot_embedded",
            "data_ai",
            "software_engineering",
            "general_flexible",
        ),
        flexibility=0.6,
    ),
    SupervisorProfile(
        code="D6826",
        name="Karen Etania Saputra, S.Kom., M.Kom.",
        keywords=(
            "general",
            "free topic",
            "software engineering",
            "flexible",
            "web development",
            "application development",
            "internship support",
        ),
        labels=(
            "software_engineering",
            "web_fullstack",
            "internship",
            "general_flexible",
        ),
        flexibility=0.95,
    ),
    SupervisorProfile(
        code="D7055",
        name="Livia Janice Widiapradja, S.Si., Ph.D.",
        keywords=(
            "studi independent",
            "specific independent study",
            "learning plan",
            "research",
            "academic writing",
            "education technology",
        ),
        labels=(
            "independent_study",
            "research",
            "education",
            "general_flexible",
        ),
        flexibility=0.35,
    ),
    SupervisorProfile(
        code="D6469",
        name="Muhammad Maulana Ramadhan, S.Kom., M.Kom.",
        keywords=(
            "game",
            "games",
            "unity",
            "interactive media",
            "general",
            "mobile development",
            "frontend",
            "creative technology",
        ),
        labels=(
            "game_interactive",
            "web_fullstack",
            "software_engineering",
            "general_flexible",
        ),
        flexibility=0.55,
    ),
    SupervisorProfile(
        code="D6836",
        name="Riccosan, S.Kom., M.Kom.",
        keywords=(
            "banking",
            "bank",
            "finance",
            "financial technology",
            "risk management",
            "data analytics",
            "enterprise system",
        ),
        labels=(
            "finance_banking",
            "data_ai",
            "software_engineering",
            "internship",
        ),
        flexibility=0.25,
    ),
    SupervisorProfile(
        code="D6408",
        name="Rissa Rahmania, S.T., M.T.",
        keywords=(
            "apple academy",
            "apple developer academy",
            "swift",
            "ios",
            "mobile app",
            "ui ux",
            "product development",
        ),
        labels=(
            "apple_mobile",
            "web_fullstack",
            "software_engineering",
            "education",
            "general_flexible",
        ),
        flexibility=0.45,
    ),
)


INDEPENDENT_TERMS = (
    "specific independent study",
    "independent study",
    "studi independent",
    "studi indepedent",
)
RESEARCH_TERMS = ("research", "fellowship", "certified research")
INTERNSHIP_TERMS = ("internship", "magang", "certified internship", "company internship")
BINUS_BANDUNG_TERMS = (
    "bina nusantara university school of computer science bandung",
    "binus bandung",
    "school of computer science bandung",
)
BINUS_INTERNAL_TERMS = (
    "bina nusantara university",
    "binus",
    "school of computer science bandung",
    "binus incubator",
    "apple developer academy binus",
    "apple developer academy",
)
NETWORK_TERMS = ("network", "jaringan", "cloud", "infrastructure", "it operation")
ENTRE_TERMS = ("entrepreneurship", "startup", "business", "venture")
DRONE_TERMS = ("drone", "uav")
GOVERNMENT_TERMS = ("government", "pemerintah", "kementerian", "dinas", "public sector")
HEALTH_TERMS = ("health", "kesehatan", "hospital", "medis", "klinik")
HOSPITAL_NICHE_TERMS = (
    "hospital",
    "rumah sakit",
    "siloam",
    "klinik",
    "medical center",
    "rsud",
    "rsia",
)
GAME_TERMS = ("game", "games", "gaming", "unity", "unreal")
BANKING_TERMS = ("bank", "banking", "perbankan", "financial")
APPLE_TERMS = ("apple academy", "apple developer academy", "swift", "ios")
AGIT_TERMS = ("agit", "astra graphia information technology")

WEB_TERMS = (
    "web",
    "frontend",
    "front end",
    "backend",
    "back end",
    "full stack",
    "website",
    "react",
    "javascript",
    "api",
)
SOFTWARE_ENGINEERING_TERMS = (
    "software",
    "application",
    "program",
    "developer",
    "engineering",
    "implementation",
    "system",
)
DATA_AI_TERMS = (
    "data science",
    "machine learning",
    "analytics",
    "analyst",
    "ai",
    "model",
    "visualization",
)
SECURITY_TERMS = ("security", "cyber", "infosec", "pentest")
IOT_TERMS = ("iot", "embedded", "microcontroller", "sensor", "drone", "uav", "robot")
EDUCATION_TERMS = ("academy", "instructor", "sekolah", "learning", "kampus", "universitas")

LABEL_TERMS: dict[str, tuple[str, ...]] = {
    "independent_study": INDEPENDENT_TERMS,
    "internship": INTERNSHIP_TERMS,
    "research": RESEARCH_TERMS,
    "binus_bandung": BINUS_BANDUNG_TERMS,
    "binus_internal_internship": BINUS_INTERNAL_TERMS,
    "network_cloud": NETWORK_TERMS,
    "entrepreneurship": ENTRE_TERMS,
    "iot_embedded": IOT_TERMS,
    "government_public": GOVERNMENT_TERMS,
    "health_medical": HEALTH_TERMS,
    "hospital_niche": HOSPITAL_NICHE_TERMS,
    "game_interactive": GAME_TERMS,
    "finance_banking": BANKING_TERMS,
    "apple_mobile": APPLE_TERMS,
    "web_fullstack": WEB_TERMS,
    "software_engineering": SOFTWARE_ENGINEERING_TERMS,
    "data_ai": DATA_AI_TERMS,
    "cyber_security": SECURITY_TERMS,
    "education": EDUCATION_TERMS,
}

BINUS_INTERNAL_ELIGIBLE_CODES = {profile.code for profile in SUPERVISOR_PROFILES}
BINUS_INTERNAL_PRIORITY_CODES = {
    "D6670",
    "D7187",
    "D6532",
    "D6469",
    "D6408",
    "D5918",
    "D6826",
    "D6184",
    "D2211",
    "D1749",
}
GOVERNMENT_NICHE_SPECIALIST_CODES = {"D6407"}
HOSPITAL_NICHE_SPECIALIST_CODES = {"D6274"}


def _unique_terms(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


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
    # else:
    #     # === HARDCODED FALLBACK (original logic, unchanged) ===
    #     overlap = sorted(student_label_set & profile_labels)
    #     if overlap:
    #         overlap_boost = min(5.0, 1.15 * len(overlap))
    #         boost += overlap_boost
    #         reasons.append(f"Multilabel match: {', '.join(overlap)}")
    #     elif profile.flexibility > 0:
    #         fallback_boost = min(1.0, 0.35 + profile.flexibility * 0.75)
    #         boost += fallback_boost
    #         reasons.append("Flexible fallback matching")

    #     government = "government_public" in student_label_set
    #     hospital_niche = "hospital_niche" in student_label_set
    #     independent = "independent_study" in student_label_set
    #     internship = "internship" in student_label_set
    #     binus_bandung = "binus_bandung" in student_label_set
    #     research = "research" in student_label_set
    #     network = "network_cloud" in student_label_set
    #     entrepreneurship = "entrepreneurship" in student_label_set
    #     drone = "iot_embedded" in student_label_set
    #     health = "health_medical" in student_label_set
    #     game = "game_interactive" in student_label_set
    #     banking = "finance_banking" in student_label_set
    #     apple = "apple_mobile" in student_label_set

    #     if government:
    #         if profile.code in GOVERNMENT_NICHE_SPECIALIST_CODES:
    #             boost += 4.2; reasons.append("Government niche specialist")
    #         else:
    #             boost -= 18.0; reasons.append("Government niche non-specialist penalty")
    #     if hospital_niche:
    #         if profile.code in HOSPITAL_NICHE_SPECIALIST_CODES:
    #             boost += 4.8; reasons.append("Hospital niche specialist")
    #         else:
    #             boost -= 20.0; reasons.append("Hospital niche non-specialist penalty")
    #     if independent and "independent_study" in profile_labels:
    #         boost += 1.2; reasons.append("Independent study alignment")
    #     if internship and "internship" in profile_labels:
    #         boost += 1.0; reasons.append("Internship alignment")
    #     if binus_bandung and "binus_bandung" in profile_labels:
    #         boost += 0.8; reasons.append("BINUS Bandung context")
    #     if research and "research" in profile_labels:
    #         research_boost = 1.2
    #         if profile.code == "D6184":
    #             research_boost += 1.4; reasons.append("Research priority (Haldi)")
    #         boost += research_boost
    #     if (network or entrepreneurship) and {"network_cloud", "entrepreneurship"} & profile_labels:
    #         network_boost = 1.2
    #         if profile.code == "D1749":
    #             network_boost += 0.9; reasons.append("Network/entrepreneurship affinity (Johan)")
    #         elif profile.code == "D2211":
    #             network_boost += 0.6; reasons.append("Network/entrepreneurship support (Abdul Haris)")
    #         boost += network_boost
    #     if (drone or government) and {"iot_embedded", "government_public"} & profile_labels:
    #         boost += 1.6; reasons.append("IoT/government specialization")
    #     if health and "health_medical" in profile_labels:
    #         boost += 2.0; reasons.append("Health specialization")
    #     if game and "game_interactive" in profile_labels:
    #         boost += 2.0; reasons.append("Games specialization")
    #     if banking and "finance_banking" in profile_labels:
    #         boost += 2.0; reasons.append("Banking specialization")
    #     if apple and "apple_mobile" in profile_labels:
    #         boost += 2.2; reasons.append("Apple Academy specialization")
    #     # === END HARDCODED FALLBACK ===

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

    if profile.flexibility > 0:
        boost += 0.25 + min(0.6, profile.flexibility * 0.5)
        reasons.append("Flexible guidance slot")

    return boost, list(dict.fromkeys(reasons))
