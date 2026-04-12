from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.embedding import get_embedding_provider
from app.config import (
    CAPACITY_PRIORITY_CODES,
    COMPANY_GROUP_BONUS,
    LABELED_STUDENT_PRIORITY_WEIGHT,
    SIMILARITY_WEIGHT,
    TARGET_MAX_CAPACITY,
    TARGET_MIN_CAPACITY,
)
from app.rules import (
    SUPERVISOR_PROFILES,
    SupervisorProfile,
    evaluate_rule_boost,
    normalize_text,
    profile_document,
    student_document,
)


@dataclass
class CapacityPlan:
    min_caps: list[int]
    max_caps: list[int]
    relaxed: bool
    note: str | None


@dataclass
class RecommendationItem:
    student: dict[str, Any]
    supervisor: SupervisorProfile
    similarity_score: float
    rule_boost: float
    group_boost: float
    final_score: float
    reasons: list[str]
    company_group_key: str | None


@dataclass
class RecommendationOutput:
    items: list[RecommendationItem]
    counts_by_supervisor: dict[str, int]
    solver: str
    objective_score: float
    capacity_plan: CapacityPlan
    solver_note: str | None
    embedding_backend: str
    embedding_model: str
    embedding_note: str | None
    supervisor_codes: list[str]
    content_similarity_matrix: np.ndarray
    hybrid_score_matrix: np.ndarray


def _rank_supervisor_indices(codes: list[str]) -> list[int]:
    priority = {code: idx for idx, code in enumerate(CAPACITY_PRIORITY_CODES)}
    return sorted(range(len(codes)), key=lambda i: (priority.get(codes[i], 999), i))


def _build_capacity_plan(
    supervisor_codes: list[str],
    student_count: int,
    target_min: int = TARGET_MIN_CAPACITY,
    target_max: int = TARGET_MAX_CAPACITY,
) -> CapacityPlan:
    supervisor_count = len(supervisor_codes)
    min_caps = [target_min for _ in supervisor_codes]
    max_caps = [target_max for _ in supervisor_codes]
    relaxed = False
    notes: list[str] = []

    ranked = _rank_supervisor_indices(supervisor_codes)

    max_total = sum(max_caps)
    if student_count > max_total:
        overflow = student_count - max_total
        relaxed = True
        notes.append(
            f"Jumlah mahasiswa {student_count} > kapasitas maksimal "
            f"{max_total} (aturan {target_min}-{target_max}). "
            f"Sistem menambah slot +1 pada {overflow} dosen prioritas."
        )
        for idx in ranked:
            if overflow <= 0:
                break
            max_caps[idx] += 1
            overflow -= 1
        loop_idx = 0
        while overflow > 0:
            idx = ranked[loop_idx % len(ranked)]
            max_caps[idx] += 1
            overflow -= 1
            loop_idx += 1

    min_total = sum(min_caps)
    if student_count < min_total:
        deficit = min_total - student_count
        relaxed = True
        notes.append(
            f"Jumlah mahasiswa {student_count} < kapasitas minimal "
            f"{min_total} (aturan {target_min}-{target_max}). "
            f"Sistem mengurangi minimum pada {deficit} dosen prioritas."
        )
        for idx in reversed(ranked):
            if deficit <= 0:
                break
            if min_caps[idx] > 0:
                min_caps[idx] -= 1
                deficit -= 1
        loop_idx = 0
        while deficit > 0:
            idx = ranked[loop_idx % len(ranked)]
            if min_caps[idx] > 0:
                min_caps[idx] -= 1
                deficit -= 1
            loop_idx += 1

    # Final safety in edge cases.
    while sum(max_caps) < student_count:
        relaxed = True
        notes.append("Slot maksimum ditambah karena masih belum cukup.")
        idx = ranked[sum(max_caps) % len(ranked)]
        max_caps[idx] += 1

    while sum(min_caps) > student_count:
        relaxed = True
        notes.append("Slot minimum dikurangi karena melebihi jumlah mahasiswa.")
        idx = ranked[sum(min_caps) % len(ranked)]
        if min_caps[idx] > 0:
            min_caps[idx] -= 1

    for idx in range(len(min_caps)):
        if min_caps[idx] > max_caps[idx]:
            min_caps[idx] = max_caps[idx]

    note = "\n".join(dict.fromkeys(notes)) if notes else None
    return CapacityPlan(
        min_caps=min_caps,
        max_caps=max_caps,
        relaxed=relaxed,
        note=note,
    )


def _company_key(partner: str | None) -> str | None:
    key = normalize_text(partner)
    if not key or key in {"-", "none", "na", "n a"}:
        return None
    return key


def _context_token(value: object) -> str:
    text = normalize_text(value)
    return text if text else "__none__"



def _apply_company_group_bonus(
    score_matrix: np.ndarray,
    students: list[dict[str, Any]],
    bonus_value: float = COMPANY_GROUP_BONUS,
) -> tuple[np.ndarray, dict[int, str]]:
    group_bonus = np.zeros_like(score_matrix, dtype=float)
    company_map: dict[str, list[int]] = {}
    student_company_key: dict[int, str] = {}

    for idx, student in enumerate(students):
        key = _company_key(student.get("partner_lecturer"))
        if not key:
            continue
        student_company_key[idx] = key
        company_map.setdefault(key, []).append(idx)

    for key, student_indices in company_map.items():
        if len(student_indices) < 2:
            continue
        topic_diversity = {
            _context_token(students[student_idx].get("position_topic"))
            for student_idx in student_indices
        }
        if len(topic_diversity) > 6:
            continue

        company_scores = score_matrix[student_indices, :]
        mean_scores = company_scores.mean(axis=0)
        ranked = np.argsort(-mean_scores)
        best_supervisor_idx = int(ranked[0])
        if len(ranked) > 1:
            margin = float(mean_scores[best_supervisor_idx] - mean_scores[int(ranked[1])])
            if margin < 0.08:
                continue
        effective_bonus = bonus_value / max(1.0, float(np.log2(len(student_indices) + 1)))
        if effective_bonus <= 0:
            continue
        for student_idx in student_indices:
            group_bonus[student_idx, best_supervisor_idx] += effective_bonus

    return group_bonus, student_company_key


# Greedy solver is used as the primary assignment strategy.
# For the scale of this system (~100-200 students, ~14 supervisors),
# greedy produces near-optimal results without requiring an external
# ILP solver dependency, making it simpler to maintain and explain.
def _solve_assignment(
    score_matrix: np.ndarray,
    min_caps: list[int],
    max_caps: list[int],
) -> tuple[np.ndarray, float]:
    student_count, supervisor_count = score_matrix.shape
    assignment = np.argmax(score_matrix, axis=1)
    counts = np.bincount(assignment, minlength=supervisor_count)
    max_iter = student_count * supervisor_count * 10

    for _ in range(max_iter):
        overfull = [j for j, count in enumerate(counts) if count > max_caps[j]]
        if not overfull:
            break
        best_move: tuple[float, int, int, int] | None = None
        for source in overfull:
            student_indices = np.where(assignment == source)[0]
            for student_idx in student_indices:
                for target in range(supervisor_count):
                    if source == target:
                        continue
                    if counts[target] >= max_caps[target]:
                        continue
                    penalty = score_matrix[student_idx, source] - score_matrix[student_idx, target]
                    if best_move is None or penalty < best_move[0]:
                        best_move = (float(penalty), int(student_idx), int(source), int(target))
        if best_move is None:
            break
        _, student_idx, source, target = best_move
        assignment[student_idx] = target
        counts[source] -= 1
        counts[target] += 1

    for _ in range(max_iter):
        underfull = [j for j, count in enumerate(counts) if count < min_caps[j]]
        if not underfull:
            break
        best_move = None
        for target in underfull:
            donors = [j for j, count in enumerate(counts) if count > min_caps[j] and j != target]
            for source in donors:
                student_indices = np.where(assignment == source)[0]
                for student_idx in student_indices:
                    penalty = score_matrix[student_idx, source] - score_matrix[student_idx, target]
                    if best_move is None or penalty < best_move[0]:
                        best_move = (float(penalty), int(student_idx), int(source), int(target))
        if best_move is None:
            break
        _, student_idx, source, target = best_move
        assignment[student_idx] = target
        counts[source] -= 1
        counts[target] += 1

    if any(counts[j] < min_caps[j] or counts[j] > max_caps[j] for j in range(supervisor_count)):
        raise RuntimeError("Greedy fallback tidak menemukan solusi yang memenuhi kapasitas.")

    objective = float(sum(score_matrix[i, assignment[i]] for i in range(student_count)))
    return assignment, objective


def generate_recommendations(
    students: list[dict[str, Any]],
    supervisor_profiles: tuple[SupervisorProfile, ...] = SUPERVISOR_PROFILES,
) -> RecommendationOutput:
    if not students:
        raise ValueError("Data mahasiswa kosong.")
    if not supervisor_profiles:
        raise ValueError("Data dosen kosong.")

    supervisor_codes = [profile.code for profile in supervisor_profiles]
    supervisor_docs = [profile_document(profile) for profile in supervisor_profiles]
    student_docs = [student_document(student) for student in students]
    embedding_provider = get_embedding_provider()
    embedding_info = embedding_provider.info
    similarity = embedding_provider.similarity_matrix(student_docs, supervisor_docs)
    weighted_similarity = similarity * SIMILARITY_WEIGHT

    student_count = len(students)
    supervisor_count = len(supervisor_profiles)
    rule_boost = np.zeros((student_count, supervisor_count), dtype=float)
    reasons_map: dict[tuple[int, int], list[str]] = {}

    for i, student in enumerate(students):
        for j, profile in enumerate(supervisor_profiles):
            boost, reasons = evaluate_rule_boost(student, profile)
            rule_boost[i, j] = boost
            if reasons:
                reasons_map[(i, j)] = reasons

    score_matrix = weighted_similarity + rule_boost
    group_boost, student_company = _apply_company_group_bonus(score_matrix, students)
    score_matrix = score_matrix + group_boost

    optimization_score_matrix = np.array(score_matrix, copy=True)
    if LABELED_STUDENT_PRIORITY_WEIGHT > 1.0:
        for i, student in enumerate(students):
            current_code = str(student.get("current_supervisor_code") or "").strip()
            if current_code:
                optimization_score_matrix[i, :] *= LABELED_STUDENT_PRIORITY_WEIGHT

    capacity_plan = _build_capacity_plan(
        supervisor_codes=supervisor_codes,
        student_count=student_count,
    )

    solver_note: str | None = None
    solver_name = "greedy"
    assignment, objective = _solve_assignment(
        score_matrix=optimization_score_matrix,
        min_caps=capacity_plan.min_caps,
        max_caps=capacity_plan.max_caps,
    )

    counts_by_supervisor: dict[str, int] = {code: 0 for code in supervisor_codes}
    items: list[RecommendationItem] = []

    for student_idx, supervisor_idx in enumerate(assignment):
        profile = supervisor_profiles[int(supervisor_idx)]
        counts_by_supervisor[profile.code] += 1

        reasons = list(reasons_map.get((student_idx, int(supervisor_idx)), []))
        if group_boost[student_idx, int(supervisor_idx)] > 0:
            reasons.append("Company cohort alignment")
        if not reasons:
            reasons = ["Content-based similarity"]
        else:
            reasons = list(dict.fromkeys(reasons))

        items.append(
            RecommendationItem(
                student=students[student_idx],
                supervisor=profile,
                similarity_score=float(similarity[student_idx, int(supervisor_idx)]),
                rule_boost=float(rule_boost[student_idx, int(supervisor_idx)]),
                group_boost=float(group_boost[student_idx, int(supervisor_idx)]),
                final_score=float(score_matrix[student_idx, int(supervisor_idx)]),
                reasons=reasons,
                company_group_key=student_company.get(student_idx),
            )
        )

    return RecommendationOutput(
        items=items,
        counts_by_supervisor=counts_by_supervisor,
        solver=solver_name,
        objective_score=float(objective),
        capacity_plan=capacity_plan,
        solver_note=solver_note,
        embedding_backend=embedding_info.backend,
        embedding_model=embedding_info.model_name,
        embedding_note=embedding_info.note,
        supervisor_codes=supervisor_codes,
        content_similarity_matrix=similarity,
        hybrid_score_matrix=score_matrix,
    )
