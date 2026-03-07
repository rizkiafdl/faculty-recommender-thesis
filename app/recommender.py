from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.embedding import get_embedding_provider
from app.config import (
    CAPACITY_PRIORITY_CODES,
    COMPANY_GROUP_BONUS,
    CONTEXT_PRIOR_MIN_SUPPORT,
    CONTEXT_PRIOR_WEIGHT,
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

try:
    import pulp

    HAS_PULP = True
except Exception:  # pragma: no cover
    HAS_PULP = False


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


def _context_views(student: dict[str, Any]) -> list[tuple[str, str]]:
    track = _context_token(student.get("track"))
    partner = _context_token(student.get("partner_lecturer"))
    topic = _context_token(student.get("position_topic"))
    schema = _context_token(student.get("work_schema"))
    return [
        ("exact", f"{track}|{partner}|{topic}|{schema}"),
        ("partner_topic", f"{track}|{partner}|{topic}"),
        ("partner", f"{track}|{partner}"),
        ("topic", f"{track}|{topic}"),
        ("track", track),
    ]


def _build_context_prior_boost(
    students: list[dict[str, Any]],
    supervisor_codes: list[str],
    base_weight: float = CONTEXT_PRIOR_WEIGHT,
    min_support: int = CONTEXT_PRIOR_MIN_SUPPORT,
) -> tuple[np.ndarray, dict[tuple[int, int], list[str]]]:
    matrix = np.zeros((len(students), len(supervisor_codes)), dtype=float)
    reasons: dict[tuple[int, int], list[str]] = {}
    if not students or not supervisor_codes or base_weight <= 0:
        return matrix, reasons

    code_to_idx = {code: idx for idx, code in enumerate(supervisor_codes)}
    level_weights = {
        "exact": 1.7 * base_weight,
        "partner_topic": 1.3 * base_weight,
        "partner": 0.9 * base_weight,
        "topic": 0.7 * base_weight,
        "track": 0.5 * base_weight,
    }
    level_min_support = {
        "exact": max(3, min_support + 1),
        "partner_topic": max(3, min_support),
        "partner": max(2, min_support - 1),
        "topic": max(2, min_support - 1),
        "track": max(2, min_support - 1),
    }

    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    totals: dict[str, dict[str, int]] = defaultdict(dict)

    for student in students:
        code = str(student.get("current_supervisor_code") or "").strip()
        if code not in code_to_idx:
            continue
        for level, key in _context_views(student):
            counts[level][key][code] += 1

    for level, group_map in counts.items():
        for key, counter in group_map.items():
            totals[level][key] = int(sum(counter.values()))

    for student_idx, student in enumerate(students):
        own_code = str(student.get("current_supervisor_code") or "").strip()
        for level, key in _context_views(student):
            group_counter = counts.get(level, {}).get(key)
            group_total = totals.get(level, {}).get(key, 0)
            if not group_counter or group_total <= 0:
                continue

            adjusted_total = group_total
            if own_code in code_to_idx and group_counter.get(own_code, 0) > 0:
                adjusted_total -= 1
            if adjusted_total < level_min_support[level]:
                continue

            support_scale = min(1.0, adjusted_total / float(level_min_support[level] + 3))
            weight = level_weights[level]
            for code, count in group_counter.items():
                if code not in code_to_idx:
                    continue
                adjusted_count = count
                if own_code == code and adjusted_count > 0:
                    adjusted_count -= 1
                if adjusted_count <= 0:
                    continue

                ratio = adjusted_count / adjusted_total
                boost = weight * ratio * support_scale
                col = code_to_idx[code]
                matrix[student_idx, col] += boost

                if ratio >= 0.33:
                    reasons.setdefault((student_idx, col), []).append(
                        f"Historical context prior ({level}: {adjusted_count}/{adjusted_total})"
                    )

    return matrix, reasons


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


def _solve_assignment_with_pulp(
    score_matrix: np.ndarray,
    min_caps: list[int],
    max_caps: list[int],
) -> tuple[np.ndarray, float]:
    if not HAS_PULP:
        raise RuntimeError("PuLP is not installed")

    student_count, supervisor_count = score_matrix.shape
    problem = pulp.LpProblem("faculty_supervisor_assignment", pulp.LpMaximize)
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i in range(student_count)
        for j in range(supervisor_count)
    }

    problem += pulp.lpSum(score_matrix[i, j] * x[(i, j)] for i in range(student_count) for j in range(supervisor_count))

    for i in range(student_count):
        problem += pulp.lpSum(x[(i, j)] for j in range(supervisor_count)) == 1

    for j in range(supervisor_count):
        problem += pulp.lpSum(x[(i, j)] for i in range(student_count)) >= min_caps[j]
        problem += pulp.lpSum(x[(i, j)] for i in range(student_count)) <= max_caps[j]

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f"PuLP solver gagal: {pulp.LpStatus[status]}")

    assignment = np.full(student_count, -1, dtype=int)
    for i in range(student_count):
        for j in range(supervisor_count):
            value = pulp.value(x[(i, j)])
            if value is not None and value > 0.5:
                assignment[i] = j
                break
        if assignment[i] < 0:
            raise RuntimeError(f"Mahasiswa index {i} tidak ter-assign")

    objective = float(pulp.value(problem.objective))
    return assignment, objective


def _solve_assignment_greedy(
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

    context_boost, context_reasons = _build_context_prior_boost(
        students=students,
        supervisor_codes=supervisor_codes,
    )

    score_matrix = weighted_similarity + rule_boost + context_boost
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
    solver_name = "pulp" if HAS_PULP else "greedy"
    if HAS_PULP:
        try:
            assignment, objective = _solve_assignment_with_pulp(
                score_matrix=optimization_score_matrix,
                min_caps=capacity_plan.min_caps,
                max_caps=capacity_plan.max_caps,
            )
        except Exception as exc:
            solver_name = "greedy"
            solver_note = f"PuLP gagal, fallback ke greedy: {exc}"
            assignment, objective = _solve_assignment_greedy(
                score_matrix=optimization_score_matrix,
                min_caps=capacity_plan.min_caps,
                max_caps=capacity_plan.max_caps,
            )
    else:
        solver_note = "PuLP tidak tersedia, menggunakan greedy fallback."
        assignment, objective = _solve_assignment_greedy(
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
        reasons.extend(context_reasons.get((student_idx, int(supervisor_idx)), []))
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
