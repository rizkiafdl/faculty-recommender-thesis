from __future__ import annotations

import math
from typing import Any

import numpy as np


def _safe_round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def evaluate_retrieval_metrics(
    score_matrix: np.ndarray,
    students: list[dict[str, Any]],
    supervisor_codes: list[str],
    label_key: str = "current_supervisor_code",
) -> dict[str, Any]:
    if score_matrix.size == 0:
        return {
            "evaluated_queries": 0,
            "mrr": None,
            "hit_at_1": None,
            "hit_at_5": None,
            "ndcg_at_5": None,
            "ndcg_at_10": None,
            "avg_rank": None,
            "avg_similarity_at_1": None,
            "avg_true_similarity": None,
            "precision_at_5": None,
        }

    code_to_idx = {code: idx for idx, code in enumerate(supervisor_codes)}
    ranks: list[int] = []
    top1_scores: list[float] = []
    true_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    hit1: list[float] = []
    hit5: list[float] = []
    ndcg5: list[float] = []
    ndcg10: list[float] = []
    precision5: list[float] = []

    for row_idx, student in enumerate(students):
        label_code = student.get(label_key)
        if not label_code:
            continue
        true_idx = code_to_idx.get(str(label_code))
        if true_idx is None:
            continue
        row = score_matrix[row_idx]
        ranked = np.argsort(-row)
        rank = int(np.where(ranked == true_idx)[0][0]) + 1
        ranks.append(rank)

        top1_scores.append(float(row[ranked[0]]))
        true_scores.append(float(row[true_idx]))

        reciprocal_ranks.append(1.0 / rank)
        hit1.append(1.0 if rank <= 1 else 0.0)
        hit5.append(1.0 if rank <= 5 else 0.0)
        ndcg5.append((1.0 / math.log2(rank + 1.0)) if rank <= 5 else 0.0)
        ndcg10.append((1.0 / math.log2(rank + 1.0)) if rank <= 10 else 0.0)
        precision5.append((1.0 / 5.0) if rank <= 5 else 0.0)

    evaluated = len(ranks)
    if evaluated == 0:
        return {
            "evaluated_queries": 0,
            "mrr": None,
            "hit_at_1": None,
            "hit_at_5": None,
            "ndcg_at_5": None,
            "ndcg_at_10": None,
            "avg_rank": None,
            "avg_similarity_at_1": None,
            "avg_true_similarity": None,
            "precision_at_5": None,
        }

    return {
        "evaluated_queries": evaluated,
        "mrr": _safe_round(float(np.mean(reciprocal_ranks))),
        "hit_at_1": _safe_round(float(np.mean(hit1))),
        "hit_at_5": _safe_round(float(np.mean(hit5))),
        "ndcg_at_5": _safe_round(float(np.mean(ndcg5))),
        "ndcg_at_10": _safe_round(float(np.mean(ndcg10))),
        "avg_rank": _safe_round(float(np.mean(ranks))),
        "avg_similarity_at_1": _safe_round(float(np.mean(top1_scores))),
        "avg_true_similarity": _safe_round(float(np.mean(true_scores))),
        "precision_at_5": _safe_round(float(np.mean(precision5))),
    }


def evaluate_assignment_match_rate(recommendation_items: list[Any]) -> dict[str, Any]:
    total = 0
    matched = 0
    for item in recommendation_items:
        current_code = str(item.student.get("current_supervisor_code") or "").strip()
        if not current_code:
            continue
        total += 1
        if current_code == item.supervisor.code:
            matched += 1
    return {
        "evaluated_students": total,
        "matched_students": matched,
        "match_rate": _safe_round((matched / total) if total else 0.0),
    }


def build_evaluation_payload(
    content_scores: np.ndarray,
    hybrid_scores: np.ndarray,
    students: list[dict[str, Any]],
    supervisor_codes: list[str],
    recommendation_items: list[Any],
) -> dict[str, Any]:
    return {
        "label_source": "current_supervisor_code",
        "content_based": evaluate_retrieval_metrics(
            score_matrix=content_scores,
            students=students,
            supervisor_codes=supervisor_codes,
        ),
        "hybrid_score": evaluate_retrieval_metrics(
            score_matrix=hybrid_scores,
            students=students,
            supervisor_codes=supervisor_codes,
        ),
        "assignment_match": evaluate_assignment_match_rate(recommendation_items),
    }

