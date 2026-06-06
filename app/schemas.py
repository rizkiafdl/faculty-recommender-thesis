from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SupervisorProfile:
    code: str
    name: str
    keywords: tuple[str, ...]
    labels: tuple[str, ...] = ()

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
    group_boost: float
    final_score: float
    rule_matches: list[str]
    company_group_key: str | None


@dataclass
class RunOverrides:
    embedding_model: str
    embedding_task: str
    enable_group_bonus: bool
    enable_extra_docs: bool
    capacity_priority_codes: list[str]
    target_min_capacity: int
    target_max_capacity: int


@dataclass
class RecommendationOutput:
    items: list[RecommendationItem]
    counts_by_supervisor: dict[str, int]
    solver_name: str
    objective_score: float
    capacity_plan: CapacityPlan
    solver_note: str | None
    embedding_backend: str
    embedding_model: str
    embedding_note: str | None
    supervisor_codes: list[str]
    content_similarity_matrix: np.ndarray
    hybrid_score_matrix: np.ndarray

@dataclass
class EmbeddingInfo:
    backend: str
    model_name: str
    note: str | None = None