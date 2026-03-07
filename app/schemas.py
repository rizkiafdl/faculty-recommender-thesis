from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ImportResponse(BaseModel):
    inserted: int
    updated: int
    total: int
    source: str


class RunResponse(BaseModel):
    run_id: int
    created_at: datetime
    total_students: int
    total_supervisors: int
    target_min_capacity: int
    target_max_capacity: int
    capacity_relaxed: bool
    capacity_note: str | None = None
    solver_name: str | None = None
    solver_note: str | None = None
    embedding_backend: str | None = None
    embedding_model: str | None = None
    objective_score: float | None = None


class RecommendationListResponse(BaseModel):
    run_id: int
    data: list[dict[str, Any]]


class SummaryResponse(BaseModel):
    run_id: int
    summary: list[dict[str, Any]]


class EvaluationResponse(BaseModel):
    run_id: int
    evaluation: dict[str, Any]
