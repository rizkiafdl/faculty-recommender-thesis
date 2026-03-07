from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np

from app.config import BENCHMARK_MODEL_NAMES, EMBEDDING_DEVICE
from app.evaluation import evaluate_retrieval_metrics
from app.rules import SupervisorProfile, profile_document, student_document

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAS_SENTENCE_TRANSFORMERS = False


def _model_list(model_names: list[str] | None) -> list[str]:
    if model_names:
        selected = [value.strip() for value in model_names if value and value.strip()]
        if selected:
            return selected
    return list(BENCHMARK_MODEL_NAMES)


def _leaderboard(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    successful = []
    for result in results:
        metrics = result.get("metrics")
        if result.get("status") != "ok" or not isinstance(metrics, dict):
            continue
        mrr = metrics.get("mrr")
        if mrr is None:
            continue
        successful.append(result)

    successful.sort(key=lambda item: float(item["metrics"]["mrr"]), reverse=True)
    board: list[dict[str, Any]] = []
    for idx, item in enumerate(successful, start=1):
        metrics = item["metrics"]
        board.append(
            {
                "rank": idx,
                "model_name": item["model_name"],
                "mrr": metrics.get("mrr"),
                "hit_at_1": metrics.get("hit_at_1"),
                "hit_at_5": metrics.get("hit_at_5"),
                "ndcg_at_5": metrics.get("ndcg_at_5"),
                "ndcg_at_10": metrics.get("ndcg_at_10"),
                "avg_rank": metrics.get("avg_rank"),
                "avg_true_similarity": metrics.get("avg_true_similarity"),
                "duration_seconds": item.get("duration_seconds"),
            }
        )
    winner = board[0]["model_name"] if board else None
    return board, winner


def benchmark_transformer_models(
    students: list[dict[str, Any]],
    supervisor_profiles: tuple[SupervisorProfile, ...],
    model_names: list[str] | None = None,
    device: str | None = None,
    label_key: str = "current_supervisor_code",
) -> dict[str, Any]:
    if not students:
        raise ValueError("Data mahasiswa kosong.")
    if not supervisor_profiles:
        raise ValueError("Data dosen kosong.")

    selected_models = _model_list(model_names)
    if not selected_models:
        raise ValueError("Daftar model benchmark kosong.")

    supervisor_codes = [profile.code for profile in supervisor_profiles]
    student_docs = [student_document(student) for student in students]
    supervisor_docs = [profile_document(profile) for profile in supervisor_profiles]
    resolved_device = (device or EMBEDDING_DEVICE).strip()

    if not HAS_SENTENCE_TRANSFORMERS:
        failed = [
            {
                "model_name": model_name,
                "status": "error",
                "error": "sentence-transformers tidak tersedia.",
                "duration_seconds": 0.0,
                "metrics": None,
            }
            for model_name in selected_models
        ]
        leaderboard, winner = _leaderboard(failed)
        return {
            "label_source": label_key,
            "device": resolved_device,
            "total_models": len(selected_models),
            "successful_models": 0,
            "failed_models": len(selected_models),
            "results": failed,
            "leaderboard": leaderboard,
            "best_model_by_mrr": winner,
        }

    results: list[dict[str, Any]] = []
    for model_name in selected_models:
        started = time.perf_counter()
        try:
            model = SentenceTransformer(model_name_or_path=model_name, device=resolved_device)
            source_vectors = model.encode(
                student_docs,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            target_vectors = model.encode(
                supervisor_docs,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            score_matrix = np.matmul(source_vectors, target_vectors.T)
            metrics = evaluate_retrieval_metrics(
                score_matrix=score_matrix,
                students=students,
                supervisor_codes=supervisor_codes,
                label_key=label_key,
            )
            elapsed = round(time.perf_counter() - started, 3)
            results.append(
                {
                    "model_name": model_name,
                    "status": "ok",
                    "error": None,
                    "duration_seconds": elapsed,
                    "metrics": metrics,
                }
            )
        except Exception as exc:  # pragma: no cover
            elapsed = round(time.perf_counter() - started, 3)
            results.append(
                {
                    "model_name": model_name,
                    "status": "error",
                    "error": str(exc)[:1000],
                    "duration_seconds": elapsed,
                    "metrics": None,
                }
            )
        finally:
            try:
                del model
            except Exception:
                pass
            gc.collect()

    leaderboard, winner = _leaderboard(results)
    success_count = sum(1 for item in results if item["status"] == "ok")
    return {
        "label_source": label_key,
        "device": resolved_device,
        "total_models": len(selected_models),
        "successful_models": success_count,
        "failed_models": len(selected_models) - success_count,
        "results": results,
        "leaderboard": leaderboard,
        "best_model_by_mrr": winner,
    }
