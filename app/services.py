from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from werkzeug.security import check_password_hash, generate_password_hash

from app import queries
from datasets.map_loader import load_supervisor_extra_docs
from app.config import (
    DEFAULT_SHEET_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_TASK,
    ENABLE_EXTRA_DOCS,
    ENABLE_GROUP_BONUS,
    ENABLE_RULE_BOOST,
    SIMILARITY_WEIGHT,
    TARGET_MAX_CAPACITY,
    TARGET_MIN_CAPACITY,
)
from app.database import Base, engine
from app.evaluation import build_evaluation_payload
from app.excel_io import read_students_from_excel_bytes, read_students_from_excel_path
from app.models import (
    AppUser,
    LabelDescription,
    Recommendation,
    RecommendationRun,
    Student,
    Supervisor,
    SupervisorLabelAffinity,
)
from app.recommender import RunOverrides, generate_recommendations
from app.rules import normalize_text, student_document, student_labels
from app.schemas import SupervisorProfile
from datasets.seed_dataset.label_descriptions import DEFAULT_LABEL_DESCRIPTIONS
from datasets.seed_dataset.seeder import seed_affinity_matrix
from datasets.seed_dataset.stopwords import PROFILE_TOKEN_STOPWORDS


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_recommendation_run_schema()


def _ensure_recommendation_run_schema() -> None:
    inspector = inspect(engine)
    if "recommendation_runs" not in inspector.get_table_names():
        return
    existing_columns = {column["name"] for column in inspector.get_columns("recommendation_runs")}
    required_ddl = {
        "capacity_bounds_json": "ALTER TABLE recommendation_runs ADD COLUMN capacity_bounds_json TEXT",
        "solver_name": "ALTER TABLE recommendation_runs ADD COLUMN solver_name VARCHAR(32)",
        "solver_note": "ALTER TABLE recommendation_runs ADD COLUMN solver_note TEXT",
        "embedding_backend": "ALTER TABLE recommendation_runs ADD COLUMN embedding_backend VARCHAR(64)",
        "embedding_model": "ALTER TABLE recommendation_runs ADD COLUMN embedding_model VARCHAR(255)",
        "evaluation_json": "ALTER TABLE recommendation_runs ADD COLUMN evaluation_json TEXT",        "pipeline_config_json": "ALTER TABLE recommendation_runs ADD COLUMN pipeline_config_json TEXT",
    }
    for column_name, ddl in required_ddl.items():
        if column_name in existing_columns:
            continue
        with engine.begin() as connection:
            connection.execute(text(ddl))


def _normalize_supervisor_code(value: str) -> str:
    return str(value or "").strip().upper()


def _normalize_username(value: str) -> str:
    return normalize_text(value).replace(" ", "")


def register_user(
    session: Session,
    username: str,
    password: str,
    full_name: str | None = None,
) -> AppUser:
    normalized_username = _normalize_username(username)
    if len(normalized_username) < 3:
        raise ValueError("Username minimal 3 karakter.")
    if len(password or "") < 6:
        raise ValueError("Password minimal 6 karakter.")

    if queries.get_user_by_username(session, normalized_username) is not None:
        raise ValueError("Username sudah digunakan.")

    user = AppUser(
        username=normalized_username,
        full_name=(full_name or "").strip() or None,
        password_hash=generate_password_hash(password),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def authenticate_user(session: Session, username: str, password: str) -> AppUser | None:
    normalized_username = _normalize_username(username)
    if not normalized_username:
        return None
    user = queries.get_user_by_username(session, normalized_username)
    if user is None:
        return None
    if not check_password_hash(user.password_hash, password):
        return None
    return user


def get_user_by_id(session: Session, user_id: int) -> AppUser | None:
    return queries.get_user_by_id(session, user_id)


def list_supervisor_profiles_for_web(session: Session) -> list[dict[str, Any]]:
    supervisors = queries.get_active_supervisors_ordered(session)
    return [
        {
            "id": supervisor.id,
            "code": supervisor.code,
            "name": supervisor.name,
            "profile_keywords": supervisor.profile_keywords or "",
            "is_active": supervisor.is_active,
        }
        for supervisor in supervisors
    ]


def update_supervisor_keywords(session: Session, supervisor_code: str, profile_keywords: str) -> None:
    code = str(supervisor_code or "").strip()
    if not code:
        raise ValueError("Kode dosen wajib diisi.")
    supervisor = queries.get_active_supervisor_by_code(session, code)
    if supervisor is None:
        raise ValueError(f"Dosen {code} tidak ditemukan.")
    supervisor.profile_keywords = (profile_keywords or "").strip()
    session.commit()


def add_or_update_supervisor(
    session: Session,
    supervisor_code: str,
    supervisor_name: str,
    profile_keywords: str = "",
) -> Supervisor:
    code = _normalize_supervisor_code(supervisor_code)
    name = str(supervisor_name or "").strip()
    if not code:
        raise ValueError("Kode dosen wajib diisi.")
    if len(code) < 3:
        raise ValueError("Kode dosen minimal 3 karakter.")
    if not name:
        raise ValueError("Nama dosen wajib diisi.")

    conflict = queries.get_supervisor_by_name_not_code(session, name, code)
    if conflict is not None:
        raise ValueError(f"Nama dosen sudah dipakai oleh kode {conflict.code}.")

    existing = queries.get_supervisor_by_code(session, code)
    if existing is None:
        supervisor = Supervisor(
            code=code,
            name=name,
            profile_keywords=(profile_keywords or "").strip(),
            is_active=True,
        )
        session.add(supervisor)
        session.commit()
        session.refresh(supervisor)
        return supervisor

    existing.name = name
    if (profile_keywords or "").strip():
        existing.profile_keywords = (profile_keywords or "").strip()
    existing.is_active = True
    session.commit()
    session.refresh(existing)
    return existing


def export_supervisor_configuration_excel(session: Session) -> tuple[bytes, str]:
    supervisors = list_supervisor_profiles_for_web(session)
    tracks = [{"track": track} for track in queries.get_distinct_student_tracks(session)]

    supervisor_df = pd.DataFrame(supervisors)
    tracks_df = pd.DataFrame(tracks)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        supervisor_df.to_excel(writer, index=False, sheet_name="supervisor_config")
        tracks_df.to_excel(writer, index=False, sheet_name="track_reference")
    output.seek(0)
    return output.read(), "supervisor_config_export.xlsx"


def load_label_descriptions(session: Session) -> list[dict]:
    rows = queries.get_all_label_descriptions(session)

    if not rows:
        return [
            {"label_name": n, "description": d, "threshold": t, "is_niche": nf}
            for n, d, t, nf in DEFAULT_LABEL_DESCRIPTIONS
        ]

    return [
        {
            "label_name": row.label_name,
            "description": row.description,
            "threshold": row.threshold,
            "is_niche": row.is_niche,
        }
        for row in rows
    ]


def save_label_description(
    session: Session,
    label_name: str,
    description: str,
    threshold: float,
    is_niche: bool,
) -> None:
    row = queries.get_label_description_by_name(session, label_name)
    if row is None:
        session.add(LabelDescription(
            label_name=label_name,
            description=description,
            threshold=threshold,
            is_niche=is_niche,
        ))
    else:
        row.description = description
        row.threshold = threshold
        row.is_niche = is_niche
    session.commit()


def reset_label_description(session: Session, label_name: str) -> None:
    for n, d, t, nf in DEFAULT_LABEL_DESCRIPTIONS:
        if n == label_name:
            save_label_description(session, label_name=n, description=d, threshold=t, is_niche=nf)
            return
    raise ValueError(f"Label '{label_name}' tidak ada di default seed.")


def list_students_for_preview(session: Session) -> list[dict]:
    return [{"student_id": r[0], "name": r[1]} for r in queries.get_students_preview(session)]


def load_affinity_lookup(session: Session) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    if queries.count_affinity_rows(session) == 0:
        return {}, {}

    affinity_index: dict[tuple[str, str], float] = {}
    niche_defaults: dict[str, float] = {}

    for affinity_row, supervisor_code in queries.get_affinity_with_supervisor_codes(session):
        if affinity_row.is_niche_penalty and supervisor_code is None:
            niche_defaults[affinity_row.label_name] = affinity_row.boost_value
        elif supervisor_code:
            affinity_index[(supervisor_code, affinity_row.label_name)] = affinity_row.boost_value

    return affinity_index, niche_defaults


def load_affinity_matrix_for_web(
    session: Session,
    supervisors: list[dict],
) -> dict[str, dict[str, float]]:
    affinity_index, niche_defaults = load_affinity_lookup(session)
    label_names = queries.get_label_names_ordered(session)

    grid: dict[str, dict[str, float]] = {}
    for label in label_names:
        grid[label] = {}
        for sup in supervisors:
            code = sup["code"]
            key = (code, label)
            if key in affinity_index:
                grid[label][code] = affinity_index[key]
            elif label in niche_defaults:
                grid[label][code] = niche_defaults[label]
            else:
                grid[label][code] = 0.0
    return grid


def save_affinity_cells(session: Session, cells: list[dict]) -> None:
    supervisor_id_by_code = queries.get_supervisor_code_id_map(session)
    for cell in cells:
        code = str(cell.get("supervisor_code") or "").strip()
        label_name = str(cell.get("label_name") or "").strip()
        boost_value = float(cell.get("boost_value", 0.0))
        if not code or not label_name:
            continue
        supervisor_id = supervisor_id_by_code.get(code)
        if supervisor_id is None:
            continue
        existing = queries.get_affinity_by_supervisor_and_label(session, supervisor_id, label_name)
        if existing:
            existing.boost_value = boost_value
        else:
            session.add(SupervisorLabelAffinity(
                supervisor_id=supervisor_id,
                label_name=label_name,
                boost_value=boost_value,
                is_niche_penalty=False,
            ))
    session.commit()


def reset_affinity_matrix(session: Session) -> None:
    queries.delete_all_affinities(session)
    session.commit()
    seed_affinity_matrix(session)


def _upsert_students(session: Session, rows: list[dict[str, Any]]) -> dict[str, int]:
    if not rows:
        return {"inserted": 0, "updated": 0, "total": 0}

    student_ids = [row["student_id"] for row in rows]
    existing_students = queries.get_students_by_student_ids(session, student_ids)

    inserted = 0
    updated = 0
    for row in rows:
        current = existing_students.get(row["student_id"])
        if current is None:
            session.add(Student(**row))
            inserted += 1
            continue

        for key, value in row.items():
            setattr(current, key, value)
        updated += 1

    session.commit()
    return {"inserted": inserted, "updated": updated, "total": len(rows)}


def import_students_from_path(
    session: Session,
    excel_path: str | Path,
    sheet_name: str = DEFAULT_SHEET_NAME,
) -> dict[str, Any]:
    rows = read_students_from_excel_path(path=excel_path, sheet_name=sheet_name)
    stats = _upsert_students(session=session, rows=rows)
    stats["source"] = str(excel_path)
    return stats


def import_students_from_bytes(
    session: Session,
    content: bytes,
    filename: str,
    sheet_name: str = DEFAULT_SHEET_NAME,
) -> dict[str, Any]:
    rows = read_students_from_excel_bytes(content=content, sheet_name=sheet_name)
    stats = _upsert_students(session=session, rows=rows)
    stats["source"] = filename
    return stats


def _supervisor_profiles_from_db(session: Session) -> list[SupervisorProfile]:
    supervisors = queries.get_active_supervisors_ordered(session)
    profiles: list[SupervisorProfile] = []
    for supervisor in supervisors:
        db_keywords = tuple(
            value.strip()
            for value in (supervisor.profile_keywords or "").split(",")
            if value.strip()
        )
        profiles.append(
            SupervisorProfile(
                code=supervisor.code,
                name=supervisor.name,
                keywords=db_keywords,
                labels=("general_flexible",),
            )
        )

    # Adaptive multilabel enrichment from historical mapping.
    if not profiles:
        return profiles

    supervisor_codes = {profile.code for profile in profiles}
    history_students = queries.get_students_by_supervisor_codes(session, supervisor_codes)

    grouped_by_code: dict[str, list[dict[str, Any]]] = {}
    for student in history_students:
        code = str(student.current_supervisor_code or "").strip()
        if not code:
            continue
        grouped_by_code.setdefault(code, []).append(
            {
                "track": student.track,
                "partner_lecturer": student.partner_lecturer,
                "position_topic": student.position_topic,
                "work_schema": student.work_schema,
            }
        )

    adaptive_profiles: list[SupervisorProfile] = []
    for profile in profiles:
        records = grouped_by_code.get(profile.code, [])
        if not records:
            adaptive_profiles.append(profile)
            continue

        token_counter: Counter[str] = Counter()
        label_counter: Counter[str] = Counter()
        for record in records:
            text_val = student_document(record)
            label_counter.update(student_labels(record))
            for token in text_val.split():
                if len(token) < 3:
                    continue
                if token.isdigit() or token in PROFILE_TOKEN_STOPWORDS:
                    continue
                token_counter[token] += 1

        learned_terms = [term for term, freq in token_counter.most_common(16) if freq >= 2]
        learned_labels = [label for label, freq in label_counter.most_common(8) if freq >= 2]

        merged_keywords = tuple(dict.fromkeys([*profile.keywords, *learned_terms]))
        merged_labels = tuple(dict.fromkeys([*profile.labels, *learned_labels]))

        adaptive_profiles.append(
            SupervisorProfile(
                code=profile.code,
                name=profile.name,
                keywords=merged_keywords,
                labels=merged_labels,
            )
        )

    return adaptive_profiles


def _students_for_recommender(session: Session) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for student in queries.get_all_students(session):
        items.append(
            {
                "student_id": student.student_id,
                "binusian_id": student.binusian_id,
                "name": student.name,
                "email": student.email,
                "phone": student.phone,
                "track": student.track,
                "gpa": student.gpa,
                "total_sks": student.total_sks,
                "partner_lecturer": student.partner_lecturer,
                "position_topic": student.position_topic,
                "duration": student.duration,
                "job_start_date": student.job_start_date,
                "job_end_date": student.job_end_date,
                "work_schema": student.work_schema,
                "enrollment_status": student.enrollment_status,
                "current_supervisor_code": student.current_supervisor_code,
                "current_supervisor_name": student.current_supervisor_name,
            }
        )
    return items


def generate_and_store_recommendations(
    session: Session,
    input_source: str = "manual-run",
    overrides: RunOverrides | None = None,
) -> RecommendationRun:
    if overrides is None:
        overrides = RunOverrides(
            embedding_model=EMBEDDING_MODEL_NAME,
            embedding_task=EMBEDDING_TASK,
            enable_rule_boost=ENABLE_RULE_BOOST,
            enable_group_bonus=ENABLE_GROUP_BONUS,
            enable_extra_docs=ENABLE_EXTRA_DOCS,
        )
    student_payload = _students_for_recommender(session)
    if not student_payload:
        raise ValueError("Belum ada data mahasiswa. Import Excel terlebih dahulu.")

    profiles = _supervisor_profiles_from_db(session)
    if not profiles:
        raise ValueError("Belum ada data dosen aktif.")

    label_descriptions = load_label_descriptions(session)
    affinity_index, niche_defaults = load_affinity_lookup(session)

    try:
        extra_supervisor_docs = load_supervisor_extra_docs()
    except Exception:
        extra_supervisor_docs = {}

    output = generate_recommendations(
        students=student_payload,
        supervisor_profiles=tuple(profiles),
        label_descriptions=label_descriptions,
        affinity_index=affinity_index,
        niche_defaults=niche_defaults,
        extra_supervisor_docs=extra_supervisor_docs,
        overrides=overrides,
    )
    evaluation_payload = build_evaluation_payload(
        content_scores=output.content_similarity_matrix,
        hybrid_scores=output.hybrid_score_matrix,
        students=student_payload,
        supervisor_codes=output.supervisor_codes,
        recommendation_items=output.items,
    )

    supervisor_by_code = queries.get_active_supervisor_map(session)
    student_by_student_id = queries.get_all_students_map(session)

    capacity_bounds = {}
    for idx, profile in enumerate(profiles):
        capacity_bounds[profile.code] = {
            "min": output.capacity_plan.min_caps[idx],
            "max": output.capacity_plan.max_caps[idx],
            "count": output.counts_by_supervisor.get(profile.code, 0),
        }

    pipeline_config = {
        "rule_boost": overrides.enable_rule_boost,
        "group_bonus": overrides.enable_group_bonus,
        "extra_docs": overrides.enable_extra_docs,
        "similarity_weight": SIMILARITY_WEIGHT,
        "embedding_model": overrides.embedding_model,
        "embedding_task": overrides.embedding_task,
    }

    note_parts: list[str] = []
    if output.solver_note:
        note_parts.append(output.solver_note)
    if output.embedding_note:
        note_parts.append(output.embedding_note)

    run = RecommendationRun(
        input_source=input_source,
        total_students=len(student_payload),
        total_supervisors=len(profiles),
        target_min_capacity=TARGET_MIN_CAPACITY,
        target_max_capacity=TARGET_MAX_CAPACITY,
        capacity_relaxed=output.capacity_plan.relaxed,
        capacity_note=output.capacity_plan.note,
        capacity_bounds_json=json.dumps(capacity_bounds),
        solver_name=output.solver_name,
        solver_note="\n".join(note_parts) if note_parts else None,
        embedding_backend=output.embedding_backend,
        embedding_model=output.embedding_model,
        evaluation_json=json.dumps(evaluation_payload),
        pipeline_config_json=json.dumps(pipeline_config),
        objective_score=output.objective_score,
    )
    session.add(run)
    session.flush()

    for item in output.items:
        student = student_by_student_id[item.student["student_id"]]
        supervisor = supervisor_by_code[item.supervisor.code]
        session.add(
            Recommendation(
                run_id=run.id,
                student_id=student.id,
                supervisor_id=supervisor.id,
                similarity_score=item.similarity_score,
                rule_boost=item.rule_boost,
                group_boost=item.group_boost,
                final_score=item.final_score,
                rule_matches="; ".join(item.rule_matches),
                company_group_key=item.company_group_key,
            )
        )

    session.commit()
    session.refresh(run)
    return run


def get_latest_run(session: Session) -> RecommendationRun | None:
    return queries.get_latest_run(session)


def list_runs(session: Session, limit: int | None = None) -> list[RecommendationRun]:
    return queries.list_runs(session, limit)


def get_run_by_id(session: Session, run_id: int) -> RecommendationRun | None:
    return queries.get_run_by_id(session, run_id)


def _resolve_run_id(session: Session, run_id: int | None) -> int:
    if run_id is not None:
        return run_id
    latest = queries.get_latest_run(session)
    if latest is None:
        raise ValueError("Belum ada hasil rekomendasi.")
    return latest.id


def list_recommendations(session: Session, run_id: int | None = None) -> tuple[RecommendationRun, list[dict[str, Any]]]:
    resolved_run_id = _resolve_run_id(session, run_id)
    run = queries.get_run_by_id(session, resolved_run_id)
    if run is None:
        raise ValueError(f"Run {resolved_run_id} tidak ditemukan.")

    data: list[dict[str, Any]] = []
    for recommendation, student, supervisor in queries.get_recommendations_with_entities(session, resolved_run_id):
        data.append(
            {
                "run_id": recommendation.run_id,
                "student_id": student.student_id,
                "student_name": student.name,
                "track": student.track,
                "gpa": student.gpa,
                "partner_lecturer": student.partner_lecturer,
                "position_topic": student.position_topic,
                "recommended_supervisor_code": supervisor.code,
                "recommended_supervisor_name": supervisor.name,
                "similarity_score": recommendation.similarity_score,
                "rule_boost": recommendation.rule_boost,
                "group_boost": recommendation.group_boost,
                "final_score": recommendation.final_score,
                "rule_matches": recommendation.rule_matches,
                "company_group_key": recommendation.company_group_key,
                "current_supervisor_code": student.current_supervisor_code,
                "current_supervisor_name": student.current_supervisor_name,
            }
        )
    return run, data


def summary_by_supervisor(session: Session, run_id: int | None = None) -> tuple[RecommendationRun, list[dict[str, Any]]]:
    resolved_run_id = _resolve_run_id(session, run_id)
    run = queries.get_run_by_id(session, resolved_run_id)
    if run is None:
        raise ValueError(f"Run {resolved_run_id} tidak ditemukan.")

    bounds = json.loads(run.capacity_bounds_json or "{}")

    summary: list[dict[str, Any]] = []
    for code, name, count in queries.get_supervisor_recommendation_counts(session, resolved_run_id):
        limit = bounds.get(code, {})
        min_cap = limit.get("min", TARGET_MIN_CAPACITY)
        max_cap = limit.get("max", TARGET_MAX_CAPACITY)
        within = min_cap <= int(count) <= max_cap
        summary.append(
            {
                "supervisor_code": code,
                "supervisor_name": name,
                "assigned_students": int(count),
                "min_capacity": int(min_cap),
                "max_capacity": int(max_cap),
                "within_capacity": within,
            }
        )
    return run, summary


def evaluation_by_run(session: Session, run_id: int | None = None) -> tuple[RecommendationRun, dict[str, Any]]:
    resolved_run_id = _resolve_run_id(session, run_id)
    run = queries.get_run_by_id(session, resolved_run_id)
    if run is None:
        raise ValueError(f"Run {resolved_run_id} tidak ditemukan.")
    payload = json.loads(run.evaluation_json or "{}")
    if not isinstance(payload, dict):
        payload = {}
    return run, payload


def export_recommendations_excel(
    session: Session,
    run_id: int | None = None,
) -> tuple[bytes, str]:
    run, recommendations = list_recommendations(session=session, run_id=run_id)
    _, summary = summary_by_supervisor(session=session, run_id=run.id)
    _, evaluation = evaluation_by_run(session=session, run_id=run.id)

    recommendations_df = pd.DataFrame(recommendations)
    summary_df = pd.DataFrame(summary)
    evaluation_rows: list[dict[str, Any]] = []
    for section, metrics in evaluation.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                evaluation_rows.append(
                    {
                        "section": section,
                        "metric": metric_name,
                        "value": value,
                    }
                )
        else:
            evaluation_rows.append(
                {
                    "section": "meta",
                    "metric": section,
                    "value": metrics,
                }
            )
    evaluation_df = pd.DataFrame(evaluation_rows)
    filename = f"rekomendasi_dosen_run_{run.id}.xlsx"

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        recommendations_df.to_excel(writer, index=False, sheet_name="recommendations")
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        evaluation_df.to_excel(writer, index=False, sheet_name="evaluation")
    output.seek(0)
    return output.read(), filename