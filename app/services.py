from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session, selectinload
from werkzeug.security import check_password_hash, generate_password_hash

from app.benchmark import benchmark_transformer_models
from app.config import DEFAULT_SHEET_NAME, TARGET_MAX_CAPACITY, TARGET_MIN_CAPACITY
from app.database import Base, engine
from app.evaluation import build_evaluation_payload
from app.excel_io import read_students_from_excel_bytes, read_students_from_excel_path
from app.models import (
    AppUser,
    Recommendation,
    RecommendationRun,
    Student,
    Supervisor,
    SupervisorCategory,
    SupervisorCategoryAssignment,
)
from app.recommender import generate_recommendations
from app.rules import LABEL_TERMS, SUPERVISOR_PROFILES, SupervisorProfile, normalize_text, student_document, student_labels

PROFILE_TOKEN_STOPWORDS = {
    "and",
    "for",
    "the",
    "with",
    "from",
    "yang",
    "dan",
    "dengan",
    "pada",
    "atau",
    "untuk",
    "internship",
    "independent",
    "study",
    "specific",
    "company",
    "project",
    "application",
    "developer",
    "software",
    "system",
    "program",
    "onsite",
    "hybrid",
    "wfo",
}

DEFAULT_CATEGORY_SUGGESTIONS = tuple(
    sorted({label.replace("_", " ") for label in LABEL_TERMS.keys()})
)


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
        "evaluation_json": "ALTER TABLE recommendation_runs ADD COLUMN evaluation_json TEXT",
    }
    for column_name, ddl in required_ddl.items():
        if column_name in existing_columns:
            continue
        with engine.begin() as connection:
            connection.execute(text(ddl))


def _normalize_supervisor_code(value: str) -> str:
    return str(value or "").strip().upper()


def seed_supervisors(session: Session) -> int:
    profile_by_code = {profile.code: profile for profile in SUPERVISOR_PROFILES}
    existing = {
        supervisor.code: supervisor
        for supervisor in session.execute(select(Supervisor)).scalars().all()
    }
    student_supervisor_rows = session.execute(
        select(Student.current_supervisor_code, Student.current_supervisor_name).where(
            Student.current_supervisor_code.is_not(None),
            func.trim(Student.current_supervisor_code) != "",
        )
    ).all()
    student_supervisor_map: dict[str, str] = {}
    for code_raw, name_raw in student_supervisor_rows:
        code = str(code_raw or "").strip()
        if not code:
            continue
        name = str(name_raw or "").strip()
        if code not in student_supervisor_map or not student_supervisor_map[code]:
            student_supervisor_map[code] = name

    changes = 0
    for code, profile in profile_by_code.items():
        keywords = ", ".join(profile.keywords)
        if code in existing:
            supervisor = existing[code]
            changed = False
            if not supervisor.name:
                supervisor.name = profile.name
                changed = True
            if not (supervisor.profile_keywords or "").strip():
                supervisor.profile_keywords = keywords
                changed = True
            if not supervisor.is_active:
                supervisor.is_active = True
                changed = True
            if changed:
                changes += 1
        else:
            session.add(
                Supervisor(
                    code=profile.code,
                    name=profile.name,
                    profile_keywords=keywords,
                    is_active=True,
                )
            )
            changes += 1

    # Keep historical supervisors active so scope dosen tidak sempit pada list statis.
    for code, history_name in student_supervisor_map.items():
        if code in profile_by_code:
            continue
        display_name = history_name or code
        if code in existing:
            supervisor = existing[code]
            if not supervisor.is_active or not supervisor.name:
                supervisor.is_active = True
                supervisor.name = display_name
                changes += 1
        else:
            session.add(
                Supervisor(
                    code=code,
                    name=display_name,
                    profile_keywords="",
                    is_active=True,
                )
            )
            changes += 1

    # Keep non-static supervisors active to support flexible manual additions from web.

    if changes:
        session.commit()
    return changes


def _normalize_username(value: str) -> str:
    return normalize_text(value).replace(" ", "")


def _normalize_category_name(value: str) -> str:
    return normalize_text(value)


def _category_to_label(value: str) -> str:
    return _normalize_category_name(value).replace(" ", "_")


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

    existing = session.execute(
        select(AppUser).where(AppUser.username == normalized_username)
    ).scalars().first()
    if existing is not None:
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
    user = session.execute(
        select(AppUser).where(AppUser.username == normalized_username)
    ).scalars().first()
    if user is None:
        return None
    if not check_password_hash(user.password_hash, password):
        return None
    return user


def get_user_by_id(session: Session, user_id: int) -> AppUser | None:
    return session.get(AppUser, user_id)


def _ensure_category_record(session: Session, category_name: str) -> SupervisorCategory:
    normalized = _normalize_category_name(category_name)
    if not normalized:
        raise ValueError("Kategori tidak boleh kosong.")

    category = session.execute(
        select(SupervisorCategory).where(SupervisorCategory.name == normalized)
    ).scalars().first()
    if category is not None:
        return category

    category = SupervisorCategory(name=normalized)
    session.add(category)
    session.flush()
    return category


def list_supervisor_profiles_for_web(session: Session) -> list[dict[str, Any]]:
    supervisors = session.execute(
        select(Supervisor)
        .where(Supervisor.is_active.is_(True))
        .options(
            selectinload(Supervisor.category_links).selectinload(SupervisorCategoryAssignment.category)
        )
        .order_by(Supervisor.code.asc())
    ).scalars().all()

    rows: list[dict[str, Any]] = []
    for supervisor in supervisors:
        categories = sorted(
            {
                (link.category.name or "").strip()
                for link in supervisor.category_links
                if link.category and (link.category.name or "").strip()
            }
        )
        rows.append(
            {
                "code": supervisor.code,
                "name": supervisor.name,
                "profile_keywords": supervisor.profile_keywords or "",
                "categories": categories,
                "categories_text": ", ".join(categories),
            }
        )
    return rows


def list_category_suggestions(session: Session) -> list[str]:
    category_values = {
        row[0]
        for row in session.execute(select(SupervisorCategory.name)).all()
        if isinstance(row[0], str) and row[0].strip()
    }
    category_values.update(DEFAULT_CATEGORY_SUGGESTIONS)

    tracks = {
        normalize_text(row[0])
        for row in session.execute(
            select(Student.track).where(Student.track.is_not(None), func.trim(Student.track) != "")
        ).all()
        if isinstance(row[0], str) and row[0].strip()
    }
    category_values.update(track for track in tracks if track)
    return sorted(category_values)


def assign_supervisor_category(session: Session, supervisor_code: str, category_name: str) -> None:
    code = str(supervisor_code or "").strip()
    if not code:
        raise ValueError("Kode dosen wajib diisi.")
    supervisor = session.execute(
        select(Supervisor).where(Supervisor.code == code, Supervisor.is_active.is_(True))
    ).scalars().first()
    if supervisor is None:
        raise ValueError(f"Dosen {code} tidak ditemukan.")

    category = _ensure_category_record(session=session, category_name=category_name)
    existing = session.execute(
        select(SupervisorCategoryAssignment).where(
            SupervisorCategoryAssignment.supervisor_id == supervisor.id,
            SupervisorCategoryAssignment.category_id == category.id,
        )
    ).scalars().first()
    if existing is not None:
        session.rollback()
        return

    session.add(
        SupervisorCategoryAssignment(
            supervisor_id=supervisor.id,
            category_id=category.id,
        )
    )
    session.commit()


def remove_supervisor_category(session: Session, supervisor_code: str, category_name: str) -> bool:
    code = str(supervisor_code or "").strip()
    normalized_category = _normalize_category_name(category_name)
    if not code or not normalized_category:
        return False

    supervisor = session.execute(
        select(Supervisor).where(Supervisor.code == code)
    ).scalars().first()
    category = session.execute(
        select(SupervisorCategory).where(SupervisorCategory.name == normalized_category)
    ).scalars().first()
    if supervisor is None or category is None:
        return False

    link = session.execute(
        select(SupervisorCategoryAssignment).where(
            SupervisorCategoryAssignment.supervisor_id == supervisor.id,
            SupervisorCategoryAssignment.category_id == category.id,
        )
    ).scalars().first()
    if link is None:
        return False

    session.delete(link)
    session.flush()

    remaining_links = session.execute(
        select(func.count(SupervisorCategoryAssignment.id)).where(
            SupervisorCategoryAssignment.category_id == category.id
        )
    ).scalar_one()
    if int(remaining_links) == 0:
        session.delete(category)
    session.commit()
    return True


def update_supervisor_keywords(session: Session, supervisor_code: str, profile_keywords: str) -> None:
    code = str(supervisor_code or "").strip()
    if not code:
        raise ValueError("Kode dosen wajib diisi.")
    supervisor = session.execute(
        select(Supervisor).where(Supervisor.code == code, Supervisor.is_active.is_(True))
    ).scalars().first()
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

    conflict = session.execute(
        select(Supervisor).where(
            func.lower(Supervisor.name) == name.lower(),
            Supervisor.code != code,
        )
    ).scalars().first()
    if conflict is not None:
        raise ValueError(f"Nama dosen sudah dipakai oleh kode {conflict.code}.")

    existing = session.execute(
        select(Supervisor).where(Supervisor.code == code)
    ).scalars().first()
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
    category_rows = [{"category": value} for value in list_category_suggestions(session)]

    tracks = [
        {"track": row[0]}
        for row in session.execute(
            select(Student.track)
            .where(Student.track.is_not(None), func.trim(Student.track) != "")
            .distinct()
            .order_by(Student.track.asc())
        ).all()
    ]

    supervisor_df = pd.DataFrame(supervisors)
    categories_df = pd.DataFrame(category_rows)
    tracks_df = pd.DataFrame(tracks)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        supervisor_df.to_excel(writer, index=False, sheet_name="supervisor_config")
        categories_df.to_excel(writer, index=False, sheet_name="category_options")
        tracks_df.to_excel(writer, index=False, sheet_name="track_reference")
    output.seek(0)
    return output.read(), "supervisor_config_export.xlsx"


def _upsert_students(session: Session, rows: list[dict[str, Any]]) -> dict[str, int]:
    if not rows:
        return {"inserted": 0, "updated": 0, "total": 0}

    student_ids = [row["student_id"] for row in rows]
    existing_students = {
        student.student_id: student
        for student in session.execute(select(Student).where(Student.student_id.in_(student_ids))).scalars().all()
    }

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
    supervisors = session.execute(
        select(Supervisor)
        .where(Supervisor.is_active.is_(True))
        .options(
            selectinload(Supervisor.category_links).selectinload(SupervisorCategoryAssignment.category)
        )
        .order_by(Supervisor.code.asc())
    ).scalars().all()
    profile_lookup = {profile.code: profile for profile in SUPERVISOR_PROFILES}
    profiles: list[SupervisorProfile] = []
    for supervisor in supervisors:
        static_profile = profile_lookup.get(supervisor.code)
        db_keywords = tuple(
            value.strip()
            for value in (supervisor.profile_keywords or "").split(",")
            if value.strip()
        )
        categories = tuple(
            sorted(
                {
                    (link.category.name or "").strip()
                    for link in supervisor.category_links
                    if link.category and (link.category.name or "").strip()
                }
            )
        )
        category_labels = tuple(
            label
            for label in (_category_to_label(category) for category in categories)
            if label
        )

        if static_profile is None:
            merged_keywords = tuple(dict.fromkeys([*db_keywords, *categories]))
            profile = SupervisorProfile(
                code=supervisor.code,
                name=supervisor.name,
                keywords=merged_keywords,
                labels=tuple(dict.fromkeys([*category_labels, "general_flexible"])),
                flexibility=0.5,
            )
            profiles.append(profile)
            continue

        merged_keywords = tuple(
            dict.fromkeys([*static_profile.keywords, *db_keywords, *categories])
        )
        merged_labels = tuple(
            dict.fromkeys([*static_profile.labels, *category_labels])
        )
        profiles.append(
            SupervisorProfile(
                code=static_profile.code,
                name=supervisor.name or static_profile.name,
                keywords=merged_keywords,
                labels=merged_labels,
                flexibility=static_profile.flexibility,
            )
        )

    # Adaptive multilabel enrichment from historical mapping.
    if not profiles:
        return profiles

    supervisor_codes = {profile.code for profile in profiles}
    history_students = session.execute(
        select(Student)
        .where(Student.current_supervisor_code.in_(supervisor_codes))
        .order_by(Student.id.asc())
    ).scalars().all()

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
            text = student_document(record)
            label_counter.update(student_labels(record))
            for token in text.split():
                if len(token) < 3:
                    continue
                if token.isdigit() or token in PROFILE_TOKEN_STOPWORDS:
                    continue
                token_counter[token] += 1

        learned_terms = [term for term, freq in token_counter.most_common(16) if freq >= 2]
        learned_labels = [label for label, freq in label_counter.most_common(8) if freq >= 2]

        merged_keywords = tuple(dict.fromkeys([*profile.keywords, *learned_terms]))
        merged_labels = tuple(dict.fromkeys([*profile.labels, *learned_labels]))

        diversity = len({label for label, freq in label_counter.items() if freq >= 2})
        flex = profile.flexibility
        if diversity >= 5:
            flex = min(0.95, flex + 0.2)
        elif diversity >= 3:
            flex = min(0.95, flex + 0.1)

        adaptive_profiles.append(
            SupervisorProfile(
                code=profile.code,
                name=profile.name,
                keywords=merged_keywords,
                labels=merged_labels,
                flexibility=flex,
            )
        )

    return adaptive_profiles


def _students_for_recommender(session: Session) -> list[dict[str, Any]]:
    students = session.execute(select(Student).order_by(Student.student_id.asc())).scalars().all()
    items: list[dict[str, Any]] = []
    for student in students:
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
) -> RecommendationRun:
    seed_supervisors(session)
    student_payload = _students_for_recommender(session)
    if not student_payload:
        raise ValueError("Belum ada data mahasiswa. Import Excel terlebih dahulu.")

    profiles = _supervisor_profiles_from_db(session)
    if not profiles:
        raise ValueError("Belum ada data dosen aktif.")

    output = generate_recommendations(students=student_payload, supervisor_profiles=tuple(profiles))
    evaluation_payload = build_evaluation_payload(
        content_scores=output.content_similarity_matrix,
        hybrid_scores=output.hybrid_score_matrix,
        students=student_payload,
        supervisor_codes=output.supervisor_codes,
        recommendation_items=output.items,
    )

    supervisor_by_code = {
        supervisor.code: supervisor
        for supervisor in session.execute(select(Supervisor).where(Supervisor.is_active.is_(True))).scalars().all()
    }
    student_by_student_id = {
        student.student_id: student for student in session.execute(select(Student)).scalars().all()
    }

    capacity_bounds = {}
    for idx, profile in enumerate(profiles):
        capacity_bounds[profile.code] = {
            "min": output.capacity_plan.min_caps[idx],
            "max": output.capacity_plan.max_caps[idx],
            "count": output.counts_by_supervisor.get(profile.code, 0),
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
        solver_name=output.solver,
        solver_note="\n".join(note_parts) if note_parts else None,
        embedding_backend=output.embedding_backend,
        embedding_model=output.embedding_model,
        evaluation_json=json.dumps(evaluation_payload),
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
                rule_matches="; ".join(item.reasons),
                company_group_key=item.company_group_key,
            )
        )

    session.commit()
    session.refresh(run)
    return run


def get_latest_run(session: Session) -> RecommendationRun | None:
    return session.execute(
        select(RecommendationRun).order_by(RecommendationRun.created_at.desc(), RecommendationRun.id.desc())
    ).scalars().first()


def list_runs(session: Session, limit: int | None = None) -> list[RecommendationRun]:
    stmt = select(RecommendationRun).order_by(RecommendationRun.id.desc())
    if limit is not None and limit > 0:
        stmt = stmt.limit(limit)
    return session.execute(stmt).scalars().all()


def get_run_by_id(session: Session, run_id: int) -> RecommendationRun | None:
    return session.get(RecommendationRun, run_id)


def _resolve_run_id(session: Session, run_id: int | None) -> int:
    if run_id is not None:
        return run_id
    latest = get_latest_run(session)
    if latest is None:
        raise ValueError("Belum ada hasil rekomendasi.")
    return latest.id


def list_recommendations(session: Session, run_id: int | None = None) -> tuple[RecommendationRun, list[dict[str, Any]]]:
    resolved_run_id = _resolve_run_id(session, run_id)
    run = get_run_by_id(session, resolved_run_id)
    if run is None:
        raise ValueError(f"Run {resolved_run_id} tidak ditemukan.")

    rows = session.execute(
        select(Recommendation, Student, Supervisor)
        .join(Student, Recommendation.student_id == Student.id)
        .join(Supervisor, Recommendation.supervisor_id == Supervisor.id)
        .where(Recommendation.run_id == resolved_run_id)
        .order_by(Supervisor.name.asc(), Student.name.asc())
    ).all()

    data: list[dict[str, Any]] = []
    for recommendation, student, supervisor in rows:
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
    run = get_run_by_id(session, resolved_run_id)
    if run is None:
        raise ValueError(f"Run {resolved_run_id} tidak ditemukan.")

    bounds = json.loads(run.capacity_bounds_json or "{}")
    count_rows = session.execute(
        select(Supervisor.code, Supervisor.name, func.count(Recommendation.id))
        .select_from(Supervisor)
        .join(Recommendation, Recommendation.supervisor_id == Supervisor.id)
        .where(Recommendation.run_id == resolved_run_id)
        .group_by(Supervisor.code, Supervisor.name)
        .order_by(Supervisor.name.asc())
    ).all()

    summary: list[dict[str, Any]] = []
    for code, name, count in count_rows:
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
    run = get_run_by_id(session, resolved_run_id)
    if run is None:
        raise ValueError(f"Run {resolved_run_id} tidak ditemukan.")
    payload = json.loads(run.evaluation_json or "{}")
    if not isinstance(payload, dict):
        payload = {}
    return run, payload


def benchmark_models(
    session: Session,
    model_names: list[str] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    seed_supervisors(session)
    student_payload = _students_for_recommender(session)
    if not student_payload:
        raise ValueError("Belum ada data mahasiswa. Import Excel terlebih dahulu.")

    profiles = _supervisor_profiles_from_db(session)
    if not profiles:
        raise ValueError("Belum ada data dosen aktif.")

    return benchmark_transformer_models(
        students=student_payload,
        supervisor_profiles=tuple(profiles),
        model_names=model_names,
        device=device,
    )


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
