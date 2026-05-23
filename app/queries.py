from __future__ import annotations

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.models import (
    AppUser,
    LabelDescription,
    Recommendation,
    RecommendationRun,
    Student,
    Supervisor,
    SupervisorLabelAffinity,
)


# ── Users ──────────────────────────────────────────────────────────────────────

def get_user_by_id(session: Session, user_id: int) -> AppUser | None:
    return session.get(AppUser, user_id)


def get_user_by_username(session: Session, username: str) -> AppUser | None:
    return session.execute(
        select(AppUser).where(AppUser.username == username)
    ).scalars().first()


# ── Supervisors ────────────────────────────────────────────────────────────────

def get_all_supervisors(session: Session) -> list[Supervisor]:
    return session.execute(select(Supervisor)).scalars().all()


def get_active_supervisors(session: Session) -> list[Supervisor]:
    return session.execute(
        select(Supervisor).where(Supervisor.is_active.is_(True))
    ).scalars().all()


def get_active_supervisors_ordered(session: Session) -> list[Supervisor]:
    return list(session.execute(
        select(Supervisor)
        .where(Supervisor.is_active.is_(True))
        .order_by(Supervisor.code.asc())
    ).scalars().all())


def get_supervisor_by_code(session: Session, code: str) -> Supervisor | None:
    return session.execute(
        select(Supervisor).where(Supervisor.code == code)
    ).scalars().first()


def get_active_supervisor_by_code(session: Session, code: str) -> Supervisor | None:
    return session.execute(
        select(Supervisor).where(Supervisor.code == code, Supervisor.is_active.is_(True))
    ).scalars().first()


def get_supervisor_by_name_not_code(session: Session, name: str, code: str) -> Supervisor | None:
    return session.execute(
        select(Supervisor).where(
            func.lower(Supervisor.name) == name.lower(),
            Supervisor.code != code,
        )
    ).scalars().first()


def get_supervisor_code_id_map(session: Session) -> dict[str, int]:
    return {
        row[0]: row[1]
        for row in session.execute(select(Supervisor.code, Supervisor.id)).all()
    }


def get_active_supervisor_map(session: Session) -> dict[str, Supervisor]:
    return {
        supervisor.code: supervisor
        for supervisor in session.execute(
            select(Supervisor).where(Supervisor.is_active.is_(True))
        ).scalars().all()
    }


def get_current_supervisor_codes_names(session: Session) -> list[tuple]:
    return session.execute(
        select(Student.current_supervisor_code, Student.current_supervisor_name).where(
            Student.current_supervisor_code.is_not(None),
            func.trim(Student.current_supervisor_code) != "",
        )
    ).all()


# ── Students ───────────────────────────────────────────────────────────────────

def get_all_students(session: Session) -> list[Student]:
    return session.execute(select(Student).order_by(Student.student_id.asc())).scalars().all()


def get_all_students_map(session: Session) -> dict[str, Student]:
    return {
        student.student_id: student
        for student in session.execute(select(Student)).scalars().all()
    }


def get_students_by_student_ids(session: Session, student_ids: list[str]) -> dict[str, Student]:
    return {
        student.student_id: student
        for student in session.execute(
            select(Student).where(Student.student_id.in_(student_ids))
        ).scalars().all()
    }


def get_students_by_supervisor_codes(session: Session, codes: set[str]) -> list[Student]:
    return session.execute(
        select(Student)
        .where(Student.current_supervisor_code.in_(codes))
        .order_by(Student.id.asc())
    ).scalars().all()


def get_distinct_student_tracks(session: Session) -> list[str]:
    return [
        row[0]
        for row in session.execute(
            select(Student.track)
            .where(Student.track.is_not(None), func.trim(Student.track) != "")
            .distinct()
            .order_by(Student.track.asc())
        ).all()
    ]


def get_non_null_tracks(session: Session) -> list[str]:
    return [
        row[0]
        for row in session.execute(
            select(Student.track).where(Student.track.is_not(None), func.trim(Student.track) != "")
        ).all()
        if isinstance(row[0], str) and row[0].strip()
    ]


def get_students_preview(session: Session) -> list[tuple]:
    return session.execute(
        select(Student.student_id, Student.name)
        .order_by(Student.name.asc())
        .limit(200)
    ).all()


# ── Label Descriptions ─────────────────────────────────────────────────────────

def get_all_label_descriptions(session: Session) -> list[LabelDescription]:
    return session.execute(
        select(LabelDescription).order_by(LabelDescription.label_name.asc())
    ).scalars().all()


def get_existing_label_names(session: Session) -> set[str]:
    return {row[0] for row in session.execute(select(LabelDescription.label_name)).all()}


def get_label_description_by_name(session: Session, label_name: str) -> LabelDescription | None:
    return session.execute(
        select(LabelDescription).where(LabelDescription.label_name == label_name)
    ).scalars().first()


def get_label_names_ordered(session: Session) -> list[str]:
    return [
        row[0]
        for row in session.execute(
            select(LabelDescription.label_name).order_by(LabelDescription.label_name.asc())
        ).all()
    ]


# ── Label Affinities ───────────────────────────────────────────────────────────

def count_affinity_rows(session: Session) -> int:
    return int(session.execute(
        select(func.count(SupervisorLabelAffinity.id))
    ).scalar_one())


def get_affinity_with_supervisor_codes(session: Session) -> list[tuple]:
    return session.execute(
        select(SupervisorLabelAffinity, Supervisor.code)
        .outerjoin(Supervisor, SupervisorLabelAffinity.supervisor_id == Supervisor.id)
    ).all()


def get_affinity_by_supervisor_and_label(
    session: Session, supervisor_id: int, label_name: str
) -> SupervisorLabelAffinity | None:
    return session.execute(
        select(SupervisorLabelAffinity).where(
            SupervisorLabelAffinity.supervisor_id == supervisor_id,
            SupervisorLabelAffinity.label_name == label_name,
        )
    ).scalars().first()


def delete_all_affinities(session: Session) -> None:
    session.execute(text("DELETE FROM supervisor_label_affinities"))


# ── Recommendation Runs ────────────────────────────────────────────────────────

def get_latest_run(session: Session) -> RecommendationRun | None:
    return session.execute(
        select(RecommendationRun).order_by(
            RecommendationRun.created_at.desc(), RecommendationRun.id.desc()
        )
    ).scalars().first()


def list_runs(session: Session, limit: int | None = None) -> list[RecommendationRun]:
    stmt = select(RecommendationRun).order_by(RecommendationRun.id.desc())
    if limit is not None and limit > 0:
        stmt = stmt.limit(limit)
    return session.execute(stmt).scalars().all()


def get_run_by_id(session: Session, run_id: int) -> RecommendationRun | None:
    return session.get(RecommendationRun, run_id)


def get_recommendations_with_entities(session: Session, run_id: int) -> list[tuple]:
    return session.execute(
        select(Recommendation, Student, Supervisor)
        .join(Student, Recommendation.student_id == Student.id)
        .join(Supervisor, Recommendation.supervisor_id == Supervisor.id)
        .where(Recommendation.run_id == run_id)
        .order_by(Supervisor.name.asc(), Student.name.asc())
    ).all()


def get_supervisor_recommendation_counts(session: Session, run_id: int) -> list[tuple]:
    return session.execute(
        select(Supervisor.code, Supervisor.name, func.count(Recommendation.id))
        .select_from(Supervisor)
        .join(Recommendation, Recommendation.supervisor_id == Supervisor.id)
        .where(Recommendation.run_id == run_id)
        .group_by(Supervisor.code, Supervisor.name)
        .order_by(Supervisor.name.asc())
    ).all()
