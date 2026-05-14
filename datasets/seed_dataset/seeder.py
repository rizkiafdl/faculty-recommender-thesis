from __future__ import annotations

from sqlalchemy.orm import Session

from app import queries
from app.models import LabelDescription, Supervisor, SupervisorCategory, SupervisorCategoryAssignment, SupervisorLabelAffinity
from datasets.seed_dataset.affinity_matrix import DEFAULT_AFFINITY_ROWS
from datasets.seed_dataset.label_descriptions import DEFAULT_LABEL_DESCRIPTIONS
from datasets.seed_dataset.supervisor_profiles import SUPERVISOR_PROFILES


def seed_supervisors(session: Session) -> int:
    profile_by_code = {profile.code: profile for profile in SUPERVISOR_PROFILES}
    existing = {
        supervisor.code: supervisor
        for supervisor in queries.get_all_supervisors(session)
    }
    student_supervisor_map: dict[str, str] = {}
    for code_raw, name_raw in queries.get_current_supervisor_codes_names(session):
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

    # Sync supervisor codes found in student records so FK integrity holds.
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

    if changes:
        session.commit()
    return changes


def seed_label_descriptions(session: Session) -> int:
    existing = queries.get_existing_label_names(session)
    changes = 0
    for label_name, description, threshold, is_niche in DEFAULT_LABEL_DESCRIPTIONS:
        if label_name in existing:
            continue
        session.add(LabelDescription(
            label_name=label_name,
            description=description,
            threshold=threshold,
            is_niche=is_niche,
        ))
        changes += 1
    if changes:
        session.commit()
    return changes


def seed_affinity_matrix(session: Session) -> int:
    if queries.count_affinity_rows(session) > 0:
        return 0

    supervisor_id_by_code = queries.get_supervisor_code_id_map(session)

    for code, label_name, boost_value, is_niche_penalty in DEFAULT_AFFINITY_ROWS:
        supervisor_id = supervisor_id_by_code.get(code) if code else None
        if code and supervisor_id is None:
            continue
        session.add(SupervisorLabelAffinity(
            supervisor_id=supervisor_id,
            label_name=label_name,
            boost_value=boost_value,
            is_niche_penalty=is_niche_penalty,
        ))

    session.commit()
    return len(DEFAULT_AFFINITY_ROWS)


def seed_supervisor_categories(session: Session) -> int:
    supervisor_by_code = {
        s.code: s for s in queries.get_all_supervisors(session)
    }

    changes = 0
    for profile in SUPERVISOR_PROFILES:
        supervisor = supervisor_by_code.get(profile.code)
        if supervisor is None:
            continue

        for label in profile.labels:
            category_name = label.replace("_", " ")

            category = queries.get_category_by_name(session, category_name)
            if category is None:
                category = SupervisorCategory(name=category_name)
                session.add(category)
                session.flush()

            existing = queries.get_category_assignment(session, supervisor.id, category.id)
            if existing is not None:
                continue

            session.add(SupervisorCategoryAssignment(
                supervisor_id=supervisor.id,
                category_id=category.id,
            ))
            changes += 1

    if changes:
        session.commit()
    return changes
