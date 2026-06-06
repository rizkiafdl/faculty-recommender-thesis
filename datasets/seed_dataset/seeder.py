from __future__ import annotations

from sqlalchemy.orm import Session

from app import queries
from app.models import Supervisor
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
