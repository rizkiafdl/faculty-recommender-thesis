from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


class Supervisor(Base):
    __tablename__ = "supervisors"

    id = Column(Integer, primary_key=True)
    code = Column(String(16), unique=True, nullable=False, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    profile_keywords = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)

    recommendations = relationship("Recommendation", back_populates="supervisor")


class AppUser(Base):
    __tablename__ = "app_users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True)
    student_id = Column(String(32), unique=True, nullable=False, index=True)
    binusian_id = Column(String(32), nullable=True)
    name = Column(String(255), nullable=False, index=True)
    email = Column(String(255), nullable=True)
    phone = Column(String(64), nullable=True)
    track = Column(String(255), nullable=True, index=True)
    gpa = Column(Float, nullable=True)
    total_sks = Column(Integer, nullable=True)
    partner_lecturer = Column(String(255), nullable=True, index=True)
    position_topic = Column(Text, nullable=True)
    duration = Column(String(64), nullable=True)
    job_start_date = Column(String(64), nullable=True)
    job_end_date = Column(String(64), nullable=True)
    work_schema = Column(String(64), nullable=True)
    enrollment_status = Column(String(64), nullable=True)
    current_supervisor_code = Column(String(16), nullable=True)
    current_supervisor_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    recommendations = relationship("Recommendation", back_populates="student")


class RecommendationRun(Base):
    __tablename__ = "recommendation_runs"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    created_by_id = Column(Integer, ForeignKey("app_users.id"), nullable=True)
    input_source = Column(String(255), nullable=True)
    total_students = Column(Integer, nullable=False)
    total_supervisors = Column(Integer, nullable=False)
    target_min_capacity = Column(Integer, nullable=False)
    target_max_capacity = Column(Integer, nullable=False)
    capacity_relaxed = Column(Boolean, nullable=False, default=False)
    capacity_note = Column(Text, nullable=True)
    capacity_bounds_json = Column(Text, nullable=True)
    solver_name = Column(String(32), nullable=True)
    solver_note = Column(Text, nullable=True)
    embedding_backend = Column(String(64), nullable=True)
    embedding_model = Column(String(255), nullable=True)
    evaluation_json = Column(Text, nullable=True)
    pipeline_config_json = Column(Text, nullable=True)
    objective_score = Column(Float, nullable=True)
    rankings_json = Column(Text, nullable=True)

    recommendations = relationship(
        "Recommendation",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class Recommendation(Base):
    __tablename__ = "recommendations"
    __table_args__ = (
        UniqueConstraint("run_id", "student_id", name="uq_run_student"),
    )

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("recommendation_runs.id"), nullable=False, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False, index=True)
    supervisor_id = Column(Integer, ForeignKey("supervisors.id"), nullable=False, index=True)
    similarity_score = Column(Float, nullable=False, default=0.0)
    group_boost = Column(Float, nullable=False, default=0.0)
    final_score = Column(Float, nullable=False)
    rule_matches = Column(Text, nullable=True)
    company_group_key = Column(String(255), nullable=True)

    run = relationship("RecommendationRun", back_populates="recommendations")
    student = relationship("Student", back_populates="recommendations")
    supervisor = relationship("Supervisor", back_populates="recommendations")
