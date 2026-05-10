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
    category_links = relationship(
        "SupervisorCategoryAssignment",
        back_populates="supervisor",
        cascade="all, delete-orphan",
    )


class AppUser(Base):
    __tablename__ = "app_users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class SupervisorCategory(Base):
    __tablename__ = "supervisor_categories"

    id = Column(Integer, primary_key=True)
    name = Column(String(128), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    supervisor_links = relationship(
        "SupervisorCategoryAssignment",
        back_populates="category",
        cascade="all, delete-orphan",
    )


class SupervisorCategoryAssignment(Base):
    __tablename__ = "supervisor_category_assignments"
    __table_args__ = (
        UniqueConstraint("supervisor_id", "category_id", name="uq_supervisor_category"),
    )

    id = Column(Integer, primary_key=True)
    supervisor_id = Column(Integer, ForeignKey("supervisors.id"), nullable=False, index=True)
    category_id = Column(Integer, ForeignKey("supervisor_categories.id"), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    supervisor = relationship("Supervisor", back_populates="category_links")
    category = relationship("SupervisorCategory", back_populates="supervisor_links")


class LabelDescription(Base):
    __tablename__ = "label_descriptions"

    id = Column(Integer, primary_key=True)
    label_name = Column(String(64), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=False)
    threshold = Column(Float, nullable=False, default=0.45)
    is_niche = Column(Boolean, nullable=False, default=False)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class SupervisorLabelAffinity(Base):
    __tablename__ = "supervisor_label_affinities"
    __table_args__ = (
        UniqueConstraint("supervisor_id", "label_name", name="uq_supervisor_label"),
    )

    id = Column(Integer, primary_key=True)
    supervisor_id = Column(Integer, ForeignKey("supervisors.id"), nullable=True, index=True)
    label_name = Column(String(64), nullable=False, index=True)
    boost_value = Column(Float, nullable=False, default=0.0)
    is_niche_penalty = Column(Boolean, nullable=False, default=False)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    supervisor = relationship("Supervisor", foreign_keys=[supervisor_id])


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
    rule_boost = Column(Float, nullable=False, default=0.0)
    group_boost = Column(Float, nullable=False, default=0.0)
    final_score = Column(Float, nullable=False)
    rule_matches = Column(Text, nullable=True)
    company_group_key = Column(String(255), nullable=True)

    run = relationship("RecommendationRun", back_populates="recommendations")
    student = relationship("Student", back_populates="recommendations")
    supervisor = relationship("Supervisor", back_populates="recommendations")
