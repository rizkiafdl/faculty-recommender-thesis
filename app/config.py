from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'recommendation.db'}")
DEFAULT_EXCEL_PATH = os.getenv(
    "DEFAULT_EXCEL_PATH",
    str(BASE_DIR / "mapping faculty supervisor.xlsx"),
)
DEFAULT_SHEET_NAME = os.getenv("DEFAULT_SHEET_NAME", "Sheet1")

TARGET_MIN_CAPACITY = int(os.getenv("TARGET_MIN_CAPACITY", "10"))
TARGET_MAX_CAPACITY = int(os.getenv("TARGET_MAX_CAPACITY", "12"))
SIMILARITY_WEIGHT = float(os.getenv("SIMILARITY_WEIGHT", "2.0"))
COMPANY_GROUP_BONUS = float(os.getenv("COMPANY_GROUP_BONUS", "1.5"))
HIGH_GPA_THRESHOLD = float(os.getenv("HIGH_GPA_THRESHOLD", "3.8"))
CURRENT_SUPERVISOR_CONTINUITY_WEIGHT = float(
    os.getenv("CURRENT_SUPERVISOR_CONTINUITY_WEIGHT", "12.0")
)
LABELED_STUDENT_PRIORITY_WEIGHT = float(os.getenv("LABELED_STUDENT_PRIORITY_WEIGHT", "2.0"))

# Transformer embedding settings (content-based recommendation).
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "answerdotai/ModernBERT-base")
EMBEDDING_FALLBACK_MODEL_NAME = os.getenv(
    "EMBEDDING_FALLBACK_MODEL_NAME",
    "sentence-transformers/all-mpnet-base-v2",
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")
# Supervisors that are considered more flexible for overflow/underflow handling.
CAPACITY_PRIORITY_CODES = [
    "D2211",  # Dr. Abdul Haris Rangkuti
    "D6184",  # Dr. Mochammad Haldi Widianto
    "D6826",  # Karen Etania Saputra
    "D1749",  # Dr. Johan Muliadi Kerta
]
