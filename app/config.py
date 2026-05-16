from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'recommendation.db').as_posix()}")
DEFAULT_EXCEL_PATH = os.getenv(
    "DEFAULT_EXCEL_PATH",
    str(BASE_DIR / "map_2026.xlsx"),
)
DEFAULT_SHEET_NAME = os.getenv("DEFAULT_SHEET_NAME", "Sheet1")

TARGET_MIN_CAPACITY = int(os.getenv("TARGET_MIN_CAPACITY", "10"))
TARGET_MAX_CAPACITY = int(os.getenv("TARGET_MAX_CAPACITY", "12"))
SIMILARITY_WEIGHT = float(os.getenv("SIMILARITY_WEIGHT", "100"))
COMPANY_GROUP_BONUS = float(os.getenv("COMPANY_GROUP_BONUS", "1.5"))
HIGH_GPA_THRESHOLD = float(os.getenv("HIGH_GPA_THRESHOLD", "3.8"))

# Transformer embedding settings (content-based recommendation).
AVAILABLE_EMBEDDING_MODELS: list[str] = [
    "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
    "intfloat/multilingual-e5-large-instruct",
]
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", AVAILABLE_EMBEDDING_MODELS[2])
EMBEDDING_TASK = os.getenv("EMBEDDING_TASK", "")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")
# Supervisors that are considered more flexible for overflow/underflow handling.
CAPACITY_PRIORITY_CODES = [
    "D2211",  # Dr. Abdul Haris Rangkuti
    "D6184",  # Dr. Mochammad Haldi Widianto
    "D6826",  # Karen Etania Saputra
    "D1749",  # Dr. Johan Muliadi Kerta
]
