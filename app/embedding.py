from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np

from app.config import EMBEDDING_DEVICE, EMBEDDING_FALLBACK_MODEL_NAME, EMBEDDING_MODEL_NAME

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    HAS_SKLEARN = False


@dataclass
class EmbeddingInfo:
    backend: str
    model_name: str
    note: str | None = None


def _resolve_embedding_device(requested_device: str) -> tuple[str, str | None]:
    requested = (requested_device or "auto").strip().lower()

    if requested in {"auto", "gpu", "rtx3070ti", "rtx 3070 ti", "3070ti"}:
        if HAS_TORCH and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            return "cuda", f"GPU aktif: {gpu_name}."
        return "cpu", "CUDA tidak tersedia; fallback ke CPU."

    if requested.startswith("cuda") or requested == "gpu":
        if HAS_TORCH and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            return requested if requested.startswith("cuda") else "cuda", f"GPU aktif: {gpu_name}."
        return "cpu", "EMBEDDING_DEVICE meminta CUDA, tetapi CUDA tidak tersedia; fallback ke CPU."

    return requested, None


class EmbeddingProvider:
    def __init__(self) -> None:
        self._lock = Lock()
        self._loaded = False
        self._model: SentenceTransformer | None = None
        self._info: EmbeddingInfo | None = None

    def _load_sentence_transformer(self) -> None:
        resolved_device, device_note = _resolve_embedding_device(EMBEDDING_DEVICE)
        if not HAS_SENTENCE_TRANSFORMERS:
            note_parts = ["sentence-transformers tidak tersedia; fallback ke TF-IDF."]
            if device_note:
                note_parts.append(device_note)
            self._info = EmbeddingInfo(
                backend="tfidf-fallback",
                model_name="tfidf",
                note=" ".join(note_parts),
            )
            self._loaded = True
            return

        errors: list[str] = []
        for model_name in (EMBEDDING_MODEL_NAME, EMBEDDING_FALLBACK_MODEL_NAME):
            try:
                model = SentenceTransformer(model_name_or_path=model_name, device=resolved_device)
                self._model = model
                note_parts: list[str] = []
                if model_name != EMBEDDING_MODEL_NAME:
                    note_parts.append(
                        f"Gagal load model utama '{EMBEDDING_MODEL_NAME}', fallback ke '{model_name}'."
                    )
                if device_note:
                    note_parts.append(device_note)
                self._info = EmbeddingInfo(
                    backend="sentence-transformers",
                    model_name=model_name,
                    note=" ".join(note_parts) if note_parts else None,
                )
                self._loaded = True
                return
            except Exception as exc:  # pragma: no cover
                errors.append(f"{model_name}: {exc}")

        if HAS_SKLEARN:
            note_parts = []
            if errors:
                note_parts.append("; ".join(errors)[:1000])
            else:
                note_parts.append("Fallback ke TF-IDF.")
            if device_note:
                note_parts.append(device_note)
            self._info = EmbeddingInfo(
                backend="tfidf-fallback",
                model_name="tfidf",
                note=" ".join(note_parts),
            )
        else:
            note_parts = []
            if errors:
                note_parts.append("; ".join(errors)[:1000])
            else:
                note_parts.append("Fallback ke token-overlap.")
            if device_note:
                note_parts.append(device_note)
            self._info = EmbeddingInfo(
                backend="token-overlap-fallback",
                model_name="token-overlap",
                note=" ".join(note_parts),
            )
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load_sentence_transformer()

    @property
    def info(self) -> EmbeddingInfo:
        self._ensure_loaded()
        assert self._info is not None
        return self._info

    def similarity_matrix(self, source_docs: list[str], target_docs: list[str]) -> np.ndarray:
        self._ensure_loaded()

        if self._model is not None:
            source_vectors = self._model.encode(
                source_docs,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            target_vectors = self._model.encode(
                target_docs,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.matmul(source_vectors, target_vectors.T)

        if HAS_SKLEARN:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            corpus = source_docs + target_docs
            tfidf = vectorizer.fit_transform(corpus)
            source = tfidf[: len(source_docs)]
            target = tfidf[len(source_docs) :]
            return cosine_similarity(source, target)

        # Last resort lightweight fallback.
        src_tokens = [set((doc or "").split()) for doc in source_docs]
        tgt_tokens = [set((doc or "").split()) for doc in target_docs]
        matrix = np.zeros((len(source_docs), len(target_docs)), dtype=float)
        for i, s_tokens in enumerate(src_tokens):
            for j, t_tokens in enumerate(tgt_tokens):
                union = s_tokens | t_tokens
                matrix[i, j] = (len(s_tokens & t_tokens) / len(union)) if union else 0.0
        return matrix


_PROVIDER: EmbeddingProvider | None = None
_PROVIDER_LOCK = Lock()


def get_embedding_provider() -> EmbeddingProvider:
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER
    with _PROVIDER_LOCK:
        if _PROVIDER is None:
            _PROVIDER = EmbeddingProvider()
    return _PROVIDER
