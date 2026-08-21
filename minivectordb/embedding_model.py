"""Sentence embeddings backed by fast-universal-sentence-encoder (USE-3).

The model is a 512-dimensional, L2-normalized multilingual sentence encoder.
It runs on CPU with NumPy only, so importing this module does not pull in
torch, transformers or onnxruntime.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np

DIMENSION = 512

SUPPORTED_LANGUAGES = (
    "ar", "de", "en", "es", "fr", "it", "ja", "ko",
    "nl", "pl", "pt", "ru", "th", "tr", "zh-cn", "zh-tw",
)


class EmbeddingModel:
    """Thin wrapper around ``usem3.USE``.

    Args:
        denoise: normalize numeric tokens before encoding, so that texts that
            differ only in quantities stay close together. On by default.
        threads: worker threads used by the encoder. ``None`` keeps the
            encoder default.
        denoise_fn: custom callable applied to the text instead of the
            built-in number normalizer.
    """

    dimension = DIMENSION

    def __init__(self, denoise: bool = True, threads: int | None = None, denoise_fn=None):
        from usem3 import USE  # imported lazily: loading it maps the model file

        kwargs = {"denoise": denoise, "denoise_fn": denoise_fn}
        if threads is not None:
            kwargs["threads"] = threads

        self.denoise = denoise
        self.model = USE(**kwargs)

    def extract_embeddings(self, text: str) -> np.ndarray:
        """Encode one string into a ``(512,)`` float32 unit vector."""
        if not isinstance(text, str):
            # Anything else iterable would be encoded as a batch and come back
            # with a shape the caller is not expecting.
            raise TypeError(f"Expected a string, got {type(text).__name__}.")
        return np.asarray(self.model.encode(text), dtype=np.float32)

    def extract_embeddings_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode many strings at once into an ``(N, 512)`` float32 matrix."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, DIMENSION), dtype=np.float32)
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Every text in the batch must be a string.")
        return np.asarray(self.model.encode(texts), dtype=np.float32)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two strings."""
        vectors = self.extract_embeddings_batch([text_a, text_b])
        return float(vectors[0] @ vectors[1])

    def encode(self, texts: Union[str, Iterable[str]]) -> np.ndarray:
        """Encode a string or an iterable of strings (alias of the encoder)."""
        if isinstance(texts, str):
            return self.extract_embeddings(texts)
        return self.extract_embeddings_batch(list(texts))
