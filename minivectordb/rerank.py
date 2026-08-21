"""Lexical reranking helpers.

These use a character n-gram hash vectorizer and a fuzzy ratio to reorder
semantic search results. Both work on plain NumPy, so no scikit-learn.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple
from zlib import crc32

import numpy as np
from rapidfuzz import fuzz

N_FEATURES = 64
NGRAM_RANGE = (1, 6)

#: Keeps the autocut ratio finite when a score sits at zero.
EPSILON = 1e-9


def hash_features(text: str, n_features: int = N_FEATURES, ngram_range: Tuple[int, int] = NGRAM_RANGE) -> np.ndarray:
    """Hash the character n-grams of ``text`` into a fixed-size count vector."""
    features = np.zeros(n_features, dtype=np.float32)
    low, high = ngram_range
    for size in range(low, high + 1):
        for start in range(len(text) - size + 1):
            bucket = crc32(text[start : start + size].encode("utf-8")) % n_features
            features[bucket] += 1.0
    return features


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def text_hash_scores(query: str, documents: Sequence[str]) -> np.ndarray:
    """Cosine similarity between hashed character n-grams of query and documents."""
    if not documents:
        return np.zeros(0, dtype=np.float32)
    query_vector = _unit(hash_features(query))
    return np.array([float(query_vector @ _unit(hash_features(doc))) for doc in documents], dtype=np.float32)


def fuzzy_scores(query: str, documents: Sequence[str]) -> np.ndarray:
    """Fuzzy partial-match ratios, scaled to ``[0, 1]``."""
    return np.array([fuzz.partial_ratio(query, doc) / 100.0 for doc in documents], dtype=np.float32)


def hybrid_rerank(
    sentences: Sequence[str],
    search_scores: Sequence[float],
    query: str,
    k: int = 5,
    weights: Tuple[float, float, float] = (0.80, 0.15, 0.05),
) -> Tuple[List[str], List[float]]:
    """Blend semantic, hashed n-gram and fuzzy scores, then keep the top ``k``.

    All three components live in ``[0, 1]``, so ``weights`` behaves as the
    proportions it looks like.
    """
    sentences = list(sentences)
    search_scores = np.asarray(search_scores, dtype=np.float32).reshape(-1)
    if not sentences or k <= 0:
        return [], []
    if search_scores.size != len(sentences):
        # A single score would otherwise be broadcast across every sentence,
        # ranking them all as though they had scored alike.
        raise ValueError(
            f"Got {len(sentences)} sentences and {search_scores.size} scores; they must match."
        )

    search_weight, hash_weight, fuzzy_weight = weights
    combined = (
        search_weight * search_scores
        + hash_weight * text_hash_scores(query, sentences)
        + fuzzy_weight * fuzzy_scores(query, sentences)
    )
    order = np.argsort(-combined, kind="stable")[:k]
    return [sentences[i] for i in order], [float(combined[i]) for i in order]


def autocut(scores: Sequence[float], threshold: float = 0.2) -> List[int]:
    """Indexes to drop after the largest relative drop in a descending score list.

    Returns an empty list when no drop exceeds ``threshold``. Inspired by
    Weaviate's autocut.
    """
    scores = list(scores)
    if len(scores) < 2:
        return []
    # Measured against the size of the previous score, not its value: cosine
    # similarity may be negative, and dividing by a negative would turn a
    # collapse into a negative "drop" that no threshold ever catches.
    drops = [
        (scores[i - 1] - scores[i]) / max(abs(scores[i - 1]), EPSILON)
        for i in range(1, len(scores))
    ]
    largest = max(drops)
    if largest > threshold:
        return list(range(drops.index(largest) + 1, len(scores)))
    return []
