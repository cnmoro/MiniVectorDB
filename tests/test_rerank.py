import numpy as np
import pytest

from minivectordb.rerank import autocut, fuzzy_scores, hash_features, hybrid_rerank, text_hash_scores

SENTENCES = ["The sky is blue", "The ocean is blue", "Programming is cool", "I like cats"]


def test_hash_features_are_fixed_size_and_stable():
    features = hash_features("hello world")
    assert features.shape == (64,)
    assert np.array_equal(features, hash_features("hello world"))
    assert not np.array_equal(features, hash_features("goodbye world"))


def test_text_hash_scores_rank_similar_text_higher():
    scores = text_hash_scores("blue sky", SENTENCES)
    assert scores.shape == (4,)
    assert scores[0] > scores[3]
    assert np.all((scores >= -1) & (scores <= 1))


def test_text_hash_scores_on_empty_input():
    assert text_hash_scores("query", []).shape == (0,)


def test_fuzzy_scores_are_normalized():
    scores = fuzzy_scores("blue", SENTENCES)
    assert np.all((scores >= 0) & (scores <= 1))
    assert scores[0] == pytest.approx(1.0)


def test_hybrid_rerank_promotes_lexical_matches():
    search_scores = [0.66, 0.62, 0.31, 0.36]
    sentences, scores = hybrid_rerank(SENTENCES, search_scores, "blue is cool", k=3)
    assert len(sentences) == len(scores) == 3
    assert sentences[0] in ("The sky is blue", "The ocean is blue")
    assert "I like cats" not in sentences
    assert scores == sorted(scores, reverse=True)


def test_hybrid_rerank_scores_stay_in_range():
    _, scores = hybrid_rerank(SENTENCES, [0.9, 0.8, 0.7, 0.6], "blue")
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_hybrid_rerank_weights_can_disable_lexical_signals():
    sentences, scores = hybrid_rerank(SENTENCES, [0.1, 0.2, 0.3, 0.9], "blue", weights=(1.0, 0.0, 0.0))
    assert sentences[0] == "I like cats"
    assert scores[0] == pytest.approx(0.9)


def test_hybrid_rerank_on_empty_input():
    assert hybrid_rerank([], [], "query") == ([], [])


def test_autocut_finds_the_largest_drop():
    assert autocut([0.9, 0.85, 0.4, 0.3]) == [2, 3]
    assert autocut([0.9, 0.88, 0.87]) == []
    assert autocut([0.9]) == []
    assert autocut([]) == []


def test_autocut_threshold_is_configurable():
    assert autocut([1.0, 0.95, 0.9], threshold=0.02) == [2]
    assert autocut([1.0, 0.9, 0.87], threshold=0.02) == [1, 2]
    assert autocut([1.0, 0.95, 0.9], threshold=0.5) == []


def test_autocut_handles_zero_scores():
    assert autocut([0.0, 0.0]) == []


def test_database_exposes_rerank_helpers(db):
    db.store_embedding(1, [1.0, 0.0])
    sentences, scores = db.hybrid_rerank_results(SENTENCES, [0.6, 0.5, 0.4, 0.3], "blue", k=2)
    assert len(sentences) == len(scores) == 2
    assert db.autocut_scores([0.9, 0.85, 0.4]) == [2]
