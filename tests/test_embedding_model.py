import numpy as np
import pytest

from minivectordb.embedding_model import DIMENSION, EmbeddingModel
from minivectordb.vector_database import VectorDatabase


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel()


def test_embedding_shape_and_norm(model):
    embedding = model.extract_embeddings("This is a sample text")
    assert embedding.shape == (DIMENSION,)
    assert embedding.dtype == np.float32
    assert float(np.linalg.norm(embedding)) == pytest.approx(1.0, abs=1e-3)


def test_embeddings_are_deterministic(model):
    assert np.array_equal(model.extract_embeddings("stable"), model.extract_embeddings("stable"))


def test_batch_matches_single(model):
    texts = ["primeiro texto", "second text"]
    batch = model.extract_embeddings_batch(texts)
    assert batch.shape == (2, DIMENSION)
    assert np.allclose(batch[0], model.extract_embeddings(texts[0]), atol=1e-5)


def test_empty_batch(model):
    assert model.extract_embeddings_batch([]).shape == (0, DIMENSION)


def test_encode_accepts_string_or_list(model):
    assert model.encode("one text").shape == (DIMENSION,)
    assert model.encode(["one", "two", "three"]).shape == (3, DIMENSION)


def test_similarity_is_multilingual(model):
    assert model.similarity("o carro é azul", "the car is blue") > 0.5
    assert model.similarity("o carro é azul", "recipe for lentil soup") < 0.4


def test_denoise_is_on_by_default(model):
    """Numbers are normalized, so quantities stop dominating the embedding."""
    noisy = EmbeddingModel(denoise=False)
    pair = ("paguei 350 reais", "paguei 12 reais")
    assert model.denoise is True
    assert model.similarity(*pair) > noisy.similarity(*pair)


def test_semantic_search_end_to_end(model):
    sentences = {
        1: "I like dogs",
        2: "I like cats",
        3: "The queen has one daughter",
        4: "Programming is cool",
    }
    with VectorDatabase() as db:
        db.store_embeddings_batch(
            list(sentences),
            model.extract_embeddings_batch(list(sentences.values())),
            [{"index": key} for key in sentences],
        )
        ids, scores, metadatas = db.find_most_similar(model.extract_embeddings("pets"), k=2)

    assert set(ids) == {1, 2}
    assert scores[0] >= scores[1]
    assert metadatas[0]["index"] in (1, 2)
