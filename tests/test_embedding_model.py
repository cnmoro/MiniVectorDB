from minivectordb.embedding_model import EmbeddingModel

model = EmbeddingModel()

def test_load_onnx_model():
    assert model.model is not None, "Model should be loaded"

def test_extract_embeddings():
    embedding = model.extract_embeddings("This is a sample text")
    assert embedding is not None, "Embedding should be extracted"

def test_embeddings_dimension():
    embedding = model.extract_embeddings("This is a sample text")

    # Should be 512
    assert len(embedding) == 512, "Embedding should have 512 dimensions"
