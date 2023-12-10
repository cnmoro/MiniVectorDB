from minivectordb.embedding_model import EmbeddingModel

non_quant_model_small = EmbeddingModel(use_quantized_onnx_model=False, e5_model_size='small')
non_quant_model_large = EmbeddingModel(use_quantized_onnx_model=False, e5_model_size='large')
quant_model = EmbeddingModel(use_quantized_onnx_model=True)

def test_load_onnx_model():
    assert non_quant_model_small.model is not None, "Non-quant small model should be loaded"
    assert non_quant_model_large.model is not None, "Non-quant large model should be loaded"
    assert quant_model.model is not None, "Onnx model should be loaded"

def test_extract_embeddings():
    embedding = quant_model.extract_embeddings("This is a sample text")
    assert embedding is not None, "Embedding should be extracted from onnx model"

    embedding = non_quant_model_small.extract_embeddings("This is a sample text")
    assert embedding is not None, "Embedding should be extracted from e5 model small"

    embedding = non_quant_model_large.extract_embeddings("This is a sample text")
    assert embedding is not None, "Embedding should be extracted from e5 model large"

def test_embeddings_dimension():
    embedding = quant_model.extract_embeddings("This is a sample text")

    # Should be 512
    assert len(embedding) == 512, "Embedding should have 512 dimensions from onnx model"

    embedding = non_quant_model_small.extract_embeddings("This is a sample text")

    # Should be 384
    assert len(embedding) == 384, "Embedding should have 384 dimensions from e5 model small"

    embedding = non_quant_model_large.extract_embeddings("This is a sample text")

    # Should be 1024
    assert len(embedding) == 1024, "Embedding should have 1024 dimensions from e5 model large"
