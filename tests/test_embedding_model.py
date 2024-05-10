from minivectordb.embedding_model import AlternativeModel, EmbeddingModel

def test_load_onnx_model():
    quant_model = EmbeddingModel(use_quantized_onnx_model=True)
    assert quant_model.model is not None, "Onnx model should be loaded"

    embedding = quant_model.extract_embeddings("This is a sample text")
    assert embedding is not None, "Embedding should be extracted from onnx model"

    embedding = quant_model.extract_embeddings("This is a sample text")

    # Should be 512
    assert len(embedding) == 512, "Embedding should have 512 dimensions from onnx model"

def test_load_onnx_model_custom_cpu_core_count():
    quant_model = EmbeddingModel(use_quantized_onnx_model=True, onnx_model_cpu_core_count=1)
    assert quant_model.model is not None, "Onnx model should be loaded"

    embedding = quant_model.extract_embeddings("This is a sample text")
    assert embedding is not None, "Embedding should be extracted from onnx model"

    embedding = quant_model.extract_embeddings("This is a sample text")

    # Should be 512
    assert len(embedding) == 512, "Embedding should have 512 dimensions from onnx model"

def test_load_small_alternative_model():
    non_quant_model_small = EmbeddingModel(use_quantized_onnx_model=False, alternative_model=AlternativeModel.small)
    assert non_quant_model_small.model is not None, "Non-quant small model should be loaded"

    embedding = non_quant_model_small.extract_embeddings("This is a sample text")
    assert embedding is not None, "Embedding should be extracted from e5 model small"

    embedding = non_quant_model_small.extract_embeddings("This is a sample text")

    # Should be 384
    assert len(embedding) == 384, "Embedding should have 384 dimensions from e5 model small"

def test_load_small_alternative_model_retrocompatibility_args():
    non_quant_model_small = EmbeddingModel(use_quantized_onnx_model=False, e5_model_size='small')
    assert non_quant_model_small.model is not None, "Non-quant small model should be loaded"

    embedding = non_quant_model_small.extract_embeddings("This is a sample text")
    assert embedding is not None, "Embedding should be extracted from e5 model small"

    embedding = non_quant_model_small.extract_embeddings("This is a sample text")

    # Should be 384
    assert len(embedding) == 384, "Embedding should have 384 dimensions from e5 model small"

def test_load_large_alternative_model():
    non_quant_model_large = EmbeddingModel(use_quantized_onnx_model=False, alternative_model=AlternativeModel.large)
    assert non_quant_model_large.model is not None, "Non-quant large model should be loaded"

    embedding = non_quant_model_large.extract_embeddings("This is a sample text")
    assert embedding is not None, "Embedding should be extracted from e5 model large"

    embedding = non_quant_model_large.extract_embeddings("This is a sample text")

    # Should be 1024
    assert len(embedding) == 1024, "Embedding should have 1024 dimensions from e5 model large"

def test_load_bgem3_alternative_model():
    non_quant_model_bgem3 = EmbeddingModel(use_quantized_onnx_model=False, alternative_model=AlternativeModel.bgem3)
    assert non_quant_model_bgem3.model is not None, "Non-quant bgem3 model should be loaded"

    embedding = non_quant_model_bgem3.extract_embeddings("This is a sample text")
    assert embedding is not None, "Embedding should be extracted from bgem3 model"

    embedding = non_quant_model_bgem3.extract_embeddings("This is a sample text")

    # Should be 1024
    assert len(embedding) == 1024, "Embedding should have 1024 dimensions from bgem3 model"
