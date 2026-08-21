"""MiniVectorDB: a small, out-of-core vector database with a built-in encoder."""
from .embedding_model import EmbeddingModel
from .vector_database import VectorDatabase

__version__ = "3.3.0"
__all__ = ["EmbeddingModel", "VectorDatabase", "__version__"]
