import numpy as np
import pytest

from minivectordb.vector_database import VectorDatabase


@pytest.fixture
def db():
    """A temporary database, removed when the test ends."""
    database = VectorDatabase()
    yield database
    database.close()


@pytest.fixture
def db_no_compact():
    """A temporary database that keeps the gaps deletions leave."""
    database = VectorDatabase(auto_compact=0, slot_reuse_delay=0)
    yield database
    database.close()


@pytest.fixture
def persistent_db(tmp_path):
    """A database on disk that can be reopened inside a test."""
    database = VectorDatabase(str(tmp_path / "store"))
    yield database
    database.close()


def random_vectors(count, dimension=8, seed=0):
    generator = np.random.default_rng(seed)
    return generator.normal(size=(count, dimension)).astype(np.float32)
