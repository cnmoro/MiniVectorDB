from minivectordb.embedding_model import EmbeddingModel
from minivectordb.vector_database import VectorDatabase
import uuid, os

model = EmbeddingModel()

def test_initialization():
    db = VectorDatabase(embedding_size = 512)
    assert db.embedding_size == 512
    assert db.embeddings.shape == (0, 512)
    assert len(db.id_map) == 0
    assert len(db.inverse_id_map) == 0

def test_store_and_retrieve_embedding():
    db = VectorDatabase(embedding_size = 2)
    db.store_embedding(1, [0.5, 0.5])
    assert len(db.id_map) == 1
    assert 1 in db.inverse_id_map

def test_delete_embedding():
    db = VectorDatabase(embedding_size=2)
    db.store_embedding(1, [0.5, 0.5])
    db.delete_embedding(1)
    assert len(db.id_map) == 0
    assert 1 not in db.inverse_id_map

def test_persist_and_load():
    storage_file_tmp = f"{uuid.uuid4()}.pkl"
    db = VectorDatabase(embedding_size=512, storage_file=storage_file_tmp)
    db.store_embedding(1, model.extract_embeddings("This is a test"))
    db.persist_to_disk()
    
    db2 = VectorDatabase(embedding_size=512, storage_file=storage_file_tmp)

    # Remove the temporary file
    os.remove(storage_file_tmp)

    assert len(db2.id_map) == 1
    assert 1 in db2.inverse_id_map

def test_valid_similarity_search():
    db = VectorDatabase(embedding_size=512)

    sentences = [
        (1, 'i like animals'),
        (2, 'i like cars'),
        (3, 'i like programming')
    ]

    for id, sentence in sentences:
        embedding = model.extract_embeddings(sentence)
        db.store_embedding(id, embedding)
    
    query_embedding = model.extract_embeddings("i like dogs")
    ids, distances = db.find_most_similar(query_embedding, k=2)
    
    # Validate return counts
    assert len(ids) == 2
    assert len(distances) == 2

    # Validate correct semantic search (dogs should be more similar to
    # animals than cars and programming)
    assert ids[0] == 1

def test_unique_id_validation():
    db = VectorDatabase(embedding_size=2)

    to_index = [
        (1, [0.5, 0.5]),
        (1, [0.5, 0.5])
    ]
    
    # Should raise ValueError
    try:
        for id, embedding in to_index:
            db.store_embedding(id, embedding)
        assert False
    except ValueError:
        assert True

def test_delete_nonexistent_id():
    db = VectorDatabase(embedding_size=2)
    try:
        db.delete_embedding(1)
        assert False
    except ValueError:
        assert True

def test_delete_embedding_rebuilds_id_map():
    db = VectorDatabase(embedding_size=2)
    db.store_embedding(1, [0.5, 0.5])
    db.store_embedding(2, [0.1, 0.1])
    db.store_embedding(3, [0.2, 0.2])

    # Ensure we have multiple embeddings
    assert len(db.id_map) == 3

    # Delete one embedding
    db.delete_embedding(2)

    # Check if the id_map was rebuilt correctly
    assert len(db.id_map) == 2
    assert db.id_map == {0: 1, 1: 3}

def test_retrieve_embedding_by_id():
    db = VectorDatabase(embedding_size=2)
    test_embedding = [0.5, 0.5]
    db.store_embedding(1, test_embedding)

    # Retrieve the embedding
    embedding = db.get_vector(1)
    assert (embedding == test_embedding).all()

def test_retrieve_embedding_by_id_nonexistent():
    db = VectorDatabase(embedding_size=2)
    try:
        db.get_vector(1)
        assert False
    except ValueError:
        assert True