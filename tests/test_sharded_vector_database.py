from minivectordb.sharded_vector_database import ShardedVectorDatabase
from minivectordb.vector_database import VectorDatabase
from minivectordb.embedding_model import EmbeddingModel
from contextlib import contextmanager
import numpy as np, shutil

model = EmbeddingModel()

@contextmanager
def delete_folder_after(folder_path='./db_shards'):
    try:
        yield
    finally:
        shutil.rmtree(folder_path)

def test_initialization():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        assert db.embedding_size is None
        assert len(db.inverse_id_map) == 0

def test_store_and_retrieve_embedding():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        db.store_embedding(1, [0.5, 0.5])
        assert db.embedding_size == 2
        assert 1 in db.inverse_id_map
        assert len(db.inverse_id_map) == 1

def test_store_embedding_with_metadata_filter():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        db.store_embedding(1, [0.5, 0.5], {"type": "abc"})
        db.store_embedding(2, [0.1, 0.1], {"type": "xyz"})

        # Retrieve the embedding with metadata filter
        ids, distances, metadatas = db.find_most_similar([0.7, 0.7], {"type": "abc"})

        # Assert that the returned ids and distances are of length 1
        assert len(ids) == 1
        assert len(distances) == 1
        assert len(metadatas) == 1
        assert ids[0] == 1

def test_store_embedding_with_metadata_filter_and_exclude_filter():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        db.store_embedding(1, [0.5, 0.5], {"type": "abc", "id": 1})
        db.store_embedding(2, [0.1, 0.1], {"type": "xyz", "id": "2"})
        db.store_embedding(3, [0.1, 0.1], {"type": "other", "id": 555})

        # Retrieve the embedding with metadata filter
        ids, distances, metadatas = db.find_most_similar(
            embedding = [0.7, 0.7],
            metadata_filter = {"type": "abc"},
            exclude_filter = {"type": "other"},
            k = 10
        )
        
        # Assert that the returned ids and distances are of length 1
        assert len(ids) == 1
        assert len(distances) == 1
        assert len(metadatas) == 1

        # Now test the exclude_filter, passing two items in a list, that should be excluded
        ids, distances, metadatas = db.find_most_similar(
            embedding = [0.7, 0.7],
            metadata_filter = {},
            exclude_filter = [
                {"type": "abc"},
                {"type": "xyz"}
            ],
            k = 10
        )

        # Assert that the returned ids and distances are of length 1
        assert len(ids) == 1
        assert len(distances) == 1
        assert len(metadatas) == 1

        # Test using exclude filter, to exclude all results
        # One by one
        seen_metadata = []
        seen_ids = set()
        it_count = 0
        while it_count < 10:
            exclude = [ { "id": id } for id in seen_ids ]

            _, _, metadatas = db.find_most_similar(
                embedding = [0.7, 0.7],
                metadata_filter = {},
                exclude_filter = exclude,
                k = 1
            )

            if len(metadatas) == 0:
                break

            # Assert that the returned ids have not been seen before
            assert metadatas[0]["id"] not in seen_ids

            seen_metadata.extend(metadatas)
            seen_ids.update([ metadata["id"] for metadata in metadatas ])
            it_count += 1

        # Assert that the seen metadata is equal to the total number of items
        assert len(seen_metadata) == 3
        assert len(seen_ids) == 3
        assert it_count == 3

def test_store_embedding_with_exclude_filter_none_remains():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        db.store_embedding(1, [0.5, 0.5], {"type": "abc"})
        db.store_embedding(3, [0.1, 0.1], {"kind": "other"})

        # Retrieve the embedding with metadata filter
        ids, distances, metadatas = db.find_most_similar(
            embedding = [0.7, 0.7],
            exclude_filter = {
                "kind": "other",
                "type": "abc"
            },
            k = 10
        )

        # Assert that the returned ids and distances are of length 1
        assert len(ids) == 0
        assert len(distances) == 0
        assert len(metadatas) == 0

def test_store_then_delete_with_stored_metadata():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        db.store_embedding(1, [0.5, 0.5], {"type": "abc"})
        db.delete_embeddings_batch(1)

        # Retrieve the embedding with metadata filter
        ids, distances, metadatas = db.find_most_similar([0.7, 0.7], {"type": "abc"})

        # Assert that the returned ids and distances are of length 0
        assert len(ids) == 0
        assert len(distances) == 0
        assert len(metadatas) == 0

def test_store_embeddings_with_multiple_metadata_filters():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        db.store_embedding('1', [0.5, 0.5], {"type": "abc", "category": "first"})
        db.store_embedding('2', [0.6, 0.6], {"type": "abc", "category": "second"})
        db.store_embedding('3', [0.7, 0.7], {"type": "xyz", "category": "first"})
        db.store_embedding('4', [0.8, 0.8], {"type": "xyz", "category": "second"})

        # Apply first filter which matches embeddings '1' and '2'
        # Apply second filter which should only match embedding '1' after the intersection
        ids, distances, metadatas = db.find_most_similar([0.5, 0.5], {"type": "abc", "category": "first"})

        # Assert that the returned ids and distances match only the first embedding
        assert len(ids) == 1
        assert len(distances) == 1
        assert len(metadatas) == 1
        assert ids[0] == '1'

def test_try_retrieve_k_higher_than_existing_embedding_count():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        db.store_embedding(1, [0.5, 0.5])
        db.store_embedding(2, [0.1, 0.1])
        
        # Retrieve 3 embeddings when only 2 exist
        ids, distances, metadatas = db.find_most_similar([0.7, 0.7], k=3)

        # Assert that the returned ids and distances are of length 2
        assert len(ids) == 2
        assert len(distances) == 2
        assert len(metadatas) == 2

def test_retrieve_embeddings_when_none_indexed():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        ids, distances, metadatas = db.find_most_similar([0.5, 0.5], k=3)

        assert len(ids) == 0
        assert len(distances) == 0
        assert len(metadatas) == 0

def test_delete_embedding():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        db.store_embedding(1, [0.5, 0.5])
        db.delete_embeddings_batch(1)
        assert 1 not in db.inverse_id_map
        assert len(db.inverse_id_map) == 0

def test_persist_and_load():
    with delete_folder_after():
        db = ShardedVectorDatabase(shard_size=2)
        db.store_embedding(1, model.extract_embeddings("This is a test 1"))
        db.store_embedding(2, model.extract_embeddings("This is a test 2"))
        db.store_embedding(3, model.extract_embeddings("This is a test 3"))
        
        db2 = ShardedVectorDatabase(shard_size=2)

        assert len(db2.inverse_id_map) == 3
        assert 1 in db2.inverse_id_map
        assert 2 in db2.inverse_id_map
        assert 3 in db2.inverse_id_map

def test_valid_similarity_search_quant():
    with delete_folder_after():
        db = ShardedVectorDatabase()

        sentences = [
            (1, 'i like animals'),
            (2, 'i like cars'),
            (3, 'i like programming')
        ]

        for id, sentence in sentences:
            embedding = model.extract_embeddings(sentence)
            db.store_embedding(id, embedding)
        
        query_embedding = model.extract_embeddings("i like dogs")
        ids, distances, metadatas = db.find_most_similar(query_embedding, k=2)
        
        # Validate return counts
        assert len(ids) == 2
        assert len(distances) == 2
        assert len(metadatas) == 2

        # Validate correct semantic search (dogs should be more similar to
        # animals than cars and programming)
        assert ids[0] == 1

def test_valid_similarity_search_non_quant_small():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        e5_model = EmbeddingModel(use_quantized_onnx_model=False, e5_model_size='small')

        sentences = [
            (1, 'i like animals'),
            (2, 'i like cars'),
            (3, 'i like programming')
        ]

        for id, sentence in sentences:
            embedding = e5_model.extract_embeddings(sentence)
            db.store_embedding(id, embedding)
        
        query_embedding = e5_model.extract_embeddings("i like dogs")
        ids, distances, metadatas = db.find_most_similar(query_embedding, k=2)
        
        # Validate return counts
        assert len(ids) == 2
        assert len(distances) == 2
        assert len(metadatas) == 2

        # Validate correct semantic search (dogs should be more similar to
        # animals than cars and programming)
        assert ids[0] == 1

def test_valid_similarity_search_non_quant_large():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        e5_model = EmbeddingModel(use_quantized_onnx_model=False, e5_model_size='large')

        sentences = [
            (1, 'i like animals'),
            (2, 'i like cars'),
            (3, 'i like programming')
        ]

        for id, sentence in sentences:
            embedding = e5_model.extract_embeddings(sentence)
            db.store_embedding(id, embedding)
        
        query_embedding = e5_model.extract_embeddings("i like dogs")
        ids, distances, metadatas = db.find_most_similar(query_embedding, k=2)
        
        # Validate return counts
        assert len(ids) == 2
        assert len(distances) == 2
        assert len(metadatas) == 2

        # Validate correct semantic search (dogs should be more similar to
        # animals than cars and programming)
        assert ids[0] == 1

def test_similarity_search_with_hybrid_reranking():
    with delete_folder_after():
        db = ShardedVectorDatabase()

        sentences = [
            (1, 'i like animals'),
            (2, 'i like cars'),
            (3, 'i like programming'),
            (4, 'technology is the future')
        ]

        for id, sentence in sentences:
            embedding = model.extract_embeddings(sentence)
            db.store_embedding(id, embedding)
        
        query = "cars and animals"
        query_embedding = model.extract_embeddings(query)
        ids, distances, _ = db.find_most_similar(query_embedding, k=3)

        # Get the sentences by ids
        sentences = [sentences[id-1][1] for id in ids]

        hybrid_reranked_results = db.hybrid_rerank_results(
            sentences, distances, query, k = 2
        )
        hybried_retrieved_sentences, hybrid_scores = hybrid_reranked_results

        # Assert that the hybrid reranked results are correct (sentence ids 1 and 2)
        assert len(hybried_retrieved_sentences) == 2
        assert len(hybrid_scores) == 2
        assert 1 in ids
        assert 2 in ids

        # Now, try to find the 4 best matches, but using the autocut parameter
        query = "technology rocks"
        query_embedding = model.extract_embeddings(query)
        ids, distances, _ = db.find_most_similar(query_embedding, k=4, autocut=True)

        # Assert that only the 4th sentence is returned
        assert len(ids) == 1
        assert ids[0] == 4

        # Now test the autocut again, but in a case where no sentence is ignored
        query = "animals, cars, programming, technology"
        query_embedding = model.extract_embeddings(query)
        ids, distances, _ = db.find_most_similar(query_embedding, k=4, autocut=True)

        # Assert that all sentences are returned
        assert len(ids) == 4
        assert 1 in ids
        assert 2 in ids
        assert 3 in ids
        assert 4 in ids

def test_unique_id_validation():
    with delete_folder_after():
        db = ShardedVectorDatabase()

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
    with delete_folder_after():
        db = ShardedVectorDatabase()
        try:
            db.delete_embeddings_batch(1)
            assert False
        except ValueError:
            assert True

def test_delete_embedding_rebuilds_id_map():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        db.store_embedding(1, [0.5, 0.5])
        db.store_embedding(2, [0.1, 0.1])
        db.store_embedding(3, [0.2, 0.2])

        # Ensure we have multiple embeddings
        assert len(db.inverse_id_map) == 3

        # Delete one embedding
        db.delete_embeddings_batch(2)

        # Check if the inverse_id_map was rebuilt correctly
        assert len(db.inverse_id_map) == 2
        assert db.inverse_id_map == {1: 0, 3: 1}

def test_retrieve_embedding_by_id():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        test_embedding = [0.5, 0.5]
        db.store_embedding(1, test_embedding)

        # Retrieve the embedding
        embedding = db.get_vector(1)
        assert (embedding == test_embedding).all()

def test_retrieve_embedding_by_id_nonexistent():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        try:
            db.get_vector(1)
            assert False
        except ValueError:
            assert True

def test_search_expansion_metadata_filters():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        embedding_size = 32

        for i in range(250):
            embedding = np.random.rand(embedding_size)
            random_num = np.random.randint(1, 5)
            db.store_embedding(f"item_{i}", embedding, metadata_dict={"num_filter": f"test_{random_num}"})
        
        # Now add just a few embeddings with a different metadata filter
        for i in range(5):
            embedding = np.random.rand(embedding_size)
            db.store_embedding(f"item_{i + 250}", embedding, metadata_dict={"num_filter": "test_99"})
        
        # Now search for embeddings with the metadata filter
        ids, _, _ = db.find_most_similar(
            embedding = np.random.rand(embedding_size),
            metadata_filter = {"num_filter": "test_99"},
            k = 2
        )

        # Assert that the returned ids and distances are of length 2
        assert len(ids) == 2

def test_search_expansion_metadata_filters_with_or_filters():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        embedding_size = 32

        for i in range(250):
            embedding = np.random.rand(embedding_size)
            random_num = np.random.randint(1, 5)
            db.store_embedding(f"item_{i}", embedding, metadata_dict={"num_filter": f"test_{random_num}"})
        
        # Now add just a few embeddings with a different metadata filter
        for i in range(5):
            embedding = np.random.rand(embedding_size)
            db.store_embedding(f"item_{i + 250}", embedding, metadata_dict={"num_filter": "test_99", "type": "test"})
        
        # Now search for embeddings with the "or" metadata filter
        ids, _, _ = db.find_most_similar(
            embedding = np.random.rand(embedding_size),
            or_filters = [
                {"num_filter": "test_99"},
                {"num_filter": "test_10"},
                {"num_filter": "test_20"}
            ],
            k = 10
        )

        # Assert that the returned ids and distances are of length 5
        assert len(ids) == 5

        # Now testing "and" filter together with "or" filter
        ids, _, _ = db.find_most_similar(
            embedding = np.random.rand(embedding_size),
            metadata_filter = {
                "type": "test"
            },
            or_filters = [
                {"num_filter": "test_99"},
                {"num_filter": "test_10"},
                {"num_filter": "test_20"}
            ],
            k = 500
        )

        # Assert that the returned ids and distances are of length 5
        assert len(ids) == 5 # Only 5 items have the "type" metadata, even though we search for 10

        # Now testing "and" filter together with "or" filter, but "or" filter is a single entry and not a list
        # Should be accepted as a list of one entry internally

        embedding = np.random.rand(embedding_size)
        db.store_embedding("item_300", embedding, metadata_dict={"num_filter": "test_101", "type": "test"})

        ids, _, _ = db.find_most_similar(
            embedding = np.random.rand(embedding_size),
            metadata_filter = {
                "type": "test"
            },
            or_filters = {"num_filter": "test_101"},
            k = 10
        )

        # Assert that the returned ids and distances are of length 1
        # 5 items have the "type" metadata, but only 1 has the "test_101" value
        # so only 1 item should be returned when computing the filters simultaneously
        assert len(ids) == 1

        # Now test with multiple "or" filters, list of dicts
        ids, _, _ = db.find_most_similar(
            embedding = np.random.rand(embedding_size),
            or_filters = [
                {"num_filter": "test_101"},
                {"num_filter": "test_99"}
            ],
            k = 7
        )

        # Assert that the returned ids and distances are of length 6
        assert len(ids) == 6

def test_search_expansion_metadata_filters_high_k_exact_count():
    with delete_folder_after():
        # Create an instance of ShardedVectorDatabase
        db = ShardedVectorDatabase()

        db.store_embedding("1", model.extract_embeddings("cat"), {'category': 'irrelevant'})
        db.store_embedding("2", model.extract_embeddings("dog"), {'category': 'irrelevant'})
        db.store_embedding("3", model.extract_embeddings("bird"), {'category': 'irrelevant'})
        db.store_embedding("4", model.extract_embeddings("lion"), {'category': 'irrelevant'})
        db.store_embedding("5", model.extract_embeddings("panther"), {'category': 'irrelevant'})
        db.store_embedding("6", model.extract_embeddings("lizard"), {'category': 'irrelevant'})
        db.store_embedding("7", model.extract_embeddings("hippo"), {'category': 'irrelevant'})

        # Only 3 relevant
        db.store_embedding("8", model.extract_embeddings("dinosaur"), {'category': 'relevant'})
        db.store_embedding("9", model.extract_embeddings("worm"), {'category': 'relevant'})
        db.store_embedding("10", model.extract_embeddings("bug"), {'category': 'relevant'})

        k = 10 # Set k to a high value

        # Define an embedding that is unlikely to match the existing embeddings
        search_embedding = model.extract_embeddings("mammoth")

        # Call find_most_similar with a high value of k
        ids, _, _ = db.find_most_similar(
            embedding = search_embedding,
            
            # Set metadata_filter to select a category that has fewer entries
            metadata_filter = {"category": "relevant"},
            k = k
        )

        # Assert that the number of found IDs is equal to the number of relevant embeddings
        assert len(ids) == 3, "Number of found IDs does not match the number of relevant embeddings"

def test_batch_indexing():
    with delete_folder_after():
        # Create an instance of ShardedVectorDatabase
        db = ShardedVectorDatabase()

        sentences = [
            'i like animals',
            'i like cars',
            'i like programming',
            'technology is the future'
        ]

        # Extract embeddings for the sentences
        embeddings = [model.extract_embeddings(sentence) for sentence in sentences]

        ids = [1, 2, 3, 4]

        # Index the embeddings
        db.store_embeddings_batch(ids, embeddings)

        # Assert that we have the correct number of embeddings
        assert len(db.inverse_id_map) == 4

        new_sentence = 'dogs and cats'
        new_embedding = model.extract_embeddings(new_sentence)
        # Find the most similar embeddings
        ids, _, _ = db.find_most_similar(new_embedding, k=1)

        # Assert that the returned IDs are correct
        assert ids[0] == 1

        # Test error on batch insert with existing id (should error out)
        try:
            db.store_embeddings_batch([1, 2], [new_embedding, new_embedding])
            assert False
        except ValueError:
            assert True
        
        # Test error on batch insert with mismatching sizes of ids and embeddings
        try:
            db.store_embeddings_batch([9, 8, 25], [new_embedding, new_embedding], [{"type": "test"}])
            assert False
        except ValueError:
            assert True
        
        # Test correct insertion with a valid metadata
        db.store_embeddings_batch([5, 6], [new_embedding, new_embedding], [{"type": "test"}, {"type": "test"}])

        assert {"type": "test"} in db.metadata

def test_hybrid_rerank_with_empty_database():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        query = "cars and animals"
        query_embedding = model.extract_embeddings(query)
        ids, distances, _ = db.find_most_similar(query_embedding, k=3)

        # Get the sentences by ids
        sentences = [sentences[id-1][1] for id in ids]

        hybrid_reranked_results = db.hybrid_rerank_results(
            sentences, distances, query, k = 2
        )
        hybried_retrieved_sentences, hybrid_scores = hybrid_reranked_results

        # Assert that the hybrid reranked results are correct (sentence ids 1 and 2)
        assert len(hybried_retrieved_sentences) == 0
        assert len(hybrid_scores) == 0

def test_trying_to_delete_nonexisting_together_with_existing_ids():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        db.store_embedding(1, [0.5, 0.5])
        db.store_embedding(2, [0.1, 0.1])
        db.store_embedding(3, [0.2, 0.2])

        # Ensure we have multiple embeddings
        assert len(db.inverse_id_map) == 3

        # Delete one embedding
        try:
            db.delete_embeddings_batch([2, 4])
            assert False
        except ValueError:
            assert True
        
        # Try to delete a None value
        try:
            db.delete_embeddings_batch(None)
            assert False
        except ValueError:
            assert True

        # Try to delete an empty list
        try:
            db.delete_embeddings_batch([])
            assert False
        except ValueError:
            assert True

def test_migrate_from_non_sharded_version():
    with delete_folder_after():
        # Create an instance of ShardedVectorDatabase
        sdb = ShardedVectorDatabase()

        # Create an instance of VectorDatabase
        vdb = VectorDatabase()

        # Index some embeddings
        vdb.store_embedding(1, [0.5, 0.5])
        vdb.store_embedding(2, [0.1, 0.1])
        vdb.store_embedding(3, [0.2, 0.2])

        # Migrate the embeddings
        sdb._convert_from_non_sharded_db(vdb)

        # Assert that the embeddings were migrated correctly
        assert len(sdb.inverse_id_map) == 3

def test_index_then_delete_everything_and_reload():
    with delete_folder_after():
        # Create an instance of ShardedVectorDatabase
        db = ShardedVectorDatabase(shard_size=50)

        # Index 1000 embeddings
        for i in range(1000):
            db.store_embedding(i, np.random.rand(64))

        # Search for half
        ids, _, _ = db.find_most_similar(np.random.rand(64), k=500)

        # Delete the embeddings
        db.delete_embeddings_batch(list(ids))

        # Close and reload the database
        db = ShardedVectorDatabase(shard_size=50)

        # Assert that we have left 500 embeddings
        assert len(db.inverse_id_map) == 500

        # Now delete everything
        ids, _, _ = db.find_most_similar(np.random.rand(64), k=500)

        # Delete the embeddings
        db.delete_embeddings_batch(list(ids))

        # Close and reload the database
        db = ShardedVectorDatabase(shard_size=50)

        # Assert that we have left 0 embeddings
        assert len(db.inverse_id_map) == 0