import threading, uuid, time, numpy as np
from minivectordb.vector_database import VectorDatabase

def test_multithreaded_simultaneous_operations():
    for _ in range(1):
        db = VectorDatabase()
        embedding_size = 64

        initial_size = 5000

        # Create a large number of embeddings
        unique_ids = [ i for i in range(initial_size) ]
        embeddings = [ np.random.rand(embedding_size) for _ in range(initial_size) ]
        metadata_dicts = [ {"num_filter": f"test_{i}"} for i in range(initial_size) ]
        db.store_embeddings_batch(unique_ids, embeddings, metadata_dicts)
        
        # Now, create multiple threads that will perform indexing, searching and deletion
        def index_thread():
            for i in range(2000):
                embedding = np.random.rand(embedding_size)
                emb_id = str(uuid.uuid4())
                db.store_embedding(f"item_{emb_id}", embedding, metadata_dict={"num_filter": f"test_{i}"})
        
        def search_thread():
            for _ in range(5000):
                _, _, _ = db.find_most_similar(
                    embedding = np.random.rand(embedding_size),
                    k = 3
                )
        
        def delete_thread():
            for i in range(500, 4000):
                db.delete_embedding(i)

        # Create the threads
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=index_thread))
            threads.append(threading.Thread(target=search_thread))

        threads.append(threading.Thread(target=delete_thread))

        print("Starting multithreaded operations")
        start = time.time()

        # Start the threads
        for thread in threads:
            thread.start()
        
        # Join the threads
        for thread in threads:
            thread.join()

        end = time.time()

        print(f"Time taken for multithreaded operations: {end - start}")

        # Assert that the number of embeddings is correct
        assert len(db.id_map) == (initial_size + 5 * 2000) - 3500
        assert len(db.inverse_id_map) == (initial_size + 5 * 2000) - 3500
        assert len(db.metadata) == (initial_size + 5 * 2000) - 3500
        assert len(db.embeddings) == (initial_size + 5 * 2000) - 3500