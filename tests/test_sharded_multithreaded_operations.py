from minivectordb.sharded_vector_database import ShardedVectorDatabase
import threading, uuid, time, numpy as np, shutil, time
from contextlib import contextmanager

@contextmanager
def delete_folder_after(folder_path="./testing_shards"):
    try:
        yield
    finally:
        shutil.rmtree(folder_path)

def test_multithreaded_simultaneous_operations():
    for _ in range(1):
        with delete_folder_after():
            db = ShardedVectorDatabase(storage_dir="./testing_shards", shard_size=777)
            embedding_size = 512

            initial_count = 7711

            # Create a large number of embeddings
            unique_ids = [ str(uuid.uuid4()) for i in range(initial_count) ]
            embeddings = [ np.random.rand(embedding_size) for _ in range(initial_count) ]
            metadata_dicts = [ {"num_filter": f"test_{i}"} for i in range(initial_count) ]
            db.store_embeddings_batch(unique_ids, embeddings, metadata_dicts)

            # Store some custom metadata that span across two shards
            new_unique_ids = [ str(uuid.uuid4()) for _ in range(825) ]
            embeddings = [ np.random.rand(embedding_size) for _ in range(825) ]
            metadata_dicts = [ {"something": "custom"} for _ in range(825) ]
            db.store_embeddings_batch(new_unique_ids, embeddings, metadata_dicts)
            
            initial_count += 825

            # Now, create multiple threads that will perform indexing, searching and deletion
            def index_thread():
                for i in range(100):
                    embedding = np.random.rand(embedding_size)
                    emb_id = str(uuid.uuid4())
                    db.store_embedding(f"item_{emb_id}", embedding, metadata_dict={"num_filter": f"test_{i}"})
            
            def search_thread():
                for _ in range(100):
                    _, _, _ = db.find_most_similar(
                        embedding = np.random.rand(embedding_size),
                        k = 3
                    )
            
            def delete_thread():
                for id in unique_ids[100:300]:
                    db.delete_embeddings_batch(id)
                
                for id in unique_ids[5000:5100]:
                    db.delete_embeddings_batch(id)
                
                final_ids1 = unique_ids[5200:5300]
                db.delete_embeddings_batch(final_ids1)

                final_ids2 = unique_ids[2200:2300]
                db.delete_embeddings_batch(final_ids2)

                delete_custom_data()

            def delete_custom_data():
                # 1 - Search for the items with custom metadata
                ids, _, _ = db.find_most_similar(
                    embedding = np.random.rand(embedding_size),
                    k = 825
                )
                ids = list(ids)

                # 2 - Delete half the items one by one
                for i in range(412):
                    db.delete_embeddings_batch(ids[i])

                # 3 - Delete the rest of the items in a batch
                db.delete_embeddings_batch(ids[412:])

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

            # Print the throughput of operations
            total_op_count = 500 * 3 + 412 + 1
            print(f"Throughput: {total_op_count / (end - start)} ops/sec")

            assert len(db.inverse_id_map) == initial_count + 500 - 500 - 825
            assert len(db.metadata) == initial_count + 500 - 500 - 825
            assert len(db.embeddings) == initial_count + 500 - 500 - 825

if __name__ == "__main__":
    test_multithreaded_simultaneous_operations()