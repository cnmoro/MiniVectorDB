from minivectordb.sharded_vector_database import ShardedVectorDatabase
from minivectordb.embedding_model import EmbeddingModel
from contextlib import contextmanager
import numpy as np, uuid, shutil
from datetime import datetime

model = EmbeddingModel()

@contextmanager
def delete_folder_after(folder_path='./db_shards'):
    try:
        yield
    finally:
        shutil.rmtree(folder_path)

def test_multifilters_options():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        embedding_size = 4

        # Add random regular embeddings
        for i in range(250):
            embedding = np.random.rand(embedding_size)
            random_num = np.random.randint(1, 5)
            db.store_embedding(f"item_{i}", embedding, metadata_dict={"num_filter": f"test_{random_num}"})
        
        # Add random embeddings with fixed value 10 and a date
        for i in range(10):
            embedding = np.random.rand(embedding_size)
            fixed_value = 10
            fixed_date = datetime(2021, 1, 1)
            db.store_embedding(f"item_{i + 250}", embedding, metadata_dict={
                "num_filter": f"test_{fixed_value}",
                "value": fixed_value,
                "date": fixed_date
            })
        
        # Add random embeddings with fixed value 20 and a date
        for i in range(10):
            embedding = np.random.rand(embedding_size)
            fixed_value = 20
            fixed_date = datetime(2022, 1, 1)
            db.store_embedding(f"item_{i + 260}", embedding, metadata_dict={
                "num_filter": f"test_{fixed_value}",
                "value": fixed_value,
                "date": fixed_date
            })

        # Testing basic filter
        value_and_filter = { "value": 10 }
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, metadata_filter=value_and_filter)
        # Assert that all results have the value 10
        for result in results[2]:
            assert result["value"] == 10
        # Assert that the number of results is 10
        assert len(results[2]) == 10

        # Testing operators filters ( gt, gte, lt, lte, ne )
        value_and_filter = { "value": { "$gte": 10 } }
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, metadata_filter=value_and_filter)
        # Assert that all results have the value 10 or 20
        for result in results[2]:
            assert result["value"] >= 10
        # Assert that the number of results is 20
        assert len(results[2]) == 20
        
        value_and_filter = { "value": { "$gte": 20 } }
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, metadata_filter=value_and_filter)
        # Assert that all results have the value 20
        for result in results[2]:
            assert result["value"] >= 20
        # Assert that the number of results is 10
        assert len(results[2]) == 10

        value_and_filter = { "value": { "$lt": 20 } }
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, metadata_filter=value_and_filter)
        # Assert that all results have the value 10
        for result in results[2]:
            assert result["value"] < 20
        # Assert that the number of results is 10
        assert len(results[2]) == 10

        value_and_filter = { "value": { "$lte": 10 } }
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, metadata_filter=value_and_filter)
        # Assert that all results have the value 10
        for result in results[2]:
            assert result["value"] <= 10
        # Assert that the number of results is 10
        assert len(results[2]) == 10

        value_and_filter = { "value": { "$ne": 10 } }
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, metadata_filter=value_and_filter)
        # Assert that all results have the value 20
        for result in results[2]:
            assert result["value"] != 10
        # Assert that the number of results is 10
        assert len(results[2]) == 10

        # Now some tests using datetime
        date_and_filter = { "date": { "$gte": datetime(2021, 1, 1) } }
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, metadata_filter=date_and_filter)
        # Assert that all results have the date greater or equal to 2021
        for result in results[2]:
            assert result["date"] >= datetime(2021, 1, 1)
        # Assert that the number of results is 20
        assert len(results[2]) == 20

        date_and_filter = { "date": { "$lt": datetime(2022, 1, 1) } }
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, metadata_filter=date_and_filter)
        # Assert that all results have the date less than 2022
        for result in results[2]:
            assert result["date"] < datetime(2022, 1, 1)
        # Assert that the number of results is 10
        assert len(results[2]) == 10

        two_filters_together = { "value": { "$gt": 15 }, "date": { "$gt": datetime(2021, 5, 5) } }
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, metadata_filter=two_filters_together)
        # Assert that all results have the value greater than 15 and the date greater than 2021-5-5
        for result in results[2]:
            assert result["value"] > 15
            assert result["date"] > datetime(2021, 5, 5)
        # Assert that the number of results is 5
        assert len(results[2]) == 10

        # Now using as the "or_filters" parameter (list of dicts)
        or_filters = [
            { "value": { "$gte": 10 } },
            { "date": { "$lte": datetime(2022, 1, 1) } }
        ]
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, or_filters=or_filters)
        # Assert that all results have the value greater or equal to 10 or the date less or equal to 2022
        for result in results[2]:
            assert result["value"] >= 10 or result["date"] <= datetime(2022, 1, 1)
        # Assert that the number of results is 30
        assert len(results[2]) == 20

        # Now test a date range using the and filter
        date_range_filter = { "date": { "$gte": datetime(2021, 1, 1), "$lte": datetime(2022, 1, 1) } }
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, metadata_filter=date_range_filter)
        # Assert that all results have the date between 2021 and 2022
        for result in results[2]:
            assert result["date"] >= datetime(2021, 1, 1)
            assert result["date"] <= datetime(2022, 1, 1)
        # Assert that the number of results is 20
        assert len(results[2]) == 20

        # Now another range
        date_range_filter = [
            { "date": { "$gte": datetime(2021, 1, 1) } },
            { "date": { "$lt": datetime(2022, 1, 1) } }
        ]
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, metadata_filter=date_range_filter)
        # Assert that all results have the date between 2021 and 2022
        for result in results[2]:
            assert result["date"] >= datetime(2021, 1, 1)
            assert result["date"] < datetime(2022, 1, 1)
        # Assert that the number of results is 10
        assert len(results[2]) == 10

        # Use the range filter, but with the "or"
        date_range_filter = [
            { "date": { "$gte": datetime(2021, 1, 1) } },
            { "date": { "$lt": datetime(2022, 1, 1) } }
        ]
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, or_filters=date_range_filter)
        # Assert that all results have the date between 2021 and 2022
        for result in results[2]:
            assert (result["date"] >= datetime(2021, 1, 1) or result["date"] < datetime(2022, 1, 1))
        # Assert that the number of results is 20
        assert len(results[2]) == 20

        # Test invalid operator on "and"
        try:
            value_and_filter = { "value": { "$invalid": 10 } }
            results = db.find_most_similar(np.random.rand(embedding_size), k=999, metadata_filter=value_and_filter)
            assert False
        except Exception as e:
            assert True

        # Test invalid operator on "or"
        try:
            or_filters = [
                { "value": { "$invalid": 10 } }
            ]
            results = db.find_most_similar(np.random.rand(embedding_size), k=999, or_filters=or_filters)
            assert False
        except Exception as e:
            assert True

def test_in_operator():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        embedding_size = 4

        # Add embeddings
        first_id = str(uuid.uuid4())
        db.store_embedding(
            first_id,
            np.random.rand(embedding_size),
            metadata_dict={
                "custom_list": [
                    "a", "b", "c"
                ]
            }
        )

        second_id = str(uuid.uuid4())
        db.store_embedding(
            second_id,
            np.random.rand(embedding_size),
            metadata_dict={
                "custom_list": [
                    "d", "e", "f"
                ]
            }
        )

        # Now test the "$in" operator
        value_and_filter = { "custom_list": { "$in": "a" } }
        results = db.find_most_similar(np.random.rand(embedding_size), k=2, metadata_filter=value_and_filter)
        # Assert that the first_id is returned
        assert first_id in results[0]
        assert len(results[0]) == 1

        value_and_filter = { "custom_list": { "$in": "d" } }
        results = db.find_most_similar(np.random.rand(embedding_size), k=2, metadata_filter=value_and_filter)
        # Assert that the second_id is returned
        assert second_id in results[0]    
        assert len(results[0]) == 1

        # Test the $in operator with the "or_filters"
        or_filters = [
            { "custom_list": { "$in": "a" } },
            { "custom_list": { "$in": "d" } }
        ]
        results = db.find_most_similar(np.random.rand(embedding_size), k=2, or_filters=or_filters)
        # Assert that both ids are returned
        assert first_id in results[0]
        assert second_id in results[0]
        assert len(results[0]) == 2
    
def test_filtering_no_results():
    with delete_folder_after():
        db = ShardedVectorDatabase()
        embedding_size = 4

        # Add random regular embeddings
        for i in range(250):
            embedding = np.random.rand(embedding_size)
            random_num = np.random.randint(1, 5)
            db.store_embedding(f"item_{i}", embedding, metadata_dict={"num_filter": f"test_{random_num}", "value": 1})
        
        # Test filtering that does not change the results
        and_filter = { "value": 2 }
        or_filter = { "value": 1 }
        results = db.find_most_similar(np.random.rand(embedding_size), k=999, or_filters=or_filter, metadata_filter=and_filter)
        # Assert that the number of results is 0
        assert len(results[2]) == 0