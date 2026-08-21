import numpy as np
import pytest

from minivectordb.vector_database import VectorDatabase
from tests.conftest import random_vectors


def test_empty_database(db):
    assert len(db) == 0
    assert db.dimension is None
    assert db.find_most_similar([0.5, 0.5]) == ([], [], [])


def test_store_and_retrieve(db):
    db.store_embedding(1, [0.5, 0.5], {"type": "abc"})
    assert db.dimension == 2
    assert len(db) == 1
    assert 1 in db and 2 not in db
    assert db.get_metadata(1) == {"type": "abc"}
    # Vectors come back L2-normalized.
    assert np.allclose(db.get_vector(1), [0.7071068, 0.7071068])


def test_store_batch(db):
    ids = list(range(10))
    db.store_embeddings_batch(ids, random_vectors(10), [{"i": i} for i in ids])
    assert len(db) == 10
    assert sorted(uid for uid, _ in db.iter_entries()) == ids


def test_store_batch_without_metadata(db):
    db.store_embeddings_batch(["a", "b"], random_vectors(2))
    assert db.get_metadata("a") == {}


def test_duplicate_id_is_rejected(db):
    db.store_embedding(1, [0.5, 0.5])
    with pytest.raises(ValueError, match="already exists"):
        db.store_embedding(1, [0.1, 0.9])
    with pytest.raises(ValueError, match="Duplicate unique IDs"):
        db.store_embeddings_batch([2, 2], random_vectors(2, dimension=2))
    assert len(db) == 1


def test_mismatched_batch_lengths(db):
    with pytest.raises(ValueError, match="different number"):
        db.store_embeddings_batch([1, 2], random_vectors(3, dimension=2))
    with pytest.raises(ValueError, match="Metadata dictionaries"):
        db.store_embeddings_batch([1, 2], random_vectors(2, dimension=2), [{"a": 1}])


def test_dimension_mismatch(db):
    db.store_embedding(1, [0.5, 0.5])
    with pytest.raises(ValueError, match="2-dimensional"):
        db.store_embedding(2, [0.5, 0.5, 0.5])


def test_unknown_id(db):
    with pytest.raises(ValueError, match="does not exist"):
        db.get_vector(99)
    with pytest.raises(ValueError, match="does not exist"):
        db.get_metadata(99)
    with pytest.raises(ValueError, match="does not exist"):
        db.delete_embedding(99)


def test_ids_of_any_hashable_type(db):
    db.store_embeddings_batch([1, "two", 3.5, True], random_vectors(4, dimension=2))
    assert {uid for uid, _ in db.iter_entries()} == {1, "two", 3.5, True}
    assert db.get_vector("two") is not None


def test_delete_and_slot_reuse():
    with VectorDatabase(slot_reuse_delay=0) as db:
        db.store_embeddings_batch([1, 2, 3], random_vectors(3, dimension=2))
        db.delete_embedding(2)
        assert len(db) == 2 and 2 not in db

        ids, _, _ = db.find_most_similar([1.0, 0.0], k=10)
        assert 2 not in ids

        # The freed slot is reused instead of growing the file.
        db.store_embedding(4, [0.3, 0.7])
        assert db.metadata_store.row_bound() == 3
        assert len(db) == 3
        ids, _, _ = db.find_most_similar([0.3, 0.7], k=1)
        assert ids == [4]


def test_freed_slots_are_left_alone_for_a_while():
    """A slot is not handed to a new vector while a search might be reading it."""
    with VectorDatabase(auto_compact=0) as db:
        db.store_embeddings_batch([1, 2, 3], random_vectors(3, dimension=2))
        db.delete_embedding(2)

        db.store_embedding(4, [0.3, 0.7])
        assert db.metadata_store.row_bound() == 4  # the file grew instead
        assert db.metadata_store.free_rows().tolist() == [1]

        ids, _, _ = db.find_most_similar([1.0, 0.0], k=10)
        assert sorted(ids) == [1, 3, 4]  # the hole is skipped


# -- compaction ------------------------------------------------------------


def vector_file_rows(db):
    import os

    return os.path.getsize(os.path.join(db.path, "vectors.bin")) // (db.dimension * 4)


def test_deletions_leave_no_gaps_behind(db):
    """Deleting compacts the file instead of leaving holes in it."""
    db.store_embeddings_batch(list(range(100)), random_vectors(100, dimension=8))
    db.delete_embeddings_batch(list(range(0, 100, 2)))

    assert len(db) == 50
    assert db.holes() == 0
    assert db.metadata_store.row_bound() == 50
    assert vector_file_rows(db) == 50  # the disk space came back

    # Everything that is left is still findable, and still itself.
    ids, _, _ = db.find_most_similar(db.get_vector(51), k=1)
    assert ids == [51]
    for unique_id in range(1, 100, 2):
        assert unique_id in db


def test_compaction_keeps_vectors_with_their_ids(db):
    vectors = random_vectors(60, dimension=8, seed=4)
    db.store_embeddings_batch(list(range(60)), vectors, [{"n": i} for i in range(60)])
    survivors = [i for i in range(60) if i % 3]

    db.delete_embeddings_batch([i for i in range(60) if i % 3 == 0])
    assert db.holes() == 0

    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    for unique_id in survivors:
        assert np.allclose(db.get_vector(unique_id), normalized[unique_id], atol=1e-6)
        assert db.get_metadata(unique_id) == {"n": unique_id}
        ids, scores, _ = db.find_most_similar(normalized[unique_id], k=1)
        assert ids == [unique_id] and scores[0] == pytest.approx(1.0, abs=1e-5)


def test_compaction_keeps_metadata_filters_working(db):
    db.store_embeddings_batch(
        list(range(40)), random_vectors(40, dimension=8), [{"bucket": i % 4} for i in range(40)]
    )
    db.delete_embeddings_batch(list(range(0, 40, 2)))

    ids, _, _ = db.find_most_similar(np.ones(8), {"bucket": 1}, k=40)
    assert sorted(ids) == [i for i in range(1, 40, 2) if i % 4 == 1]


def test_compaction_can_be_left_to_the_caller(db_no_compact):
    db_no_compact.store_embeddings_batch(list(range(20)), random_vectors(20, dimension=8))
    db_no_compact.delete_embeddings_batch(list(range(10)))

    assert db_no_compact.holes() == 10
    assert db_no_compact.compact() == 10  # ten rows moved down
    assert db_no_compact.holes() == 0
    assert len(db_no_compact) == 10
    assert vector_file_rows(db_no_compact) == 10

    assert db_no_compact.compact() == 0  # nothing left to do
    ids, _, _ = db_no_compact.find_most_similar(db_no_compact.get_vector(15), k=1)
    assert ids == [15]


def test_compaction_of_an_empty_database(db_no_compact):
    assert db_no_compact.compact() == 0
    db_no_compact.store_embeddings_batch([1, 2], random_vectors(2, dimension=4))
    db_no_compact.delete_embeddings_batch([1, 2])
    assert db_no_compact.compact() == 0
    assert db_no_compact.holes() == 0
    assert len(db_no_compact) == 0
    assert db_no_compact.find_most_similar([1.0, 0.0, 0.0, 0.0]) == ([], [], [])


def test_compaction_survives_a_reopen(tmp_path):
    path = str(tmp_path / "compacted")
    with VectorDatabase(path) as db:
        db.store_embeddings_batch(list(range(30)), random_vectors(30, dimension=8), [{"n": i} for i in range(30)])
        db.delete_embeddings_batch(list(range(15)))

    with VectorDatabase(path) as reopened:
        assert len(reopened) == 15 and reopened.holes() == 0
        ids, _, _ = reopened.find_most_similar(np.ones(8), {"n": {"$gte": 20}}, k=30)
        assert sorted(ids) == list(range(20, 30))


# -- updating --------------------------------------------------------------


def test_update_replaces_the_vector_in_place(db):
    db.store_embeddings_batch([1, 2], [[1.0, 0.0], [0.0, 1.0]], [{"n": 1}, {"n": 2}])
    db.update_embedding(1, [0.0, 1.0])

    assert len(db) == 2
    assert db.holes() == 0
    assert db.metadata_store.row_bound() == 2  # no new slot was used
    assert np.allclose(db.get_vector(1), [0.0, 1.0])
    assert db.get_metadata(1) == {"n": 1}  # metadata untouched

    ids, scores, _ = db.find_most_similar([0.0, 1.0], k=2)
    assert set(ids) == {1, 2} and scores[0] == pytest.approx(1.0, abs=1e-6)


def test_update_replaces_metadata_whole(db):
    db.store_embedding(1, [1.0, 0.0], {"colour": "red", "size": 4})
    db.update_embedding(1, metadata={"colour": "blue"})

    assert db.get_metadata(1) == {"colour": "blue"}
    assert np.allclose(db.get_vector(1), [1.0, 0.0])  # vector untouched
    assert db.find_most_similar([1.0, 0.0], {"colour": "blue"}, k=1)[0] == [1]
    assert db.find_most_similar([1.0, 0.0], {"colour": "red"}, k=1)[0] == []
    assert db.find_most_similar([1.0, 0.0], {"size": 4}, k=1)[0] == []


def test_update_both_at_once_and_in_batches(db):
    db.store_embeddings_batch([1, 2, 3], random_vectors(3, dimension=2), [{"n": i} for i in [1, 2, 3]])
    db.update_embeddings_batch([1, 3], [[1.0, 0.0], [0.0, 1.0]], [{"n": 10}, {"n": 30}])

    assert db.get_metadata(1) == {"n": 10} and db.get_metadata(3) == {"n": 30}
    assert np.allclose(db.get_vector(3), [0.0, 1.0])
    assert db.get_metadata(2) == {"n": 2}


def test_update_rejects_bad_arguments(db):
    db.store_embedding(1, [1.0, 0.0])
    with pytest.raises(ValueError, match="does not exist"):
        db.update_embedding(99, [0.0, 1.0])
    with pytest.raises(ValueError, match="Nothing to update"):
        db.update_embedding(1)
    with pytest.raises(ValueError, match="2-dimensional"):
        db.update_embedding(1, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="different number"):
        db.update_embeddings_batch([1], [[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(db.get_vector(1), [1.0, 0.0])  # nothing changed


def test_upsert_inserts_and_replaces(db):
    db.store_embedding(1, [1.0, 0.0], {"n": 1})
    db.upsert_embeddings_batch([1, 2], [[0.0, 1.0], [1.0, 0.0]], [{"n": 100}, {"n": 2}])

    assert len(db) == 2
    assert db.holes() == 0
    assert db.get_metadata(1) == {"n": 100}
    assert np.allclose(db.get_vector(1), [0.0, 1.0])
    assert np.allclose(db.get_vector(2), [1.0, 0.0])

    db.upsert_embedding(3, [0.5, 0.5])
    assert len(db) == 3


def test_delete_batch(db):
    ids = list(range(20))
    db.store_embeddings_batch(ids, random_vectors(20), [{"even": i % 2 == 0} for i in ids])
    db.delete_embeddings_batch([i for i in ids if i % 2])
    assert len(db) == 10
    found, _, _ = db.find_most_similar(random_vectors(1)[0], k=20)
    assert all(uid % 2 == 0 for uid in found)


def test_delete_batch_is_atomic(db):
    db.store_embeddings_batch([1, 2], random_vectors(2, dimension=2))
    with pytest.raises(ValueError, match="does not exist"):
        db.delete_embeddings_batch([1, 99])
    assert len(db) == 2


def test_search_ranks_by_cosine_similarity(db):
    db.store_embedding("same", [1.0, 0.0])
    db.store_embedding("close", [0.9, 0.1])
    db.store_embedding("orthogonal", [0.0, 1.0])

    ids, scores, _ = db.find_most_similar([1.0, 0.0], k=3)
    assert ids == ["same", "close", "orthogonal"]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    assert scores[2] == pytest.approx(0.0, abs=1e-6)


def test_search_k_larger_than_database(db):
    db.store_embeddings_batch([1, 2], random_vectors(2, dimension=2))
    ids, _, _ = db.find_most_similar([1.0, 0.0], k=100)
    assert len(ids) == 2


def test_search_with_non_positive_k(db):
    db.store_embedding(1, [1.0, 0.0])
    assert db.find_most_similar([1.0, 0.0], k=0) == ([], [], [])


def test_search_matches_brute_force(db):
    vectors = random_vectors(500, dimension=16, seed=7)
    db.store_embeddings_batch(list(range(500)), vectors)

    query = random_vectors(1, dimension=16, seed=99)[0]
    ids, scores, _ = db.find_most_similar(query, k=5)

    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    expected = np.argsort(-(normalized @ (query / np.linalg.norm(query))))[:5]
    assert ids == expected.tolist()
    assert scores[0] == pytest.approx(float(normalized[expected[0]] @ (query / np.linalg.norm(query))), abs=1e-5)


def test_search_spans_many_chunks(db, monkeypatch):
    import minivectordb._vector_store as vector_store

    monkeypatch.setattr(vector_store, "CHUNK_BYTES", 256)  # a few rows per chunk
    vectors = random_vectors(300, dimension=8, seed=3)
    db.store_embeddings_batch(list(range(300)), vectors)

    assert db.vector_store.chunk_rows < 300
    ids, _, _ = db.find_most_similar(vectors[42], k=1)
    assert ids == [42]


def test_sync_survives_a_database_growing_underneath_it(db, monkeypatch):
    """Another process may extend the database between the two reads in _sync."""
    db.store_embeddings_batch([1, 2], [[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(db.metadata_store, "free_rows", lambda: np.array([0, 7, 99], dtype=np.int64))

    db._version = None  # force a refresh
    ids, _, _ = db.find_most_similar([0.0, 1.0], k=5)
    assert ids == [2]  # row 0 was masked out, the rows past the end ignored


def test_scores_come_from_a_fresh_read(db, monkeypatch):
    """A slot rewritten during the scan is scored as what it holds now."""
    from minivectordb._vector_store import normalize

    db.store_embeddings_batch([1, 2], [[1.0, 0.0], [0.0, 1.0]])
    scan = db.vector_store.search

    def racing_scan(*args, **kwargs):
        rows, scores = scan(*args, **kwargs)
        # Stand in for another process claiming the winning slot mid-scan.
        db.vector_store.write(rows[:1], normalize(np.array([[0.0, 1.0]], dtype=np.float32)))
        return rows, np.full(scores.shape, 99.0, dtype=np.float32)

    monkeypatch.setattr(db.vector_store, "search", racing_scan)
    ids, scores, _ = db.find_most_similar([1.0, 0.0], k=1)
    assert ids == [1]
    assert scores[0] == pytest.approx(0.0, abs=1e-6)  # not the 99.0 the scan claimed


def test_reading_one_vector_retries_when_its_row_moves(db, monkeypatch):
    """Compaction between "which row?" and "read that row" is caught."""
    db.store_embeddings_batch([1, 2], [[1.0, 0.0], [0.0, 1.0]])
    row_for = db.metadata_store.row_for
    calls = []

    def moving_row_for(unique_id):
        calls.append(unique_id)
        return 999 if len(calls) == 1 else row_for(unique_id)  # a row it has left

    versions = iter([0, 1, 1, 1])  # the layout changed during the first read
    monkeypatch.setattr(db.metadata_store, "row_for", moving_row_for)
    monkeypatch.setattr(db.metadata_store, "layout_version", lambda: next(versions))

    assert np.allclose(db.get_vector(1), [1.0, 0.0])
    assert len(calls) == 2  # it looked again


def test_iterating_entries_walks_ids_not_rows(db):
    """Rows move under compaction, so paging by row could skip entries."""
    db.store_embeddings_batch([f"id-{i:03d}" for i in range(50)], random_vectors(50, dimension=4))
    db.delete_embeddings_batch([f"id-{i:03d}" for i in range(0, 50, 2)])

    paged = [uid for _, uid, _ in db.metadata_store.iter_entries(page_size=7)]
    assert paged == sorted(paged) and len(paged) == len(set(paged)) == 25
    assert sorted(uid for uid, _ in db.iter_entries()) == [f"id-{i:03d}" for i in range(1, 50, 2)]


def test_a_search_that_keeps_being_compacted_stays_correct(db, monkeypatch):
    """When rows keep moving, results may be short but never wrong."""
    db.store_embeddings_batch([1, 2, 3], [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    versions = iter(range(1000))  # every check sees a different layout
    monkeypatch.setattr(db.metadata_store, "layout_version", lambda: next(versions))

    ids, scores, _ = db.find_most_similar([1.0, 0.0], k=3)
    assert ids == [1, 2, 3]  # rows did not really move, so nothing is dropped
    assert scores[0] == pytest.approx(1.0, abs=1e-6)

    # Now the rows really are gone by the time the check runs.
    monkeypatch.setattr(db.metadata_store, "fetch", lambda rows: {})
    assert db.find_most_similar([1.0, 0.0], k=3) == ([], [], [])


def test_autocut_trims_weak_results(db):
    db.store_embedding("a", [1.0, 0.0])
    db.store_embedding("b", [0.99, 0.01])
    db.store_embedding("c", [0.2, 0.98])

    ids, _, _ = db.find_most_similar([1.0, 0.0], k=3, autocut=True)
    assert ids == ["a", "b"]


def test_persistence_across_reopen(tmp_path):
    path = str(tmp_path / "store")
    with VectorDatabase(path) as db:
        db.store_embeddings_batch([1, 2], [[1.0, 0.0], [0.0, 1.0]], [{"n": 1}, {"n": 2}])

    with VectorDatabase(path) as reopened:
        assert len(reopened) == 2
        assert reopened.dimension == 2
        ids, _, metadatas = reopened.find_most_similar([1.0, 0.0], k=1)
        assert ids == [1] and metadatas == [{"n": 1}]

        ids, _, _ = reopened.find_most_similar([1.0, 0.0], {"n": 2}, k=1)
        assert ids == [2]


def test_deletes_survive_reopen(tmp_path):
    path = str(tmp_path / "store")
    with VectorDatabase(path) as db:
        db.store_embeddings_batch([1, 2, 3], random_vectors(3, dimension=2))
        db.delete_embedding(2)

    with VectorDatabase(path, slot_reuse_delay=0) as reopened:
        assert len(reopened) == 2
        ids, _, _ = reopened.find_most_similar([1.0, 0.0], k=10)
        assert 2 not in ids
        reopened.store_embedding(4, [0.5, 0.5])
        assert reopened.metadata_store.row_bound() == 3


def test_temporary_database_is_removed():
    import os

    db = VectorDatabase()
    db.store_embedding(1, [1.0, 0.0])
    path = db.path
    assert os.path.isdir(path)
    db.close()
    assert not os.path.exists(path)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_quantized_storage(tmp_path, dtype):
    path = str(tmp_path / dtype)
    vectors = random_vectors(50, dimension=32, seed=11)
    with VectorDatabase(path, dtype=dtype) as db:
        db.store_embeddings_batch(list(range(50)), vectors)
        ids, scores, _ = db.find_most_similar(vectors[7], k=1)
        assert ids == [7]
        assert scores[0] > 0.99

    with VectorDatabase(path) as reopened:
        assert reopened.dtype == dtype  # the on-disk precision wins over the default


def test_file_size_matches_dtype(tmp_path):
    import os

    with VectorDatabase(str(tmp_path / "small"), dtype="int8") as db:
        db.store_embeddings_batch(list(range(10)), random_vectors(10, dimension=64))
        db.flush()
        size = os.path.getsize(os.path.join(db.path, "vectors.bin"))
    # 1 byte per dimension, rounded up to the minimum capacity.
    assert size == 1024 * 64


def test_rejects_unknown_dtype(tmp_path):
    with pytest.raises(ValueError, match="Unsupported dtype"):
        VectorDatabase(str(tmp_path / "bad"), dtype="float64")


@pytest.mark.parametrize("gather_ratio", [0, 1000])
def test_filtered_search_paths_agree(db, monkeypatch, gather_ratio):
    """Gathering scattered rows and masking a full scan must return the same results."""
    import minivectordb._vector_store as vector_store

    monkeypatch.setattr(vector_store, "GATHER_RATIO", gather_ratio)
    vectors = random_vectors(200, dimension=8, seed=5)
    db.store_embeddings_batch(list(range(200)), vectors, [{"bucket": i % 4} for i in range(200)])

    ids, scores, _ = db.find_most_similar(vectors[3], {"bucket": 3}, k=5)
    assert len(ids) == 5
    assert all(uid % 4 == 3 for uid in ids)
    assert scores == sorted(scores, reverse=True)
