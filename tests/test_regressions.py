"""Regressions found by adversarially attacking the library.

Each test here stands for a defect that was real: a repro existed, it failed,
and the fix is what makes it pass. The names say what went wrong.
"""
import gc
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

from minivectordb._metadata_store import MetadataStore, encode
from minivectordb._vector_store import VectorStore
from minivectordb.embedding_model import EmbeddingModel
from minivectordb.rerank import autocut, hybrid_rerank
from minivectordb.vector_database import VectorDatabase
from tests.conftest import random_vectors


# -- writes that reached the file before anything checked them --------------


def test_upsert_checks_the_vector_width(db):
    """A too-wide vector used to be written straight over the next row."""
    db.store_embeddings_batch(["a", "b"], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="4-dimensional"):
        db.upsert_embeddings_batch(["a"], [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])

    assert np.allclose(db.get_vector("a"), [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(db.get_vector("b"), [0.0, 1.0, 0.0, 0.0])  # the neighbour is untouched


def test_the_vector_store_refuses_a_mismatched_block():
    """The last line of defence, below the database's own checks."""
    store = VectorStore("/dev/null", 4, "float32")
    with pytest.raises(ValueError, match="shape"):
        store.write(np.array([0, 1]), np.zeros((2, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="negative"):
        store.read(np.array([-1]))


def test_a_failed_batch_changes_nothing(tmp_path):
    """A call that raises must not leave a write behind, on disk or in memory."""
    path = str(tmp_path / "atomic")
    with VectorDatabase(path) as db:
        db.store_embeddings_batch(["a", "b"], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

        # A batch mixing an existing id with a badly shaped vector: the
        # existing id used to be rewritten before the failure was noticed.
        with pytest.raises(ValueError):
            db.upsert_embeddings_batch(["a", "z"], [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]])

        assert len(db) == 2 and "z" not in db
        assert np.allclose(db.get_vector("a"), [1.0, 0.0, 0.0, 0.0])

    with VectorDatabase(path) as reopened:  # and nothing was persisted either
        assert len(reopened) == 2
        assert np.allclose(reopened.get_vector("a"), [1.0, 0.0, 0.0, 0.0])


def test_an_empty_vector_does_not_break_the_database(db):
    with pytest.raises(ValueError, match="at least one dimension"):
        db.store_embedding("nothing", [])
    # The instance used to be left with a zero-width store and raise for ever.
    db.store_embedding("something", [1.0, 2.0, 3.0, 4.0])
    assert len(db) == 1


def test_a_dimension_of_zero_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least one dimension"):
        VectorDatabase(str(tmp_path / "zero"), dimension=0)
    with pytest.raises(ValueError, match="at least one dimension"):
        VectorStore(str(tmp_path / "zero.bin"), 0)


# -- updating an entry while someone else reads it --------------------------


def test_updating_moves_the_entry_to_a_fresh_slot():
    """Rows are never rewritten under a reader.

    The file is written before the metadata is committed, so a reader cannot
    tell that a rewrite is under way. An update therefore writes somewhere
    new and repoints the entry, leaving the old row alone. (Compaction is off
    here, or it would tidy the entry straight back to where it started.)
    """
    with VectorDatabase(auto_compact=0) as db:
        _assert_update_relocates(db)


def _assert_update_relocates(db):
    db.store_embedding("a", [1.0, 0.0])
    before = db.metadata_store.row_for("a")
    old = db.vector_store.read(np.array([before]))[0].copy()

    db.update_embedding("a", [0.0, 1.0], {"generation": 2})
    after = db.metadata_store.row_for("a")

    assert after != before
    assert np.allclose(db.vector_store.read(np.array([before]))[0], old)  # old row untouched
    assert np.allclose(db.get_vector("a"), [0.0, 1.0])
    assert db.get_metadata("a") == {"generation": 2}
    assert db.metadata_store.free_rows().tolist() == [before]


def test_updates_do_not_pile_up_gaps(db):
    """The slot an update leaves behind is reclaimed, not leaked."""
    db.store_embeddings_batch(list(range(20)), random_vectors(20, dimension=8))
    for round_number in range(5):
        db.update_embeddings_batch(list(range(20)), random_vectors(20, dimension=8, seed=round_number + 1))

    assert len(db) == 20
    assert db.holes() == 0
    assert db.metadata_store.row_bound() == 20


def test_a_search_pairs_each_score_with_its_own_metadata(db):
    """Score and document must come from the same generation of an entry."""
    db.store_embedding("hot", [1.0, 0.0])
    query = np.array([1.0, 0.0], dtype=np.float32)

    stop = threading.Event()
    mismatches = []

    def writer():
        generation = 0
        while not stop.is_set():
            generation += 1
            angle = generation % 90 * 0.01
            vector = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            db.update_embedding("hot", vector, {"expected": float(vector @ query)})

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for _ in range(300):
            ids, scores, metadatas = db.find_most_similar(query, k=1)
            for score, metadata in zip(scores, metadatas):
                if abs(score - metadata["expected"]) > 1e-3:
                    mismatches.append((score, metadata))
    finally:
        stop.set()
        thread.join()

    assert not mismatches


# -- the free-row mask, tested on its own ----------------------------------


def test_deleted_rows_are_masked_out_of_the_scan(db_no_compact):
    """Checked directly, not through a second safety net downstream."""
    db_no_compact.store_embeddings_batch([1, 2, 3], random_vectors(3, dimension=4))
    row = db_no_compact.metadata_store.row_for(2)
    db_no_compact.delete_embedding(2)
    db_no_compact._sync()

    assert db_no_compact._active is not None
    assert db_no_compact._active[row] == False  # noqa: E712 - the point is the value
    assert db_no_compact._active.sum() == 2

    # And the scan itself skips it, before any metadata lookup can hide it.
    rows, _ = db_no_compact.vector_store.search(
        np.ones(4, dtype=np.float32), k=10, n_rows=db_no_compact._row_bound, active=db_no_compact._active
    )
    assert row not in rows.tolist()


def test_a_zero_vector_stays_zero(db):
    """Normalizing a zero vector must not produce NaN."""
    db.store_embedding("zero", [0.0, 0.0, 0.0, 0.0])
    db.store_embedding("real", [1.0, 0.0, 0.0, 0.0])

    stored = db.get_vector("zero")
    assert np.all(stored == 0.0) and not np.isnan(stored).any()

    ids, scores, _ = db.find_most_similar([1.0, 0.0, 0.0, 0.0], k=2)
    assert ids[0] == "real"
    assert not any(np.isnan(score) for score in scores)


def test_int8_keeps_its_precision(tmp_path):
    """A quantization regression should fail here, not just a severe one."""
    vectors = random_vectors(40, dimension=32, seed=8)
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    with VectorDatabase(str(tmp_path / "int8"), dtype="int8") as db:
        db.store_embeddings_batch(list(range(40)), vectors)
        errors = [np.abs(db.get_vector(i) - normalized[i]).max() for i in range(40)]

    # 8 bits over [-1, 1] is a step of 1/127, so nothing may be off by more.
    assert max(errors) <= 1.0 / 127
    assert np.mean(errors) < 1.0 / 200


def test_scores_stay_inside_the_documented_range(tmp_path):
    """A quantized vector is a rounded unit vector, so it is not quite unit.

    Scoring it as it stands used to put the similarity above 1. What is left
    is float32 rounding, a hundred-millionth either way.
    """
    with VectorDatabase(str(tmp_path / "range"), dtype="int8") as db:
        vectors = random_vectors(200, dimension=2, seed=12)
        db.store_embeddings_batch(list(range(200)), vectors)
        for index in range(0, 200, 20):
            _, scores, _ = db.find_most_similar(db.get_vector(index), k=5)
            assert all(-1.0 - 1e-6 <= score <= 1.0 + 1e-6 for score in scores)
            assert scores == sorted(scores, reverse=True)

        # And the quantization must not flatten the ranking into ties.
        _, scores, _ = db.find_most_similar(db.get_vector(3), k=10)
        assert len(set(round(score, 7) for score in scores)) >= 5


# -- metadata filters -------------------------------------------------------


def test_a_key_holding_an_empty_list_still_exists(db):
    db.store_embedding("empty", [1.0, 0.0], {"tags": []})
    db.store_embedding("full", [0.0, 1.0], {"tags": ["x"]})
    db.store_embedding("absent", [1.0, 1.0], {"other": 1})
    query = [1.0, 0.0]

    assert sorted(db.find_most_similar(query, {"tags": {"$exists": True}}, k=9)[0]) == ["empty", "full"]
    assert db.find_most_similar(query, {"tags": {"$exists": False}}, k=9)[0] == ["absent"]
    assert db.find_most_similar(query, {"tags": {"$contains": "x"}}, k=9)[0] == ["full"]


def test_contains_does_not_match_the_json_quoting(db):
    db.store_embedding("plain", [1.0, 0.0], {"name": "plain text"})
    db.store_embedding("lines", [0.0, 1.0], {"name": "two\nlines"})
    query = [1.0, 0.0]

    assert db.find_most_similar(query, {"name": {"$contains": '"'}}, k=9)[0] == []
    assert db.find_most_similar(query, {"name": {"$contains": "\\"}}, k=9)[0] == []
    assert db.find_most_similar(query, {"name": {"$contains": "ain te"}}, k=9)[0] == ["plain"]
    assert db.find_most_similar(query, {"name": {"$contains": "\n"}}, k=9)[0] == ["lines"]


def test_every_operator_in_a_condition_counts(db):
    db.store_embeddings_batch(
        ["low", "mid", "high"], [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], [{"price": p} for p in (5, 50, 500)]
    )
    query = [1.0, 0.0]

    assert db.find_most_similar(query, {"price": {"$gt": 1, "$lt": 10}}, k=9)[0] == ["low"]
    assert sorted(db.find_most_similar(query, {"price": {"$gte": 5, "$lte": 50}}, k=9)[0]) == ["low", "mid"]
    assert db.find_most_similar(query, {"price": {"$gt": 1, "$lt": 10, "$ne": 5}}, k=9)[0] == []
    with pytest.raises(ValueError, match="Empty operator"):
        db.find_most_similar(query, {"price": {}})


def test_ordering_compares_within_one_type(db):
    db.store_embeddings_batch(
        ["null", "bool", "text"], [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], [{"k": None}, {"k": True}, {"k": "hello"}]
    )
    # Comparing the JSON text across types used to order them by their first
    # character, which said nothing about the values.
    assert db.find_most_similar([1.0, 0.0], {"k": {"$gt": "a"}}, k=9)[0] == ["text"]
    assert db.find_most_similar([1.0, 0.0], {"k": {"$gt": None}}, k=9)[0] == []


def test_ne_matches_documents_without_the_key(db):
    """Documented behaviour that nothing used to exercise."""
    db.store_embeddings_batch(
        ["a", "b", "c"], [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], [{"type": "x"}, {"type": "y"}, {"other": 1}]
    )
    assert sorted(db.find_most_similar([1.0, 0.0], {"type": {"$ne": "x"}}, k=9)[0]) == ["b", "c"]
    assert sorted(db.find_most_similar([1.0, 0.0], {"type": {"$nin": ["x"]}}, k=9)[0]) == ["b", "c"]


def test_negative_zero_is_one_number(db):
    db.store_embedding("neg", [1.0, 0.0], {"n": -0.0})
    assert db.find_most_similar([1.0, 0.0], {"n": 0.0}, k=9)[0] == ["neg"]
    assert db.find_most_similar([1.0, 0.0], {"n": {"$gte": 0.0}}, k=9)[0] == ["neg"]


def test_ids_that_json_cannot_carry_are_refused(db):
    with pytest.raises(ValueError, match="JSON-encodable"):
        db.store_embedding(b"bytes", [1.0, 0.0])
    with pytest.raises(ValueError, match="JSON-encodable"):
        db.store_embedding("ok", [1.0, 0.0], {"value": {1, 2}})
    assert len(db) == 0


def test_ids_that_encode_alike_are_the_same_id(db):
    """A tuple and a list of the same items used to raise a raw sqlite error."""
    db.store_embedding((1, 2), [1.0, 0.0])
    with pytest.raises(ValueError, match="already exists"):
        db.store_embedding([1, 2], [0.0, 1.0])
    with pytest.raises(ValueError, match="Duplicate"):
        db.store_embeddings_batch([(3, 4), [3, 4]], [[1.0, 0.0], [0.0, 1.0]])


def test_list_ids_work_through_every_path(db):
    """An id that cannot go in a set used to crash update and upsert."""
    db.store_embedding(["a", 1], [1.0, 0.0], {"n": 1})
    db.update_embedding(["a", 1], [0.0, 1.0])
    db.upsert_embedding(["a", 1], [1.0, 1.0], {"n": 2})
    assert db.get_metadata(["a", 1]) == {"n": 2}
    db.delete_embedding(["a", 1])
    assert len(db) == 0


def test_update_rejects_duplicate_ids_like_its_siblings(db):
    db.store_embedding("a", [1.0, 0.0])
    with pytest.raises(ValueError, match="Duplicate"):
        db.update_embeddings_batch(["a", "a"], [[0.0, 1.0], [1.0, 1.0]])
    assert np.allclose(db.get_vector("a"), [1.0, 0.0])


def test_large_batches_cross_the_query_chunk_boundary(db):
    """More ids in one call than fit in a single SQL IN (...) list."""
    ids = [f"id-{index}" for index in range(1500)]
    db.store_embeddings_batch(ids, random_vectors(1500, dimension=4), [{"n": i} for i in range(1500)])
    assert len(db) == 1500

    db.update_embeddings_batch(ids, metadatas=[{"n": i, "seen": True} for i in range(1500)])
    assert db.get_metadata("id-1400") == {"n": 1400, "seen": True}

    db.upsert_embeddings_batch(ids[:1000], random_vectors(1000, dimension=4))
    assert len(db) == 1500

    db.delete_embeddings_batch(ids)
    assert len(db) == 0


# -- resources --------------------------------------------------------------


def test_a_big_k_does_not_pull_in_the_whole_collection(db):
    """The scan's memory used to follow the collection once k got large."""
    db.store_embeddings_batch(list(range(5000)), random_vectors(5000, dimension=8))
    import minivectordb._vector_store as vector_store

    chunk = db.vector_store.chunk_rows
    assert chunk >= 1
    ids, scores, _ = db.find_most_similar(np.ones(8), k=4000)
    assert len(ids) == len(set(ids)) == 4000
    assert scores == sorted(scores, reverse=True)


def test_a_temporary_database_cleans_up_after_itself():
    """Its directory used to survive until the process ended."""
    db = VectorDatabase()
    db.store_embedding(1, [1.0, 0.0])
    path = db.path
    assert os.path.isdir(path)

    del db
    gc.collect()
    assert not os.path.exists(path)


def test_compaction_reclaims_space_a_crash_left_behind(db_no_compact, monkeypatch):
    """A compaction killed between its two steps used to leak the file space."""
    db_no_compact.store_embeddings_batch(list(range(100)), random_vectors(100, dimension=8))
    db_no_compact.delete_embeddings_batch(list(range(50)))

    monkeypatch.setattr(db_no_compact.vector_store, "truncate", lambda rows: None)
    db_no_compact.compact()  # moves the entries, then "dies" before shrinking

    size = os.path.getsize(os.path.join(db_no_compact.path, "vectors.bin"))
    assert db_no_compact.holes() == 0  # metadata says it is compacted
    assert size > 50 * 8 * 4  # but the file was never shortened

    monkeypatch.undo()
    db_no_compact.compact()  # a later compaction has to finish the job
    assert os.path.getsize(os.path.join(db_no_compact.path, "vectors.bin")) == 50 * 8 * 4
    assert len(db_no_compact) == 50


# -- durability and forking -------------------------------------------------


def test_vectors_are_flushed_before_the_metadata_that_points_at_them(db, monkeypatch):
    """The ordering the crash-safety guarantee rests on.

    A process kill cannot show this: unflushed writes still sit in the page
    cache, where every other process can read them. So the order itself is
    what gets checked.
    """
    db.store_embedding("seed", [1.0, 0.0])  # opens the vector store
    order = []
    sync = db.vector_store.sync
    add = db.metadata_store.add

    monkeypatch.setattr(db.vector_store, "sync", lambda: (order.append("sync"), sync())[1])
    monkeypatch.setattr(db.metadata_store, "add", lambda records: (order.append("add"), add(records))[1])

    db.store_embedding("a", [1.0, 0.0])
    assert order == ["sync", "add"]

    order.clear()
    db.update_embedding("a", [0.0, 1.0])
    assert order[0] == "sync"


FORK_CONCURRENT = """
import os
import sys

import numpy as np

from minivectordb.vector_database import VectorDatabase

path = sys.argv[1]
db = VectorDatabase(path)
db.store_embedding("parent-0", np.eye(8)[0])

children = []
for child_index in range(3):
    pid = os.fork()
    if pid == 0:
        # Both sides write through handles inherited across the fork.
        for step in range(20):
            db.store_embedding(f"child{child_index}-{step}", np.random.rand(8))
            db.find_most_similar(np.random.rand(8), k=3)
        os._exit(0)
    children.append(pid)

for step in range(20):
    db.store_embedding(f"parent-{step + 1}", np.random.rand(8))
    db.find_most_similar(np.random.rand(8), k=3)

for pid in children:
    assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 0

assert len(db) == 1 + 20 + 3 * 20, len(db)
db.close()
print("ok")
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_parent_and_children_write_at_the_same_time(tmp_path):
    """The fork test used to let the child finish before the parent resumed."""
    result = subprocess.run(
        [sys.executable, "-c", FORK_CONCURRENT, str(tmp_path / "forked")],
        env=dict(os.environ, PYTHONPATH=os.getcwd()),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# -- reranking and the encoder ---------------------------------------------


def test_rerank_refuses_a_score_list_of_the_wrong_length():
    """One score used to be broadcast across every sentence."""
    with pytest.raises(ValueError, match="must match"):
        hybrid_rerank(["apple pie", "banana bread", "car repair"], [0.99], "recipe")
    with pytest.raises(ValueError, match="must match"):
        hybrid_rerank(["one", "two"], [0.1, 0.2, 0.3], "query")


def test_rerank_with_a_non_positive_k_returns_nothing():
    assert hybrid_rerank(["a", "b", "c"], [0.1, 0.5, 0.9], "query", k=-1) == ([], [])
    assert hybrid_rerank(["a", "b"], [0.1, 0.5], "query", k=0) == ([], [])


def test_autocut_sees_a_collapse_below_zero():
    """Cosine scores may be negative, and the drop was measured by sign."""
    assert autocut([-0.05, -0.9]) == [1]
    assert autocut([0.5, -0.1, -0.15]) == [1, 2]
    assert autocut([0.0, -0.5]) == [1]
    assert autocut([0.9, 0.88]) == []


def test_the_encoder_takes_strings_only():
    model = EmbeddingModel()
    with pytest.raises(TypeError, match="Expected a string"):
        model.extract_embeddings({"a": 1, "b": 2})  # iterable, so it used to encode as a batch
    with pytest.raises(TypeError, match="must be a string"):
        model.extract_embeddings_batch(["fine", 7])
    assert model.extract_embeddings("fine").shape == (512,)
