"""Randomized workloads with every operation running at the same time.

Each worker owns a namespace of ids that nothing else touches, so it can
record exactly what it stored and check its own results as it goes. When the
workers are done, the database must hold precisely the union of what they
recorded -- same ids, same vectors, same metadata, and no gaps.
"""
import json
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

from minivectordb.vector_database import VectorDatabase

DIMENSION = 8

WORKER = '''
import json
import sys

import numpy as np

sys.path.insert(0, {repository!r})
from minivectordb.vector_database import VectorDatabase

path, worker, steps, compactor = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
generator = np.random.default_rng(worker)
expected = {{}}
counter = 0

def check(db, unique_id, vector):
    """Our own ids are ours alone, so they must read back exactly."""
    stored = db.get_vector(unique_id)
    assert np.allclose(stored, vector, atol=1e-3), (unique_id, stored[:3], vector[:3])

with VectorDatabase(path) as db:
    for step in range(steps):
        operation = generator.choice(["insert", "insert", "batch", "update", "upsert", "delete", "search"])
        live = sorted(expected)

        if operation in ("insert", "batch"):
            size = 1 if operation == "insert" else int(generator.integers(2, 12))
            ids, vectors, metadatas = [], [], []
            for _ in range(size):
                counter += 1
                unique_id = f"w{{worker}}-{{counter}}"
                vector = generator.normal(size={dimension}).astype(np.float32)
                vector /= np.linalg.norm(vector)
                metadata = {{"worker": worker, "bucket": int(generator.integers(0, 4))}}
                ids.append(unique_id); vectors.append(vector); metadatas.append(metadata)
            db.store_embeddings_batch(ids, vectors, metadatas)
            for unique_id, vector, metadata in zip(ids, vectors, metadatas):
                expected[unique_id] = (vector, metadata)
            check(db, ids[0], expected[ids[0]][0])

        elif operation == "update" and live:
            unique_id = str(generator.choice(live))
            vector = generator.normal(size={dimension}).astype(np.float32)
            vector /= np.linalg.norm(vector)
            metadata = {{"worker": worker, "bucket": int(generator.integers(0, 4)), "updated": True}}
            db.update_embedding(unique_id, vector, metadata)
            expected[unique_id] = (vector, metadata)
            check(db, unique_id, vector)

        elif operation == "upsert":
            unique_id = str(generator.choice(live)) if live and generator.random() < 0.5 else f"w{{worker}}-u{{step}}"
            vector = generator.normal(size={dimension}).astype(np.float32)
            vector /= np.linalg.norm(vector)
            metadata = {{"worker": worker, "bucket": int(generator.integers(0, 4))}}
            db.upsert_embedding(unique_id, vector, metadata)
            expected[unique_id] = (vector, metadata)
            check(db, unique_id, vector)

        elif operation == "delete" and live:
            victims = [str(unique_id) for unique_id in generator.choice(live, size=min(3, len(live)), replace=False)]
            db.delete_embeddings_batch(victims)
            for unique_id in victims:
                del expected[unique_id]
                assert unique_id not in db

        else:
            query = generator.normal(size={dimension})
            query /= np.linalg.norm(query)
            found, scores, metadatas = db.find_most_similar(query, k=10)
            assert len(found) == len(scores) == len(metadatas) == len(set(found))
            for unique_id, score, metadata in zip(found, scores, metadatas):
                if unique_id in expected:   # only our own ids are stable to check
                    vector, stored_metadata = expected[unique_id]
                    assert score == np.float32(vector @ query).item() or abs(score - float(vector @ query)) < 1e-3, unique_id
                    assert metadata == stored_metadata, unique_id

            found, _, _ = db.find_most_similar(query, {{"worker": worker}}, k=1000)
            assert set(found) == set(expected), (len(found), len(expected))

        if compactor and step % 8 == 7:
            db.compact()

with open(f"{{path}}/expected-{{worker}}.json", "w") as handle:
    json.dump({{unique_id: [vector.tolist(), metadata] for unique_id, (vector, metadata) in expected.items()}}, handle)
'''


def run_workers(path, workers, steps, compactor=False):
    script = WORKER.format(repository=os.getcwd(), dimension=DIMENSION)
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, path, str(worker), str(steps), "1" if compactor and worker == 0 else "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for worker in range(workers)
    ]
    for worker, process in enumerate(processes):
        stdout, stderr = process.communicate(timeout=300)
        assert process.returncode == 0, f"worker {worker} failed:\n{stderr or stdout}"


def verify(path, workers):
    """The database must hold exactly what the workers recorded."""
    expected = {}
    for worker in range(workers):
        with open(f"{path}/expected-{worker}.json") as handle:
            expected.update(json.load(handle))

    with VectorDatabase(path) as db:
        assert len(db) == len(expected)
        assert {unique_id for unique_id, _ in db.iter_entries()} == set(expected)

        for unique_id, (vector, metadata) in expected.items():
            assert np.allclose(db.get_vector(unique_id), vector, atol=1e-3), unique_id
            assert db.get_metadata(unique_id) == metadata, unique_id

        # Searching still finds each vector, and the file has no gaps left.
        for unique_id in list(expected)[:20]:
            found, scores, _ = db.find_most_similar(expected[unique_id][0], k=1)
            assert scores[0] == pytest.approx(1.0, abs=1e-3), unique_id
        db.compact()
        assert db.holes() == 0
        assert db.metadata_store.row_bound() == len(expected)
    return expected


def test_processes_running_every_operation_at_once(tmp_path):
    path = str(tmp_path / "stress")
    os.makedirs(path)
    run_workers(path, workers=5, steps=400)
    expected = verify(path, workers=5)
    assert len(expected) > 500  # the run actually did something


def test_processes_while_one_of_them_compacts(tmp_path):
    """Closing the gaps while other processes read and write is safe."""
    path = str(tmp_path / "compacting")
    os.makedirs(path)
    run_workers(path, workers=5, steps=400, compactor=True)
    verify(path, workers=5)


def test_threads_running_every_operation_at_once(db):
    """The same workload through threads sharing one instance."""
    expected = {}
    guard = threading.Lock()
    errors = []

    def worker(index):
        generator = np.random.default_rng(index)
        mine = {}
        try:
            for step in range(150):
                unique_id = f"t{index}-{step}"
                vector = generator.normal(size=DIMENSION).astype(np.float32)
                vector /= np.linalg.norm(vector)
                db.store_embedding(unique_id, vector, {"thread": index})
                mine[unique_id] = vector

                if step % 3 == 0:  # update one of ours in place
                    target = f"t{index}-{step - step % 3}"
                    replacement = generator.normal(size=DIMENSION).astype(np.float32)
                    replacement /= np.linalg.norm(replacement)
                    db.update_embedding(target, replacement)
                    mine[target] = replacement
                if step % 5 == 0:  # and drop another
                    victim = f"t{index}-{step - step % 5}"
                    if victim in mine:
                        db.delete_embedding(victim)
                        del mine[victim]
                if step % 7 == 0:
                    found, scores, _ = db.find_most_similar(vector, k=5)
                    assert len(found) == len(set(found)) == len(scores)
                    for found_id, score in zip(found, scores):
                        if found_id in mine:
                            assert abs(score - float(mine[found_id] @ vector)) < 1e-3, found_id
        except Exception as error:  # pragma: no cover - only on a real failure
            errors.append(error)
        with guard:
            expected.update(mine)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert len(db) == len(expected)
    assert {unique_id for unique_id, _ in db.iter_entries()} == set(expected)
    for unique_id, vector in expected.items():
        assert np.allclose(db.get_vector(unique_id), vector, atol=1e-3), unique_id

    db.compact()
    assert db.holes() == 0
