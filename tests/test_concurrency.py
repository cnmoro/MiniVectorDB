"""The database is guarded by one lock, so threads may share an instance."""
import threading
import uuid

import numpy as np
import pytest

from tests.conftest import random_vectors

THREADS = 8
PER_THREAD = 50


def run_threads(target, count=THREADS):
    errors = []

    def wrapper(index):
        try:
            target(index)
        except Exception as error:  # pragma: no cover - only on a real failure
            errors.append(error)

    threads = [threading.Thread(target=wrapper, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors


def test_concurrent_writes(db):
    vectors = random_vectors(THREADS * PER_THREAD, dimension=8)

    def writer(index):
        start = index * PER_THREAD
        ids = [f"{index}-{i}" for i in range(PER_THREAD)]
        db.store_embeddings_batch(ids, vectors[start : start + PER_THREAD], [{"thread": index}] * PER_THREAD)

    run_threads(writer)
    assert len(db) == THREADS * PER_THREAD
    ids, _, _ = db.find_most_similar(vectors[0], {"thread": 0}, k=PER_THREAD)
    assert len(ids) == PER_THREAD


def test_concurrent_single_writes(db):
    def writer(index):
        for _ in range(PER_THREAD):
            db.store_embedding(str(uuid.uuid4()), np.random.rand(8), {"thread": index})

    run_threads(writer)
    assert len(db) == THREADS * PER_THREAD


def test_concurrent_reads_and_writes(db):
    ids = [f"seed-{i}" for i in range(200)]
    db.store_embeddings_batch(ids, random_vectors(200, dimension=8), [{"kind": "seed"}] * 200)

    def worker(index):
        for step in range(20):
            if index % 2:
                db.store_embedding(f"{index}-{step}", np.random.rand(8), {"kind": "new"})
            else:
                found, scores, _ = db.find_most_similar(np.random.rand(8), k=5)
                assert len(found) == len(scores) <= 5

    run_threads(worker)
    assert len(db) == 200 + (THREADS // 2) * 20


def test_concurrent_deletes(db):
    ids = list(range(THREADS * PER_THREAD))
    db.store_embeddings_batch(ids, random_vectors(len(ids), dimension=8))

    def deleter(index):
        db.delete_embeddings_batch(ids[index * PER_THREAD : (index + 1) * PER_THREAD])

    run_threads(deleter)
    assert len(db) == 0
    assert db.find_most_similar(np.random.rand(8), k=5) == ([], [], [])


def test_duplicate_ids_across_threads(db):
    """Exactly one writer wins when threads race on the same id."""
    accepted = []
    lock = threading.Lock()

    def writer(index):
        try:
            db.store_embedding("shared", np.random.rand(8))
        except ValueError:
            return
        with lock:
            accepted.append(index)

    run_threads(writer)
    assert len(accepted) == 1
    assert len(db) == 1
