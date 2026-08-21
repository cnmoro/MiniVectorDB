"""Several processes may share one database directory.

The workers here are separate interpreters, the way a web server's worker
processes or two pods on the same volume would be.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from minivectordb import _metadata_store
from minivectordb._metadata_store import (
    LOCAL_FILESYSTEMS,
    SHARED_STORAGE_VARIABLE,
    filesystem_type,
    is_network_storage,
)
from minivectordb.vector_database import VectorDatabase

WRITER = """
import sys
import numpy as np
from minivectordb.vector_database import VectorDatabase

path, worker, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
reuse_delay = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
generator = np.random.default_rng(worker)
with VectorDatabase(path, slot_reuse_delay=reuse_delay) as db:
    for index in range(count):
        db.store_embedding(
            f"{worker}-{index}",
            generator.normal(size=8).astype(np.float32),
            {"worker": worker, "index": index},
        )
"""

MIXED = """
import sys
import numpy as np
from minivectordb.vector_database import VectorDatabase

path, worker, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
generator = np.random.default_rng(worker)
with VectorDatabase(path) as db:
    ids = [f"{worker}-{index}" for index in range(count)]
    db.store_embeddings_batch(ids, generator.normal(size=(count, 8)), [{"worker": worker}] * count)
    db.delete_embeddings_batch(ids[: count // 2])
    for _ in range(5):
        found, scores, _ = db.find_most_similar(generator.normal(size=8), k=5)
        assert len(found) == len(set(found)) == len(scores)
    assert len(db.find_most_similar(np.ones(8), {"worker": worker}, k=count)[0]) == count - count // 2
"""

UPDATER = """
import sys
import numpy as np
from minivectordb.vector_database import VectorDatabase

path = sys.argv[1]
with VectorDatabase(path) as db:
    for unique_id in sys.argv[2:]:
        db.update_embedding(unique_id, np.eye(8)[3], {"updated": True})
"""

GAP_DELETER = """
import sys
from minivectordb.vector_database import VectorDatabase

path = sys.argv[1]
with VectorDatabase(path, auto_compact=0) as db:   # leave the gaps behind
    db.delete_embeddings_batch(sys.argv[2:])
"""

COMPACTOR = """
import sys
from minivectordb.vector_database import VectorDatabase

path = sys.argv[1]
with VectorDatabase(path, auto_compact=0) as db:
    print(db.compact())
"""

DELETER = """
import sys
from minivectordb.vector_database import VectorDatabase

path = sys.argv[1]
with VectorDatabase(path) as db:
    db.delete_embeddings_batch(sys.argv[2:])
"""


def expected_vector(worker, index):
    """The vector worker ``worker`` stores at ``index``, normalized."""
    generator = np.random.default_rng(worker)
    for _ in range(index):
        generator.normal(size=8)
    vector = generator.normal(size=8).astype(np.float32)
    return vector / np.linalg.norm(vector)


def run(script, *args, wait=True):
    environment = dict(os.environ, PYTHONPATH=os.getcwd())
    process = subprocess.Popen(
        [sys.executable, "-c", script, *[str(arg) for arg in args]],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not wait:
        return process
    stdout, stderr = process.communicate(timeout=120)
    assert process.returncode == 0, stderr or stdout
    process.stdout_text = stdout
    return process


def test_concurrent_writers_do_not_corrupt_each_other(tmp_path):
    path = str(tmp_path / "shared")
    workers, per_worker = 5, 40

    processes = [run(WRITER, path, worker, per_worker, wait=False) for worker in range(workers)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stderr or stdout

    with VectorDatabase(path) as db:
        assert len(db) == workers * per_worker
        # Every id kept the vector its own writer stored: no writer overwrote
        # a row slot handed to another one.
        for worker in range(workers):
            for index in range(per_worker):
                stored = db.get_vector(f"{worker}-{index}")
                assert np.allclose(stored, expected_vector(worker, index), atol=1e-6)
        # No slot was handed out twice, so the file has exactly as many rows.
        assert db.metadata_store.row_bound() == workers * per_worker


def test_writes_from_another_process_are_visible(persistent_db):
    persistent_db.store_embedding("local", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert len(persistent_db) == 1

    run(WRITER, persistent_db.path, 7, 3)

    # The same open instance sees them, without reopening the database.
    assert len(persistent_db) == 4
    ids, _, _ = persistent_db.find_most_similar(expected_vector(7, 1), k=1)
    assert ids == ["7-1"]

    ids, _, _ = persistent_db.find_most_similar(np.ones(8), {"worker": 7}, k=10)
    assert sorted(ids) == ["7-0", "7-1", "7-2"]


def test_deletes_from_another_process_are_visible(persistent_db):
    run(WRITER, persistent_db.path, 2, 4)
    assert len(persistent_db) == 4

    run(DELETER, persistent_db.path, "2-1", "2-3")

    assert len(persistent_db) == 2
    ids, _, _ = persistent_db.find_most_similar(expected_vector(2, 1), k=10)
    assert sorted(ids) == ["2-0", "2-2"]
    assert "2-1" not in persistent_db


def test_updates_from_another_process_are_visible(persistent_db):
    run(WRITER, persistent_db.path, 2, 4)
    assert len(persistent_db) == 4

    run(UPDATER, persistent_db.path, "2-1", "2-2")

    assert len(persistent_db) == 4  # an update is not an insert
    assert persistent_db.holes() == 0  # and leaves no gap behind
    assert np.allclose(persistent_db.get_vector("2-1"), np.eye(8)[3])
    assert persistent_db.get_metadata("2-1") == {"updated": True}

    ids, scores, _ = persistent_db.find_most_similar(np.eye(8)[3], {"updated": True}, k=10)
    assert sorted(ids) == ["2-1", "2-2"]
    assert scores[0] == pytest.approx(1.0, abs=1e-6)


def test_compaction_in_another_process_is_picked_up(persistent_db):
    run(WRITER, persistent_db.path, 3, 10)
    run(GAP_DELETER, persistent_db.path, *[f"3-{index}" for index in range(5)])
    assert len(persistent_db) == 5
    assert persistent_db.holes() == 5

    moved = run(COMPACTOR, persistent_db.path).stdout_text
    assert int(moved) > 0

    assert persistent_db.holes() == 0
    assert len(persistent_db) == 5
    ids, _, _ = persistent_db.find_most_similar(expected_vector(3, 7), k=1)
    assert ids == ["3-7"]


def test_slots_freed_in_one_process_are_reused_by_another(persistent_db):
    run(WRITER, persistent_db.path, 1, 4)
    run(DELETER, persistent_db.path, "1-0", "1-1")
    run(WRITER, persistent_db.path, 5, 2, 0)  # reuse freed slots at once

    assert len(persistent_db) == 4
    assert persistent_db.metadata_store.row_bound() == 4  # the file did not grow
    ids, _, _ = persistent_db.find_most_similar(expected_vector(5, 0), k=1)
    assert ids == ["5-0"]


def test_reader_keeps_working_while_a_writer_runs(persistent_db):
    run(WRITER, persistent_db.path, 3, 5)
    query = np.ones(8)

    writer = run(WRITER, persistent_db.path, 4, 30, wait=False)
    seen = []
    while writer.poll() is None:
        ids, scores, _ = persistent_db.find_most_similar(query, k=5)
        assert len(ids) == len(scores) == len(set(ids))
        seen.append(len(persistent_db))
    stdout, stderr = writer.communicate(timeout=120)
    assert writer.returncode == 0, stderr or stdout

    assert len(persistent_db) == 35
    assert max(seen or [0]) <= 35  # never reports rows that are not committed


def test_mixed_workload_across_processes(tmp_path):
    """Writes, deletes and searches from five processes at the same time."""
    path = str(tmp_path / "mixed")
    workers, per_worker = 5, 40

    processes = [run(MIXED, path, worker, per_worker, wait=False) for worker in range(workers)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=180)
        assert process.returncode == 0, stderr or stdout

    with VectorDatabase(path) as db:
        assert len(db) == workers * (per_worker - per_worker // 2)
        for worker in range(workers):
            ids, _, _ = db.find_most_similar(np.ones(8), {"worker": worker}, k=per_worker)
            assert len(ids) == per_worker - per_worker // 2
            for unique_id in ids:
                assert db.get_vector(unique_id).shape == (8,)


def test_a_killed_writer_leaves_the_database_usable(tmp_path):
    """A pod that dies mid-write must not corrupt the shared directory."""
    path = str(tmp_path / "killed")
    run(WRITER, path, 1, 5)

    victim = run(WRITER, path, 2, 4000, wait=False)
    while True:
        with VectorDatabase(path) as watcher:
            if len(watcher) > 10:
                break
    victim.kill()
    victim.communicate(timeout=60)

    with VectorDatabase(path) as db:
        # Whatever the writer committed is intact and readable.
        assert len(db) > 10
        for worker, index in [(1, 0), (1, 4)]:
            assert np.allclose(db.get_vector(f"{worker}-{index}"), expected_vector(worker, index), atol=1e-6)
        ids, _, _ = db.find_most_similar(expected_vector(1, 3), k=1)
        assert ids == ["1-3"]

        db.store_embedding("after-crash", np.eye(8)[7])
        assert "after-crash" in db


CHURN = """
import sys
import numpy as np
from minivectordb.vector_database import VectorDatabase

path, rounds = sys.argv[1], int(sys.argv[2])
generator = np.random.default_rng(99)
with VectorDatabase(path) as db:
    for round_number in range(rounds):
        ids = [f"churn-{round_number}-{i}" for i in range(50)]
        db.store_embeddings_batch(ids, generator.normal(size=(50, 8)))
        db.delete_embeddings_batch(ids)
"""


def test_scores_stay_true_while_another_process_recycles_slots(persistent_db):
    """The score of a result always belongs to the vector that result holds.

    A deleted slot handed to a new vector must never leave a search reporting
    the old vector's score against the new vector's id.
    """
    run(WRITER, persistent_db.path, 8, 60)
    churn = run(CHURN, persistent_db.path, 60, wait=False)

    query = np.ones(8) / np.sqrt(8)
    checked = 0
    while churn.poll() is None:
        ids, scores, _ = persistent_db.find_most_similar(query, k=10)
        for unique_id, score in zip(ids, scores):
            try:
                vector = persistent_db.get_vector(unique_id)
            except ValueError:
                continue  # deleted between the search and this check
            assert score == pytest.approx(float(vector @ query), abs=1e-3), unique_id
            checked += 1
    stdout, stderr = churn.communicate(timeout=120)
    assert churn.returncode == 0, stderr or stdout
    assert checked > 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_database_opened_before_fork_is_safe(tmp_path):
    """A preloading server forks after opening the database."""
    db = VectorDatabase(str(tmp_path / "forked"))
    db.store_embedding("parent-0", np.eye(8)[0])

    child = os.fork()
    if child == 0:  # pragma: no cover - runs in the child process
        status = 0
        try:
            db.store_embedding("child-0", np.eye(8)[1])
            assert len(db) == 2
        except BaseException:
            status = 1
        finally:
            os._exit(status)

    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0

    db.store_embedding("parent-1", np.eye(8)[2])
    assert len(db) == 3
    ids, _, _ = db.find_most_similar(np.eye(8)[1], k=1)
    assert ids == ["child-0"]
    db.close()


def test_local_storage_uses_a_write_ahead_log(persistent_db):
    assert persistent_db.shared_storage is False
    assert persistent_db.metadata_store.journal_mode == "WAL"


def test_shared_storage_avoids_the_write_ahead_log(tmp_path):
    with VectorDatabase(str(tmp_path / "nfs"), shared_storage=True) as db:
        assert db.shared_storage is True
        assert db.metadata_store.journal_mode == "TRUNCATE"
        mode = db.metadata_store.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "truncate"

        db.store_embedding(1, [1.0, 0.0])
        ids, _, _ = db.find_most_similar([1.0, 0.0], k=1)
        assert ids == [1]


def test_filesystem_detection(tmp_path):
    assert filesystem_type(str(tmp_path)) in LOCAL_FILESYSTEMS
    assert is_network_storage(str(tmp_path)) is False


@pytest.mark.parametrize("filesystem", ["ext4", "xfs", "btrfs", "zfs", "overlay", "tmpfs"])
def test_disks_of_a_single_machine_are_local(monkeypatch, tmp_path, filesystem):
    monkeypatch.setattr(_metadata_store, "filesystem_type", lambda path: filesystem)
    assert is_network_storage(str(tmp_path)) is False
    with VectorDatabase(str(tmp_path / filesystem)) as db:
        assert db.metadata_store.journal_mode == "WAL"


@pytest.mark.parametrize(
    "filesystem",
    [
        "nfs4",              # AWS EFS, GCP Filestore, NFS-backed RWX claims
        "cifs",              # Azure Files
        "ceph",              # OpenShift Data Foundation
        "glusterfs",
        "fuse.juicefs",      # a FUSE driver we never listed
        "csi-vendor-x",      # a storage driver naming itself
        "virtiofs",
        "9p",
    ],
)
def test_anything_not_a_local_disk_is_treated_as_shared(monkeypatch, tmp_path, filesystem):
    """An unrecognized volume must not be guessed to be local."""
    monkeypatch.setattr(_metadata_store, "filesystem_type", lambda path: filesystem)
    assert is_network_storage(str(tmp_path)) is True
    with VectorDatabase(str(tmp_path / "volume")) as db:
        assert db.shared_storage is True
        assert db.metadata_store.journal_mode == "TRUNCATE"

        db.store_embedding(1, [1.0, 0.0])
        assert db.find_most_similar([1.0, 0.0], k=1)[0] == [1]


def test_unreadable_mount_information_falls_back_safely(monkeypatch, tmp_path):
    monkeypatch.setattr(_metadata_store, "filesystem_type", lambda path: "")
    # On Linux, where shared volumes are deployed, not knowing means shared.
    monkeypatch.setattr(os.path, "exists", lambda path: True)
    assert is_network_storage(str(tmp_path)) is True
    # Where there is no mount information at all there is no cluster either.
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    assert is_network_storage(str(tmp_path)) is False


@pytest.mark.parametrize("value,shared", [("1", True), ("true", True), ("0", False), ("no", False)])
def test_environment_variable_overrides_detection(monkeypatch, tmp_path, value, shared):
    """Operators can correct the decision without touching the application."""
    monkeypatch.setattr(_metadata_store, "filesystem_type", lambda path: "nfs4")
    monkeypatch.setenv(SHARED_STORAGE_VARIABLE, value)
    with VectorDatabase(str(tmp_path / "override")) as db:
        assert db.shared_storage is shared


def test_the_decision_is_recorded_for_every_process(monkeypatch, tmp_path):
    path = str(tmp_path / "recorded")
    with VectorDatabase(path, shared_storage=True) as db:
        db.store_embedding(1, [1.0, 0.0])

    # A later process detects a local disk, but the database already knows
    # better, so both journal the same way.
    monkeypatch.setattr(_metadata_store, "filesystem_type", lambda p: "ext4")
    with VectorDatabase(path) as reopened:
        assert reopened.shared_storage is True
        assert reopened.metadata_store.journal_mode == "TRUNCATE"
        assert len(reopened) == 1


def test_an_explicit_argument_corrects_a_recorded_decision(tmp_path):
    path = str(tmp_path / "corrected")
    with VectorDatabase(path, shared_storage=False) as db:
        db.store_embedding(1, [1.0, 0.0])
        assert db.metadata_store.journal_mode == "WAL"

    with VectorDatabase(path, shared_storage=True) as fixed:
        assert fixed.metadata_store.journal_mode == "TRUNCATE"
        assert len(fixed) == 1

    with VectorDatabase(path) as reopened:
        assert reopened.shared_storage is True  # the correction stuck


def test_a_database_held_in_wal_mode_refuses_to_open_as_shared(tmp_path):
    path = str(tmp_path / "conflict")
    with VectorDatabase(path, shared_storage=False) as holder:
        holder.store_embedding(1, [1.0, 0.0])
        with pytest.raises(RuntimeError, match="write-ahead log"):
            VectorDatabase(path, shared_storage=True)
