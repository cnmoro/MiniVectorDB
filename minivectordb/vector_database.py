"""An out-of-core vector database.

Vectors are kept in a file of fixed-size rows and metadata in SQLite, so the
working set is bounded by the search chunk size rather than by the number of
indexed vectors. Everything is written as it arrives; there is no separate
"build the index" step and nothing to rebuild after a restart.

Several processes may open the same directory at once. Writers take SQLite's
write lock while they allocate row slots and store vectors, and readers check
a version counter before each search, so a database can be shared by the
workers of a web server or by pods on one network volume.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import weakref
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ._metadata_store import SLOT_REUSE_DELAY_SECONDS, MetadataStore, encode
from ._vector_store import DTYPES, VectorStore, normalize
from .rerank import autocut as autocut_scores
from .rerank import hybrid_rerank

VECTOR_FILE = "vectors.bin"
METADATA_FILE = "metadata.sqlite3"

#: Rows moved per read/write round trip while compacting.
MOVE_BATCH = 1024

#: How many times a search starts over when compaction moves entries under it.
SEARCH_ATTEMPTS = 3

SearchResult = Tuple[List[Any], List[float], List[Dict[str, Any]]]


class VectorDatabase:
    """Store embeddings with metadata and search them by cosine similarity.

    Args:
        path: directory holding the database. When omitted the database lives
            in a temporary directory that is removed on ``close()``.
        dtype: on-disk vector precision. ``"float32"`` is exact,
            ``"float16"`` halves the file, ``"int8"`` quarters it with a small
            loss of recall. Fixed on creation.
        dimension: vector size. Detected from the first insertion if omitted.
        shared_storage: set it to say whether the directory is on storage that
            other machines can reach. It is detected from the filesystem type
            when left out, and decides how SQLite journals its writes.
        slot_reuse_delay: seconds a deleted row slot is left alone before a new
            vector may take it. It has to outlast the longest search running
            anywhere against this database.
        auto_compact: close the gaps left by deletions and updates once they
            reach this share of the file, and give the disk space back. Set it
            to 0 to compact only when ``compact()`` is called; a value of 1 or
            more means the same thing, since gaps can never be all of a file.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        dtype: str = "float32",
        dimension: Optional[int] = None,
        shared_storage: Optional[bool] = None,
        slot_reuse_delay: float = SLOT_REUSE_DELAY_SECONDS,
        auto_compact: float = 0.2,
    ):
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(DTYPES)}")
        if dimension is not None and int(dimension) <= 0:
            raise ValueError(f"Vectors need at least one dimension, got {dimension}.")

        self._temporary = path is None
        self.path = path or tempfile.mkdtemp(prefix="minivectordb-")
        os.makedirs(self.path, exist_ok=True)
        # A database with nowhere to live gets a temporary directory. Tie its
        # removal to this object, so a caller who never reaches close() -- an
        # exception, a dropped reference, the interpreter shutting down --
        # does not leave it behind.
        self._cleanup = (
            weakref.finalize(self, shutil.rmtree, self.path, True) if self._temporary else None
        )
        self.lock = threading.RLock()
        self.slot_reuse_delay = slot_reuse_delay
        self.auto_compact = auto_compact

        self.metadata_store = MetadataStore(
            os.path.join(self.path, METADATA_FILE), shared_storage=shared_storage
        )
        stored_dtype = self.metadata_store.get_config("dtype")
        stored_dimension = self.metadata_store.get_config("dimension")
        self.dtype = stored_dtype or dtype
        self.dimension = stored_dimension or dimension
        # Opening an existing database takes no write lock, so readers never
        # queue behind each other or behind a writer.
        if stored_dtype is None or (self.dimension and stored_dimension is None):
            with self.metadata_store.transaction():
                self.metadata_store.set_config("dtype", self.dtype)
                if self.dimension:
                    self.metadata_store.set_config("dimension", self.dimension)

        self.vector_store: Optional[VectorStore] = None
        self._version: Optional[Tuple[int, int]] = None
        self._row_bound = 0
        self._active: Optional[np.ndarray] = None
        if self.dimension:
            self._open_vector_store()
        self._sync()

    @property
    def shared_storage(self) -> bool:
        """Whether the database is treated as reachable from other machines."""
        return self.metadata_store.shared_storage

    def _open_vector_store(self) -> None:
        # The precision on disk wins: another process may have created the
        # database with a different one.
        self.dtype = self.metadata_store.get_config("dtype") or self.dtype
        self.vector_store = VectorStore(os.path.join(self.path, VECTOR_FILE), self.dimension, self.dtype)

    def _sync(self) -> None:
        """Pick up writes made by this or any other process.

        Everything cached here is derived from two cheap values: the highest
        row slot ever allocated, and the slots that deletions have freed. Both
        are re-read only when the database version says something changed.
        """
        version = self.metadata_store.version()
        if version == self._version:
            return

        self._version = version
        if self.dimension is None:
            self.dimension = self.metadata_store.get_config("dimension")
            if self.dimension:
                self._open_vector_store()
        if self.vector_store is not None and self.metadata_store.shared_storage:
            self.vector_store.reopen()  # revalidate the network filesystem cache

        # Read the freed slots before the bound, never after: another process
        # may extend the database between the two reads, and a slot it frees
        # up there must not land outside the mask built here.
        free_rows = self.metadata_store.free_rows()
        self._row_bound = self.metadata_store.row_bound()
        if free_rows.size:
            # Freed slots still hold a row in the file; mask them out of scans.
            masked = free_rows[free_rows < self._row_bound]
            self._active = np.ones(self._row_bound, dtype=bool)
            self._active[masked] = False
        else:
            self._active = None

    # -- writing -----------------------------------------------------------

    def store_embedding(self, unique_id: Any, embedding: Sequence[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store one embedding. Raises ``ValueError`` if the id already exists."""
        self.store_embeddings_batch([unique_id], [embedding], [metadata or {}])

    def store_embeddings_batch(
        self,
        unique_ids: Sequence[Any],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Store many embeddings in one transaction."""
        unique_ids, vectors, metadatas = self._check_batch(unique_ids, embeddings, metadatas)
        if not unique_ids:
            return

        with self.lock, self.metadata_store.transaction():
            # Inside the write transaction nothing else can allocate rows, so
            # the checks below cannot go stale before the commit.
            existing = self.metadata_store.existing_uids(unique_ids)
            if existing:
                raise ValueError(f"Unique ID already exists: {sorted(existing)[0]}")
            self._insert(unique_ids, vectors, metadatas)

        with self.lock:
            self._sync()

    def upsert_embedding(
        self, unique_id: Any, embedding: Sequence[float], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store one embedding, replacing it if the id is already there."""
        self.upsert_embeddings_batch([unique_id], [embedding], [metadata or {}])

    def upsert_embeddings_batch(
        self,
        unique_ids: Sequence[Any],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Store many embeddings, replacing any id that is already there."""
        unique_ids, vectors, metadatas = self._check_batch(unique_ids, embeddings, metadatas)
        if not unique_ids:
            return

        with self.lock, self.metadata_store.transaction():
            rows = self.metadata_store.rows_for(unique_ids)
            encoded = [encode(unique_id) for unique_id in unique_ids]
            known = [index for index, uid in enumerate(encoded) if uid in rows]
            fresh = [index for index, uid in enumerate(encoded) if uid not in rows]

            if known:
                self._overwrite(
                    [rows[encoded[index]] for index in known],
                    vectors[known],
                    [metadatas[index] for index in known],
                )
            if fresh:
                self._insert(
                    [unique_ids[index] for index in fresh],
                    vectors[fresh],
                    [metadatas[index] for index in fresh],
                )

        with self.lock:
            self._sync()
        if known:
            self._compact_if_needed()

    def update_embedding(
        self,
        unique_id: Any,
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Replace the vector, the metadata, or both, of an existing id."""
        self.update_embeddings_batch(
            [unique_id],
            None if embedding is None else [embedding],
            None if metadata is None else [metadata],
        )

    def update_embeddings_batch(
        self,
        unique_ids: Sequence[Any],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Replace vectors, metadata, or both, of ids that already exist.

        Metadata is replaced whole, not merged. A new vector goes into a
        newly claimed row slot and the entry is moved to it, so a search
        reading the old slot is never shown a half-written row; the slot left
        behind is reclaimed like any other gap.
        """
        unique_ids = list(unique_ids)
        if not unique_ids:
            return
        if embeddings is None and metadatas is None:
            raise ValueError("Nothing to update: pass embeddings, metadata, or both.")

        if len({encode(unique_id) for unique_id in unique_ids}) != len(unique_ids):
            raise ValueError("Duplicate unique IDs in the same batch.")

        vectors = None
        if embeddings is not None:
            vectors = self._check_vectors(embeddings, len(unique_ids))
        if metadatas is not None:
            metadatas = [dict(meta or {}) for meta in metadatas]
            if len(metadatas) != len(unique_ids):
                raise ValueError("Metadata dictionaries must be provided for all unique IDs.")

        with self.lock, self.metadata_store.transaction():
            rows = self.metadata_store.rows_for(unique_ids)
            encoded = [encode(unique_id) for unique_id in unique_ids]
            missing = [unique_ids[index] for index, uid in enumerate(encoded) if uid not in rows]
            if missing:
                raise ValueError(f"Unique ID does not exist: {missing[0]!r}")
            self._overwrite([rows[uid] for uid in encoded], vectors, metadatas)

        with self.lock:
            self._sync()
        if vectors is not None:
            self._compact_if_needed()

    def _check_batch(self, unique_ids, embeddings, metadatas):
        """Validate a batch of ids, vectors and metadata that belong together."""
        unique_ids = list(unique_ids)
        vectors = self._check_vectors(embeddings, len(unique_ids))
        metadatas = [dict(meta or {}) for meta in (metadatas or [{}] * len(unique_ids))]
        if len(metadatas) != len(unique_ids):
            raise ValueError("Metadata dictionaries must be provided for all unique IDs.")
        if len({encode(unique_id) for unique_id in unique_ids}) != len(unique_ids):
            raise ValueError("Duplicate unique IDs in the same batch.")
        return unique_ids, vectors, metadatas

    def _check_vectors(self, embeddings, count: int) -> np.ndarray:
        """Shape a batch of embeddings, refusing anything the file cannot hold.

        Every check happens before a single byte is written: a vector of the
        wrong width would otherwise run over the row that follows it, and a
        write that has already happened cannot be rolled back with the
        metadata transaction.
        """
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise ValueError(f"Embeddings must be one vector or a list of vectors, got {vectors.ndim} dimensions.")
        if vectors.shape[0] != count:
            raise ValueError("Got a different number of unique ids and embeddings.")
        if vectors.shape[1] == 0:
            raise ValueError("Embeddings need at least one dimension.")
        expected = self.dimension if self.dimension is not None else self.metadata_store.get_config("dimension")
        if expected is not None and vectors.shape[1] != expected:
            raise ValueError(f"Expected {expected}-dimensional embeddings, got {vectors.shape[1]}.")
        return vectors

    def _insert(self, unique_ids, vectors, metadatas) -> None:
        """Claim row slots and fill them. The write transaction must be held."""
        stored_dimension = self.metadata_store.get_config("dimension")
        if stored_dimension is None:
            stored_dimension = int(vectors.shape[1])
            self.metadata_store.set_config("dimension", stored_dimension)
        if vectors.shape[1] != stored_dimension:
            raise ValueError(f"Expected {stored_dimension}-dimensional embeddings, got {vectors.shape[1]}.")
        if self.dimension is None:
            store = VectorStore(os.path.join(self.path, VECTOR_FILE), stored_dimension, self.dtype)
            self.dimension, self.vector_store = stored_dimension, store

        rows = self.metadata_store.allocate_rows(len(unique_ids), self.slot_reuse_delay)
        self.vector_store.write(np.asarray(rows, dtype=np.int64), normalize(vectors))
        self.vector_store.sync()  # the vectors must be readable before the commit
        self.metadata_store.add(list(zip(rows, unique_ids, metadatas)))

    def _overwrite(self, rows, vectors, metadatas) -> None:
        """Give existing entries a new vector, new metadata, or both.

        A new vector goes into a newly claimed slot and the entry is moved to
        it, rather than being written over where it already sits. A row's
        contents never change while an entry points at it, so a search reading
        that row cannot end up pairing one generation's score with another
        generation's document: the file is written before the metadata is
        committed, and no reader can see far enough inside a writer's
        transaction to know that a rewrite is under way.

        The write transaction must be held.
        """
        if vectors is not None:
            fresh = self.metadata_store.allocate_rows(len(rows), self.slot_reuse_delay)
            self.vector_store.write(np.asarray(fresh, dtype=np.int64), normalize(vectors))
            self.vector_store.sync()  # readable before the entry points at it
            self.metadata_store.relocate(list(zip(rows, fresh)))
            rows = fresh
        if metadatas is not None:
            self.metadata_store.replace_metadata(list(zip(rows, metadatas)))

    # -- deleting ----------------------------------------------------------

    def delete_embedding(self, unique_id: Any) -> None:
        """Delete one embedding. Raises ``ValueError`` if the id is unknown."""
        self.delete_embeddings_batch([unique_id])

    def delete_embeddings_batch(self, unique_ids: Sequence[Any]) -> None:
        """Delete many embeddings, freeing their row slots for reuse."""
        unique_ids = list(unique_ids)
        with self.lock, self.metadata_store.transaction():
            existing = self.metadata_store.existing_uids(unique_ids)
            missing = [unique_id for unique_id in unique_ids if encode(unique_id) not in existing]
            if missing:
                raise ValueError(f"Unique ID does not exist: {missing[0]!r}")

            # The rows are freed in the metadata and nothing else is done to
            # them. Blanking them here would be a write outside the write
            # lock, which another process's compaction could be filling at
            # that very moment; the free list already keeps them out of
            # searches, and compaction overwrites or truncates them away.
            rows = self.metadata_store.remove(unique_ids)

        with self.lock:
            self._sync()
        if rows:
            self._compact_if_needed()

    # -- compacting --------------------------------------------------------

    def holes(self) -> int:
        """Row slots the file holds that no entry uses."""
        with self.lock:
            self._sync()
            return max(0, self._row_bound - len(self))

    def _compact_if_needed(self) -> None:
        if self.auto_compact <= 0:
            return
        bound = self._row_bound
        if bound and (bound - len(self)) >= self.auto_compact * bound:
            self.compact()

    def compact(self) -> int:
        """Close the gaps deletions left, and give the disk space back.

        Entries at the end of the file move into the free slots in front of
        them, then the file is truncated. Returns how many rows it moved.

        Searches running at the same time stay correct: a moved entry is
        found in its new slot, and the copy left at the old one is dropped
        because no entry claims that slot any more. A search that is reading
        the file while it is compacted may miss the entries being moved.
        """
        moves = []
        with self.lock, self.metadata_store.transaction():
            if self.vector_store is None:
                return 0
            live = self.metadata_store.count()
            bound = self.metadata_store.row_bound()
            # When there are no gaps there is nothing to move, but the file may
            # still be longer than the database needs: a compaction killed
            # between its two steps leaves that space behind, and the truncate
            # below is what reclaims it.
            if bound > live:
                moves = self._plan_and_move(live)

        self._truncate_to(live)
        with self.lock:
            self._sync()
        return len(moves)

    def _plan_and_move(self, live: int) -> list:
        """Move the entries above ``live`` into the gaps below it."""
        free_rows = self.metadata_store.free_rows()
        all_rows = self.metadata_store.all_rows()
        free = free_rows[free_rows < live]
        tail = all_rows[all_rows >= live][::-1]
        width = min(free.size, tail.size)
        sources, targets = tail[:width], free[:width]

        for start in range(0, width, MOVE_BATCH):
            batch = slice(start, start + MOVE_BATCH)
            self.vector_store.write(targets[batch], self.vector_store.read(sources[batch]))
        self.vector_store.sync()

        moves = list(zip(sources.tolist(), targets.tolist()))
        self.metadata_store.move_rows(moves)
        self.metadata_store.reset_free_rows(live)
        self.metadata_store.bump_layout_version()  # searches in flight must start over
        return moves

    def _truncate_to(self, live: int) -> None:
        """Shrink the file to ``live`` rows, if that is still the right size.

        Shrinking needs the write lock again, and needs the database to still
        end where the compaction left it: another process may have appended
        rows in between, and those must not be cut off.
        """
        with self.lock, self.metadata_store.transaction():
            if self.vector_store is not None and self.metadata_store.row_bound() == live:
                self.vector_store.truncate(live)

    # -- reading -----------------------------------------------------------

    def get_vector(self, unique_id: Any) -> np.ndarray:
        """Return the stored vector for an id, as a unit vector.

        A quantized vector is a rounded unit vector, so it is normalized on
        the way out. That keeps this in step with the scores a search
        reports, which are similarities to exactly this vector.
        """
        with self.lock:
            for _ in range(SEARCH_ATTEMPTS):
                # Which row holds the entry, and what that row holds, are two
                # reads; compaction between them would answer for a row the
                # entry has just left.
                layout = self.metadata_store.layout_version()
                row = self.metadata_store.row_for(unique_id)
                if row is None:
                    raise ValueError("Unique ID does not exist.")
                vector = normalize(self.vector_store.read(np.array([row], dtype=np.int64)))[0]
                if self.metadata_store.layout_version() == layout:
                    break
            return vector

    def get_metadata(self, unique_id: Any) -> Dict[str, Any]:
        """Return the metadata document for an id."""
        with self.lock:
            metadata = self.metadata_store.metadata_for(unique_id)
            if metadata is None:
                raise ValueError("Unique ID does not exist.")
            return metadata

    def iter_entries(self) -> Iterable[Tuple[Any, Dict[str, Any]]]:
        """Iterate over ``(unique_id, metadata)`` pairs without loading them all."""
        for _, unique_id, metadata in self.metadata_store.iter_entries():
            yield unique_id, metadata

    def __len__(self) -> int:
        with self.lock:
            return self.metadata_store.count()

    def __contains__(self, unique_id: Any) -> bool:
        with self.lock:
            return self.metadata_store.contains(unique_id)

    # -- searching ---------------------------------------------------------

    def find_most_similar(
        self,
        embedding: Sequence[float],
        metadata_filter: Any = None,
        exclude_filter: Any = None,
        or_filters: Any = None,
        k: int = 5,
        autocut: bool = False,
    ) -> SearchResult:
        """Return the ``k`` closest ``(ids, scores, metadatas)`` to ``embedding``.

        Scores are cosine similarities in ``[-1, 1]``, highest first. Each
        filter accepts a dictionary or a list of dictionaries; values may be
        plain values or operator documents such as ``{"$gt": 10}``.
        """
        empty: SearchResult = ([], [], [])
        with self.lock:
            if k <= 0:
                return empty
            for _ in range(SEARCH_ATTEMPTS):
                layout = self.metadata_store.layout_version()
                found = self._search_once(embedding, metadata_filter, exclude_filter, or_filters, k)
                if found is None:
                    return empty
                if found[3].all() and self.metadata_store.layout_version() == layout:
                    break
                # Something was written where this search was reading: entries
                # moved by compaction, or rewritten in place. Take it again.
            else:
                found = self._settled(found)  # keep only what stayed still
            if found is None:
                return empty
            rows, scores, entries = found[:3]

        ids: List[Any] = []
        distances: List[float] = []
        metadatas: List[Dict[str, Any]] = []
        for position in np.argsort(-scores, kind="stable"):
            row = int(rows[position])
            unique_id, metadata = entries[row]
            ids.append(unique_id)
            distances.append(float(scores[position]))
            metadatas.append(metadata)

        if autocut:
            dropped = set(autocut_scores(distances))
            if dropped:
                ids = [value for i, value in enumerate(ids) if i not in dropped]
                distances = [value for i, value in enumerate(distances) if i not in dropped]
                metadatas = [value for i, value in enumerate(metadatas) if i not in dropped]

        return ids, distances, metadatas

    def _search_once(self, embedding, metadata_filter, exclude_filter, or_filters, k):
        """One pass: choose candidate rows, then score them from a fresh read."""
        self._sync()  # pick up anything another process wrote
        if self.vector_store is None:
            return None

        candidate_rows = self.metadata_store.filter_rows(metadata_filter, exclude_filter, or_filters)
        if candidate_rows is not None and candidate_rows.size == 0:
            return None

        query = normalize(np.asarray(embedding, dtype=np.float32))[0]
        rows, _ = self.vector_store.search(
            query,
            k=k,
            n_rows=self._row_bound,
            active=self._active,
            candidate_rows=candidate_rows,
        )
        if rows.size == 0:
            return None

        # The scan picks the candidates; their scores are then taken from a
        # fresh read of those few rows. A slot that another process gave to a
        # different vector while the scan was running is therefore scored as
        # what it holds now, never as what it held before.
        entries = self.metadata_store.fetch(rows.tolist())
        rows = np.array([row for row in rows.tolist() if row in entries], dtype=np.int64)
        if rows.size == 0:
            return None
        # Normalize what was read before scoring it: a quantized vector is a
        # rounded unit vector, so its own length is only nearly 1, and the
        # scores have to stay the cosine similarities they are documented to
        # be. For float32 this changes nothing.
        scores = normalize(self.vector_store.read(rows)) @ query

        # Read the entries again around the vector read. An entry rewritten in
        # place between the two reads would otherwise be reported with the new
        # vector's score beside the old document -- neither stale nor fresh,
        # but half of each.
        after = self.metadata_store.fetch(rows.tolist())
        settled = np.array([after.get(row) == entries[row] for row in rows.tolist()], dtype=bool)
        return rows, scores, entries, settled

    def _settled(self, found):
        """Drop results that were being written while they were read.

        Only reached when writers kept changing the rows this search was
        looking at for as long as it ran. Such a search may come back short,
        but never with a score and a document that disagree.
        """
        if found is None:
            return None
        rows, scores, entries, settled = found
        if not settled.any():
            return None
        return rows[settled], scores[settled], entries, settled[settled]

    def hybrid_rerank_results(
        self,
        sentences: Sequence[str],
        search_scores: Sequence[float],
        query: str,
        k: int = 5,
        weights: Tuple[float, float, float] = (0.80, 0.15, 0.05),
    ) -> Tuple[List[str], List[float]]:
        """Rerank search results with hashed character n-grams and fuzzy matching."""
        return hybrid_rerank(sentences, search_scores, query, k=k, weights=weights)

    @staticmethod
    def autocut_scores(scores: Sequence[float], threshold: float = 0.2) -> List[int]:
        """Indexes to drop after the largest relative drop in ``scores``."""
        return autocut_scores(scores, threshold)

    # -- lifecycle ---------------------------------------------------------

    def flush(self) -> None:
        """Force pending writes to storage. Writes are committed as they happen."""
        with self.lock:
            if self.vector_store is not None:
                self.vector_store.sync()

    def close(self) -> None:
        """Close the files. A temporary database is deleted here."""
        with self.lock:
            if self.vector_store is not None:
                self.vector_store.close()
                self.vector_store = None
            self.metadata_store.close()
            if self._cleanup is not None:
                self._cleanup()  # removes the temporary directory, once

    def __enter__(self) -> "VectorDatabase":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
