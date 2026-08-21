"""Out-of-core vector storage.

Vectors live in a single file on disk, one fixed-size row per slot. Reads go
through explicit positional I/O into reusable buffers rather than a memory
map, so resident memory stays bounded by the chunk size no matter how large
the file grows.

Positional reads and writes also make the file safe to share between
processes: each writer touches only the row slots it was given, and the file
only ever grows. Nothing about the file is cached in memory except its size,
which is re-read whenever the database changes.
"""
from __future__ import annotations

import os
import threading
from typing import Iterator, List, Optional, Tuple

import numpy as np

#: Supported on-disk precisions. Vectors are always read back as float32.
DTYPES = {
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
    "int8": np.dtype(np.int8),
}

#: Vectors are unit-normalized before writing, so int8 rows scale by 127.
INT8_SCALE = 127.0

#: Bytes held in memory per chunk while scanning. This is the memory ceiling.
CHUNK_BYTES = 8 << 20

MIN_CAPACITY = 1024

#: Above ``1 / GATHER_RATIO`` of the store, a masked full scan beats gathering
#: scattered rows one run at a time.
GATHER_RATIO = 20

_HAS_POSITIONAL_IO = hasattr(os, "pread") and hasattr(os, "pwrite")


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so that an inner product equals cosine similarity."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms > 0)
    return vectors


def _contiguous_runs(rows: np.ndarray) -> List[Tuple[int, int]]:
    """Split sorted row numbers into ``(start, length)`` runs of adjacent rows."""
    if rows.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    return [(int(part[0]), int(part.size)) for part in np.split(rows, breaks)]


class VectorStore:
    """A growable, disk-resident matrix of ``dim``-sized rows."""

    def __init__(self, path: str, dim: int, dtype: str = "float32"):
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(DTYPES)}")
        if int(dim) <= 0:
            raise ValueError(f"Vectors need at least one dimension, got {dim}.")
        self.path = path
        self.dim = int(dim)
        self.dtype_name = dtype
        self.dtype = DTYPES[dtype]
        self.row_bytes = self.dim * self.dtype.itemsize
        self._io_lock = threading.Lock()
        self._pid = os.getpid()
        self._handle = self._open()

    def _open(self):
        if not os.path.exists(self.path):
            with open(self.path, "ab"):
                pass
        return open(self.path, "r+b", buffering=0)

    @property
    def _file(self):
        """The file handle, reopened if this process inherited it from a fork."""
        if os.getpid() != self._pid:
            self._pid = os.getpid()
            self._handle = self._open()
        return self._handle

    def reopen(self) -> None:
        """Reopen the file so a shared filesystem revalidates its cache.

        Network filesystems guarantee fresh data on open (close-to-open
        consistency), so a reader that another process has written to calls
        this before reading again.
        """
        with self._io_lock:
            handle, self._handle = self._handle, self._open()
            self._pid = os.getpid()
            handle.close()

    @property
    def capacity(self) -> int:
        """Row slots the file currently has room for, read from the filesystem."""
        return os.fstat(self._file.fileno()).st_size // self.row_bytes

    # -- positional I/O ----------------------------------------------------

    def _pread(self, offset: int, buffer: memoryview) -> None:
        if _HAS_POSITIONAL_IO:
            read = os.preadv(self._file.fileno(), [buffer], offset)
        else:  # pragma: no cover - Windows fallback
            with self._io_lock:
                self._file.seek(offset)
                read = self._file.readinto(buffer)
        if read < len(buffer):
            # Past the end of the file: a slot no writer has reached yet.
            buffer[read:] = bytes(len(buffer) - read)

    def _pwrite(self, offset: int, data: memoryview) -> None:
        if _HAS_POSITIONAL_IO:
            os.pwritev(self._file.fileno(), [data], offset)
        else:  # pragma: no cover - Windows fallback
            with self._io_lock:
                self._file.seek(offset)
                self._file.write(data)

    # -- file management ---------------------------------------------------

    def reserve(self, rows_needed: int) -> None:
        """Make sure at least ``rows_needed`` slots exist, growing geometrically.

        The file is only ever extended, and the caller holds the database
        write lock, so concurrent writers cannot truncate each other's rows.
        """
        capacity = self.capacity
        if rows_needed <= capacity:
            return
        os.ftruncate(self._file.fileno(), max(rows_needed, capacity * 2, MIN_CAPACITY) * self.row_bytes)

    def truncate(self, rows: int) -> None:
        """Shrink the file to ``rows`` slots, giving the space back.

        Only ever called by compaction, which holds the database write lock
        and has already moved every live row below ``rows``.
        """
        if rows * self.row_bytes < os.fstat(self._file.fileno()).st_size:
            os.ftruncate(self._file.fileno(), rows * self.row_bytes)

    def sync(self) -> None:
        """Flush written rows to the storage device.

        Called before the metadata is committed, so a reader that can see an
        entry can always read the vector behind it.
        """
        self._file.flush()
        os.fsync(self._file.fileno())

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        if not self._handle.closed:
            if os.getpid() == self._pid:
                self._handle.flush()
            self._handle.close()

    # -- encoding ----------------------------------------------------------

    def _encode(self, vectors: np.ndarray) -> np.ndarray:
        if self.dtype_name == "int8":
            return np.clip(np.rint(vectors * INT8_SCALE), -127, 127).astype(np.int8)
        return np.ascontiguousarray(vectors, dtype=self.dtype)

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        decoded = raw.astype(np.float32)
        if self.dtype_name == "int8":
            decoded /= INT8_SCALE
        return decoded

    @property
    def value_scale(self) -> float:
        """What a stored number has to be multiplied by to mean what it says."""
        return 1.0 / INT8_SCALE if self.dtype_name == "int8" else 1.0

    # -- reads and writes --------------------------------------------------

    def write(self, rows: np.ndarray, vectors: np.ndarray) -> None:
        """Write already-normalized ``vectors`` into the given row slots.

        The shapes are checked here as well as by the caller: a vector of the
        wrong width would be written at the right offset with the wrong
        length, running over the rows that follow it.
        """
        rows = self._check_rows(rows)
        if rows.size == 0:
            return
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape != (rows.size, self.dim):
            raise ValueError(
                f"Expected vectors of shape {(rows.size, self.dim)}, got {vectors.shape}."
            )
        self.reserve(int(rows.max()) + 1)
        order = np.argsort(rows, kind="stable")
        encoded = self._encode(vectors)[order]
        position = 0
        for start, length in _contiguous_runs(rows[order]):
            block = np.ascontiguousarray(encoded[position : position + length])
            self._pwrite(start * self.row_bytes, memoryview(block).cast("B"))
            position += length

    def read(self, rows: np.ndarray) -> np.ndarray:
        """Read the given rows, in the order asked, as a float32 matrix."""
        rows = self._check_rows(rows)
        if rows.size == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        order = np.argsort(rows, kind="stable")
        raw = np.empty((rows.size, self.dim), dtype=self.dtype)
        view = memoryview(raw).cast("B")
        position = 0
        for start, length in _contiguous_runs(rows[order]):
            byte_start = position * self.row_bytes
            self._pread(start * self.row_bytes, view[byte_start : byte_start + length * self.row_bytes])
            position += length

        decoded = self._decode(raw)
        return decoded[np.argsort(order, kind="stable")]

    def _check_rows(self, rows: np.ndarray) -> np.ndarray:
        """Row numbers must be whole and non-negative to be byte offsets."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and int(rows.min()) < 0:
            raise ValueError(f"Row numbers cannot be negative, got {int(rows.min())}.")
        return rows

    @property
    def chunk_rows(self) -> int:
        return max(1, CHUNK_BYTES // self.row_bytes)

    def iter_chunks(self, n_rows: int) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(start_row, block)`` blocks covering ``[0, n_rows)``.

        Blocks come back in the dtype they are stored in, undecoded: widening
        a chunk to float32 costs more than the arithmetic that follows it, and
        for a float32 store it is a copy of the whole chunk for nothing. Scale
        the result of whatever you compute by ``value_scale`` instead.

        The same buffer is reused for every chunk, so each block is only valid
        until the next iteration.
        """
        step = self.chunk_rows
        buffer = np.empty((step, self.dim), dtype=self.dtype)
        view = memoryview(buffer).cast("B")
        for start in range(0, n_rows, step):
            length = min(step, n_rows - start)
            self._pread(start * self.row_bytes, view[: length * self.row_bytes])
            yield start, buffer[:length]

    # -- search ------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int,
        n_rows: int,
        active: Optional[np.ndarray] = None,
        candidate_rows: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the ``k`` best ``(rows, scores)`` by cosine similarity.

        ``candidate_rows`` restricts the search to those row numbers; without
        it the whole store is streamed and ``active`` masks out free slots.
        """
        if k <= 0 or n_rows <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        query = normalize(query)[0]
        best_rows = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float32)

        def keep(rows: np.ndarray, scores: np.ndarray) -> None:
            """Merge a chunk's results into the running best k."""
            nonlocal best_rows, best_scores
            if scores.size > k:
                top = np.argpartition(-scores, k - 1)[:k]
                rows, scores = rows[top], scores[top]
            best_rows = np.concatenate((best_rows, rows))
            best_scores = np.concatenate((best_scores, scores.astype(np.float32, copy=False)))
            if best_scores.size > 2 * k:
                # Trim as we go: holding every chunk's candidates until the end
                # would make a large k cost as much memory as the collection.
                top = np.argpartition(-best_scores, k - 1)[:k]
                best_rows, best_scores = best_rows[top], best_scores[top]

        if candidate_rows is not None:
            candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
            if candidate_rows.size * GATHER_RATIO >= n_rows:
                # Too many scattered rows to gather one by one: stream the
                # whole store instead and mask the rows we do not want.
                active = np.zeros(n_rows, dtype=bool)
                active[candidate_rows] = True
                candidate_rows = None

        scale = self.value_scale
        if candidate_rows is None:
            for start, block in self.iter_chunks(n_rows):
                # Multiply in float32 straight out of the stored dtype, and
                # put the scale on the scores: one pass over the chunk instead
                # of three.
                scores = np.matmul(block, query, dtype=np.float32)
                if scale != 1.0:
                    scores *= scale
                rows = np.arange(start, start + scores.size, dtype=np.int64)
                if active is not None:
                    mask = active[start : start + scores.size]
                    rows, scores = rows[mask], scores[mask]
                if scores.size:
                    keep(rows, scores)
        else:
            candidate_rows = np.sort(candidate_rows)
            step = self.chunk_rows
            for start in range(0, candidate_rows.size, step):
                rows = candidate_rows[start : start + step]
                keep(rows, self.read(rows) @ query)

        order = np.argsort(-best_scores, kind="stable")[:k]
        return best_rows[order], best_scores[order]
