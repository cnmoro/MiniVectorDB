"""SQLite-backed metadata storage and filtering.

Identifiers, metadata documents and the inverted index all live in a single
SQLite file, so metadata never has to be held in memory. Filters are resolved
to row numbers by indexed queries and combined with set algebra.

SQLite is also what makes several processes able to share one database
directory: row slots are handed out inside a write transaction, so its file
locking serializes writers without any lock of our own.
"""
from __future__ import annotations

import functools
import json
import os
import sqlite3
import threading
import weakref
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS entries (
    row INTEGER PRIMARY KEY,
    uid TEXT NOT NULL UNIQUE,
    meta TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta_index (
    key TEXT NOT NULL,
    value TEXT,
    num REAL,
    row INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS free_rows (row INTEGER PRIMARY KEY, freed_at REAL NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS meta_index_kv ON meta_index (key, value);
CREATE INDEX IF NOT EXISTS meta_index_kn ON meta_index (key, num);
CREATE INDEX IF NOT EXISTS meta_index_row ON meta_index (row);
"""

#: Filesystems that only one machine can mount, and where SQLite's
#: write-ahead log is therefore safe. Anything else -- a network filesystem, a
#: FUSE driver, a volume type not listed here -- is assumed to be reachable
#: from more than one machine, because guessing "local" is the answer that
#: corrupts data when it is wrong. Storage drivers report names this list
#: cannot hope to enumerate, so the unknown case has to be the safe one.
LOCAL_FILESYSTEMS = frozenset({
    "apfs", "bcachefs", "btrfs", "exfat", "ext2", "ext3", "ext4", "f2fs",
    "hfs", "hfsplus", "jfs", "msdos", "nilfs2", "ntfs", "ntfs3", "overlay",
    "ramfs", "reiserfs", "tmpfs", "ufs", "vfat", "xfs", "zfs",
})

#: Environment variable that overrides the detection, for deployments where
#: the application code cannot be changed.
SHARED_STORAGE_VARIABLE = "MINIVECTORDB_SHARED_STORAGE"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})

#: How long a writer waits for another process to release the database lock.
BUSY_TIMEOUT_SECONDS = 30.0

#: How long a freed row slot is left alone before another insertion may claim
#: it. A search that is already reading the file has to finish within this
#: time, or it could score a slot that has been given to a different vector.
SLOT_REUSE_DELAY_SECONDS = 300.0

OPERATORS = ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$contains", "$exists")

_COMPARISONS = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


def synchronized(method):
    """Serialize connection use, so threads never interleave statements."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)

    return wrapper


def encode(value: Any) -> str:
    """JSON-encode a value so equality and ordering are type-aware."""
    if isinstance(value, float) and value == 0.0:
        value = 0.0  # -0.0 and 0.0 are the same number; keep one spelling
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError as error:
        raise ValueError(
            f"Ids and metadata values must be JSON-encodable, got {type(value).__name__}: {error}"
        ) from error


def decode(value: str) -> Any:
    return json.loads(value)


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def filesystem_type(path: str) -> str:
    """The filesystem type ``path`` sits on, or ``""`` when it cannot be told."""
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as mounts:
            entries = [line.split() for line in mounts]
    except OSError:  # pragma: no cover - not Linux
        return ""

    target = os.path.realpath(path)
    best_point, best_type = "", ""
    for fields in entries:
        if "-" not in fields:
            continue  # pragma: no cover - unexpected mountinfo format
        separator = fields.index("-")
        mount_point, mount_type = fields[4], fields[separator + 1]
        if (target == mount_point or target.startswith(mount_point.rstrip("/") + "/")) and len(
            mount_point
        ) >= len(best_point):
            best_point, best_type = mount_point, mount_type
    return best_type


def environment_override() -> Optional[bool]:
    """The value of ``MINIVECTORDB_SHARED_STORAGE``, or ``None`` if unset."""
    value = os.environ.get(SHARED_STORAGE_VARIABLE, "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return None


def is_network_storage(path: str) -> bool:
    """Whether ``path`` may be reachable from more than one machine.

    A filesystem is treated as local only when it is one that a single
    machine mounts on its own disks. A mount this cannot identify -- a CSI
    driver with a name of its own, a FUSE volume, anything on a system
    without ``/proc`` mount information to read -- counts as shared, so the
    cost of not knowing is some write speed rather than a corrupt database.
    """
    filesystem = filesystem_type(path)
    if not filesystem:
        return not _looks_like_a_single_machine()
    return filesystem not in LOCAL_FILESYSTEMS


def _looks_like_a_single_machine() -> bool:
    """Whether a path of unknown filesystem is worth treating as local.

    Mount information can only be read on Linux, which is where shared
    volumes are deployed. On a developer machine running macOS or Windows
    there is nothing to read and no cluster either.
    """
    return not os.path.exists("/proc/self/mountinfo")


class MetadataStore:
    """Row identifiers, metadata documents and the metadata inverted index.

    The connection belongs to the process that opened it. After a fork the
    child reconnects on first use, so a database opened before ``fork()`` (a
    preloading web server, for example) stays safe to use in the workers.
    """

    def __init__(self, path: str, shared_storage: Optional[bool] = None):
        self.path = path
        self.lock = threading.RLock()
        self._requested_shared = environment_override() if shared_storage is None else shared_storage
        self.shared_storage = False
        self.journal_mode = "DELETE"
        self._local_version = 0
        self._pid = os.getpid()
        self._conn = self._connect()
        # Close the connection even if the database is dropped without being
        # closed. The list is what the finalizer sees, so it can be repointed
        # after a fork without the finalizer holding on to this object.
        self._open_connections = [self._conn]
        weakref.finalize(self, _close_all, self._open_connections)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=BUSY_TIMEOUT_SECONDS,
            isolation_level=None,  # transactions are opened explicitly
        )
        connection.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_SECONDS * 1000)}")
        connection.executescript(SCHEMA)
        self._agree_on_storage_mode(connection)
        try:
            # Changing the mode needs every other connection to be closed.
            connection.execute(f"PRAGMA journal_mode = {self.journal_mode}")
        except sqlite3.OperationalError:
            pass  # someone else is using the database; the mode is checked below
        connection.execute("PRAGMA synchronous = FULL" if self.shared_storage else "PRAGMA synchronous = NORMAL")

        self.journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0].upper()
        if self.shared_storage and self.journal_mode == "WAL":
            connection.close()
            raise RuntimeError(
                f"{self.path} is on shared storage but is open in write-ahead log mode, "
                "which does not work between machines. Close every process using this "
                "database, then open it again."
            )
        return connection

    def _agree_on_storage_mode(self, connection: sqlite3.Connection) -> None:
        """Settle whether this database is on shared storage, once, for everyone.

        Processes sharing a database must journal the same way, so the answer
        is recorded in the database itself. Detection only fills it in the
        first time; an explicit argument or the environment variable can
        correct a database that was created with the wrong answer.
        """
        row = connection.execute("SELECT value FROM config WHERE key = 'shared_storage'").fetchone()
        stored = decode(row[0]) if row else None

        if self._requested_shared is not None:
            self.shared_storage = self._requested_shared
        elif stored is not None:
            self.shared_storage = stored
        else:
            self.shared_storage = is_network_storage(os.path.dirname(self.path) or ".")

        if stored != self.shared_storage:
            connection.execute(
                "INSERT INTO config (key, value) VALUES ('shared_storage', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (encode(self.shared_storage),),
            )
        # A write-ahead log is faster but is confined to one machine.
        self.journal_mode = "TRUNCATE" if self.shared_storage else "WAL"

    @property
    def conn(self) -> sqlite3.Connection:
        """The connection, reopened if this process inherited it from a fork."""
        if os.getpid() != self._pid:
            self._pid = os.getpid()
            self._local_version = 0
            self._conn.close()  # only drops this process's handle, not the parent's
            self._conn = self._connect()
            self._open_connections[:] = [self._conn]
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Hold the cross-process write lock for the duration of the block."""
        with self.lock:
            connection = self.conn
            connection.execute("BEGIN IMMEDIATE")  # waits for other processes
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")
                self._local_version += 1

    def version(self) -> Tuple[int, int]:
        """A value that changes whenever this or another process writes.

        ``PRAGMA data_version`` only reports commits from other connections,
        so our own writes are counted separately.
        """
        with self.lock:
            return self.conn.execute("PRAGMA data_version").fetchone()[0], self._local_version

    # -- config ------------------------------------------------------------

    @synchronized
    def get_config(self, key: str) -> Optional[Any]:
        row = self.conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
        return decode(row[0]) if row else None

    @synchronized
    def set_config(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO config (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, encode(value)),
        )

    # -- rows --------------------------------------------------------------

    @synchronized
    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    @synchronized
    def row_bound(self) -> int:
        """One past the highest row ever allocated."""
        return self.get_config("row_bound") or 0

    @synchronized
    def allocate_rows(self, count: int, reuse_delay: float = SLOT_REUSE_DELAY_SECONDS) -> List[int]:
        """Hand out row slots, reusing slots freed long enough ago first."""
        reused = [
            r
            for (r,) in self.conn.execute(
                "SELECT row FROM free_rows WHERE freed_at <= strftime('%s', 'now') - ? LIMIT ?",
                (reuse_delay, count),
            )
        ]
        if reused:
            self.conn.executemany("DELETE FROM free_rows WHERE row = ?", [(r,) for r in reused])
        bound = self.row_bound()
        fresh = list(range(bound, bound + count - len(reused)))
        if fresh:
            self.set_config("row_bound", fresh[-1] + 1)
        return reused + fresh

    @synchronized
    def contains(self, uid: Any) -> bool:
        return self.conn.execute("SELECT 1 FROM entries WHERE uid = ?", (encode(uid),)).fetchone() is not None

    @synchronized
    def existing_uids(self, uids: Sequence[Any]) -> set:
        """Which of these ids already exist, as their encoded form.

        Encoded rather than decoded: an id may be a list, which cannot go in
        a set, and two ids are the same id exactly when they encode alike.
        """
        found = set()
        for chunk in _chunks(list(uids), 512):
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT uid FROM entries WHERE uid IN ({placeholders})", [encode(u) for u in chunk]
            )
            found.update(r[0] for r in rows)
        return found

    @synchronized
    def row_for(self, uid: Any) -> Optional[int]:
        row = self.conn.execute("SELECT row FROM entries WHERE uid = ?", (encode(uid),)).fetchone()
        return row[0] if row else None

    @synchronized
    def add(self, records: Sequence[Tuple[int, Any, Dict[str, Any]]]) -> None:
        """Insert ``(row, uid, metadata)`` records and index their metadata."""
        try:
            self.conn.executemany(
                "INSERT INTO entries (row, uid, meta) VALUES (?, ?, ?)",
                [(row, encode(uid), encode(meta)) for row, uid, meta in records],
            )
        except sqlite3.IntegrityError as error:
            # Two ids that encode alike are the same id, a tuple and a list of
            # the same items among them.
            clashing = self.existing_uids([uid for _, uid, _ in records])
            raise ValueError(
                f"Unique ID already exists: {sorted(clashing)[0] if clashing else error}"
            ) from error
        index_rows = _index_rows([(row, meta) for row, _, meta in records])
        if index_rows:
            self.conn.executemany(
                "INSERT INTO meta_index (key, value, num, row) VALUES (?, ?, ?, ?)", index_rows
            )

    @synchronized
    def metadata_for(self, uid: Any) -> Optional[Dict[str, Any]]:
        """The metadata of one id, read in a single query.

        Looking up the row first and the document second would break if
        compaction moved the entry between the two.
        """
        row = self.conn.execute("SELECT meta FROM entries WHERE uid = ?", (encode(uid),)).fetchone()
        return decode(row[0]) if row else None

    @synchronized
    def rows_for(self, uids: Sequence[Any]) -> Dict[str, int]:
        """Map each id that exists, in encoded form, to the row it occupies."""
        found: Dict[str, int] = {}
        for chunk in _chunks(list(uids), 512):
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT uid, row FROM entries WHERE uid IN ({placeholders})", [encode(u) for u in chunk]
            )
            found.update((uid, row) for uid, row in rows)
        return found

    @synchronized
    def replace_metadata(self, records: Sequence[Tuple[int, Dict[str, Any]]]) -> None:
        """Replace the metadata document of rows, reindexing what changed."""
        if not records:
            return
        rows = [row for row, _ in records]
        for chunk in _chunks(rows, 512):
            placeholders = ",".join("?" * len(chunk))
            self.conn.execute(f"DELETE FROM meta_index WHERE row IN ({placeholders})", chunk)
        self.conn.executemany(
            "UPDATE entries SET meta = ? WHERE row = ?", [(encode(meta), row) for row, meta in records]
        )
        self.conn.executemany(
            "INSERT INTO meta_index (key, value, num, row) VALUES (?, ?, ?, ?)", _index_rows(records)
        )

    @synchronized
    def move_rows(self, moves: Sequence[Tuple[int, int]]) -> None:
        """Move entries from one row slot to another, freeing the old slots."""
        if not moves:
            return
        self.conn.executemany("UPDATE entries SET row = ? WHERE row = ?", [(new, old) for old, new in moves])
        self.conn.executemany("UPDATE meta_index SET row = ? WHERE row = ?", [(new, old) for old, new in moves])

    @synchronized
    def relocate(self, moves: Sequence[Tuple[int, int]]) -> None:
        """Move entries to new row slots and free the slots they came from."""
        if not moves:
            return
        self.move_rows(moves)
        self.conn.executemany(
            "INSERT OR IGNORE INTO free_rows (row, freed_at) VALUES (?, strftime('%s', 'now'))",
            [(old,) for old, _ in moves],
        )

    @synchronized
    def reset_free_rows(self, row_bound: int) -> None:
        """Declare every slot below ``row_bound`` taken and drop the rest."""
        self.conn.execute("DELETE FROM free_rows")
        self.set_config("row_bound", row_bound)

    @synchronized
    def layout_version(self) -> int:
        """Counts the times entries have been moved between row slots.

        A search reads this before and after it works. Row numbers mean
        something different on either side of a move, so a search that
        straddles one has to be taken again.
        """
        return self.get_config("layout_version") or 0

    @synchronized
    def bump_layout_version(self) -> None:
        self.set_config("layout_version", self.layout_version() + 1)

    @synchronized
    def remove(self, uids: Sequence[Any]) -> List[int]:
        """Delete entries and return the row slots they occupied."""
        rows: List[int] = []
        for chunk in _chunks(list(uids), 512):
            placeholders = ",".join("?" * len(chunk))
            encoded = [encode(u) for u in chunk]
            rows.extend(
                r[0]
                for r in self.conn.execute(
                    f"SELECT row FROM entries WHERE uid IN ({placeholders})", encoded
                )
            )
            self.conn.execute(f"DELETE FROM entries WHERE uid IN ({placeholders})", encoded)
        for chunk in _chunks(rows, 512):
            placeholders = ",".join("?" * len(chunk))
            self.conn.execute(f"DELETE FROM meta_index WHERE row IN ({placeholders})", chunk)
            self.conn.executemany(
                "INSERT OR IGNORE INTO free_rows (row, freed_at) VALUES (?, strftime('%s', 'now'))",
                [(r,) for r in chunk],
            )
        return rows

    @synchronized
    def fetch(self, rows: Sequence[int]) -> Dict[int, Tuple[Any, Dict[str, Any]]]:
        """Map row numbers to their ``(uid, metadata)`` pairs."""
        found: Dict[int, Tuple[Any, Dict[str, Any]]] = {}
        for chunk in _chunks(list(rows), 512):
            placeholders = ",".join("?" * len(chunk))
            for row, uid, meta in self.conn.execute(
                f"SELECT row, uid, meta FROM entries WHERE row IN ({placeholders})", chunk
            ):
                found[row] = (decode(uid), decode(meta))
        return found

    def iter_entries(self, page_size: int = 1000) -> Iterable[Tuple[int, Any, Dict[str, Any]]]:
        """Stream entries a page at a time, holding the lock only per page.

        Pages walk the ids, not the row numbers: an id stays put for the life
        of its entry, while compaction moves rows around underneath.
        """
        last_uid = ""
        while True:
            with self.lock:
                page = self.conn.execute(
                    "SELECT row, uid, meta FROM entries WHERE uid > ? ORDER BY uid LIMIT ?",
                    (last_uid, page_size),
                ).fetchall()
            if not page:
                return
            for row, uid, meta in page:
                yield row, decode(uid), decode(meta)
            last_uid = page[-1][1]

    @synchronized
    def free_rows(self) -> np.ndarray:
        """Row slots that are allocated in the file but hold no entry."""
        return np.unique(
            np.fromiter((r for (r,) in self.conn.execute("SELECT row FROM free_rows")), dtype=np.int64)
        )

    @synchronized
    def all_rows(self) -> np.ndarray:
        return np.unique(
            np.fromiter((r for (r,) in self.conn.execute("SELECT row FROM entries")), dtype=np.int64)
        )

    # -- filtering ---------------------------------------------------------

    def _condition_rows(self, key: str, condition: Any) -> Set[int]:
        """Row numbers matching one ``{key: condition}`` pair.

        An operator document may carry several operators, and every one of
        them has to hold: ``{"$gt": 1, "$lt": 10}`` is a range, not a ``$gt``
        with something ignored after it.
        """
        if not isinstance(condition, dict):
            return self._query("SELECT row FROM meta_index WHERE key = ? AND value = ?", (key, encode(condition)))
        if not condition:
            raise ValueError(f"Empty operator document for key {key!r}.")

        matched: Optional[np.ndarray] = None
        for operator, operand in condition.items():
            rows = self._operator_rows(key, operator, operand)
            matched = rows if matched is None else np.intersect1d(matched, rows, assume_unique=True)
            if matched.size == 0:
                return matched
        return matched

    def _operator_rows(self, key: str, operator: str, operand: Any) -> Set[int]:
        """Row numbers matching a single operator against one key."""
        if operator == "$eq":
            return self._condition_rows(key, operand)
        if operator == "$ne":
            return self._all_except(
                self._query("SELECT row FROM meta_index WHERE key = ? AND value = ?", (key, encode(operand)))
            )
        if operator in _COMPARISONS:
            comparison = _COMPARISONS[operator]
            number = _numeric(operand)
            if number is not None:
                return self._query(
                    f"SELECT row FROM meta_index WHERE key = ? AND num IS NOT NULL AND num {comparison} ?",
                    (key, number),
                )
            # Comparing JSON text only means anything within one type: the
            # encoding of a string, a boolean and null sort by their first
            # character, which says nothing about the values.
            encoded = encode(operand)
            return self._query(
                f"SELECT row FROM meta_index WHERE key = ? AND num IS NULL AND value {comparison} ? "
                f"AND substr(value, 1, 1) = ?",
                (key, encoded, encoded[:1]),
            )
        if operator in ("$in", "$nin"):
            values = list(operand) if isinstance(operand, (list, tuple, set)) else [operand]
            matched = np.unique(
                np.concatenate(
                    [
                        self._query("SELECT row FROM meta_index WHERE key = ? AND value = ?", (key, encode(value)))
                        for value in values
                    ]
                    or [np.empty(0, dtype=np.int64)]
                )
            )
            return matched if operator == "$in" else self._all_except(matched)
        if operator == "$contains":
            # List members are indexed one by one, so equality covers lists;
            # strings additionally match on substring.
            matched = self._query("SELECT row FROM meta_index WHERE key = ? AND value = ?", (key, encode(operand)))
            if isinstance(operand, str):
                # Match the encoded form against the encoded values, minus its
                # surrounding quotes. Comparing the raw text would let a bare
                # quote match the quoting of every string in the index.
                needle = encode(operand)[1:-1]
                matched = np.union1d(
                    matched,
                    self._query(
                        "SELECT row FROM meta_index WHERE key = ? AND value LIKE ? ESCAPE '\\' "
                        "AND substr(value, 1, 1) = '\"'",
                        (key, f"%{_escape_like(needle)}%"),
                    ),
                )
            return matched
        if operator == "$exists":
            present = self._query("SELECT row FROM meta_index WHERE key = ?", (key,))
            return present if operand else self._all_except(present)
        raise ValueError(f"Invalid operator: {operator}. Expected one of {', '.join(OPERATORS)}")

    def _query(self, sql: str, params: Sequence[Any]) -> np.ndarray:
        """Row numbers as a sorted array.

        Row ids travel as ``int64`` arrays rather than Python sets: a filter
        that matches a million rows costs 8 MB this way and about 64 MB as a
        set, and a search must not carry the collection in memory.
        """
        return np.unique(np.fromiter((row[0] for row in self.conn.execute(sql, params)), dtype=np.int64))

    def _all_except(self, excluded: np.ndarray) -> np.ndarray:
        return np.setdiff1d(self.all_rows(), excluded, assume_unique=True)

    def _clause_rows(self, clause: Dict[str, Any]) -> Optional[np.ndarray]:
        """Rows matching every pair of one filter dictionary (AND)."""
        matched: Optional[np.ndarray] = None
        for key, condition in clause.items():
            rows = self._condition_rows(key, condition)
            matched = rows if matched is None else np.intersect1d(matched, rows, assume_unique=True)
            if matched.size == 0:
                return matched
        return matched

    def filter_rows(
        self,
        metadata_filter: Any = None,
        exclude_filter: Any = None,
        or_filters: Any = None,
    ) -> Optional[np.ndarray]:
        """Resolve the three filter kinds to row numbers.

        Returns ``None`` when no filter applies, meaning "every row".
        """
        with self.lock:
            matched: Optional[np.ndarray] = None

            for clause in _as_clauses(metadata_filter):
                rows = self._clause_rows(clause)
                if rows is not None:
                    matched = rows if matched is None else np.intersect1d(matched, rows, assume_unique=True)

            clauses = _as_clauses(or_filters)
            if clauses:
                found = [self._clause_rows(clause) for clause in clauses]
                alternatives = np.unique(
                    np.concatenate([rows for rows in found if rows is not None and rows.size] or [np.empty(0, dtype=np.int64)])
                )
                matched = alternatives if matched is None else np.intersect1d(matched, alternatives, assume_unique=True)

            for clause in _as_clauses(exclude_filter):
                rows = self._clause_rows(clause)
                if rows is None or rows.size == 0:
                    continue
                matched = (
                    self._all_except(rows)
                    if matched is None
                    else np.setdiff1d(matched, rows, assume_unique=True)
                )

            return matched

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        with self.lock:
            if os.getpid() == self._pid:
                self._conn.close()
            self._open_connections.clear()


def _close_all(connections: List[sqlite3.Connection]) -> None:
    """Close connections a dropped store left open."""
    for connection in connections:
        try:
            connection.close()
        except sqlite3.Error:  # pragma: no cover - already unusable
            pass
    connections.clear()


def _index_rows(records: Sequence[Tuple[int, Dict[str, Any]]]) -> List[Tuple[str, Optional[str], Optional[float], int]]:
    """One inverted-index row per metadata value, list values one by one.

    An empty list still gets a row, with a NULL value: without it the key
    would have no rows at all and would look as though it were not there.
    """
    indexed: List[Tuple[str, Optional[str], Optional[float], int]] = []
    for row, meta in records:
        for key, value in meta.items():
            items = list(value) if isinstance(value, (list, tuple, set)) else [value]
            if not items:
                indexed.append((key, None, None, row))  # present, but holds nothing
                continue
            indexed.extend((key, encode(item), _numeric(item), row) for item in items)
    return indexed


def _as_clauses(filters: Any) -> List[Dict[str, Any]]:
    """Normalize a dict / list of dicts / ``None`` into a list of non-empty dicts."""
    if not filters:
        return []
    if isinstance(filters, dict):
        filters = [filters]
    return [clause for clause in filters if clause]


def _escape_like(value: str) -> str:
    for char in ("\\", "%", "_"):
        value = value.replace(char, "\\" + char)
    return value


def _chunks(items: List[Any], size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
