[![codecov](https://codecov.io/gh/cnmoro/MiniVectorDB/graph/badge.svg?token=DGHUUFI9H2)](https://codecov.io/gh/cnmoro/MiniVectorDB)
[![Downloads](https://static.pepy.tech/badge/minivectordb-ooc)](https://pepy.tech/project/minivectordb-ooc)
[![Downloads](https://static.pepy.tech/badge/minivectordb-ooc/month)](https://pepy.tech/project/minivectordb-ooc)

## **MiniVectorDB-OOC**

A vector database that does not keep its index in memory, with a built-in multilingual sentence encoder.

* **Out of core.** Vectors live in a file and are streamed in fixed-size chunks. A 200k x 512 index is searched in ~38 ms, and a 1M x 512 index in ~191 ms, while the process holds a couple of hundred MB whatever the size. Nothing is rebuilt at startup.
* **Light.** The only dependencies are NumPy, RapidFuzz and the encoder. No torch, no transformers, no FAISS. The encoder itself is ~33 MB and runs on CPU.
* **Shared.** Several processes can open the same directory: the workers of a web server, or pods on one volume. There is no server to run.
* **Simple.** Store vectors with metadata, filter with a MongoDB-like query language, and rerank with lexical signals.

### **Installation**

```plaintext
pip install minivectordb-ooc
```

`minivectordb-ooc` is the out-of-core rewrite and is published under its own name. `pip install minivectordb` still installs the old in-memory library, which keeps its whole index in RAM and is not maintained here.

The import is unchanged, so moving over is a change of one line in your requirements:

```python
from minivectordb import EmbeddingModel, VectorDatabase
```

Because both packages provide the same `minivectordb` module, **install one or the other, never both** — pip will let them overwrite each other's files. Uninstall the old one first:

```plaintext
pip uninstall minivectordb && pip install minivectordb-ooc
```

### **Quickstart**

```python
from minivectordb import EmbeddingModel, VectorDatabase

model = EmbeddingModel()               # 512-dim, denoise=True by default
db = VectorDatabase("./my_index")      # a directory; omit it for a temporary database

sentences = {
    1: ("I like dogs", {"animal": "dog", "price": 10}),
    2: ("I like cats", {"animal": "cat", "price": 20}),
    3: ("The queen has one daughter", {"royalty": "queen"}),
    4: ("Programming is cool", {"topic": "software", "tags": ["work", "fun"]}),
}

db.store_embeddings_batch(
    list(sentences),
    model.extract_embeddings_batch([text for text, _ in sentences.values()]),
    [metadata for _, metadata in sentences.values()],
)

ids, scores, metadatas = db.find_most_similar(model.extract_embeddings("pets"), k=2)
for unique_id, score, metadata in zip(ids, scores, metadatas):
    print(f"{unique_id}: {sentences[unique_id][0]!r} score={score:.3f} {metadata}")

# 2: 'I like cats' score=0.413 {'animal': 'cat', 'price': 20}
# 1: 'I like dogs' score=0.408 {'animal': 'dog', 'price': 10}

db.close()
```

Scores are cosine similarities in `[-1, 1]`, highest first, and always the similarity to the vector as it is stored — with `dtype="int8"` that is the quantized vector, not the one you passed in. Every write is committed as it happens, so reopening `VectorDatabase("./my_index")` picks up where you left off.

### **The encoder**

`EmbeddingModel` wraps [fast-universal-sentence-encoder](https://github.com/cnmoro/fast-universal-sentence-encoder-3), a NumPy-only port of Google's Universal Sentence Encoder Multilingual v3. It produces 512-dimensional L2-normalized vectors for 16 languages: `ar`, `de`, `en`, `es`, `fr`, `it`, `ja`, `ko`, `nl`, `pl`, `pt`, `ru`, `th`, `tr`, `zh-cn`, `zh-tw`.

```python
model = EmbeddingModel(denoise=True, threads=4)

model.extract_embeddings("one text")               # (512,) float32
model.extract_embeddings_batch(["one", "two"])     # (2, 512) float32
model.similarity("o carro é azul", "the car is blue")   # 0.93
```

`denoise=True` (the default) normalizes numeric tokens before encoding, so texts that differ only in quantities stay close together:

| texts | `denoise=False` | `denoise=True` |
| --- | --- | --- |
| "paguei 350 reais" / "paguei 12 reais" | 0.78 | 1.00 |

Pass `denoise=False` to keep the raw embeddings, or `denoise_fn=...` to supply your own preprocessing. You can also store vectors from any other model; the dimension is taken from the first insertion.

### **Metadata filtering**

`find_most_similar` accepts three filter arguments, each a dictionary or a list of dictionaries:

* `metadata_filter` — every condition must match (AND).
* `or_filters` — at least one dictionary must match (OR), intersected with `metadata_filter`.
* `exclude_filter` — anything matching is removed.

```python
db.find_most_similar(
    query_embedding,
    metadata_filter={"price": {"$lte": 20}},
    or_filters=[{"animal": "dog"}, {"animal": "cat"}],
    exclude_filter={"animal": "cat"},
    k=5,
)
```

Values are matched by type, so `5` and `"5"` are different. Supported operators:

| operator | matches |
| --- | --- |
| `{"$eq": v}` / plain value | equal to `v` |
| `{"$ne": v}` | not equal to `v`, including documents without the key |
| `{"$gt": v}`, `{"$gte": v}`, `{"$lt": v}`, `{"$lte": v}` | ordered comparison (numbers, or strings against strings) |
| `{"$in": [a, b]}` | equal to any listed value |
| `{"$nin": [a, b]}` | equal to none of the listed values |
| `{"$contains": v}` | list metadata containing `v`, or string metadata containing `v` as a substring |
| `{"$exists": True/False}` | key present or absent |

A condition may carry several operators, and all of them have to hold: `{"price": {"$gt": 1, "$lt": 10}}` is a range.

List values are indexed element by element, so `{"tags": {"$contains": "work"}}` is an indexed lookup, not a scan. A key whose value is an empty list still counts as present for `$exists`.

Two details worth knowing about ordering comparisons: they only compare values of the same kind, so `{"$gt": "a"}` never matches a number, a boolean or `null` stored under the same key; and they order numbers as 64-bit floats, so integers beyond 2^53 are compared by their nearest float.

### **Reranking and autocut**

`hybrid_rerank_results` blends the semantic score with a hashed character n-gram similarity and a fuzzy ratio. All three signals are in `[0, 1]`, so the weights behave as proportions. Pass one score per sentence; a mismatched list is refused rather than stretched to fit.

```python
ids, scores, _ = db.find_most_similar(model.extract_embeddings("blue is cool"), k=6)
sentences = [text for text, _ in (sentences[i] for i in ids)]

best, blended = db.hybrid_rerank_results(
    sentences=sentences,
    search_scores=scores,
    query="blue is cool",
    k=4,
    weights=(0.80, 0.15, 0.05),   # semantic, n-gram, fuzzy
)
```

`find_most_similar(..., autocut=True)` drops the tail of the result list after the largest relative score drop, so a query with only two good matches returns two results instead of `k`.

### **Several processes, one database**

A database directory can be opened by any number of processes at once, with no server and no extra configuration:

```python
# app.py, served with: uvicorn app:app --workers 5
from fastapi import FastAPI
from minivectordb import EmbeddingModel, VectorDatabase

app = FastAPI()
model = EmbeddingModel()
db = VectorDatabase("/data/index")     # every worker opens the same directory

@app.post("/documents/{document_id}")
def add(document_id: str, text: str):
    db.store_embedding(document_id, model.extract_embeddings(text), {"text": text})

@app.get("/search")
def search(query: str):
    ids, scores, metadatas = db.find_most_similar(model.extract_embeddings(query), k=5)
    return [{"id": i, "score": s, **m} for i, s, m in zip(ids, scores, metadatas)]
```

This works because the database keeps almost nothing in memory:

* **Writers take SQLite's write lock** while they claim row slots and store their vectors, so two processes are never given the same slot. A writer that has to wait waits up to 30 seconds before giving up.
* **Readers check a version counter** before each search, which costs one SQLite pragma. When another process has written, the reader re-reads two small values and sees the new vectors immediately. There is no cache to invalidate and no index to rebuild.
* **Vectors are written by offset** and the file only ever grows, so writers never move data another process is reading.
* **Scores are read, not remembered.** The scan only chooses which rows are worth returning; their scores come from a second read of those few rows, taken together with their metadata. A row that another process has given to a different vector is scored as what it holds now.
* **A row is never rewritten while an entry points at it.** Updating an entry writes the new vector somewhere else and then repoints the entry, so a reader cannot catch a row half way through being replaced. This matters because the file is written *before* the metadata is committed: without it, a search could pair a fresh vector's score with the document it had a moment ago.
* **A deleted slot is left alone** for five minutes before an insertion may claim it, which is far longer than any search takes to read the file. `slot_reuse_delay` sets the wait; on a database whose searches can be slow, raise it.
* **Compaction is announced.** Closing the gaps moves entries between slots, so it bumps a counter that every search checks before and after it runs. A search that straddles a compaction simply takes itself again.
* **A crash is safe.** Vectors are flushed before the metadata that points at them is committed, and SQLite rolls back the transaction of a process that dies mid-write. The slots it had claimed are reused later.
* **Forking is safe.** A database opened before `fork()` reconnects in each child on first use, so a preloading server can open it at import time.

An instance is also safe to share between threads, so a thread pool and a process pool can be mixed freely.

What a search does **not** promise is a frozen snapshot: entries another process deletes or updates while the search is reading can leave it returning fewer than `k` results. What it does promise is that every result is an entry that existed, carrying the score of the vector that entry holds and the metadata document that belongs to it — never one entry's score beside another's document.

#### Shared volumes

Pods on one ReadWriteMany volume work the same way, with one difference: SQLite's write-ahead log needs shared memory that only exists on a single machine, so on shared storage MiniVectorDB journals to a rollback file instead and flushes every write.

Which mode to use is decided **the safe way round**. A volume counts as local only when its filesystem is one a single machine mounts on its own disks (`ext4`, `xfs`, `btrfs`, `zfs`, `overlay`, `tmpfs` and the like). Anything else counts as shared: NFS, CIFS and CephFS, but equally a FUSE driver or a CSI driver that reports a name of its own — the kind of mount an OpenShift claim can appear as. Guessing "local" is the answer that corrupts data when it is wrong, so an unrecognized volume is never guessed to be local; the cost of not knowing is some write speed.

The answer is recorded in the database the first time it is created, so every process that opens it afterwards journals the same way, whatever each one detects.

You can decide instead of leaving it to detection, either in the code:

```python
db = VectorDatabase("/mnt/shared/index", shared_storage=True)
```

or, when the application cannot be changed, in the deployment:

```yaml
env:
  - name: MINIVECTORDB_SHARED_STORAGE
    value: "1"
```

Either one also corrects a database that was created with the wrong answer. If a database is already open in write-ahead log mode when a process asks for shared storage, opening it fails with an explicit error rather than proceeding unsafely.

**Set it yourself whenever you can.** Detection is a fallback, and the one case it cannot see is a volume that looks like an ordinary disk but is reachable from more than one machine.

The volume must also support POSIX file locking, which is what serializes the writers. NFSv4 provides it; an NFSv3 export mounted with `nolock`, or any filesystem where `flock`/`fcntl` locks are ignored, cannot safely take concurrent writers from more than one node. If you cannot guarantee locking, keep writes on one pod and mount the volume read-only elsewhere.

### **Updates and deletions**

Updating gives an entry a new vector without ever rewriting the row another process might be reading:

```python
db.update_embedding("doc-1", new_vector)                 # vector only
db.update_embedding("doc-1", metadata={"state": "read"}) # metadata only, replaced whole
db.update_embeddings_batch(ids, vectors, metadatas)      # both, in one transaction

db.upsert_embedding("doc-1", vector, {"state": "new"})   # insert it, or replace it
db.upsert_embeddings_batch(ids, vectors, metadatas)
```

A new vector goes into a newly claimed slot and the entry is moved to it, leaving the slot it came from free. That is what lets a search running at the same time stay coherent (see below), and the freed slot is reclaimed the same way a deletion's is. Replacing only the metadata touches no vector at all.

Deleting frees the row slot the entry held; the bytes stay in the file until something writes over them. Those slots do not stay empty: once the gaps reach a fifth of the file, deleting or updating closes them and hands the disk space back, in the same call. Compaction is also what physically removes a deleted vector, by writing another entry over it or truncating it away.

```python
db.store_embeddings_batch(list(range(100)), vectors)
db.delete_embeddings_batch(list(range(0, 100, 2)))   # delete half

len(db)      # 50
db.holes()   # 0 -- the file holds 50 rows, not 100
```

Compaction moves the entries at the end of the file into the gaps in front of them and truncates what is left over. On 100k vectors of 512 dimensions, deleting a quarter of them compacts in 0.4 s and takes the file from 328 MB to 154 MB.

Small deletions do not pay for this — they leave their slots for the next insertion to reuse, and compaction waits until the gaps are worth closing. `auto_compact` is that threshold; set it to `0` to compact only when you ask (a value of `1` or more means the same, since gaps can never be the whole file):

```python
db = VectorDatabase("./my_index", auto_compact=0)
db.store_embeddings_batch(ids, vectors)      # 1000 of them, say
db.delete_embeddings_batch(ids[:500])
db.holes()     # 500
db.compact()   # returns how many entries it moved
db.holes()     # 0
```

Searches running at that moment, in this process or another, stay correct — that is what the layout counter above is for.

### **Performance**

Measured on an AMD Ryzen 9 9950X3D (16 cores), 30 GB RAM, ext4 on NVMe, Python 3.14, NumPy 2.5, one process, `float32`, `k=10`, medians over 40 queries with the page cache warm. The scripts are in the repository history; every number below came off this machine, and the ones marked *extrapolated* say so.

Nothing here is an approximate index: a search reads every live vector. That makes the cost easy to predict — it is the size of the data divided by how fast the machine can stream it — and it means recall is always exact.

#### Search

| vectors | dim | file | p50 | p95 | queries/s | with a filter |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 128 | 5 MB | 0.65 ms | 1.5 ms | 1,539 | 1.6 ms |
| 50,000 | 128 | 41 MB | 2.26 ms | 6.6 ms | 443 | 5.6 ms |
| 100,000 | 128 | 82 MB | 5.42 ms | 23.8 ms | 184 | 10.4 ms |
| 200,000 | 128 | 164 MB | 9.3 ms | 19.6 ms | 108 | 20.9 ms |
| 1,000,000 | 128 | 655 MB | 45.5 ms | — | 22 | — |
| 10,000 | 512 | 20 MB | 2.03 ms | 11.3 ms | 493 | 2.6 ms |
| 50,000 | 512 | 164 MB | 10.3 ms | 14.4 ms | 97 | 12.0 ms |
| 100,000 | 512 | 328 MB | 17.2 ms | 30.7 ms | 58 | 22.4 ms |
| 200,000 | 512 | 655 MB | 37.9 ms | 99.1 ms | 26 | 60.0 ms |
| 1,000,000 | 512 | 2,621 MB | 190.9 ms | — | 5.2 | — |

The filtered column is a filter that keeps a tenth of the collection. Filtering is not free: the rows still have to be found in SQLite and either gathered or masked, which is why it costs more than the plain scan rather than a tenth of it.

#### Insert, update and delete

| vectors | dim | batched insert | one at a time |
| ---: | ---: | ---: | ---: |
| 100,000 | 128 | 87,085 vectors/s | 4.8 ms each |
| 200,000 | 128 | 88,618 vectors/s | 4.5 ms each |
| 1,000,000 | 128 | 96,329 vectors/s | — |
| 100,000 | 512 | 65,514 vectors/s | 4.4 ms each |
| 200,000 | 512 | 64,723 vectors/s | 4.6 ms each |
| 1,000,000 | 512 | 64,872 vectors/s | — |
| 10,000,000 | 128 | 146,164 vectors/s | — |

Every row above stores one metadata document per vector; the 10M row stores none, which is most of why it is faster. Indexing metadata is a real part of insert cost.

A single `store_embedding` costs about 4.5 ms on this disk and barely moves with the collection size, because almost all of it is one `fsync`: the vector is on the platter before the metadata that points at it is committed. **Batch your writes** — `store_embeddings_batch` pays that cost once for the whole batch, which is the difference between 200 writes a second and 90,000.

Deleting costs 1.3 ms per vector at 100k x 128 and 7.4 ms at 200k x 512 when it triggers compaction; without compaction it is a metadata-only operation. Updating writes one row and repoints one entry.

#### Mixed workloads

Four reader threads and, where stated, one thread inserting 100-vector batches continuously — a deliberately write-saturated mix.

| | 100,000 x 128 | 200,000 x 512 |
| --- | --- | --- |
| 1 reader | 5.4 ms p50, 184 q/s | 37.9 ms p50, 26 q/s |
| 4 readers | 25.6 ms p50, 127 q/s | 172.9 ms p50, 20 q/s |
| 4 readers + writer | 33.2 ms p50, 97 q/s, 1,312 writes/s | 208.3 ms p50, 17 q/s, 215 writes/s |
| 4 readers + writer + deletes | 27.1 ms p50, 119 q/s, 693 writes/s | 164.0 ms p50, 23 q/s, 143 writes/s |

**Reads do not scale across threads inside one process.** One lock covers a search, so four reader threads take about four times as long each and the throughput stays flat. This is deliberate — the lock is what keeps a search coherent while other threads write — and it is not a limit on the database, only on the process: several processes on the same directory read in parallel with no lock between them, which is exactly how a multi-worker web server or a set of pods uses it. Give each worker its own `VectorDatabase`.

#### Vector size

At 50,000 vectors, sweeping the dimension:

| dim | file | insert | search p50 | queries/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 41 MB | 92,197/s | 2.26 ms | 443 |
| 384 | 123 MB | 74,715/s | 7.0 ms | 143 |
| 512 | 164 MB | 66,509/s | 10.3 ms | 97 |
| 768 | 246 MB | 55,315/s | 14.2 ms | 71 |
| 1536 | 492 MB | 35,401/s | 32.1 ms | 31 |

#### Precision

200,000 x 512, the same data at each precision:

| dtype | file | search p50 |
| --- | ---: | ---: |
| `float32` | 655 MB | 32.5 ms |
| `int8` | 164 MB | 42.7 ms |
| `float16` | 328 MB | 217.7 ms |

(One run, so compare these three with each other rather than against the tables above — repeat runs on this machine vary by about 15%.)

Quarter the file does not mean quarter the time. The scan multiplies straight out of the stored dtype, but anything that is not already `float32` still has to be widened, and NumPy's `float16` conversion is slow enough to cost more than the bytes it saves. **Choose a smaller dtype to fit more on disk, not to go faster** — and if you do, prefer `int8`, which is both smaller and quicker than `float16`.

#### Working out your own numbers

Search time on this machine follows the bytes it has to read:

```
latency ≈ 0.4 ms + (vectors × dim × bytes_per_value) / 10.5 GB/s
```

From about 50 MB upward that lands within roughly 10% of every row measured above, because the scan runs at memory-copy speed once the file is in the page cache. Below 50 MB the fixed half-millisecond of query overhead dominates and the formula over-estimates. Two checks against collections larger than anything in the tables:

| vectors | dim | file | predicted | measured | peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | 512 | 2.6 GB | 195 ms | 190.9 ms | 203 MB |
| 10,000,000 | 128 | 5.2 GB | 488 ms | 544.7 ms | 195 MB |

Ten million vectors, a 5.2 GB file, and the process holds 195 MB. That is the whole point of the design.

Extrapolating the same way, and marked clearly as arithmetic rather than measurement:

| vectors | dim | file | extrapolated p50 | notes |
| ---: | ---: | ---: | ---: | --- |
| 1,000,000 | 1536 | 6.1 GB | ~590 ms | fits in page cache on a 30 GB machine |
| 10,000,000 | 512 | 20.5 GB | ~2.0 s | only if 20 GB of page cache is free |
| 10,000,000 | 1536 | 61 GB | ~6 s | needs the storage to stream at 10 GB/s; on a disk that reads at 2 GB/s, ~30 s |

The caveat matters more than the arithmetic: **the formula holds while the file is in the page cache.** Once the collection is larger than free RAM, a search is bounded by how fast the storage streams, not by the CPU, and the same query can be several times slower. At that size, use a smaller dtype, split the collection across directories, or accept the read cost.

### **Storage and memory**

A database is a directory with two files: `vectors.bin` (raw fixed-size rows) and `metadata.sqlite3` (ids, metadata documents and the metadata index).

Searching reads `vectors.bin` in 8 MB chunks with positional reads, so peak memory follows the size of the answer you ask for and not the size of the collection: scanning 400k vectors for the best 10 and for the best 400,000 both leave the peak where it was. Building the result list itself is what a very large `k` pays for. Deleted rows are freed and reused by later insertions instead of rewriting the file.

Lower precision trades a little recall for a smaller file:

```python
db = VectorDatabase("./my_index", dtype="int8")   # "float32" (default), "float16", "int8"
```

| dtype | bytes per 512-dim vector | 1M vectors |
| --- | --- | --- |
| `float32` | 2048 | 2.0 GB |
| `float16` | 1024 | 1.0 GB |
| `int8` | 512 | 0.5 GB |

The precision is fixed when the database is created and is remembered on reopen.

### **API**

```python
db = VectorDatabase(path=None, dtype="float32", dimension=None,
                    shared_storage=None, slot_reuse_delay=300.0, auto_compact=0.2)

db.store_embedding(unique_id, embedding, metadata=None)
db.store_embeddings_batch(unique_ids, embeddings, metadatas=None)
db.update_embedding(unique_id, embedding=None, metadata=None)
db.update_embeddings_batch(unique_ids, embeddings=None, metadatas=None)
db.upsert_embedding(unique_id, embedding, metadata=None)
db.upsert_embeddings_batch(unique_ids, embeddings, metadatas=None)
db.delete_embedding(unique_id)
db.delete_embeddings_batch(unique_ids)

db.holes()                        # row slots deletions and updates left unused
db.compact()                      # close them and shrink the file

db.get_vector(unique_id)          # the stored, normalized vector
db.get_metadata(unique_id)
db.iter_entries()                 # streams (unique_id, metadata) pairs
len(db), unique_id in db

db.find_most_similar(embedding, metadata_filter=None, exclude_filter=None,
                     or_filters=None, k=5, autocut=False)
db.hybrid_rerank_results(sentences, search_scores, query, k=5, weights=(0.80, 0.15, 0.05))
db.autocut_scores(scores, threshold=0.2)   # the indexes find_most_similar(autocut=True) drops

db.flush()                        # writes are committed as they happen
db.close()                        # or use VectorDatabase(...) as a context manager
```

`shared_storage` is detected from the filesystem when left out; `slot_reuse_delay` is how many seconds a deleted row slot is kept out of use; `auto_compact` is the share of unused slots that triggers compaction.

Ids may be any JSON-encodable value: integers, strings, floats or booleans. A database instance is safe to share between threads.

### **License**

This project is licensed under the MIT License.
