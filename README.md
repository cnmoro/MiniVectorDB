[![codecov](https://codecov.io/gh/cnmoro/MiniVectorDB/graph/badge.svg?token=DGHUUFI9H2)](https://codecov.io/gh/cnmoro/MiniVectorDB)

[![Downloads](https://static.pepy.tech/badge/minivectordb)](https://pepy.tech/project/minivectordb)

[![Downloads](https://static.pepy.tech/badge/minivectordb/month)](https://pepy.tech/project/minivectordb)

[![Downloads](https://static.pepy.tech/badge/minivectordb/week)](https://pepy.tech/project/minivectordb)

## **MiniVectorDB**

This is a Python project aimed at providing an extremely simple yet powerful vector database that uses FAISS internally, while also providing functionality for extracting embeddings, using an integrated ONNX model - but also integrated with the e5 multilingual embedding models. It is now possible to index vectors with metadata (which can be used for querying), and also rerank results using a hybrid approach (rank_bm25 + fuzzy string similarity). Please check out the code snippets below.

Integrated model link in Huggingface: [universal-sentence-encoder-multilingual-3-onnx-quantized](https://huggingface.co/WiseIntelligence/universal-sentence-encoder-multilingual-3-onnx-quantized)

### **Installation**

```plaintext
pip install minivectordb
```

### **Quantized ONNX Model Supported Languages**

```python
["en", "pt", "ar", "zh", "fr", "de", "it", "ja", "ko", "nl", "ps", "es", "th", "tr", "ru"]
```

### **Usage**

```python
from minivectordb.embedding_model import EmbeddingModel
from minivectordb.vector_database import VectorDatabase

# Embedding size will be automatically registered on the first insertion
# You can use your own model, such as ada-v2
vector_db = VectorDatabase()

# Three models are offered:
# Google's Universal Sentence Encoder (ONNX)
# intfloat's e5 multilingual model (SMALL or LARGE)
# Additional parameters on model constructor:
# use_quantized_onnx_model (True / False)
# e5_model_size ('small', 'large'), used if use_quantized_onnx_model is False
# (note: e5 models are downloaded automatically. the onnx model is built-in)
model = EmbeddingModel()

# Text identifier, sentences and metadata
# Basic example 
sentences_with_metadata = [
    (1,  "I like dogs", {"animal": "dog", "like": True}),
    (2,  "I like cats", {"animal": "cat", "like": True}),
    (3,  "The king has three kids", {"royalty": "king"}),
    (4,  "The queen has one daughter", {"royalty": "queen"}),
    (5,  "Programming is cool", {"topic": "programming", "sentiment": "positive"}),
    (6,  "Software development is cool", {"topic": "software development", "sentiment": "positive"}),
    (7,  "Being a developer is stressful", {"topic": "software development", "sentiment": "negative"}),
    (8,  "Being a developer is a job", {"topic": "software development", "sentiment": "neutral"}),
    (9,  "I like to ride my bicycle", {"activity": "riding", "object": "bicycle"}),
    (10,  "I like to ride my scooter", {"activity": "riding", "object": "scooter"}),
    (11,  "The sky is blue", {"color": "blue", "object": "sky"}),
    (12, "The ocean is blue", {"color": "blue", "object": "ocean"})
]

for id, sentence, metadata in sentences_with_metadata:
    sentence_embedding = model.extract_embeddings(sentence)
    vector_db.store_embedding(id, sentence_embedding, metadata)

## Basic Semantic Search
query = "animals"
query_embedding = model.extract_embeddings(query)
search_results = vector_db.find_most_similar(query_embedding, k = 2)

ids, distances, metadatas = search_results
for id, dist, metadata in zip(ids, distances, metadatas):
    print(f"ID: {id}, Sentence: \"{sentences_with_metadata[id-1][1]}\", Distance: {dist}, Metadata: {metadata}")

# Results:
# ID: 1, Sentence: "I like dogs", Distance: 0.4143948554992676, Metadata: {'animal': 'dog', 'like': True}
# ID: 2, Sentence: "I like cats", Distance: 0.3983381986618042, Metadata: {'animal': 'cat', 'like': True}

## Hybrid Reranking with BM25 and Fuzzy Ratios
query = "blue is cool"
query_embedding = model.extract_embeddings(query)
search_results = vector_db.find_most_similar(query_embedding, k = 6) # Note that we are fetching 6 results here
ids, distances, metadata = search_results

# Results:
# ID: 11, Sentence: "The sky is blue", Distance: 0.6656221747398376, Metadata: {'color': 'blue', 'object': 'sky'}
# ID: 12, Sentence: "The ocean is blue", Distance: 0.6223428845405579, Metadata: {'color': 'blue', 'object': 'ocean'}
# ID: 2, Sentence: "I like cats", Distance: 0.3566429018974304, Metadata: {'animal': 'cat', 'like': True}
# ID: 1, Sentence: "I like dogs", Distance: 0.3240365982055664, Metadata: {'animal': 'dog', 'like': True}
# ID: 5, Sentence: "Programming is cool", Distance: 0.3074682354927063, Metadata: {'topic': 'programming', 'sentiment': 'positive'}
# ID: 6, Sentence: "Software development is cool", Distance: 0.22255833446979523, Metadata: {'topic': 'software development', 'sentiment': 'positive'}

sentences = [sentences_with_metadata[id-1][1] for id in ids]
hybrid_reranked_results = vector_db.hybrid_rerank_results(
    sentences = sentences,
    search_scores = distances,
    query = query,
    k = 4 # Now we are reducing the scope to 4 results
)
hybried_retrieved_sentences, hybrid_scores = hybrid_reranked_results

for sentence, score in zip(hybried_retrieved_sentences, hybrid_scores):
    print(f"Sentence: \"{sentence}\", Score: {score}")

# With the reranking we get the following results:
# Sentence: "Programming is cool", Score: 4.37548599419139
# Sentence: "Software development is cool", Score: 4.291912408770172
# Sentence: "The ocean is blue", Score: 3.2117400547872474
# Sentence: "The sky is blue", Score: 3.1463634988676

# We have successfully reranked the results to get the most relevant results first.
# Note that we have removed the results with good scores, but that are not relevant to the query.
# (e.g. "I like cats", "I like dogs")

##################################################################

## Semantic Search with Metadata Filtering
query_embedding = model.extract_embeddings("king")
metadata_filter = {"royalty": "queen"}
search_results = vector_db.find_most_similar(query_embedding, metadata_filter, k = 2)

ids, distances, metadatas = search_results
for id, dist, metadata in zip(ids, distances, metadatas):
    print(f"ID: {id}, Sentence: \"{sentences_with_metadata[id-1][1]}\", Distance: {dist}, Metadata: {metadata}")

# We searched for "king" but filtered by "queen" so we should get the queen sentence
# ID: 4, Sentence: "The queen has one daughter", Distance: 0.3122280240058899, Metadata: {'royalty': 'queen'}
    
##################################################################
    
## Semantic Search with Metadata Filtering and also using the "OR" Filtering operator
query_embedding = model.extract_embeddings("programming")
metadata_filter = {"topic": "software development"}
or_filters = [
    {"sentiment": "positive"},
    {"sentiment": "negative"}
] # This could be a list of dicts, or a single dict

search_results = vector_db.find_most_similar(query_embedding, metadata_filter, k = 2, or_filters = or_filters)
ids, distances, metadatas = search_results
for id, dist, metadata in zip(ids, distances, metadatas):
    print(f"ID: {id}, Sentence: \"{sentences_with_metadata[id-1][1]}\", Distance: {dist}, Metadata: {metadata}")

# We searched for "programming" and filtered by "software development" and allow both sentiments
# ID: 6, Sentence: "Software development is cool", Distance: 0.3860135078430176, Metadata: {'topic': 'software development', 'sentiment': 'positive'}
# ID: 7, Sentence: "Being a developer is stressful", Distance: 0.21792981028556824, Metadata: {'topic': 'software development', 'sentiment': 'negative'}

##################################################################
    
# Save the database to disk
# The database file will be automatically loaded if exists on disk
# File path is "db.pkl" by default, saved to the current working directory
# Customizable by parameter "storage_file" on VectorDatabase constructor
vector_db.persist_to_disk()
```

## **License**

This project is licensed under the MIT License.
