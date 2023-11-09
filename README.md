[![codecov](https://codecov.io/gh/cnmoro/MiniVectorDB/graph/badge.svg?token=DGHUUFI9H2)](https://codecov.io/gh/cnmoro/MiniVectorDB)

## **MiniVectorDB**

This is a Python project aimed at extracting embeddings from textual data and performing semantic search. It's a simple yet powerful system combining a small quantized ONNX model with FAISS indexing for fast similarity search. As the model is small and also running in ONNX runtime with quantization, we get lightning fast speed.

Model link in Huggingface: [universal-sentence-encoder-multilingual-3-onnx-quantized](https://huggingface.co/WiseIntelligence/universal-sentence-encoder-multilingual-3-onnx-quantized)

### **Installation**

```plaintext
pip install minivectordb
```

### **Supported Languages**

```python
["en", "pt", "ar", "zh", "fr", "de", "it", "ja", "ko", "nl", "ps", "es", "th", "tr", "ru"]
```

### **Usage**

```python
from minivectordb.embedding_model import EmbeddingModel
from minivectordb.vector_database import VectorDatabase

vector_db = VectorDatabase(embedding_size = 512)
model = EmbeddingModel()

# Text identifier, sentences and metadata
sentences_with_metadata = [
    (1,  "I like dogs", {"animal": "dog", "like": True}),
    (2,  "I like cats", {"animal": "cat", "like": True}),
    (3,  "The king has three kids", {"royalty": "king"}),
    (4,  "The queen has one daughter", {"royalty": "queen"}),
    (5,  "Programming is cool", {"topic": "programming", "sentiment": "positive"}),
    (6,  "Software development is cool", {"topic": "software development", "sentiment": "positive"}),
    (7,  "I like to ride my bicycle", {"activity": "riding", "object": "bicycle"}),
    (8,  "I like to ride my scooter", {"activity": "riding", "object": "scooter"}),
    (9,  "The sky is blue", {"color": "blue", "object": "sky"}),
    (10, "The ocean is blue", {"color": "blue", "object": "ocean"})
]

for id, sentence, metadata in sentences_with_metadata:
    sentence_embedding = model.extract_embeddings(sentence)
    vector_db.store_embedding(id, sentence_embedding, metadata)

## Basic Semantic Search

query_embedding = model.extract_embeddings("cats")
search_results = vector_db.find_most_similar(query_embedding, k = 2)

ids, distances, metadatas = search_results
for id, dist, metadata in zip(ids, distances, metadatas):
    print(f"ID: {id}, Sentence: \"{sentences_with_metadata[id-1][1]}\", Distance: {dist}, Metadata: {metadata}")

# ID: 2, Sentence: "I like cats", Distance: 0.6755620241165161, Metadata: {'animal': 'cat', 'like': True}
# ID: 1, Sentence: "I like dogs", Distance: 0.41838371753692627, Metadata: {'animal': 'dog', 'like': True}

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

# Save the database to disk
vector_db.persist_to_disk()
```

## **License**

This project is licensed under the MIT License.