## **MiniVectorDB**

This is a Python project aimed at extracting embeddings from textual data and performing semantic search. It's a simple yet powerful system combining a small quantized ONNX model with FAISS indexing for fast similarity search. As the model is small and also running in ONNX runtime with quantization, we get lightning fast speed.

Model link in Huggingface: [universal-sentence-encoder-multilingual-3-onnx-quantized](https://huggingface.co/WiseIntelligence/universal-sentence-encoder-multilingual-3-onnx-quantized)

## **Features**

*   **Embedding Model**: Load the ONNX model to extract embeddings from text.
*   **Vector Database**: Store and manage textual embeddings, perform fast similarity searches using FAISS.

## **Getting Started**

### **Prerequisites**

*   Python 3.11
*   ONNX Runtime + Extensions
*   FAISS
*   NumPy
*   pytest (for testing)

or use `pip install -r requirements.txt`

### **Installation**

```plaintext
pip install minivectordb
```

### **Usage**

```python
from minivectordb.embedding_model import EmbeddingModel
from minivectordb.vector_database import VectorDatabase

vector_db = VectorDatabase(embedding_size = 512)
model = EmbeddingModel()

# Text identifier and sentences
sentences = [
    (1,  "I like dogs"),
    (2,  "I like cats"),
    (3,  "The king has three kids"),
    (4,  "The queen has one daughter"),
    (5,  "Programming is cool"),
    (6,  "Software development is cool"),
    (7,  "I like to ride my bicycle"),
    (8,  "I like to ride my scooter"),
    (9,  "The sky is blue"),
    (10, "The ocean is blue")
]

for id, sentence in sentences:
    sentence_embedding = model.extract_embeddings(sentence)
    vector_db.store_embedding(id, sentence_embedding)

##  Semantic Search

query_embedding = model.extract_embeddings("I like cats")
search_results = vector_db.find_most_similar(query_embedding, k = 5)

ids, distances = search_results
for id, dist in zip(ids, distances):
    print("ID:", id, "Distance:", dist)

# Output:
# ID: 2 Distance: 1.0
# ID: 1 Distance: 0.7593117
# ID: 8 Distance: 0.42757708
# ID: 7 Distance: 0.41723043
# ID: 5 Distance: 0.27484077

##################################################################

query_embedding = model.extract_embeddings("I am a programmer")
search_results = vector_db.find_most_similar(query_embedding, k = 5)

ids, distances = search_results
for id, dist in zip(ids, distances):
    print("ID:", id, "Distance:", dist)

# Output:
# ID: 5 Distance: 0.6494667
# ID: 6 Distance: 0.47456568
# ID: 1 Distance: 0.31276548
# ID: 2 Distance: 0.28922778
# ID: 7 Distance: 0.21100259

# Save the database to disk
vector_db.persist_to_disk()
```

## **Testing**

Ensure you have **pytest** and **pytest-cov** installed. Run the tests using:

```plaintext
pytest --cov=minivectordb
```

For detailed coverage reports:

```plaintext
pytest --cov=minivectordb --cov-report=term-missing
```

## **Contributing**

1.  Fork the repository on GitHub.
2.  Clone your fork locally: **git clone https://github.com/yourusername/your-repo-name.git**
3.  Create a new branch for your feature or fix: **git checkout -b your-branch-name**
4.  Commit your changes and push to your fork: **git push origin your-branch-name**
5.  Create a new Pull Request from your fork to the main repository.

## **License**

This project is licensed under the MIT License.