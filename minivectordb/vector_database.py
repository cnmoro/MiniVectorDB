import numpy as np, faiss, pickle, os

class VectorDatabase:
    def __init__(self, storage_file='db.pkl', embedding_size=512):
        self.embedding_size = embedding_size
        self.storage_file = storage_file
        self.embeddings = np.zeros((0, self.embedding_size), dtype=np.float32)
        self.id_map = {}  # Maps embedding row number to unique id
        self.inverse_id_map = {}  # Maps unique id to embedding row number
        self.index = None
        self._embeddings_changed = False
        self._load_database()

    def _load_database(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.id_map = data['id_map']
                self.inverse_id_map = data['inverse_id_map']
                self._build_index()

    def _build_index(self):
        self.index = faiss.IndexFlatIP(self.embedding_size)  # Change L2 to IP
        if self.embeddings.shape[0] > 0:
            # Normalize embeddings before adding them
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)

    def store_embedding(self, unique_id, embedding):
        if unique_id in self.inverse_id_map:
            raise ValueError("Unique ID already exists.")
        
        row_num = self.embeddings.shape[0]
        self.embeddings = np.vstack([self.embeddings, embedding])
        self.id_map[row_num] = unique_id
        self.inverse_id_map[unique_id] = row_num
        
        # Set the flag to true because we've modified embeddings
        self._embeddings_changed = True
    
    def get_vector(self, unique_id):
        if unique_id not in self.inverse_id_map:
            raise ValueError("Unique ID does not exist.")
        
        row_num = self.inverse_id_map[unique_id]
        return self.embeddings[row_num]

    def delete_embedding(self, unique_id):
        if unique_id not in self.inverse_id_map:
            raise ValueError("Unique ID does not exist.")
        
        row_num = self.inverse_id_map[unique_id]
        self.embeddings = np.delete(self.embeddings, row_num, 0)
        del self.id_map[row_num]
        del self.inverse_id_map[unique_id]
        # Rebuild id maps
        new_id_map = {}
        for i, uid in enumerate(self.id_map.values()):
            new_id_map[i] = uid
        self.id_map = new_id_map
        self.inverse_id_map = {v: k for k, v in self.id_map.items()}
        
        # Set the flag to true because we've modified embeddings
        self._embeddings_changed = True

    def find_most_similar(self, embedding, k=5):
        # Check if embeddings have changed and rebuild index if needed
        if self._embeddings_changed:
            self._build_index()
            self._embeddings_changed = False  # Reset the flag

        # Normalize the query embedding
        embedding = np.array([embedding])
        faiss.normalize_L2(embedding)
        distances, indices = self.index.search(embedding, k)
        ids = [self.id_map[idx] for idx in indices[0]]
        return ids, distances[0]
    
    def persist_to_disk(self):
        with open(self.storage_file, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'id_map': self.id_map,
                'inverse_id_map': self.inverse_id_map
            }, f)
