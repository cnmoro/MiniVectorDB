import numpy as np, faiss, pickle, os
from collections import defaultdict

class VectorDatabase:
    def __init__(self, storage_file='db.pkl', embedding_size=512):
        self.embedding_size = embedding_size
        self.storage_file = storage_file
        self.embeddings = np.zeros((0, self.embedding_size), dtype=np.float32)
        self.metadata = []  # Stores dictionaries of metadata
        self.id_map = {}  # Maps embedding row number to unique id
        self.inverse_id_map = {}  # Maps unique id to embedding row number
        self.inverted_index = defaultdict(set)  # Inverted index for metadata
        self.index = None
        self._embeddings_changed = False
        self._load_database()

    def _convert_ndarray_float32(self, ndarray):
        return np.array(ndarray, dtype=np.float32)

    def _load_database(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.metadata = data['metadata']
                self.id_map = data['id_map']
                self.inverse_id_map = data['inverse_id_map']
                self.inverted_index = data.get('inverted_index', defaultdict(set))
                self._build_index()

    def _build_index(self):
        self.index = faiss.IndexFlatIP(self.embedding_size)  # Inner Product (cosine similarity)
        if self.embeddings.shape[0] > 0:
            faiss.normalize_L2(self.embeddings)  # Normalize for cosine similarity
            self.index.add(self.embeddings)
            self._embeddings_changed = False

    def get_vector(self, unique_id):
        if unique_id not in self.inverse_id_map:
            raise ValueError("Unique ID does not exist.")
        
        row_num = self.inverse_id_map[unique_id]
        return self.embeddings[row_num]

    def store_embedding(self, unique_id, embedding, metadata_dict={}):
        if unique_id in self.inverse_id_map:
            raise ValueError("Unique ID already exists.")

        embedding = self._convert_ndarray_float32(embedding)
        row_num = self.embeddings.shape[0]
        self.embeddings = np.vstack([self.embeddings, embedding])
        self.metadata.append(metadata_dict)
        self.id_map[row_num] = unique_id
        self.inverse_id_map[unique_id] = row_num

        # Update the inverted index
        for key, _ in metadata_dict.items():
            self.inverted_index[key].add(unique_id)

        self._embeddings_changed = True

    def delete_embedding(self, unique_id):
        if unique_id not in self.inverse_id_map:
            raise ValueError("Unique ID does not exist.")

        row_num = self.inverse_id_map[unique_id]
        # Delete the embedding and metadata
        self.embeddings = np.delete(self.embeddings, row_num, 0)
        metadata_to_delete = self.metadata.pop(row_num)

        # Update the inverted index
        for key, _ in metadata_to_delete.items():
            self.inverted_index[key].discard(unique_id)
            if not self.inverted_index[key]:  # If the set is empty, remove the key
                del self.inverted_index[key]

        # Delete from id_map and inverse_id_map
        del self.id_map[row_num]
        del self.inverse_id_map[unique_id]

        # Update the id_map and inverse_id_map to reflect the new indices
        new_id_map = {}
        new_inverse_id_map = {}
        for index, uid in enumerate(self.id_map.values()):
            new_id_map[index] = uid
            new_inverse_id_map[uid] = index

        self.id_map = new_id_map
        self.inverse_id_map = new_inverse_id_map

        # Since we've modified the embeddings, we must rebuild the index before the next search
        self._embeddings_changed = True

    def _get_filtered_indices(self, metadata_filter):
        if not metadata_filter:
            return set(self.inverse_id_map.values())

        # Start with the full set of IDs and narrow down with each metadata filter
        filtered_indices = None
        for key, value in metadata_filter.items():
            indices = {self.inverse_id_map[uid] for uid in self.inverted_index.get(key, set()) if self.metadata[self.inverse_id_map[uid]].get(key) == value}
            if filtered_indices is None:
                filtered_indices = indices
            else:
                filtered_indices &= indices

            if not filtered_indices:
                break

        return filtered_indices if filtered_indices is not None else set()

    def find_most_similar(self, embedding, metadata_filter={}, k=5):
        embedding = self._convert_ndarray_float32(embedding)
        embedding = np.array([embedding])
        faiss.normalize_L2(embedding)

        filtered_indices = self._get_filtered_indices(metadata_filter)

        # If embeddings or metadata have changed, rebuild the index
        if self._embeddings_changed:
            self._build_index()

        # If no filtered indices, return empty results
        if not filtered_indices:
            return [], [], []

        # Search in the FAISS index
        distances, indices = self.index.search(embedding, k)
        # Filter the results based on the filtered indices
        filtered_results = [(self.id_map[idx], dist, self.metadata[idx]) for idx, dist in zip(indices[0], distances[0]) if idx in filtered_indices]

        # Sort the filtered results by distance and take top k
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        # Unzip the results into separate lists
        ids, distances, metadatas = zip(*filtered_results[:k])
        
        return ids, distances, metadatas

    def persist_to_disk(self):
        with open(self.storage_file, 'wb') as f:
            data = {
                'embeddings': self.embeddings,
                'metadata': self.metadata,
                'id_map': self.id_map,
                'inverse_id_map': self.inverse_id_map,
                'inverted_index': self.inverted_index
            }
            pickle.dump(data, f)
