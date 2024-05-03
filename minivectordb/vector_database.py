import numpy as np, faiss, pickle, os, threading
from operator import gt, ge, lt, le, ne
from collections import defaultdict
from rank_bm25 import BM25Okapi
from thefuzz import fuzz

class VectorDatabase:
    def __init__(self, storage_file='db.pkl'):
        self.embedding_size = None
        self.storage_file = storage_file
        self.embeddings = None
        self.metadata = []  # Stores dictionaries of metadata
        self.id_map = {}  # Maps embedding row number to unique id
        self.inverse_id_map = {}  # Maps unique id to embedding row number
        self.inverted_index = defaultdict(set)  # Inverted index for metadata
        self.index = None
        self._embeddings_changed = False
        self.lock = threading.Lock()
        self._load_database()

    def _convert_ndarray_float32(self, ndarray):
        return np.array(ndarray, dtype=np.float32)

    def _convert_ndarray_float32_batch(self, ndarrays):
        return [np.array(arr, dtype=np.float32) for arr in ndarrays]

    def _load_database(self):
        if os.path.exists(self.storage_file):
            with self.lock:
                with open(self.storage_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data['embeddings']
                    self.embedding_size = data['embeddings'].shape[1] if data['embeddings'] is not None else None
                    self.metadata = data['metadata']
                    self.id_map = data['id_map']
                    self.inverse_id_map = data['inverse_id_map']
                    self.inverted_index = data.get('inverted_index', defaultdict(set))
            if self.embedding_size is not None:
                self._build_index()

    def _build_index(self):
        with self.lock:
            self.index = faiss.IndexFlatIP(self.embedding_size)
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

        if self.embedding_size is None:
            self.embedding_size = embedding.shape[0]

        if self.embeddings is None:
            self.embeddings = np.zeros((0, self.embedding_size), dtype=np.float32)

        row_num = self.embeddings.shape[0]

        with self.lock:
            self.embeddings = np.vstack([self.embeddings, embedding])
            self.metadata.append(metadata_dict)
            self.id_map[row_num] = unique_id
            self.inverse_id_map[unique_id] = row_num

            # Update the inverted index
            for key, _ in metadata_dict.items():
                self.inverted_index[key].add(unique_id)

            self._embeddings_changed = True

    def store_embeddings_batch(self, unique_ids, embeddings, metadata_dicts=[]):
        for uid in unique_ids:
            if uid in self.inverse_id_map:
                raise ValueError("Unique ID already exists.")
        
        if self.embedding_size is None:
            self.embedding_size = embeddings[0].shape[0]
        
        if self.embeddings is None:
            self.embeddings = np.zeros((0, self.embedding_size), dtype=np.float32)
        
        if len(metadata_dicts) < len(unique_ids) and len(metadata_dicts) > 0:
            raise ValueError("Metadata dictionaries must be provided for all unique IDs.")

        if metadata_dicts == []:
            metadata_dicts = [{} for _ in range(len(unique_ids))]
        
        # Convert all embeddings to float32
        embeddings = self._convert_ndarray_float32_batch(embeddings)

        row_nums = list(range(self.embeddings.shape[0], self.embeddings.shape[0] + len(embeddings)))
        
        # Stack the embeddings with a single operation
        with self.lock:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.metadata.extend(metadata_dicts)
            self.id_map.update({row_num: unique_id for row_num, unique_id in zip(row_nums, unique_ids)})
            self.inverse_id_map.update({unique_id: row_num for row_num, unique_id in zip(row_nums, unique_ids)})

            # Update the inverted index
            for i, metadata_dict in enumerate(metadata_dicts):
                for key, _ in metadata_dict.items():
                    self.inverted_index[key].add(unique_ids[i])

            self._embeddings_changed = True

    def delete_embedding(self, unique_id):
        if unique_id not in self.inverse_id_map:
            raise ValueError("Unique ID does not exist.")

        with self.lock:
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

    def _apply_or_filter(self, or_filters):
        result_indices = set()
        for filter in or_filters:
            key_indices = set()
            for key, value in filter.items():
                # Check if the value is a dictionary containing operators
                if isinstance(value, dict):
                    op = next(iter(value))  # Get the operator
                    op_value = value[op]  # Get the value for the operator
                    op_func = {
                        "$gt": gt,
                        "$gte": ge,
                        "$lt": lt,
                        "$lte": le,
                        "$ne": ne,
                        "$in": lambda x, y: x in y,
                    }.get(op, None)
                    if op_func is None:
                        raise ValueError(f"Invalid operator: {op}")

                    # Create a copy of the set for iteration
                    inverted_index_copy = self.inverted_index.get(key, set()).copy()
                    key_indices.update({self.inverse_id_map[uid] for uid in inverted_index_copy
                                        if op_func(self.metadata[self.inverse_id_map[uid]].get(key, None), op_value)})
                else:
                    # Create a copy of the set for iteration
                    inverted_index_copy = self.inverted_index.get(key, set()).copy()
                    key_indices.update({self.inverse_id_map[uid] for uid in inverted_index_copy
                                        if self.metadata[self.inverse_id_map[uid]].get(key, None) == value})
            result_indices |= key_indices

        return result_indices

    def _apply_and_filter(self, and_filters, filtered_indices):
        for metadata_filter in and_filters:
            for key, value in metadata_filter.items():
                # Check if the value is a dictionary containing operators
                if isinstance(value, dict):
                    op = next(iter(value))  # Get the operator
                    op_value = value[op]  # Get the value for the operator
                    op_func = {
                        "$gt": gt,
                        "$gte": ge,
                        "$lt": lt,
                        "$lte": le,
                        "$ne": ne,
                        "$in": lambda x, y: y in x,
                    }.get(op, None)
                    if op_func is None:
                        raise ValueError(f"Invalid operator: {op}")

                    indices = {self.inverse_id_map[uid] for uid in self.inverted_index.get(key, set())
                            if op_func(self.metadata[self.inverse_id_map[uid]].get(key, None), op_value)}
                else:
                    indices = {self.inverse_id_map[uid] for uid in self.inverted_index.get(key, set())
                            if self.metadata[self.inverse_id_map[uid]].get(key, None) == value}

                if filtered_indices is None:
                    filtered_indices = indices
                else:
                    # Create a copy of filtered_indices for iteration
                    for index in filtered_indices.copy():
                        if index not in indices:
                            filtered_indices.remove(index)

                if not filtered_indices:
                    break
        
        return filtered_indices
    
    def _apply_exclude_filter(self, exclude_filter, filtered_indices):
        for exclude in exclude_filter:
            for key, value in exclude.items():
                # Create a copy of the set for iteration
                inverted_index_copy = self.inverted_index.get(key, set()).copy()
                exclude_indices = {self.inverse_id_map[uid] for uid in inverted_index_copy
                                   if self.metadata[self.inverse_id_map[uid]].get(key, None) == value}
                filtered_indices -= exclude_indices
                if not filtered_indices:
                    break

        return filtered_indices

    def _get_filtered_indices(self, metadata_filters, exclude_filter, or_filters):
        # Initialize filtered_indices with all indices if metadata_filters is not provided
        filtered_indices = set(self.inverse_id_map.values()) if not metadata_filters else None

        # Check if metadata_filters is a dict, if so, convert to list of dicts
        if isinstance(metadata_filters, dict):
            metadata_filters = [metadata_filters]

        # Apply metadata_filters (AND)
        if metadata_filters:
            filtered_indices = self._apply_and_filter(metadata_filters, filtered_indices)

        # Apply OR filters
        if or_filters:
            # Remove all empty dictionaries from or_filters
            if isinstance(or_filters, dict):
                or_filters = [or_filters]
            or_filters = [or_filter for or_filter in or_filters if or_filter]
            if or_filters:
                temp_indices = self._apply_or_filter(or_filters)
                if filtered_indices is None:
                    filtered_indices = temp_indices
                else:
                    filtered_indices &= temp_indices

        # Apply exclude_filter
        if exclude_filter:
            # Check if exclude_filter is a dict, if so, convert to list of dicts
            if isinstance(exclude_filter, dict):
                exclude_filter = [exclude_filter]
            filtered_indices = self._apply_exclude_filter(exclude_filter, filtered_indices)

        return filtered_indices if filtered_indices is not None else set()

    def _calculate_bm25_scores(self, query, documents):
        if len(documents) == 0:
            return []
        tokenized_query = query.split()
        bm25 = BM25Okapi([doc.split() for doc in documents])
        return bm25.get_scores(tokenized_query)

    def _calculate_fuzzy_ratios(self, query, documents):
        return [fuzz.partial_ratio(query, doc) for doc in documents]
    
    def hybrid_rerank_results(self, sentences, search_scores, query, k=5, weights=(0.80, 0.15, 0.05)):
        bm25_scores = self._calculate_bm25_scores(query, sentences)
        fuzzy_scores = self._calculate_fuzzy_ratios(query, sentences)

        # If bm25_scores is empty, return the sentences and search_scores as is
        if len(bm25_scores) == 0:
            return sentences[:k], search_scores[:k]

        # Combine scores
        search_weight, bm25_weight, fuzzy_weight = weights
        combined_scores = search_weight * np.array(search_scores) + bm25_weight * np.array(bm25_scores) + fuzzy_weight * np.array(fuzzy_scores)

        # Merge sentences and scores
        sentences = np.array(sentences)
        combined_scores = np.array(combined_scores)
        combined_results = np.column_stack((sentences, combined_scores))

        # Sort by combined scores and sort by combined scores descending
        combined_results = combined_results[combined_results[:, 1].argsort()[::-1]]

        # Unzip the results into separate lists
        sentences, combined_scores = zip(*combined_results)

        # Trim results to requested k
        return sentences[:k], combined_scores[:k]

    def autocut_scores(self, score_list):
        """
        This function takes a list of scores and determines if there is a significant drop in the scores.
        If there is a drop greater than 20%, it returns the indices of the scores that would be removed.
        If there is no such drop, it returns an empty list.

        Inspired by weaviate's golden ragtriever autocut feature.
        This is a basic implementation and can be improved.
        """
        # Find the percentage of each score decrease
        score_decreases = []
        for i in range(1, len(score_list)):
            score_decreases.append((score_list[i-1] - score_list[i]) / score_list[i-1])
        
        # Find the highest score decrease
        max_score_decrease = max(score_decreases)

        if max_score_decrease > 0.2:
            # Return all the indexes that would be removed if cut at the index of the max_score_decrease
            return list(range(score_decreases.index(max_score_decrease) + 1, len(score_list)))
        
        return []

    def find_most_similar(self, embedding, metadata_filter=None, exclude_filter=None, or_filters=None, k=5, autocut=False):
        """ or_filters could be a list of dictionaries, where each dictionary contains key-value pairs for OR filters.
        or it could be a single dictionary, which will be equivalent to a list with a single dictionary."""
        embedding = self._convert_ndarray_float32(embedding)
        embedding = np.array([embedding])
        faiss.normalize_L2(embedding)

        filtered_indices = self._get_filtered_indices(metadata_filter, exclude_filter, or_filters)

        # If embeddings or metadata have changed, rebuild the index
        if self._embeddings_changed:
            self._build_index()

        # If no filtered indices, return empty results
        if not filtered_indices:
            return [], [], []

        # Determine the maximum number of possible matches
        max_possible_matches = min(k, len(filtered_indices))

        found_results = []
        search_k = max_possible_matches

        # Check if filtered_indices corresponds to all possible matches
        if len(filtered_indices) == self.embeddings.shape[0]:
            # Simply perform the search
            distances, indices = self.index.search(embedding, search_k)

            for idx, dist in zip(indices[0], distances[0]):
                found_results.append((self.id_map[idx], dist, self.metadata[idx]))
        else:
            # Otherwise, we create a new index with only the filtered indices
            filtered_embeddings = self.embeddings[list(filtered_indices)]
            filtered_index = faiss.IndexFlatIP(self.embedding_size)
            filtered_index.add(filtered_embeddings)

            distances, indices = filtered_index.search(embedding, search_k)

            for idx, dist in zip(indices[0], distances[0]):
                found_results.append((self.id_map[list(filtered_indices)[idx]], dist, self.metadata[list(filtered_indices)[idx]]))

        # Unzip the results into separate lists
        ids, distances, metadatas = zip(*found_results) if found_results else ([], [], [])

        if autocut and len(distances) > 1:
            # Remove results that are not within 20% of the best result
            remove_indexes = self.autocut_scores(distances)
            if remove_indexes:
                ids = [ids[i] for i in range(len(ids)) if i not in remove_indexes]
                distances = [distances[i] for i in range(len(distances)) if i not in remove_indexes]
                metadatas = [metadatas[i] for i in range(len(metadatas)) if i not in remove_indexes]

        return ids, distances, metadatas

    def persist_to_disk(self):
        with self.lock:
            with open(self.storage_file, 'wb') as f:
                data = {
                    'embeddings': self.embeddings,
                    'metadata': self.metadata,
                    'id_map': self.id_map,
                    'inverse_id_map': self.inverse_id_map,
                    'inverted_index': self.inverted_index
                }
                pickle.dump(data, f)
