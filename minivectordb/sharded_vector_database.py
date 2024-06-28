from sklearn.feature_extraction.text import HashingVectorizer
from minivectordb.vector_database import VectorDatabase
import numpy as np, faiss, pickle, os, threading
from operator import gt, ge, lt, le, ne
from collections import defaultdict
from thefuzz import fuzz

class ShardedVectorDatabase:
    def __init__(self, storage_dir='db_shards', shard_size=5000):
        self.hash_vectorizer = HashingVectorizer(ngram_range=(1, 6), analyzer='char', n_features=64)
        self.embedding_size = None
        self.storage_dir = storage_dir
        self.shard_size = shard_size
        self.embeddings = None
        self.metadata = []
        self.unique_ids = []
        self.inverse_id_map = {}
        self.inverted_index = defaultdict(set)
        self.index = None
        self._embeddings_changed = False
        self.lock = threading.Lock()
        self.box_item_map = {}  # New attribute to track embeddings in shards
        self.inverse_box_item_map = {}  # New attribute to track which shard an embedding is in
        self._load_database()

    def _convert_from_non_sharded_db(self, non_sharded_db_object: VectorDatabase):
        # Transfer embeddings and metadata
        embeddings = non_sharded_db_object.embeddings
        metadata = non_sharded_db_object.metadata
        unique_ids = [non_sharded_db_object.id_map[i] for i in range(len(embeddings))]

        self.store_embeddings_batch(unique_ids, embeddings, metadata)
        del non_sharded_db_object

    def _convert_ndarray_float32(self, ndarray):
        return np.array(ndarray, dtype=np.float32)

    def _convert_ndarray_float32_batch(self, ndarrays):
        return [np.array(arr, dtype=np.float32) for arr in ndarrays]

    def _load_database(self):
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        
        shard_files = [f for f in os.listdir(self.storage_dir) if f.endswith('.pkl')]
        for shard_file in shard_files:
            with self.lock:
                with open(os.path.join(self.storage_dir, shard_file), 'rb') as f:
                    data = pickle.load(f)
                    if self.embeddings is None:
                        self.embeddings = data['embeddings']
                    else:
                        self.embeddings = np.vstack([self.embeddings, data['embeddings']])
                    self.metadata.extend(data['metadata'])
                    self.unique_ids.extend(data['unique_ids'])
                    self.inverted_index.update(data['inverted_index'])
                    self._update_box_item_map(data['unique_ids'], shard_file)
        
        self.inverse_id_map = {uid: i for i, uid in enumerate(self.unique_ids)}
        
        if self.embeddings is not None and self.embeddings.shape[0] > 0:
            self.embedding_size = self.embeddings.shape[1]
            self._build_index()
    
    def _update_box_item_map(self, unique_ids, shard_file):
        shard_id = int(os.path.basename(shard_file).split('_')[1].split('.')[0])
        self.box_item_map[shard_id] = unique_ids
        for uid in unique_ids:
            self.inverse_box_item_map[uid] = shard_id

    def _build_index(self):
        self.index = faiss.IndexFlatIP(self.embedding_size)
        if self.embeddings.shape[0] > 0:
            faiss.normalize_L2(self.embeddings)  # Normalize for cosine similarity
            self.index.add(self.embeddings)
            self._embeddings_changed = False

    def get_vector(self, unique_id):
        with self.lock:
            if unique_id not in self.inverse_id_map:
                raise ValueError("Unique ID does not exist.")
            
            index = self.inverse_id_map[unique_id]
            shard_id = self.inverse_box_item_map[unique_id]
            shard_file = os.path.join(self.storage_dir, f'shard_{shard_id}.pkl')
            with open(shard_file, 'rb') as f:
                data = pickle.load(f)
                return data['embeddings'][index]
    
    def _get_available_shard_id(self):
        for shard_id, items in self.box_item_map.items():
            if len(items) < self.shard_size:
                return shard_id
        return len(self.box_item_map)

    def store_embedding(self, unique_id, embedding, metadata_dict={}):
        with self.lock:
            if unique_id in self.inverse_id_map:
                raise ValueError("Unique ID already exists.")

            embedding = self._convert_ndarray_float32(embedding)

            if self.embedding_size is None:
                self.embedding_size = embedding.shape[0]

            if self.embeddings is None:
                self.embeddings = np.zeros((0, self.embedding_size), dtype=np.float32)

            self.embeddings = np.vstack([self.embeddings, embedding])
            self.metadata.append(metadata_dict)
            self.unique_ids.append(unique_id)
            new_index = len(self.unique_ids) - 1
            self.inverse_id_map[unique_id] = new_index

            for key, value in metadata_dict.items():
                self.inverted_index[key].add(unique_id)

            self._embeddings_changed = True
            shard_id = self._get_available_shard_id()
            if shard_id not in self.box_item_map:
                self.box_item_map[shard_id] = []
            self.box_item_map[shard_id].append(unique_id)
            self.inverse_box_item_map[unique_id] = shard_id
            self._persist_to_shard(shard_id, unique_id, embedding, metadata_dict)

    def _persist_to_shard(self, shard_id, unique_id, embedding, metadata_dict):
        shard_file = os.path.join(self.storage_dir, f'shard_{shard_id}.pkl')
        if os.path.exists(shard_file):
            with open(shard_file, 'rb') as f:
                data = pickle.load(f)
        else:
            data = {'embeddings': np.zeros((0, self.embedding_size), dtype=np.float32),
                    'metadata': [], 'unique_ids': [], 'inverted_index': defaultdict(set)}

        data['embeddings'] = np.vstack([data['embeddings'], embedding])
        data['metadata'].append(metadata_dict)
        data['unique_ids'].append(unique_id)
        for key, value in metadata_dict.items():
            data['inverted_index'][key].add(unique_id)

        with open(shard_file, 'wb') as f:
            pickle.dump(data, f)
            
    def _persist_to_shard_multiple(self, shard_id, unique_ids, embeddings, metadata_dicts):
        shard_file = os.path.join(self.storage_dir, f'shard_{shard_id}.pkl')
        if os.path.exists(shard_file):
            with open(shard_file, 'rb') as f:
                data = pickle.load(f)
        else:
            data = {'embeddings': np.zeros((0, self.embedding_size), dtype=np.float32),
                    'metadata': [], 'unique_ids': [], 'inverted_index': defaultdict(set)}
            
        data['embeddings'] = np.vstack([data['embeddings'], embeddings])
        data['metadata'].extend(metadata_dicts)
        data['unique_ids'].extend(unique_ids)
        
        for metadata_dict, unique_id in zip(metadata_dicts, unique_ids):
            for key, value in metadata_dict.items():
                data['inverted_index'][key].add(unique_id)

        with open(shard_file, 'wb') as f:
            pickle.dump(data, f)

    def _remove_embeddings_from_shard(self, shard_id, unique_ids):
        shard_file = os.path.join(self.storage_dir, f'shard_{shard_id}.pkl')
        with open(shard_file, 'rb') as f:
            data = pickle.load(f)

        unique_ids_set = set(unique_ids)
        indices_to_keep = [idx for idx, uid in enumerate(data['unique_ids']) if uid not in unique_ids_set]

        # if len(indices_to_keep) == len(data['unique_ids']):
        #     return

        data['embeddings'] = data['embeddings'][indices_to_keep]
        data['metadata'] = [data['metadata'][i] for i in indices_to_keep]
        data['unique_ids'] = [data['unique_ids'][i] for i in indices_to_keep]

        for uid in unique_ids_set:
            for key, ids in list(data['inverted_index'].items()):
                if uid in ids:
                    ids.discard(uid)
                    if not ids:
                        del data['inverted_index'][key]

        with open(shard_file, 'wb') as f:
            pickle.dump(data, f)

        self.box_item_map[shard_id] = data['unique_ids']
        for uid in unique_ids_set:
            del self.inverse_box_item_map[uid]

    def delete_embeddings_batch(self, unique_ids):
        with self.lock:
            if not isinstance(unique_ids, list):
                unique_ids = [unique_ids]

            # If no unique IDs are provided, raise ValueError
            if not unique_ids:
                raise ValueError("No unique IDs provided.")
            
            # If none of the unique IDs exist, raise ValueError
            if not all(uid in self.inverse_id_map for uid in unique_ids):
                raise ValueError("One or more unique IDs do not exist.")

            unique_ids = [uid for uid in unique_ids if uid is not None]

            shard_groups = defaultdict(list)
            for unique_id in unique_ids:
                shard_id = self.inverse_box_item_map[unique_id]
                shard_groups[shard_id].append(unique_id)

            for shard_id, shard_unique_ids in shard_groups.items():
                self._remove_embeddings_from_shard(shard_id, shard_unique_ids)

            indices_to_keep = [i for i, uid in enumerate(self.unique_ids) if uid not in unique_ids]
            self.embeddings = self.embeddings[indices_to_keep]
            self.metadata = [self.metadata[i] for i in indices_to_keep]
            self.unique_ids = [uid for uid in self.unique_ids if uid not in unique_ids]

            for uid in unique_ids:
                for key, ids in list(self.inverted_index.items()):
                    ids.discard(uid)
                    if not ids:
                        del self.inverted_index[key]

            self.inverse_id_map = {uid: i for i, uid in enumerate(self.unique_ids)}
            self._embeddings_changed = True

    def store_embeddings_batch(self, unique_ids, embeddings, metadata_dicts=[]):
        with self.lock:
            if len(unique_ids) != len(embeddings):
                raise ValueError("Number of unique IDs must match number of embeddings.")

            for uid in unique_ids:
                if uid in self.inverse_id_map:
                    raise ValueError(f"Unique ID {uid} already exists.")
            
            if self.embedding_size is None:
                self.embedding_size = embeddings[0].shape[0]
            
            if self.embeddings is None:
                self.embeddings = np.zeros((0, self.embedding_size), dtype=np.float32)
            
            if len(metadata_dicts) < len(unique_ids):
                metadata_dicts.extend([{} for _ in range(len(unique_ids) - len(metadata_dicts))])
            
            embeddings = self._convert_ndarray_float32_batch(embeddings)
            current_count = len(self.unique_ids)
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.metadata.extend(metadata_dicts)
            self.unique_ids.extend(unique_ids)

            self.inverse_id_map.update({uid: i for i, uid in enumerate(unique_ids, start=current_count)})

            for uid, metadata_dict in zip(unique_ids, metadata_dicts):
                for key, value in metadata_dict.items():
                    self.inverted_index[key].add(uid)

            self._embeddings_changed = True

            shard_groups = defaultdict(list)
            for i, (uid, embedding, metadata_dict) in enumerate(zip(unique_ids, embeddings, metadata_dicts)):
                shard_id = self._get_available_shard_id()
                shard_groups[shard_id].append((uid, embedding, metadata_dict))
                if shard_id not in self.box_item_map:
                    self.box_item_map[shard_id] = []
                self.box_item_map[shard_id].append(uid)
                self.inverse_box_item_map[uid] = shard_id

            for shard_id, shard_data in shard_groups.items():
                uids, embeddings, metadata_dicts = zip(*shard_data)
                self._persist_to_shard_multiple(shard_id, uids, embeddings, metadata_dicts)

    def _apply_or_filter(self, or_filters):
        result_indices = set()
        for filter in or_filters:
            key_indices = set()
            for key, value in filter.items():
                # Check if the value is a dictionary containing operators
                if isinstance(value, dict):
                    op = next(iter(value)) # Get the operator
                    op_value = value[op] # Get the value for the operator
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

                    try:
                        # Create a copy of the set for iteration
                        inverted_index_copy = self.inverted_index.get(key, set()).copy()

                        key_indices_update = set()

                        # Iterate over each user ID in the inverted index copy
                        for uid in inverted_index_copy:
                            # Get the corresponding index for the user ID from the inverse_id_map
                            if uid not in self.inverse_id_map:
                                continue

                            inverse_id = self.inverse_id_map[uid]

                            metadata = self.metadata[inverse_id]

                            # Get the value for the key from the metadata, if it doesn't exist, return None
                            metadata_value = metadata.get(key, None)

                            # Check if the operation function returns True when applied to the metadata value and the operation value
                            if op_func(metadata_value, op_value):
                                # If it does, add the index to the key_indices_update set
                                key_indices_update.add(inverse_id)

                        # Update the key_indices set with the key_indices_update set
                        key_indices.update(key_indices_update)
                    except KeyError:
                        continue
                else:
                    try:
                        # Create a copy of the set for iteration
                        inverted_index_copy = self.inverted_index.get(key, set()).copy()

                        key_indices_update = set()

                        # Iterate over each user ID in the inverted index copy
                        for uid in inverted_index_copy:
                            # Get the corresponding index for the user ID from the inverse_id_map
                            if uid not in self.inverse_id_map:
                                continue

                            inverse_id = self.inverse_id_map[uid]

                            metadata = self.metadata[inverse_id]

                            # Get the value for the key from the metadata, if it doesn't exist, return None
                            metadata_value = metadata.get(key, None)

                            # Check if the metadata value matches the given value
                            if metadata_value == value:
                                # If it does, add the index to the key_indices_update set
                                key_indices_update.add(inverse_id)

                        # Update the key_indices set with the key_indices_update set
                        key_indices.update(key_indices_update)
                    except KeyError:
                        continue
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

                    try:
                        indices = set()

                        # Get the set of user IDs from the inverted index for the given key. If the key is not present, return an empty set.
                        uids = self.inverted_index.get(key, set())

                        # Iterate over each user ID in the set
                        for uid in uids:
                            # Get the corresponding index for the user ID from the inverse_id_map
                            if uid not in self.inverse_id_map:
                                continue

                            inverse_id = self.inverse_id_map[uid]

                            metadata = self.metadata[inverse_id]

                            # Get the value for the key from the metadata, if it doesn't exist, return None
                            metadata_value = metadata.get(key, None)

                            # Check if the operation function returns True when applied to the metadata value and the operation value
                            if op_func(metadata_value, op_value):
                                # If it does, add the index to the indices set
                                indices.add(inverse_id)
                    except KeyError:
                        indices = set()
                else:
                    try:
                        indices = set()

                        # Get the set of user IDs from the inverted index for the given key. If the key is not present, return an empty set.
                        uids = self.inverted_index.get(key, set())

                        # Iterate over each user ID in the set
                        for uid in uids:

                            # Get the corresponding index for the user ID from the inverse_id_map
                            if uid not in self.inverse_id_map:
                                continue

                            inverse_id = self.inverse_id_map[uid]

                            metadata = self.metadata[inverse_id]

                            # Check if the key exists in the metadata and if its value matches the given value
                            if metadata.get(key, None) == value:
                                # If it does, add the index to the indices set
                                indices.add(inverse_id)

                    except KeyError:
                        indices = set()

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
                try:
                    # Create a copy of the set for iteration
                    inverted_index_copy = self.inverted_index.get(key, set()).copy()

                    exclude_indices = set()

                    # Iterate over each user ID in the inverted index copy
                    for uid in inverted_index_copy:
                        # Get the corresponding index for the user ID from the inverse_id_map
                        if uid not in self.inverse_id_map:
                            continue

                        inverse_id = self.inverse_id_map[uid]

                        metadata = self.metadata[inverse_id]

                        # Get the value for the key from the metadata, if it doesn't exist, return None
                        metadata_value = metadata.get(key, None)

                        # Check if the metadata value matches the given value
                        if metadata_value == value:
                            # If it does, add the index to the exclude_indices set
                            exclude_indices.add(inverse_id)
                except KeyError:
                    exclude_indices = set()
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

    def _fetch_hash_text_features(self, text):
        # Fit the vectorizer to the data and transform the text into vectors
        X = self.hash_vectorizer.fit_transform([text])
        dense_matrix = X.toarray()
        fixed_size_matrix = np.sum(dense_matrix, axis=0)
        return fixed_size_matrix.tolist()

    def _calculate_text_hash_scores(self, query, documents):
        if len(documents) == 0:
            return []
        
        query_vector = self._fetch_hash_text_features(query)
        documents_vectors = [self._fetch_hash_text_features(doc) for doc in documents]
        
        # Normalize the query vector
        query_vector /= np.linalg.norm(query_vector)
        
        # Normalize each document vector and calculate cosine similarity
        similarity_scores = [np.dot(query_vector, doc_vector / np.linalg.norm(doc_vector)) for doc_vector in documents_vectors]
        
        return similarity_scores

    def _calculate_fuzzy_ratios(self, query, documents):
        return [fuzz.partial_ratio(query, doc) for doc in documents]
    
    def hybrid_rerank_results(self, sentences, search_scores, query, k=5, weights=(0.80, 0.15, 0.05)):
        try:
            text_hash_scores = self._calculate_text_hash_scores(query, sentences)
            fuzzy_scores = self._calculate_fuzzy_ratios(query, sentences)

            # If text_hash_scores is empty, return the sentences and search_scores as is
            if len(text_hash_scores) == 0:
                return sentences[:k], search_scores[:k]

            # Combine scores
            search_weight, text_hash_weight, fuzzy_weight = weights
            combined_scores = search_weight * np.array(search_scores) + text_hash_weight * np.array(text_hash_scores) + fuzzy_weight * np.array(fuzzy_scores)

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
        except Exception:
            # Return trimmed results
            return sentences[:k], search_scores[:k]

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
        if self.embeddings is None:
            return [], [], []

        embedding = self._convert_ndarray_float32(embedding)
        embedding = np.array([embedding])
        faiss.normalize_L2(embedding)

        if self._embeddings_changed:
            with self.lock:
                self._build_index()
        
        with self.lock:
            filtered_indices = self._get_filtered_indices(metadata_filter, exclude_filter, or_filters)

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
                if idx == -1:
                    continue  # Skip processing for non-existent indices

                uid = self.unique_ids[idx]
                found_results.append((uid, dist, self.metadata[idx]))
        else:
            # Otherwise, we create a new index with only the filtered indices
            filtered_indices_np = np.array(list(filtered_indices), dtype=np.int32)
            filtered_embeddings = np.take(self.embeddings, filtered_indices_np, axis=0)

            filtered_index = faiss.IndexFlatIP(self.embedding_size)
            filtered_index.add(filtered_embeddings)

            distances, indices = filtered_index.search(embedding, search_k)

            for idx, dist in zip(indices[0], distances[0]):
                if idx == -1:
                    continue  # Skip processing for non-existent indices

                uid = self.unique_ids[filtered_indices_np[idx]]
                found_results.append((uid, dist, self.metadata[filtered_indices_np[idx]]))

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
