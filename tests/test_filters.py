import pytest

DOCUMENTS = [
    (1, [1.0, 0.0], {"type": "fruit", "name": "apple", "price": 5, "tags": ["red", "sweet"]}),
    (2, [0.9, 0.1], {"type": "fruit", "name": "banana", "price": 10, "tags": ["yellow", "sweet"]}),
    (3, [0.8, 0.2], {"type": "vegetable", "name": "carrot", "price": 15, "tags": ["orange"]}),
    (4, [0.7, 0.3], {"type": "vegetable", "name": "potato", "price": 20}),
    (5, [0.6, 0.4], {"type": "grain", "price": 25.5, "organic": True}),
]

QUERY = [1.0, 0.0]


@pytest.fixture
def catalog(db):
    db.store_embeddings_batch(
        [uid for uid, _, _ in DOCUMENTS],
        [vector for _, vector, _ in DOCUMENTS],
        [metadata for _, _, metadata in DOCUMENTS],
    )
    return db


def found(db, **kwargs):
    ids, _, _ = db.find_most_similar(QUERY, k=10, **kwargs)
    return sorted(ids)


def test_no_filter_returns_everything(catalog):
    assert found(catalog) == [1, 2, 3, 4, 5]


def test_equality(catalog):
    assert found(catalog, metadata_filter={"type": "fruit"}) == [1, 2]
    assert found(catalog, metadata_filter={"organic": True}) == [5]


def test_equality_is_type_aware(catalog):
    assert found(catalog, metadata_filter={"price": 5}) == [1]
    assert found(catalog, metadata_filter={"price": "5"}) == []


def test_multiple_keys_are_combined_with_and(catalog):
    assert found(catalog, metadata_filter={"type": "fruit", "price": 10}) == [2]
    assert found(catalog, metadata_filter={"type": "fruit", "price": 15}) == []


def test_list_of_filters_is_combined_with_and(catalog):
    assert found(catalog, metadata_filter=[{"type": "fruit"}, {"name": "apple"}]) == [1]


def test_unknown_key_matches_nothing(catalog):
    assert found(catalog, metadata_filter={"missing": "value"}) == []


def test_comparison_operators(catalog):
    assert found(catalog, metadata_filter={"price": {"$gt": 15}}) == [4, 5]
    assert found(catalog, metadata_filter={"price": {"$gte": 15}}) == [3, 4, 5]
    assert found(catalog, metadata_filter={"price": {"$lt": 15}}) == [1, 2]
    assert found(catalog, metadata_filter={"price": {"$lte": 15}}) == [1, 2, 3]


def test_comparison_on_strings(catalog):
    assert found(catalog, metadata_filter={"name": {"$gt": "c"}}) == [3, 4]
    assert found(catalog, metadata_filter={"name": {"$lt": "c"}}) == [1, 2]


def test_equality_and_inequality_operators(catalog):
    assert found(catalog, metadata_filter={"type": {"$eq": "grain"}}) == [5]
    assert found(catalog, metadata_filter={"type": {"$ne": "fruit"}}) == [3, 4, 5]


def test_in_and_nin(catalog):
    assert found(catalog, metadata_filter={"type": {"$in": ["fruit", "grain"]}}) == [1, 2, 5]
    assert found(catalog, metadata_filter={"type": {"$nin": ["fruit", "grain"]}}) == [3, 4]
    assert found(catalog, metadata_filter={"type": {"$in": "grain"}}) == [5]


def test_contains_matches_list_members_and_substrings(catalog):
    assert found(catalog, metadata_filter={"tags": {"$contains": "sweet"}}) == [1, 2]
    assert found(catalog, metadata_filter={"tags": {"$contains": "red"}}) == [1]
    assert found(catalog, metadata_filter={"name": {"$contains": "arro"}}) == [3]


def test_exists(catalog):
    assert found(catalog, metadata_filter={"tags": {"$exists": True}}) == [1, 2, 3]
    assert found(catalog, metadata_filter={"tags": {"$exists": False}}) == [4, 5]


def test_invalid_operator(catalog):
    with pytest.raises(ValueError, match="Invalid operator"):
        catalog.find_most_similar(QUERY, {"price": {"$between": [1, 2]}})


def test_or_filters(catalog):
    assert found(catalog, or_filters=[{"type": "grain"}, {"name": "apple"}]) == [1, 5]
    assert found(catalog, or_filters={"type": "grain"}) == [5]


def test_or_filters_are_intersected_with_and_filters(catalog):
    assert found(
        catalog,
        metadata_filter={"type": "vegetable"},
        or_filters=[{"price": {"$gt": 18}}, {"name": "carrot"}],
    ) == [3, 4]


def test_exclude_filter(catalog):
    assert found(catalog, exclude_filter={"type": "fruit"}) == [3, 4, 5]
    assert found(catalog, exclude_filter=[{"type": "fruit"}, {"name": "carrot"}]) == [4, 5]


def test_exclude_filter_with_operators(catalog):
    assert found(catalog, exclude_filter={"price": {"$gte": 15}}) == [1, 2]


def test_all_filter_kinds_together(catalog):
    assert found(
        catalog,
        metadata_filter={"price": {"$lte": 20}},
        or_filters=[{"type": "fruit"}, {"type": "vegetable"}],
        exclude_filter={"name": "apple"},
    ) == [2, 3, 4]


def test_empty_filters_are_ignored(catalog):
    assert found(catalog, metadata_filter={}, exclude_filter=[], or_filters=[{}]) == [1, 2, 3, 4, 5]


def test_filters_follow_deletions(catalog):
    catalog.delete_embedding(1)
    assert found(catalog, metadata_filter={"type": "fruit"}) == [2]
    assert found(catalog, metadata_filter={"type": {"$ne": "fruit"}}) == [3, 4, 5]


def test_filtered_search_respects_k(catalog):
    ids, scores, _ = catalog.find_most_similar(QUERY, {"type": {"$ne": "grain"}}, k=2)
    assert ids == [1, 2]
    assert scores[0] > scores[1]
