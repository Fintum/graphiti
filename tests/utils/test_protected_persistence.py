"""A rewrite keeps what other writers put on an entity: authority, provenance and invalidators."""

import re

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graph_queries import get_range_indices
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_bulk_query,
    get_entity_edge_save_query,
)
from graphiti_core.models.nodes.node_db_queries import get_entity_node_save_query
from graphiti_core.utils.protected_persistence import (
    STORED_EDGE_TYPES,
    STORED_NODE_LABELS,
    preserving_update,
    unioned,
)


def union(variable: str, key: str) -> str:
    return f'{variable}.{key} = {unioned(key, f"{variable}.{key}")}'


def test_rewriting_an_edge_keeps_every_episode_that_invalidated_it():
    assert union('e', 'invalidated_by') in get_entity_edge_save_query(GraphProvider.FALKORDB)
    assert union('r', 'invalidated_by') in get_entity_edge_save_bulk_query(GraphProvider.FALKORDB)


def test_invalidators_are_unioned_like_the_episodes_of_the_edge():
    rewrite = preserving_update('e', '$edge_data')

    assert union('e', 'episodes') in rewrite and union('e', 'invalidated_by') in rewrite


def test_a_node_rewrite_carries_no_invalidators():
    assert 'invalidated_by' not in get_entity_node_save_query(
        GraphProvider.FALKORDB, labels='Entity'
    )
    assert 'invalidated_by' not in preserving_update('n', '$entity_data')


def test_identity_reads_cover_exactly_the_labels_and_edge_types_with_a_uuid_index():
    indexed = [query for query in get_range_indices(GraphProvider.FALKORDB) if '.uuid' in query]

    labels = {match for query in indexed for match in re.findall(r'\(n:(\w+)\)', query)}
    edge_types = {match for query in indexed for match in re.findall(r'\[e:(\w+)\]', query)}

    assert sorted(STORED_NODE_LABELS) == sorted(labels)
    assert sorted(STORED_EDGE_TYPES) == sorted(edge_types)


def test_relates_to_episodes_is_indexed_on_its_own_statement():
    assert 'CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.episodes)' in get_range_indices(
        GraphProvider.FALKORDB
    )
