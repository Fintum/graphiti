"""A rewrite keeps what other writers put on an entity: authority, provenance and invalidators."""

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.edges.edge_db_queries import (
    get_entity_edge_save_bulk_query,
    get_entity_edge_save_query,
)
from graphiti_core.models.nodes.node_db_queries import get_entity_node_save_query
from graphiti_core.utils.protected_persistence import preserving_update


def union(variable: str, key: str) -> str:
    prior, current = f'authority.{key}', f'{variable}.{key}'
    return (
        f'{current} = CASE WHEN {prior} IS NULL THEN {current} WHEN {current} IS NULL THEN {prior} '
        f'ELSE {prior} + [item IN {current} WHERE NOT item IN {prior}] END'
    )


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
