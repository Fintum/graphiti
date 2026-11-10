"""FalkorDB keeps what other writers put on a list exactly as before, and a hub's list costs one pass, not a square."""

import contextlib
import os
import time
from typing import Any

import pytest
from falkordb import FalkorDB

from graphiti_core.models.nodes.node_db_queries import _falkordb_entity_node_bulk_query
from graphiti_core.utils.protected_persistence import unioned

pytestmark = pytest.mark.skipif(
    os.getenv('DISABLE_FALKORDB') is not None, reason='FalkorDB disabled for this run'
)

GRAPH_NAME = 'test_protected_union'
SQUARE = 'CASE WHEN prior IS NULL THEN current WHEN current IS NULL THEN prior ELSE prior + [item IN current WHERE NOT item IN prior] END'
CASES = {
    'the stored list then new items': (['a', 'b', 'c'], ['a', 'b', 'c', 'd', 'e']),
    'nothing new': (['a', 'b'], ['a', 'b']),
    'a new item already stored': (['a', 'b'], ['a', 'b', 'a']),
    'a new item twice': (['a'], ['a', 'x', 'x']),
    'another order': (['a', 'b', 'c'], ['c', 'a', 'd']),
    'a shorter payload': (['a', 'b', 'c'], ['a', 'd']),
    'a payload that drops the head': (['a', 'b', 'c'], ['b', 'c', 'd']),
    'a stored list with a repetition': (['a', 'a', 'b'], ['a', 'b', 'c']),
    'nothing stored': ([], ['a', 'b']),
    'an empty payload': (['a', 'b'], []),
    'both empty': ([], []),
    'no stored list': (None, ['a']),
    'no payload list': (['a'], None),
    'neither': (None, None),
}


@pytest.fixture
def graph():
    client = FalkorDB(host=os.getenv('FALKORDB_HOST', 'localhost'), port=int(os.getenv('FALKORDB_PORT', '6379')))
    with contextlib.suppress(Exception):
        client.select_graph(GRAPH_NAME).delete()
    graph = client.select_graph(GRAPH_NAME)
    yield graph
    with contextlib.suppress(Exception):
        graph.delete()


def merged(graph, expression: str, prior: Any, current: Any) -> Any:
    query = (f'WITH {{provenance: $prior}} AS authority, $current AS current '
             f'WITH authority, current, coalesce(authority.provenance, []) AS prior_provenance, authority.provenance AS prior '
             f'RETURN {expression}')
    return graph.query(query, {'prior': prior, 'current': current}).result_set[0][0]


@pytest.mark.parametrize('case', CASES)
def test_every_list_merges_exactly_as_the_full_check_did(graph, case):
    prior, current = CASES[case]
    assert merged(graph, unioned('provenance', 'current'), prior, current) == merged(graph, SQUARE, prior, current)


@pytest.mark.parametrize('stored', [20_000, 25_000])
def test_a_hub_keeps_its_whole_provenance_and_gains_the_new_episode_in_one_pass(graph, stored):
    provenance = [f'episode-{index:06d}' for index in range(stored)]
    graph.query('CREATE (:Entity {uuid: $uuid, name: $name, provenance: $provenance})',
                {'uuid': 'hub', 'name': 'Hub', 'provenance': provenance})
    node = {'uuid': 'hub', 'name': 'Hub', 'group_id': GRAPH_NAME, 'summary': '', 'labels': ['Entity', 'Person'],
            'provenance': [*provenance, 'episode-new'], 'name_embedding': [0.1, 0.2, 0.3]}
    started = time.monotonic()
    graph.query(_falkordb_entity_node_bulk_query(('Entity', 'Person')), {'nodes': [node]})
    elapsed = time.monotonic() - started
    saved = graph.query("MATCH (n:Entity {uuid: 'hub'}) RETURN n.provenance").result_set[0][0]
    assert saved == [*provenance, 'episode-new']
    assert elapsed < 1.0, f'{stored} stored episodes took {elapsed:.2f}s'


@pytest.mark.parametrize('stored', [20_000, 25_000])
def test_an_edge_cited_by_thousands_keeps_them_all_in_one_pass(graph, stored):
    episodes = [f'episode-{index:06d}' for index in range(stored)]
    started = time.monotonic()
    merged_episodes = merged(graph, unioned('provenance', 'current'), episodes, [*episodes, 'episode-new'])
    elapsed = time.monotonic() - started
    assert merged_episodes == [*episodes, 'episode-new']
    assert elapsed < 1.0, f'{stored} episodes took {elapsed:.2f}s'
