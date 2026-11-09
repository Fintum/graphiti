"""FalkorDB plans of the identity reads of a bulk write: each one by a uuid index, none a scan of the whole graph."""

import asyncio
import contextlib
import os
import time
from typing import Any

import pytest
from falkordb import FalkorDB

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.graph_queries import get_range_indices
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.protected_persistence import validate_entity_writes

pytestmark = pytest.mark.skipif(
    os.getenv('DISABLE_FALKORDB') is not None, reason='FalkorDB disabled for this run'
)

GRAPH_NAME = 'test_identity_reads_plan'
SCANNED_NODES = (
    'MATCH (n) WHERE n.uuid IN $uuids RETURN properties(n) AS properties, labels(n) AS labels'
)
SCANNED_EDGES = (
    'MATCH (s)-[e]->(t) WHERE e.uuid IN $uuids '
    'RETURN properties(e) AS properties, s.uuid AS source, t.uuid AS target'
)
CITING = 'MATCH ()-[e:RELATES_TO]->() WHERE $episode IN e.episodes RETURN e.uuid ORDER BY e.uuid'
MIXED_GRAPH = """
CREATE (a:Entity:Person {uuid:'a', name:'Ada'}), (b:Entity {uuid:'b', name:'Apollo'}),
       (episode:Episodic {uuid:'ep1'}), (next:Episodic {uuid:'ep2'}),
       (team:Community {uuid:'c1'}), (saga:Saga {uuid:'s1'}),
       (:Entity {uuid:'twin'}), (:Episodic {uuid:'twin'}), (:Entity:Episodic {uuid:'both'}),
       (a)-[:RELATES_TO {uuid:'r1', episodes:['ep1']}]->(b),
       (b)-[:RELATES_TO {uuid:'r2', episodes:['ep0', 'ep1']}]->(a),
       (a)-[:RELATES_TO {uuid:'r3', episodes:['ep0'], invalidated_by:['ep1']}]->(b),
       (episode)-[:MENTIONS {uuid:'m1'}]->(a),
       (team)-[:HAS_MEMBER {uuid:'h1'}]->(a),
       (saga)-[:HAS_EPISODE {uuid:'he1'}]->(episode),
       (episode)-[:NEXT_EPISODE {uuid:'n1'}]->(next),
       (a)-[:RELATES_TO {uuid:'shared'}]->(b), (episode)-[:MENTIONS {uuid:'shared'}]->(b)
"""
NODE_UUIDS = ['a', 'b', 'ep1', 'ep2', 'c1', 's1', 'twin', 'both', 'nowhere']
EDGE_UUIDS = ['r1', 'r2', 'r3', 'm1', 'h1', 'he1', 'n1', 'shared', 'nowhere']


class RecordingDriver:
    provider = GraphProvider.FALKORDB

    def __init__(self):
        self.sent: list[str] = []

    async def execute_query(self, query: str, **params: Any):
        self.sent.append(query)
        return [], None, None


def _identity_reads() -> tuple[str, str]:
    driver = RecordingDriver()
    node = EntityNode(uuid='a', name='Ada', group_id='g', labels=['Entity'])
    edge = EntityEdge(
        uuid='r1',
        group_id='g',
        source_node_uuid='a',
        target_node_uuid='b',
        name='LEADS',
        fact='f',
        created_at=utc_now(),
    )
    asyncio.run(validate_entity_writes(driver, [node], [edge]))  # type: ignore[arg-type]
    node_read = next(query for query in driver.sent if 'labels(n) AS labels' in query)
    edge_read = next(query for query in driver.sent if 's.uuid AS source' in query)
    return node_read, edge_read


STORED_NODES, STORED_EDGES = _identity_reads()


def _operational(graph) -> None:
    for _ in range(50):
        statuses = graph.query('CALL db.indexes() YIELD status RETURN status').result_set
        if all(row[0] == 'OPERATIONAL' for row in statuses):
            return
        time.sleep(0.1)
    raise AssertionError(f'indices still building: {statuses}')


@pytest.fixture
def graph():
    try:
        client = FalkorDB(
            host=os.getenv('FALKORDB_HOST', 'localhost'),
            port=int(os.getenv('FALKORDB_PORT', '6379')),
            socket_connect_timeout=2,
        )
        client.list_graphs()
    except Exception:
        pytest.skip('no FalkorDB listening')
    with contextlib.suppress(Exception):
        client.select_graph(GRAPH_NAME).delete()
    graph = client.select_graph(GRAPH_NAME)
    for statement in get_range_indices(GraphProvider.FALKORDB):
        graph.query(statement)
    graph.query(MIXED_GRAPH)
    _operational(graph)
    yield graph
    graph.delete()


def _edge_rows(graph, query: str) -> list[tuple[str, str, str, str]]:
    rows = graph.query(query, {'uuids': EDGE_UUIDS}).result_set
    return sorted(
        (props['uuid'], repr(sorted(props.items())), source, target)
        for props, source, target in rows
    )


def test_the_node_read_goes_through_the_uuid_index_of_each_label(graph):
    plan = str(graph.explain(STORED_NODES, {'uuids': NODE_UUIDS}))

    assert 'Node By Index Scan' in plan, plan
    assert 'All Node Scan' not in plan and 'Label Scan' not in plan, plan


def test_the_edge_read_goes_through_the_uuid_index_of_each_edge_type(graph):
    plan = str(graph.explain(STORED_EDGES, {'uuids': EDGE_UUIDS}))

    assert 'Edge By Index Scan' in plan, plan
    assert 'All Node Scan' not in plan, plan


def test_the_node_read_returns_the_rows_of_the_full_scan_in_the_same_order(graph):
    indexed = graph.query(STORED_NODES, {'uuids': NODE_UUIDS}).result_set

    assert indexed == graph.query(SCANNED_NODES, {'uuids': NODE_UUIDS}).result_set
    assert [row[0]['uuid'] for row in indexed].count('twin') == 2
    assert [row[0]['uuid'] for row in indexed].count('both') == 1


def test_the_edge_read_returns_the_rows_of_the_full_scan_across_edge_types(graph):
    indexed = _edge_rows(graph, STORED_EDGES)

    assert indexed == _edge_rows(graph, SCANNED_EDGES)
    assert [row[0] for row in indexed].count('shared') == 2


def test_the_edges_citing_an_episode_are_found_through_the_episodes_index(graph):
    plan = str(graph.explain(CITING, {'episode': 'ep1'}))

    assert 'Edge By Index Scan' in plan and 'All Node Scan' not in plan, plan
    assert graph.query(CITING, {'episode': 'ep1'}).result_set == [['r1'], ['r2']]
