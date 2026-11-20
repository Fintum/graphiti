"""FalkorDB keeps a curator's discard mark when a deduplicated restatement rewrites the edge or the node."""

import contextlib
import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from falkordb import FalkorDB

from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

pytestmark = pytest.mark.skipif(
    os.getenv('DISABLE_FALKORDB') is not None, reason='FalkorDB disabled for this run'
)

GRAPH_NAME = 'test_discard_mark'
GROUP = 'group'
CREATED = datetime(2025, 1, 1, tzinfo=timezone.utc)
EMBEDDING = [0.1, 0.2, 0.3]
DISCARD_MARK = {
    'discarded_at': '2026-11-19T10:00:00+00:00',
    'discarded_by': 'suppression-1',
    'kept_episodes': ['ep-board'],
}
EDGE = "MATCH ()-[x:RELATES_TO {uuid: 'costs'}]->()"
NODE = "MATCH (x:Entity {uuid: 'pricing'})"

Write = Callable[[FalkorDriver, list[EntityNode], list[EntityEdge]], Awaitable[None]]


async def save_one_by_one(driver: FalkorDriver, nodes: list[EntityNode], edges: list[EntityEdge]):
    for node in nodes:
        await node.save(driver)
    for edge in edges:
        await edge.save(driver)


async def save_in_bulk(driver: FalkorDriver, nodes: list[EntityNode], edges: list[EntityEdge]):
    await add_nodes_and_edges_bulk(driver, [], [], nodes, edges, MagicMock())


WRITES = pytest.mark.parametrize(
    'write', [save_one_by_one, save_in_bulk], ids=['one by one', 'in bulk']
)
RESTATEMENTS = pytest.mark.parametrize(
    'restated',
    [{}, {'kept_episodes': ['ep-stale']}],
    ids=['silent on kept episodes', 'with other kept episodes'],
)


def entity(uuid: str, episodes: list[str], **attributes: object) -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=uuid.title(),
        group_id=GROUP,
        labels=['Entity', 'Topic'],
        created_at=CREATED,
        name_embedding=EMBEDDING,
        attributes={'provenance': episodes, **attributes},
    )


def price(episodes: list[str], **attributes: object) -> EntityEdge:
    return EntityEdge(
        uuid='costs',
        group_id=GROUP,
        source_node_uuid='plan',
        target_node_uuid='pricing',
        name='COSTS',
        fact='The plan costs 99 €',
        episodes=episodes,
        created_at=CREATED,
        valid_at=CREATED,
        fact_embedding=EMBEDDING,
        attributes={'provenance': episodes, **attributes},
    )


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
    yield graph
    with contextlib.suppress(Exception):
        graph.delete()


@pytest.fixture
def driver(graph):
    return FalkorDriver(
        host=os.getenv('FALKORDB_HOST', 'localhost'),
        port=int(os.getenv('FALKORDB_PORT', '6379')),
        database=GRAPH_NAME,
    )


def discard(graph, match: str) -> None:
    marks = ', '.join(f'x.{key} = ${key}' for key in DISCARD_MARK)
    graph.query(f'{match} SET {marks}', DISCARD_MARK)


def stored(graph, match: str) -> list:
    marks = ', '.join(f'x.{key}' for key in DISCARD_MARK)
    return graph.query(f'{match} RETURN {marks}, x.provenance').result_set


@WRITES
@RESTATEMENTS
async def test_a_deduplicated_restatement_keeps_the_discard_mark_of_the_edge(
    graph, driver, write: Write, restated: dict
):
    await write(driver, [entity('plan', ['ep1']), entity('pricing', ['ep1'])], [price(['ep1'])])
    discard(graph, EDGE)

    await write(driver, [], [price(['ep1', 'ep2'], **restated)])

    assert stored(graph, EDGE) == [[*DISCARD_MARK.values(), ['ep1', 'ep2']]]


@WRITES
@RESTATEMENTS
async def test_a_deduplicated_restatement_keeps_the_discard_mark_of_the_node(
    graph, driver, write: Write, restated: dict
):
    await write(driver, [entity('pricing', ['ep1'])], [])
    discard(graph, NODE)

    await write(driver, [entity('pricing', ['ep1', 'ep2'], **restated)], [])

    assert stored(graph, NODE) == [[*DISCARD_MARK.values(), ['ep1', 'ep2']]]
