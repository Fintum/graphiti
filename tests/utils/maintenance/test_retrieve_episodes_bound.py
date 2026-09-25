"""retrieve_episodes bounds the reference outside FalkorDB's valid_at range index."""

import re
from datetime import datetime, timezone
from typing import Any, cast

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.driver.falkordb.operations.episode_node_ops import FalkorEpisodeNodeOperations
from graphiti_core.driver.kuzu.operations.episode_node_ops import KuzuEpisodeNodeOperations
from graphiti_core.driver.neo4j.operations.episode_node_ops import Neo4jEpisodeNodeOperations
from graphiti_core.driver.neptune.operations.episode_node_ops import NeptuneEpisodeNodeOperations
from graphiti_core.utils.maintenance.graph_data_operations import retrieve_episodes

REFERENCE = datetime(2026, 9, 23, 11, 30, tzinfo=timezone.utc)
FALKOR_BOUND = 'toString(e.valid_at) <= $reference_time'
INDEXED_BOUND = 'e.valid_at <= $reference_time'
INDEXED_COMPARISON = re.compile(r'(?<!toString\()e\.valid_at\s*[<>]')
PATHS = [pytest.param(None, id='graph'), pytest.param('thread', id='saga')]
OTHER_PROVIDERS = [GraphProvider.NEO4J, GraphProvider.KUZU, GraphProvider.NEPTUNE]
OTHER_OPERATIONS = [
    Neo4jEpisodeNodeOperations,
    KuzuEpisodeNodeOperations,
    NeptuneEpisodeNodeOperations,
]


class RecordingDriver:
    graph_operations_interface = None

    def __init__(self, provider: GraphProvider):
        self.provider = provider
        self.queries: list[tuple[str, dict[str, Any]]] = []

    async def execute_query(self, query: str, **params: Any):
        self.queries.append((query, params))
        return [], [], None

    @property
    def query(self) -> str:
        [(query, _)] = self.queries
        return query

    @property
    def params(self) -> dict[str, Any]:
        [(_, params)] = self.queries
        return params


async def retrieved_by(provider: GraphProvider, saga: str | None) -> RecordingDriver:
    driver = RecordingDriver(provider)
    await retrieve_episodes(cast(GraphDriver, driver), REFERENCE, 4, ['group'], saga=saga)
    return driver


async def retrieved_by_operations(
    operations: Any, provider: GraphProvider, saga: str | None
) -> RecordingDriver:
    executor = RecordingDriver(provider)
    await operations.retrieve_episodes(executor, REFERENCE, 4, ['group'], None, saga)
    return executor


def where_lines(query: str) -> list[str]:
    return [line.strip() for line in query.splitlines() if line.strip().startswith('WHERE')]


@pytest.mark.asyncio
@pytest.mark.parametrize('saga', PATHS)
async def test_falkordb_bounds_the_reference_without_its_valid_at_index(saga):
    driver = await retrieved_by(GraphProvider.FALKORDB, saga)

    assert where_lines(driver.query) == [f'WHERE {FALKOR_BOUND}']
    assert INDEXED_COMPARISON.search(driver.query) is None
    assert driver.params['reference_time'] == REFERENCE


@pytest.mark.asyncio
@pytest.mark.parametrize('saga', PATHS)
async def test_falkordb_keeps_the_most_recent_first_and_the_last_n_cap(saga):
    driver = await retrieved_by(GraphProvider.FALKORDB, saga)

    assert 'ORDER BY e.valid_at DESC' in driver.query and 'LIMIT $num_episodes' in driver.query
    assert driver.params['num_episodes'] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize('saga', PATHS)
@pytest.mark.parametrize('provider', OTHER_PROVIDERS)
async def test_the_other_providers_keep_their_indexed_bound(provider, saga):
    driver = await retrieved_by(provider, saga)

    assert where_lines(driver.query) == [f'WHERE {INDEXED_BOUND}']
    assert 'toString' not in driver.query


@pytest.mark.asyncio
@pytest.mark.parametrize('saga', PATHS)
async def test_the_falkordb_episode_operations_bound_the_reference_without_the_index(saga):
    executor = await retrieved_by_operations(
        FalkorEpisodeNodeOperations(), GraphProvider.FALKORDB, saga
    )

    assert where_lines(executor.query) == [f'WHERE {FALKOR_BOUND}']
    assert INDEXED_COMPARISON.search(executor.query) is None
    assert 'ORDER BY e.valid_at DESC' in executor.query and executor.params['num_episodes'] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize('saga', PATHS)
@pytest.mark.parametrize('operations', OTHER_OPERATIONS)
async def test_the_other_episode_operations_keep_their_indexed_bound(operations, saga):
    executor = await retrieved_by_operations(operations(), GraphProvider.NEO4J, saga)

    assert where_lines(executor.query) == [f'WHERE {INDEXED_BOUND}']
    assert 'toString' not in executor.query
