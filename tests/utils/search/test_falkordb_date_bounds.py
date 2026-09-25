"""Edge search date filters compare outside FalkorDB's RELATES_TO range index."""

import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, cast

import pytest

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.driver.falkordb.operations.search_ops import (
    FalkorSearchOperations,
    _build_falkor_fulltext_query,
)
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
    edge_search_filter_query_constructor,
)
from graphiti_core.search.search_utils import (
    edge_bfs_search,
    edge_fulltext_search,
    edge_similarity_search,
    get_edge_invalidation_candidates,
    get_relevant_edges,
)

EARLIER = datetime(2026, 9, 23, 9, tzinfo=timezone.utc)
REFERENCE = datetime(2026, 9, 23, 11, 30, tzinfo=timezone.utc)
LATER = datetime(2026, 9, 23, 14, tzinfo=timezone.utc)
DATE_FIELDS = ['valid_at', 'invalid_at', 'created_at', 'expired_at']
NULL_CHECKS = [ComparisonOperator.is_null, ComparisonOperator.is_not_null]
INDEXED_COMPARISON = re.compile(
    r'(?<!toString\()e\.(valid_at|invalid_at|created_at|expired_at)\s*(<|>|=)'
)
DATE_PARAMETER = re.compile(r'\$((?:valid|invalid|created|expired)_at_\w+)')
PARAMETER = re.compile(r'\$(\w+)')
OTHER_PROVIDERS = [GraphProvider.NEO4J, GraphProvider.KUZU, GraphProvider.NEPTUNE]
VECTOR = [0.1, 0.2, 0.3]
PROBE = EntityEdge(
    group_id='group',
    source_node_uuid='ana',
    target_node_uuid='luis',
    created_at=REFERENCE,
    name='LED',
    fact='Ana led the Apollo pilot',
    fact_embedding=VECTOR,
)
ENDPOINTS = (
    'n.uuid IN [edge.source_node_uuid, edge.target_node_uuid] '
    'OR m.uuid IN [edge.target_node_uuid, edge.source_node_uuid]'
)


def dated(operator: ComparisonOperator, date: datetime = REFERENCE) -> list[list[DateFilter]]:
    return [[DateFilter(date=date, comparison_operator=operator)]]


EVERY_DATE = SearchFilters(
    valid_at=dated(ComparisonOperator.less_than_equal),
    invalid_at=dated(ComparisonOperator.less_than),
    created_at=dated(ComparisonOperator.greater_than_equal),
    expired_at=dated(ComparisonOperator.not_equals),
)


class RecordingDriver:
    search_interface = None
    graph_operations_interface = None

    def __init__(self, provider: GraphProvider):
        self.provider = provider
        self.queries: list[tuple[str, dict[str, Any]]] = []

    async def execute_query(self, cypher: str, /, **params: Any):
        self.queries.append((cypher, params))
        return [], [], None

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        return _build_falkor_fulltext_query(query, group_ids, max_query_length)

    @property
    def query(self) -> str:
        [(query, _)] = self.queries
        return query

    @property
    def params(self) -> dict[str, Any]:
        [(_, params)] = self.queries
        return params


Search = Callable[[Any, SearchFilters], Awaitable[Any]]
OPERATIONS = FalkorSearchOperations()
EDGE_SEARCHES: dict[str, Search] = {
    'fulltext': lambda driver, filters: edge_fulltext_search(
        driver, 'Apollo pilot', filters, ['group']
    ),
    'similarity': lambda driver, filters: edge_similarity_search(
        driver, VECTOR, None, None, filters, ['group']
    ),
    'bfs': lambda driver, filters: edge_bfs_search(driver, ['ana'], 2, filters, ['group']),
    'relevant': lambda driver, filters: get_relevant_edges(driver, [PROBE], filters),
    'invalidation': lambda driver, filters: get_edge_invalidation_candidates(
        driver, [PROBE], filters
    ),
    'operations-fulltext': lambda driver, filters: OPERATIONS.edge_fulltext_search(
        driver, 'Apollo pilot', filters, ['group']
    ),
    'operations-similarity': lambda driver, filters: OPERATIONS.edge_similarity_search(
        driver, VECTOR, None, None, filters, ['group']
    ),
    'operations-bfs': lambda driver, filters: OPERATIONS.edge_bfs_search(
        driver, ['ana'], 2, filters, ['group']
    ),
}


async def searched_by(
    path: str, provider: GraphProvider, filters: SearchFilters
) -> RecordingDriver:
    driver = RecordingDriver(provider)
    await EDGE_SEARCHES[path](cast(GraphDriver, driver), filters)
    return driver


def flattened(query: str) -> str:
    return ' '.join(query.split())


@pytest.mark.parametrize('operator', list(ComparisonOperator), ids=lambda operator: operator.name)
@pytest.mark.parametrize('field', DATE_FIELDS)
def test_falkordb_compares_every_date_filter_outside_its_range_index(field, operator):
    filters = SearchFilters.model_validate({field: dated(operator)})

    queries, params = edge_search_filter_query_constructor(filters, GraphProvider.FALKORDB)

    if operator in NULL_CHECKS:
        assert queries == [f'((e.{field} {operator.value}))']
        assert params == {}
    else:
        [name] = params
        assert queries == [f'((toString(e.{field}) {operator.value} ${name}))']
        assert params[name] == REFERENCE


@pytest.mark.asyncio
@pytest.mark.parametrize('path', EDGE_SEARCHES)
async def test_every_falkordb_edge_search_path_keeps_its_dates_off_the_index(path):
    driver = await searched_by(path, GraphProvider.FALKORDB, EVERY_DATE)

    assert INDEXED_COMPARISON.search(driver.query) is None
    assert all(f'toString(e.{field})' in driver.query for field in DATE_FIELDS)
    assert [driver.params[name] for name in DATE_PARAMETER.findall(driver.query)] == [REFERENCE] * 4


@pytest.mark.parametrize('field', DATE_FIELDS)
@pytest.mark.parametrize('provider', OTHER_PROVIDERS)
def test_the_other_providers_keep_their_date_filters(provider, field):
    filters = SearchFilters.model_validate({field: dated(ComparisonOperator.less_than_equal)})

    queries, params = edge_search_filter_query_constructor(filters, provider)

    [name] = params
    assert queries == [f'((e.{field} <= ${name}))']
    assert params[name] == REFERENCE


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', [GraphProvider.FALKORDB, GraphProvider.NEO4J])
async def test_invalidation_candidates_filter_both_endpoints(provider):
    filters = SearchFilters(valid_at=dated(ComparisonOperator.greater_than_equal))

    driver = await searched_by('invalidation', provider, filters)

    assert f'WHERE ({ENDPOINTS}) AND (' in flattened(driver.query)


@pytest.mark.parametrize('field', DATE_FIELDS)
@pytest.mark.parametrize('provider', [GraphProvider.FALKORDB, GraphProvider.NEO4J])
def test_two_or_groups_keep_their_own_dates(provider, field):
    window = [
        DateFilter(date=EARLIER, comparison_operator=ComparisonOperator.greater_than_equal),
        DateFilter(date=REFERENCE, comparison_operator=ComparisonOperator.less_than_equal),
    ]
    after = [DateFilter(date=LATER, comparison_operator=ComparisonOperator.greater_than)]
    filters = SearchFilters.model_validate({field: [window, after]})

    [query], params = edge_search_filter_query_constructor(filters, provider)

    assert [params[name] for name in PARAMETER.findall(query)] == [EARLIER, REFERENCE, LATER]


@pytest.mark.parametrize(
    'filters',
    [
        SearchFilters(),
        SearchFilters(edge_uuids=['first', 'second']),
        SearchFilters(edge_types=['LED'], node_labels=['Person'], edge_uuids=['first']),
    ],
    ids=['empty', 'edge-uuids', 'types-labels-uuids'],
)
def test_without_date_filters_falkordb_filters_match_neo4j(filters):
    assert edge_search_filter_query_constructor(
        filters, GraphProvider.FALKORDB
    ) == edge_search_filter_query_constructor(filters, GraphProvider.NEO4J)
