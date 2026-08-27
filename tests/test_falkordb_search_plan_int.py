"""FalkorDB execution plans for the search Cypher: the regression that took production down was a label scan per hit."""

import os

import pytest
from falkordb import FalkorDB

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb_driver import index_already_exists_error
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices, get_vector_indices
from graphiti_core.models.edges.edge_db_queries import (
    get_falkordb_edge_fulltext_query,
    get_falkordb_edge_vector_query,
)
from graphiti_core.models.nodes.node_db_queries import get_falkordb_node_vector_query

pytestmark = pytest.mark.skipif(
    os.getenv('DISABLE_FALKORDB') is not None, reason='FalkorDB disabled for this run'
)

GRAPH_NAME = 'test_search_plan'
GROUP = 'g'
DIMENSION = 3
FULLTEXT_PARAMS = {'query': f'(@group_id:"{GROUP}") (Olatz)', 'limit': 20, 'group_ids': [GROUP]}
VECTOR_PARAMS = {'search_vector': [1.0, 0.05, 0.0], 'limit': 10, 'min_score': 0.6, 'group_ids': [GROUP]}


def _client() -> FalkorDB:
    return FalkorDB(
        host=os.getenv('FALKORDB_HOST', 'localhost'),
        port=int(os.getenv('FALKORDB_PORT', '6379')),
        socket_connect_timeout=2,
    )


def _build_indices(graph) -> None:
    statements = (
        get_range_indices(GraphProvider.FALKORDB)
        + get_fulltext_indices(GraphProvider.FALKORDB)
        + get_vector_indices(GraphProvider.FALKORDB, DIMENSION)
    )
    for statement in statements:
        try:
            graph.query(statement)
        except Exception as error:
            if not index_already_exists_error(error):
                raise


@pytest.fixture
def graph():
    try:
        client = _client()
        client.list_graphs()
    except Exception:
        pytest.skip('no FalkorDB listening')
    try:
        client.select_graph(GRAPH_NAME).delete()
    except Exception:
        pass
    graph = client.select_graph(GRAPH_NAME)
    graph.query(
        "CREATE (a:Entity {uuid:'n1', group_id:$g, name:'Olatz', name_embedding: vecf32([1.0, 0.0, 0.0])})"
        "-[:RELATES_TO {uuid:'e1', group_id:$g, name:'ATTENDED', fact:'Olatz attended the plan review',"
        " fact_embedding: vecf32([1.0, 0.0, 0.0])}]->"
        "(b:Entity {uuid:'n2', group_id:$g, name:'Plan review', name_embedding: vecf32([0.0, 1.0, 0.0])})",
        {'g': GROUP},
    )
    _build_indices(graph)
    yield graph
    graph.delete()


def test_edge_fulltext_plan_has_no_label_scan(graph):
    plan = str(graph.explain(get_falkordb_edge_fulltext_query(['e.group_id IN $group_ids']), FULLTEXT_PARAMS))

    assert 'Node By Label Scan' not in plan
    assert 'ProcedureCall' in plan


def test_edge_fulltext_query_returns_the_edge_with_its_endpoints(graph):
    result = graph.query(get_falkordb_edge_fulltext_query(['e.group_id IN $group_ids']), FULLTEXT_PARAMS)

    rows = result.result_set
    assert len(rows) == 1
    uuid, source_uuid, target_uuid = rows[0][0], rows[0][1], rows[0][2]
    assert (uuid, source_uuid, target_uuid) == ('e1', 'n1', 'n2')


def test_relationship_index_ignores_spanish_stopwords_server_side(graph):
    graph.query("MATCH ()-[e:RELATES_TO {uuid:'e1'}]->() SET e.fact = 'el plan de Olatz'")

    stopword_hits = graph.query(
        "CALL db.idx.fulltext.queryRelationships('RELATES_TO', $q) YIELD relationship RETURN count(relationship)",
        {'q': 'de'},
    ).result_set[0][0]
    real_hits = graph.query(
        "CALL db.idx.fulltext.queryRelationships('RELATES_TO', $q) YIELD relationship RETURN count(relationship)",
        {'q': 'Olatz'},
    ).result_set[0][0]

    assert (stopword_hits, real_hits) == (0, 1)


def test_edge_vector_plan_uses_the_procedure_not_a_scan(graph):
    plan = str(graph.explain(get_falkordb_edge_vector_query(['e.group_id IN $group_ids'], k=20), VECTOR_PARAMS))

    assert 'ProcedureCall' in plan
    assert 'Node By Label Scan' not in plan and 'Conditional Traverse' not in plan


def test_edge_vector_query_returns_the_similar_edge_with_a_similarity_score(graph):
    rows = graph.query(get_falkordb_edge_vector_query(['e.group_id IN $group_ids'], k=20), VECTOR_PARAMS).result_set

    assert [row[0] for row in rows] == ['e1']


def test_node_vector_query_ranks_the_closest_node_first(graph):
    rows = graph.query(get_falkordb_node_vector_query(['n.group_id IN $group_ids'], k=20), VECTOR_PARAMS).result_set

    assert rows[0][0] == 'n1'
