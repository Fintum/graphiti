"""Shape of the FalkorDB search Cypher: linear plans only (no re-MATCH of a relationship the procedure already yielded)."""

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb import STOPWORDS, STOPWORDS_ES
from graphiti_core.driver.falkordb.operations.search_ops import _build_falkor_fulltext_query
from graphiti_core.graph_queries import get_fulltext_indices, get_vector_indices
from graphiti_core.models.edges.edge_db_queries import (
    get_falkordb_edge_fulltext_query,
    get_falkordb_edge_vector_query,
)
from graphiti_core.models.nodes.node_db_queries import get_falkordb_node_vector_query


def test_edge_fulltext_query_uses_yielded_relationship_endpoints():
    query = get_falkordb_edge_fulltext_query(['e.group_id IN $group_ids'])

    assert '{uuid: rel.uuid}' not in query
    assert 'YIELD relationship AS e, score' in query
    assert 'startNode(e) AS n' in query and 'endNode(e) AS m' in query


def test_edge_fulltext_query_limits_before_return():
    query = get_falkordb_edge_fulltext_query([])

    assert query.index('ORDER BY score DESC') < query.index('RETURN')
    assert query.index('LIMIT $limit') < query.index('RETURN')


def test_edge_fulltext_query_applies_filters_after_endpoints_exist():
    query = get_falkordb_edge_fulltext_query(['e.group_id IN $group_ids', 'n.uuid = $source_uuid'])

    where_at = query.index('WHERE e.group_id IN $group_ids AND n.uuid = $source_uuid')
    assert query.index('startNode(e) AS n') < where_at


def test_spanish_function_words_are_stopwords():
    for word in ('de', 'la', 'el', 'y', 'en', 'fue', 'del', 'los', 'las', 'con', 'para', 'por', 'que', 'se'):
        assert word in STOPWORDS_ES
        assert word in STOPWORDS


def test_fulltext_query_drops_spanish_stopwords():
    query = _build_falkor_fulltext_query('El plan de alimentación y entrenamiento de Olatz fue revisado', ['org_x'])

    assert query == '(@group_id:"org\\_x") (plan | alimentación | entrenamiento | Olatz | revisado)'


def test_relationship_fulltext_index_declares_stopwords():
    relationship_index = next(
        statement for statement in get_fulltext_indices(GraphProvider.FALKORDB) if 'RELATES_TO' in statement
    )

    assert 'OPTIONS' in relationship_index and 'stopwords' in relationship_index
    assert "'de'" in relationship_index and "'the'" in relationship_index


def test_vector_indices_carry_the_embedder_dimension():
    statements = get_vector_indices(GraphProvider.FALKORDB, dimension=1024)

    assert len(statements) == 2
    assert all('dimension:1024' in statement for statement in statements)
    assert any('RELATES_TO' in statement and 'fact_embedding' in statement for statement in statements)
    assert any('(n:Entity)' in statement and 'name_embedding' in statement for statement in statements)
    assert get_vector_indices(GraphProvider.NEO4J, dimension=1024) == []


def test_edge_vector_query_uses_the_index_and_keeps_similarity_semantics():
    query = get_falkordb_edge_vector_query(['e.group_id IN $group_ids'], k=80)

    assert "db.idx.vector.queryRelationships('RELATES_TO', 'fact_embedding', 80, vecf32($search_vector))" in query
    assert '(2 - distance) / 2 AS score' in query
    assert 'WHERE score > $min_score AND e.group_id IN $group_ids' in query
    assert 'vec.cosineDistance' not in query
    assert 'MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)' not in query


def test_node_vector_query_uses_the_index():
    query = get_falkordb_node_vector_query(['n.group_id IN $group_ids'], k=40)

    assert "db.idx.vector.queryNodes('Entity', 'name_embedding', 40, vecf32($search_vector))" in query
    assert 'WHERE score > $min_score AND n.group_id IN $group_ids' in query
