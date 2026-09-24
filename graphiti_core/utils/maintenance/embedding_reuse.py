"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import EntityEdge, create_entity_edge_embeddings
from graphiti_core.embedder import EmbedderClient
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, create_entity_node_embeddings


def search_text(text: str) -> str:
    """The text a similarity search embeds for a name or a fact."""
    return text.replace('\n', ' ')


def keep_name_vectors(nodes: list[EntityNode], vectors: list[list[float]]) -> None:
    """Leave on each node the vector its name was searched with, when it is that name's vector."""
    for node, vector in zip(nodes, vectors, strict=True):
        if search_text(node.name) == node.name:
            node.name_embedding = vector


def fact_search_vector(edge: EntityEdge) -> list[float] | None:
    """The vector a similarity search for the edge's fact would compute, when the edge has it."""
    return edge.fact_embedding if search_text(edge.fact) == edge.fact else None


async def load_name_embeddings(driver: GraphDriver, nodes: list[EntityNode]) -> None:
    """Give each node without a vector the one stored for its uuid and name, in a single query."""
    missing = [node for node in nodes if node.name_embedding is None]
    if not missing:
        return
    stored = await _stored_vectors(
        driver, _names_query(driver.provider), [node.uuid for node in missing]
    )
    for node in missing:
        node.name_embedding = stored.get((node.uuid, node.name))


async def load_fact_embeddings(driver: GraphDriver, edges: list[EntityEdge]) -> None:
    """Give each edge without a vector the one stored for its uuid and fact, in a single query."""
    missing = [edge for edge in edges if edge.fact_embedding is None]
    if not missing:
        return
    stored = await _stored_vectors(
        driver, _facts_query(driver.provider), [edge.uuid for edge in missing]
    )
    for edge in missing:
        edge.fact_embedding = stored.get((edge.uuid, edge.fact))


async def embed_missing_names(embedder: EmbedderClient, nodes: list[EntityNode]) -> None:
    """Embed in one batch the names of the nodes still without a vector."""
    await create_entity_node_embeddings(
        embedder, [node for node in nodes if node.name_embedding is None]
    )


async def embed_missing_facts(embedder: EmbedderClient, edges: list[EntityEdge]) -> None:
    """Embed in one batch the facts of the edges still without a vector."""
    await create_entity_edge_embeddings(
        embedder, [edge for edge in edges if edge.fact_embedding is None]
    )


async def reuse_fact_embeddings(clients: GraphitiClients, edges: list[EntityEdge]) -> None:
    """Load the stored vectors of the edges without one, then embed the facts still missing."""
    await load_fact_embeddings(clients.driver, edges)
    await embed_missing_facts(clients.embedder, edges)


async def _stored_vectors(
    driver: GraphDriver, query: str, uuids: list[str]
) -> dict[tuple[str, str], list[float]]:
    """The stored vectors by uuid and the text they were stored with; empty ones are left out."""
    records, _, _ = await driver.execute_query(
        query, uuids=list(dict.fromkeys(uuids)), routing_='r'
    )
    return {
        (record['uuid'], record['text']): list(record['vector'])
        for record in records
        if record['vector']
    }


def _stored_vector(provider: GraphProvider, variable: str) -> str:
    if provider == GraphProvider.NEPTUNE:
        return f'[x IN split({variable}, ",") | toFloat(x)]'
    return variable


def _names_query(provider: GraphProvider) -> str:
    vector = _stored_vector(provider, 'n.name_embedding')
    return (
        'MATCH (n:Entity) WHERE n.uuid IN $uuids '
        f'RETURN n.uuid AS uuid, n.name AS text, {vector} AS vector'
    )


def _facts_query(provider: GraphProvider) -> str:
    path = (
        '(:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(:Entity)'
        if provider == GraphProvider.KUZU
        else '(:Entity)-[e:RELATES_TO]->(:Entity)'
    )
    vector = _stored_vector(provider, 'e.fact_embedding')
    return (
        f'MATCH {path} WHERE e.uuid IN $uuids '
        f'RETURN e.uuid AS uuid, e.fact AS text, {vector} AS vector'
    )
