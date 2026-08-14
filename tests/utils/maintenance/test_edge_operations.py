from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.edge_operations import (
    extract_edges,
    resolve_extracted_edge,
    resolve_extracted_edges,
    search_related_edges,
)


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.generate_response = AsyncMock()
    return client


@pytest.fixture
def mock_extracted_edge():
    return EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='test_edge',
        group_id='group_1',
        fact='Test fact',
        episodes=['episode_1'],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )


@pytest.fixture
def mock_related_edges():
    return [
        EntityEdge(
            source_node_uuid='source_uuid_2',
            target_node_uuid='target_uuid_2',
            name='related_edge',
            group_id='group_1',
            fact='Related fact',
            episodes=['episode_2'],
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
            valid_at=datetime.now(timezone.utc) - timedelta(days=1),
            invalid_at=None,
        )
    ]


@pytest.fixture
def mock_existing_edges():
    return [
        EntityEdge(
            source_node_uuid='source_uuid_3',
            target_node_uuid='target_uuid_3',
            name='existing_edge',
            group_id='group_1',
            fact='Existing fact',
            episodes=['episode_3'],
            created_at=datetime.now(timezone.utc) - timedelta(days=2),
            valid_at=datetime.now(timezone.utc) - timedelta(days=2),
            invalid_at=None,
        )
    ]


@pytest.fixture
def mock_current_episode():
    return EpisodicNode(
        uuid='episode_1',
        content='Current episode content',
        valid_at=datetime.now(timezone.utc),
        name='Current Episode',
        group_id='group_1',
        source='message',
        source_description='Test source description',
    )


@pytest.fixture
def mock_previous_episodes():
    return [
        EpisodicNode(
            uuid='episode_2',
            content='Previous episode content',
            valid_at=datetime.now(timezone.utc) - timedelta(days=1),
            name='Previous Episode',
            group_id='group_1',
            source='message',
            source_description='Test source description',
        )
    ]


# Run the tests
if __name__ == '__main__':
    pytest.main([__file__])


@pytest.mark.asyncio
async def test_resolve_extracted_edge_exact_fact_short_circuit(
    mock_llm_client,
    mock_existing_edges,
    mock_current_episode,
):
    extracted = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='test_edge',
        group_id='group_1',
        fact='Related fact',
        episodes=['episode_1'],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    related_edges = [
        EntityEdge(
            source_node_uuid='source_uuid',
            target_node_uuid='target_uuid',
            name='related_edge',
            group_id='group_1',
            fact=' related FACT  ',
            episodes=['episode_2'],
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
            valid_at=None,
            invalid_at=None,
        )
    ]

    resolved_edge, duplicate_edges, invalidated = await resolve_extracted_edge(
        mock_llm_client,
        extracted,
        related_edges,
        mock_existing_edges,
        mock_current_episode,
        edge_type_candidates=None,
    )

    assert resolved_edge is related_edges[0]
    assert resolved_edge.episodes.count(mock_current_episode.uuid) == 1
    assert duplicate_edges == []
    assert invalidated == []
    mock_llm_client.generate_response.assert_not_called()


class OccurredAtEdge(BaseModel):
    """Edge model stub for OCCURRED_AT."""


@pytest.mark.asyncio
async def test_resolve_extracted_edges_keeps_unknown_names(monkeypatch):
    from graphiti_core.utils.maintenance import edge_operations as edge_ops

    monkeypatch.setattr(edge_ops, 'create_entity_edge_embeddings', AsyncMock(return_value=None))
    monkeypatch.setattr(EntityEdge, 'get_between_nodes', AsyncMock(return_value=[]))

    async def immediate_gather(*aws, max_coroutines=None):
        return [await aw for aw in aws]

    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)
    monkeypatch.setattr(edge_ops, 'search', AsyncMock(return_value=SearchResults()))

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'duplicate_facts': [],
            'contradicted_facts': [],
        }
    )

    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=llm_client,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )

    source_node = EntityNode(
        uuid='source_uuid',
        name='User Node',
        group_id='group_1',
        labels=['User'],
    )
    target_node = EntityNode(
        uuid='target_uuid',
        name='Topic Node',
        group_id='group_1',
        labels=['Topic'],
    )

    extracted_edge = EntityEdge(
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        name='INTERACTED_WITH',
        group_id='group_1',
        fact='User interacted with topic',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Episode content',
        valid_at=datetime.now(timezone.utc),
    )

    edge_types = {'OCCURRED_AT': OccurredAtEdge}
    edge_type_map = {('Event', 'Entity'): ['OCCURRED_AT']}

    resolved_edges, invalidated_edges, new_edges = await resolve_extracted_edges(
        clients,
        [extracted_edge],
        episode,
        [source_node, target_node],
        edge_types,
        edge_type_map,
    )

    assert resolved_edges[0].name == 'INTERACTED_WITH'
    assert invalidated_edges == []
    assert new_edges == resolved_edges  # No duplicates, so all edges are new


@pytest.mark.asyncio
async def test_resolve_extracted_edge_uses_integer_indices_for_duplicates(mock_llm_client):
    """Test that resolve_extracted_edge correctly uses integer indices for LLM duplicate detection."""
    # Mock LLM to return duplicate_facts with integer indices
    mock_llm_client.generate_response.return_value = {
        'duplicate_facts': [0, 1],  # LLM identifies first two related edges as duplicates
        'contradicted_facts': [],
    }

    extracted_edge = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='test_edge',
        group_id='group_1',
        fact='User likes yoga',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Episode content',
        valid_at=datetime.now(timezone.utc),
    )

    # Create multiple related edges - LLM should receive these with integer indices
    related_edge_0 = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='test_edge',
        group_id='group_1',
        fact='User enjoys yoga',
        episodes=['episode_1'],
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
        valid_at=None,
        invalid_at=None,
    )

    related_edge_1 = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='test_edge',
        group_id='group_1',
        fact='User practices yoga',
        episodes=['episode_2'],
        created_at=datetime.now(timezone.utc) - timedelta(days=2),
        valid_at=None,
        invalid_at=None,
    )

    related_edge_2 = EntityEdge(
        source_node_uuid='source_uuid',
        target_node_uuid='target_uuid',
        name='test_edge',
        group_id='group_1',
        fact='User loves swimming',
        episodes=['episode_3'],
        created_at=datetime.now(timezone.utc) - timedelta(days=3),
        valid_at=None,
        invalid_at=None,
    )

    related_edges = [related_edge_0, related_edge_1, related_edge_2]

    resolved_edge, invalidated, duplicates = await resolve_extracted_edge(
        mock_llm_client,
        extracted_edge,
        related_edges,
        [],
        episode,
        edge_type_candidates=None,
    )

    # Verify LLM was called
    mock_llm_client.generate_response.assert_called_once()

    # Verify the system correctly identified duplicates using integer indices
    # The LLM returned [0, 1], so related_edge_0 and related_edge_1 should be marked as duplicates
    assert len(duplicates) == 2
    assert related_edge_0 in duplicates
    assert related_edge_1 in duplicates
    assert invalidated == []

    # Verify that the resolved edge is one of the duplicates (the first one found)
    # Check UUID since the episode list gets modified
    assert resolved_edge.uuid == related_edge_0.uuid
    assert episode.uuid in resolved_edge.episodes


@pytest.mark.asyncio
async def test_resolve_extracted_edges_fast_path_deduplication(monkeypatch):
    """Test that resolve_extracted_edges deduplicates exact matches before parallel processing."""
    from graphiti_core.utils.maintenance import edge_operations as edge_ops

    monkeypatch.setattr(edge_ops, 'create_entity_edge_embeddings', AsyncMock(return_value=None))
    monkeypatch.setattr(EntityEdge, 'get_between_nodes', AsyncMock(return_value=[]))

    # Track how many times resolve_extracted_edge is called
    resolve_call_count = 0

    async def mock_resolve_extracted_edge(
        llm_client,
        extracted_edge,
        related_edges,
        existing_edges,
        episode,
        edge_type_candidates=None,
    ):
        nonlocal resolve_call_count
        resolve_call_count += 1
        return extracted_edge, [], []

    # Mock semaphore_gather to execute awaitable immediately
    async def immediate_gather(*aws, max_coroutines=None):
        results = []
        for aw in aws:
            results.append(await aw)
        return results

    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)
    monkeypatch.setattr(edge_ops, 'search', AsyncMock(return_value=SearchResults()))
    monkeypatch.setattr(edge_ops, 'resolve_extracted_edge', mock_resolve_extracted_edge)

    llm_client = MagicMock()
    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=llm_client,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )

    source_node = EntityNode(
        uuid='source_uuid',
        name='Assistant',
        group_id='group_1',
        labels=['Entity'],
    )
    target_node = EntityNode(
        uuid='target_uuid',
        name='User',
        group_id='group_1',
        labels=['Entity'],
    )

    # Create 3 identical edges
    edge1 = EntityEdge(
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        name='recommends',
        group_id='group_1',
        fact='assistant recommends yoga poses',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    edge2 = EntityEdge(
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        name='recommends',
        group_id='group_1',
        fact='  Assistant Recommends YOGA Poses  ',  # Different whitespace/case
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    edge3 = EntityEdge(
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        name='recommends',
        group_id='group_1',
        fact='assistant recommends yoga poses',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )

    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Episode content',
        valid_at=datetime.now(timezone.utc),
    )

    resolved_edges, invalidated_edges, new_edges = await resolve_extracted_edges(
        clients,
        [edge1, edge2, edge3],
        episode,
        [source_node, target_node],
        {},
        {},
    )

    # Fast path should have deduplicated the 3 identical edges to 1
    # So resolve_extracted_edge should only be called once
    assert resolve_call_count == 1
    assert len(resolved_edges) == 1
    assert invalidated_edges == []
    assert new_edges == resolved_edges  # All edges are new (no graph duplicates)


class InterpersonalRelationship(BaseModel):
    """A relationship between two people."""


class LocatedIn(BaseModel):
    """A relationship indicating something is located in a place."""


def test_edge_type_signatures_map_preserves_multiple_signatures():
    """Test that edge types used across multiple node type pairs preserve all signatures.

    This tests the fix for the bug where dict comprehension would overwrite
    previous signatures when the same edge type appeared in multiple node pairs.
    """
    # Edge type map where the same edge type is used for multiple node pair signatures
    # This is the scenario that was broken before the fix
    edge_type_map: dict[tuple[str, str], list[str]] = {
        ('Person', 'Person'): ['InterpersonalRelationship'],
        ('Person', 'Entity'): ['InterpersonalRelationship'],  # Same type, different signature
        ('Person', 'City'): ['LocatedIn'],
        ('Entity', 'City'): ['LocatedIn'],  # Same type, different signature
    }

    edge_types: dict[str, type[BaseModel]] = {
        'InterpersonalRelationship': InterpersonalRelationship,
        'LocatedIn': LocatedIn,
    }

    # Build the mapping the same way as in extract_edges (the fixed implementation)
    edge_type_signatures_map: dict[str, list[tuple[str, str]]] = {}
    for signature, edge_type_names in edge_type_map.items():
        for edge_type in edge_type_names:
            if edge_type not in edge_type_signatures_map:
                edge_type_signatures_map[edge_type] = []
            edge_type_signatures_map[edge_type].append(signature)

    # Verify InterpersonalRelationship has BOTH signatures preserved
    assert 'InterpersonalRelationship' in edge_type_signatures_map
    interpersonal_signatures = edge_type_signatures_map['InterpersonalRelationship']
    assert len(interpersonal_signatures) == 2
    assert ('Person', 'Person') in interpersonal_signatures
    assert ('Person', 'Entity') in interpersonal_signatures

    # Verify LocatedIn has BOTH signatures preserved
    assert 'LocatedIn' in edge_type_signatures_map
    located_signatures = edge_type_signatures_map['LocatedIn']
    assert len(located_signatures) == 2
    assert ('Person', 'City') in located_signatures
    assert ('Entity', 'City') in located_signatures

    # Verify the edge_types_context structure
    edge_types_context = [
        {
            'fact_type_name': type_name,
            'fact_type_signatures': edge_type_signatures_map.get(type_name, [('Entity', 'Entity')]),
            'fact_type_description': type_model.__doc__,
        }
        for type_name, type_model in edge_types.items()
    ]

    # Verify the context has the correct structure with plural 'fact_type_signatures'
    for ctx in edge_types_context:
        assert 'fact_type_signatures' in ctx
        assert isinstance(ctx['fact_type_signatures'], list)
        assert len(ctx['fact_type_signatures']) == 2  # Each type has 2 signatures


def test_edge_type_signatures_map_single_signature_still_works():
    """Test that edge types with a single signature still work correctly."""
    edge_type_map: dict[tuple[str, str], list[str]] = {
        ('Person', 'Organization'): ['WorksAt'],
        ('Person', 'City'): ['LivesIn'],
    }

    edge_types: dict[str, type[BaseModel]] = {
        'WorksAt': BaseModel,
        'LivesIn': BaseModel,
    }

    # Build the mapping
    edge_type_signatures_map: dict[str, list[tuple[str, str]]] = {}
    for signature, edge_type_names in edge_type_map.items():
        for edge_type in edge_type_names:
            if edge_type not in edge_type_signatures_map:
                edge_type_signatures_map[edge_type] = []
            edge_type_signatures_map[edge_type].append(signature)

    # Verify each edge type has exactly one signature
    assert len(edge_type_signatures_map['WorksAt']) == 1
    assert ('Person', 'Organization') in edge_type_signatures_map['WorksAt']

    assert len(edge_type_signatures_map['LivesIn']) == 1
    assert ('Person', 'City') in edge_type_signatures_map['LivesIn']

    # Verify the context structure
    edge_types_context = [
        {
            'fact_type_name': type_name,
            'fact_type_signatures': edge_type_signatures_map.get(type_name, [('Entity', 'Entity')]),
            'fact_type_description': type_model.__doc__,
        }
        for type_name, type_model in edge_types.items()
    ]

    for ctx in edge_types_context:
        assert 'fact_type_signatures' in ctx
        assert isinstance(ctx['fact_type_signatures'], list)
        assert len(ctx['fact_type_signatures']) == 1


@pytest.mark.asyncio
async def test_extract_edges_drops_self_edges(monkeypatch):
    """Self-edges (source == target) are dropped during extraction."""
    from graphiti_core.prompts.extract_edges import Edge as ExtractedEdge
    from graphiti_core.prompts.extract_edges import ExtractedEdges

    alice = EntityNode(
        uuid='alice_uuid',
        name='Alice',
        group_id='group_1',
        labels=['Person'],
    )
    bob = EntityNode(
        uuid='bob_uuid',
        name='Bob',
        group_id='group_1',
        labels=['Person'],
    )

    # LLM returns one valid edge and one self-edge
    llm_response = ExtractedEdges(
        edges=[
            ExtractedEdge(
                source_entity_name='Alice',
                target_entity_name='Bob',
                relation_type='CONGRATULATED',
                fact='Alice congratulated Bob',
                valid_at=None,
                invalid_at=None,
            ),
            ExtractedEdge(
                source_entity_name='Alice',
                target_entity_name='Alice',
                relation_type='FEELS_HAPPY',
                fact='Alice feels happy',
                valid_at=None,
                invalid_at=None,
            ),
        ]
    ).model_dump()

    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(return_value=llm_response)

    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=mock_llm,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )

    episode = EpisodicNode(
        uuid='ep_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Alice congratulated Bob. Alice feels happy.',
        valid_at=datetime.now(timezone.utc),
    )

    edges = await extract_edges(
        clients,
        episode,
        [alice, bob],
        [],
        {},
        group_id='group_1',
    )

    # Only the Alice -> Bob edge survives; the Alice -> Alice self-edge is dropped
    assert len(edges) == 1
    assert edges[0].source_node_uuid == 'alice_uuid'
    assert edges[0].target_node_uuid == 'bob_uuid'
    assert edges[0].name == 'CONGRATULATED'


@pytest.mark.asyncio
async def test_extract_edges_keeps_valid_edges_with_same_name_different_nodes(monkeypatch):
    """Edges between different nodes that happen to share a name are NOT self-edges."""
    from graphiti_core.prompts.extract_edges import Edge as ExtractedEdge
    from graphiti_core.prompts.extract_edges import ExtractedEdges

    alice = EntityNode(
        uuid='alice_uuid',
        name='Alice',
        group_id='group_1',
        labels=['Person'],
    )
    paris = EntityNode(
        uuid='paris_uuid',
        name='Paris',
        group_id='group_1',
        labels=['City'],
    )

    llm_response = ExtractedEdges(
        edges=[
            ExtractedEdge(
                source_entity_name='Alice',
                target_entity_name='Paris',
                relation_type='LIVES_IN',
                fact='Alice lives in Paris',
                valid_at=None,
                invalid_at=None,
            ),
        ]
    ).model_dump()

    mock_llm = MagicMock()
    mock_llm.generate_response = AsyncMock(return_value=llm_response)

    clients = SimpleNamespace(
        driver=MagicMock(),
        llm_client=mock_llm,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )

    episode = EpisodicNode(
        uuid='ep_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Alice lives in Paris.',
        valid_at=datetime.now(timezone.utc),
    )

    edges = await extract_edges(
        clients,
        episode,
        [alice, paris],
        [],
        {},
        group_id='group_1',
    )

    assert len(edges) == 1
    assert edges[0].source_node_uuid == 'alice_uuid'
    assert edges[0].target_node_uuid == 'paris_uuid'


class _EmploymentEdge(BaseModel):
    """Edge attribute schema mirroring a customer's EMPLOYMENT relation."""

    title: str | None = None
    is_current: str | None = None


@pytest.mark.asyncio
async def test_resolve_extracted_edge_overcap_attribute_preserves_prior(monkeypatch):
    """End-to-end: when the LLM bleeds meta-reasoning into one attribute, the
    cap drops that field and the merge logic falls back to the prior on-edge
    value — without affecting fields the LLM legitimately updated."""
    from graphiti_core.utils.maintenance import edge_operations as edge_ops

    # Avoid the timestamps LLM call that follows attribute extraction.
    monkeypatch.setattr(edge_ops, '_extract_edge_timestamps', AsyncMock(return_value=None))

    # The LLM returns a clean update for `title` but bleeds 9 KB of meta-reasoning into
    # `is_current` — the kind of failure the customer reported.
    bleed = (
        'true (implied by context, but no new information explicitly stated. '
        'I should ideally remove it or set it to null. However, the instruction '
        'is to preserve existing values...) '
    ) * 30
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'title': 'Senior Engineer', 'is_current': bleed}
    )

    extracted_edge = EntityEdge(
        source_node_uuid='person_uuid',
        target_node_uuid='company_uuid',
        name='EMPLOYMENT',
        group_id='group_1',
        fact='Sam was promoted to Senior Engineer at Northwind',
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
        attributes={'title': 'Engineer', 'is_current': 'true'},
    )

    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Sam was promoted to Senior Engineer.',
        valid_at=datetime.now(timezone.utc),
    )

    resolved, dupes, invalidated = await resolve_extracted_edge(
        llm_client,
        extracted_edge,
        related_edges=[],
        existing_edges=[],
        episode=episode,
        edge_type_candidates={'EMPLOYMENT': _EmploymentEdge},
    )

    # Title should reflect the legitimate LLM update.
    assert resolved.attributes['title'] == 'Senior Engineer'
    # is_current was bleed-dropped, so the prior value must be preserved (not the
    # 9 KB rant, not None, not absent).
    assert resolved.attributes['is_current'] == 'true'
    assert dupes == []
    assert invalidated == []


def _entity_edge(source_uuid: str, target_uuid: str, fact: str) -> EntityEdge:
    return EntityEdge(
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        name='RELATES',
        group_id='group_1',
        fact=fact,
        episodes=[],
        created_at=datetime.now(timezone.utc),
        valid_at=None,
        invalid_at=None,
    )


def _entity_node(uuid: str) -> EntityNode:
    return EntityNode(uuid=uuid, name=f'Node {uuid}', group_id='group_1', labels=['Entity'])


def _episode() -> EpisodicNode:
    return EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source='message',
        source_description='desc',
        content='Episode content',
        valid_at=datetime.now(timezone.utc),
    )


def _mock_clients() -> SimpleNamespace:
    return SimpleNamespace(
        driver=MagicMock(),
        llm_client=MagicMock(),
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )


def _patch_resolve_extracted_edges_deps(monkeypatch, candidates_by_pair, related_by_fact):
    """Wire resolve_extracted_edges onto in-memory stubs and return (search spy, observed pairings)."""
    from graphiti_core.utils.maintenance import edge_operations as edge_ops

    monkeypatch.setattr(edge_ops, 'create_entity_edge_embeddings', AsyncMock(return_value=None))

    async def immediate_gather(*aws, max_coroutines=None):
        return [await aw for aw in aws]

    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)

    async def fake_get_between_nodes(driver, source_node_uuid, target_node_uuid):
        return list(candidates_by_pair[(source_node_uuid, target_node_uuid)])

    monkeypatch.setattr(EntityEdge, 'get_between_nodes', fake_get_between_nodes)

    async def fake_search(clients, query, group_ids=None, config=None, search_filter=None):
        if search_filter is not None and search_filter.edge_uuids is not None:
            return SearchResults(edges=list(related_by_fact.get(query, [])))
        return SearchResults()

    search_spy = AsyncMock(side_effect=fake_search)
    monkeypatch.setattr(edge_ops, 'search', search_spy)

    observed: list[tuple[str, list[str]]] = []

    async def record_resolve(
        llm_client,
        extracted_edge,
        related_edges,
        existing_edges,
        episode,
        edge_type_candidates=None,
    ):
        observed.append((extracted_edge.uuid, [edge.uuid for edge in related_edges]))
        return extracted_edge, [], []

    monkeypatch.setattr(edge_ops, 'resolve_extracted_edge', record_resolve)

    return search_spy, observed


@pytest.mark.asyncio
async def test_search_related_edges_skips_query_without_candidates(monkeypatch):
    """An empty edge_uuids filter can never match, so no search may be issued at all."""
    from graphiti_core.utils.maintenance import edge_operations as edge_ops

    search_spy = AsyncMock(return_value=SearchResults())
    monkeypatch.setattr(edge_ops, 'search', search_spy)

    results = await search_related_edges(
        _mock_clients(), _entity_edge('n1', 'n2', 'alice works at acme'), []
    )

    assert search_spy.await_count == 0
    # Equivalent to what the unsatisfiable filtered search would have returned.
    assert results == SearchResults()
    assert results.edges == []
    assert results.nodes == []
    assert results.episodes == []
    assert results.communities == []


@pytest.mark.asyncio
async def test_search_related_edges_queries_with_candidate_uuids(monkeypatch):
    """With candidates present the filtered search must still be issued unchanged."""
    from graphiti_core.utils.maintenance import edge_operations as edge_ops

    candidate = _entity_edge('n1', 'n2', 'alice worked at acme')
    expected = SearchResults(edges=[candidate])
    search_spy = AsyncMock(return_value=expected)
    monkeypatch.setattr(edge_ops, 'search', search_spy)

    extracted_edge = _entity_edge('n1', 'n2', 'alice works at acme')
    results = await search_related_edges(_mock_clients(), extracted_edge, [candidate])

    assert results is expected
    assert search_spy.await_count == 1
    assert search_spy.await_args.kwargs['search_filter'] == SearchFilters(
        edge_uuids=[candidate.uuid]
    )


@pytest.mark.asyncio
async def test_resolve_extracted_edges_issues_no_filtered_search_for_new_node_pairs(monkeypatch):
    """New node pairs have no candidates, so no edge_uuids-filtered search may reach the graph."""
    edge_a = _entity_edge('n1', 'n2', 'alice works at acme')
    edge_b = _entity_edge('n3', 'n4', 'bob lives in berlin')
    candidates_by_pair = {('n1', 'n2'): [], ('n3', 'n4'): []}

    search_spy, observed = _patch_resolve_extracted_edges_deps(monkeypatch, candidates_by_pair, {})

    resolved_edges, invalidated_edges, new_edges = await resolve_extracted_edges(
        _mock_clients(),
        [edge_a, edge_b],
        _episode(),
        [_entity_node(uuid) for uuid in ('n1', 'n2', 'n3', 'n4')],
        {},
        {},
    )

    edge_uuid_filters = [
        call.kwargs['search_filter'].edge_uuids for call in search_spy.await_args_list
    ]
    # Not a single search carries an (unsatisfiable) edge_uuids filter...
    assert [f for f in edge_uuid_filters if f is not None] == []
    # ...so only the unfiltered invalidation-candidate pass runs: N queries instead of 2N.
    assert search_spy.await_count == 2

    # And the outcome is exactly what the unsatisfiable search would have produced.
    assert observed == [(edge_a.uuid, []), (edge_b.uuid, [])]
    assert [edge.uuid for edge in resolved_edges] == [edge_a.uuid, edge_b.uuid]
    assert invalidated_edges == []
    assert new_edges == resolved_edges


@pytest.mark.asyncio
async def test_resolve_extracted_edges_filters_search_by_candidate_uuids(monkeypatch):
    """When candidates exist the filtered search keeps running with their uuids."""
    extracted_edge = _entity_edge('n1', 'n2', 'alice works at acme')
    candidate_one = _entity_edge('n1', 'n2', 'alice worked at acme')
    candidate_two = _entity_edge('n1', 'n2', 'alice joined acme')
    candidates_by_pair = {('n1', 'n2'): [candidate_one, candidate_two]}
    related_by_fact = {'alice works at acme': [candidate_one]}

    search_spy, observed = _patch_resolve_extracted_edges_deps(
        monkeypatch, candidates_by_pair, related_by_fact
    )

    await resolve_extracted_edges(
        _mock_clients(),
        [extracted_edge],
        _episode(),
        [_entity_node('n1'), _entity_node('n2')],
        {},
        {},
    )

    filtered_calls = [
        call
        for call in search_spy.await_args_list
        if call.kwargs['search_filter'].edge_uuids is not None
    ]
    assert len(filtered_calls) == 1
    assert filtered_calls[0].args[1] == 'alice works at acme'
    assert filtered_calls[0].kwargs['search_filter'] == SearchFilters(
        edge_uuids=[candidate_one.uuid, candidate_two.uuid]
    )
    assert observed == [(extracted_edge.uuid, [candidate_one.uuid])]


@pytest.mark.asyncio
async def test_resolve_extracted_edges_pairs_each_result_with_its_own_edge(monkeypatch):
    """Mixed candidate/no-candidate edges keep result order and per-edge correspondence."""
    edge_a = _entity_edge('n1', 'n2', 'alice works at acme')
    edge_b = _entity_edge('n3', 'n4', 'bob lives in berlin')
    edge_c = _entity_edge('n1', 'n4', 'alice knows dave')
    candidate_a = _entity_edge('n1', 'n2', 'alice worked at acme')
    candidate_c = _entity_edge('n1', 'n4', 'alice met dave')

    candidates_by_pair = {
        ('n1', 'n2'): [candidate_a],
        ('n3', 'n4'): [],
        ('n1', 'n4'): [candidate_c],
    }
    related_by_fact = {
        'alice works at acme': [candidate_a],
        'alice knows dave': [candidate_c],
    }

    search_spy, observed = _patch_resolve_extracted_edges_deps(
        monkeypatch, candidates_by_pair, related_by_fact
    )

    extracted_edges = [edge_a, edge_b, edge_c]
    resolved_edges, _, _ = await resolve_extracted_edges(
        _mock_clients(),
        extracted_edges,
        _episode(),
        [_entity_node(uuid) for uuid in ('n1', 'n2', 'n3', 'n4')],
        {},
        {},
    )

    assert len(observed) == len(extracted_edges)
    assert observed == [
        (edge_a.uuid, [candidate_a.uuid]),
        (edge_b.uuid, []),
        (edge_c.uuid, [candidate_c.uuid]),
    ]
    assert [edge.uuid for edge in resolved_edges] == [edge.uuid for edge in extracted_edges]
    # The skipped edge never contributed an unsatisfiable filter.
    edge_uuid_filters = [
        call.kwargs['search_filter'].edge_uuids for call in search_spy.await_args_list
    ]
    assert [] not in edge_uuid_filters
