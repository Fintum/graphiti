from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.identity_constraints import merge_provenance
from graphiti_core.utils.maintenance import edge_operations
from graphiti_core.utils.maintenance.edge_operations import (
    _settle_edge_validity,
    extract_edges,
    resolve_extracted_edge,
    resolve_extracted_edges,
    search_related_edges,
)
from graphiti_core.utils.maintenance.resolution_decisions import EdgeQuestion, EdgeVerdict


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
            name='test_edge',
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

    # Track how many edges reach resolution
    question_count = 0

    def count_edge_question(extracted_edge, related_edges, existing_edges, *slot_policy):
        nonlocal question_count
        question_count += 1
        return EdgeQuestion(extracted_edge, (), ())

    # Mock semaphore_gather to execute awaitable immediately
    async def immediate_gather(*aws, max_coroutines=None):
        results = []
        for aw in aws:
            results.append(await aw)
        return results

    monkeypatch.setattr(edge_ops, 'semaphore_gather', immediate_gather)
    monkeypatch.setattr(edge_ops, 'search', AsyncMock(return_value=SearchResults()))
    monkeypatch.setattr(edge_ops, '_edge_question', count_edge_question)
    monkeypatch.setattr(edge_ops, '_extract_edge_timestamps', AsyncMock(return_value=None))

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
    # So only one edge should reach resolution
    assert question_count == 1
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

    async def fake_search(
        clients, query, group_ids=None, config=None, search_filter=None, query_vector=None
    ):
        if search_filter is not None and search_filter.edge_uuids is not None:
            return SearchResults(edges=list(related_by_fact.get(query, [])))
        return SearchResults()

    search_spy = AsyncMock(side_effect=fake_search)
    monkeypatch.setattr(edge_ops, 'search', search_spy)

    observed: list[tuple[str, list[str]]] = []

    def record_question(extracted_edge, related_edges, existing_edges, *slot_policy):
        observed.append((extracted_edge.uuid, [edge.uuid for edge in related_edges]))
        return EdgeQuestion(extracted_edge, (), ())

    monkeypatch.setattr(edge_ops, '_edge_question', record_question)
    monkeypatch.setattr(edge_ops, '_extract_edge_timestamps', AsyncMock(return_value=None))

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


EARLIER = datetime(2025, 1, 1, tzinfo=timezone.utc)
MIDDLE = datetime(2025, 3, 1, tzinfo=timezone.utc)
LATER = datetime(2025, 6, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)
DISCARD_MARK = {
    'discarded_at': '2026-11-19T10:00:00+00:00',
    'discarded_by': 'suppression-1',
    'kept_episodes': ['episode-of-board'],
}
OTHER_MARK = {
    'discarded_at': '2026-11-20T09:00:00+00:00',
    'discarded_by': 'suppression-2',
    'kept_episodes': ['episode-of-board', 'another-episode'],
}
SALES = {'org_public': True, 'directory': False, 'acl_key': 'slack:sales'}
BOARD = {'org_public': True, 'directory': False, 'acl_key': 'slack:board'}


def _price(fact: str, valid_at: datetime, **attributes: object) -> EntityEdge:
    return EntityEdge(
        source_node_uuid='plan',
        target_node_uuid='price',
        name='COSTS',
        group_id='group_1',
        fact=fact,
        episodes=['episode_1'],
        created_at=EARLIER,
        valid_at=valid_at,
        invalid_at=None,
        attributes=dict(attributes),
    )


def _validity(edge: EntityEdge) -> tuple:
    return edge.invalid_at, edge.expired_at, edge.attributes.get('invalidated_by')


def _mark(edge: EntityEdge) -> dict:
    return {key: edge.attributes.get(key) for key in DISCARD_MARK}


def test_a_discarded_restatement_invalidates_no_older_visible_edge():
    discarded = _price('The plan costs 99 €', LATER, **DISCARD_MARK)
    visible = _price('The plan costs 49 €', EARLIER)

    assert _settle_edge_validity(discarded, [visible]) == []
    assert _validity(visible) == (None, None, None)


def test_a_discarded_restatement_is_not_expired_by_a_newer_visible_edge():
    discarded = _price('The plan costs 99 €', EARLIER, **DISCARD_MARK)
    visible = _price('The plan costs 49 €', LATER)

    assert _settle_edge_validity(discarded, [visible]) == []
    assert _validity(discarded) == (None, None, None)
    assert _validity(visible) == (None, None, None)


def test_a_newer_discarded_edge_never_expires_the_new_edge():
    new = _price('The plan costs 49 €', EARLIER)
    discarded = _price('The plan costs 99 €', LATER, **DISCARD_MARK)

    assert _settle_edge_validity(new, [discarded]) == []
    assert _validity(new) == (None, None, None)


def test_a_newer_visible_edge_still_expires_the_new_edge_past_a_discarded_one(monkeypatch):
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)
    new = _price('The plan costs 49 €', EARLIER)
    discarded = _price('The plan costs 99 €', MIDDLE, **DISCARD_MARK)
    visible = _price('The plan costs 59 €', LATER)

    assert _settle_edge_validity(new, [visible, discarded]) == []
    assert (new.invalid_at, new.expired_at) == (LATER, NOW)


def test_a_new_edge_still_invalidates_an_older_discarded_edge(monkeypatch):
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)
    new = _price('The plan costs 49 €', LATER)
    discarded = _price('The plan costs 99 €', EARLIER, **DISCARD_MARK)

    assert _settle_edge_validity(new, [discarded]) == [discarded]
    assert (discarded.invalid_at, discarded.expired_at) == (LATER, NOW)
    assert _validity(new) == (None, None, None)


@pytest.mark.asyncio
async def test_a_restatement_absorbed_by_a_discarded_edge_keeps_its_mark_and_invalidates_nothing(
    mock_llm_client,
):
    mock_llm_client.generate_response.return_value = {
        'duplicate_facts': [0],
        'contradicted_facts': [1],
    }
    restatement = _price('The plan now costs 99 €', LATER)
    discarded = _price('The plan costs 99 €', LATER, **DISCARD_MARK)
    visible = _price('The plan costs 49 €', EARLIER)

    resolved, invalidated, _ = await resolve_extracted_edge(
        mock_llm_client, restatement, [discarded], [visible], _episode()
    )

    assert resolved.uuid == discarded.uuid
    assert _mark(resolved) == DISCARD_MARK
    assert invalidated == []
    assert _validity(visible) == (None, None, None)


@pytest.mark.asyncio
async def test_a_discarded_restatement_marks_the_unmarked_copy_of_its_audience_and_invalidates_nothing(
    mock_llm_client,
):
    mock_llm_client.generate_response.return_value = {
        'duplicate_facts': [0],
        'contradicted_facts': [1],
    }
    restatement = _price('The plan now costs 99 €', LATER, **SALES, **DISCARD_MARK)
    copy = _price('The plan costs 99 €', LATER, **SALES)
    visible = _price('The plan costs 49 €', EARLIER, **SALES)

    resolved, invalidated, _ = await resolve_extracted_edge(
        mock_llm_client, restatement, [copy], [visible], _episode()
    )

    assert [edge.fact for edge in invalidated] == []
    assert _validity(visible) == (None, None, None)
    assert resolved.uuid == copy.uuid
    assert _mark(resolved) == DISCARD_MARK


@pytest.mark.asyncio
async def test_a_discarded_copy_keeps_its_own_mark_against_another_incoming_mark(mock_llm_client):
    mock_llm_client.generate_response.return_value = {
        'duplicate_facts': [0],
        'contradicted_facts': [],
    }
    restatement = _price('The plan now costs 99 €', LATER, **SALES, **OTHER_MARK)
    copy = _price('The plan costs 99 €', LATER, **SALES, **DISCARD_MARK)

    resolved, _, _ = await resolve_extracted_edge(
        mock_llm_client, restatement, [copy], [], _episode()
    )

    assert resolved.uuid == copy.uuid
    assert _mark(resolved) == DISCARD_MARK


def test_an_edge_takes_the_kept_episodes_with_the_discard_mark_of_its_audience():
    stored = {**SALES, 'provenance': ['episode_1']}

    merged = merge_provenance(stored, {**SALES, **DISCARD_MARK, 'provenance': ['episode_2']})

    assert merged == {**SALES, **DISCARD_MARK, 'provenance': ['episode_1', 'episode_2']}


def test_an_edge_never_takes_the_discard_mark_of_another_audience():
    stored = {**SALES, 'provenance': ['episode_1']}

    merged = merge_provenance(stored, {**BOARD, **DISCARD_MARK, 'provenance': ['episode_2']})

    assert merged == {**SALES, 'provenance': ['episode_1', 'episode_2']}


SLOTS = [
    pytest.param('IMPLEMENTS', 'lumora', 'project', False, id='another relation on one end'),
    pytest.param('CONTRACTS', 'harinera', 'lumora', False, id='another relation on no end'),
    pytest.param('RESPONSIBLE_FOR', 'brezo', 'dashboards', False, id='the same relation on no end'),
    pytest.param('RESPONSIBLE_FOR', 'brezo', 'project', False, id='the same relation on one end'),
    pytest.param('LEADS', 'alba', 'project', True, id='the same two ends under another relation'),
    pytest.param('LED_BY', 'project', 'alba', True, id='the same two ends reversed'),
]


def _responsibility(name: str, source: str, target: str, valid_at: datetime) -> EntityEdge:
    return EntityEdge(
        source_node_uuid=source,
        target_node_uuid=target,
        name=name,
        group_id='group_1',
        fact=f'{source} {name} {target}',
        episodes=['episode_1'],
        created_at=EARLIER,
        valid_at=valid_at,
        invalid_at=None,
    )


def _contradicting_everything(mock_llm_client) -> None:
    mock_llm_client.generate_response.return_value = {
        'duplicate_facts': [],
        'contradicted_facts': [0],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('name, source, target, same_slot', SLOTS)
async def test_a_new_edge_ends_an_older_one_only_in_its_slot(
    mock_llm_client, monkeypatch, name, source, target, same_slot
):
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)
    _contradicting_everything(mock_llm_client)
    new = _responsibility('RESPONSIBLE_FOR', 'alba', 'project', LATER)
    older = _responsibility(name, source, target, EARLIER)

    _, invalidated, _ = await resolve_extracted_edge(mock_llm_client, new, [], [older], _episode())

    assert [edge.uuid for edge in invalidated] == ([older.uuid] if same_slot else [])
    assert (older.invalid_at, older.expired_at) == ((LATER, NOW) if same_slot else (None, None))


@pytest.mark.asyncio
@pytest.mark.parametrize('name, source, target, same_slot', SLOTS)
async def test_a_newer_edge_ends_the_new_one_only_in_its_slot(
    mock_llm_client, monkeypatch, name, source, target, same_slot
):
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)
    _contradicting_everything(mock_llm_client)
    new = _responsibility('RESPONSIBLE_FOR', 'alba', 'project', EARLIER)
    newer = _responsibility(name, source, target, LATER)

    resolved, invalidated, _ = await resolve_extracted_edge(
        mock_llm_client, new, [], [newer], _episode()
    )

    assert invalidated == []
    assert (resolved.invalid_at, resolved.expired_at) == (
        (LATER, NOW) if same_slot else (None, None)
    )


@pytest.mark.asyncio
async def test_the_decisions_are_offered_only_the_stored_edges_in_the_slot(monkeypatch):
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)
    new = _responsibility('RESPONSIBLE_FOR', 'alba', 'project', LATER)
    stored = {slot.id: _responsibility(*slot.values[:3], EARLIER) for slot in SLOTS}
    in_slot = [stored[slot.id].uuid for slot in SLOTS if slot.values[3]]

    async def get_between_nodes(driver, source_node_uuid, target_node_uuid):
        return []

    async def search(
        clients, query, group_ids=None, config=None, search_filter=None, query_vector=None
    ):
        return SearchResults(edges=list(stored.values()))

    offered: list[list[str]] = []

    async def resolve_edges(episode, questions):
        offered.extend([edge.uuid for edge in question.invalidations] for question in questions)
        return [EdgeVerdict(None, question.invalidations) for question in questions]

    monkeypatch.setattr(EntityEdge, 'get_between_nodes', get_between_nodes)
    monkeypatch.setattr(edge_operations, 'search', search)
    monkeypatch.setattr(
        edge_operations, 'create_entity_edge_embeddings', AsyncMock(return_value=None)
    )
    decisions = SimpleNamespace(
        resolve_edges=resolve_edges, date_edges=AsyncMock(return_value=None)
    )

    _, invalidated, _ = await resolve_extracted_edges(
        _mock_clients(),
        [new],
        _episode(),
        [
            _entity_node(uuid)
            for uuid in ('alba', 'project', 'lumora', 'harinera', 'brezo', 'dashboards')
        ],
        {},
        {},
        decisions=decisions,
    )

    assert offered == [in_slot]
    assert [edge.uuid for edge in invalidated] == in_slot


@dataclass(frozen=True)
class _Functional:
    anchor: str
    labels: frozenset[str]


FUNCTIONAL = {
    'HAS_STATUS': _Functional('source', frozenset({'Task'})),
    'RESPONSIBLE_FOR': _Functional('target', frozenset({'Task'})),
    'REPORTS_TO': _Functional('source', frozenset({'Person'})),
}
LABELS = {
    'task': 'Task',
    'project': 'Project',
    'open': 'Status',
    'done': 'Status',
    'alba': 'Person',
    'brezo': 'Person',
    'ciro': 'Person',
    'bug': 'Challenge',
    'leak': 'Challenge',
    'pipedrive': 'Company',
}
FORMS = [
    pytest.param(
        ('HAS_STATUS', 'task', 'done'),
        ('HAS_STATUS', 'task', 'open'),
        True,
        id='a status change of a task, on the source',
    ),
    pytest.param(
        ('RESPONSIBLE_FOR', 'brezo', 'task'),
        ('RESPONSIBLE_FOR', 'alba', 'task'),
        True,
        id='a new owner of the same task, on the target',
    ),
    pytest.param(
        ('RESPONSIBLE_FOR', 'brezo', 'project'),
        ('RESPONSIBLE_FOR', 'alba', 'project'),
        False,
        id='an owner of a target that is no task',
    ),
    pytest.param(
        ('AFFECTS', 'leak', 'pipedrive'),
        ('AFFECTS', 'bug', 'pipedrive'),
        False,
        id='the same target under a relation off the list',
    ),
    pytest.param(
        ('RESPONSIBLE_FOR', 'alba', 'task'),
        ('RESPONSIBLE_FOR', 'alba', 'project'),
        False,
        id='the same source of a relation anchored on its target',
    ),
    pytest.param(
        ('REPORTS_TO', 'brezo', 'ciro'),
        ('REPORTS_TO', 'alba', 'brezo'),
        False,
        id='crossed: the end in common on the other side',
    ),
    pytest.param(
        ('HAS_STATUS', 'task', 'done'),
        ('BLOCKED_AS', 'done', 'task'),
        True,
        id='the same two ends under another relation, reversed',
    ),
]


async def _ended_in_slot(monkeypatch, new, older, functional_slots):
    """What resolve_extracted_edges ends for the new edge when the decisions contradict all."""
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)

    async def get_between_nodes(driver, source_node_uuid, target_node_uuid):
        return []

    async def search(
        clients, query, group_ids=None, config=None, search_filter=None, query_vector=None
    ):
        return SearchResults(edges=[older])

    async def resolve_edges(episode, questions):
        return [EdgeVerdict(None, question.invalidations) for question in questions]

    monkeypatch.setattr(EntityEdge, 'get_between_nodes', get_between_nodes)
    monkeypatch.setattr(edge_operations, 'search', search)
    monkeypatch.setattr(
        edge_operations, 'create_entity_edge_embeddings', AsyncMock(return_value=None)
    )
    decisions = SimpleNamespace(
        resolve_edges=resolve_edges, date_edges=AsyncMock(return_value=None)
    )
    nodes = [
        EntityNode(uuid=uuid, name=uuid, group_id='group_1', labels=['Entity', label])
        for uuid, label in LABELS.items()
    ]
    _, invalidated, _ = await resolve_extracted_edges(
        _mock_clients(),
        [new],
        _episode(),
        nodes,
        {},
        {},
        decisions=decisions,
        functional_slots=functional_slots,
    )
    return [edge.uuid for edge in invalidated]


@pytest.mark.asyncio
@pytest.mark.parametrize('new_form, older_form, same_slot', FORMS)
async def test_a_functional_slot_is_one_value_on_its_anchor(
    monkeypatch, new_form, older_form, same_slot
):
    new = _responsibility(*new_form, LATER)
    older = _responsibility(*older_form, EARLIER)

    ended = await _ended_in_slot(monkeypatch, new, older, FUNCTIONAL)

    assert ended == ([older.uuid] if same_slot else [])
    assert (older.invalid_at, older.expired_at) == ((LATER, NOW) if same_slot else (None, None))


@pytest.mark.asyncio
@pytest.mark.parametrize('new_form, older_form, same_slot', FORMS)
async def test_without_functional_slots_only_the_same_two_ends_fill_a_slot(
    monkeypatch, new_form, older_form, same_slot
):
    new = _responsibility(*new_form, LATER)
    older = _responsibility(*older_form, EARLIER)
    same_ends = {new.source_node_uuid, new.target_node_uuid} == {
        older.source_node_uuid,
        older.target_node_uuid,
    }

    ended = await _ended_in_slot(monkeypatch, new, older, {})

    assert ended == ([older.uuid] if same_ends else [])
