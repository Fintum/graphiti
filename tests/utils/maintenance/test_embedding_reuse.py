import hashlib
import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search import search as search_module
from graphiti_core.utils.maintenance import edge_operations, node_operations

EARLIER = datetime(2025, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)
GROUP = 'group'
# (prompt_name, messages) and outcome of the episode on 5202f54, before any vector was reused.
PROMPTS_SHA = 'e588d80353cad8ae034517b0334a692498aceeb32780f50547b78bc9cc50ef7a'
OUTCOME_SHA = '91a5865349a5e40092902b9f809916e7549d9468ea3ec24970c66fcaa82e434b'

STORED_NODES = {'ana': 'Ana', 'marta': 'Marta', 'program': 'Apollo Program'}
STORED_EDGES = {
    'owns': ('ana', 'program', 'OWNS', 'Ana owns the Apollo Program'),
    'marta-leads': ('marta', 'program', 'LEADS', 'Marta leads the Apollo Program'),
}
STORED_FACTS = {uuid: stored[-1] for uuid, stored in STORED_EDGES.items()}
EXTRACTED_NODES = [
    ('ana', 'Ana'), ('x-apollo', 'Apollo'), ('x-zeta', 'Zeta Initiative'), ('luis', 'Luis')
]
EXTRACTED_EDGES = [
    ('again', 'ana', 'x-apollo', 'OWNS', 'Ana owns the Apollo Program'),
    ('owner', 'ana', 'x-apollo', 'OWNS', 'Ana is the owner of the Apollo Program'),
    ('update', 'luis', 'x-apollo', 'LEADS', 'Luis now leads Apollo'),
    ('new', 'ana', 'x-zeta', 'LEADS', 'Ana leads the Zeta Initiative'),
]
EXTRACTED_FACTS = [extracted[-1] for extracted in EXTRACTED_EDGES]
SEMANTIC_HITS = {'Apollo': ['program']}
INVALIDATION_HITS = {'Luis now leads Apollo': ['marta-leads']}
NODE_VERDICTS = [{'id': 0, 'name': 'Apollo', 'duplicate_candidate_id': 0}]
EDGE_VERDICTS = {
    'Ana is the owner of the Apollo Program': {'duplicate_facts': [0], 'contradicted_facts': []},
    'Luis now leads Apollo': {'duplicate_facts': [], 'contradicted_facts': [0]},
}
DATES = {'Luis now leads Apollo': {'valid_at': '2026-02-15T00:00:00Z'}}


def vector_of(text: str) -> list[float]:
    return [byte / 255 for byte in hashlib.sha256(text.encode()).digest()[:4]]


class CountingEmbedder:
    """Gives every text a vector of its own and records the texts of each call it receives."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def create(self, input_data):
        self.calls.append(list(input_data))
        return vector_of(input_data[0])

    async def create_batch(self, input_data_list):
        self.calls.append(list(input_data_list))
        return [vector_of(text) for text in input_data_list]


class StoredGraph:
    """Answers the vector loads with the stored texts, whose vectors are theirs unless legacy."""

    provider = GraphProvider.NEO4J

    def __init__(self, legacy: frozenset[str] = frozenset()) -> None:
        self.legacy = legacy
        self.queries: list[str] = []

    async def execute_query(self, query, **params):
        self.queries.append(query)
        texts = STORED_FACTS if 'RELATES_TO' in query else STORED_NODES
        vectors = {uuid: vector_of(text) for uuid, text in texts.items() if uuid not in self.legacy}
        rows = [
            {'uuid': uuid, 'text': texts[uuid], 'vector': vectors.get(uuid)}
            for uuid in params['uuids']
            if uuid in texts
        ]
        return rows, None, None


def tagged(content: str, tag: str) -> str:
    [found] = re.findall(f'<{tag}>\n(.*?)\n</{tag}>', content, re.DOTALL)
    return found


class RecordingLlm:
    """Answers the resolution prompts of the episode from the tables above and records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    async def generate_response(self, messages, prompt_name=None, **options):
        content = messages[-1].content
        self.calls.append((prompt_name, tuple((item.role, item.content) for item in messages)))
        if prompt_name == 'dedupe_nodes.nodes':
            return {'entity_resolutions': NODE_VERDICTS}
        if prompt_name == 'dedupe_edges.resolve_edge':
            return dict(EDGE_VERDICTS[tagged(content, 'NEW FACT')])
        if prompt_name == 'extract_edges.extract_timestamps':
            return dict(DATES.get(tagged(content, 'FACT'), {}))
        return {'summaries': []}


class Hydrated(NamedTuple):
    nodes: list[EntityNode]
    edges: list[EntityEdge]
    invalidated: list[EntityEdge]
    uuid_map: dict[str, str]
    llm: RecordingLlm
    embedder: CountingEmbedder
    searched: list[tuple[str, list[float] | None]]


def node(uuid: str, name: str) -> EntityNode:
    return EntityNode(uuid=uuid, name=name, group_id=GROUP, labels=['Entity'], created_at=EARLIER)


def edge(uuid: str, source: str, target: str, name: str, fact: str, stored=False) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id=GROUP,
        source_node_uuid=source,
        target_node_uuid=target,
        name=name,
        fact=fact,
        episodes=['earlier' if stored else 'episode'],
        created_at=EARLIER if stored else NOW,
        valid_at=EARLIER if stored else None,
    )


def stored_edge(uuid: str) -> EntityEdge:
    return edge(uuid, *STORED_EDGES[uuid], stored=True)


def episode(uuid: str, content: str, told_at: datetime) -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid,
        name=uuid,
        group_id=GROUP,
        source=EpisodeType.message,
        source_description='chat',
        content=content,
        valid_at=told_at,
        created_at=told_at,
    )


def install(monkeypatch, searched: list[tuple[str, list[float] | None]]) -> None:
    hits = {tuple(vector_of(name)): uuids for name, uuids in SEMANTIC_HITS.items()}
    searching = edge_operations.search

    async def get_by_uuids(driver, uuids, group_id=None):
        return [node(uuid, STORED_NODES[uuid]) for uuid in uuids if uuid in STORED_NODES]

    async def similar_nodes(driver, search_vector, search_filter, group_ids=None, *limits):
        return [node(uuid, STORED_NODES[uuid]) for uuid in hits.get(tuple(search_vector), [])]

    async def get_between_nodes(driver, source_node_uuid, target_node_uuid):
        pair = (source_node_uuid, target_node_uuid)
        return [stored_edge(uuid) for uuid, stored in STORED_EDGES.items() if stored[:2] == pair]

    async def edge_search(driver, encoder, query, vector, group_ids, config, search_filter, *rest):
        uuids = search_filter.edge_uuids
        if uuids is None:
            uuids = INVALIDATION_HITS.get(query, [])
        found = [stored_edge(uuid) for uuid in uuids]
        return found, [1.0] * len(found)

    async def recorded_search(clients, query, *args, **kwargs):
        searched.append((query, kwargs.get('query_vector')))
        return await searching(clients, query, *args, **kwargs)

    monkeypatch.setattr(EntityNode, 'get_by_uuids', get_by_uuids)
    monkeypatch.setattr(node_operations, 'node_similarity_search', similar_nodes)
    monkeypatch.setattr(EntityEdge, 'get_between_nodes', get_between_nodes)
    monkeypatch.setattr(search_module, 'edge_search', edge_search)
    monkeypatch.setattr(edge_operations, 'search', recorded_search)
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)


async def hydrate(monkeypatch, graph: StoredGraph | None = None, nodes=(), edges=()) -> Hydrated:
    llm, embedder, searched = RecordingLlm(), CountingEmbedder(), []
    install(monkeypatch, searched)
    driver = graph or StoredGraph()
    clients = SimpleNamespace(driver=driver, llm_client=llm, embedder=embedder, cross_encoder=None)
    result = await Graphiti.resolve_and_hydrate_nodes_edges(
        SimpleNamespace(clients=clients),
        group_id=GROUP,
        episode=episode('episode', 'Luis now leads Apollo; Ana owns it.', NOW),
        extracted_nodes=[node(*extracted) for extracted in [*EXTRACTED_NODES, *nodes]],
        extracted_edges=[edge(*extracted) for extracted in [*EXTRACTED_EDGES, *edges]],
        previous_episodes=[episode('earlier', 'Marta set up the Apollo Program.', EARLIER)],
    )
    return Hydrated(*result, llm, embedder, searched)


def outcome(hydrated: Hydrated) -> list[Any]:
    return [
        [(item.uuid, item.labels, item.summary, item.attributes) for item in hydrated.nodes],
        [
            (item.uuid, item.fact, item.episodes, item.valid_at, item.invalid_at, item.expired_at)
            for item in [*hydrated.edges, *hydrated.invalidated]
        ],
        hydrated.uuid_map,
    ]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, default=str, ensure_ascii=False).encode()).hexdigest()


@pytest.mark.asyncio
async def test_an_episode_embeds_unstored_names_in_one_batch_and_facts_in_another(monkeypatch):
    hydrated = await hydrate(monkeypatch)

    assert hydrated.embedder.calls == [['Apollo', 'Zeta Initiative', 'Luis'], EXTRACTED_FACTS]


@pytest.mark.asyncio
async def test_every_node_and_fact_keeps_the_vector_of_its_own_text(monkeypatch):
    hydrated = await hydrate(monkeypatch)

    nodes, edges = hydrated.nodes, [*hydrated.edges, *hydrated.invalidated]
    assert [item.uuid for item in nodes] == ['ana', 'program', 'x-zeta', 'luis']
    assert [item.uuid for item in edges] == ['owns', 'owns', 'update', 'new', 'marta-leads']
    assert [item.name_embedding for item in nodes] == [vector_of(item.name) for item in nodes]
    assert [item.fact_embedding for item in edges] == [vector_of(item.fact) for item in edges]


@pytest.mark.asyncio
async def test_every_search_for_a_fact_is_given_the_vector_of_that_fact(monkeypatch):
    hydrated = await hydrate(monkeypatch)

    queries = [query for query, _ in hydrated.searched]
    assert sorted(queries) == sorted([*EXTRACTED_FACTS, *EXTRACTED_FACTS[:2]])
    assert [query for query, vector in hydrated.searched if vector != vector_of(query)] == []


@pytest.mark.asyncio
async def test_reusing_the_vectors_changes_neither_the_llm_calls_nor_the_outcome(monkeypatch):
    hydrated = await hydrate(monkeypatch)

    assert digest([[name, messages] for name, messages in hydrated.llm.calls]) == PROMPTS_SHA
    assert digest(outcome(hydrated)) == OUTCOME_SHA


@pytest.mark.asyncio
async def test_a_stored_node_without_a_vector_costs_one_more_batch_with_only_its_name(monkeypatch):
    hydrated = await hydrate(monkeypatch, StoredGraph(legacy=frozenset({'program'})))

    assert hydrated.embedder.calls[2:] == [['Apollo Program']]
    assert hydrated.nodes[1].name_embedding == vector_of('Apollo Program')


@pytest.mark.asyncio
async def test_a_name_and_a_fact_with_a_newline_are_embedded_with_todays_texts(monkeypatch):
    notes = ('x-notes', 'Kickoff\nNotes')
    wrote = ('notes', 'ana', 'x-notes', 'WROTE', 'Ana wrote the\nkickoff notes')

    hydrated = await hydrate(monkeypatch, nodes=[notes], edges=[wrote])

    assert hydrated.embedder.calls == [
        ['Apollo', 'Zeta Initiative', 'Luis', 'Kickoff Notes'],
        [*EXTRACTED_FACTS, wrote[-1]],
        ['Ana wrote the kickoff notes'],
        [notes[-1]],
    ]
    assert hydrated.nodes[-1].name_embedding == vector_of(notes[-1])
    assert hydrated.edges[-1].fact_embedding == vector_of(wrote[-1])


@pytest.mark.asyncio
async def test_a_stored_vector_is_reused_only_for_the_text_it_was_stored_with():
    from graphiti_core.utils.maintenance import embedding_reuse

    graph, embedded = StoredGraph(), node('marta', 'Marta')
    embedded.name_embedding = [0.5]
    renamed, same = node('ana', 'Ana Ruiz'), node('program', 'Apollo Program')
    restated, kept = edge('owns', 'ana', 'program', 'OWNS', 'Ana owns Apollo'), stored_edge('owns')

    await embedding_reuse.load_name_embeddings(graph, [renamed, same, embedded])
    await embedding_reuse.load_fact_embeddings(graph, [restated, kept])
    await embedding_reuse.load_name_embeddings(graph, [same, embedded])

    names = [item.name_embedding for item in (renamed, same, embedded)]
    assert names == [None, vector_of('Apollo Program'), [0.5]]
    assert [restated.fact_embedding, kept.fact_embedding] == [None, vector_of(kept.fact)]
    assert len(graph.queries) == 2
