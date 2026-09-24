import re
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphiti_core import graphiti as graphiti_module
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.dedupe_nodes import NodeResolutions
from graphiti_core.prompts.extract_edges import EdgeTimestamps
from graphiti_core.search.search_config import SearchResults
from graphiti_core.utils.maintenance import edge_operations, node_operations
from graphiti_core.utils.maintenance.edge_operations import (
    resolve_extracted_edge,
    resolve_extracted_edges,
)
from graphiti_core.utils.maintenance.node_operations import resolve_extracted_nodes
from graphiti_core.utils.maintenance.resolution_decisions import (
    EdgeQuestion,
    EdgeVerdict,
    LlmResolutionDecisions,
    NodeQuestion,
)

EARLIER = datetime(2025, 1, 1, tzinfo=timezone.utc)
RECENT = datetime(2025, 6, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)
GROUP = 'group'


class Project(BaseModel):
    """An initiative with a goal and someone leading it."""


class Topic(BaseModel):
    """A subject people keep discussing."""


class Leads(BaseModel):
    """Someone leads an initiative."""

    role: str | None = None


ENTITY_TYPES: dict[str, type[BaseModel]] = {'Project': Project, 'Topic': Topic}
EDGE_TYPES: dict[str, type[BaseModel]] = {'LEADS': Leads}
EDGE_TYPE_MAP = {('Entity', 'Entity'): ['LEADS']}

SEMANTIC_HITS = {
    'Apollo Program': ['program'],
    'Apollo': ['program'],
    'Artemis': ['mission'],
    'Budget Review': ['budget'],
    'Zeta Initiative': [],
}
NODE_ANSWERS = [
    {'id': 0, 'name': 'Apollo', 'duplicate_candidate_id': 0},
    {'id': 1, 'name': 'Artemis', 'duplicate_candidate_id': 1},
    {'id': 2, 'name': 'Budget Review', 'duplicate_candidate_id': -1},
]

EDGES_BETWEEN = {
    ('ana', 'budget'): ['owns'],
    ('ana', 'review'): ['manages'],
    ('ana', 'note'): ['private'],
}
INVALIDATION_HITS = {
    'Luis now leads Apollo': ['marta-leads'],
    'Marta chaired the board in 2020': ['luis-chairs'],
    'Ana shares the note again': ['private'],
}
EDGE_ANSWERS = {
    'Ana is in charge of the budget review': {'duplicate_facts': [0], 'contradicted_facts': []},
    'Luis now leads Apollo': {'duplicate_facts': [], 'contradicted_facts': [0]},
    'Marta chaired the board in 2020': {'duplicate_facts': [], 'contradicted_facts': [0]},
}
TIMESTAMP_ANSWERS = {
    'Ana leads the Zeta Initiative': {'valid_at': '2026-02-01T00:00:00Z'},
    'Luis now leads Apollo': {'valid_at': '2026-02-15T00:00:00Z'},
    'Marta chaired the board in 2020': {'valid_at': '2020-01-01T00:00:00Z'},
}
ATTRIBUTE_ANSWERS = {
    'Ana leads the Zeta Initiative': {'role': 'sponsor'},
    'Luis now leads Apollo': {'role': 'lead'},
}


class Call(NamedTuple):
    prompt_name: str
    messages: tuple[tuple[str, str], ...]
    response_model: type[BaseModel] | None
    model_size: ModelSize
    group_id: str | None
    attribute_extraction: bool


def expected_call(
    prompt_name: str,
    context: dict[str, Any],
    response_model: type[BaseModel],
    model_size: ModelSize = ModelSize.small,
    attribute_extraction: bool = False,
) -> Call:
    family, version = prompt_name.split('.')
    messages = getattr(getattr(prompt_library, family), version)(context)
    return Call(
        prompt_name,
        tuple((message.role, message.content) for message in messages),
        response_model,
        model_size,
        None,
        attribute_extraction,
    )


def tagged(content: str, tag: str) -> str:
    match = re.search(f'<{tag}>\n(.*?)\n</{tag}>', content, re.DOTALL)
    assert match is not None, f'prompt without <{tag}>'
    return match.group(1)


class RecordingLlm:
    """Answers each prompt from tables keyed by what the prompt asks about; records every call."""

    def __init__(
        self,
        nodes: list[dict[str, Any]] | None = None,
        edges: dict[str, dict[str, Any]] | None = None,
        timestamps: dict[str, dict[str, Any]] | None = None,
        attributes: dict[str, dict[str, Any]] | None = None,
    ):
        self.answers: dict[str, Callable[[str], dict[str, Any]]] = {
            'dedupe_nodes.nodes': lambda content: {'entity_resolutions': nodes or []},
            'dedupe_edges.resolve_edge': lambda content: (edges or {})[tagged(content, 'NEW FACT')],
            'extract_edges.extract_timestamps': lambda content: (timestamps or {}).get(
                tagged(content, 'FACT'), {}
            ),
            'extract_edges.extract_attributes': lambda content: (attributes or {})[
                tagged(content, 'FACT')
            ],
        }
        self.calls: list[Call] = []

    async def generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=None,
        model_size=ModelSize.medium,
        group_id=None,
        prompt_name=None,
        *,
        attribute_extraction=False,
    ):
        self.calls.append(
            Call(
                prompt_name,
                tuple((message.role, message.content) for message in messages),
                response_model,
                model_size,
                group_id,
                attribute_extraction,
            )
        )
        return self.answers[prompt_name](messages[-1].content)


class ScriptedDecisions:
    """Answers from scripted verdicts and dates undated edges at the episode time.

    Records every question it is asked and every batch of edges it dates.
    """

    def __init__(
        self,
        nodes: dict[str, Callable[[NodeQuestion, list[NodeQuestion]], EntityNode | None]]
        | None = None,
        edges: dict[str, Callable[[EdgeQuestion], EdgeVerdict]] | None = None,
    ):
        self.nodes = nodes or {}
        self.edges = edges or {}
        self.node_questions: list[list[NodeQuestion]] = []
        self.edge_questions: list[list[EdgeQuestion]] = []
        self.dated: list[list[str]] = []

    async def resolve_nodes(self, episode, previous, questions):
        self.node_questions.append(questions)
        return [
            self.nodes.get(question.extracted.name, lambda *_: None)(question, questions)
            for question in questions
        ]

    async def resolve_edges(self, episode, questions):
        self.edge_questions.append(questions)
        return [
            self.edges.get(question.extracted.fact, lambda _: EdgeVerdict(None, ()))(question)
            for question in questions
        ]

    async def date_edges(self, episode, edges):
        self.dated.append([edge.uuid for edge in edges])
        for edge in edges:
            edge.valid_at = edge.valid_at or episode.valid_at


def clients(llm: RecordingLlm) -> GraphitiClients:
    return GraphitiClients.model_construct(
        driver=SimpleNamespace(provider=GraphProvider.NEO4J),
        llm_client=llm,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
    )


def episode() -> EpisodicNode:
    return EpisodicNode(
        uuid='episode',
        name='episode',
        group_id=GROUP,
        source=EpisodeType.message,
        source_description='chat',
        content='Luis now leads Apollo and Ana is in charge of the budget review.',
        valid_at=NOW,
        created_at=NOW,
    )


def previous_episodes() -> list[EpisodicNode]:
    return [
        EpisodicNode(
            uuid='earlier',
            name='earlier',
            group_id=GROUP,
            source=EpisodeType.message,
            source_description='chat',
            content='Marta leads Apollo.',
            valid_at=EARLIER,
            created_at=EARLIER,
        )
    ]


def entity(uuid: str, name: str, label: str, summary: str = '', **attributes: Any) -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=name,
        group_id=GROUP,
        labels=['Entity', label],
        summary=summary,
        attributes=attributes,
        created_at=EARLIER,
    )


def stored_entities() -> dict[str, EntityNode]:
    return {
        'ana': entity('ana', 'Ana', 'Person', 'Ana as stored.', user_id=7),
        'program': entity('program', 'Apollo Program', 'Project', 'Apollo Program as stored.'),
        'mission': entity('mission', 'Artemis Mission', 'Project', 'Artemis Mission as stored.'),
        'budget': entity('budget', 'Quarterly Budget', 'Topic', 'Quarterly Budget as stored.'),
    }


def extracted_entities() -> list[EntityNode]:
    return [
        entity('ana', 'Ana', 'Person', user_id=7),
        entity('x-program', 'Apollo Program', 'Project'),
        entity('x-apollo', 'Apollo', 'Project'),
        entity('x-artemis', 'Artemis', 'Project'),
        entity('x-review', 'Budget Review', 'Topic'),
        entity('x-zeta', 'Zeta Initiative', 'Project'),
    ]


def relation(
    uuid: str,
    source: str,
    target: str,
    name: str,
    fact: str,
    valid_at: datetime | None = None,
    episode_uuid: str = 'episode',
    **attributes: Any,
) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id=GROUP,
        source_node_uuid=source,
        target_node_uuid=target,
        name=name,
        fact=fact,
        episodes=[episode_uuid],
        created_at=EARLIER,
        valid_at=valid_at,
        attributes={'provenance': [episode_uuid], **attributes},
    )


def stored_relations() -> dict[str, EntityEdge]:
    return {
        edge.uuid: edge
        for edge in [
            relation('owns', 'ana', 'budget', 'OWNS', 'Ana owns the budget', EARLIER, 'earlier'),
            relation(
                'manages',
                'ana',
                'review',
                'MANAGES',
                'Ana manages the budget review',
                EARLIER,
                'earlier',
                note='weekly',
            ),
            relation(
                'marta-leads', 'marta', 'apollo', 'LEADS', 'Marta leads Apollo', EARLIER, 'earlier'
            ),
            relation(
                'luis-chairs', 'luis', 'board', 'CHAIRS', 'Luis chairs the board', RECENT, 'earlier'
            ),
            relation(
                'private',
                'ana',
                'note',
                'SHARES',
                'Ana shares the note',
                EARLIER,
                'earlier',
                owner_user_id=8,
            ),
        ]
    }


def extracted_relations() -> list[EntityEdge]:
    return [
        relation('new', 'ana', 'zeta', 'LEADS', 'Ana leads the Zeta Initiative'),
        relation('repeat', 'ana', 'zeta', 'LEADS', '  ana LEADS the zeta initiative '),
        relation('literal', 'ana', 'budget', 'OWNS', ' ana owns the BUDGET '),
        relation('paraphrase', 'ana', 'review', 'MANAGES', 'Ana is in charge of the budget review'),
        relation('update', 'luis', 'apollo', 'LEADS', 'Luis now leads Apollo'),
        relation('backfill', 'marta', 'board', 'CHAIRED', 'Marta chaired the board in 2020'),
        relation(
            'scoped', 'ana', 'note', 'SHARES', 'Ana shares the note again', NOW, owner_user_id=7
        ),
    ]


def endpoints() -> list[EntityNode]:
    return [
        entity(uuid, uuid.title(), 'Entity')
        for uuid in ('ana', 'zeta', 'budget', 'review', 'luis', 'apollo', 'marta', 'board', 'note')
    ]


async def resolve_node_fixture(monkeypatch, llm: RecordingLlm, **options):
    stored = stored_entities()

    async def get_by_uuids(driver, uuids, group_id=None):
        return [stored[uuid].model_copy(deep=True) for uuid in uuids if uuid in stored]

    async def semantic_search(clients, nodes):
        return [
            [stored[uuid].model_copy(deep=True) for uuid in SEMANTIC_HITS[node.name]]
            for node in nodes
        ]

    monkeypatch.setattr(EntityNode, 'get_by_uuids', get_by_uuids)
    monkeypatch.setattr(node_operations, '_semantic_candidate_search', semantic_search)
    return await resolve_extracted_nodes(
        clients(llm), extracted_entities(), episode(), previous_episodes(), ENTITY_TYPES, **options
    )


async def resolve_edge_fixture(monkeypatch, llm: RecordingLlm, **options):
    stored = stored_relations()

    async def get_between_nodes(driver, source_node_uuid, target_node_uuid):
        pair = (source_node_uuid, target_node_uuid)
        return [stored[uuid].model_copy(deep=True) for uuid in EDGES_BETWEEN.get(pair, [])]

    async def search(clients, query, group_ids=None, config=None, search_filter=None):
        uuids = (
            search_filter.edge_uuids
            if search_filter is not None and search_filter.edge_uuids is not None
            else INVALIDATION_HITS.get(query, [])
        )
        return SearchResults(edges=[stored[uuid].model_copy(deep=True) for uuid in uuids])

    monkeypatch.setattr(EntityEdge, 'get_between_nodes', get_between_nodes)
    monkeypatch.setattr(edge_operations, 'search', search)
    monkeypatch.setattr(
        edge_operations, 'create_entity_edge_embeddings', AsyncMock(return_value=None)
    )
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)
    return await resolve_extracted_edges(
        clients(llm),
        extracted_relations(),
        episode(),
        endpoints(),
        EDGE_TYPES,
        EDGE_TYPE_MAP,
        **options,
    )


def todays_node_prompt() -> Call:
    return expected_call(
        'dedupe_nodes.nodes',
        {
            'extracted_nodes': [
                {
                    'id': 0,
                    'name': 'Apollo',
                    'entity_type': ['Entity', 'Project'],
                    'entity_type_description': Project.__doc__,
                },
                {
                    'id': 1,
                    'name': 'Artemis',
                    'entity_type': ['Entity', 'Project'],
                    'entity_type_description': Project.__doc__,
                },
                {
                    'id': 2,
                    'name': 'Budget Review',
                    'entity_type': ['Entity', 'Topic'],
                    'entity_type_description': Topic.__doc__,
                },
            ],
            'existing_nodes': [
                {
                    'candidate_id': 0,
                    'name': 'Apollo Program',
                    'entity_types': ['Entity', 'Project'],
                    'summary': 'Apollo Program as stored.',
                },
                {
                    'candidate_id': 1,
                    'name': 'Artemis Mission',
                    'entity_types': ['Entity', 'Project'],
                    'summary': 'Artemis Mission as stored.',
                },
                {
                    'candidate_id': 2,
                    'name': 'Quarterly Budget',
                    'entity_types': ['Entity', 'Topic'],
                    'summary': 'Quarterly Budget as stored.',
                },
            ],
            'episode_content': episode().content,
            'previous_episodes': [
                {'content': 'Marta leads Apollo.', 'timestamp': EARLIER.isoformat()}
            ],
        },
        NodeResolutions,
        ModelSize.medium,
    )


def todays_edge_prompts() -> Counter[Call]:
    def resolve_edge(new_fact: str, duplicates: list[str], invalidations: list[str]) -> Call:
        return expected_call(
            'dedupe_edges.resolve_edge',
            {
                'existing_edges': [{'idx': i, 'fact': fact} for i, fact in enumerate(duplicates)],
                'new_edge': new_fact,
                'edge_invalidation_candidates': [
                    {'idx': len(duplicates) + i, 'fact': fact}
                    for i, fact in enumerate(invalidations)
                ],
            },
            EdgeDuplicate,
        )

    return Counter(
        [
            resolve_edge(
                'Ana is in charge of the budget review', ['Ana manages the budget review'], []
            ),
            resolve_edge('Luis now leads Apollo', [], ['Marta leads Apollo']),
            resolve_edge('Marta chaired the board in 2020', [], ['Luis chairs the board']),
            *[
                expected_call(
                    'extract_edges.extract_attributes',
                    {
                        'fact': fact,
                        'reference_time': NOW,
                        'existing_attributes': {},
                    },
                    Leads,
                    attribute_extraction=True,
                )
                for fact in ATTRIBUTE_ANSWERS
            ],
            *[
                expected_call(
                    'extract_edges.extract_timestamps',
                    {'fact': fact, 'reference_time': NOW.isoformat()},
                    EdgeTimestamps,
                )
                for fact in TIMESTAMP_ANSWERS
            ],
        ]
    )


def dated(edge: EntityEdge) -> tuple:
    return (
        edge.uuid,
        edge.valid_at,
        edge.invalid_at,
        edge.expired_at,
        edge.episodes,
        edge.attributes,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('explicit_default', [False, True])
async def test_default_node_resolution_sends_todays_single_prompt_and_keeps_its_result(
    monkeypatch, explicit_default
):
    llm = RecordingLlm(nodes=NODE_ANSWERS)
    decisions = LlmResolutionDecisions(llm, ENTITY_TYPES) if explicit_default else None

    resolved, uuid_map, duplicates = await resolve_node_fixture(
        monkeypatch, llm, decisions=decisions
    )

    assert llm.calls == [todays_node_prompt()]
    assert [node.uuid for node in resolved] == [
        'ana',
        'program',
        'program',
        'mission',
        'x-review',
        'x-zeta',
    ]
    assert uuid_map == {
        'ana': 'ana',
        'x-program': 'program',
        'x-apollo': 'program',
        'x-artemis': 'mission',
        'x-review': 'x-review',
        'x-zeta': 'x-zeta',
    }
    assert [(extracted.uuid, resolved.uuid) for extracted, resolved in duplicates] == [
        ('x-program', 'program'),
        ('x-apollo', 'program'),
        ('x-artemis', 'mission'),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('explicit_default', [False, True])
async def test_default_edge_resolution_sends_todays_prompts_and_keeps_its_result(
    monkeypatch, explicit_default
):
    llm = RecordingLlm(
        edges=EDGE_ANSWERS, timestamps=TIMESTAMP_ANSWERS, attributes=ATTRIBUTE_ANSWERS
    )
    decisions = LlmResolutionDecisions(llm) if explicit_default else None

    resolved, invalidated, new = await resolve_edge_fixture(monkeypatch, llm, decisions=decisions)

    assert Counter(llm.calls) == todays_edge_prompts()
    assert [dated(edge) for edge in resolved] == [
        (
            'new',
            datetime(2026, 2, 1, tzinfo=timezone.utc),
            None,
            None,
            ['episode'],
            {'role': 'sponsor', 'provenance': ['episode']},
        ),
        (
            'owns',
            EARLIER,
            None,
            None,
            ['earlier', 'episode'],
            {'provenance': ['earlier', 'episode']},
        ),
        (
            'manages',
            EARLIER,
            None,
            None,
            ['earlier', 'episode'],
            {'provenance': ['earlier', 'episode']},
        ),
        (
            'update',
            datetime(2026, 2, 15, tzinfo=timezone.utc),
            None,
            None,
            ['episode'],
            {'role': 'lead', 'provenance': ['episode']},
        ),
        (
            'backfill',
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            RECENT,
            NOW,
            ['episode'],
            {'provenance': ['episode']},
        ),
        ('scoped', NOW, None, None, ['episode'], {'provenance': ['episode'], 'owner_user_id': 7}),
    ]
    assert [(edge.uuid, edge.invalid_at, edge.expired_at) for edge in invalidated] == [
        ('marta-leads', datetime(2026, 2, 15, tzinfo=timezone.utc), NOW)
    ]
    assert [edge.uuid for edge in new] == ['new', 'update', 'backfill', 'scoped']


@pytest.mark.asyncio
async def test_injected_decisions_replace_the_dedupe_and_timestamp_prompts(monkeypatch):
    llm = RecordingLlm(attributes=ATTRIBUTE_ANSWERS)
    decisions = ScriptedDecisions(
        nodes={'Apollo': lambda question, _: question.candidates[0]},
        edges={
            'Ana is in charge of the budget review': lambda question: EdgeVerdict(
                question.duplicates[0], ()
            ),
            'Luis now leads Apollo': lambda question: EdgeVerdict(None, question.invalidations),
            'Marta chaired the board in 2020': lambda question: EdgeVerdict(
                None, question.invalidations
            ),
        },
    )

    _, uuid_map, _ = await resolve_node_fixture(monkeypatch, llm, decisions=decisions)
    resolved, invalidated, new = await resolve_edge_fixture(monkeypatch, llm, decisions=decisions)

    assert {call.prompt_name for call in llm.calls} == {'extract_edges.extract_attributes'}
    assert [
        [question.extracted.uuid for question in batch] for batch in decisions.node_questions
    ] == [['x-apollo', 'x-artemis', 'x-review']]
    assert [
        [
            (
                question.extracted.uuid,
                [edge.uuid for edge in question.duplicates],
                [edge.uuid for edge in question.invalidations],
            )
            for question in batch
        ]
        for batch in decisions.edge_questions
    ] == [
        [
            ('paraphrase', ['manages'], []),
            ('update', [], ['marta-leads']),
            ('backfill', [], ['luis-chairs']),
        ]
    ]
    assert decisions.dated == [['new', 'update', 'backfill', 'scoped']]
    assert uuid_map['x-apollo'] == 'program'
    assert uuid_map['x-artemis'] == 'x-artemis'
    assert [edge.uuid for edge in resolved] == [
        'new',
        'owns',
        'manages',
        'update',
        'backfill',
        'scoped',
    ]
    assert [(edge.uuid, edge.invalid_at, edge.expired_at) for edge in invalidated] == [
        ('marta-leads', NOW, NOW),
        ('luis-chairs', NOW, NOW),
    ]
    assert [edge.uuid for edge in new] == ['new', 'update', 'backfill', 'scoped']


@pytest.mark.asyncio
async def test_node_verdicts_count_only_for_offered_candidates_of_a_compatible_identity(
    monkeypatch,
):
    """A verdict may name any candidate offered in the call, as the LLM could.

    A node nobody offered, or one of another identity, leaves the extracted node new.
    """
    stranger = entity('stranger', 'Apollo Moon', 'Project')
    decisions = ScriptedDecisions(
        nodes={
            'Apollo': lambda question, _: stranger,
            'Artemis': lambda question, questions: questions[0].candidates[0],
            'Budget Review': lambda question, questions: questions[0].candidates[0],
        }
    )

    resolved, uuid_map, duplicates = await resolve_node_fixture(
        monkeypatch, RecordingLlm(), decisions=decisions
    )

    assert uuid_map['x-apollo'] == 'x-apollo'
    assert uuid_map['x-artemis'] == 'program'
    assert uuid_map['x-review'] == 'x-review'
    assert 'stranger' not in [node.uuid for node in resolved]
    assert [(extracted.uuid, resolved.uuid) for extracted, resolved in duplicates] == [
        ('x-program', 'program'),
        ('x-artemis', 'program'),
    ]


@pytest.mark.asyncio
async def test_edge_verdicts_ignore_duplicates_and_contradictions_the_question_did_not_offer(
    monkeypatch,
):
    stranger = relation(
        'stranger', 'ana', 'review', 'MANAGES', 'Ana manages the budget review', EARLIER, 'earlier'
    )
    decisions = ScriptedDecisions(
        edges={
            'Ana is in charge of the budget review': lambda question: EdgeVerdict(
                stranger, (stranger,)
            ),
            'Luis now leads Apollo': lambda question: EdgeVerdict(question.invalidations[0], ()),
        }
    )

    resolved, invalidated, new = await resolve_edge_fixture(
        monkeypatch, RecordingLlm(attributes=ATTRIBUTE_ANSWERS), decisions=decisions
    )

    assert [edge.uuid for edge in resolved] == [
        'new',
        'owns',
        'paraphrase',
        'update',
        'backfill',
        'scoped',
    ]
    assert invalidated == []
    assert [edge.uuid for edge in new] == ['new', 'paraphrase', 'update', 'backfill', 'scoped']
    assert decisions.dated == [['new', 'paraphrase', 'update', 'backfill', 'scoped']]
    assert (stranger.episodes, stranger.invalid_at) == (['earlier'], None)


@pytest.mark.asyncio
async def test_resolve_extracted_edge_still_composes_question_verdict_and_application(monkeypatch):
    """add_triplet resolves one edge against two overlapping searches.

    Every duplicate the LLM names still comes back, as before the split into phases.
    """
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)
    heads = relation('heads', 'luis', 'apollo', 'LEADS', 'Luis heads Apollo')
    leads = relation(
        'luis-leads', 'luis', 'apollo', 'LEADS', 'Luis leads Apollo', RECENT, 'earlier'
    )
    runs = relation('luis-runs', 'luis', 'apollo', 'LEADS', 'Luis runs Apollo', RECENT, 'earlier')
    marta = relation(
        'marta-leads', 'marta', 'apollo', 'LEADS', 'Marta leads Apollo', EARLIER, 'earlier'
    )
    llm = RecordingLlm(
        edges={'Luis heads Apollo': {'duplicate_facts': [0, 1], 'contradicted_facts': [3]}}
    )

    resolved, invalidated, duplicates = await resolve_extracted_edge(
        llm, heads, [leads, runs], [leads.model_copy(deep=True), marta], episode(), None
    )

    assert llm.calls == [
        expected_call(
            'dedupe_edges.resolve_edge',
            {
                'existing_edges': [
                    {'idx': 0, 'fact': 'Luis leads Apollo'},
                    {'idx': 1, 'fact': 'Luis runs Apollo'},
                ],
                'new_edge': 'Luis heads Apollo',
                'edge_invalidation_candidates': [
                    {'idx': 2, 'fact': 'Luis leads Apollo'},
                    {'idx': 3, 'fact': 'Marta leads Apollo'},
                ],
            },
            EdgeDuplicate,
        )
    ]
    assert resolved is leads
    assert duplicates == [leads, runs]
    assert leads.episodes == ['earlier', 'episode']
    assert invalidated == [marta]
    assert (marta.invalid_at, marta.expired_at) == (RECENT, NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize('injected', [False, True])
async def test_resolve_and_hydrate_hands_the_decisions_to_node_and_edge_resolution(
    monkeypatch, injected
):
    decisions = ScriptedDecisions() if injected else None
    received: dict[str, Any] = {}

    async def resolve_nodes(*args, **kwargs):
        received['nodes'] = kwargs['decisions']
        return [], {}, []

    async def resolve_edges(*args, **kwargs):
        received['edges'] = kwargs['decisions']
        return [], [], []

    monkeypatch.setattr(graphiti_module, 'resolve_extracted_nodes', resolve_nodes)
    monkeypatch.setattr(graphiti_module, 'resolve_extracted_edges', resolve_edges)
    monkeypatch.setattr(
        graphiti_module, 'extract_attributes_from_nodes', AsyncMock(return_value=[])
    )

    await Graphiti.resolve_and_hydrate_nodes_edges(
        SimpleNamespace(clients=clients(RecordingLlm())),
        group_id=GROUP,
        episode=episode(),
        extracted_nodes=[],
        extracted_edges=[],
        decisions=decisions,
    )

    assert received == {'nodes': decisions, 'edges': decisions}
