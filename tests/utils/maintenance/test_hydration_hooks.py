import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ValidationError

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_config import SearchResults
from graphiti_core.utils.maintenance import edge_operations, node_operations
from graphiti_core.utils.maintenance.hydration_hooks import (
    LlmEdgeAttributes,
    LlmNodeAttributes,
    _node_changes,
)
from graphiti_core.utils.maintenance.identity_constraints import (
    PROTECTED_ATTRIBUTES,
    protected_attributes,
)

EARLIER = datetime(2025, 1, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)
GROUP = 'group'
# (prompt_name, messages) and outcome of the episode without protected attributes, taken on 9e82c2c.
PROMPTS_SHA = 'cf3ff50774074d1b719792d3c01e1269a3996a258461739f10b9981603173ef1'
OUTCOME_SHA = '931f20a02e3fee53bd9f05e1731e4dec63584e5f023cf061df5339d511b63b0c'


class Person(BaseModel):
    """Someone taking part in the conversation."""

    role: str | None = None


class Project(BaseModel):
    """An initiative with a goal and someone leading it."""

    status: str | None = None
    department: str | None = None


class Topic(BaseModel):
    """A subject people keep discussing."""


class Document(BaseModel):
    """The record an episode comes from."""

    source: str


class Leads(BaseModel):
    """Someone leads an initiative."""

    role: str | None = None


class Owns(BaseModel):
    """Someone owns an initiative."""


ENTITY_TYPES: dict[str, type[BaseModel]] = {
    'Person': Person,
    'Project': Project,
    'Topic': Topic,
    'Document': Document,
}
EDGE_TYPES: dict[str, type[BaseModel]] = {'LEADS': Leads, 'OWNS': Owns}
EDGE_TYPE_MAP = {('Entity', 'Entity'): ['LEADS', 'OWNS']}

STORED_MARKS: dict[str, Any] = {
    'provenance': ['earlier'],
    'evidence_refs': ['earlier#1'],
    'source_key': 'gmail:inbox',
    'revision_id': 'revision-1',
    'policy_version': 1,
    'last_activity_at': EARLIER.isoformat(),
    'deleted_at': None,
    'org_public': False,
    'directory': False,
    'owner_user_id': 7,
    'acl_key': 'gmail:inbox',
    'task_id': None,
    'organization_id': None,
    'identity_key': None,
}
EXTRACTED_MARKS = {
    **STORED_MARKS,
    'provenance': ['episode'],
    'evidence_refs': ['episode#1'],
    'revision_id': 'revision-2',
    'last_activity_at': NOW.isoformat(),
}
MERGED_MARKS = {
    **STORED_MARKS,
    'provenance': ['earlier', 'episode'],
    'evidence_refs': ['earlier#1', 'episode#1'],
}
LONG_SUMMARY = ' '.join(['Apollo Program keeps a long record.'] * 60)

SEMANTIC_HITS = {'Apollo': ['program'], 'Budget Review': ['budget'], 'Quarterly Budget': ['budget']}
EDGES_BETWEEN = {
    ('ana', 'program'): ['owns'],
    ('marta', 'program'): ['marta-leads'],
    ('ana', 'budget'): ['ana-chairs'],
}
INVALIDATION_HITS = {'Luis now leads Apollo': ['marta-leads']}

NODE_VERDICTS = [
    {'id': 0, 'name': 'Apollo', 'duplicate_candidate_id': 0},
    {'id': 1, 'name': 'Budget Review', 'duplicate_candidate_id': -1},
]
EDGE_VERDICTS = {
    'Ana is the owner of the Apollo Program': {'duplicate_facts': [0], 'contradicted_facts': []},
    'Luis now leads Apollo': {'duplicate_facts': [], 'contradicted_facts': [0]},
}
DATES = {
    'Luis now leads Apollo': {'valid_at': '2026-02-15T00:00:00Z'},
    'Ana leads the Zeta Initiative': {'valid_at': '2026-02-01T00:00:00Z'},
}
NODE_ATTRIBUTES = {
    'Ana': {'role': 'finance lead'},
    'Marta': {'role': 'founder'},
    'Apollo Program': {'status': 'active'},
    'Zeta Initiative': {'status': 'planned'},
    'Luis': {'role': 'lead'},
    'Kickoff Notes': {'source': 'meeting'},
}
EDGE_ATTRIBUTES = {
    'Luis now leads Apollo': {'role': 'lead'},
    'Ana leads the Zeta Initiative': {'role': 'sponsor'},
}
SUMMARIES = [
    {'name': 'Apollo Program', 'summary': 'Apollo Program, now led by Luis.'},
    {'name': 'Kickoff Notes', 'summary': 'Notes of the Apollo kickoff.'},
]
TODAYS_CALLS = Counter(
    {
        ('dedupe_nodes.nodes', 'NodeResolutions', ModelSize.medium, None, False): 1,
        ('dedupe_edges.resolve_edge', 'EdgeDuplicate', ModelSize.small, None, False): 2,
        ('extract_edges.extract_attributes', 'Leads', ModelSize.small, None, True): 2,
        ('extract_edges.extract_timestamps', 'EdgeTimestamps', ModelSize.small, None, False): 3,
        ('extract_nodes.extract_attributes', 'Person', ModelSize.small, GROUP, True): 3,
        ('extract_nodes.extract_attributes', 'Project', ModelSize.small, GROUP, True): 2,
        ('extract_nodes.extract_attributes', 'Document', ModelSize.small, GROUP, True): 1,
        (
            'extract_nodes.extract_summaries_batch',
            'SummarizedEntities',
            ModelSize.small,
            GROUP,
            False,
        ): 1,
    }
)


class World(NamedTuple):
    stored_nodes: dict[str, EntityNode]
    stored_edges: dict[str, EntityEdge]
    nodes: list[EntityNode]
    edges: list[EntityEdge]


class Call(NamedTuple):
    prompt_name: str
    messages: tuple[tuple[str, str], ...]
    response_model: str
    model_size: ModelSize
    group_id: str | None
    attribute_extraction: bool


def tagged(content: str, tag: str) -> str:
    match = re.search(f'<{tag}>\n(.*?)\n</{tag}>', content, re.DOTALL)
    assert match is not None, f'prompt without <{tag}>'
    return match.group(1)


class RecordingLlm:
    """Answers every prompt of the episode from the tables above and records each call."""

    def __init__(self) -> None:
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
        content = messages[-1].content
        self.calls.append(
            Call(
                prompt_name,
                tuple((message.role, message.content) for message in messages),
                response_model.__name__,
                model_size,
                group_id,
                attribute_extraction,
            )
        )
        if prompt_name == 'dedupe_nodes.nodes':
            return {'entity_resolutions': NODE_VERDICTS}
        if prompt_name == 'dedupe_edges.resolve_edge':
            return dict(EDGE_VERDICTS[tagged(content, 'NEW FACT')])
        if prompt_name == 'extract_edges.extract_timestamps':
            return dict(DATES.get(tagged(content, 'FACT'), {}))
        if prompt_name == 'extract_edges.extract_attributes':
            return dict(EDGE_ATTRIBUTES[tagged(content, 'FACT')])
        if prompt_name == 'extract_nodes.extract_attributes':
            name = re.search(r"'name': '([^']*)'", tagged(content, 'ENTITY'))
            assert name is not None
            return dict(NODE_ATTRIBUTES[name.group(1)])
        assert prompt_name == 'extract_nodes.extract_summaries_batch'
        return {'summaries': SUMMARIES}


class NodeHook:
    def __init__(self, answers: dict[str, dict[str, Any]]) -> None:
        self.answers = answers
        self.calls: list[tuple[Any, ...]] = []

    async def __call__(self, episode, previous, questions):
        self.calls.append((episode, previous, questions))
        return [dict(self.answers.get(question.node.name, {})) for question in questions]


class EdgeHook:
    def __init__(self, answers: dict[str, dict[str, Any]]) -> None:
        self.answers = answers
        self.calls: list[tuple[Any, ...]] = []

    async def __call__(self, episode, questions):
        self.calls.append((episode, questions))
        return [dict(self.answers.get(question.edge.fact, {})) for question in questions]


class SummaryHook:
    def __init__(self, rejected: set[str]) -> None:
        self.rejected = rejected
        self.calls: list[list[Any]] = []

    async def __call__(self, changes):
        self.calls.append(changes)
        return [change.node.uuid not in self.rejected for change in changes]


class EpisodeBlindAcl:
    """Lets every node and edge through and fails if it is asked about previous episodes."""

    async def allowed_episodes(self, episodes):
        raise AssertionError('explicit previous episodes reached the acl')

    async def allowed_nodes(self, nodes):
        return nodes

    async def allowed_edges(self, edges):
        return edges


class Embedder:
    async def create_batch(self, input_data_list):
        return [[0.5, 0.5] for _ in input_data_list]


def entity(
    uuid: str, name: str, label: str, marks: dict[str, Any], summary: str = '', **attributes: Any
) -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=name,
        group_id=GROUP,
        labels=['Entity', label],
        summary=summary,
        attributes={**marks, **attributes},
        created_at=EARLIER,
    )


def relation(
    uuid: str,
    source: str,
    target: str,
    name: str,
    fact: str,
    marks: dict[str, Any],
    stored: bool = False,
    **attributes: Any,
) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id=GROUP,
        source_node_uuid=source,
        target_node_uuid=target,
        name=name,
        fact=fact,
        episodes=['earlier' if stored else 'episode'],
        created_at=EARLIER,
        valid_at=EARLIER if stored else None,
        attributes={**marks, **attributes},
    )


def world(marked: bool = False) -> World:
    stored, extracted = (STORED_MARKS, EXTRACTED_MARKS) if marked else ({}, {})
    ana, notes = ({'user_id': 7}, {'document_id': 'notes'}) if marked else ({}, {})
    stored_nodes = [
        entity('ana', 'Ana', 'Person', {**stored, **ana}, 'Ana runs finance.', role='analyst'),
        entity('marta', 'Marta', 'Person', stored, 'Marta founded Apollo.', role='founder'),
        entity(
            'program',
            'Apollo Program',
            'Project',
            stored,
            LONG_SUMMARY,
            status='active',
            department='R&D',
        ),
        entity('budget', 'Quarterly Budget', 'Topic', stored, 'Quarterly Budget as stored.'),
    ]
    stored_edges = [
        relation(
            'owns',
            'ana',
            'program',
            'OWNS',
            'Ana owns the Apollo Program',
            stored,
            True,
            note='weekly',
        ),
        relation(
            'marta-leads',
            'marta',
            'program',
            'LEADS',
            'Marta leads the Apollo Program',
            stored,
            True,
            role='founder',
        ),
        relation(
            'ana-chairs',
            'ana',
            'budget',
            'LEADS',
            'Ana leads the quarterly budget',
            stored,
            True,
            role='chair',
        ),
    ]
    return World(
        {node.uuid: node for node in stored_nodes},
        {edge.uuid: edge for edge in stored_edges},
        [
            entity('ana', 'Ana', 'Person', {**extracted, **ana}),
            entity('marta', 'Marta', 'Person', extracted),
            entity('x-apollo', 'Apollo', 'Project', extracted),
            entity('x-review', 'Budget Review', 'Topic', extracted),
            entity('x-budget', 'Quarterly Budget', 'Topic', extracted),
            entity('x-zeta', 'Zeta Initiative', 'Project', extracted),
            entity('luis', 'Luis', 'Person', extracted),
            entity('x-notes', 'Kickoff Notes', 'Document', {**extracted, **notes}),
        ],
        [
            relation(
                'lead-again',
                'ana',
                'x-budget',
                'LEADS',
                ' ana leads the QUARTERLY budget ',
                extracted,
            ),
            relation(
                'owner',
                'ana',
                'x-apollo',
                'OWNS',
                'Ana is the owner of the Apollo Program',
                extracted,
            ),
            relation(
                'update',
                'luis',
                'x-apollo',
                'LEADS',
                'Luis now leads Apollo',
                extracted,
                role='interim',
            ),
            relation('new', 'ana', 'x-zeta', 'LEADS', 'Ana leads the Zeta Initiative', extracted),
            relation(
                'review',
                'ana',
                'x-review',
                'RUNS',
                'Ana runs the budget review',
                extracted,
                cadence='weekly',
            ),
        ],
    )


def episode() -> EpisodicNode:
    return EpisodicNode(
        uuid='episode',
        name='episode',
        group_id=GROUP,
        source=EpisodeType.message,
        source_description='chat',
        content='Luis now leads Apollo. Ana owns it, leads Zeta and runs the budget review.',
        valid_at=NOW,
        created_at=NOW,
    )


def earlier(
    uuid: str = 'earlier', content: str = 'Earlier, Marta set up the Apollo Program.'
) -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid,
        name=uuid,
        group_id=GROUP,
        source=EpisodeType.message,
        source_description='chat',
        content=content,
        valid_at=EARLIER,
        created_at=EARLIER,
    )


def install(monkeypatch, stored: World) -> None:
    async def get_by_uuids(driver, uuids, group_id=None):
        return [
            stored.stored_nodes[uuid].model_copy(deep=True)
            for uuid in uuids
            if uuid in stored.stored_nodes
        ]

    async def semantic_search(clients, nodes):
        return [
            [
                stored.stored_nodes[uuid].model_copy(deep=True)
                for uuid in SEMANTIC_HITS.get(node.name, [])
            ]
            for node in nodes
        ]

    async def get_between_nodes(driver, source_node_uuid, target_node_uuid):
        uuids = EDGES_BETWEEN.get((source_node_uuid, target_node_uuid), [])
        return [stored.stored_edges[uuid].model_copy(deep=True) for uuid in uuids]

    async def search(clients, query, group_ids=None, config=None, search_filter=None):
        uuids = (
            search_filter.edge_uuids
            if search_filter is not None and search_filter.edge_uuids is not None
            else INVALIDATION_HITS.get(query, [])
        )
        return SearchResults(
            edges=[stored.stored_edges[uuid].model_copy(deep=True) for uuid in uuids]
        )

    monkeypatch.setattr(EntityNode, 'get_by_uuids', get_by_uuids)
    monkeypatch.setattr(node_operations, '_semantic_candidate_search', semantic_search)
    monkeypatch.setattr(EntityEdge, 'get_between_nodes', get_between_nodes)
    monkeypatch.setattr(edge_operations, 'search', search)
    monkeypatch.setattr(
        edge_operations, 'create_entity_edge_embeddings', AsyncMock(return_value=None)
    )
    monkeypatch.setattr(edge_operations, 'utc_now', lambda: NOW)


def graph(llm: RecordingLlm) -> SimpleNamespace:
    clients = GraphitiClients.model_construct(
        driver=SimpleNamespace(provider=GraphProvider.NEO4J),
        llm_client=llm,
        embedder=Embedder(),
        cross_encoder=None,
    )
    return SimpleNamespace(clients=clients, retrieve_episodes=AsyncMock(return_value=[earlier()]))


async def hydrate(monkeypatch, llm: RecordingLlm, marked: bool = False, graphiti=None, **options):
    episode_world = world(marked)
    install(monkeypatch, episode_world)
    return await Graphiti.resolve_and_hydrate_nodes_edges(
        graphiti or graph(llm),
        group_id=GROUP,
        episode=episode(),
        extracted_nodes=episode_world.nodes,
        extracted_edges=episode_world.edges,
        entity_types=ENTITY_TYPES,
        edge_types=EDGE_TYPES,
        edge_type_map=EDGE_TYPE_MAP,
        **{'get_previous_episodes': True, **options},
    )


def outcome(result, unmarked: bool = False) -> list[Any]:
    nodes, edges, invalidated, uuid_map = result

    def attributes(item) -> dict[str, Any]:
        return {
            key: value
            for key, value in item.attributes.items()
            if not (unmarked and key in PROTECTED_ATTRIBUTES)
        }

    return [
        [(node.uuid, node.labels, node.summary, attributes(node)) for node in nodes],
        [
            (
                edge.uuid,
                edge.fact,
                edge.episodes,
                edge.valid_at,
                edge.invalid_at,
                edge.expired_at,
                attributes(edge),
            )
            for edge in edges
        ],
        [
            (edge.uuid, edge.valid_at, edge.invalid_at, edge.expired_at, attributes(edge))
            for edge in invalidated
        ],
        uuid_map,
    ]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, default=str, ensure_ascii=False).encode()).hexdigest()


def prompt_names(llm: RecordingLlm) -> set[str]:
    return {call.prompt_name for call in llm.calls}


@pytest.mark.asyncio
async def test_without_hooks_an_episode_sends_todays_prompts_and_gets_todays_result(monkeypatch):
    llm = RecordingLlm()

    result = await hydrate(monkeypatch, llm)

    assert digest([[call.prompt_name, call.messages] for call in llm.calls]) == PROMPTS_SHA
    assert Counter(call[:1] + call[2:] for call in llm.calls) == TODAYS_CALLS
    assert digest(outcome(result)) == OUTCOME_SHA


@pytest.mark.asyncio
async def test_protected_attributes_stay_out_of_every_prompt_and_in_what_is_persisted(monkeypatch):
    plain, marked = RecordingLlm(), RecordingLlm()

    plain_result = await hydrate(monkeypatch, plain)
    marked_result = await hydrate(monkeypatch, marked, marked=True)

    assert [call.messages for call in marked.calls] == [call.messages for call in plain.calls]
    assert outcome(marked_result, unmarked=True) == outcome(plain_result)
    nodes, edges, invalidated, _ = marked_result
    assert {node.uuid: protected_attributes(node.attributes) for node in nodes} == {
        'ana': {**MERGED_MARKS, 'user_id': 7},
        'marta': MERGED_MARKS,
        'program': STORED_MARKS,
        'x-review': EXTRACTED_MARKS,
        'budget': STORED_MARKS,
        'x-zeta': EXTRACTED_MARKS,
        'luis': EXTRACTED_MARKS,
        'x-notes': {**EXTRACTED_MARKS, 'document_id': 'notes'},
    }
    assert {
        edge.uuid: protected_attributes(edge.attributes) for edge in [*edges, *invalidated]
    } == {
        'ana-chairs': MERGED_MARKS,
        'owns': MERGED_MARKS,
        'update': EXTRACTED_MARKS,
        'new': EXTRACTED_MARKS,
        'review': EXTRACTED_MARKS,
        'marta-leads': STORED_MARKS,
    }


@pytest.mark.asyncio
async def test_todays_attribute_prompts_given_as_hooks_behave_as_no_hooks(monkeypatch):
    default, explicit = RecordingLlm(), RecordingLlm()

    by_default = await hydrate(monkeypatch, default, marked=True)
    by_hooks = await hydrate(
        monkeypatch,
        explicit,
        marked=True,
        node_attributes=LlmNodeAttributes(explicit),
        edge_attributes=LlmEdgeAttributes(explicit),
    )

    assert explicit.calls == default.calls
    assert outcome(by_hooks) == outcome(by_default)


@pytest.mark.asyncio
async def test_a_node_hook_replaces_the_prompt_in_one_call_and_cannot_write_protected_attributes(
    monkeypatch,
):
    llm = RecordingLlm()
    hook = NodeHook(
        {
            **NODE_ATTRIBUTES,
            'Ana': {
                'role': 'finance lead',
                'provenance': ['forged'],
                'user_id': 999,
                'owner_user_id': None,
            },
            'Apollo Program': {'status': 'paused'},
        }
    )

    nodes, _, _, _ = await hydrate(monkeypatch, llm, marked=True, node_attributes=hook)

    assert 'extract_nodes.extract_attributes' not in prompt_names(llm)
    [(given_episode, previous, questions)] = hook.calls
    assert (given_episode.uuid, [item.uuid for item in previous]) == ('episode', ['earlier'])
    assert [(question.node.uuid, question.entity_type) for question in questions] == [
        ('ana', Person),
        ('marta', Person),
        ('program', Project),
        ('x-zeta', Project),
        ('luis', Person),
        ('x-notes', Document),
    ]
    attributes = {node.uuid: node.attributes for node in nodes}
    assert attributes['ana'] == {'role': 'finance lead', **MERGED_MARKS, 'user_id': 7}
    assert attributes['program'] == {'status': 'paused', 'department': 'R&D', **STORED_MARKS}
    assert (attributes['budget'], attributes['x-review']) == (STORED_MARKS, EXTRACTED_MARKS)


@pytest.mark.asyncio
async def test_a_node_hook_that_leaves_a_required_attribute_empty_fails_validation(monkeypatch):
    hook = NodeHook({**NODE_ATTRIBUTES, 'Kickoff Notes': {}})

    with pytest.raises(ValidationError):
        await hydrate(monkeypatch, RecordingLlm(), node_attributes=hook)


@pytest.mark.asyncio
async def test_an_edge_hook_answers_the_typed_edges_that_are_not_literal_duplicates_in_one_call(
    monkeypatch,
):
    llm = RecordingLlm()
    hook = EdgeHook(
        {
            'Luis now leads Apollo': {},
            'Ana leads the Zeta Initiative': {
                'role': 'sponsor',
                'provenance': ['forged'],
                'owner_user_id': 99,
                'note': 'not a field',
            },
        }
    )

    _, edges, _, _ = await hydrate(monkeypatch, llm, marked=True, edge_attributes=hook)

    assert 'extract_edges.extract_attributes' not in prompt_names(llm)
    [(given_episode, questions)] = hook.calls
    assert given_episode.uuid == 'episode'
    assert [
        (question.edge.uuid, question.edge_type, question.source.uuid, question.target.uuid)
        for question in questions
    ] == [('update', Leads, 'luis', 'program'), ('new', Leads, 'ana', 'x-zeta')]
    assert {edge.uuid: edge.attributes for edge in edges} == {
        'ana-chairs': {**MERGED_MARKS, 'role': 'chair'},
        'owns': MERGED_MARKS,
        'update': EXTRACTED_MARKS,
        'new': {'role': 'sponsor', **EXTRACTED_MARKS},
        'review': {**EXTRACTED_MARKS, 'cadence': 'weekly'},
    }


@pytest.mark.asyncio
async def test_the_edge_hook_sees_a_stored_endpoint_the_episode_did_not_resolve(monkeypatch):
    install(monkeypatch, world())
    hook = EdgeHook({})
    update = relation('update', 'luis', 'program', 'LEADS', 'Luis now leads Apollo', {})

    await edge_operations.resolve_extracted_edges(
        graph(RecordingLlm()).clients,
        [update],
        episode(),
        [entity('luis', 'Luis', 'Person', {})],
        EDGE_TYPES,
        EDGE_TYPE_MAP,
        edge_attributes=hook,
    )

    [(_, [question])] = hook.calls
    assert (question.source.uuid, question.target.uuid, question.target.name) == (
        'luis',
        'program',
        'Apollo Program',
    )


@pytest.mark.asyncio
async def test_explicit_previous_episodes_are_used_as_given_without_retrieval_or_acl(monkeypatch):
    llm = RecordingLlm()
    graphiti = graph(llm)
    given = [earlier('given', 'Only this came before.')]
    hook = NodeHook(NODE_ATTRIBUTES)

    await hydrate(
        monkeypatch,
        llm,
        graphiti=graphiti,
        acl=EpisodeBlindAcl(),
        get_previous_episodes=False,
        previous_episodes=given,
        node_attributes=hook,
    )

    graphiti.retrieve_episodes.assert_not_called()
    assert [previous for _, previous, _ in hook.calls] == [given]
    shown = [
        call.messages[-1][1]
        for call in llm.calls
        if call.prompt_name in {'dedupe_nodes.nodes', 'extract_nodes.extract_summaries_batch'}
    ]
    assert len(shown) == 2
    assert all(
        'Only this came before.' in prompt and 'Marta set up' not in prompt for prompt in shown
    )


@pytest.mark.asyncio
async def test_previous_episodes_cannot_be_given_and_retrieved_at_once(monkeypatch):
    llm = RecordingLlm()

    with pytest.raises(ValueError):
        await hydrate(monkeypatch, llm, get_previous_episodes=True, previous_episodes=[])

    assert llm.calls == []


@pytest.mark.asyncio
async def test_the_summary_policy_sees_each_node_once_and_rejected_ones_get_no_facts_or_summary(
    monkeypatch,
):
    llm = RecordingLlm()
    policy = SummaryHook(rejected={'program', 'luis'})

    nodes, _, _, _ = await hydrate(monkeypatch, llm, should_summarize_node=policy)

    [changes] = policy.calls
    assert [
        (
            change.node.uuid,
            [edge.uuid for edge in change.new_facts],
            [edge.uuid for edge in change.invalidated_facts],
        )
        for change in changes
    ] == [
        ('ana', ['new', 'review'], []),
        ('marta', [], ['marta-leads']),
        ('program', ['update'], ['marta-leads']),
        ('x-review', ['review'], []),
        ('budget', [], []),
        ('x-zeta', ['new'], []),
        ('luis', ['update'], []),
        ('x-notes', [], []),
    ]
    summaries = {node.uuid: node.summary for node in nodes}
    assert (summaries['program'], summaries['luis']) == (LONG_SUMMARY, '')
    assert (
        summaries['ana']
        == 'Ana runs finance.\nAna leads the Zeta Initiative\nAna runs the budget review'
    )
    [summarized] = [
        call.messages[-1][1]
        for call in llm.calls
        if call.prompt_name == 'extract_nodes.extract_summaries_batch'
    ]
    assert [entity['name'] for entity in json.loads(tagged(summarized, 'ENTITIES'))] == [
        'Kickoff Notes'
    ]


def test_a_node_resolved_twice_is_one_change():
    apollo = entity('program', 'Apollo Program', 'Project', {})
    update = relation('update', 'luis', 'program', 'LEADS', 'Luis now leads Apollo', {})

    changes = _node_changes([apollo, apollo.model_copy()], [update], [])

    assert [(change.node, change.new_facts, change.invalidated_facts) for change in changes] == [
        (apollo, (update,), ())
    ]
