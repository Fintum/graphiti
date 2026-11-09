import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.maintenance import node_operations
from graphiti_core.utils.maintenance.identity_constraints import IDENTITY_FIELDS
from graphiti_core.utils.maintenance.name_copies import (
    ADMISSION_TRANCHE,
    STORED_COPIES,
    lowered_spellings,
    stored_copies,
)
from graphiti_core.utils.maintenance.node_operations import (
    NODE_DEDUP_CANDIDATE_LIMIT,
    _collect_candidate_nodes,
    resolve_extracted_nodes,
)

FIRST_SEEN = datetime(2026, 1, 5, tzinfo=timezone.utc)


def copy_of(name: str, minutes_later: int, kind: str = 'Company', **attributes: Any) -> EntityNode:
    return EntityNode(
        name=name,
        group_id='org',
        labels=['Entity', kind],
        created_at=FIRST_SEEN + timedelta(minutes=minutes_later),
        summary=f'{name}, as stored',
        attributes=attributes,
    )


@dataclass
class OnlyAdmits:
    """Admits some uuids, shows some of them trimmed, and can break once the page is through."""

    uuids: set[str]
    trimmed: set[str] = field(default_factory=set)
    breaks_after: int | None = None
    shown: list[list[str]] = field(default_factory=list)

    async def allowed_nodes(self, nodes: list[EntityNode]) -> list[EntityNode]:
        if self.breaks_after is not None and len(self.shown) >= self.breaks_after:
            raise RuntimeError('the evidence query timed out')
        self.shown.append([node.uuid for node in nodes])
        admitted = [node for node in nodes if node.uuid in self.uuids]
        return [
            node.model_copy(update={'summary': '', 'attributes': {}})
            if node.uuid in self.trimmed
            else node
            for node in admitted
        ]

    async def allowed_episodes(self, episodes: list[EpisodicNode]) -> list[EpisodicNode]:
        return episodes

    async def allowed_edges(self, edges: list[Any]) -> list[Any]:
        return edges


class CopiesGraph:
    """Answers the name lookup, grouped by spelling and oldest first, and the load by uuid."""

    def __init__(self, stored: list[EntityNode], provider: GraphProvider = GraphProvider.FALKORDB):
        self.provider = provider
        self.search_interface = None
        self.stored = {node.uuid: node for node in stored}
        self.lookup_breaks = False
        self.asked: list[dict[str, Any]] = []
        self.loaded: list[list[str]] = []
        self.graph_operations_interface = SimpleNamespace(node_get_by_uuids=self.by_uuids)

    async def execute_query(
        self, query: str, **params: Any
    ) -> tuple[list[dict[str, Any]], None, None]:
        if query != STORED_COPIES:
            return [], None, None
        self.asked.append(params)
        if self.lookup_breaks:
            raise RuntimeError('Query timed out')
        spellings: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for node in sorted(
            self.stored.values(), key=lambda node: (node.created_at.isoformat(), node.uuid)
        ):
            if node.group_id in params['group_ids'] and node.name.lower() in params['names']:
                spellings.setdefault((node.group_id, node.name.lower()), []).append(
                    {
                        'uuid': node.uuid,
                        'name': node.name,
                        'created_at': node.created_at.isoformat(),
                        'labels': list(node.labels),
                        **{key: node.attributes.get(key) for key in IDENTITY_FIELDS},
                    }
                )
        rows = [
            {'group_id': group_id, 'stored': len(copies), 'copies': copies[: params['limit']]}
            for (group_id, _), copies in spellings.items()
        ]
        return rows, None, None

    async def by_uuids(
        self, cls: Any, driver: Any, uuids: list[str], group_id: str | None = None
    ) -> list[EntityNode]:
        self.loaded.append(list(uuids))
        return [self.stored[uuid].model_copy(deep=True) for uuid in uuids if uuid in self.stored]


def clients_over(graph: CopiesGraph) -> GraphitiClients:
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock()
    return GraphitiClients.model_construct(
        driver=graph, embedder=MagicMock(), cross_encoder=MagicMock(), llm_client=llm_client
    )


def searching(monkeypatch: pytest.MonkeyPatch, page: list[EntityNode]) -> None:
    async def search(clients: Any, extracted: list[EntityNode]) -> list[list[EntityNode]]:
        return [list(page) for _ in extracted]

    monkeypatch.setattr(node_operations, '_semantic_candidate_search', search)


def hidden_crowd(
    count: int = NODE_DEDUP_CANDIDATE_LIMIT, first_minute: int = 0
) -> list[EntityNode]:
    return [copy_of('Kiboa', minute) for minute in range(first_minute, first_minute + count)]


def extracted(name: str = 'Kiboa', kind: str = 'Company') -> EntityNode:
    return EntityNode(name=name, group_id='org', labels=['Entity', kind])


async def candidates_of(
    graph: CopiesGraph, acl: OnlyAdmits, node: EntityNode | None = None
) -> list[EntityNode]:
    (found,) = await _collect_candidate_nodes(clients_over(graph), [node or extracted()], None, acl)
    return found


@pytest.mark.asyncio
async def test_a_page_full_of_hidden_copies_offers_the_visible_copy_the_cut_left_out(monkeypatch):
    crowd, beyond = hidden_crowd(), copy_of('Kiboa', 40)
    visible = copy_of('KIBOA', 30, industry='retail')
    graph = CopiesGraph([*crowd, beyond, visible])
    acl = OnlyAdmits({visible.uuid})
    searching(monkeypatch, crowd)
    node = extracted()

    found = await candidates_of(graph, acl, node)

    assert [(copy.uuid, copy.attributes) for copy in found] == [
        (visible.uuid, {'industry': 'retail'})
    ]
    assert graph.loaded == [[node.uuid], [visible.uuid]]
    assert acl.shown[-2] == [visible.uuid, beyond.uuid]
    assert set(graph.asked[0]['names']) == {'kiboa'}


@pytest.mark.asyncio
async def test_a_copy_loaded_whole_is_offered_as_the_acl_shows_it(monkeypatch):
    crowd, directory = hidden_crowd(), copy_of('Kiboa', 30, industry='retail')
    searching(monkeypatch, crowd)

    found = await candidates_of(
        CopiesGraph([*crowd, directory]), OnlyAdmits({directory.uuid}, {directory.uuid})
    )

    assert [(copy.uuid, copy.summary, copy.attributes) for copy in found] == [
        (directory.uuid, '', {})
    ]


@pytest.mark.asyncio
async def test_two_visible_copies_past_the_cut_resolve_to_the_oldest_without_the_llm(monkeypatch):
    crowd, newer, oldest = hidden_crowd(), copy_of('Kiboa', 90), copy_of('Kiboa', 60)
    graph = CopiesGraph([*crowd, newer, oldest])
    clients = clients_over(graph)
    searching(monkeypatch, crowd)
    node = extracted()

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [node],
        acl=OnlyAdmits({newer.uuid, oldest.uuid}),
        episode=EpisodicNode(
            name='m',
            group_id='org',
            source=EpisodeType.message,
            source_description='t',
            content='Kiboa',
            valid_at=FIRST_SEEN,
        ),
    )

    assert [found.uuid for found in resolved] == [oldest.uuid]
    assert uuid_map == {node.uuid: oldest.uuid}
    clients.llm_client.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_admission_goes_oldest_first_by_tranches_until_the_limit_is_met(monkeypatch):
    crowd, copies = hidden_crowd(), hidden_crowd(2 * ADMISSION_TRANCHE + 100, first_minute=100)
    visible = copies[:20]
    acl = OnlyAdmits({copy.uuid for copy in visible})
    graph = CopiesGraph([*crowd, *copies])
    searching(monkeypatch, crowd)

    found = await candidates_of(graph, acl)

    assert [copy.uuid for copy in found] == [
        copy.uuid for copy in visible[:NODE_DEDUP_CANDIDATE_LIMIT]
    ]
    assert [len(shown) for shown in acl.shown] == [
        NODE_DEDUP_CANDIDATE_LIMIT,
        ADMISSION_TRANCHE,
        NODE_DEDUP_CANDIDATE_LIMIT,
    ]


@pytest.mark.asyncio
async def test_admission_reaches_the_newest_copy_past_every_tranche(monkeypatch):
    crowd, copies = hidden_crowd(), hidden_crowd(2 * ADMISSION_TRANCHE + 100, first_minute=100)
    acl = OnlyAdmits({copies[-1].uuid})
    searching(monkeypatch, crowd)

    found = await candidates_of(CopiesGraph([*crowd, *copies]), acl)

    assert [copy.uuid for copy in found] == [copies[-1].uuid]
    assert [len(shown) for shown in acl.shown[1:-1]] == [ADMISSION_TRANCHE, ADMISSION_TRANCHE, 100]


@pytest.mark.asyncio
async def test_older_copies_of_another_type_never_take_the_places_of_a_compatible_one(monkeypatch):
    crowd = hidden_crowd()
    topics = [
        copy_of('Kiboa', 100 + minute, 'Topic') for minute in range(NODE_DEDUP_CANDIDATE_LIMIT + 1)
    ]
    company = copy_of('Kiboa', 500)
    searching(monkeypatch, crowd)

    found = await candidates_of(
        CopiesGraph([*crowd, *topics, company]),
        OnlyAdmits({company.uuid} | {t.uuid for t in topics}),
    )

    assert [copy.uuid for copy in found] == [company.uuid]


@pytest.mark.asyncio
@pytest.mark.parametrize('broken', ['lookup', 'admission'])
async def test_a_failing_copies_step_is_logged_and_the_candidates_stay_those_of_the_page(
    monkeypatch, caplog, broken
):
    shown = copy_of('Kiboa', 7)
    crowd, visible = [*hidden_crowd(NODE_DEDUP_CANDIDATE_LIMIT - 1), shown], copy_of('Kiboa', 30)
    graph = CopiesGraph([*crowd, visible])
    graph.lookup_breaks = broken == 'lookup'
    acl = OnlyAdmits({shown.uuid, visible.uuid}, breaks_after=1 if broken == 'admission' else None)
    searching(monkeypatch, crowd)

    with caplog.at_level(logging.ERROR, logger='graphiti_core.utils.maintenance.name_copies'):
        found = await candidates_of(graph, acl)

    assert [copy.uuid for copy in found] == [shown.uuid]
    assert (
        len(caplog.records) == 1
        and 'keeping the similarity candidates' in caplog.records[0].getMessage()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'page',
    [
        pytest.param(
            [*hidden_crowd(NODE_DEDUP_CANDIDATE_LIMIT - 1), copy_of('Kiboa Labs', 99)],
            id='a-full-page-ending-on-another-name',
        ),
        pytest.param(hidden_crowd(NODE_DEDUP_CANDIDATE_LIMIT - 1), id='a-short-page-of-copies'),
    ],
)
async def test_a_page_the_name_does_not_crowd_asks_the_graph_nothing_more(monkeypatch, page):
    graph = CopiesGraph(page)
    searching(monkeypatch, page)
    node = extracted()

    found = await candidates_of(graph, OnlyAdmits(set()), node)

    assert (found, graph.asked, graph.loaded) == ([], [], [[node.uuid]])


def test_a_stored_copy_is_looked_up_as_written_with_its_spaces_collapsed_and_folded():
    assert lowered_spellings('Transfórmate  la Vida') == {
        'transfórmate  la vida',
        'transfórmate la vida',
        'transformate la vida',
    }


@pytest.mark.asyncio
async def test_a_provider_without_the_lookup_reports_no_copies():
    graph = CopiesGraph(hidden_crowd(), GraphProvider.KUZU)

    assert await stored_copies(graph, [extracted()]) == {}  # type: ignore[arg-type]
    assert graph.asked == []
