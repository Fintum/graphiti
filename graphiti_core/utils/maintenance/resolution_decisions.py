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

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupResolutionState,
    _build_candidate_indexes,
)


@dataclass(frozen=True)
class NodeQuestion:
    """An extracted node the deterministic passes left open, with its own candidates.

    The candidates have a compatible identity, are visible under the enrichment ACL and keep
    the order of the semantic search.
    """

    extracted: EntityNode
    candidates: tuple[EntityNode, ...]


@dataclass(frozen=True)
class EdgeQuestion:
    """An extracted edge with the stored edges it may restate or contradict.

    Duplicates share both endpoints, the relation name and a compatible scope; invalidation
    candidates come from the hybrid search and share a compatible scope.
    """

    extracted: EntityEdge
    duplicates: tuple[EntityEdge, ...]
    invalidations: tuple[EntityEdge, ...]


@dataclass(frozen=True)
class EdgeVerdict:
    """The offered edge that the extracted edge restates, if any, and the ones it contradicts."""

    duplicate: EntityEdge | None
    contradicted: tuple[EntityEdge, ...]


class ResolutionDecisions(Protocol):
    """The judgement calls of node and edge resolution.

    The fork asks only what its deterministic passes leave open and validates every answer:
    a verdict naming anything the question did not offer, or a node of another identity, is
    ignored, and temporal invalidation keeps its own rules.
    """

    async def resolve_nodes(
        self,
        episode: EpisodicNode | None,
        previous: list[EpisodicNode],
        questions: list[NodeQuestion],
    ) -> list[EntityNode | None]:
        """Name, per question, the candidate the extracted node is, or None when it is new."""
        ...

    async def resolve_edges(
        self, episode: EpisodicNode, questions: list[EdgeQuestion]
    ) -> list[EdgeVerdict]:
        """Return one verdict per question, in order."""
        ...

    async def date_edges(self, episode: EpisodicNode, edges: list[EntityEdge]) -> None:
        """Set valid_at and invalid_at in place on the new edges that still lack them."""
        ...


class LlmResolutionDecisions:
    """Today's prompts: one dedupe_nodes.nodes for all the open nodes, one dedupe_edges.resolve_edge
    per open edge and one extract_edges.extract_timestamps per new edge still without dates.

    The prompt code stays in node_operations and edge_operations, which build on this module,
    so it is imported where it is used.
    """

    def __init__(
        self, llm_client: LLMClient, entity_types: dict[str, type[BaseModel]] | None = None
    ) -> None:
        self.llm_client = llm_client
        self.entity_types = entity_types

    async def resolve_nodes(
        self,
        episode: EpisodicNode | None,
        previous: list[EpisodicNode],
        questions: list[NodeQuestion],
    ) -> list[EntityNode | None]:
        """Offer the LLM every open node against the candidates of all of them, as one prompt."""
        from graphiti_core.utils.maintenance.node_operations import (
            _merge_candidate_nodes,
            _resolve_with_llm,
        )

        extracted = [question.extracted for question in questions]
        offered = _merge_candidate_nodes(
            [candidate for question in questions for candidate in question.candidates], None
        )
        state = DedupResolutionState(
            resolved_nodes=[None] * len(extracted),
            uuid_map={},
            unresolved_indices=list(range(len(extracted))),
        )
        await _resolve_with_llm(
            self.llm_client,
            extracted,
            _build_candidate_indexes(offered),
            state,
            episode,
            previous,
            self.entity_types,
        )
        return [
            None if resolved is None or resolved.uuid == node.uuid else resolved
            for node, resolved in zip(extracted, state.resolved_nodes, strict=True)
        ]

    async def resolve_edges(
        self, episode: EpisodicNode, questions: list[EdgeQuestion]
    ) -> list[EdgeVerdict]:
        """Ask the dedupe prompt once per question; the first duplicate it names is the verdict."""
        from graphiti_core.utils.maintenance.edge_operations import _resolve_edge_with_llm

        answers = await semaphore_gather(
            *[_resolve_edge_with_llm(self.llm_client, question) for question in questions]
        )
        return [
            EdgeVerdict(duplicates[0] if duplicates else None, tuple(contradicted))
            for duplicates, contradicted in answers
        ]

    async def date_edges(self, episode: EpisodicNode, edges: list[EntityEdge]) -> None:
        """Ask the timestamps prompt for every edge without dates, against the episode time."""
        from graphiti_core.utils.maintenance.edge_operations import _extract_edge_timestamps

        await semaphore_gather(
            *[_extract_edge_timestamps(self.llm_client, edge, episode) for edge in edges]
        )
