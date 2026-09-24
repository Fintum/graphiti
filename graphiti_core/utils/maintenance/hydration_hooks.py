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

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodicNode


@dataclass(frozen=True)
class AttributeQuestion:
    """A resolved node whose entity type declares attributes, with that type."""

    node: EntityNode
    entity_type: type[BaseModel]


@dataclass(frozen=True)
class EdgeAttributeQuestion:
    """A resolved edge whose relation type declares attributes, with that type and its endpoints.

    An endpoint is None only when it is neither among the resolved nodes nor stored.
    """

    edge: EntityEdge
    edge_type: type[BaseModel]
    source: EntityNode | None
    target: EntityNode | None


@dataclass(frozen=True)
class NodeChange:
    """A resolved node with its new facts and the stored facts the episode invalidates on it."""

    node: EntityNode
    new_facts: tuple[EntityEdge, ...]
    invalidated_facts: tuple[EntityEdge, ...]


class NodeAttributes(Protocol):
    """Proposes the attributes of the typed nodes of an episode: one answer per question, in order.

    The fork caps each answer, overlays it on the node's attributes, keeps the protected ones and
    validates the result against the entity type, so an answer cannot write a protected attribute
    and a missing required field fails the episode.
    """

    async def __call__(
        self,
        episode: EpisodicNode | list[EpisodicNode] | None,
        previous: list[EpisodicNode],
        questions: list[AttributeQuestion],
    ) -> list[dict[str, Any]]: ...


class EdgeAttributes(Protocol):
    """Proposes the attributes of the typed edges of an episode: one answer per question, in order.

    Literal duplicates are never asked. The fork caps each answer and lets it replace the edge's
    inferred attributes, keeping the protected ones.
    """

    async def __call__(
        self, episode: EpisodicNode, questions: list[EdgeAttributeQuestion]
    ) -> list[dict[str, Any]]: ...


class SummaryPolicy(Protocol):
    """Says, per resolved node and in order, whether to append facts to it and summarize it."""

    async def __call__(self, changes: list[NodeChange]) -> list[bool]: ...


class LlmNodeAttributes:
    """Today's extract_nodes.extract_attributes prompt, once per question.

    The prompt code stays in node_operations, which builds on this module, so it is imported where
    it is used.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def __call__(
        self,
        episode: EpisodicNode | list[EpisodicNode] | None,
        previous: list[EpisodicNode],
        questions: list[AttributeQuestion],
    ) -> list[dict[str, Any]]:
        from graphiti_core.utils.maintenance.node_operations import _ask_entity_attributes

        return await semaphore_gather(
            *[
                _ask_entity_attributes(
                    self.llm_client, question.node, episode, previous, question.entity_type
                )
                for question in questions
            ]
        )


class LlmEdgeAttributes:
    """Today's extract_edges.extract_attributes prompt, once per question."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def __call__(
        self, episode: EpisodicNode, questions: list[EdgeAttributeQuestion]
    ) -> list[dict[str, Any]]:
        from graphiti_core.utils.maintenance.edge_operations import _ask_edge_attributes

        return await semaphore_gather(
            *[
                _ask_edge_attributes(self.llm_client, question.edge, episode, question.edge_type)
                for question in questions
            ]
        )


def _node_changes(
    nodes: list[EntityNode], new_edges: list[EntityEdge], invalidated_edges: list[EntityEdge]
) -> list[NodeChange]:
    """One change per distinct resolved node, in order, with the new and invalidated facts on it."""
    distinct: dict[str, EntityNode] = {}
    for node in nodes:
        distinct.setdefault(node.uuid, node)

    def touching(edges: list[EntityEdge], uuid: str) -> tuple[EntityEdge, ...]:
        return tuple(
            edge for edge in edges if uuid in (edge.source_node_uuid, edge.target_node_uuid)
        )

    return [
        NodeChange(node, touching(new_edges, uuid), touching(invalidated_edges, uuid))
        for uuid, node in distinct.items()
    ]


async def _summary_filter(
    policy: SummaryPolicy | None,
    nodes: list[EntityNode],
    new_edges: list[EntityEdge],
    invalidated_edges: list[EntityEdge],
) -> Callable[[EntityNode], Awaitable[bool]] | None:
    """Ask the policy once about every resolved node and let the summaries through by uuid.

    No policy means no filter: every node is appended to and summarized as before.
    """
    if policy is None:
        return None
    changes = _node_changes(nodes, new_edges, invalidated_edges)
    verdicts = await policy(changes) if changes else []
    approved = {change.node.uuid for change, keep in zip(changes, verdicts, strict=True) if keep}

    async def approved_node(node: EntityNode) -> bool:
        return node.uuid in approved

    return approved_node
