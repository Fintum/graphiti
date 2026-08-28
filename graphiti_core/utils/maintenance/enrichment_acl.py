"""Caller-supplied ACL over the graph material an episode may be enriched with."""

from typing import Protocol

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode


class EnrichmentAcl(Protocol):
    """Narrows the candidates a new episode is crossed against; the caller owns what 'allowed' means."""

    async def allowed_episodes(self, episodes: list[EpisodicNode]) -> list[EpisodicNode]: ...

    async def allowed_edges(self, edges: list[EntityEdge]) -> list[EntityEdge]: ...

    async def allowed_nodes(self, nodes: list[EntityNode]) -> list[EntityNode]: ...


async def acl_episodes(
    acl: EnrichmentAcl | None, episodes: list[EpisodicNode]
) -> list[EpisodicNode]:
    """No acl means the unfiltered behaviour."""
    if acl is None or not episodes:
        return episodes
    return await acl.allowed_episodes(episodes)


async def acl_edges(acl: EnrichmentAcl | None, edges: list[EntityEdge]) -> list[EntityEdge]:
    """No acl means the unfiltered behaviour."""
    if acl is None or not edges:
        return edges
    return await acl.allowed_edges(edges)


async def acl_nodes(acl: EnrichmentAcl | None, nodes: list[EntityNode]) -> list[EntityNode]:
    """No acl means the unfiltered behaviour."""
    if acl is None or not nodes:
        return nodes
    return await acl.allowed_nodes(nodes)
