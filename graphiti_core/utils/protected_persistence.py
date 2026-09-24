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

from __future__ import annotations

from typing import TYPE_CHECKING

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.utils.identity_constraints import (
    PROTECTED_ATTRIBUTES,
    IDENTITY_FIELDS,
    SCOPE_FIELDS,
    IdentityConflict,
    require_same_identity,
    scopes_compatible,
)

if TYPE_CHECKING:
    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode


def preserving_update(variable: str, payload: str, carry: str = '') -> str:
    variables = f'{variable}, {carry}' if carry else variable
    assignments = []
    fields = PROTECTED_ATTRIBUTES | ({'episodes', 'invalidated_by'} if variable != 'n' else set())
    for key in sorted(fields):
        prior, current = f'authority.{key}', f'{variable}.{key}'
        value = f'coalesce({prior}, {current})'
        if key in {'provenance', 'evidence_refs', 'episodes', 'invalidated_by'}:
            value = (f'CASE WHEN {prior} IS NULL THEN {current} WHEN {current} IS NULL THEN {prior} '
                     f'ELSE {prior} + [item IN {current} WHERE NOT item IN {prior}] END')
        assignments.append(f'{current} = {value}')
    if variable == 'n':
        assignments.append(
            'n.name = CASE WHEN authority.user_id IS NOT NULL OR authority.directory = true '
            'THEN coalesce(authority.name, n.name) ELSE n.name END'
        )
    return (f'WITH {variables}, properties({variable}) AS authority '
            f'SET {variable} = {payload} SET ' + ', '.join(assignments))


async def validate_entity_writes(
    driver: GraphDriver, nodes: list[EntityNode], edges: list[EntityEdge],
    episode_uuids: tuple[str, ...] = (),
) -> None:
    if driver.provider != GraphProvider.FALKORDB:
        return
    if set(episode_uuids) & {node.uuid for node in nodes}:
        raise IdentityConflict('incompatible_episode_entity_identity')
    if episode_uuids:
        collisions = (await driver.execute_query(
            'MATCH (n:Entity) WHERE n.uuid IN $uuids RETURN n.uuid AS uuid', uuids=list(episode_uuids),
        ))[0]
        if collisions:
            raise IdentityConflict('incompatible_episode_entity_identity')
    by_uuid: dict[str, EntityNode] = {}
    by_identity: dict[tuple[str, str, str], str] = {}
    for node in nodes:
        require_same_identity(node, node)
        for key in IDENTITY_FIELDS:
            if node.attributes.get(key) is None:
                continue
            identity = (node.group_id, key, str(node.attributes[key]))
            if by_identity.get(identity, node.uuid) != node.uuid:
                raise IdentityConflict('duplicate_incoming_identity')
            by_identity[identity] = node.uuid
        if node.uuid in by_uuid:
            require_same_identity(node, by_uuid[node.uuid])
        by_uuid[node.uuid] = node
    if by_uuid:
        rows = (await driver.execute_query(
            'MATCH (n) WHERE n.uuid IN $uuids RETURN properties(n) AS properties, labels(n) AS labels',
            uuids=list(by_uuid),
        ))[0]
        seen: set[str] = set()
        for row in rows:
            properties = row['properties']
            if properties['uuid'] in seen:
                raise IdentityConflict('duplicate_stored_uuid')
            seen.add(properties['uuid'])
            node = by_uuid[properties['uuid']]
            stored = node.model_copy(update={
                'attributes': properties, 'labels': row['labels'],
                'group_id': properties.get('group_id', ''),
            })
            require_same_identity(node, stored)
        identities = [
            {'uuid': node.uuid, 'group_id': node.group_id, 'kind': key, 'value': str(node.attributes[key])}
            for node in nodes for key in IDENTITY_FIELDS if node.attributes.get(key) is not None
        ]
        if identities:
            conditions = ' OR '.join(
                f"(identity.kind = '{key}' AND toString(n.{key}) = identity.value)" for key in sorted(IDENTITY_FIELDS)
            )
            collisions = (await driver.execute_query(
                'UNWIND $identities AS identity MATCH (n:Entity) '
                f'WHERE n.group_id = identity.group_id AND ({conditions}) AND n.uuid <> identity.uuid '
                'RETURN n.uuid AS uuid', identities=identities,
            ))[0]
            if collisions:
                raise IdentityConflict('noncanonical_stored_identity')
    if edges:
        incoming_edges: dict[str, EntityEdge] = {}
        for edge in edges:
            other = incoming_edges.get(edge.uuid)
            if other is not None and (other.source_node_uuid, other.target_node_uuid, other.name, other.group_id) != (
                edge.source_node_uuid, edge.target_node_uuid, edge.name, edge.group_id,
            ):
                raise IdentityConflict('incompatible_edge_identity')
            if other is not None and not scopes_compatible(other.attributes, edge.attributes):
                raise IdentityConflict('incompatible_edge_scope')
            incoming_edges[edge.uuid] = edge
        rows = (await driver.execute_query(
            'MATCH (s)-[e]->(t) WHERE e.uuid IN $uuids '
            'RETURN properties(e) AS properties, s.uuid AS source, t.uuid AS target',
            uuids=[edge.uuid for edge in edges],
        ))[0]
        stored_edges = {row['properties']['uuid']: row for row in rows}
        if len(stored_edges) != len(rows):
            raise IdentityConflict('duplicate_stored_edge_uuid')
        for edge in edges:
            stored_edge = stored_edges.get(edge.uuid)
            if stored_edge is None:
                continue
            attributes = stored_edge['properties']
            if (stored_edge['source'] != edge.source_node_uuid
                    or stored_edge['target'] != edge.target_node_uuid
                    or attributes.get('name') != edge.name
                    or attributes.get('group_id') != edge.group_id):
                raise IdentityConflict('incompatible_edge_identity')
            if any(key in edge.attributes and attributes.get(key) is not None
                   and edge.attributes[key] != attributes[key] for key in SCOPE_FIELDS):
                raise IdentityConflict('incompatible_edge_scope')
