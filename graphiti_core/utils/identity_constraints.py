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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphiti_core.nodes import EntityNode

IDENTITY_FIELDS = frozenset({'user_id', 'task_id', 'document_id', 'organization_id', 'identity_key'})
SCOPE_FIELDS = frozenset({'org_public', 'directory', 'owner_user_id', 'acl_key'})
PROTECTED_ATTRIBUTES = IDENTITY_FIELDS | SCOPE_FIELDS | frozenset({
    'provenance', 'source_key', 'revision_id', 'evidence_refs', 'policy_version',
    'last_activity_at', 'deleted_at',
})
STRUCTURAL_ATTRIBUTES = frozenset({
    'uuid', 'group_id', 'labels', 'name', 'summary', 'attributes', 'name_embedding',
    'episodes', 'source_node_uuid', 'target_node_uuid', 'source_uuid', 'target_uuid',
    'created_at', 'valid_at', 'invalid_at', 'expired_at', 'reference_time', 'fact',
    'fact_embedding',
})
IDENTITY_TYPES = {'user_id': 'Person', 'task_id': 'Task', 'document_id': 'Document', 'organization_id': 'Company'}


class IdentityConflict(ValueError):
    pass


def protected_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in attributes.items() if key in PROTECTED_ATTRIBUTES}


def inferred_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in attributes.items()
        if key not in PROTECTED_ATTRIBUTES | STRUCTURAL_ATTRIBUTES
    }


def prompt_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in attributes.items() if key not in PROTECTED_ATTRIBUTES}


def preserve_authority(prior: dict[str, Any], inferred: dict[str, Any]) -> dict[str, Any]:
    return {**inferred_attributes(inferred), **protected_attributes(prior)}


def identities_compatible(left: EntityNode, right: EntityNode) -> bool:
    if left.group_id != right.group_id:
        return False
    left_types = set(left.labels) - {'Entity'}
    right_types = set(right.labels) - {'Entity'}
    left_types.update(kind for key, kind in IDENTITY_TYPES.items() if left.attributes.get(key) is not None)
    right_types.update(kind for key, kind in IDENTITY_TYPES.items() if right.attributes.get(key) is not None)
    if left_types and right_types and left_types != right_types:
        return False
    left_ids = {key: str(left.attributes[key]) for key in IDENTITY_FIELDS if left.attributes.get(key) is not None}
    right_ids = {key: str(right.attributes[key]) for key in IDENTITY_FIELDS if right.attributes.get(key) is not None}
    if any(left_ids[key] != right_ids[key] for key in left_ids.keys() & right_ids.keys()):
        return False
    if left.uuid == right.uuid:
        return True
    if left_ids or right_ids or (left_types | right_types) & {'Person', 'Task', 'Document'}:
        return False
    return True


def require_same_identity(left: EntityNode, right: EntityNode) -> None:
    for node in (left, right):
        declared = set(node.labels) - {'Entity'}
        implied = {kind for key, kind in IDENTITY_TYPES.items() if node.attributes.get(key) is not None}
        if len(implied) > 1 or (declared and implied and declared != implied):
            raise IdentityConflict('incompatible_identity_type')
    if not identities_compatible(left, right):
        raise IdentityConflict('incompatible_entity_identity')


def scopes_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key, False if key in {'org_public', 'directory'} else None)
               == right.get(key, False if key in {'org_public', 'directory'} else None) for key in SCOPE_FIELDS)


def merge_provenance(prior: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(prior)
    for key in ('provenance', 'evidence_refs'):
        previous, current = prior.get(key, []), incoming.get(key, [])
        if isinstance(previous, list) and isinstance(current, list) and (previous or current):
            merged[key] = list(dict.fromkeys([*previous, *current]))
    return merged
