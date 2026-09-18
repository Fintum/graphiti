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

from graphiti_core.nodes import EntityNode
from graphiti_core.utils.maintenance.identity_constraints import require_same_identity


def inject_known_nodes(
    extracted: list[EntityNode], known: list[EntityNode], document: EntityNode | None = None,
) -> list[EntityNode]:
    nodes: dict[str, EntityNode] = {}
    for node in ([document] if document is not None else []) + known + extracted:
        if node.uuid in nodes:
            require_same_identity(nodes[node.uuid], node)
        else:
            nodes[node.uuid] = node
    return list(nodes.values())
