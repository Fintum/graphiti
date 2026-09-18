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

from collections import defaultdict
from typing import Any

from graphiti_core.nodes import EntityNode
from graphiti_core.utils.maintenance.identity_constraints import IDENTITY_FIELDS


class EdgeEndpoints:
    def __init__(self, nodes: list[EntityNode]) -> None:
        self.by_uuid = {node.uuid: node for node in nodes}
        self.by_name: defaultdict[str, list[EntityNode]] = defaultdict(list)
        for node in self.by_uuid.values():
            self.by_name[node.name].append(node)

    def resolve(self, uuid: str | None, name: str) -> EntityNode | None:
        if uuid is not None:
            node = self.by_uuid.get(uuid)
            return node if node is not None and node.name == name else None
        candidates = self.by_name.get(name, [])
        return candidates[0] if len(candidates) == 1 else None

    def context(self) -> list[dict[str, Any]]:
        return [{
            'uuid': node.uuid, 'name': node.name, 'entity_types': node.labels,
            **{key: value for key, value in node.attributes.items() if key in IDENTITY_FIELDS},
        } for node in self.by_uuid.values()]
