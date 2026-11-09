"""Copies of an extracted name that the candidate cut left out, offered if the ACL admits them.

The similarity search keeps the closest candidates and the ACL filters them afterwards, so enough
hidden copies of a name crowd a visible one out of the cut. A full page that ends on the name is
the sign: its copies are then read by name as light rows, oldest first, admitted by tranches and
loaded whole only when admitted, because a whole node carries its vector and costs a few
milliseconds a row. The step only adds candidates: when it fails, the similarity ones stay.
"""

import logging
from typing import Any

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact, by_age
from graphiti_core.utils.maintenance.enrichment_acl import EnrichmentAcl, acl_nodes
from graphiti_core.utils.maintenance.identity_constraints import (
    IDENTITY_FIELDS,
    identities_compatible,
)

logger = logging.getLogger(__name__)

MAX_COPIES_PER_NAME = 5000
ADMISSION_TRANCHE = 250
NAMED_PROVIDERS = (GraphProvider.FALKORDB, GraphProvider.NEO4J)
IDENTITY_COLUMNS = ', '.join(f'{field}: n.{field}' for field in sorted(IDENTITY_FIELDS))

STORED_COPIES = f"""
MATCH (n:Entity)
WHERE n.group_id IN $group_ids AND toLower(n.name) IN $names
WITH n ORDER BY n.created_at, n.uuid
WITH n.group_id AS group_id, toLower(n.name) AS spelling, collect({{
    uuid: n.uuid, name: n.name, created_at: n.created_at, labels: labels(n), {IDENTITY_COLUMNS}
}}) AS copies
RETURN group_id, size(copies) AS stored, copies[..$limit] AS copies
"""

CopyKey = tuple[str, str]


def copy_key(node: EntityNode) -> CopyKey:
    return node.group_id, _normalize_string_exact(node.name)


def crowded(page: list[EntityNode], node: EntityNode, limit: int) -> bool:
    """A full page that ends on the node's own name may have cut off more copies of it."""
    return len(page) >= limit and copy_key(page[-1]) == copy_key(node)


def lowered_spellings(name: str) -> set[str]:
    """How a stored copy of the name reads lowercased: as written, spaces collapsed, and folded."""
    lowered = name.lower()
    return {lowered, ' '.join(lowered.split()), _normalize_string_exact(name)}


async def stored_copies(
    driver: GraphDriver, nodes: list[EntityNode]
) -> dict[CopyKey, list[EntityNode]]:
    """Stubs of the oldest copies of each node's name, by copy key; none if the lookup fails."""
    if not nodes or driver.provider not in NAMED_PROVIDERS:
        return {}
    try:
        records, _, _ = await driver.execute_query(
            STORED_COPIES,
            group_ids=sorted({node.group_id for node in nodes}),
            names=sorted({spelling for node in nodes for spelling in lowered_spellings(node.name)}),
            limit=MAX_COPIES_PER_NAME,
            routing_='r',
        )
        return _by_name(records)
    except Exception:
        logger.exception(
            'Looking up the copies of crowded names failed; keeping the similarity candidates'
        )
        return {}


async def admitted_copies(
    driver: GraphDriver,
    acl: EnrichmentAcl | None,
    node: EntityNode,
    stubs: list[EntityNode],
    offered: set[str],
    limit: int,
) -> list[EntityNode]:
    """The oldest compatible copies not offered yet that the ACL admits, loaded as it shows them."""
    fresh = [
        stub for stub in stubs if stub.uuid not in offered and identities_compatible(node, stub)
    ]
    try:
        admitted = await _first_admitted(acl, fresh, limit)
        if not admitted:
            return []
        loaded = await EntityNode.get_by_uuids(driver, [stub.uuid for stub in admitted])
        return sorted(await acl_nodes(acl, loaded), key=by_age)
    except Exception:
        logger.exception(
            'Admitting the copies of a crowded name failed; keeping the similarity candidates'
        )
        return []


async def _first_admitted(
    acl: EnrichmentAcl | None, stubs: list[EntityNode], limit: int
) -> list[EntityNode]:
    """Admit the stubs oldest first, a tranche at a time, until the limit is met or they run out."""
    admitted: list[EntityNode] = []
    for start in range(0, len(stubs), ADMISSION_TRANCHE):
        admitted += await acl_nodes(acl, stubs[start : start + ADMISSION_TRANCHE])
        if len(admitted) >= limit:
            break
    return sorted(admitted, key=by_age)[:limit]


def _by_name(records: list[dict[str, Any]]) -> dict[CopyKey, list[EntityNode]]:
    copies: dict[CopyKey, dict[str, EntityNode]] = {}
    for record in records:
        if record['stored'] > MAX_COPIES_PER_NAME:
            logger.warning(
                'A name has %d stored copies in %s; only the oldest %d are offered',
                record['stored'],
                record['group_id'],
                MAX_COPIES_PER_NAME,
            )
        for copy in record['copies']:
            if copy['created_at'] is not None:
                stub = _stub(copy, record['group_id'])
                copies.setdefault(copy_key(stub), {})[stub.uuid] = stub
    return {
        key: sorted(found.values(), key=by_age)[:MAX_COPIES_PER_NAME]
        for key, found in copies.items()
    }


def _stub(copy: dict[str, Any], group_id: str) -> EntityNode:
    """What admission and identity need of a copy: labels and identity fields, no text or vector."""
    return EntityNode(
        uuid=copy['uuid'],
        name=copy['name'],
        group_id=group_id,
        labels=[
            label for label in copy['labels'] if label != 'Entity_' + group_id.replace('-', '')
        ],
        created_at=parse_db_date(copy['created_at']),  # type: ignore[arg-type]
        attributes={field: copy[field] for field in IDENTITY_FIELDS if copy.get(field) is not None},
    )
