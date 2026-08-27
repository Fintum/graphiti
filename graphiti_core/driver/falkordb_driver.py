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

import asyncio
import datetime
import logging
import re
import threading
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from falkordb import Graph as FalkorGraph
    from falkordb.asyncio import FalkorDB
else:
    try:
        from falkordb import Graph as FalkorGraph
        from falkordb.asyncio import FalkorDB
    except ImportError:
        # If falkordb is not installed, raise an ImportError
        raise ImportError(
            'falkordb is required for FalkorDriver. '
            'Install it with: pip install graphiti-core[falkordb]'
        ) from None

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.falkordb import STOPWORDS as STOPWORDS
from graphiti_core.driver.falkordb.operations.community_edge_ops import (
    FalkorCommunityEdgeOperations,
)
from graphiti_core.driver.falkordb.operations.community_node_ops import (
    FalkorCommunityNodeOperations,
)
from graphiti_core.driver.falkordb.operations.entity_edge_ops import FalkorEntityEdgeOperations
from graphiti_core.driver.falkordb.operations.entity_node_ops import FalkorEntityNodeOperations
from graphiti_core.driver.falkordb.operations.episode_node_ops import FalkorEpisodeNodeOperations
from graphiti_core.driver.falkordb.operations.episodic_edge_ops import FalkorEpisodicEdgeOperations
from graphiti_core.driver.falkordb.operations.graph_ops import FalkorGraphMaintenanceOperations
from graphiti_core.driver.falkordb.operations.has_episode_edge_ops import (
    FalkorHasEpisodeEdgeOperations,
)
from graphiti_core.driver.falkordb.operations.next_episode_edge_ops import (
    FalkorNextEpisodeEdgeOperations,
)
from graphiti_core.driver.falkordb.operations.saga_node_ops import FalkorSagaNodeOperations
from graphiti_core.driver.falkordb.operations.search_ops import FalkorSearchOperations
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices, get_vector_indices
from graphiti_core.helpers import validate_group_ids
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings

logger = logging.getLogger(__name__)

# FalkorDB has no CREATE INDEX ... IF NOT EXISTS, so re-creating an index is reported as an
# error whose wording depends on the index kind and the server version ("Attribute 'x' is
# already indexed", "Index already exists", ...). Matching "already indexed"/"already exists"
# instead of one literal substring keeps index creation tolerant across those variants.
_INDEX_ALREADY_EXISTS_PATTERN = re.compile(r'already\s+(?:been\s+)?(?:index|exist)', re.IGNORECASE)

_INDEX_LABEL_PATTERN = re.compile(r"label:\s*'([^']+)'")
_INDEX_ENTITY_PATTERN = re.compile(r'[(\[][ne]:(\w+)')

# Guards index creation per (FalkorDB client, database). Keyed weakly so that drivers and
# their clients stay collectable; the mutex is a plain threading lock because the registry is
# also read from worker threads that each drive their own event loop.
_INDEX_STATE_MUTEX = threading.RLock()
_INDEX_STATE: 'WeakKeyDictionary[Any, dict[str, _IndexBuildState]]' = WeakKeyDictionary()


def index_already_exists_error(error: BaseException) -> bool:
    """Whether a FalkorDB error only means the index is already there."""
    return _INDEX_ALREADY_EXISTS_PATTERN.search(str(error)) is not None


def describe_index_query(query: str) -> str:
    """Short log-friendly identifier for an index creation query."""
    collapsed = ' '.join(query.split())
    kind = 'fulltext index' if 'FULLTEXT' in collapsed.upper() else 'index'

    label_match = _INDEX_LABEL_PATTERN.search(collapsed)
    if label_match:
        return f'{kind} on {label_match.group(1)}'

    entity_match = _INDEX_ENTITY_PATTERN.search(collapsed)
    if entity_match:
        return f'{kind} on {entity_match.group(1)}'

    return collapsed[:100]


class _IndexBuildState:
    """Bookkeeping shared by every driver pointing at the same (client, database)."""

    __slots__ = ('built', '_locks')

    def __init__(self) -> None:
        self.built = False
        self._locks: list[tuple[asyncio.AbstractEventLoop, asyncio.Lock]] = []

    def lock_for_running_loop(self) -> asyncio.Lock:
        """Return the lock bound to the running loop, creating it on first use there.

        An asyncio.Lock binds itself to whichever loop first awaits it, so a single shared
        lock would raise "attached to a different loop" as soon as the process starts a
        second loop. Workers call asyncio.run() once per job, i.e. many short-lived loops in
        one process, so a lock is kept per live loop instead. Closed loops are pruned on
        access, which keeps the list at one entry per loop actually running right now.
        """
        loop = asyncio.get_running_loop()
        with _INDEX_STATE_MUTEX:
            self._locks = [entry for entry in self._locks if not entry[0].is_closed()]
            for bound_loop, lock in self._locks:
                if bound_loop is loop:
                    return lock

            lock = asyncio.Lock()
            self._locks.append((loop, lock))
            return lock


def _get_index_build_state(client: Any, database: str) -> _IndexBuildState:
    with _INDEX_STATE_MUTEX:
        try:
            per_database = _INDEX_STATE.setdefault(client, {})
        except TypeError:
            # Client cannot be weakly referenced: fall back to state attached to the client.
            per_database = client.__dict__.setdefault('_graphiti_index_build_state', {})

        state = per_database.get(database)
        if state is None:
            state = _IndexBuildState()
            per_database[database] = state

        return state


def _strip_nul_bytes(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace('\x00', '')
    if isinstance(value, dict):
        return {key: _strip_nul_bytes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_nul_bytes(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_nul_bytes(item) for item in value)
    return value


class FalkorDriverSession(GraphDriverSession):
    provider = GraphProvider.FALKORDB

    def __init__(self, graph: FalkorGraph):
        self.graph = graph

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # No cleanup needed for Falkor, but method must exist
        pass

    async def close(self):
        # No explicit close needed for FalkorDB, but method must exist
        pass

    async def execute_write(self, func, *args, **kwargs):
        # Directly await the provided async function with `self` as the transaction/session
        return await func(self, *args, **kwargs)

    async def run(self, query: str | list, **kwargs: Any) -> Any:
        # FalkorDB does not support argument for Label Set, so it's converted into an array of queries
        if isinstance(query, list):
            for cypher, params in query:
                params = convert_datetimes_to_strings(params)
                params = _strip_nul_bytes(params)
                await self.graph.query(str(cypher), params)  # type: ignore[reportUnknownArgumentType]
        else:
            params = dict(kwargs)
            params = convert_datetimes_to_strings(params)
            params = _strip_nul_bytes(params)
            await self.graph.query(str(query), params)  # type: ignore[reportUnknownArgumentType]
        # Assuming `graph.query` is async (ideal); otherwise, wrap in executor
        return None


class FalkorDriver(GraphDriver):
    provider = GraphProvider.FALKORDB
    default_group_id: str = '_'
    fulltext_syntax: str = '@'  # FalkorDB uses a redisearch-like syntax for fulltext queries
    aoss_client: None = None

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        username: str | None = None,
        password: str | None = None,
        falkor_db: FalkorDB | None = None,
        database: str = 'default_db',
    ):
        """
        Initialize the FalkorDB driver.

        FalkorDB is a multi-tenant graph database.
        To connect, provide the host and port.
        The default parameters assume a local (on-premises) FalkorDB instance.

        Args:
        host (str): The host where FalkorDB is running.
        port (int): The port on which FalkorDB is listening.
        username (str | None): The username for authentication (if required).
        password (str | None): The password for authentication (if required).
        falkor_db (FalkorDB | None): An existing FalkorDB instance to use instead of creating a new one.
        database (str): The name of the database to connect to. Defaults to 'default_db'.
        """
        super().__init__()
        self._database = database
        if falkor_db is not None:
            # If a FalkorDB instance is provided, use it directly
            self.client = falkor_db
        else:
            self.client = FalkorDB(host=host, port=port, username=username, password=password)

        # Instantiate FalkorDB operations
        self._entity_node_ops = FalkorEntityNodeOperations()
        self._episode_node_ops = FalkorEpisodeNodeOperations()
        self._community_node_ops = FalkorCommunityNodeOperations()
        self._saga_node_ops = FalkorSagaNodeOperations()
        self._entity_edge_ops = FalkorEntityEdgeOperations()
        self._episodic_edge_ops = FalkorEpisodicEdgeOperations()
        self._community_edge_ops = FalkorCommunityEdgeOperations()
        self._has_episode_edge_ops = FalkorHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = FalkorNextEpisodeEdgeOperations()
        self._search_ops = FalkorSearchOperations()
        self._graph_ops = FalkorGraphMaintenanceOperations()

        # Index creation is NOT scheduled here. Constructing a driver must not start
        # unsupervised work: a fire-and-forget task outlives close(), dies with
        # "Task exception was never retrieved", and turns every instantiation into a write
        # storm against FalkorDB. Callers own the lifecycle and await
        # build_indices_and_constraints() explicitly (it is idempotent, see below).

    # --- Operations properties ---

    @property
    def entity_node_ops(self) -> EntityNodeOperations:
        return self._entity_node_ops

    @property
    def episode_node_ops(self) -> EpisodeNodeOperations:
        return self._episode_node_ops

    @property
    def community_node_ops(self) -> CommunityNodeOperations:
        return self._community_node_ops

    @property
    def saga_node_ops(self) -> SagaNodeOperations:
        return self._saga_node_ops

    @property
    def entity_edge_ops(self) -> EntityEdgeOperations:
        return self._entity_edge_ops

    @property
    def episodic_edge_ops(self) -> EpisodicEdgeOperations:
        return self._episodic_edge_ops

    @property
    def community_edge_ops(self) -> CommunityEdgeOperations:
        return self._community_edge_ops

    @property
    def has_episode_edge_ops(self) -> HasEpisodeEdgeOperations:
        return self._has_episode_edge_ops

    @property
    def next_episode_edge_ops(self) -> NextEpisodeEdgeOperations:
        return self._next_episode_edge_ops

    @property
    def search_ops(self) -> SearchOperations:
        return self._search_ops

    @property
    def graph_ops(self) -> GraphMaintenanceOperations:
        return self._graph_ops

    def _get_graph(self, graph_name: str | None) -> FalkorGraph:
        # FalkorDB requires a non-None database name for multi-tenant graphs; the default is "default_db"
        if graph_name is None:
            graph_name = self._database
        return self.client.select_graph(graph_name)

    async def execute_query(self, cypher_query_, **kwargs: Any):
        graph = self._get_graph(self._database)

        # Convert datetime objects to ISO strings (FalkorDB does not support datetime objects directly)
        params = convert_datetimes_to_strings(dict(kwargs))
        params = _strip_nul_bytes(params)

        try:
            result = await graph.query(cypher_query_, params)  # type: ignore[reportUnknownArgumentType]
        except Exception as e:
            if 'already indexed' in str(e):
                # check if index already exists
                logger.info(f'Index already exists: {e}')
                return None
            logger.error(f'Error executing FalkorDB query: {e}\n{cypher_query_}\n{params}')
            raise

        # Convert the result header to a list of strings
        header = [h[1] for h in result.header]

        # Convert FalkorDB's result format (list of lists) to the format expected by Graphiti (list of dicts)
        records = []
        for row in result.result_set:
            record = {}
            for i, field_name in enumerate(header):
                if i < len(row):
                    record[field_name] = row[i]
                else:
                    # If there are more fields in header than values in row, set to None
                    record[field_name] = None
            records.append(record)

        return records, header, None

    def session(self, database: str | None = None) -> GraphDriverSession:
        return FalkorDriverSession(self._get_graph(database))

    async def close(self) -> None:
        """Close the driver connection."""
        if hasattr(self.client, 'aclose'):
            await self.client.aclose()  # type: ignore[reportUnknownMemberType]
        elif hasattr(self.client.connection, 'aclose'):
            await self.client.connection.aclose()
        elif hasattr(self.client.connection, 'close'):
            await self.client.connection.close()

    async def delete_all_indexes(self) -> None:
        result = await self.execute_query('CALL db.indexes()')
        if not result:
            return

        records, _, _ = result
        drop_tasks = []

        for record in records:
            label = record['label']
            entity_type = record['entitytype']

            for field_name, index_type in record['types'].items():
                if 'RANGE' in index_type:
                    drop_tasks.append(self.execute_query(f'DROP INDEX ON :{label}({field_name})'))
                elif 'FULLTEXT' in index_type:
                    if entity_type == 'NODE':
                        drop_tasks.append(
                            self.execute_query(
                                f'DROP FULLTEXT INDEX FOR (n:{label}) ON (n.{field_name})'
                            )
                        )
                    elif entity_type == 'RELATIONSHIP':
                        drop_tasks.append(
                            self.execute_query(
                                f'DROP FULLTEXT INDEX FOR ()-[e:{label}]-() ON (e.{field_name})'
                            )
                        )

        if drop_tasks:
            await asyncio.gather(*drop_tasks)

    @property
    def indices_built(self) -> bool:
        """Whether the last build left every index for this database in place."""
        return _get_index_build_state(self.client, self._database).built

    async def _execute_index_query(self, query: str) -> bool:
        """Execute one index creation query, reporting whether the index ended up in place."""
        try:
            await self.execute_query(query)
            return True
        except Exception as e:
            if index_already_exists_error(e):
                logger.debug(
                    'Index already exists on %s: %s', self._database, describe_index_query(query)
                )
                return True

            logger.warning(
                'Could not create %s on FalkorDB database %s: %s',
                describe_index_query(query),
                self._database,
                e,
            )
            return False

    async def build_indices_and_constraints(
        self, delete_existing: bool = False, vector_dimension: int | None = None
    ):
        """Create the range, fulltext and vector indices, tolerating individual failures.

        Every query is attempted even if an earlier one failed, so a rejected index never
        hides the ones queued behind it. A completed build is remembered per
        (client, database), so repeat calls are free; `delete_existing` forces a rebuild.
        """
        state = _get_index_build_state(self.client, self._database)

        async with state.lock_for_running_loop():
            if state.built and not delete_existing:
                logger.debug('Indices already built for FalkorDB database %s', self._database)
                return

            if delete_existing:
                state.built = False
                await self.delete_all_indexes()

            index_queries = (
                get_range_indices(self.provider)
                + get_fulltext_indices(self.provider)
                + (get_vector_indices(self.provider, vector_dimension) if vector_dimension else [])
            )
            failed = [
                query for query in index_queries if not await self._execute_index_query(query)
            ]

            if not failed:
                state.built = True
                return

            if len(failed) == len(index_queries):
                # Nothing at all got through: this is a dead connection or an unusable
                # database, not a rejected index, and the caller must hear about it.
                raise RuntimeError(
                    f'Failed to create any index on FalkorDB database {self._database} '
                    f'({len(failed)} queries failed); see warnings above'
                )

            # Partial failure: the graph stays usable, searches backed by the missing indices
            # do not. Surface it loudly but do not abort the caller's start-up, and leave the
            # build unmarked so a later call retries the missing ones.
            logger.error(
                'Built FalkorDB indices on %s with %d of %d failures: %s',
                self._database,
                len(failed),
                len(index_queries),
                ', '.join(describe_index_query(query) for query in failed),
            )

    def clone(self, database: str) -> 'GraphDriver':
        """
        Returns a shallow copy of this driver with a different default database.
        Reuses the same connection (e.g. FalkorDB, Neo4j).
        """
        if database == self._database:
            cloned = self
        elif database == self.default_group_id:
            cloned = FalkorDriver(falkor_db=self.client)
        else:
            # Create a new instance of FalkorDriver with the same connection but a different database
            cloned = FalkorDriver(falkor_db=self.client, database=database)

        return cloned

    async def health_check(self) -> None:
        """Check FalkorDB connectivity by running a simple query."""
        try:
            await self.execute_query('MATCH (n) RETURN 1 LIMIT 1')
            return None
        except Exception as e:
            print(f'FalkorDB health check failed: {e}')
            raise

    @staticmethod
    def convert_datetimes_to_strings(obj):
        if isinstance(obj, dict):
            return {k: FalkorDriver.convert_datetimes_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FalkorDriver.convert_datetimes_to_strings(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(FalkorDriver.convert_datetimes_to_strings(item) for item in obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    def sanitize(self, query: str) -> str:
        """
        Replace FalkorDB special characters with whitespace.
        Based on FalkorDB tokenization rules: ,.<>{}[]"':;!@#$%^&*()-+=~
        """
        # FalkorDB separator characters that break text into tokens
        separator_map = str.maketrans(
            {
                ',': ' ',
                '.': ' ',
                '<': ' ',
                '>': ' ',
                '{': ' ',
                '}': ' ',
                '[': ' ',
                ']': ' ',
                '"': ' ',
                "'": ' ',
                ':': ' ',
                ';': ' ',
                '!': ' ',
                '@': ' ',
                '#': ' ',
                '$': ' ',
                '%': ' ',
                '^': ' ',
                '&': ' ',
                '*': ' ',
                '(': ' ',
                ')': ' ',
                '-': ' ',
                '+': ' ',
                '=': ' ',
                '~': ' ',
                '?': ' ',
                '|': ' ',
                '/': ' ',
                '\\': ' ',
            }
        )
        sanitized = query.translate(separator_map)
        # Clean up multiple spaces
        sanitized = ' '.join(sanitized.split())
        return sanitized

    def build_fulltext_query(
        self, query: str, group_ids: list[str] | None = None, max_query_length: int = 128
    ) -> str:
        """
        Build a fulltext query string for FalkorDB using RedisSearch syntax.
        FalkorDB uses RedisSearch-like syntax where:
        - Field queries use @ prefix: @field:value
        - Multiple values for same field: (@field:value1|value2)
        - Text search doesn't need @ prefix for content fields
        - AND is implicit with space: (@group_id:value) (text)
        - OR uses pipe within parentheses: (@group_id:value1|value2)
        """
        validate_group_ids(group_ids)

        if group_ids is None or len(group_ids) == 0:
            group_filter = ''
        else:
            # Quote group_ids and escape non-alphanumeric chars (e.g. '_' in the
            # default group_id, or hyphens). RediSearch treats these as token
            # separators/operators, which otherwise causes a syntax error or a
            # failure to match. group_ids are restricted to [a-zA-Z0-9_-] by
            # validate_group_ids above.
            escaped_group_ids = [
                '"' + re.sub(r'([^a-zA-Z0-9])', r'\\\1', gid) + '"' for gid in group_ids
            ]
            group_values = '|'.join(escaped_group_ids)
            group_filter = f'(@group_id:{group_values})'

        sanitized_query = self.sanitize(query)

        # Remove stopwords and empty tokens from the sanitized query
        query_words = sanitized_query.split()
        filtered_words = [word for word in query_words if word and word.lower() not in STOPWORDS]
        sanitized_query = ' | '.join(filtered_words)

        # If the query is too long return no query
        if len(sanitized_query.split(' ')) + len(group_ids or '') >= max_query_length:
            return ''

        full_query = group_filter + ' (' + sanitized_query + ')'

        return full_query
