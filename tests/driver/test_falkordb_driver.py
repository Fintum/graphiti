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
import gc
import logging
import os
import unittest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices

try:
    from graphiti_core.driver.falkordb_driver import (
        FalkorDriver,
        FalkorDriverSession,
        describe_index_query,
        index_already_exists_error,
    )

    HAS_FALKORDB = True
except ImportError:
    FalkorDriver = None
    HAS_FALKORDB = False


class TestFalkorDriver:
    """Comprehensive test suite for FalkorDB driver."""

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB'):
            self.driver = FalkorDriver()
        self.driver.client = self.mock_client

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_init_with_connection_params(self):
        """Test initialization with connection parameters."""
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB') as mock_falkor_db:
            driver = FalkorDriver(
                host='test-host', port='1234', username='test-user', password='test-pass'
            )
            assert driver.provider == GraphProvider.FALKORDB
            mock_falkor_db.assert_called_once_with(
                host='test-host', port='1234', username='test-user', password='test-pass'
            )

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_init_with_falkor_db_instance(self):
        """Test initialization with a FalkorDB instance."""
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB') as mock_falkor_db_class:
            mock_falkor_db = MagicMock()
            driver = FalkorDriver(falkor_db=mock_falkor_db)
            assert driver.provider == GraphProvider.FALKORDB
            assert driver.client is mock_falkor_db
            mock_falkor_db_class.assert_not_called()

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_provider(self):
        """Test driver provider identification."""
        assert self.driver.provider == GraphProvider.FALKORDB

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_get_graph_with_name(self):
        """Test _get_graph with specific graph name."""
        mock_graph = MagicMock()
        self.mock_client.select_graph.return_value = mock_graph

        result = self.driver._get_graph('test_graph')

        self.mock_client.select_graph.assert_called_once_with('test_graph')
        assert result is mock_graph

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_get_graph_with_none_defaults_to_default_database(self):
        """Test _get_graph with None defaults to default_db."""
        mock_graph = MagicMock()
        self.mock_client.select_graph.return_value = mock_graph

        result = self.driver._get_graph(None)

        self.mock_client.select_graph.assert_called_once_with('default_db')
        assert result is mock_graph

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_success(self):
        """Test successful query execution."""
        mock_graph = MagicMock()
        mock_result = MagicMock()
        mock_result.header = [('col1', 'column1'), ('col2', 'column2')]
        mock_result.result_set = [['row1col1', 'row1col2']]
        mock_graph.query = AsyncMock(return_value=mock_result)
        self.mock_client.select_graph.return_value = mock_graph

        result = await self.driver.execute_query('MATCH (n) RETURN n', param1='value1')

        mock_graph.query.assert_called_once_with('MATCH (n) RETURN n', {'param1': 'value1'})

        result_set, header, summary = result
        assert result_set == [{'column1': 'row1col1', 'column2': 'row1col2'}]
        assert header == ['column1', 'column2']
        assert summary is None

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_handles_index_already_exists_error(self):
        """Test handling of 'already indexed' error."""
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=Exception('Index already indexed'))
        self.mock_client.select_graph.return_value = mock_graph

        with patch('graphiti_core.driver.falkordb_driver.logger') as mock_logger:
            result = await self.driver.execute_query('CREATE INDEX ...')

            mock_logger.info.assert_called_once()
            assert result is None

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_propagates_other_exceptions(self):
        """Test that other exceptions are properly propagated."""
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(side_effect=Exception('Other error'))
        self.mock_client.select_graph.return_value = mock_graph

        with patch('graphiti_core.driver.falkordb_driver.logger') as mock_logger:
            with pytest.raises(Exception, match='Other error'):
                await self.driver.execute_query('INVALID QUERY')

            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_converts_datetime_parameters(self):
        """Test that datetime objects in kwargs are converted to ISO strings."""
        mock_graph = MagicMock()
        mock_result = MagicMock()
        mock_result.header = []
        mock_result.result_set = []
        mock_graph.query = AsyncMock(return_value=mock_result)
        self.mock_client.select_graph.return_value = mock_graph

        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        await self.driver.execute_query(
            'CREATE (n:Node) SET n.created_at = $created_at', created_at=test_datetime
        )

        call_args = mock_graph.query.call_args[0]
        assert call_args[1]['created_at'] == test_datetime.isoformat()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_query_strips_nul_bytes_from_parameters(self):
        mock_graph = MagicMock()
        mock_result = MagicMock()
        mock_result.header = []
        mock_result.result_set = []
        mock_graph.query = AsyncMock(return_value=mock_result)
        self.mock_client.select_graph.return_value = mock_graph

        await self.driver.execute_query(
            'CREATE (n:Node) SET n.content = $content',
            content='Seamless Recruiting \x00 Onboarding',
            nested={'values': ['ok\x00', 'clean']},
        )

        call_args = mock_graph.query.call_args[0]
        assert call_args[1]['content'] == 'Seamless Recruiting  Onboarding'
        assert call_args[1]['nested'] == {'values': ['ok', 'clean']}

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_session_creation(self):
        """Test session creation with specific database."""
        mock_graph = MagicMock()
        self.mock_client.select_graph.return_value = mock_graph

        session = self.driver.session()

        assert isinstance(session, FalkorDriverSession)
        assert session.graph is mock_graph

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_session_creation_with_none_uses_default_database(self):
        """Test session creation with None uses default database."""
        mock_graph = MagicMock()
        self.mock_client.select_graph.return_value = mock_graph

        session = self.driver.session()

        assert isinstance(session, FalkorDriverSession)

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_close_calls_connection_close(self):
        """Test driver close method calls connection close."""
        mock_connection = MagicMock()
        mock_connection.close = AsyncMock()
        self.mock_client.connection = mock_connection

        # Ensure hasattr checks work correctly
        del self.mock_client.aclose  # Remove aclose if it exists

        with patch('builtins.hasattr') as mock_hasattr:
            # hasattr(self.client, 'aclose') returns False
            # hasattr(self.client.connection, 'aclose') returns False
            # hasattr(self.client.connection, 'close') returns True
            mock_hasattr.side_effect = lambda obj, attr: (
                attr == 'close' and obj is mock_connection
            )

            await self.driver.close()

        mock_connection.close.assert_called_once()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_delete_all_indexes(self):
        """Test delete_all_indexes method."""
        with patch.object(self.driver, 'execute_query', new_callable=AsyncMock) as mock_execute:
            # Return None to simulate no indexes found
            mock_execute.return_value = None

            await self.driver.delete_all_indexes()

            mock_execute.assert_called_once_with('CALL db.indexes()')


class TestFalkorDriverSession:
    """Test FalkorDB driver session functionality."""

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_graph = MagicMock()
        self.session = FalkorDriverSession(self.mock_graph)

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_session_async_context_manager(self):
        """Test session can be used as async context manager."""
        async with self.session as s:
            assert s is self.session

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_close_method(self):
        """Test session close method doesn't raise exceptions."""
        await self.session.close()  # Should not raise

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_execute_write_passes_session_and_args(self):
        """Test execute_write method passes session and arguments correctly."""

        async def test_func(session, *args, **kwargs):
            assert session is self.session
            assert args == ('arg1', 'arg2')
            assert kwargs == {'key': 'value'}
            return 'result'

        result = await self.session.execute_write(test_func, 'arg1', 'arg2', key='value')
        assert result == 'result'

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_run_single_query_with_parameters(self):
        """Test running a single query with parameters."""
        self.mock_graph.query = AsyncMock()

        await self.session.run('MATCH (n) RETURN n', param1='value1', param2='value2')

        self.mock_graph.query.assert_called_once_with(
            'MATCH (n) RETURN n', {'param1': 'value1', 'param2': 'value2'}
        )

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_run_multiple_queries_as_list(self):
        """Test running multiple queries passed as list."""
        self.mock_graph.query = AsyncMock()

        queries = [
            ('MATCH (n) RETURN n', {'param1': 'value1'}),
            ('CREATE (n:Node)', {'param2': 'value2'}),
        ]

        await self.session.run(queries)

        assert self.mock_graph.query.call_count == 2
        calls = self.mock_graph.query.call_args_list
        assert calls[0][0] == ('MATCH (n) RETURN n', {'param1': 'value1'})
        assert calls[1][0] == ('CREATE (n:Node)', {'param2': 'value2'})

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_run_converts_datetime_objects_to_iso_strings(self):
        """Test that datetime objects are converted to ISO strings."""
        self.mock_graph.query = AsyncMock()
        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        await self.session.run(
            'CREATE (n:Node) SET n.created_at = $created_at', created_at=test_datetime
        )

        self.mock_graph.query.assert_called_once()
        call_args = self.mock_graph.query.call_args[0]
        assert call_args[1]['created_at'] == test_datetime.isoformat()

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_run_strips_nul_bytes_from_list_query_parameters(self):
        self.mock_graph.query = AsyncMock()

        queries = [
            (
                'CREATE (n:Node) SET n.content = $content',
                {'content': 'a\x00b', 'items': ('x\x00', 'y')},
            )
        ]

        await self.session.run(queries)

        self.mock_graph.query.assert_called_once_with(
            'CREATE (n:Node) SET n.content = $content',
            {'content': 'ab', 'items': ('x', 'y')},
        )


class TestDatetimeConversion:
    """Test datetime conversion utility function."""

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_convert_datetime_dict(self):
        """Test datetime conversion in nested dictionary."""
        from graphiti_core.driver.falkordb_driver import convert_datetimes_to_strings

        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        input_dict = {
            'string_val': 'test',
            'datetime_val': test_datetime,
            'nested_dict': {'nested_datetime': test_datetime, 'nested_string': 'nested_test'},
        }

        result = convert_datetimes_to_strings(input_dict)

        assert result['string_val'] == 'test'
        assert result['datetime_val'] == test_datetime.isoformat()
        assert result['nested_dict']['nested_datetime'] == test_datetime.isoformat()
        assert result['nested_dict']['nested_string'] == 'nested_test'

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_convert_datetime_list_and_tuple(self):
        """Test datetime conversion in lists and tuples."""
        from graphiti_core.driver.falkordb_driver import convert_datetimes_to_strings

        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Test list
        input_list = ['test', test_datetime, ['nested', test_datetime]]
        result_list = convert_datetimes_to_strings(input_list)
        assert result_list[0] == 'test'
        assert result_list[1] == test_datetime.isoformat()
        assert result_list[2][1] == test_datetime.isoformat()

        # Test tuple
        input_tuple = ('test', test_datetime)
        result_tuple = convert_datetimes_to_strings(input_tuple)
        assert isinstance(result_tuple, tuple)
        assert result_tuple[0] == 'test'
        assert result_tuple[1] == test_datetime.isoformat()

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_convert_single_datetime(self):
        """Test datetime conversion for single datetime object."""
        from graphiti_core.driver.falkordb_driver import convert_datetimes_to_strings

        test_datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = convert_datetimes_to_strings(test_datetime)
        assert result == test_datetime.isoformat()

    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    def test_convert_other_types_unchanged(self):
        """Test that non-datetime types are returned unchanged."""
        from graphiti_core.driver.falkordb_driver import convert_datetimes_to_strings

        assert convert_datetimes_to_strings('string') == 'string'
        assert convert_datetimes_to_strings(123) == 123
        assert convert_datetimes_to_strings(None) is None
        assert convert_datetimes_to_strings(True) is True


INDEX_QUERIES = (
    get_range_indices(GraphProvider.FALKORDB) + get_fulltext_indices(GraphProvider.FALKORDB)
    if HAS_FALKORDB
    else []
)
INDEX_QUERY_COUNT = len(INDEX_QUERIES)
RELATES_TO_FULLTEXT_QUERY = INDEX_QUERIES[-1] if INDEX_QUERIES else ''
LOGGER_NAME = 'graphiti_core.driver.falkordb_driver'


class RecordingExecutor:
    """Stands in for FalkorDriver.execute_query, recording queries and failing on demand."""

    def __init__(self, failing_positions: set[int] | None = None, error: Exception | None = None):
        self.queries: list[str] = []
        self.failing_positions = failing_positions or set()
        self.error = error or Exception('index rejected by FalkorDB')

    async def __call__(self, query: str, **kwargs: Any):
        self.queries.append(query)
        # Yield so concurrent callers actually get a chance to interleave.
        await asyncio.sleep(0)
        if len(self.queries) in self.failing_positions:
            raise self.error
        return [], [], None

    @property
    def index_queries(self) -> list[str]:
        return [query for query in self.queries if query in INDEX_QUERIES]


def build_driver(client: MagicMock | None = None, database: str = 'default_db'):
    """FalkorDriver wired to a recording executor, with index deletion stubbed out."""
    client = client if client is not None else MagicMock()
    driver = FalkorDriver(falkor_db=client, database=database)
    executor = RecordingExecutor()
    driver.execute_query = executor  # type: ignore[method-assign]
    driver.delete_all_indexes = AsyncMock()  # type: ignore[method-assign]
    return driver, executor


@unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
class TestFalkorDriverIndexLifecycle:
    """Index creation must be caller-driven, resilient, idempotent and loop-agnostic."""

    @pytest.mark.asyncio
    async def test_construction_schedules_no_background_work(self):
        """Constructing inside a running loop must not spawn an unsupervised task."""
        client = MagicMock()
        client.aclose = AsyncMock()
        tasks_before = asyncio.all_tasks()

        driver = FalkorDriver(falkor_db=client)
        await asyncio.sleep(0)

        assert asyncio.all_tasks() == tasks_before
        client.select_graph.assert_not_called()

        await driver.close()
        await asyncio.sleep(0)

        assert asyncio.all_tasks() == tasks_before

    @pytest.mark.asyncio
    async def test_construction_leaves_no_unretrieved_exception(self):
        """No background work means nothing can die with 'Task exception was never retrieved'."""
        loop = asyncio.get_running_loop()
        unhandled: list[dict] = []
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

        try:
            client = MagicMock()
            client.aclose = AsyncMock()
            driver = FalkorDriver(falkor_db=client)
            await driver.close()

            for _ in range(3):
                await asyncio.sleep(0)
            gc.collect()
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(None)

        assert unhandled == []

    @pytest.mark.asyncio
    async def test_every_index_is_attempted_when_one_fails(self, caplog):
        """A rejected index must not hide the twelve queued behind it."""
        driver, _ = build_driver()
        executor = RecordingExecutor(failing_positions={2}, error=Exception('boom on index 2'))
        driver.execute_query = executor  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            await driver.build_indices_and_constraints()

        assert executor.queries == list(INDEX_QUERIES)
        assert len(executor.queries) == INDEX_QUERY_COUNT
        assert RELATES_TO_FULLTEXT_QUERY in executor.queries
        assert executor.queries[-1] == RELATES_TO_FULLTEXT_QUERY

        warnings = [
            record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING
        ]
        assert any('index on Episodic' in message for message in warnings)

    @pytest.mark.asyncio
    async def test_partial_failure_does_not_mark_the_build_as_done(self):
        """A build that lost an index stays retryable instead of caching the hole."""
        driver, _ = build_driver()
        executor = RecordingExecutor(failing_positions={2})
        driver.execute_query = executor  # type: ignore[method-assign]

        await driver.build_indices_and_constraints()
        await driver.build_indices_and_constraints()

        assert len(executor.queries) == 2 * INDEX_QUERY_COUNT

    @pytest.mark.asyncio
    async def test_indices_built_reports_a_complete_build(self):
        """Consumers memoizing the build need to tell a complete one from a partial one."""
        driver, _ = build_driver()
        driver.execute_query = RecordingExecutor()  # type: ignore[method-assign]

        assert driver.indices_built is False
        await driver.build_indices_and_constraints()
        assert driver.indices_built is True

    @pytest.mark.asyncio
    async def test_indices_built_stays_false_after_a_partial_build(self):
        """A missing index must not look like a finished build to the consumer."""
        driver, _ = build_driver()
        driver.execute_query = RecordingExecutor(failing_positions={2})  # type: ignore[method-assign]

        await driver.build_indices_and_constraints()

        assert driver.indices_built is False

    @pytest.mark.asyncio
    async def test_total_failure_raises(self):
        """Nothing getting through is a dead connection, not a rejected index."""
        driver, _ = build_driver()
        executor = RecordingExecutor(
            failing_positions=set(range(1, INDEX_QUERY_COUNT + 1)),
            error=ConnectionError('Connection closed by server'),
        )
        driver.execute_query = executor  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match='Failed to create any index'):
            await driver.build_indices_and_constraints()

        assert len(executor.queries) == INDEX_QUERY_COUNT

    @pytest.mark.asyncio
    async def test_repeated_build_runs_the_queries_once(self):
        driver, executor = build_driver()

        await driver.build_indices_and_constraints()
        await driver.build_indices_and_constraints()
        await driver.build_indices_and_constraints()

        assert len(executor.queries) == INDEX_QUERY_COUNT

    @pytest.mark.asyncio
    async def test_delete_existing_forces_a_rebuild(self):
        driver, executor = build_driver()

        await driver.build_indices_and_constraints()
        await driver.build_indices_and_constraints(delete_existing=True)

        assert len(executor.queries) == 2 * INDEX_QUERY_COUNT
        driver.delete_all_indexes.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_concurrent_builds_run_the_queries_once(self):
        driver, executor = build_driver()

        await asyncio.gather(
            driver.build_indices_and_constraints(),
            driver.build_indices_and_constraints(),
            driver.build_indices_and_constraints(),
        )

        assert len(executor.queries) == INDEX_QUERY_COUNT

    @pytest.mark.asyncio
    async def test_driver_instances_sharing_client_and_database_build_once(self):
        """clone() and per-request drivers reuse the guard instead of re-running the storm."""
        client = MagicMock()
        first, first_executor = build_driver(client=client, database='org_42')
        second, second_executor = build_driver(client=client, database='org_42')

        await first.build_indices_and_constraints()
        await second.build_indices_and_constraints()

        assert len(first_executor.queries) == INDEX_QUERY_COUNT
        assert second_executor.queries == []

    @pytest.mark.asyncio
    async def test_databases_are_guarded_independently(self):
        """One graph per organisation: building org A must not skip org B."""
        client = MagicMock()
        org_a, executor_a = build_driver(client=client, database='org_a')
        org_b, executor_b = build_driver(client=client, database='org_b')

        await org_a.build_indices_and_constraints()
        await org_b.build_indices_and_constraints()

        assert len(executor_a.queries) == INDEX_QUERY_COUNT
        assert len(executor_b.queries) == INDEX_QUERY_COUNT

    @pytest.mark.asyncio
    async def test_clients_are_guarded_independently(self):
        first, first_executor = build_driver(client=MagicMock())
        second, second_executor = build_driver(client=MagicMock())

        await first.build_indices_and_constraints()
        await second.build_indices_and_constraints()

        assert len(first_executor.queries) == INDEX_QUERY_COUNT
        assert len(second_executor.queries) == INDEX_QUERY_COUNT

    def test_build_survives_many_event_loops(self):
        """The worker pattern: one asyncio.run() per job, many loops in one process."""
        driver, _ = build_driver(database='worker_db')
        executor = RecordingExecutor()
        driver.execute_query = executor  # type: ignore[method-assign]

        async def build_twice_concurrently():
            # Contending for the lock is what forces it to bind to the running loop.
            await asyncio.gather(
                driver.build_indices_and_constraints(delete_existing=True),
                driver.build_indices_and_constraints(delete_existing=True),
            )

        for expected_runs in (2, 4, 6):
            asyncio.run(build_twice_concurrently())
            assert len(executor.queries) == expected_runs * INDEX_QUERY_COUNT

    @pytest.mark.asyncio
    async def test_index_already_exists_is_tolerated_end_to_end(self):
        """Wordings execute_query does not swallow are still fine for index creation."""
        client = MagicMock()
        graph = MagicMock()
        graph.query = AsyncMock(side_effect=Exception('Index already exists'))
        client.select_graph.return_value = graph
        driver = FalkorDriver(falkor_db=client, database='already_indexed_db')

        await driver.build_indices_and_constraints()

        assert graph.query.await_count == INDEX_QUERY_COUNT

        # The build counts as complete, so it is not retried.
        await driver.build_indices_and_constraints()
        assert graph.query.await_count == INDEX_QUERY_COUNT

    @pytest.mark.parametrize(
        'message',
        [
            "Attribute 'name' is already indexed",
            'Index already exists',
            'Fulltext index already exists',
            'Attribute has already been indexed',
            'ALREADY INDEXED',
        ],
    )
    def test_index_already_exists_error_matches_falkordb_wordings(self, message):
        assert index_already_exists_error(Exception(message)) is True

    @pytest.mark.parametrize(
        'message',
        [
            'Connection closed by server',
            'Syntax error at offset 12',
            'Redis is loading the dataset in memory',
        ],
    )
    def test_index_already_exists_error_rejects_real_failures(self, message):
        assert index_already_exists_error(Exception(message)) is False

    def test_describe_index_query_names_the_index(self):
        assert describe_index_query(INDEX_QUERIES[0]) == 'index on Entity'
        assert describe_index_query(RELATES_TO_FULLTEXT_QUERY) == 'fulltext index on RELATES_TO'
        assert describe_index_query(INDEX_QUERIES[9]) == 'fulltext index on Episodic'


# Simple integration test
class TestFalkorDriverIntegration:
    """Simple integration test for FalkorDB driver."""

    @pytest.mark.asyncio
    @unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
    async def test_basic_integration_with_real_falkordb(self):
        """Basic integration test with real FalkorDB instance."""
        pytest.importorskip('falkordb')

        falkor_host = os.getenv('FALKORDB_HOST', 'localhost')
        falkor_port = os.getenv('FALKORDB_PORT', '6379')

        try:
            driver = FalkorDriver(host=falkor_host, port=falkor_port)

            # Test basic query execution
            result = await driver.execute_query('RETURN 1 as test')
            assert result is not None

            result_set, header, summary = result
            assert header == ['test']
            assert result_set == [{'test': 1}]

            await driver.close()

        except Exception as e:
            pytest.skip(f'FalkorDB not available for integration test: {e}')
