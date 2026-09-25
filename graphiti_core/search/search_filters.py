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

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.helpers import validate_node_labels


class ComparisonOperator(Enum):
    equals = '='
    not_equals = '<>'
    greater_than = '>'
    less_than = '<'
    greater_than_equal = '>='
    less_than_equal = '<='
    is_null = 'IS NULL'
    is_not_null = 'IS NOT NULL'


class DateFilter(BaseModel):
    date: datetime | None = Field(default=None, description='A datetime to filter on')
    comparison_operator: ComparisonOperator = Field(
        description='Comparison operator for date filter'
    )


class PropertyFilter(BaseModel):
    property_name: str = Field(description='Property name')
    property_value: str | int | float | None = Field(
        default=None, description='Value you want to match on for the property'
    )
    comparison_operator: ComparisonOperator = Field(
        description='Comparison operator for the property'
    )


class SearchFilters(BaseModel):
    node_labels: list[str] | None = Field(
        default=None, description='List of node labels to filter on'
    )
    edge_types: list[str] | None = Field(
        default=None, description='List of edge types to filter on'
    )
    valid_at: list[list[DateFilter]] | None = Field(default=None)
    invalid_at: list[list[DateFilter]] | None = Field(default=None)
    created_at: list[list[DateFilter]] | None = Field(default=None)
    expired_at: list[list[DateFilter]] | None = Field(default=None)
    edge_uuids: list[str] | None = Field(default=None)
    property_filters: list[PropertyFilter] | None = Field(default=None)

    @field_validator('node_labels')
    @classmethod
    def validate_node_label_filters(cls, value: list[str] | None) -> list[str] | None:
        validate_node_labels(value)
        return value


def cypher_to_opensearch_operator(op: ComparisonOperator) -> str:
    mapping = {
        ComparisonOperator.greater_than: 'gt',
        ComparisonOperator.less_than: 'lt',
        ComparisonOperator.greater_than_equal: 'gte',
        ComparisonOperator.less_than_equal: 'lte',
    }
    return mapping.get(op, op.value)


def node_search_filter_query_constructor(
    filters: SearchFilters,
    provider: GraphProvider,
) -> tuple[list[str], dict[str, Any]]:
    filter_queries: list[str] = []
    filter_params: dict[str, Any] = {}

    if filters.node_labels is not None:
        # Defense-in-depth for model_construct()/other validation bypasses.
        validate_node_labels(filters.node_labels)
        if provider == GraphProvider.KUZU:
            node_label_filter = 'list_has_all(n.labels, $labels)'
            filter_params['labels'] = filters.node_labels
        else:
            node_labels = '|'.join(filters.node_labels)
            node_label_filter = 'n:' + node_labels
        filter_queries.append(node_label_filter)

    return filter_queries, filter_params


def date_filter_query_constructor(
    value_name: str, param_name: str, operator: ComparisonOperator
) -> str:
    query = '(' + value_name + ' '

    if operator == ComparisonOperator.is_null or operator == ComparisonOperator.is_not_null:
        query += operator.value + ')'
    else:
        query += operator.value + ' ' + param_name + ')'

    return query


def _date_filter_query(
    field: str,
    or_groups: list[list[DateFilter]],
    provider: GraphProvider,
    filter_params: dict[str, Any],
) -> str:
    # FalkorDB's range index over ISO date strings ignores upper bounds: compare outside it.
    compared = f'toString(e.{field})' if provider == GraphProvider.FALKORDB else f'e.{field}'
    or_queries: list[str] = []
    for i, and_filters in enumerate(or_groups):
        and_queries: list[str] = []
        for j, date_filter in enumerate(and_filters):
            operator = date_filter.comparison_operator
            param_name = f'{field}_{i}_{j}'
            if operator in (ComparisonOperator.is_null, ComparisonOperator.is_not_null):
                value_name = f'e.{field}'
            else:
                value_name = compared
                filter_params[param_name] = date_filter.date
            and_queries.append(
                date_filter_query_constructor(value_name, f'${param_name}', operator)
            )
        or_queries.append(' AND '.join(and_queries))
    return '(' + ' OR '.join(or_queries) + ')'


def edge_search_filter_query_constructor(
    filters: SearchFilters,
    provider: GraphProvider,
) -> tuple[list[str], dict[str, Any]]:
    filter_queries: list[str] = []
    filter_params: dict[str, Any] = {}

    if filters.edge_types is not None:
        edge_types = filters.edge_types
        filter_queries.append('e.name in $edge_types')
        filter_params['edge_types'] = edge_types

    if filters.edge_uuids is not None:
        filter_queries.append('e.uuid in $edge_uuids')
        filter_params['edge_uuids'] = filters.edge_uuids

    if filters.node_labels is not None:
        # Defense-in-depth for model_construct()/other validation bypasses.
        validate_node_labels(filters.node_labels)
        if provider == GraphProvider.KUZU:
            node_label_filter = (
                'list_has_all(n.labels, $labels) AND list_has_all(m.labels, $labels)'
            )
            filter_params['labels'] = filters.node_labels
        else:
            node_labels = '|'.join(filters.node_labels)
            node_label_filter = 'n:' + node_labels + ' AND m:' + node_labels
        filter_queries.append(node_label_filter)

    date_filters = {
        'valid_at': filters.valid_at,
        'invalid_at': filters.invalid_at,
        'created_at': filters.created_at,
        'expired_at': filters.expired_at,
    }
    for field, or_groups in date_filters.items():
        if or_groups is not None:
            filter_queries.append(_date_filter_query(field, or_groups, provider, filter_params))

    return filter_queries, filter_params
