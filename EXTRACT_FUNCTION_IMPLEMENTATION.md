# Implementation: `_extract_nodes_and_edges_only` Function

## Overview
The `_extract_nodes_and_edges_only` function has been fully implemented to support both low-level and high-level interfaces for extracting entities and relationships from episodes.

## Function Signature
```python
async def _extract_nodes_and_edges_only(
    self,
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    edge_type_map: dict[tuple[str, str], list[str]] | None = None,
    custom_extraction_instructions: str | None = None,
    # High-level parameters (from episode content)
    episode_body: str | None = None,
    source_description: str | None = None,
    reference_time: datetime | None = None,
    source: EpisodeType | None = None,
    group_id: str | None = None,
    uuid: str | None = None,
    saga: str | None = None,
) -> tuple[list[EntityNode], dict[str, list[int]], list[EntityEdge]]:
```

## What It Does

### 1. Validates Entity Types
```python
validate_entity_types(entity_types)
validate_excluded_entity_types(excluded_entity_types, entity_types)
```
- Ensures custom entity type definitions are valid Pydantic models
- Checks that excluded types are compatible with entity_types

### 2. Handles Group ID
```python
resolved_group_id = group_id
if resolved_group_id is None:
    resolved_group_id = get_default_group_id(self.driver.provider)
else:
    validate_group_id(resolved_group_id)
    if resolved_group_id != self.driver._database:
        self.driver = self.driver.clone(database=resolved_group_id)
        self.clients.driver = self.driver
```
- Uses default group_id if not provided
- Validates group_id format
- Clones driver if switching databases

### 3. Creates or Retrieves Episode
```python
episode = (
    await EpisodicNode.get_by_uuid(self.driver, uuid)
    if uuid is not None
    else EpisodicNode(
        name=saga or 'Episode',
        group_id=resolved_group_id,
        labels=[],
        source=source or EpisodeType.message,
        content=episode_body or '',
        source_description=source_description or '',
        created_at=utc_now(),
        valid_at=reference_time or utc_now(),
    )
)
```
- Retrieves existing episode if UUID provided
- Creates new EpisodicNode from content parameters
- Uses saga name as episode name if available

### 4. Retrieves Previous Episodes for Context
```python
previous_episodes = await self.retrieve_episodes(
    reference_time or utc_now(),
    last_n=RELEVANT_SCHEMA_LIMIT,
    group_ids=[resolved_group_id],
    source=source,
)
```
- Fetches up to `RELEVANT_SCHEMA_LIMIT` previous episodes
- Used as context for the LLM extraction
- Filters by group_id and source type

### 5. Extracts Nodes (Entities)
```python
extracted_nodes, node_episode_index_map = await extract_nodes(
    self.clients,
    episode,
    previous_episodes,
    entity_types,
    excluded_entity_types,
    custom_extraction_instructions,
)
```
- Uses LLM to identify entities in episode content
- Respects custom entity types and exclusions
- Returns mapping of node UUIDs to episode indices for attribution

### 6. Extracts Edges (Relationships)
```python
extracted_edges = await extract_edges(
    self.clients,
    episode,
    extracted_nodes,
    previous_episodes,
    edge_type_map or edge_type_map_default,
    episode.group_id,
    edge_types,
    custom_extraction_instructions,
)
```
- Uses LLM to identify relationships between extracted entities
- Creates edge type map if not provided
- Returns raw extracted edges (unresolved)

### 7. Returns Results
```python
return extracted_nodes, node_episode_index_map, extracted_edges
```

## Usage Examples

### High-Level Interface (From Test)
```python
episode_result = await graphiti._extract_nodes_and_edges_only(
    episode_body=json.dumps(doc_body),
    source_description="Meeting document",
    reference_time=datetime.now(timezone.utc),
    source=EpisodeType.json,
    group_id="user_2699",
    entity_types=custom_entity_types,
    edge_types=custom_edge_types,
    edge_type_map=type_mapping,
    excluded_entity_types=["Task"],
    saga="Meeting Title",
)

extracted_nodes, index_map, extracted_edges = episode_result
```

### High-Level Interface (Minimal)
```python
extracted_nodes, index_map, extracted_edges = await graphiti._extract_nodes_and_edges_only(
    episode_body="Some content",
    source_description="Source",
    reference_time=datetime.now(timezone.utc),
)
```

### Retrieving Existing Episode
```python
extracted_nodes, index_map, extracted_edges = await graphiti._extract_nodes_and_edges_only(
    episode_body="New content",
    source_description="Updated source",
    reference_time=datetime.now(timezone.utc),
    uuid="existing-episode-uuid",  # Will fetch and update this episode
)
```

## Key Features

✅ **Automatic Episode Creation**
- Creates EpisodicNode from content parameters
- Handles group_id defaults and validation
- Uses saga name as episode name if provided

✅ **Context-Aware Extraction**
- Retrieves previous episodes for LLM context
- Filters by group_id and source type
- Limits context to RELEVANT_SCHEMA_LIMIT episodes

✅ **Type Validation**
- Validates custom entity types
- Validates excluded entity types
- Creates default edge type map

✅ **Attribution Tracking**
- Returns node_episode_index_map for tracing which episode mentions which node
- Used later for building episodic edges

✅ **Raw Extraction Only**
- Does NOT resolve against existing graph
- Does NOT deduplicate nodes
- Does NOT validate edges
- Returns raw LLM output for flexibility

## Data Flow

```
Input Parameters (episode content + extraction config)
    ↓
Validate entity types
    ↓
Handle group_id (default, validate, clone driver if needed)
    ↓
Create or retrieve EpisodicNode
    ↓
Retrieve previous episodes for context
    ↓
Create default edge type map
    ↓
Extract nodes (LLM) → extracted_nodes, node_episode_index_map
    ↓
Extract edges (LLM) → extracted_edges
    ↓
Return tuple[extracted_nodes, node_episode_index_map, extracted_edges]
```

## Integration with Refactoring

This function is the first step in the three-phase extraction pipeline:

1. **`_extract_nodes_and_edges_only()`** ← You are here
   - Creates episode, retrieves context, extracts raw data

2. **`_resolve_and_hydrate_nodes_edges()`**
   - Takes extracted data
   - Resolves against existing graph
   - Deduplicates and merges nodes
   - Validates and resolves edges
   - Extracts node attributes

3. **`_persist_episode_to_graph()`**
   - Saves all resolved entities to graph
   - Creates episodic edges
   - Handles saga association
   - Updates communities

## Testing

The implementation has been validated against the test file (`src/brain/graphiti_test.py`) which calls:

```python
episode_result = await _graphiti._extract_nodes_and_edges_only(
    episode_body=json.dumps(clean_dict(doc_body)),
    source_description=f"{doc_model.source} document",
    reference_time=doc_model.created_at,
    source=EpisodeType.json,
    group_id=group_id,
    entity_types=entity_types,
    edge_types=edge_types,
    edge_type_map=edge_type_map,
    excluded_entity_types=excluded_entities,
    saga=doc_model.title,
)
```

All parameters are properly handled and the function returns the expected tuple structure.