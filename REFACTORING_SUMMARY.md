# Refactoring Summary: `add_episode` Function Decomposition

## Overview
The original `add_episode()` function (line 1419) remains **completely intact and unchanged**. Three new focused helper functions have been created to decompose its logic into smaller, testable, and composable components.

## New Helper Functions

### 1. `_extract_nodes_and_edges_only()` (Line 680)
**Purpose:** Validate entity types and extract nodes and edges from episode without resolution.

**Responsibilities:**
- Validates entity type definitions
- Validates excluded entity types
- Extracts entities/nodes from episode content via LLM
- Extracts relationships/edges from episode content via LLM
- Returns raw extraction results (no graph resolution)

**Input:**
- `episode`: The episode to extract from
- `previous_episodes`: Prior episodes for LLM context
- `entity_types`, `excluded_entity_types`: Entity type filters
- `edge_types`, `edge_type_map`: Relationship type definitions
- `custom_extraction_instructions`: Additional LLM guidance

**Output:**
```python
tuple[
    list[EntityNode],          # extracted_nodes - raw nodes from LLM
    dict[str, list[int]],      # node_episode_index_map - node→episode attribution
    list[EntityEdge]           # extracted_edges - raw edges from LLM
]
```

---

### 2. `_resolve_and_hydrate_nodes_edges()` (Line 753)
**Purpose:** Handle group_id, resolve nodes and edges, and extract node attributes.

**Responsibilities:**
- Handles `group_id` (default assignment, validation, driver cloning)
- Resolves extracted nodes against existing graph (deduplication/merging)
- Resolves extracted edges against existing graph (conflict detection)
- Extracts node attributes (summaries, descriptions)
- Returns resolved entities ready for persistence

**Input:**
- `episode`: The episode being processed
- `extracted_nodes`, `extracted_edges`: Raw extraction from step 1
- `previous_episodes`: Prior episodes for resolution context
- `entity_types`, `edge_types`, `edge_type_map`: Type definitions
- `group_id`: Optional graph partition ID
- `custom_extraction_instructions`: LLM guidance

**Output:**
```python
tuple[
    str,                       # resolved_group_id - final group ID
    list[EntityNode],          # resolved_nodes - deduplicated nodes
    list[EntityEdge],          # resolved_edges - valid edges
    list[EntityEdge],          # invalidated_edges - contradicted edges
    dict[str, str],            # uuid_map - extracted→resolved UUID mapping
    list[EntityNode]           # hydrated_nodes - nodes with attributes
]
```

---

### 3. `_persist_episode_to_graph()` (Line 853)
**Purpose:** Persist episode data to graph, handle saga association, and update communities.

**Responsibilities:**
- Builds episodic edges (episode→entity relationships)
- Saves episode, nodes, edges, and episodic edges to graph
- Optionally associates episode with a saga (creates HAS_EPISODE and NEXT_EPISODE edges)
- Optionally updates community summaries
- Returns all persisted entities for result wrapping

**Input:**
- `episode`: The episode to persist
- `hydrated_nodes`: Resolved nodes with attributes
- `entity_edges`: All entity edges (resolved + invalidated)
- `node_episode_index_map`: Attribution mapping
- `group_id`: Graph partition ID
- `saga`: Optional saga name or node
- `saga_previous_episode_uuid`: Optional previous episode in saga
- `update_communities`: Whether to regenerate community summaries

**Output:**
```python
tuple[
    list[EpisodicEdge],        # episodic_edges - episode→entity links
    EpisodicNode,              # episode - updated episode node
    list[CommunityNode],       # communities - updated community nodes
    list[CommunityEdge]        # community_edges - community relationships
]
```

---

## Composition Function

### `_add_episode_composed()` (Line 1269)
Demonstrates how the three helper functions compose together to achieve the same result as `add_episode()`.

**Flow:**
```
Input: episode metadata and content
  ↓
1. retrieve_episodes() → previous_episodes
2. create/get episode node
  ↓
3. _extract_nodes_and_edges_only()
   → extracted_nodes, node_episode_index_map, extracted_edges
  ↓
4. _resolve_and_hydrate_nodes_edges()
   → resolved_group_id, resolved_nodes, resolved_edges, 
     invalidated_edges, uuid_map, hydrated_nodes
  ↓
5. _persist_episode_to_graph()
   → episodic_edges, episode, communities, community_edges
  ↓
6. Wrap in AddEpisodeResults
Output: AddEpisodeResults (identical to add_episode())
```

---

## Comparison: Original vs. Refactored

| Aspect | Original `add_episode()` | Refactored Approach |
|--------|--------------------------|---------------------|
| **Location** | Line 1419 (unchanged) | Line 1269 (`_add_episode_composed`) |
| **Complexity** | ~250 lines, many concerns | Three focused functions (50-100 lines each) |
| **Testability** | Test entire pipeline | Test each phase independently |
| **Reusability** | Monolithic | Each helper is independently useful |
| **Functionality** | ✅ Preserved | ✅ Identical |
| **API** | ✅ Unchanged | New alternative with same signature |

---

## Usage Examples

### Using Original Function (Unchanged)
```python
result = await graphiti.add_episode(
    name="Meeting notes",
    episode_body="...",
    source_description="...",
    reference_time=datetime.now(timezone.utc),
)
```

### Using Composed Function (New Option)
```python
# Same signature and result
result = await graphiti._add_episode_composed(
    name="Meeting notes",
    episode_body="...",
    source_description="...",
    reference_time=datetime.now(timezone.utc),
)
```

### Using Helper Functions Separately (Advanced)
```python
# Just extraction phase
extracted_nodes, index_map, extracted_edges = await graphiti._extract_nodes_and_edges_only(
    episode=my_episode,
    previous_episodes=context_episodes,
    entity_types=custom_types,
    excluded_entity_types=None,
    edge_types=None,
    edge_type_map=None,
)

# Then later: resolve and hydrate
group_id, resolved_nodes, resolved_edges, invalidated_edges, uuid_map, hydrated = \
    await graphiti._resolve_and_hydrate_nodes_edges(
    episode=my_episode,
    extracted_nodes=extracted_nodes,
    extracted_edges=extracted_edges,
    previous_episodes=context_episodes,
    entity_types=custom_types,
    edge_types=None,
    edge_type_map=None,
    group_id=None,
)

# Finally: persist
episodic_edges, episode, communities, community_edges = await graphiti._persist_episode_to_graph(
    episode=my_episode,
    hydrated_nodes=hydrated,
    entity_edges=resolved_edges + invalidated_edges,
    node_episode_index_map=index_map,
    group_id=group_id,
    saga=None,
    saga_previous_episode_uuid=None,
    update_communities=False,
)
```

---

## Benefits

1. **Separation of Concerns**
   - Extraction logic isolated from resolution logic
   - Persistence logic separate from transformation logic

2. **Testability**
   - Each phase can be unit tested independently
   - Mock/test specific components without full pipeline

3. **Reusability**
   - Use extraction phase in custom pipelines
   - Apply resolution logic to different data sources
   - Bulk operations can reuse helper functions

4. **Maintainability**
   - Smaller functions are easier to understand and modify
   - Changes to one phase don't require rewriting entire function
   - Clear contracts (inputs/outputs) between phases

5. **Flexibility**
   - Process extraction results before resolution
   - Insert custom validation/filtering between steps
   - Compose with other operations

---

## Backward Compatibility

✅ **100% Backward Compatible**
- Original `add_episode()` is completely unchanged
- All existing code continues to work without modification
- New helper functions and `_add_episode_composed()` are additions only
