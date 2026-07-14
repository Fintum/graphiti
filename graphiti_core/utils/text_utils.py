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

import json
import re
from typing import TYPE_CHECKING

from thefuzz import fuzz  # type: ignore


if TYPE_CHECKING:
    from graphiti_core.nodes import EpisodicNode

# Maximum length for entity/community summaries
MAX_SUMMARY_CHARS = 1000


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """
    Truncate text at or about max_chars while respecting sentence boundaries.

    Attempts to truncate at the last complete sentence before max_chars.
    If no sentence boundary is found before max_chars, truncates at max_chars.

    Args:
        text: The text to truncate
        max_chars: Maximum number of characters

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_chars:
        return text

    # Find all sentence boundaries (., !, ?) up to max_chars
    truncated = text[:max_chars]

    # Look for sentence boundaries: period, exclamation, or question mark followed by space or end
    sentence_pattern = r'[.!?](?:\s|$)'
    matches = list(re.finditer(sentence_pattern, truncated))

    if matches:
        # Truncate at the last sentence boundary found
        last_match = matches[-1]
        return text[: last_match.end()].rstrip()

    # No sentence boundary found, truncate at max_chars
    return truncated.rstrip()


def concatenate_episodes(episodes: list[EpisodicNode]) -> str:
    """Concatenate episode contents with enumerated headers.

    When given a single episode, returns its content as-is.
    When given multiple episodes, each is prefixed with an ``[Episode N]``
    header so the LLM can distinguish where one ends and the next begins.
    """
    if len(episodes) == 1:
        return episodes[0].content
    parts: list[str] = []
    for i, ep in enumerate(episodes):
        timestamp = ep.valid_at.isoformat() if ep.valid_at else 'unknown'
        parts.append(f'[Episode {i}] (timestamp: {timestamp})\n{ep.content}')
    return '\n\n'.join(parts)


def split_episode(episode: EpisodicNode) -> list[dict]:
    episode_body: dict = json.loads(episode.content)

    content_chapters = episode_body.pop("chapters", None)
    content_tasks = episode_body.pop("tasks", None)
    content_participants = episode_body.pop("participants", None)
    content_main = episode_body

    episode_main = episode.model_copy()
    episode_chapters = episode.model_copy()
    episode_tasks = episode.model_copy()
    episode_participants = episode.model_copy()

    episode_main.content = json.dumps(content_main)
    episode_chapters.content = json.dumps(content_chapters)
    episode_tasks.content = json.dumps(content_tasks)
    episode_participants.content = json.dumps(content_participants)

    return [
        {"episode": episode_participants, "name": "participants", "is_empty": content_participants is None, "entities": ["Person"]}, 
        {"episode": episode_main, "name": "main", "is_empty": False, "entities": ["Person", "Company", "Project", "Signal", "Decision"]}, 
        {"episode": episode_tasks, "name": "tasks", "is_empty": False, "entities": ["Task"]}, 
        {"episode": episode_chapters, "name": "chapters", "is_empty": False, "entities": ["Person", "Company", "Project", "Signal"]}, 
    ]




def compare_names(
        target: str, 
        choice: str,
        clean_names: bool = False
    ) -> int:
    """Use fuzz library to get the similarity ratio."""
    # compare_names('Ernesto Hernandez', 'Ernesto')
    # cutoff >= 60

    if clean_names:
        pattern = r'[0-9/()]'
        target = re.sub(pattern, '', target.replace('y','i').lower()).strip()
        choice = re.sub(pattern, '', choice.replace('y','i').lower()).strip()

    # return int(difflib.SequenceMatcher(None, target, choice).quick_ratio() * 100)
    score: int = fuzz.ratio(target, choice)
    return score



def compare_names_list(
        name: str, 
        targets: list[str],
        clean_names: bool = False,
        threshold: int = 70,
    ) -> str | None:

    for target in targets:
        if compare_names(name, target, clean_names) > threshold:
            return target
        if name in target:
            return target
    return None