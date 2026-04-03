"""Prompt template loader utility with caching."""

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Load and cache a prompt template from the prompts directory.

    Each file is read from disk only once; subsequent calls return
    the cached string.

    Args:
        name: Filename (with or without ``.txt`` extension).

    Returns:
        The prompt text content.
    """
    path = PROMPTS_DIR / name
    if not path.suffix:
        path = path.with_suffix(".txt")
    return path.read_text(encoding="utf-8")
