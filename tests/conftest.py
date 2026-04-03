"""Shared test fixtures."""

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def sample_snapshot() -> dict:
    """Return a complete sample snapshot for testing."""
    return json.loads((FIXTURES_DIR / "sample_snapshot.json").read_text())


@pytest.fixture()
def empty_snapshot() -> dict:
    """Return a minimal / empty snapshot."""
    return {}
