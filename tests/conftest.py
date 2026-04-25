"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from cybersec import CybersecEnvironment


@pytest.fixture
def env() -> CybersecEnvironment:
    return CybersecEnvironment()
