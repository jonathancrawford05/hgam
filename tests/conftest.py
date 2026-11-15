"""Fixture loading. Fixtures are mgcv outputs written by oracle/make_fixtures.R."""

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture():
    def _load(name: str) -> dict:
        with (FIXTURES / f"{name}.json").open() as fh:
            return json.load(fh)

    return _load
