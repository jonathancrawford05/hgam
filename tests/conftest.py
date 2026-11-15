"""Fixture loading. Fixtures are mgcv outputs written by oracle/make_fixtures.R.

CI's ``oracle`` job sets HGAM_FIXTURE_DIR to fixtures regenerated inside the pinned image, so the suite is
held to mgcv itself and not to whatever was committed.
"""

import json
import os
from pathlib import Path

import pytest

FIXTURES = Path(os.environ.get("HGAM_FIXTURE_DIR") or Path(__file__).parent / "fixtures")


@pytest.fixture(scope="session")
def load_fixture():
    def _load(name: str) -> dict:
        with (FIXTURES / f"{name}.json").open() as fh:
            return json.load(fh)

    return _load
