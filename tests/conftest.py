"""Fixture loading and the key-result margin report. Fixtures are mgcv outputs (oracle/make_fixtures.R).

CI's ``oracle`` job sets HGAM_FIXTURE_DIR to fixtures regenerated inside the pinned image, so the suite is
held to mgcv itself and not to whatever was committed. Every test module reads fixtures through ``FIXTURES``
(``from conftest import FIXTURES`` or the ``load_fixture`` fixture), never through its own path.

With HGAM_REPORT_MARGINS set, the session ends with a table of the largest difference each test measured for
each key result (the ``margin`` fixture), against the fixtures in use; under GitHub Actions it is also written
to the job summary. The table reports; the tests' own assertions decide.
"""

import json
import os
from pathlib import Path

import pytest

FIXTURES = Path(os.environ.get("HGAM_FIXTURE_DIR") or Path(__file__).parent / "fixtures")

# (item, quantity) -> [largest value, gate, fixture]
_MARGINS: dict[tuple[str, str], list] = {}


@pytest.fixture
def load_fixture():
    def _load(name: str) -> dict:
        with (FIXTURES / f"{name}.json").open() as fh:
            return json.load(fh)

    return _load


@pytest.fixture
def margin():
    """``margin(item, quantity, value, gate, fixture)``: record a measured difference; the largest is kept."""

    def _record(item: str, quantity: str, value: float, gate: str, fixture: str) -> None:
        entry = _MARGINS.setdefault((item, quantity), [float(value), gate, fixture])
        entry[0] = max(entry[0], float(value))

    return _record


def _fixture_source(name: str) -> str:
    try:
        meta = json.loads((FIXTURES / f"{name}.json").read_text()).get("meta", {})
    except (OSError, ValueError):
        return "?"
    return f"tier {meta.get('tier', '?')}, mgcv {meta.get('mgcv', '?')}"


def pytest_terminal_summary(terminalreporter) -> None:
    if not os.environ.get("HGAM_REPORT_MARGINS") or not _MARGINS:
        return
    lines = [
        f"Key-result margins against the fixtures in `{FIXTURES}`",
        "",
        "| Item | Quantity | Largest difference | Gate | Fixture (source) |",
        "|---|---|---|---|---|",
    ]
    for (item, quantity), (value, gate, fixture) in _MARGINS.items():
        lines.append(
            f"| {item} | {quantity} | {value:.2e} | {gate} | `{fixture}` ({_fixture_source(fixture)}) |"
        )
    terminalreporter.write_sep("=", "key-result margins")
    for line in lines:
        terminalreporter.write_line(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write("\n".join(lines) + "\n")
