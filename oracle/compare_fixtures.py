"""Compare committed fixtures with fixtures regenerated inside the pinned oracle image.

Usage: python oracle/compare_fixtures.py <committed_dir> <oracle_dir>

A committed tier 3 fixture must reproduce: structure exactly, numbers to 1e-8 relative. A committed tier 1
fixture (made with a local R) is reported but does not fail: the oracle copy replaces it for the tests.
A fixture the generator no longer produces, or one that is missing from the commit, fails.
"""

import json
import math
import sys
from pathlib import Path

RTOL, ATOL = 1e-8, 1e-10


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key == "meta":
                continue
            if key not in a or key not in b:
                yield f"{path}/{key}: present on one side only"
            else:
                yield from differences(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield f"{path}: length {len(a)} vs {len(b)}"
        else:
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                yield from differences(x, y, f"{path}[{i}]")
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        if a != b:
            yield f"{path}: {a!r} vs {b!r}"
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL):
            yield f"{path}: {a!r} vs {b!r}"
    elif a != b:
        yield f"{path}: {a!r} vs {b!r}"


def main(committed_dir: str, oracle_dir: str) -> int:
    committed = {p.name: p for p in Path(committed_dir).glob("*.json")}
    oracle = {p.name: p for p in Path(oracle_dir).glob("*.json")}
    failed = False
    for name in sorted(set(committed) | set(oracle)):
        if name not in committed or name not in oracle:
            print(f"FAIL {name}: present on one side only")
            failed = True
            continue
        a, b = json.loads(committed[name].read_text()), json.loads(oracle[name].read_text())
        diffs = list(differences(a, b))
        tier = a.get("meta", {}).get("tier")
        if not diffs:
            print(f"ok   {name} (committed tier {tier})")
        elif tier == 3:
            print(f"FAIL {name}: committed tier 3 fixture does not reproduce ({len(diffs)} differences)")
            for d in diffs[:5]:
                print("     ", d)
            failed = True
        else:
            print(
                f"note {name}: committed tier {tier} fixture differs from the oracle in {len(diffs)} places; "
                "the oracle copy is used for the tests."
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
