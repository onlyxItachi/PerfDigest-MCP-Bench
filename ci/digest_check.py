"""CI digest driver — digest captured reports through the RELEASED perfdigest-mcp.

Usage: python ci/digest_check.py <format>:<path> [<format>:<path> ...]

For every report it prints the unit list and the top unit's digest (so the CI
log doubles as an eval record), then enforces the honesty contract:

  * at least one unit parses (a capture that produced nothing fails the job);
  * no digested metric is fabricated: every value is a real number, the honest
    absence sentinel, or None — a hardware counter the VM could not measure
    (perf's "<not supported>") must NEVER surface as 0.0.

This script deliberately uses only the installed PyPI package, never repo code:
the bench repo evaluates released behavior. (Backend registration goes through
the server's private hook until the entry-point plugin API lands — main repo
issue #6.)
"""

import json
import math
import sys

from perfdigest.adapters import registry
from perfdigest.core.digest import build_digest
from perfdigest.server.app import _register_backends

NOT_AVAILABLE = "not_available_in_this_export"


def check(fmt: str, path: str) -> None:
    backend = registry.get_backend(fmt)
    units = backend.load_units(path)
    assert units, f"{fmt}: no units parsed from {path}"

    listing = [
        {"name": u.name, "index": u.index, "duration_us": u.duration_us, "domain": u.domain}
        for u in units
    ]
    print(f"\n===== {fmt} :: {path} — {len(units)} units =====")
    print(json.dumps(listing[:10], indent=1))

    top = max(units, key=lambda u: (u.duration_us is not None, u.duration_us or 0.0))
    digest = build_digest(top, fmt, list(backend.default_core_set)).to_dict()
    print(f"--- digest of {top.name!r} ---")
    print(json.dumps(digest, indent=1))

    for name, value in digest["metrics"].items():
        ok = (
            value is None
            or value == NOT_AVAILABLE
            or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))
        )
        assert ok, f"{fmt}: metric {name!r} has fabricated/non-finite value {value!r}"


def main() -> None:
    pairs = [arg.split(":", 1) for arg in sys.argv[1:]]
    assert pairs, "usage: digest_check.py <format>:<path> ..."
    _register_backends()
    for fmt, path in pairs:
        check(fmt, path)
    print(f"\nOK — {len(pairs)} report(s) digested honestly.")


if __name__ == "__main__":
    main()
