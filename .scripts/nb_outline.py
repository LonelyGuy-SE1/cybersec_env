"""Quick outline of a notebook (cell idx, type, first non-empty line)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def outline(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    print(f"=== {path}  ({len(nb['cells'])} cells) ===")
    for i, c in enumerate(nb["cells"]):
        src = c["source"]
        if isinstance(src, list):
            src = "".join(src)
        first = next((l for l in src.splitlines() if l.strip()), "<empty>")
        print(f"[{i:>3}] {c['cell_type']:>8} | {first[:100]}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        outline(Path(arg))
