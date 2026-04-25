"""Quick overview of the results notebook: cell types, sources, and outputs.

Used once after pulling the Colab-produced notebook so the assistant can
ground next-step advice in the actual numbers from the run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_PATH = Path("cybersec_grpo_results_iter_1.ipynb")


def text_of(field):
    if isinstance(field, list):
        return "".join(field)
    return field or ""


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]
    print(f"total cells: {len(cells)}\n")

    for idx, cell in enumerate(cells):
        kind = cell["cell_type"]
        src = text_of(cell.get("source", ""))
        head = src.strip().splitlines()[:2]
        head_str = " | ".join(head) if head else "(empty)"
        print(f"[{idx:>3d}] {kind:<8s}  {head_str[:120]}")

        if kind == "code":
            for out in cell.get("outputs", []):
                ot = out.get("output_type", "")
                if ot == "stream":
                    txt = text_of(out.get("text", ""))
                    if txt.strip():
                        for line in txt.rstrip().splitlines():
                            print(f"        | {line[:140]}")
                elif ot in ("execute_result", "display_data"):
                    data = out.get("data", {})
                    if "text/plain" in data:
                        txt = text_of(data["text/plain"])
                        if txt.strip():
                            for line in txt.rstrip().splitlines():
                                print(f"        > {line[:140]}")
                elif ot == "error":
                    print(f"        ! {out.get('ename')}: {text_of(out.get('evalue', ''))[:140]}")
                    for line in (out.get("traceback") or [])[-5:]:
                        for sub in line.rstrip().splitlines()[-3:]:
                            print(f"        ! {sub[:140]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
