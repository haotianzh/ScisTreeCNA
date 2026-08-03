#!/usr/bin/env python3
"""Flatten terminal control sequences in executed notebook outputs.

ScisTreeCNA logs progress with `rich`, which drives a live spinner using ANSI
escapes and carriage returns. That renders fine in a terminal and as unreadable
noise in a static HTML page, so this script replays the control sequences the way
a terminal would and writes back plain text.

What it does, per stdout/stderr stream output:
  - applies carriage returns (a later segment overwrites the earlier one)
  - strips ANSI CSI/OSC escapes (colour, cursor movement, erase-line)
  - drops leftover spinner-only frames and blank runs

Usage:
    python docs/tools/clean_notebook_output.py docs/tutorials/*.ipynb

Run it after re-executing the notebooks. See docs/README.md.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_ONLY = re.compile(rf"^[{SPINNER_CHARS}]\s*(NNI Searching)?\s*$")


def render(text: str) -> str:
    """Replay `text` as a terminal would and return the resulting plain text."""
    text = OSC.sub("", text)

    lines: list[str] = []
    for raw in text.split("\n"):
        # A carriage return resets to column 0; whatever follows overwrites
        # what came before, so only the final segment survives.
        segment = raw.rsplit("\r", 1)[-1]
        segment = CSI.sub("", segment)
        lines.append(segment.rstrip())

    kept: list[str] = []
    for line in lines:
        if SPINNER_ONLY.match(line.strip()):
            continue
        # collapse runs of blank lines
        if not line.strip() and kept and not kept[-1].strip():
            continue
        kept.append(line)

    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def clean_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text())
    changed = False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream":
                continue
            original = "".join(output.get("text", []))
            if "\x1b" not in original and "\r" not in original:
                continue
            cleaned = render(original)
            if cleaned != original:
                output["text"] = [l + "\n" for l in cleaned.split("\n")]
                changed = True

    if changed:
        path.write_text(json.dumps(nb, indent=1))
    return changed


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv]
    if not paths:
        print(__doc__)
        return 2
    for path in paths:
        status = "cleaned" if clean_notebook(path) else "unchanged"
        print(f"{status:>10}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
