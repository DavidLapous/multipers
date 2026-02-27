#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

from Cython import Tempita


def process_template(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    rendered = Tempita.Template(content).substitute()
    output = path.with_suffix("")
    output.write_text(rendered, encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit("Usage: process_tempita.py <file1.tp> [file2.tp ...]")

    for arg in argv[1:]:
        process_template(Path(arg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
