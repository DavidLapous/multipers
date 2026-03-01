#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from Cython import Tempita


def _output_path(
    path: Path, source_root: Path | None, output_root: Path | None
) -> Path:
    if output_root is None:
        return path.with_suffix("")
    if source_root is None:
        source_root = Path.cwd()
    relative_path = path.resolve().relative_to(source_root.resolve())
    return (output_root.resolve() / relative_path).with_suffix("")


def process_template(
    path: Path, source_root: Path | None, output_root: Path | None
) -> None:
    content = path.read_text(encoding="utf-8")
    rendered = Tempita.Template(content).substitute()
    output = _output_path(path, source_root, output_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = output.read_text(encoding="utf-8") if output.exists() else None
    if previous != rendered:
        output.write_text(rendered, encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Render Tempita templates into .pyx/.pxd outputs"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Source tree root used to preserve relative paths",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root directory (defaults to in-place generation)",
    )
    parser.add_argument("templates", nargs="+", type=Path)
    args = parser.parse_args(argv[1:])

    for path in args.templates:
        process_template(path, args.source_root, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
