#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


BEGIN_MARKER = "  # BEGIN GENERATED SDIST EXT INCLUDE"
END_MARKER = "  # END GENERATED SDIST EXT INCLUDE"
LICENSE_NAMES = ("LICENSE", "COPYING", "COPYING.LESSER")


def _repo_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if not (root / "CMakeLists.txt").is_file():
        raise RuntimeError(f"Missing CMakeLists.txt in {root}")
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"Missing pyproject.toml in {root}")
    if not (root / "multipers").is_dir():
        raise RuntimeError(f"Missing multipers/ in {root}")
    return root


def _required_ext_files(root: Path) -> set[str]:
    build_dir = root / "build"
    if not (build_dir / ".ninja_deps").is_file():
        raise RuntimeError("Missing build/.ninja_deps, run a local build first")

    try:
        output = subprocess.check_output(
            ["ninja", "-C", str(build_dir), "-t", "deps"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Could not find `ninja` on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"`ninja -C build -t deps` failed:\n{exc.output}") from exc

    ext_root = (root / "ext").resolve()
    required: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("ninja:", "warning:")):
            continue
        if line.endswith(("(VALID)", "(STALE)", "(MISSING)")):
            continue
        if line.startswith("CMakeFiles/") and ": #deps" in line:
            continue

        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (build_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()

        if candidate.is_file() and ext_root in candidate.parents:
            required.add(candidate.relative_to(root).as_posix())

    return required


def _vendored_license_files(root: Path, required_ext_files: set[str]) -> set[str]:
    ext_root = root / "ext"
    licenses: set[str] = set()

    for rel_path in required_ext_files:
        current = (root / rel_path).parent
        while current == ext_root or ext_root in current.parents:
            for license_name in LICENSE_NAMES:
                candidate = current / license_name
                if candidate.is_file():
                    licenses.add(candidate.relative_to(root).as_posix())
            if current == ext_root:
                break
            current = current.parent

    return licenses


def _generated_block(
    required_ext_files: set[str], vendored_license_files: set[str]
) -> str:
    lines = [
        BEGIN_MARKER,
        "  # Generated from `ninja -C build -t deps`; update with",
        "  # `python tools/update_sdist_ext_whitelist.py --write`.",
    ]
    for path in sorted(required_ext_files):
        lines.append(f'  "{path}",')
    if vendored_license_files:
        lines.append("  # Vendored license files kept in the sdist.")
        for path in sorted(vendored_license_files):
            lines.append(f'  "{path}",')
    lines.append(END_MARKER)
    return "\n".join(lines)


def _replace_generated_block(pyproject_text: str, generated_block: str) -> str:
    try:
        before, rest = pyproject_text.split(BEGIN_MARKER, 1)
        _, after = rest.split(END_MARKER, 1)
    except ValueError as exc:
        raise RuntimeError(
            "Could not find generated ext include markers in pyproject.toml"
        ) from exc
    return before + generated_block + after


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check or update the generated ext include block in pyproject.toml from "
            "the local Ninja dependency closure."
        )
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite pyproject.toml; without this flag the script only checks",
    )
    args = parser.parse_args()

    root = _repo_root()
    pyproject_path = root / "pyproject.toml"
    required_ext_files = _required_ext_files(root)
    vendored_license_files = _vendored_license_files(root, required_ext_files)
    generated_block = _generated_block(required_ext_files, vendored_license_files)

    original = pyproject_path.read_text(encoding="utf-8")
    updated = _replace_generated_block(original, generated_block)

    summary = (
        f"{len(required_ext_files)} ext files + "
        f"{len(vendored_license_files)} license files"
    )
    if updated == original:
        print(f"{pyproject_path} already matches Ninja deps: {summary}")
        return 0

    if not args.write:
        print(
            f"{pyproject_path} is out of date: {summary}. "
            "Run `python tools/update_sdist_ext_whitelist.py --write`."
        )
        return 1

    pyproject_path.write_text(updated, encoding="utf-8")
    print(f"Updated {pyproject_path}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
