#!/usr/bin/env python3
"""Check that compiled multipers extension modules share templates with the core library.

The invariant being verified (no class or module names are hardcoded):

1. A module must not carry its own copy of a project symbol that
   ``libmultipers_core`` already defines (no duplicated template instantiations).
2. Every project symbol a module imports must be provided by the core library
   (or by the module itself).
3. A module that imports symbols from core must actually be linked against it.
4. At least one module must use the core, otherwise the shared-template setup
   is silently broken.

Modules are discovered by globbing the installed package, so adding, removing
or renaming a module requires no change here.  Optionally, a manifest
(``--manifest modules.json``) lists the modules the build is *expected* to
produce; the check then also fails if one of them is missing.  Manifest format::

    {"modules": ["_slicer_nanobind", "_simplex_tree_multi_nanobind"]}

On Windows, symbol inspection is not attempted: the core DLL must exist and
every extension module (``*.pyd``) must import successfully.

Run it from OUTSIDE the source tree so the *installed* package is inspected::

    cd "$(mktemp -d)" && python /path/to/tools/check_core_ownership.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Only the project's C++ namespace is hardcoded (Itanium mangling of `Gudhi::`,
# with the extra leading underscore used by Mach-O).  Standard library, libc
# and Python symbols never match, so they are ignored automatically.
DEFAULT_PREFIXES = ("_ZN5Gudhi", "__ZN5Gudhi")
CORE_MARKER = "multipers_core"

# nm letters meaning "undefined" (uppercase U, lowercase weak-undefined w / v).
_UNDEFINED_TYPES = {"U", "w", "v"}


# --------------------------------------------------------------------------- #
# Symbol extraction
# --------------------------------------------------------------------------- #
def _parse_nm(out: str) -> list[tuple[str, str]]:
    rows = []
    for line in out.splitlines():
        m = re.match(r"^\s*([0-9A-Fa-f]+)?\s*([A-Za-z])\s+(.+)$", line)
        if m:
            rows.append((m.group(2), m.group(3).strip()))
    return rows


def _parse_readelf(out: str) -> list[tuple[str, str]]:
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 8 or not parts[0].rstrip(":").isdigit():
            continue
        rows.append(("U" if parts[6] == "UND" else "T", parts[7]))
    return rows


def read_symbols(path: Path) -> list[tuple[str, str]]:
    """Return ``(type, name)`` pairs of the dynamic symbols of ``path``."""
    if sys.platform.startswith("linux"):
        commands = [
            (["nm", "-D", str(path)], _parse_nm),
            (["readelf", "-Ws", str(path)], _parse_readelf),
        ]
    else:
        commands = [(["nm", "-g", str(path)], _parse_nm)]

    errors = []
    for command, parser in commands:
        if shutil.which(command[0]) is None:
            errors.append(f"{command[0]} not available")
            continue
        proc = subprocess.run(command, text=True, capture_output=True)
        rows = parser(proc.stdout)
        if rows:
            return rows
        detail = proc.stderr.strip() or "no parseable symbols"
        errors.append(f"{' '.join(command)} -> {detail}")
    raise RuntimeError(f"Unable to inspect symbols in {path}: " + " ; ".join(errors))


def split_symbols(path: Path, prefixes: tuple[str, ...]) -> tuple[set[str], set[str]]:
    """Return ``(defined, undefined)`` project symbols of ``path``."""
    defined, undefined = set(), set()
    for typ, name in read_symbols(path):
        if not name.startswith(prefixes):
            continue
        (undefined if typ in _UNDEFINED_TYPES else defined).add(name)
    return defined, undefined


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def locate_package_dir(package_name: str) -> Path:
    spec = importlib.util.find_spec(package_name)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"Unable to locate installed {package_name} package")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def module_stem(path: Path) -> str:
    """`_slicer_nanobind.cpython-313-x86_64-linux-gnu.so` -> `_slicer_nanobind`."""
    return path.name.split(".")[0]


def load_manifest(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return set(json.loads(path.read_text())["modules"])


def check_manifest(manifest: set[str], modules: list[Path]) -> list[str]:
    built = {module_stem(p) for p in modules}
    return [
        f"manifest lists module '{name}' but no compiled file was found"
        for name in sorted(manifest - built)
    ]


# --------------------------------------------------------------------------- #
# Platform checks
# --------------------------------------------------------------------------- #
def check_unix(pkg: Path, prefixes, manifest: set[str], check_duplicates: bool) -> list[str]:
    cores = sorted(
        p for pattern in ("lib*multipers_core*.so", "lib*multipers_core*.dylib")
        for p in pkg.glob(pattern)
    )
    if not cores:
        return [f"Missing {CORE_MARKER} shared library in {pkg}"]
    core = cores[0]

    core_defs, _ = split_symbols(core, prefixes)
    if not core_defs:
        return [f"{core.name} exports no project symbols"]

    modules = sorted(p for p in pkg.glob("*.so") if CORE_MARKER not in p.name)
    if not modules:
        return [f"No compiled extension modules found in {pkg}"]

    errors = check_manifest(manifest, modules)

    if sys.platform == "darwin":
        dep_cmd = ["otool", "-L"]
    else:
        dep_cmd = ["ldd"]

    using_core = 0
    for mod in modules:
        local, undef = split_symbols(mod, prefixes)

        if check_duplicates:
            dup = sorted(local & core_defs)
            if dup:
                errors.append(
                    f"{mod.name}: {len(dup)} project symbol(s) duplicated from "
                    f"{core.name}, e.g. {dup[:3]}"
                )

        missing = sorted(undef - core_defs - local)
        if missing:
            errors.append(
                f"{mod.name}: {len(missing)} unresolved project symbol(s) not "
                f"provided by {core.name}, e.g. {missing[:3]}"
            )

        if undef & core_defs:
            using_core += 1
            deps = subprocess.check_output(dep_cmd + [str(mod)], text=True)
            if CORE_MARKER not in deps:
                errors.append(f"{mod.name} uses core symbols but is not linked against {core.name}")

    if using_core == 0:
        errors.append(
            "no module imports anything from the core library: "
            "the shared-template setup looks broken"
        )
    print(f"checked {len(modules)} module(s) against {core.name}; {using_core} use the core")
    return errors


def check_windows(pkg: Path, manifest: set[str]) -> list[str]:
    cores = sorted(pkg.glob(f"*{CORE_MARKER}*.dll"))
    if not cores:
        return [f"Missing multipers core DLL in {pkg}"]

    modules = sorted(pkg.glob("*.pyd"))
    if not modules:
        return [f"No compiled extension modules found in {pkg}"]

    errors = check_manifest(manifest, modules)
    for mod in modules:
        name = f"{pkg.name}.{module_stem(mod)}"
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report any import failure
            errors.append(f"cannot import {name}: {exc}")
    print(f"imported {len(modules)} module(s); core DLL: {cores[0].name}")
    return errors


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", default="multipers", help="installed package name")
    ap.add_argument("--package-dir", type=Path, help="inspect this directory instead")
    ap.add_argument("--manifest", type=Path, help="JSON list of expected modules")
    ap.add_argument(
        "--prefix", action="append", dest="prefixes",
        help="mangled-name prefix identifying project symbols (repeatable)",
    )
    ap.add_argument(
        "--no-duplicate-check", action="store_true",
        help="skip check 1 (e.g. if inline/weak symbols are legitimately duplicated)",
    )
    args = ap.parse_args()

    prefixes = tuple(args.prefixes) if args.prefixes else DEFAULT_PREFIXES
    pkg = args.package_dir.resolve() if args.package_dir else locate_package_dir(args.package)
    manifest = load_manifest(args.manifest)

    if sys.platform == "win32":
        errors = check_windows(pkg, manifest)
    else:
        errors = check_unix(pkg, prefixes, manifest, not args.no_duplicate_check)

    for err in errors:
        print(f"::error::{err}", file=sys.stderr)
    if errors:
        return 1
    print("core template ownership checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
