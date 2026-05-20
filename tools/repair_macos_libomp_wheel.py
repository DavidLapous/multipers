from __future__ import annotations

import base64
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


LIBOMP_ID = "@rpath/libomp.dylib"
LIBOMP_BASENAME = "libomp.dylib"


def _run(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def _mach_o_deps(path: Path) -> list[str]:
    try:
        output = _run(["otool", "-L", str(path)])
    except subprocess.CalledProcessError:
        return []
    deps = []
    for line in output.splitlines()[1:]:
        stripped = line.strip()
        if stripped:
            deps.append(stripped.split(" ", 1)[0])
    return deps


def _rpaths(path: Path) -> set[str]:
    try:
        output = _run(["otool", "-l", str(path)])
    except subprocess.CalledProcessError:
        return set()
    paths = set()
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("path "):
            paths.add(stripped.split(" ", 2)[1])
    return paths


def _install_name_tool(*args: str) -> None:
    subprocess.check_call(["install_name_tool", *args])


def _codesign(path: Path) -> None:
    if shutil.which("codesign") is not None:
        subprocess.check_call(["codesign", "--force", "--sign", "-", str(path)])


def _loader_rpath(binary: Path, dylibs_dir: Path) -> str:
    rel = os.path.relpath(dylibs_dir, binary.parent)
    if rel == ".":
        return "@loader_path"
    return f"@loader_path/{rel}"


def _repair_libomp(root: Path) -> bool:
    package_dir = root / "multipers"
    dylibs_dir = package_dir / ".dylibs"
    libomp = dylibs_dir / LIBOMP_BASENAME
    if not libomp.exists():
        return False

    _install_name_tool("-id", LIBOMP_ID, str(libomp))

    repaired = True
    modified = {libomp}
    for binary in package_dir.rglob("*"):
        if not binary.is_file() or binary == libomp:
            continue
        deps = _mach_o_deps(binary)
        libomp_deps = [dep for dep in deps if Path(dep).name == LIBOMP_BASENAME]
        if not libomp_deps:
            continue
        for dep in libomp_deps:
            if dep != LIBOMP_ID:
                _install_name_tool("-change", dep, LIBOMP_ID, str(binary))
                repaired = True
                modified.add(binary)
        rpath = _loader_rpath(binary, dylibs_dir)
        if rpath not in _rpaths(binary):
            _install_name_tool("-add_rpath", rpath, str(binary))
            repaired = True
            modified.add(binary)
    for binary in sorted(modified):
        _codesign(binary)
    return repaired


def _rewrite_record(root: Path) -> None:
    (record_path,) = root.glob("*.dist-info/RECORD")
    record_rel = record_path.relative_to(root).as_posix()
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == record_rel:
            rows.append([rel, "", ""])
            continue
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        rows.append([rel, f"sha256={digest.decode('ascii')}", str(len(data))])
    with record_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def _rewrite_wheel(root: Path, wheel: Path) -> None:
    tmp_wheel = wheel.with_suffix(wheel.suffix + ".tmp")
    with zipfile.ZipFile(tmp_wheel, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(root).as_posix())
    tmp_wheel.replace(wheel)


def repair_wheel(wheel: Path) -> bool:
    if "macosx" not in wheel.name:
        return False
    with tempfile.TemporaryDirectory(prefix="multipers-libomp-wheel-") as tmp:
        root = Path(tmp) / "wheel"
        root.mkdir()
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(root)
        changed = _repair_libomp(root)
        if changed:
            _rewrite_record(root)
            _rewrite_wheel(root, wheel)
    return changed


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: repair_macos_libomp_wheel.py WHEEL [WHEEL ...]", file=sys.stderr)
        return 2
    if not shutil.which("otool") or not shutil.which("install_name_tool"):
        print("otool and install_name_tool are required", file=sys.stderr)
        return 1
    for arg in sys.argv[1:]:
        wheel = Path(arg)
        changed = repair_wheel(wheel)
        status = "repaired" if changed else "unchanged"
        print(f"{wheel}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
