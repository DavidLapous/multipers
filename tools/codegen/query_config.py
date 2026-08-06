#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def load_options():
    options_path = Path(__file__).resolve().parents[2] / "options.py"
    spec = importlib.util.spec_from_file_location("multipers_user_options", options_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {options_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("cmake",), required=True)
    args = parser.parse_args()
    has_flat = "Degree_rips_bifiltration" in getattr(load_options(), "FILTRATION_CONTAINERS", ())
    print("1" if has_flat else "0")


if __name__ == "__main__":
    main()
