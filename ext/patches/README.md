# External Patches

This directory stores patch files for vendored libraries under `ext/` when
Multipers needs a local compatibility or workflow patch but we do not want to
edit the vendored checkout directly by hand.

Patches are organized by concern (log control, features, optimizations) and are
applied as build-time overlays — vendored `ext/` trees are never modified.

Current patch groups:

- **Log gating** (`*_runtime_logs.patch`): routes `bool verbose` globals through
  `runtime_flag<Bit>` for runtime backend-log control
- **Features** (`*_features.patch`): algorithmic patches (windowed free
  resolution, optimized column operations, etc.)
- **Optimization** (`*_edge_copy_reducer.patch`): narrow perf improvements
  (timer guards, scratch-vector reuse)

Rules:

- prefer generated patch files over ad-hoc manual edits in vendored trees
- patch files should be reproducible from scripts when practical
- normal builds generate build-local patch files under
  `build/generated_ext_patches/` and apply them to build-local overlays through
  `cmake/ApplyExtPatchOverlay.cmake`
- tracked patch files here are review/check artifacts refreshed explicitly, not
  files that normal builds rewrite in place
- keep patch scope minimal and backend-specific
- profiling/stat patches that exist only for downstream optimization should live
  in the downstream overlay, not in `multipers/ext/patches/`

Current generator:

- `ext/patches/generate_ext_patches.py`

Available targets:

| Library name | Generates | Concern |
|---|---|---|
| `mpfree` | `mpfree_runtime_logs.patch` | Log gating |
| `function_delaunay` | `function_delaunay_runtime_logs.patch` | Log gating |
| `multi_critical_logs` | `multi_critical_runtime_logs.patch` | Log gating |
| `multi_critical_features` | `multi_critical_features.patch` | Features |
| `deg_rips` | `deg_rips_edge_copy_reducer.patch` | Optimization |

Example:

```bash
python ext/patches/generate_ext_patches.py function_delaunay
```

Configured-build targets:

```bash
cmake --build build --target multipers_generate_ext_patches
cmake --build build --target multipers_check_ext_patches
```

`multipers_check_ext_patches` compares generated build-local patches against the
tracked review artifacts above; it is meant to fail if a tracked patch needs to
be refreshed.
