from __future__ import annotations

import os
import pickle
import sys
from itertools import product
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.codegen import _registry as registry
import options as user_options


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


VERBOSE = _env_flag("MULTIPERS_TEMPITA_GRID_VERBOSE", default=False)

OUTPUT_ROOT = Path(
    os.environ.get("MULTIPERS_TEMPITA_GRID_OUTPUT_ROOT", str(REPO_ROOT))
).resolve()
TEMPITA_CACHE_DIR = Path(
    os.environ.get("MULTIPERS_TEMPITA_CACHE_DIR", str(REPO_ROOT / "build" / "tmp"))
).resolve()


def _print_section_header(title: str) -> None:
    if not VERBOSE:
        return
    print("#----------------------")
    print("#----------------------")
    print(title)
    print("#----------------------")
    print("#----------------------")


def _unique(values):
    return list(dict.fromkeys(values))


def _require_known(name: str, values: list[str], known: set[str]) -> None:
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown {name}: {unknown}")


def _normalize_key(key: str) -> str:
    normalized = key.strip().lower()
    aliases = {
        "backend": "backend",
        "backends": "backend",
        "column": "column",
        "columns": "column",
        "value_type": "value_type",
        "value_types": "value_type",
        "filtration_container": "filtration_container",
        "filtration": "filtration_container",
        "container": "filtration_container",
        "vine": "vine",
        "is_vine": "vine",
        "vineyard": "vine",
        "vineyards": "vine",
        "kcritical": "kcritical",
        "is_kcritical": "kcritical",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown rule key '{key}'")
    return aliases[normalized]


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value '{raw}'")


def _coerce_bool_list(name: str, values: list[Any]) -> list[bool]:
    out: list[bool] = []
    for value in values:
        if isinstance(value, bool):
            out.append(value)
        elif isinstance(value, str):
            out.append(_parse_bool(value))
        else:
            raise ValueError(
                f"{name} entries must be bool or string, got {type(value)!r}"
            )
    out = _unique(out)
    if not out:
        raise ValueError(f"{name} cannot be empty")
    return out


def _split_non_empty(raw: str, sep: str) -> list[str]:
    return [chunk.strip() for chunk in raw.split(sep) if chunk.strip()]


def _build_short_token_axis() -> dict[str, str]:
    token_axis: dict[str, str] = {}

    def bind(token: str, axis: str) -> None:
        existing = token_axis.get(token)
        if existing is not None and existing != axis:
            raise ValueError(
                f"Ambiguous short-rule token '{token}' (axes: {existing}, {axis})"
            )
        token_axis[token] = axis

    for token in registry.BACKEND_REGISTRY:
        bind(token, "backend")
    for token in registry.COLUMN_REGISTRY:
        bind(token, "column")
    for token in registry.VALUE_TYPE_REGISTRY:
        bind(token, "value_type")
    for token in registry.FILTRATION_CONTAINER_SHORT_NAMES:
        bind(token, "filtration_container")

    return token_axis


SHORT_TOKEN_AXIS = _build_short_token_axis()


KNOWN_AXIS_VALUES = {
    "backend": set(registry.BACKEND_REGISTRY),
    "column": set(registry.COLUMN_REGISTRY),
    "value_type": set(registry.VALUE_TYPE_REGISTRY),
    "filtration_container": set(registry.FILTRATION_CONTAINER_SHORT_NAMES),
    "vine": {True, False},
    "kcritical": {True, False},
}


def _parse_rule_value(axis: str, token: str) -> Any:
    if axis in {"vine", "kcritical"}:
        return _parse_bool(token)
    return token


def _parse_assignment(expression: str, allow_short_lhs: bool) -> tuple[str, set[Any]]:
    expression = expression.strip()
    if not expression:
        raise ValueError("Empty rule assignment")

    if "=" in expression:
        key, raw_values = expression.split("=", 1)
        axis = _normalize_key(key)
        values = {
            _parse_rule_value(axis, token)
            for token in _split_non_empty(raw_values, "|")
        }
        if not values:
            raise ValueError(f"Missing values in assignment '{expression}'")
        unknown = values - KNOWN_AXIS_VALUES[axis]
        if unknown:
            raise ValueError(f"Unknown values {sorted(unknown)} for axis '{axis}'")
        return axis, values

    if not allow_short_lhs:
        raise ValueError(f"Expected key=value assignment, got '{expression}'")

    tokens = _split_non_empty(expression, "|")
    if not tokens:
        raise ValueError(f"Invalid short rule segment '{expression}'")

    axes = set()
    for token in tokens:
        axis = SHORT_TOKEN_AXIS.get(token)
        if axis is None:
            raise ValueError(f"Unknown short-rule token '{token}'")
        axes.add(axis)

    if len(axes) != 1:
        raise ValueError(f"Mixed-axis short rule segment '{expression}'")

    axis = axes.pop()
    values = {_parse_rule_value(axis, token) for token in tokens}
    unknown = values - KNOWN_AXIS_VALUES[axis]
    if unknown:
        raise ValueError(f"Unknown values {sorted(unknown)} for axis '{axis}'")
    return axis, values


def _parse_rule(rule_text: str) -> dict[str, Any]:
    if "=>" not in rule_text:
        raise ValueError(f"Invalid rule '{rule_text}'. Expected 'lhs => rhs'.")

    raw_when, raw_require = rule_text.split("=>", 1)
    when: dict[str, set[Any]] = {}
    require: dict[str, set[Any]] = {}

    for segment in _split_non_empty(raw_when, ","):
        axis, values = _parse_assignment(segment, allow_short_lhs=True)
        when.setdefault(axis, set()).update(values)

    for segment in _split_non_empty(raw_require, ","):
        axis, values = _parse_assignment(segment, allow_short_lhs=False)
        require.setdefault(axis, set()).update(values)

    if not when:
        raise ValueError(f"Invalid rule '{rule_text}': empty left-hand side")
    if not require:
        raise ValueError(f"Invalid rule '{rule_text}': empty right-hand side")

    return {
        "raw": rule_text,
        "when": when,
        "require": require,
    }


def _parse_constraint(constraint_text: str) -> dict[str, set[Any]]:
    constraints: dict[str, set[Any]] = {}
    for segment in _split_non_empty(constraint_text, ","):
        axis, values = _parse_assignment(segment, allow_short_lhs=False)
        constraints.setdefault(axis, set()).update(values)
    if not constraints:
        raise ValueError(f"Invalid constraint '{constraint_text}': empty expression")
    return constraints


def _matches(candidate: dict[str, Any], constraints: dict[str, set[Any]]) -> bool:
    return all(candidate[key] in allowed for key, allowed in constraints.items())


def _write_text_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _render_extern_templates_header(type_names: list[str]) -> str:
    lines = [
        "#pragma once",
        "",
        "// Auto-generated by tools/tempita_grid_gen.py. Do not edit.",
        "",
    ]
    lines.extend(f"extern template class {type_name};" for type_name in type_names)
    lines.append("")
    return "\n".join(lines)


def _render_instantiations_include(type_names: list[str]) -> str:
    lines = [
        "// Auto-generated by tools/tempita_grid_gen.py. Do not edit.",
        "",
    ]
    lines.extend(f"template class {type_name};" for type_name in type_names)
    lines.append("")
    return "\n".join(lines)


value_type_ids = _unique(list(user_options.VALUE_TYPES))
column_ids = _unique(list(user_options.COLUMNS))
matrix_types = _unique(list(user_options.BACKENDS))
filtration_containers = _unique(list(user_options.FILTRATION_CONTAINERS))
vineyards_values = _coerce_bool_list("VINE", list(user_options.VINE))
kcritical_options = _coerce_bool_list("KCRITICAL", list(user_options.KCRITICAL))

if not value_type_ids:
    raise ValueError("VALUE_TYPES cannot be empty")
if not column_ids:
    raise ValueError("COLUMNS cannot be empty")
if not matrix_types:
    raise ValueError("BACKENDS cannot be empty")
if not filtration_containers:
    raise ValueError("FILTRATION_CONTAINERS cannot be empty")

_require_known("value type IDs", value_type_ids, set(registry.VALUE_TYPE_REGISTRY))
_require_known("column IDs", column_ids, set(registry.COLUMN_REGISTRY))
_require_known("backend IDs", matrix_types, set(registry.BACKEND_REGISTRY))
_require_known(
    "filtration container IDs",
    filtration_containers,
    set(registry.FILTRATION_CONTAINER_SHORT_NAMES),
)

coarsened_value_type_id = getattr(
    user_options,
    "COARSENED_VALUE_TYPE",
    getattr(user_options, "COARSENED_VALUE_TYPE", None),
)
if coarsened_value_type_id is None:
    raise ValueError(
        "Missing COARSENED_VALUE_TYPE (or COARSENED_VALUE_TYPE) in options"
    )

real_value_type_id = user_options.REAL_VALUE_TYPE

_require_known(
    "coarsenned value type alias",
    [coarsened_value_type_id],
    set(registry.VALUE_TYPE_REGISTRY),
)
_require_known(
    "real value type alias",
    [real_value_type_id],
    set(registry.VALUE_TYPE_REGISTRY),
)

value_types = [
    registry.VALUE_TYPE_REGISTRY[value_type_id] for value_type_id in value_type_ids
]
columns_name = [registry.COLUMN_REGISTRY[column_id] for column_id in column_ids]
short_filtration_container = dict(registry.FILTRATION_CONTAINER_SHORT_NAMES)
COARSENED_VALUE_TYPE = registry.VALUE_TYPE_REGISTRY[coarsened_value_type_id]
REAL_VALUE_TYPE = registry.VALUE_TYPE_REGISTRY[real_value_type_id]

if COARSENED_VALUE_TYPE not in value_types:
    raise ValueError(
        f"COARSENED_VALUE_TYPE={coarsened_value_type_id!r} must be enabled in VALUE_TYPES"
    )
if REAL_VALUE_TYPE not in value_types:
    raise ValueError(
        f"REAL_VALUE_TYPE={real_value_type_id!r} must be enabled in VALUE_TYPES"
    )

value_type_id_by_tuple = {
    registry.VALUE_TYPE_REGISTRY[value_type_id]: value_type_id
    for value_type_id in value_type_ids
}
column_id_by_cpp = {
    registry.COLUMN_REGISTRY[column_id]: column_id for column_id in column_ids
}
parsed_rules = [
    _parse_rule(rule_text) for rule_text in _unique(list(user_options.RULES))
]

required_slicer_combination_specs = [
    {
        "raw": constraint_text,
        "constraints": _parse_constraint(constraint_text),
    }
    for constraint_text in _unique(
        list(getattr(user_options, "REQUIRED_SLICER_COMBINATIONS", []))
    )
]


def get_cfiltration_type(container, dtype, is_kcritical, co=False):
    return f"Gudhi::multi_filtration::{container}<{dtype[0]},false,!{str(is_kcritical).lower()}>"


def get_python_filtration_type(container, dtype, is_kcritical, co=False):
    return f"{'K' if is_kcritical else ''}{short_filtration_container[container]}_{dtype[2]}"


def _build_combination_candidate(
    backend_type, is_vine, is_kcritical, value_type, column_type, filtration_container
) -> dict[str, Any]:
    return {
        "backend": backend_type,
        "vine": is_vine,
        "kcritical": is_kcritical,
        "value_type": value_type_id_by_tuple[value_type],
        "column": column_id_by_cpp[column_type[1]],
        "filtration_container": filtration_container,
    }


def check_combination(
    backend_type, is_vine, is_kcritical, value_type, column_type, filtration_container
):
    candidate = _build_combination_candidate(
        backend_type,
        is_vine,
        is_kcritical,
        value_type,
        column_type,
        filtration_container,
    )
    for rule in parsed_rules:
        if _matches(candidate, rule["when"]) and not _matches(
            candidate, rule["require"]
        ):
            return False
    return True


def get_slicer_python_class_names(
    backend_type, is_vine, is_kcritical, value_type, column_type, filtration_container
):
    ctype, pytype, short_type = value_type
    col_idx, col = column_type
    python_type = f"_{'K' if is_kcritical else ''}{short_filtration_container[filtration_container]}Slicer_{backend_type}{col_idx}{'_vine' if is_vine else ''}_{short_type}"
    ctype_name = f"multipers::tmp_interface::TrucPythonInterface<multipers::tmp_interface::BackendsEnum::{backend_type},{'true' if is_vine else 'false'},{'true' if is_kcritical else 'false'},{ctype},{col},multipers::tmp_interface::Filtration_containers_strs::{filtration_container}>"
    return python_type, ctype_name


def get_slicer(
    backend_type, is_vine, is_kcritical, value_type, column_type, filtration_container
):
    ctype, pytype, short_type = value_type
    col_idx, col = column_type
    python_type, ctype_name = get_slicer_python_class_names(
        backend_type,
        is_vine,
        is_kcritical,
        value_type,
        column_type,
        filtration_container,
    )
    filtration_type = get_python_filtration_type(
        filtration_container, value_type, is_kcritical
    )
    return {
        "TRUC_TYPE": ctype_name,
        "C_TEMPLATE_TYPE": "C" + python_type,
        "PYTHON_TYPE": python_type,
        "IS_SIMPLICIAL": False,
        "IS_VINE": is_vine,
        "IS_KCRITICAL": is_kcritical,
        "C_VALUE_TYPE": ctype,
        "PY_VALUE_TYPE": pytype,
        "COLUMN_TYPE": column_id_by_cpp[col],
        "SHORT_VALUE_TYPE": short_type,
        "SHORT_FILTRATION_TYPE": short_filtration_container[filtration_container],
        "FILTRATION_TYPE": filtration_type,
        "FILTRATION_CONTAINER_STR": filtration_container,
        "PERS_BACKEND_TYPE": backend_type,
        "IS_FLOAT": short_type[0] == "f",
        "REAL_PY_CLASS_NAME": get_slicer_python_class_names(
            backend_type,
            is_vine,
            is_kcritical,
            REAL_VALUE_TYPE,
            column_type,
            filtration_container,
        )[0],
        "COARSENNED_PY_CLASS_NAME": get_slicer_python_class_names(
            backend_type,
            is_vine,
            is_kcritical,
            COARSENED_VALUE_TYPE,
            column_type,
            filtration_container,
        )[0],
    }


enabled_combinations = [
    kwargs
    for backend_type, is_vine, is_kcritical, value_type, column_type, filtration_container in product(
        matrix_types,
        vineyards_values,
        kcritical_options,
        value_types,
        enumerate(columns_name),
        filtration_containers,
    )
    if check_combination(
        **(
            kwargs := {
                "backend_type": backend_type,
                "is_vine": is_vine,
                "is_kcritical": is_kcritical,
                "value_type": value_type,
                "column_type": column_type,
                "filtration_container": filtration_container,
            }
        )
    )
]

slicers = [get_slicer(**kwargs) for kwargs in enabled_combinations]

enabled_candidates = [
    _build_combination_candidate(**kwargs) for kwargs in enabled_combinations
]

enabled_filtration_combinations = {
    (
        value_type_id_by_tuple[kwargs["value_type"]],
        kwargs["filtration_container"],
        kwargs["is_kcritical"],
    )
    for kwargs in enabled_combinations
}


def _is_filtration_combo_enabled(
    value_type, filtration_container, is_kcritical
) -> bool:
    return (
        value_type_id_by_tuple[value_type],
        filtration_container,
        is_kcritical,
    ) in enabled_filtration_combinations


TEMPITA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


_print_section_header("Value_types")
if VERBOSE:
    print(*value_types, sep="\n")
with (TEMPITA_CACHE_DIR / "_value_types.pkl").open("wb") as f:
    pickle.dump(value_types, f)

_print_section_header("Filtrations")
Filtrations = [
    {
        "python": get_python_filtration_type(F, T, K),
        "c": get_cfiltration_type(F, T, K),
        "c_value_type": T[0],
        "py_value_type": T[1],
        "short_value_type": T[2],
        "container": F,
        "multicritical": K,
    }
    for F, T, K in product(filtration_containers, value_types, kcritical_options)
    if _is_filtration_combo_enabled(
        value_type=T,
        filtration_container=F,
        is_kcritical=K,
    )
]
if VERBOSE:
    print(*Filtrations, sep="\n")
with (TEMPITA_CACHE_DIR / "_filtration_names.pkl").open("wb") as f:
    pickle.dump(Filtrations, f)

_print_section_header("Slicers")
if VERBOSE:
    print(*[s["PYTHON_TYPE"] for s in slicers], sep="\n")
    print("----------------------")
    print(*slicers, sep="\n")
with (TEMPITA_CACHE_DIR / "_slicer_names.pkl").open("wb") as f:
    pickle.dump(slicers, f)

with (TEMPITA_CACHE_DIR / "_codegen_defaults.pkl").open("wb") as f:
    pickle.dump(
        {
            "default_column_type": column_ids[0],
        },
        f,
    )

_print_section_header("SimplexTrees")


def get_simplextree_class_name(is_kcritical, value_type, filtration_container):
    ctype, pytype, short_type = value_type
    python_class_name = (
        "_SimplexTreeMulti_"
        + short_filtration_container[filtration_container]
        + "_"
        + ("K" if is_kcritical else "")
        + short_type
    )
    return python_class_name


def get_simplextree(is_kcritical, value_type, filtration_container):
    ctype, pytype, short_type = value_type
    python_filtration = get_python_filtration_type(
        filtration_container, value_type, is_kcritical
    )
    one_k_python_filtration = get_python_filtration_type(
        filtration_container, value_type, False
    )
    python_class_name = get_simplextree_class_name(
        is_kcritical, value_type, filtration_container
    )
    coarsenned_class_name = get_simplextree_class_name(
        is_kcritical, COARSENED_VALUE_TYPE, filtration_container
    )
    real_class_name = get_simplextree_class_name(
        is_kcritical, REAL_VALUE_TYPE, filtration_container
    )
    return {
        "IS_KCRITICAL": is_kcritical,
        "CTYPE": ctype,
        "PY_VALUE_TYPE": pytype,
        "PYTYPE": pytype,
        "SHORT_VALUE_TYPE": short_type,
        "SHORT_FILTRATION_TYPE": short_filtration_container[filtration_container],
        "PyFil": python_filtration,
        "CFil": get_cfiltration_type(filtration_container, value_type, is_kcritical),
        "FILTRATION_CONTAINER_STR": filtration_container,
        "IS_FLOAT": short_type[0] == "f",
        "PY_CLASS_NAME": python_class_name,
        "COARSENNED_PY_CLASS_NAME": coarsenned_class_name,
        "REAL_PY_CLASS_NAME": real_class_name,
        "ST_INTERFACE": (
            "Simplex_tree_multi_interface[" + python_filtration + ", " + ctype + "]"
        ),
        "C2P_Fil": f"{python_filtration}_2_python",
        "P2C_Fil": f"python_2_{python_filtration}",
        "P2C_1KFil": f"python_2_{one_k_python_filtration}",
        "C2P_vFil": f"vect_{python_filtration}_2_python",
        "OneCriticalFil": one_k_python_filtration,
    }


st_list = [
    get_simplextree(*args)
    for args in product(kcritical_options, value_types, filtration_containers)
    if _is_filtration_combo_enabled(
        value_type=args[1],
        filtration_container=args[2],
        is_kcritical=args[0],
    )
]
if VERBOSE:
    print(*st_list, sep="\n")


filtration_python_types = {f["python"] for f in Filtrations}
if not filtration_python_types:
    raise ValueError(
        "No filtration types remain after applying options/rules. "
        "Please relax VALUE_TYPES/FILTRATION_CONTAINERS/KCRITICAL or RULES."
    )

slicer_python_types = {s["PYTHON_TYPE"] for s in slicers}
if not slicer_python_types:
    raise ValueError(
        "No slicer types remain after applying options/rules. "
        "Please relax BACKENDS/COLUMNS/VINE/KCRITICAL/FILTRATION_CONTAINERS or RULES."
    )

missing_required_slicer_combinations = [
    spec["raw"]
    for spec in required_slicer_combination_specs
    if not any(
        _matches(candidate, spec["constraints"]) for candidate in enabled_candidates
    )
]
if missing_required_slicer_combinations:
    raise ValueError(
        "Options/rules violate REQUIRED_SLICER_COMBINATIONS from options.py. "
        f"Missing combinations: {missing_required_slicer_combinations}."
    )

missing_coarsened_aliases = []
for slicer in slicers:
    coarsenned = slicer["COARSENNED_PY_CLASS_NAME"]
    if coarsenned not in slicer_python_types:
        missing_coarsened_aliases.append((slicer["PYTHON_TYPE"], coarsenned))
if missing_coarsened_aliases:
    examples = ", ".join(f"{src}->{dst}" for src, dst in missing_coarsened_aliases[:6])
    raise ValueError(
        "Options/rules break coarsening API requirements (missing coarsened slicer aliases). "
        f"Examples: {examples}"
    )

with (TEMPITA_CACHE_DIR / "_simplextrees_.pkl").open("wb") as f:
    pickle.dump(st_list, f)


simplextree_instantiation_types = _unique(
    [
        "Gudhi::multiparameter::python_interface::Simplex_tree_multi_interface<"
        + st["CFil"]
        + ", "
        + st["CTYPE"]
        + ">"
        for st in st_list
    ]
)

filtration_instantiation_types = _unique(
    [filtration["c"] for filtration in Filtrations]
)


def _bool_cpp(value: bool) -> str:
    return "true" if value else "false"


def _slicer_type_for_instantiation(slicer: dict[str, Any]) -> str:
    filtration_type = (
        "multipers::tmp_interface::filtration_options<"
        f"multipers::tmp_interface::Filtration_containers_strs::{slicer['FILTRATION_CONTAINER_STR']},"
        f"{_bool_cpp(slicer['IS_KCRITICAL'])},"
        f"{slicer['C_VALUE_TYPE']}>"
    )
    filtration_type = "".join(filtration_type)
    backend_type = (
        "multipers::tmp_interface::PersBackendOpts<"
        f"multipers::tmp_interface::BackendsEnum::{slicer['PERS_BACKEND_TYPE']},"
        f"{_bool_cpp(slicer['IS_VINE'])},"
        f"multipers::tmp_interface::Available_columns::{slicer['COLUMN_TYPE']},"
        f"{filtration_type}>"
    )
    backend_type = "".join(backend_type)
    return f"Gudhi::multi_persistence::Slicer<{filtration_type}, {backend_type}>"


slicer_instantiation_types = _unique(
    [_slicer_type_for_instantiation(slicer) for slicer in slicers]
)

_write_text_if_changed(
    OUTPUT_ROOT / "multipers/gudhi/filtrations_extern_templates.h",
    _render_extern_templates_header(filtration_instantiation_types),
)
_write_text_if_changed(
    OUTPUT_ROOT / "multipers/gudhi/simplextree_multi_extern_templates.h",
    _render_extern_templates_header(simplextree_instantiation_types),
)
_write_text_if_changed(
    OUTPUT_ROOT / "multipers/gudhi/slicer_extern_templates.h",
    _render_extern_templates_header(slicer_instantiation_types),
)
_write_text_if_changed(
    OUTPUT_ROOT / "tools/core/filtrations_instantiations.inc",
    _render_instantiations_include(filtration_instantiation_types),
)
_write_text_if_changed(
    OUTPUT_ROOT / "tools/core/simplextree_instantiations.inc",
    _render_instantiations_include(simplextree_instantiation_types),
)
_write_text_if_changed(
    OUTPUT_ROOT / "tools/core/slicer_instantiations.inc",
    _render_instantiations_include(slicer_instantiation_types),
)

print(
    "[tempita-grid] "
    f"value_types={len(value_types)} "
    f"filtrations={len(Filtrations)} "
    f"slicers={len(slicers)} "
    f"simplextrees={len(st_list)} "
    f"filtration_instantiations={len(filtration_instantiation_types)} "
    f"slicer_instantiations={len(slicer_instantiation_types)} "
    f"simplextree_instantiations={len(simplextree_instantiation_types)}"
)
