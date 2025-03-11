import pickle
from itertools import product

### OPTIONS

## Columns of the matrix backend.
# with ordered by their performance on some synthetic benchmarks.
columns_name = [  # only one column is necessary
    "Available_columns::" + stuff
    for stuff in (
        "INTRUSIVE_SET", # At least one of these is necessary. This one is the default
        # "SET",
        # "HEAP",
        # "UNORDERED_SET",
        "NAIVE_VECTOR",
        # "VECTOR",
        # "INTRUSIVE_LIST",
        # "LIST",
        # "SMALL_VECTOR",
    )
]

## Value types : CTYPE, PYTHON_TYPE, short
value_types = [
    ("int32_t", "np.int32",   "i32"),  # necessary
    ("int64_t", "np.int64",   "i64"),
    ("float",   "np.float32", "f32"), 
    ("double",  "np.float64", "f64"),  # necessary
]

## True needed for MMA, and False is default value
vineyards_values = [
    #
    True,
    False,
]

## Kcritical Filtrations. Default is False.
kcritical_options = [
    #
    True,
    False, # necessary
]

##
matrix_types = [
    #
    "Matrix", # necessary
    # "Graph",
    # "Clement",
    "GudhiCohomology",
]


# Removes some impossible / unnecessary combinations
def check_combination(backend_type, is_vine, is_kcritical, value_type, column_type):
    if backend_type in ["Clement", "Graph"]:
        if not is_vine:
            return False
    if backend_type in ["GudhiCohomology"]:
        if is_vine:
            return False
    if backend_type in ["Graph", "GudhiCohomology"]:
        if column_type[0] != 0:
            return False
    return True


def get_slicer(backend_type, is_vine, is_kcritical, value_type, column_type):
    stuff = locals()
    ctype, pytype, short_type = value_type
    col_idx, col = column_type
    PYTHON_TYPE = f"_{'K' if is_kcritical else ''}Slicer_{backend_type}{col_idx}{'_vine' if is_vine else ''}_{short_type}"
    CTYPE = f"TrucPythonInterface<BackendsEnum::{backend_type},{'true' if is_vine else 'false'},{'true' if is_kcritical else 'false'},{ctype},{col}>"
    IS_SIMPLICIAL = False
    IS_VINE = is_vine
    IS_KCRITICAL = is_kcritical
    FILTRATION_TYPE = (
        ("Multi_critical_filtration" if is_kcritical else "One_critical_filtration")
        + "["
        + ctype
        + "]"
    )
    return {
        "TRUC_TYPE": CTYPE,
        "C_TEMPLATE_TYPE": "C" + PYTHON_TYPE,
        "PYTHON_TYPE": PYTHON_TYPE,
        "IS_SIMPLICIAL": IS_SIMPLICIAL,
        "IS_VINE": IS_VINE,
        "IS_KCRITICAL": IS_KCRITICAL,
        "C_VALUE_TYPE": ctype,
        "PY_VALUE_TYPE": pytype,
        "COLUMN_TYPE": col.split("::")[1],
        "SHORT_VALUE_TYPE": short_type,
        "FILTRATION_TYPE": FILTRATION_TYPE,
        "PERS_BACKEND_TYPE": backend_type,
        "IS_FLOAT": short_type[0] == "f",
    }


# {{for CTYPE_H, CTYPE,PYTHON_TYPE, IS_SIMPLICIAL, IS_VINE, IS_KCRITICAL, C_VALUE_TYPE, PY_VALUE_TYPE, COL, SHORT,FILTRATION_TYPE  in slicers}}
slicers = [
    get_slicer(**kwargs)
    for backend_type, is_vine, is_kcritical, value_type, column_type in product(
        matrix_types,
        vineyards_values,
        kcritical_options,
        value_types,
        enumerate(columns_name),
    )
    if check_combination(
        **(
            kwargs := {
                "backend_type": backend_type,
                "is_vine": is_vine,
                "is_kcritical": is_kcritical,
                "value_type": value_type,
                "column_type": column_type,
            }
        )
    )
]

for D in slicers:
    print(D)

with open("build/tmp/_slicer_names.pkl", "wb") as f:
    pickle.dump(slicers, f)

## Simplextree

Filtrations_types = [
    (
        ("Multi_critical_filtration", True)
        if kcritical
        else ("One_critical_filtration", False)
    )
    for kcritical in kcritical_options
]


## CTYPE, PYTYPE, SHORT, FILTRATION
to_iter = [
    (
        CTYPE,
        PYTYPE,
        SHORT,
        Filtration + "[" + CTYPE + "]",
        is_kcritical,
        ("K" if is_kcritical else "") + "F" + SHORT,
    )
    for (CTYPE, PYTYPE, SHORT), (Filtration, is_kcritical) in product(
        value_types, Filtrations_types
    )
]


with open("build/tmp/_simplextrees_.pkl", "wb") as f:
    pickle.dump(to_iter, f)
