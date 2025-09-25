import pickle
from itertools import product

### OPTIONS

## Columns of the matrix backend.
# with ordered by their performance on some synthetic benchmarks.
columns_name = [  # only one column is necessary
    "multipers::tmp_interface::Available_columns::" + stuff
    for stuff in (
        "INTRUSIVE_SET",  # At least one of these is necessary. This one is the default
        # "SET",
        # "HEAP",
        # "UNORDERED_SET",
        # "NAIVE_VECTOR",
        # "VECTOR",
        # "INTRUSIVE_LIST",
        # "LIST",
        # "SMALL_VECTOR",
    )
]

## Value types : CTYPE, PYTHON_TYPE, short
value_types = [
    ("int32_t", "np.int32", "i32"),  # necessary
    ("int64_t", "np.int64",   "i64"),
    ("float",   "np.float32", "f32"),
    ("double", "np.float64", "f64"),  # necessary
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
    False,  # necessary
]

##
matrix_types = [
    #
    "Matrix",  # necessary
    # "Graph",
    # "Clement",
    # "GudhiCohomology",
]

filtration_containers = [
    "Dynamic_multi_parameter_filtration",
    "Degree_rips_bifiltration",
    "Multi_parameter_filtration",
]
short_filtration_container = {
    "Dynamic_multi_parameter_filtration": "Dynamic",
    "Degree_rips_bifiltration": "Flat",
    "Multi_parameter_filtration": "Contiguous",
}


def get_cfiltration_type(container, dtype, is_kcritical, co=False):
    return f"Gudhi::multi_filtration::{container}<{dtype[0]},false,!{str(is_kcritical).lower()}>"


def get_python_filtration_type(container, dtype, is_kcritical, co=False):
    return f"{'K' if is_kcritical else ''}{short_filtration_container[container]}_{dtype[2]}"


# Removes some impossible / unnecessary combinations
def check_combination(
    backend_type, is_vine, is_kcritical, value_type, column_type, filtration_container
):
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


def get_slicer(
    backend_type, is_vine, is_kcritical, value_type, column_type, filtration_container
):
    stuff = locals()
    ctype, pytype, short_type = value_type
    col_idx, col = column_type
    PYTHON_TYPE = f"_{'K' if is_kcritical else ''}{short_filtration_container[filtration_container]}Slicer_{backend_type}{col_idx}{'_vine' if is_vine else ''}_{short_type}"
    CTYPE = f"multipers::tmp_interface::TrucPythonInterface<multipers::tmp_interface::BackendsEnum::{backend_type},{'true' if is_vine else 'false'},{'true' if is_kcritical else 'false'},{ctype},{col},multipers::tmp_interface::Filtration_containers_strs::{filtration_container}>"
    IS_SIMPLICIAL = False
    IS_VINE = is_vine
    IS_KCRITICAL = is_kcritical
    # FILTRATION_TYPE = (
    #     ("Multi_critical_filtration" if is_kcritical else "One_critical_filtration")
    #     + "["
    #     + ctype
    #     + "]"
    # )
    FILTRATION_TYPE = get_python_filtration_type(
        filtration_container, value_type, is_kcritical
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
        "COLUMN_TYPE": col.split("::")[3],
        "SHORT_VALUE_TYPE": short_type,
        "FILTRATION_TYPE": FILTRATION_TYPE,
        "FILTRATION_CONTAINER_STR": filtration_container,
        "PERS_BACKEND_TYPE": backend_type,
        "IS_FLOAT": short_type[0] == "f",
    }


# {{for CTYPE_H, CTYPE,PYTHON_TYPE, IS_SIMPLICIAL, IS_VINE, IS_KCRITICAL, C_VALUE_TYPE, PY_VALUE_TYPE, COL, SHORT,FILTRATION_TYPE  in slicers}}
slicers = [
    get_slicer(**kwargs)
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


print("#----------------------")
print("Value_types")
print("#----------------------")
print(*value_types, sep="\n")
with open("build/tmp/_value_types.pkl", "wb") as f:
    pickle.dump(value_types, f)

print("#----------------------")
print("Filtrations")
print("#----------------------")
Filtrations = [
    {
        "python": get_python_filtration_type(F, T, K),
        "c": get_cfiltration_type(F, T, K),
        "c_value_type": T[0],
        "short_value_type":T[2],
        "container": F,
        "multicritical":K,
    }
    for F, T, K in product(filtration_containers, value_types, kcritical_options)
]
print(*Filtrations, sep="\n")
with open("build/tmp/_filtration_names.pkl", "wb") as f:
    pickle.dump(Filtrations, f)


print("#----------------------")
print("Slicers")
print("#----------------------")
print(*slicers, sep="\n")
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
