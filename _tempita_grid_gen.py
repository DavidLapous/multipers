import os
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
    # ("int64_t", "np.int64",   "i64"),
    # ("float",   "np.float32", "f32"),
    ("double", "np.float64", "f64"),  # necessary
]
COARSENNED_VALUE_TYPE = ("int32_t", "np.int32", "i32")
REAL_VALUE_TYPE = ("double", "np.float64", "f64")


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
    # "Dynamic_multi_parameter_filtration",
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
    if filtration_container == "flat":
        if not is_kcritical:
            return False
        if value_type[0] == "f":
            return False
    return True


def get_slicer_python_class_names(
    backend_type, is_vine, is_kcritical, value_type, column_type, filtration_container
):
    stuff = locals()
    ctype, pytype, short_type = value_type
    col_idx, col = column_type
    PYTHON_TYPE = f"_{'K' if is_kcritical else ''}{short_filtration_container[filtration_container]}Slicer_{backend_type}{col_idx}{'_vine' if is_vine else ''}_{short_type}"
    CTYPE = f"multipers::tmp_interface::TrucPythonInterface<multipers::tmp_interface::BackendsEnum::{backend_type},{'true' if is_vine else 'false'},{'true' if is_kcritical else 'false'},{ctype},{col},multipers::tmp_interface::Filtration_containers_strs::{filtration_container}>"
    return PYTHON_TYPE, CTYPE


def get_slicer(
    backend_type, is_vine, is_kcritical, value_type, column_type, filtration_container
):
    stuff = locals()
    ctype, pytype, short_type = value_type
    col_idx, col = column_type
    PYTHON_TYPE, CTYPE = get_slicer_python_class_names(
        backend_type,
        is_vine,
        is_kcritical,
        value_type,
        column_type,
        filtration_container,
    )
    FILTRATION_TYPE = get_python_filtration_type(
        filtration_container, value_type, is_kcritical
    )
    return {
        "TRUC_TYPE": CTYPE,
        "C_TEMPLATE_TYPE": "C" + PYTHON_TYPE,
        "PYTHON_TYPE": PYTHON_TYPE,
        "IS_SIMPLICIAL": False,
        "IS_VINE": is_vine,
        "IS_KCRITICAL": is_kcritical,
        "C_VALUE_TYPE": ctype,
        "PY_VALUE_TYPE": pytype,
        "COLUMN_TYPE": col.split("::")[3],
        "SHORT_VALUE_TYPE": short_type,
        "SHORT_FILTRATION_TYPE": short_filtration_container[filtration_container],
        "FILTRATION_TYPE": FILTRATION_TYPE,
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
            COARSENNED_VALUE_TYPE,
            column_type,
            filtration_container,
        )[0],
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


os.makedirs("build/tmp", exist_ok=True)


print("#----------------------")
print("#----------------------")
print("Value_types")
print("#----------------------")
print("#----------------------")
print(*value_types, sep="\n")
with open("build/tmp/_value_types.pkl", "wb") as f:
    pickle.dump(value_types, f)

print("#----------------------")
print("#----------------------")
print("Filtrations")
print("#----------------------")
print("#----------------------")
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
]
print(*Filtrations, sep="\n")
with open("build/tmp/_filtration_names.pkl", "wb") as f:
    pickle.dump(Filtrations, f)


print("#----------------------")
print("#----------------------")
print("Slicers")
print("#----------------------")
print("#----------------------")
print(*slicers, sep="\n")
with open("build/tmp/_slicer_names.pkl", "wb") as f:
    pickle.dump(slicers, f)


print("#----------------------")
print("#----------------------")
print("SimplexTrees")
print("#----------------------")
print("#----------------------")


def get_simplextree_class_name(is_kcritical, value_type, filtration_container):
    ctype, pytype, short_type = value_type
    python_filtration = get_python_filtration_type(
        filtration_container, value_type, is_kcritical
    )
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
    IS_KCRITICAL = is_kcritical
    python_filtration = get_python_filtration_type(
        filtration_container, value_type, is_kcritical
    )
    OneKpython_filtration = get_python_filtration_type(
        filtration_container, value_type, False
    )
    python_class_name = get_simplextree_class_name(
        is_kcritical, value_type, filtration_container
    )
    coarsenned_class_name = get_simplextree_class_name(
        is_kcritical, COARSENNED_VALUE_TYPE, filtration_container
    )
    real_class_name = get_simplextree_class_name(
        is_kcritical, REAL_VALUE_TYPE, filtration_container
    )
    return {
        "IS_KCRITICAL": IS_KCRITICAL,
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
        "P2C_1KFil": f"python_2_{OneKpython_filtration}",
        "C2P_vFil": f"vect_{python_filtration}_2_python",
        "OneCriticalFil": OneKpython_filtration,
    }


st_list = [
    get_simplextree(*args)
    for args in product(kcritical_options, value_types, filtration_containers)
]
print(*st_list, sep="\n")

with open("build/tmp/_simplextrees_.pkl", "wb") as f:
    pickle.dump(st_list, f)
