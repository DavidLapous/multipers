import pickle
from itertools import product

### OPTIONS

## Columns of the matrix backend.
# with ordered by their performance on some synthetic benchmarks.
columns_name = [  # only one column is necessary
    "Available_columns::" + stuff
    for stuff in (
        "INTRUSIVE_SET",
        # "SET",
        "HEAP",
        # "UNORDERED_SET",
        "NAIVE_VECTOR",
        # "VECTOR",
        # "INTRUSIVE_LIST",
        # "LIST",
    )
]

## Value types : CTYPE, PYTHON_TYPE, short
value_types = [
    ("int32_t", "np.int32", "i32"),  # necessary
    ("int64_t", "np.int64", "i64"),
    ("float", "np.float32", "f32"),  # necessary for mma (TODO: fixme)
    ("double", "np.float64", "f64"),  # necessary
]

## True, False necessary
vineyards_values = [
    #
    True,
    False,
]

## Kcritical Filtrations
kcritical_options = [
    #
    True,
    False,
]

##
matrix_types = [
    #
    "RU",
    "Clement",
]

##  Slicers : CPP NAME, CTYPE, PYTHON_TYPE, IS_SIMPLICIAL, IS_VINE, IS_KCRITICAL, CVALUE_TYPE, PYVALUE_TYPE, COLUMN_TYPE, SHORT_DTYPE
clement_slicers = [  # this is temporarily necessary
    (
        "GeneralVineClementTruc<>",
        "GeneralVineClementTruc",
        "_SlicerClement",
        False,
        True,
        False,
        "float",
        "np.float32",
        None,
        "f32",
        "Finitely_critical_multi_filtration[float]",
    ),
    (
        "SimplicialVineGraphTruc",
        "SimplicialVineGraphTruc",
        "_SlicerVineGraph",
        True,
        True,
        False,
        "float",
        "np.float32",
        None,
        "f32",
        "Finitely_critical_multi_filtration[float]",
    ),
    (
        "SimplicialVineMatrixTruc<>",
        "SimplicialVineMatrixTruc",
        "_SlicerVineSimplicial",
        True,
        True,
        False,
        "float",
        "np.float32",
        None,
        "f32",
        "Finitely_critical_multi_filtration[float]",
    ),
    (
        "SimplicialNoVineMatrixTruc<>",
        "SimplicialNoVineMatrixTruc",
        "_SlicerNoVineSimplicial",
        True,
        False,
        False,
        "float",
        "np.float32",
        None,
        "f32",
        "Finitely_critical_multi_filtration[float]",
    ),
]


def get_slicer(is_vine, is_kcritical, value_type, column_type):
    ctype, pytype, short_type = value_type
    col_idx, col = column_type
    PYTHON_TYPE = f"_{'K' if is_kcritical else ''}Slicer{col_idx}{'_vine' if is_vine else ''}_{short_type}"
    CTYPE = f"MatrixTrucPythonInterface<{'true' if is_vine else 'false'},{'true' if is_kcritical else 'false'},{ctype},{col}>"
    IS_SIMPLICIAL = False
    IS_VINE = is_vine
    IS_KCRITICAL = is_kcritical
    FILTRATION_TYPE = (
        (
            "KCriticalFiltration"
            if is_kcritical
            else "Finitely_critical_multi_filtration"
        )
        + "["
        + ctype
        + "]"
    )
    return (
        CTYPE,
        "C" + PYTHON_TYPE,
        PYTHON_TYPE,
        IS_SIMPLICIAL,
        IS_VINE,
        IS_KCRITICAL,
        ctype,
        pytype,
        col.split("::")[1],
        short_type,
        FILTRATION_TYPE,
    )


matrix_slicers = [
    get_slicer(is_vine, is_kcritical, value_type, column_type)
    for is_vine, is_kcritical, value_type, column_type in product(
        vineyards_values, kcritical_options, value_types, enumerate(columns_name)
    )
]
slicers = []
if "clement" in matrix_types:
    slicers += clement_slicers
if "RU" in matrix_types:
    slicers += matrix_slicers

with open("build/tmp/_slicer_names.pkl", "wb") as f:
    pickle.dump(slicers, f)

## Simplextree

Filtrations_types = [
    (
        ("KCriticalFiltration", True)
        if kcritical
        else ("Finitely_critical_multi_filtration", False)
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
