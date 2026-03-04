"""Internal registry for Tempita code-generation option IDs."""

# ID -> (C++ type, Python dtype symbol, short suffix)
VALUE_TYPE_REGISTRY = {
    "int32": ("int32_t", "np.int32", "i32"),
    "int64": ("int64_t", "np.int64", "i64"),
    "float32": ("float", "np.float32", "f32"),
    "float64": ("double", "np.float64", "f64"),
}

# ID -> C++ enum value
COLUMN_REGISTRY = {
    "INTRUSIVE_SET": "multipers::tmp_interface::Available_columns::INTRUSIVE_SET",
    "SET": "multipers::tmp_interface::Available_columns::SET",
    "HEAP": "multipers::tmp_interface::Available_columns::HEAP",
    "UNORDERED_SET": "multipers::tmp_interface::Available_columns::UNORDERED_SET",
    "NAIVE_VECTOR": "multipers::tmp_interface::Available_columns::NAIVE_VECTOR",
    "VECTOR": "multipers::tmp_interface::Available_columns::VECTOR",
    "INTRUSIVE_LIST": "multipers::tmp_interface::Available_columns::INTRUSIVE_LIST",
    "LIST": "multipers::tmp_interface::Available_columns::LIST",
}

BACKEND_REGISTRY = (
    "Matrix",
    "Graph",
    "Clement",
    "GudhiCohomology",
)

# Container ID -> short name used in generated class names
FILTRATION_CONTAINER_SHORT_NAMES = {
    "Dynamic_multi_parameter_filtration": "Dynamic",
    "Degree_rips_bifiltration": "Flat",
    "Multi_parameter_filtration": "Contiguous",
}
