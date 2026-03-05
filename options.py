"""Editable code-generation options for Tempita templates.

Comment/uncomment option IDs directly in this file.
Verbose C++/Python metadata is resolved from tools/codegen/_registry.py.
"""

# Value type IDs
VALUE_TYPES = [
    "int32",  # required
    "float64",  # required for MMA
    "int64",
    "float32",
]

# Persistence matrix column IDs
COLUMNS = [
    "INTRUSIVE_LIST",
    # "UNORDERED_SET",
    # "INTRUSIVE_SET",
    # "SET",
    # "HEAP",
    # "NAIVE_VECTOR",
    # "VECTOR",
    # "PHAT_VECTOR",
    # "PHAT_BIT_TREE",
    # "LIST",
]

# Backend IDs
BACKENDS = [
    "Matrix",  # default
    # "Graph",
    # "Clement",
    "GudhiCohomology",
]

# Vineyard support options
VINE = [
    True,  # needed for MMA
    False,
]

# k-critical options
KCRITICAL = [
    True,  # needed for multi-critical filtrations
    False,
]

# Filtration container IDs
FILTRATION_CONTAINERS = [
    "Degree_rips_bifiltration",  # needed for DegreeRips
    "Multi_parameter_filtration",
    # "Dynamic_multi_parameter_filtration",
]

# Aliases used by generated Python wrappers
COARSENNED_VALUE_TYPE = "int32"
REAL_VALUE_TYPE = "float64"

# Rule format (short style):
#   "<lhs> => key=value[, key=value]"
# where <lhs> can be:
#   - short tokens: "Graph|Clement"
#   - explicit key: "backend=Graph|Clement"
RULES = [
    "Degree_rips_bifiltration => kcritical=True",
    "Graph|Clement => vine=True",
    "GudhiCohomology => vine=False",
    "Graph|GudhiCohomology => column=INTRUSIVE_LIST",
]

# Hard requirements validated after RULES are applied.
# Each entry means: at least one generated slicer must satisfy all key=value pairs.
# This keeps bridge/coarsening assumptions explicit and editable from this file.
DEFAULT_BRIDGE_COLUMN = COLUMNS[0] if COLUMNS else "INTRUSIVE_LIST"
REQUIRED_SLICER_COMBINATIONS = [
    f"backend=Matrix, vine=False, kcritical=False, value_type=float64, filtration_container=Multi_parameter_filtration, column={DEFAULT_BRIDGE_COLUMN}",
    f"backend=Matrix, vine=False, kcritical=False, value_type=int32, filtration_container=Multi_parameter_filtration, column={DEFAULT_BRIDGE_COLUMN}",
    f"backend=Matrix, vine=False, kcritical=True, value_type=float64, filtration_container=Multi_parameter_filtration, column={DEFAULT_BRIDGE_COLUMN}",
    f"backend=Matrix, vine=False, kcritical=True, value_type=int32, filtration_container=Multi_parameter_filtration, column={DEFAULT_BRIDGE_COLUMN}",
]
