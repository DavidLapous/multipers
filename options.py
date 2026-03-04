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
    "UNORDERED_SET",
    # "CHATTY_COLUMN",
    # "CHATTY_COLUMN_FLIP",
    # "CHATTY_COLUMN_ADAPTIVE",
    # "PHAT_VECTOR",
    # "PHAT_BIT_TREE",
    # "INTRUSIVE_LIST",
    # "INTRUSIVE_SET",
    # "SET",
    # "HEAP",
    # "NAIVE_VECTOR",
    # "VECTOR",
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
    "Graph|GudhiCohomology => column=INTRUSIVE_SET",
]
