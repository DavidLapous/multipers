"""Build-time config helpers."""

try:
    from multipers.multiparameter_module_approximation import (
        TRACE_MMA_OPS as TRACE_MMA_OPS,
    )
except Exception:
    TRACE_MMA_OPS = False
