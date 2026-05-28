"""
System-wide configuration parameters and invariant limits.
"""

# Hard recursion limit for state dynamics and Ct evaluations
MAX_RECURSION_DEPTH: int = 128

# Minimum allowable metabolic reserve before action suppression kicks in
MIN_RESERVE: float = 1e-4

# Default safety epsilon for numerical stabilization
DEFAULT_EPSILON: float = 1e-6