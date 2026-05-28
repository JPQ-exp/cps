"""
Type definitions representing strict state schemas.
"""
from typing import NamedTuple, Dict, Any, List
import numpy as np

class StateAuditLog(NamedTuple):
    """
    Immutable snapshot of the system state for deterministic debugging and auditing.
    """
    state: np.ndarray
    action: np.ndarray
    reserve: float
    margin: float
    timestamp: float
    metadata: Dict[str, Any]