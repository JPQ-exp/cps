"""
Coupled Persistence dynamics scaling metrics.
"""
import numpy as np
from typing import Union

def compute_persistence(
    reserve: float,
    margin: float,
    coordination_matrix: np.ndarray,
    alpha: float = 1.2,
    beta: float = 0.5,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Computes the multiscale Persistence Operator matrix P(R, \psi, C).
    
    Based on equation (13):
        P(R, \psi, C) = C(t) * ( \sqrt{\alpha * R} / \sqrt{\beta * \psi(x)} )
        
    Parameters:
        reserve: Current metabolic reserve (R)
        margin: Viability margin distance \psi(x)
        coordination_matrix: Active coordination matrix C(t) derived from COL
        alpha: Metabolic allocation intensity scale parameter (\alpha > 0)
        beta: Metabolic cost scale factor (\beta > 0)
        eps: Stabilization factor preventing division-by-zero boundaries
    """
    # Keep mathematical variables positive and valid
    safe_reserve = max(0.0, reserve)
    safe_margin = max(margin, eps)
    
    numerator = np.sqrt(alpha * safe_reserve)
    denominator = np.sqrt(beta * safe_margin)
    
    scaling_factor = numerator / (denominator + eps)
    
    return coordination_matrix * scaling_factor