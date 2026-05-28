"""
Core viability tracking, margins, and boundaries.
Provides a strictly explicit ViabilityField to prevent hidden viability functions.
"""
import numpy as np
from typing import Callable, Optional

class ViabilityField:
    """
    Represents the viability zone V = {x in X : \Pi(x) >= 0}.
    Requires explicit user-defined boundary and margin functions.
    """
    def __init__(
        self, 
        boundary_fn: Callable[[np.ndarray], float], 
        margin_fn: Optional[Callable[[np.ndarray], float]] = None,
        gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ):
        if boundary_fn is None:
            raise ValueError(
                "Viability boundaries must be defined explicitly. "
                "Automatic/hidden boundaries are disabled for system safety."
            )
        self._boundary_fn = boundary_fn
        self._margin_fn = margin_fn if margin_fn is not None else boundary_fn
        self._gradient_fn = gradient_fn

    def viability(self, x: np.ndarray) -> float:
        """
        Evaluates the raw fitness functional \Pi(x).
        """
        return self._boundary_fn(x)

    def margin(self, x: np.ndarray) -> float:
        """
        Computes the viability margin \psi(x), representing the 
        distance to the boundary of the viability zone.
        """
        return self._margin_fn(x)

    def gradient(self, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Computes the gradient \nabla\Pi(x). Falls back to central finite 
        differences if an analytical gradient function is not provided.
        """
        if self._gradient_fn is not None:
            return self._gradient_fn(x)
        
        # Central difference approximation
        grad = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            x_plus = x.copy().astype(float)
            x_minus = x.copy().astype(float)
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self._boundary_fn(x_plus) - self._boundary_fn(x_minus)) / (2.0 * eps)
        return grad


# Standalone function mappings as required by the specifications
def viability(x: np.ndarray, field: ViabilityField) -> float:
    """Standalone wrapper to evaluate \Pi(x) on an explicit ViabilityField."""
    return field.viability(x)


def margin(x: np.ndarray, field: ViabilityField) -> float:
    """Standalone wrapper to evaluate the margin \psi(x) on an explicit ViabilityField."""
    return field.margin(x)


def gradient(x: np.ndarray, field: ViabilityField, eps: float = 1e-4) -> np.ndarray:
    """Standalone wrapper to evaluate the gradient \nabla\Pi(x) on an explicit ViabilityField."""
    return field.gradient(x, eps)