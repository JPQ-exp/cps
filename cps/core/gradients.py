"""
Regularized and normalized viability gradients.
"""
import numpy as np

def normalized_gradient(grad: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalizes the raw viability gradient to guarantee bounded directional actions.
    
    Based on equations (8) and (9):
        \hat{\nabla}\Pi(x) = \nabla\Pi(x) / ( ||\nabla\Pi(x)||_2 + \epsilon )
    
    Parameters:
        grad: Raw gradient vector \nabla\Pi(x)
        eps: Regularization parameter (\varepsilon > 0)
    """
    norm = np.linalg.norm(grad)
    return grad / (norm + eps)