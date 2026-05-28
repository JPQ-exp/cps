"""
Coupled Persistence System (CPS) multiscale state dynamics flow.
"""
import numpy as np

def compute_augmented_potential_gradient(
    normalized_grad: np.ndarray,
    perturbation: np.ndarray,
    intensity_scalar: float,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Computes the augmented potential gradient \nabla V*(x, R, z).
    
    Based on Equation (15) and Algorithm 1:
        \nabla V* = \hat{\nabla}\Pi(x) + ( 1 / (I_t + \varepsilon) ) * e
        
    Where:
        - \hat{\nabla}\Pi(x) is `normalized_grad`
        - e is the drift/regulatory perturbation vector `perturbation`
        - I_t is the metabolic intensity scalar `intensity_scalar`
    """
    return normalized_grad + (1.0 / (intensity_scalar + eps)) * perturbation


def evaluate_state_velocity(
    persistence_matrix: np.ndarray,
    augmented_gradient: np.ndarray,
    use_absolute_coupling: bool = False
) -> np.ndarray:
    """
    Evaluates the continuous coordinate velocity vector \dot{x}.
    
    Based on Equation (16) and Algorithm 1:
        \dot{x} = P * \nabla V*   (or P * |\nabla V*| for rectified COL actions)
    
    Parameters:
        persistence_matrix: Bounded persistence transformation matrix (P)
        augmented_gradient: Gradient of augmented potential landscape (\nabla V*)
        use_absolute_coupling: If True, uses element-wise absolute potential values 
                               to match Algorithmic rectifications of COL.
    """
    if use_absolute_coupling:
        return np.dot(persistence_matrix, np.abs(augmented_gradient))
    return np.dot(persistence_matrix, augmented_gradient)