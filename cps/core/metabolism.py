"""
Thermodynamic reserve dynamics and metabolic dissipation tracking.
"""
import numpy as np

def update_reserve(
    reserve: float,
    action_cost: float,
    gain: float = 0.0,
    decay_rate: float = 0.0,
    dt: float = 1.0
) -> float:
    """
    Updates the thermodynamic metabolic reserve density R.
    
    Based on the differential equation (4):
        \dot{R} = \sigma(z, e) - \ell*R - c(u)
    
    Where:
        - \sigma(z, e) is represented by `gain`
        - \ell is the intrinsic dissipation rate represented by `decay_rate`
        - c(u) is the metabolic action cost represented by `action_cost`
    
    Parameters:
        reserve: Current metabolic reserve (R)
        action_cost: The computed metabolic cost c(u) associated with control action magnitude
        gain: Energy assimilation rate from environmental resources
        decay_rate: Intrinsic baseline maintenance decay rate
        dt: Time delta step
    """
    # R_dot = gain - decay_rate * reserve - action_cost
    reserve_dot = gain - (decay_rate * reserve) - action_cost
    
    # Apply Euler integration step
    next_reserve = reserve + reserve_dot * dt
    
    # Physical systems cannot have negative thermodynamic reserve.
    # The actual boundary collapse is monitored via Exception handlers.
    return max(0.0, next_reserve)