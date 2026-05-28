"""
Integration test and verification simulation for Coupled Persistence System (CPS).
Replicates Algorithm 1 and evaluates performance against the FEP Delayed Baseline.
"""
import numpy as np
import sys
import os

# Ensure local 'cps' directory is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from cps import (
    ViabilityField,
    ConstrainedOrthantLattice,
    Automaton,
    Ct,
    Coordinator,
    InstabilityError,
    ReserveExhaustionError
)

# Set deterministic seed for reproducibility
np.random.seed(42)

def run_simulation(agent_type: str, steps: int = 500) -> dict:
    """
    Runs a comparative simulation trial of a specified agent type.
    """
    # 1. Experimental Setup Parameters
    D = 6
    dt = 0.01
    tau = 25  # Sensory Latency Delay
    alpha = 1.2
    beta = 0.5
    
    # Initialize state trajectories
    x = np.ones(D, dtype=float)
    reserve = 2.0
    alive = True
    
    # Track delay buffer for delayed baseline evaluation
    state_history = [np.copy(x)]
    
    # Dynamic center tracking
    # Center oscillates slowly simulating periodic resource fluctuations
    def get_center(t_idx: int) -> np.ndarray:
        return np.ones(D) + 0.1 * np.sin(0.05 * t_idx)

    # 2. Setup Explicit Viability Field
    # Center state will be bound dynamically at runtime
    current_center = np.ones(D)
    
    def pi_boundary_fn(state: np.ndarray) -> float:
        # \Pi(x) = 1.2 - \sum_{i=1}^D 0.9 * (x_i - c_i)^2
        return 1.2 - np.sum(0.9 * (state - current_center) ** 2)

    field = ViabilityField(boundary_fn=pi_boundary_fn)

    # 3. Setup CPS Structures if running CPS-COL
    col = ConstrainedOrthantLattice(dimension=D, hysteresis_threshold=0.03, max_flip_rate=0.45)
    ct = Ct(delay=tau)
    automaton = Automaton(state=x, reserve=reserve, viability_field=field)
    coordinator = Coordinator(automaton=automaton, col=col, ct=ct, alpha=alpha, beta=beta)
    
    # Metric Tracking Containers
    velocities = []
    costs = []
    margins = []
    reserves = [reserve]
    jitter_flips = 0
    prev_velocity = np.zeros(D)
    
    # 4. Simulation Iteration Loop
    for t in range(steps):
        # Update current center to simulate non-stationary tracking target
        current_center = get_center(t)
        
        # Calculate current margin
        psi = field.margin(x)
        margins.append(psi)
        
        # Check boundary condition collapse
        if reserve <= 0.0 or psi < -0.4:
            alive = False
            break
            
        # Select controller
        if agent_type == "CPS-COL":
            try:
                # Stochastic environmental drift perturbation
                e = np.random.normal(0, 0.5, size=D)
                
                # Take transition step
                coordinator.step(dt=dt, environmental_drift=e)
                
                # Sync loop state back to local trackers
                x = automaton.observe()
                reserve = automaton.reserve
                x_dot = automaton.velocity
                
            except (InstabilityError, ReserveExhaustionError) as err:
                print(f"[{agent_type}] Critical Safety Interrupt at step {t}: {err}")
                alive = False
                break
                
        elif agent_type == "FEP-Baseline":
            # Delayed sensory-state retrieval
            if t > tau:
                x_delayed = state_history[t - tau]
                t_delayed = t - tau
            else:
                x_delayed = state_history[0]
                t_delayed = 0
                
            # Temporarily shift center state context to t - \tau
            current_center = get_center(t_delayed)
            raw_grad_delayed = field.gradient(x_delayed)
            
            # Compute unconstrained feedback command: \dot{x} = 2.0 * \nabla\Pi_delayed
            x_dot = 2.0 * raw_grad_delayed
            
            # Restore current center context
            current_center = get_center(t)
            
            # Apply stochastic Euler-Maruyama state transition
            noise = np.random.normal(0, 0.1, size=D)
            x_next = x + (x_dot + noise) * dt
            
            # Update reserves: R(t+1) = R(t) + (0.8 - 0.02 * R(t) - beta * ||x_dot||^2) * dt
            action_cost = beta * (np.linalg.norm(x_dot) ** 2)
            reserve_next = reserve + (0.8 - 0.02 * reserve - action_cost) * dt
            
            # Commit local variables
            x = x_next
            reserve = max(0.0, reserve_next)
            
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        # Record trajectory and measure jitter transitions
        state_history.append(np.copy(x))
        velocities.append(x_dot)
        costs.append(beta * (np.linalg.norm(x_dot) ** 2))
        reserves.append(reserve)
        
        # Calculate high-frequency change (sign variance proxy for jitter)
        if t > 0:
            sign_changes = np.sum(np.sign(x_dot) != np.sign(prev_velocity))
            jitter_flips += sign_changes
        prev_velocity = np.copy(x_dot)

    # 5. Compute Comparative Performance Metrics
    survival_rate = 100.0 if alive else 0.0
    mean_margin = np.mean(margins)
    final_reserve = reserves[-1]
    
    # Mean Metabolic Efficiency: Ratio of path travel relative to energy spent
    total_movement = np.sum([np.linalg.norm(v) for v in velocities])
    total_cost = np.sum(costs) * dt
    efficiency = total_movement / (total_cost + 1e-6)
    
    # Actuator Jitter (Hz) proxy: normalized flips per coordinate step
    jitter = jitter_flips / (steps * D)

    return {
        "Agent": agent_type,
        "Survival Rate": f"{survival_rate}%",
        "Mean Margin": f"{mean_margin:.4f}",
        "Final Reserve": f"{final_reserve:.4f}",
        "Metabolic Efficiency": f"{efficiency:.4f}",
        "Jitter Index": f"{jitter:.4f}"
    }

if __name__ == "__main__":
    print("=" * 60)
    print("CPS GITHUB IMPORT SIMULATION & VERIFICATION TRIAL")
    print("=" * 60)
    
    print("\n[Running Trial 1: FEP-Baseline (Delayed Baseline)]")
    baseline_results = run_simulation("FEP-Baseline", steps=500)
    
    print("\n[Running Trial 2: Proposed CPS-COL (Coupled Persistence)]")
    cps_results = run_simulation("CPS-COL", steps=500)
    
    print("\n" + "=" * 60)
    print("COMPARATIVE EVALUATION SUMMARY (Table 1 Benchmark)")
    print("=" * 60)
    for k in ["Survival Rate", "Mean Margin", "Final Reserve", "Metabolic Efficiency", "Jitter Index"]:
        print(f"{k:<25} | Baseline: {baseline_results[k]:<12} | CPS-COL: {cps_results[k]:<12}")
    print("=" * 60)