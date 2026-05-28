"""
cps_tutorial.py
Minimal self-contained simulation of a 2D agent tracking a homeostatic boundary.
"""
import numpy as np
import matplotlib.pyplot as plt  # Optional: for plotting results

# Import the core CPS primitives
from cps import (
    ViabilityField,
    ConstrainedOrthantLattice,
    Automaton,
    Ct,
    Coordinator,
)
from cps.exceptions import CPSError

# ---------------------------------------------------------
# Step 1: Define the Explicit Viability Envelope
# ---------------------------------------------------------
# The agent is considered viable if it remains inside a circle of radius 2.0.
# Mathematically: \Pi(x) = 4.0 - (x_0^2 + x_1^2) >= 0
def circular_safety_boundary(state: np.ndarray) -> float:
    squared_distance = np.sum(state ** 2)
    return 4.0 - squared_distance

# Wrap the boundary function in the explicit ViabilityField class [1].
# This class automatically falls back to numerical central differences to
# compute viability gradients when an analytical gradient is not provided [1, 2].
viability_field = ViabilityField(boundary_fn=circular_safety_boundary)


# ---------------------------------------------------------
# Step 2: Initialize System Components
# ---------------------------------------------------------
# Start the agent near the boundary at [1.2, 1.2] with a 5.0 unit energy reserve.
initial_state = np.array([1.2, 1.2])
initial_reserve = 5.0

# 1. Instantiate the Automaton (the state-carrying process) [1]
agent = Automaton(
    state=initial_state,
    reserve=initial_reserve,
    viability_field=viability_field,
    min_reserve=1e-4  # Hard reserve floor below which actions are suppressed [1]
)

# 2. Instantiate the Constrained Orthant Lattice (COL) [1]
# The lattice suppresses high-frequency coordinate oscillations (metabolic jitter) [2]
col_filter = ConstrainedOrthantLattice(
    dimension=2,
    hysteresis_threshold=0.05,  # Filters out micro-adjustments smaller than 0.05
    max_flip_rate=0.4           # Bails out if sign-flips occur in > 40% of steps
)

# 3. Instantiate the Transition Core (Ct) with a 5-step sensory delay queue [2]
transition_core = Ct(
    delay=5,
    max_depth=128
)

# 4. Bind everything together inside the Coordinator [1]
coordinator = Coordinator(
    automaton=agent,
    col=col_filter,
    ct=transition_core,
    alpha=1.2,                # Energy-to-action allocation scaling factor [2]
    beta=0.5,                 # Kinetic cost modifier for action expenditure [2]
    rest_matrix_scale=0.01,   # Minimal background dissipation
    max_action_magnitude=3.0, # Bounded action rate limit for stability [1]
    enable_audit_log=True     # Instructs the loop to log immutable step traces [1]
)


# ---------------------------------------------------------
# Step 3: Run the Coordination Loop
# ---------------------------------------------------------
# Seed the delay buffer with initial states to populate the lag queue [2]
for _ in range(transition_core.delay + 1):
    transition_core.record_state(agent.observe())

dt = 0.05
simulation_steps = 200

print("Starting simulation loop...")
try:
    for step in range(simulation_steps):
        # Retrieve the delayed state at t - \tau [2]
        delayed_state = transition_core.get_delayed_state(fallback_state=agent.observe())
        
        # Simulate a noisy environmental drift (e.g. constant wind blowing towards North-East)
        # combined with standard random white noise
        constant_drift = np.array([0.3, 0.3])
        random_gusts = np.random.normal(0, 0.1, size=2)
        environmental_force = constant_drift + random_gusts
        
        # Execute single coupled persistence step
        # This resolves gradients, applies COL filtering, runs Ct, and updates reserves [1, 2]
        coordinator.step(dt=dt, environmental_drift=environmental_force)
        
        # Query status
        if not agent.alive:
            print(f"Agent collapsed at step {step} due to boundary crossing or energy starvation.")
            break
            
    else:
        print("Simulation completed successfully. Agent remained viable [2].")

except CPSError as err:
    print(f"Safety interrupt triggered: {err} [1]")


# ---------------------------------------------------------
# Step 4: Extract and Visualize the Audit Logs
# ---------------------------------------------------------
# Reconstructing trajectories from the immutable audit logs [1]
audit_records = coordinator.audit_log

timestamps = [log.timestamp for log in audit_records]
states = np.array([log.state for log in audit_records])
actions = np.array([log.action for log in audit_records])
reserves = [log.reserve for log in audit_records]
margins = [log.margin for log in audit_records]

print("\n--- Final Diagnostics ---")
print(f"Steps Executed:       {len(audit_records)}")
print(f"Final State:          {states[-1]}")
print(f"Final Reserves:       {reserves[-1]:.4f} units")
print(f"Final Viability Margin: {margins[-1]:.4f}")

# Optional: Plot the state trajectory and reserves over time using matplotlib
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left Plot: Spatial Trajectory
    # Draw the circular safety boundary (radius = 2.0)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), 'r--', label='Viability Boundary (r=2.0)')
    ax1.plot(states[:, 0], states[:, 1], 'b-', label='Agent Trajectory')
    ax1.scatter(states[0, 0], states[0, 1], color='green', marker='o', label='Start')
    ax1.scatter(states[-1, 0], states[-1, 1], color='red', marker='x', label='End')
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")
    ax1.set_title("2D Boundary Trajectory Tracking")
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')
    
    # Right Plot: Metabolic Reserves and Safety Margin
    ax2.plot(timestamps, reserves, 'g-', label='Metabolic Reserves (R)')
    ax2.plot(timestamps, margins, 'orange', label='Safety Margin (\u03c8)')
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Reserves & Viability Margins Over Time")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("cps_stabilization_results.png")
    print("\nPlot saved successfully as 'cps_stabilization_results.png'.")
    plt.show()
    
except Exception as plot_err:
    print(f"\nCould not generate plot: {plot_err}")
    print("If matplotlib is not installed, you can install it using 'pip install matplotlib'.")
