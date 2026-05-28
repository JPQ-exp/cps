# `cps` — Developer Usage & Integration Guide

This guide provides a detailed walkthrough for integrating, configuring, and operating the Coupled Persistence System (`cps`) within your software architectures . It covers the system's operational workflow, step-by-step API integration, and patterns for handling edge cases like latency delays and error recovery .

---

## 1. Operational Workflow Overview

A single execution step in the Coupled Persistence System follows a deterministic, sequential pipeline :

```text
  [ 1. Observe State ] ---> [ 2. Calculate Margin & Gradient ]
                                          |
                                          v
  [ 4. Build COL Matrix ] <--- [ 3. Compute Metabolic Intensity ]
           |
           v
  [ 5. Project Velocity ] ---> [ 6. Run Ct Transition ] ---> [ 7. Deduct Energy Cost ]
```

1. **Observe**: The current state vector $x(t)$ is retrieved from the `Automaton` .
2. **Evaluate**: The safety margin $\psi(x)$ and regularized direction vector $\hat{\nabla}\Pi(x)$ are calculated using the user-defined `ViabilityField` .
3. **Intensity**: The metabolic scalar $I_t$ is computed, balancing available reserve capital $R(t)$ against boundary proximity $\psi(x)$ .
4. **COL Alignment**: The raw directional vector is projected using the Orthant Lattice generators to yield a stable coordination matrix $C(t)$ .
5. **Velocity**: The state trajectory velocity $\dot{x}$ is calculated, incorporating any external environmental drift or noise perturbations .
6. **Ct Transition**: An Euler-integration state transition step is executed, checking trajectory buffers for delayed coordinates if latency is configured .
7. **Expenditure**: The cost $c(u)$ of the action is computed and deducted from the metabolic reserve $R$ .

---

## 2. Step-by-Step API Implementation

### Step 1: Setting up the `ViabilityField`
The system requires an explicit safety boundary . You must provide a Python callable that accepts a 1D NumPy array and returns a float representing the distance or safety metric (where values $\geq 0$ represent the viable envelope) .

```python
import numpy as np
from cps import ViabilityField

# Define a 3D ellipsoidal safety envelope
# State must remain inside: 1.5 * x^2 + y^2 + 0.8 * z^2 <= 9.0
def ellipsoid_boundary(state: np.ndarray) -> float:
    scaled_dist = 1.5 * state[0]**2 + state**2 + 0.8 * state**2
    return 9.0 - scaled_dist

# Instantiate the field. You can optionally supply an analytical gradient function
# to bypass numerical central finite difference computations.
viability_field = ViabilityField(boundary_fn=ellipsoid_boundary)

# Verify margins manually
state_sample = np.array([1.0, 1.0, 1.0])
print("Current Margin:", viability_field.margin(state_sample))
print("Numerical Gradient:", viability_field.gradient(state_sample))
```

### Step 2: Configuring the `Automaton`
The `Automaton` represents the state-holding agent . It must be initialized with a starting state, a positive starting reserve, and your configured `ViabilityField` .

```python
from cps import Automaton

initial_state = np.array([0.0, 0.0, 0.0]) # Starting at center of the safe ellipsoid
initial_reserve = 5.0                     # Initial thermodynamic resource budget

agent = Automaton(
    state=initial_state,
    reserve=initial_reserve,
    viability_field=viability_field,
    min_reserve=1e-4                      # Absolute lower-bound safety reserve floor
)
```

### Step 3: Setting up the `ConstrainedOrthantLattice` (COL)
The COL manages directional contradictions . The `hysteresis_threshold` determines the minimum action magnitude required to switch coordinate signs, while `max_flip_rate` prevents high-frequency limit cycles .

```python
from cps import ConstrainedOrthantLattice

col_filter = ConstrainedOrthantLattice(
    dimension=3,
    hysteresis_threshold=0.05,  # Damps any oscillations with magnitude smaller than 0.05
    max_flip_rate=0.4,          # Triggers InstabilityError if sign-flips occur in >40% of window steps
    rolling_window=20           # Window size for monitoring sign-flip frequency
)
```

### Step 4: Configuring the Transition Core (`Ct`)
If your control loop experiences sensory lag or communications delay, instantiate `Ct` with a non-zero `delay` setting to establish historical state tracking queues .

```python
from cps import Ct

# Setup transition dynamics with a 15-step lag tracking queue
transition_core = Ct(
    delay=15,
    max_depth=128  # Limits recursive transformations to prevent stack overflow issues
)
```

### Step 5: Initializing and Running the `Coordinator`
The `Coordinator` drives the simulation loop by executing the discrete transitions and managing metabolic depletion .

```python
from cps import Coordinator

coordinator = Coordinator(
    automaton=agent,
    col=col_filter,
    ct=transition_core,
    alpha=1.2,                # Metabolic allocation intensity modifier
    beta=0.5,                 # Scale modifier for kinetic action cost
    rest_matrix_scale=0.01,   # Scale of rest/dissipation identity matrix when passive
    max_action_magnitude=5.0  # Safe rate limit on potential gradients
)
```

---

## 3. Advanced Integration Patterns

### Pattern A: Handling Sensory Latency Delays
In environments with significant communications lag, you must record intermediate trajectory steps to ensure the delay buffer stays populated .

```python
# Seed the delay buffer with initial states before starting the loop
for _ in range(transition_core.delay + 1):
    transition_core.record_state(agent.observe())

dt = 0.01
for step in range(100):
    # Retrieve the state at t - tau to compute lag-aware feedback
    delayed_state = transition_core.get_delayed_state(fallback_state=agent.observe())
    
    # Calculate external drift or targeting force acting on the delayed state
    drift_force = np.array([0.1, -0.05, 0.02])
    
    # Execute step
    coordinator.step(dt=dt, environmental_drift=drift_force)
```

### Pattern B: Auditing System Trajectories
If `enable_audit_log` is active, the `Coordinator` logs every transition step inside an internal list of immutable `StateAuditLog` traces . This is useful for post-incident diagnostics or debugging state instabilities .

```python
# Run several steps...
for _ in range(10):
    coordinator.step(dt=0.01)

# Inspect the accumulated logs
for log in coordinator.audit_log[-5:]:  # Print last 5 states
    print(f"Time: {log.timestamp:.2f}s | Reserve: {log.reserve:.3f} | Margin: {log.margin:.3f}")
    # Extract metadata properties used during the step calculations
    intensity = log.metadata["intensity"]
    print(f"  Metabolic Intensity: {intensity:.4f}")
```

### Pattern C: Error Recovery & Safe Shutdown
In complex systems, external disturbances can force the state vector toward boundary thresholds, exhausting resource reserves or causing high-frequency oscillations . Wrap execution loops in explicit exception handlers to perform graceful shutdowns .

```python
from cps.exceptions import InstabilityError, ReserveExhaustionError

try:
    while agent.alive:
        # Simulate extreme external disturbance forces
        extreme_disturbance = np.random.normal(0, 1.5, size=3)
        coordinator.step(dt=0.01, environmental_drift=extreme_disturbance)
        
except InstabilityError as err:
    # Handle actuator chatter/jitter errors before physical hardware damage occurs
    print(f"Safety Halt: High-frequency controller jitter detected: {err}")
    # Perform physical shutdown sequence here
    
except ReserveExhaustionError as err:
    # Handle resource exhaustion
    print(f"Safety Halt: Insufficient metabolic energy reserves: {err}")
    # Switch hardware to low-power hibernation state
    
finally:
    print(f"Final State: {agent.observe()} | Final Reserve Balance: {agent.reserve:.4f}")
```

---

## 4. Troubleshooting & Stabilization Guidelines

If you encounter issues during loop integration, review these standard configuration adjustments:

| Symptom | Probable Cause | Corrective Action |
| :--- | :--- | :--- |
| **`InstabilityError` raised early** | Hysteresis threshold is set too low for the noise level of the drift signal. | Increase `col.hysteresis_threshold` or apply a low-pass filter to environmental observations. |
| **System collapses rapidly (reserves deplete to 0)** | The metabolic scaling coefficient `beta` is too high, or joint velocities are excessive. | Reduce the step size `dt` or lower the `max_action_magnitude` boundary limit. |
| **Agent drifts outside safe margins without correcting** | The persistence operator is underpowered. | Increase `alpha` to allocate more metabolic reserve energy to boundary correction actions. |