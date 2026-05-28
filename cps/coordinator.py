"""
Global persistence coordinator orchestrating multiscale flows and metabolic updates.
"""
import numpy as np
from typing import Optional, List
from cps.automaton import Automaton
from cps.col import ConstrainedOrthantLattice
from cps.ct import Ct
from cps.config import MIN_RESERVE
from cps.exceptions import ReserveExhaustionError
from cps.core.metabolism import update_reserve
from cps.core.dynamics import compute_augmented_potential_gradient, evaluate_state_velocity
from cps.types import StateAuditLog


class Coordinator:
    """
    Manages the synchronization loop between the Constrained Orthant Lattice,
    persistence operator scaling, metabolic dissipation, and delay updates.
    """
    def __init__(
        self,
        automaton: Automaton,
        col: ConstrainedOrthantLattice,
        ct: Ct,
        alpha: float = 1.2,
        beta: float = 0.5,
        rest_matrix_scale: float = 0.01,
        max_action_magnitude: float = 10.0,
        max_action_frequency: float = 1.0,
        enable_audit_log: bool = True
    ):
        self.automaton = automaton
        self.col = col
        self.ct = ct
        self.alpha = alpha
        self.beta = beta

        self.dimension = automaton.state.shape[0]
        self.rest_matrix = rest_matrix_scale * np.eye(self.dimension)

        self.max_action_magnitude = max_action_magnitude
        self.max_action_frequency = max_action_frequency

        self.enable_audit_log = enable_audit_log
        self.audit_log: List[StateAuditLog] = []
        self.time_elapsed = 0.0

    def step(self, dt: float, environmental_drift: Optional[np.ndarray] = None) -> None:
        """
        Performs a single complete coordination transition cycle:
        Gather -> COL composition -> Integration -> Metabolic cost update.
        """
        if not self.automaton.alive:
            return

        if self.automaton.reserve <= MIN_RESERVE:
            raise ReserveExhaustionError("System reserve has depleted below allowable thresholds.")

        # 1. Inspect state variables
        x = self.automaton.observe()
        psi = self.automaton.viability_field.margin(x)
        raw_grad = self.automaton.viability_field.gradient(x)

        # 2. Compute metabolic intensity scaling metric (I_t)
        R = self.automaton.reserve
        intensity_denominator = np.sqrt(self.beta) * max(abs(psi), 0.05) + 1e-6
        I_t = np.sqrt(self.alpha * R) / intensity_denominator

        # 3. Handle environmental drift
        e = environmental_drift if environmental_drift is not None else np.zeros(self.dimension)

        # 4. Formulate augmented potentials
        norm_grad = raw_grad / (np.linalg.norm(raw_grad) + 1e-6)
        nabla_V_star = compute_augmented_potential_gradient(
            normalized_grad=norm_grad,
            perturbation=e,
            intensity_scalar=I_t,
            eps=1e-6
        )

        # Apply safety amplitude rate-limits
        grad_magnitude = np.linalg.norm(nabla_V_star)
        if grad_magnitude > self.max_action_magnitude:
            nabla_V_star = (nabla_V_star / grad_magnitude) * self.max_action_magnitude

        # 5. Build generators and construct coordinate scale operator matrix C(t)
        T_pos, T_neg = self.col.build_generators()
        P = self.col.compose_tensor(
            gradient_vector=nabla_V_star,
            intensity_scalar=I_t,
            rest_matrix=self.rest_matrix,
            T_pos=T_pos,
            T_neg=T_neg
        )

        # 6. Evaluate State Velocity (applying absolute rectifications of COL)
        x_dot = evaluate_state_velocity(
            persistence_matrix=P,
            augmented_gradient=nabla_V_star,
            use_absolute_coupling=True
        )

        # 7. Step state trajectory via Ct transition transform
        x_next = self.ct.transition(x, x_dot, R, dt)
        self.ct.record_state(x_next)

        # 8. Deduct metabolic action cost
        action_cost = self.beta * (np.linalg.norm(x_dot) ** 2)
        R_next = update_reserve(
            reserve=R,
            action_cost=action_cost,
            gain=0.8,
            decay_rate=0.02,
            dt=dt
        )

        # 9. Update Agent state values
        self.automaton.update(x_next, R_next)
        self.time_elapsed += dt

        # 10. Append trace to optional audit log
        if self.enable_audit_log:
            self.audit_log.append(
                StateAuditLog(
                    state=x,
                    action=x_dot,
                    reserve=R,
                    margin=psi,
                    timestamp=self.time_elapsed,
                    metadata={
                        "intensity": I_t,
                        "augmented_potential": nabla_V_star,
                        "coordination_matrix": P
                    }
                )
            )