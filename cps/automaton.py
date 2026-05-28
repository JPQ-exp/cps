"""
Autonomous agent dynamical process state abstraction.
"""
import numpy as np
from cps.core.viability import ViabilityField
from cps.core.gradients import normalized_gradient


class Automaton:
    """
    Encapsulates state x(t), metabolic reserve R(t), and velocity.
    Evaluates actions purely in normalized viability gradient fields.
    """
    def __init__(
        self,
        state: np.ndarray,
        reserve: float,
        viability_field: ViabilityField,
        min_reserve: float = 1e-4
    ):
        self.state = np.array(state, dtype=float)
        self.reserve = float(reserve)
        self.viability_field = viability_field
        self.min_reserve = min_reserve

        self.velocity = np.zeros_like(self.state)
        self.alive = True

    def observe(self) -> np.ndarray:
        """
        Exposes current state vector configuration.
        """
        return np.copy(self.state)

    def propose_action(self, eps: float = 1e-6) -> np.ndarray:
        """
        Returns the raw, normalized viability-climbing action vector \hat{\nabla}\Pi(x).
        Does not issue direct environmental mutations.
        """
        if not self.alive:
            return np.zeros_like(self.state)

        raw_grad = self.viability_field.gradient(self.state)
        return normalized_gradient(raw_grad, eps=eps)

    def update(self, next_state: np.ndarray, next_reserve: float) -> None:
        """
        Applies mathematical updates and evaluates metabolic/viability boundaries.
        """
        if not self.alive:
            return

        self.velocity = next_state - self.state
        self.state = np.array(next_state, dtype=float)
        self.reserve = float(next_reserve)

        # Check viability margins
        psi = self.viability_field.margin(self.state)

        # Boundary condition collapse check
        if self.reserve <= self.min_reserve or psi < -0.4:
            self.alive = False