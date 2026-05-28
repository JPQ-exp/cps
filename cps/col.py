"""
Constrained Orthant Lattice (COL) algebraic projection layer.
Prevents contradictory activations and mitigates high-frequency metabolic jitter.
"""
import numpy as np
from collections import deque
from typing import Tuple, List
from cps.exceptions import InstabilityError


class ConstrainedOrthantLattice:
    """
    Implements algebraic exclusion constraints and hysteresis to prevent
    mutually destructive actuator activation loops.
    """
    def __init__(
        self,
        dimension: int,
        hysteresis_threshold: float = 0.05,
        max_flip_rate: float = 0.4,
        rolling_window: int = 20
    ):
        self.dimension = dimension
        self.hysteresis_threshold = hysteresis_threshold
        self.max_flip_rate = max_flip_rate
        self.rolling_window = rolling_window

        # Hysteresis state tracking
        self.previous_signs = np.zeros(dimension, dtype=float)
        self.previous_actions = np.zeros(dimension, dtype=float)

        # Rolling tracking of directional sign flips to detect instability
        self.flip_history = [deque(maxlen=rolling_window) for _ in range(dimension)]

    def project(self, u_raw: np.ndarray) -> np.ndarray:
        """
        Projects raw abstract actions into a stable non-contradictory action vector.
        Suppresses weaker changes below the hysteresis threshold.
        """
        u_stable = np.copy(u_raw)
        current_signs = np.sign(u_raw)

        for i in range(self.dimension):
            val = u_raw[i]
            prev_sign = self.previous_signs[i]

            # Detect dimensional sign flip
            if prev_sign != 0.0 and current_signs[i] != 0.0 and current_signs[i] != prev_sign:
                if np.abs(val) < self.hysteresis_threshold:
                    # Suppress weak signals attempting to flip direction
                    u_stable[i] = 0.0
                    current_signs[i] = 0.0
                    self.flip_history[i].append(0)
                else:
                    # Legitimate sign flip recorded
                    self.flip_history[i].append(1)
            else:
                self.flip_history[i].append(0)

            # Contradiction Detection: compute rolling sign oscillation rate
            if len(self.flip_history[i]) >= self.rolling_window:
                flip_rate = sum(self.flip_history[i]) / self.rolling_window
                if flip_rate > self.max_flip_rate:
                    raise InstabilityError(
                        f"Unstable state trajectory detected in dimension {i}. "
                        f"Oscillation rate {flip_rate:.3f} exceeds limit {self.max_flip_rate}."
                    )

        # Commit current signs and actions to state history
        for i in range(self.dimension):
            if u_stable[i] != 0.0:
                self.previous_signs[i] = np.sign(u_stable[i])
            self.previous_actions[i] = u_stable[i]

        return u_stable

    def build_generators(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Creates positive and negative orthogonal projection operators T_i^+ and T_i^-.
        Each operator represents directional coordinate activation slices.
        """
        T_pos = []
        T_neg = []
        for i in range(self.dimension):
            # Form base orthogonal projection matrix
            e_i = np.zeros((self.dimension, 1))
            e_i[i, 0] = 1.0
            T_pos.append(np.dot(e_i, e_i.T))
            T_neg.append(-np.dot(e_i, e_i.T))
        return T_pos, T_neg

    def compose_tensor(
        self,
        gradient_vector: np.ndarray,
        intensity_scalar: float,
        rest_matrix: np.ndarray,
        T_pos: List[np.ndarray],
        T_neg: List[np.ndarray]
    ) -> np.ndarray:
        """
        Assembles the coordinate scaling operator C(t) using generator maps.
        Implements Equation (11) and Step 13-21 of Algorithm 1.
        """
        T_comp = np.zeros_like(rest_matrix, dtype=float)
        alpha_sum = 0.0

        for i in range(self.dimension):
            # Choose active positive/negative slice depending on gradient direction
            sigma_i = T_pos[i] if gradient_vector[i] > 0 else T_neg[i]
            alpha_i = intensity_scalar / self.dimension

            T_comp += alpha_i * sigma_i
            alpha_sum += alpha_i

        # Maintain mass conservation / structural dissipation
        T_0 = max(0.0, 1.0 - alpha_sum)
        return T_0 * rest_matrix + T_comp