"""
Recursive persistence state transitions and delayed sensory buffers.
"""
import numpy as np
from collections import deque
from cps.config import MAX_RECURSION_DEPTH
from cps.exceptions import RecursionLimitError


class Ct:
    """
    Handles delayed state recording and manages state trajectory updates.
    """
    def __init__(self, delay: int = 25, max_depth: int = MAX_RECURSION_DEPTH):
        self.delay = delay
        self.max_depth = max_depth
        # Set buffer size to delay + 1 to hold the delayed state at index 0
        self.buffer = deque(maxlen=delay + 1 if delay > 0 else 1)

    def record_state(self, state: np.ndarray) -> None:
        """
        Records the current state configuration into the sensory buffer history.
        """
        self.buffer.append(np.copy(state))

    def get_delayed_state(self, fallback_state: np.ndarray) -> np.ndarray:
        """
        Returns state at t - \tau if buffer is filled, otherwise returns fallback.
        """
        if len(self.buffer) > self.delay:
            return self.buffer[0]
        return np.copy(fallback_state)

    def transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reserve: float,
        dt: float,
        depth: int = 0
    ) -> np.ndarray:
        """
        Performs Euler integration transition step with bounded recursion guards.
        """
        if depth > self.max_depth:
            raise RecursionLimitError(
                f"Transition step calculation exceeded maximum "
                f"allowable depth of {self.max_depth} levels."
            )

        # Standard Euler flow mapping: x(t+dt) = x(t) + dt * \dot{x}
        x_next = state + dt * action
        return x_next