"""
Coupled Persistence System (CPS)
A Python library for viability-aware coordination and stabilization of autonomous dynamical systems.
"""

from cps.core.viability import ViabilityField, viability, margin, gradient
from cps.core.persistence import compute_persistence
from cps.core.metabolism import update_reserve
from cps.core.gradients import normalized_gradient
from cps.core.dynamics import compute_augmented_potential_gradient, evaluate_state_velocity

from cps.automaton import Automaton
from cps.col import ConstrainedOrthantLattice
from cps.ct import Ct
from cps.coordinator import Coordinator

from cps.exceptions import (
    CPSError,
    InstabilityError,
    RecursionLimitError,
    ReserveExhaustionError,
    ViabilityCollapseError
)

__version__ = "0.1.0"