"""
Explicit custom exceptions for the Coupled Persistence System.
Provides structured error boundaries to halt execution in unsafe configurations.
"""

class CPSError(Exception):
    """
    Base exception class for all Coupled Persistence System errors.
    """
    pass


class InstabilityError(CPSError):
    """
    Raised when contradiction detection identifies critical instabilities,
    such as high-frequency sign reversals (metabolic jitter) or excessive
    action oscillation [1].
    """
    pass


class RecursionLimitError(CPSError):
    """
    Raised when recursive persistence updates (Ct) or state transitions 
    exceed the hardcoded maximum depth boundary [1].
    """
    pass


class ReserveExhaustionError(CPSError):
    """
    Raised when thermodynamic metabolic reserves drop below critical safety 
    thresholds, requiring action suppression or system shutdown [1].
    """
    pass


class ViabilityCollapseError(CPSError):
    """
    Raised when the system exits the defined superlevel set of viability,
    crossing into terminal collapse regions.
    """
    pass