"""eqsys — Nonlinear equation system solver."""

from eqsys.status import SolveStatus, ValidationStatus
from eqsys.system import EquationSystem
from eqsys.var import Var

__all__ = ["EquationSystem", "Var", "SolveStatus", "ValidationStatus"]
