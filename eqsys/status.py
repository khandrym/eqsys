from enum import Enum


class SolveStatus(Enum):
    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    ERROR = "error"


class ValidationStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
