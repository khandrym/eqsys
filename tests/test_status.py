from eqsys.status import SolveStatus, ValidationStatus


def test_solve_status_values():
    assert SolveStatus.CONVERGED.value == "converged"
    assert SolveStatus.NOT_CONVERGED.value == "not_converged"
    assert SolveStatus.ERROR.value == "error"


def test_validation_status_values():
    assert ValidationStatus.VALID.value == "valid"
    assert ValidationStatus.INVALID.value == "invalid"
    assert ValidationStatus.WARNING.value == "warning"
