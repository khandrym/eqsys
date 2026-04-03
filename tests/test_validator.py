from eqsys.validator import validate_system, ValidationIssue
from eqsys.status import ValidationStatus
from eqsys.var import Var
from eqsys.tracker import DependencyTracker


class _MockSystem:
    """Minimal system-like object for Var tracking in tests."""

    def __init__(self):
        self._tracker = DependencyTracker()


def _make_run_equation(mock: _MockSystem):
    def run_equation(eq_func):
        mock._tracker.start()
        try:
            result = eq_func()
        except Exception:
            mock._tracker.stop()
            raise
        deps = mock._tracker.stop()
        return result, deps

    return run_equation


def _run_validation(variables, equations, mock=None):
    if mock is None:
        mock = _MockSystem()
    return validate_system(variables, equations, _make_run_equation(mock))


def test_valid_system():
    mock = _MockSystem()
    x = Var("x", guess=1.0, system=mock)
    variables = [x]

    def eq():
        return x() - 1.0

    equations = [("eq1", eq)]

    status, issues = _run_validation(variables, equations, mock)
    assert status == ValidationStatus.VALID
    assert len([i for i in issues if i.level == "error"]) == 0


def test_empty_system_is_valid():
    status, issues = _run_validation([], [])
    assert status == ValidationStatus.VALID


def test_dimension_mismatch():
    mock = _MockSystem()
    x = Var("x", guess=1.0, system=mock)
    y = Var("y", guess=2.0, system=mock)
    variables = [x, y]

    def eq():
        return x() - 1.0

    equations = [("eq1", eq)]

    status, issues = _run_validation(variables, equations, mock)
    assert status == ValidationStatus.INVALID
    assert any("dimension" in i.message.lower() or "equations" in i.message.lower()
               for i in issues if i.level == "error")


def test_duplicate_variable_names():
    mock = _MockSystem()
    x1 = Var("x", guess=1.0, system=mock)
    x2 = Var("x", guess=2.0, system=mock)
    variables = [x1, x2]

    def eq1():
        return x1() - 1.0

    def eq2():
        return x2() - 2.0

    equations = [("eq1", eq1), ("eq2", eq2)]

    status, issues = _run_validation(variables, equations, mock)
    assert status == ValidationStatus.INVALID
    assert any("duplicate" in i.message.lower() and "variable" in i.message.lower()
               for i in issues if i.level == "error")


def test_duplicate_equation_names():
    mock = _MockSystem()
    x = Var("x", guess=1.0, system=mock)
    y = Var("y", guess=2.0, system=mock)
    variables = [x, y]

    def eq1():
        return x() - 1.0

    def eq2():
        return y() - 2.0

    equations = [("eq1", eq1), ("eq1", eq2)]

    status, issues = _run_validation(variables, equations, mock)
    assert status == ValidationStatus.INVALID
    assert any("duplicate" in i.message.lower() and "equation" in i.message.lower()
               for i in issues if i.level == "error")


def test_equation_raises_exception():
    mock = _MockSystem()
    x = Var("x", guess=0.0, system=mock)
    variables = [x]

    def eq():
        return 1.0 / x()  # ZeroDivisionError

    equations = [("bad_eq", eq)]

    status, issues = _run_validation(variables, equations, mock)
    assert status == ValidationStatus.INVALID
    assert any("bad_eq" in i.message for i in issues if i.level == "error")


def test_equation_returns_non_numeric():
    mock = _MockSystem()
    x = Var("x", guess=1.0, system=mock)
    variables = [x]

    def eq():
        return "not a number"

    equations = [("string_eq", eq)]

    status, issues = _run_validation(variables, equations, mock)
    assert status == ValidationStatus.INVALID
    assert any("string_eq" in i.message for i in issues if i.level == "error")


def test_equation_reads_no_variables():
    mock = _MockSystem()
    x = Var("x", guess=1.0, system=mock)
    variables = [x]

    def eq():
        return 42.0

    equations = [("const_eq", eq)]

    status, issues = _run_validation(variables, equations, mock)
    # Both warning (reads no vars) and error (x unused)
    assert status == ValidationStatus.INVALID
    assert any("const_eq" in i.message for i in issues if i.level == "warning")
    assert any("x" in i.message for i in issues if i.level == "error")


def test_variable_not_used():
    mock = _MockSystem()
    x = Var("x", guess=1.0, system=mock)
    y = Var("y", guess=2.0, system=mock)
    variables = [x, y]

    def eq1():
        return x() - 1.0

    def eq2():
        return x() * 2 - 2.0

    equations = [("eq1", eq1), ("eq2", eq2)]

    status, issues = _run_validation(variables, equations, mock)
    assert status == ValidationStatus.INVALID
    assert any("y" in i.message for i in issues if i.level == "error")


def test_guess_out_of_bounds():
    mock = _MockSystem()
    x = Var("x", guess=200.0, system=mock, min_value=0.0, max_value=100.0)
    variables = [x]

    def eq():
        return x() - 50.0

    equations = [("eq1", eq)]

    status, issues = _run_validation(variables, equations, mock)
    assert status == ValidationStatus.INVALID
    assert any("guess" in i.message.lower() for i in issues if i.level == "error")
