import numpy as np
from eqsys.var import Var
from eqsys.solver import newton_raphson
from eqsys.diagnostics import DiagnosticsData
from eqsys.status import SolveStatus


def _make_linear_system():
    """System: x = 3, y = 5. Residuals: x - 3, y - 5."""
    x = Var("x", guess=0.0, system=None)
    y = Var("y", guess=0.0, system=None)
    variables = [x, y]

    def eq1():
        return x() - 3.0

    def eq2():
        return y() - 5.0

    equations = [("eq1", eq1), ("eq2", eq2)]
    deps = {"eq1": {"x"}, "eq2": {"y"}}
    return variables, equations, deps


def test_solver_converges_linear():
    variables, equations, deps = _make_linear_system()
    diag = DiagnosticsData()

    def eval_fn():
        residuals = np.array([eq() for _, eq in equations])
        return residuals, deps

    status = newton_raphson(eval_fn, equations, variables, diag)
    assert status == SolveStatus.CONVERGED
    np.testing.assert_allclose(variables[0].value, 3.0, atol=1e-7)
    np.testing.assert_allclose(variables[1].value, 5.0, atol=1e-7)


def test_solver_converges_nonlinear():
    """System: x^2 = 4 -> x = 2 (starting from guess=1)."""
    x = Var("x", guess=1.0, system=None)
    variables = [x]

    def eq():
        return x() ** 2 - 4.0

    equations = [("eq1", eq)]
    deps = {"eq1": {"x"}}
    diag = DiagnosticsData()

    def eval_fn():
        residuals = np.array([eq()])
        return residuals, deps

    status = newton_raphson(eval_fn, equations, variables, diag)
    assert status == SolveStatus.CONVERGED
    np.testing.assert_allclose(variables[0].value, 2.0, atol=1e-7)


def test_solver_not_converged():
    """System that doesn't converge in 2 iterations."""
    x = Var("x", guess=100.0, system=None)
    variables = [x]

    def eq():
        return x() ** 3 - 1.0

    equations = [("eq1", eq)]
    deps = {"eq1": {"x"}}
    diag = DiagnosticsData()

    def eval_fn():
        return np.array([eq()]), deps

    status = newton_raphson(eval_fn, equations, variables, diag, max_iter=2)
    assert status == SolveStatus.NOT_CONVERGED


def test_solver_error_on_exception():
    """Solver returns ERROR if equation raises."""
    x = Var("x", guess=0.0, system=None)
    variables = [x]

    def eq():
        return 1.0 / x()  # ZeroDivisionError on first call

    equations = [("eq1", eq)]
    deps = {"eq1": {"x"}}
    diag = DiagnosticsData()

    def eval_fn():
        return np.array([eq()]), deps

    status = newton_raphson(eval_fn, equations, variables, diag)
    assert status == SolveStatus.ERROR


def test_solver_clamp_max_step():
    """max_step limits the Newton step."""
    x = Var("x", guess=0.0, system=None, max_step=0.5)
    variables = [x]

    def eq():
        return x() - 100.0

    equations = [("eq1", eq)]
    deps = {"eq1": {"x"}}
    diag = DiagnosticsData()

    def eval_fn():
        return np.array([eq()]), deps

    status = newton_raphson(eval_fn, equations, variables, diag, max_iter=5)
    # After 5 iterations with max_step=0.5, x should be at most 2.5
    assert variables[0].value <= 2.5 + 1e-10


def test_solver_clamp_min_max_value():
    """min_value/max_value bound the variable."""
    x = Var("x", guess=5.0, system=None, min_value=0.0, max_value=10.0)
    variables = [x]

    def eq():
        return x() - 100.0  # wants to push x to 100

    equations = [("eq1", eq)]
    deps = {"eq1": {"x"}}
    diag = DiagnosticsData()

    def eval_fn():
        return np.array([eq()]), deps

    status = newton_raphson(eval_fn, equations, variables, diag)
    assert variables[0].value <= 10.0


def test_solver_diagnostics_populated():
    variables, equations, deps = _make_linear_system()
    diag = DiagnosticsData()

    def eval_fn():
        residuals = np.array([eq() for _, eq in equations])
        return residuals, deps

    newton_raphson(eval_fn, equations, variables, diag)
    assert diag.iterations > 0
    assert diag.converged is True
    assert len(diag.residual_history) > 0


def test_solver_on_iteration_callback():
    variables, equations, deps = _make_linear_system()
    diag = DiagnosticsData()
    calls = []

    def monitor(info, control):
        calls.append(info.iteration)

    def eval_fn():
        residuals = np.array([eq() for _, eq in equations])
        return residuals, deps

    newton_raphson(eval_fn, equations, variables, diag, on_iteration=monitor)
    assert len(calls) > 0


def test_solver_on_iteration_stop():
    """on_iteration can stop the solver via control.stop."""
    x = Var("x", guess=0.0, system=None, max_step=1.0)
    variables = [x]

    def eq():
        return x() - 100.0

    equations = [("eq1", eq)]
    deps = {"eq1": {"x"}}
    diag = DiagnosticsData()

    def monitor(info, control):
        if info.iteration >= 3:
            control.stop = True

    def eval_fn():
        return np.array([eq()]), deps

    status = newton_raphson(eval_fn, equations, variables, diag, on_iteration=monitor)
    assert status == SolveStatus.NOT_CONVERGED
    assert diag.iterations <= 4
