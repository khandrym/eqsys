import math
import numpy as np
from eqsys import EquationSystem, SolveStatus, ValidationStatus


def test_system_creation():
    sys = EquationSystem("test")
    assert sys.name == "test"


def test_system_repr():
    sys = EquationSystem("test")
    x = sys.var("x", guess=1.0)
    sys.add_equation("eq1", lambda: x() - 1.0)
    assert 'EquationSystem("test"' in repr(sys)
    assert "vars=1" in repr(sys)
    assert "eqs=1" in repr(sys)


def test_var_repr():
    sys = EquationSystem("test")
    x = sys.var("x", guess=42.0)
    assert 'Var("x"' in repr(x)
    assert "42.0" in repr(x)


def test_diagnostics_is_snapshot():
    """diagnostics() returns a copy, not the mutable internal object."""
    sys = EquationSystem("test")
    x = sys.var("x", guess=0.0)
    sys.add_equation("eq1", lambda: x() - 5.0)
    sys.solve()

    diag1 = sys.diagnostics()
    assert diag1.converged is True
    iterations_before = diag1.iterations

    # Solve again — should not affect the snapshot
    x.value = 0.0
    sys.solve()
    assert diag1.iterations == iterations_before


def test_system_var_and_equation():
    sys = EquationSystem("test")
    x = sys.var("x", guess=1.0)
    assert x() == 1.0

    def eq():
        return x() - 5.0

    sys.add_equation("eq1", eq)


def test_system_validate_valid():
    sys = EquationSystem("test")
    x = sys.var("x", guess=1.0)

    def eq():
        return x() - 5.0

    sys.add_equation("eq1", eq)
    status = sys.validate()
    assert status == ValidationStatus.VALID


def test_system_validate_details():
    sys = EquationSystem("test")
    x = sys.var("x", guess=1.0)
    y = sys.var("y", guess=2.0)

    def eq():
        return x() - 5.0

    sys.add_equation("eq1", eq)
    status = sys.validate()
    assert status == ValidationStatus.INVALID
    details = sys.validation_details()
    assert len(details) > 0


def test_system_solve_linear():
    sys = EquationSystem("linear")
    x = sys.var("x", guess=0.0)
    y = sys.var("y", guess=0.0)

    def eq1():
        return x() - 3.0

    def eq2():
        return y() - 7.0

    sys.add_equation("eq1", eq1)
    sys.add_equation("eq2", eq2)

    status = sys.solve()
    assert status == SolveStatus.CONVERGED
    np.testing.assert_allclose(x(), 3.0, atol=1e-7)
    np.testing.assert_allclose(y(), 7.0, atol=1e-7)


def test_system_solve_nonlinear():
    sys = EquationSystem("nonlinear")
    x = sys.var("x", guess=1.0)

    def eq():
        return x() ** 2 - 4.0

    sys.add_equation("eq1", eq)

    status = sys.solve()
    assert status == SolveStatus.CONVERGED
    np.testing.assert_allclose(x(), 2.0, atol=1e-7)


def test_system_diagnostics():
    sys = EquationSystem("test")
    x = sys.var("x", guess=0.0)

    def eq():
        return x() - 5.0

    sys.add_equation("eq1", eq)
    sys.solve()

    diag = sys.diagnostics()
    assert diag.iterations > 0
    assert diag.converged is True


def test_system_on_iteration():
    sys = EquationSystem("test")
    x = sys.var("x", guess=0.0)

    def eq():
        return x() - 5.0

    sys.add_equation("eq1", eq)

    calls = []

    def monitor(info, control):
        calls.append(info.iteration)

    sys.solve(on_iteration=monitor)
    assert len(calls) > 0


def test_system_pipe_flow():
    """Integration test: pipe flow from CONCEPT.md."""
    sys = EquationSystem("pipe_flow")

    P1   = sys.var("P1",   guess=101325.0)
    P2   = sys.var("P2",   guess=100000.0)
    flow = sys.var("flow", guess=1.0, min_value=1e-6)

    D = 0.1
    L = 5.0
    rho = 998.0
    mu = 1e-3

    def inlet_pressure():
        return P1() - 101325.0

    def pipe_pressure_drop():
        area = math.pi * D ** 2 / 4
        v = flow() / (rho * area)
        Re = rho * v * D / mu
        if Re < 2300:
            f = 64 / Re
        else:
            f = 0.02
        dp = f * L / D * rho * v ** 2 / 2
        return P2() - (P1() - dp)

    def outlet_pressure():
        return P2() - 100000.0

    sys.add_equation("inlet_pressure", inlet_pressure)
    sys.add_equation("pipe_pressure_drop", pipe_pressure_drop)
    sys.add_equation("outlet_pressure", outlet_pressure)

    status = sys.solve()
    assert status == SolveStatus.CONVERGED
    np.testing.assert_allclose(P1(), 101325.0, atol=1e-4)
    np.testing.assert_allclose(P2(), 100000.0, atol=1e-4)
    assert flow() > 0


def test_system_multiple_systems():
    """Variables from one system used as constants in another."""
    sys_a = EquationSystem("system_a")
    T = sys_a.var("T", guess=300.0)

    def eq_a():
        return T() - 350.0

    sys_a.add_equation("temperature", eq_a)
    sys_a.solve()

    sys_b = EquationSystem("system_b")
    Q = sys_b.var("Q", guess=0.0)

    def eq_b():
        return Q() - (T() - 273.15)  # T() from sys_a is a constant here

    sys_b.add_equation("heat", eq_b)
    status = sys_b.solve()

    assert status == SolveStatus.CONVERGED
    np.testing.assert_allclose(T(), 350.0, atol=1e-7)
    np.testing.assert_allclose(Q(), 350.0 - 273.15, atol=1e-7)


def test_system_branching():
    """System where branching changes dependencies between iterations."""
    sys = EquationSystem("branching")
    x = sys.var("x", guess=0.5)

    def eq():
        if x() < 1.0:
            return x() - 2.0  # pushes x towards 2.0
        else:
            return x() ** 2 - 4.0  # x = 2.0

    sys.add_equation("eq1", eq)
    status = sys.solve()
    assert status == SolveStatus.CONVERGED
    np.testing.assert_allclose(x(), 2.0, atol=1e-7)
