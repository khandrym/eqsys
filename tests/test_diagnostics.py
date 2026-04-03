import pytest
from eqsys.diagnostics import DiagnosticsData, IterationInfo, SolverControl


def test_diagnostics_empty():
    d = DiagnosticsData()
    assert d.iterations == 0
    assert d.final_norm == 0.0
    assert d.converged is False
    assert d.residual_history == []


def test_diagnostics_record_iteration():
    d = DiagnosticsData()
    d.record_iteration(
        norm=100.0,
        residuals={"eq1": 80.0, "eq2": 60.0},
        values={"x": 1.0, "y": 2.0},
    )
    d.record_iteration(
        norm=1.0,
        residuals={"eq1": 0.8, "eq2": 0.6},
        values={"x": 1.5, "y": 2.5},
    )
    assert d.iterations == 2
    assert d.residual_history == [100.0, 1.0]
    assert d.final_norm == 1.0


def test_diagnostics_worst_equations():
    d = DiagnosticsData()
    d.record_iteration(
        norm=100.0,
        residuals={"eq1": 80.0, "eq2": 60.0, "eq3": 90.0},
        values={"x": 1.0},
    )
    worst = d.worst_equations()
    assert worst[0] == ("eq3", 90.0)
    assert worst[1] == ("eq1", 80.0)
    assert worst[2] == ("eq2", 60.0)


def test_diagnostics_variable_history():
    d = DiagnosticsData()
    d.record_iteration(norm=10.0, residuals={}, values={"x": 1.0, "y": 2.0})
    d.record_iteration(norm=1.0, residuals={}, values={"x": 1.5, "y": 2.5})
    assert d.variable_history("x") == [1.0, 1.5]
    assert d.variable_history("y") == [2.0, 2.5]


def test_diagnostics_reset():
    d = DiagnosticsData()
    d.record_iteration(norm=10.0, residuals={}, values={"x": 1.0})
    d.reset()
    assert d.iterations == 0
    assert d.residual_history == []


def test_iteration_info():
    info = IterationInfo(iteration=5, norm=1.2e-3, values={"x": 1.0}, residuals={"eq1": 0.5})
    assert info.iteration == 5
    assert info.norm == 1.2e-3
    assert info.values == {"x": 1.0}
    assert info.residuals == {"eq1": 0.5}


def test_solver_control_default():
    ctrl = SolverControl()
    assert ctrl.stop is False


def test_solver_control_stop():
    ctrl = SolverControl()
    ctrl.stop = True
    assert ctrl.stop is True


def test_diagnostics_variable_history_unknown_var():
    d = DiagnosticsData()
    d.record_iteration(norm=10.0, residuals={}, values={"x": 1.0})
    with pytest.raises(KeyError, match="nonexistent"):
        d.variable_history("nonexistent")
