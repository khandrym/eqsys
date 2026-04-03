import pytest


def test_var_creation_and_call():
    """Var returns guess value when called."""
    from eqsys.var import Var
    v = Var("pressure", guess=101325.0, system=None)
    assert v() == 101325.0
    assert v.name == "pressure"


def test_var_default_parameters():
    """Optional parameters have sensible defaults."""
    from eqsys.var import Var
    v = Var("x", guess=1.0, system=None)
    assert v.min_value == -1e15
    assert v.max_value == 1e15
    assert v.max_step == 1e10
    assert v.deriv_step == 1e-8


def test_var_custom_parameters():
    """All parameters can be customized."""
    from eqsys.var import Var
    v = Var("x", guess=5.0, system=None,
            min_value=0.0, max_value=100.0,
            max_step=1.0, deriv_step=1e-6)
    assert v.min_value == 0.0
    assert v.max_value == 100.0
    assert v.max_step == 1.0
    assert v.deriv_step == 1e-6


def test_var_value_update():
    """Var value can be updated and read back."""
    from eqsys.var import Var
    v = Var("x", guess=1.0, system=None)
    assert v() == 1.0
    v.value = 5.0
    assert v() == 5.0


def test_var_returns_float():
    """Var always returns a float."""
    from eqsys.var import Var
    v = Var("x", guess=1, system=None)
    result = v()
    assert isinstance(result, float)
