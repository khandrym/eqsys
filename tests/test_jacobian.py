import numpy as np
from scipy import sparse
from eqsys.jacobian import compute_jacobian
from eqsys.var import Var


def test_jacobian_simple_linear():
    """Jacobian of f(x,y) = 2x + 3y is [[2, 3]]."""
    x = Var("x", guess=1.0, system=None)
    y = Var("y", guess=2.0, system=None)
    vars_list = [x, y]

    def eq():
        return 2.0 * x() + 3.0 * y()

    equations = [("eq1", eq)]
    deps = {"eq1": {"x", "y"}}
    residuals = np.array([2.0 * 1.0 + 3.0 * 2.0])

    J = compute_jacobian(equations, vars_list, deps, residuals)

    assert sparse.issparse(J)
    J_dense = J.toarray()
    assert J_dense.shape == (1, 2)
    np.testing.assert_allclose(J_dense[0, 0], 2.0, atol=1e-4)
    np.testing.assert_allclose(J_dense[0, 1], 3.0, atol=1e-4)


def test_jacobian_sparsity():
    """Jacobian skips pairs not in dependency map."""
    x = Var("x", guess=1.0, system=None)
    y = Var("y", guess=2.0, system=None)
    vars_list = [x, y]

    def eq1():
        return x() * 2.0

    def eq2():
        return y() * 3.0

    equations = [("eq1", eq1), ("eq2", eq2)]
    deps = {"eq1": {"x"}, "eq2": {"y"}}
    residuals = np.array([2.0, 6.0])

    J = compute_jacobian(equations, vars_list, deps, residuals)

    J_dense = J.toarray()
    assert J_dense.shape == (2, 2)
    np.testing.assert_allclose(J_dense[0, 0], 2.0, atol=1e-4)
    assert J_dense[0, 1] == 0.0
    assert J_dense[1, 0] == 0.0
    np.testing.assert_allclose(J_dense[1, 1], 3.0, atol=1e-4)


def test_jacobian_nonlinear():
    """Jacobian of f(x) = x^2 at x=3 is approximately 6."""
    x = Var("x", guess=3.0, system=None)
    vars_list = [x]

    def eq():
        return x() ** 2

    equations = [("eq1", eq)]
    deps = {"eq1": {"x"}}
    residuals = np.array([9.0])

    J = compute_jacobian(equations, vars_list, deps, residuals)

    J_dense = J.toarray()
    np.testing.assert_allclose(J_dense[0, 0], 6.0, atol=1e-4)


def test_jacobian_custom_deriv_step():
    """Uses per-variable deriv_step."""
    x = Var("x", guess=3.0, system=None, deriv_step=1e-6)
    vars_list = [x]

    def eq():
        return x() ** 2

    equations = [("eq1", eq)]
    deps = {"eq1": {"x"}}
    residuals = np.array([9.0])

    J = compute_jacobian(equations, vars_list, deps, residuals)

    J_dense = J.toarray()
    np.testing.assert_allclose(J_dense[0, 0], 6.0, atol=1e-3)
