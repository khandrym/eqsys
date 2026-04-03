from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from eqsys.var import Var


def compute_jacobian(
    equations: list[tuple[str, callable]],
    variables: list[Var],
    deps: dict[str, set[str]],
    residuals: np.ndarray,
) -> coo_matrix:
    var_index = {v.name: j for j, v in enumerate(variables)}
    rows = []
    cols = []
    data = []

    for i, (eq_name, eq_func) in enumerate(equations):
        eq_deps = deps.get(eq_name, set())
        for var_name in eq_deps:
            j = var_index[var_name]
            var = variables[j]

            original = var.value
            var.value = original + var.deriv_step
            f_perturbed = eq_func()
            var.value = original

            dfdx = (f_perturbed - residuals[i]) / var.deriv_step
            rows.append(i)
            cols.append(j)
            data.append(dfdx)

    n_eq = len(equations)
    n_var = len(variables)
    return coo_matrix((data, (rows, cols)), shape=(n_eq, n_var)).tocsc()
