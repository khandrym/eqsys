from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.sparse.linalg import spsolve

from eqsys.diagnostics import DiagnosticsData, IterationInfo, SolverControl
from eqsys.jacobian import compute_jacobian
from eqsys.status import SolveStatus
from eqsys.var import Var


def newton_raphson(
    eval_fn: Callable,
    equations: list[tuple[str, Callable]],
    variables: list[Var],
    diagnostics: DiagnosticsData,
    tol: float = 1e-8,
    max_iter: int = 100,
    on_iteration: Callable[[IterationInfo, SolverControl], None] | None = None,
) -> SolveStatus:
    diagnostics.reset()

    for iteration in range(max_iter):
        try:
            F, deps = eval_fn()
        except Exception:
            return SolveStatus.ERROR

        norm = float(np.linalg.norm(F))
        converged = norm < tol

        residuals_dict = {
            eq_name: float(F[i]) for i, (eq_name, _) in enumerate(equations)
        }
        values_dict = {v.name: v.value for v in variables}
        diagnostics.record_iteration(
            norm=norm, residuals=residuals_dict, values=values_dict
        )

        if on_iteration is not None:
            info = IterationInfo(
                iteration=iteration,
                norm=norm,
                values=dict(values_dict),
                residuals=dict(residuals_dict),
            )
            control = SolverControl()
            on_iteration(info, control)
            if control.stop:
                return SolveStatus.NOT_CONVERGED

        if converged:
            diagnostics.converged = True
            return SolveStatus.CONVERGED

        try:
            J = compute_jacobian(equations, variables, deps, F)
            dx = spsolve(J, -F)
        except Exception:
            return SolveStatus.ERROR

        for k, var in enumerate(variables):
            step = float(dx[k])
            step = max(-var.max_step, min(var.max_step, step))
            new_val = var.value + step
            new_val = max(var.min_value, min(var.max_value, new_val))
            var.value = new_val

    return SolveStatus.NOT_CONVERGED
