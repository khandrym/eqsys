from __future__ import annotations

import copy
from typing import Callable

from eqsys.diagnostics import DiagnosticsData, IterationInfo, SolverControl
from eqsys.solver import newton_raphson
from eqsys.status import SolveStatus, ValidationStatus
from eqsys.tracker import DependencyTracker
from eqsys.validator import ValidationIssue, validate_system
from eqsys.var import Var

import numpy as np


class EquationSystem:
    def __init__(self, name: str):
        self.name = name
        self._variables: list[Var] = []
        self._equations: list[tuple[str, Callable]] = []
        self._tracker = DependencyTracker()
        self._diagnostics = DiagnosticsData()
        self._validation_issues: list[ValidationIssue] = []

    def __repr__(self) -> str:
        return (
            f'EquationSystem("{self.name}", '
            f"vars={len(self._variables)}, eqs={len(self._equations)})"
        )

    def var(
        self,
        name: str,
        guess: float,
        min_value: float = -1e15,
        max_value: float = 1e15,
        max_step: float = 1e10,
        deriv_step: float = 1e-8,
    ) -> Var:
        v = Var(
            name=name,
            guess=guess,
            system=self,
            min_value=min_value,
            max_value=max_value,
            max_step=max_step,
            deriv_step=deriv_step,
        )
        self._variables.append(v)
        return v

    def add_equation(self, name: str, func: Callable) -> None:
        self._equations.append((name, func))

    def _run_equation(self, eq_func: Callable) -> tuple[object, set[str]]:
        self._tracker.start()
        try:
            result = eq_func()
        except Exception:
            self._tracker.stop()
            raise
        deps = self._tracker.stop()
        return result, deps

    def validate(self) -> ValidationStatus:
        status, self._validation_issues = validate_system(
            self._variables, self._equations, self._run_equation
        )
        return status

    def validation_details(self) -> list[ValidationIssue]:
        return list(self._validation_issues)

    def solve(
        self,
        tol: float = 1e-8,
        max_iter: int = 100,
        on_iteration: Callable[[IterationInfo, SolverControl], None] | None = None,
    ) -> SolveStatus:
        def eval_fn():
            residuals = []
            deps = {}
            for eq_name, eq_func in self._equations:
                self._tracker.start()
                try:
                    r = eq_func()
                finally:
                    eq_deps = self._tracker.stop()
                residuals.append(r)
                deps[eq_name] = eq_deps
            return np.array(residuals, dtype=float), deps

        return newton_raphson(
            eval_fn=eval_fn,
            equations=self._equations,
            variables=self._variables,
            diagnostics=self._diagnostics,
            tol=tol,
            max_iter=max_iter,
            on_iteration=on_iteration,
        )

    def diagnostics(self) -> DiagnosticsData:
        return copy.deepcopy(self._diagnostics)
