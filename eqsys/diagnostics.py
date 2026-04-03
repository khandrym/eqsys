from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class IterationInfo:
    iteration: int
    norm: float
    values: dict[str, float]
    residuals: dict[str, float]


@dataclass
class SolverControl:
    stop: bool = False


class DiagnosticsData:
    def __init__(self):
        self.reset()

    def reset(self):
        self.converged = False
        self.residual_history: list[float] = []
        self._residuals_per_eq: list[dict[str, float]] = []
        self._values_per_iter: list[dict[str, float]] = []

    @property
    def iterations(self) -> int:
        return len(self.residual_history)

    @property
    def final_norm(self) -> float:
        if not self.residual_history:
            return 0.0
        return self.residual_history[-1]

    def record_iteration(
        self,
        norm: float,
        residuals: dict[str, float],
        values: dict[str, float],
    ):
        self.residual_history.append(norm)
        self._residuals_per_eq.append(dict(residuals))
        self._values_per_iter.append(dict(values))

    def worst_equations(self) -> list[tuple[str, float]]:
        if not self._residuals_per_eq:
            return []
        last = self._residuals_per_eq[-1]
        return sorted(last.items(), key=lambda x: abs(x[1]), reverse=True)

    def variable_history(self, var_name: str) -> list[float]:
        if self._values_per_iter and var_name not in self._values_per_iter[0]:
            raise KeyError(f'Variable "{var_name}" not found in diagnostics history')
        return [v[var_name] for v in self._values_per_iter]
