from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eqsys.system import EquationSystem


class Var:
    def __init__(
        self,
        name: str,
        guess: float,
        system: EquationSystem | None,
        min_value: float = -1e15,
        max_value: float = 1e15,
        max_step: float = 1e10,
        deriv_step: float = 1e-8,
    ):
        self.name = name
        self.value = float(guess)
        self._system = system
        self.min_value = min_value
        self.max_value = max_value
        self.max_step = max_step
        self.deriv_step = deriv_step

    def __repr__(self) -> str:
        return f'Var("{self.name}", value={self.value})'

    def __call__(self) -> float:
        if self._system is not None and self._system._tracker.active:
            self._system._tracker.register(self.name)
        return self.value
