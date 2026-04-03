from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Callable

from eqsys.status import ValidationStatus
from eqsys.var import Var


@dataclass
class ValidationIssue:
    level: str  # "error" or "warning"
    message: str


def validate_system(
    variables: list[Var],
    equations: list[tuple[str, Callable]],
    run_equation: Callable[[Callable], tuple[object, set[str]]],
) -> tuple[ValidationStatus, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []

    # Empty system is valid
    if not variables and not equations:
        return ValidationStatus.VALID, issues

    # Duplicate variable names
    var_names = [v.name for v in variables]
    var_counts = Counter(var_names)
    for name, count in var_counts.items():
        if count > 1:
            issues.append(ValidationIssue(
                "error", f'Duplicate variable name "{name}" ({count} times)'))

    # Duplicate equation names
    eq_names = [name for name, _ in equations]
    eq_counts = Counter(eq_names)
    for name, count in eq_counts.items():
        if count > 1:
            issues.append(ValidationIssue(
                "error", f'Duplicate equation name "{name}" ({count} times)'))

    # Dimension check
    if len(equations) != len(variables):
        issues.append(ValidationIssue(
            "error",
            f"System has {len(equations)} equations but {len(variables)} variables"))

    # Guess bounds check
    for var in variables:
        if var.value < var.min_value or var.value > var.max_value:
            issues.append(ValidationIssue(
                "error",
                f'Variable "{var.name}" guess {var.value} is outside '
                f'bounds [{var.min_value}, {var.max_value}]'))

    # Trial run — call each equation via run_equation, capture deps
    all_used_vars: set[str] = set()
    for eq_name, eq_func in equations:
        try:
            result, deps = run_equation(eq_func)
        except Exception as e:
            issues.append(ValidationIssue(
                "error",
                f'Equation "{eq_name}" raised {type(e).__name__}: {e}'))
            continue

        if not isinstance(result, (int, float)):
            issues.append(ValidationIssue(
                "error",
                f'Equation "{eq_name}" returned {type(result).__name__}, expected numeric'))

        if not deps:
            issues.append(ValidationIssue(
                "warning",
                f'Equation "{eq_name}" does not read any variables'))

        all_used_vars.update(deps)

    # Variables not used by any equation
    for var in variables:
        if var.name not in all_used_vars:
            issues.append(ValidationIssue(
                "error",
                f'Variable "{var.name}" is not used by any equation'))

    # Determine status
    has_errors = any(i.level == "error" for i in issues)
    has_warnings = any(i.level == "warning" for i in issues)

    if has_errors:
        return ValidationStatus.INVALID, issues
    if has_warnings:
        return ValidationStatus.WARNING, issues
    return ValidationStatus.VALID, issues
