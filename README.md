# eqsys

Nonlinear equation system solver with automatic dependency tracking.

Write only residual functions in plain Python — the system handles variable tracking, sparse Jacobian computation, and Newton-Raphson iteration automatically.

## Installation

```bash
pip install eqsys
```

## Quick Example

```python
import math
from eqsys import EquationSystem

sys = EquationSystem("pipe_flow")

# Variables — name and initial guess are required
P1   = sys.var("P1",   guess=101325.0)
P2   = sys.var("P2",   guess=100000.0)
flow = sys.var("flow", guess=1.0, min_value=0.0)

# Constants
D, L, rho, mu = 0.1, 5.0, 998.0, 1e-3

# Equations — plain Python functions returning residuals
def inlet_pressure():
    return P1() - 101325.0

def pipe_pressure_drop():
    area = math.pi * D ** 2 / 4
    v = flow() / (rho * area)
    Re = rho * v * D / mu
    f = 64 / Re if Re < 2300 else 0.02
    dp = f * L / D * rho * v ** 2 / 2
    return P2() - (P1() - dp)

def outlet_pressure():
    return P2() - 100000.0

sys.add_equation("inlet_pressure", inlet_pressure)
sys.add_equation("pipe_pressure_drop", pipe_pressure_drop)
sys.add_equation("outlet_pressure", outlet_pressure)

# Validate and solve
sys.validate()
status = sys.solve()

print(f"P1 = {P1()}")    # 101325.0
print(f"P2 = {P2()}")    # 100000.0
print(f"flow = {flow()}")  # ~12.77
```

## Key Features

- **Minimal API** — write residual functions, the system does the rest
- **Automatic dependency tracking** — no need to declare which variables each equation uses
- **Sparse Jacobian** — only computes non-zero partial derivatives
- **Per-variable solver parameters** — bounds, max step, derivative step
- **Strict validation** — catches dimension mismatches, unused variables, equation errors
- **Convergence diagnostics** — residual history, worst equations, variable trajectories
- **Real-time monitoring** — `on_iteration` callback with stop control
- **Multiple systems** — variables from one system can be used as constants in another

## API

### Variables

```python
x = sys.var("x",
    guess=1.0,           # required — initial value
    min_value=-1e15,     # lower bound (default)
    max_value=1e15,      # upper bound (default)
    max_step=1e10,       # max change per iteration (default)
    deriv_step=1e-8,     # finite difference step (default)
)

x()  # returns current float value (guess before solve, solution after)
```

### Equations

```python
sys.add_equation("name", func)  # func() -> float (residual)
```

Any Python code is allowed inside equations: `if/else`, `math.*`, loops, function calls.

### Validation

```python
status = sys.validate()  # ValidationStatus.VALID / INVALID / WARNING
details = sys.validation_details()  # list of ValidationIssue
```

Checks: dimension match, duplicate names, equation errors, unused variables, guess bounds.

### Solving

```python
status = sys.solve(tol=1e-8, max_iter=100)  # SolveStatus.CONVERGED / NOT_CONVERGED / ERROR
```

No exceptions — check the returned status.

### Diagnostics

```python
diag = sys.diagnostics()
diag.iterations           # number of iterations
diag.converged            # bool
diag.residual_history     # [norm_0, norm_1, ...]
diag.worst_equations()    # [("eq_name", residual), ...] sorted by |residual|
diag.variable_history("x")  # [x_0, x_1, ...]
```

### Real-Time Monitoring

```python
def monitor(info, control):
    print(f"Iter {info.iteration}: norm={info.norm:.2e}")
    if info.iteration > 50:
        control.stop = True  # abort solve

sys.solve(on_iteration=monitor)
```

## Requirements

- Python 3.11+
- numpy
- scipy

## License

MIT
