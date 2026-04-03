import pytest
from eqsys import EquationSystem, SolveStatus


def _build_linear_system(n: int) -> tuple[EquationSystem, list]:
    """Build a system of n equations: x_i = i+1 for i in 0..n-1."""
    sys = EquationSystem(f"bench_{n}")
    variables = []
    for i in range(n):
        v = sys.var(f"x{i}", guess=0.0)
        variables.append(v)

    for i in range(n):
        target = float(i + 1)
        var = variables[i]

        def make_eq(v=var, t=target):
            def eq():
                return v() - t
            return eq

        sys.add_equation(f"eq{i}", make_eq())

    return sys, variables


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_benchmark_solve(benchmark, n):
    """Benchmark solve time for N-equation linear system."""
    sys, variables = _build_linear_system(n)

    def run():
        for v in variables:
            v.value = 0.0
        return sys.solve()

    result = benchmark(run)
    assert result == SolveStatus.CONVERGED


@pytest.mark.benchmark
@pytest.mark.parametrize("n", [10, 100, 500, 1000])
def test_benchmark_tracker_overhead(benchmark, n):
    """Benchmark tracker overhead: eval with tracking for N variables."""
    sys = EquationSystem(f"tracker_bench_{n}")
    variables = []
    for i in range(n):
        v = sys.var(f"x{i}", guess=float(i))
        variables.append(v)

    def read_all_vars():
        sys._tracker.start()
        total = 0.0
        for v in variables:
            total += v()
        sys._tracker.stop()
        return total

    benchmark(read_all_vars)
