"""Small-instance Shannon LP for degree and mixed-norm statistics."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math
from time import perf_counter

from ortools.linear_solver import pywraplp

from .numeric import round_up


@dataclass(frozen=True)
class MixedNorm:
    blocks: tuple[tuple[str, ...], ...]
    exponents: tuple[float, ...]

    def __post_init__(self):
        attributes = [attr for block in self.blocks for attr in block]
        if len(self.blocks) != len(self.exponents) or any(not block for block in self.blocks):
            raise ValueError("Each nonempty block needs one exponent")
        if len(set(attributes)) != len(attributes) or any(not p > 0 for p in self.exponents):
            raise ValueError("Blocks must be disjoint and exponents must be positive")

    @classmethod
    def degree(cls, projection, key, q):
        if not q > 0:
            raise ValueError("Degree exponent must be positive")
        rest = tuple(attr for attr in projection if attr not in key)
        if not set(key) <= set(projection):
            raise ValueError("Key must be contained in the projection")
        pairs = [(tuple(key), q), (rest, 1)]
        return cls(tuple(block for block, _ in pairs if block),
                   tuple(p for block, p in pairs if block))

    def expression(self, bits):
        """Map nonempty prefix masks to exact rational coefficients."""
        expression = {}
        prefix = 0
        for block, exponent in zip(self.blocks, self.exponents):
            coefficient = Fraction(0) if math.isinf(exponent) else 1 / Fraction(str(exponent))
            if prefix:
                expression[prefix] -= coefficient
            for attr in block:
                prefix |= bits[attr]
            expression[prefix] = coefficient
        return {mask: value for mask, value in expression.items() if value}


@dataclass(frozen=True)
class Bound:
    status: str
    log2_bound: float | None = None
    log2_primal: float | None = None
    input_weights: tuple[str, ...] = ()
    n_variables: int = 0
    n_constraints: int = 0
    n_nonzeros: int = 0
    build_ms: float = 0
    solve_ms: float = 0
    verify_ms: float = 0
    compatibility_ms: float = 0
    order: tuple[str, ...] = ()


def solve(attributes, budgets, target, time_limit_ms=None):
    """Solve over query attributes with atom-guarded (MixedNorm, log2 bound) rows.

    A finite bound is returned only after verifying the recovered dual using
    rational arithmetic. This prototype can report unverified on valid LPs.
    """
    start = perf_counter()
    bits = {attr: 1 << index for index, attr in enumerate(attributes)}
    if len(bits) != len(attributes):
        raise ValueError("Query attributes must be unique")
    goal = target.expression(bits)
    inputs = [(stat.expression(bits), budget) for stat, budget in budgets]
    if any(math.isnan(budget) or budget == math.inf for _, budget in inputs):
        raise ValueError("Budgets must be finite or certify an empty atom")
    if any(budget < 0 for _, budget in inputs):
        return Bound("empty")

    solver = pywraplp.Solver.CreateSolver("GLOP")
    solver.SetNumThreads(1)
    if time_limit_ms is not None:
        solver.SetTimeLimit(time_limit_ms)
    universe = (1 << len(attributes)) - 1
    variables = {mask: solver.NumVar(-solver.infinity(), solver.infinity(), f"h{mask}")
                 for mask in range(1, universe + 1)}
    rows = []

    def add_row(expression, budget=0):
        row = solver.RowConstraint(-solver.infinity(), budget)
        for mask, coefficient in expression.items():
            row.SetCoefficient(variables[mask], float(coefficient))
        rows.append((row, expression, budget))

    for expression, budget in inputs:
        add_row(expression, budget)

    # Elemental inequalities: H(i | W-i) >= 0 and I(i;j | K) >= 0.
    for i in bits.values():
        expression = {universe: -1}
        if universe ^ i:
            expression[universe ^ i] = 1
        add_row(expression)
    for i, j in combinations(bits.values(), 2):
        remaining = universe ^ i ^ j
        subset = remaining
        while True:
            expression = {subset | i | j: 1, subset | i: -1, subset | j: -1}
            if subset:
                expression[subset] = 1
            add_row(expression)
            if not subset:
                break
            subset = (subset - 1) & remaining

    objective = solver.Objective()
    for mask, coefficient in goal.items():
        objective.SetCoefficient(variables[mask], float(coefficient))
    objective.SetMaximization()
    solve_start = perf_counter()
    status = solver.Solve()
    verify_start = perf_counter()
    metrics = dict(n_variables=len(variables), n_constraints=len(rows),
                   n_nonzeros=sum(len(expression) for _, expression, _ in rows),
                   build_ms=1000 * (solve_start - start),
                   solve_ms=1000 * (verify_start - solve_start))
    if status != solver.OPTIMAL:
        if time_limit_ms is not None and status in (solver.NOT_SOLVED, solver.FEASIBLE):
            return Bound("timeout", **metrics)
        # GLOP can report INFEASIBLE for a primal unbounded model in presolve.
        # h=0 is feasible for every nonnegative budget in this formulation.
        if status in (solver.UNBOUNDED, solver.INFEASIBLE):
            return Bound("unbounded", **metrics)
        return Bound("solver_error", **metrics)

    weights = [Fraction(row.dual_value()).limit_denominator(10**6) for row, _, _ in rows]
    residual = goal.copy()
    cost = Fraction(0)
    for weight, (_, expression, budget) in zip(weights, rows):
        if weight:
            cost += weight * Fraction(budget)
            for mask, coefficient in expression.items():
                residual[mask] = residual.get(mask, 0) - weight * coefficient
    metrics.update(log2_primal=objective.Value(), verify_ms=1000 * (perf_counter() - verify_start))
    if any(weight < 0 for weight in weights) or any(residual.values()):
        return Bound("unverified", **metrics)
    upper = round_up(cost)
    return Bound("optimal", upper, input_weights=tuple(map(str, weights[:len(inputs)])), **metrics)
