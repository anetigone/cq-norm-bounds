"""Coverage dual and the compatible network-flow LP (paper §VII)."""

from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
import math
from time import perf_counter

from ortools.graph.python import max_flow
from ortools.linear_solver import linear_solver_pb2, pywraplp

from .numeric import certificate_cost
from .shannon import Bound


def compatible_order(bits, expressions, goal):
    """Return a preserving order for negative input sets and target scopes."""
    residual = set(goal)
    residual.update(mask for row in expressions for mask, coeff in row.items() if coeff < 0)
    remaining = list(bits.values())
    order = []
    while nonsingletons := [mask for mask in residual if mask.bit_count() > 1]:
        common = nonsingletons[0]
        for mask in nonsingletons[1:]:
            common &= mask
        if not common:
            return None
        chosen = common & -common
        remaining.remove(chosen)
        order.append(chosen)
        residual = {mask & ~chosen for mask in residual}
    return order + remaining


def step_value(expression, support):
    return sum(coeff for mask, coeff in expression.items() if mask & support)


def split_terms(expressions, goal):
    positive = defaultdict(list)
    negative = defaultdict(list)
    for index, expression in enumerate(expressions):
        for mask, coeff in expression.items():
            (positive if coeff > 0 else negative)[mask].append((index, abs(coeff)))
    for mask, coeff in goal.items():
        (negative if coeff > 0 else positive)[mask].append((None, abs(coeff)))
    return positive, negative


def networks(order, expressions, goal):
    """Yield (arcs, demand) with affine capacities; None means a forcing arc."""
    positive, negative = split_terms(expressions, goal)
    nodes = {bit: index + 2 for index, bit in enumerate(order)}
    coverage = []
    for node, (mask, terms) in enumerate(positive.items(), start=len(order) + 2):
        coverage.append((0, node, terms))
        coverage.extend((node, nodes[bit], None) for bit in order if mask & bit)
    rewards = [(mask.bit_count() - 1, next(bit for bit in reversed(order) if mask & bit), terms)
               for mask, terms in negative.items()]
    demand = [term for terms in negative.values() for term in terms]
    for index, bit in enumerate(order):
        unary = defaultdict(list)
        for prefix_size, last, terms in rewards:
            if index >= prefix_size:
                unary[last].extend(terms)
        arcs = coverage + [(nodes[key], 1, terms) for key, terms in unary.items()]
        arcs.extend((0, nodes[key], None) for key in order[:index])
        arcs.append((nodes[bit], 1, None))
        yield arcs, demand


def add_flow(solver, arcs, demand, affine):
    balance = defaultdict(list)
    for start, end, capacity in arcs:
        flow = solver.NumVar(0, solver.infinity(), "")
        balance[start].append(flow)
        balance[end].append(-flow)
        if capacity is not None:
            solver.Add(flow <= affine(capacity))
    value = solver.NumVar(0, solver.infinity(), "")
    balance[0].append(-value)
    balance[1].append(value)
    for net in balance.values():
        solver.Add(solver.Sum(net) == 0)
    solver.Add(value >= affine(demand))


def verify_flow(networks_to_check, weights):
    """Check the cut condition using exact, integer-scaled max flows."""
    def evaluate(terms):
        return sum((coeff if index is None else coeff * weights[index] for index, coeff in terms), Fraction(0))

    for arcs, demand in networks_to_check:
        required = evaluate(demand)
        if not required:
            continue
        capacities = [evaluate(terms) if terms is not None else None for _, _, terms in arcs]
        denominator = math.lcm(required.denominator, *(c.denominator for c in capacities if c is not None))
        threshold = int(required * denominator)
        # Each arc can be capped at the required flow. This preserves whether
        # the demand can be met and also gives a finite forcing capacity.
        if threshold * len(arcs) >= 2**63:
            return False
        graph = max_flow.SimpleMaxFlow()
        for (start, end, _), capacity in zip(arcs, capacities):
            limit = threshold if capacity is None else min(threshold, int(capacity * denominator))
            graph.add_arc_with_capacity(start, end, limit)
        if graph.solve(0, 1) != graph.OPTIMAL or graph.optimal_flow() < threshold:
            return False
    return True


def _empty(rhs):
    if any(not math.isfinite(value) and value != -math.inf for value in rhs):
        raise ValueError("Budgets must be finite or certify an empty atom")
    return any(value < 0 for value in rhs)


def solve(attributes, budgets, target, *, method, time_limit_ms=None):
    """Build and solve a one-off certified coverage or flow bound."""
    if method not in ("coverage", "flow"):
        raise ValueError(f"Unknown method {method}")
    budgets = tuple(budgets)
    rhs = tuple(value for _, value in budgets)
    if _empty(rhs):
        return Bound("empty")
    program = FlowProgram(attributes, tuple(stat for stat, _ in budgets), target,
                          method=method, time_limit_ms=time_limit_ms)
    return replace(program.solve(rhs), build_ms=program.build_ms)


class FlowProgram:
    """Fixed ordered statistics and target. Only the objective depends on budgets."""

    def __init__(self, attributes, statistics, target, *, method="flow", time_limit_ms=None):
        if method not in ("coverage", "flow"):
            raise ValueError(f"Unknown method {method}")
        self.method = method
        self.time_limit_ms = time_limit_ms
        self.rhs = (0,) * len(statistics)
        self.solver = None
        self.build_ms = 0
        self.metrics = {}
        self._build(tuple(attributes), statistics, target)

    def _build(self, attributes, statistics, target):
        method = self.method
        time_limit_ms = self.time_limit_ms
        start = perf_counter()
        bits = {attr: 1 << index for index, attr in enumerate(attributes)}
        if len(bits) != len(attributes):
            raise ValueError("Query attributes must be unique")
        goal = target.expression(bits)
        expressions = [stat.expression(bits) for stat in statistics]
        order = compatible_order(bits, expressions, goal)
        self.metrics["compatibility_ms"] = 1000 * (perf_counter() - start)
        if order is None:
            return

        build_start = perf_counter()
        solver = self.solver = pywraplp.Solver.CreateSolver("GLOP")
        solver.SetNumThreads(1)
        if time_limit_ms is not None:
            solver.SetTimeLimit(time_limit_ms)
        self.weights = [solver.NumVar(0, solver.infinity(), "") for _ in statistics]

        def affine(terms):
            return solver.Sum(float(coeff) if index is None else float(coeff) * self.weights[index]
                              for index, coeff in terms)

        self.coverage_rows = []
        self.flow_networks = []
        if method == "coverage":
            for support in range(1, 1 << len(bits)):
                required = step_value(goal, support)
                terms = [(index, coeff) for index, row in enumerate(expressions)
                         if (coeff := step_value(row, support))]
                self.coverage_rows.append((terms, required))
                solver.Add(affine(terms) >= float(required))
        else:
            self.flow_networks = list(networks(order, expressions, goal))
            for arcs, demand in self.flow_networks:
                add_flow(solver, arcs, demand, affine)
        self.objective = solver.Objective()
        self.objective.SetMinimization()
        model = linear_solver_pb2.MPModelProto()
        solver.ExportModelToProto(model)
        self.model_bytes = model.ByteSize()
        self.metrics.update(n_variables=solver.NumVariables(), n_constraints=solver.NumConstraints(),
                            n_nonzeros=sum(len(row.var_index) for row in model.constraint),
                            order=tuple(attributes[bit.bit_length() - 1] for bit in order))
        self.build_ms = 1000 * (perf_counter() - build_start)

    def solve(self, rhs):
        rhs = tuple(rhs)
        if len(rhs) != len(self.rhs):
            raise ValueError("Budgets must match the compiled statistic order")
        if _empty(rhs):
            return Bound("empty")
        solver = self.solver
        if solver is None:
            return Bound("incompatible", **self.metrics)
        for weight, previous, budget in zip(self.weights, self.rhs, rhs):
            if budget != previous:
                self.objective.SetCoefficient(weight, budget)
        self.rhs = rhs
        solve_start = perf_counter()
        # GLOP's default incrementality parameter is ON. Keeping this solver
        # alive lets its backend retain state across objective changes.
        status = solver.Solve()
        verify_start = perf_counter()
        metrics = {**self.metrics, "solve_ms": 1000 * (verify_start - solve_start)}
        if status == solver.INFEASIBLE:
            return Bound("unbounded", **metrics)
        if self.time_limit_ms is not None and status in (solver.NOT_SOLVED, solver.FEASIBLE):
            return Bound("timeout", **metrics)
        if status != solver.OPTIMAL:
            return Bound("solver_error", **metrics)

        exact = [Fraction(weight.solution_value()).limit_denominator(10**6) for weight in self.weights]
        valid = all(weight >= 0 for weight in exact)
        if valid and self.method == "coverage":
            valid = all(sum(exact[index] * coeff for index, coeff in terms) >= demand
                        for terms, demand in self.coverage_rows)
        if valid and self.method == "flow":
            valid = verify_flow(self.flow_networks, exact)
        if not valid:
            return Bound("unverified", verify_ms=1000 * (perf_counter() - verify_start), **metrics)
        upper = certificate_cost(exact, rhs)
        return Bound("optimal", upper, input_weights=tuple(map(str, exact)),
                     verify_ms=1000 * (perf_counter() - verify_start), **metrics)
