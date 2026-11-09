"""First synthetic E1/E2 run: two projected paths and one downstream join."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import duckdb
import ortools

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from cqnorm.shannon import MixedNorm, solve
from cqnorm import coverage_flow

ORDERS = (0.5, 1, 1.5, 2, 4, 8, math.inf)
FORMULATIONS = {"shannon": "Shannon-General", "coverage": "Coverage-Enumerate", "flow": "Flow-General"}
V_SQL = "SELECT DISTINCT x, z FROM r JOIN s USING (y)"
Q_SQL = f"SELECT DISTINCT x, z, u FROM ({V_SQL}) v JOIN t USING (x)"


def degree(projection, key="", q=1):
    return MixedNorm.degree(tuple(projection), tuple(key), q)


def log_norm(connection, sql, norm):
    """Evaluate a statistic on the distinct projection, innermost block first."""
    columns = [column for block in norm.blocks for column in block]
    rows = connection.execute(f"SELECT DISTINCT {', '.join(columns) or '1'} FROM ({sql})").fetchall()
    if not rows:
        return -math.inf
    values = {row: 0.0 for row in rows} if columns else {(): 0.0}
    for block, exponent in reversed(list(zip(norm.blocks, norm.exponents))):
        groups = defaultdict(list)
        for row, value in values.items():
            groups[row[:-len(block)]].append(value)
        values = {}
        for prefix, group in groups.items():
            maximum = max(group)
            if math.isinf(exponent):
                values[prefix] = maximum
            else:
                total = math.fsum(math.exp2(exponent * (value - maximum)) for value in group)
                values[prefix] = maximum + math.log2(total) / exponent
    return values[()]


def collect_budgets(connection, table, statistics):
    budgets = []
    for stat in statistics:
        value = log_norm(connection, f"SELECT * FROM {table}", stat)
        # Allow for floating-point norm extraction on these synthetic inputs.
        budget = math.nextafter(value + 1e-12, math.inf) if value > 0 else value
        budgets.append((stat, budget))
    return budgets


def norm_json(norm):
    return {"blocks": norm.blocks,
            "exponents": ["inf" if math.isinf(q) else q for q in norm.exponents]}


def measure(rows, case, method, attributes, budgets, target, solver):
    start = perf_counter()
    if solver == "shannon":
        result = solve(attributes, budgets, target)
    else:
        result = coverage_flow.solve(attributes, budgets, target, method=solver)
    estimate_ms = 1000 * (perf_counter() - start)
    budget_rows = [{**norm_json(stat), "log2_budget": log_budget} for stat, log_budget in budgets]
    rows.append({"case": case, "method": method, "target": norm_json(target),
                 "input_statistics_sha256": hashlib.sha256(json.dumps(budget_rows).encode()).hexdigest(),
                 "input_statistics": budget_rows, **asdict(result),
                 "estimate_ms": estimate_ms})
    if result.status != "optimal":
        raise RuntimeError(f"{case}/{method} failed: {result.status}")
    return result.log2_bound


def section_xa(connection, rows, solver="shannon"):
    # Paper's tight family: z encodes both y and z0.
    x_size, y_size, z_size = 3, 5, 7
    connection.execute("CREATE OR REPLACE TABLE r AS SELECT x, y FROM range(?) a(x), range(?) b(y)",
                       [x_size, y_size])
    connection.execute("CREATE OR REPLACE TABLE s AS SELECT y, y * ? + z0 AS z "
                       "FROM range(?) a(y), range(?) b(z0)", [z_size, y_size, z_size])
    start = perf_counter()
    budgets = collect_budgets(connection, "r", [degree("xy")])
    budgets += collect_budgets(connection, "s", [degree("yz", "y", 2)])
    stats_ms = 1000 * (perf_counter() - start)
    target = degree("xz", "x", 2)
    measure(rows, "section_xa", FORMULATIONS[solver], "xyz", budgets, target, solver)
    return {"x_size": x_size, "y_size": y_size, "z0_size": z_size,
            "base_statistics_ms": stats_ms}, x_size**0.5 * y_size * z_size


def heavy_key(connection, rows, solver="shannon"):
    m = 100
    n = m**2
    connection.execute("CREATE OR REPLACE TABLE r AS SELECT i AS x, i AS y FROM range(?) a(i)", [n + 1])
    for table, key, value in [("s", "y", "z"), ("t", "x", "u")]:
        connection.execute(f"CREATE OR REPLACE TABLE {table} AS "
                           f"SELECT i AS {key}, 0 AS {value} FROM range(?) a(i) "
                           f"UNION ALL SELECT 0, j FROM range(1, ?) b(j)", [n + 1, m])
    start = perf_counter()
    r_stats = [degree("xy"), degree("xy", "x", math.inf), degree("xy", "y", math.inf)]
    view_inputs = collect_budgets(connection, "r", r_stats)
    view_inputs += collect_budgets(connection, "s", [degree("yz", "y", q) for q in ORDERS])
    t_inputs = collect_budgets(connection, "t", [degree("xu", "x", q) for q in ORDERS])
    stats_ms = 1000 * (perf_counter() - start)
    view_profile = {}
    for q in ORDERS:
        target = degree("xz", "x", q)
        view_profile[q] = (target, measure(rows, "heavy_key_view", FORMULATIONS[solver], "xyz", view_inputs, target, solver))

    target = degree("xzu")
    final_bounds = {}
    for method, orders in [("Card-only", (1,)), ("Card+Max", (1, math.inf)),
                           ("Degree-profile", (1, 2, 4, 8, math.inf))]:
        budgets = [view_profile[q] for q in orders] + t_inputs
        final_bounds[method] = measure(rows, "heavy_key_outer", method, "xzu", budgets, target, solver)
    final_bounds["Global-General"] = measure(rows, "heavy_key_outer", "Global-General", "xyzu",
                                             view_inputs + t_inputs, target, solver)
    return {"heavy_degree": m, "singleton_keys": n, "base_statistics_ms": stats_ms}, final_bounds


def record_truth(connection, rows, queries):
    for row in rows:
        if row["case"] not in queries:
            continue
        blocks = tuple(map(tuple, row["target"]["blocks"]))
        exponents = tuple(float(q) for q in row["target"]["exponents"])
        start = perf_counter()
        truth = log_norm(connection, queries[row["case"]], MixedNorm(blocks, exponents))
        row["truth_ms"] = 1000 * (perf_counter() - start)
        gap = row["log2_bound"] - truth
        row.update(log2_truth=truth, log2_gap=gap, bound_over_truth=math.exp2(gap),
                   safety_violation=gap < -1e-9)


def git_state(directory):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(directory), *args], text=True, stderr=subprocess.DEVNULL).strip()

    try:
        return {"dirty": bool(git("status", "--porcelain"))}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"dirty": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=FORMULATIONS, default="shannon")
    args = parser.parse_args()
    rows = []
    with duckdb.connect(config={"threads": 1}) as connection:
        xa, xa_truth = section_xa(connection, rows, args.solver)
        record_truth(connection, rows, {"section_xa": V_SQL})
        heavy, final_bounds = heavy_key(connection, rows, args.solver)
        record_truth(connection, rows, {"heavy_key_view": V_SQL, "heavy_key_outer": Q_SQL})
    for row in rows:
        if row["case"] == "heavy_key_outer":
            row["bound_over_global"] = math.exp2(row["log2_bound"] - final_bounds["Global-General"])
    if any(row["safety_violation"] for row in rows):
        raise RuntimeError("Shannon bound below DuckDB truth")
    expected = {"section_xa": math.log2(xa_truth), "heavy_key_outer": math.log2(2 * heavy["singleton_keys"])}
    for row in rows:
        if row["case"] in expected and abs(row["log2_truth"] - expected[row["case"]]) > 1e-9:
            raise RuntimeError(f"Unexpected DuckDB truth for {row['case']}")
        tight = row["case"] != "heavy_key_outer" or row["method"] in ("Degree-profile", "Global-General")
        if tight and abs(row["log2_gap"]) > 1e-9:
            raise RuntimeError(f"Expected a tight bound for {row['case']}/{row['method']}")

    run_id = "projected_path_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = PROJECT / "results" / "synthetic" / run_id
    output.mkdir(parents=True)
    report = {"run_id": run_id, "status": "passed", "dataset": "deterministic projected paths",
              "seed": None, "semantics": "set", "solver": "GLOP", "formulation": args.solver, "threads": 1,
              "python": sys.version, "duckdb": duckdb.__version__, "ortools": ortools.__version__,
              "project": git_state(PROJECT), "baseline": git_state(PROJECT.parent / "LpBound"),
              "queries": {"view": V_SQL, "outer": Q_SQL}, "section_xa": xa,
              "heavy_key": heavy, "results": rows}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output / "report.json")
    for row in rows:
        print(f"{row['case']:16} {row['method']:16} {row['target']['exponents']!s:12} "
              f"bound/truth={row['bound_over_truth']:.9g}")


if __name__ == "__main__":
    main()
