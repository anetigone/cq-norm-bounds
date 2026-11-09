"""E3 prototype run with common log budgets and a shared GLOP backend."""

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import math
import random
import sys
from time import perf_counter

from run_projected_path import PROJECT, git_state, heavy_key, norm_json, section_xa

import duckdb
import ortools

from cqnorm import coverage_flow, shannon
from cqnorm.shannon import MixedNorm

METHODS = ("shannon", "coverage", "flow")
SEEDS = tuple(range(5))


def path(n, rng):
    attributes = tuple(f"x{i}" for i in range(n))
    budgets = []
    for left, right in zip(attributes, attributes[1:]):
        pair = (left, right)
        budgets.append((MixedNorm.degree(pair, (), 1), rng.randint(16, 32) / 4))
        budgets.append((MixedNorm.degree(pair, (left,), rng.choice((2, 4, math.inf))), rng.randint(4, 16) / 4))
    return attributes, budgets, MixedNorm.degree(attributes, (), 1)


def mixed(rng):
    attributes = tuple("abcd")
    stats = [MixedNorm.degree(tuple("abd"), tuple("ab"), 2),
             MixedNorm.degree(tuple("acd"), tuple("ac"), math.inf),
             MixedNorm((("a", "b"), ("c",), ("d",)), (2, math.inf, 1)),
             MixedNorm.degree(attributes, (), 1)]
    budgets = [(stat, rng.randint(12, 40) / 4) for stat in stats]
    q = rng.choice((0.5, 1.5, 2, 4, math.inf))
    return attributes, budgets, MixedNorm((("a",), ("b",), ("c", "d")), (q, math.inf, 1))


def compare(results, case, model, time_limit_ms, methods=METHODS, expected="optimal", seed=None):
    attributes, budgets, target = model
    rows = []
    for method in methods:
        start = perf_counter()
        if method == "shannon":
            result = shannon.solve(attributes, budgets, target, time_limit_ms)
        else:
            result = coverage_flow.solve(attributes, budgets, target, method=method, time_limit_ms=time_limit_ms)
        total_ms = 1000 * (perf_counter() - start)
        rows.append({"case": case, "seed": seed, "n_attributes": len(attributes), "method": method,
                     "attributes": list(attributes), "target": norm_json(target),
                     "budgets": [{**norm_json(stat), "log2_budget": "-inf" if value == -math.inf else value}
                                 for stat, value in budgets], "total_ms": total_ms, **asdict(result)})
    results.extend(rows)
    expected_statuses = [expected] * len(rows) if isinstance(expected, str) else expected
    if any(row["status"] != status and not (case.startswith("scale_") and row["status"] == "timeout")
           for row, status in zip(rows, expected_statuses)):
        raise RuntimeError(f"Unexpected status for {case}: {[(r['method'], r['status']) for r in rows]}")
    values = [row["log2_bound"] for row in rows if row["status"] == "optimal"]
    if values and max(values) - min(values) > 1e-8:
        raise RuntimeError(f"Solver disagreement for {case}: {values}")
    return values


def decode(item):
    return MixedNorm(tuple(map(tuple, item["blocks"])), tuple(float(q) for q in item["exponents"]))


def run_cases(rows, limit):
    synthetic = []
    with duckdb.connect(config={"threads": 1}) as connection:
        section_xa(connection, synthetic)
        heavy_key(connection, synthetic)
    for row in synthetic:
        if row["case"] == "heavy_key_outer":
            attributes = "xyzu" if row["method"] == "Global-General" else "xzu"
        else:
            attributes = "xyz"
        budgets = [(decode(item), item["log2_budget"]) for item in row["input_statistics"]]
        case = f"{row['case']}/{row['method']}/{row['target']['exponents']}"
        compare(rows, case, (attributes, budgets, decode(row["target"])), limit)

    for seed in SEEDS:
        rng = random.Random(seed)
        compare(rows, "random_path", path(6, rng), limit, seed=seed)
        compare(rows, "compound_mixed", mixed(rng), limit, seed=seed)

    attributes = tuple('abc')
    target = MixedNorm((('a',), ('b',), ('c',)), (4, 2, 1))
    crossing = [(MixedNorm.degree(attributes, ('a', 'b'), 4), 2),
                (MixedNorm.degree(attributes, ('a', 'c'), math.inf), 1),
                (MixedNorm.degree(('a', 'c'), (), 1), 3),
                (MixedNorm.degree(('b', 'c'), (), 1), 4)]
    for name, inputs, expected in (
        ('crossing_active', crossing, 8 / 3),
        ('crossing_without_ab', crossing[1:], 7 / 2),
        ('crossing_without_ac', [crossing[0], *crossing[2:]], 23 / 8),
    ):
        values = compare(rows, name, (attributes, inputs, target), limit)
        if any(abs(value - expected) > 1e-8 for value in values):
            raise RuntimeError(f'Unexpected ablation value for {name}: {values}')
    card = MixedNorm.degree(tuple("abcd"), (), 1)
    incompatible = [(MixedNorm.degree(tuple("abc"), tuple("ab"), 2), 2),
                    (MixedNorm.degree(tuple("acd"), tuple("cd"), 2), 2), (card, 4)]
    compare(rows, "incompatible_keys", ("abcd", incompatible, card), limit,
            expected=("optimal", "incompatible", "incompatible"))
    compare(rows, "unbounded", ("abcd", [], card), limit, expected="unbounded")
    compare(rows, "empty_atom", ("abcd", [(card, -math.inf)], card), limit, expected="empty")
    compare(rows, "empty_head", ("", [], MixedNorm((), ())), limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit-ms", type=int, default=5000)
    args = parser.parse_args()
    if args.time_limit_ms <= 0:
        parser.error("--time-limit-ms must be positive")
    run_id = "solver_agreement_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = PROJECT / "results" / "synthetic" / run_id
    output.mkdir(parents=True)
    rows = []
    report = {"run_id": run_id, "status": "failed", "solver": "GLOP", "threads": 1,
              "solver_time_limit_ms": args.time_limit_ms, "seeds": SEEDS,
              "python": sys.version, "duckdb": duckdb.__version__, "ortools": ortools.__version__,
              "project": git_state(PROJECT), "baseline": git_state(PROJECT.parent / "LpBound"),
              "timing": "single run per model; total includes model construction and certificate verification",
              "results": rows}
    try:
        run_cases(rows, args.time_limit_ms)
        print("Small-instance comparisons passed", flush=True)
        for n in (4, 6, 8, 10, 20, 40, 80, 160):
            methods = METHODS if n <= 10 else ("flow",)
            compare(rows, f"scale_{n}", path(n, random.Random(0)), args.time_limit_ms, methods, seed=0)
            print(f"n={n}: " + ", ".join(f"{row['method']}={row['status']} ({row['total_ms']:.1f} ms)"
                                       for row in rows[-len(methods):]), flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
