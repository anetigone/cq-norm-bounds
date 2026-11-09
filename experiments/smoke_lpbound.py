"""Run the original Berge LP solver and verify a small join against DuckDB."""

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent / "LpBound" / "src"))


def main():
    import duckdb
    from lpbound.solver.berge_lp_solver import run_berge_lp_solver

    connection = duckdb.connect()
    truth = connection.execute(
        "SELECT count(*) FROM (VALUES (1,1),(1,2),(2,3)) AS r(x,a) "
        "JOIN (VALUES (1,1),(1,2),(1,3),(2,4)) AS s(x,b) USING(x)"
    ).fetchone()[0]
    estimate, _ = run_berge_lp_solver(
        join_pool_map={("R", "x"): 1, ("S", "x"): 1},
        join_pool_alias_map={"R": [1], "S": [1]},
        aliases=["R", "S"],
        statistics_dict={
            ("R", "x"): [2, 3, math.sqrt(5), 2],
            ("S", "x"): [2, 4, math.sqrt(10), 3],
        },
        domain_size_statistics={},
    )
    assert truth == 7
    assert estimate >= truth - 1e-8
    assert math.isclose(estimate, math.sqrt(50), rel_tol=1e-7)
    print(json.dumps({"truth": truth, "lpbound": estimate,
                      "status": "passed", "scope": "original Python Berge solver only"}))


if __name__ == "__main__":
    main()
