"""Compare Flow and the original Python Berge solver on matched JOBjoin inputs."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import ortools

from job_views import Atom, connected, degree_log, histogram_sql, parse_jobjoin, scope, imdb_identity
from run_jobjoin_boundary import TOLERANCE
from run_jobjoin_oracles import read
from run_jobjoin_views import PROJECT, degree, git_state, norm_json, open_imdb, padded, query, write_json
from run_lpbound_joblight import BASELINE
from cqnorm.coverage_flow import solve
from lpbound.solver.berge_lp_solver import run_berge_lp_solver


ORDERS = (1, 2, 3, 4, math.inf)
WORKLOAD = BASELINE / 'benchmarks/workloads/jobjoin/jobjoinQueries.sql'


def model(db, atoms, histograms, timeout):
    pools = {var: index for index, var in enumerate(sorted(scope(atoms)), 1)}
    if not connected(atoms) or sum(len(atom.scope) for atom in atoms) != len(atoms) + len(pools) - 1:
        return None
    appearances = Counter(var for atom in atoms for var in atom.scope)
    if min(appearances.values()) < 2:
        raise ValueError('A join pool is not shared by two aliases')
    aliases = [atom.alias for atom in atoms]
    statistics, alias_map, pool_map = {}, {}, {}
    budgets, attributes = [], set(pools)
    for atom in atoms:
        # A unique row ID makes full set cardinality equal to full SQL bag cardinality.
        private = f'row_{atom.alias}'
        identified = Atom(atom.table, atom.alias, (*atom.columns, ('id', private)))
        attributes.add(private)
        head = tuple(sorted(identified.scope))
        alias_map[atom.alias] = [pools[var] for var in sorted(atom.scope)]
        for index, var in enumerate(sorted(atom.scope)):
            pair = (atom.alias, var)
            pool_map[pair] = pools[var]
            sql = histogram_sql(identified.sql(), (var,))
            if sql not in histograms:
                rows, elapsed = query(db, sql, timeout)
                if not rows:
                    raise ValueError('Matched benchmark expects nonempty base projections')
                histograms[sql] = {'histogram': rows, 'ms': elapsed}
            histogram = histograms[sql]['histogram']
            logs = [padded(degree_log(histogram, q)) for q in ORDERS]
            statistics[pair] = np.array([0., *map(math.exp2, logs)])
            # Normalize Flow's budgets through the same linear-scale values passed to Berge.
            budgets.extend((degree(head, (var,), q), math.log2(norm))
                           for q, norm in zip(ORDERS, statistics[pair][1:], strict=True) if q != 1 or index == 0)
    return sorted(attributes), budgets, (pool_map, alias_map, aliases, statistics, {})


def measure(attributes, budgets, berge_args, repeats):
    target = degree(attributes)

    def run_flow():
        result = solve(attributes, budgets, target, method='flow', time_limit_ms=20000)
        if result.status != 'optimal':
            raise ValueError(f'Flow failed: {result.status}')
        return result

    def run_berge():
        value, _ = run_berge_lp_solver(*berge_args)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'Berge failed: {value}')
        return math.log2(value)

    methods = {'Flow': run_flow, 'LpBound-Python-Berge': run_berge}
    samples = {name: [] for name in methods}
    values = {}
    for i in range(repeats + 2):
        order = list(methods)
        if i % 2:
            order.reverse()
        for name in order:
            start = perf_counter()
            value = methods[name]()
            ms = (perf_counter() - start) * 1000
            if i >= 2:
                samples[name].append(ms)
            values[name] = value
        if abs(values['Flow'].log2_bound - values['LpBound-Python-Berge']) > TOLERANCE:
            raise ValueError(f'Matched Flow and Berge bounds disagree: {values}')
    return {'log2_bound': values['Flow'].log2_bound, 'flow_model': asdict(values['Flow']),
            'lpbound_log2_bound': values['LpBound-Python-Berge'],
            'times_ms': samples, 'median_ms': {name: median(times) for name, times in samples.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=11)
    parser.add_argument('--sql-timeout', type=float, default=90)
    args = parser.parse_args()
    validation = read(PROJECT / 'results/data_validation/imdb_job.json')
    baseline = git_state(BASELINE)
    if validation['status'] != 'passed' or args.repeats < 1 or args.sql_timeout <= 0:
        raise ValueError('Expected validated IMDb and positive timing options')
    tables = {row['table']: row for row in validation['tables']}
    run_id = 'matched_berge_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = PROJECT / 'results/lpbound' / run_id
    output.mkdir(parents=True)
    write_json(output / 'config.json', {'project': git_state(PROJECT), 'baseline': baseline, 'ortools': ortools.__version__,
                                       'runner_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                                       'data_identity': imdb_identity(tables),
                                       'workload_sha256': sha256(WORKLOAD.read_bytes()).hexdigest(),
                                       'orders': [*ORDERS[:-1], 'inf'], 'repeats': args.repeats, 'warmups': 2,
                                       'scope': 'solver wall time with prepared identical statistics; Flow verifies its certificate'})
    print(f'Run directory: {output}', flush=True)
    histograms, results = {}, []
    with open_imdb(output, tables, 4, '4GB') as db:
        for qid, sql in enumerate(WORKLOAD.read_text(encoding='utf-8').splitlines(), 1):
            if not sql.strip():
                continue
            atoms = parse_jobjoin(sql)
            inputs = model(db, atoms, histograms, args.sql_timeout)
            row = {'query_id': qid, 'sql': sql, 'atoms': len(atoms), 'status': 'not_berge_acyclic'}
            if inputs is not None:
                attributes, budgets, berge_args = inputs
                row.update(status='passed', attributes=attributes, input_statistics=[
                    {**norm_json(stat), 'log2_budget': log} for stat, log in budgets],
                    **measure(attributes, budgets, berge_args, args.repeats))
            results.append(row)
            write_json(output / f'query_{qid:02d}.json', row)
            print(f'{qid:02d}: {row["status"]}', flush=True)
    write_json(output / 'statistics.json', histograms)
    valid = [r['median_ms'] for r in results if r['status'] == 'passed']
    ratios = [r['Flow'] / r['LpBound-Python-Berge'] for r in valid]
    write_json(output / 'summary.json', {'statuses': dict(Counter(r['status'] for r in results)),
                                        'median_query_ms': {name: median(r[name] for r in valid) for name in valid[0]},
                                        'flow_over_berge': {'median': median(ratios), 'min': min(ratios), 'max': max(ratios)},
                                        'statistics_sql_ms': sum(row['ms'] for row in histograms.values())})


if __name__ == '__main__':
    main()
