"""Measure degree propagation through a reconverging DAG and certificate updates."""

import argparse
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import math
from pathlib import Path
from random import Random
from statistics import median
import sys
from time import perf_counter

import duckdb
import ortools

from job_views import histogram_sql
from run_projected_path import log_norm
from run_jobjoin_views import PROFILES, PROJECT, degree, git_state, norm_json, padded, write_json
from cqnorm.coverage_flow import solve
from cqnorm.numeric import certificate_cost, histogram_log_upper


VIEW_SQL = 'SELECT DISTINCT x, y FROM r JOIN s USING (z)'
LEFT_SQL = f'SELECT x, y, u FROM ({VIEW_SQL}) JOIN a USING (y)'
RIGHT_SQL = f'SELECT x, y, v FROM ({VIEW_SQL}) JOIN b USING (y)'
NESTED_SQL = f'SELECT x, y, u, v FROM ({LEFT_SQL}) JOIN ({RIGHT_SQL}) USING (x, y)'
FLAT_SQL = f'SELECT DISTINCT x, y, u, v FROM r JOIN s USING (z) JOIN a USING (y) JOIN b USING (y)'
TERMINAL = degree('xyuv')
ORDERS = PROFILES['Degree-profile']
TOLERANCE = 1e-8


def flow(attributes, inputs, target):
    result = solve(tuple(attributes), inputs, target, method='flow', time_limit_ms=5000)
    if result.status != 'optimal':
        raise RuntimeError(f'Flow failed on {attributes}/{target}: {result.status}')
    return result


def setup(connection, seed, peak, alignment):
    keys = list(range(64))
    Random(seed).shuffle(keys)
    heavy_left = keys[:8]
    heavy_right = {'aligned': heavy_left, 'partial': keys[4:12], 'disjoint': keys[8:16]}[alignment]
    connection.execute('CREATE OR REPLACE TABLE r AS SELECT * FROM range(16) x(x), range(2) z(z)')
    connection.execute('CREATE OR REPLACE TABLE s AS SELECT * FROM range(2) z(z), range(64) y(y)')
    for table, payload, heavy in [('a', 'u', heavy_left), ('b', 'v', heavy_right)]:
        connection.execute(f'CREATE OR REPLACE TABLE {table} AS '
                           f'SELECT y, {payload} FROM range(64) t(y), '
                           f'range(?) p({payload}) WHERE {payload} = 0 OR y IN '
                           '(SELECT unnest(?))', [peak, heavy])
    return heavy_left, heavy_right


def base_statistics(connection):
    targets = {'r': (degree('xz'), degree('xz', ('z',), math.inf)),
               's': (degree('zy'), degree('zy', ('z',), math.inf)),
               'a': tuple(degree('yu', ('y',), q) for q in ORDERS),
               'b': tuple(degree('yv', ('y',), q) for q in ORDERS)}
    budgets = {}
    for table, rows in targets.items():
        histograms = {}
        budgets[table] = []
        for stat in rows:
            key = stat.blocks[0] if len(stat.blocks) > 1 else ()
            if key not in histograms:
                histograms[key] = connection.execute(histogram_sql(f'SELECT * FROM {table}', key)).fetchall()
            budgets[table].append((stat, histogram_log_upper(histograms[key], stat.exponents[0])))
    return budgets


def exact_branch_inputs(connection):
    inputs = []
    for head, sql in [('xyu', LEFT_SQL), ('xyv', RIGHT_SQL)]:
        for q in ORDERS:
            stat = degree(head, ('x', 'y'), q)
            inputs.append((stat, padded(log_norm(connection, sql, stat))))
    return inputs


def upper_dot(weights, inputs):
    return certificate_cost(weights, (value for _, value in inputs))


def propagate(base, orders, retained=None):
    profiles = dict(base)
    certificates = {}
    stages = [('view', 'xyz', ('r', 's'), [degree('xy', ('y',), q) for q in orders]),
              ('left', 'xyu', ('view', 'a'), [degree('xyu', ('x', 'y'), q) for q in orders]),
              ('right', 'xyv', ('view', 'b'), [degree('xyv', ('x', 'y'), q) for q in orders]),
              ('root', 'xyuv', ('left', 'right'), [TERMINAL])]
    for name, attributes, children, targets in stages:
        inputs = [row for child in children for row in profiles[child]]
        profiles[name] = []
        certificates[name] = []
        for index, target in enumerate(targets):
            if retained is None:
                result = flow(attributes, inputs, target)
                value = result.log2_bound
                certificates[name].append(tuple(map(Fraction, result.input_weights)))
            else:
                value = upper_dot(retained[name][index], inputs)
            profiles[name].append((target, value))
    return profiles['root'][0][1], certificates


def trial(connection, seed, peak, alignment):
    heavy_left, heavy_right = setup(connection, seed, peak, alignment)
    base = base_statistics(connection)
    flat_budgets = [row for rows in base.values() for row in rows]
    bounds = {'Global-General': flow('xyzuv', flat_budgets, TERMINAL).log2_bound}
    for method, orders in PROFILES.items():
        bounds[method], _ = propagate(base, orders)
    # All joins here produce a nonempty set. The truth oracle is evaluated after inference.
    start = perf_counter()
    count = connection.execute(f'SELECT count(*) FROM ({NESTED_SQL})').fetchone()[0]
    difference = connection.execute(f'WITH nested AS ({NESTED_SQL}), flat AS ({FLAT_SQL}) '
                                    'SELECT count(*) FROM ((SELECT * FROM nested EXCEPT SELECT * FROM flat) '
                                    'UNION ALL (SELECT * FROM flat EXCEPT SELECT * FROM nested))').fetchone()[0]
    truth_ms = 1000 * (perf_counter() - start)
    expected = 16 * sum((peak if key in heavy_left else 1) * (peak if key in heavy_right else 1)
                        for key in range(64))
    if difference or count != expected:
        raise RuntimeError('Flattened SQL, nested SQL and analytic count disagree')
    log_truth = math.log2(count)
    bounds['Exact-profile'] = flow('xyuv', exact_branch_inputs(connection), TERMINAL).log2_bound
    global_bound = bounds['Global-General']
    if (min(bounds.values()) < log_truth - TOLERANCE
            or any(bounds[method] < global_bound - TOLERANCE for method in PROFILES)
            or bounds['Exact-profile'] > bounds['Degree-profile'] + TOLERANCE):
        raise RuntimeError('Bound consistency check failed')
    return {'seed': seed, 'peak': peak, 'alignment': alignment, 'truth': count, 'truth_ms': truth_ms,
            'difference_rows': difference, 'log2_bounds': bounds,
            'bound_over_truth': {key: math.exp2(value - log_truth) for key, value in bounds.items()},
            'bound_over_global': {key: math.exp2(value - global_bound) for key, value in bounds.items()}}


def changed_budgets(base, scenario, delta):
    if scenario == 'uniform-log-scale':
        scale = 1 + delta / 4
        return {name: [(stat, value * scale) for stat, value in rows] for name, rows in base.items()}
    cardinality = degree('yu')
    return {**base, 'a': [(stat, value if stat == cardinality else value + delta) for stat, value in base['a']]}


def update_trials(connection, repeats):
    setup(connection, 0, 32, 'aligned')
    base = base_statistics(connection)
    old, certificate = propagate(base, ORDERS)
    if abs(old - propagate(base, ORDERS, certificate)[0]) > TOLERANCE:
        raise RuntimeError('Stored certificate fails at original budgets')
    results = []
    for scenario in ('uniform-log-scale', 'local-moment-relaxation'):
        for delta in (0.5, 1.0, 2.0, 4.0):
            updated = changed_budgets(base, scenario, delta)
            timings = {'reuse': [], 'reoptimize': []}
            values = {}
            for iteration in range(repeats + 2):
                methods = tuple(timings) if iteration % 2 == 0 else tuple(reversed(timings))
                for method in methods:
                    start = perf_counter()
                    value, _ = propagate(updated, ORDERS, certificate if method == 'reuse' else None)
                    elapsed = 1000 * (perf_counter() - start)
                    if method in values and abs(values[method] - value) > TOLERANCE:
                        raise RuntimeError('Certificate update is not repeatable')
                    values[method] = value
                    if iteration >= 2:
                        timings[method].append(elapsed)
            gap = values['reuse'] - values['reoptimize']
            if gap < -TOLERANCE or (scenario == 'uniform-log-scale' and abs(gap) > TOLERANCE):
                raise RuntimeError('Invalid old-certificate bound on changed budgets')
            results.append({'scenario': scenario, 'delta': delta, 'log2_bounds': values,
                            'reuse_over_reoptimize': math.exp2(gap), 'timings_ms': timings,
                            'median_ms': {method: median(times) for method, times in timings.items()}})
    return {'base_statistics': {name: [{**norm_json(stat), 'log2_budget': value} for stat, value in rows]
                                for name, rows in base.items()},
            'certificates': {node: [[str(weight) for weight in row] for row in rows]
                             for node, rows in certificate.items()}, 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=31)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    run_id = 'diamond_updates_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = PROJECT / 'results/synthetic' / run_id
    output.mkdir(parents=True)
    print(f'Run directory: {output}', flush=True)
    with duckdb.connect(config={'threads': 1}) as connection:
        rows = [trial(connection, seed, peak, alignment) for seed in range(5) for peak in (4, 16, 32)
                for alignment in ('aligned', 'partial', 'disjoint')]
        updates = update_trials(connection, args.repeats)
    write_json(output / 'report.json', {'run_id': run_id, 'status': 'passed', 'project': git_state(PROJECT),
                                        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                        'python': sys.version, 'duckdb': duckdb.__version__, 'ortools': ortools.__version__,
                                        'threads': 1, 'semantics': 'set', 'repeats': args.repeats, 'warmups': 2,
                                        'nested_sql': NESTED_SQL, 'flat_sql': FLAT_SQL, 'diamond': rows, 'updates': updates})
    print(f'{len(rows)} diamond cases and {len(updates["results"])} budget updates passed', flush=True)


if __name__ == '__main__':
    main()
