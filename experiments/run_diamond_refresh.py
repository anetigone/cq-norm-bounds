"""Measure certificate refresh after tuple mutations and a changed-input statistics scan."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import duckdb
import ortools

from job_views import histogram_sql
from run_diamond_updates import (NESTED_SQL, ORDERS, TERMINAL, TOLERANCE, base_statistics, flow, propagate,
                                 setup, upper_dot)
from run_jobjoin_views import degree, git_state, query, write_json
from run_stats_views import PROJECT, digest, run_directory
from cqnorm.numeric import histogram_log_upper

MUTATIONS = (
    ('insert-heavy', 'INSERT INTO a SELECT y, u FROM unnest(?) t(y), range(32, 64) p(u)', 256),
    ('insert-cold', 'INSERT INTO a SELECT y, u FROM unnest(?) t(y), range(1, 9) p(u)', 64),
    ('delete-heavy', 'DELETE FROM a WHERE y IN (SELECT unnest(?)) AND u >= 1', 504),
    ('delete-cold', 'DELETE FROM a WHERE y IN (SELECT unnest(?)) AND u >= 1', 64),
)


def benchmark(db, repeats):
    initial_heavy, _ = setup(db, 0, 32, 'aligned')
    initial = base_statistics(db)
    start = perf_counter()
    _, certificates = propagate(initial, ORDERS)
    profile_seed_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    flat = flow('xyzuv', [row for rows in initial.values() for row in rows], TERMINAL)
    flat_weights = tuple(map(Fraction, flat.input_weights))
    flat_seed_ms = (perf_counter() - start) * 1000
    sparse = lambda weights: [(i, str(w)) for i, w in enumerate(weights) if w]
    compact = {node: [sparse(weights) for weights in rows] for node, rows in certificates.items()}
    payloads = {'local': compact, 'flat': sparse(flat_weights)}
    setup_report = {'profile_seed_ms': profile_seed_ms, 'flat_seed_ms': flat_seed_ms,
                    'payload_bytes': {name: len(json.dumps(payload, separators=(',', ':')).encode())
                                      for name, payload in payloads.items()}, 'payloads': payloads}
    stats_sql = histogram_sql('SELECT y, u FROM a', ('y',))
    records = [{'batch': name, 'sql': sql, 'affected_rows': affected, 'trials': []}
               for name, sql, affected in MUTATIONS]
    for iteration in range(repeats + 2):
        heavy, _ = setup(db, 0, 32, 'aligned')
        cold = sorted(set(range(64)) - set(heavy))[:8]
        if heavy != initial_heavy:
            raise ValueError('Reset changed the input keys')
        for row in records:
            keys = heavy if row['batch'].endswith('heavy') else cold
            start = perf_counter()
            [(affected,)] = db.execute(row['sql'], [keys]).fetchall()
            mutation_ms = 1000 * (perf_counter() - start)
            if affected != row['affected_rows']:
                raise ValueError(f'Unexpected mutation count for {row["batch"]}: {affected}')
            start = perf_counter()
            histogram, sql_ms = query(db, stats_sql, 30)
            updated = {**initial, 'a': [(degree('yu', ('y',), q), histogram_log_upper(histogram, q))
                                        for q in ORDERS]}
            statistics_ms = 1000 * (perf_counter() - start)
            times, bounds = {}, {}
            methods = {
                'reuse': lambda: propagate(updated, ORDERS, certificates)[0],
                'reoptimize': lambda: propagate(updated, ORDERS)[0],
                'flat_certificate': lambda: upper_dot(flat_weights, [row for rows in updated.values() for row in rows]),
                'flat_reoptimize': lambda: flow('xyzuv', [row for rows in updated.values() for row in rows],
                                                TERMINAL).log2_bound,
            }
            order = tuple(methods)
            offset = iteration % len(order)
            for method in order[offset:] + order[:offset]:
                start = perf_counter()
                bounds[method] = methods[method]()
                times[method] = 1000 * (perf_counter() - start)
            truth = db.execute(f'SELECT count(*) FROM ({NESTED_SQL})').fetchone()[0]
            log_truth = math.log2(truth)
            gap = bounds['reuse'] - bounds['reoptimize']
            flat_gap = bounds['flat_certificate'] - bounds['flat_reoptimize']
            if (min(gap, flat_gap, bounds['reoptimize'] - bounds['flat_reoptimize']) < -TOLERANCE
                    or min(bounds.values()) < log_truth - TOLERANCE):
                raise ValueError('Updated certificate violates the expected bound ordering')
            row.update(truth=truth, bounds=bounds, reuse_over_reoptimize=math.exp2(gap),
                       flat_certificate_over_reoptimize=math.exp2(flat_gap),
                       local_certificate_over_flat_certificate=math.exp2(bounds['reuse'] - bounds['flat_certificate']),
                       reuse_over_truth=math.exp2(bounds['reuse'] - log_truth),
                       reoptimize_over_truth=math.exp2(bounds['reoptimize'] - log_truth))
            if iteration >= 2:
                row['trials'].append({'mutation_ms': mutation_ms, 'statistics_sql_ms': sql_ms,
                                      'statistics_ms': statistics_ms, **{f'{m}_ms': t for m, t in times.items()},
                                      **{f'{m}_total_ms': mutation_ms + statistics_ms + t for m, t in times.items()}})
    for row in records:
        row['median_ms'] = {phase: median(t[phase] for t in row['trials']) for phase in row['trials'][0]}
    return records, stats_sql, setup_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=11)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Repetitions must be positive')
    output = run_directory('diamond_refresh', 'synthetic')
    print(f'Run directory: {output}', flush=True)
    with duckdb.connect(config={'threads': 1}) as db:
        records, stats_sql, setup_report = benchmark(db, args.repeats)
    write_json(output / 'report.json', {'runner_sha256': digest(Path(__file__)), 'project': git_state(PROJECT),
                                        'system': platform.platform(), 'python': platform.python_version(),
                                        'duckdb': duckdb.__version__, 'ortools': ortools.__version__,
                                        'threads': 1, 'repeats': args.repeats, 'warmups': 2,
                                        'statistics_sql': stats_sql, 'certificate_setup': setup_report, 'batches': records})


if __name__ == '__main__':
    main()
