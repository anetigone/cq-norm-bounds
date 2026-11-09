"""Separate errors in inferred boundary statistics from residual bound slack."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median, quantiles
import sys
from time import perf_counter

import duckdb
import ortools

from run_jobjoin_boundary import (POLICIES, TOLERANCE, decode_plan, exact_profiles, solve_policy)
from run_jobjoin_views import PROJECT, base_budgets, git_state, load_histograms, open_imdb, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checked_terminal(rows, truth):
    if any(row['status'] != 'optimal' or row.get('safety_violation') for row in rows):
        raise ValueError('An inferred profile row failed or fell below its exact value')
    bound = rows[-1]['log2_bound']
    if rows[-1]['stage'] != 'E2' or bound < truth - TOLERANCE:
        raise ValueError('Invalid terminal upper bound')
    return bound


def compare_policy(stages, tail, head, budgets, policy, limit, oracle, archived, truth, global_bound):
    online, _, _ = solve_policy(stages, tail, head, budgets, policy, limit)
    online_bound = checked_terminal(online, truth)
    if abs(online_bound - archived['log2_bound']) > TOLERANCE:
        raise ValueError(f'Archived online bound changed for {policy}')

    local, _, _ = solve_policy(stages, tail, head, budgets, policy, limit, oracle)
    exact_bound = checked_terminal(local, truth)
    checkpoints = [online_bound]
    for layer in range(1, len(stages)):
        reset, _, _ = solve_policy(stages, tail, head, budgets, policy, limit, oracle, exact_after=layer)
        checkpoints.append(checked_terminal(reset, truth))
    checkpoints.append(exact_bound)
    gains = [before - after for before, after in zip(checkpoints, checkpoints[1:])]
    if min(gains) < -TOLERANCE:
        raise ValueError(f'Later exact profile gives a looser bound for {policy}')

    layers = []
    for propagated, one_step in zip(online[:-1], local[:-1], strict=True):
        online_value = propagated['log2_bound']
        local_value = one_step['log2_bound']
        exact_value = one_step['log2_truth']
        if online_value < local_value - TOLERANCE:
            raise ValueError(f'Propagated layer below exact-input inference for {policy}')
        layers.append({'stage': one_step['stage'], 'key': one_step['key'], 'q': one_step['q'],
                       'log2_exact': exact_value, 'log2_local': local_value, 'log2_propagated': online_value})
    return {'online_over_truth': math.exp2(online_bound - truth),
            'exact_profile_over_truth': math.exp2(exact_bound - truth),
            'online_over_exact_profile': math.exp2(online_bound - exact_bound),
            'online_over_global': math.exp2(online_bound - global_bound),
            'exact_profile_over_global': math.exp2(exact_bound - global_bound),
            'log2_bound_by_reset_layer': checkpoints, 'log2_reduction_per_layer': gains,
            'layer_statistics': layers}


def distribution(values):
    return {'median': median(values), 'p90': quantiles(values, n=10, method='inclusive')[-1] if len(values) > 1 else values[0],
            'max': max(values)}


def aggregate(reports):
    summary = {}
    for policy in POLICIES:
        rows = [row['policies'][policy] for row in reports if row['status'] == 'passed']
        if not rows:
            continue
        summary[policy] = {name: distribution([row[name] for row in rows]) for name in (
            'online_over_truth', 'exact_profile_over_truth', 'online_over_exact_profile',
            'online_over_global', 'exact_profile_over_global')}
        summary[policy].update(n=len(rows),
                               improved=sum(row['online_over_exact_profile'] > math.exp2(TOLERANCE) for row in rows),
                               exact_profile_below_global=sum(row['exact_profile_over_global'] < math.exp2(-TOLERANCE) for row in rows),
                               inference_larger_than_residual=sum(row['online_over_exact_profile'] > row['exact_profile_over_truth']
                                                                  for row in rows),
                               by_layer=[{'improved': sum(gain > TOLERANCE for gain in gains),
                                          'log2_reduction': distribution(gains)}
                                         for gains in zip(*(row['log2_reduction_per_layer'] for row in rows))])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('boundary_run', type=Path)
    parser.add_argument('--query-ids', nargs='+', type=int)
    parser.add_argument('--sql-timeout', type=float, default=90)
    args = parser.parse_args()
    if args.sql_timeout <= 0:
        parser.error('--sql-timeout must be positive')
    boundary_dir = args.boundary_run.resolve()
    boundary_config = read(boundary_dir / 'config.json')
    depth_dir = Path(boundary_config['depth_run'])
    source = Path(boundary_config['source_run'])
    config = read(source / 'config.json')
    settings = config['args']
    cache = load_histograms(source)
    run_id = 'jobjoin_oracles_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = PROJECT / 'results/jobjoin_views' / run_id
    output.mkdir(parents=True)
    write_json(output / 'config.json', {'run_id': run_id, 'project': git_state(PROJECT),
                                        'runner_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                                        'boundary_runner_sha256': sha256(Path(__file__).with_name('run_jobjoin_boundary.py').read_bytes()).hexdigest(),
                                        'boundary_run': str(boundary_dir), 'depth_run': str(depth_dir), 'source_run': str(source),
                                        'query_ids': args.query_ids, 'sql_timeout': args.sql_timeout,
                                        'python': sys.version, 'duckdb': duckdb.__version__, 'ortools': ortools.__version__})
    print(f'Run directory: {output}', flush=True)
    reports = []
    with open_imdb(output, config['tables'], settings['duckdb_threads'], settings['duckdb_memory']) as connection:
        for path in sorted(boundary_dir.glob('query_*.json')):
            old = read(path)
            query_id = old['query_id']
            if args.query_ids and query_id not in args.query_ids:
                continue
            report = {'query_id': query_id, 'status': 'failed'}
            reports.append(report)
            started = perf_counter()
            try:
                if old['status'] != 'passed' or not old['sample']['flat_rows'] or old['sample']['difference_rows']:
                    raise ValueError('Archived boundary plan lacks a passed nonempty semantic check')
                depth = next(row for row in read(depth_dir / path.name) if row['depth'] == 4 and row['status'] == 'passed')
                prior = read(source / path.name)
                split, stages, tail = decode_plan(prior, depth)
                cached = len(cache)
                budgets = base_budgets(connection, split.atoms, cache, args.sql_timeout)
                if len(cache) != cached:
                    raise ValueError('Archived base summary is incomplete')
                truth = math.log2(depth['true_cardinality'])
                global_bound = next(row['log2_bound'] for row in prior['results'] if row['method'] == 'Global-General')
                report.update(stages=depth['stages'], tail=depth['tail'], truth=depth['true_cardinality'],
                              log2_global=global_bound, sample_flat_rows=old['sample']['flat_rows'])
                oracle, report['extraction'] = exact_profiles(connection, stages, tail, args.sql_timeout)
                archived = {row['method']: row for row in old['results']}
                report['policies'] = {policy: compare_policy(stages, tail, split.head, budgets, policy,
                                                           settings['lp_time_limit_ms'], oracle, archived[policy], truth,
                                                           global_bound) for policy in POLICIES}
                report['status'] = 'passed'
            except duckdb.InterruptException as error:
                report.update(status='sql_timeout', error=str(error))
            except (duckdb.Error, ValueError) as error:
                report.update(error=str(error))
            report['wall_ms'] = (perf_counter() - started) * 1000
            write_json(output / path.name, report)
            print(f'{query_id:02d}: {report["status"]} ({report["wall_ms"] / 1000:.1f}s)' +
                  (f': {report["error"]}' if 'error' in report else ''), flush=True)
    statuses = dict(Counter(row['status'] for row in reports))
    write_json(output / 'summary.json', {'run_id': run_id, 'statuses': statuses, 'policies': aggregate(reports)})
    if not reports or statuses != {'passed': len(reports)}:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
