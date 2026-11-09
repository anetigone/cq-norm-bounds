"""Sweep projected JOBjoin view depths using a completed depth-one run."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import duckdb

from job_views import check_chain_sql, derive, parse_jobjoin, scope
from run_jobjoin_views import (PROFILES, PROJECT, base_budgets, compare_truth, degree, estimate,
                               git_state, load_histograms, open_imdb, paired_improvement, query, summarize, write_json)


def chain(split, depth, sizes):
    """Extend the original view by one connected atom per layer."""
    if len(split.right) < depth:
        return None
    frontier = scope(split.left)
    tail = list(split.right)
    stages = [(split.left, split.left_head, split.keys)]
    for _ in range(depth - 1):
        def score(atom):
            after = frontier | atom.scope
            outside = scope([other for other in tail if other != atom])
            return len(after & outside), sizes[atom.table], atom.alias

        added = min((atom for atom in tail if frontier & atom.scope), key=score)
        tail.remove(added)
        available = frontier | added.scope
        key = tuple(sorted(available & scope(tail)))
        head = tuple(sorted(set(key) | (available & set(split.head))))
        stages.append(((added,), head, key))
        frontier = frozenset(head)
    return stages, tail


def flatten_budgets(atoms, budgets):
    return [entry for atom in atoms for entry in budgets[atom.alias]]


def run_chain(stages, tail, head, budgets, report, lp_timeout, global_bound):
    rows = report['results']
    truth = math.log2(report['true_cardinality']) if report['true_cardinality'] else -math.inf
    variant = f"depth_{report['depth']}"
    for method, orders in PROFILES.items():
        profile = []
        previous = set()
        profile_ms = 0
        for index, (atoms, view_head, key) in enumerate(stages, 1):
            inputs = profile + flatten_budgets(atoms, budgets)
            attributes = sorted(previous | scope(atoms))
            profile = []
            for order in orders:
                stat = estimate(rows, f'layer_{index}', variant, method, attributes, inputs,
                                degree(view_head, key, order), lp_timeout)
                profile_ms += rows[-1]['total_ms']
                profile.append(stat)
            if any(stat is None for stat in profile):
                rows.append({'stage': 'E2', 'variant': variant, 'method': method, 'status': 'profile_unavailable'})
                break
            previous = set(view_head)
        else:
            attributes = sorted(previous | scope(tail))
            estimate(rows, 'E2', variant, method, attributes, profile + flatten_budgets(tail, budgets),
                     degree(head), lp_timeout)
            row = rows[-1]
            row['profile_ms'] = profile_ms
            compare_truth(row, truth)
            if row['status'] == 'optimal' and global_bound is not None:
                gap = row['log2_bound'] - global_bound
                row['bound_over_global'] = math.exp2(gap)
                row['below_global'] = gap < -1e-8
    if any(row.get('safety_violation') or row.get('below_global') for row in rows):
        raise ValueError('Upper-bound or compositional consistency check failed')


def matched_depth_comparison(reports):
    comparisons = {}
    for depth in sorted({report['depth'] for report in reports}):
        pairs = [report['degree_profile_pair'] for report in reports
                 if report['depth'] == depth and report['status'] == 'passed' and 'degree_profile_pair' in report]
        if not pairs:
            continue
        gaps = [current - previous for previous, current in pairs]
        comparisons[depth] = {'n': len(pairs), 'strictly_looser': sum(gap > 1e-8 for gap in gaps),
                              'strictly_tighter': sum(gap < -1e-8 for gap in gaps),
                              'median_inflation': math.exp2(median(gaps)),
                              'max_inflation': math.exp2(max(gaps))}
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_run', type=Path)
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 4])
    args = parser.parse_args()
    if any(depth < 2 for depth in args.depths):
        parser.error('Depths must be at least 2')
    source = args.source_run.resolve()
    config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    source_args = config['args']
    cache = load_histograms(source)
    sizes = {name: table['rows'] for name, table in config['tables'].items()}
    run_id = 'jobjoin_depth_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = PROJECT / 'results/jobjoin_views' / run_id
    output.mkdir(parents=True)
    write_json(output / 'config.json', {'run_id': run_id, 'source_run': str(source), 'depths': args.depths,
                                        'project': git_state(PROJECT), 'source_config': config})
    print(f'Run directory: {output}', flush=True)
    reports = []
    with open_imdb(output, config['tables'], source_args['duckdb_threads'], source_args['duckdb_memory']) as connection:
        for query_id in config['query_ids']:
            prior = json.loads((source / f'query_{query_id:02d}.json').read_text(encoding='utf-8'))
            if prior['status'] != 'passed':
                continue
            split = derive(parse_jobjoin(prior['source_sql']), sizes, source_args.get('view_atoms', 2),
                           config.get('categorical_columns'))
            depth_one = next(row for row in prior['results'] if row['stage'] == 'E2'
                             and row['variant'] == 'projected' and row['method'] == 'Degree-profile')
            timeout = source_args['sql_timeout']
            budgets = base_budgets(connection, split.atoms, cache, timeout)
            for depth in sorted(set(args.depths)):
                report = {'query_id': query_id, 'depth': depth, 'results': [],
                          'true_cardinality': prior['truth']['outer']['rows']}
                reports.append(report)
                plan = chain(split, depth, sizes)
                if plan is None:
                    report['status'] = 'too_few_atoms'
                    continue
                stages, tail = plan
                report['stages'] = [{'atoms': [atom.alias for atom in atoms], 'head': head, 'key': key}
                                    for atoms, head, key in stages]
                report['tail'] = [atom.alias for atom in tail]
                global_bound = next(row.get('log2_bound') for row in prior['results'] if row['method'] == 'Global-General')
                start = perf_counter()
                try:
                    run_chain(stages, tail, split.head, budgets, report, source_args['lp_time_limit_ms'], global_bound)
                    current = next(row for row in report['results']
                                   if row['stage'] == 'E2' and row['method'] == 'Degree-profile')
                    if depth_one['status'] == current['status'] == 'optimal':
                        report['degree_profile_pair'] = [depth_one['log2_bound'], current['log2_bound']]
                    sql = check_chain_sql(split, prior['source_sql'], [(atoms, head) for atoms, head, _ in stages], tail)
                    check, elapsed = query(connection, sql, timeout)
                    count, difference = check[0]
                    report.update(sample_result_rows=count, sample_difference_rows=difference,
                                  sample_check_ms=elapsed, sample_check_sql=sql)
                    if difference:
                        raise ValueError('Sampled flat and nested queries disagree')
                    report['status'] = 'passed'
                except duckdb.InterruptException as error:
                    report.update(status='sql_timeout', error=str(error))
                except (duckdb.Error, ValueError) as error:
                    report.update(status='failed', error=str(error))
                report['wall_ms'] = (perf_counter() - start) * 1000
                print(f'{query_id:02d} depth={depth}: {report["status"]} ({report["wall_ms"] / 1000:.1f}s)', flush=True)
            write_json(output / f'query_{query_id:02d}.json', [report for report in reports if report['query_id'] == query_id])

    valid = [{**report, 'results': [row for row in report['results'] if row['stage'] == 'E2']}
             for report in reports if report['status'] == 'passed']
    write_json(output / 'summary.json', {'run_id': run_id, 'source_run': str(source),
                                        'statuses': {depth: dict(Counter(r['status'] for r in reports if r['depth'] == depth))
                                                     for depth in sorted(set(args.depths))},
                                        'groups': summarize(valid), 'paired_improvement': paired_improvement(valid),
                                        'matched_depth_comparison': matched_depth_comparison(reports)})
    if any(report['status'] == 'failed' for report in reports) or not valid:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
