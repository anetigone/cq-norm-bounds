"""Compare shared-profile reuse with flattened Flow on real IMDb consumers."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import json
import math
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import duckdb
import ortools

from job_views import Atom, degree_counts_sql, degree_log, same_imdb_input
from run_jobjoin_boundary import encoded_size
from run_jobjoin_mixed import A, B, ATTRIBUTES, CASES, MIXED, mixed_histogram
from run_jobjoin_views import PROJECT, degree, git_state, norm_json, open_imdb, query, write_json
from cqnorm.coverage_flow import solve
from cqnorm.numeric import log2_count_upper
from cqnorm.shannon import MixedNorm


CONSUMERS = tuple((kind, years) for years in ('', ' >= 2000', ' < 2000', ' >= 2010') for kind in (1, 2))
METHODS = ('Global-General', 'Reuse-full', 'Reuse-mixed-only')
COMPACT_METHODS = ('Reuse-compact-preselected', 'Reuse-compact-with-selection')
TOLERANCE = 1e-8


def predicates():
    return [f'kind_id = {kind}' + (f' AND production_year{years}' if years else '') for kind, years in CONSUMERS]


def decade_predicates(start, width, windows):
    return [f'kind_id = {kind} AND production_year BETWEEN {year} AND {year + width - 1}'
            for year in range(start, start + width * windows, width) for kind in (1, 2)]


def checked_bound(inputs):
    result = solve(ATTRIBUTES, inputs, MIXED, method='flow', time_limit_ms=5000)
    if result.status != 'optimal':
        raise ValueError(f'Consumer LP failed: {result.status}')
    return result


def infer_profile(base, targets):
    profile = []
    for target in targets:
        result = solve(ATTRIBUTES, base, target, method='flow', time_limit_ms=5000)
        if result.status != 'optimal':
            raise ValueError(f'Profile LP failed: {result.status}')
        profile.append((target, result.log2_bound))
    return profile


def select_profile(full, training):
    """Keep the mixed row and at most one ordinary row used on training."""
    certificates = [checked_bound([*full, consumer]).input_weights for consumer in training]
    uses = Counter(index for weights in certificates for index, weight in enumerate(weights[:len(full)])
                   if full[index][0] != MIXED and Fraction(weight) > 0)
    retained = {next(index for index, (stat, _) in enumerate(full) if stat == MIXED)}
    if uses:
        retained.add(min(uses, key=lambda index: (-uses[index], index)))
    selected = [row for index, row in enumerate(full) if index in retained]
    return selected, {'selected_indices': sorted(retained), 'ordinary_use_counts': dict(uses),
                      'training_weights': certificates}


def timed_run(method, base, targets, consumers, *, selected=(), training=()):
    start = perf_counter()
    profile = []
    setup = {}
    if method == 'Reuse-compact-with-selection':
        full = infer_profile(base, targets)
        selection_start = perf_counter()
        profile, decision = select_profile(full, training)
        setup = {'candidate_profile_ms': 1000 * (selection_start - start),
                 'selection_ms': 1000 * (perf_counter() - selection_start),
                 'selected_indices': decision['selected_indices']}
    elif method == 'Reuse-compact-preselected':
        profile = infer_profile(base, selected)
    elif method != 'Global-General':
        requested = targets if method == 'Reuse-full' else (MIXED,)
        profile = infer_profile(base, requested)
    profile_ms = 1000 * (perf_counter() - start) if profile else 0
    bounds, cumulative_ms = [], []
    for consumer in consumers:
        bounds.append(checked_bound([*(profile or base), consumer]))
        cumulative_ms.append(1000 * (perf_counter() - start))
    return {'profile_ms': profile_ms, 'cumulative_ms': cumulative_ms, 'bounds': bounds,
            'profile_items': len(profile), 'profile_bytes': encoded_size(profile) if profile else 0, **setup}


def time_summary(trials):
    global_trials = trials['Global-General']
    summary = {}
    for method, runs in trials.items():
        prefix = []
        for index in range(len(runs[0]['cumulative_ms'])):
            durations = [run['cumulative_ms'][index] for run in runs]
            differences = [time - reference['cumulative_ms'][index] for time, reference in zip(durations, global_trials)]
            prefix.append({'consumers': index + 1, 'median_cumulative_ms': median(durations),
                           'median_paired_difference_ms': median(differences),
                           'faster_repeats': sum(difference < 0 for difference in differences)})
        earlier = [row['consumers'] for row in prefix if row['median_paired_difference_ms'] < 0]
        sustained = next((row['consumers'] for index, row in enumerate(prefix)
                          if all(later['median_paired_difference_ms'] < 0 for later in prefix[index:])), None)
        summary[method] = {'median_profile_ms': median(run['profile_ms'] for run in runs),
                           'first_faster_prefix': earlier[0] if earlier else None,
                           'sustained_break_even': sustained, 'prefix': prefix}
        for phase in ('candidate_profile_ms', 'selection_ms'):
            if phase in runs[0]:
                summary[method][f'median_{phase}'] = median(run[phase] for run in runs)
    return summary


def base_model(prior):
    saved = next(row['input_statistics'] for row in prior['results']
                 if row['stage'] == 'view' and row['variant'] == 'mixed-base')
    base = [(MixedNorm(tuple(map(tuple, row['blocks'])), tuple(float(q) for q in row['exponents'])),
             float(row['log2_budget'])) for row in saved]
    # The last row is the category dimension; all preceding fact summaries
    # are the same statistics requested at the shared-view boundary.
    if base[-1][0] != degree((B,)) or base[-2][0] != MIXED:
        raise ValueError('Unexpected archived mixed-base model')
    return saved, base, tuple(stat for stat, _ in base[:-1])


def collect_truth(connection, fact, dimension, columns, selectors, timeout):
    atom = Atom(fact, 'r', tuple(zip(columns, ATTRIBUTES)))
    dimension_sql = f'SELECT DISTINCT id AS {B} FROM {dimension} WHERE id IS NOT NULL'
    grouped = degree_counts_sql(f'SELECT * FROM ({atom.sql()}) NATURAL JOIN ({dimension_sql})', (A, B))
    _, setup_ms = query(connection, f'CREATE OR REPLACE TEMP TABLE reuse_truth AS {grouped}', timeout)
    rows = []
    for selector in selectors:
        filtered = (f'SELECT g.* FROM reuse_truth g WHERE EXISTS '
                    f'(SELECT 1 FROM title t WHERE t.id = g.{A} AND {selector})')
        sql = mixed_histogram(filtered)
        histogram, ms = query(connection, sql, timeout)
        rows.append({'truth_sql': sql, 'truth_ms': ms, 'truth_empty': not histogram,
                     'log2_truth': degree_log(histogram, 2) if histogram else None})
    return {'grouping_sql': grouped, 'setup_ms': setup_ms, 'consumers': rows}


def consumer_budgets(connection, selectors, timeout):
    consumers, consumer_stats = [], []
    for selector in selectors:
        sql = f'SELECT count(DISTINCT id) FROM title WHERE {selector}'
        [(count,)], ms = query(connection, sql, timeout)
        if not count:
            raise ValueError(f'Consumer title predicate has empty support: {selector}')
        consumers.append((degree((A,)), log2_count_upper(count)))
        consumer_stats.append({'predicate': selector, 'stats_sql': sql, 'stats_ms': ms,
                               'title_rows': count, 'log2_budget': consumers[-1][1]})
    return consumers, consumer_stats


def run_case(connection, prior, case, repeats, warmups, timeout, compact=False):
    fact, dimension, columns = case
    saved, base, targets = base_model(prior)
    selectors = decade_predicates(2000, 5, 4) if compact else predicates()
    training, selected, training_report = (), (), {}
    methods = METHODS + COMPACT_METHODS if compact else METHODS
    if compact:
        training, stats = consumer_budgets(connection, decade_predicates(1960, 10, 4), timeout)
        profile, decision = select_profile(infer_profile(base, targets), training)
        selected = tuple(stat for stat, _ in profile)
        training_report = {'training': {'consumers': stats, **decision,
                                        'title_stats_ms': sum(row['stats_ms'] for row in stats),
                                        'selected_statistics': [norm_json(stat) for stat in selected]}}
    consumers, consumer_stats = consumer_budgets(connection, selectors, timeout)

    trials = {method: [] for method in methods}
    for iteration in range(warmups + repeats):
        for offset in range(len(methods)):
            method = methods[(iteration + offset) % len(methods)]
            result = timed_run(method, base, targets, consumers, selected=selected, training=training)
            if method == 'Reuse-compact-with-selection' and result['selected_indices'] != decision['selected_indices']:
                raise ValueError('Training selection changed between repeats')
            if iteration >= warmups:
                trials[method].append(result)
    truth = collect_truth(connection, fact, dimension, columns, selectors, timeout)
    estimates = []
    for method, runs in trials.items():
        for index, (result, exact) in enumerate(zip(runs[0]['bounds'], truth['consumers'])):
            if any(abs(run['bounds'][index].log2_bound - result.log2_bound) > TOLERANCE for run in runs):
                raise ValueError(f'Inconsistent repeat for {method}/{index + 1}')
            global_bound = trials['Global-General'][0]['bounds'][index].log2_bound
            value = result.log2_bound
            log_truth = exact['log2_truth']
            if value < global_bound - TOLERANCE or (log_truth is not None and value < log_truth - TOLERANCE):
                raise ValueError(f'Bound consistency failed for {method}/{index + 1}')
            if not compact and index < 2 and method == 'Reuse-full':
                earlier = next(row['log2_bound'] for row in prior['results'] if row['stage'] == 'downstream'
                               and row['method'] == 'Degree+Mixed' and row['variant'] == 'mixed-base'
                               and row['kind_id'] == index + 1 and row['target_name'] == 'mixed')
                if abs(value - earlier) > TOLERANCE:
                    raise ValueError('Existing mixed consumer bound did not reproduce')
            full_bound = trials['Reuse-full'][0]['bounds'][index].log2_bound
            if method != 'Global-General' and value < full_bound - TOLERANCE:
                raise ValueError(f'Profile subset bound below full profile for {method}/{index + 1}')
            if method in COMPACT_METHODS:
                other = trials[COMPACT_METHODS[0]][0]['bounds'][index].log2_bound
                if abs(value - other) > TOLERANCE:
                    raise ValueError('Preselected and freshly selected profiles disagree')
            estimates.append({'method': method, 'consumer': index + 1, **asdict(result), **exact,
                              'bound_over_truth': math.exp2(value - log_truth) if log_truth is not None else None,
                              'bound_over_global': math.exp2(value - global_bound),
                              'bound_over_full': math.exp2(value - full_bound)})
    summary = time_summary(trials)
    for method, runs in trials.items():
        ratios = [row['bound_over_global'] for row in estimates if row['method'] == method]
        full_ratios = [row['bound_over_full'] for row in estimates if row['method'] == method]
        nonempty = [row['bound_over_full'] for row in estimates if row['method'] == method and not row['truth_empty']]
        summary[method].update(profile_items=runs[0]['profile_items'], profile_bytes=runs[0]['profile_bytes'],
                               median_bound_over_global=median(ratios), max_bound_over_global=max(ratios),
                               matches_full=sum(abs(math.log2(ratio)) <= TOLERANCE for ratio in full_ratios),
                               nonempty_consumers=len(nonempty),
                               nonempty_matches_full=sum(abs(math.log2(ratio)) <= TOLERANCE for ratio in nonempty),
                               max_bound_over_full=max(full_ratios))
    timings = {method: [{key: value for key, value in run.items() if key != 'bounds'} for run in runs]
               for method, runs in trials.items()}
    return {'case': fact, 'base_inputs': saved, 'consumers': consumer_stats, 'truth': truth,
            'estimates': estimates, 'summary': summary, 'timings': timings, **training_report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mixed_run', type=Path)
    parser.add_argument('--repeats', type=int, default=31)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--sql-timeout', type=float, default=90)
    parser.add_argument('--compact', action='store_true', help='train on 1960–1999 and time held-out consumers from 2000–2019')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.sql_timeout <= 0:
        parser.error('Invalid repeats, warmups, or SQL timeout')
    source = args.mixed_run.resolve()
    archived = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    validation = json.loads((PROJECT / 'results/data_validation/imdb_job.json').read_text(encoding='utf-8'))
    tables = {row['table']: row for row in validation['tables']}
    if validation['status'] != 'passed' or not same_imdb_input(archived, tables):
        raise ValueError('The mixed run and local data validation use different snapshots')
    prefix = 'jobjoin_compact_' if args.compact else 'jobjoin_reuse_'
    run_id = prefix + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = PROJECT / 'results/jobjoin_views' / run_id
    output.mkdir(parents=True)
    write_json(output / 'config.json', {'run_id': run_id, 'mixed_run': str(source), 'mixed_run_config': archived,
                                        'project': git_state(PROJECT), 'warmups': args.warmups, 'repeats': args.repeats,
                                        'compact': args.compact,
                                        'python': sys.version, 'duckdb': duckdb.__version__, 'ortools': ortools.__version__,
                                        'sql_timeout': args.sql_timeout, 'lp_threads': 1, 'lp_timeout_ms': 5000,
                                        'timing_scope': 'online LP construction, solving and verification, including cold profile build'})
    print(f'Run directory: {output}', flush=True)
    reports = []
    with open_imdb(output, tables, 4, '4GB') as connection:
        for case in CASES:
            prior = json.loads((source / f'{case[0]}.json').read_text(encoding='utf-8'))
            if prior['status'] != 'passed':
                raise ValueError(f'Archived mixed case did not pass: {case[0]}')
            report = run_case(connection, prior, case, args.repeats, args.warmups, args.sql_timeout, args.compact)
            reports.append(report)
            write_json(output / f'{case[0]}.json', report)
            print(f'{case[0]}: passed', flush=True)
    pooled = {}
    for method in reports[0]['timings']:
        pooled[method] = []
        for same_round in zip(*(report['timings'][method] for report in reports)):
            cumulative = [sum(times) for times in zip(*(trial['cumulative_ms'] for trial in same_round))]
            phases = {phase: sum(trial[phase] for trial in same_round)
                      for phase in ('profile_ms', 'candidate_profile_ms', 'selection_ms') if phase in same_round[0]}
            pooled[method].append({'cumulative_ms': cumulative, **phases})
    write_json(output / 'summary.json', {'run_id': run_id, 'status': 'passed', 'cases': {r['case']: r['summary'] for r in reports},
                                        'pooled_time': time_summary(pooled)})


if __name__ == '__main__':
    main()
