"""Compare two ordinary rows against one ordinary and one mixed row on IMDb."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import duckdb
import ortools

from job_views import Atom, degree_log, histogram_sql, imdb_identity, same_imdb_input
from run_jobjoin_boundary import encoded_size
from run_jobjoin_mixed import A, B, C, ATTRIBUTES, CASES, MIXED
from run_jobjoin_reuse import (TOLERANCE, checked_bound, collect_truth, consumer_budgets,
                               decade_predicates, infer_profile)
from run_jobjoin_views import PROJECT, degree, git_state, norm_json, open_imdb, padded, query, write_json
from cqnorm import shannon


def read_inputs(prior, variant):
    rows = next(row['input_statistics'] for row in prior['results']
                if row['stage'] == 'view' and row['variant'] == variant)
    return [(shannon.MixedNorm(tuple(map(tuple, row['blocks'])), tuple(map(float, row['exponents']))),
             row['log2_budget']) for row in rows]


def select_pair(candidates, training):
    """Minimize mean log upper bound on training consumers in each class."""
    start = perf_counter()
    best = None
    for pair in candidates:
        bounds = [checked_bound([*pair, consumer]).log2_bound for consumer in training]
        score = math.fsum(bounds) / len(bounds)
        if best is None or score < best[0] - TOLERANCE:
            best = (score, pair)
    score, pair = best
    return pair, {'training_mean_log2_bound': score, 'selection_ms': 1000 * (perf_counter() - start)}


def compare_profiles(base, targets, training, consumers, truth):
    start = perf_counter()
    profile = infer_profile(base, targets)
    profile_ms = 1000 * (perf_counter() - start)
    ordinary, mixed = profile[:-1], profile[-1]
    if mixed[0] != MIXED:
        raise ValueError('Expected the mixed statistic last')
    choices = {'two-ordinary': combinations(ordinary, 2),
               'one-ordinary-one-mixed': ((row, mixed) for row in ordinary)}
    results = {}
    global_bounds = [checked_bound([*base, consumer]).log2_bound for consumer in consumers]
    for name, candidates in choices.items():
        pair, selection = select_pair(candidates, training)
        estimates = []
        for consumer, exact, reference in zip(consumers, truth, global_bounds):
            value = checked_bound([*pair, consumer]).log2_bound
            log_truth = exact['log2_truth']
            if value < reference - TOLERANCE or (log_truth is not None and value < log_truth - TOLERANCE):
                raise ValueError('Invalid held-out bound')
            estimates.append({'log2_bound': value, 'log2_truth': log_truth, 'truth_empty': exact['truth_empty'],
                              'log2_global': reference})
        results[name] = {**selection, 'profile_items': len(pair), 'profile_bytes': encoded_size(pair),
                         'profile': [{**norm_json(stat), 'log2_budget': value} for stat, value in pair],
                         'estimates': estimates}
    return {'candidate_profile_ms': profile_ms, **results}


def crossing_case(connection, case, base, consumers, truths, timeout):
    """Measure a real {A,B}/{A,C} input pair with the same mixed target."""
    fact, _, columns = case
    atom = Atom(fact, 'r', tuple(zip(columns, ATTRIBUTES)))
    crossed = degree(ATTRIBUTES, (A, C), math.inf)
    sql = histogram_sql(atom.sql(), (A, C))
    hist, extract_ms = query(connection, sql, timeout)
    log_budget = padded(degree_log(hist, math.inf))
    if not any(stat.blocks[0] == (A, B) for stat, _ in base):
        raise ValueError('Expected an {A,B} base statistic')
    rows = []
    for consumer, exact in zip(consumers, truths):
        inputs = [*base, consumer, (crossed, log_budget)]
        flow = checked_bound(inputs)
        reference = shannon.solve(ATTRIBUTES, inputs, MIXED, time_limit_ms=5000)
        original = checked_bound(inputs[:-1]).log2_bound
        truth = exact['log2_truth']
        if (reference.status != 'optimal' or abs(reference.log2_bound - flow.log2_bound) > TOLERANCE
                or flow.log2_bound > original + TOLERANCE
                or (truth is not None and flow.log2_bound < truth - TOLERANCE)):
            raise ValueError('Crossing-key correctness check failed')
        rows.append({'flow_order': flow.order, 'log2_flow': flow.log2_bound, 'log2_shannon': reference.log2_bound,
                     'log2_ordinary': original, 'log2_truth': truth})
    return {'added_statistic': norm_json(crossed), 'log2_budget': log_budget, 'statistics_sql': sql,
            'extract_ms': extract_ms, 'estimates': rows}


def summarize(cases):
    summary = {}
    for variant in ('ordinary-base', 'mixed-base'):
        gaps = []
        for case in cases:
            estimates = case['treatments'][variant]
            first = estimates['two-ordinary']['estimates']
            second = estimates['one-ordinary-one-mixed']['estimates']
            gaps.extend(left['log2_bound'] - right['log2_bound'] for left, right in zip(first, second)
                        if not left['truth_empty'])
        summary[variant] = {'nonempty': len(gaps), 'mixed_tighter': sum(gap > TOLERANCE for gap in gaps),
                            'ordinary_tighter': sum(gap < -TOLERANCE for gap in gaps),
                            'median_ordinary_over_mixed': math.exp2(median(gaps)),
                            'min_ordinary_over_mixed': math.exp2(min(gaps)),
                            'max_ordinary_over_mixed': math.exp2(max(gaps))}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mixed_run', type=Path)
    parser.add_argument('--sql-timeout', type=float, default=90)
    args = parser.parse_args()
    if args.sql_timeout <= 0:
        parser.error('--sql-timeout must be positive')
    source = args.mixed_run.resolve()
    archived = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    validation = json.loads((PROJECT / 'results/data_validation/imdb_job.json').read_text(encoding='utf-8'))
    tables = {row['table']: row for row in validation['tables']}
    if validation['status'] != 'passed' or not same_imdb_input(archived, tables):
        raise ValueError('The mixed run and local data validation use different snapshots')
    run_id = 'jobjoin_equal_budget_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = PROJECT / 'results/jobjoin_views' / run_id
    output.mkdir(parents=True)
    training_sql = decade_predicates(1960, 10, 4)
    held_out_sql = decade_predicates(2000, 5, 4)
    write_json(output / 'config.json', {'run_id': run_id, 'project': git_state(PROJECT),
                                        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                        'mixed_run': source.name, 'data_identity': imdb_identity(tables),
                                        'python': sys.version, 'duckdb': duckdb.__version__, 'ortools': ortools.__version__,
                                        'lp_threads': 1, 'duckdb_threads': 4, 'sql_timeout': args.sql_timeout,
                                        'semantics': 'set', 'training': training_sql, 'held_out': held_out_sql})
    print(f'Run directory: {output}', flush=True)
    reports = []
    with open_imdb(output, tables, 4, '4GB') as connection:
        training, training_stats = consumer_budgets(connection, training_sql, args.sql_timeout)
        consumers, consumer_stats = consumer_budgets(connection, held_out_sql, args.sql_timeout)
        for case in CASES:
            prior = json.loads((source / f'{case[0]}.json').read_text(encoding='utf-8'))
            if prior['status'] != 'passed':
                raise ValueError(f'Archived mixed case failed: {case[0]}')
            truth = collect_truth(connection, *case, held_out_sql, args.sql_timeout)
            ordinary_base = read_inputs(prior, 'ordinary-base')
            mixed_base = read_inputs(prior, 'mixed-base')
            targets = tuple(stat for stat, _ in ordinary_base[:-1]) + (MIXED,)
            treatments = {variant: compare_profiles(base, targets, training, consumers, truth['consumers'])
                          for variant, base in [('ordinary-base', ordinary_base), ('mixed-base', mixed_base)]}
            crossing = crossing_case(connection, case, ordinary_base, consumers, truth['consumers'], args.sql_timeout)
            report = {'case': case[0], 'treatments': treatments, 'crossing': crossing,
                      'training_stats': training_stats, 'consumer_stats': consumer_stats, 'truth': truth}
            reports.append(report)
            write_json(output / f'{case[0]}.json', report)
            print(f'{case[0]}: passed', flush=True)
    write_json(output / 'summary.json', {'run_id': run_id, 'status': 'passed', 'groups': summarize(reports),
                                        'crossing': {case['case']: case['crossing'] for case in reports}})


if __name__ == '__main__':
    main()
