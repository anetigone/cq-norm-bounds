"""Measure setup, reuse, and bound quality on held-out STATS consumers."""

import argparse
from datetime import date
from fractions import Fraction
import json
import math
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import duckdb
import ortools

from job_views import Atom, degree_counts_sql
from run_jobjoin_boundary import encoded_size
from run_jobjoin_mixed import A, B, ATTRIBUTES, MIXED, mixed_histogram, ordinary_budgets
from run_jobjoin_oracles import read
from run_jobjoin_reuse import (COMPACT_METHODS, METHODS, TOLERANCE, checked_bound, infer_profile,
                              select_profile, time_summary, timed_run)
from run_jobjoin_views import degree, git_state, norm_json, query, write_json
from run_stats_mixed import CASES, DIMENSION, TEST, TRAIN, consumers, exact_mixed_truth
from run_stats_views import DATA, PROJECT, digest, load_data, run_directory
from cqnorm.coverage_flow import FlowProgram
from cqnorm.numeric import histogram_log_upper, log2_count_upper, round_up
from cqnorm.shannon import Bound


GLOBAL = METHODS[0]
FULL = METHODS[1]
PRESELECTED, COLD = COMPACT_METHODS
FLAT_MODEL, FLAT_CERT = 'Flat-model-reuse', 'Flat-certificate'
PROFILE_MODEL, PROFILE_CERT = 'Profile-model-reuse', 'Profile-certificate'
FLAT_METHODS = (GLOBAL, FLAT_MODEL, FLAT_CERT)
STRATEGIES = METHODS + COMPACT_METHODS + (FLAT_MODEL, FLAT_CERT, PROFILE_MODEL, PROFILE_CERT)


def window_stats(db, years, *, granularity, timeout):
    start, stop = years
    half_year = granularity == 'half-year'
    month_bucket = 'CASE WHEN month(creationdate) <= 6 THEN 1 ELSE 7 END' if half_year else 'month(creationdate)'
    day_bucket = 'CASE WHEN day(creationdate) <= 15 THEN 1 ELSE 16 END' if granularity == 'half-month' else '1'
    sql = (f'SELECT year(creationdate) AS y, {month_bucket} AS m, {day_bucket} AS d, posttypeid, count(*) FROM posts '
           f"WHERE posttypeid IN (1, 2) AND creationdate >= '{start}-01-01' AND creationdate < '{stop}-01-01' "
           'GROUP BY y, m, d, posttypeid ORDER BY y, m, d, posttypeid')
    groups, elapsed = query(db, sql, timeout)
    rows = []
    step = 6 if half_year else 1
    for year, month, day, kind, count in groups:
        lower = date(year, month, day)
        if granularity == 'half-month' and day == 1:
            upper = date(year, month, 16)
        else:
            upper = date(year + (month + step - 1) // 12, (month + step - 1) % 12 + 1, 1)
        rows.append({'post_type': kind, 'from': str(lower), 'until': str(upper), 'base_rows': count,
                     'predicate': f"posttypeid = {kind} AND creationdate >= '{lower}' AND creationdate < '{upper}'"})
    if not rows:
        raise ValueError('Missing consumer support statistics')
    return rows, {'sql': sql, 'ms': elapsed}


def extract_base(db, atom, timeout):
    start = perf_counter()
    projection_sql = f'CREATE OR REPLACE TEMP TABLE mixed_base AS {atom.sql()}'
    _, projection_ms = query(db, projection_sql, timeout)
    ordinary, extractions = ordinary_budgets(db, timeout)
    mixed_sql = mixed_histogram(degree_counts_sql('SELECT * FROM mixed_base', (A, B)))
    histogram, mixed_ms = query(db, mixed_sql, timeout)
    dimension_sql = f'SELECT count(*) FROM ({DIMENSION})'
    [(count,)], dimension_ms = query(db, dimension_sql, timeout)
    if not histogram or not count:
        raise ValueError('Empty producer base statistics')
    fact = [*ordinary, (MIXED, histogram_log_upper(histogram, 2))]
    dimension = degree((A,)), log2_count_upper(count)
    ordinary_ms = sum(r['ms'] for r in extractions)
    budget_ms = 1000 * (perf_counter() - start) - projection_ms - ordinary_ms - mixed_ms - dimension_ms
    return fact, dimension, {'projection_sql': projection_sql, 'projection_ms': projection_ms,
                             'ordinary_queries': extractions, 'ordinary_ms': ordinary_ms, 'budget_ms': budget_ms,
                             'mixed_sql': mixed_sql, 'mixed_ms': mixed_ms,
                             'dimension_sql': dimension_sql, 'dimension_ms': dimension_ms}


def verify_archived_truth(rows, archived):
    # The outer squared 2-norm is additive over disjoint sets of posts.
    for reference in archived:
        within = [r for r in rows if r['post_type'] == reference['post_type'] and
                  reference['from'] <= r['from'] < reference['until']]
        count = sum(r['base_rows'] for r in within)
        squares = math.fsum(math.exp2(2 * r['log2_truth']) for r in within if not r['truth_empty'])
        if count != reference['base_rows'] or bool(squares) == reference['truth_empty']:
            raise ValueError('Window statistics did not roll up to archived half-year counts')
        if squares and abs(math.log2(squares) / 2 - reference['log2_truth']) > TOLERANCE:
            raise ValueError('Window mixed truth did not roll up to the archived half-year truth')


def cached_run(method, base, batch, selected):
    start = perf_counter()
    profile = infer_profile(base, selected) if method in (PROFILE_MODEL, PROFILE_CERT) else []
    profile_ms = 1000 * (perf_counter() - start) if profile else 0
    inputs = profile or base
    static = tuple(value for _, value in inputs)
    consumer_stat = batch[0][0]
    if any(stat != consumer_stat for stat, _ in batch):
        raise ValueError('Cached template requires the same consumer statistic')
    phases = {}
    if method in (FLAT_MODEL, PROFILE_MODEL):
        build_start = perf_counter()
        model = FlowProgram(ATTRIBUTES, (*[stat for stat, _ in inputs], consumer_stat), MIXED, time_limit_ms=5000)
        phases.update(model_setup_ms=1000 * (perf_counter() - build_start), model_proto_bytes=model.model_bytes)
    else:
        seed_start = perf_counter()
        seed = checked_bound([*inputs, batch[0]])
        weights = tuple(map(Fraction, seed.input_weights))
        sparse = [(i, str(weight)) for i, weight in enumerate(weights) if weight]
        constant = sum((weight * Fraction(value) for weight, value in zip(weights[:-1], static, strict=True)),
                       Fraction(0))
        phases.update(seed_ms=1000 * (perf_counter() - seed_start),
                      certificate_bytes=len(json.dumps(sparse, separators=(',', ':')).encode()),
                      evaluator_bytes=len(json.dumps([str(constant), str(weights[-1])], separators=(',', ':')).encode()),
                      certificate_weights=sparse)
    bounds, elapsed = [], []
    for index, (_, value) in enumerate(batch):
        if method in (FLAT_MODEL, PROFILE_MODEL):
            bound = model.solve((*static, value))
        elif index == 0:
            bound = seed
        else:
            bound = Bound('optimal', round_up(constant + weights[-1] * Fraction(value)))
        if bound.status != 'optimal':
            raise ValueError(f'{method}: {bound.status}')
        bounds.append(bound)
        elapsed.append(1000 * (perf_counter() - start))
    return {'profile_ms': profile_ms, 'profile_items': len(profile), 'profile_bytes': encoded_size(profile) if profile else 0,
            'bounds': bounds, 'cumulative_ms': elapsed, **phases}


def accuracy(trials, truth):
    estimates, summaries = {}, {}
    global_logs = trials[GLOBAL][0]['log2_bounds']
    full_logs = trials[FULL][0]['log2_bounds']
    preselected = trials[PRESELECTED][0]['log2_bounds']
    for method, runs in trials.items():
        first = runs[0]
        logs = first['log2_bounds']
        cached_certificate = method in (FLAT_CERT, PROFILE_CERT)
        if not cached_certificate and any(abs(a - b) > TOLERANCE
                                          for run in runs for a, b in zip(logs, run['log2_bounds'], strict=True)):
            raise ValueError(f'Consumer bounds changed across repeats for {method}')
        if method == COLD and any(abs(a - b) > TOLERANCE for a, b in zip(logs, preselected, strict=True)):
            raise ValueError('Preselected and freshly selected bounds disagree')
        reference_logs = preselected if method in (PROFILE_MODEL, PROFILE_CERT) else global_logs
        if method in (FLAT_MODEL, PROFILE_MODEL) and any(abs(a - b) > TOLERANCE
                                                        for a, b in zip(logs, reference_logs, strict=True)):
            raise ValueError(f'Cached model disagrees with fresh solve: {method}')
        if cached_certificate:
            gaps = [v - ref for run in runs for v, ref in zip(run['log2_bounds'], reference_logs, strict=True)]
            if min(gaps) < -TOLERANCE:
                raise ValueError(f'Cached certificate is tighter than reoptimization: {method}')
        per_window = []
        for value, reference, full, exact in zip(logs, global_logs, full_logs, truth, strict=True):
            log_truth = exact['log2_truth']
            if (value < reference - TOLERANCE or (method not in FLAT_METHODS and value < full - TOLERANCE) or
                    (log_truth is not None and value < log_truth - TOLERANCE)):
                raise ValueError(f'Invalid bound for {method}')
            per_window.append({'log2_bound': value, 'bound_over_global': math.exp2(value - reference),
                               'bound_over_full': math.exp2(value - full),
                               'bound_over_truth': math.exp2(value - log_truth) if log_truth is not None else None})
        estimates[method] = per_window
        nonempty = [r for r in per_window if r['bound_over_truth'] is not None]
        summaries[method] = {'profile_items': first['profile_items'], 'profile_bytes': first['profile_bytes'],
                             'nonempty_matches_full': sum(abs(math.log2(r['bound_over_full'])) <= TOLERANCE for r in nonempty),
                             'nonempty_max_over_full': max(r['bound_over_full'] for r in nonempty),
                             'nonempty_median_over_global': median(r['bound_over_global'] for r in nonempty),
                             'nonempty_max_over_global': max(r['bound_over_global'] for r in nonempty),
                             'nonempty_median_over_truth': median(r['bound_over_truth'] for r in nonempty)}
        if cached_certificate:
            nonempty_gaps = [v - ref for run in runs
                             for v, ref, row in zip(run['log2_bounds'], reference_logs, truth, strict=True)
                             if not row['truth_empty']]
            summaries[method].update(certificate_match_fraction=sum(abs(g) <= TOLERANCE for g in nonempty_gaps)
                                                                / len(nonempty_gaps),
                                     certificate_median_over_resolve=math.exp2(median(nonempty_gaps)),
                                     certificate_max_over_resolve=math.exp2(max(nonempty_gaps)))
    return estimates, summaries


def paired_costs(trials):
    comparisons = {}
    for reference in FLAT_METHODS:
        comparisons[reference] = {}
        for method in (PRESELECTED, COLD, PROFILE_MODEL, PROFILE_CERT):
            rows = [[a - b for a, b in zip(run['cumulative_ms'], ref['cumulative_ms'], strict=True)]
                    for run, ref in zip(trials[method], trials[reference], strict=True)]

            def sustained(deltas):
                last_loss = max((i for i, delta in enumerate(deltas) if delta >= 0), default=-1)
                return last_loss + 2 if last_loss < len(deltas) - 1 else None

            prefixes = [sustained(row) for row in rows]
            comparisons[reference][method] = {
                'paired_median_break_even': sustained([median(values) for values in zip(*rows)]),
                'per_trial_break_even': prefixes,
                'faster_at_batch_end': sum(row[-1] < 0 for row in rows),
                'batch_delta_median_ms': median(row[-1] for row in rows)}
    return comparisons


def run_view(db, fact, columns, archived, args):
    timeout = args.sql_timeout
    atom = Atom(fact, 'r', tuple(zip(columns, ATTRIBUTES, strict=True)))
    train, train_query = window_stats(db, TRAIN, granularity='half-year', timeout=timeout)
    test, test_query = window_stats(db, TEST, granularity=args.test_granularity, timeout=timeout)
    archived_train = {(r['from'], r['post_type']): r['base_rows'] for r in archived['training'] if r['base_rows']}
    if {(r['from'], r['post_type']): r['base_rows'] for r in train} != archived_train:
        raise ValueError('Training support differs from the archived STATS run')
    fact_inputs, dimension, sql = extract_base(db, atom, timeout)
    base = [*fact_inputs, dimension]
    targets = tuple(stat for stat, _ in fact_inputs)
    training, testing = consumers(train), consumers(test)
    selected_profile, decision = select_profile(infer_profile(base, targets), training)
    selected = tuple(stat for stat, _ in selected_profile)
    for consumer in testing:
        # The consumer predicate implies the producer's posts restriction.
        flat = checked_bound([*fact_inputs, consumer]).log2_bound
        redundant = checked_bound([*base, consumer]).log2_bound
        if abs(flat - redundant) > TOLERANCE:
            raise ValueError('Simplifying the flat posts restriction changed the reference bound')

    rng = Random(args.seed)
    trials = {method: [] for method in STRATEGIES}
    acquisitions = []
    for iteration in range(args.warmups + args.repeats):
        inputs, dim, times = extract_base(db, atom, timeout)
        train_now, train_time = window_stats(db, TRAIN, granularity='half-year', timeout=timeout)
        test_now, test_time = window_stats(db, TEST, granularity=args.test_granularity, timeout=timeout)
        if inputs != fact_inputs or dim != dimension or train_now != train or test_now != test:
            raise ValueError('Extracted statistics changed across rounds')
        conversion_start = perf_counter()
        testing_now = consumers(test_now)
        test_budget_ms = 1000 * (perf_counter() - conversion_start)
        conversion_start = perf_counter()
        training_now = consumers(train_now)
        train_budget_ms = 1000 * (perf_counter() - conversion_start)
        positions = list(range(len(test)))
        rng.shuffle(positions)
        batch = [testing_now[i] for i in positions]
        fact_ms = sum(times[k] for k in ('projection_ms', 'ordinary_ms', 'mixed_ms', 'budget_ms'))
        costs = {'fact_ms': fact_ms, 'dimension_ms': times['dimension_ms'],
                 'test_support_ms': test_time['ms'] + test_budget_ms,
                 'training_support_ms': train_time['ms'] + train_budget_ms, 'window_order': positions}
        for offset in range(len(STRATEGIES)):
            method = STRATEGIES[(iteration + offset) % len(STRATEGIES)]
            model = inputs if method in FLAT_METHODS else [*inputs, dim]
            if method in (FLAT_MODEL, FLAT_CERT, PROFILE_MODEL, PROFILE_CERT):
                trial = cached_run(method, model, batch, selected)
            else:
                trial = timed_run(method, model, targets, batch, selected=selected, training=training_now)
            if method == COLD and trial['selected_indices'] != decision['selected_indices']:
                raise ValueError('Training selected a different profile during timing')
            if iteration < args.warmups:
                continue
            bounds = [None] * len(positions)
            for i, result in zip(positions, trial.pop('bounds'), strict=True):
                bounds[i] = result.log2_bound
            acquisition_ms = fact_ms + costs['test_support_ms']
            if method not in FLAT_METHODS:
                acquisition_ms += costs['dimension_ms']
            if method == COLD:
                acquisition_ms += costs['training_support_ms']
            trial.update(log2_bounds=bounds, acquisition_ms=acquisition_ms)
            trials[method].append(trial)
        if iteration >= args.warmups:
            acquisitions.append(costs)

    grouped = degree_counts_sql(f'SELECT * FROM mixed_base NATURAL JOIN ({DIMENSION})', (A, B))
    truth_ms = exact_mixed_truth(db, grouped, test, timeout)
    verify_archived_truth(test, archived['test'])
    estimates, tightness = accuracy(trials, test)
    lp_only = time_summary(trials)
    including_sql = {m: [{**r, 'cumulative_ms': [t + r['acquisition_ms'] for t in r['cumulative_ms']]} for r in runs]
                     for m, runs in trials.items()}
    total = time_summary(including_sql)
    for method, runs in trials.items():
        total[method]['median_acquisition_ms'] = median(r['acquisition_ms'] for r in runs)
        total[method]['median_consumer_lp_ms'] = median((r['cumulative_ms'][-1] - r['profile_ms']) / len(test) for r in runs)
        for phase in ('model_setup_ms', 'model_proto_bytes', 'seed_ms', 'certificate_bytes', 'evaluator_bytes'):
            if phase in runs[0]:
                total[method][f'median_{phase}'] = median(r[phase] for r in runs)
    return {'view': fact, 'columns': columns, 'train': train, 'test': test, 'nonempty_outputs': sum(not r['truth_empty'] for r in test),
            'selected_statistics': [norm_json(stat) for stat in selected], 'selection': decision,
            'sql': {**sql, 'training_support_sql': train_query['sql'], 'test_support_sql': test_query['sql'],
                    'truth_group_sql': grouped, 'truth_group_ms': truth_ms},
            'timings': trials, 'acquisitions': acquisitions, 'estimates': estimates, 'tightness': tightness,
            'lp_only': lp_only, 'including_sql': total, 'paired_costs': paired_costs(including_sql),
            'median_sql_phases_ms': {phase: median(r[phase] for r in acquisitions)
                                     for phase in ('fact_ms', 'dimension_ms', 'test_support_ms', 'training_support_ms')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mixed_run', type=Path)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--repeats', type=int, default=31)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--seed', type=int, default=108)
    parser.add_argument('--sql-timeout', type=float, default=30)
    parser.add_argument('--test-granularity', choices=('month', 'half-month'), default='month')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.sql_timeout <= 0:
        parser.error('Invalid repeats, warmups, or SQL timeout')
    mixed = args.mixed_run.resolve()
    output = run_directory('stats_cached_reuse')
    print(f'Run directory: {output}', flush=True)
    cases = {}
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB', 'temp_directory': str(output / 'spill')}) as db:
        data = load_data(db, args.data)
        archived_tables = read(mixed / 'config.json')['data']['tables']
        comparable = {name: {key: row[key] for key in ('rows', 'columns', 'file')}
                      for name, row in archived_tables.items()}
        if data['tables'] != comparable:
            raise ValueError('STATS data differ from the archived mixed run')
        write_json(output / 'config.json', {'mixed_run': str(mixed), 'data': data, 'project': git_state(PROJECT),
                                            'runner_sha256': digest(Path(__file__)), 'duckdb': duckdb.__version__,
                                            'ortools': ortools.__version__, 'training_years': TRAIN, 'test_years': TEST,
                                            'test_granularity': args.test_granularity,
                                            'repeats': args.repeats, 'warmups': args.warmups, 'seed': args.seed,
                                            'sql_timeout': args.sql_timeout, 'duckdb_threads': 4, 'lp_threads': 1})
        for name, columns in CASES:
            report = run_view(db, name, columns, read(mixed / f'{name}.json'), args)
            write_json(output / f'{name}.json', report)
            cases[name] = {k: report[k] for k in ('nonempty_outputs', 'tightness', 'lp_only', 'including_sql',
                                                 'paired_costs', 'median_sql_phases_ms')}
            cases[name]['consumers'] = len(report['test'])
            print(f'{name}: {len(report["test"])} consumers, truth rollup passed', flush=True)
    write_json(output / 'summary.json', {'views': cases})


if __name__ == '__main__':
    main()
