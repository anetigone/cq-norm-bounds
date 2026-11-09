"""Evaluate byte-capped mixed profiles and crossing keys on unseen IMDb producers."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from itertools import combinations
import math
from pathlib import Path
from statistics import median
from time import perf_counter

from job_views import Atom, degree_counts_sql, histogram_sql, imdb_identity
from run_jobjoin_boundary import TOLERANCE, encoded_size
from run_jobjoin_mixed import A, B, C, ATTRIBUTES, MIXED, ordinary_budgets
from run_jobjoin_oracles import read
from run_jobjoin_reuse import consumer_budgets, decade_predicates
from run_jobjoin_views import PROJECT, degree, git_state, norm_json, open_imdb, query, write_json
from cqnorm import coverage_flow, shannon
from cqnorm.numeric import histogram_log_upper, log2_count_upper


CASES = (
    ('aka_title', 'kind_type', ('movie_id', 'kind_id', 'id')),
    ('complete_cast', 'comp_cast_type', ('movie_id', 'subject_id', 'status_id')),
    ('movie_info', 'info_type', ('movie_id', 'info_type_id', 'id')),
    ('movie_keyword', 'keyword', ('movie_id', 'keyword_id', 'id')),
    ('movie_link', 'link_type', ('movie_id', 'link_type_id', 'linked_movie_id')),
)
TARGETS = (MIXED, shannon.MixedNorm(((A,), (B,), (C,)), (4, 2, 1)),
           shannon.MixedNorm(((A,), (B,), (C,)), (math.inf, 2, 1)))
AC = degree(ATTRIBUTES, (A, C), math.inf)
BYTE_CAP = 192
LP_LIMIT_MS = 5000


def measured_norm(db, grouped, target, timeout):
    start = perf_counter()
    outer, inner, _ = target.exponents
    per_a = 'max(degree)' if math.isinf(inner) else 'sum(degree::HUGEINT * degree)'
    sql = (f'SELECT moment, count(*) FROM (SELECT {A}, {per_a} AS moment '
           f'FROM ({grouped}) GROUP BY {A}) GROUP BY moment')
    hist, ms = query(db, sql, timeout)
    if not hist:
        value = None
    elif math.isinf(inner):
        value = histogram_log_upper(hist, outer)
    elif math.isinf(outer):
        value = log2_count_upper(max(moment for moment, _ in hist), 2)
    else:
        # (4,2,1) gives the fourth root of the sum of squared inner moments.
        value = log2_count_upper(sum(count * moment**2 for moment, count in hist), 4)
    return value, {'sql': sql, 'ms': ms, 'total_ms': (perf_counter() - start) * 1000}


def flow(inputs, target):
    return coverage_flow.solve(ATTRIBUTES, inputs, target, method='flow', time_limit_ms=LP_LIMIT_MS)


def checked_flow(inputs, target):
    result = flow(inputs, target)
    if result.status != 'optimal':
        raise ValueError(f'Expected compatible optimal LP, got {result.status}: {target}')
    return result.log2_bound


def profile_rows(profile):
    return [{**norm_json(stat), 'log2_budget': value} for stat, value in profile]


def select(candidates, training, target):
    start = perf_counter()
    best = None
    eligible = 0
    unbounded = 0
    for pair in candidates:
        size = encoded_size(pair)
        if size > BYTE_CAP:
            continue
        eligible += 1
        bounds = [flow([*pair, consumer], target) for consumer in training]
        if any(result.status == 'unbounded' for result in bounds):
            unbounded += 1
            continue
        if any(result.status != 'optimal' for result in bounds):
            raise ValueError(f'Failed training LP: {[result.status for result in bounds]}')
        mean = math.fsum(result.log2_bound for result in bounds) / len(training)
        # Prefer smaller profiles on equal training bounds, then retain enumeration order.
        if best is None or mean < best[0] - TOLERANCE or (abs(mean - best[0]) <= TOLERANCE and size < best[1]):
            best = (mean, size, pair)
    if best is None:
        raise ValueError('No candidate profile fits the predeclared byte cap')
    mean, size, pair = best
    return pair, {'training_mean_log2_bound': mean, 'bytes': size, 'eligible_pairs': eligible,
                  'unbounded_pairs': unbounded,
                  'selection_ms': (perf_counter() - start) * 1000, 'profile': profile_rows(pair)}


def compare_profiles(base, ordinary_targets, target, training, consumers, truths):
    start = perf_counter()
    ordinary = [(stat, checked_flow(base, stat)) for stat in ordinary_targets]
    ordinary_profile_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    mixed = target, checked_flow(base, target)
    mixed_row_ms = (perf_counter() - start) * 1000
    control, control_meta = select(combinations(ordinary, 2), training, target)
    candidate, candidate_meta = select(((row, mixed) for row in ordinary), training, target)
    # Training decides whether to keep mixed. Ties retain the ordinary profile.
    use_mixed = candidate_meta['training_mean_log2_bound'] < control_meta['training_mean_log2_bound'] - TOLERANCE
    estimates = []
    for consumer, truth in zip(consumers, truths, strict=True):
        start = perf_counter()
        global_bound = checked_flow([*base, consumer], target)
        global_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        ordinary_bound = checked_flow([*control, consumer], target)
        ordinary_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        mixed_bound = checked_flow([*candidate, consumer], target)
        mixed_ms = (perf_counter() - start) * 1000
        if min(ordinary_bound, mixed_bound) < global_bound - TOLERANCE:
            raise ValueError('Profile tighter than same-input global LP')
        if truth is not None and min(global_bound, ordinary_bound, mixed_bound) < truth - TOLERANCE:
            raise ValueError('Mixed norm estimate below truth')
        estimates.append({'truth': truth, 'global': global_bound, 'ordinary': ordinary_bound, 'forced': mixed_bound,
                          'selected': mixed_bound if use_mixed else ordinary_bound,
                          'online_ms': {'flat': global_ms, 'ordinary': ordinary_ms, 'mixed': mixed_ms}})
    return {'ordinary': control_meta, 'mixed': candidate_meta, 'selected_mixed': use_mixed,
            'profile_inference_ms': ordinary_profile_ms + mixed_row_ms,
            'ordinary_candidate_ms': ordinary_profile_ms, 'extra_mixed_candidate_ms': mixed_row_ms,
            'estimates': estimates}


def timed_solver(solver, inputs, target):
    start = perf_counter()
    result = solver(inputs, target)
    return {**asdict(result), 'wall_ms': (perf_counter() - start) * 1000}


def cross_ablation(base, added, consumers, targets, truths):
    """Drop either crossing row group and probe an incompatible third key."""
    rows = []
    ab_rows = [row for row in base if row[0].blocks[0] == (A, B)]
    if not ab_rows:
        raise ValueError('Missing AB degree statistics')
    for target in targets:
        for consumer, truth in zip(consumers, truths[target], strict=True):
            inputs = [*base, consumer, added[0]]
            control = checked_flow(inputs[:-1], target)
            without_ab = checked_flow([row for row in inputs if row not in ab_rows], target)
            compatible = timed_solver(flow, inputs, target)
            reference = timed_solver(lambda rows, goal: shannon.solve(ATTRIBUTES, rows, goal, LP_LIMIT_MS), inputs, target)
            if compatible['status'] != 'optimal' or reference['status'] != 'optimal':
                raise ValueError('Crossing program failed')
            full = compatible['log2_bound']
            if (abs(full - reference['log2_bound']) > TOLERANCE or full > min(control, without_ab) + TOLERANCE
                    or (truth is not None and full < truth - TOLERANCE)):
                raise ValueError('Crossing ablation failed soundness or monotonicity')
            rejected = timed_solver(flow, [*inputs, added[1]], target)
            if rejected['status'] != 'incompatible':
                raise ValueError('Three different two-attribute keys should be incompatible')
            fallback = timed_solver(lambda rows, goal: shannon.solve(ATTRIBUTES, rows, goal, LP_LIMIT_MS),
                                    [*inputs, added[1]], target)
            if (fallback['status'] != 'optimal' or fallback['log2_bound'] > full + TOLERANCE
                    or (truth is not None and fallback['log2_bound'] < truth - TOLERANCE)):
                raise ValueError('Shannon fallback failed soundness or monotonicity')
            rows.append({'target': norm_json(target), 'truth': truth, 'without_ac': control, 'without_ab': without_ab,
                         'flow': compatible, 'shannon': reference, 'three_keys': rejected, 'fallback': fallback})
    return rows


def run_case(db, case, training, consumers, predicates, timeout):
    fact, dimension, columns = case
    atom = Atom(fact, 'r', tuple(zip(columns, ATTRIBUTES, strict=True)))
    _, base_ms = query(db, f'CREATE OR REPLACE TEMP TABLE mixed_base AS {atom.sql()}', timeout)
    ordinary, extractions = ordinary_budgets(db, timeout)
    dim_sql = f'SELECT DISTINCT id AS {B} FROM {dimension} WHERE id IS NOT NULL'
    [(size,)], dimension_ms = query(db, f'SELECT count(*) FROM ({dim_sql})', timeout)
    base = [*ordinary, (degree((B,)), log2_count_upper(size))]
    source = 'SELECT * FROM mixed_base'
    _, group_ms = query(db, f'CREATE OR REPLACE TEMP TABLE mixed_fact_groups AS {degree_counts_sql(source, (A, B))}', timeout)
    _, view_ms = query(db, f'CREATE OR REPLACE TEMP TABLE mixed_view_groups AS '
                          f'SELECT * FROM mixed_fact_groups NATURAL JOIN ({dim_sql})', timeout)
    added = []
    for stat in (AC, degree(ATTRIBUTES, (B, C), math.inf)):
        sql = histogram_sql(source, stat.blocks[0])
        hist, ms = query(db, sql, timeout)
        added.append((stat, histogram_log_upper(hist, math.inf)))
        extractions.append({'sql': sql, 'ms': ms})
    result = {'case': case, 'base_ms': base_ms, 'dimension_ms': dimension_ms, 'group_ms': group_ms,
              'truth_view_ms': view_ms, 'extracts': extractions, 'targets': []}
    truths = {}
    for target in TARGETS:
        base_norm, extract = measured_norm(db, 'SELECT * FROM mixed_fact_groups', target, timeout)
        observed = []
        for predicate in predicates:
            grouped = (f'SELECT g.* FROM mixed_view_groups g JOIN title t ON g.{A} = t.id WHERE {predicate}')
            truth, sql = measured_norm(db, grouped, target, timeout)
            observed.append({'log2_truth': truth, **sql})
        truths[target] = [row['log2_truth'] for row in observed]
        treatments = {'ordinary-base': base, 'mixed-base': [*base, (target, base_norm)]}
        result['targets'].append({'target': norm_json(target), 'base_mixed_extraction': extract, 'truth': observed,
                                  'treatments': {label: compare_profiles(rows, [stat for stat, _ in ordinary],
                                                                        target, training, consumers, truths[target])
                                                 for label, rows in treatments.items()}})
    result['crossing'] = cross_ablation(base, added, consumers, TARGETS, truths)
    return result


def summarize(cases):
    profiles = {}
    for label in ('ordinary-base', 'mixed-base'):
        per_view = {}
        for case in cases:
            targets = [row['treatments'][label] for row in case['targets']]
            estimates = [e for target in targets for e in target['estimates'] if e['truth'] is not None]
            counts = {'nonempty': len(estimates), 'selected_targets': sum(t['selected_mixed'] for t in targets)}
            for method in ('forced', 'selected'):
                gaps = [e['ordinary'] - e[method] for e in estimates]
                counts[method] = {'wins': sum(g > TOLERANCE for g in gaps), 'losses': sum(g < -TOLERANCE for g in gaps),
                                  'ties': sum(abs(g) <= TOLERANCE for g in gaps),
                                  'max_reduction': math.exp2(max(gaps, default=0)),
                                  'max_inflation': math.exp2(-min(gaps, default=0))}
            per_view[case['case'][0]] = counts
        profiles[label] = per_view
    rows = [row for case in cases for row in case['crossing']]
    return {'profiles': profiles, 'crossing': {'programs': len(rows),
            'flow_passed': sum(r['flow']['status'] == 'optimal' for r in rows),
            'ac_useful': sum(r['without_ac'] > r['flow']['log2_bound'] + TOLERANCE for r in rows),
            'ab_useful': sum(r['without_ab'] > r['flow']['log2_bound'] + TOLERANCE for r in rows),
            'both_useful': sum(min(r['without_ac'], r['without_ab']) > r['flow']['log2_bound'] + TOLERANCE for r in rows),
            'three_key_fallbacks': dict(Counter(r['fallback']['status'] for r in rows)),
            'fallback_median_ms': median(r['fallback']['wall_ms'] for r in rows),
            'flow_median_ms': median(r['flow']['wall_ms'] for r in rows)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sql-timeout', type=float, default=90)
    args = parser.parse_args()
    validation = read(PROJECT / 'results/data_validation/imdb_job.json')
    if validation['status'] != 'passed' or args.sql_timeout <= 0:
        raise ValueError('Validated IMDb data and a positive timeout are required')
    tables = {row['table']: row for row in validation['tables']}
    run_id = 'mixed_holdout_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = PROJECT / 'results/jobjoin_views' / run_id
    output.mkdir(parents=True)
    train = decade_predicates(1960, 10, 4)
    held_out = decade_predicates(2000, 5, 4)
    write_json(output / 'config.json', {'project': git_state(PROJECT), 'runner_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                                       'byte_cap': BYTE_CAP, 'cases': CASES, 'targets': list(map(norm_json, TARGETS)),
                                       'data_identity': imdb_identity(tables),
                                       'train': train, 'held_out': held_out, 'sql_timeout': args.sql_timeout})
    print(f'Run directory: {output}', flush=True)
    cases = []
    with open_imdb(output, tables, 4, '4GB') as db:
        training, train_stats = consumer_budgets(db, train, args.sql_timeout)
        consumers, test_stats = consumer_budgets(db, held_out, args.sql_timeout)
        write_json(output / 'consumer_statistics.json', {'train': train_stats, 'test': test_stats})
        for case in CASES:
            result = run_case(db, case, training, consumers, held_out, args.sql_timeout)
            cases.append(result)
            write_json(output / f'{case[0]}.json', result)
            print(f'{case[0]}: passed', flush=True)
    write_json(output / 'summary.json', {'status': 'passed', **summarize(cases)})


if __name__ == '__main__':
    main()
