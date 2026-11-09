"""E4 on IMDb views with ordinary versus mixed boundary summaries."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from statistics import median
from time import perf_counter

import duckdb
import ortools

from job_views import Atom, degree_counts_sql, degree_log, histogram_sql, imdb_identity
from run_jobjoin_boundary import encoded_size
from run_jobjoin_views import (ORDERS, PROJECT, compare_truth, degree, estimate, git_state, open_imdb,
                               query, write_json)
from cqnorm.shannon import MixedNorm
from cqnorm.numeric import histogram_log_upper, log2_count_upper

A, B, C = ATTRIBUTES = ('v00', 'v01', 'v02')
MIXED = MixedNorm(((A,), (B,), (C,)), (2, math.inf, 1))
CARD = degree(ATTRIBUTES)
KEYS = ((A,), (B,), (C,), (A, B))
CASES = (
    ('movie_companies', 'company_type', ('movie_id', 'company_type_id', 'company_id')),
    ('cast_info', 'role_type', ('movie_id', 'role_id', 'person_id')),
    ('movie_info_idx', 'info_type', ('movie_id', 'info_type_id', 'id')),
)
KINDS = (1, 2)


def mixed_histogram(grouped):
    return (f'SELECT max_degree, count(*) FROM '
            f'(SELECT {A}, max(degree) AS max_degree FROM ({grouped}) GROUP BY {A}) GROUP BY max_degree')


def ordinary_budgets(connection, timeout):
    budgets, extracts = [], []
    for index, key in enumerate(KEYS):
        sql = histogram_sql('SELECT * FROM mixed_base', key)
        histogram, elapsed = query(connection, sql, timeout)
        extracts.append({'sql': sql, 'ms': elapsed})
        for q in ORDERS if index == 0 else ORDERS[1:]:
            budgets.append((degree(ATTRIBUTES, key, q), histogram_log_upper(histogram, q)))
    return budgets, extracts


def solve_case(connection, report, fact, dimension, columns, sql_timeout, lp_timeout):
    atom = Atom(fact, 'r', tuple(zip(columns, ATTRIBUTES)))
    materialize = f'CREATE OR REPLACE TEMP TABLE mixed_base AS {atom.sql()}'
    _, base_ms = query(connection, materialize, sql_timeout)
    ordinary, extracts = ordinary_budgets(connection, sql_timeout)
    mixed_sql = mixed_histogram(degree_counts_sql('SELECT * FROM mixed_base', (A, B)))
    histogram, mixed_ms = query(connection, mixed_sql, sql_timeout)
    measured_mixed = (MIXED, histogram_log_upper(histogram, 2))
    dim_sql = f'SELECT DISTINCT id AS {B} FROM {dimension} WHERE id IS NOT NULL'
    [(dim_size,)], dim_ms = query(connection, f'SELECT count(*) FROM ({dim_sql})', sql_timeout)
    dim_budget = (degree((B,)), log2_count_upper(dim_size))
    title_budgets = {}
    title_extracts = []
    for kind in KINDS:
        title_sql = f'SELECT count(DISTINCT id) FROM title WHERE kind_id = {kind}'
        [(size,)], elapsed = query(connection, title_sql, sql_timeout)
        title_budgets[kind] = (degree((A,)), log2_count_upper(size))
        title_extracts.append({'kind_id': kind, 'rows': size, 'sql': title_sql, 'ms': elapsed})
    report['base_statistics'] = {'materialize_sql': materialize, 'materialize_ms': base_ms,
                                 'ordinary_histograms': extracts, 'ordinary_bytes': encoded_size(ordinary),
                                 'mixed_sql': mixed_sql, 'mixed_ms': mixed_ms,
                                 'ordinary_plus_mixed_bytes': encoded_size([*ordinary, measured_mixed]),
                                 'dimension_sql': dim_sql, 'dimension_ms': dim_ms, 'title': title_extracts}
    rows = report['results']
    profiles = {}
    for treatment, base in (('ordinary-base', ordinary), ('mixed-base', [*ordinary, measured_mixed])):
        view_input = [*base, dim_budget]
        degree_profile = []
        for target, _ in ordinary:
            degree_profile.append(estimate(rows, 'view', treatment, 'Degree-profile', ATTRIBUTES, view_input, target, lp_timeout))
        ordinary_ms = sum(row['total_ms'] for row in rows[-len(ordinary):])
        mixed_stat = estimate(rows, 'view', treatment, 'Degree+Mixed', ATTRIBUTES, view_input, MIXED, lp_timeout)
        extra_ms = rows[-1]['total_ms']
        if mixed_stat is None or any(stat is None for stat in degree_profile):
            raise ValueError(f'View profile unavailable for {treatment}')
        profiles[treatment] = {'Degree-profile': degree_profile, 'Degree+Mixed': [*degree_profile, mixed_stat]}
        report['footprint'][treatment] = {name: {'items': len(profile), 'json_bytes': encoded_size(profile),
                                                'view_inference_ms': ordinary_ms + (extra_ms if name == 'Degree+Mixed' else 0)}
                                         for name, profile in profiles[treatment].items()}
        for kind, title_budget in title_budgets.items():
            for name, profile in profiles[treatment].items():
                for target_name, target in (('mixed', MIXED), ('cardinality', CARD)):
                    estimate(rows, 'downstream', treatment, name, ATTRIBUTES, [*profile, title_budget], target, lp_timeout)
                    rows[-1].update(kind_id=kind, target_name=target_name)
                    if rows[-1]['status'] != 'optimal':
                        raise ValueError(f'Downstream profile unavailable for {treatment}/{name}')

    # All online bounds above use base statistics only. Exact group counts below
    # serve solely as truth, including both downstream title filters.
    grouped = degree_counts_sql(f'SELECT * FROM mixed_base NATURAL JOIN ({dim_sql})', (A, B))
    materialize_truth = f'CREATE OR REPLACE TEMP TABLE mixed_truth AS {grouped}'
    _, truth_ms = query(connection, materialize_truth, sql_timeout)
    report['truth'] = {'materialize_sql': materialize_truth, 'materialize_ms': truth_ms, 'consumers': {}}
    view_hist, _ = query(connection, mixed_histogram('SELECT * FROM mixed_truth'), sql_timeout)
    for row in rows:
        if row['stage'] == 'view' and row['method'] == 'Degree+Mixed':
            compare_truth(row, degree_log(view_hist, 2))
    for kind in KINDS:
        filtered = (f'SELECT g.* FROM mixed_truth g WHERE EXISTS '
                    f'(SELECT 1 FROM title t WHERE t.id = g.{A} AND t.kind_id = {kind})')
        histogram, mixed_truth_ms = query(connection, mixed_histogram(filtered), sql_timeout)
        [(cardinality,)], card_ms = query(connection, f'SELECT sum(degree::HUGEINT) FROM ({filtered})', sql_timeout)
        truth = {'mixed': degree_log(histogram, 2), 'cardinality': math.log2(cardinality)}
        report['truth']['consumers'][kind] = {'sql': filtered, 'mixed_ms': mixed_truth_ms, 'card_ms': card_ms,
                                              'cardinality': cardinality, 'log2_mixed_truth': truth['mixed']}
        for row in rows:
            if row.get('kind_id') == kind:
                compare_truth(row, truth[row['target_name']])
    if any(row.get('safety_violation') for row in rows):
        raise ValueError('Downstream estimate below exact truth')


def summarize(reports):
    pairs = []
    for report in reports:
        rows = [row for row in report['results'] if row['stage'] == 'downstream']
        controls = {(row['variant'], row['kind_id'], row['target_name']): row for row in rows if row['method'] == 'Degree-profile'}
        for row in rows:
            if row['method'] == 'Degree+Mixed':
                treatment, kind, target = key = row['variant'], row['kind_id'], row['target_name']
                before = controls[key]
                pairs.append({'case': report['case'], 'base': treatment, 'kind_id': kind, 'target': target,
                              'ordinary_over_truth': before['bound_over_truth'], 'mixed_over_truth': row['bound_over_truth'],
                              'log2_reduction': before['log2_bound'] - row['log2_bound']})
    groups = []
    for treatment in ('ordinary-base', 'mixed-base'):
        for target in ('mixed', 'cardinality'):
            subset = [pair for pair in pairs if pair['base'] == treatment and pair['target'] == target]
            if subset:
                groups.append({'base': treatment, 'target': target, 'n': len(subset),
                               'tighter': sum(pair['log2_reduction'] > 1e-8 for pair in subset),
                               'median_reduction': math.exp2(median(pair['log2_reduction'] for pair in subset)),
                               'median_ordinary_over_truth': median(pair['ordinary_over_truth'] for pair in subset),
                               'median_mixed_over_truth': median(pair['mixed_over_truth'] for pair in subset)})
    return groups, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sql-timeout', type=float, default=90)
    args = parser.parse_args()
    if args.sql_timeout <= 0:
        parser.error('--sql-timeout must be positive')
    validation = json.loads((PROJECT / 'results/data_validation/imdb_job.json').read_text(encoding='utf-8'))
    if validation['status'] != 'passed':
        raise ValueError('IMDb validation has not passed')
    tables = {row['table']: row for row in validation['tables']}
    run_id = 'jobjoin_mixed_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = PROJECT / 'results/jobjoin_views' / run_id
    output.mkdir(parents=True)
    write_json(output / 'config.json', {'run_id': run_id, 'cases': CASES, 'kind_ids': KINDS,
                                        'sql_timeout': args.sql_timeout, 'duckdb': duckdb.__version__,
                                        'ortools': ortools.__version__, 'project': git_state(PROJECT),
                                        'data_identity': imdb_identity(tables),
                                        'duckdb_threads': 4, 'duckdb_memory': '4GB', 'lp_threads': 1,
                                        'lp_timeout_ms': 5000, 'seed': None, 'semantics': 'set'})
    print(f'Run directory: {output}', flush=True)
    reports = []
    with open_imdb(output, tables, 4, '4GB') as connection:
        for fact, dimension, columns in CASES:
            report = {'case': fact, 'status': 'failed', 'results': [], 'footprint': {}}
            reports.append(report)
            start = perf_counter()
            try:
                solve_case(connection, report, fact, dimension, columns, args.sql_timeout, 5000)
                report['status'] = 'passed'
            except duckdb.InterruptException as error:
                report.update(status='sql_timeout', error=str(error))
            except (duckdb.Error, ValueError) as error:
                report.update(error=str(error))
            report['wall_ms'] = (perf_counter() - start) * 1000
            write_json(output / f'{fact}.json', report)
            print(f'{fact}: {report["status"]} ({report["wall_ms"] / 1000:.1f}s)' +
                  (f': {report["error"]}' if 'error' in report else ''), flush=True)
    valid = [report for report in reports if report['status'] == 'passed']
    groups, pairs = summarize(valid)
    write_json(output / 'summary.json', {'run_id': run_id, 'statuses': dict(Counter(r['status'] for r in reports)),
                                        'groups': groups, 'pairs': pairs,
                                        'footprints': {r['case']: r['footprint'] for r in valid}})
    if len(valid) != len(reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
