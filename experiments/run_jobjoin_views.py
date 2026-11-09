"""Run E1/E2 on deterministic, one-boundary views derived from all JOBjoin queries."""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from threading import Timer
from time import perf_counter

import duckdb
import ortools

from job_views import (check_split_sql, degree_log, derive, histogram_sql, outer_cardinality_sql,
                       parse_jobjoin, project_join, scope)
from run_projected_path import PROJECT, git_state, norm_json
from cqnorm import coverage_flow, shannon
from cqnorm.numeric import histogram_log_upper
from cqnorm.shannon import MixedNorm

ORDERS = (1, 2, 4, 8, math.inf)
PROFILES = {"Card-only": (1,), "Card+Max": (1, math.inf), "Degree-profile": ORDERS}
PROJECTED_METHOD = 'Degree+Projected-base'
PROJECTED_GLOBAL = 'Global+Projected-base'


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def query(connection, sql, timeout):
    start = perf_counter()
    timer = Timer(timeout, connection.interrupt)
    timer.start()
    try:
        return connection.execute(sql).fetchall(), (perf_counter() - start) * 1000
    finally:
        timer.cancel()
        timer.join()


def padded(log_value):
    return math.nextafter(log_value + 1e-12, math.inf) if log_value > 0 else log_value


def degree(head, keys=(), q=1):
    return MixedNorm.degree(tuple(head), keys if q != 1 else (), q)


def physical_groups(atom):
    groups = defaultdict(list)
    for column, variable in atom.columns:
        groups[variable].append(column)
    return {tuple(sorted(columns)): variable for variable, columns in groups.items()}


def base_budgets(connection, atoms, cache, timeout):
    budgets = {}
    for atom in atoms:
        head = tuple(sorted(atom.scope))
        groups = physical_groups(atom)
        for group, key in sorted(groups.items()):
            signature = (atom.table, tuple(sorted(groups)), group)
            if signature not in cache:
                sql = histogram_sql(atom.sql(), (key,))
                histogram, elapsed = query(connection, sql, timeout)
                cache[signature] = {"table": atom.table, "column_groups": signature[1], "key": group,
                                    "sql": sql, "extract_ms": elapsed,
                                    "histogram": histogram}
        entries = []
        for index, (group, key) in enumerate(sorted(groups.items())):
            histogram = cache[(atom.table, tuple(sorted(groups)), group)]["histogram"]
            for q in ORDERS if index == 0 else ORDERS[1:]:
                if len(head) == 1 and q != 1:
                    # Norms on a full singleton key still provide support size via q=1.
                    continue
                entries.append((degree(head, (key,), q), histogram_log_upper(histogram, q)))
        budgets[atom.alias] = entries
    return budgets


def projected_base_budgets(connection, split, cache, timeout):
    """Measure only the needed base projections, preserving original atom guards."""
    result = []
    for atom in split.left:
        head = tuple(sorted(atom.scope & set(split.left_head)))
        if not head or set(head) == atom.scope:
            continue
        key = tuple(variable for variable in split.keys if variable in head)
        if key != split.keys or len(head) == len(key):
            key = ()
        sql = histogram_sql(atom.sql(output=head), key)
        groups = {variable: columns for columns, variable in physical_groups(atom).items()}
        signature = (atom.table, tuple(sorted(groups.values())), tuple(sorted(groups[v] for v in head)),
                     tuple(sorted(groups[v] for v in key)))
        if signature not in cache:
            histogram, elapsed = query(connection, sql, timeout)
            cache[signature] = {'table': atom.table, 'projection': signature[2], 'key': signature[3],
                                'column_groups': signature[1], 'sql': sql, 'extract_ms': elapsed,
                                'histogram': histogram}
        histogram = cache[signature]['histogram']
        result.extend((degree(head, key, q), histogram_log_upper(histogram, q)) for q in (ORDERS if key else (1,)))
    return result


def estimate(rows, stage, variant, method, attributes, budgets, target, lp_timeout, *, shannon_max_attributes=8):
    start = perf_counter()
    bound = coverage_flow.solve(attributes, budgets, target, method='flow', time_limit_ms=lp_timeout)
    backend = 'flow'
    if bound.status == 'incompatible' and len(attributes) <= shannon_max_attributes:
        bound = shannon.solve(attributes, budgets, target, lp_timeout)
        backend = 'shannon_fallback'
    budget_rows = [{**norm_json(stat), 'log2_budget': '-inf' if value == -math.inf else value}
                   for stat, value in budgets]
    row = {'stage': stage, 'variant': variant, 'method': method, 'formulation': backend,
           'attributes': list(attributes), 'target': norm_json(target),
           'input_statistics_sha256': hashlib.sha256(json.dumps(budget_rows).encode()).hexdigest(),
           'input_statistics': budget_rows, 'total_ms': (perf_counter() - start) * 1000, **asdict(bound)}
    rows.append(row)
    if bound.status == 'empty':
        return target, -math.inf
    if bound.status == 'optimal':
        return target, bound.log2_bound
    return None


def compare_truth(row, truth):
    row['log2_truth'] = truth if truth != -math.inf else None
    row['truth_empty'] = truth == -math.inf
    if row['status'] == 'empty':
        row['safety_violation'] = truth != -math.inf
    elif row['status'] == 'optimal' and truth != -math.inf:
        gap = row['log2_bound'] - truth
        row.update(log2_gap=gap, bound_over_truth=math.exp2(gap), safety_violation=gap < -1e-8)


def run_query(connection, split, report, cache, projected_cache, args):
    before = len(cache)
    stats_start = perf_counter()
    budgets = base_budgets(connection, split.atoms, cache, args.sql_timeout)
    report.update(new_statistics_groups=len(cache) - before,
                  base_statistics_ms=(perf_counter() - stats_start) * 1000)
    left = [entry for atom in split.left for entry in budgets[atom.alias]]
    right = [entry for atom in split.right for entry in budgets[atom.alias]]
    left_attributes, right_attributes = (scope(side) for side in (split.left, split.right))
    target = degree(split.head)
    rows = report['results']
    timeout = args.lp_time_limit_ms
    estimate(rows, 'E2', 'flat', 'Global-General', sorted(left_attributes | right_attributes),
             left + right, target, timeout)
    right_sql = project_join(split.right, split.right_head)
    views = {'projected': split.left_head, 'retain_witnesses': tuple(sorted(left_attributes))}
    report['truth_sql'] = {'right': right_sql}
    # Complete online estimates before materializing any derived truth.
    for variant, head in views.items():
        profile = {}
        costs = {}
        for q in ORDERS:
            profile[q] = estimate(rows, 'E1', variant, 'Flow-General', sorted(left_attributes),
                                  left, degree(head, split.keys, q), timeout)
            costs[q] = rows[-1]['total_ms']
        attributes = sorted(set(head) | right_attributes)
        for method, orders in PROFILES.items():
            if any(profile[q] is None for q in orders):
                rows.append({'stage': 'E2', 'variant': variant, 'method': method, 'status': 'profile_unavailable'})
                continue
            estimate(rows, 'E2', variant, method, attributes, [profile[q] for q in orders] + right, target, timeout)
            rows[-1]['profile_ms'] = sum(costs[q] for q in orders)
            rows[-1]['summary_bytes'] = len(json.dumps([(norm_json(stat), '-inf' if value == -math.inf else value)
                                                       for stat, value in (profile[q] for q in orders)]).encode())

    if args.projected_base:
        extra_start = perf_counter()
        extra = projected_base_budgets(connection, split, projected_cache, args.sql_timeout)
        report['projected_base_statistics_ms'] = (perf_counter() - extra_start) * 1000
        report['projected_base_rows'] = len(extra)
        report['projected_base_summary_bytes'] = len(json.dumps([(norm_json(stat), '-inf' if value == -math.inf else value)
                                                                 for stat, value in extra]).encode())
        estimate(rows, 'E2', 'flat', PROJECTED_GLOBAL, sorted(left_attributes | right_attributes),
                 left + extra + right, target, timeout)
        profile = []
        profile_ms = 0
        for q in ORDERS:
            profile.append(estimate(rows, 'E1', 'projected', 'Flow+Projected-base', sorted(left_attributes),
                                    left + extra, degree(split.left_head, split.keys, q), timeout))
            profile_ms += rows[-1]['total_ms']
        if any(stat is None for stat in profile):
            rows.append({'stage': 'E2', 'variant': 'projected', 'method': PROJECTED_METHOD, 'status': 'profile_unavailable'})
        else:
            estimate(rows, 'E2', 'projected', PROJECTED_METHOD, sorted(set(split.left_head) | right_attributes),
                     profile + right, target, timeout)
            rows[-1]['profile_ms'] = profile_ms

    truths = report['truth']
    for variant, head in views.items():
        sql = project_join(split.left, head)
        report['truth_sql'][variant] = sql
        histogram, elapsed = query(connection, histogram_sql(sql, split.keys), args.sql_timeout)
        variant_truth = {q: degree_log(histogram, q) for q in ORDERS}
        check_sql = check_split_sql(split, report['source_sql'], variant == 'retain_witnesses')
        check, check_ms = query(connection, check_sql, args.sql_timeout)
        sample_rows, sample_difference = check[0]
        truths[variant] = {'view_rows': sum(degree * count for degree, count in histogram),
                           'view_truth_ms': elapsed, 'sample_difference_rows': sample_difference,
                           'sample_result_rows': sample_rows,
                           'sample_check_ms': check_ms, 'sample_check_sql': check_sql}
        if sample_difference:
            raise ValueError(f'Sampled flat and nested queries disagree for {variant}')
        for row in rows:
            if row['stage'] == 'E1' and row['variant'] == variant:
                q = float(row['target']['exponents'][0])
                compare_truth(row, variant_truth[q])
        # Diagnostic: only this method receives measured view statistics.
        exact = [(degree(head, split.keys, q), padded(variant_truth[q])) for q in ORDERS]
        estimate(rows, 'E2', variant, 'Exact-profile', sorted(set(head) | right_attributes),
                 exact + right, target, timeout)

    outer_sql = outer_cardinality_sql(report['truth_sql']['projected'], right_sql, split.keys)
    cardinality, elapsed = query(connection, outer_sql, args.sql_timeout)
    count = cardinality[0][0]
    truths['outer'] = {'rows': count, 'truth_ms': elapsed, 'sql': outer_sql}
    truth = math.log2(count) if count else -math.inf
    global_row = next(row for row in rows if row['method'] == 'Global-General')
    extra_global_row = next((row for row in rows if row['method'] == PROJECTED_GLOBAL), global_row)
    for row in rows:
        if row['stage'] != 'E2':
            continue
        compare_truth(row, truth)
        reference = extra_global_row if row['method'] in (PROJECTED_METHOD, PROJECTED_GLOBAL) else global_row
        if row['status'] == reference['status'] == 'optimal':
            gap = row['log2_bound'] - reference['log2_bound']
            row['bound_over_global'] = math.exp2(gap)
            row['below_global'] = (row['method'] in PROFILES or row['method'] == PROJECTED_METHOD) and gap < -1e-8

    if any(row.get('safety_violation') or row.get('below_global') for row in rows):
        raise ValueError('Upper-bound or compositional consistency check failed')


def summarize(reports):
    groups = defaultdict(list)
    for report in reports:
        for row in report['results']:
            exponent = row.get('target', {}).get('exponents', [None])[0] if row['stage'] == 'E1' else None
            groups[row['stage'], row['variant'], row['method'], exponent].append(row)

    def distribution(values):
        if not values:
            return None
        ordered = sorted(values)
        return {'n': len(ordered), 'median': statistics.median(ordered),
                'p90': ordered[math.ceil(0.90 * len(ordered)) - 1],
                'p95': ordered[math.ceil(0.95 * len(ordered)) - 1], 'max': ordered[-1]}

    summary = []
    for (stage, variant, method, q), rows in groups.items():
        summary.append({'stage': stage, 'variant': variant, 'method': method, 'q': q,
                        'statuses': dict(Counter(row['status'] for row in rows)),
                        'empty_truth': sum(row.get('truth_empty', False) for row in rows),
                        'safety_violations': sum(row.get('safety_violation', False) for row in rows),
                        'bound_over_truth': distribution([row['bound_over_truth'] for row in rows
                                                          if 'bound_over_truth' in row]),
                        'bound_over_global': distribution([row['bound_over_global'] for row in rows
                                                           if 'bound_over_global' in row]),
                        'estimation_ms': distribution([row['total_ms'] + row.get('profile_ms', 0)
                                                       for row in rows if 'total_ms' in row])})
    return summary


def paired_improvement(reports):
    groups = defaultdict(list)
    for report in reports:
        variants = defaultdict(dict)
        for row in report['results']:
            if row['stage'] == 'E2' and row['status'] == 'optimal':
                variants[row['variant']][row['method']] = row['log2_bound']
        for variant, bounds in variants.items():
            if 'Card+Max' in bounds and 'Degree-profile' in bounds:
                groups[variant].append(bounds['Card+Max'] - bounds['Degree-profile'])
    return {variant: {'n': len(gaps), 'strictly_tighter': sum(gap > 1e-8 for gap in gaps),
                      'strictly_looser': sum(gap < -1e-8 for gap in gaps),
                      'median_reduction_factor': math.exp2(statistics.median(gaps)),
                      'max_reduction_factor': math.exp2(max(gaps))}
            for variant, gaps in groups.items()}


@contextmanager
def open_imdb(output, tables, threads, memory):
    data = PROJECT / 'data/imdb_job/data'
    with duckdb.connect(config={'threads': threads, 'memory_limit': memory,
                                'temp_directory': str(output / 'spill')}) as connection:
        for name, table in tables.items():
            path = data / (name + '.parquet')
            if path.stat().st_size != table['bytes']:
                raise ValueError(f'Parquet size changed since validation: {path}')
            path_sql = path.as_posix().replace("'", "''")
            connection.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path_sql}')")
        yield connection


def load_histograms(source):
    histograms = json.loads((source / 'base_statistics.json').read_text(encoding='utf-8'))
    return {(entry['table'], tuple(map(tuple, entry['column_groups'])), tuple(entry['key'])): entry
            for entry in histograms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-ids', nargs='+', type=int)
    parser.add_argument('--view-atoms', type=int, default=2)
    parser.add_argument('--categorical-payload', action='store_true')
    parser.add_argument('--projected-base', action='store_true')
    parser.add_argument('--reuse-statistics-from', type=Path)
    parser.add_argument('--sql-timeout', type=float, default=60)
    parser.add_argument('--lp-time-limit-ms', type=int, default=5000)
    parser.add_argument('--duckdb-threads', type=int, default=4)
    parser.add_argument('--duckdb-memory', default='4GB')
    args = parser.parse_args()
    if args.sql_timeout <= 0 or args.lp_time_limit_ms <= 0 or args.duckdb_threads <= 0:
        parser.error('Timeouts and thread count must be positive')
    if args.view_atoms < 2:
        parser.error('--view-atoms must be at least 2')

    baseline = PROJECT.parent / 'LpBound'
    source = baseline / 'benchmarks/workloads/jobjoin/jobjoinQueries.sql'
    source_bytes = source.read_bytes()
    queries = source_bytes.decode().splitlines()
    query_ids = sorted(set(args.query_ids or range(1, len(queries) + 1)))
    if not set(query_ids) <= set(range(1, len(queries) + 1)):
        parser.error('Query IDs must be valid 1-based source line numbers')
    validation_path = PROJECT / 'results/data_validation/imdb_job.json'
    validation = json.loads(validation_path.read_text(encoding='utf-8'))
    if validation['status'] != 'passed':
        raise ValueError('IMDb validation has not passed')
    tables = {table['table']: table for table in validation['tables']}
    sizes = {name: table['rows'] for name, table in tables.items()}
    categorical = None
    if args.categorical_payload:
        metadata = json.loads((PROJECT / 'data/imdb_job/metadata/job.json').read_text(encoding='utf-8'))
        categorical = {name: sorted(column['name'] for column in table['schema']
                                    if column['type'] == 'INTEGER' and not column['nullable'])
                       for name, table in metadata['tables'].items()}
    cache = {}
    if args.reuse_statistics_from:
        statistics_source = args.reuse_statistics_from.resolve()
        original = json.loads((statistics_source / 'config.json').read_text(encoding='utf-8'))
        if original['tables'] != tables or original['base_stats'] != 'exact projected singleton degree profiles':
            raise ValueError('Cached statistics use a different data snapshot or statistics protocol')
        cache = load_histograms(statistics_source)

    run_id = 'jobjoin_views_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = PROJECT / 'results' / 'jobjoin_views' / run_id
    output.mkdir(parents=True)
    run_args = {**vars(args), 'reuse_statistics_from': str(args.reuse_statistics_from.resolve()) if args.reuse_statistics_from else None}
    config = {'run_id': run_id, 'args': run_args, 'query_ids': query_ids, 'seed': None,
              'source': str(source), 'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
              'data_validation': str(validation_path), 'tables': tables,
              'project': git_state(PROJECT), 'baseline': git_state(baseline),
              'python': sys.version, 'duckdb': duckdb.__version__, 'ortools': ortools.__version__,
              'lp_threads': 1, 'semantics': 'set', 'view_depth': 1, 'base_stats': 'exact projected singleton degree profiles',
              'categorical_columns': categorical,
              'sample_equivalence': 'symmetric difference on first boundary variable in [1,64]',
              'outer_truth': 'exact dot product of both projected sides grouped by retained boundary key'}
    write_json(output / 'config.json', config)
    print(f'Run directory: {output}', flush=True)
    reports = []
    projected_cache = {}
    with open_imdb(output, tables, args.duckdb_threads, args.duckdb_memory) as connection:
        for query_id in query_ids:
            sql = queries[query_id - 1]
            report = {'query_id': query_id, 'source_sql': sql, 'results': [], 'truth': {}}
            reports.append(report)
            start = perf_counter()
            try:
                split = derive(parse_jobjoin(sql), sizes, args.view_atoms, categorical)
                if split is None:
                    report['status'] = 'no_eligible_cut'
                else:
                    report.update(split=asdict(split), self_join=len({atom.table for atom in split.atoms}) < len(split.atoms))
                    run_query(connection, split, report, cache, projected_cache, args)
                    report['status'] = 'passed'
            except duckdb.InterruptException as error:
                report.update(status='sql_timeout', error=str(error))
            except (duckdb.Error, ValueError) as error:
                report.update(status='failed', error=str(error))
            report['wall_ms'] = (perf_counter() - start) * 1000
            write_json(output / f'query_{query_id:02d}.json', report)
            print(f'{query_id:02d}: {report["status"]} ({report["wall_ms"] / 1000:.1f}s)', flush=True)

    write_json(output / 'base_statistics.json', list(cache.values()))
    if args.projected_base:
        write_json(output / 'projected_base_statistics.json', list(projected_cache.values()))
    passed = [report for report in reports if report['status'] == 'passed']
    collapsed = [report for report in passed if report['truth']['projected']['view_rows'] < report['truth']['retain_witnesses']['view_rows']]
    all_rows = [row for report in reports for row in report['results']]
    write_json(output / 'summary.json', {'run_id': run_id, 'queries': dict(Counter(report['status'] for report in reports)),
                                        'status_by_query': {r['query_id']: r['status'] for r in reports},
                                        'projection_reduced_rows': [report['query_id'] for report in collapsed],
                                        'all_solver_statuses': dict(Counter(row['status'] for row in all_rows)),
                                        'safety_violations': sum(row.get('safety_violation', False) for row in all_rows),
                                        'below_global': sum(row.get('below_global', False) for row in all_rows),
                                        'groups': summarize(passed), 'paired_improvement': paired_improvement(passed),
                                        'collapsed_view_groups': summarize(collapsed)})
    if any(report['status'] == 'failed' for report in reports) or not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
