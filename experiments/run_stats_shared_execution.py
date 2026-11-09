"""Select and execute plans over a reused STATS view in DuckDB."""

import argparse
from collections import Counter
from datetime import date
from fractions import Fraction
import math
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import duckdb
import ortools

from cqnorm.coverage_flow import FlowProgram
from cqnorm.numeric import certificate_cost, histogram_log_upper, log2_count_upper
from job_views import Atom, histogram_sql
from run_jobjoin_mixed import A, ATTRIBUTES, B
from run_jobjoin_views import ORDERS, degree, git_state, query, write_json
from run_stats_mixed import CASES, DIMENSION, TEST
from run_stats_reuse import window_stats
from run_stats_views import DATA, PROJECT, digest, load_data, run_directory


FORCED = ('window-first', 'category-first')
NATIVE = ('DuckDB-shared', 'DuckDB-inline')
PLANS = (*FORCED, *NATIVE)
INFER = ('Card+Max', 'Degree-profile', 'Flat-model', 'Flat-certificate')
METHODS = (*INFER, *NATIVE)
TOL = 1e-8  # log2, also used for deterministic near-ties in plan scoring
LP_MS = 5000
SQL_FAILURES = (duckdb.InterruptException, duckdb.OutOfMemoryException)


def elapsed(start):
    return 1000 * (perf_counter() - start)


def sql_status(error):
    return 'sql_timeout' if isinstance(error, duckdb.InterruptException) else 'out_of_memory'


def window_roster():
    for year in range(*TEST):
        for month in range(1, 13):
            lower = date(year, month, 1)
            upper = date(year + month // 12, month % 12 + 1, 1)
            for kind in (1, 2):
                yield {'from': str(lower), 'until': str(upper), 'post_type': kind}


def support_key(row):
    return row['from'], row['post_type']


def predicate(row):
    return (f"p.posttypeid = {row['post_type']} AND p.creationdate >= '{row['from']}' "
            f"AND p.creationdate < '{row['until']}'")


def producer_sql(atom, condition='p.posttypeid IN (1, 2)', categories=()):
    # Expose the join and filters together to DuckDB; deduplicate only the
    # surviving output, so the native inline baseline has no early DISTINCT.
    columns = {variable: f'r.{column}' for column, variable in atom.columns}
    projection = ', '.join(f'{column} AS {var}' for var, column in columns.items())
    guards = ' AND '.join(f'{column} IS NOT NULL' for column in columns.values())
    if categories:
        condition += f" AND {columns[B]} IN ({', '.join(map(str, categories))})"
    return (f'SELECT DISTINCT {projection} FROM {atom.table} r JOIN posts p ON p.id = {columns[A]} '
            f'WHERE {condition} AND {guards}')


def workload(db, atom, timeout):
    # Select labels using producer input alone, before observing any test-window
    # cardinalities or running any execution candidates. Keep all windows.
    labels_sql = f'SELECT {B} FROM ({atom.sql()}) GROUP BY {B} ORDER BY count(*) DESC, {B} LIMIT 4'
    ranked, _ = query(db, labels_sql, timeout)
    labels = [int(label) for label, in ranked]
    if len(labels) < 2:
        raise ValueError('Expected at least two producer input categories')
    variants = (('top', labels[:1]), ('second', labels[1:2]), ('top-four', labels))
    consumers = []
    for row in window_roster():
        for variant, subset in variants:
            p = predicate(row)
            b = f"r.{B} IN ({', '.join(map(str, subset))})"
            v = 'e2e_shared r'
            s = 'e2e_intermediate r'
            join = f'JOIN posts p ON p.id = r.{A}'
            consumers.append({**row, 'variant': variant, 'categories': subset, 'sql': {
                FORCED[0]: [f'SELECT r.* FROM {v} {join} WHERE {p}', f'SELECT count(*) FROM {s} WHERE {b}'],
                FORCED[1]: [f'SELECT r.* FROM {v} WHERE {b}', f'SELECT count(*) FROM {s} {join} WHERE {p}'],
                NATIVE[0]: [None, f'SELECT count(*) FROM {v} {join} WHERE {p} AND {b}'],
                NATIVE[1]: [None, f'SELECT count(*) FROM ({producer_sql(atom, p, subset)})'],
            }})
    return consumers, labels_sql


def inputs(db, source, timeout):
    start = perf_counter()
    projection = f'CREATE TEMP TABLE e2e_stats_fact AS {source}'
    rows, histograms = [], []
    try:
        _, projection_ms = query(db, projection, timeout)
        for k, key in enumerate((A, B)):
            sql = histogram_sql('SELECT * FROM e2e_stats_fact', (key,))
            histogram, ms = query(db, sql, timeout)
            histograms.append({'sql': sql, 'ms': ms, 'histogram': histogram})
            for q in ORDERS if k == 0 else ORDERS[1:]:
                rows.append((degree(ATTRIBUTES, (key,), q), histogram_log_upper(histogram, q)))
    finally:
        db.execute('DROP TABLE IF EXISTS e2e_stats_fact')
    dim_sql = f'SELECT count(*) FROM ({DIMENSION})'
    [(dim,)], dim_ms = query(db, dim_sql, timeout)
    counts, support_query = window_stats(db, TEST, granularity='month', timeout=timeout)
    counts = {support_key(row): row['base_rows'] for row in counts}
    rows.append((degree((A,)), log2_count_upper(dim)))
    return tuple(rows), counts, {'ms': elapsed(start), 'projection_sql': projection,
                                'projection_ms': projection_ms, 'histograms': histograms,
                                'dimension_sql': dim_sql, 'dimension_ms': dim_ms, 'support': support_query}


def estimate(base, support, consumers, method):
    start = perf_counter()
    profile_ms = 0
    target = degree(ATTRIBUTES)
    profile = []
    if method in INFER[:2]:
        targets = [target, *(degree(ATTRIBUTES, (key,), q) for key in (A, B)
                             for q in (ORDERS[1:] if method == INFER[1] else (math.inf,)))]
        for stat in targets:
            model = FlowProgram(ATTRIBUTES, tuple(s for s, _ in base), stat, time_limit_ms=LP_MS)
            result = model.solve(v for _, v in base)
            if result.status != 'optimal':
                return {'status': result.status, 'phase': 'profile', 'ms': elapsed(start)}
            profile.append((stat, result.log2_bound))
        profile_ms = elapsed(start)
    fixed = profile or base
    budgets = tuple(value for _, value in fixed)
    models = [FlowProgram(ATTRIBUTES, (*[s for s, _ in fixed], degree((key,))), target,
                          time_limit_ms=LP_MS) for key in (A, B)]
    model_ms = elapsed(start) - profile_ms
    certificates, memo = {}, {}
    scores, choices = [], []
    for i, consumer in enumerate(consumers):
        counts = (support.get(support_key(consumer), 0), len(consumer['categories']))
        row = []
        for j, count in enumerate(counts):
            if not count:
                row.append(None)  # zero rows, not a log budget to send to GLOP
                continue
            signature = (j, count)
            if signature not in memo:
                rhs = (*budgets, log2_count_upper(count))
                if j in certificates:
                    memo[signature] = certificate_cost(certificates[j][1], rhs)
                else:
                    result = models[j].solve(rhs)
                    if result.status != 'optimal':
                        return {'status': result.status, 'phase': 'consumer', 'consumer': i, 'ms': elapsed(start)}
                    memo[signature] = result.log2_bound
                    if method == 'Flat-certificate':
                        certificates[j] = (i, tuple(map(Fraction, result.input_weights)))
            row.append(memo[signature])
        scores.append(row)
        choices.append(FORCED[0 if row[0] is None or row[0] <= row[1] + TOL else 1])
    return {'status': 'passed', 'ms': elapsed(start), 'profile_ms': profile_ms, 'model_ms': model_ms,
            'scores': scores, 'choices': choices, 'unique_nonempty_budgets': len(memo),
            'certificates': [{'candidate': FORCED[j], 'seed_index': i, 'weights': list(map(str, weights))}
                             for j, (i, weights) in certificates.items()]}


def outcome(db, plan, consumer, expected, timeout):
    start = perf_counter()
    first, last = consumer['sql'][plan]
    rows = None
    build_ms = terminal_ms = None
    try:
        if first:
            [(rows,)], build_ms = query(db, f'CREATE TEMP TABLE e2e_intermediate AS {first}', timeout)
            if rows != expected[plan]:
                raise ValueError(f'{plan} CTAS wrote {rows} rows; expected {expected[plan]}: {consumer}')
        remaining = timeout - (perf_counter() - start)
        if remaining <= 0:
            raise duckdb.InterruptException('candidate deadline')
        [(count,)], terminal_ms = query(db, last, remaining)
        if count != expected['output']:
            raise ValueError(f'{plan} output count {count} != reference {expected["output"]}: {consumer}')
        status = 'passed'
    except SQL_FAILURES as error:
        status, count = sql_status(error), None
    finally:
        if first:
            db.execute('DROP TABLE IF EXISTS e2e_intermediate')
    return {'status': status, 'ms': elapsed(start), 'rows': rows, 'build_ms': build_ms,
            'terminal_ms': terminal_ms, 'output_rows': count}


def materialize(db, atom, timeout):
    start = perf_counter()
    try:
        [(count,)], _ = query(db, f'CREATE TEMP TABLE e2e_shared AS {producer_sql(atom)}', timeout)
    except SQL_FAILURES as error:
        db.execute('DROP TABLE IF EXISTS e2e_shared')
        return {'status': sql_status(error), 'ms': elapsed(start)}
    return {'status': 'passed', 'rows': count, 'ms': elapsed(start)}


def truth(db, source, consumers, timeout):
    first, last = f'{TEST[0]}-01-01', f'{TEST[1]}-01-01'
    # Equivalent grouped reference evaluated independently of all timed plans.
    sql = (f"SELECT date_trunc('month', p.creationdate)::DATE, p.posttypeid, r.{B}, count(*) "
           f'FROM ({source}) r JOIN posts p ON p.id = r.{A} WHERE p.posttypeid IN (1, 2) '
           f"AND p.creationdate >= '{first}' AND p.creationdate < '{last}' GROUP BY 1, 2, 3")
    rows, _ = query(db, sql, timeout)
    counts = {(str(month), kind, label): n for month, kind, label, n in rows}
    windows = Counter()
    for (month, kind, _), n in counts.items():
        windows[month, kind] += n
    category_sql = (f'SELECT r.{B}, count(*) FROM ({source}) r JOIN posts p ON p.id = r.{A} '
                    'WHERE p.posttypeid IN (1, 2) GROUP BY 1')
    category_totals = dict(query(db, category_sql, timeout)[0])
    return [{'output': sum(counts.get((*support_key(row), label), 0) for label in row['categories']),
             FORCED[0]: windows[support_key(row)], FORCED[1]: sum(category_totals.get(label, 0) for label in row['categories'])}
            for row in consumers], {'window_category': sql, 'all_categories': category_sql}


def characterize(db, atom, consumers, expected, args):
    timeout = args.sql_timeout
    sample, rounds = {}, []
    rng = Random(args.seed)
    for iteration in range(args.warmups + args.repeats):
        shared = materialize(db, atom, timeout)
        if not sample and shared['status'] == 'passed':
            if db.execute('SELECT count(*) FROM e2e_shared').fetchone()[0] != shared['rows']:
                raise ValueError('CTAS row count differs from materialized rows')
            for plan, (first, last) in consumers[0]['sql'].items():
                sample[plan] = db.execute('EXPLAIN ' + (f'CREATE TEMP TABLE e2e_intermediate AS {first}' if first else last)).fetchone()[1]
        runs = [{} for _ in consumers]
        indexes = list(range(len(consumers)))
        rng.shuffle(indexes)
        for i in indexes:
            order = list(PLANS)
            rng.shuffle(order)
            for plan in order:
                runs[i][plan] = (outcome(db, plan, consumers[i], expected[i], timeout)
                                 if plan == NATIVE[1] or shared['status'] == 'passed' else
                                 {'status': 'blocked_by_shared', 'rows': None})
        if shared['status'] == 'passed':
            db.execute('DROP TABLE e2e_shared')
        rounds.append({'warmup': iteration < args.warmups, 'shared': shared, 'order': indexes, 'runs': runs})
    return rounds, sample


def batches(db, atom, consumers, expected, args):
    rng = Random(args.seed + 1)
    rounds = []
    for iteration in range(args.warmups + args.repeats):
        indexes = list(range(len(consumers)))
        rng.shuffle(indexes)
        methods = list(METHODS)
        rng.shuffle(methods)
        by_method = {}
        for method in methods:
            start = perf_counter()
            acquisition = scoring = shared = None
            if method in INFER:
                try:
                    base, support, acquisition = inputs(db, atom.sql(), args.sql_timeout)
                except SQL_FAILURES as error:
                    by_method[method] = {'status': sql_status(error), 'phase': 'acquisition', 'ms': elapsed(start)}
                    continue
                scoring = estimate(base, support, consumers, method)
                if scoring['status'] != 'passed':
                    by_method[method] = {'status': scoring['status'], 'phase': 'estimate', 'ms': elapsed(start),
                                         'acquisition': acquisition, 'estimate': scoring}
                    continue
                choices = scoring['choices']
            else:
                choices = [method] * len(consumers)
            if method != NATIVE[1]:
                shared = materialize(db, atom, args.sql_timeout)
                if shared['status'] != 'passed':
                    by_method[method] = {'status': shared['status'], 'phase': 'materialization', 'ms': elapsed(start),
                                         'acquisition': acquisition, 'estimate': scoring, 'shared': shared}
                    continue
            runs = [None] * len(consumers)
            for i in indexes:
                runs[i] = outcome(db, choices[i], consumers[i], expected[i], args.sql_timeout)
            if shared:
                db.execute('DROP TABLE e2e_shared')
            by_method[method] = {'status': 'passed' if all(r['status'] == 'passed' for r in runs) else 'incomplete',
                                 'ms': elapsed(start), 'acquisition': acquisition,
                                 'estimate': scoring, 'shared': shared, 'runs': runs}
        rounds.append({'warmup': iteration < args.warmups, 'order': indexes, 'methods': by_method})
    return rounds


def summarize(consumers, expected, reference, batches):
    timed = [r for r in reference if not r['warmup']]
    paired = [r['methods'] for r in batches if not r['warmup']]
    oracle, measurements = [], []
    for i, row in enumerate(consumers):
        times = {}
        for plan in PLANS:
            samples = [r['runs'][i][plan] for r in timed]
            if all(x['status'] == 'passed' for x in samples):
                times[plan] = median(x['ms'] for x in samples)
        best = min(FORCED, key=lambda plan: times[plan]) if all(p in times for p in FORCED) else None
        oracle.append(best)
        counts = {p: expected[i][p] for p in FORCED}
        measurements.append({'consumer': i, **{k: v for k, v in row.items() if k != 'sql'},
                             'output_rows': expected[i]['output'], 'candidate_median_ms': times, 'intermediate_rows': counts,
                             'fastest_forced': best, 'exact_row_choice': min(FORCED, key=counts.get)})
    summary = {}
    card = next((r[INFER[0]]['estimate']['choices'] for r in paired if r[INFER[0]]['status'] == 'passed'), None)
    for method in METHODS:
        successful = [r[method] for r in paired if r[method]['status'] == 'passed']
        ms = [r['ms'] for r in successful]
        first = successful[0] if successful else None
        ratios, comparison = [], []
        changed = None
        if first:
            choices = first['estimate']['choices'] if method in INFER else [method] * len(consumers)
            if method in INFER:
                if any(r['estimate']['choices'] != choices for r in successful):
                    raise ValueError(f'{method} choices changed across timing repeats')
                changed = sum(a != b for a, b in zip(card, choices, strict=True)) if card is not None else None
                for row, logs in zip(measurements, first['estimate']['scores'], strict=True):
                    row.setdefault('log2_scores', {})[method] = logs
                    for plan, upper in zip(FORCED, logs, strict=True):
                        n = row['intermediate_rows'].get(plan)
                        if n and (upper is None or upper + TOL < math.log2(n)):
                            raise ValueError(f'Bound below intermediate count: {method}, {plan}, {row}')
            for row, chosen, best in zip(measurements, choices, oracle, strict=True):
                times = row['candidate_median_ms']
                row.setdefault('choices', {})[method] = chosen
                if best and chosen in times:
                    ratios.append(times[chosen] / times[best])
                    comparison.append((row['output_rows'] > 0, ratios[-1]))
        nonempty_ratios = [r for nonempty, r in comparison if nonempty]
        statuses = Counter(run['status'] for r in paired for run in r[method].get('runs', []))
        batch_ratios = [r[method]['ms'] / r[INFER[0]]['ms'] for r in paired
                        if r[method]['status'] == r[INFER[0]]['status'] == 'passed']
        summary[method] = {'batch_statuses': dict(Counter(r[method]['status'] for r in paired)),
                           'selected_sql_statuses': dict(statuses), 'median_total_ms': median(ms) if ms else None,
                           'total_samples_ms': ms, 'paired_batch_ratio_vs_card': batch_ratios, 'changes_vs_card': changed,
                           'median_vs_fastest_forced': median(ratios) if ratios else None,
                           'median_nonempty_vs_fastest_forced': median(nonempty_ratios) if nonempty_ratios else None,
                           'worst_vs_fastest_forced': max(ratios, default=None),
                           'median_estimate_ms': median(r['estimate']['ms'] for r in successful) if first and first['estimate'] else None,
                           'median_statistics_ms': median(r['acquisition']['ms'] for r in successful) if first and first['acquisition'] else None,
                           'median_shared_ms': median(r['shared']['ms'] for r in successful) if first and first['shared'] else None,
                           'median_selected_ms': median(sum(q['ms'] for q in r['runs']) for r in successful) if first else None}
    spreads = [max(row['candidate_median_ms'][p] for p in FORCED) / row['candidate_median_ms'][best]
               for row, best in zip(measurements, oracle, strict=True) if best]
    return {'consumers': len(consumers), 'empty_output': sum(not row['output'] for row in expected), 'methods': summary,
            'reference_materialization_statuses': dict(Counter(r['shared']['status'] for r in timed)),
            'candidate_statuses': {p: dict(Counter(row[p]['status'] for r in timed for row in r['runs'])) for p in PLANS},
            'candidate_spread': {'paired': len(spreads), 'at_least_1_25': sum(x >= 1.25 for x in spreads),
                                 'median': median(spreads) if spreads else None, 'max': max(spreads, default=None)},
            'exact_row_agreement': sum(row['exact_row_choice'] == best for row, best in zip(measurements, oracle, strict=True) if best)}, measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--seed', type=int, default=157)
    parser.add_argument('--sql-timeout', type=float, default=20)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 1 or args.sql_timeout <= 0:
        parser.error('Expected positive repeats, warm-ups and SQL timeout')
    output = run_directory('stats_shared_execution')
    print(f'Run directory: {output}', flush=True)
    cases = {}
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB', 'temp_directory': str(output / 'spill')}) as db:
        data = load_data(db, args.data)
        write_json(output / 'config.json', {'data': data, 'project': git_state(PROJECT), 'runner_sha256': digest(Path(__file__)),
                                            'duckdb': duckdb.__version__, 'ortools': ortools.__version__, 'duckdb_threads': 4,
                                            'repeats': args.repeats, 'warmups': args.warmups, 'seed': args.seed,
                                            'sql_timeout': args.sql_timeout, 'lp_limit_ms': LP_MS, 'years': TEST,
                                            'producers': CASES, 'plans': PLANS, 'methods': METHODS,
                                            'workload_rule': 'every calendar month/type × top, second, top-four source labels',
                                            'score': 'minimize estimated CTAS rows, window-first at a log2 near-tie'})
        for name, columns in CASES:
            atom = Atom(name, 'r', tuple(zip(columns, ATTRIBUTES, strict=True)))
            source = atom.sql()
            consumers, labels_sql = workload(db, atom, args.sql_timeout)
            expected, truth_sql = truth(db, source, consumers, args.sql_timeout)
            reference, explain = characterize(db, atom, consumers, expected, args)
            print(f'{name}: all candidates characterized, measuring complete batches', flush=True)
            workflows = batches(db, atom, consumers, expected, args)
            cases[name], measurements = summarize(consumers, expected, reference, workflows)
            write_json(output / f'{name}.json', {'summary': cases[name], 'consumers': consumers, 'measured': measurements,
                                                  'sql': {'source': source, 'producer': producer_sql(atom), 'labels': labels_sql,
                                                          'truth': truth_sql, 'explain': explain},
                                                  'reference': reference, 'batches': workflows})
            write_json(output / 'summary.json', {'cases': cases})
            print(f'{name}: {len(consumers)} consumers; {cases[name]["candidate_spread"]}', flush=True)


if __name__ == '__main__':
    main()
