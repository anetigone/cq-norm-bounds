"""Separate mixed from ordinary degree summaries across a projected join."""

from datetime import datetime, timedelta, timezone
import hashlib
from itertools import combinations
import math
from pathlib import Path
import sys

import duckdb
import ortools

from job_views import degree_log, histogram_sql
from run_jobjoin_boundary import encoded_size
from run_jobjoin_views import ORDERS, PROJECT, degree, git_state, padded, write_json
from run_projected_path import log_norm, measure, norm_json
from cqnorm.shannon import MixedNorm


MATRICES = {'larger': ((4, 1, 1), (3, 2, 5)), 'smaller': ((3, 2, 1), (4, 1, 5))}
R_SQL = 'SELECT * FROM r'
S_SQL = 'SELECT * FROM s'
T_SQL = 'SELECT * FROM t'
JOIN_SQL = 'SELECT a, b, x, y, d FROM r JOIN s USING (a, b)'
VIEW_SQL = f'SELECT DISTINCT a, b, x, y FROM ({JOIN_SQL})'
QUERY_SQL = f'SELECT a, b, x, y, e FROM ({VIEW_SQL}) JOIN t USING (y)'
FLAT_SQL = f'SELECT DISTINCT a, b, x, y, e FROM ({JOIN_SQL}) JOIN t USING (y)'
BASE_SOURCES = {'r': (R_SQL, 'abx'), 's': (S_SQL, 'abyd'), 't': (T_SQL, 'ye')}
TOLERANCE = 1e-8


def mixed(payload, q):
    return MixedNorm((('a',), ('b',), tuple(payload)), (q, math.inf, 1))


def subsets(attributes):
    return [part for size in range(len(attributes) + 1) for part in combinations(attributes, size)]


def scopes(attributes):
    for projection in subsets(attributes):
        for key in subsets(projection):
            yield projection, key


def ordinary_histograms(connection, sql, attributes):
    """Exact degree multisets for every key inside every projection."""
    rows = []
    for projection, key in scopes(attributes):
        columns = ', '.join(projection) or '1 AS unit'
        projected = f'SELECT DISTINCT {columns} FROM ({sql})'
        histogram = sorted(connection.execute(histogram_sql(projected, key)).fetchall())
        rows.append({'projection': projection, 'key': key, 'histogram': histogram})
    return rows


def orders(projection, key):
    if not projection or key == projection:
        return ()
    return ORDERS[1:] if key else (1,)


def ordinary_budgets(histograms):
    rows = []
    for row in histograms:
        projection, key, histogram = row['projection'], row['key'], row['histogram']
        rows.extend((degree(projection, key, q), padded(degree_log(histogram, q)))
                    for q in orders(projection, key))
    return rows


def setup(connection, matrix):
    r = [(a, b) for a, row in enumerate(matrix) for b, n in enumerate(row) for _ in range(n)]
    connection.execute('CREATE OR REPLACE TABLE r(a INTEGER, b INTEGER, x INTEGER)')
    connection.executemany('INSERT INTO r VALUES (?, ?, ?)', [(a, b, x) for x, (a, b) in enumerate(r)])
    connection.execute('CREATE OR REPLACE TABLE s AS SELECT a, b, 2 * (3 * a + b) + j AS y, d '
                       'FROM range(2) aa(a), range(3) bb(b), range(2) yy(j), range(3) dd(d)')
    connection.execute('CREATE OR REPLACE TABLE t AS SELECT y, e FROM (SELECT DISTINCT y FROM s), range(2) ee(e)')


def trial(connection, name, matrix):
    setup(connection, matrix)
    histograms = {table: ordinary_histograms(connection, sql, attributes)
                  for table, (sql, attributes) in BASE_SOURCES.items()}
    base = {table: ordinary_budgets(rows) for table, rows in histograms.items()}
    mixed_inputs = [(stat, padded(log_norm(connection, sql, stat)))
                    for stat, sql in [(mixed('x', 4), R_SQL), (mixed('y', 4), S_SQL)]]
    view_goal, goal = mixed('xy', 2), mixed('xye', 2)
    rows = []
    inferred_mixed = measure(rows, name, 'Producer-Mixed-Flow', 'abxyd', mixed_inputs, view_goal, 'flow')
    reference = measure(rows, name, 'Producer-Mixed-Shannon', 'abxyd', mixed_inputs, view_goal, 'shannon')
    if abs(inferred_mixed - reference) > TOLERANCE:
        raise RuntimeError('Flow and Shannon disagree on the mixed producer certificate')

    producer = base['r'] + base['s'] + mixed_inputs
    inferred_ordinary = []
    targets = [degree(projection, key, q) for projection, key in scopes('abxy') for q in orders(projection, key)]
    for stat in targets:
        result = measure(rows, name, 'Producer-Ordinary', 'abxyd', producer, stat, 'shannon')
        # Keep one copy of the common input rows and retain each LP's certificate.
        if inferred_ordinary:
            del rows[-1]['input_statistics']
        inferred_ordinary.append((stat, result))
    mixed_row = (view_goal, inferred_mixed)
    cardinality = next(row for row in inferred_ordinary if row[0] == degree('abxy'))
    maximum = next(row for row in inferred_ordinary if row[0] == degree('abxy', ('a', 'b'), math.inf))
    profiles = {'Card+Max': [cardinality, maximum], 'Card+Mixed': [cardinality, mixed_row],
                'Inferred-ordinary': inferred_ordinary, 'Inferred-ordinary+Mixed': [*inferred_ordinary, mixed_row]}
    downstream = {}
    for method, profile in profiles.items():
        value = measure(rows, name, method, 'abxye', profile + base['t'], goal, 'shannon')
        downstream[method] = {'log2_bound': value, 'profile_items': len(profile), 'profile_bytes': encoded_size(profile)}
    for method, inputs in [('Global-ordinary', base['r'] + base['s']), ('Global+Mixed', producer)]:
        value = measure(rows, name, method, 'abxyde', inputs + base['t'], goal, 'shannon')
        downstream[method] = {'log2_bound': value}

    histograms['view'] = ordinary_histograms(connection, VIEW_SQL, 'abxy')
    exact_ordinary = ordinary_budgets(histograms['view'])
    for method, profile in [('Exact-ordinary', exact_ordinary),
                            ('Exact-ordinary+Inferred-mixed', [*exact_ordinary, mixed_row])]:
        value = measure(rows, name, method, 'abxye', profile + base['t'], goal, 'shannon')
        downstream[method] = {'log2_bound': value, 'profile_items': len(profile), 'profile_bytes': encoded_size(profile)}

    join_rows = connection.execute(f'SELECT count(*) FROM ({JOIN_SQL})').fetchone()[0]
    view_rows = connection.execute(f'SELECT count(*) FROM ({VIEW_SQL})').fetchone()[0]
    consumer_rows = connection.execute(f'SELECT count(*) FROM ({QUERY_SQL})').fetchone()[0]
    difference = connection.execute(f'WITH q AS ({QUERY_SQL}), flat AS ({FLAT_SQL}) '
                                    'SELECT count(*) FROM ((SELECT * FROM q EXCEPT SELECT * FROM flat) '
                                    'UNION ALL (SELECT * FROM flat EXCEPT SELECT * FROM q))').fetchone()[0]
    truths = {label: log_norm(connection, sql, target) for label, sql, target in [
        ('input_mixed', R_SQL, mixed('x', 2)), ('view', VIEW_SQL, view_goal), ('consumer', QUERY_SQL, goal)]}
    expected = 4 * math.sqrt(sum(max(row)**2 for row in matrix))
    if (difference or join_rows != 3 * view_rows or consumer_rows != 2 * view_rows
            or abs(truths['consumer'] - math.log2(expected)) > TOLERANCE):
        raise RuntimeError('Set projection or downstream truth check failed')
    if inferred_mixed < truths['view'] - TOLERANCE:
        raise RuntimeError('Producer mixed bound below truth')
    for row in downstream.values():
        gap = row['log2_bound'] - truths['consumer']
        if gap < -TOLERANCE:
            raise RuntimeError('Consumer bound below truth')
        row.update(bound=math.exp2(row['log2_bound']), bound_over_truth=math.exp2(gap))
    return {'matrix': matrix, 'histograms': histograms, 'join_rows': join_rows, 'view_rows': view_rows,
            'consumer_rows': consumer_rows,
            'difference_rows': difference, 'log2_truth': truths, 'downstream': downstream, 'results': rows}


def main():
    with duckdb.connect(config={'threads': 1}) as connection:
        trials = {name: trial(connection, name, matrix) for name, matrix in MATRICES.items()}
    larger, smaller = trials['larger'], trials['smaller']
    if larger['histograms'] != smaller['histograms']:
        raise RuntimeError('Ordinary degree multisets distinguish the two instances')
    for method in ('Exact-ordinary', 'Global-ordinary'):
        if abs(larger['downstream'][method]['log2_bound'] - smaller['downstream'][method]['log2_bound']) > TOLERANCE:
            raise RuntimeError(f'{method} received different ordinary information')
    for method in ('Card+Mixed', 'Inferred-ordinary+Mixed', 'Exact-ordinary+Inferred-mixed'):
        if smaller['downstream'][method]['log2_bound'] >= larger['log2_truth']['consumer'] - TOLERANCE:
            raise RuntimeError(f'{method} does not beat the ordinary information floor')
    run_id = 'mixed_separation_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = PROJECT / 'results/synthetic' / run_id
    output.mkdir(parents=True)
    write_json(output / 'report.json', {'run_id': run_id, 'status': 'passed', 'project': git_state(PROJECT),
                                        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                        'python': sys.version, 'duckdb': duckdb.__version__, 'ortools': ortools.__version__,
                                        'threads': 1, 'semantics': 'set', 'target': norm_json(mixed('xye', 2)),
                                        'sql': {'join': JOIN_SQL, 'view': VIEW_SQL, 'query': QUERY_SQL, 'flat': FLAT_SQL},
                                        'cases': trials})
    print(f'Ordinary degree multisets match on every scope. See {output / "report.json"}')


if __name__ == '__main__':
    main()
