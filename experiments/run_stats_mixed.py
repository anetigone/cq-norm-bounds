"""Repeat the equal-item mixed profile comparison on a separate STATS dataset."""

import argparse
from datetime import date
import math
from pathlib import Path

import duckdb
import ortools

from job_views import Atom, degree_counts_sql, degree_log
from run_jobjoin_equal_budget import compare_profiles, summarize
from run_jobjoin_mixed import A, B, ATTRIBUTES, MIXED, mixed_histogram, ordinary_budgets
from run_jobjoin_views import degree, git_state, query, write_json
from run_stats_views import DATA, PROJECT, digest, load_data, run_directory
from cqnorm.numeric import histogram_log_upper, log2_count_upper


# Post, category, and payload. The view projects away other event fields.
CASES = (
    ('comments', ('postid', 'score', 'userid')),
    ('posthistory', ('postid', 'posthistorytypeid', 'userid')),
    ('postlinks', ('postid', 'linktypeid', 'relatedpostid')),
    ('votes', ('postid', 'votetypeid', 'id')),
)
DIMENSION = f'SELECT id AS {A} FROM posts WHERE posttypeid IN (1, 2)'
TRAIN = (2010, 2013)
TEST = (2013, 2015)


def windows(start, stop):
    for year in range(start, stop):
        for lower, upper in ((date(year, 1, 1), date(year, 7, 1)), (date(year, 7, 1), date(year + 1, 1, 1))):
            for kind in (1, 2):
                yield {'post_type': kind, 'from': str(lower), 'until': str(upper),
                       'predicate': f"posttypeid = {kind} AND creationdate >= '{lower}' AND creationdate < '{upper}'"}


def consumer_summaries(db, batch, timeout):
    rows = []
    for window in windows(*batch):
        sql = f'SELECT count(*) FROM posts WHERE {window["predicate"]}'
        [(count,)], elapsed = query(db, sql, timeout)
        rows.append({**window, 'base_rows': count, 'statistics_sql': sql, 'extract_ms': elapsed})
    return rows


def consumers(rows):
    return [(degree((A,)), log2_count_upper(row['base_rows'])) for row in rows if row['base_rows']]


def exact_mixed_truth(db, grouped_sql, test, timeout):
    _, elapsed = query(db, f'CREATE OR REPLACE TEMP TABLE mixed_groups AS {grouped_sql}', timeout)
    for row in test:
        selected = (f'SELECT g.* FROM mixed_groups g JOIN posts p ON g.{A} = p.id '
                    f'WHERE {row["predicate"]}')
        sql = mixed_histogram(selected)
        histogram, ms = query(db, sql, timeout)
        if histogram and not row['base_rows']:
            raise ValueError('Empty consumer input had nonempty output')
        row.update(log2_truth=degree_log(histogram, 2) if histogram else None, truth_empty=not histogram,
                   truth_sql=sql, truth_ms=ms)
    return elapsed


def run_case(db, fact, columns, train, test, timeout):
    atom = Atom(fact, 'r', tuple(zip(columns, ATTRIBUTES, strict=True)))
    atom_sql = atom.sql()
    _, base_ms = query(db, f'CREATE OR REPLACE TEMP TABLE mixed_base AS {atom_sql}', timeout)
    ordinary, extracts = ordinary_budgets(db, timeout)
    mixed_sql = mixed_histogram(degree_counts_sql('SELECT * FROM mixed_base', (A, B)))
    histogram, mixed_ms = query(db, mixed_sql, timeout)
    if not histogram:
        raise ValueError(f'Empty mixed base in {fact}')
    mixed_row = MIXED, histogram_log_upper(histogram, 2)
    dim_sql = f'SELECT count(*) FROM ({DIMENSION})'
    [(dim_size,)], dim_ms = query(db, dim_sql, timeout)
    dim = degree((A,)), log2_count_upper(dim_size)
    targets = [stat for stat, _ in ordinary] + [MIXED]
    cases = [dict(row) for row in test]
    view_sql = f'SELECT * FROM mixed_base NATURAL JOIN ({DIMENSION})'
    grouped_sql = degree_counts_sql(view_sql, (A, B))
    truth_ms = exact_mixed_truth(db, grouped_sql, cases, timeout)
    # The shared selection routine only uses train inputs when choosing rows.
    active = [row for row in cases if row['base_rows']]
    treatments = {name: compare_profiles(base, targets, consumers(train), consumers(active), active)
                  for name, base in (('ordinary-base', [*ordinary, dim]), ('mixed-base', [*ordinary, dim, mixed_row]))}
    return {'case': fact, 'columns': columns, 'training': train, 'test': cases, 'treatments': treatments,
            'sql': {'base': atom_sql, 'view': view_sql, 'exact_groups': grouped_sql,
                    'ordinary': extracts, 'mixed': mixed_sql, 'dimension': dim_sql},
            'sql_ms': {'base': base_ms, 'mixed': mixed_ms, 'dimension': dim_ms, 'truth_groups': truth_ms}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--sql-timeout', type=float, default=30)
    args = parser.parse_args()
    if args.sql_timeout <= 0:
        parser.error('--sql-timeout must be positive')
    output = run_directory('stats_mixed')
    print(f'Run directory: {output}', flush=True)
    reports = []
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB', 'temp_directory': str(output / 'spill')}) as db:
        data = load_data(db, args.data)
        write_json(output / 'config.json', {'data': data, 'train_years': TRAIN, 'test_years': TEST, 'cases': CASES,
                                            'dimension_sql': DIMENSION, 'runner_sha256': digest(Path(__file__)),
                                            'project': git_state(PROJECT), 'duckdb': duckdb.__version__,
                                            'ortools': ortools.__version__, 'sql_timeout': args.sql_timeout})
        train = consumer_summaries(db, TRAIN, args.sql_timeout)
        test = consumer_summaries(db, TEST, args.sql_timeout)
        for fact, columns in CASES:
            report = run_case(db, fact, columns, train, test, args.sql_timeout)
            write_json(output / f'{fact}.json', report)
            reports.append(report)
            print(f'{fact}: passed', flush=True)
    summary = summarize(reports)
    for name in summary:
        summary[name]['per_view'] = {row['case']: summarize([row])[name] for row in reports}
        for profile in ('two-ordinary', 'one-ordinary-one-mixed'):
            sizes = [row['treatments'][name][profile]['profile_bytes'] for row in reports]
            summary[name][profile + '_json_bytes'] = [min(sizes), max(sizes)]
    write_json(output / 'summary.json', {'cases': len(reports), 'consumers': len(CASES) * len(test),
                                         'empty_inputs': len(CASES) * sum(not row['base_rows'] for row in test),
                                         'empty_outputs': sum(row['truth_empty'] for r in reports for row in r['test']),
                                         'comparisons': summary})


if __name__ == '__main__':
    main()
