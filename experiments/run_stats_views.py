"""Test degree-profile propagation on projected set CQs derived from STATS-CEB."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import file_digest, sha256
import math
from pathlib import Path
import re
from types import SimpleNamespace

import duckdb
import ortools

from job_views import COLUMN, check_chain_sql, degree_counts_sql, derive, parse_jobjoin, scope
from run_jobjoin_boundary import solve_policy
from run_jobjoin_depth import chain, matched_depth_comparison, run_chain
from run_jobjoin_views import (PROJECT, base_budgets, compare_truth, git_state, paired_improvement,
                              query, run_query, summarize, write_json)


DATA = PROJECT / 'data/stats_ceb/datasets/stats_simplified'
WORKLOAD = PROJECT.parent / 'LpBound/benchmarks/workloads/stats/statsQueries.sql'
FILES = ('badges', 'comments', 'postHistory', 'postLinks', 'posts', 'tags', 'users', 'votes')
CATEGORIES = ('posttypeid', 'votetypeid', 'posthistorytypeid', 'linktypeid', 'score')
LOCAL = re.compile(rf"({COLUMN})\s*(<=|>=|=|<|>)\s*(-?\d+|'[0-9 :.-]+')", re.I)
TOLERANCE = 1e-8


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def run_directory(prefix, family='stats_ceb'):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    output = PROJECT / 'results' / family / f'{prefix}_{stamp}'
    output.mkdir(parents=True)
    return output


def load_data(db, folder):
    ddl = folder / 'stats.sql'
    db.execute(re.sub(r'\bSERIAL PRIMARY KEY\b', 'INTEGER', ddl.read_text(encoding='utf-8')))
    tables = {}
    for name in FILES:
        path = folder / f'{name}.csv'
        table = name.lower()
        sql_path = path.resolve().as_posix().replace("'", "''")
        db.execute(f"COPY {table} FROM '{sql_path}' (FORMAT CSV, HEADER, NULL '')")
        rows, ids = db.execute(f'SELECT count(*), count(DISTINCT id) FROM {table}').fetchone()
        if not rows or rows != ids:
            raise ValueError(f'Nonempty table with unique nonnull id expected: {table}')
        with path.open(encoding='utf-8-sig') as stream:
            header = stream.readline().strip().lower().split(',')
        columns = [row[1].lower() for row in db.execute(f'PRAGMA table_info({table})').fetchall()]
        if columns != header:
            raise ValueError(f'CSV column order differs from schema for {table}: {header}, {columns}')
        tables[table] = {'rows': rows, 'columns': columns, 'file': path.name}
    return {'tables': tables}


def parse_source(sql):
    """Keep equijoins as CQ variables and apply all single-table predicates to base atoms."""
    statement = re.fullmatch(r'SELECT\s+\*\s+FROM\s+(.+?)\s+WHERE\s+(.+);', sql.strip(), re.I | re.S)
    if statement is None:
        raise ValueError(f'Unrecognized STATS statement: {sql}')
    equalities, predicates = [], {}
    for term in re.split(r'\s+AND\s+', statement[2], flags=re.I):
        term = term.strip()
        if re.fullmatch(rf'{COLUMN}\s*=\s*{COLUMN}', term, re.I):
            equalities.append(term.lower())
        else:
            local = LOCAL.fullmatch(term)
            if local is None:
                raise ValueError(f'Unsupported STATS predicate: {term}')
            qualified, op, literal = local.groups()
            alias, column = qualified.lower().split('.')
            predicates.setdefault(alias, []).append(f'{column} {op} {literal}')
    atoms = parse_jobjoin(f'SELECT * FROM {statement[1].lower()} WHERE {" AND ".join(equalities)};')
    if not set(predicates) <= {atom.alias for atom in atoms} or not all(atom.columns for atom in atoms):
        raise ValueError('Predicates or joins did not cover the expected aliases')
    return atoms, predicates


def filtered_split(db, split, predicates):
    atoms = {}
    for atom in split.atoms:
        conditions = ' AND '.join(sorted(predicates.get(atom.alias, ())))
        if not conditions:
            atoms[atom.alias] = atom
            continue
        statement = f'SELECT * FROM {atom.table} WHERE {conditions}'
        name = 'filtered_' + sha256(statement.encode()).hexdigest()[:24]
        db.execute(f'CREATE TEMP VIEW IF NOT EXISTS {name} AS {statement}')
        atoms[atom.alias] = replace(atom, table=name)

    def remap(side):
        return tuple(atoms[atom.alias] for atom in side)

    return replace(split, atoms=remap(split.atoms), left=remap(split.left), right=remap(split.right))


def matched_key(db, split, truth_sql, timeout):
    keys = ', '.join(split.keys)
    left = degree_counts_sql(truth_sql['projected'], split.keys)
    right = degree_counts_sql(truth_sql['right'], split.keys)
    sql = (f'SELECT {keys} FROM ({left}) l JOIN ({right}) r USING ({keys}) '
           f'ORDER BY l.degree::HUGEINT * r.degree, {keys} LIMIT 1')
    return query(db, sql, timeout)[0]


def semantic_check(db, split, sql, stages, tail, sample, timeout):
    available = set()
    for i, (atoms, head, _) in enumerate(stages):
        available.update(scope(atoms))
        future = scope([atom for later, _, _ in stages[i + 1:] for atom in later] + list(tail))
        if not (available & (future | set(split.head))) <= set(head) <= available:
            raise ValueError('View drops an output or future join variable')
        available = set(head)
    check = check_chain_sql(split, sql, [(atoms, head) for atoms, head, _ in stages], tail, sample)
    [(rows, difference)], elapsed = query(db, check, timeout)
    if not rows or difference:
        raise ValueError(f'Nonempty flat SQL sample disagreed: {rows=}, {difference=}')
    return {'matched_keys': sample, 'rows': rows, 'difference_rows': difference, 'sql': check, 'ms': elapsed}


def run_case(db, row, split, sizes, cache, args, depths):
    run_query(db, split, row, cache, {}, args)
    count = row['truth']['outer']['rows']
    sample = matched_key(db, split, row['truth_sql'], args.sql_timeout) if count else None
    first = [(split.left, split.left_head, split.keys)]
    if count:
        row['semantic_check'] = semantic_check(db, split, row['source_sql'], first, split.right, sample, args.sql_timeout)
    by_method = {r['method']: r for r in row['results'] if r['stage'] == 'E2' and r['variant'] == 'projected'}
    global_row = next(r for r in row['results'] if r['method'] == 'Global-General')
    if any(r['status'] not in ('optimal', 'empty') for r in row['results']):
        raise ValueError('Depth-one solve did not finish')
    budgets = base_budgets(db, split.atoms, cache, args.sql_timeout)
    for depth in depths:
        report = {'query_id': row['query_id'], 'depth': depth, 'results': [], 'true_cardinality': count}
        row['depths'].append(report)
        plan = chain(split, depth, sizes)
        if plan is None:
            report['status'] = 'too_few_atoms'
            continue
        stages, tail = plan
        report.update(stages=[{'atoms': [a.alias for a in atoms], 'head': head, 'key': key}
                              for atoms, head, key in stages], tail=[a.alias for a in tail])
        try:
            run_chain(stages, tail, split.head, budgets, report, args.lp_time_limit_ms, global_row.get('log2_bound'))
            terminal = next(r for r in report['results'] if r['method'] == 'Degree-profile' and r['stage'] == 'E2')
            if count and terminal['status'] == 'optimal':
                report['degree_profile_pair'] = [by_method['Degree-profile']['log2_bound'], terminal['log2_bound']]
            if depth == 4:
                for policy in ('Join-key', 'All-singletons'):
                    estimates, footprint, ms = solve_policy(stages, tail, split.head, budgets, policy, args.lp_time_limit_ms)
                    last = estimates[-1]
                    compare_truth(last, math.log2(count) if count else -math.inf)
                    if last['status'] == global_row['status'] == 'optimal':
                        gap = last['log2_bound'] - global_row['log2_bound']
                        last.update(bound_over_global=math.exp2(gap), below_global=gap < -TOLERANCE)
                    if last.get('safety_violation') or last.get('below_global'):
                        raise ValueError(f'Invalid {policy} bound')
                    last.update(boundary_items=sum(r['items'] for r in footprint),
                                boundary_bytes=sum(r['json_bytes'] for r in footprint), policy_total_ms=ms)
                    report['results'].extend(estimates)
            if any(r['status'] not in ('optimal', 'empty') for r in report['results']):
                raise ValueError('Chain solve did not finish')
            if count:
                report['semantic_check'] = semantic_check(db, split, row['source_sql'], stages, tail, sample, args.sql_timeout)
            report['status'] = 'passed'
        except (duckdb.Error, ValueError) as error:
            report.update(status='sql_timeout' if isinstance(error, duckdb.InterruptException) else 'failed', error=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--workload', type=Path, default=WORKLOAD)
    parser.add_argument('--query-ids', type=int, nargs='+')
    parser.add_argument('--sql-timeout', type=float, default=30)
    parser.add_argument('--lp-time-limit-ms', type=int, default=5000)
    args = parser.parse_args()
    if args.sql_timeout <= 0 or args.lp_time_limit_ms <= 0:
        parser.error('Time limits must be positive')
    statements = [sql.strip() + ';' for sql in args.workload.read_text(encoding='utf-8').split(';') if sql.strip()]
    if args.query_ids and not all(1 <= i <= len(statements) for i in args.query_ids):
        parser.error('Query IDs are one-based workload indices')
    output = run_directory('stats_views')
    print(f'Run directory: {output}', flush=True)
    reports, cache = [], {}
    settings = SimpleNamespace(sql_timeout=args.sql_timeout, lp_time_limit_ms=args.lp_time_limit_ms, projected_base=False)
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB', 'temp_directory': str(output / 'spill')}) as db:
        data = load_data(db, args.data)
        sizes = {name: meta['rows'] for name, meta in data['tables'].items()}
        categories = {name: meta['columns'] for name, meta in data['tables'].items()}
        write_json(output / 'config.json', {'data': data, 'workload': str(args.workload.resolve()),
                                            'workload_sha256': digest(args.workload), 'runner_sha256': digest(Path(__file__)),
                                            'project': git_state(PROJECT), 'query_ids': args.query_ids, 'settings': vars(settings),
                                            'duckdb': duckdb.__version__, 'ortools': ortools.__version__, 'categories': CATEGORIES})
        for i, sql in enumerate(statements, 1):
            if args.query_ids and i not in args.query_ids:
                continue
            row = {'query_id': i, 'source_sql': sql, 'results': [], 'truth': {}, 'depths': []}
            reports.append(row)
            try:
                atoms, filters = parse_source(sql)
                split = derive(atoms, sizes, categorical_columns=categories, categorical_priority=CATEGORIES)
                if split is None:
                    row['status'] = 'ineligible_split'
                else:
                    guarded = filtered_split(db, split, filters)
                    guarded_sizes = {a.table: sizes[b.table] for a, b in zip(guarded.atoms, split.atoms, strict=True)}
                    row.update(split=asdict(split), filters=filters, guarded_split=asdict(guarded))
                    run_case(db, row, guarded, guarded_sizes, cache, settings, (2, 4))
                    row['status'] = 'passed'
            except (duckdb.Error, ValueError) as error:
                row.update(status='sql_timeout' if isinstance(error, duckdb.InterruptException) else 'failed', error=str(error))
            write_json(output / f'query_{i:03d}.json', row)
            print(f'{i:03d}: {row["status"]}' + (f': {row["error"]}' if 'error' in row else ''), flush=True)
    passed = [r for r in reports if r['status'] == 'passed']
    depths = [d for r in passed for d in r['depths']]
    depth_ok = [d for d in depths if d['status'] == 'passed']
    nonempty = [r for r in passed if r['truth']['outer']['rows']]
    shrunk = [r for r in nonempty if r['truth']['projected']['view_rows'] < r['truth']['retain_witnesses']['view_rows']]
    statuses = dict(Counter(r['status'] for r in reports))
    write_json(output / 'summary.json', {
        'statuses': statuses, 'nonempty': len(nonempty), 'view_shrinks_nonempty': len(shrunk),
        'paired_nonempty': paired_improvement(nonempty), 'paired_shrinking_views': paired_improvement(shrunk),
        'depth_statuses': {d: dict(Counter(r['status'] for r in depths if r['depth'] == d)) for d in (2, 4)},
        'depth_pairwise': matched_depth_comparison(depth_ok), 'methods': summarize([*passed, *depth_ok])})
    if not passed or set(statuses) - {'passed', 'ineligible_split'} or any(r['status'] not in ('passed', 'too_few_atoms') for r in depths):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
