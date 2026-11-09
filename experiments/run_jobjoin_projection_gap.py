"""Measure the effect of retaining the final join key in depth-four JOBjoin."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import math
from pathlib import Path

import duckdb

from job_views import degree_counts_sql, joined_degree_counts_sql, project_join, scope
from run_jobjoin_boundary import TOLERANCE, decode_plan
from run_jobjoin_depth import flatten_budgets
from run_jobjoin_oracles import distribution, read
from run_jobjoin_views import (PROJECT, base_budgets, degree, estimate, git_state, load_histograms,
                               open_imdb, padded, query, write_json)


def left_counts(stages, key, factorized, materialize):
    prefix = [atom for atoms, _, _ in stages[:-1] for atom in atoms]
    atoms, head, _ = stages[-1]
    if not factorized:
        materialize('probe_view', project_join([*prefix, *atoms], head))
        return degree_counts_sql('SELECT * FROM probe_view', key)
    old_head = set(stages[-2][1])
    new_scope = scope(atoms)
    join = tuple(sorted(old_head & new_scope))
    left_head = tuple(sorted(old_head & set(head)))
    right_head = tuple(sorted(new_scope & set(head)))
    materialize('probe_view', project_join(prefix, tuple(sorted(old_head))))
    return joined_degree_counts_sql(f'SELECT DISTINCT {", ".join(left_head)} FROM probe_view',
                                    project_join(atoms, right_head),
                                    left_head, right_head, join, key)


def components(atoms):
    """Split the tail by shared variables, keeping the input order within each part."""
    parts = []
    for atom in atoms:
        matches = [part for part in parts if atom.scope & scope(part)]
        parts = [part for part in parts if part not in matches]
        parts.append([item for part in matches for item in part] + [atom])
    return parts


def multiply_histograms(left, right):
    product = Counter()
    for degree_l, frequency_l in left.items():
        for degree_r, frequency_r in right.items():
            product[degree_l * degree_r] += frequency_l * frequency_r
    return product


def sorted_dot(left, right):
    """Dot product after sorting both multisets, padding the shorter one with zeros."""
    left = iter(sorted(left.items(), reverse=True))
    right = iter(sorted(right.items(), reverse=True))
    dl, nl = next(left, (0, 0))
    dr, nr = next(right, (0, 0))
    value = 0
    while nl and nr:
        paired = min(nl, nr)
        value += paired * dl * dr
        nl, nr = nl - paired, nr - paired
        if not nl:
            dl, nl = next(left, (0, 0))
        if not nr:
            dr, nr = next(right, (0, 0))
    return value


def probe(connection, stages, tail, head, budgets, oracle, settings, timeout, statements):
    def run(label, sql):
        entry = {'label': label, 'sql': sql}
        statements.append(entry)
        result, entry['ms'] = query(connection, sql, timeout)
        return result

    def materialize(name, sql):
        run(name, f'CREATE OR REPLACE TEMP TABLE {name} AS {sql}')

    def histogram(name):
        return dict(run(f'{name}_histogram', f'SELECT degree, count(*) FROM {name} GROUP BY degree'))

    _, view_head, key = stages[-1]
    target = tuple(sorted(set(head) | set(key)))
    # With the key retained, the two projected sides form a product per key.
    # Keep independent tail components separate to avoid enumerating their Cartesian product.
    materialize('probe_left', left_counts(stages, key, oracle['extraction'][-1]['factorized'], materialize))
    connection.execute('DROP TABLE probe_view')
    tail_histogram = {1: 1}
    sources, product = 'probe_left AS l', 'l.degree::HUGEINT'
    for i, part in enumerate(components(tail)):
        part_head = tuple(sorted(scope(part) & set(target)))
        part_key = tuple(sorted(scope(part) & set(key)))
        if not part_head:
            raise ValueError('A tail component projects to an empty attribute set')
        name = f'probe_tail_{i}'
        materialize(name, degree_counts_sql(project_join(part, part_head), part_key))
        tail_histogram = multiply_histograms(tail_histogram, histogram(name))
        sources += f' JOIN {name} USING ({", ".join(part_key)})' if part_key else f' CROSS JOIN {name}'
        product += f' * {name}.degree'
    [(witness,)] = run('matching_keys', f'SELECT sum({product}) FROM {sources}')
    aligned_max = sorted_dot(histogram('probe_left'), tail_histogram)
    truth = oracle['truth']
    if witness is None or witness < truth or aligned_max < witness or (set(key) <= set(head) and witness != truth):
        raise ValueError('Terminal cardinality check failed')

    old = oracle['policies']['All-singletons']
    profile = [(degree(view_head, tuple(row['key']), float(row['q'])), padded(row['log2_exact']))
               for row in old['layer_statistics'] if row['stage'] == 'layer_4']
    rows = []
    estimate(rows, 'E2', 'depth_4', 'Exact-profile with retained key', sorted(set(view_head) | scope(tail)),
             profile + flatten_budgets(tail, budgets), degree(target), settings['lp_time_limit_ms'])
    retained = rows[0]
    projected_bound = old['log2_bound_by_reset_layer'][-1]
    if retained['status'] != 'optimal' or retained['log2_bound'] < max(math.log2(witness), projected_bound) - TOLERANCE:
        raise ValueError('Retained-key bound failed soundness or monotonicity check')
    return {'truth': truth, 'retained_key_truth': witness, 'max_aligned_truth': aligned_max,
            'terminal_key': key, 'head': head, 'retained_key_head': target,
            'sql_ms': sum(entry['ms'] for entry in statements),
            'retained_key_bound': retained, 'log2_projected_bound': projected_bound,
            'projection_factor': witness / truth, 'alignment_factor': aligned_max / witness,
            'retained_bound_over_truth': math.exp2(retained['log2_bound'] - math.log2(witness)),
            'projected_bound_over_truth': old['exact_profile_over_truth'],
            'retained_over_projected_bound': math.exp2(retained['log2_bound'] - projected_bound)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('oracle_run', type=Path)
    parser.add_argument('--sql-timeout', type=float, default=90)
    parser.add_argument('--query-ids', nargs='+', type=int, help='Override the set of queries that project out a key component')
    args = parser.parse_args()
    if args.sql_timeout <= 0:
        parser.error('--sql-timeout must be positive')
    oracle_dir = args.oracle_run.resolve()
    oracle_config = read(oracle_dir / 'config.json')
    depth = Path(oracle_config['depth_run'])
    source = Path(oracle_config['source_run'])
    config = read(source / 'config.json')
    settings = config['args']
    cache = load_histograms(source)
    run_id = 'jobjoin_projection_gap_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = PROJECT / 'results/jobjoin_views' / run_id
    output.mkdir(parents=True)
    write_json(output / 'config.json', {'run_id': run_id, 'project': git_state(PROJECT), 'oracle_run': str(oracle_dir),
                                        'runner_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                                        'query_ids': args.query_ids, 'sql_timeout': args.sql_timeout})
    print(f'Run directory: {output}', flush=True)
    reports = []
    with open_imdb(output, config['tables'], settings['duckdb_threads'], settings['duckdb_memory']) as connection:
        for path in sorted(oracle_dir.glob('query_*.json')):
            oracle = read(path)
            query_id = oracle['query_id']
            if oracle['status'] != 'passed' or (args.query_ids and query_id not in args.query_ids):
                continue
            archived = next(row for row in read(depth / path.name) if row['depth'] == 4 and row['status'] == 'passed')
            split, stages, tail = decode_plan(read(source / path.name), archived)
            if not args.query_ids and set(stages[-1][2]) <= set(split.head):
                continue
            row = {'query_id': query_id, 'status': 'failed', 'statements': []}
            reports.append(row)
            try:
                budgets = base_budgets(connection, split.atoms, cache, args.sql_timeout)
                row.update(probe(connection, stages, tail, split.head, budgets, oracle, settings,
                                 args.sql_timeout, row['statements']))
                row['status'] = 'passed'
            except duckdb.InterruptException as error:
                row.update(status='sql_timeout', error=str(error))
            except (duckdb.Error, ValueError) as error:
                row['error'] = str(error)
            finally:
                for name in ('probe_view', 'probe_left', *(f'probe_tail_{i}' for i in range(len(tail)))):
                    connection.execute(f'DROP TABLE IF EXISTS {name}')
            write_json(output / path.name, row)
            print(f'{query_id:02d}: {row["status"]}' + (f': {row["error"]}' if 'error' in row else ''), flush=True)
    statuses = dict(Counter(row['status'] for row in reports))
    passed = [row for row in reports if row['status'] == 'passed']
    metrics = {name: distribution([row[name] for row in passed]) for name in (
        'projection_factor', 'alignment_factor', 'retained_bound_over_truth', 'projected_bound_over_truth',
        'retained_over_projected_bound')} if passed else {}
    write_json(output / 'summary.json', {'run_id': run_id, 'statuses': statuses, 'metrics': metrics})
    if not reports or len(passed) != len(reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
