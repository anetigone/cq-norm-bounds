"""Cross base-statistic acquisition with boundary policy on the fixed depth-one workloads."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import duckdb
import ortools

from job_views import derive, parse_jobjoin, scope
from run_jobjoin_oracles import read
from run_jobjoin_views import (PROFILES, PROJECT, base_budgets, compare_truth, degree, estimate,
                               git_state, open_imdb, projected_base_budgets, write_json)
from run_stats_views import DATA, digest, filtered_split, load_data, parse_source, run_directory


def checked(rows, stage, method, attributes, budgets, target, limit):
    value = estimate(rows, stage, 'projected', method, attributes, budgets, target, limit)
    if value is None:
        raise ValueError(f'{method} {stage}: {rows[-1]["status"]}')
    return value


def evaluate(db, split, truth, cache, projected_cache, args):
    before, start = len(cache), perf_counter()
    base = base_budgets(db, split.atoms, cache, args.sql_timeout)
    ordinary_ms = (perf_counter() - start) * 1000
    ordinary_sql = list(cache.values())[before:]
    before, start = len(projected_cache), perf_counter()
    extra = projected_base_budgets(db, split, projected_cache, args.sql_timeout)
    projected_ms = (perf_counter() - start) * 1000
    projected_sql = list(projected_cache.values())[before:]
    left = [row for atom in split.left for row in base[atom.alias]]
    right = [row for atom in split.right for row in base[atom.alias]]
    target = degree(split.head)
    truth_log = math.log2(truth)
    cells, traces = {}, {}
    for label, added in (('ordinary', []), ('projected', extra)):
        producer = left + added
        rows = traces[label] = []
        _, flat = checked(rows, 'E2', 'Flat', sorted(scope(split.atoms)), producer + right, target, args.lp_limit)
        methods = cells[label] = {}
        for method, orders in [('Flat', ()), *((name, PROFILES[name]) for name in ('Card+Max', 'Degree-profile'))]:
            start = perf_counter()
            profile = [checked(rows, 'E1', method, sorted(scope(split.left)), producer,
                               degree(split.left_head, split.keys, q), args.lp_limit) for q in orders]
            profile_ms = (perf_counter() - start) * 1000 if orders else 0
            if orders:
                checked(rows, 'E2', method, sorted(set(split.left_head) | scope(split.right)),
                        profile + right, target, args.lp_limit)
            terminal = rows[-1]
            compare_truth(terminal, truth_log)
            if terminal['safety_violation'] or terminal['log2_bound'] < flat - 1e-8:
                raise ValueError('Bound violates truth or same-input flat reference')
            methods[method] = {'bound_over_truth': terminal['bound_over_truth'],
                               'log2_bound': terminal['log2_bound'],
                               'profile_ms': profile_ms, 'online_ms': terminal['total_ms']}
        if methods['Degree-profile']['log2_bound'] > methods['Card+Max']['log2_bound'] + 1e-8:
            raise ValueError('Adding profile rows loosened the bound')
    for method in cells['ordinary']:
        if cells['projected'][method]['log2_bound'] > cells['ordinary'][method]['log2_bound'] + 1e-8:
            raise ValueError('Adding projected statistics loosened the bound')
    return {'truth': truth, 'ordinary_ms': ordinary_ms, 'projected_ms': projected_ms,
            'ordinary_sql_ms': sum(row['extract_ms'] for row in ordinary_sql),
            'projected_sql_ms': sum(row['extract_ms'] for row in projected_sql),
            'ordinary_sql_count': len(ordinary_sql), 'projected_sql_count': len(projected_sql),
            'projected_statistic_rows': len(extra), 'cells': cells, 'traces': traces}


def run(db, args, config, output):
    imdb = args.workload == 'imdb'
    if imdb:
        sizes = {name: meta['rows'] for name, meta in config['tables'].items()}
        columns = config['categorical_columns']
    else:
        data = load_data(db, args.data)
        if data != config['data']:
            raise ValueError('STATS snapshot differs from the archived truth run')
        sizes = {name: meta['rows'] for name, meta in data['tables'].items()}
        columns = {name: meta['columns'] for name, meta in data['tables'].items()}
    cache, projected_cache, reports = {}, {}, []
    for path in sorted(args.source.glob('query_*.json')):
        old = read(path)
        row = {'query_id': old['query_id'], 'status': old['status']}
        reports.append(row)
        atoms, filters = (parse_jobjoin(old['source_sql']), {}) if imdb else parse_source(old['source_sql'])
        options = {} if imdb else {'categorical_priority': config['categories']}
        split = derive(atoms, sizes, categorical_columns=columns, **options)
        if split is None:
            if old['status'] not in ('no_eligible_cut', 'ineligible_split'):
                raise ValueError(f'Eligibility changed for query {old["query_id"]}')
            row['status'] = ('no_categorical_payload' if derive(atoms, sizes, **options) is not None
                             else 'no_structural_split')
        elif old['status'] == 'passed':
            try:
                if not imdb:
                    split = filtered_split(db, split, filters)
                archived_split = old['split' if imdb else 'guarded_split']
                # Normalize tuples to their JSON representation before using archived truth.
                if json.loads(json.dumps(asdict(split))) != archived_split:
                    raise ValueError('Reconstructed split differs from the archived truth run')
                row.update(evaluate(db, split, old['truth']['outer']['rows'], cache, projected_cache, args))
                row['projection_shrinks'] = (old['truth']['projected']['view_rows']
                                             < old['truth']['retain_witnesses']['view_rows'])
            except (duckdb.Error, ValueError) as error:
                row.update(status='sql_timeout' if isinstance(error, duckdb.InterruptException) else 'failed',
                           error=str(error))
        write_json(output / path.name, row)
        print(f'{row["query_id"]:03d}: {row["status"]} {row.get("error", "")}', flush=True)
    passed = [row for row in reports if row['status'] == 'passed']
    status = dict(Counter(row['status'] for row in reports))
    write_json(output / 'base_statistics.json', list(cache.values()))
    write_json(output / 'projected_base_statistics.json', list(projected_cache.values()))
    summary = {'status_counts': status, 'projection_shrinks': sum(row['projection_shrinks'] for row in passed)}
    for phase in ('ordinary_ms', 'projected_ms', 'ordinary_sql_ms', 'projected_sql_ms',
                  'ordinary_sql_count', 'projected_sql_count'):
        summary[phase] = sum(row[phase] for row in passed)
    for label in ('ordinary', 'projected'):
        summary[label] = {}
        for method in ('Flat', 'Card+Max', 'Degree-profile'):
            rows = [row['cells'][label][method] for row in passed]
            if rows:
                summary[label][method] = {phase: median(r[phase] for r in rows)
                                           for phase in ('bound_over_truth', 'profile_ms', 'online_ms')}
        if passed:
            summary[label]['paired_cardmax_over_degree'] = median(
                math.exp2(r['cells'][label]['Card+Max']['log2_bound']
                          - r['cells'][label]['Degree-profile']['log2_bound']) for r in passed)
    write_json(output / 'summary.json', summary)
    if not passed or set(status) - {'passed', 'no_categorical_payload', 'no_structural_split'}:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workload', choices=('imdb', 'stats'))
    parser.add_argument('source', type=Path)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--sql-timeout', type=float, default=90)
    parser.add_argument('--lp-limit', type=int, default=5000)
    args = parser.parse_args()
    if args.sql_timeout <= 0 or args.lp_limit <= 0:
        parser.error('Time limits must be positive')
    config = read(args.source / 'config.json')
    family = 'jobjoin_views' if args.workload == 'imdb' else 'stats_ceb'
    output = run_directory(f'projected_ablation_{args.workload}', family)
    print(f'Run directory: {output}', flush=True)
    write_json(output / 'config.json', {'source': str(args.source.resolve()), 'source_config': config,
                                        'runner_sha256': digest(Path(__file__)), 'project': git_state(PROJECT),
                                        'python': platform.python_version(), 'system': platform.platform(),
                                        'duckdb': duckdb.__version__, 'ortools': ortools.__version__,
                                        'sql_timeout': args.sql_timeout, 'lp_limit_ms': args.lp_limit,
                                        'sql_threads': 4, 'memory_limit': '4GB', 'lp_threads': 1})
    connection = (open_imdb(output, config['tables'], 4, '4GB') if args.workload == 'imdb'
                  else duckdb.connect(config={'threads': 4, 'memory_limit': '4GB',
                                               'temp_directory': str(output / 'spill')}))
    with connection as db:
        run(db, args, config, output)


if __name__ == '__main__':
    main()
