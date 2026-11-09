"""Scale a compatible crossing-key family with a mixed-norm output."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter

import ortools

from run_jobjoin_views import degree, git_state, norm_json, write_json
from run_scale_shapes import peak_rss_mib
from run_stats_views import PROJECT, digest, run_directory
from cqnorm import coverage_flow, shannon


def program(size, variant):
    payload = tuple(f'c{i}' for i in range(size))
    group = tuple(f'b{i}' for i in range(size))
    attributes = ('a', *group, *payload)
    target = shannon.MixedNorm((('a',), group, payload), (4, 2, 1))
    rows = []
    for i, (b, c) in enumerate(zip(group, payload, strict=True)):
        scope = ('a', b, c)
        if i or variant != 'drop-ab':
            rows.append((degree(scope, ('a', b), 4), 2))
        if i or variant != 'drop-ac':
            rows.append((degree(scope, ('a', c), math.inf), 1))
        rows.extend(((degree(('a', c)), 2), (degree((b, c)), 4)))
    return attributes, rows, target


def measure(size, method, variant, limit):
    attributes, budgets, target = program(size, variant)
    start = perf_counter()
    bound = (shannon.solve(attributes, budgets, target, limit) if method == 'shannon'
             else coverage_flow.solve(attributes, budgets, target, method='flow', time_limit_ms=limit))
    total_ms = 1000 * (perf_counter() - start)
    return {'gadgets': size, 'attributes': len(attributes), 'method': method, 'variant': variant,
            'statistic_count': len(budgets), 'target': norm_json(target),
            'peak_rss_mib': peak_rss_mib(), 'total_ms': total_ms, **asdict(bound)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time-limit-ms', type=int, default=20000)
    parser.add_argument('--worker', nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.time_limit_ms < 1:
        parser.error('Time limit must be positive')
    if args.worker:
        size, method, variant = args.worker
        print(json.dumps(measure(int(size), method, variant, args.time_limit_ms), allow_nan=False))
        return

    output = run_directory('crossing_mixed_scale', 'synthetic')
    print(f'Run directory: {output}', flush=True)
    report = {'system': platform.platform(), 'python': platform.python_version(), 'ortools': ortools.__version__,
              'project': git_state(PROJECT), 'runner_sha256': digest(Path(__file__)),
              'solver': 'single-thread GLOP', 'time_limit_ms': args.time_limit_ms,
              'flow_full_repeats': 3, 'trials': [], 'comparisons': [], 'scale': []}
    for size in (1, 2, 4, 8, 16, 32, 64):
        cases = [('flow', 'full')]
        if size <= 4:
            cases.append(('shannon', 'full'))
        if size <= 8 or size == 32:
            cases.extend((('flow', 'drop-ab'), ('flow', 'drop-ac')))
        rows = {}
        for method, variant in cases:
            repeated = []
            for _ in range(report['flow_full_repeats'] if (method, variant) == ('flow', 'full') else 1):
                stdout = subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
                                                  '--time-limit-ms', str(args.time_limit_ms),
                                                  '--worker', str(size), method, variant], text=True)
                repeated.append(json.loads(stdout.splitlines()[-1]))
            report['trials'].extend(repeated)
            if any((r['status'], r['log2_bound']) != (repeated[0]['status'], repeated[0]['log2_bound'])
                   for r in repeated):
                raise ValueError(f'Repeats disagree at size {size}')
            rows[method, variant] = repeated[0]
            if (method, variant) == ('flow', 'full'):
                report['scale'].append({'attributes': repeated[0]['attributes'], 'status': repeated[0]['status'],
                                        **{phase: median(r[phase] for r in repeated) for phase in
                                           ('build_ms', 'solve_ms', 'verify_ms', 'peak_rss_mib')}})
        full = rows['flow', 'full']['log2_bound']
        comparison = {'gadgets': size}
        if full is not None:
            for case, label in ((('shannon', 'full'), 'shannon_minus_flow'),
                                (('flow', 'drop-ab'), 'log2_inflation_without_ab'),
                                (('flow', 'drop-ac'), 'log2_inflation_without_ac')):
                other = rows.get(case, {}).get('log2_bound')
                if other is not None:
                    comparison[label] = other - full
        report['comparisons'].append(comparison)
        write_json(output / 'report.json', report)
        print(f'{2 * size + 1} attributes: {[(case, row["status"]) for case, row in rows.items()]}', flush=True)
        if (any(row['status'] not in ('optimal', 'timeout', 'unverified') for row in rows.values())
                or abs(comparison.get('shannon_minus_flow', 0)) > 1e-8
                or any(comparison.get(label, 0) < -1e-8 for label in
                       ('log2_inflation_without_ab', 'log2_inflation_without_ac'))):
            raise ValueError(f'Unexpected solver behavior for size {size}: {comparison}')
    for comparison in report['comparisons']:
        if 'log2_inflation_without_ab' in comparison and not all(
                comparison.get(label, 0) > 1e-8 for label in
                ('log2_inflation_without_ab', 'log2_inflation_without_ac')):
            raise ValueError(f'Both crossing statistics should affect the bound: {comparison}')


if __name__ == '__main__':
    main()
