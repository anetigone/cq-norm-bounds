"""Measure certified Flow on multiple topologies and summary densities."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
from random import Random
from statistics import median
import subprocess
import sys
from time import perf_counter

import ortools

from run_jobjoin_views import degree, git_state, norm_json, write_json
from run_stats_views import PROJECT, digest, run_directory
from cqnorm.coverage_flow import solve


def peak_rss_mib():
    if sys.platform == 'win32':
        from ctypes import POINTER, Structure, WinDLL, byref, c_size_t, sizeof
        from ctypes.wintypes import BOOL, DWORD, HANDLE

        class Counters(Structure):
            _fields_ = [('size', DWORD), ('faults', DWORD)] + [
                (field, c_size_t) for field in ('peak_working_set', 'working_set', 'peak_paged_pool',
                                              'paged_pool', 'peak_nonpaged_pool', 'nonpaged_pool',
                                              'pagefile', 'peak_pagefile')]

        kernel = WinDLL('kernel32')
        kernel.GetCurrentProcess.restype = HANDLE
        api = WinDLL('psapi').GetProcessMemoryInfo
        api.argtypes = [HANDLE, POINTER(Counters), DWORD]
        api.restype = BOOL
        counters = Counters(size=sizeof(Counters))
        if not api(HANDLE(kernel.GetCurrentProcess()), byref(counters), counters.size):
            raise OSError('GetProcessMemoryInfo failed')
        return counters.peak_working_set / 2**20
    from resource import RUSAGE_SELF, getrusage
    return getrusage(RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == 'darwin' else 1024)


def worker(shape, density, n, seed, limit):
    attributes = tuple(f'x{i}' for i in range(n))
    rng = Random(seed)
    inputs = []
    for i in range(1, n):
        parent = {'path': i - 1, 'star': 0, 'binary-tree': (i - 1) // 2}[shape]
        edge = (attributes[parent], attributes[i])
        left, right = rng.randint(2, 4), rng.randint(2, 4)
        # Each budget is realizable by a complete bipartite input on its edge.
        inputs.append((degree(edge), left + right))
        for j, key in enumerate(edge):
            for q in ((2, 4, 8, math.inf) if density == 'dense' else (math.inf,) if j == 0 else ()):
                support, fanout = (left, right) if j == 0 else (right, left)
                inputs.append((degree(edge, (key,), q), support / q + fanout))
    target = degree(attributes)
    start = perf_counter()
    bound = solve(attributes, inputs, target, method='flow', time_limit_ms=limit)
    elapsed = (perf_counter() - start) * 1000
    return {'shape': shape, 'density': density, 'n': n, 'seed': seed, 'total_ms': elapsed,
            'peak_rss_mib': peak_rss_mib(), 'statistic_count': len(inputs),
            'inputs': [{**norm_json(stat), 'log2_budget': value} for stat, value in inputs], **asdict(bound)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time-limit-ms', type=int, default=5000)
    parser.add_argument('--worker', nargs=4, metavar=('SHAPE', 'DENSITY', 'N', 'SEED'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.time_limit_ms <= 0:
        parser.error('Time limit must be positive')
    if args.worker:
        shape, density, n, seed = args.worker
        print(json.dumps(worker(shape, density, int(n), int(seed), args.time_limit_ms), allow_nan=False))
        return
    output = run_directory('flow_scale_shapes', 'synthetic')
    print(f'Run directory: {output}', flush=True)
    report = {'system': platform.platform(), 'python': platform.python_version(), 'ortools': ortools.__version__,
              'project': git_state(PROJECT), 'runner_sha256': digest(Path(__file__)),
              'solver': 'single-thread GLOP', 'time_limit_ms': args.time_limit_ms,
              'memory': 'fresh process high-water RSS including interpreter and native libraries',
              'trials': [], 'summary': []}
    for shape in ('path', 'star', 'binary-tree'):
        for density in ('sparse', 'dense'):
            for n in (20, 40, 80):
                trials = []
                for seed in range(3):
                    stdout = subprocess.check_output(
                        [sys.executable, str(Path(__file__).resolve()), '--time-limit-ms', str(args.time_limit_ms),
                         '--worker', shape, density, str(n), str(seed)], text=True)
                    trials.append(json.loads(stdout.splitlines()[-1]))
                report['trials'].extend(trials)
                optimal = [row for row in trials if row['status'] == 'optimal']
                report['summary'].append({'shape': shape, 'density': density, 'n': n,
                                          'statuses': [r['status'] for r in trials],
                                          **{phase: median(row[phase] for row in optimal) if optimal else None
                                             for phase in ('build_ms', 'solve_ms', 'verify_ms', 'peak_rss_mib')}})
                write_json(output / 'report.json', report)
                print(f'{shape} {density} {n}: {[r["status"] for r in trials]}', flush=True)
    if any(row['status'] not in ('optimal', 'timeout') for row in report['trials']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
