"""Match depths and terminal heads on the archived IMDb depth-four query family."""

import argparse
import math
from pathlib import Path

from job_views import scope
from run_jobjoin_boundary import decode_plan, solve_policy
from run_jobjoin_depth import flatten_budgets
from run_jobjoin_oracles import checked_terminal, distribution, read
from run_jobjoin_views import PROJECT, base_budgets, degree, estimate, load_histograms, write_json
from run_stats_views import digest, run_directory

POLICIES = ('Join-key', 'All-singletons')
TOL = 1e-8


def validate_prefix(split, stages, tail):
    order = [atom for atoms, _, _ in stages for atom in atoms] + list(tail)
    if sorted(atom.alias for atom in order) != sorted(atom.alias for atom in split.atoms):
        raise ValueError('A query atom is missing or repeated')
    available = frozenset()
    for index, (atoms, head, key) in enumerate(stages):
        outside = scope([atom for later, _, _ in stages[index + 1:] for atom in later] + list(tail))
        available |= scope(atoms)
        live = available & outside
        if set(key) != live or set(head) != live | (available & set(split.head)):
            raise ValueError('A projection drops a variable that is still live')
        available = frozenset(head)


def solve_flat(split, head, budgets, limit, truth):
    rows = []
    estimate(rows, 'E2', 'depth_4', 'Flat', sorted(scope(split.atoms)),
             flatten_budgets(split.atoms, budgets), degree(head), limit)
    return checked_terminal(rows, truth)


def solve_depth(stages, tail, head, budgets, policy, limit, truth):
    rows, _, _ = solve_policy(stages, tail, head, budgets, policy, limit)
    return checked_terminal(rows, truth)


def analyze(prior, oracle, projection, cache, limit):
    split, stages, tail = decode_plan(prior, oracle)
    if len(stages) != 4:
        raise ValueError('Expected a depth-four plan')
    budgets = base_budgets(None, split.atoms, cache, 0)
    truth = math.log2(oracle['truth'])
    flat = solve_flat(split, split.head, budgets, limit, truth)
    if abs(flat - oracle['log2_global']) > TOL:
        raise ValueError('Archived flattened bound changed')
    prefixes = []
    for depth in range(1, 5):
        prefix = stages[:depth]
        remainder = [atom for atoms, _, _ in stages[depth:] for atom in atoms] + list(tail)
        validate_prefix(split, prefix, remainder)
        inferred = {policy: solve_depth(prefix, remainder, split.head, budgets, policy, limit, truth)
                    for policy in POLICIES}
        if min(inferred.values()) < flat - TOL:
            raise ValueError('Inferred profile beat the same-input flattened bound')
        prefixes.append(inferred)
    profile = oracle['policies']['All-singletons']['log2_bound_by_reset_layer']
    if abs(profile[0] - prefixes[-1]['All-singletons']) > TOL:
        raise ValueError('Archived inferred bound changed')
    projected = {'truth': oracle['truth'], 'inferred': profile[0], 'exact': profile[-1], 'flat': flat}
    retained = projected.copy()
    if projection is not None:
        if projection['status'] != 'passed' or projection['truth'] != projected['truth']:
            raise ValueError('Missing matching terminal projection truth')
        head = tuple(sorted(set(split.head) | set(stages[-1][2])))
        retained_truth = math.log2(projection['retained_key_truth'])
        retained = {'truth': projection['retained_key_truth'],
                    'inferred': solve_depth(stages, tail, head, budgets, 'All-singletons', limit, retained_truth),
                    'exact': projection['retained_key_bound']['log2_bound'],
                    'flat': solve_flat(split, head, budgets, limit, retained_truth)}
        if tuple(projection['retained_key_head']) != head or retained['exact'] > retained['inferred'] + TOL:
            raise ValueError('Retained-key comparison is inconsistent')
    elif not set(stages[-1][2]) <= set(split.head):
        raise ValueError('Missing terminal projection diagnostic')
    return {'query_id': oracle['query_id'], 'prefix_log2_bounds': prefixes, 'projected_head': projected,
            'retained_head': retained, 'drops_key': projection is not None,
            'log2_reduction_by_reset_layer': [a - b for a, b in zip(profile, profile[1:])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('oracles', type=Path)
    parser.add_argument('projection', type=Path)
    args = parser.parse_args()
    config = read(args.source / 'config.json')
    cache = load_histograms(args.source)
    output = run_directory('matched_depth')
    print(f'Run directory: {output}', flush=True)
    write_json(output / 'config.json', {**{name: str(path.resolve()) for name, path in vars(args).items()},
                                        'runner_sha256': digest(Path(__file__))})
    reports = []
    for path in sorted(args.oracles.glob('query_*.json')):
        oracle = read(path)
        if oracle['status'] != 'passed':
            raise ValueError(f'Oracle failed for {path.name}')
        gap = args.projection / path.name
        row = analyze(read(args.source / path.name), oracle, read(gap) if gap.exists() else None,
                      cache, config['args']['lp_time_limit_ms'])
        reports.append(row)
        write_json(output / path.name, row)
    dropped = [row for row in reports if row['drops_key']]
    write_json(output / 'summary.json', {
        'n': len(reports), 'drops_key': len(dropped),
        'matched_depth': {policy: [distribution([
            math.exp2(row['prefix_log2_bounds'][i][policy] - row['projected_head']['flat'])
            for row in reports]) for i in range(4)] for policy in POLICIES},
        'projection_pairs': {head: {method: distribution([
            math.exp2(row[head][method]) / row[head]['truth'] for row in dropped])
            for method in ('inferred', 'exact', 'flat')} for head in ('projected_head', 'retained_head')},
        'exact_tighter_than_flat': sum(row['projected_head']['exact'] < row['projected_head']['flat'] - TOL
                                      for row in reports),
        'exact_looser_than_flat': sum(row['projected_head']['exact'] > row['projected_head']['flat'] + TOL
                                     for row in reports),
    })


if __name__ == '__main__':
    main()
