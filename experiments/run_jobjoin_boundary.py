"""Compare boundary statistics on the archived four-layer JOBjoin plans."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import duckdb

from job_views import (Atom, Split, check_chain_sql, degree_log, histogram_sql, joined_histogram_sql,
                       project_join, scope)
from run_jobjoin_depth import flatten_budgets
from run_jobjoin_views import (ORDERS, PROJECT, base_budgets, compare_truth, degree, estimate, git_state,
                               load_histograms, norm_json, open_imdb, padded, query, summarize, write_json)

POLICIES = ('Join-key', 'All-singletons', 'Demand-keys')
BACKWARD_DEMAND = 'Backward-demand'
DIAGNOSTIC_IDS = (23, 22, 27)
TOLERANCE = 1e-8


def profile_requests(key, consumers, next_key, policy):
    extra = set(key) if len(key) > 1 else set()
    if policy == 'Join-key':
        extra.clear()
    elif policy == 'Demand-keys':
        boundary = set(key)
        uses = [boundary & atom.scope for atom in consumers]
        uses.append(boundary & set(next_key))
        # A consumer may join on a proper part or need that part for its own
        # output degrees. Full-boundary requests already have a profile.
        used = set().union(*(use for use in uses if use < boundary))
        extra &= used
    return [((), 1), *((key, q) for q in ORDERS[1:]),
            *(((variable,), q) for variable in sorted(extra) for q in ORDERS[1:])]


def decode_plan(prior, archived):
    raw = prior['split']
    atoms = tuple(Atom(row['table'], row['alias'], tuple(map(tuple, row['columns']))) for row in raw['atoms'])
    aliases = {atom.alias: atom for atom in atoms}

    def select(rows):
        return tuple(aliases[row['alias']] for row in rows)

    split = Split(atoms, select(raw['left']), select(raw['right']), tuple(raw['keys']),
                  tuple(raw['left_head']), tuple(raw['right_head']))
    stages = [(tuple(aliases[name] for name in row['atoms']), tuple(row['head']), tuple(row['key']))
              for row in archived['stages']]
    tail = tuple(aliases[name] for name in archived['tail'])
    return split, stages, tail


def requests_for(stages, tail, policy, *, backward_orders=ORDERS[1:]):
    downstream = [(atoms, key) for atoms, _, key in stages[1:]] + [(tail, ())]
    local = 'Demand-keys' if policy == BACKWARD_DEMAND else policy
    requests = [profile_requests(key, consumers, next_key, local)
                for (_, _, key), (consumers, next_key) in zip(stages, downstream)]
    if policy != BACKWARD_DEMAND:
        return requests

    orders = frozenset(backward_orders)
    if not orders <= set(ORDERS[1:]):
        raise ValueError('Backward demand orders must be non-cardinality profile orders')
    # A requested singleton at the next boundary can depend on the same
    # singleton here, even when the next join uses a different variable.
    # Carry requests through every boundary where the variable is still live.
    needed = set()
    for i in range(len(stages) - 1, -1, -1):
        (_, _, key), (consumers, next_key) = stages[i], downstream[i]
        direct = set(requests[i])
        additions = {((v,), q) for v in needed & set(key) for q in orders}
        requests[i] = [row for row in profile_requests(key, consumers, next_key, 'All-singletons')
                       if row in direct or row in additions]
        needed = {key[0] for key, _ in requests[i] if len(key) == 1}
    return requests


def encoded_size(profile):
    payload = [{**norm_json(stat), 'log2_bound': '-inf' if value == -math.inf else value}
               for stat, value in profile]
    return len(json.dumps(payload, separators=(',', ':'), allow_nan=False).encode('utf-8'))


def solve_policy(stages, tail, head, budgets, policy, limit, oracle=None, *, exact_after=None,
                 backward_orders=ORDERS[1:], shannon_max_attributes=8):
    start = perf_counter()
    rows, footprint = [], []
    previous = set()
    profile = []
    requests = requests_for(stages, tail, policy, backward_orders=backward_orders)
    for index, ((atoms, view_head, _), requested) in enumerate(zip(stages, requests)):
        inputs = profile + flatten_budgets(atoms, budgets)
        attributes = sorted(previous | scope(atoms))
        profile = []
        for key, q in requested:
            stat = degree(view_head, key, q)
            result = estimate(rows, f'layer_{index + 1}', 'depth_4', policy, attributes, inputs, stat, limit,
                              shannon_max_attributes=shannon_max_attributes)
            rows[-1].update(key=key, q='inf' if q == math.inf else q)
            if oracle is not None:
                truth = oracle[index][key, q]
                compare_truth(rows[-1], truth)
            if oracle is not None and (exact_after is None or exact_after == index + 1):
                profile.append((stat, padded(truth)))
            else:
                profile.append(result)
        if any(stat is None for stat in profile):
            rows.append({'stage': 'E2', 'variant': 'depth_4', 'method': policy, 'status': 'profile_unavailable'})
            break
        footprint.append({'layer': index + 1, 'items': len(profile), 'json_bytes': encoded_size(profile)})
        previous = set(view_head)
    else:
        estimate(rows, 'E2', 'depth_4', policy, sorted(previous | scope(tail)),
                 profile + flatten_budgets(tail, budgets), degree(head), limit,
                 shannon_max_attributes=shannon_max_attributes)
    total_ms = (perf_counter() - start) * 1000
    return rows, footprint, total_ms


def exact_profiles(connection, stages, tail, timeout):
    """Compute the complete layer profiles used in the three diagnostic queries."""
    requests = requests_for(stages, tail, 'All-singletons')
    prefix = []
    truths, extraction = [], []
    for index, ((atoms, head, _), requested) in enumerate(zip(stages, requests)):
        prefix.extend(atoms)
        direct = project_join(prefix, head)
        old_head = set(stages[index - 1][1]) if index else set()
        join_key = tuple(sorted(old_head & scope(atoms)))
        factorized = bool(index and join_key and set(join_key) <= set(head) and index == len(stages) - 1)
        if factorized:
            left_head = tuple(sorted(old_head & set(head)))
            right_head = tuple(sorted(scope(atoms) & set(head)))
            left_sql = f'SELECT DISTINCT {", ".join(left_head)} FROM exact_view'
            right_sql = project_join(atoms, right_head)
        else:
            _, materialize_ms = query(connection, f'CREATE OR REPLACE TEMP TABLE exact_view AS {direct}', timeout)
        histograms = {}
        timings = {}
        for key in dict.fromkeys(key for key, _ in requested):
            if factorized:
                sql = joined_histogram_sql(left_sql, right_sql, left_head, right_head, join_key, key)
            else:
                sql = histogram_sql('SELECT * FROM exact_view', key)
            histograms[key], elapsed = query(connection, sql, timeout)
            timings[str(key)] = {'sql': sql, 'ms': elapsed}
        truths.append({(key, q): degree_log(histograms[key], q) for key, q in requested})
        extraction.append({'layer': index + 1, 'factorized': factorized, 'materialize_sql': None if factorized else direct,
                           'materialize_ms': 0 if factorized else materialize_ms, 'histogram_queries': timings})
    connection.execute('DROP TABLE exact_view')
    return truths, extraction


def check_nonempty_sample(connection, split, prior, stages, tail, timeout, sample_size, cached=None):
    key = ', '.join(split.keys)
    left = project_join(split.left, split.keys)
    right = project_join(split.right, split.keys)
    selection = f'SELECT {key} FROM ({left}) l JOIN ({right}) r USING ({key}) ORDER BY {key} LIMIT {sample_size}'
    stage_heads = [(atoms, head) for atoms, head, _ in stages]
    if cached is not None:
        sql = check_chain_sql(split, prior['source_sql'], stage_heads, tail, cached['values'])
        if (cached['selection_sql'] != selection or cached['sql'] != sql
                or not cached['flat_rows'] or cached['difference_rows']):
            raise ValueError('Cached semantic check does not match the current plan')
        return cached
    matched, selection_ms = query(connection, selection, timeout)
    if not matched:
        raise ValueError('No matching keys for a query with nonempty archived truth')
    sql = check_chain_sql(split, prior['source_sql'], stage_heads, tail, matched)
    [(count, difference)], check_ms = query(connection, sql, timeout)
    if not count or difference:
        raise ValueError(f'Nonempty key sample failed: {count} flat rows, {difference} differences')
    return {'keys': split.keys, 'values': matched, 'selection_sql': selection, 'selection_ms': selection_ms,
            'flat_rows': count, 'difference_rows': difference, 'sql': sql, 'check_ms': check_ms}


def aggregate(reports):
    result = {}
    for policy in POLICIES:
        rows = [row for report in reports for row in report['results']
                if row['method'] == policy and row['status'] == 'optimal']
        result[policy] = {'n': len(rows)}
        if rows:
            result[policy].update(
                median_bound_over_depth_one=math.exp2(median(row['log2_over_depth_one'] for row in rows)),
                median_bound_over_global=median(row['bound_over_global'] for row in rows),
                median_bound_over_truth=median(row['bound_over_truth'] for row in rows),
                tighter_than_join_key=sum(row['log2_improvement'] > TOLERANCE for row in rows),
                looser_than_join_key=sum(row['log2_improvement'] < -TOLERANCE for row in rows),
                max_items_per_boundary=max(layer['items'] for row in rows for layer in row['footprint']),
                max_bytes_per_boundary=max(layer['json_bytes'] for row in rows for layer in row['footprint']),
                median_total_items=median(sum(layer['items'] for layer in row['footprint']) for row in rows),
                median_total_bytes=median(sum(layer['json_bytes'] for layer in row['footprint']) for row in rows),
                median_total_ms=median(row['total_ms'] for row in rows))
    return result


def summarize_oracles(reports):
    results = {}
    for report in reports:
        if 'oracle' not in report:
            continue
        results[report['query_id']] = {}
        for policy, rows in report['oracle']['policies'].items():
            online = report['layers'][policy]
            per_layer = []
            for index in range(1, len(report['stages']) + 1):
                stage = f'layer_{index}'
                local = [row for row in rows if row['stage'] == stage]
                propagated = [row for row in online if row['stage'] == stage]
                per_layer.append({'layer': index, 'max_local_over_truth': max(r['bound_over_truth'] for r in local),
                                  'max_propagated_over_truth': max(r['bound_over_truth'] for r in propagated),
                                  'max_propagated_over_local': max(r['propagated_over_local'] for r in local)})
            results[report['query_id']][policy] = {'layers': per_layer,
                                                  'terminal_exact_profile_over_truth': rows[-1]['bound_over_truth']}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('depth_run', type=Path)
    parser.add_argument('--query-ids', nargs='+', type=int)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--sample-size', type=int, default=3)
    parser.add_argument('--reuse-samples-from', type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.sample_size < 1:
        parser.error('Repeat and sample counts must be positive')
    archived_dir = args.depth_run.resolve()
    depth_config = json.loads((archived_dir / 'config.json').read_text(encoding='utf-8'))
    source = Path(depth_config['source_run'])
    config = depth_config['source_config']
    settings = config['args']
    timeout = settings['sql_timeout']
    limit = settings['lp_time_limit_ms']
    cache = load_histograms(source)
    sample_dir = args.reuse_samples_from.resolve() if args.reuse_samples_from else None
    if sample_dir is not None:
        sample_config = json.loads((sample_dir / 'config.json').read_text(encoding='utf-8'))
        if sample_config['depth_run'] != str(archived_dir) or sample_config['sample_size'] != args.sample_size:
            raise ValueError('Cached checks use different four-layer plans or sample size')
    run_id = 'jobjoin_boundary_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = PROJECT / 'results/jobjoin_views' / run_id
    output.mkdir(parents=True)
    write_json(output / 'config.json', {'run_id': run_id, 'depth_run': str(archived_dir), 'source_run': str(source),
                                        'project': git_state(PROJECT), 'repeats': args.repeats,
                                        'sample_size': args.sample_size, 'diagnostic_ids': DIAGNOSTIC_IDS,
                                        'query_ids': args.query_ids, 'policies': POLICIES,
                                        'reused_samples_from': str(sample_dir) if sample_dir else None,
                                        'demand_rule': 'singleton components of proper overlaps with next atoms or next output key'})
    print(f'Run directory: {output}', flush=True)
    reports = []
    with open_imdb(output, config['tables'], settings['duckdb_threads'], settings['duckdb_memory']) as connection:
        for path in sorted(archived_dir.glob('query_*.json')):
            archived = next((row for row in json.loads(path.read_text(encoding='utf-8'))
                             if row['depth'] == 4 and row['status'] == 'passed'), None)
            if archived is None or (args.query_ids and archived['query_id'] not in args.query_ids):
                continue
            query_id = archived['query_id']
            prior = json.loads((source / path.name).read_text(encoding='utf-8'))
            split, stages, tail = decode_plan(prior, archived)
            report = {'query_id': query_id, 'stages': archived['stages'], 'tail': archived['tail'],
                      'final_head': split.head, 'results': [], 'layers': {}, 'status': 'failed'}
            reports.append(report)
            try:
                cached = len(cache)
                budgets = base_budgets(connection, split.atoms, cache, timeout)
                if len(cache) != cached:
                    raise ValueError('Archived base summary is incomplete')
                one, old_bound = archived['degree_profile_pair']
                global_bound = next(row['log2_bound'] for row in prior['results'] if row['method'] == 'Global-General')
                truth = math.log2(archived['true_cardinality'])
                runs = {policy: [] for policy in POLICIES}
                for repeat in range(args.repeats):
                    for offset in range(len(POLICIES)):
                        policy = POLICIES[(repeat + offset) % len(POLICIES)]
                        runs[policy].append(solve_policy(stages, tail, split.head, budgets, policy, limit))
                for policy, samples in runs.items():
                    rows, footprint, _ = samples[0]
                    result = rows[-1]
                    result.update(total_ms=median(sample[2] for sample in samples),
                                  repeat_ms=[sample[2] for sample in samples], footprint=footprint)
                    report['results'].append(result)
                    report['layers'][policy] = rows[:-1]
                    compare_truth(result, truth)
                    if result['status'] != 'optimal' or any(s[0][-1].get('log2_bound') != result.get('log2_bound') for s in samples):
                        raise ValueError(f'Incomplete or inconsistent repeat for {policy}')
                    bound = result['log2_bound']
                    result.update(log2_over_depth_one=bound - one, log2_improvement=old_bound - bound,
                                  bound_over_global=math.exp2(bound - global_bound))
                    if result['safety_violation'] or bound < global_bound - TOLERANCE:
                        raise ValueError(f'Invalid upper bound for {policy}')
                    if policy == 'Join-key' and abs(bound - old_bound) > TOLERANCE:
                        raise ValueError('Archived Join-key bound failed to reproduce')
                cached_sample = None
                if sample_dir is not None and (sample_dir / path.name).is_file():
                    cached_sample = json.loads((sample_dir / path.name).read_text(encoding='utf-8')).get('sample')
                report['sample'] = check_nonempty_sample(connection, split, prior, stages, tail, timeout,
                                                         args.sample_size, cached=cached_sample)
                if query_id in DIAGNOSTIC_IDS:
                    oracle, extraction = exact_profiles(connection, stages, tail, timeout)
                    report['oracle'] = {'extraction': extraction, 'policies': {}}
                    for policy in POLICIES:
                        local, _, _ = solve_policy(stages, tail, split.head, budgets, policy, limit, oracle)
                        compare_truth(local[-1], truth)
                        if any(row.get('safety_violation') or row['status'] != 'optimal' for row in local):
                            raise ValueError(f'Invalid local Exact-profile result for {policy}')
                        for online, row in zip(report['layers'][policy], local[:-1]):
                            q = float(row['q'])
                            compare_truth(online, oracle[int(row['stage'].split('_')[1]) - 1][tuple(row['key']), q])
                            if online.get('safety_violation'):
                                raise ValueError(f'Propagated layer estimate below exact truth for {policy}')
                            row['propagated_over_local'] = math.exp2(online['log2_bound'] - row['log2_bound'])
                        report['oracle']['policies'][policy] = local
                report['status'] = 'passed'
            except duckdb.InterruptException as error:
                report.update(status='sql_timeout', error=str(error))
            except (duckdb.Error, ValueError) as error:
                report.update(error=str(error))
            write_json(output / path.name, report)
            print(f'{query_id:02d}: {report["status"]}' + (f': {report["error"]}' if 'error' in report else ''), flush=True)
    passed = [report for report in reports if report['status'] == 'passed']
    write_json(output / 'summary.json', {'run_id': run_id, 'statuses': dict(Counter(r['status'] for r in reports)),
                                        'policies': aggregate(passed), 'groups': summarize(passed),
                                        'nonempty_checks': sum('sample' in report for report in reports),
                                        'oracle_diagnostics': summarize_oracles(passed)})
    if not reports or len(passed) != len(reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
