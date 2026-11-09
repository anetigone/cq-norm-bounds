"""Export final CSV results and plotting data; keep original run logs local."""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean

from analyze_cached_certificates import analyze as analyze_certificates


PROJECT = Path(__file__).resolve().parents[1]
SOURCES = {
    'ablation_imdb': 'jobjoin_views/projected_ablation_imdb_20261108_130446_769932Z/summary.json',
    'ablation_stats': 'stats_ceb/projected_ablation_stats_20261108_130502_736177Z/summary.json',
    'matched_depth': 'stats_ceb/matched_depth_20261108_125626_817174Z/summary.json',
    'mixed_holdout': 'jobjoin_views/mixed_holdout_20261108_214822_526172/summary.json',
    'mixed_separation': 'synthetic/mixed_separation_20261107_192142_475348/report.json',
    'reuse_monthly': 'stats_ceb/stats_cached_reuse_20261108_134303_142564Z/summary.json',
    'reuse_half_month': 'stats_ceb/stats_cached_reuse_20261108_134411_585345Z/summary.json',
    'crossing_scale': 'synthetic/crossing_mixed_scale_20261108_135636_488969Z/report.json',
    'shape_scale': 'synthetic/flow_scale_shapes_20261108_130008_162564Z/report.json',
    'end_to_end': 'stats_ceb/stats_shared_execution_20261109_025531_951983Z/summary.json',
    'diamond_refresh': 'synthetic/diamond_refresh_20261108_134802_446840Z/report.json',
    'solver_agreement': 'synthetic/solver_agreement_20261108_220651_835016/report.json',
    'matched_berge': 'lpbound/matched_berge_20261108_153954_552797/summary.json',
    'joblight': 'lpbound/joblight_20261106_204514/reference_comparison.json',
}
# These are implementation details or individual execution traces, rather than final result columns.
OMIT = {'sql', 'truth_sql', 'query', 'statistics_sql', 'input_weights', 'training_weights',
        'extracts', 'input_statistics', 'inputs', 'payloads', 'profile', 'budgets',
        'columns', 'blocks', 'exponents', 'order', 'times_ms', 'timings', 'trials',
        'training', 'train', 'traces', 'project', 'commit', 'baseline_commit', 'run_id'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def flattened(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in OMIT and 'sha256' not in key:
                yield from flattened(item, f'{prefix}.{key}' if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flattened(item, f'{prefix}[{index}]')
    elif value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and (len(value) > 300 or re.match(r'^(?:[A-Za-z]:[/\\]|/)', value)):
            return
        # JSON scalars preserve the distinction between null, booleans, strings and numbers.
        yield prefix, json.dumps(value, allow_nan=False)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(source, value):
    return ({'source': source, 'metric': key, 'value_json': scalar} for key, scalar in flattened(value))


def export(root, output, sources=SOURCES):
    locations = {name: root / 'results' / relative for name, relative in sources.items()}
    reports = {name: read(path) for name, path in locations.items()}
    provenance = {'format_version': 1, 'scope': 'Final result data; original run logs are not distributed.',
                  'sources': {name: {'run': path.parent.name}
                              for name, path in locations.items()}}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'sources.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    write_csv(output / 'summary_metrics.csv', ('source', 'metric', 'value_json'),
              (row for name, report in reports.items() for row in metrics(name, report)))

    for name in ('ablation_imdb', 'ablation_stats', 'matched_depth', 'matched_berge'):
        rows = []
        for path in sorted(locations[name].parent.glob('query_*.json')):
            record = read(path)
            if name == 'matched_berge':
                record = {k: v for k, v in record.items() if k not in {'attributes', 'flow_model'}}
            rows.extend({'query_id': record['query_id'], 'metric': key, 'value_json': value}
                        for key, value in flattened(record) if key != 'query_id')
        write_csv(output / f'{name}_queries.csv', ('query_id', 'metric', 'value_json'), rows)

    depth = [read(path) for path in sorted(locations['matched_depth'].parent.glob('query_*.json'))]
    depth_points = []
    for row in depth:
        for i, bounds in enumerate(row['prefix_log2_bounds'], 1):
            for policy, value in bounds.items():
                depth_points.append({'kind': 'query', 'query_id': row['query_id'], 'depth': i,
                                     'policy': policy, 'bound_over_flat': math.exp2(value - row['projected_head']['flat'])})
    for policy, values in reports['matched_depth']['matched_depth'].items():
        depth_points.extend({'kind': 'median', 'query_id': '', 'depth': i, 'policy': policy,
                             'bound_over_flat': row['median']} for i, row in enumerate(values, 1))
    write_csv(output / 'plot_depth.csv', ('kind', 'query_id', 'depth', 'policy', 'bound_over_flat'), depth_points)
    write_csv(output / 'plot_projection.csv', ('query_id', 'drops_key', 'head', 'method', 'bound_over_truth'),
              ({'query_id': row['query_id'], 'drops_key': int(row['drops_key']), 'head': head,
                'method': method, 'bound_over_truth': math.exp2(row[head][method]) / row[head]['truth']}
               for row in depth for head in ('projected_head', 'retained_head')
               for method in ('inferred', 'exact', 'flat')))
    reuse_points = []
    for label, key in (('Monthly', 'reuse_monthly'), ('Half-month', 'reuse_half_month')):
        producers = list(reports[key]['views'].values())
        for method in producers[0]['including_sql']:
            prefixes = [view['including_sql'][method]['prefix'] for view in producers]
            for rows in zip(*prefixes, strict=True):
                reuse_points.append({'granularity': label, 'method': method, 'consumers': rows[0]['consumers'],
                                     'mean_cumulative_ms': mean(row['median_cumulative_ms'] for row in rows)})
    write_csv(output / 'plot_reuse.csv', ('granularity', 'method', 'consumers', 'mean_cumulative_ms'), reuse_points)
    write_csv(output / 'plot_crossing_scale.csv',
              ('attributes', 'status', 'build_ms', 'solve_ms', 'verify_ms', 'peak_rss_mib'),
              reports['crossing_scale']['scale'])

    scale_fields = ('attributes', 'n', 'gadgets', 'shape', 'density', 'seed', 'method', 'variant',
                    'status', 'statistic_count', 'log2_bound', 'n_variables', 'n_constraints',
                    'n_nonzeros', 'build_ms', 'solve_ms', 'verify_ms', 'peak_rss_mib', 'total_ms')
    for name in ('crossing_scale', 'shape_scale'):
        write_csv(output / f'{name}.csv', scale_fields,
                  ({key: trial.get(key) for key in scale_fields} for trial in reports[name]['trials']))

    e2e_rows, e2e_fields = [], set()
    for producer in reports['end_to_end']['cases']:
        for consumer in read(locations['end_to_end'].parent / f'{producer}.json')['measured']:
            row = {'producer': producer, **{key: value for key, value in flattened(consumer)}}
            e2e_rows.append(row)
            e2e_fields.update(row)
    write_csv(output / 'end_to_end_consumers.csv', ['producer', *sorted(e2e_fields - {'producer'})], e2e_rows)
    paired = read(locations['end_to_end'].parent / 'paired_analysis.json')
    write_csv(output / 'end_to_end_paired.csv', ('source', 'metric', 'value_json'), metrics('end_to_end', paired))
    certificates = analyze_certificates([locations['reuse_monthly'].parent, locations['reuse_half_month'].parent])
    for run in certificates['runs']:
        run['run'] = Path(run['run']).name
    write_csv(output / 'certificate_comparison.csv', ('source', 'metric', 'value_json'),
              metrics('fixed_certificates', certificates))

    holdout = []
    for producer in reports['mixed_holdout']['profiles']['ordinary-base']:
        record = read(locations['mixed_holdout'].parent / f'{producer}.json')
        holdout.extend({'producer': producer, 'metric': key, 'value_json': value}
                       for key, value in flattened({'targets': record['targets'], 'crossing': record['crossing']}))
    write_csv(output / 'mixed_holdout.csv', ('producer', 'metric', 'value_json'), holdout)
    print(f'Exported {len(list(output.glob("*.csv")))} CSV files to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--output', type=Path, default=PROJECT / 'results/paper')
    parser.add_argument('--runs', type=Path, help='JSON mapping of result names to rerun report paths.')
    args = parser.parse_args()
    sources = read(args.runs) if args.runs else SOURCES
    if set(sources) != set(SOURCES):
        parser.error('Run mapping must contain exactly the result names in SOURCES')
    export(args.root.resolve(), args.output.resolve(), sources)


if __name__ == '__main__':
    main()
