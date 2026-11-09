"""Generate the supplement's PGFPlots coordinates from archived results."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

COLORS = ('blue!75!black', 'orange!80!black', 'teal!80!black')


def series(points, style, legend=None):
    coordinates = ' '.join(f'({x},{y:.7g})' for x, y in points)
    label = rf'\addlegendentry{{{legend}}}' if legend else ''
    return rf'\addplot+[{style}] coordinates {{{coordinates}}};' + '\n' + label


def axis(settings, plots):
    return ('\\begin{tikzpicture}\n\\begin{axis}[diagnostic,' + settings + ']\n'
            + '\n'.join(plots) + '\n\\end{axis}\n\\end{tikzpicture}\n')


def depth_plot(rows, summary):
    depths = range(1, 5)
    curves = [[(i, math.exp2(row['prefix_log2_bounds'][i - 1]['All-singletons']
                            - row['projected_head']['flat'])) for i in depths] for row in rows]
    medians = {policy: [(i, values[i - 1]['median']) for i in depths]
               for policy, values in summary['matched_depth'].items()}
    return depth_points_plot(curves, medians)


def depth_points_plot(curves, medians):
    plots = [series(points, 'gray!50, thin, mark=none, forget plot') for points in curves]
    for (policy, points), color, marker in zip(medians.items(), COLORS, ('square*', '*')):
        plots.append(series(points, f'{color}, very thick, mark={marker}', policy))
    return axis('ymode=log,log basis y=10, ymin=0.9,ymax=500,xtick={1,2,3,4},'
                'xlabel={Composition depth},ylabel={Bound / flat bound},legend pos=north west', plots)


def projection_plot(rows, ratios=False):
    dropped = [r for r in rows if r['drops_key']]
    ids = ','.join(str(r['query_id']) for r in dropped)
    panels = []
    for index, (head, title) in enumerate((('projected_head', 'Original output'),
                                          ('retained_head', 'Final join key retained'))):
        plots = [series([(i, 1) for i in (1, len(dropped))], 'black, dashed, mark=none', 'Truth' if not index else None)]
        for method, color, marker, offset in zip(('inferred', 'exact', 'flat'), COLORS,
                                                  ('*', 'square*', 'triangle*'), (-0.16, 0, 0.16)):
            points = [(i + offset, row[head][method] if ratios else math.exp2(row[head][method]) / row[head]['truth'])
                      for i, row in enumerate(dropped, 1)]
            label = {'inferred': 'Inferred', 'exact': 'Exact', 'flat': 'Flat'}[method]
            plots.append(series(points, f'{color}, thin, mark={marker}, mark size=1.3pt, only marks',
                                label if not index else None))
        common = ('ymode=log,log basis y=10,ymin=0.85,ymax=6500,ylabel={Bound / truth},'
                  'ytick={1,10,100,1000},height=3.8cm,xmin=0.5,xmax=12.5,xtick={1,...,12},')
        labels = (r'xticklabels=\empty,legend style={at={(0.5,1.40)},anchor=south},legend columns=4' if not index
                  else f'xticklabels={{{ids}}},xlabel={{JOBjoin query ID}}')
        panels.append(axis(common + f'title={{{title}}},' + labels, plots))
    return '\n'.join(panels)


def reuse_plot(monthly, half_month):
    points = {}
    for label, summary in (('Monthly', monthly), ('Half-month', half_month)):
        producers = list(summary['views'].values())
        points[label] = {}
        for method in producers[0]['including_sql']:
            prefixes = [view['including_sql'][method]['prefix'] for view in producers]
            points[label][method] = [(rows[0]['consumers'], mean(row['median_cumulative_ms'] for row in rows))
                                     for rows in zip(*prefixes, strict=True)]
    return reuse_points_plot(points)


def reuse_points_plot(points):
    panels = []
    methods = (
        ('Global-General', 'blue!65!black, dashed', 'Flat fresh'),
        ('Flat-model-reuse', 'blue!65!black', 'Flat model'),
        ('Flat-certificate', 'black', 'Flat certificate'),
        ('Reuse-compact-preselected', 'orange!85!black, dashed', 'Profile fresh'),
        ('Profile-model-reuse', 'orange!85!black', 'Profile model'),
        ('Reuse-compact-with-selection', 'teal!85!black, dotted', 'Select on demand'),
    )
    for i, (label, curves) in enumerate(points.items()):
        plots = []
        for method, color, title in methods:
            plots.append(series(curves[method], f'{color}, thick, mark=none', title if i == 0 else None))
        panels.append(axis(f'title={{{label}}},ymin=0,ymax=230,xmin=1,ylabel={{Cumulative time (ms)}},'
                           + ('height=4cm,legend style={at={(0.5,1.37)},anchor=south},legend columns=2' if i == 0
                              else 'height=4cm,xlabel={Consumers per producer}'), plots))
    return '\n'.join(panels)


def crossing_scale_plot(report):
    rows = [row for row in report['scale'] if row['status'] == 'optimal']
    plots = [series([(row['attributes'], row[phase]) for row in rows], f'{color}, thick, mark={mark}', title)
             for phase, color, mark, title in zip(('build_ms', 'solve_ms', 'verify_ms'), COLORS,
                                                   ('*', 'square*', 'triangle*'), ('Build', 'Solve', 'Verify'))]
    ticks = ','.join(str(row['attributes']) for row in rows)
    return axis(f'xmode=log,ymode=log,log basis y=10,xtick={{{ticks}}},xticklabels={{{ticks}}},'
                'xlabel={Attributes},ylabel={Time (ms)},legend pos=north west', plots)


def paper_result_plots(folder):
    def csv_rows(name):
        with (folder / name).open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))

    curves, medians = {}, {}
    for row in csv_rows('plot_depth.csv'):
        point = int(row['depth']), float(row['bound_over_flat'])
        if row['kind'] == 'median':
            medians.setdefault(row['policy'], []).append(point)
        elif row['policy'] == 'All-singletons':
            curves.setdefault(row['query_id'], []).append(point)
    projected = {}
    for row in csv_rows('plot_projection.csv'):
        value = projected.setdefault(row['query_id'], {'query_id': int(row['query_id']),
                                                       'drops_key': row['drops_key'] == '1',
                                                       'projected_head': {}, 'retained_head': {}})
        value[row['head']][row['method']] = float(row['bound_over_truth'])
    reuse = {}
    for row in csv_rows('plot_reuse.csv'):
        reuse.setdefault(row['granularity'], {}).setdefault(row['method'], []).append(
            (int(row['consumers']), float(row['mean_cumulative_ms'])))
    scale = [{'attributes': int(row['attributes']), 'status': row['status'],
              **{key: float(row[key]) for key in ('build_ms', 'solve_ms', 'verify_ms')}}
             for row in csv_rows('plot_crossing_scale.csv')]
    return (('depth', depth_points_plot(list(curves.values()), medians)),
            ('projection', projection_plot(list(projected.values()), ratios=True)),
            ('reuse', reuse_points_plot(reuse)),
            ('crossingscale', crossing_scale_plot({'scale': scale})))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-results', type=Path,
                        help='Regenerate figures from the published final CSV files.')
    parser.add_argument('paths', type=Path, nargs='+')
    args = parser.parse_args()
    if args.paper_results:
        if len(args.paths) != 1:
            parser.error('--paper-results requires one output path')
        plots = paper_result_plots(args.paper_results)
    else:
        if len(args.paths) != 5:
            parser.error('Expected depth, monthly, half-month, crossing and output paths')
        depth, monthly, half_month, crossing, _ = args.paths
        rows = [read(path) for path in sorted(depth.glob('query_*.json'))]
        plots = (('depth', depth_plot(rows, read(depth / 'summary.json'))),
                 ('projection', projection_plot(rows)),
                 ('reuse', reuse_plot(read(monthly / 'summary.json'), read(half_month / 'summary.json'))),
                 ('crossingscale', crossing_scale_plot(read(crossing / 'report.json'))))
    tex = '% Generated by experiments/render_diagnostic_plots.py.\n'
    for name, plot in plots:
        tex += rf'\newcommand{{\diagnostic{name}}}{{%' + '\n' + plot + '}\n'
    args.paths[-1].write_text(tex, encoding='utf-8')


if __name__ == '__main__':
    main()
