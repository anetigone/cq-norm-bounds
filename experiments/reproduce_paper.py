"""Re-run the paper experiments and export fresh results to results/reproduced."""

import json
import os
from pathlib import Path
import subprocess
import sys

from analyze_stats_shared_execution import analyze
from export_paper_results import export


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def run(script, *args):
    print(f'Running {script}', flush=True)
    subprocess.run([sys.executable, str(ROOT / 'experiments' / f'{script}.py'), *map(str, args)],
                   cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, check=True)


def experiment(family, prefix, script, *args, report='summary.json'):
    runs = RESULTS / family
    before = set(runs.glob(f'{prefix}_*'))
    run(script, *args)
    completed = [path / report for path in set(runs.glob(f'{prefix}_*')) - before
                 if (path / report).is_file()]
    if len(completed) != 1:
        raise RuntimeError(f'Expected one new {report} in {family}/{prefix}_*')
    return completed[0]


def main():
    run('validate_imdb_job')
    categorical = experiment('jobjoin_views', 'jobjoin_views', 'run_jobjoin_views',
                             '--categorical-payload', '--projected-base', '--sql-timeout', 90).parent
    imdb = experiment('jobjoin_views', 'projected_ablation_imdb', 'run_projected_ablation', 'imdb', categorical)
    stats_views = experiment('stats_ceb', 'stats_views', 'run_stats_views').parent
    stats = experiment('stats_ceb', 'projected_ablation_stats', 'run_projected_ablation', 'stats', stats_views)

    views = experiment('jobjoin_views', 'jobjoin_views', 'run_jobjoin_views', '--sql-timeout', 90).parent
    depth = experiment('jobjoin_views', 'jobjoin_depth', 'run_jobjoin_depth', views).parent
    boundary = experiment('jobjoin_views', 'jobjoin_boundary', 'run_jobjoin_boundary', depth).parent
    oracles = experiment('jobjoin_views', 'jobjoin_oracles', 'run_jobjoin_oracles', boundary).parent
    projection = experiment('jobjoin_views', 'jobjoin_projection_gap', 'run_jobjoin_projection_gap', oracles).parent
    matched = experiment('stats_ceb', 'matched_depth', 'run_matched_depth', views, oracles, projection)

    mixed = experiment('stats_ceb', 'stats_mixed', 'run_stats_mixed').parent
    monthly = experiment('stats_ceb', 'stats_cached_reuse', 'run_stats_reuse', mixed)
    half_month = experiment('stats_ceb', 'stats_cached_reuse', 'run_stats_reuse', mixed,
                            '--test-granularity', 'half-month')
    end_to_end = experiment('stats_ceb', 'stats_shared_execution', 'run_stats_shared_execution')
    (end_to_end.parent / 'paired_analysis.json').write_text(
        json.dumps(analyze(end_to_end.parent), indent=2, allow_nan=False) + '\n', encoding='utf-8')

    joblight = experiment('lpbound', 'joblight', 'run_lpbound_joblight').parent
    run('summarize_lpbound_joblight', joblight)
    sources = {
        'ablation_imdb': imdb,
        'ablation_stats': stats,
        'matched_depth': matched,
        'mixed_holdout': experiment('jobjoin_views', 'mixed_holdout', 'run_mixed_holdout'),
        'mixed_separation': experiment('synthetic', 'mixed_separation', 'run_mixed_separation', report='report.json'),
        'reuse_monthly': monthly,
        'reuse_half_month': half_month,
        'crossing_scale': experiment('synthetic', 'crossing_mixed_scale', 'run_crossing_scale', report='report.json'),
        'shape_scale': experiment('synthetic', 'flow_scale_shapes', 'run_scale_shapes', report='report.json'),
        'end_to_end': end_to_end,
        'diamond_refresh': experiment('synthetic', 'diamond_refresh', 'run_diamond_refresh', report='report.json'),
        'solver_agreement': experiment('synthetic', 'solver_agreement', 'run_solver_agreement',
                                       '--time-limit-ms', 20000, report='report.json'),
        'matched_berge': experiment('lpbound', 'matched_berge', 'run_lpbound_matched'),
        'joblight': joblight / 'reference_comparison.json',
    }
    export(ROOT, RESULTS / 'reproduced', sources)


if __name__ == '__main__':
    main()
