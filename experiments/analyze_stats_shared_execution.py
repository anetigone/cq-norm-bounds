"""Summarize paired plan changes and their costs from a shared-execution run."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median


CARD, DEGREE = 'Card+Max', 'Degree-profile'
WINDOW, CATEGORY = 'window-first', 'category-first'
SHARED, INLINE = 'DuckDB-shared', 'DuckDB-inline'


def center(values):
    values = list(values)
    return median(values) if values else None


def changes(rows, method):
    return [r for r in rows if CARD in r['choices'] and method in r['choices'] and
            r['choices'][CARD] != r['choices'][method]]


def compare_changes(rows):
    pairs = [r for r in rows if r['fastest_forced']]
    ratios = [r['candidate_median_ms'][r['choices'][DEGREE]] / r['candidate_median_ms'][r['choices'][CARD]] for r in pairs]
    return {'changes': len(rows), 'nonempty': sum(r['output_rows'] > 0 for r in rows),
            'matches_exact_rows': sum(r['choices'][DEGREE] == r['exact_row_choice'] for r in rows),
            'times_available': len(pairs), 'matches_fastest': sum(r['choices'][DEGREE] == r['fastest_forced'] for r in pairs),
            'median_new_over_old': center(ratios), 'faster_at_least_1_25x': sum(x <= .8 for x in ratios),
            'slower_at_least_1_25x': sum(x >= 1.25 for x in ratios)}


def analyze(folder):
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))['cases']
    all_rows, counts, per_producer = [], Counter(), {}
    for name, case in summary.items():
        detail = json.loads((folder / f'{name}.json').read_text(encoding='utf-8'))
        rows = [r for r in detail['measured'] if r.get('choices')]
        timed = [r['methods'] for r in detail['batches'] if not r['warmup']]
        all_rows.extend(rows)
        counts['consumers'] += case['consumers']
        counts['empty_output'] += case['empty_output']
        counts['timed_forced_pairs'] += case['candidate_spread']['paired']
        counts['forced_spread_at_least_1_25'] += case['candidate_spread']['at_least_1_25']
        counts['exact_rows_matches_fastest'] += case['exact_row_agreement']
        changed = changes(rows, DEGREE)
        median_phases = {}
        if changed:
            matched = [r for r in changed if r['fastest_forced']]
            for plan in (WINDOW, CATEGORY):
                runs = [[rep['runs'][row['consumer']][plan] for rep in detail['reference'] if not rep['warmup']]
                        for row in matched]
                median_phases[plan] = {phase: center(center(q[phase] for q in trial) for trial in runs)
                                       for phase in ('build_ms', 'terminal_ms')}
        per_producer[name] = {'degree_changes': compare_changes(changed),
                              'changed_median_phase_ms': median_phases,
                              'flat_model_changes_vs_degree': sum(r['choices']['Flat-model'] != r['choices'][DEGREE]
                                                                   for r in rows if {'Flat-model', DEGREE} <= r['choices'].keys()),
                              'paired_inline_over_shared': center(r[INLINE]['ms'] / r[SHARED]['ms'] for r in timed
                                                                  if r[INLINE]['status'] == r[SHARED]['status'] == 'passed'),
                              'methods': {method: {**metrics,
                                                   'median_paired_batch_ratio_vs_card': center(metrics['paired_batch_ratio_vs_card'])}
                                          for method, metrics in case['methods'].items()}}

    methods = tuple(next(iter(summary.values()))['methods'])
    regret = {}
    for method in methods:
        rows = [r for r in all_rows if r['fastest_forced'] and r.get('choices', {}).get(method) in r['candidate_median_ms']]
        ratios = [r['candidate_median_ms'][r['choices'][method]] / r['candidate_median_ms'][r['fastest_forced']] for r in rows]
        regret[method] = {'available': len(ratios), 'median': center(ratios),
                          'geomean': math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else None,
                          'max': max(ratios, default=None),
                          'at_least_1_25': sum(x >= 1.25 for x in ratios),
                          'matches_exact_rows': sum(r['choices'][method] == r['exact_row_choice'] for r in rows)}
    return {'counts': counts, 'per_producer': per_producer, 'degree_changes': compare_changes(changes(all_rows, DEGREE)),
            'ratios_to_fastest_forced': regret}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2, allow_nan=False))
