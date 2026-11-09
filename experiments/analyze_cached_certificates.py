"""Compare archived STATS certificates without running SQL or the solver."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median


TOLERANCE = 1e-8


def comparison(rows):
    outcomes = Counter()
    gains = []
    for flat, compact in rows:
        gap = flat - compact
        if gap > TOLERANCE:
            outcomes['win'] += 1
            gains.append(math.exp2(gap))
        elif gap < -TOLERANCE:
            outcomes['loss'] += 1
        else:
            outcomes['tie'] += 1
    return {
        'compact_wins_ties_losses': [outcomes[k] for k in ('win', 'tie', 'loss')],
        'flat_over_compact_on_wins': (
            {'min': min(gains), 'median': median(gains), 'max': max(gains)} if gains else None),
        **{name + '_over_fresh': {
            'median': math.exp2(median(row[index] for row in rows)),
            'max': math.exp2(max(row[index] for row in rows))}
           for index, name in enumerate(('flat', 'compact'))},
    }


def analyze(runs):
    overall = []
    including_empty = []
    reports = []
    for directory in runs:
        summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        nonempty, all_rows = [], []
        producers = {}
        for producer in summary['views']:
            view = json.loads((directory / f'{producer}.json').read_text(encoding='utf-8'))
            windows = view['test']
            methods = ('Global-General', 'Flat-certificate', 'Profile-certificate')
            paired = zip(*(view['timings'][method] for method in methods), strict=True)
            observed_flat = [[] for _ in windows]
            flat_weights = set()
            wins_per_trial = []
            win_windows = set()
            producer_rows = []
            for fresh, flat, compact in paired:
                flat_weights.add(tuple(map(tuple, flat['certificate_weights'])))
                win_count = 0
                logs = (r['log2_bounds'] for r in (fresh, flat, compact))
                for index, (window, reference, f, c) in enumerate(zip(windows, *logs, strict=True)):
                    normalized = (f - reference, c - reference)
                    if min(normalized) < -TOLERANCE:
                        raise ValueError('Cached certificate is tighter than the fresh flat optimum')
                    observed_flat[index].append(normalized[0])
                    all_rows.append(normalized)
                    if window['truth_empty']:
                        continue
                    producer_rows.append(normalized)
                    if f - c > TOLERANCE:
                        win_count += 1
                        win_windows.add(index)
                wins_per_trial.append(win_count)
            producers[producer] = {
                **comparison(producer_rows),
                'distinct_nonempty_windows_with_a_win': len(win_windows),
                'trials_with_a_win': sum(count > 0 for count in wins_per_trial),
                'wins_match_fresh': sum(f - c > TOLERANCE and abs(c) <= TOLERANCE for f, c in producer_rows),
                'distinct_observed_flat_certificates': len(flat_weights),
                'observed_flat_envelope_matches_fresh': sum(
                    abs(min(values)) <= TOLERANCE for values in observed_flat),
                'windows_including_empty': len(windows),
            }
            nonempty.extend(producer_rows)
        reports.append({'run': str(directory), 'nonempty': comparison(nonempty),
                        'including_empty': comparison(all_rows), 'producers': producers})
        overall.extend(nonempty)
        including_empty.extend(all_rows)
    return {'nonempty': comparison(overall), 'including_empty': comparison(including_empty), 'runs': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps(analyze(args.runs), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
