"""Download IMDb JOB and STATS from their public benchmark repositories."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen


PROJECT = Path(__file__).resolve().parents[1]


def fetch(path, url):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(url, headers={'User-Agent': 'cqnorm-artifact/1.0'})
        with urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.',
                                             suffix='.download', delete=False) as stream:
                temporary = Path(stream.name)
                shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
        if path.exists():
            raise FileExistsError(f'Destination appeared during download: {path}')
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('imdb', 'stats', 'all'), default='all')
    parser.add_argument('--root', type=Path, default=PROJECT,
                        help='Project root in which to create data/ (default: this project).')
    parser.add_argument('--plan', action='store_true', help='Print download URLs and destinations.')
    args = parser.parse_args()
    datasets = json.loads((PROJECT / 'configs/datasets.json').read_text(encoding='utf-8'))
    selected = datasets if args.dataset == 'all' else {args.dataset: datasets[args.dataset]}
    for dataset in selected.values():
        for filename in dataset['files']:
            path = args.root / dataset['destination'] / filename
            url = dataset['base_url'] + filename
            if args.plan:
                print(f'{url} -> {path}')
            elif not path.exists():
                print(f'Downloading {path}', flush=True)
                fetch(path, url)


if __name__ == '__main__':
    main()
