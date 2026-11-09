"""Check the IMDb JOB Parquet schema and row counts against its metadata."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def quote_name(value):
    return '"' + value.replace('"', '""') + '"'


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def validate(root, output):
    manifest = json.loads((root / 'metadata/job.json').read_text(encoding='utf-8'))
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='2GB'")
    report = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_root': str(root), 'source_url': manifest.get('source_data_url'),
        'duckdb_version': duckdb.__version__, 'tables': [], 'errors': [],
        'scope': 'File size, schema and row counts from the upstream metadata.',
    }
    for name, expected in manifest['tables'].items():
        record = {'table': name, 'expected_rows': expected['rows']}
        report['tables'].append(record)
        try:
            path = root / expected['file']
            record['bytes'] = path.stat().st_size
            if record['bytes'] != expected['bytes']:
                raise ValueError('File size differs from manifest')
            schema = con.execute(f'DESCRIBE SELECT * FROM read_parquet({literal(path)})').fetchall()
            actual_schema = [(row[0], row[1]) for row in schema]
            expected_schema = [(row['name'], row['type']) for row in expected['schema']]
            if actual_schema != expected_schema:
                raise ValueError(f'Schema differs: {actual_schema}')
            rows = con.execute(f'SELECT count(*) FROM read_parquet({literal(path)})').fetchone()[0]
            record['rows'] = rows
            if rows != expected['rows']:
                raise ValueError('Row count differs from manifest')
            record['status'] = 'passed'
            print(f'{name}: {rows:,} rows, size/schema/row count passed', flush=True)
        except Exception as error:
            record.update(status='failed', error=str(error))
            report['errors'].append(f'{name}: {error}')
            print(f'{name}: FAILED: {error}', flush=True)

    report['unexpected_parquet_files'] = sorted(
        str(p.relative_to(root)) for p in (root / 'data').glob('*.parquet')
        if str(p.relative_to(root)).replace('\\', '/') not in
        {v['file'] for v in manifest['tables'].values()}
    )
    incomplete = list((root / '.cache').rglob('*.incomplete'))
    report['incomplete_cache_files'] = len(incomplete)
    report['incomplete_cache_bytes'] = sum(p.stat().st_size for p in incomplete)
    if not report['errors']:
        for name in ('title', 'movie_companies', 'movie_info_idx'):
            file = root / manifest['tables'][name]['file']
            con.execute(f'CREATE TEMP VIEW {quote_name(name)} AS SELECT * FROM read_parquet({literal(file)})')
        project = Path(__file__).resolve().parents[1]
        query = (project.parent / 'LpBound/benchmarks/workloads/joblight/joblightQueries.sql').read_text().splitlines()[0]
        count_query = re.sub(r'^SELECT\s+\*', 'SELECT COUNT(*)', query, count=1, flags=re.IGNORECASE)
        report['joblight_query_1'] = {
            'query': query, 'true_cardinality': con.execute(count_query).fetchone()[0],
        }
        print(f"JOBlight query 1 truth: {report['joblight_query_1']['true_cardinality']:,}", flush=True)
    report['status'] = 'passed' if not report['errors'] else 'failed'
    report['total_rows'] = sum(t.get('rows', 0) for t in report['tables'])
    report['total_bytes'] = sum(t.get('bytes', 0) for t in report['tables'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{report['status']}: {len(report['tables'])} tables; report: {output}", flush=True)
    con.close()
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=project / 'data/imdb_job')
    parser.add_argument('--output', type=Path, default=project / 'results/data_validation/imdb_job.json')
    args = parser.parse_args()
    raise SystemExit(validate(args.root.resolve(), args.output.resolve()))
