"""Run unmodified LpBound's Python JOBlight estimator on validated JOB Parquet.

Only storage paths and Parquet ingestion are adapted. The original DDL, optional indexes,
statistics SQL generator, statistic fetcher and Berge LP solver are used.
"""

import argparse
import csv
import importlib.metadata
import json
import math
import re
import statistics
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT.parent / 'LpBound'
sys.path.insert(0, str(BASELINE / 'src'))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def quote(value):
    return '"' + str(value).replace('"', '""') + '"'


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def main(args):
    from lpbound.config.paths import LpBoundPaths
    from lpbound.config.lpbound_config import LpBoundConfig
    from lpbound.config.benchmark_schema import load_benchmark_schema
    from lpbound.acyclic.stat_generator.sql_statistics_generator import generate_all_sql_for_benchmark
    from lpbound.acyclic.lpbound import estimate

    run_id = 'joblight_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output = PROJECT / 'results/lpbound' / run_id
    output.mkdir(parents=True)
    cache = PROJECT / ('data/cache/lpbound_indexed_v4' if args.build_indexes else 'data/cache/lpbound_no_secondary_indexes_v4')
    (cache / 'duckdb').mkdir(parents=True, exist_ok=True)
    LpBoundPaths.DATA_DIR = cache
    LpBoundPaths.DATASETS_DIR = PROJECT / 'data'
    LpBoundPaths.OUTPUT_DIR = output
    LpBoundPaths.GENERATED_SQL_DIR = output / 'statistics_sql'
    LpBoundPaths.GENERATED_SQL_DIR.mkdir()
    (output / 'lp').mkdir()

    validation = json.loads((PROJECT / 'results/data_validation/imdb_job.json').read_text(encoding='utf-8'))
    if validation['status'] != 'passed':
        raise RuntimeError('Dataset validation has not passed')
    source_root = Path(validation['dataset_root'])
    source_manifest = json.loads((source_root / 'metadata/job.json').read_text(encoding='utf-8'))
    config = LpBoundConfig(benchmark_name='joblight')
    schema = load_benchmark_schema(config)
    data_checks = {t['table']: t for t in validation['tables']}
    ddl = (BASELINE / 'data/sql_scripts/duckdb/create_queries_joblight.sql').read_text()
    cache_inputs = {
        'ddl': ddl,
        'data_root': str(source_root),
        'data': {name: [data_checks[name.lower()]['bytes'],
                        (source_root / source_manifest['tables'][name.lower()]['file']).stat().st_mtime_ns]
                 for name in schema['relations']},
        'import': 'original-ddl-parquet-v2', 'secondary_indexes': args.build_indexes,
    }
    run_config = {
        'run_id': run_id, 'python': sys.version, 'python_executable': sys.executable,
        'data_validation_report': str(PROJECT / 'results/data_validation/imdb_job.json'),
        'dataset_root': str(source_root), 'threads': args.threads, 'memory_limit': args.memory_limit,
        'estimation_repeats': args.repeats, 'estimator': 'original Python Berge LP / OR-Tools GLOP',
        'primary_keys': True, 'secondary_indexes': args.build_indexes,
        'lpbound_config': vars(config),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['duckdb', 'numpy', 'pandas', 'ortools', 'sqlparse', 'protobuf']},
        'scope': '70 full JOBlight queries; SQL bag-row cardinality; no subquery-plan injection or C++ timing.',
        'timing_notes': 'Estimation includes statistic lookup, connection access, SQL parsing, LP construction and solving; truth and preprocessing are separate.',
    }
    save_json(output / 'config.json', run_config)
    print(f'Run directory: {output}', flush=True)
    database = cache / 'duckdb/joblight_duckdb.db'
    marker_path = cache / 'joblight_import.json'
    marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
    con = duckdb.connect(str(database))
    con.execute(f'SET threads={args.threads}')
    con.execute(f'SET memory_limit={literal(args.memory_limit)}')
    try:
        stage_start = time.perf_counter()
        import_cache_reused = marker.get('inputs') == cache_inputs
        if not import_cache_reused:
            # Import is transactional, so an interrupted first import can be retried.
            if con.execute('SHOW TABLES').fetchall():
                raise RuntimeError('Cache has tables but no matching completed import marker; preserve it and choose a fresh cache')
            con.execute('BEGIN')
            con.execute(ddl)
            for name, columns in schema['relations'].items():
                meta = source_manifest['tables'][name.lower()]
                path = source_root / meta['file']
                if path.stat().st_size != meta['bytes']:
                    raise RuntimeError(f'Input file size changed: {path}')
                names = ', '.join(quote(c) for c in columns)
                con.execute(f'INSERT INTO {quote(name)} ({names}) SELECT {names} FROM read_parquet({literal(path)})')
                rows = con.execute(f'SELECT count(*) FROM {quote(name)}').fetchone()[0]
                if rows != meta['rows']:
                    raise RuntimeError(f'Import count mismatch: {name}')
                print(f'Imported {name}: {rows:,} rows', flush=True)
            if args.build_indexes:
                for statement in (BASELINE / 'data/sql_scripts/duckdb/fkindexes_joblight.sql').read_text().split(';'):
                    if statement.strip():
                        print(f'Building index: {statement.strip()}', flush=True)
                        con.execute(statement)
            con.execute('COMMIT')
            marker = {'inputs': cache_inputs, 'import_seconds': time.perf_counter() - stage_start}
            save_json(marker_path, marker)
            print(f"Import completed (secondary indexes={args.build_indexes}): {marker['import_seconds']:.2f}s", flush=True)
        else:
            print('Reusing verified base-table cache', flush=True)
        run_config['import_cache_reused'] = import_cache_reused
        run_config['original_import_seconds'] = marker['import_seconds']

        stage_start = time.perf_counter()
        commands = generate_all_sql_for_benchmark(con, config)
        (output / 'statistics_sql/joblight.sql').write_text('\n\n'.join(c['sql'] for c in commands), encoding='utf-8')
        generator_seconds = time.perf_counter() - stage_start
        by_tag = defaultdict(float)
        timings = []
        with (output / 'statistics_progress.jsonl').open('w', encoding='utf-8') as progress:
            for i, command in enumerate(commands, 1):
                started = time.perf_counter()
                try:
                    con.execute(command['sql'])
                except Exception:
                    (output / 'failed_statistics.sql').write_text(command['sql'], encoding='utf-8')
                    raise
                elapsed = time.perf_counter() - started
                by_tag[command['tag']] += elapsed
                timing = {'index': i, 'tag': command['tag'], 'seconds': elapsed}
                timings.append(timing)
                progress.write(json.dumps(timing) + '\n')
                progress.flush()
                if i % 20 == 0 or elapsed > 3 or i == len(commands):
                    print(f'Statistics {i}/{len(commands)}: {command["tag"]} ({elapsed:.2f}s)', flush=True)
        stats_seconds = time.perf_counter() - stage_start
        (output / 'statistics').mkdir()
        statistics_bytes = 0
        for (table,) in con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE NOT temporary AND schema_name='main'"
        ).fetchall():
            if table.upper() in schema['relations']:
                continue
            dest = output / 'statistics' / f'{table}.parquet'
            con.execute(f'COPY {quote(table)} TO {literal(dest)} (FORMAT PARQUET)')
            statistics_bytes += dest.stat().st_size
        norms_rows = con.execute('SELECT count(*) FROM norms').fetchone()[0]
        save_json(output / 'preprocessing.json', {
            'original_import_seconds': marker['import_seconds'], 'sql_generation_seconds': generator_seconds,
            'statistics_seconds': stats_seconds, 'seconds_by_tag': dict(by_tag), 'commands': len(commands),
            'norms_rows': norms_rows, 'exported_statistics_parquet_bytes': statistics_bytes,
        })
        con.execute('CHECKPOINT')
        con.close()
        print(f'Statistics ready: {norms_rows} norm rows, {stats_seconds:.2f}s', flush=True)

        queries = [q for q in (LpBoundPaths.WORKLOADS_DIR / 'joblight/joblightQueries.sql').read_text().splitlines() if q.strip()]
        if len(queries) != 70:
            raise RuntimeError(f'Expected 70 JOBlight queries, got {len(queries)}')
        # A read-only connection remains open to use identical DuckDB settings and
        # avoid repeated cold database opens in the upstream statistic fetcher.
        con = duckdb.connect(str(database), read_only=True)
        con.execute(f'SET threads={args.threads}')
        con.execute(f'SET memory_limit={literal(args.memory_limit)}')
        estimate(queries[0], config)  # one warm-up, excluded from recorded times
        records = []
        columns = ['query_id', 'truth', 'estimate', 'bound_ratio', 'log2_bound_ratio',
                   'estimation_ms_median', 'truth_seconds', 'status', 'error']
        with (output / 'queries.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for query_id, query in enumerate(queries, 1):
                row = {'query_id': query_id, 'status': 'failed', 'error': ''}
                try:
                    estimates, milliseconds = [], []
                    for _ in range(args.repeats):
                        started = time.perf_counter()
                        value = estimate(query, config)
                        milliseconds.append((time.perf_counter() - started) * 1000)
                        estimates.append(float(value))
                    value = estimates[0]
                    if not all(math.isfinite(v) and math.isclose(v, value, rel_tol=1e-8) for v in estimates):
                        raise RuntimeError('Nonfinite or inconsistent estimates')
                    count_sql = re.sub(r'^SELECT\s+\*', 'SELECT COUNT(*)', query, count=1, flags=re.IGNORECASE)
                    if count_sql == query:
                        raise RuntimeError('Unexpected query head')
                    started = time.perf_counter()
                    truth = con.execute(count_sql).fetchone()[0]
                    row.update(truth=truth, estimate=value, truth_seconds=time.perf_counter() - started,
                               estimation_ms_median=statistics.median(milliseconds))
                    if truth:
                        row.update(bound_ratio=value / truth, log2_bound_ratio=math.log2(value / truth))
                    row['status'] = 'underestimate' if value < truth * (1 - 1e-7) else 'passed'
                    (output / 'lp' / f'{query_id:03}.json').write_text(json.dumps({
                        'query': query, 'estimates': estimates, 'estimation_ms': milliseconds,
                    }, indent=2), encoding='utf-8')
                except Exception as error:
                    row['error'] = str(error)
                    (output / f'query_{query_id:03}_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                records.append(row)
                writer.writerow(row)
                stream.flush()
                print(f"Query {query_id}/70: {row['status']}, truth={row.get('truth')}, bound={row.get('estimate')}", flush=True)
        completed = [r for r in records if 'truth' in r]
        ratios = [r['bound_ratio'] for r in completed if r['truth'] > 0]
        ms = [r['estimation_ms_median'] for r in completed]
        summary = {
            'run_id': run_id, 'queries': len(records), 'completed': len(completed),
            'failed': sum(r['status'] == 'failed' for r in records),
            'underestimates': sum(r['status'] == 'underestimate' for r in records),
            'empty_results': sum(r.get('truth') == 0 for r in records),
            'bound_ratio': {label: float(np.percentile(ratios, percentile)) for label, percentile in
                            [('median', 50), ('p90', 90), ('p95', 95), ('max', 100)]} if ratios else {},
            'estimation_ms': {label: float(np.percentile(ms, percentile)) for label, percentile in
                              [('median', 50), ('p95', 95)]} if ms else {},
            'statistics_seconds': stats_seconds, 'sql_generation_seconds': generator_seconds,
            'truth_total_seconds': sum(r.get('truth_seconds', 0) for r in records),
            'scope': run_config['scope'],
        }
        summary['status'] = 'passed' if summary['failed'] == summary['underestimates'] == 0 else 'needs_review'
        save_json(output / 'summary.json', summary)
        save_json(output / 'config.json', run_config)
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if summary['status'] == 'passed' else 1
    except Exception:
        (output / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        con.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--memory-limit', default='4GB')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--build-indexes', action='store_true', help='Also build upstream secondary indexes; slow on DuckDB 0.10.1')
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1:
        parser.error('threads and repeats must be positive')
    raise SystemExit(main(args))
