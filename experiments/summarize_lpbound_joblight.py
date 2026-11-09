"""Compare a completed local run with author-provided JOBlight references."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def summarize(run):
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    preprocessing = json.loads((run / 'preprocessing.json').read_text(encoding='utf-8'))
    rows = read_rows(run / 'queries.csv')
    official_truth = {int(r['QueryID']): int(r['truecardinality_Estimate']) for r in
                      read_rows(PROJECT / 'docs/references/joblight_truth.csv')}
    official_bounds = {int(r['QueryID']): float(r['lpbound_Estimate']) for r in
                       read_rows(PROJECT / 'docs/references/joblight_lpbound.csv')}
    mismatch = []
    comparison = []
    for row in rows:
        number = int(row['query_id'])
        if not row['truth'] or int(row['truth']) != official_truth[number]:
            mismatch.append(number)
        if row['estimate']:
            local, published = float(row['estimate']), official_bounds[number]
            comparison.append({'query_id': number, 'local': local, 'published': published,
                               'local_over_published': local / published,
                               'matches_rtol_1e_5': math.isclose(local, published, rel_tol=1e-5)})
    result = {
        'truth_mismatch_query_ids': mismatch,
        'truth_match_count': len(rows) - len(mismatch),
        'published_estimate_match_count_rtol_1e_5': sum(c['matches_rtol_1e_5'] for c in comparison),
        'estimate_comparison': comparison,
    }
    (run / 'reference_comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    err = summary['bound_ratio']
    ms = summary['estimation_ms']
    top = sorted((r for r in rows if r['bound_ratio']), key=lambda r: float(r['bound_ratio']), reverse=True)[:5]
    lines = [
        '# LpBound JOBlight 本地基线', '',
        f"- 运行批次：`{summary['run_id']}`。",
        '- 数据：liangyc/job 的 JOB IMDB Parquet；此前已通过文件大小、SHA256、schema 和全列扫描检查。',
        f"- 导入适配：Parquet 插入官方 DDL 建立的六张表，主键约束保留；辅助索引构建：{config.get('secondary_indexes', True)}。统计生成、统计查找及 Berge LP 使用原版代码。",
        f"- 环境：Python {config['python'].split()[0]}，DuckDB {config['packages']['duckdb']}，OR-Tools {config['packages']['ortools']}。",
        f"- 参数：{config['threads']} 个 DuckDB 线程，内存限制 {config['memory_limit']}；p=1..10 加最大 degree，5000 MCV，128 histogram buckets。",
        '- 范围：70 条完整 JOBlight 查询，SQL bag 行数；没有跑子查询注入、PostgreSQL 执行计划或 C++ 时间实验。', '',
        '| 指标 | 本机结果 |', '| --- | ---: |',
        f"| 成功完成 | {summary['completed']}/{summary['queries']} |",
        f"| 与作者真值一致 | {len(rows)-len(mismatch)}/{len(rows)} |",
        f"| 执行失败 | {summary['failed']} |",
        f"| 低估 | {summary['underestimates']} |",
        f"| 上界/真值 中位数 | {err.get('median', float('nan')):.4f} |",
        f"| 上界/真值 P95 | {err.get('p95', float('nan')):.4f} |",
        f"| 上界/真值 最大值 | {err.get('max', float('nan')):.4f} |",
        f"| 单查询估计时间 中位数（ms） | {ms.get('median', float('nan')):.3f} |",
        f"| 单查询估计时间 P95（ms） | {ms.get('p95', float('nan')):.3f} |",
        f"| 初次导入（含主键；辅助索引按配置）（s） | {preprocessing['original_import_seconds']:.3f} |",
        f"| 统计构建合计，含 SQL 生成（s） | {preprocessing['statistics_seconds']:.3f} |",
        f"| 其中 SQL 生成（s） | {preprocessing['sql_generation_seconds']:.3f} |", '',
        f"估计时间先预热一条查询，每条测 {config['estimation_repeats']} 次并取中位数；整体指标为各查询中位数的分位数。统计查找、SQL 解析和 LP 构建/求解均包含在估计时间内。真值执行和预处理单列。",
        '',
        '这是 Windows Python Berge 实现的本机运行结果；论文中的 C++ 加速时间需要另行编译、同机测量后比较。', '',
        f"作者给定的估计与本机估计在相对容差 1e-5 内一致的查询数：{result['published_estimate_match_count_rtol_1e_5']}/{len(comparison)}。详细差异保存在 reference_comparison.json。", '',
        '## 上界最松的五条查询', '',
        '| 查询 | 真值 | 上界 | 上界/真值 |', '| --- | ---: | ---: | ---: |',
    ]
    lines += [f"| {r['query_id']} | {int(r['truth']):,} | {float(r['estimate']):,.3f} | {float(r['bound_ratio']):.3f} |" for r in top]
    lines += ['', '## 结果文件', '',
              '- queries.csv：全部逐查询真值、估计及耗时。',
              '- config.json：环境、代码版本、数据来源及参数。',
              '- preprocessing.json / statistics_progress.jsonl：预处理计时。',
              '- statistics_sql/joblight.sql：实际执行的原版统计 SQL。',
              '- statistics/：统计表快照。',
              '- docs/references/：作者提供的真值与估计，附上游来源。', '']
    if mismatch:
        lines += [f'需要复核的真值差异查询：{mismatch}。', '']
    report = '\n'.join(lines)
    (run / 'report.md').write_text(report, encoding='utf-8')
    (PROJECT / 'docs/lpbound_joblight_baseline.md').write_text(
        report + f'\n本次运行结果目录：`{run}`。\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'estimate_comparison'}, indent=2))
    print(f'Report: {run / "report.md"}')
    return 0 if not mismatch and summary['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    raise SystemExit(summarize(parser.parse_args().run.resolve()))
