import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import scipy
from scipy import stats

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_json(out, name, value):
    with (out / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def save_csv(out, name, rows):
    with (out / name).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def input_manifest(root):
    return [dict(file=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in sorted((root / 'inputCSV').glob('*.csv'))]


def holm_two(values):
    small, large = sorted(range(2), key=lambda i: values[i])
    result = [0., 0.]
    result[small] = min(1., 2 * values[small])
    result[large] = min(1., max(result[small], values[large]))
    return result


def manual_kruskal(x, y):
    values = list(x) + list(y)
    counts = Counter(values)
    positions, offset = {}, 0
    for value, count in sorted(counts.items()):
        positions[value] = offset + (count + 1) / 2
        offset += count
    ranks = [positions[v] for v in values]
    n, n1, n2 = len(values), len(x), len(y)
    correction = 1 - sum(t ** 3 - t for t in counts.values()) / (n ** 3 - n)
    if correction == 0:
        raise ValueError('All values identical; Kruskal-Wallis statistic is undefined.')
    rank1, rank2 = sum(ranks[:n1]), sum(ranks[n1:])
    h = (12 / (n * (n + 1)) * (rank1 ** 2 / n1 + rank2 ** 2 / n2) - 3 * (n + 1)) / correction
    p = math.erfc(math.sqrt(max(0., h) / 2))
    return h, p, ranks, correction, sum(t > 1 for t in counts.values())


def self_check():
    np.testing.assert_allclose(holm_two([.04, .01]), [.04, .02])
    np.testing.assert_allclose(holm_two([.4972237262, .3143095320]), [.6286190640, .6286190640])
    for x, y, expected in [([1, 2, 3], [4, 5, 6], 27 / 7), ([1, 1, 2], [2, 3, 3], 10 / 3)]:
        h, p, _, _, _ = manual_kruskal(x, y)
        result = stats.kruskal(x, y, nan_policy='raise')
        np.testing.assert_allclose(h, expected, rtol=1e-12)
        np.testing.assert_allclose([h, p], [result.statistic, result.pvalue], rtol=1e-12)


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def verify_raw(root, participants, audit):
    lookup = {(r['source'], int(r['line'])): r for r in audit}
    values, checked = defaultdict(list), 0
    for path in sorted((root / 'inputCSV').glob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for line, raw in enumerate(csv.DictReader(stream), 2):
                if raw['phase'] != 'practice' or raw['conditionId'] not in ('C2', 'C3'):
                    continue
                n = int(raw['trialInCondition'])
                period = 'early' if 1 <= n <= 100 else 'late' if 301 <= n <= 400 else None
                if period is None:
                    continue
                row = lookup[(path.name, line)]
                assert (row['condition'], row['participant'], row['period']) == (raw['conditionId'], raw['participantId'], period)
                included = raw['countedAsTrial'].lower() == 'true' and finite(raw['coreExitMs']) is not None
                mt = finite(raw['coreMtMs'])
                valid = included and mt is not None and mt > 0
                assert included == (row['included'] == 'True') and valid == (row['valid_MT'] == 'True')
                checked += 1
                if valid:
                    values[(raw['conditionId'], raw['participantId'], period)].append(mt)
    assert checked == len(audit)
    for row in participants:
        computed = {}
        for period in ('early', 'late'):
            data = values[(row['condition'], row['participant'], period)]
            assert len(data) == int(row[period + '_valid_rows'])
            computed[period] = statistics.mean(data)
            np.testing.assert_allclose(computed[period], float(row[period + '_mean_ms']), rtol=0, atol=1e-10)
        np.testing.assert_allclose(computed['late'] - computed['early'], float(row['change_ms']), rtol=0, atol=1e-10)
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--out', type=Path, default=HERE)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.check:
        print('Known H values, tie correction, chi-square(df=1) tail and Holm checks passed. Numeric cores: 1.')
        return
    if args.root is None:
        parser.error('--root is required')
    root, out = args.root.resolve(), args.out.resolve()
    source = root / 'output/c2_c3_mt_between_20261114_152209'
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ('protocol.json', 'results.json', 'participant_means.csv')):
        raise FileExistsError('Use a new output directory. Existing results must not be overwritten.')
    inputs = input_manifest(root)
    source_manifest = json.loads((source / 'manifest.json').read_text())
    assert inputs == source_manifest['inputs'], 'CSV changed since the verified Welch analysis; stop before using previous summaries.'
    source_checks = json.loads((source / 'checksums.json').read_text())
    names = ('participant_means.csv', 'trial_audit.csv', 'exclusion_summary.csv')
    for name in names:
        assert digest(source / name) == source_checks[name], f'Previous artifact changed: {name}'
    protocol = dict(created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                    status='User-requested exploratory rank analysis after Welch results were viewed; does not replace the original tests.',
                    phase='practice', metric='coreMtMs', unit='ms', conditions=['C2', 'C3'], participants_each=10,
                    early=[1, 100], late=[301, 400], base='countedAsTrial=true AND finite coreExitMs',
                    valid_MT='finite and positive', additional_exclusions='none', imputation='none', transformation='none',
                    aggregation='Arithmetic mean of valid MT rows per participant and period; retain valid retries; participants equally weighted.',
                    endpoints=['participant change = late mean minus early mean', 'participant late mean'],
                    test='Kruskal-Wallis H, independent samples; tie correction; nondirectional; chi-square approximation with df=1.',
                    null='Identical group distributions; not generally a test of arithmetic means or medians alone.',
                    multiplicity='Holm across the two additional Kruskal-Wallis tests only.', alpha=.05,
                    interpretation='No replacement of original Welch conclusions; no claim that historical analytic multiplicity is eliminated.',
                    source=str(source.relative_to(root)), numeric_cores=1)
    save_json(out, 'protocol.json', protocol)
    print('Numeric cores: 1. Checking raw CSV against the previously verified participant summaries.', flush=True)
    participants, audit = read_csv(source / 'participant_means.csv'), read_csv(source / 'trial_audit.csv')
    groups = {c: [r for r in participants if r['condition'] == c] for c in ('C2', 'C3')}
    assert all(len(g) == 10 for g in groups.values())
    assert not {r['participant'] for r in groups['C2']} & {r['participant'] for r in groups['C3']}
    checked = verify_raw(root, participants, audit)
    results, rank_rows = [], []
    for endpoint, label, column in [('change', '変化量（後期−初期）', 'change_ms'), ('late', '後期100試行の平均', 'late_mean_ms')]:
        x, y = [[float(r[column]) for r in groups[c]] for c in ('C2', 'C3')]
        result = stats.kruskal(x, y, nan_policy='raise')
        h, p, ranks, correction, ties = manual_kruskal(x, y)
        np.testing.assert_allclose([h, p], [result.statistic, result.pvalue], rtol=1e-12, atol=1e-12)
        results.append(dict(endpoint=endpoint, label=label, n_C2=len(x), n_C3=len(y), H=float(result.statistic), df=1,
                            p_raw=float(result.pvalue), C2_mean_ms=statistics.mean(x), C3_mean_ms=statistics.mean(y),
                            C2_median_ms=statistics.median(x), C3_median_ms=statistics.median(y),
                            C2_mean_rank=statistics.mean(ranks[:len(x)]), C3_mean_rank=statistics.mean(ranks[len(x):]),
                            tie_correction=correction, tied_value_groups=ties))
        for row, rank in zip(groups['C2'] + groups['C3'], ranks):
            rank_rows.append(dict(endpoint=endpoint, condition=row['condition'], participant=row['participant'],
                                  value_ms=float(row[column]), pooled_rank=rank))
    for row, p in zip(results, holm_two([r['p_raw'] for r in results])):
        row.update(p_holm=p, significant=bool(p < .05))
    for name in names:
        with (source / name).open('rb') as src, (out / name).open('xb') as dst:
            shutil.copyfileobj(src, dst)
    save_csv(out, 'tests.csv', results)
    save_csv(out, 'participant_ranks.csv', rank_rows)
    save_json(out, 'results.json', results)
    assert inputs == input_manifest(root), 'Inputs changed during analysis.'
    save_json(out, 'manifest.json', dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, numeric_cores=1,
              inputs=inputs, raw_rows_checked=checked, participants_checked=len(participants),
              source_manifest_sha256=digest(source / 'manifest.json'), source_files={name: digest(source / name) for name in names},
              script_sha256=digest(Path(__file__)), protocol_sha256=digest(out / 'protocol.json'),
              checks=['input hashes match structurally validated dataset', 'previous artifact hashes verified',
                      'raw CSV selection and participant mean recomputation', 'known H with and without ties',
                      'SciPy versus independent midranks, H and chi-square(df=1) tail', 'two-test Holm known values', 'input hashes unchanged']))
    lines = ['C2とC3のCoreMT Kruskal-Wallis追加探索分析', '',
             '前回のWelch検定結果を閲覧した後に依頼された追加の探索的分析。既存の主分析・副次分析を置き換えない。',
             '対象はpractice、C2・C3各10名。初期1〜100、後期301〜400試行。参加者別の算術平均を使い、変化量は後期−初期。',
             '基本採用はcountedAsTrial=trueかつcoreExitMs欠損なし。MTは有限かつ正。success=falseのみでは除外しない。追加除外・変換・補完なし。',
             'Kruskal-Wallis H統計量に同順位補正を適用し、自由度1のχ²近似で方向を限定しないp値を求めた。今回の2検定にHolm補正（5%）。',
             '算術平均の差を検定したものではない。分布の形・広がりが同じという条件なしに、中央値差だけの検定と解釈しない。', '']
    for row in results:
        lines.extend([row['label'], f"H(1)={row['H']:.8f}, 未補正p={row['p_raw']:.10g}, Holm補正後p={row['p_holm']:.10g}",
                      f"平均順位 C2={row['C2_mean_rank']:.3f}, C3={row['C3_mean_rank']:.3f}。同順位値のグループ数={row['tied_value_groups']}。",
                      f"記述値: 参加者値の平均 C2={row['C2_mean_ms']:.3f}, C3={row['C3_mean_ms']:.3f} ms; 中央値 C2={row['C2_median_ms']:.3f}, C3={row['C3_median_ms']:.3f} ms。",
                      'Holm補正後5%で有意。' if row['significant'] else 'Holm補正後5%では条件間差の十分な証拠は得られなかった。', ''])
    lines.extend(['元CSVの全4168対象記録と20名の前後平均・変化量を照合し、前回と一致。',
                  '基本除外はC2 124件、C3 54件。採用はC2 1996件、C3 1994件。参加者除外0、基本採用後MT計算不能0。',
                  '非有意は同等性の証明ではない。今回の補正は2件の追加検定に限られ、過去の方法選択・繰り返し検定全体の誤り率を保証しない。',
                  'p値はχ²近似であり、正確な置換p値ではない。参加者単位の独立性が必要であり、個々の試行を独立標本としていない。',
                  '入力CSV、UI、過去の結果は変更していない。使用コア数1。',
                  '方法資料: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html'])
    with (out / 'report_ja.txt').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print('Raw rows and means verified:', checked, len(participants))


if __name__ == '__main__':
    main()
