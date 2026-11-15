import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/steering_between_matplotlib')
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
from analysis_snapshot import load_rows

HERE = Path(__file__).resolve().parent
COLORS = {'C2': 'tab:orange', 'C3': 'tab:green'}
PERIODS = {'early': (1, 100), 'late': (301, 400)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_manifest(root):
    return [dict(file=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in sorted((root / 'inputCSV').glob('*.csv'))]


def save_json(out, name, value):
    with (out / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def save_csv(out, name, rows):
    with (out / name).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(np.asarray(values)[order] * np.arange(len(values), 0, -1)))
    return adjusted


def welch(x, y):
    n1, n2 = len(x), len(y)
    mean1, mean2 = statistics.mean(x), statistics.mean(y)
    v1, v2 = statistics.variance(x) / n1, statistics.variance(y) / n2
    se = math.sqrt(v1 + v2)
    if se == 0:
        raise ValueError('Zero standard error requires explicit handling.')
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    contrast = mean1 - mean2
    t = contrast / se
    p = 2 * stats.t.sf(abs(t), df)
    critical = stats.t.ppf(.975, df)
    result = stats.ttest_ind(x, y, equal_var=False, alternative='two-sided')
    ci = result.confidence_interval(.95)
    np.testing.assert_allclose([result.statistic, result.df, result.pvalue, ci.low, ci.high],
                              [t, df, p, contrast - critical * se, contrast + critical * se], rtol=1e-11, atol=1e-11)
    return dict(n_C2=n1, n_C3=n2, C2_mean_ms=mean1, C3_mean_ms=mean2,
                C2_sd_ms=statistics.stdev(x), C3_sd_ms=statistics.stdev(y), difference_ms=contrast,
                se_ms=se, ci95_low_ms=float(ci.low), ci95_high_ms=float(ci.high),
                t=float(result.statistic), df=float(result.df), p_raw=float(result.pvalue))


def self_check():
    np.testing.assert_allclose(holm([.04, .01]), [.04, .02])
    np.testing.assert_allclose(holm([.01, .01]), [.02, .02])
    np.testing.assert_allclose(holm([.8, .9]), [1, 1])
    result = welch([1., 2., 3.], [4., 5., 6.])
    np.testing.assert_allclose([result['difference_ms'], result['t'], result['df']], [-3, -3 / math.sqrt(2 / 3), 4])
    reverse = welch([4., 5., 6.], [1., 2., 3.])
    np.testing.assert_allclose([reverse['t'], reverse['p_raw'], reverse['ci95_low_ms']],
                              [-result['t'], result['p_raw'], -result['ci95_high_ms']])


def collect(rows):
    members = {c: sorted({r['participant'] for r in rows if r['phase'] == 'practice' and r['condition'] == c}) for c in COLORS}
    if any(len(p) != 10 for p in members.values()) or set(members['C2']) & set(members['C3']):
        raise ValueError(f'Expected disjoint groups of 10: {members}')
    values, numbers, counts = defaultdict(list), defaultdict(list), defaultdict(Counter)
    audit = []
    for row in rows:
        c, pid, n = row['condition'], row['participant'], row['n']
        if row['phase'] != 'practice' or c not in COLORS:
            continue
        period = next((p for p, (low, high) in PERIODS.items() if low <= n <= high), None)
        if period is None:
            continue
        reasons = []
        if not row['countedAsTrial']:
            reasons.append('countedAsTrial=false')
        if row['coreExitMs'] is None:
            reasons.append('coreExitMs missing')
        mt = row['coreMtMs']['MT']
        included, valid = not reasons, not reasons and mt is not None
        audit.append(dict(condition=c, participant=pid, period=period, source=row['source'], line=row['line'],
                          N=n, attempt=row['attempt'], success=row['success'], countedAsTrial=row['countedAsTrial'],
                          coreExitMs=row['coreExitMs'], core_MT_ms=mt, included=included, valid_MT=valid,
                          reason='; '.join(reasons) or ('MT unavailable' if mt is None else '')))
        count = counts[(c, period)]
        count['records'] += 1
        count['counted_false'] += not row['countedAsTrial']
        count['core_exit_missing'] += row['coreExitMs'] is None
        count['both_reasons'] += len(reasons) == 2
        count['excluded_unique'] += not included
        count['included_rows'] += included
        count['invalid_MT_included'] += included and not valid
        count['valid_MT_rows'] += valid
        count['included_success_false'] += included and not row['success']
        if valid:
            values[(c, pid, period)].append(mt)
            numbers[(c, pid, period)].append(n)
    participants = []
    for c, pids in members.items():
        for pid in pids:
            record = dict(condition=c, participant=pid)
            for period in PERIODS:
                data = values[(c, pid, period)]
                if not data:
                    raise ValueError(f'No valid MT: {c}, {pid}, {period}')
                ns = numbers[(c, pid, period)]
                low, high = PERIODS[period]
                record.update({period + '_mean_ms': float(np.mean(data)), period + '_valid_rows': len(data),
                               period + '_unique_N': len(set(ns)),
                               period + '_missing_N': ';'.join(map(str, sorted(set(range(low, high + 1)) - set(ns))))})
            record['change_ms'] = record['late_mean_ms'] - record['early_mean_ms']
            participants.append(record)
    return participants, audit, [dict(condition=c, period=p, **counts[(c, p)]) for c in COLORS for p in PERIODS]


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def verify_raw(root, participants, audit):
    records = {(r['source'], r['line']): r for r in audit}
    values = defaultdict(list)
    checked = 0
    for path in sorted((root / 'inputCSV').glob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for line, raw in enumerate(csv.DictReader(stream), 2):
                key = (path.name, line)
                if key not in records:
                    continue
                row = records[key]
                included = raw['countedAsTrial'].lower() == 'true' and finite(raw['coreExitMs']) is not None
                mt = finite(raw['coreMtMs'])
                valid = included and mt is not None and mt > 0
                assert included == row['included'] and valid == row['valid_MT']
                assert raw['participantId'] == row['participant'] and int(raw['trialInCondition']) == row['N']
                if valid:
                    values[(row['condition'], row['participant'], row['period'])].append(mt)
                checked += 1
    assert checked == len(audit)
    for row in participants:
        for period in PERIODS:
            data = values[(row['condition'], row['participant'], period)]
            assert len(data) == row[period + '_valid_rows']
            np.testing.assert_allclose(statistics.mean(data), row[period + '_mean_ms'], rtol=0, atol=1e-10)
    return checked


def draw_diagnostics(out, participants, metric, stamp):
    column, label = ('change_ms', 'Late minus early core MT (ms)') if metric == 'change' else ('late_mean_ms', 'Late mean core MT (ms)')
    fig = plt.figure(figsize=(16, 10), dpi=100, layout='constrained')
    grid = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(grid[0, :])
    for i, c in enumerate(COLORS):
        values = [r[column] for r in participants if r['condition'] == c]
        offsets = np.linspace(-.12, .12, len(values))
        ax.scatter(i + offsets, values, color=COLORS[c], s=65, alpha=.85)
        ax.scatter(i, np.mean(values), marker='D', color='black', s=80, zorder=3,
                   label='Group mean' if i == 0 else None)
        qq = fig.add_subplot(grid[1, i])
        (theoretical, ordered), (slope, intercept, _) = stats.probplot(values, dist='norm')
        qq.scatter(theoretical, ordered, color=COLORS[c], s=55)
        qq.plot(theoretical, intercept + slope * theoretical, color='gray')
        qq.set(xlabel='Theoretical normal quantiles', ylabel=label, title=f'{c}: Q-Q plot (n={len(values)})')
        qq.grid(alpha=.2)
    ax.set(xticks=[0, 1], xticklabels=['C2 (n=10)', 'C3 (n=10)'], ylabel=label,
           title='Each point = one participant; black diamonds = equal-weight means', xlim=(-.45, 1.45))
    if metric == 'change':
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.grid(axis='y', alpha=.2)
    ax.legend()
    fig.suptitle(f'Practice core MT | {"Primary: change" if metric == "change" else "Secondary: late mean"}\n'
                 'Early 1-100; late 301-400 | countedAsTrial=true & finite coreExitMs | No outlier / boundary exclusion', fontsize=14)
    name = f'core_MT_{metric}_C2-C3_practice_1-100_vs_301-400_base-only_{stamp}.png'
    with (out / name).open('xb') as stream:
        fig.savefig(stream, format='png', dpi=100)
    plt.close(fig)
    return name


def report_text(results, summaries, diagnostics):
    lines = ['C2とC3のCoreMT条件間比較 検定計画と結果', '',
             '研究目的は初期から後期への平均変化量の条件間差（主分析）、および後期平均の条件間差（副次分析）の検証である。',
             '20名をC2・C3に無作為割り当て（各10名）。各条件で400試行、100試行ごとに1分休憩、教示は「早く正確に」（ユーザー申告）。',
             '対象はpracticeのcoreMtMs（ms）。初期は元の試行番号1〜100、後期は301〜400。両端を含む。',
             '参加者・区間ごとの有効CoreMT行の算術平均を計算し、各参加者を等重みで扱う。変化量は後期−初期。',
             '基本採用条件はcountedAsTrial=trueかつcoreExitMsが有限。success=falseのみでは除外しない。',
             'CoreMTは有限かつ正の値を有効とする。追加の外れ値除外・境界周辺除外・変換・欠損補完なし。元の番号を保持する。',
             '主分析・副次分析とも対応のない両側Welchのt検定。帰無仮説は条件間母平均差=0、対立仮説は差≠0。',
             '比較方向はC2−C3。両検定を一つの検定群としてHolm補正し、家族内有意水準5%。主分析の有意性に関わらず両結果を報告する。',
             '95%信頼区間は個別・未補正のWelch区間。分散は条件別に推定し、等分散を仮定しない。',
             '参加者間の独立性と各群の参加者値の分布を前提とする。同一参加者内の試行を独立参加者として扱わない。',
             '正規性・等分散性検定による検定の切り替えは行わず、追加の補助検定は設けない。全点と条件別Q-Q図を診断に用いる。', '', '結果']
    for r in results:
        lines.extend([f"{r['role']}：{r['label']}",
                      f"C2 n={r['n_C2']}、平均={r['C2_mean_ms']:.6f} ms、参加者間SD={r['C2_sd_ms']:.6f} ms",
                      f"C3 n={r['n_C3']}、平均={r['C3_mean_ms']:.6f} ms、参加者間SD={r['C3_sd_ms']:.6f} ms",
                      f"条件間差（C2−C3）={r['difference_ms']:.6f} ms、95% CI [{r['ci95_low_ms']:.6f}, {r['ci95_high_ms']:.6f}] ms",
                      f"Welch t={r['t']:.8f}, df={r['df']:.8f}, 未補正p={r['p_raw']:.10g}, Holm補正後p={r['p_holm']:.10g}",
                      'Holm補正後5%で有意。' if r['significant'] else 'Holm補正後5%では、条件間差の十分な統計的証拠は得られなかった。', ''])
    lines.append('除外記録（理由は重複するため単純合計しない）')
    for r in summaries:
        lines.append(f"{r['condition']} {r['period']}: 対象={r['records']}, 基本除外={r['excluded_unique']}, 有効MT={r['valid_MT_rows']}, "
                     f"counted=false={r['counted_false']}, coreExit欠損={r['core_exit_missing']}, 両理由={r['both_reasons']}, "
                     f"採用後MT計算不能={r['invalid_MT_included']}, success=false採用={r['included_success_false']}")
    lines.extend(['', '解釈の範囲',
                  '非有意は同等性や差がないことの証明ではない。CIが許容する差の範囲と各群10名という精度の限界を併記する。',
                  '条件内の短縮が両群で有意であったことから、条件間の変化量差を推論しない。今回の主分析がその直接比較である。',
                  '後期平均の差は練習による変化量の差とは異なる。初期1〜100試行も割り当て条件下で測定されており、介入前ベースラインではない。',
                  '本結果だけで保持・転移を伴う学習や、全参加者の改善を主張しない。誤り率・正確性の維持は今回検定していない。',
                  '結果閲覧後の再解析であり、収集前の事前登録や独立した確認的検証ではない。',
                  '今回の2検定へのHolm補正は過去の分析選択全体を補正するものではなく、既存の検定群や結果を置き換えない。',
                  '入力CSV・通常UI・過去の出力を変更しない。数値計算は1コア。', '',
                  '検証: 小データによるWelch・Holm既知値、SciPyと手計算のt・自由度・p・CI、元CSV別読み取りによる採用判定・参加者平均を照合。',
                  '診断図: ' + ', '.join(diagnostics),
                  '方法資料: https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm',
                  '多重性資料: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--out', type=Path, default=HERE)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.check:
        print('Known-value Welch/Holm and contrast direction checks passed. Numeric cores: 1.')
        return
    if args.root is None:
        parser.error('--root is required for analysis')
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise FileExistsError('Choose a new output directory; previous results must not be overwritten.')
    stamp = datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    protocol = dict(created_jst=stamp, status='Retrospective reanalysis after prior results were viewed; not preregistered.',
                    phase='practice', metric='coreMtMs', unit='ms', conditions=['C2', 'C3'], participants_each=10,
                    early=[1, 100], late=[301, 400], base='countedAsTrial=true AND finite coreExitMs',
                    valid_MT='finite and positive', success_exclusion=False, outliers='none', boundary_exclusion=False,
                    transformation='none', imputation='none', aggregation='Arithmetic mean of valid rows per participant and period; equal participant weight; retain valid retries.',
                    primary='Between-condition difference in participant changes (late minus early)',
                    secondary='Between-condition difference in participant late means', contrast='C2 minus C3',
                    test='Independent two-sided Welch t-test for each endpoint', alpha=.05,
                    multiplicity='Holm across exactly the primary and secondary tests; report both regardless of primary significance.',
                    confidence='Pointwise unadjusted 95% Welch CI', diagnostics='All participant points and within-group Q-Q plots for both endpoints; no test switching.',
                    supplementary_tests='none', missing='Stop if a participant has no valid MT in either period; report missing records without imputation.', numeric_cores=1)
    save_json(out, 'protocol.json', protocol)
    inputs = input_manifest(root)
    print('Numeric cores: 1. Validating CSV and recomputing participant means.', flush=True)
    rows = load_rows(root / 'inputCSV')
    participants, audit, summaries = collect(rows)
    checked = verify_raw(root, participants, audit)
    prior = root / 'output/c2_c3_mt_paired_20261114/rerun_20261114_144205'
    prior_check = 'not available or input hashes differ'
    if (prior / 'manifest.json').exists() and json.loads((prior / 'manifest.json').read_text())['inputs'] == inputs:
        with (prior / 'participant_means.csv').open(encoding='utf-8-sig', newline='') as stream:
            old = {(r['condition'], r['participant']): r for r in csv.DictReader(stream)}
        for r in participants:
            old_row = old[(r['condition'], r['participant'])]
            fields = ('early_mean_ms', 'late_mean_ms', 'change_ms')
            np.testing.assert_allclose([r[k] for k in fields], [float(old_row[k]) for k in fields], rtol=0, atol=1e-10)
        prior_check = 'all 20 participant early, late and change means match prior paired analysis'
    results = []
    for role, metric, label, column in [('主分析', 'change', '変化量の条件間差', 'change_ms'), ('副次分析', 'late', '後期平均の条件間差', 'late_mean_ms')]:
        groups = [[r[column] for r in participants if r['condition'] == c] for c in COLORS]
        results.append(dict(role=role, endpoint=metric, label=label, **welch(*groups)))
    for r, p in zip(results, holm([r['p_raw'] for r in results])):
        r.update(p_holm=float(p), significant=bool(p < .05))
    descriptive = []
    for c in COLORS:
        for endpoint, column in [('early', 'early_mean_ms'), ('late', 'late_mean_ms'), ('change', 'change_ms')]:
            values = [r[column] for r in participants if r['condition'] == c]
            descriptive.append(dict(condition=c, endpoint=endpoint, n=len(values), mean_ms=statistics.mean(values),
                                    sd_ms=statistics.stdev(values), min_ms=min(values), max_ms=max(values)))
    for name, data in [('participant_means.csv', participants), ('trial_audit.csv', audit), ('exclusion_summary.csv', summaries),
                       ('descriptives.csv', descriptive), ('tests.csv', results)]:
        save_csv(out, name, data)
    save_json(out, 'results.json', results)
    diagnostics = [draw_diagnostics(out, participants, endpoint, stamp) for endpoint in ('change', 'late')]
    assert inputs == input_manifest(root), 'Inputs changed during analysis.'
    save_json(out, 'manifest.json', dict(created_jst=stamp, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
              matplotlib=matplotlib.__version__, numeric_cores=1, inputs=inputs, loaded_rows=len(rows), selected_rows=len(audit),
              included_rows=sum(r['included'] for r in audit), raw_rows_checked=checked, prior_check=prior_check,
              script_sha256=digest(Path(__file__)), analysis_snapshot_sha256=digest(HERE / 'analysis_snapshot.py'),
              protocol_sha256=digest(out / 'protocol.json'), checks=['Welch/Holm known values and contrast direction',
              'SciPy versus explicit Welch statistic, degrees of freedom, p and CI', 'independent CSV scalar audit and participant means',
              'input hashes unchanged']))
    report = report_text(results, summaries, diagnostics)
    with (out / 'report_ja.txt').open('x', encoding='utf-8') as stream:
        stream.write(report)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print('Raw rows verified:', checked, '| Prior check:', prior_check)


if __name__ == '__main__':
    main()
