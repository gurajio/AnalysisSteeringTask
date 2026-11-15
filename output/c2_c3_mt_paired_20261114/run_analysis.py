import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/c2_c3_mt_paired_matplotlib')
import csv
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from analysis import load_rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(name, value):
    with (HERE / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def save_csv(name, rows):
    with (HERE / name).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm(values):
    values = np.asarray(values)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def signed_rank(delta):
    nonzero = delta[delta != 0]
    if not len(nonzero):
        return 0., 1., 1., 1
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    totals = np.asarray(list(itertools.product((0, 1), repeat=len(nonzero)))) @ ranks
    exact = min(1., 2 * min(np.mean(totals <= positive), np.mean(totals >= positive)))
    tied = len(set(np.abs(nonzero))) != len(nonzero)
    method = stats.PermutationMethod(n_resamples=np.inf) if tied or len(nonzero) != len(delta) else 'exact'
    result = stats.wilcoxon(delta, alternative='two-sided', zero_method='wilcox', method=method)
    np.testing.assert_allclose(result.pvalue, exact, rtol=0, atol=1e-14)
    return float(result.statistic), float(result.pvalue), float(exact), len(totals)


def main():
    if (HERE / 'results.json').exists():
        raise FileExistsError('Existing results must not be overwritten; use a new output directory.')
    np.testing.assert_allclose(holm([.04, .01]), [.04, .02])
    stamp = datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    inputs = [dict(file=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in sorted((ROOT / 'inputCSV').glob('*.csv'))]
    protocol = dict(date='2026-11-14 Asia/Tokyo', status='Retrospective analysis after earlier results were viewed; not preregistered.',
                    phase='practice', conditions=['C2', 'C3'], participants_each=10, metric='coreMtMs', unit='ms',
                    early=[1, 100], late=[301, 400], base='countedAsTrial=true AND finite coreExitMs',
                    boundary_exclusion=False, mt_outlier_exclusion='none', success_exclusion=False,
                    aggregation='Mean of valid positive finite MT rows within each participant and period; equal participant weights. Retain retries.',
                    contrast='late minus early', primary='paired t, two-sided', supplementary='Wilcoxon signed-rank, two-sided, exact sign enumeration',
                    multiplicity='Holm across C2/C3 separately for each method; supplementary results cannot replace the primary results.',
                    alpha=.05, confidence='95% pointwise unadjusted t CI for mean change',
                    diagnostics='Participant differences, Q-Q plots, Shapiro-Wilk and sample skewness; no test switching or participant deletion.',
                    missing='No imputation; report invalid MT rows; stop if any participant has no valid MT in either period.',
                    zero_difference='Wilcoxon drops zero differences; record count. All zeros give p=1.', numeric_cores=1)
    save_json('protocol.json', protocol)
    print('Numeric cores: 1; reading and validating CSV once; no model fitting.', flush=True)
    rows = load_rows(ROOT / 'inputCSV')
    selected = [r for r in rows if r['phase'] == 'practice' and r['condition'] in ('C2', 'C3') and (1 <= r['n'] <= 100 or 301 <= r['n'] <= 400)]
    audit, participants = [], []
    groups = {c: sorted({r['participant'] for r in selected if r['condition'] == c}) for c in ('C2', 'C3')}
    assert all(len(g) == 10 for g in groups.values()), groups
    assert not set(groups['C2']) & set(groups['C3'])
    for condition, members in groups.items():
        for pid in members:
            record = dict(condition=condition, participant=pid)
            for period, low, high in [('early', 1, 100), ('late', 301, 400)]:
                window = [r for r in selected if r['condition'] == condition and r['participant'] == pid and low <= r['n'] <= high]
                valid, included = [], []
                for row in window:
                    reasons = []
                    if not row['countedAsTrial']:
                        reasons.append('countedAsTrial=false')
                    if row['coreExitMs'] is None:
                        reasons.append('coreExitMs missing')
                    mt = row['coreMtMs']['MT']
                    if not reasons:
                        included.append(row)
                        if mt is not None:
                            valid.append(mt)
                    audit.append(dict(condition=condition, participant=pid, period=period, source=row['source'], line=row['line'],
                                      N=row['n'], attempt=row['attempt'], core_MT_ms=mt, included=not reasons,
                                      valid_MT=not reasons and mt is not None, reason='; '.join(reasons) or ('MT unavailable' if mt is None else '')))
                assert valid, (condition, pid, period, 'No valid MT')
                record.update({period + '_mean_ms': float(np.mean(valid)), period + '_valid_rows': len(valid),
                               period + '_excluded_rows': len(window) - len(included), period + '_invalid_MT_rows': len(included) - len(valid),
                               period + '_unique_N': len({r['n'] for r in included})})
            record['change_ms'] = record['late_mean_ms'] - record['early_mean_ms']
            participants.append(record)
    results, arrays = [], {}
    for condition in groups:
        members = [r for r in participants if r['condition'] == condition]
        early = np.array([r['early_mean_ms'] for r in members])
        late = np.array([r['late_mean_ms'] for r in members])
        delta = late - early
        arrays[condition] = (early, late, delta)
        assert np.std(delta, ddof=1) > 0, 'Degenerate paired t-test requires explicit handling.'
        test = stats.ttest_rel(late, early, alternative='two-sided')
        se = np.std(delta, ddof=1) / math.sqrt(len(delta))
        manual_t = np.mean(delta) / se
        np.testing.assert_allclose([test.statistic, test.pvalue], [manual_t, 2 * stats.t.sf(abs(manual_t), len(delta)-1)], rtol=1e-12)
        ci = test.confidence_interval(.95)
        w, wp, ep, combinations = signed_rank(delta)
        normal = stats.shapiro(delta)
        results.append(dict(condition=condition, n=len(delta), early_mean_ms=float(early.mean()), late_mean_ms=float(late.mean()),
                            change_mean_ms=float(delta.mean()), change_sd_ms=float(delta.std(ddof=1)), change_median_ms=float(np.median(delta)),
                            ci95_low_ms=float(ci.low), ci95_high_ms=float(ci.high), t=float(test.statistic), df=int(test.df), t_p=float(test.pvalue),
                            wilcoxon_W=w, wilcoxon_p=wp, enumerated_p=ep, sign_combinations=combinations,
                            zero_differences=int(np.sum(delta == 0)), absolute_ties=len(delta) - len(np.unique(np.abs(delta))),
                            negative_differences=int(np.sum(delta < 0)), shapiro_W=float(normal.statistic), shapiro_p=float(normal.pvalue),
                            difference_skew=float(stats.skew(delta, bias=False)), difference_min_ms=float(delta.min()), difference_max_ms=float(delta.max())))
    for prefix in ('t', 'wilcoxon'):
        adjusted = holm([r[prefix + '_p'] for r in results])
        for record, value in zip(results, adjusted):
            record[prefix + '_p_holm'] = float(value)
            record[prefix + '_significant'] = bool(value < .05)
    prior = ROOT / 'output/c2_c3_validation_20261113'
    prior_manifest = json.loads((prior / 'manifest.json').read_text())
    unchanged = inputs == prior_manifest['inputs']
    if unchanged:
        with (prior / 'participant_summary.csv').open(encoding='utf-8-sig', newline='') as stream:
            old = {(r['condition'], r['participant']): r for r in csv.DictReader(stream) if r['variant'] == 'base' and r['metric'] == 'MT'}
        assert len(old) == len(participants)
        for row in participants:
            previous = old[(row['condition'], row['participant'])]
            np.testing.assert_allclose([row['early_mean_ms'], row['late_mean_ms']], [float(previous['early']), float(previous['late'])], rtol=0, atol=1e-10)
    after = [dict(file=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in sorted((ROOT / 'inputCSV').glob('*.csv'))]
    assert inputs == after, 'Inputs changed during analysis.'
    save_csv('participant_means.csv', participants)
    save_csv('trial_audit.csv', audit)
    save_csv('tests.csv', results)
    save_json('results.json', results)
    save_json('manifest.json', dict(created_jst=stamp, python=sys.version, numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                                    numeric_cores=1, inputs=inputs, script_sha256=digest(Path(__file__)), analysis_sha256=digest(ROOT / 'analysis.py'),
                                    protocol_sha256=digest(HERE / 'protocol.json'), loaded_rows=len(rows), selected_rows=len(selected),
                                    included_rows=sum(r['included'] for r in audit), invalid_MT_rows=sum(r['included'] and not r['valid_MT'] for r in audit),
                                    previous_inputs_identical=unchanged, previous_participant_means_match=unchanged,
                                    checks=['paired t versus closed-form calculation', 'Wilcoxon versus all sign combinations', 'Holm known values', 'input hashes unchanged']))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=100, layout='constrained')
    for column, condition in enumerate(groups):
        early, late, delta = arrays[condition]
        color = {'C2': 'tab:orange', 'C3': 'tab:green'}[condition]
        ax = axes[0, column]
        for e, l in zip(early, late):
            ax.plot([0, 1], [e, l], marker='o', color=color, alpha=.5, linewidth=1)
        ax.plot([0, 1], [early.mean(), late.mean()], color='black', linewidth=3, marker='D', label='Equal-weight mean')
        ax.set(xticks=[0, 1], xticklabels=['Early: trials 1-100', 'Late: trials 301-400'], ylabel='Participant mean core MT (ms)', title=f'{condition}: paired means (n=10)')
        ax.legend()
        ax.grid(alpha=.2)
        ax = axes[1, column]
        (theoretical, ordered), (slope, intercept, correlation) = stats.probplot(delta, dist='norm')
        ax.scatter(theoretical, ordered, color=color)
        ax.plot(theoretical, intercept + slope * theoretical, color='gray')
        ax.set(xlabel='Theoretical normal quantiles', ylabel='Late minus early core MT (ms)', title=f'{condition}: Q-Q plot of participant changes')
        ax.grid(alpha=.2)
    fig.suptitle('Practice core MT | countedAsTrial=true & finite coreExitMs\nNo MT-outlier or boundary exclusion | Each line / point = one participant', fontsize=14)
    image_name = f'core_MT_paired_QQ_C2-C3_practice_1-100_vs_301-400_base-only_{stamp}.png'
    fig.savefig(HERE / image_name, dpi=100)
    plt.close(fig)
    report = ['C2・C3 条件内のcore MT前後比較', '', 'practice、前半1〜100試行、後半301〜400試行。基本除外あり、MT外れ値・境界周辺除外なし。',
              '参加者ごとに有効試行の算術平均を計算。差は後半−前半。各群10名、参加者を等重みとする。',
              '主解析は対応のある両側t検定。補助解析は両側Wilcoxon符号付き順位検定。各方法でC2・C3の2検定を別々にHolm補正（5%）。',
              '平均差の95%信頼区間は個別・未補正。Wilcoxonは同じ平均差の仮説を検定するものではない。', '']
    for r in results:
        report.extend([f"{r['condition']}（n={r['n']}）", f"前半平均 {r['early_mean_ms']:.3f} ms、後半平均 {r['late_mean_ms']:.3f} ms",
                       f"平均差 {r['change_mean_ms']:.3f} ms、95% CI [{r['ci95_low_ms']:.3f}, {r['ci95_high_ms']:.3f}] ms",
                       f"t({r['df']})={r['t']:.6f}, p={r['t_p']:.9g}, Holm p={r['t_p_holm']:.9g}",
                       f"Wilcoxon W={r['wilcoxon_W']:.3f}, exact p={r['wilcoxon_p']:.9g}, Holm p={r['wilcoxon_p_holm']:.9g}（{r['sign_combinations']}通りを全列挙して照合）",
                       f"Shapiro–Wilk W={r['shapiro_W']:.6f}, p={r['shapiro_p']:.6f}、標本歪度={r['difference_skew']:.3f}",
                       f"差の範囲 [{r['difference_min_ms']:.3f}, {r['difference_max_ms']:.3f}] ms、短縮した参加者 {r['negative_differences']}/10、差0={r['zero_differences']}、絶対値同順位={r['absolute_ties']}", ''])
    report.extend(['解釈上の注意', '正規性検定は診断用であり、非有意でも正規性を証明しない。10名では分布形状の確認に限界がある。Wilcoxonにも差の対称性の仮定がある。',
                   '本解析のHolm補正は今回指定したMTの条件内2比較に対するもの。過去の多数の解析全体への補正ではなく、事後的な再解析として扱う。',
                   'C2とC3の改善量の大小は今回検定していない。前後変化だけから練習の因果効果を単独で確定しない。',
                   f'入力CSVの変更なし。既存解析と入力一致={unchanged}。一致時は20名の前後平均も照合済み。',
                   f'対象行={len(audit)}、採用行={sum(r["included"] for r in audit)}、採用行内MT計算不能={sum(r["included"] and not r["valid_MT"] for r in audit)}。',
                   f'図: {image_name}', '使用コア数: 1。数式・全列挙・Holm既知値の検証を実施。'])
    with (HERE / 'report_ja.txt').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(report) + '\n')
    print('\n'.join(report), flush=True)


if __name__ == '__main__':
    main()
