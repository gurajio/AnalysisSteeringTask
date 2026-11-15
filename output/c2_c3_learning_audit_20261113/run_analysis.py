import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
from scipy import stats
from scipy.linalg import helmert

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from analysis import aggregate, analyze, load_rows, mt_outliers

METRICS = ('MT', 'ER', 'TPe')
UNITS = {'MT': 'ms', 'ER': 'percentage points', 'TPe': 's^-1'}


def holm(values):
    values = np.asarray(values)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def contrast(x, y, paired=False):
    x, y = np.asarray(x), np.asarray(y)
    result = stats.ttest_rel(x, y) if paired else stats.ttest_ind(x, y, equal_var=False)
    difference = float(np.mean(x) - np.mean(y))
    if paired:
        se = np.std(x - y, ddof=1) / np.sqrt(len(x))
        df = len(x) - 1
    else:
        vx, vy = np.var(x, ddof=1) / len(x), np.var(y, ddof=1) / len(y)
        se = np.sqrt(vx + vy)
        df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    assert se > 0 and np.isfinite(result.statistic)
    np.testing.assert_allclose([result.statistic, result.df], [difference / se, df])
    ci = result.confidence_interval(.95)
    np.testing.assert_allclose([ci.low, ci.high], difference + stats.t.ppf([.025, .975], df) * se)
    return dict(method='paired t / two-level RMANOVA' if paired else 'Welch t', estimate=difference,
                ci_low=float(ci.low), ci_high=float(ci.high), statistic=float(result.statistic),
                F=float(result.statistic ** 2) if paired else None, df1=1 if paired else None,
                df2=float(result.df), p_raw=float(result.pvalue), mean_c2=float(x.mean()), mean_c3=float(y.mean()))


def profile_test(y):
    a, n, b = y.shape
    assert a == 2 and b == 4
    grand = y.mean()
    cell = y.mean(axis=1)
    condition = y.mean(axis=(1, 2))
    block = y.mean(axis=(0, 1))
    subject = y.mean(axis=2)
    ss = n * np.sum((cell - condition[:, None] - block + grand) ** 2)
    error = np.sum((y - subject[:, :, None] - cell[:, None, :] + condition[:, None, None]) ** 2)
    df1, df2 = (a - 1) * (b - 1), a * (n - 1) * (b - 1)
    covariance = sum(np.cov(g, rowvar=False, ddof=1) * (n - 1) for g in y) / (a * (n - 1))
    center = np.eye(b) - np.ones((b, b)) / b
    centered = center @ covariance @ center
    epsilon = float(np.clip(np.trace(centered) ** 2 / ((b - 1) * np.trace(centered @ centered)), 1 / (b - 1), 1))
    f = float((ss / df1) / (error / df2))
    orthogonal = y @ helmert(b).T
    np.testing.assert_allclose(error, sum(np.sum((g - g.mean(axis=0)) ** 2) for g in orthogonal))
    np.testing.assert_allclose(ss, n * sum(np.sum((g.mean(axis=0) - orthogonal.mean(axis=(0, 1))) ** 2) for g in orthogonal))
    return dict(method='mixed ANOVA interaction, always GG corrected', estimate=None, ci_low=None, ci_high=None,
                statistic=f, F=f, df1=df1 * epsilon, df2=df2 * epsilon, p_raw=float(stats.f.sf(f, df1 * epsilon, df2 * epsilon)),
                epsilon=epsilon, partial_eta_squared=float(ss / (ss + error)), SS_effect=float(ss), SS_error=float(error))


def variance_test(x, y):
    result = stats.levene(x, y, center='median')
    check = stats.f_oneway(np.abs(x - np.median(x)), np.abs(y - np.median(y)))
    np.testing.assert_allclose([result.statistic, result.pvalue], [check.statistic, check.pvalue])
    return dict(method='Brown-Forsythe median-centered', estimate=float(np.std(x, ddof=1) / np.std(y, ddof=1)),
                ci_low=None, ci_high=None, statistic=float(result.statistic), F=float(result.statistic),
                df1=1, df2=len(x) + len(y) - 2, p_raw=float(result.pvalue))


def fit_line(group, metric):
    valid = [r for r in group if r['coreMtMs'][metric] is not None]
    n = np.array([r['n'] for r in valid], dtype=float)
    values = np.array([r['coreMtMs'][metric] for r in valid], dtype=float)
    assert len(values) >= 3 and len(set(n)) >= 2
    design = np.column_stack((np.ones(len(n)), n - n.mean()))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ coefficients
    sd = float(np.sqrt(np.sum(residual ** 2) / (len(values) - 2)))
    assert sd > 0
    np.testing.assert_allclose(coefficients[1], stats.linregress(n, values).slope)
    return float(coefficients[1]), sd


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    np.testing.assert_allclose(holm([.01, .04, .03, .8]), [.04, .09, .09, .8])
    np.testing.assert_allclose(holm([.001] + [.5] * 27)[0], .028)
    paired = contrast([2., 4., 7.], [1., 2., 4.], True)
    np.testing.assert_allclose(paired['F'], 12.)
    np.testing.assert_allclose(contrast([1., 2., 3.], [4., 5., 6.])['statistic'] ** 2, 13.5)
    h = helmert(4)
    residual = h.T @ h
    y = np.stack((10 + residual + h[0], 10 + residual - h[0]))
    p = profile_test(y)
    np.testing.assert_allclose([p['F'], p['epsilon'], p['df1'], p['df2']], [8, 1, 3, 18])
    print('Known-value tests passed.', flush=True)


def main():
    self_test()
    protocol = json.loads((HERE / 'protocol.json').read_text())
    files = sorted((ROOT / 'inputCSV').glob('*.csv'))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    prior = json.loads((ROOT / 'output/c2_c3_all_metrics_100_anova_20261113/results.json').read_text())
    assert manifest == prior['input_sha256'], 'Input CSV differs from previous analysis; inspect before running'
    rows = load_rows(ROOT / 'inputCSV')
    selected = [r for r in rows if r['phase'] == 'practice' and r['condition'] in ('C2', 'C3') and 1 <= r['n'] <= 400]
    members = {c: sorted({r['participant'] for r in selected if r['condition'] == c}) for c in ('C2', 'C3')}
    assert all(len(ids) == 10 for ids in members.values()) and not set(members['C2']) & set(members['C3'])
    basic = [r for r in selected if r['countedAsTrial'] and r['coreExitMs'] is not None]
    outliers = mt_outliers(basic, 'coreMtMs', 'trial1_5sd')
    tests, blocks, people, audit, missing, diagnostics, checks, bootstraps, influence = [], [], [], [], [], [], [], [], []
    for pipeline in ('basic_only', 'sensitivity_1_5sd'):
        kept = basic if pipeline == 'basic_only' else [r for r in basic if id(r) not in outliers]
        for r in selected:
            reasons = []
            if not r['countedAsTrial']:
                reasons.append('countedAsTrial=false')
            if r['coreExitMs'] is None:
                reasons.append('coreExitMs missing')
            if pipeline != 'basic_only' and id(r) in outliers:
                reasons.append('condition/trial MT outside mean +/- 1.5 sample SD')
            if reasons:
                audit.append(dict(pipeline=pipeline, participant=r['participant'], condition=r['condition'], trial=r['n'],
                                  source=r['source'], line=r['line'], reason='; '.join(reasons)))
        groups = defaultdict(list)
        for r in kept:
            groups[r['condition'], r['participant'], (r['n'] - 1) // 100].append(r)
            for metric in METRICS:
                if r['coreMtMs'][metric] is None:
                    missing.append(dict(pipeline=pipeline, metric=metric, condition=r['condition'], participant=r['participant'], source=r['source'], line=r['line']))
        local = []
        for metric in METRICS:
            for c, ids in members.items():
                for participant in ids:
                    item = dict(pipeline=pipeline, metric=metric, unit=UNITS[metric], condition=c, participant=participant)
                    for block in range(4):
                        group = groups[c, participant, block]
                        value = aggregate(group, metric, 'coreMtMs')
                        if value is None:
                            raise ValueError(f'Unavailable participant/block: {pipeline}/{metric}/{c}/{participant}/{block+1}')
                        item[f'b{block+1}'] = value
                        blocks.append(dict(pipeline=pipeline, metric=metric, condition=c, participant=participant, block=block+1,
                                           start=block*100+1, end=(block+1)*100, value=value, rows=len(group),
                                           valid=sum(r['coreMtMs'][metric] is not None for r in group)))
                    item['early'], item['late'] = item['b1'], item['b4']
                    item['change'] = item['late'] - item['early']
                    if metric != 'ER':
                        item['slope_early'], item['residual_sd_early'] = fit_line(groups[c, participant, 0], metric)
                        item['slope_late'], item['residual_sd_late'] = fit_line(groups[c, participant, 3], metric)
                        item['log_sd_late'] = float(np.log(item['residual_sd_late']))
                    local.append(item)
        people.extend(local)
        for metric in METRICS:
            units = UNITS[metric]
            group = {c: [r for r in local if r['metric'] == metric and r['condition'] == c] for c in members}
            keys = ('early', 'late', 'change') + (('slope_early', 'slope_late', 'log_sd_late') if metric != 'ER' else ())
            arrays = {c: {key: np.array([r[key] for r in group[c]]) for key in keys} for c in members}
            def add(name, result, unit=units, priority='secondary'):
                assert np.isfinite(result['p_raw'])
                tests.append(dict(pipeline=pipeline, metric=metric, comparison=name, priority=priority, unit=unit, **result))
            for key in ('change', 'early', 'late'):
                add(key + ': C2-C3', contrast(arrays['C2'][key], arrays['C3'][key]), priority='primary' if key == 'change' else 'secondary')
            for c in members:
                result = contrast(arrays[c]['late'], arrays[c]['early'], True)
                result['mean_late'], result['mean_early'] = result.pop('mean_c2'), result.pop('mean_c3')
                add(c + ': late-early', result)
            if metric != 'ER':
                for key in ('slope_early', 'slope_late'):
                    add(key + ': C2-C3', contrast(arrays['C2'][key], arrays['C3'][key]), units + '/trial')
                result = contrast(arrays['C2']['log_sd_late'], arrays['C3']['log_sd_late'])
                result.update(ratio=float(np.exp(result['estimate'])), ratio_ci_low=float(np.exp(result['ci_low'])), ratio_ci_high=float(np.exp(result['ci_high'])))
                add('late residual SD: C2/C3', result, 'log SD ratio (C2/C3)')
                for key in ('late', 'change'):
                    add(key + ' between-person SD: C2/C3', variance_test(arrays['C2'][key], arrays['C3'][key]), 'SD ratio (C2/C3)')
            y = np.array([[[r[f'b{b}'] for b in range(1, 5)] for r in group[c]] for c in members])
            add('condition x 4-block profile', profile_test(y), 'F statistic')
            for c in members:
                for key in keys:
                    x = arrays[c][key]
                    diagnostics.append(dict(pipeline=pipeline, metric=metric, condition=c, quantity=key, n=len(x), mean=float(x.mean()),
                                            sd=float(x.std(ddof=1)), min=float(x.min()), max=float(x.max()), skew=float(stats.skew(x, bias=False)), zero_count=int((x == 0).sum())))
            if pipeline == 'basic_only':
                x, y = arrays['C2']['change'], arrays['C3']['change']
                boot = stats.bootstrap((x, y), lambda x, y, axis: x.mean(axis=axis)-y.mean(axis=axis), vectorized=True,
                                       n_resamples=10000, batch=1000, method='BCa', rng=np.random.default_rng(20261113 + METRICS.index(metric)))
                assert np.isfinite([boot.confidence_interval.low, boot.confidence_interval.high]).all()
                bootstraps.append(dict(metric=metric, method='participant BCa bootstrap; unadjusted 95%', resamples=10000,
                                       estimate=float(x.mean()-y.mean()), ci_low=float(boot.confidence_interval.low), ci_high=float(boot.confidence_interval.high)))
                for c in members:
                    for i, r in enumerate(group[c]):
                        estimate = np.delete(x, i).mean()-y.mean() if c == 'C2' else x.mean()-np.delete(y, i).mean()
                        influence.append(dict(metric=metric, removed_condition=c, removed_participant=r['participant'], estimate=float(estimate)))
        batch = [t for t in tests if t['pipeline'] == pipeline]
        assert len(batch) == 28
        for t, p in zip(batch, holm([r['p_raw'] for r in batch])):
            t['p_holm28'] = float(p)
        for block in range(4):
            options = dict(participants=sum(members.values(), []), conditions=list(members), phase='practice', scope='coreMtMs',
                           mode='mean', bin=100, exclude=True, trialStart=block*100+1, trialEnd=(block+1)*100,
                           mtOutlier='none' if pipeline == 'basic_only' else 'trial1_5sd', excludeBoundaryTrials=False)
            existing = analyze(rows, options, include_fits=False)
            actual_audit = {(r['source'], r['line']) for r in audit if r['pipeline'] == pipeline and block*100+1 <= r['trial'] <= (block+1)*100}
            assert actual_audit == {(r['source'], r['line']) for r in existing['excluded']}
            for summary in existing['summary']:
                for metric in METRICS:
                    expected = np.mean([r[f'b{block+1}'] for r in local if r['metric'] == metric and r['condition'] == summary['condition']])
                    np.testing.assert_allclose(summary[metric], expected)
            checks.append(dict(pipeline=pipeline, block=block+1, selected=existing['selected'], included=existing['included'], missing={m: existing['missing'][m] for m in METRICS}))
        print(pipeline, 'completed; selected', len(selected), 'retained', len(kept), 'planned tests', len(batch), flush=True)
    for t, p in zip(tests, holm([r['p_raw'] for r in tests])):
        t['p_holm56_reference'] = float(p)
    current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert current == manifest
    for name, records in [('tests.csv', tests), ('participant_summary.csv', people), ('participant_blocks.csv', blocks), ('excluded_rows.csv', audit),
                          ('diagnostics.csv', diagnostics), ('primary_bootstrap.csv', bootstraps), ('leave_one_out.csv', influence)]:
        write_csv(name, records)
    result = dict(protocol=protocol, timestamp_utc=datetime.now(timezone.utc).isoformat(), numpy=np.__version__, scipy=scipy.__version__,
                  input_sha256=manifest, analysis_sha256=hashlib.sha256((ROOT/'analysis.py').read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), checks=checks, missing_rows=missing,
                  tests=tests, bootstrap=bootstraps, original_selected=len(selected), basic_excluded=len(selected)-len(basic), sensitivity_outliers=len(outliers))
    with (HERE/'results.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    for t in tests:
        if t['priority'] == 'primary' or t['p_holm28'] < .05:
            print(json.dumps(t, ensure_ascii=False))
    print('All checks passed; no changes to input CSV. Missing metric rows:', len(missing))


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        self_test()
    else:
        main()
