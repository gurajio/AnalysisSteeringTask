import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import load_rows, aggregate, analyze, mt_outliers


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(np.asarray(values)[order] * np.arange(len(values), 0, -1)))
    return adjusted.tolist()


def compare(label, x, y, paired=False):
    difference = float(np.mean(x) - np.mean(y))
    if paired:
        values = np.column_stack((y, x))
        grand = values.mean()
        ss = len(x) * np.sum((values.mean(axis=0) - grand) ** 2)
        residual = values - values.mean(axis=1, keepdims=True) - values.mean(axis=0) + grand
        error = np.sum(residual ** 2)
        df = len(x) - 1
        se = np.sqrt(2 * error / df / len(x))
        check = stats.ttest_rel(x, y)
        name = 'one-factor RMANOVA'
    else:
        grand = np.concatenate((x, y)).mean()
        ss = len(x) * (x.mean() - grand) ** 2 + len(y) * (y.mean() - grand) ** 2
        error = np.sum((x - x.mean()) ** 2) + np.sum((y - y.mean()) ** 2)
        df = len(x) + len(y) - 2
        se = np.sqrt(error / df * (1 / len(x) + 1 / len(y)))
        check = stats.ttest_ind(x, y, equal_var=True)
        name = 'mixed ANOVA interaction' if label.startswith('change') else 'mixed-design simple condition effect at late; specific error'
        if label.startswith('change'):
            ss, error = ss / 2, error / 2
    f = float(ss / (error / df))
    p = float(stats.f.sf(f, 1, df))
    np.testing.assert_allclose([f, p], [check.statistic ** 2, check.pvalue], rtol=1e-10)
    ci = difference + stats.t.ppf([.025, .975], df) * se
    return dict(comparison=label, test=name, difference=difference, ci95_low=float(ci[0]),
                ci95_high=float(ci[1]), F=f, df1=1, df2=df, p_raw=p,
                partial_eta_squared=float(ss / (ss + error)), SS_effect=float(ss), SS_error=float(error))


def mixed_components(arrays):
    y = np.array([np.column_stack((arrays[c]['early'], arrays[c]['late'])) for c in ('C2', 'C3')])
    a, n, b = y.shape
    assert a == b == 2
    grand = y.mean()
    cell = y.mean(axis=1)
    condition = y.mean(axis=(1, 2))
    time = y.mean(axis=(0, 1))
    subject = y.mean(axis=2)
    terms = [
        ('condition', b * n * np.sum((condition - grand) ** 2), a - 1),
        ('subject(condition)', b * np.sum((subject - condition[:, None]) ** 2), a * (n - 1)),
        ('time', a * n * np.sum((time - grand) ** 2), b - 1),
        ('condition:time', n * np.sum((cell - condition[:, None] - time + grand) ** 2), (a - 1) * (b - 1)),
        ('time:subject(condition)', np.sum((y - subject[:, :, None] - cell[:, None, :] + condition[:, None, None]) ** 2), a * (n - 1) * (b - 1))]
    np.testing.assert_allclose(sum(ss for _, ss, _ in terms), np.sum((y - grand) ** 2))
    assert sum(df for _, _, df in terms) == y.size - 1
    return [dict(source=name, SS=float(ss), df=df, MS=float(ss / df)) for name, ss, df in terms]


def main():
    from collections import defaultdict
    from statistics import mean, stdev
    np.testing.assert_allclose(holm([.01, .04, .03, .8]), [.04, .09, .09, .8])
    np.testing.assert_allclose(compare('fixture', np.array([2., 4., 7.]), np.array([1., 2., 4.]), True)['F'], 12.)
    np.testing.assert_allclose(compare('fixture', np.array([1., 2., 3.]), np.array([4., 5., 6.]))['F'], 13.5)
    rows = load_rows(ROOT / 'inputCSV')
    selected = [r for r in rows if r['phase'] == 'practice' and r['condition'] in ('C2', 'C3') and (1 <= r['n'] <= 100 or 301 <= r['n'] <= 400)]
    members = {c: sorted({r['participant'] for r in selected if r['condition'] == c}) for c in ('C2', 'C3')}
    assert all(len(ids) == 10 for ids in members.values()) and not set(members['C2']) & set(members['C3'])
    basic = [r for r in selected if r['countedAsTrial'] and r['coreExitMs'] is not None]
    outliers = mt_outliers(basic, 'coreMtMs', 'trial1_5sd')
    groups = defaultdict(list)
    for r in basic:
        if r['coreMtMs']['MT'] is not None:
            groups[r['condition'], r['n']].append(r)
    verified = set()
    for group in groups.values():
        values = [r['coreMtMs']['MT'] for r in group]
        if len(values) >= 2 and stdev(values) > 0:
            low, high = mean(values) - 1.5 * stdev(values), mean(values) + 1.5 * stdev(values)
            verified.update(id(r) for r in group if r['coreMtMs']['MT'] < low or r['coreMtMs']['MT'] > high)
    assert verified == outliers
    kept = [r for r in basic if id(r) not in outliers]
    audit = []
    for r in selected:
        reasons = []
        if not r['countedAsTrial']:
            reasons.append('countedAsTrial=false')
        if r['coreExitMs'] is None:
            reasons.append('coreExitMs missing')
        if id(r) in outliers:
            reasons.append('same trial MT outside mean +/- 1.5 SD')
        if reasons:
            audit.append(dict(condition=r['condition'], participant=r['participant'], trial=r['n'], source=r['source'], line=r['line'], reason='; '.join(reasons)))
    missing = []
    participants, summaries, tests, components = [], [], [], []
    units = {'MT': 'ms', 'ER': 'percent (differences: percentage points)', 'TPe': 's^-1'}
    for metric, unit in units.items():
        local = []
        for c, ids in members.items():
            for participant in ids:
                item = dict(metric=metric, unit=unit, condition=c, participant=participant)
                for half, low, high in [('early', 1, 100), ('late', 301, 400)]:
                    group = [r for r in kept if r['condition'] == c and r['participant'] == participant and low <= r['n'] <= high]
                    for r in group:
                        if r['coreMtMs'][metric] is None:
                            missing.append(dict(metric=metric, condition=c, participant=participant, trial=r['n'], source=r['source'], line=r['line']))
                    item[half] = aggregate(group, metric, 'coreMtMs')
                    item[half + '_rows'] = len(group)
                    item[half + '_valid'] = sum(r['coreMtMs'][metric] is not None for r in group)
                    if item[half] is None:
                        raise ValueError(f'Participant estimate unavailable; no silent participant deletion: {metric}/{c}/{participant}/{half}')
                item['change'] = item['late'] - item['early']
                local.append(item)
        participants.extend(local)
        arrays = {c: {key: np.array([r[key] for r in local if r['condition'] == c]) for key in ('early', 'late', 'change')} for c in members}
        results = [compare(c + ': late - early', arrays[c]['late'], arrays[c]['early'], True) for c in members]
        results += [compare('late: C2 - C3', arrays['C2']['late'], arrays['C3']['late']), compare('change: C2 - C3', arrays['C2']['change'], arrays['C3']['change'])]
        parts = mixed_components(arrays)
        np.testing.assert_allclose(results[3]['F'], parts[3]['MS'] / parts[4]['MS'])
        for t in results:
            assert np.isfinite(t['F']) and np.isfinite(t['p_raw'])
            tests.append(dict(metric=metric, unit=unit, **t))
        components.extend(dict(metric=metric, **part) for part in parts)
        for c, values in arrays.items():
            summary = dict(metric=metric, unit=unit, condition=c, n=len(values['early']))
            for key, a in values.items():
                summary[key + '_mean'] = float(a.mean())
                summary[key + '_sd'] = float(a.std(ddof=1))
                summary[key + '_min'] = float(a.min())
                summary[key + '_max'] = float(a.max())
                summary[key + '_skew'] = float(stats.skew(a, bias=False))
            summaries.append(summary)
    assert len(tests) == 12
    adjusted = holm([t['p_raw'] for t in tests])
    for t, p in zip(tests, adjusted):
        t['p_holm12'] = p
    checks = []
    for low, high, half in [(1, 100, 'early'), (301, 400, 'late')]:
        options = dict(participants=sum(members.values(), []), conditions=list(members), phase='practice', scope='coreMtMs', mode='mean', bin=100,
                       exclude=True, trialStart=low, trialEnd=high, mtOutlier='trial1_5sd', excludeBoundaryTrials=False)
        existing = analyze(rows, options, include_fits=False)
        assert {(r['source'], r['line']) for r in existing['excluded']} == {(r['source'], r['line']) for r in audit if low <= r['trial'] <= high}
        for item in existing['summary']:
            for metric in units:
                summary = next(s for s in summaries if s['metric'] == metric and s['condition'] == item['condition'])
                np.testing.assert_allclose(item[metric], summary[half + '_mean'])
        checks.append(dict(range=[low, high], selected=existing['selected'], included=existing['included'], missing={m: existing['missing'][m] for m in units}))
    prior = json.loads((ROOT / 'output/c2_c3_practice_100_anova_trial1_5sd_20261113/results.json').read_text())
    for old, new in zip(prior['tests'], tests[:4]):
        np.testing.assert_allclose([old['F'], old['p_raw']], [new['F'], new['p_raw']])
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'inputCSV').glob('*.csv'))}
    assert manifest == prior['input_sha256']
    result = dict(settings=dict(phase='practice', scope='coreMtMs', early=[1, 100], late=[301, 400], basic_exclusion=True,
                  mt_outlier='trial1_5sd', boundary_exclusion=False, common_exclusion=True, multiplicity='Holm over all 12 tests', alpha=.05,
                  model='2x2 mixed ANOVA; specific late error for simple effect; equal group variances', ci='unadjusted 95%', cores=1,
                  analysis_status='additional exploratory analysis; correction does not cover earlier analyses'),
                  summaries=summaries, tests=tests, validation=checks, basic_excluded=len(selected)-len(basic), outlier_rows=len(outliers),
                  included=len(kept), missing=missing, input_sha256=manifest)
    folder = Path(__file__).resolve().parent
    for name, records in [('participants.csv', participants), ('summaries.csv', summaries), ('tests.csv', tests), ('excluded_rows.csv', audit), ('mixed_components.csv', components)]:
        with (folder / name).open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    with (folder / 'results.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    for s in summaries:
        print(s)
    for t in tests:
        print(t)
    print('validation', checks, 'basic_excluded', result['basic_excluded'], 'outlier_rows', len(outliers), 'included', len(kept))


if __name__ == '__main__':
    main()
