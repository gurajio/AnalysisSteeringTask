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
    return dict(comparison=label, test=name, difference_ms=difference, ci95_low_ms=float(ci[0]),
                ci95_high_ms=float(ci[1]), F=f, df1=1, df2=df, p_raw=p,
                partial_eta_squared=float(ss / (ss + error)), SS_effect=float(ss), SS_error=float(error))


def mixed_components(arrays):
    y = np.array([np.column_stack((arrays[c]['early_ms'], arrays[c]['late_ms'])) for c in ('C2', 'C3')])
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
    np.testing.assert_allclose(holm([.01, .04, .03, .8]), [.04, .09, .09, .8])
    np.testing.assert_allclose(compare('fixture', np.array([2., 4., 7.]), np.array([1., 2., 4.]), True)['F'], 12.)
    np.testing.assert_allclose(compare('fixture', np.array([1., 2., 3.]), np.array([4., 5., 6.]))['F'], 13.5)
    rows = load_rows(ROOT / 'inputCSV')
    selected = [r for r in rows if r['phase'] == 'practice' and r['condition'] in ('C2', 'C3') and 1 <= r['n'] <= 400]
    members = {c: sorted({r['participant'] for r in selected if r['condition'] == c}) for c in ('C2', 'C3')}
    assert not set(members['C2']) & set(members['C3'])
    eligible = [r for r in selected if (r['n'] <= 100 or r['n'] >= 301) and r['countedAsTrial'] and r['coreExitMs'] is not None]
    outliers = mt_outliers(eligible, 'coreMtMs', 'trial1_5sd')
    from collections import defaultdict
    from statistics import mean, stdev
    groups = defaultdict(list)
    for row in eligible:
        if row['coreMtMs']['MT'] is not None:
            groups[row['condition'], row['n']].append(row)
    verified = set()
    for group in groups.values():
        values = [r['coreMtMs']['MT'] for r in group]
        if len(values) >= 2 and stdev(values) > 0:
            low, high = mean(values) - 1.5 * stdev(values), mean(values) + 1.5 * stdev(values)
            verified.update(id(r) for r in group if r['coreMtMs']['MT'] < low or r['coreMtMs']['MT'] > high)
    assert verified == outliers
    participants, audit, checks = [], [], []
    for condition, ids in members.items():
        for participant in ids:
            item = dict(condition=condition, participant=participant)
            for half, low, high in [('early', 1, 100), ('late', 301, 400)]:
                raw = [r for r in selected if r['condition'] == condition and r['participant'] == participant and low <= r['n'] <= high]
                valid = []
                for row in raw:
                    reasons = []
                    if not row['countedAsTrial']:
                        reasons.append('countedAsTrial=false')
                    if row['coreExitMs'] is None:
                        reasons.append('coreExitMs missing')
                    if id(row) in outliers:
                        reasons.append('same trial MT outside mean +/- 1.5 SD')
                    if reasons or row['coreMtMs']['MT'] is None:
                        audit.append(dict(condition=condition, participant=participant, half=half, source=row['source'],
                                          line=row['line'], trial=row['n'], reason='; '.join(reasons) or 'invalid core MT',
                                          status='excluded' if reasons else 'MT unavailable'))
                    else:
                        valid.append(row)
                if not valid:
                    raise ValueError(f'Missing participant half: {condition}/{participant}/{half}')
                item[half + '_ms'] = aggregate(valid, 'MT', 'coreMtMs')
                item[half + '_rows'] = len(valid)
                item[half + '_unique_trials'] = len({r['n'] for r in valid})
                item[half + '_raw_rows'] = len(raw)
                item[half + '_outlier_rows'] = sum(id(r) in outliers for r in raw)
                item[half + '_basic_excluded'] = sum(not r['countedAsTrial'] or r['coreExitMs'] is None for r in raw)
                item[half + '_invalid_mt'] = sum(r['countedAsTrial'] and r['coreExitMs'] is not None and r['coreMtMs']['MT'] is None for r in raw)
            item['change_ms'] = item['late_ms'] - item['early_ms']
            participants.append(item)
    arrays = {c: {key: np.array([p[key] for p in participants if p['condition'] == c]) for key in ('early_ms', 'late_ms', 'change_ms')} for c in members}
    tests = [compare(c + ': late - early', arrays[c]['late_ms'], arrays[c]['early_ms'], True) for c in members]
    tests += [compare('late: C2 - C3', arrays['C2']['late_ms'], arrays['C3']['late_ms']),
              compare('change: C2 - C3', arrays['C2']['change_ms'], arrays['C3']['change_ms'])]
    components = mixed_components(arrays)
    np.testing.assert_allclose(tests[3]['F'], components[3]['MS'] / components[4]['MS'])
    for test, adjusted in zip(tests, holm([t['p_raw'] for t in tests])):
        test['p_holm'] = adjusted
    summaries = {}
    for c, values in arrays.items():
        summaries[c] = {key: dict(n=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)), min=float(a.min()), max=float(a.max()),
                                  skew=float(stats.skew(a, bias=False))) for key, a in values.items()}
    for low, high, key in [(1, 100, 'early_ms'), (301, 400, 'late_ms')]:
        options = dict(participants=sum(members.values(), []), conditions=list(members), phase='practice', scope='coreMtMs',
                       mode='mean', bin=100, exclude=True, trialStart=low, trialEnd=high, mtOutlier='trial1_5sd', excludeBoundaryTrials=False)
        from analysis import trial_range
        assert trial_range(options) == (low, high)
        existing = analyze(rows, options, include_fits=False)
        assert {(e['source'], e['line']) for e in existing['excluded']} == {(e['source'], e['line']) for e in audit if low <= e['trial'] <= high and e['status'] == 'excluded'}
        for summary in existing['summary']:
            np.testing.assert_allclose(summary['MT'], summaries[summary['condition']][key]['mean'])
        checks.append(dict(range=[low, high], selected=existing['selected'], included=existing['included'], missing_mt=existing['missing']['MT']))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'inputCSV').glob('*.csv'))}
    result = dict(settings=dict(phase='practice', metric='coreMtMs', early=[1, 100], late=[301, 400],
                               basic_exclusion=True, outlier_exclusion='trial1_5sd', boundary_exclusion=False,
                               unit='participant', aggregation='arithmetic mean of valid rows within participant and half; equal participant weighting',
                               alternative='F upper-tail; nondirectional contrasts', alpha=.05, multiplicity='Holm over all four tests', model='balanced 2x2 mixed ANOVA; specific late error for simple effect; equal group variances; two time levels', ci='unadjusted 95%', cores=1, analysis_status='additional analysis after viewing previous analyses; Holm covers these four tests only'),
                  summaries=summaries, tests=tests, mixed_components=components, validation=checks, outlier_rows=len(outliers), input_sha256=manifest)
    folder = Path(__file__).resolve().parent
    for filename, records in [('participants.csv', participants), ('tests.csv', tests), ('excluded_rows.csv', audit), ('mixed_components.csv', components)]:
        with (folder / filename).open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    with (folder / 'results.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(summaries=summaries, tests=tests, mixed_components=components, validation=checks, outlier_rows=len(outliers)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
