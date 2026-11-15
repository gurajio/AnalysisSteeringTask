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
from analysis import load_rows, aggregate, analyze


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(np.asarray(values)[order] * np.arange(len(values), 0, -1)))
    return adjusted.tolist()


def compare(label, x, y, paired=False):
    result = stats.ttest_rel(x, y) if paired else stats.ttest_ind(x, y, equal_var=False)
    difference = float(np.mean(x) - np.mean(y))
    if paired:
        se = np.std(x - y, ddof=1) / np.sqrt(len(x))
        df = len(x) - 1
    else:
        vx, vy = np.var(x, ddof=1) / len(x), np.var(y, ddof=1) / len(y)
        se = np.sqrt(vx + vy)
        df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    np.testing.assert_allclose([result.statistic, result.df], [difference / se, df])
    ci = result.confidence_interval(.95)
    np.testing.assert_allclose([ci.low, ci.high], difference + stats.t.ppf([.025, .975], df) * se)
    return dict(comparison=label, test='paired t' if paired else 'Welch t', difference_ms=difference,
                ci95_low_ms=float(ci.low), ci95_high_ms=float(ci.high), t=float(result.statistic),
                df=float(result.df), p_raw=float(result.pvalue))


def main():
    np.testing.assert_allclose(holm([.01, .04, .03, .8]), [.04, .09, .09, .8])
    rows = load_rows(ROOT / 'inputCSV')
    selected = [r for r in rows if r['phase'] == 'practice' and r['condition'] in ('C2', 'C3') and 1 <= r['n'] <= 400]
    members = {c: sorted({r['participant'] for r in selected if r['condition'] == c}) for c in ('C2', 'C3')}
    assert not set(members['C2']) & set(members['C3'])
    participants, audit, checks = [], [], []
    for condition, ids in members.items():
        for participant in ids:
            item = dict(condition=condition, participant=participant)
            for half, low, high in [('early', 1, 50), ('late', 350, 400)]:
                raw = [r for r in selected if r['condition'] == condition and r['participant'] == participant and low <= r['n'] <= high]
                valid = []
                for row in raw:
                    reasons = []
                    if not row['countedAsTrial']:
                        reasons.append('countedAsTrial=false')
                    if row['coreExitMs'] is None:
                        reasons.append('coreExitMs missing')
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
                item[half + '_invalid_mt'] = sum(r['countedAsTrial'] and r['coreExitMs'] is not None and r['coreMtMs']['MT'] is None for r in raw)
            item['change_ms'] = item['late_ms'] - item['early_ms']
            participants.append(item)
    arrays = {c: {key: np.array([p[key] for p in participants if p['condition'] == c]) for key in ('early_ms', 'late_ms', 'change_ms')} for c in members}
    tests = [compare(c + ': late - early', arrays[c]['late_ms'], arrays[c]['early_ms'], True) for c in members]
    tests += [compare('late: C2 - C3', arrays['C2']['late_ms'], arrays['C3']['late_ms']),
              compare('change: C2 - C3', arrays['C2']['change_ms'], arrays['C3']['change_ms'])]
    for test, adjusted in zip(tests, holm([t['p_raw'] for t in tests])):
        test['p_holm'] = adjusted
    summaries = {}
    for c, values in arrays.items():
        summaries[c] = {key: dict(n=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)), min=float(a.min()), max=float(a.max()),
                                  skew=float(stats.skew(a, bias=False))) for key, a in values.items()}
    for low, high, key in [(1, 50, 'early_ms'), (350, 400, 'late_ms')]:
        options = dict(participants=sum(members.values(), []), conditions=list(members), phase='practice', scope='coreMtMs',
                       mode='mean', bin=100, exclude=True, trialStart=low, trialEnd=high, mtOutlier='none', excludeBoundaryTrials=False)
        from analysis import trial_range
        assert trial_range(options) == (low, high)
        existing = analyze(rows, options, include_fits=False)
        for summary in existing['summary']:
            np.testing.assert_allclose(summary['MT'], summaries[summary['condition']][key]['mean'])
        checks.append(dict(range=[low, high], selected=existing['selected'], included=existing['included'], missing_mt=existing['missing']['MT']))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'inputCSV').glob('*.csv'))}
    result = dict(settings=dict(phase='practice', metric='coreMtMs', early=[1, 50], late=[350, 400],
                               basic_exclusion=True, outlier_exclusion=False, boundary_exclusion=False,
                               unit='participant', aggregation='arithmetic mean of valid rows within participant and half; equal participant weighting',
                               alternative='two-sided', alpha=.05, multiplicity='Holm over all four tests', ci='unadjusted 95%', cores=1, analysis_status='additional analysis after viewing 200-trial and 100-trial results; Holm covers these four tests only'),
                  summaries=summaries, tests=tests, validation=checks, input_sha256=manifest)
    folder = Path(__file__).resolve().parent
    for filename, records in [('participants.csv', participants), ('tests.csv', tests), ('excluded_rows.csv', audit)]:
        with (folder / filename).open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    with (folder / 'results.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(summaries=summaries, tests=tests, validation=checks, participants=participants), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
