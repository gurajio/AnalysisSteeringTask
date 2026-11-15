import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ['STEERING_ROOT']).resolve() if 'STEERING_ROOT' in os.environ else next(
    p for p in HERE.parents if (p / 'inputCSV').is_dir())


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def write_csv(name, records):
    with (HERE / name).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    with (HERE / 'trial_audit.csv').open(encoding='utf-8-sig', newline='') as stream:
        audit = {(r['source'], int(r['line'])): r for r in csv.DictReader(stream)}
    counts, values, numbers = defaultdict(Counter), defaultdict(list), defaultdict(list)
    geometry, members = defaultdict(set), defaultdict(set)
    checked = 0
    for path in sorted((ROOT / 'inputCSV').glob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for line, row in enumerate(csv.DictReader(stream), 2):
                c, pid = row['conditionId'], row['participantId']
                if row['phase'] != 'practice' or c not in ('C2', 'C3'):
                    continue
                members[c].add(pid)
                geometry[c].add((row['amplitude'], row['width']))
                n = int(row['trialInCondition'])
                period = 'early' if 1 <= n <= 100 else 'late' if 301 <= n <= 400 else None
                if period is None:
                    continue
                a = audit[(path.name, line)]
                counted = row['countedAsTrial'].lower() == 'true'
                exit_valid = finite(row['coreExitMs']) is not None
                included = counted and exit_valid
                mt = finite(row['coreMtMs'])
                valid = included and mt is not None and mt > 0
                assert (a['included'] == 'True') == included
                assert (a['valid_MT'] == 'True') == valid
                assert (a['condition'], a['participant'], a['period'], int(a['N'])) == (c, pid, period, n)
                checked += 1
                count = counts[(c, period)]
                count['records'] += 1
                count['excluded_counted_false'] += not counted
                count['excluded_core_exit_missing'] += not exit_valid
                count['excluded_both_reasons'] += not counted and not exit_valid
                count['excluded_unique'] += not included
                count['included_rows'] += included
                count['invalid_MT_included'] += included and not valid
                count['valid_MT_rows'] += valid
                count['included_success_false'] += included and row['success'].lower() == 'false'
                if valid:
                    values[(c, pid, period)].append(mt)
                    numbers[(c, pid, period)].append(n)
    assert checked == len(audit)
    assert all(len(members[c]) == 10 for c in ('C2', 'C3'))
    assert not members['C2'] & members['C3']
    with (HERE / 'participant_means.csv').open(encoding='utf-8-sig', newline='') as stream:
        means = list(csv.DictReader(stream))
    for row in means:
        for period in ('early', 'late'):
            key = (row['condition'], row['participant'], period)
            assert values[key]
            np.testing.assert_allclose(statistics.mean(values[key]), float(row[period + '_mean_ms']), rtol=0, atol=1e-10)
            assert len(values[key]) == int(row[period + '_valid_rows'])
    results = json.loads((HERE / 'results.json').read_text())
    for row in results:
        c = row['condition']
        diffs = [statistics.mean(values[(c, p, 'late')]) - statistics.mean(values[(c, p, 'early')]) for p in sorted(members[c])]
        mean, se = statistics.mean(diffs), statistics.stdev(diffs) / math.sqrt(len(diffs))
        critical = stats.t.ppf(.975, len(diffs) - 1)
        np.testing.assert_allclose([mean, mean - critical * se, mean + critical * se],
                                  [row['change_mean_ms'], row['ci95_low_ms'], row['ci95_high_ms']], rtol=0, atol=1e-10)
    for prefix in ('t', 'wilcoxon'):
        low, high = sorted(results, key=lambda r: r[prefix + '_p'])
        np.testing.assert_allclose([low[prefix + '_p_holm'], high[prefix + '_p_holm']],
                                  [min(1, 2 * low[prefix + '_p']), min(1, max(2 * low[prefix + '_p'], high[prefix + '_p']))])
    summaries = [dict(condition=c, period=p, **counts[(c, p)]) for c in ('C2', 'C3') for p in ('early', 'late')]
    write_csv('exclusion_summary.csv', summaries)
    coverage = []
    for key, ns in sorted(numbers.items()):
        c, pid, period = key
        expected = set(range(1, 101)) if period == 'early' else set(range(301, 401))
        coverage.append(dict(condition=c, participant=pid, period=period, valid_rows=len(ns),
                             valid_unique_N=len(set(ns)), repeated_valid_N=len(ns)-len(set(ns)),
                             missing_valid_N=';'.join(map(str, sorted(expected-set(ns))))))
    write_csv('coverage.csv', coverage)
    manifest = json.loads((HERE / 'manifest.json').read_text())
    current = [dict(file=p.name, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in sorted((ROOT / 'inputCSV').glob('*.csv'))]
    assert manifest['inputs'] == current
    verification = dict(status='passed', raw_rows_checked=checked, participant_means_checked=len(means),
                        geometry={c: sorted(g) for c, g in geometry.items()}, numeric_cores=1,
                        min_valid_rows=min(len(v) for v in values.values()), max_valid_rows=max(len(v) for v in values.values()),
                        checks=['independent scalar CSV audit', 'participant means against statistics.mean',
                                'confidence intervals against explicit t formula', 'two-test Holm formula', 'input hashes unchanged'])
    with (HERE / 'verification.json').open('x', encoding='utf-8') as stream:
        json.dump(verification, stream, ensure_ascii=False, indent=2)
    print(json.dumps(verification, ensure_ascii=False))
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == '__main__':
    main()
