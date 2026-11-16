import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import analysis

MODELS = ('power_rate', 'exponential')
CONDITIONS = ('C2', 'C3')
OPTIONS = dict(phase='practice', scope='coreMtMs', mode='mean', bin=10,
               trialStart=1, trialEnd=400, exclude=True, mtOutlier='none',
               excludeBoundaryTrials=False, conditions=list(CONDITIONS))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    rows = analysis.load_rows(ROOT / 'inputCSV')
    groups = defaultdict(lambda: defaultdict(list))
    audit = []
    members = defaultdict(set)
    for row in rows:
        if row['phase'] != 'practice' or row['condition'] not in CONDITIONS or not 1 <= row['n'] <= 400:
            continue
        condition, person = row['condition'], row['participant']
        members[condition].add(person)
        reasons = []
        if not row['countedAsTrial']:
            reasons.append('countedAsTrial=false')
        if row['coreExitMs'] is None:
            reasons.append('coreExitMs missing')
        mt = row['coreMtMs']['MT']
        audit.append(dict(source=row['source'], line=row['line'], condition=condition,
                          participant=person, n=row['n'], mt=mt, excluded=' / '.join(reasons)))
        if not reasons and mt is not None:
            groups[condition, person][row['n']].append(mt)
    matrices = {}
    members = {c: sorted(members[c]) for c in CONDITIONS}
    for condition in CONDITIONS:
        matrix = np.full((len(members[condition]), 400), np.nan)
        for i, person in enumerate(members[condition]):
            for n, values in groups[condition, person].items():
                matrix[i, n - 1] = analysis.average(values)
        matrices[condition] = matrix
    return rows, members, matrices, audit


def observations(matrix, subset):
    values = matrix[list(subset)]
    counts = np.sum(np.isfinite(values), axis=0)
    valid = counts > 0
    return np.arange(1, 401)[valid], np.nansum(values, axis=0)[valid] / counts[valid]


def fit_subset(model, condition, subset, members, matrices):
    x, y = observations(matrices[condition], subset)
    fit = analysis.fit_regression_model(x, y, model)
    result = dict(model=model, condition=condition,
                  participants=[members[condition][i] for i in subset], count=len(subset), **fit)
    if fit['ok']:
        params = [fit['parameters'][key] for key in (('a', 'b') if model == 'power_rate' else ('a', 'b', 'c'))]
        result['endpoint'] = float(analysis.regression_value(model, 400, params))
    result.pop('x', None)
    result.pop('y', None)
    return result


def write_csv(name, rows):
    if rows:
        with (DEST / name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    started = time.monotonic()
    rows, members, matrices, audit = prepare()
    print('Members:', members, 'cores=1', flush=True)
    count = sum(2 ** len(members[c]) - 1 for c in CONDITIONS) * len(MODELS)
    sample_start = time.monotonic()
    for model in MODELS:
        for condition in CONDITIONS:
            size = len(members[condition])
            for subset in [tuple(range(size)), *itertools.islice(itertools.combinations(range(size), size - 1), 4)]:
                fit_subset(model, condition, subset, members, matrices)
    elapsed = time.monotonic() - sample_start
    print(f'Benchmark 20 fits={elapsed:.2f}s; {count} fits estimated={elapsed / 20 * count:.1f}s', flush=True)
    if '--run' not in sys.argv:
        return
    protocol = dict(options=OPTIONS, models=list(MODELS), endpointN=400, toleranceMs=50,
                    objectives=['maximum total', 'maximum equal group sizes'], members=members,
                    cores=1, selection='exhaustive nonempty subsets in each condition',
                    tieRule='report all maximum-size matches; representative has smallest absolute endpoint difference',
                    interpretation='Outcome-selected exploratory subsets, not evidence of population equivalence',
                    inputs={p.name: digest(p) for p in sorted((ROOT / 'inputCSV').glob('*.csv'))},
                    analysisSha256=digest(ROOT / 'analysis.py'))
    (DEST / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    write_csv('trial_audit.csv', audit)
    all_fits, results = [], []
    for model in MODELS:
        fits = {}
        for condition in CONDITIONS:
            fits[condition] = []
            size = len(members[condition])
            for n in range(size, 0, -1):
                for subset in itertools.combinations(range(size), n):
                    fits[condition].append(fit_subset(model, condition, subset, members, matrices))
                print(f'{model} {condition} size={n} done; elapsed={time.monotonic() - started:.1f}s', flush=True)
            all_fits.extend(fits[condition])
        for equal in (False, True):
            best_size, matches = -1, []
            for left in fits['C2']:
                if not left['ok']:
                    continue
                for right in fits['C3']:
                    if not right['ok'] or (equal and left['count'] != right['count']):
                        continue
                    total = left['count'] + right['count']
                    if total < best_size:
                        continue
                    delta = left['endpoint'] - right['endpoint']
                    if abs(delta) <= 50:
                        match = dict(model=model, objective='equal' if equal else 'total', total=total,
                                     C2=left, C3=right, difference=delta)
                        if total > best_size:
                            best_size, matches = total, []
                        matches.append(match)
            matches.sort(key=lambda item: (abs(item['difference']), item['C2']['participants'], item['C3']['participants']))
            results.append(dict(model=model, objective='equal' if equal else 'total', maximumTotal=best_size,
                                maximumMatches=len(matches), matches=matches))
    (DEST / 'fits.json').write_text(json.dumps(all_fits, ensure_ascii=False, indent=2))
    (DEST / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    match_rows = []
    for result in results:
        for i, match in enumerate(result['matches'], 1):
            match_rows.append(dict(model=match['model'], objective=match['objective'], rank=i,
                                   total=match['total'], C2_count=match['C2']['count'], C3_count=match['C3']['count'],
                                   C2_ids=' '.join(match['C2']['participants']), C3_ids=' '.join(match['C3']['participants']),
                                   C2_endpoint=match['C2']['endpoint'], C3_endpoint=match['C3']['endpoint'],
                                   difference=match['difference']))
    write_csv('maximum_matches.csv', match_rows)
    for result in results:
        if result['matches']:
            match = result['matches'][0]
            for condition in CONDITIONS:
                selected = match[condition]
                options = dict(OPTIONS, participants=selected['participants'], conditions=[condition])
                reference = analysis.analyze(rows, options)['fits'][0]
                subset = [members[condition].index(p) for p in selected['participants']]
                x, y = observations(matrices[condition], subset)
                np.testing.assert_array_equal(x, reference['observedX'])
                np.testing.assert_allclose(y, reference['observedY'], rtol=1e-13, atol=1e-10)
                check = analysis.fit_regression_model(reference['observedX'], reference['observedY'], result['model'])
                for key, value in selected['parameters'].items():
                    np.testing.assert_allclose(value, check['parameters'][key], rtol=1e-8, atol=1e-8)
            print(json.dumps(match, ensure_ascii=False), flush=True)
    failures = sum(not fit['ok'] for fit in all_fits)
    verification = dict(fits=len(all_fits), failures=failures, elapsedSeconds=time.monotonic() - started,
                        representativeChecks='observations and fit parameters matched analysis.analyze and fit_regression_model',
                        inputHashesUnchanged=all(digest(ROOT / 'inputCSV' / name) == value for name, value in protocol['inputs'].items()))
    (DEST / 'verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
