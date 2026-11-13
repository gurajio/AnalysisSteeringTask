import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

BOUNDARY_WINDOWS = ((96, 105), (196, 205), (296, 305))
CORE_A = {'C1': 84., 'C2': 253., 'C3': 315.}
MT_OUTLIERS = ('none', 'trial3sd', 'curve3sd', 'trial1_5sd', 'curve1_5sd')


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def clip_points(points, start, end):
    if start is None or end is None or end <= start or len(points) < 2:
        return None
    if start < points[0, 0] or end > points[-1, 0]:
        return None
    inner = points[(points[:, 0] > start) & (points[:, 0] < end)]
    edges = np.array([[t, np.interp(t, points[:, 0], points[:, 1]),
                       np.interp(t, points[:, 0], points[:, 2])] for t in (start, end)])
    return np.vstack((edges[0], inner, edges[1]))


def trajectory_metrics(points, mt):
    if points is None or len(points) < 2 or mt is None or mt <= 0:
        return None
    coords = points[:, 1:3]
    perpendicular = coords @ np.array([0.5, math.sqrt(3) / 2])
    we = 4.133 * float(np.std(perpendicular, ddof=1))
    ae = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
    if we <= 1e-9 or ae <= 0:
        return None
    return ae / we / (mt / 1000)


def core_ip(mt, amplitude, width):
    values = [number(v) for v in (mt, amplitude, width)]
    if any(v is None or v <= 0 for v in values):
        return None
    mt, amplitude, width = values
    value = mt * (width / amplitude)
    return value if math.isfinite(value) and value > 0 else None


def event_region(events, start, end):
    if not events:
        return 'none'
    if start is None or end is None or end <= start:
        return 'unknown'
    inside = any(e['startMs'] <= end and e['endMs'] >= start for e in events)
    outside = any(e['startMs'] < start or e['endMs'] > end for e in events)
    return 'both' if inside and outside else 'inside' if inside else 'outside'


def load_rows(folder):
    rows, seen = [], set()
    required = {'participantId', 'phase', 'conditionId', 'trialInCondition', 'attemptNumber',
                'success', 'mtMs', 'coreMtMs', 'coreEntryMs', 'coreExitMs', 'errorType',
                'trajectory', 'deviationEvents', 'startedAtIso', 'countedAsTrial'}
    for path in sorted(Path(folder).glob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{path.name}: 必須列が不足しています: {sorted(required - set(reader.fieldnames or []))}')
            for line, raw in enumerate(reader, 2):
                try:
                    key = tuple(raw[k] for k in ('participantId', 'phase', 'conditionId', 'attemptNumber', 'startedAtIso'))
                    if key in seen:
                        raise ValueError('同一試行が複数記録されています')
                    seen.add(key)
                    if raw['success'].lower() not in ('true', 'false'):
                        raise ValueError('successが真偽値ではありません')
                    if raw['countedAsTrial'].lower() not in ('true', 'false'):
                        raise ValueError('countedAsTrialが真偽値ではありません')
                    n = int(raw['trialInCondition'])
                    if n < 1:
                        raise ValueError('試行番号が1未満です')
                    samples = json.loads(raw['trajectory'])
                    points = np.array([[p['t'], p['x'], p['y']] for p in samples], dtype=float).reshape(-1, 3)
                    if not np.isfinite(points).all() or (len(points) and np.any(np.diff(points[:, 0]) < 0)):
                        raise ValueError('軌跡の座標または時刻が不正です')
                    if len(points):
                        keep = np.r_[np.diff(points[:, 0]) != 0, True]
                        points = points[keep]
                    events = json.loads(raw['deviationEvents'])
                    for event in events:
                        if not all(number(event.get(k)) is not None for k in ('startMs', 'endMs')) or event['endMs'] < event['startMs']:
                            raise ValueError('逸脱イベントの時刻が不正です')
                    entry, exit_time = number(raw['coreEntryMs']), number(raw['coreExitMs'])
                    region = event_region(events, entry, exit_time)
                    mt, core_mt = number(raw['mtMs']), number(raw['coreMtMs'])
                    core = clip_points(points, entry, exit_time)
                    deviation = raw['errorType'] == 'deviation'
                    core_error = None if deviation and region in ('unknown', 'none') else int(deviation and region in ('inside', 'both'))
                    rows.append(dict(participant=raw['participantId'], phase=raw['phase'], condition=raw['conditionId'],
                                     n=n, success=raw['success'].lower() == 'true',
                                     countedAsTrial=raw['countedAsTrial'].lower() == 'true', coreExitMs=exit_time,
                                     source=path.name, line=line,
                                     attempt=raw['attemptNumber'], errorType=raw['errorType'], region=region,
                                     IP=core_ip(core_mt, raw.get('amplitude'), raw.get('width')),
                                     coreMtMs=dict(MT=core_mt if core_mt and core_mt > 0 else None,
                                                   TPe=trajectory_metrics(core, core_mt), ER=core_error),
                                     mtMs=dict(MT=mt if mt and mt > 0 else None,
                                               TPe=trajectory_metrics(points, mt), ER=int(deviation))))
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'{path.name}:{line}: {exc}') from exc
    return rows


def average(values):
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def aggregate(rows, metric, scope):
    values = [r.get('IP') if metric == 'IP' else r[scope][metric] for r in rows]
    if metric == 'ER':
        return 100 * float(np.mean(values)) if values and all(v is not None for v in values) else None
    return average(values)


def curve(n, params):
    a0, b, e, a = params if len(params) == 4 else (params[0], params[1], 0., params[2])
    return a0 + b * (np.asarray(n) + e) ** (-a)


def fit_curve(x, y, unit_scale=1, model="shifted", a_bound="nonnegative", fixed_a=None):
    if model not in ("shifted", "power"):
        raise ValueError("回帰式が不正です")
    if a_bound not in ('nonnegative', 'free'):
        raise ValueError('Aの制約が不正です')
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float) * unit_scale
    if len(x) < 8 or len(np.unique(x)) < 8:
        return dict(ok=False, message='回帰には8点以上の異なる試行番号が必要です')
    scale = max(float(np.mean(y)), 1)
    upper = [np.inf, np.inf, float(max(x) * 10), 5]
    lower = [0, 0, 0, .001]
    if model == 'power':
        lower, upper = [0, 0, .001], [np.inf, np.inf, 5]
    if a_bound == 'free':
        lower[0] = -np.inf
    if fixed_a is not None:
        lower, upper = lower[1:], upper[1:]
    def expand(p):
        return np.r_[fixed_a * unit_scale, p] if fixed_a is not None else p
    best = None
    for e, a in [(0, .3), (10, .8), (100, 1.5)]:
        guess = ([max(float(min(y)) * .5, 0), scale, a] if model == 'power' else
                 [max(float(min(y)) * .5, 0), scale * (1 + e) ** a, min(e, upper[2] / 2), a])
        if fixed_a is not None:
            guess = guess[1:]
        fit = least_squares(lambda p: (curve(x, expand(p)) - y) / scale, guess,
                            bounds=(lower, upper), x_scale='jac', max_nfev=1500)
        if fit.success and (best is None or fit.cost < best.cost):
            best = fit
    if best is None:
        return dict(ok=False, message='回帰が収束しませんでした')
    fitted = expand(best.x)
    pred = curve(x, fitted)
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    grid = np.linspace(float(min(x)), float(max(x)), 200)
    condition = float(np.linalg.cond(best.jac))
    warning = 'パラメータの識別が不安定です' if not math.isfinite(condition) or condition > 1e8 else ''
    if np.any(best.active_mask):
        warning = (warning + ' / ' if warning else '') + 'パラメータが制約境界に達しています'
    params = fitted if model == 'shifted' else np.insert(fitted, 2, 0.)
    return dict(ok=True, aFixed=fixed_a is not None, parameters=dict(zip(('A', 'B', 'E', 'a'), map(float, params / [unit_scale, unit_scale, 1, 1]))),
                r2=1 - sse / sst if sst > 1e-12 else None, rmse=math.sqrt(sse / len(x)) / unit_scale,
                points=len(x), warning=warning, x=grid.tolist(), y=(curve(grid, fitted) / unit_scale).tolist())


def trial_range(options):
    start, end = options.get('trialStart', 1), options.get('trialEnd')
    if type(start) is not int or start < 1:
        raise ValueError('開始試行は1以上の整数で指定してください')
    if end is not None and (type(end) is not int or end < start):
        raise ValueError('終了試行は開始試行以上の整数で指定してください')
    return start, end


def boundary_trial(n):
    return any(start <= n <= end for start, end in BOUNDARY_WINDOWS)


def valid_mt(row, scope):
    value = row[scope]['MT']
    return value is not None and math.isfinite(value) and value > 0


def fixed_core_a(condition, scope, a_bound):
    if a_bound != 'fixed_core' or scope != 'coreMtMs':
        return None
    if condition not in CORE_A:
        raise ValueError(f'{condition}: core MTの固定Aが未定義です')
    return CORE_A[condition]


def mt_outliers(rows, scope, method, model="shifted", a_bound="nonnegative"):
    if method == 'none':
        return set()
    threshold = 1.5 if method in ('trial1_5sd', 'curve1_5sd') else 3
    found = set()
    if method in ('trial3sd', 'trial1_5sd'):
        groups = defaultdict(list)
        for row in rows:
            if valid_mt(row, scope):
                groups[(row['condition'], row['n'])].append(row)
        for group in groups.values():
            if len(group) < 2:
                continue
            values = np.asarray([row[scope]['MT'] for row in group], dtype=float)
            mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
            if sd > 0:
                found.update(id(row) for row, value in zip(group, values) if abs(value - mean) > threshold * sd)
        return found
    groups = defaultdict(list)
    for row in rows:
        if valid_mt(row, scope):
            groups[(row['condition'], row['participant'])].append(row)
    for group in groups.values():
        per_n = defaultdict(list)
        for row in group:
            per_n[row['n']].append(row[scope]['MT'])
        x = sorted(per_n)
        fit = fit_curve(x, [average(per_n[n]) for n in x], model=model,
                        a_bound='nonnegative' if a_bound == 'fixed_core' else a_bound,
                        fixed_a=fixed_core_a(group[0]['condition'], scope, a_bound))
        if not fit['ok']:
            continue
        params = [fit['parameters'][key] for key in ('A', 'B', 'E', 'a')]
        residuals = np.asarray([row[scope]['MT'] - float(curve(row['n'], params)) for row in group])
        if len(residuals) < 2:
            continue
        mean, sd = float(np.mean(residuals)), float(np.std(residuals, ddof=1))
        if sd > 0:
            found.update(id(row) for row, value in zip(group, residuals) if abs(value - mean) > threshold * sd)
    return found


def analyze(rows, options):
    start, end = trial_range(options)
    model = options.get('curveModel', 'shifted')
    a_bound = options.get('aBound', 'nonnegative')
    if a_bound not in ('nonnegative', 'free', 'fixed_core'):
        raise ValueError('Aの制約が不正です')
    if model not in ('shifted', 'power'):
        raise ValueError('回帰式が不正です')
    scope, phase, mode, width = options['scope'], options['phase'], options['mode'], options['bin']
    selected = [r for r in rows if r['participant'] in options['participants'] and r['phase'] == phase
                and r['condition'] in options['conditions'] and r['n'] >= start
                and (end is None or r['n'] <= end)]
    included, excluded = [], []
    for row in selected:
        reasons = []
        if options['exclude'] and not row['countedAsTrial']:
            reasons.append('countedAsTrial=false')
        if options['exclude'] and row['coreExitMs'] is None:
            reasons.append('coreExitMs欠損')
        if options.get('excludeBoundaryTrials', False) and boundary_trial(row['n']):
            reasons.append('100・200・300試行周辺10試行')
        if reasons:
            excluded.append(dict(participant=row['participant'], condition=row['condition'], n=row['n'],
                                 attempt=row['attempt'], source=row['source'], line=row['line'],
                                 reason=' / '.join(reasons)))
        else:
            included.append(row)
    outlier_method = options.get('mtOutlier', 'none')
    if outlier_method not in MT_OUTLIERS:
        raise ValueError('MT外れ値除外の設定が不正です')
    outlier_ids = mt_outliers(included, scope, outlier_method, model=model, a_bound=a_bound)
    if outlier_ids:
        threshold = 1.5 if outlier_method in ('trial1_5sd', 'curve1_5sd') else 3
        reason = (f'同一試行番号MTが平均±{threshold}SD超過' if outlier_method in ('trial3sd', 'trial1_5sd')
                  else f'習熟曲線MT残差が平均±{threshold}SD超過')
        kept = []
        for row in included:
            if id(row) in outlier_ids:
                excluded.append(dict(participant=row['participant'], condition=row['condition'], n=row['n'],
                                     attempt=row['attempt'], source=row['source'], line=row['line'], reason=reason))
            else:
                kept.append(row)
        included = kept
    groups = defaultdict(list)
    for r in included:
        groups[(r['condition'], r['participant'])].append(r)
    summaries, series, fits, inverse_fits = [], [], [], []
    for condition in options['conditions']:
        members = {p: group for (c, p), group in groups.items() if c == condition}
        if not members:
            continue
        labels = sorted(members) if mode == 'individual' else ['Mean']
        for label in labels:
            active = members if label == 'Mean' else {label: members[label]}
            summary = dict(condition=condition, participant=label, rows=sum(map(len, active.values())), participants=len(active))
            for metric in ('MT', 'ER', 'TPe', 'IP'):
                estimates = [aggregate(g, metric, scope) for g in active.values()]
                summary[metric] = average(estimates) if metric != 'ER' or all(v is not None for v in estimates) else None
                summary[metric + 'Participants'] = sum(v is not None for v in estimates)
            summaries.append(summary)
            bins = sorted({(r['n'] - 1) // width for group in active.values() for r in group})
            line = dict(condition=condition, participant=label, x=[], MT=[], ER=[], TPe=[], IP=[], counts=[])
            for block in bins:
                block_groups = [[r for r in group if (r['n'] - 1) // width == block] for group in active.values()]
                block_groups = [g for g in block_groups if g]
                left = max(block * width + 1, start)
                right = min((block + 1) * width, end) if end is not None else (block + 1) * width
                line['x'].append((left + right) / 2)
                line['counts'].append(sum(map(len, block_groups)))
                for metric in ('MT', 'ER', 'TPe', 'IP'):
                    values = [aggregate(g, metric, scope) for g in block_groups]
                    line[metric].append(average(values) if metric != 'ER' or all(v is not None for v in values) else None)
            series.append(line)
            if phase == 'practice':
                for metric, target in [('MT', fits), ('TPe', inverse_fits)]:
                    per_n = defaultdict(list)
                    for group in active.values():
                        local = defaultdict(list)
                        for r in group:
                            value = r[scope][metric]
                            if value is not None and math.isfinite(value) and value > 0:
                                value = 1 / value if metric == 'TPe' else value
                                if math.isfinite(value):
                                    local[r['n']].append(value)
                        for n, values in local.items():
                            per_n[n].append(average(values))
                    x = sorted(per_n)
                    y = [average(per_n[n]) for n in x]
                    target.append(dict(condition=condition, participant=label, observedX=x, observedY=y,
                                       **fit_curve(x, y, unit_scale=1000 if metric == 'TPe' else 1, model=model,
                                                   a_bound='nonnegative' if a_bound == 'fixed_core' else a_bound,
                                                   fixed_a=fixed_core_a(condition, scope, a_bound) if metric == 'MT' else None)))
    regions = {key: sum(r['region'] == key for r in included if r['errorType'] == 'deviation')
               for key in ('inside', 'outside', 'both', 'unknown', 'none')}
    missing = {metric: sum(r[scope][metric] is None for r in included) for metric in ('MT', 'ER', 'TPe')}
    missing['IP'] = sum(r.get('IP') is None for r in included)
    return dict(summary=summaries, series=series, fits=fits, inverseFits=inverse_fits, excluded=excluded, regions=regions,
                trialRange=dict(start=start, end=end), missing=missing, selected=len(selected), included=len(included), options=options)
