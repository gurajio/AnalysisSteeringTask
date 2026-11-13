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
REGRESSION_MODELS = {
    'power_rate': dict(title='べき乗則と習熟率', plotTitle='Power law and learning rate', formula='T̄(N) = aN^b,  b = log(r) / log(2)', parameters=('a', 'b', 'r')),
    'asymptotic_power': dict(title='終端値付きべき乗則', plotTitle='Asymptotic power law', formula='T(n) = a + bn^(-c)', parameters=('a', 'b', 'c')),
    'exponential': dict(title='指数関数的習熟', plotTitle='Exponential learning', formula='T(n) = a + b exp(-cn)', parameters=('a', 'b', 'c')),
    'lower_limit': dict(title='作業時間の下限', plotTitle='Lower-bound task time', formula='T(n) = (a - c)n^(-b) + c', parameters=('a', 'b', 'c')),
    'unlearnable': dict(title='習熟できない作業時間', plotTitle='Unlearnable task time', formula='T(n) = a[M + (1 - M)n^(-b)]', parameters=('a', 'M', 'b')),
    'experience': dict(title='過去の作業経験', plotTitle='Prior work experience', formula='T(n) = a(n + B)^(-b)', parameters=('a', 'B', 'b')),
    'changing_rate': dict(title='変化する習熟率', plotTitle='Changing learning rate', formula='T(n) = a n^[b / (1 + n/c)]', parameters=('a', 'b', 'c')),
}


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


def regression_value(model, n, params):
    n = np.asarray(n, dtype=float)
    if model == 'power_rate':
        return params[0] * n ** params[1]
    if model == 'asymptotic_power':
        return params[0] + params[1] * n ** (-params[2])
    if model == 'exponential':
        return params[0] + params[1] * np.exp(-params[2] * n)
    if model == 'lower_limit':
        return params[1] * n ** (-params[2]) + params[0]
    if model == 'unlearnable':
        return params[0] * (params[1] + (1 - params[1]) * n ** (-params[2]))
    if model == 'experience':
        return params[0] * (n + params[1]) ** (-params[2])
    if model == 'changing_rate':
        exponent = params[1] / (1 + n / params[2])
        return np.exp(np.clip(np.log(params[0]) + exponent * np.log(n), -700, 700))
    raise ValueError('回帰モデルが不正です')


def regression_setup(model, x, y):
    first, minimum = float(y[0]), float(np.min(y))
    scale, last_n = max(float(np.mean(y)), 1), float(max(x))
    if model == 'power_rate':
        guesses = [[max(first / x[0] ** b, 1e-9), b] for b in (-.1, -.5, -1.5)]
        return [0, -np.inf], [np.inf, 0], guesses
    if model in ('asymptotic_power', 'lower_limit'):
        guesses = []
        for c in (.3, .8, 1.5):
            floor = max(minimum * .8, 0)
            span = max((first - floor) * x[0] ** c, scale * .1)
            guesses.append([floor, span, c])
        return [0, 0, .001], [np.inf, np.inf, 5], guesses
    if model == 'exponential':
        guesses = []
        for c in (1 / last_n, 5 / last_n, min(.1, 20 / last_n)):
            floor = max(minimum * .8, 0)
            span = max((first - floor) * math.exp(min(c * x[0], 50)), scale * .1)
            guesses.append([floor, span, c])
        return [0, 0, .000001], [np.inf, np.inf, 5], guesses
    if model == 'unlearnable':
        guesses = []
        for m, b in ((.2, .3), (.5, .8), (.8, 1.5)):
            divisor = m + (1 - m) * x[0] ** (-b)
            guesses.append([max(first / divisor, 1e-9), m, b])
        return [0, 0, .001], [np.inf, 1, 5], guesses
    if model == 'experience':
        guesses = [[max(first * (x[0] + shift) ** b, 1e-9), shift, b]
                   for shift, b in ((0, .3), (10, .8), (100, 1.5))]
        return [0, 0, .001], [np.inf, last_n * 10, 5], guesses
    if model == 'changing_rate':
        guesses = []
        for b, c in ((-.1, 10), (-.5, 100), (-1.5, max(last_n, 1))):
            c = min(c, last_n * 10)
            divisor = x[0] ** (b / (1 + x[0] / c))
            guesses.append([max(first / divisor, 1e-9), b, c])
        return [1e-12, -np.inf, .001], [np.inf, np.inf, last_n * 10], guesses
    raise ValueError('回帰モデルが不正です')


def public_parameters(model, fitted):
    if model == 'power_rate':
        return dict(a=float(fitted[0]), b=float(fitted[1]), r=float(2 ** fitted[1]))
    if model == 'lower_limit':
        return dict(a=float(fitted[0] + fitted[1]), b=float(fitted[2]), c=float(fitted[0]))
    names = REGRESSION_MODELS[model]['parameters']
    return dict(zip(names, map(float, fitted)))


def fit_regression_model(x, y, model):
    if model not in REGRESSION_MODELS:
        raise ValueError('回帰モデルが不正です')
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 8 or len(np.unique(x)) < 8:
        return dict(ok=False, message='回帰には8点以上の異なる試行番号が必要です')
    scale = max(float(np.mean(y)), 1)
    lower, upper, guesses = regression_setup(model, x, y)
    best = None
    for guess in guesses:
        try:
            fit = least_squares(lambda p: (regression_value(model, x, p) - y) / scale, guess,
                                bounds=(lower, upper), x_scale='jac', max_nfev=1500)
        except (ValueError, FloatingPointError):
            continue
        if fit.success and np.isfinite(fit.fun).all() and (best is None or fit.cost < best.cost):
            best = fit
    if best is None:
        return dict(ok=False, message='回帰が収束しませんでした')
    pred = regression_value(model, x, best.x)
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    grid = np.linspace(float(min(x)), float(max(x)), 200)
    condition = float(np.linalg.cond(best.jac))
    warning = 'パラメータの識別が不安定です' if not math.isfinite(condition) or condition > 1e8 else ''
    if np.any(best.active_mask):
        warning = (warning + ' / ' if warning else '') + 'パラメータが制約境界に達しています'
    return dict(ok=True, parameters=public_parameters(model, best.x),
                r2=1 - sse / sst if sst > 1e-12 else None, rmse=math.sqrt(sse / len(x)),
                points=len(x), warning=warning, x=grid.tolist(),
                y=regression_value(model, grid, best.x).tolist())


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


def analyze(rows, options, include_fits=True):
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
            if include_fits and phase == 'practice':
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


def analyze_regressions(rows, options):
    result = analyze(rows, options)
    fits = {key: [] for key in REGRESSION_MODELS}
    if options['phase'] == 'practice':
        for observed in result['fits']:
            for key in REGRESSION_MODELS:
                fits[key].append(dict(condition=observed['condition'], participant=observed['participant'],
                                      observedX=observed['observedX'], observedY=observed['observedY'],
                                      **fit_regression_model(observed['observedX'], observed['observedY'], key)))
    result['regressionFits'] = fits
    result['regressionModels'] = [dict(key=key, **details) for key, details in REGRESSION_MODELS.items()]
    return result
