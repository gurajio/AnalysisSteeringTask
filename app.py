import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))

import argparse
import base64
import io
import json
import re
import secrets
import webbrowser
from collections import OrderedDict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from analysis import MT_OUTLIERS, REGRESSION_MODELS, analyze, analyze_regressions, load_rows, trial_range

ROWS = []
RESULTS = OrderedDict()
COLORS = {}
CONDITION_COLORS = {'C1': '#1f77b4', 'C2': '#ff7f0e', 'C3': '#2ca02c'}


def plot_result(result, key):
    learning = key in ('learning', 'learning_inverse_TPe')
    metric = ('1/TPe' if key == 'learning_inverse_TPe' else 'MT') if learning else key.split('_')[-1]
    fig = Figure(figsize=(16, 10), dpi=100, layout='constrained')
    ax = fig.subplots()
    options = result['options']
    conditions = options['conditions']
    labels = sorted({s['participant'] for s in result['summary']})
    styles = ['-', '--', ':', '-.']
    if key.startswith('mean_'):
        if options['mode'] == 'mean':
            for i, condition in enumerate(conditions):
                row = next((s for s in result['summary'] if s['participant'] == 'Mean' and s['condition'] == condition), None)
                value = row[metric] if row and row[metric] is not None else float('nan')
                ax.bar(i, value, width=.65, label=condition, color=CONDITION_COLORS.get(condition, '#0f766e'))
        else:
            for j, label in enumerate(labels):
                xs, ys = [], []
                for i, condition in enumerate(conditions):
                    row = next((s for s in result['summary'] if s['participant'] == label and s['condition'] == condition), None)
                    xs.append(i + (j - (len(labels) - 1) / 2) * .65 / max(len(labels), 1))
                    ys.append(row[metric] if row and row[metric] is not None else float('nan'))
                ax.bar(xs, ys, width=.65 / max(len(labels), 1), label=label, color=COLORS.get(label, '#0f766e'))
        ax.set_xticks(range(len(conditions)), conditions)
        ax.set_xlabel('Condition')
    elif learning:
        for fit in result['inverseFits' if metric == '1/TPe' else 'fits']:
            label = f"{fit['participant']} / {fit['condition']}"
            color = (CONDITION_COLORS.get(fit['condition'], '#0f766e') if options['mode'] == 'mean'
                     else COLORS.get(fit['participant'], '#0f766e'))
            ax.scatter(fit['observedX'], fit['observedY'], s=10, alpha=.3, color=color)
            if fit['ok']:
                ax.plot(fit['x'], fit['y'], color=color, linestyle=styles[conditions.index(fit['condition']) % 4], label=label)
        ax.set_xlabel('Original trial number within condition (N)')
    else:
        for line in result['series']:
            y = [v if v is not None else float('nan') for v in line[metric]]
            color = (CONDITION_COLORS.get(line['condition'], '#0f766e') if options['mode'] == 'mean'
                     else COLORS.get(line['participant'], '#0f766e'))
            ax.plot(line['x'], y, marker='o', markersize=4, color=color,
                    linestyle=styles[conditions.index(line['condition']) % 4], label=f"{line['participant']} / {line['condition']}")
        ax.set_xlabel(f"Original trial number / {options['bin']}-trial bins (midpoint)")
    start, end = trial_range(options)
    if not key.startswith('mean_') and (start > 1 or end is not None):
        last = end if end is not None else max(
            [start] + [x for line in result['series'] for x in line['x']]
            + [x for fit in result['fits'] for x in fit['observedX']])
        ax.set_xlim(start - 1, max(last, start))
    unit = {'MT': 'ms', 'ER': '%', 'TPe': '1/s', '1/TPe': 's', 'IP': 'ms'}[metric]
    ax.set_ylabel(f'{metric} ({unit})')
    plot_scope = 'coreMtMs / (amplitude / width)' if metric == 'IP' else options['scope']
    ax.set_title(f"{'Learning curve: ' + metric if learning else key.replace('_', ' ').title()} | {options['phase']} | {plot_scope}", loc='left', fontsize=20)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    if metric == 'ER':
        ax.set_ylim(0, 100)
    elif not (learning and options.get('aBound') == 'free'):
        ax.set_ylim(bottom=0)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=10)
    if not result['summary']:
        ax.text(.5, .5, 'No data for the current selection', transform=ax.transAxes, ha='center')
    if metric == 'IP' and result['summary'] and all(v is None for line in result['series'] for v in line['IP']):
        ax.text(.5, .5, 'IP unavailable: valid core MT and nominal ID required', transform=ax.transAxes, ha='center')
    boundary_note = 'Exclude boundary trials' if options['excludeBoundaryTrials'] else 'Keep boundary trials'
    outlier_note = {'none': 'MT outliers: none', 'trial3sd': 'MT outliers: trial mean ±3SD',
                    'curve3sd': 'MT outliers: curve residual ±3SD',
                    'trial1_5sd': 'MT outliers: trial mean ±1.5SD',
                    'curve1_5sd': 'MT outliers: curve residual ±1.5SD'}[options['mtOutlier']]
    formula = 'A+B*N^(-a)' if options.get('curveModel') == 'power' else 'A+B*(N+E)^(-a)'
    bound_note = 'A unrestricted' if options.get('aBound') == 'free' else 'A >= 0'
    if options.get('aBound') == 'fixed_core':
        bound_note = 'Core MT A fixed: C1=84, C2=253, C3=315 ms; other fits A >= 0'
    fig.supxlabel(f"Model: {formula} | {bound_note}\n"
                  f"{'Keep counted trials with coreExitMs' if options['exclude'] else 'All attempts'} | {boundary_note} | {outlier_note} | {options['mode']} | "
                  f"trials={start}-{end if end is not None else 'all'} | rows={result['included']} | participants={','.join(options['participants'])}", fontsize=10)
    data = io.BytesIO()
    fig.savefig(data, format='png', dpi=100)
    return data.getvalue()


def plot_regression(result, model):
    fig = Figure(figsize=(16, 10), dpi=100, layout='constrained')
    ax = fig.subplots()
    options = result['options']
    conditions = options['conditions']
    styles = ['-', '--', ':', '-.']
    for fit in result['regressionFits'][model]:
        color = (CONDITION_COLORS.get(fit['condition'], '#0f766e') if options['mode'] == 'mean'
                 else COLORS.get(fit['participant'], '#0f766e'))
        label = f"{fit['participant']} / {fit['condition']}"
        ax.scatter(fit['observedX'], fit['observedY'], s=10, alpha=.3, color=color)
        if fit['ok']:
            ax.plot(fit['x'], fit['y'], color=color,
                    linestyle=styles[conditions.index(fit['condition']) % len(styles)], label=label)
    start, end = trial_range(options)
    if start > 1 or end is not None:
        last = end if end is not None else max([start] + [x for fit in result['regressionFits'][model] for x in fit['observedX']])
        ax.set_xlim(start - 1, max(last, start))
    details = REGRESSION_MODELS[model]
    ax.set_xlabel('Original trial number within condition (N)')
    ax.set_ylabel('MT (ms)')
    ax.set_title(f"{details['plotTitle']} | {details['formula']} | {options['scope']}", loc='left', fontsize=20)
    ax.set_ylim(bottom=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=10)
    if not result['regressionFits'][model]:
        message = 'Regression is available for practice data only' if options['phase'] != 'practice' else 'No data for the current selection'
        ax.text(.5, .5, message, transform=ax.transAxes, ha='center')
    boundary = 'Exclude boundary trials' if options['excludeBoundaryTrials'] else 'Keep boundary trials'
    fig.supxlabel(f"{'Keep counted trials with coreExitMs' if options['exclude'] else 'All attempts'} | {boundary} | "
                  f"outliers={options['mtOutlier']} | {options['mode']} | trials={start}-{end if end is not None else 'all'} | "
                  f"rows={result['included']} | participants={','.join(options['participants'])}", fontsize=10)
    data = io.BytesIO()
    fig.savefig(data, format='png', dpi=100)
    return data.getvalue()


def metadata():
    participants = sorted({r['participant'] for r in ROWS})
    conditions = sorted({r['condition'] for r in ROWS})
    by_condition = {condition: sorted({r['participant'] for r in ROWS
                                       if r['phase'] == 'practice' and r['condition'] == condition})
                    for condition in conditions}
    assigned = {participant for members in by_condition.values() for participant in members}
    unassigned = sorted(set(participants) - assigned)
    return dict(participants=participants, conditions=conditions, participantsByCondition=by_condition,
                unassignedParticipants=unassigned, rows=len(ROWS),
                files=len({r['source'] for r in ROWS}))


def validate(data):
    start, end = trial_range(data)
    data = dict(data, trialStart=start, trialEnd=end,
                excludeBoundaryTrials=data.get('excludeBoundaryTrials', False),
                mtOutlier=data.get('mtOutlier', 'none'), curveModel=data.get('curveModel', 'shifted'),
                aBound=data.get('aBound', 'nonnegative'))
    if data['curveModel'] not in ('shifted', 'power'):
        raise ValueError('回帰式が不正です')
    if data['aBound'] not in ('nonnegative', 'free', 'fixed_core'):
        raise ValueError('Aの制約が不正です')
    meta = metadata()
    if data.get('scope') not in ('coreMtMs', 'mtMs') or data.get('phase') not in ('practice', 'post'):
        raise ValueError('対象区間またはphaseが不正です')
    if data.get('mode') not in ('individual', 'mean') or data.get('bin') not in (10, 50, 100):
        raise ValueError('表示方法または区間幅が不正です')
    if not isinstance(data.get('exclude'), bool) or not isinstance(data.get('excludeBoundaryTrials'), bool):
        raise ValueError('除外設定が不正です')
    if data.get('mtOutlier') not in MT_OUTLIERS:
        raise ValueError('MT外れ値除外の設定が不正です')
    for key in ('participants', 'conditions'):
        if not isinstance(data.get(key), list) or any(v not in meta[key] for v in data[key]):
            raise ValueError('選択が不正です')
        data[key] = sorted(set(data[key]))
    return {k: data[k] for k in ('scope', 'phase', 'mode', 'bin', 'exclude', 'excludeBoundaryTrials', 'mtOutlier', 'curveModel', 'aBound',
                                  'participants', 'conditions', 'trialStart', 'trialEnd')}


class Handler(BaseHTTPRequestHandler):
    def respond(self, code, body, mime='application/json; charset=utf-8'):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def trusted(self):
        host = self.headers.get('Host', '')
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        origin = self.headers.get('Origin')
        return host in allowed and (origin is None or origin in {'http://' + h for h in allowed})

    def do_GET(self):
        if not self.trusted():
            return self.respond(403, dict(error='ローカル接続のみ許可しています'))
        if self.path == '/':
            self.respond(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/regression':
            self.respond(200, (ROOT / 'regression.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/api/meta':
            self.respond(200, metadata())
        else:
            self.respond(404, dict(error='Not found'))

    def do_POST(self):
        global ROWS
        if not self.trusted():
            return self.respond(403, dict(error='ローカル接続のみ許可しています'))
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 65536:
                raise ValueError('リクエストのサイズが不正です')
            data = json.loads(self.rfile.read(length))
            if self.path == '/api/reload':
                loaded = load_rows(ROOT / 'inputCSV')
                ROWS = loaded
                set_colors()
                RESULTS.clear()
                return self.respond(200, metadata())
            if self.path == '/api/analyze':
                options = validate(data)
                result = analyze(ROWS, options, include_fits=False)
                keys = [f'{kind}_{metric}' for kind in ('mean', 'trend') for metric in ('MT', 'ER', 'TPe')]
                keys.append('trend_IP')
                charts = {k: plot_result(result, k) for k in keys}
                result.pop('fits')
                result.pop('inverseFits')
                token = secrets.token_hex(16)
                RESULTS[token] = (options, charts)
                while len(RESULTS) > 4:
                    RESULTS.popitem(last=False)
                result['token'] = token
                result['charts'] = {k: base64.b64encode(v).decode() for k, v in charts.items()}
                return self.respond(200, result)
            if self.path == '/api/regression':
                options = validate(data)
                result = analyze_regressions(ROWS, options)
                charts = ({f'regression_{key}': plot_regression(result, key) for key in REGRESSION_MODELS}
                          if options['phase'] == 'practice' else {})
                token = secrets.token_hex(16)
                RESULTS[token] = (options, charts)
                while len(RESULTS) > 4:
                    RESULTS.popitem(last=False)
                result['token'] = token
                result['charts'] = {k: base64.b64encode(v).decode() for k, v in charts.items()}
                return self.respond(200, result)
            if self.path == '/api/save':
                if data.get('token') not in RESULTS:
                    raise ValueError('表示の保存期限が切れました。再分析してください')
                options, charts = RESULTS[data['token']]
                key = data.get('key')
                if key not in charts:
                    raise ValueError('図が見つかりません')
                stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
                boundary = 'boundary-excluded' if options['excludeBoundaryTrials'] else 'boundary-kept'
                if key == 'trend_IP':
                    key_name = 'trend_IP_coreMtMs_nominalID_ms'
                else:
                    key_name = key
                name = f"{key_name}_{options['curveModel']}_A-{options['aBound']}_trials{options['trialStart']}-{options['trialEnd'] if options['trialEnd'] is not None else 'all'}_{options['phase']}_{options['scope']}_{'-'.join(options['conditions'])}_{options['mode']}_bin{options['bin']}_{'counted-core-exit' if options['exclude'] else 'all'}_{boundary}_{options['mtOutlier']}_P{'-'.join(options['participants'])}_{stamp}.png"
                name = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
                if len(name.encode()) > 240:
                    name = name[:170] + '_' + stamp + '.png'
                folder = ROOT / 'output'
                folder.mkdir(exist_ok=True)
                path = folder / name
                with path.open('xb') as stream:
                    stream.write(charts[key])
                return self.respond(200, dict(path=str(path)))
            self.respond(404, dict(error='Not found'))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.respond(400, dict(error=str(exc)))
        except Exception as exc:
            self.log_error('Analysis failed: %s', exc)
            self.respond(500, dict(error='分析中にエラーが発生しました。ターミナルのログを確認してください'))


def set_colors():
    global COLORS
    palette = matplotlib.colormaps['tab20']
    COLORS = {p: palette(i % 20) for i, p in enumerate(metadata()['participants'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    print('使用コア数: 1（数値計算の内部並列化も1に制限）', flush=True)
    ROWS = load_rows(ROOT / 'inputCSV')
    set_colors()
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{args.port}'
    print(f"{len(ROWS)}行を読み込みました。{url}  終了: Ctrl+C", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
