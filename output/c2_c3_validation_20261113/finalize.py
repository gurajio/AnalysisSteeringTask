import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/steering-validation-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/steering-validation-cache')
from pathlib import Path
import csv
import hashlib
import html
import json
import math
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_analysis import save_csv, json_save

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
COLORS={'C2':'#ff7f0e','C3':'#2ca02c'}
LABELS={'change_C2-C3':'変化量の条件差','late_C2-C3':'後期成績の条件差','change_C2':'C2の前後変化','change_C3':'C3の前後変化',
        'early_C2-C3':'初期成績の条件差','early_slope_C2-C3':'初期傾きの条件差','late_slope_C2-C3':'後期傾きの条件差',
        'log_late_sd_C2-C3':'後期・対数残差SDの条件差','late_variance':'後期平均の個人間分散','change_variance':'変化量の個人間分散','curve_shape':'曲線形状の条件差'}


def read_csv(name):
    with (HERE/name).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def holm(values):
    p=np.asarray(values,float); order=np.argsort(p); result=np.empty(len(p))
    result[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return result


def fmt(value, places=3):
    try:
        x=float(value)
        if not math.isfinite(x):return '—'
        return f'{x:.{places}f}'
    except (ValueError,TypeError):return '—'


def pformat(value):
    try:return '<.0001' if float(value)<.0001 else f'{float(value):.4f}'
    except (ValueError,TypeError):return '—'


def participant_plot(records,metric):
    fig,axes=plt.subplots(1,2,figsize=(16,10),dpi=100)
    unit={'MT':'ms','TPe':'s^-1','ER':'%'}[metric]
    rng=np.random.default_rng(20261113)
    for i,c in enumerate(('C2','C3')):
        group=[r for r in records if r['condition']==c and r['metric']==metric and r['variant']=='base']
        y=np.asarray([[float(r['early']),float(r['late'])] for r in group])
        for row in y:axes[0].plot([0,1],row,color=COLORS[c],alpha=.3,linewidth=1)
        means=y.mean(axis=0);ci=stats.t.ppf(.975,9)*y.std(axis=0,ddof=1)/np.sqrt(10)
        axes[0].errorbar([i*.045,1+i*.045],means,yerr=ci,color=COLORS[c],label=c+' mean (95% CI)',linewidth=3,marker='o',capsize=5)
        change=y[:,1]-y[:,0]
        axes[1].scatter(i+rng.uniform(-.1,.1,len(change)),change,color=COLORS[c],s=45,alpha=.7)
        axes[1].errorbar(i+.2,change.mean(),yerr=stats.t.ppf(.975,9)*change.std(ddof=1)/np.sqrt(10),color=COLORS[c],fmt='D',capsize=6,markersize=8)
    axes[0].set(xticks=[0,1],xticklabels=['Early: 1-100','Late: 301-400'],ylabel=f'{metric} ({unit})',title='Participant means and group means')
    axes[0].legend(frameon=False)
    axes[1].set(xticks=[0,1],xticklabels=['C2','C3'],ylabel=f'Late minus early ({"percentage points" if metric=="ER" else unit})',title='Changes: each dot is one participant')
    axes[1].axhline(0,color='#555555',linewidth=1)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'{metric}: practice, core interval | C2 vs C3, n = 10 each',fontsize=20)
    fig.text(.5,.025,'Basic eligibility applied; no MT outlier removal; no rest-boundary removal.\nIntervals are individual 95% confidence intervals; they are not multiplicity-adjusted.',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(HERE/f'participants_{metric}.png',dpi=100);plt.close(fig)


def curve_plot(metric,diagnostics):
    r=read_csv(f'gamm_curve_base_{metric}.csv')
    arrays={k:np.array([float(row[k]) for row in r]) for k in r[0]}
    fig,axes=plt.subplots(2,1,figsize=(16,10),dpi=100,sharex=True)
    for c in ('C2','C3'):
        axes[0].plot(arrays['N'],arrays[c],color=COLORS[c],label=c)
        axes[0].fill_between(arrays['N'],arrays[c+'_low'],arrays[c+'_high'],color=COLORS[c],alpha=.15)
    axes[0].set(ylabel={'MT':'MT (ms)','TPe':'TPe (s^-1)','ER':'Error probability (%)'}[metric],title='Condition curves; pointwise 95% intervals')
    axes[0].legend(frameon=False)
    axes[1].plot(arrays['N'],arrays['centered_difference'],color='#333333')
    axes[1].fill_between(arrays['N'],arrays['simultaneous_low'],arrays['simultaneous_high'],color='#777777',alpha=.2)
    axes[1].axhline(0,color='#555555',linestyle='--')
    axes[1].set(xlabel='Original practice trial number N',ylabel='Centered C2 - C3 ('+{'MT':'ms','TPe':'s^-1','ER':'logit'}[metric]+')',title='Shape difference after subtracting its 1-400 mean; simultaneous 95% band')
    for ax in axes:
        for boundary in (100.5,200.5,300.5):ax.axvline(boundary,color='#666666',linestyle=':',alpha=.5)
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    diag=diagnostics[f'gamm_base_{metric}']
    status='NUMERICAL CONVERGENCE FAILED' if not diag['converged'] else 'Exploratory model; inspect basis and residual diagnostics'
    fig.suptitle(f'{metric}: GAMM | {status}',fontsize=17)
    fig.text(.5,.02,'N = 1-400; basic eligibility; no MT outlier removal; participant smooths and within-block AR correction.\nER curves set participant effects to zero; ER AR correction and all curve tests are approximate.',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(HERE/f'curve_{metric}.png',dpi=100);plt.close(fig)


def main():
    np.testing.assert_allclose(holm([.01,.04,.03,.8]),[.04,.09,.09,.8])
    diagnostics=json.loads((HERE/'model_diagnostics.json').read_text())
    tests=read_csv('continuous_tests.csv')+read_csv('model_tests.csv')
    for r in tests:
        r['status']='estimated'
        if r['test']=='curve_shape' or (r['metric']=='ER' and 'slope' in r['test']):
            d=diagnostics[f"gamm_{r['variant']}_{r['metric']}"]
            r['status']='exploratory_approximation'
            kcheck=read_csv(f"gamm_kcheck_{r['variant']}_{r['metric']}.csv")
            if any(float(v['p-value'])<.05 for v in kcheck if v.get('p-value') not in (None,'NA','')):
                r['status']='basis_diagnostic_warning_reference_only'
            if not d['converged'] or not d['initial_converged']:
                r['model_reference_p']=r['p_raw'];r['p_raw']='';r['status']='not_estimable_nonconvergence'
    for variant in ('base','sensitivity'):
        for family,size in [('primary',1),('secondary',11),('exploratory',18)]:
            rows=[r for r in tests if r['variant']==variant and r['family']==family]
            assert len(rows)==size,(variant,family,len(rows))
            for r,p in zip(rows,holm([float(r['p_raw']) if r['p_raw'] else 1 for r in rows])):
                r['family_size']=size;r['p_adjusted']=float(p) if r['p_raw'] else ''
    tests.sort(key=lambda r:(r['variant']!='base',{'primary':0,'secondary':1,'exploratory':2}[r['family']],r['metric'],r['test']))
    save_csv('all_tests.csv',tests)
    records=read_csv('participant_summary.csv');descriptives=read_csv('descriptives.csv')
    for metric in ('MT','TPe','ER'):
        participant_plot(records,metric);curve_plot(metric,diagnostics)
    primary=next(r for r in tests if r['variant']=='base' and r['family']=='primary')
    sensitivity=next(r for r in tests if r['variant']=='sensitivity' and r['family']=='primary')
    robust=json.loads((HERE/'robustness.json').read_text())
    significant=[r for r in tests if r['variant']=='base' and r['family']=='secondary' and r['p_adjusted']!='' and r['p_adjusted']<.05]
    shape=next(r for r in tests if r['variant']=='base' and r['metric']=='MT' and r['test']=='curve_shape')
    lines=[
      'C2・C3 練習成績の統計再解析 — 2026-11-13',
      '結果閲覧後に計画を固定した再解析。収集前の事前登録や独立した確認的検証ではない。',
      '',
      f"主分析：MTの（後期−初期）の条件差 C2−C3 = {fmt(primary['estimate'],1)} ms。",
      f"95%CI [{fmt(primary['ci_low'],1)}, {fmt(primary['ci_high'],1)}] ms、F(1,18)={fmt(primary['F'])}、p={pformat(primary['p_raw'])}、Hedges g={fmt(primary['standardized'])}。",
      'MT改善量の条件差を支持する十分な証拠は得られなかった。差の不存在・同等性は証明していない。',
      '主分析はこの1検定のみ。副次的な群内変化・後期差を主分析の代替にしない。',
      '',
      '設計・対象',
      'ユーザー申告：無作為割り当て、各群10人を事前に決定、装置変更なし、100試行ごとに1分休憩。割り当て手順の記録自体は未監査。',
      'C2：幅70・距離1350、C3：幅35・距離675。公称IDは同じ。practiceのみ、C1とpostは分析対象外。',
      '独立な推論単位は各群10人、計20人。初期1〜100、後期301〜400、曲線1〜400。初期は練習前の独立した測定ではない。',
      '基本採用条件 countedAsTrial=true かつ coreExitMs欠損なし。core MT・core軌跡TPe・core内deviation ERを使用。',
      'C2/C3候補8,309行、基本除外326行、採用7,983行。除外理由は重複あり。入力CSVは変更していない。',
      '採用行に3指標の計算不能値なし。各人の初期・後期の採用数は94〜100。全員を対応のある解析に使用した。',
      '±1.5SD感度分析はさらに829行を除外し7,154行。比較する集団・ばらつき・ER自体を変える処理なので本解析に昇格させない。',
      '',
      '参加者を等重みとした記述統計（平均 ± 参加者間SD）',
      '指標 | 条件 | 初期 | 後期 | 後期−初期',
    ]
    for metric in ('MT','TPe','ER'):
        for c in ('C2','C3'):
            vals=[next(r for r in descriptives if r['variant']=='base' and r['metric']==metric and r['condition']==c and r['quantity']==q) for q in ('early','late','change')]
            lines.append(f"{metric} | {c} | "+' | '.join(f"{fmt(v['mean'],2)} ± {fmt(v['sd'],2)}" for v in vals))
    lines += ['', '単位：MTはms、TPeはs^-1、ERの初期・後期は%、変化量はパーセントポイント。', '',
      '副次分析（11検定を一括Holm補正）',
      '副次分析にはMT/TPe/ERの後期差・各群内変化9検定とTPe/ERの変化量の条件差2検定を含む。',
      'Holm補正後p<.05：'+('、'.join(r['metric']+' '+LABELS[r['test']]+' (p='+pformat(r['p_adjusted'])+')' for r in significant) if significant else '該当なし'),
      '群内で有意か否かの違いから、条件間の改善量の差を結論しない。', '',
      f"探索的なMT形状の共同Wald検定はHolm補正p={pformat(shape['p_adjusted'])}。これは休憩ブロックごとの段差と連続的変化を含む条件差で、改善量の優越や非線形な学習機構の差を直接示さない。",
      '最終モデルでは全GAMMが収束し、基底次元診断に5%基準の警告はなかった。ただしMTのAR補正後にも小さな残存自己相関があり、近似モデルへの依存は残る。', '',
      '主分析の感度・頑健性確認',
      f"Welch p={pformat(robust[0]['welch']['p_raw'])}。参加者ラベルの全184,756通りの置換検定 p={pformat(robust[0]['permutation_p'])}。",
      f"参加者単位の層別bootstrap 9,999回のpercentile 95%CI [{fmt(robust[0]['bootstrap_percentile_ci'][0],1)}, {fmt(robust[0]['bootstrap_percentile_ci'][1],1)}] ms。",
      f"±1.5SD感度分析：差={fmt(sensitivity['estimate'],1)} ms、95%CI [{fmt(sensitivity['ci_low'],1)}, {fmt(sensitivity['ci_high'],1)}]、p={pformat(sensitivity['p_raw'])}。",
      '置換検定はラベル交換可能性を仮定する。無作為化にブロック等の制約があった場合は、その方式に合わせた再計算が必要。', '',
      '方法と解釈上の制約',
      'MT/TPeの交互作用は参加者変化量の等分散t検定の二乗と一致する2×2混合計画ANOVA。後期の単純効果には後期固有の群内分散を使用した。',
      '群内変化は2時点RMANOVA＝対応t検定の二乗。初期と区間内傾きはWelch。残差SDは後期直線回帰残差の標本SDを対数変換した。',
      '個人間分散は後期平均・変化量の双方について中央値Levene（Brown–Forsythe）。連続指標の初期・傾き・ばらつき12検定、ER初期1検定、曲線3検定、ER傾き2検定を探索18検定としてHolm補正。',
      'ERは二項GLMM（参加者ランダム切片＋独立な時期ランダム傾き）。参加者効果を正規分布で積分した周辺ERの差を報告。ロジット交互作用をERの改善ポイント差として使っていない。',
      'ERのCIとp値は1,999回のパラメトリックbootstrapのSEによる正規近似Wald。percentile CIもCSVに保存。20人での近似推論であり、正確検定ではない。',
      'bootstrapは全回で有限な値を返したが、一部に収束・Hessianの警告がある。警告付きの推定も保持しており、有効件数は警告のない収束件数を意味しない。境界推定の件数も診断JSONに記録した。',
      '曲線は条件×休憩ブロック、共通・差の平滑項k=10、参加者の平滑項k=5を固定。自己相関は初回モデル残差から1回推定し、休憩・欠落番号でAR系列を切った。',
      '形状は全400試行で中心化した条件差の近似共同Wald検定。成績水準の一定差を除く。ER形状はlogit尺度、ER傾きは確率尺度。',
      '二項GAMMのAR補正はworking residualに対するGEE近似であり、厳密な自己相関付き二項尤度ではない。GAMMの確率曲線は参加者効果0で、GLMMの周辺ERとは異なる推定対象。',
      'GAMMのk診断は離散値・残存相関にも影響される。警告がある曲線検定は参考値に留める。収束不成立なら検定不能とし、補正計算ではp=1を割り当て検定群の大きさを保つ。',
      '各CIは特記以外多重性未補正。形状差の帯のみ400試行内で同時95%、3指標全体に対する同時帯ではない。',
      '各群10人ではCIが広く、正規性・等分散・ランダム効果分布の仮定の検証力も限定される。非有意は同等性の証明ではない。',
      '区間・主指標を定める前に結果を閲覧している。今回のHolm補正は、過去に試した区間・除外・指標の選択による多重性を取り消さない。',
      '休憩・時間経過も含む初期から後期への変化を測っており、疲労回復等を除いた純粋な学習効果とは断定しない。',
      'トップカンファレンスへの採択は保証できない。論文では選択経緯、全検定、除外、モデル警告を開示し、条件間改善量の優越を主張しない。', '',
      'モデル診断'
    ]
    for key,d in diagnostics.items():
        if key.startswith('gamm'):
            lines.append(f"{key}: rows={d['rows']}, converged={d['converged']}, iterations={d['iterations']}, rho={fmt(d['rho'])}, whitened lag1={fmt(d.get('whitened_lag1'))}")
        else:
            warns=d.get('bootstrap_messages',{}).get('factory-warning',{})
            lines.append(f"{key}: singular={d['singular']}, bootstrap有効={d['bootstrap_valid']}/1999, 失敗={d['bootstrap_fail']}, 警告件数={sum(warns.values()) if isinstance(warns,dict) else 0}, Pearson比={fmt(d['pearson_dispersion'])}")
    lines += ['', '全検定（本解析／感度解析を区別；CIの尺度はmethodとprotocol参照）',
              '解析 | 指標 | 比較 | 区分 | 推定値 | 95%CI | 生p | 補正p | 状態']
    for r in tests:
        lines.append(' | '.join([r['variant'],r['metric'],LABELS.get(r['test'],r['test']),r['family'],fmt(r.get('estimate')),
                                  f"[{fmt(r.get('ci_low'))}, {fmt(r.get('ci_high'))}]",pformat(r['p_raw']),pformat(r['p_adjusted']),r['status']]))
    lines += ['', '検証・再現',
      '既存test_analysis.py 28テスト成功。新規コードのt/F恒等式・傾き既知値・Holm補正既知値を確認。通常UIの初期・後期集計と3指標・両解析で照合。',
      'Pythonの8個のANOVA対比をRのaov（参加者内誤差項を含む）と照合。ERの40点Gauss–Hermite積分を独立した適応積分と照合し、Holm補正もRのp.adjustと一致した。',
      'R結果の出力処理と解析群フィルターの実装不具合を検出して修正し、R解析を再実行した。無効な途中出力はinvalid_variant_filter/に隔離し、最終結果には使用していない。',
      'Rフィルターには期間データ40行、試行データ7,983/7,154行、参加者20人、参加者×試行の重複なしの検査を追加。',
      '入力26ファイルのSHA-256、固定計画のSHA-256、解析開始時のanalysis.pyのスナップショット、Python/Rのバージョン、R環境ロック、ログ、除外一覧を同梱。数値計算は1コア。',
      '再実行（リポジトリルート）:',
      '.venv/bin/python output/c2_c3_validation_20261113/run_analysis.py',
      '.cache/validation-r/bin/Rscript output/c2_c3_validation_20261113/models.R output/c2_c3_validation_20261113',
      '.venv/bin/python output/c2_c3_validation_20261113/finalize.py',
      '既存bootstrapキャッシュはモデル推定値との一致を検証して再利用する。入力・計画を変えて別解析する場合は新しい出力フォルダを使用する。',
      '', '一次資料',
      'Brown–Forsythe: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html',
      'GLMM: https://lme4.github.io/lme4/reference/glmer.html',
      'Bootstrap: https://lme4.github.io/lme4/reference/bootMer.html',
      'GAMM/AR近似: https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/bam.html',
      '平滑項検定の限界: https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/summary.gam.html',
      '解析選択と透明な報告: https://doi.org/10.1080/00031305.2016.1154108',
    ]
    report='\n'.join(lines)+'\n';(HERE/'report_ja.txt').write_text(report,encoding='utf-8')
    figures=''.join(f'<figure><img src="{name}_{m}.png" alt="{m} {name}"></figure>' for name in ('participants','curve') for m in ('MT','TPe','ER'))
    (HERE/'report.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>C2・C3 統計再解析</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#222}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}img{width:100%;height:auto}figure{margin:32px 0}a{color:#176aaa}</style><h1>C2・C3 練習成績の統計再解析</h1><p><a href="all_tests.csv">全検定CSV</a> · <a href="protocol.json">固定した計画</a> · <a href="participant_summary.csv">参加者別集計</a> · <a href="excluded_rows.csv">除外一覧</a></p><pre>'+html.escape(report)+'</pre>'+figures+'</html>',encoding='utf-8')
    manifest=json.loads((HERE/'manifest.json').read_text())
    for r in manifest['inputs']:assert hashlib.sha256((ROOT/'inputCSV'/r['file']).read_bytes()).hexdigest()==r['sha256']
    assert hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()==manifest['protocol_sha256']
    assert hashlib.sha256((HERE/'analysis_snapshot.py').read_bytes()).hexdigest()==manifest['analysis_snapshot_sha256']
    json_save('verification.json',dict(input_hashes_unchanged=True,protocol_unchanged=True,families={'primary':1,'secondary':11,'exploratory':18},tests_per_variant=30,
              scripts_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('run_analysis.py','models.R','finalize.py')}))
    print('\n'.join(lines[:8]));print('Holm significant secondary:',[(r['metric'],r['test'],r['p_adjusted']) for r in significant])


if __name__=='__main__':main()
