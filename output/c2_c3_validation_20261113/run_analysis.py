import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
from scipy import stats, special

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from analysis_snapshot import load_rows, mt_outliers, analyze


def save_csv(name, records):
    keys = list(dict.fromkeys(k for r in records for k in r))
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def json_save(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def contrast(x, y, method):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if method == 'paired':
        test = stats.ttest_rel(x, y)
        df = len(x) - 1
        sd = np.std(x - y, ddof=1)
    else:
        test = stats.ttest_ind(x, y, equal_var=method == 'pooled')
        df = len(x) + len(y) - 2
        sd = np.sqrt(((len(x)-1)*np.var(x, ddof=1)+(len(y)-1)*np.var(y, ddof=1))/df)
    ci = test.confidence_interval(.95)
    correction = math.exp(special.gammaln(df/2) - .5*math.log(df/2) - special.gammaln((df-1)/2))
    estimate = float(np.mean(x) - np.mean(y))
    return dict(estimate=estimate, ci_low=float(ci.low), ci_high=float(ci.high),
                statistic=float(test.statistic), F=float(test.statistic**2) if method != 'welch' else '',
                df1=1 if method != 'welch' else '', df2=float(test.df), p_raw=float(test.pvalue),
                standardized=correction*estimate/sd if sd else '', effect_type='corrected dz' if method=='paired' else 'Hedges g',
                n1=len(x), n2=len(y), method={'paired':'RMANOVA / paired t', 'pooled':'mixed ANOVA contrast / pooled t', 'welch':'Welch t'}[method])


def line_values(rows, metric):
    valid = [(r['n'], r['coreMtMs'][metric]) for r in rows if r['coreMtMs'][metric] is not None]
    x, y = np.asarray(valid, float).T
    fit = stats.linregress(x, y)
    residuals = y - fit.intercept - fit.slope*x
    sd = float(np.std(residuals, ddof=1))
    return float(fit.slope), sd


def continuous_tests(records, variant):
    tests = []
    for metric in ('MT', 'TPe'):
        groups = {c: sorted([r for r in records if r['condition']==c and r['metric']==metric], key=lambda r:r['participant']) for c in ('C2','C3')}
        values = {c: {k: np.asarray([r[k] for r in g]) for k in ('early','late','change','early_slope','late_slope','late_sd')} for c,g in groups.items()}
        a,b = values['C2'],values['C3']
        def add(label, family, result):
            tests.append(dict(variant=variant, metric=metric, test=label, family=family, **result))
        add('change_C2-C3', 'primary' if metric=='MT' else 'secondary', contrast(a['change'],b['change'],'pooled'))
        add('late_C2-C3','secondary',contrast(a['late'],b['late'],'pooled'))
        for c,v in values.items():
            add('change_'+c,'secondary',contrast(v['late'],v['early'],'paired'))
        for key in ('early','early_slope','late_slope'):
            add(key+'_C2-C3','exploratory',contrast(a[key],b[key],'welch'))
        if min(np.min(a['late_sd']),np.min(b['late_sd'])) > 0:
            result = contrast(np.log(a['late_sd']),np.log(b['late_sd']),'welch')
            result.update(sd_ratio=math.exp(result['estimate']), ratio_ci_low=math.exp(result['ci_low']),ratio_ci_high=math.exp(result['ci_high']))
        else:
            result = dict(p_raw='', method='unavailable: zero residual SD; no offset added')
        add('log_late_sd_C2-C3','exploratory',result)
        for key in ('late','change'):
            result = stats.levene(a[key],b[key],center='median')
            add(key+'_variance','exploratory',dict(method='Brown-Forsythe (median Levene)',statistic=float(result.statistic),F=float(result.statistic),df1=1,df2=18,p_raw=float(result.pvalue),variance_ratio=float(np.var(a[key],ddof=1)/np.var(b[key],ddof=1))))
    return tests


def robustness(records, variant):
    groups = {c: [r for r in records if r['condition']==c and r['metric']=='MT'] for c in ('C2','C3')}
    a,b = [np.asarray([r['change'] for r in groups[c]]) for c in ('C2','C3')]
    rng = np.random.default_rng(20261113)
    boot = a[rng.integers(len(a), size=(9999,len(a)))].mean(axis=1)-b[rng.integers(len(b),size=(9999,len(b)))].mean(axis=1)
    perm = stats.permutation_test((a,b),lambda x,y,axis:np.mean(x,axis=axis)-np.mean(y,axis=axis),
                                  vectorized=True,n_resamples=np.inf,batch=2000,alternative='two-sided')
    influence = []
    for c in ('C2','C3'):
        for i,r in enumerate(groups[c]):
            x,y=(np.delete(a,i),b) if c=='C2' else (a,np.delete(b,i))
            influence.append(dict(variant=variant,omitted=r['participant'],condition=c,**contrast(x,y,'pooled')))
    return dict(variant=variant,welch=contrast(a,b,'welch'),permutation_p=float(perm.pvalue),permutations=len(perm.null_distribution),bootstrap_percentile_ci=np.quantile(boot,[.025,.975]).tolist()),influence


def self_test():
    x=np.array([1.,2.,4.,8.]); y=np.array([2.,5.,7.,9.])
    test=contrast(x,y,'pooled'); f=stats.f_oneway(x,y)
    np.testing.assert_allclose([test['F'],test['p_raw']],[f.statistic,f.pvalue])
    paired=contrast(y,x,'paired')
    np.testing.assert_allclose(paired['F'],len(x)*np.mean(y-x)**2/np.var(y-x,ddof=1))
    rows=[dict(n=i,coreMtMs={'MT':2*i+3}) for i in range(1,10)]
    np.testing.assert_allclose(line_values(rows,'MT'),[2.,0.],atol=1e-12)
    print('Numeric identity tests passed',flush=True)


def prepare():
    self_test()
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,numeric_cores=1)
    manifest['inputs']=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted((ROOT/'inputCSV').glob('*.csv'))]
    manifest['protocol_sha256']=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
    manifest['analysis_snapshot_sha256']=hashlib.sha256((HERE/'analysis_snapshot.py').read_bytes()).hexdigest()
    json_save('manifest.json',manifest)
    rows=load_rows(ROOT/'inputCSV')
    selected=[r for r in rows if r['phase']=='practice' and r['condition'] in ('C2','C3') and 1<=r['n']<=400]
    base=[r for r in selected if r['countedAsTrial'] and r['coreExitMs'] is not None]
    excluded=[]
    for r in selected:
        reasons=[]
        if not r['countedAsTrial']: reasons.append('countedAsTrial=false')
        if r['coreExitMs'] is None: reasons.append('coreExitMs missing')
        if reasons: excluded.append({k:r[k] for k in ('source','line','participant','condition','n')}|dict(stage='base',reason='; '.join(reasons)))
    outliers=mt_outliers(base,'coreMtMs','trial1_5sd')
    for r in base:
        if id(r) in outliers: excluded.append({k:r[k] for k in ('source','line','participant','condition','n')}|dict(stage='sensitivity_only',reason='within-condition same-N MT +/-1.5 sample SD'))
    summaries=[]; periods=[]; trials=[]; tests=[]; audits=[]; robust=[]; influence=[]; descriptions=[]
    for variant,active in [('base',base),('sensitivity',[r for r in base if id(r) not in outliers])]:
        groups=defaultdict(list)
        for r in active:
            groups[(r['condition'],r['participant'])].append(r)
            trials.append(dict(variant=variant,participant=r['participant'],condition=r['condition'],N=r['n'],block=(r['n']-1)//100+1,**r['coreMtMs']))
        local=[]
        for (condition,pid),group in sorted(groups.items()):
            assert len({r['n'] for r in group})==len(group), 'Repeated N requires explicit model handling'
            windows={name:[r for r in group if lo<=r['n']<=hi] for name,lo,hi in [('early',1,100),('late',301,400)]}
            assert all(windows.values()), 'Missing period requires explicit review'
            for name,sub in windows.items():
                invalid={m:sum(r['coreMtMs'][m] is None or not math.isfinite(r['coreMtMs'][m]) for r in sub) for m in ('MT','TPe','ER')}
                assert not any(invalid.values()), 'Missing metric requires explicit review'
                audits.append(dict(variant=variant,condition=condition,participant=pid,window=name,rows=len(sub),**{m+'_invalid':v for m,v in invalid.items()}))
                periods.append(dict(variant=variant,condition=condition,participant=pid,late=int(name=='late'),total=len(sub),errors=sum(r['coreMtMs']['ER'] for r in sub)))
            for metric in ('MT','TPe','ER'):
                unit=100 if metric=='ER' else 1
                vals={name:float(np.mean([r['coreMtMs'][metric] for r in sub]))*unit for name,sub in windows.items()}
                record=dict(variant=variant,participant=pid,condition=condition,metric=metric,**vals,change=vals['late']-vals['early'])
                if metric!='ER':
                    for name,sub in windows.items():
                        slope,sd=line_values(sub,metric);record[name+'_slope']=slope;record[name+'_sd']=sd
                local.append(record)
        summaries.extend(local)
        tests.extend(continuous_tests(local,variant))
        result,loo=robustness(local,variant);robust.append(result);influence.extend(loo)
        for metric in ('MT','TPe','ER'):
            for condition in ('C2','C3'):
                members=[r for r in local if r['metric']==metric and r['condition']==condition]
                for key in ('early','late','change'):
                    values=np.array([r[key] for r in members]);sd=float(np.std(values,ddof=1));ci=stats.t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values))
                    descriptions.append(dict(variant=variant,metric=metric,condition=condition,quantity=key,n=len(values),mean=float(values.mean()),sd=sd,ci_low=float(values.mean()-ci),ci_high=float(values.mean()+ci)))
        options=dict(scope='coreMtMs',phase='practice',mode='mean',bin=100,participants=sorted({r['participant'] for r in active}),conditions=['C2','C3'],trialStart=1,trialEnd=400,exclude=True,excludeBoundaryTrials=False,mtOutlier='none' if variant=='base' else 'trial1_5sd')
        check=analyze(rows,options,include_fits=False)
        assert check['included']==len(active)
        for series in check['series']:
            for metric in ('MT','TPe','ER'):
                for i,key in [(0,'early'),(3,'late')]:
                    expected=np.mean([r[key] for r in local if r['metric']==metric and r['condition']==series['condition']])
                    np.testing.assert_allclose(series[metric][i],expected)
        print(variant,'rows',len(active),'participants',len(groups),'UI aggregate crosscheck passed',flush=True)
    for name,records in [('participant_summary.csv',summaries),('period_counts.csv',periods),('trial_metrics.csv',trials),('continuous_tests.csv',tests),('audit.csv',audits),('excluded_rows.csv',excluded),('leave_one_out.csv',influence),('descriptives.csv',descriptions)]:
        save_csv(name,records)
    json_save('robustness.json',robust)
    json_save('data_audit.json',dict(total_files=26,total_rows=len(rows),candidate_rows=len(selected),included_rows=len(base),excluded_rows=len(selected)-len(base),sensitivity_removed=len(outliers),tests_existing=28))


if __name__=='__main__':
    prepare()
