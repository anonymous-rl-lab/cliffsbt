#!/usr/bin/env python3
"""Version-7 post hoc summaries for the Cliff NMI manuscript.

This script reuses only the frozen CURE-OR and CIFAR/TorchSig evidence already
contained in the v6 repository. It does not alter any preregistered H1--H3
model, path, threshold, identity split or decision. The v7 additions are:
  1. classifier-head-seed summaries for the strictly nested warning proxy test;
  2. complete-seed bootstrap ranges for the timely-rate differences;
  3. calibration false-alarm-budget sensitivity with fitted score models fixed;
  4. per-focal-seed trained-peer boundary effect-size summaries.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P = argparse.ArgumentParser()
P.add_argument('--repo-root', type=Path, required=True)
P.add_argument('--v6-diagnostics', type=Path, required=True)
P.add_argument('--v6-builder', type=Path, required=True)
P.add_argument('--out-dir', type=Path, required=True)
P.add_argument('--bootstrap-seed', type=int, default=20260818)
P.add_argument('--bootstrap-replicates', type=int, default=20000)
A = P.parse_args()
ROOT=A.repo_root.resolve(); V6=A.v6_diagnostics.resolve(); OUT=A.out_dir.resolve(); OUT.mkdir(parents=True,exist_ok=True)

# Import the released v6 reconstruction/fitting code without running its main().
old_argv=sys.argv[:]
sys.argv=['build_v6_diagnostics.py','--repo-root',str(ROOT),'--out-dir',str(OUT/'_v6_import_scratch')]
spec=importlib.util.spec_from_file_location('v6builder', A.v6_builder.resolve())
mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod)
sys.argv=old_argv

cal_paths, conf_paths = mod.reconstruct_paths()

VARIANTS = [
    ('static_state','Static current telemetry'),
    ('static_plus_net','Static + net prediction-state transport'),
    ('static_plus_persistence','Static + persistent departure'),
    ('current_state','Static + both proxy channels'),
    ('full34','Full temporal chart'),
]

models={}; thresholds={}; cal_scores={}; conf_scores={}; metrics={}
for kind,label in VARIANTS:
    model,thr,cal_metric,_selection=mod.fit_refit_logistic(cal_paths,kind)
    models[kind]=model; thresholds[kind]=thr
    # Score calibration and confirmation with the selected model.
    cs=mod.score_logistic(cal_paths,kind,model)
    ts=mod.score_logistic(conf_paths,kind,model)
    cal_scores[kind]=cs; conf_scores[kind]=ts
    metrics[kind]=(mod.eval_paths(cal_paths,cs,thr),mod.eval_paths(conf_paths,ts,thr))

# 1) Per-classifier-head-seed outcomes for the nested current-state analysis.
seed_rows=[]; path_rows=[]
for kind,label in VARIANTS[:4]:
    conf_eval=metrics[kind][1]
    frame=pd.DataFrame(conf_eval['rows'])
    for seed,g in frame.groupby('seed'):
        cliff=g[g.event.notna()]; ctrl=g[g.event.isna()]
        seed_rows.append({
            'kind':kind,'method':label,'seed':int(seed),
            'cliffs':int(len(cliff)),'timely':int(cliff.timely.sum()),
            'controls':int(len(ctrl)),'false':int(ctrl.false_alarm.sum()),
            'timely_rate':float(cliff.timely.mean()) if len(cliff) else np.nan,
            'false_rate':float(ctrl.false_alarm.mean()) if len(ctrl) else np.nan,
            'median_lead':float(cliff.loc[cliff.timely,'lead'].median()) if cliff.timely.any() else np.nan,
        })
    for r in conf_eval['rows']:
        path_rows.append({'kind':kind,'method':label,**r})
seed_df=pd.DataFrame(seed_rows)
seed_df.to_csv(OUT/'warning_nested_channel_by_seed.csv',index=False)
path_df=pd.DataFrame(path_rows)
path_df.to_csv(OUT/'warning_nested_channel_path_rows.csv',index=False)

# Pairwise path discordance relative to static telemetry.
static=path_df[path_df.kind=='static_state'][['id','seed','schedule_id','family','event','timely','false_alarm']].rename(columns={'timely':'static_timely','false_alarm':'static_false'})
disc=[]
for kind,label in VARIANTS[1:4]:
    x=path_df[path_df.kind==kind][['id','timely','false_alarm']].rename(columns={'timely':'variant_timely','false_alarm':'variant_false'})
    m=static.merge(x,on='id',validate='one_to_one')
    for _,r in m.iterrows():
        disc.append({
            'variant':kind,'method':label,'id':r.id,'seed':int(r.seed),
            'schedule_id':int(r.schedule_id),'family':int(r.family),'event':r.event,
            'static_timely':bool(r.static_timely),'variant_timely':bool(r.variant_timely),
            'static_false':bool(r.static_false),'variant_false':bool(r.variant_false),
            'timely_gain':int(bool(r.variant_timely))-int(bool(r.static_timely)),
            'false_gain':int(bool(r.variant_false))-int(bool(r.static_false)),
        })
pd.DataFrame(disc).to_csv(OUT/'warning_nested_channel_path_discordance.csv',index=False)

# 2) Complete-seed bootstrap of timely-rate and false-rate differences.
rng=np.random.default_rng(A.bootstrap_seed)
seeds=sorted(seed_df.seed.unique())
seed_lookup={(r.kind,int(r.seed)):r for _,r in seed_df.iterrows()}
boot_rows=[]; summary=[]
for kind,label in VARIANTS[1:4]:
    raw=[]
    for seed in seeds:
        s=seed_lookup[('static_state',seed)]; v=seed_lookup[(kind,seed)]
        raw.append({'seed':seed,
                    'timely_diff':(v.timely-s.timely)/s.cliffs,
                    'false_diff':(v.false-s.false)/s.controls})
    rawdf=pd.DataFrame(raw)
    reps=[]
    for _ in range(A.bootstrap_replicates):
        chosen=rng.choice(seeds,size=len(seeds),replace=True)
        # Pooled within resampled complete model clusters.
        st_t=va_t=st_c=va_f=st_f=va_c=0
        for seed in chosen:
            s=seed_lookup[('static_state',int(seed))]; v=seed_lookup[(kind,int(seed))]
            st_t+=s.timely; va_t+=v.timely; st_c+=s.cliffs; va_c+=v.cliffs
            st_f+=s.false; va_f+=v.false
        # controls are identical per method within seed.
        ctrl=sum(seed_lookup[('static_state',int(seed))].controls for seed in chosen)
        reps.append(((va_t/va_c)-(st_t/st_c),(va_f/ctrl)-(st_f/ctrl)))
    reps=np.asarray(reps)
    s0=seed_df[seed_df.kind=='static_state']; v0=seed_df[seed_df.kind==kind]
    observed=(v0.timely.sum()/v0.cliffs.sum())-(s0.timely.sum()/s0.cliffs.sum())
    observed_false=(v0.false.sum()/v0.controls.sum())-(s0.false.sum()/s0.controls.sum())
    summary.append({
        'variant':kind,'method':label,
        'observed_timely_rate_difference':observed,
        'timely_difference_p2_5':float(np.quantile(reps[:,0],.025)),
        'timely_difference_p97_5':float(np.quantile(reps[:,0],.975)),
        'observed_false_rate_difference':observed_false,
        'false_difference_p2_5':float(np.quantile(reps[:,1],.025)),
        'false_difference_p97_5':float(np.quantile(reps[:,1],.975)),
        'positive_seed_directions':int((rawdf.timely_diff>0).sum()),
        'zero_seed_directions':int((rawdf.timely_diff==0).sum()),
        'negative_seed_directions':int((rawdf.timely_diff<0).sum()),
    })
pd.DataFrame(summary).to_csv(OUT/'warning_nested_channel_seed_cluster_ranges.csv',index=False)

# Pooled summary at 7.5% operating point.
pooled=[]
for kind,label in VARIANTS:
    ce,te=metrics[kind]
    pooled.append({'kind':kind,'method':label,'threshold':thresholds[kind],
                   'cal_timely':ce['timely'],'cal_cliffs':ce['cliffs'],'cal_false':ce['false'],'cal_controls':ce['controls'],'cal_median_lead':ce['median_lead'],
                   'conf_timely':te['timely'],'conf_cliffs':te['cliffs'],'conf_false':te['false'],'conf_controls':te['controls'],'conf_median_lead':te['median_lead']})
pd.DataFrame(pooled).to_csv(OUT/'warning_nested_channel_summary.csv',index=False)

# 3) Threshold recalibration across false-alarm budgets while the fitted score models remain fixed.
def choose_threshold_budget(paths,scores,budget):
    vals=np.concatenate([np.asarray(v,float)[1:] for v in scores.values()])
    grid=np.unique(np.quantile(vals,np.linspace(0,1,2001)))
    grid=np.concatenate([grid,[np.nextafter(np.max(vals),np.inf)]])
    best=None
    for thr in grid:
        met=mod.eval_paths(paths,scores,float(thr))
        if met['false_rate'] <= budget + 1e-12:
            key=(met['timely_rate'],met['median_lead'] if not np.isnan(met['median_lead']) else -1,-met['false_rate'],-float(thr))
            if best is None or key>best[0]: best=(key,float(thr),met)
    if best is None: raise RuntimeError('No threshold')
    return best[1],best[2]

budgets=[.05,.075,.10,.15,.20]
budget_rows=[]
for kind,label in VARIANTS:
    for b in budgets:
        thr,cm=choose_threshold_budget(cal_paths,cal_scores[kind],b)
        tm=mod.eval_paths(conf_paths,conf_scores[kind],thr)
        budget_rows.append({'kind':kind,'method':label,'false_budget':b,'threshold':thr,
                            'cal_timely':cm['timely'],'cal_false':cm['false'],'cal_cliffs':cm['cliffs'],'cal_controls':cm['controls'],'cal_median_lead':cm['median_lead'],
                            'conf_timely':tm['timely'],'conf_false':tm['false'],'conf_cliffs':tm['cliffs'],'conf_controls':tm['controls'],'conf_median_lead':tm['median_lead']})
pd.DataFrame(budget_rows).to_csv(OUT/'warning_false_budget_sensitivity.csv',index=False)

# 4) Per-focal-seed trained-peer summaries from the released v6 normalized ledger.
peer=pd.read_csv(V6/'trained_peer_boundary_normalized_rows.csv')
peer_rows=[]
for (domain,seed),g in peer.groupby(['domain','focal_seed']):
    for subset,mask in [('all',np.ones(len(g),bool)),('active',g.active.astype(bool).to_numpy()),('cliff',g.cliff.astype(bool).to_numpy())]:
        x=g.loc[mask]
        if x.empty: continue
        peer_rows.append({'domain':domain,'focal_seed':int(seed),'subset':subset,'n':len(x),
                          'peer_rmse_mean':float(x.peer_rmse.mean()),'peer_rmse_median':float(x.peer_rmse.median()),
                          'peer_nrmse_rms_median':float(x.peer_nrmse_rms.median()),
                          'anchor_risk_gap_median':float(np.median(np.abs(x.focal_baseline_risk-x.peer_baseline_risk)))})
pd.DataFrame(peer_rows).to_csv(OUT/'trained_peer_boundary_by_seed.csv',index=False)

# Human-readable report.
seed_piv=seed_df.pivot(index='seed',columns='kind',values='timely').reset_index()
report=['# Cliff NMI v7 post hoc diagnostic addendum','',
        'All analyses reuse frozen outputs. They were specified after confirmation reveal and do not modify the preregistered CURE-OR H1--H3 decision.','',
        '## Nested prediction-state transport proxy by classifier-head seed','',seed_piv.to_markdown(index=False),'',
        '## Complete-seed descriptive ranges','',pd.DataFrame(summary).to_markdown(index=False,floatfmt='.4f'),'',
        '## False-alarm-budget sensitivity','',pd.DataFrame(budget_rows).to_markdown(index=False,floatfmt='.4f'),'',
        '## Trained-peer mismatch by focal seed','',pd.DataFrame(peer_rows).to_markdown(index=False,floatfmt='.4f'),'']
(OUT/'V7_DIAGNOSTIC_ADDENDUM.md').write_text('\n'.join(report),encoding='utf-8')

meta={'status':'post_hoc_committed_output_diagnostics','repo_root':str(ROOT),'v6_diagnostics':str(V6),'bootstrap_seed':A.bootstrap_seed,'bootstrap_replicates':A.bootstrap_replicates,
      'files':sorted([p.name for p in OUT.iterdir() if p.is_file()])}
(OUT/'v7_diagnostic_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(pd.DataFrame(pooled).to_string(index=False))
print(pd.DataFrame(summary).to_string(index=False))
