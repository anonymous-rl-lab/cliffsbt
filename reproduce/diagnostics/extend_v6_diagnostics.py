#!/usr/bin/env python3
"""Derived v6 summaries from committed outputs.

Adds (i) a strictly nested current-state channel ablation and (ii) domain-specific
normalization of trained-peer boundary reconstruction error. No model, path,
identity split, registered alarm, or preregistered decision is changed.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

P=argparse.ArgumentParser()
P.add_argument('--repo-root',type=Path,required=True)
P.add_argument('--v6-output',type=Path,required=True)
a=P.parse_args()
root=a.repo_root.resolve(); out=a.v6_output.resolve(); out.mkdir(parents=True,exist_ok=True)

# 1) Strictly nested current-state channel ablation.
base=pd.read_csv(out/'warning_fair_baselines.csv')
labels=[
 'Static current telemetry (refit)',
 'Static current + net departure-recovery (refit)',
 'Static current + persistent departure (refit)',
 'Current active-state telemetry (refit)',
]
nested=base[base.method.isin(labels)].copy()
order={x:i for i,x in enumerate(labels)}
nested['order']=nested.method.map(order)
nested=nested.sort_values('order').drop(columns='order')
nested['delta_timely_vs_static']=nested.conf_timely-int(nested.iloc[0].conf_timely)
nested['delta_false_vs_static']=nested.conf_false-int(nested.iloc[0].conf_false)
nested.to_csv(out/'warning_nested_channel_ablation.csv',index=False)

# 2) Domain-specific normalized trained-peer boundary diagnostics.
peer=pd.read_csv(out/'trained_peer_boundary_placebo_rows.csv')
rows=[]

# CURE-OR focal risk increments from frozen path table.
cure=root/'cure_or'
frozen=pd.read_csv(cure/'raw_outputs/path_level_results.csv')
for _,r in peer[peer.domain=='CURE-OR'].iterrows():
    m=(frozen.seed==int(r.focal_seed))&(frozen.schedule_id==int(str(r.schedule)))&(frozen.family==int(str(r.family)))
    fr=frozen[m].iloc[0]
    risk=np.asarray(json.loads(fr.risk),float); delta=np.diff(risk)
    rms=float(np.sqrt(np.mean(delta**2))); mae=float(np.mean(np.abs(delta)))
    rows.append({**r.to_dict(),'incorrect_rmse':np.nan,'focal_delta_rms':rms,'focal_delta_mae':mae,
                 'peer_nrmse_rms':float(r.peer_rmse/rms) if rms>0 else np.nan,
                 'peer_rmse_over_mae':float(r.peer_rmse/mae) if mae>0 else np.nan})

# CIFAR-10-C: add fixed incorrect-partition RMSE and focal step magnitudes.
arr=np.load(root/'round13_second_domain/results/cifar10c_official_v1/paired_outputs.npz',allow_pickle=True)
pred=arr['predictions']; labels_arr=arr['labels']; pseudo=arr['pseudo_labels']; seeds=arr['seeds']; corruptions=arr['corruptions']
peer_c=peer[peer.domain=='CIFAR-10-C'].reset_index(drop=True)
k=0
for si,seed in enumerate(seeds):
    for ci,corr in enumerate(corruptions):
        r=peer_c.iloc[k]; k+=1
        err=pred[si,ci]!=labels_arr[None,:]
        risk=err.mean(axis=1); delta=np.diff(risk)
        pseudo_wrong=pred[si,ci]!=pseudo[None,:]
        pseudo_net=((~pseudo_wrong[:-1])&pseudo_wrong[1:]).mean(axis=1)-(pseudo_wrong[:-1]&(~pseudo_wrong[1:])).mean(axis=1)
        incorrect=float(np.sqrt(np.mean((delta-pseudo_net)**2)))
        rms=float(np.sqrt(np.mean(delta**2))); mae=float(np.mean(np.abs(delta)))
        rows.append({**r.to_dict(),'incorrect_rmse':incorrect,'focal_delta_rms':rms,'focal_delta_mae':mae,
                     'peer_nrmse_rms':float(r.peer_rmse/rms) if rms>0 else np.nan,
                     'peer_rmse_over_mae':float(r.peer_rmse/mae) if mae>0 else np.nan})

nr=pd.DataFrame(rows)
nr.to_csv(out/'trained_peer_boundary_normalized_rows.csv',index=False)
summary=[]
for domain,g0 in nr.groupby('domain'):
    for subset,mask in [('all',np.ones(len(g0),bool)),('active',g0.active.to_numpy(bool)),('cliff',g0.cliff.to_numpy(bool))]:
        g=g0.loc[mask]
        if not len(g): continue
        summary.append({
          'domain':domain,'subset':subset,'n':len(g),
          'self_rmse_mean':float(g.self_rmse.mean()),
          'peer_rmse_mean':float(g.peer_rmse.mean()),'peer_rmse_median':float(g.peer_rmse.median()),
          'incorrect_rmse_mean':float(g.incorrect_rmse.mean()) if g.incorrect_rmse.notna().any() else np.nan,
          'incorrect_rmse_median':float(g.incorrect_rmse.median()) if g.incorrect_rmse.notna().any() else np.nan,
          'focal_delta_rms_mean':float(g.focal_delta_rms.mean()),'focal_delta_rms_median':float(g.focal_delta_rms.median()),
          'peer_nrmse_rms_median':float(g.peer_nrmse_rms.median()),
          'peer_rmse_over_mae_median':float(g.peer_rmse_over_mae.median()),
          'baseline_risk_gap_median':float(np.median(np.abs(g.focal_baseline_risk-g.peer_baseline_risk))),
        })
su=pd.DataFrame(summary)
su.to_csv(out/'trained_peer_boundary_normalized_summary.csv',index=False)

# CIFAR-specific three-level ordering, using cell means on the same 45-cell field.
cif=su[(su.domain=='CIFAR-10-C') & (su.subset.isin(['all','cliff']))].copy()
three=[]
for _,r in cif.iterrows():
    three.append({'domain':'CIFAR-10-C','subset':r['subset'],'incorrect_partition_rmse':r['incorrect_rmse_mean'],
                  'trained_peer_rmse':r['peer_rmse_mean'],'focal_self_rmse':r['self_rmse_mean'],
                  'peer_nrmse_rms_median':r['peer_nrmse_rms_median'],
                  'peer_rmse_over_mae_median':r['peer_rmse_over_mae_median']})
pd.DataFrame(three).to_csv(out/'cifar_boundary_specificity_gradient.csv',index=False)

# Markdown addendum.
def md(df,fmt='.4f'):
    return df.to_markdown(index=False,floatfmt=fmt).replace(' nan ',' — ').replace(' nan|',' —|')
report=['# Cliff NMI v6 diagnostic addendum','',
'All analyses below reuse committed outputs. The nested warning comparison refits each feature set only on calibration identities under the same false-alarm budget. The trained-peer normalization is arithmetic on frozen risk and prediction arrays.','',
'## Strictly nested current-state channel ablation','',md(nested[['method','cal_timely','cal_false','conf_timely','conf_false','conf_median_lead','delta_timely_vs_static','delta_false_vs_static']],'.0f'),'',
'## Domain-specific trained-peer normalization','',md(su,'.6f'),'',
'## CIFAR boundary-specificity gradient','',md(pd.DataFrame(three),'.6f'),'']
(out/'V6_DIAGNOSTIC_ADDENDUM.md').write_text('\n'.join(report),encoding='utf-8')
print(nested[['method','conf_timely','conf_false','delta_timely_vs_static']].to_string(index=False))
print(su.to_string(index=False))
