#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

P=argparse.ArgumentParser(); P.add_argument('--evidence-dir',type=Path,default=Path(__file__).resolve().parents[1]/'evidence'/'compact'); A=P.parse_args(); E=A.evidence_dir
checks=[]
def ck(name,cond,detail=''):
    checks.append((name,bool(cond),detail));
    if not cond: print('FAIL',name,detail)

r11=pd.read_csv(E/'torchsig_round11c_paired_effects.csv')
ck('round11c_rows',len(r11)==10,str(len(r11)))
ck('round11c_closure',float(r11.maximum_flux_accounting_error_baseline.max())<1e-12)
ck('round11c_all_baseline_cross',int(r11.relative_cliff_crossed_baseline.sum())==10)
ck('round11c_reduction_all_positive',bool((r11.end_risk_reduction>0).all()) and bool((r11.incident_crossing_reduction>0).all()))
ck('round11c_persistence',abs(float(r11.incident_persistence_baseline.median())-0.9525)<5e-4,str(r11.incident_persistence_baseline.median()))
ck('round11c_entropy',abs(float(r11.first_crossing_entropy_baseline.median())-0.8789)<5e-4,str(r11.first_crossing_entropy_baseline.median()))

cif=pd.read_csv(E/'cifar10c_endpoint_by_seed.csv')
ck('cifar_cells',len(cif)==45,str(len(cif)))
ck('cifar_endpoint_mean',abs(float(cif.endpoint_risk_increase.mean())-0.26858)<2e-5,str(cif.endpoint_risk_increase.mean()))
ck('cifar_all_seed_cross_families',int(cif.groupby('corruption').crossed_headroom_0_15.sum().eq(3).sum())==11)

paths=pd.read_csv(E/'cure_or_path_level_results.csv'); seed=pd.read_csv(E/'cure_or_seed_level_results.csv')
ck('cure_paths',len(paths)==150,str(len(paths)))
ck('cure_cliffs',int(paths.event.notna().sum())==72,str(paths.event.notna().sum()))
ck('cure_controls',int(paths.event.isna().sum())==78,str(paths.event.isna().sum()))
ck('cure_timely',int(paths.timely.sum())==71,str(paths.timely.sum()))
ck('cure_false',int(paths.false_alarm.sum())==3,str(paths.false_alarm.sum()))
ck('cure_closure',float(paths.max_closure_error.max())<1e-12,str(paths.max_closure_error.max()))
ck('cure_repair_eligible',int(seed.repair_eligible.sum())==4,str(seed.repair_eligible.sum()))
ck('cure_mean_risk_gain',abs(float(seed.repair_mean_risk_gain.mean())-0.06203076)<1e-7,str(seed.repair_mean_risk_gain.mean()))

nest=pd.read_csv(E/'v7_diagnostics'/'warning_nested_channel_summary.csv').set_index('kind')
expected={'static_state':(62,3),'static_plus_net':(71,3),'static_plus_persistence':(62,3),'current_state':(70,3)}
for k,(t,f) in expected.items():
    ck('nested_'+k,int(nest.loc[k,'conf_timely'])==t and int(nest.loc[k,'conf_false'])==f,str(nest.loc[k,['conf_timely','conf_false']].to_dict()))

dec=pd.read_csv(E/'training_gain_decomposition.csv').set_index('regime')
for k in dec.index:
    ck('decomposition_'+k,abs(dec.loc[k,'terminal_gain']-(dec.loc[k,'anchor_gain']+dec.loc[k,'future_sbt_suppression']))<1e-12)

r13=pd.read_csv(E/'cifar_repair_seed_results.csv').groupby('arm')[['endpoint_error_mean','risk_area_mean']].mean()
ck('repair_reversal_endpoint',float(r13.loc['coverage','endpoint_error_mean'])>float(r13.loc['hazard','endpoint_error_mean']))
ck('repair_reversal_area',float(r13.loc['coverage','risk_area_mean'])>float(r13.loc['hazard','risk_area_mean']))

failed=[x for x in checks if not x[1]]
print(json.dumps({'checks':len(checks),'passed':len(checks)-len(failed),'failed':[{'name':n,'detail':d} for n,_,d in failed]},indent=2))
raise SystemExit(bool(failed))
