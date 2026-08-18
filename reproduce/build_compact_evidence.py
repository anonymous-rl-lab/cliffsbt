#!/usr/bin/env python3
"""Rebuild the compact GitHub evidence tables from the full evidence repository.

The committed compact tables are already included. This script is for provenance
checking when the 145 MB full archive is available.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd

P=argparse.ArgumentParser()
P.add_argument('--full-repo-root',type=Path,required=True)
P.add_argument('--v6-diagnostics',type=Path,required=True)
P.add_argument('--v7-diagnostics',type=Path,required=True)
P.add_argument('--out-dir',type=Path,required=True)
A=P.parse_args(); root=A.full_repo_root.resolve(); out=A.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)

def cp(rel,name): shutil.copy2(root/rel,out/name)
cp('torchsig/results/round11c_official_source_flux/paired_effects.csv','torchsig_round11c_paired_effects.csv')
cp('torchsig/results/formal_precursor_mechanism_knockout_v3/matched_pair_metrics.csv','torchsig_temporal_order_pairs.csv')
cp('torchsig/results/formal_round10_training_intervention_v1/trajectory_summary.csv','torchsig_round10_trajectory_summary.csv')
cp('torchsig/results/formal_round10_training_intervention_v1/geometry_summary.csv','torchsig_round10_geometry_summary.csv')
cp('torchsig/results/round12c_coverage_vs_hazard_pilot/selection_summary.csv','torchsig_round12c_selection_summary.csv')
cp('torchsig/results/round12c_coverage_vs_hazard_pilot/seed_paired_effects.csv','torchsig_round12c_seed_effects.csv')
cp('cure_or/raw_outputs/path_level_results.csv','cure_or_path_level_results.csv')
cp('cure_or/raw_outputs/seed_level_results.csv','cure_or_seed_level_results.csv')
arr=np.load(root/'round13_second_domain/results/cifar10c_official_v1/paired_outputs.npz',allow_pickle=True)
rows=[]
for ci,c in enumerate(arr['corruptions']):
  for si,seed in enumerate(arr['seeds']):
    risk=(arr['predictions'][si,ci]!=arr['labels'][None,:]).mean(axis=1)
    rows.append({'corruption':str(c),'seed':int(seed),'clean_risk':risk[0],'endpoint_risk':risk[-1],
                 'endpoint_risk_increase':risk[-1]-risk[0],'crossed_headroom_0_15':bool(np.any(risk>=risk[0]+.15))})
pd.DataFrame(rows).to_csv(out/'cifar10c_endpoint_by_seed.csv',index=False)

# Round 13E compact selection and seed-level result tables.
r13root=root/'round13_second_domain/results/round13e_formal_v1'
r13=json.loads((r13root/'summary.json').read_text())
sel_full=json.loads((r13root/'selections.json').read_text())
sel_rows=[]
for arm,stats in r13['selection_stats'].items():
  vals=[x.get('hazard_score') for x in sel_full['selections'][arm] if x.get('hazard_score') is not None]
  sel_rows.append({'arm':arm,'budget':stats['budget'],'hazard_hit_rate':stats['hazard_hit_rate'],
                   'unique_fragments':stats['unique_fragments'],'families':stats['families'],
                   'classes':stats['classes'],'severities':stats['severities'],
                   'mean_boundary_pressure':float(np.mean(vals)) if vals else np.nan})
pd.DataFrame(sel_rows).to_csv(out/'cifar_repair_selection_summary.csv',index=False)
result_rows=[]
for seed,arms in r13['results'].items():
  for arm,obj in arms.items():
    m=obj['metrics']; result_rows.append({'seed':int(seed),'arm':arm,'clean_error':m['clean_error'],
      'endpoint_error_mean':m['endpoint_error_mean'],'risk_area_mean':m['risk_area_mean'],
      'crossing_fraction':m['crossing_fraction'],'max_accounting_error':m['max_accounting_error']})
pd.DataFrame(result_rows).to_csv(out/'cifar_repair_seed_results.csv',index=False)

# Exact Round 10 anchor-versus-future decomposition.
traj=pd.read_csv(root/'torchsig/results/formal_round10_training_intervention_v1/trajectory_summary.csv')
means=traj.groupby('regime')[['start_risk','end_risk']].mean()
dec=[]
for reg in ['random_broad','cliff_aware']:
  terminal=float(means.loc['baseline','end_risk']-means.loc[reg,'end_risk'])
  anchor=float(means.loc['baseline','start_risk']-means.loc[reg,'start_risk'])
  dec.append({'regime':reg,'terminal_gain':terminal,'anchor_gain':anchor,'future_sbt_suppression':terminal-anchor})
pd.DataFrame(dec).to_csv(out/'training_gain_decomposition.csv',index=False)

# Threshold sensitivity from frozen trajectories/alarms.
ledger=pd.read_csv(root/'torchsig/results/formal_round10_training_intervention_v1/shared_deployment_risk_ledger.csv')
trows=[]
for beta in [.10,.125,.15,.175,.20,.225,.25,.275,.30,.325,.35]:
  row={'boundary':beta}
  for reg,g in ledger.groupby('regime'):
    total=0
    for _,q in g.groupby(['replicate_seed','path']):
      risk=q.sort_values('time_index').risk.to_numpy()
      total += int(any(risk[i]>=beta and risk[i+1]>=beta for i in range(len(risk)-1)))
    row[reg]=total
  trows.append(row)
pd.DataFrame(trows)[['boundary','support_depleted','baseline','random_broad','cliff_aware']].to_csv(out/'torchsig_threshold_sensitivity.csv',index=False)

cure=pd.read_csv(root/'cure_or/raw_outputs/path_level_results.csv')
crows=[]
for beta in [.35,.375,.4,.425,.45,.475,.5,.525,.55,.575,.6,.625,.65,.675,.7]:
  cliffs=controls=timely=false=0
  for _,r in cure.iterrows():
    risk=np.asarray(json.loads(r.risk),float)
    if risk[0]>=beta: continue
    event=None
    for t in range(len(risk)):
      if risk[t]>=beta and np.all(risk[t:]>=beta): event=t; break
    alarm=None if pd.isna(r.hybrid25_alarm) else int(r.hybrid25_alarm)
    if event is None:
      controls+=1; false+=int(alarm is not None)
    else:
      cliffs+=1; timely+=int(alarm is not None and alarm<event)
  timely_exact=timely/cliffs if cliffs else np.nan; false_exact=false/controls if controls else np.nan
  crows.append({'boundary':beta,'cliffs':cliffs,'controls':controls,'timely_count':timely,'false_count':false,
                'timely':round(timely_exact,3) if cliffs else np.nan,'false':round(false_exact,3) if controls else np.nan,
                'timely_exact':timely_exact,'false_exact':false_exact})
pd.DataFrame(crows).to_csv(out/'cure_or_threshold_sensitivity.csv',index=False)

for base,label in [(A.v6_diagnostics,'v6_diagnostics'),(A.v7_diagnostics,'v7_diagnostics')]:
  dest=out/label; dest.mkdir(exist_ok=True)
  for p in base.iterdir():
    if p.is_file() and p.suffix.lower() in {'.csv','.json','.md'}: shutil.copy2(p,dest/p.name)
print(out)
