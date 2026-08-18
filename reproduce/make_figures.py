from __future__ import annotations
from pathlib import Path
import argparse, json, math, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
warnings.filterwarnings('ignore', message='constrained_layout not applied.*')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
from matplotlib.lines import Line2D

P=argparse.ArgumentParser(description='Rebuild Cliff NMI v7 figures from frozen evidence outputs.')
P.add_argument('--evidence-dir',type=Path,required=True,help='Repository evidence/compact directory')
P.add_argument('--out-dir',type=Path,required=True,help='Output figures directory')
A=P.parse_args()
E=A.evidence_dir.resolve()
DIAG=E/'v6_diagnostics'
V7DIAG=E/'v7_diagnostics'
OUT=A.out_dir.resolve()
MAIN_OUT=OUT/'main'
ED_OUT=OUT/'extended_data'
MAIN_OUT.mkdir(parents=True,exist_ok=True)
ED_OUT.mkdir(parents=True,exist_ok=True)

# restrained, colorblind-friendly palette
BLUE='#2F6B9A'; ORANGE='#D97A2B'; GREEN='#3E8E55'; RED='#B64B4B'; PURPLE='#7B61A8'; TEAL='#3C8D8D'; GOLD='#C89A3D'; GRAY='#6F7780'; LIGHT='#DCE3E8'; DARK='#222222'

mpl.rcParams.update({
 'font.family':'DejaVu Sans','font.size':8.2,'axes.titlesize':9.2,'axes.labelsize':8.4,
 'xtick.labelsize':7.4,'ytick.labelsize':7.4,'legend.fontsize':7.0,
 'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,
 'savefig.dpi':400,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':0.8,
})

def panel(ax,label):
    ax.text(-0.12,1.08,label,transform=ax.transAxes,fontsize=10,fontweight='bold',va='top')

def save(fig,name):
    target=ED_OUT if name.startswith('Extended_Data_') else MAIN_OUT
    fig.savefig(target/f'{name}.png',bbox_inches='tight',facecolor='white')
    fig.savefig(target/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)

# ---------------- Figure 1: conceptual framework ----------------
fig=plt.figure(figsize=(7.2,4.7),constrained_layout=True)
gs=fig.add_gridspec(2,3,height_ratios=[1.15,0.85],wspace=0.48,hspace=0.62)
ax=fig.add_subplot(gs[0,0]); panel(ax,'a')
t=np.linspace(0,10,200); r=0.12+0.013*t+0.0011*t**2; beta=0.25
ax.plot(t,r,color=BLUE,lw=2.2)
ax.axhline(beta,color=RED,ls='--',lw=1.3)
tc=t[np.argmax(r>=beta)]; ax.axvline(tc,color=GRAY,ls=':',lw=1)
ax.fill_between(t,beta,r,where=r>=beta,color=RED,alpha=.14)
ax.text(tc+.18,.118,'first crossing',rotation=90,va='bottom',fontsize=7.0,color=GRAY)
ax.text(.25,beta+.008,'risk boundary',color=RED,fontsize=7.2)
ax.set(xlabel='deployment time',ylabel='risk'); ax.set_title('Operationally abrupt,\nmechanistically cumulative',fontsize=8.1,pad=7)
ax.set_xlim(0,10); ax.set_ylim(.1,.36); ax.set_xticks([0,2,4,6,8,10]); ax.set_yticks([.1,.15,.2,.25,.3,.35])

ax=fig.add_subplot(gs[0,1]); panel(ax,'b')
steps=np.arange(1,9); inc=np.array([.025,.038,.030,.047,.040,.050,.031,.026]); rec=np.array([.010,.014,.019,.016,.018,.015,.023,.020]); net=inc-rec
ax.bar(steps-.18,inc,width=.36,color=ORANGE,label=r'incident $J^+$')
ax.bar(steps+.18,rec,width=.36,color=TEAL,label=r'recovery $J^-$')
ax2=ax.twinx(); ax2.plot(steps,np.cumsum(net),color=BLUE,marker='o',ms=3,lw=1.7,label='cumulative SBT')
head=.13; ax2.axhline(head,color=RED,ls='--',lw=1.1); ax2.text(6.45,head+.010,'headroom',color=RED,fontsize=7)
ax.set(xlabel='transition',ylabel='transport mass'); ax.set_title('Incident and recovery resolve\nthe risk increment',fontsize=8.1,pad=7); ax2.set_ylabel('cumulative net transport')
ax.set_xticks(steps); ax.set_ylim(0,.06); ax2.set_ylim(0,.18)
handles=[Line2D([0],[0],color=ORANGE,lw=6),Line2D([0],[0],color=TEAL,lw=6),Line2D([0],[0],color=BLUE,lw=1.8)]
ax.legend(handles,['incident','recovery','cumulative SBT'],loc='upper center',frameon=False,bbox_to_anchor=(.5,-.18),ncol=3)

ax=fig.add_subplot(gs[0,2]); panel(ax,'c'); ax.axis('off'); ax.set_title('One object,\nthree evidential levels',fontsize=8.1,pad=7)
boxes=[(.08,.70,.84,.20,BLUE,'Formation','exact paired accounting'),(.08,.40,.84,.20,PURPLE,'Observation','dynamically sufficient telemetry'),(.08,.10,.84,.20,GREEN,'Control','training-support intervention')]
for x,y,w,h,c,title,sub in boxes:
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=.02',facecolor=c,edgecolor='none',alpha=.10,transform=ax.transAxes)
    ax.add_patch(p); ax.text(x+.04,y+.12,title,transform=ax.transAxes,fontweight='bold',color=c,fontsize=9); ax.text(x+.04,y+.055,sub,transform=ax.transAxes,fontsize=7.1,color=DARK)
for y1,y2 in [(.70,.60),(.40,.30)]:
    ax.add_patch(FancyArrowPatch((.5,y1),(.5,y2),transform=ax.transAxes,arrowstyle='-|>',mutation_scale=10,color=GRAY,lw=1))

# bottom wide panel: equal net change, different turnover
ax=fig.add_subplot(gs[1,:]); panel(ax,'d')
ax.set_title('Equal aggregate increments can hide different identity-level ledgers',pad=7)
ax.set_xlim(-.2,10.2); ax.set_ylim(-.4,6.1); ax.set_yticks([]); ax.set_xlabel('deployment state')
# Ledger A: two incident transitions, no recovery (net +2 identities)
for row,(tcross,recover) in enumerate([(3.0,None),(6.0,None),(None,None),(None,None)]):
    y=5.25-row*.62
    ax.plot([0,10],[y,y],color=BLUE,lw=2)
    if tcross is not None:
        ax.plot([tcross,10],[y,y],color=ORANGE,lw=2); ax.scatter([tcross],[y],s=20,facecolor='white',edgecolor=DARK,zorder=3)
ax.text(.15,5.70,'ledger A: incident 2, recovery 0, net +2',fontsize=7.2,fontweight='bold')
# Ledger B: five incident transitions and three recoveries (same net +2)
ys=[]
for row,(tcross,recover) in enumerate([(1.5,7.4),(2.5,None),(3.6,8.6),(4.6,None),(5.5,9.2)]):
    y=2.55-row*.48; ys.append(y)
    ax.plot([0,tcross],[y,y],color=BLUE,lw=2); ax.plot([tcross,recover if recover is not None else 10],[y,y],color=ORANGE,lw=2)
    ax.scatter([tcross],[y],s=20,facecolor='white',edgecolor=DARK,zorder=3)
    if recover is not None:
        ax.plot([recover,10],[y,y],color=BLUE,lw=2); ax.scatter([recover],[y],s=20,facecolor='white',edgecolor=DARK,zorder=3)
ax.text(.15,2.98,'ledger B: incident 5, recovery 3, net +2',fontsize=7.2,fontweight='bold')
ax.text(8.15,5.68,'same net risk change',fontsize=7.2,color=RED)
ax.text(7.55,-.08,'blue: correct   orange: error',fontsize=7.0)
ax.spines['left'].set_visible(False); ax.spines['bottom'].set_position(('data',-.18))
save(fig,'Figure_1_SBT_framework')

# ---------------- Figure 2: formation ----------------
pe=pd.read_csv(E/'torchsig_round11c_paired_effects.csv')
fig,axs=plt.subplots(2,2,figsize=(7.2,5.35),constrained_layout=True)
ax=axs[0,0]; panel(ax,'a')
for _,r in pe.iterrows():
    ax.plot([0,1],[r.end_risk_baseline,r.end_risk_aware],color=LIGHT,lw=1)
    ax.scatter(0,r.end_risk_baseline,color=ORANGE,s=22,zorder=3); ax.scatter(1,r.end_risk_aware,color=GREEN,s=22,zorder=3)
ax.set_xticks([0,1],['baseline','Cliff-aware']); ax.set_ylabel('endpoint error risk'); ax.set_title('Training reduces endpoint risk on paired streams')
ax.text(.52,.93,'10 seed–path pairs',transform=ax.transAxes,fontsize=7.2)

ax=axs[0,1]; panel(ax,'b')
colors=np.where(pe.path=='noise',BLUE,ORANGE)
ax.scatter(pe.first_crossing_entropy_baseline,pe.incident_persistence_baseline,c=colors,s=30,alpha=.9)
ax.set(xlabel='normalized first-crossing entropy',ylabel='endpoint persistence',title='Crossings are distributed along ordered paths')
ax.legend([Line2D([0],[0],marker='o',color='w',markerfacecolor=BLUE,markersize=5),Line2D([0],[0],marker='o',color='w',markerfacecolor=ORANGE,markersize=5)],['noise','mixed gradient'],frameon=False,loc='lower right')

# CIFAR endpoint risk increase from paired outputs
cifar=pd.read_csv(E/'cifar10c_endpoint_by_seed.csv')
corrs=np.asarray(sorted(cifar.corruption.unique()))
seed_values=sorted(cifar.seed.unique())
inc=np.asarray([[float(cifar[(cifar.corruption==c)&(cifar.seed==s)].endpoint_risk_increase.iloc[0]) for s in seed_values] for c in corrs])
cross=[int(cifar[cifar.corruption==c].crossed_headroom_0_15.astype(bool).sum()) for c in corrs]
ax=axs[1,0]; panel(ax,'c')
order=np.argsort(inc.mean(axis=1)); y=np.arange(len(corrs));
barcols=[RED if cross[i]==3 else GOLD if cross[i]>0 else GRAY for i in order]
ax.barh(y,inc.mean(axis=1)[order],color=barcols,alpha=.9)
ax.errorbar(inc.mean(axis=1)[order],y,xerr=[inc.mean(axis=1)[order]-inc.min(axis=1)[order],inc.max(axis=1)[order]-inc.mean(axis=1)[order]],fmt='none',ecolor=DARK,lw=.7,capsize=2)
ax.axvline(.15,color=BLUE,ls='--',lw=1)
ax.set_yticks(y,[str(c).replace('_',' ') for c in corrs[order]]); ax.set_xlabel('endpoint risk increase'); ax.set_title('Risk growth is not sufficient for a cliff')
ax.legend([Rectangle((0,0),1,1,color=RED),Rectangle((0,0),1,1,color=GOLD),Rectangle((0,0),1,1,color=GRAY)],['3/3 cross','1–2/3 cross','0/3 cross'],frameon=False,loc='lower right')

ax=axs[1,1]; panel(ax,'d')
names=['incorrect\npartition','closest-risk\ntrained peer']; vals=[.0748696,.0153888]
ax.bar(np.arange(2),vals,color=[GRAY,PURPLE],width=.58)
ax.axhline(0,color=BLUE,lw=1.2,label='focal self: exact zero reference')
ax.set_xticks(np.arange(2),names); ax.set_ylabel('CIFAR reconstruction RMSE'); ax.set_title('Complementary boundary placebos')
ax.set_ylim(0,.082)
for i,v in enumerate(vals): ax.text(i,v+.002,f'{v:.3f}',ha='center',fontsize=7)
ax.text(.47,.92,'peer NRMSE = 0.165\nanchor-risk gap = 0.0002',transform=ax.transAxes,fontsize=6.7,va='top')
ax.legend(frameon=False,loc='upper right',bbox_to_anchor=(1.0,.64),fontsize=6.3)
save(fig,'Figure_2_paired_formation')

# ---------------- Figure 3: warning ----------------
knock=pd.read_csv(E/'torchsig_temporal_order_pairs.csv')
base=pd.read_csv(DIAG/'warning_fair_baselines.csv')
nest=pd.read_csv(V7DIAG/'warning_nested_channel_summary.csv')
diff=pd.read_csv(DIAG/'cure_or_cliff_difficulty_summary.csv')
fig,axs=plt.subplots(2,2,figsize=(7.2,5.25),constrained_layout=True)
ax=axs[0,0]; panel(ax,'a')
# thin paired lines sampled all
for _,r in knock.iterrows():
    ax.plot([0,1],[r.shuffle_forecast_mean,r.ordered_forecast_mean],color=LIGHT,lw=.5,alpha=.55)
ax.scatter(np.zeros(len(knock)),knock.shuffle_forecast_mean,color=GRAY,s=7,alpha=.5)
ax.scatter(np.ones(len(knock)),knock.ordered_forecast_mean,color=BLUE,s=7,alpha=.5)
ax.set_xticks([0,1],['fixed-terminal\nshuffle','ordered\nhistory']); ax.set_ylabel('integrated forecast'); ax.set_title('Temporal order adds prospective signal')
ax.text(.04,.93,f'mean difference = {knock.forecast_difference.mean():.3f}',transform=ax.transAxes,fontsize=7.2)

ax=axs[0,1]; panel(ax,'b')
short={
'Registered Hybrid25':'registered','Time only (refit)':'time','Static current telemetry (refit)':'static','Current active-state telemetry (refit)':'active','Entropy-margin trend (refit)':'entropy','Unsigned shift (refit)':'shift','Hybrid25 full temporal (refit)':'temporal refit','Estimated current risk':'risk','Estimated risk + slope':'risk+slope','Risk-proxy CUSUM':'CUSUM'}
comparison_methods=list(short)
base_cmp=base[base.method.isin(comparison_methods)].copy()
label_offsets={'Registered Hybrid25':(-7,-18),'Current active-state telemetry (refit)':(18,8),'Static current telemetry (refit)':(-34,2),'Hybrid25 full temporal (refit)':(-52,-15),'Estimated current risk':(7,3),'Estimated risk + slope':(7,3),'Risk-proxy CUSUM':(7,3),'Entropy-margin trend (refit)':(7,-11),'Unsigned shift (refit)':(7,4),'Time only (refit)':(7,4)}
for _,r in base_cmp.iterrows():
    x=r.conf_false/r.conf_controls; y=r.conf_timely/r.conf_cliffs
    is_reg=r.method=='Registered Hybrid25'; is_active='active-state' in r.method
    c=RED if is_reg else BLUE if is_active else GRAY
    marker='o' if is_reg else 'D' if is_active else 'o'
    z=7 if (is_reg or is_active) else 3
    ax.scatter(x,y,s=38,color=c,marker=marker,zorder=z,edgecolor='white',linewidth=.4)
    if not (is_reg or is_active):
        off=label_offsets.get(r.method,(4,3)); ax.annotate(short.get(r.method,r.method),(x,y),xytext=off,textcoords='offset points',fontsize=5.7,arrowprops=dict(arrowstyle='-',lw=.45,color=c) if abs(off[0])+abs(off[1])>16 else None)
handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=RED,markersize=6,label='registered'),Line2D([0],[0],marker='D',color='w',markerfacecolor=BLUE,markersize=5.5,label='one-step augmented')]
ax.legend(handles=handles,frameon=False,loc='center left',bbox_to_anchor=(0.19,0.54),fontsize=6.0)
ax.axvline(.075,color=GRAY,ls='--',lw=.8); ax.set(xlabel='confirmation false-alarm rate',ylabel='timely-warning rate',title='Post hoc matched-calibration comparison')
ax.set_xlim(-.015,.38); ax.set_ylim(-.03,1.08)

ax=axs[1,0]; panel(ax,'c')
summary=nest.set_index('kind')
kind_order=['static_state','static_plus_net','static_plus_persistence','current_state']
labels=['static','+ net proxy','+ persistence','+ both']; timely=summary.loc[kind_order,'conf_timely'].to_numpy(); false=summary.loc[kind_order,'conf_false'].to_numpy()
ax.bar(np.arange(4),timely/72,color=[GRAY,ORANGE,TEAL,BLUE],width=.62)
ax.set_xticks(np.arange(4),labels,rotation=12); ax.set_ylim(0,1.15); ax.set_ylabel('timely-warning rate'); ax.set_title('Nested one-step state ablation',pad=10)
for i,(tval,fval) in enumerate(zip(timely,false)):
    ax.text(i,tval/72+.025,f'{int(tval)}/72',ha='center',fontsize=6.8,fontweight='bold')
    ax.text(i,.05,f'FA {int(fval)}/78',ha='center',fontsize=6.2,color=GRAY)
ax=axs[1,1]; panel(ax,'d')
sub=diff[(diff.role=='confirmation') & (diff.stratifier.isin(['endpoint_overshoot','pre_event_slope']))].copy()
# plot grouped rates
x=[]; y=[]; labs=[]; cols=[]
for strat,c in [('endpoint_overshoot',PURPLE),('pre_event_slope',GREEN)]:
    g=sub[sub.stratifier==strat]
    for _,r in g.iterrows():
        x.append(len(x)); y.append(r.timely_rate); labs.append(('overshoot ' if strat=='endpoint_overshoot' else 'slope ')+str(r.stratum)); cols.append(c)
    x.append(len(x)); y.append(np.nan); labs.append(''); cols.append(c)
valid=[i for i,v in enumerate(y) if not np.isnan(v)]
ax.bar(valid,[y[i] for i in valid],color=[cols[i] for i in valid],width=.7)
ax.set_xticks(valid,[labs[i] for i in valid],rotation=35,ha='right'); ax.set_ylim(0,1.05); ax.set_ylabel('timely-warning rate'); ax.set_title('Performance spans observed difficulty strata')
ax.axhline(.75,color=GRAY,ls='--',lw=.8)
save(fig,'Figure_3_warning_observability')

# ---------------- Figure 4: training support control ----------------
traj=pd.read_csv(E/'torchsig_round10_trajectory_summary.csv')
geo=pd.read_csv(E/'torchsig_round10_geometry_summary.csv')
# aggregate paths within seed
agg=traj.groupby(['replicate_seed','regime']).agg(end_risk=('end_risk','mean'),risk_auc=('risk_auc','mean')).reset_index()
reg_order=['support_depleted','baseline','random_broad','cliff_aware']; reg_labels=['support\ndepleted','baseline','random\nbroad','Cliff-aware']; reg_cols=[RED,ORANGE,TEAL,GREEN]
fig,axs=plt.subplots(2,2,figsize=(7.2,5.2),constrained_layout=True)
for ax,col,title,ylabel,letter in [(axs[0,0],'end_risk','Terminal risk','risk','a'),(axs[0,1],'risk_auc','Path risk area','risk area','b')]:
    panel(ax,letter)
    for i,reg in enumerate(reg_order):
        g=agg[agg.regime==reg][col]
        ax.scatter(np.full(len(g),i)+np.linspace(-.08,.08,len(g)),g,color=reg_cols[i],s=20,zorder=3)
        ax.plot([i-.18,i+.18],[g.mean(),g.mean()],color=DARK,lw=1.4)
    ax.set_xticks(range(4),reg_labels); ax.set_ylabel(ylabel); ax.set_title(title)
    if col=='end_risk': ax.axhline(.1998458772,color=GRAY,ls='--',lw=.8)
ax=axs[1,0]; panel(ax,'c')
bas=geo[geo.regime=='baseline'].iloc[0]
vals=[]
for reg in reg_order:
    r=geo[geo.regime==reg].iloc[0]; vals.append([r.b_norm_mean/bas.b_norm_mean,r.H_frobenius_mean/bas.H_frobenius_mean])
vals=np.asarray(vals); x=np.arange(4); w=.34
ax.bar(x-w/2,vals[:,0],w,label=r'$\|b\|$',color=BLUE); ax.bar(x+w/2,vals[:,1],w,label=r'$\|H\|_F$',color=PURPLE)
ax.set_xticks(x,reg_labels); ax.set_ylabel('ratio to baseline'); ax.set_title('Training reshapes local risk geometry'); ax.set_yscale('log'); ax.legend(frameon=False)
ax=axs[1,1]; panel(ax,'d')
# exact decomposition values
dec=pd.read_csv(E/'training_gain_decomposition.csv').set_index('regime'); labels=['random broad','Cliff-aware']; anchor=np.array([dec.loc['random_broad','anchor_gain'],dec.loc['cliff_aware','anchor_gain']]); future=np.array([dec.loc['random_broad','future_sbt_suppression'],dec.loc['cliff_aware','future_sbt_suppression']])
x=np.arange(2)
ax.bar(x,anchor,color=TEAL,label='anchor-risk gain'); ax.bar(x,future,bottom=anchor,color=GREEN,label='future SBT suppression')
ax.set_xticks(x,labels); ax.set_ylabel('terminal-risk gain'); ax.set_title('Terminal gain separates anchor and future transport')
for i in range(2): ax.text(i,anchor[i]+future[i]+.008,f'{anchor[i]+future[i]:.3f}',ha='center',fontsize=7)
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2)
save(fig,'Figure_4_training_support_control')

# ---------------- Figure 5: repair allocation reversal ----------------
rf_sel=pd.read_csv(E/'torchsig_round12c_selection_summary.csv')
rf_eff=pd.read_csv(E/'torchsig_round12c_seed_effects.csv')
r13_sel=pd.read_csv(E/'cifar_repair_selection_summary.csv')
r13_res=pd.read_csv(E/'cifar_repair_seed_results.csv')
fig,axs=plt.subplots(2,2,figsize=(7.2,5.25),constrained_layout=True)
arm_order=['random_unstratified','hazard_concentrated','coverage_random','coverage_hazard']; arm_labels=['random','hazard','coverage','coverage+hazard']; arm_cols=[GRAY,RED,BLUE,GREEN]
ax=axs[0,0]; panel(ax,'a')
mean_sel=rf_sel.groupby('arm').agg(coverage=('coverage_fraction','mean'),hazard=('mean_local_hazard_score','mean'),hit=('true_local_incident_precision','mean')).loc[arm_order]
for i,(arm,r) in enumerate(mean_sel.iterrows()):
    ax.scatter(r.coverage,r.hazard,s=80+220*r.hit,color=arm_cols[i],alpha=.85,edgecolor='white',lw=.7)
    ax.annotate(arm_labels[i],(r.coverage,r.hazard),xytext=(4,3),textcoords='offset points',fontsize=6.7)
ax.set(xlabel='occupied deployment-cell fraction',ylabel='mean local hazard',title='Sparse RF manipulation')
handles=[plt.scatter([],[],s=80+220*v,color=GRAY,alpha=.55,edgecolor='white') for v in (.1,.4,.8)]
ax.legend(handles,['10%','40%','80% hit'],title='incident hit rate',frameon=False,loc='lower right',fontsize=6.0,title_fontsize=6.2)

ax=axs[0,1]; panel(ax,'b')
for i,arm in enumerate(arm_order):
    g=rf_eff[rf_eff.arm==arm].mean_end_risk_reduction
    ax.scatter(np.full(len(g),i)+np.linspace(-.08,.08,len(g)),g,color=arm_cols[i],s=20)
    ax.plot([i-.18,i+.18],[g.mean(),g.mean()],color=DARK,lw=1.4)
ax.set_xticks(range(4),arm_labels,rotation=18); ax.set_ylabel('terminal-risk reduction'); ax.set_title('Tested sparse-field contrast favours coverage')

ax=axs[1,0]; panel(ax,'c')
names=['hazard','coverage']; fragments=[float(r13_sel.set_index('arm').loc[a,'unique_fragments']) for a in names]; pressure=[float(r13_sel.set_index('arm').loc[a,'mean_boundary_pressure']) for a in names]
x=np.arange(2); ax.bar(x,fragments,color=[RED,BLUE],width=.55); ax.set_xticks(x,names); ax.set_ylabel('occupied fragments'); ax.set_title('Both image arms already span the domain')
ax2=ax.twinx(); ax2.plot(x,pressure,color=DARK,marker='o',lw=1.3); ax2.set_ylabel('mean boundary pressure')

ax=axs[1,1]; panel(ax,'d')
# use averages from manuscript (exact summary aggregation)
names=['baseline','random','coverage','hazard']; armmean=r13_res.groupby('arm')[['endpoint_error_mean','risk_area_mean']].mean(); endpoint=[float(armmean.loc[a,'endpoint_error_mean']) for a in names]; area=[float(armmean.loc[a,'risk_area_mean']) for a in names]; cols=[GRAY,GOLD,BLUE,RED]
x=np.arange(4); w=.34
ax.bar(x-w/2,endpoint,w,color=cols,label='endpoint error'); ax.bar(x+w/2,area,w,color=cols,alpha=.45,label='risk area')
ax.set_xticks(x,names,rotation=15); ax.set_ylabel('error'); ax.set_title('Prospective ordering reverses\nin CIFAR-10-C',fontsize=8.4,pad=7)
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2)
save(fig,'Figure_5_repair_reversal')

# ---------------- Figure 6: CURE-OR serial loop ----------------
paths=pd.read_csv(E/'cure_or_path_level_results.csv'); seeds_df=pd.read_csv(E/'cure_or_seed_level_results.csv')
fig=plt.figure(figsize=(7.2,5.15),constrained_layout=True); gs=fig.add_gridspec(2,3,height_ratios=[1,1],wspace=.62,hspace=.68)
ax=fig.add_subplot(gs[0,0]); panel(ax,'a')
incident=[]; recovery_headroom=[]; cliff_flag=[]
for _,r in paths.iterrows():
    risk=np.asarray(json.loads(r.risk),float); f=np.asarray(json.loads(r.forward),float)/50; rec=np.asarray(json.loads(r.recovery),float)/50
    incident.append(float(np.sum(f))); recovery_headroom.append(float(np.sum(rec)+(0.5-risk[0])))
    # persistence-confirmed event is stored in the frozen table
    cliff_flag.append(not pd.isna(r.event))
incident=np.asarray(incident); recovery_headroom=np.asarray(recovery_headroom); cliff_flag=np.asarray(cliff_flag)
ax.scatter(incident[~cliff_flag],recovery_headroom[~cliff_flag],s=17,color=GRAY,alpha=.65,label='controls')
ax.scatter(incident[cliff_flag],recovery_headroom[cliff_flag],s=18,color=RED,alpha=.72,label='persistent cliffs')
lo=min(incident.min(),recovery_headroom.min())-.02; hi=max(incident.max(),recovery_headroom.max())+.02
ax.plot([lo,hi],[lo,hi],color=DARK,lw=1)
ax.set(xlabel='cumulative incident transport',ylabel='recovery + initial headroom'); ax.set_title('Headroom exhaustion\nacross 150 paths',fontsize=8.5,pad=7); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
ax.legend(frameon=False,loc='upper left',fontsize=6.2)
ax=fig.add_subplot(gs[0,1]); panel(ax,'b')
x=np.arange(len(seeds_df)); ax.bar(x-.18,seeds_df.timely_rate,.36,color=BLUE,label='timely rate'); ax.bar(x+.18,seeds_df.false_rate,.36,color=RED,label='false-alarm rate')
ax.set_xticks(x,seeds_df.seed.astype(str)); ax.set_ylim(0,1.08); ax.set_xlabel('classifier-head seed'); ax.set_ylabel('rate'); ax.set_title('Head-seed warning\nperformance',fontsize=8.5,pad=7); ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2)

ax=fig.add_subplot(gs[0,2]); panel(ax,'c')
cols=[GREEN if e else GRAY for e in seeds_df.repair_eligible]
ax.bar(x,seeds_df.repair_mean_risk_gain,color=cols); ax.set_xticks(x,seeds_df.seed.astype(str)); ax.set_ylabel('mean deployment-risk gain'); ax.set_xlabel('classifier-head seed'); ax.set_title('Guarded update\nand abstention',fontsize=8.5,pad=7)
for i,r in seeds_df.iterrows():
    if not r.repair_eligible: ax.text(i,.003,'baseline retained',ha='center',rotation=90,va='bottom',fontsize=6.5)

ax=fig.add_subplot(gs[1,:]); panel(ax,'d'); ax.axis('off'); ax.set_title('Preregistered serial formation–warning–control loop',fontsize=9,pad=8)
items=[('Paired\nformation','72 cliffs / 78 controls',BLUE),('Identity-anchored\noutcome-blind\nwarning','71/72 timely\n3/78 false',PURPLE),('No-harm\ngate','4 deploy / 1 abstain',ORANGE),('Guarded\ncontrol','5 model–family cliffs\nremoved; 0 introduced',GREEN)]
xs=[.015,.265,.515,.765]; wbox=.205
for i,(title,sub,c) in enumerate(items):
    box=FancyBboxPatch((xs[i],.26),wbox,.47,boxstyle='round,pad=.012,rounding_size=.018',transform=ax.transAxes,facecolor=c,alpha=.10,edgecolor=c,lw=1)
    ax.add_patch(box); ax.text(xs[i]+wbox/2,.56,title,ha='center',va='center',transform=ax.transAxes,fontweight='bold',color=c,fontsize=7.0); ax.text(xs[i]+wbox/2,.36,sub,ha='center',va='center',transform=ax.transAxes,fontsize=6.2)
    if i<3: ax.add_patch(FancyArrowPatch((xs[i]+wbox+.005,.495),(xs[i+1]-.008,.495),transform=ax.transAxes,arrowstyle='-|>',mutation_scale=9,color=GRAY,lw=1))
save(fig,'Figure_6_CURE_OR_closed_loop')

# ---------------- Extended Data figures ----------------
# ED1 threshold sensitivity
t10=pd.read_csv(E/'torchsig_threshold_sensitivity.csv')
ct=pd.read_csv(E/'cure_or_threshold_sensitivity.csv')
fig,axs=plt.subplots(1,2,figsize=(7.2,2.7),constrained_layout=True)
ax=axs[0]; panel(ax,'a')
for col,label,c in [('support_depleted','support depleted',RED),('baseline','baseline',ORANGE),('random_broad','random broad',TEAL),('cliff_aware','Cliff-aware',GREEN)]: ax.plot(t10.boundary,t10[col]/10,marker='o',ms=3,label=label,color=c)
ax.set(xlabel='operational boundary',ylabel='crossing fraction',title='Training intervention across thresholds'); ax.legend(frameon=False)
ax=axs[1]; panel(ax,'b'); ax.plot(ct.boundary,ct.timely,color=BLUE,marker='o',ms=3,label='timely'); ax.plot(ct.boundary,ct.false,color=RED,marker='o',ms=3,label='false alarm'); ax.axvline(.5,color=GRAY,ls='--',lw=.8); ax.set(xlabel='event boundary',ylabel='rate',title='Registered warning is calibration conditional'); ax.legend(frameon=False)
save(fig,'Extended_Data_Figure_1_threshold_sensitivity')

# ED2 normalized peer distributions
nr=pd.read_csv(DIAG/'trained_peer_boundary_normalized_rows.csv')
fig,axs=plt.subplots(1,2,figsize=(7.2,2.75),constrained_layout=True)
for ax,domain,letter in [(axs[0],'CIFAR-10-C','a'),(axs[1],'CURE-OR','b')]:
    panel(ax,letter); g=nr[(nr.domain==domain)&(nr.active)]
    vals=g.peer_nrmse_rms.replace([np.inf,-np.inf],np.nan).dropna()
    ax.hist(vals,bins=12,color=PURPLE,alpha=.75); ax.axvline(vals.median(),color=DARK,ls='--',lw=1); ax.set(xlabel='trained-peer RMSE / focal RMS increment',ylabel='paths or cells',title=f'{domain}: normalized peer error')
save(fig,'Extended_Data_Figure_2_peer_boundary_normalization')

# ED3 warning operating-budget sensitivity
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),constrained_layout=True)
ax=axs[0]; panel(ax,'a')
base2=base[base.method.isin(comparison_methods)].copy(); base2['false_rate']=base2.conf_false/base2.conf_controls; base2['timely_rate']=base2.conf_timely/base2.conf_cliffs
order=base2.sort_values('timely_rate').index; y=np.arange(len(order));
ax.scatter(base2.loc[order,'timely_rate'],y,color=BLUE,s=30,label='timely'); ax.scatter(base2.loc[order,'false_rate'],y,color=RED,s=30,label='false alarm')
ax.set_yticks(y,[short.get(m,m) for m in base2.loc[order,'method']]); ax.set_xlabel('confirmation rate'); ax.set_title('7.5% calibration operating point'); ax.legend(frameon=False,loc='lower right')
ax=axs[1]; panel(ax,'b')
bud=pd.read_csv(V7DIAG/'warning_false_budget_sensitivity.csv')
for kind,label,c,mark in [('static_state','static',GRAY,'o'),('static_plus_net','+ net proxy',ORANGE,'o'),('current_state','+ both proxies',BLUE,'s'),('full34','full temporal',PURPLE,'^')]:
    g=bud[bud.kind==kind].sort_values('false_budget')
    ax.plot(g.conf_false/g.conf_controls,g.conf_timely/g.conf_cliffs,marker=mark,color=c,label=label,lw=1.4,ms=4)
ax.set(xlabel='confirmation false-alarm rate',ylabel='timely-warning rate',title='Fixed scores, calibration-budget sensitivity'); ax.legend(frameon=False,fontsize=6.2)
save(fig,'Extended_Data_Figure_3_warning_baselines')

# ED4 difficulty calibration vs confirmation
rows=pd.read_csv(DIAG/'cure_or_cliff_difficulty_rows.csv')
fig,axs=plt.subplots(1,3,figsize=(7.2,2.6),constrained_layout=True)
for ax,var,title,letter in zip(axs,['initial_headroom','pre_event_slope','endpoint_overshoot'],['initial headroom','pre-event slope','endpoint overshoot'],['a','b','c']):
    panel(ax,letter)
    data=[rows[(rows.role=='calibration')][var].dropna(),rows[(rows.role=='confirmation')][var].dropna()]
    bp=ax.boxplot(data,tick_labels=['calibration','confirmation'],patch_artist=True,widths=.55,showfliers=False)
    bp['boxes'][0].set_facecolor(GRAY); bp['boxes'][1].set_facecolor(BLUE)
    ax.set_title(title); ax.tick_params(axis='x',rotation=20)
save(fig,'Extended_Data_Figure_4_warning_difficulty')

# ED5 classifier-head-seed nested proxy effects
seed=pd.read_csv(V7DIAG/'warning_nested_channel_by_seed.csv')
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9),constrained_layout=True)
ax=axs[0]; panel(ax,'a')
order=['static_state','static_plus_net','static_plus_persistence','current_state']; labels=['static','+ net proxy','+ persistence','+ both']; cols=[GRAY,ORANGE,TEAL,BLUE]
seeds=sorted(seed.seed.unique()); x=np.arange(len(seeds)); width=.18
for j,(kind,label,c) in enumerate(zip(order,labels,cols)):
    g=seed[seed.kind==kind].set_index('seed').loc[seeds]
    ax.bar(x+(j-1.5)*width,g.timely,width,color=c,label=label)
ax.set_xticks(x,[str(v) for v in seeds]); ax.set_xlabel('classifier-head seed'); ax.set_ylabel('timely cliff count'); ax.set_title('Nested proxy effect by head seed'); ax.legend(frameon=False,ncol=2,fontsize=6.1)
ax=axs[1]; panel(ax,'b')
static=seed[seed.kind=='static_state'].set_index('seed').loc[seeds]
for j,(kind,label,c) in enumerate(zip(order[1:],labels[1:],cols[1:])):
    g=seed[seed.kind==kind].set_index('seed').loc[seeds]
    diff=g.timely/g.cliffs-static.timely/static.cliffs
    ax.plot(x,diff,marker='o',color=c,label=label,lw=1.2)
ax.axhline(0,color=GRAY,lw=.8); ax.set_xticks(x,[str(v) for v in seeds]); ax.set_xlabel('classifier-head seed'); ax.set_ylabel('timely-rate difference vs static'); ax.set_title('Gain is distributed across head seeds'); ax.legend(frameon=False,fontsize=6.1)
save(fig,'Extended_Data_Figure_5_nested_proxy_by_seed')

print('created',len(list(MAIN_OUT.glob('*.png')))+len(list(ED_OUT.glob('*.png'))),'png figures in',OUT)
