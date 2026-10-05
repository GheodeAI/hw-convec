"""Month/area matching, retained-event diagnostics and pair+year FE OLS."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.environment import ORDER, LABELS, UNITS
from common.io import dump
from common.matching import COV, match
from common.stats import bh, fit_difference
from common.style import BLUE, ORANGE, apply_style

# Importing the figure script used to apply the journal style as a side effect.
apply_style()

def selection(c,kept,out):
    r=c.loc[kept];x=c.loc[~c.index.isin(kept)]
    variables=COV+['max_daily_land_area_km2'];rows=[]
    for v in variables:
        scale=np.sqrt((r[v].var()+x[v].var())/2)
        rows.append(dict(variable=v,retained_n=len(r),excluded_n=len(x),retained_mean=r[v].mean(),excluded_mean=x[v].mean(),
                         retained_median=r[v].median(),excluded_median=x[v].median(),
                         smd_retained_minus_excluded=(r[v].mean()-x[v].mean())/scale,
                         ks_distance=stats.ks_2samp(r[v],x[v]).statistic))
    tab=pd.DataFrame(rows);tab.to_csv(out/'retained_vs_excluded_compound.csv',index=False)
    # Fixed geographic bands are descriptive, not administrative regions. The
    # outer edges span the EUCLID CG grid (lat 35-72, lon -11-33), not the
    # narrower study-region box: at low absolute thresholds some event centres
    # fall in 35-36N / -11 to -10E, which the old [36,...] / [-10,...] edges
    # rejected outright.
    bands=[]
    for v,edges in [('center_latitude',[35,45,55,72.5]),('center_longitude',[-11,10,20,40])]:
        assert c[v].between(edges[0],edges[-1],inclusive='left').all()
        for lo,hi in zip(edges[:-1],edges[1:]):
            nr=int(((r[v]>=lo)&(r[v]<hi)).sum());nx=int(((x[v]>=lo)&(x[v]<hi)).sum())
            bands.append(dict(coordinate=v,lower=lo,upper=hi,retained_n=nr,excluded_n=nx,
                              retained_group_fraction=nr/len(r),excluded_group_fraction=nx/len(x),
                              retention_within_band=nr/(nr+nx) if nr+nx else np.nan))
    pd.DataFrame(bands).to_csv(out/'retention_geographic_bands.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(14,8));hist=[]
    for g in [r,x]:
        h,xe,ye=np.histogram2d(g.center_longitude,g.center_latitude,bins=[np.arange(-10,40.01,2.5),np.arange(36,72.01,2)])
        hist.append(h.T/len(g)*100)
    vmax=max(h.max() for h in hist)
    for ax,h,title in zip(axs[0,:2],hist,[f'Retained compound (n={len(r)})',f'Excluded compound (n={len(x)})']):
        mesh=ax.pcolormesh(xe,ye,h,cmap='Reds',vmin=0,vmax=vmax);ax.set(title=title,xlabel='Longitude (°E)',ylabel='Latitude (°N)')
        fig.colorbar(mesh,ax=ax).set_label('Within-group events (%)')
    lat=pd.DataFrame(bands).query('coordinate == "center_latitude"')
    axs[0,2].bar(np.arange(3)-.2,lat.retained_group_fraction*100,width=.38,color=BLUE,label='Retained')
    axs[0,2].bar(np.arange(3)+.2,lat.excluded_group_fraction*100,width=.38,color=ORANGE,label='Excluded')
    axs[0,2].set_xticks(range(3),['35–45°N','45–55°N','55–72.5°N']);axs[0,2].set(title='Latitude composition',ylabel='Within-group events (%)');axs[0,2].legend(fontsize=9)
    for ax,v,label in zip(axs[1],['intensity','log_max_daily_land_area','duration'],['Intensity (°C)','ln(peak daily land area / km²)','Duration (days)']):
        for g,col,name in [(r,BLUE,'Retained'),(x,ORANGE,'Excluded')]:
            a=np.sort(g[v]);ax.step(a,np.arange(1,len(a)+1)/len(a),c=col,label=name)
        ax.set(xlabel=label,ylabel='Cumulative probability');ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.suptitle('Selection among eligible compound events: retained versus excluded')
    fig.tight_layout();fig.savefig(out/'retained_vs_excluded_compound.png',dpi=200);plt.close(fig)
    return tab


def run(source,out):
    out.mkdir(parents=True,exist_ok=False)
    sources=[source/'tables/regional_events.csv',source/'matching_ablation_v1/area_only_pairs.csv',source/'tables/event_environment_anomaly.csv']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    e=pd.read_csv(source/'area_year_diagnostics_v1/events_with_peak_daily_land_area.csv')
    e=e[e.start_month.between(4,9)&(e.cg_coverage_fraction==1)].set_index('event_id')
    pairs,cov,covariance=match(e,'exact_month',base_cov=COV)
    old=pd.read_csv(source/'matching_ablation_v1/area_only_pairs.csv')
    assert pairs[['compound_event_id','control_event_id']].equals(old[['compound_event_id','control_event_id']])
    pairs.to_csv(out/'match_pairs.csv',index=False)
    a=e.loc[pairs.compound_event_id];b=e.loc[pairs.control_event_id];c=e[e.has_lightning];n=e[~e.has_lightning]
    assert (a.start_month.values==b.start_month.values).all()
    rows=[]
    for v in COV+['max_daily_land_area_km2']:
        sd=np.sqrt((c[v].var()+n[v].var())/2)
        rows.append(dict(variable=v,smd_before=(c[v].mean()-n[v].mean())/sd,smd_after=(a[v].mean()-b[v].mean())/sd))
    balance=pd.DataFrame(rows);balance.to_csv(out/'match_balance.csv',index=False)
    assert (balance.iloc[:5].smd_after.abs()<.1).all(), 'Five-variable balance check failed; stop before regression'
    print('Five-variable balance passed; pairs',len(pairs),flush=True)
    selected=selection(c,a.index,out)
    aa=a.reset_index();bb=b.reset_index();aa['compound']=1;bb['compound']=0
    aa['match_set_id']=pairs.match_set_id.values;bb['match_set_id']=pairs.match_set_id.values
    env=pd.read_csv(source/'tables/event_environment_anomaly.csv').set_index('event_id')
    long=pd.concat([aa,bb],ignore_index=True).merge(env[ORDER],left_on='event_id',right_index=True,validate='one_to_one')
    long.to_csv(out/'matched_events_model_input.csv',index=False)
    year_rows=[]
    for year in range(2006,2026):
        pc=(c.year==year).mean();pn=(n.year==year).mean();denom=np.sqrt((pc*(1-pc)+pn*(1-pn))/2)
        year_rows.append(dict(year=year,compound_n=int((a.year==year).sum()),noncompound_n=int((b.year==year).sum()),
                              smd_after=((a.year==year).mean()-(b.year==year).mean())/denom))
    yt=pd.DataFrame(year_rows);yt.to_csv(out/'matched_year_balance.csv',index=False)
    y1=env.loc[a.index,ORDER].values;y0=env.loc[b.index,ORDER].values
    assert np.isfinite(y1).all() and np.isfinite(y0).all(), 'Missing environment values require variable-specific complete pairs'
    Y=y1-y0;years=np.arange(2007,2026)
    Z=(a.year.values[:,None]==years).astype(float)-(b.year.values[:,None]==years).astype(float)
    D=np.column_stack([np.ones(len(pairs)),Z])
    beta,vcov,residual,correction=fit_difference(D,Y)
    unadj,unvcov,_,uncorrection=fit_difference(np.ones((len(pairs),1)),Y)
    # Independent check against the literal 2G-row dummy-variable OLS model.
    g=len(pairs);F=np.column_stack([np.tile(np.eye(g),(2,1)),np.r_[np.ones(g),np.zeros(g)],
                                  np.vstack([(a.year.values[:,None]==years),(b.year.values[:,None]==years)])])
    P=np.linalg.pinv(F);levelY=np.vstack([y1,y0]);fullbeta=P@levelY;fullresid=levelY-F@fullbeta
    assert np.allclose(fullbeta[g:],beta,rtol=1e-7,atol=1e-9)
    cluster_beta_score=P[g,:g,None]*fullresid[:g]+P[g,g:,None]*fullresid[g:]
    full_se=np.sqrt(correction*np.sum(cluster_beta_score**2,axis=0))
    se=np.sqrt(vcov[:,0,0]);assert np.allclose(full_se,se,rtol=1e-7,atol=1e-10)
    critical=stats.t.ppf(.975,g-1);results=[];coefficients=[]
    for j,v in enumerate(ORDER):
        use=np.sqrt(unvcov[j,0,0]);scale=np.sqrt((env.loc[c.index,v].var()+env.loc[n.index,v].var())/2)
        results.append(dict(variable=v,n_pairs=g,n_events=2*g,difference=beta[0,j],cluster_se=se[j],
                   ci95_lower=beta[0,j]-critical*se[j],ci95_upper=beta[0,j]+critical*se[j],
                   p=2*stats.t.sf(abs(beta[0,j]/se[j]),g-1),standardized_difference=beta[0,j]/scale,
                   standardized_ci95_lower=(beta[0,j]-critical*se[j])/scale,standardized_ci95_upper=(beta[0,j]+critical*se[j])/scale,
                   pair_FE_only_difference=unadj[0,j],pair_FE_only_ci95_lower=unadj[0,j]-critical*use,pair_FE_only_ci95_upper=unadj[0,j]+critical*use,
                   cr1_correction=correction,ci_df=g-1))
        for k,term in enumerate(['compound']+[f'year_{yr}' for yr in years]):
            coefficients.append(dict(variable=v,term=term,coefficient=beta[k,j],cluster_se=np.sqrt(vcov[j,k,k])))
    result=pd.DataFrame(results);result['p_bh']=bh(result.p);result.to_csv(out/'environment_pair_year_FE.csv',index=False)
    pd.DataFrame(coefficients).to_csv(out/'all_year_FE_coefficients.csv',index=False)
    np.savez_compressed(out/'regression_arrays.npz',design_difference=D,outcomes_difference=Y,coefficients=beta,cluster_covariances=vcov)
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for ax,r in zip(axs.flat,result.itertuples()):
        ax.errorbar([0,1],[r.pair_FE_only_difference,r.difference],yerr=[[r.pair_FE_only_difference-r.pair_FE_only_ci95_lower,r.difference-r.ci95_lower],
                     [r.pair_FE_only_ci95_upper-r.pair_FE_only_difference,r.ci95_upper-r.difference]],fmt='o',c=BLUE,capsize=4)
        ax.set_xticks([0,1],['Pair FE','Pair + year FE']);ax.set_xlim(-.4,1.4);ax.set(title=LABELS[r.variable],ylabel=UNITS[r.variable]);ax.axhline(0,c='gray',lw=.8);ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'Compound − non-compound anomalies: {g} matched pairs\n95% CI with standard errors clustered by matched pair')
    fig.tight_layout();fig.savefig(out/'environment_pair_year_FE_95CI.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));yy=np.arange(5)
    ax.scatter(balance.iloc[:5].smd_before.abs(),yy,label='Before',color=ORANGE)
    ax.scatter(balance.iloc[:5].smd_after.abs(),yy,label='After',color=BLUE,marker='s')
    ax.set_yticks(yy,['Latitude','Longitude','Duration','Intensity','log(peak daily land area)']);ax.invert_yaxis();ax.axvline(.1,c='gray',ls='--')
    ax.set(xlabel='Absolute SMD',title='Exact start-month matching; five-variable Mahalanobis distance');ax.legend();fig.tight_layout();fig.savefig(out/'matching_balance.png',dpi=200);plt.close(fig)
    summary=dict(pairs=g,compound_eligible=len(c),compound_retained=g,compound_excluded=len(c)-g,retention=g/len(c),
        max_five_matching_abs_smd=float(balance.iloc[:5].smd_after.abs().max()),max_year_abs_smd_before_regression=float(yt.smd_after.abs().max()),
        model='environment = matched-pair FE + start-year FE + beta * compound',
        difference_design_rank=int(np.linalg.matrix_rank(D)),full_design_rank=g+int(np.linalg.matrix_rank(D)),
        n_clusters=g,cluster='matched pair',cr1_factor=correction,ci_distribution=f't({g-1})',
        finite_sample_convention='full dummy-variable OLS CR1, counting pair FE in K; conservative relative to dropping nested pair FE from degrees-of-freedom correction',
        validation='coefficients and cluster SE agree with literal full dummy-variable OLS to numerical tolerance',
        scope='inference for retained/matchable compound events; year FE cannot repair exclusion-related selection')
    dump(out/'summary.json',summary)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in sources)
    dump(out/'validation.json',dict(status='passed',original_inputs_sha256=hashes))
    (out/'README.md').write_text('''# 月份精确匹配＋面积距离，匹配对和年份固定效应

沿用 area_only 的 837 对：开始月份精确匹配，纬度、经度、持续时间、强度、ln(最大单日陆地面积) 的 Mahalanobis 距离 ≤0.75，1:1 不放回，全局距离升序贪心。年份不用于匹配。匹配五变量的 SMD 检查在回归前执行，不是要求未匹配原始样本先平衡。

模型：Y_ij = alpha_pair + gamma_startyear + beta Compound_ij + error_ij。月份已在匹配对内相同，由匹配对 FE 吸收。2006 为年份参考；每个环境变量分别拟合。环境 Y 沿用事件陆地并集覆盖在事件期间的时间/网格均值异常，异常基准未更改。

以复合减非复合差分精确吸收 pair FE，设计为截距＋19 个年份指示变量之差；截距即原模型的 beta。不是只给匹配对指定一个年份。已使用显式 2G 行、G 个 pair FE、复合指示和 19 个 year FE 的 OLS 独立验证系数与聚类标准误。

标准误为匹配对聚类 CR1，有限样本因子 G/(G−1) × (N−1)/(N−K)，其中 N=2G，K=G+20（包含 pair FE），95% CI 用 t(G−1)。采用完整哑变量模型的常规 CR1 口径；相比某些软件不计嵌套于聚类内的 pair FE 的校正，这一口径更保守。没有按年份或日期再聚类；单变量 CI 未做多重校正，另附六变量 BH p 值。Pair FE only 比较也采用同一完整模型 CR1 规则。

保留与剔除诊断仅针对原先合格的复合事件，不把非暖季或 CG 覆盖不全事件混入剔除组。方向是保留减剔除，用两组 pooled SD，并报告中位数、KS 距离和固定经纬度带。空间图为各组内部归一化比例，不是地区内保留率；地区内保留率另见 retention_geographic_bands.csv。

年份回归调整不消除因匹配筛选产生的代表性差异，也不保证模型能推广到全部 1536 个复合事件。原事件、原匹配及既有图件均未覆盖。
''')
    print(json.dumps(summary,indent=2),flush=True)
    print(result[['variable','difference','cluster_se','ci95_lower','ci95_upper','p_bh']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.source,args.output)
