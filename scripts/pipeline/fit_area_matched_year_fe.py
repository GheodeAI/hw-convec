"""Month/area matching, retained-event diagnostics and pair+year FE OLS."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.environment import ORDER
from common.io import dump
from common.matching import COV, match
from common.stats import bh, fit_difference


def selection(c,kept,out):
    r=c.loc[kept];x=c.loc[~c.index.isin(kept)]
    variables=COV+['max_daily_land_area_km2'];rows=[]
    for v in variables:
        scale=np.sqrt((r[v].var()+x[v].var())/2)
        rows.append(dict(variable=v,retained_n=len(r),excluded_n=len(x),retained_mean=r[v].mean(),excluded_mean=x[v].mean(),
                         retained_median=r[v].median(),excluded_median=x[v].median(),
                         smd_retained_minus_excluded=(r[v].mean()-x[v].mean())/scale,
                         ks_distance=stats.ks_2samp(r[v],x[v]).statistic))
    tab=pd.DataFrame(rows);tab.to_csv(out/'tables/retained_vs_excluded_compound.csv',index=False)
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
    pd.DataFrame(bands).to_csv(out/'tables/retention_geographic_bands.csv',index=False)
    return tab


def run(source,out):
    out.mkdir(parents=True,exist_ok=False)
    (out/'tables').mkdir()
    (out/'data').mkdir()
    sources=[source/'events/regional_events.csv',source/'matching/comparison/area_only_pairs.csv',source/'environment/tables/event_environment_anomaly.csv']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    e=pd.read_csv(source/'matching/diagnostics/tables/events_with_peak_daily_land_area.csv')
    e=e[e.start_month.between(4,9)&(e.cg_coverage_fraction==1)].set_index('event_id')
    pairs,cov,covariance=match(e,'exact_month',base_cov=COV)
    old=pd.read_csv(source/'matching/comparison/area_only_pairs.csv')
    assert pairs[['compound_event_id','control_event_id']].equals(old[['compound_event_id','control_event_id']])
    pairs.to_csv(out/'tables/match_pairs.csv',index=False)
    a=e.loc[pairs.compound_event_id];b=e.loc[pairs.control_event_id];c=e[e.has_lightning];n=e[~e.has_lightning]
    assert (a.start_month.values==b.start_month.values).all()
    rows=[]
    for v in COV+['max_daily_land_area_km2']:
        sd=np.sqrt((c[v].var()+n[v].var())/2)
        rows.append(dict(variable=v,smd_before=(c[v].mean()-n[v].mean())/sd,smd_after=(a[v].mean()-b[v].mean())/sd))
    balance=pd.DataFrame(rows);balance.to_csv(out/'tables/match_balance.csv',index=False)
    assert (balance.iloc[:5].smd_after.abs()<.1).all(), 'Five-variable balance check failed; stop before regression'
    print('Five-variable balance passed; pairs',len(pairs),flush=True)
    selected=selection(c,a.index,out)
    aa=a.reset_index();bb=b.reset_index();aa['compound']=1;bb['compound']=0
    aa['match_set_id']=pairs.match_set_id.values;bb['match_set_id']=pairs.match_set_id.values
    env=pd.read_csv(source/'environment/tables/event_environment_anomaly.csv').set_index('event_id')
    long=pd.concat([aa,bb],ignore_index=True).merge(env[ORDER],left_on='event_id',right_index=True,validate='one_to_one')
    long.to_csv(out/'tables/matched_events_model_input.csv',index=False)
    year_rows=[]
    for year in range(2006,2026):
        pc=(c.year==year).mean();pn=(n.year==year).mean();denom=np.sqrt((pc*(1-pc)+pn*(1-pn))/2)
        year_rows.append(dict(year=year,compound_n=int((a.year==year).sum()),noncompound_n=int((b.year==year).sum()),
                              smd_after=((a.year==year).mean()-(b.year==year).mean())/denom))
    yt=pd.DataFrame(year_rows);yt.to_csv(out/'tables/matched_year_balance.csv',index=False)
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
    result=pd.DataFrame(results);result['p_bh']=bh(result.p);result.to_csv(out/'tables/environment_pair_year_FE.csv',index=False)
    pd.DataFrame(coefficients).to_csv(out/'tables/all_year_FE_coefficients.csv',index=False)
    np.savez_compressed(out/'data/regression_arrays.npz',design_difference=D,outcomes_difference=Y,coefficients=beta,cluster_covariances=vcov)
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
    (out/'README.md').write_text(f'''# Exact-month, area-aware matching with pair and start-year fixed effects

{g} matched pairs from the area_only design: exact start month; Mahalanobis distance <= 0.75 on centre latitude, centre longitude, duration, intensity and ln(peak daily land area); 1:1 without replacement; global ascending-distance greedy assignment. Start year is not used in matching. The five-covariate SMD check runs before the regression.

Model: Y_ij = alpha_pair + gamma_startyear + beta * Compound_ij + error_ij, fitted separately for each environmental variable. Start month is identical within pairs and absorbed by the pair effects; 2006 is the reference year. Y is the event-mean anomaly over the event's land footprint and duration.

The pair effects are removed exactly by taking compound minus non-compound differences; the design is an intercept plus 19 differenced year indicators, and the intercept equals beta. Coefficients and cluster standard errors are verified against the explicit 2G-row dummy-variable OLS.

Standard errors are CR1 clustered by matched pair, with factor G/(G-1) x (N-1)/(N-K), N = 2G, K = G + 20 (pair effects counted), and t(G-1) intervals. No further clustering by year or date. Per-variable intervals are unadjusted; BH-adjusted p values across the six variables are reported alongside.

Retention diagnostics compare retained with excluded compound events among the {len(c)} eligible compound events only, using pooled SDs, medians, KS distances and fixed latitude/longitude bands (retention_geographic_bands.csv). Year adjustment does not remove selection differences introduced by matching.
''')
    print(json.dumps(summary,indent=2),flush=True)
    print(result[['variable','difference','cluster_se','ci95_lower','ci95_upper','p_bh']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.source,args.output)
