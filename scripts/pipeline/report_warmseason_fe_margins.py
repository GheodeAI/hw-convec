"""Full/retained FE audit and joint two-way-cluster delta-method margins."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.analysis_inputs import input_path
from common.fe_logit import fit_absorbed_logit


def daily(source, year):
    with np.load(input_path(source, f'daily_{year}.npz')) as d:
        dates=pd.DatetimeIndex(d['dates'])
        keep=(dates.month>=4)&(dates.month<=9)
        return dates[keep],d['hw'][keep],d['cg'][keep]


def run(source):
    out=source/'occurrence'
    out.mkdir(parents=True, exist_ok=True)
    (out/'tables').mkdir(exist_ok=True)
    (out/'data').mkdir(exist_ok=True)
    if (out/'tables/adjusted_effects_95CI.csv').exists():
        raise FileExistsError(f'Probability report already exists: {out}')
    previous=pd.read_csv(next((source/'occurrence/tables').glob('cg_hw_fe_logit_*_summary.csv'))).iloc[0]
    nc=int(previous.land_cells); ng=nc*6; ny=20
    counts=np.zeros((ng,ny,2),dtype=np.uint16); successes=np.zeros_like(counts)
    cells=np.arange(nc)
    for yi,year in enumerate(range(2006,2026)):
        dates,hw,cg=daily(source,year)
        for mi,month in enumerate(range(4,10)):
            h=hw[dates.month==month]; y=cg[dates.month==month].astype(bool); g=cells*6+mi
            counts[g,yi,1]=h.sum(0); counts[g,yi,0]=len(h)-h.sum(0)
            successes[g,yi,1]=(y & h).sum(0); successes[g,yi,0]=(y & ~h).sum(0)
        print('Counted',year,flush=True)
    n=counts.sum((1,2)); nh=counts[:,:,1].sum(1); s=successes.sum((1,2))
    exposure=(nh>0)&(nh<n); outcome=(s>0)&(s<n); retained=exposure&outcome
    assert int(n.sum())==previous.total_sample_records
    assert int(n[retained].sum())==previous.retained_sample_records
    assert int(nh.sum())==previous.total_heatwave_records
    assert int(retained.sum())==previous.retained_cell_month_strata
    c=counts[retained]; sy=successes[retained]
    g,yr,h=np.nonzero(c); totals=c[g,yr,h].astype(float); yy=sy[g,yr,h].astype(float)
    h=h.astype(float); G=int(retained.sum()); N=int(c.sum())
    fit=fit_absorbed_logit(g,yr,h,totals,yy,G,ny,100,1e-9)
    assert fit.converged and np.isclose(fit.beta,previous.heatwave_logit_coefficient,atol=1e-9)
    # Hessian blocks for alpha (cell-month) and b=(19 year effects, beta).
    w=fit.weights
    A=np.bincount(g,weights=w,minlength=G)
    Wgy=np.bincount(g*ny+yr,weights=w,minlength=G*ny).reshape(G,ny)
    B=np.column_stack([Wgy[:,1:],np.bincount(g,weights=w*h,minlength=G)])
    C=np.zeros((20,20)); C[:19,:19]=np.diag(Wgy[:,1:].sum(0))
    C[:19,19]=np.bincount(yr,weights=w*h,minlength=ny)[1:]; C[19,:19]=C[:19,19]
    C[19,19]=np.sum(w*h*h)
    schur=C-B.T@(B/A[:,None])
    weights=c.sum(2).astype(float)/N; gamma=np.r_[0.,fit.year_effects]
    ga=np.zeros((G,3)); gb=np.zeros((20,3)); margins=[]
    for j in [0,1]:
        p=expit(fit.alpha[:,None]+gamma[None,:]+j*fit.beta)
        margins.append(float(np.sum(weights*p)))
        deriv=weights*p*(1-p); ga[:,j]=deriv.sum(1)
        gb[:19,j]=deriv.sum(0)[1:]; gb[19,j]=j*deriv.sum()
    gb[19,2]=1.
    vb=np.linalg.solve(schur,gb-B.T@(ga/A[:,None]))
    va=(ga-B@vb)/A[:,None]
    assert np.allclose(A[:,None]*va+B@vb,ga,atol=1e-10)
    assert np.allclose(B.T@va+C@vb,gb,atol=1e-10)
    assert np.allclose(margins,[previous.adjusted_p_cg_nonheatwave,previous.adjusted_p_cg_heatwave],atol=1e-10)
    mapping=np.full(ng,-1,dtype=int); mapping[retained]=np.arange(G)
    cell_scores=np.zeros((nc,3)); date_scores=[]; intersection=np.zeros((3,3))
    for yi,year in enumerate(range(2006,2026)):
        dates,hw,cg=daily(source,year)
        for mi,month in enumerate(range(4,10)):
            keep=retained[cells*6+mi]; fi=mapping[cells[keep]*6+mi]
            mask=dates.month==month
            hday=hw[mask][:,keep].astype(float); yday=cg[mask][:,keep].astype(float)
            residual=yday-expit(fit.alpha[fi][None,:]+gamma[yi]+fit.beta*hday)
            vy=vb[yi-1] if yi>0 else np.zeros(3)
            infl=(va[fi][None,:,:]+vy[None,None,:]+hday[:,:,None]*vb[19])*residual[:,:,None]
            cell_scores[keep]+=infl.sum(0); date_scores.append(infl.sum(1))
            flat=infl.reshape(-1,3); intersection+=flat.T@flat
        print('Cluster influences',year,flush=True)
    ds=np.vstack(date_scores); active=retained.reshape(nc,6).any(1)
    common=(N-1)/(N-(G+20)); gc=int(active.sum()); gd=len(ds)
    cov_cell=common*gc/(gc-1)*(cell_scores[active].T@cell_scores[active])
    cov_date=common*gd/(gd-1)*(ds.T@ds)
    cov_intersection=common*N/(N-1)*intersection
    covariance=cov_cell+cov_date-cov_intersection
    assert np.all(np.linalg.eigvalsh(covariance)>0)
    assert np.isclose(np.sqrt(covariance[2,2]),previous.heatwave_coefficient_two_way_cluster_se,rtol=1e-7)
    np.savez_compressed(out/'data/margin_cluster_covariance.npz',covariance=covariance,cell_component=cov_cell,date_component=cov_date,intersection_component=cov_intersection)
    z=norm.ppf(.975); p0,p1=margins; delta=p1-p0; rr=p1/p0
    gradients=[np.array([1.,0,0]),np.array([0.,1,0]),np.array([-1.,1,0]),np.array([-1/p0,1/p1,0]),np.array([0.,0,1.])]
    effects=[]
    for name,value,gradient,logscale in zip(['p_nonheatwave','p_heatwave','probability_difference','occurrence_ratio','odds_ratio'],[p0,p1,delta,rr,np.exp(fit.beta)],gradients,[False,False,False,True,True]):
        se=float(np.sqrt(gradient@covariance@gradient))
        lower,upper=(np.exp(np.log(value)+np.array([-1,1])*z*se) if logscale else value+np.array([-1,1])*z*se)
        effects.append(dict(metric=name,estimate=value,ci95_lower=lower,ci95_upper=upper,se_on_log_scale=se if logscale else np.nan,se=se*value if logscale else se))
    eff=pd.DataFrame(effects); eff.to_csv(out/'tables/adjusted_effects_95CI.csv',index=False)
    previous.to_frame('value').rename_axis('metric').to_csv(out/'tables/original_model_summary.csv')
    raw=[]
    for label,sel in [('Unadjusted — full sample',np.ones(ng,dtype=bool)),('Unadjusted — FE-informative sample',retained)]:
        totals2=counts[sel].sum((0,1)); succ2=successes[sel].sum((0,1)); rates=succ2/totals2
        for j in [0,1]:
            raw.append(dict(result=label,heatwave=j,n_records=int(totals2[j]),cg_records=int(succ2[j]),probability=rates[j],probability_difference=rates[1]-rates[0],occurrence_ratio=rates[1]/rates[0]))
    raw=pd.DataFrame(raw); raw.to_csv(out/'tables/unadjusted_probabilities.csv',index=False)
    # Mutually exclusive sequential exclusions, plus joint criteria to expose overlap.
    reasons=np.select([~exposure,exposure&~outcome],['no_heatwave_variation','no_CG_variation_after_exposure_filter'],default='retained')
    audit=pd.DataFrame(dict(reason=reasons,records=n,heatwave_records=nh,nonheatwave_records=n-nh)).groupby('reason').agg(strata=('records','size'),records=('records','sum'),heatwave_records=('heatwave_records','sum'),nonheatwave_records=('nonheatwave_records','sum'))
    audit.to_csv(out/'tables/sequential_attrition.csv')
    pd.DataFrame(dict(exposure_varies=exposure,outcome_varies=outcome,records=n)).groupby(['exposure_varies','outcome_varies']).agg(strata=('records','size'),records=('records','sum')).to_csv(out/'tables/joint_exclusion_diagnostics.csv')
    with np.load(input_path(source, 'grids.npz')) as grid:
        ilat,ilon=np.unravel_index(grid['cg_ids'],grid['cg_land'].shape)
        lat=grid['cg_latitude'][ilat]; lon=grid['cg_longitude'][ilon]
    strata=pd.DataFrame(dict(cell_id=np.repeat(cells,6),latitude=np.repeat(lat,6),longitude=np.repeat(lon,6),month=np.tile(np.arange(4,10),nc),original_records=n,retained_records=np.where(retained,n,0),original_heatwave_records=nh,retained_heatwave_records=np.where(retained,nh,0),retained=retained,reason=reasons))
    strata.to_csv(out/'tables/cell_month_sample_diagnostics.csv',index=False)
    month=strata.groupby('month').agg(original_strata=('retained','size'),retained_strata=('retained','sum'),original_records=('original_records','sum'),retained_records=('retained_records','sum'),original_heatwave_records=('original_heatwave_records','sum'),retained_heatwave_records=('retained_heatwave_records','sum'))
    month['retention_percent']=100*month.retained_records/month.original_records
    month['retained_sample_share_percent']=100*month.retained_records/N
    month.to_csv(out/'tables/sample_distribution_by_month.csv')
    spatial=strata.groupby(['cell_id','latitude','longitude']).agg(original_records=('original_records','sum'),retained_records=('retained_records','sum'),retained_month_strata=('retained','sum')).reset_index()
    spatial['retention_percent']=100*spatial.retained_records/spatial.original_records
    spatial.to_csv(out/'tables/sample_distribution_by_cell.csv',index=False)
    bands=[]
    for coord,edges in [('latitude',[35,45,55,65,75]),('longitude',[-15,0,15,30,45])]:
        for lo,hi in zip(edges[:-1],edges[1:]):
            sub=spatial[(spatial[coord]>=lo)&(spatial[coord]<hi)]
            original=int(sub.original_records.sum()); kept=int(sub.retained_records.sum())
            bands.append(dict(coordinate=coord,lower=lo,upper=hi,original_records=original,retained_records=kept,retention_percent=100*kept/original if original else np.nan,retained_sample_share_percent=100*kept/N))
        assert sum(r['original_records'] for r in bands if r['coordinate']==coord)==int(n.sum())
    pd.DataFrame(bands).to_csv(out/'tables/sample_distribution_by_geographic_band.csv',index=False)
    from plot_warmseason_probability_report import plot
    plot(out)
    metadata=dict(status='passed',original_records=int(n.sum()),retained_records=N,retained_fraction=N/int(n.sum()),original_strata=ng,retained_strata=G,cell_clusters=gc,date_clusters=gd,
        CI='Joint delta-method CGM two-way cell + UTC date - cell-date sandwich, full-model gradients including all cell-month and year effects; CR1 identical to original model. Probability and probability difference use normal-scale Wald intervals; RR and OR use log-scale Wald intervals.',
        standardization='Fixed empirical cell-month-year distribution of retained observations; uncertainty conditional on this retained sample and its selection. No extrapolation to excluded strata.',
        validation='Original counts, beta, margins and beta two-way SE reproduced; Hessian influence linear equations verified; joint covariance positive definite.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'README.md').write_text('# Warm-season FE probability report\n\n'+metadata['CI']+'\n\n'+metadata['standardization']+'\n\nAll outputs use original CG-grid daily caches, not the regional-event matching sample. Spatial distributions are descriptive coordinate bands and cell-level records, not administrative regions. Sequential exclusion counts do not double count; joint diagnostics additionally report overlapping exclusion conditions. No existing figures or original model outputs were overwritten.\n')
    print(eff.to_string(index=False),flush=True)
    print(month.to_string(),flush=True)
    print(pd.DataFrame(bands).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    run(parser.parse_args().source)
