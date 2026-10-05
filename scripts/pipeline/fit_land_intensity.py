#!/usr/bin/env python3
"""Categorical and ordered quintile FE logits with event-cluster inference."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import YEARS
from common.io import dump

def blocks(g,z,w):
    ng=int(g.max())+1
    a=np.bincount(g,weights=w,minlength=ng)
    membership=sparse.csr_matrix((np.ones(len(g)),(np.arange(len(g)),g)),shape=(len(g),ng))
    b=(membership.T@z.multiply(w[:,None])).toarray()
    c=(z.T@z.multiply(w[:,None])).toarray()
    schur=c-b.T@(b/a[:,None])
    return a,b,schur

def fit(g,z,y):
    n=np.bincount(g); s=np.bincount(g,weights=y)
    alpha=np.log(s/(n-s)); coef=np.zeros(z.shape[1]); trace=[]
    def ll(a,b):
        eta=a[g]+z@b
        return float(np.sum(y*eta-np.logaddexp(0,eta)))
    current=ll(alpha,coef)
    for it in range(60):
        p=expit(alpha[g]+z@coef); w=p*(1-p); resid=y-p
        a,b,schur=blocks(g,z,w)
        sg=np.bincount(g,weights=resid,minlength=len(alpha))
        db=np.linalg.solve(schur,np.asarray(z.T@resid).ravel()-b.T@(sg/a))
        da=(sg-b@db)/a
        step=1.
        while ll(alpha+step*da,coef+step*db)<current-1e-7:
            step/=2
            if step<2**-20: raise RuntimeError('Intensity FE line search failed')
        alpha+=step*da; coef+=step*db; current=ll(alpha,coef)
        maxstep=max(abs(da).max(),abs(db).max())*step
        trace.append(dict(iteration=it+1,log_likelihood=current,max_step=maxstep))
        print('Intensity IRLS',it+1,current,maxstep,flush=True)
        if maxstep<1e-7: break
    else: raise RuntimeError('Intensity FE failed convergence')
    p=expit(alpha[g]+z@coef); a,b,schur=blocks(g,z,p*(1-p))
    return alpha,coef,a,b,schur,y-p,trace

def influence(g,z,a,b,schur,resid,grad_a,grad_b):
    v_b=np.linalg.solve(schur,grad_b-b.T@(grad_a/a[:,None]))
    v_a=(grad_a-b@v_b)/a[:,None]
    return (v_a[g]+z@v_b)*resid[:,None]

def main(out):
    events=pd.read_csv(out/'tables/regional_events.csv').set_index('event_id')
    chunks=[]
    for year in YEARS:
        with np.load(out/f'data/daily_{year}.npz') as d:
            dates=pd.DatetimeIndex(d['dates']); h=d['hw']; ex=d['excess']; lab=d['event']
            valid=h & (ex>0) & np.isfinite(ex) & (lab>0)
            valid&=((dates.month>=4)&(dates.month<=9))[:,None]
            t,c=np.nonzero(valid)
            chunks.append(pd.DataFrame(dict(cell=c,date=dates.values[t],month=dates.month.values[t],year=year,
                         exposure=ex[t,c],cg=d['cg'][t,c],event_id=lab[t,c]+int(d['offset']))))
    df=pd.concat(chunks,ignore_index=True)
    edges=np.quantile(df.exposure,[0,.2,.4,.6,.8,1])
    df['q']=np.searchsorted(edges[1:-1],df.exposure,side='left')+1
    df['duration']=events.loc[df.event_id,'duration'].to_numpy()
    df.to_parquet(out/'data/intensity_sample.parquet',index=False)
    raw=df.groupby('q').agg(n=('cg','size'),cg_days=('cg','sum'),p_cg=('cg','mean'),mean_excess=('exposure','mean')).reset_index()
    raw.to_csv(out/'tables/quintile_raw.csv',index=False)
    strata=df.cell.to_numpy()*12+df.month.to_numpy()
    _,gi=np.unique(strata,return_inverse=True)
    n=np.bincount(gi); s=np.bincount(gi,weights=df.cg)
    keep=(s>0)&(s<n)
    retain=keep[gi]
    audit=dict(original_n=len(df),retained_n=int(retain.sum()),dropped_n=int((~retain).sum()),
               original_strata=len(n),retained_strata=int(keep.sum()),quintile_edges=edges.tolist(),
               sample='CG land centres mapped to land ERA5 regional heatwave cells; April–September; positive Tw minus combined threshold',
               exclusion='drop cell-month strata with CG constant; omit non-positive excess (gap days) from quintiles',
               original_scripts='absent; categorical quintile and ordered FE reimplemented from output metadata')
    df=df.loc[retain].copy(); _,g=np.unique(strata[retain],return_inverse=True)
    retained_raw=df.groupby('q').agg(n=('cg','size'),cg_days=('cg','sum'),p_cg=('cg','mean')).reset_index()
    retained_raw.to_csv(out/'tables/quintile_raw_retained.csv',index=False)
    coverage=df.assign(stratum=g).groupby('stratum').q.nunique()
    coverage.value_counts().sort_index().rename_axis('n_quintiles').rename('n_strata').to_csv(out/'checks/quintile_stratum_coverage.csv')
    q=df.q.to_numpy(); year=df.year.to_numpy()-2006; duration=df.duration.to_numpy(float)
    dm,ds=duration.mean(),duration.std(); duration=(duration-dm)/ds
    nobs=len(df); row=np.arange(nobs)
    def design(ordered=False):
        kq=1 if ordered else 4
        r=[row[year>0],row]; c=[year[year>0]-1,np.full(nobs,19)]; v=[np.ones(np.sum(year>0)),duration]
        if ordered: r.append(row); c.append(np.full(nobs,20)); v.append(q.astype(float)-1)
        else: r.append(row[q>1]); c.append(20+q[q>1]-2); v.append(np.ones(np.sum(q>1)))
        return sparse.csr_matrix((np.concatenate(v),(np.concatenate(r),np.concatenate(c))),shape=(nobs,20+kq))
    z=design(); y=df.cg.to_numpy(float)
    alpha,coef,a,b,schur,resid,trace=fit(g,z,y)
    event_codes,eg=np.unique(df.event_id,return_inverse=True)
    # Five standardization margins, and four categorical log odds coefficients.
    ga=np.zeros((len(a),9)); gb=np.zeros((len(coef),9)); margins=[]
    observed_q=np.where(q==1,0,coef[20+np.maximum(q-2,0)])
    base=alpha[g]+z@coef-observed_q
    nuisance=z[:,:20]
    for j in range(5):
        p=expit(base+(0 if j==0 else coef[19+j])); margins.append(p.mean())
        derivative=p*(1-p)/nobs
        ga[:,j]=np.bincount(g,weights=derivative,minlength=len(a))
        gb[:20,j]=np.asarray(nuisance.T@derivative).ravel()
        if j: gb[19+j,j]=derivative.sum()
    gb[20:,5:]=np.eye(4)
    infl=influence(g,z,a,b,schur,resid,ga,gb)
    scores=np.column_stack([np.bincount(eg,weights=infl[:,j]) for j in range(9)])
    covariance=scores.T@scores*len(event_codes)/(len(event_codes)-1)
    se=np.sqrt(np.diag(covariance)); margins=np.asarray(margins)
    diff=margins[4]-margins[0]; dse=np.sqrt(covariance[4,4]+covariance[0,0]-2*covariance[4,0])
    result=pd.DataFrame(dict(q=np.arange(1,6),adjusted_probability=margins,se=se[:5],ci95_lower=margins-1.96*se[:5],ci95_upper=margins+1.96*se[:5]))
    result.to_csv(out/'tables/quintile_adjusted.csv',index=False)
    rng=np.random.default_rng(20260914)
    boot=np.empty((1000,5))
    for k in range(1000):
        multiplicity=rng.multinomial(len(event_codes),np.full(len(event_codes),1/len(event_codes)))
        boot[k]=margins+(multiplicity-1)@scores[:,:5]
    result['bootstrap_ci95_lower']=np.quantile(boot,.025,axis=0)
    result['bootstrap_ci95_upper']=np.quantile(boot,.975,axis=0)
    result.to_csv(out/'tables/quintile_adjusted.csv',index=False)
    np.save(out/'data/quintile_cluster_bootstrap.npy',boot)
    pd.DataFrame(dict(q=np.arange(2,6),log_odds=coef[20:],se=se[5:],odds_ratio=np.exp(coef[20:]),
                     or_ci95_lower=np.exp(coef[20:]-1.96*se[5:]),or_ci95_upper=np.exp(coef[20:]+1.96*se[5:]))).to_csv(out/'tables/quintile_coefficients.csv',index=False)
    pd.DataFrame(trace).to_csv(out/'checks/quintile_convergence.csv',index=False)
    del infl
    zo=design(True)
    ao,co,aa,bb,ss,rr,tr=fit(g,zo,y)
    grad_a=np.zeros((len(aa),1)); grad_b=np.zeros((len(co),1)); grad_b[-1]=1
    oi=influence(g,zo,aa,bb,ss,rr,grad_a,grad_b)[:,0]
    oscore=np.bincount(eg,weights=oi); ose=np.sqrt(np.sum(oscore**2)*len(event_codes)/(len(event_codes)-1))
    audit.update(duration_mean=dm,duration_sd=ds,regional_event_clusters=len(event_codes),
                 q5_minus_q1=diff,q5_minus_q1_ci95_lower=diff-1.96*dse,q5_minus_q1_ci95_upper=diff+1.96*dse,
                 ordered_beta=float(co[-1]),ordered_or=float(np.exp(co[-1])),ordered_beta_se=ose,
                 ordered_beta_ci95_lower=float(co[-1]-1.96*ose),ordered_beta_ci95_upper=float(co[-1]+1.96*ose),
                 ordered_p=float(2*stats.norm.sf(abs(co[-1]/ose))),
                 inference='regional-event CR1 cluster sandwich, all nuisance effects included in margin delta method; normal 95% CI')
    dump(out/'tables/quintile_diagnostics.json',audit)
    np.save(out/'data/quintile_margin_coefficient_covariance.npy',covariance)
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.plot(raw.q,100*raw.p_cg,'o--',label='Unadjusted (all eligible HW days)'); ax.plot(retained_raw.q,100*retained_raw.p_cg,'s:',label='Unadjusted (retained sample)'); ax.errorbar(np.arange(1,6),100*margins,yerr=196*se[:5],fmt='o-',capsize=4,label='FE-adjusted (retained sample)'); ax.set_xticks(range(1,6),[f'Q{i}' for i in range(1,6)]); ax.set_ylabel('CG occurrence probability (%)'); ax.set_xlabel('Land heatwave intensity quintile'); ax.legend(); fig.tight_layout(); fig.savefig(out/'figures/quintile_probability.png',dpi=200); plt.close(fig)
    print(audit,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); main(p.parse_args().out)
