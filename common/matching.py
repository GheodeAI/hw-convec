"""Compound/non-compound regional event matching (Mahalanobis, caliper 0.75)."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

COV=['center_latitude','center_longitude','duration','intensity','log_max_daily_land_area']
NAMES=['Centre latitude','Centre longitude','Duration','Intensity','log(peak daily land area)','Peak daily land area (raw)']


def match(e,mode,base_cov=None):
    """1:1 greedy matching without replacement in global ascending distance order.

    The covariance is that of the full eligible sample. `mode` sets the exact
    strata: 'exact_month' (primary), 'exact_year_month' or 'within_one_year'.
    """
    assert mode in ['exact_year_month','within_one_year','exact_month']
    cov=list(COV if base_cov is None else base_cov)+(['year'] if mode=='within_one_year' else [])
    covariance=np.cov(e[cov].to_numpy(),rowvar=False);vi=np.linalg.pinv(covariance)
    edges=[]
    groups=['year','start_month'] if mode=='exact_year_month' else ['start_month']
    for _,g in e.groupby(groups):
        a=g[g.has_lightning];b=g[~g.has_lightning]
        d=cdist(a[cov],b[cov],metric='mahalanobis',VI=vi)
        allowed=d<=.75
        if mode=='within_one_year':allowed &= abs(a.year.values[:,None]-b.year.values[None,:])<=1
        i,j=np.nonzero(allowed)
        edges.extend(zip(d[i,j].tolist(),a.index.values[i].tolist(),b.index.values[j].tolist()))
    used_c=set();used_n=set();rows=[]
    for distance,c,n in sorted(edges):
        if c in used_c or n in used_n:continue
        used_c.add(c);used_n.add(n)
        rows.append(dict(match_set_id=len(rows)+1,compound_event_id=c,control_event_id=n,mahalanobis_distance=distance))
    pairs=pd.DataFrame(rows)
    assert not pairs.compound_event_id.duplicated().any() and not pairs.control_event_id.duplicated().any()
    assert (pairs.mahalanobis_distance<=.75).all()
    return pairs,cov,covariance
