"""Statistical helpers shared across analyses."""
import numpy as np


def bh(p):
    """Benjamini-Hochberg adjusted p-values; NaNs are passed through."""
    p=np.asarray(p); result=np.full(len(p),np.nan); valid=np.flatnonzero(np.isfinite(p)); order=valid[np.argsort(p[valid])]
    result[order]=np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1].clip(0,1)
    return result


def fit_difference(D,Y):
    """Exact pair-FE elimination; full-level-model CR1 finite-sample factor."""
    g=len(D);rank=np.linalg.matrix_rank(D)
    assert rank==D.shape[1]
    bread=np.linalg.inv(D.T@D);coef=bread@D.T@Y;resid=Y-D@coef
    # Full model has 2G rows and G pair intercepts plus rank(D) coefficients.
    correction=(g/(g-1))*((2*g-1)/(g-rank))
    cov=[]
    for j in range(Y.shape[1]):
        score=D*resid[:,j,None]
        cov.append(correction*bread@(score.T@score)@bread)
    return coef,np.asarray(cov),resid,correction
