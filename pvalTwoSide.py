import math
import numpy as np
from scipy.stats import binom
from scipy.optimize import minimize_scalar
from Tulap import ptulap
import scipy
from pvalue import pvalLeft, pvalRight

def pvalTulap(Z, size, theta, b, q):
    reps = len(Z)
    pval = [0] * reps
    values = range(0, size+1)

    B = binom.pmf(k=values, n=size, p=theta)

    for r in range(reps):
        F = ptulap(t=values-Z[r], m=0, b=b, q=q)
        pval[r] = np.dot(F.T, B)
    return pval


def pvalTwoSide(Z, size, theta, b, q):
    T = abs(Z - size * theta)
    pval = np.subtract(pvalTulap(Z=T+size*theta, size=size, theta=theta, b=b, q=q), pvalTulap(Z=size*theta-T, size=size, theta=theta, b=b, q=q))
    pval = list(pval)
    pval = [x + 1 for x in pval]
    return pval