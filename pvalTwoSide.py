import math
import numpy as np
from scipy.stats import binom
from scipy.optimize import minimize_scalar
from Tulap import ptulap
import scipy
from pvalue import pvalLeft, pvalRight


def pvalTwoSide(Z, size, theta, b, q):
    T = abs(Z - size * theta)
    pval = np.subtract(pvalRight(Z=T+size*theta, size=size, theta=theta, b=b, q=q), pvalRight(Z=size*theta-T, size=size, theta=theta, b=b, q=q))

    return pval+1