import math
import numpy as np
from scipy.stats import binom
from scipy.optimize import minimize_scalar
from Optimizer import custmin
from Tulap import ptulap
import scipy
from pvalue import pvalLeft, pvalRight
from pvalTwoSide import pvalTwoSide

def CITwoSide(alpha, Z, size, b, q):
    mle = Z/size
    mle = max(min(mle, 1), 0)
    #TODO: chage pvalRight to pvalTwoSide
    CIobj2 = lambda x: (pvalTwoSide(Z=Z, theta=x, size=size, b=b, q=q) - alpha)

    if mle > 0:
        L = minimize_scalar(fun=CIobj2, method=custmin, bracket=(0, mle)) #, args=mle/2
        L = L.x
    else:
        L = 0
    
    if mle < 1:
        U = minimize_scalar(fun=CIobj2, method=custmin, bracket=(mle, 1)) #, args=((1-mle)/2)
        U = U.x
    else:
        U = 1
    
    CI = [L, U]
    return CI