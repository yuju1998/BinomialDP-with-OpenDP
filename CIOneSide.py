import math
import numpy as np
from scipy.stats import binom
from scipy.optimize import minimize, minimize_scalar
from Tulap import ptulap
import scipy
from pvalue import pvalLeft, pvalRight
from Optimizer import custmin

def CIobjLower(x, alpha, Z, size, b, q):
    obj = (np.array(pvalRight(Z=Z, size=size, theta=x, b=b, q=q)) - alpha) ** 2
    return obj

def CIobjUpper(x, alpha, Z, size, b, q):
    obj = (np.array(pvalRight(Z=Z, size=size, theta=x, b=b, q=q)) - (1-alpha)) ** 2
    return obj

def CILower(alpha, Z, size, b, q):
    CIobj = lambda x: ((pvalRight(Z=Z, size=size, theta=x, b=b, q=q)) - alpha) ** 2
    L = minimize_scalar(fun=CIobj, method=custmin, bracket=(0, 1))    # args already set in CIobj
    return L.x

def CIUpper(alpha, Z, size, b, q):
    CIobj = lambda x: ((pvalRight(Z=Z, size=size, theta=x, b=b, q=q)) - (1-alpha)) ** 2
    U = minimize_scalar(fun=CIobj, method=custmin, bracket=(0, 1))
    return U.x