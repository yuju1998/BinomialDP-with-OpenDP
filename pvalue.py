import math
import numpy as np
from scipy.stats import binom
from Tulap import ptulap
import scipy

#' @title Calculating UMP One-Sided p-values
#' @name pvalueOneSide
#' @aliases pval
#' @aliases pvalleft
#' @aliases pvalright

r'''
Description: Calculating UMP one-sided p-values for binomial data under \eqn{(\epsilon, \delta)}-DP
Parameters:
    Z: A Binomial sample with Tulap noise
    size: The number of trials in Binomial distribution (parameter n in Binomial(n, \eqn{\theta}))
    theta: The success probability for each trial (parameter \eqn{\theta} in Binomial(n, \eqn{\theta}))
    b: Discrete Laplace noise parameters, obtained by \eqn{exp(-\epsilon)}
    q: The truncated quantiles
Return: A vector of one-sided p-values
See Also: Asymptotically unbiased DP two-sided p-values (\code{\link{pvalTwoSide}})
References: Awan, Jordan Alexander, and Aleksandra Slavkovic. 2020. "Differentially Private Inference for Binomial Data". Journal of Privacy and Confidentiality 10 (1). \url{https://doi.org/10.29012/jpc.725}.
Examples:
    set.seed(2020)
    sample <- rbinom(1, 10, 0.2) + rtulap(1, 0, 0.3, 0.05)
    pvalLeft(sample, size = 10, theta = 0.5,b = 0.3, q = 0.05)
    pvalRight(sample, size = 10, theta = 0.5,b = 0.3, q = 0.05)
'''



def pvalRight(Z, size, theta, b, q):
    reps = Z.size
    if reps > 1:
        pval = [0] * reps
        values = np.array(range(size))

        B = binom.pmf(k=values, n=size, p=theta)

        for r in range(reps):
            F = ptulap(t=values-Z[r], m=0, b=b, q=q)
            pval[r] = np.dot(F.T, B)
        return pval
    else:
        pval = [0]
        values = np.array(range(size))
        B = binom.pmf(k=values, n=size, p=theta)
        F = ptulap(t=values-Z, m=0, b=b, q=q)
        pval[0] = np.dot(F.T, B)
        return pval[0]

def pvalLeft(Z, size, theta, b, q):
    reps = Z.size
    if reps > 1:
        pval = [0] * reps
        values = range(size+1)

        B = binom.pmf(k=values, n=size, p=theta)

        for r in range(reps):
            F = 1 - ptulap(t=values-Z[r], m=0, b=b, q=q)
            pval[r] = np.dot(F.T, B)
        
        return pval

    else:
        pval = [0]
        values = range(size+1)

        B = binom.pmf(k=values, n=size, p=theta)
        F = 1 - ptulap(t=values-Z, m=0, b=b, q=q)
        pval[0] = np.dot(F.T, B)

        return pval[0]

