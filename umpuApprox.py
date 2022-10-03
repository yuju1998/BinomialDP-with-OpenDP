import math
import numpy as np
from scipy.stats import binom
from Tulap import ptulap
import scipy

class umpuApprox:
    def __init__(self, theta, size, alpha, epsilon, delta):
        self.theta = theta
        self.size = size
        self.alpha = alpha
        self.epsilon = epsilon
        self.delta = delta

        self.b = math.exp(-epsilon)
        self.q = 2 * delta * self.b / (1-self.b+2*delta*self.b)
        self.values = np.array(range(0, size+1))
        self.B = binom.pmf(k=self.values, n=size, p=theta)
        self.BX = self.B * (self.values - size*theta)
        self.k = size * theta
        self.greaterK = np.greater_equal(self.values, self.k)

    ### Search over k for unbiasedness
    def obj(self, s):
        F1 = ptulap(t=self.values - self.k - s, m=0, b=self.b, q=self.q)
        F2 = ptulap(t=self.k - self.values - s, m=0, b=self.b, q=self.q)
        phi = F1 * self.greaterK + F2 * (1-self.greaterK)
        return np.dot(self.B, phi) - self.alpha
    
    def numpu(self):
        lower = -1
        upper = 1
        while self.obj(lower) < 0:
            lower *= 2
        while self.obj(upper) > 0:
            upper *= 2

        result = scipy.optimize.brentq(self.obj, lower, upper)  # scipy.optimize.brentq(function, min, max)
        s = result

        F1 = ptulap(t=self.values - self.k - s, m=0, b=self.b, q=self.q)
        F2 = ptulap(t=self.k - self.values - s, m=0, b=self.b, q=self.q)
        phi = F1*self.greaterK + F2*(1-self.greaterK)
        return phi
