import math
import numpy as np
from scipy.stats import binom
from Tulap import ptulap
import scipy

class UMPU:
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
        self.s = 0

    ### Search over s for alpha level
    def miniObj(self, s, k, greaterK):
        F1 = ptulap(t=self.values - k - s, m=0, b=self.b, q=self.q)
        F2 = ptulap(t=k - self.values - s, m=0, b=self.b, q=self.q)
        phi = F1 * greaterK + F2 * (1-greaterK)
        return np.dot(self.B, phi) - self.alpha

    ### Search over k for unbiasedness
    def obj(self, k):
        greaterK = np.greater_equal(self.values, k)
        lower = -1
        upper = 1
        while self.miniObj(s=lower, k=k, greaterK=greaterK) < 0:
            lower *= 2
        while self.miniObj(s=upper, k=k, greaterK=greaterK) > 0:
            upper *= 2
        miniResult = scipy.optimize.brentq(self.miniObj, lower, upper, args=(k, greaterK)) # scipy.optimize.brentq(function, min, max)
        s = miniResult #root
       
        F1 = ptulap(t=self.values - k - s, m=0, b=self.b, q=self.q)
        F2 = ptulap(t=k - self.values - s, m=0, b=self.b, q=self.q)
        phi = F1 * greaterK + F2 * (1-greaterK)
        return np.dot(self.BX, phi)
    
    def umpu(self):
        lower = -self.size
        upper = 2 * self.size
        while self.obj(lower) < 0:
            lower *= 2
        while self.obj(upper) > 0:
            upper *= 2
        result = scipy.optimize.brentq(self.obj, lower, upper)  # scipy.optimize.brentq(function, min, max)
        k = result #root
        greaterK = np.greater_equal(self.values, k)

        lower = -1
        upper = 1
        while self.miniObj(s=lower, k=k, greaterK=greaterK) < 0:
            lower *= 2
        while self.miniObj(s=upper, k=k, greaterK=greaterK) > 0:
            upper *= 2
        miniResult = scipy.optimize.brentq(self.miniObj, lower, upper, args=(k, greaterK))
        s = miniResult

        F1 = ptulap(t=self.values - k - s, m=0, b=self.b, q=self.q)
        F2 = ptulap(t=k - self.values - s, m=0, b=self.b, q=self.q)
        phi = F1 * greaterK + F2 * (1-greaterK)
        return phi
        
    