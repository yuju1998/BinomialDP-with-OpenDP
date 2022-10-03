import math
import numpy as np
from scipy.stats import binom
from Tulap import ptulap
import scipy

class UMP:
    def __init__(self, theta, size, alpha, epsilon, delta):
        self.theta = theta
        self.size = size
        self.alpha = alpha
        self.epsilon = epsilon
        self.delta = delta

        self.b = math.exp(-epsilon)
        self.q = 2 * delta * self.b/(1-self.b+2*delta*self.b)
        self.values = list(range(0, size+1))
        self.B = binom.pmf(k=self.values, n=size, p=theta)



    def obj(self, s):
        values = np.array(self.values)
        phi = ptulap(t=values-s, m=0, b=self.b, q=self.q)
        return np.dot(self.B, phi) - self.alpha


    def umpLeft(self):
        lower = -1
        upper = 1

        while self.obj(lower) < 0:
            lower *= 2
        while self.obj(upper) > 0:
            upper *= 2
        root = scipy.optimize.brentq(self.obj, lower, upper)  # scipy.optimize.brentq(function, min, max)
        s = root
        values = np.array(self.values)
        phi = ptulap(t=values-s, m=0, b=self.b, q=self.q)
        return phi
    
    def umpRight(self):
        lower = -1
        upper = 1

        while self.obj(lower) < 0:
            lower *= 2
        while self.obj(upper) > 0:
            upper *= 2
        root = scipy.optimize.brentq(self.obj, lower, upper)  # scipy.optimize.brentq(function, min, max)
        s = root
        values = np.array(self.values)
        phi = ptulap(t=values-s, m=0, b=self.b, q=self.q)
        return 1-phi