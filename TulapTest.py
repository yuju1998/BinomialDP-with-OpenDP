import random
from Tulap import rTulap
import math
import matplotlib.pyplot as plt
from scipy.stats import kde
import seaborn as sns
import numpy as np
import numpy.random as random
random.seed(100)
rand = rTulap(n=10000, m=30, b=math.exp(-1), q=0.06)
density = kde.gaussian_kde(rand)
x = np.arange(24, 36, 0.1)
plt.plot(x, density(x))
plt.xticks(np.arange(24, 36, 1))
plt.show()

