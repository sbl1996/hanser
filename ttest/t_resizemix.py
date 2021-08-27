import numpy as np
import matplotlib.pyplot as plt


xs1 = np.random.uniform(0.1, 0.8, size=(100000,))
ys1 = xs1 ** 2
plt.hist(ys1, bins=100)

xs2 = np.random.uniform(0.2, 0.8, size=(100000,))
ys2 = xs2 ** 2
plt.hist(ys2, bins=100)