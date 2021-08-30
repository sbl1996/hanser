import numpy as np
import matplotlib.pyplot as plt

n = 50000

for b in [0.5, 0.6, 0.7, 0.8]:
    xs = np.random.uniform(0.1, b, size=(n,))
    ys = xs ** 2
    plt.hist(ys, bins=100, density=True)

xs = np.random.beta(0.4, 0.4, size=(n * 2,))
xs = xs[(xs > 0.01) & (xs < 0.5)]
plt.hist(xs, bins=100, density=True)

plt.legend([0.5, 0.6, 0.7, 0.8, "beta 0.4"])