import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# desired peak
x_peak = 1.0
sigma = 0.9

# compute mu
mu = np.log(x_peak) + sigma**2

# create distribution
s = sigma  # 's' is shape parameter in scipy
scale = np.exp(mu)  # 'scale' is exp(mu)

# generate values
x = np.linspace(0.01, 30, 1000)
pdf = lognorm.pdf(x, s, scale=scale)

# plot
plt.plot(x, pdf)
plt.title("Log-normal distribution with peak at x = 10")
plt.xlabel("x")
plt.ylabel("Density")
plt.show()