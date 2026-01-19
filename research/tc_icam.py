'''
the nonlinear compression in icam
'''


import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 1000, 1000)

La = 0.2 * x
k = 1.0 / (5 * La + 1)
FL = 0.2 * k**4 * (5 * La) + 0.1 * (1 - k**4) ** 2 * (5 * La) ** (1 / 3)
FL[FL < 0.3] = 0.3


y = x ** (FL)

plt.plot(x, y, label="Forward")
# plt.xscale("log")
plt.show()