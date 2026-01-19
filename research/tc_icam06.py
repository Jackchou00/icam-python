import matplotlib.pyplot as plt
import numpy as np

input = np.logspace(-1, 10, 100000)
p = 0.7
white = 1
L_A = 0.2 * white
k = 1.0 / (5 * L_A + 1)
F_L = 0.2 * k**4 * (5 * L_A) + 0.1 * (1 - k**4) ** 2 * (5 * L_A) ** (1 / 3)

R_a = 400 * (F_L * input / white) ** p / (27.13 + (F_L * input / white) ** p) + 0.1


fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].plot(input, R_a)
axs[0].set_ylim(0, 400)
axs[0].set_xscale("log")
axs[0].set_title("Log Scale")

axs[1].plot(input, R_a)
axs[1].set_ylim(0, 400)
axs[1].set_title("Linear Scale")

axs[2].plot(input, R_a)
axs[2].set_ylim(0, 400)
axs[2].set_xlim(0, 10000)


plt.show()
