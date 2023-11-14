

import numpy as np
import matplotlib.pyplot as plt

c0 = np.genfromtxt('checkpoint_OG2_8000ep_c0.csv', delimiter=',', skip_header=1)[:,2:]
c1 = np.genfromtxt('checkpoint_OG2_8000ep_c1.csv', delimiter=',', skip_header=1)[:,2:]

if False:
    for i, c in enumerate(c0):
        plt.plot(c, alpha=.01, color='C0')
        if i == 50:
            break
    for i, c in enumerate(c1):
        plt.plot(c, alpha=.01, color='C1')
        if i == 50:
            break

c0_avg=np.mean(c0, axis=0)
c1_avg=np.mean(c1, axis=0)

plt.plot(c0_avg, alpha=.5, color='C0')
plt.plot(c1_avg, alpha=.5, color='C1')
plt.show()
