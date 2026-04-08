import math
import matplotlib.pyplot as plt
import env
import numpy as np

env = env.ENV()
x = range(env.n_tus)
y = []

tu_state = np.zeros(env.n_tus)
numerator = 0
denominator = 0
for n in range(env.n_tus):
    for i in range(env.n_tus):
        numerator += tu_state[i]
        denominator += (tu_state[i])**2
    if denominator != 0:
        y.append(numerator**2/denominator)
    elif denominator == 0:
        y.append(0)
    tu_state[n]=1
plt.figure(3,figsize=(7.125,5.6))
plt.plot(x, y, 'd-', color='r', linewidth=3, markersize='10')
plt.show()