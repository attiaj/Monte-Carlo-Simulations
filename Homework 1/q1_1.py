import numpy as np
import matplotlib.pyplot as plt

N = np.arange(0,1000,1)

xn = []
for n in N:
    x = np.random.uniform(0,1,n)
    y = (1-x**2)**(3/2)
    int = sum(y)/n
    xn.append(int)

plt.plot(N,xn)
plt.show()