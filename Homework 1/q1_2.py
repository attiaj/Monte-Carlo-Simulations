import numpy as np
import matplotlib.pyplot as plt

N = np.arange(1,500,1)

xn = []
for nx in N:
    ex = np.random.uniform(0,1)
    x = -np.log(ex)
    
    yn = []
    for ny in N:
        y = np.random.uniform(0,x,ny)
        ey = np.exp(-y)
        inty = sum(ey)/ny
        yn.append(inty)
    
    intx = sum(yn)/nx
    xn.append(intx)

plt.plot(N,xn)
plt.gca().set_ylim([0,2])
plt.show()