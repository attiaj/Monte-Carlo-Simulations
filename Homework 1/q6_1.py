import numpy as np
import matplotlib.pyplot as plt

sims = 1000
Nmax = 1000

N = Nmax*np.ones(sims)
r = 0.05
sigma = 0.4
s = 100
mu = 0.2

u = np.exp(mu/Nmax + sigma*np.sqrt(1/Nmax))
d = 1/u

thresh = (np.exp(r/Nmax)-d)/(u-d)

sRatio = []
n = 0
while n < Nmax:
    p = np.random.uniform(0,1,Nmax)

    i = 0
    rProd = 1
    while i < Nmax:
        if p[i] <= thresh:
            rProd *= u
        else:
            rProd *= d
        i += 1

    sRatio.append(rProd)
    n += 1

logSRatio = np.log(sRatio)

print(np.mean(logSRatio),np.std(logSRatio))
print(r-(sigma**2)/2)
print(sigma)

plt.hist(logSRatio,100)
plt.show()