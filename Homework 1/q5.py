import numpy as np
import matplotlib.pyplot as plt

N = 1000
l = 1/800
threshold = 50000

sims = 10000
exceeded = 0

s = 0
while s <= sims:
    claimSum = 0
    i=0
    while i < N:
        p = np.random.uniform(0,1)
        if p < 0.05:
            u = np.random.uniform(0,1)
            ci = -np.log(u)/l
            claimSum += ci
        i += 1
    if claimSum >= threshold:
        exceeded += 1
    s += 1

var = exceeded/sims

print(var)