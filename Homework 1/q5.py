import numpy as np
import matplotlib.pyplot as plt

# Given constants
N = 1000
l = 1/800
threshold = 50000

sims = 10**4 # Number of simulations
exceeded = 0 # Number of simulations which have so far exceeded the threshold

s = 0 # Simulation index
while s < sims: # Iterate over all simulations
    claimSum = 0 # Total claim size in the next month
    i=0 # ith policy holded
    while i < N: # Iterate over all policy holders
        p = np.random.uniform(0,1)
        if p < 0.05: # 5% of the time, the ith policy holder will submit a claim
            u = np.random.uniform(0,1) # Generate uniform
            ci = -np.log(u)/l # Transform uniform into exponentially distributed claim size
            claimSum += ci # Add ith policy holder's claim to the total
        i += 1
    if claimSum >= threshold: # Count the number of times the sum of claims is above the threshold
        exceeded += 1
    s += 1

var = exceeded/sims # Percentage of times the sum of claims is above the threshold

print(var)