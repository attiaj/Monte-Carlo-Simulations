## First you need to take logs of the function given

## Turns the question into a sum

## Therefore, the question becomes the sum of ln(Ui) from i=1-->n

## By doing logs, this becomes just -5 on the right with exponent rules

""" 
We get: Sum(i=1-->n)(ln(Ui)) >=-5
And we know 0<=Ui<=1 for all i>0
Which will make a negative number when you apply natural logs
So now multiply by -1 on each side and get:
Sum(i=1-->n)(-ln(Ui)) <=5
"""

## Simulation of E[N] using previous work and -ln
import numpy as np
import math
np.random.seed(15)

threshold = 5 ## we found this to be the maximum value
reps = 100000
N_vals = np.zeros(reps, dtype=int)

for i in range(reps):
    total = 0
    n = 0
    while total < threshold:
        U = np.random.uniform()
        total += -np.log(U)  # -log(U) ~ Exp(1)
        n += 1
    N_vals[i] = n - 1  # because max n with sum < threshold

# Estimate E[N]
E_N = np.mean(N_vals)
print("Estimated E[N] =", E_N)

## We found this expected value to be close to 5

## Now we need to do part 2

## We can see that the number of n is related to the Poisson Process
## Specifically, the increments are exponential
## This is a key property of the Poisson process, where events occur continuously and independently at a constant average rate

## Therefore, this is a lambda=5 poisson probability
## Need to find the probablities for each k<=20 to stay below the threshold of 5
k_max = 20
counts = np.array([np.sum(N_vals == k) for k in range(k_max+1)])
probs = counts / reps

for k in range(k_max+1):
    print(f"P[N={k}] ~ {probs[k]:.4f}")


## Now compare this to the Poisson Distribution
## P[N=k] = (5^k * e^(-5)) / k!


reps = len(N_vals)

# Compute theoretical Poisson probabilities manually
lambda_val = 5
poisson_probs = np.array([ (lambda_val**k * np.exp(-lambda_val)) / math.factorial(k) 
                           for k in range(k_max+1) ])

# Print side by side
print("k\tSimulated\tPoisson(λ=5)")
for k in range(k_max+1):
    print(f"{k}\t{probs[k]:.4f}\t\t{poisson_probs[k]:.4f}")

## In conclusion we found how likely each number is below the threshold of 5
## We can see that this virtually matches the Poisson distribution with λ=5
## The estimated value therefore is about 5 which confirms that N follows approximately
## a Poisson distribution with λ=5