import numpy as np
import matplotlib.pyplot as plt

# List of progressively larger sample sizes
N = []
i = 1
while i <= 5:
    N.append(10**i)
    i += 1

xn = [] # List of approximations of total integral per sample size
for n in N: # Iterate over sample size
    ex = np.random.uniform(0,1,n) # Generate n many uniform variables
    X = -np.log(ex) # Now those variables are between 0 and infinity instead of 0 and 1
    
    yn = [] # List of approximations of inner integral per x value
    # Use a fixed number of y samples per x to avoid O(n²) complexity
    # Using n samples would be too slow for large n
    m = min(1000, n)  # Use at most 1000 samples per x, or n if n < 1000
    for x in X: # Iterate over generated x values
        y = np.random.uniform(0,x,m) # Generate uniform y in bounds
        ey = np.exp(-y) # Transform uniform to integrand
        inty = x*sum(ey)/m # Average integrand value times size of integral domain
        yn.append(inty) # Add to list of inner integrals
    
    intx = sum(yn)/n # Average result of inner integral—sampling by exponential distributions cancels out exterior integrand
    xn.append(intx) # Add to list of total integrals

print(xn[-1]) # Print most accurate value
plt.plot(N,xn) # Plot approximation vs sample size
plt.xscale('log') # Rescale x axis
plt.show() # Show graph