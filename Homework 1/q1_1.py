import numpy as np
import matplotlib.pyplot as plt

# List of progressively larger sample sizes
N = []
i = 1
while i <= 8:
    N.append(10**i)
    i += 1

xn = [] # List of approximations per sample size
for n in N: # Iterate over sample size
    x = np.random.uniform(0,1,n) # Generate n many uniform x
    y = (1-x**2)**(3/2) # Transform uniform variable into integrand
    int = sum(y)/n # Average of generated integrands
    xn.append(int) # Add to list

print(xn[-1]) # Print most accurate value
plt.plot(N,xn) # Plot approximation vs sample size
plt.xscale('log') # Rescale x axis
plt.show() # Show graph