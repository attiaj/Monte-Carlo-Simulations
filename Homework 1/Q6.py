import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
S0 = 100
K = 100
T = 1
r = 0.05
sigma = 0.4
mu_list = [-0.2, 0, 0.2, r - sigma**2 / 2]
N = 500        # large number of steps
M = 100000     # Monte Carlo paths

for mu in mu_list:
    u = np.exp(mu/N + sigma*np.sqrt(1/N))
    d = 1/u
    p = (np.exp(r/N) - d) / (u - d)
    
    # simulate log returns
    Z = np.random.choice([np.log(u), np.log(d)], size=(M, N), p=[p, 1-p])
    X = Z.sum(axis=1)  # log(S_N / S0)
    
    # statistics
    mean_X = np.mean(X)
    std_X = np.std(X)
    print(f"mu={mu}: mean={mean_X:.4f}, std={std_X:.4f}")
    
    # histogram
    plt.hist(X, bins=100, density=True, alpha=0.5)
    plt.title(f"Histogram of ln(S_N/S0), mu={mu}")
    plt.show()
    
    # call price
    S_N = S0 * np.exp(X)
    call_payoff = np.maximum(S_N - K, 0)
    C0 = np.exp(-r*T) * np.mean(call_payoff)
    print(f"mu={mu}: Call price ≈ {C0:.4f}")
