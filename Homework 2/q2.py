import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

S0 = 100
r = 0.05
dt = 0.001
M = 800000   # reduced so code runs I was having a lot of issues when I made it really big

K_vals = np.arange(80, 121)
T_vals = np.arange(0.25, 2.01, 0.25)


# Volatility function
def sigma(t, x):
    x_safe = np.maximum(x, 1e-8)
    return 0.5 * np.exp(-t) * (100 / x_safe)**0.3


# Black-Scholes Call formula we were given
def bs_call(S, K, T, r, sigma):
    if sigma <= 0:
        return max(S-K,0)

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# How we find implied volatility
def implied_vol(price, S, K, T, r):

    intrinsic = max(S - K*np.exp(-r*T),0)

    if price < intrinsic or price > S:
        return np.nan

    f = lambda sig: bs_call(S,K,T,r,sig) - price

    try:
        return brentq(f,1e-6,5)
    except:
        return np.nan

# Store it
price_surface = np.zeros((len(T_vals),len(K_vals)))
iv_surface = np.zeros((len(T_vals),len(K_vals)))
ci_low_surface = np.zeros((len(T_vals),len(K_vals)))
ci_high_surface = np.zeros((len(T_vals),len(K_vals)))

# Actual Simulation

for i,T in enumerate(T_vals):

    N = int(T/dt)

    S_paths = np.full(M, S0, dtype=float)

    for k in range(N):

        Z = np.random.normal(size=M)
        vol = sigma(k*dt,S_paths)

        S_paths *= np.exp(
            (r - 0.5*vol**2)*dt +
            vol*np.sqrt(dt)*Z
        )

        # keep prices positive because we can't have negative price
        S_paths = np.maximum(S_paths,1e-8)

    discount = np.exp(-r*T)

    for j,K in enumerate(K_vals):

        payoff = np.maximum(S_paths-K,0)
        discounted_payoff = discount*payoff

        price = np.mean(discounted_payoff)

        std = np.std(discounted_payoff)
        se = std/np.sqrt(M)

        ci_low = price - 1.96*se
        ci_high = price + 1.96*se

        iv = implied_vol(price,S0,K,T,r)

        price_surface[i,j] = price
        iv_surface[i,j] = iv
        ci_low_surface[i,j] = ci_low
        ci_high_surface[i,j] = ci_high

# Results
for i,T in enumerate(T_vals):

    print(f"\nMaturity T = {T:.2f}")
    print("K | Price | CI Low | CI High | Implied Vol")
    print("-"*50)

    for j,K in enumerate(K_vals):
        print(f"{K:3d} | {price_surface[i,j]:6.3f} | {ci_low_surface[i,j]:7.3f} | {ci_high_surface[i,j]:7.3f} | {iv_surface[i,j]:6.3f}")

# Implied Volatility Surface
K_grid, T_grid = np.meshgrid(K_vals,T_vals)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111,projection='3d')

ax.plot_surface(K_grid,T_grid,iv_surface,cmap='viridis')

ax.set_xlabel("Strike K")
ax.set_ylabel("Maturity T")
ax.set_zlabel("Implied Volatility")

ax.set_title("Implied Volatility Surface")

plt.show()