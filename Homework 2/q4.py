import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

np.random.seed(42)

# Call price we got in class
def bs_call(S, K, r, T, sigma):
    if sigma <= 0:
        return max(S - K * np.exp(-r * T), 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# Implied Volatility

def implied_vol(price, S, K, r, T):
    f = lambda sigma: bs_call(S, K, r, T, sigma) - price
    try:
        return brentq(f, 1e-6, 5)
    except ValueError:
        return np.nan


# Heston Model with variables
def heston_mc(S0, X0, r, kappa, theta, sigma, rho, T, paths=50000, dt=1/365):
    steps = int(T / dt)

    S = np.full(paths, float(S0))
    X = np.full(paths, float(X0))

    for _ in range(steps):
        Z1 = np.random.normal(size=paths)
        Z2 = np.random.normal(size=paths)

        dB = np.sqrt(dt) * Z1
        dW = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

        X_curr = X.copy()

        X = np.maximum(
            X + kappa * (theta - X) * dt + sigma * np.sqrt(np.maximum(X, 0)) * dB,
            0
        )

        S = S * np.exp((r - 0.5 * X_curr) * dt + np.sqrt(np.maximum(X_curr, 0)) * dW)

    return S


# Parameters
S0 = 100
r = 0.05
T = 1
X0 = 0.2
kappa = 3
theta = 0.2

strikes = np.arange(90, 121)

sigma_vals = np.sqrt(2 * theta * kappa) * np.array([0.35, 0.75, 1])
rho_vals = [-0.2, 0, 0.2]

fig, axs = plt.subplots(3, 3, figsize=(12, 10))

for i, sigma in enumerate(sigma_vals):
    for j, rho in enumerate(rho_vals):

        ST = heston_mc(S0, X0, r, kappa, theta, sigma, rho, T)

        iv = []
        logm = []

        for K in strikes:
            call = np.mean(np.maximum(ST - K, 0)) * np.exp(-r * T)
            iv.append(implied_vol(call, S0, K, r, T))
            logm.append(np.log(K * np.exp(-r * T) / S0))

        ax = axs[i, j]
        ax.plot(logm, iv, 'o-')
        ax.set_title(f"sigma={sigma:.2f}, rho={rho}")
        ax.set_xlabel("log(K * e^(-rT) / S0)")
        ax.set_ylabel("Implied Vol")

plt.tight_layout()
plt.show()


# Lookback
def lookback_spread_mc(S0, X0, r, kappa, theta, sigma, rho, T, m, paths=10000):
    dt = T / m

    S = np.full(paths, float(S0))
    X = np.full(paths, float(X0))

    Smax = S.copy()
    Smin = S.copy()

    for _ in range(m):
        Z1 = np.random.normal(size=paths)
        Z2 = np.random.normal(size=paths)

        dB = np.sqrt(dt) * Z1
        dW = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

        X_curr = X.copy()

        X = np.maximum(
            X + kappa * (theta - X) * dt + sigma * np.sqrt(np.maximum(X, 0)) * dB,
            0
        )

        S = S * np.exp((r - 0.5 * X_curr) * dt + np.sqrt(np.maximum(X_curr, 0)) * dW)

        Smax = np.maximum(Smax, S)
        Smin = np.minimum(Smin, S)

    payoff = np.exp(-r * T) * (Smax - Smin)
    return np.mean(payoff)


# Parameters and results for part 2
S0_2   = 100
X0_2   = 0.04
r2     = 0.03
kappa2 = 2
theta2 = 0.04
sigma2 = 0.5
rho2   = -0.7
T2     = 0.5

print("Lookback Spread Option Prices:")
for m in [2, 3, 6, 12]:
    price = lookback_spread_mc(S0_2, X0_2, r2, kappa2, theta2, sigma2, rho2, T2, m)
    print(f"  m={m:>2}: {price:.4f}")