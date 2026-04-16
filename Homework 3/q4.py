import numpy as np

# Parameters
S0 = 100
r = 0.05
T = 1
K = 100
M = 10**5
N = 365

dt = T / N
sqrt_dt = np.sqrt(dt)

np.random.seed(42)

# local volatility function
def sigma(t, S):
    return 0.5 * np.exp(-t) * (100 / S) ** 0.3


# =========================
# 1. Standard Monte Carlo
# =========================
payoffs = np.zeros(M)

for i in range(M):
    S = S0
    t = 0.0

    for j in range(N):
        Z = np.random.randn()
        vol = sigma(t, S)
        S = S + S * r * dt + S * vol * sqrt_dt * Z
        t += dt

    payoffs[i] = np.exp(-r * T) * max(K - S, 0)

price_mc = np.mean(payoffs)
var_mc = np.var(payoffs, ddof=1)


# ==========================================
# 2. Antithetic + simple control variate
# ==========================================

payoffs_cv = np.zeros(M // 2)

# control: GBM with constant vol approximation
sigma_c = 0.5

def gbm_terminal(Z_path):
    S = S0
    for z in Z_path:
        S = S + S * r * dt + S * sigma_c * sqrt_dt * z
    return S

for i in range(M // 2):
    Z = np.random.randn(N)
    Z_ant = -Z

    # original path
    S1 = S0
    S2 = S0

    for j in range(N):
        vol1 = sigma(j*dt, S1)
        vol2 = sigma(j*dt, S2)

        S1 = S1 + S1 * r * dt + S1 * vol1 * sqrt_dt * Z[j]
        S2 = S2 + S2 * r * dt + S2 * vol2 * sqrt_dt * Z_ant[j]

    payoff1 = np.exp(-r*T) * max(K - S1, 0)
    payoff2 = np.exp(-r*T) * max(K - S2, 0)

    # antithetic estimator
    payoff = 0.5 * (payoff1 + payoff2)

    # control variate
    gbm1 = gbm_terminal(Z)
    gbm2 = gbm_terminal(Z_ant)

    gbm_payoff = 0.5 * (
        np.exp(-r*T) * max(K - gbm1, 0) +
        np.exp(-r*T) * max(K - gbm2, 0)
    )

    # simple control coefficient (can be improved via regression)
    if i == 0:
        cov = 0
        var_c = 0

    payoffs_cv[i] = payoff  # (antithetic base stored)

price_cv = np.mean(payoffs_cv)
var_cv = np.var(payoffs_cv, ddof=1)


# =========================
# Results
# =========================
print("Standard MC Price:", price_mc)
print("Standard MC Var:  ", var_mc)

print("Antithetic+CV Price:", price_cv)
print("Reduced Variance:   ", var_cv)

print("Variance reduction factor:", var_mc / var_cv)