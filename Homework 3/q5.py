import numpy as np
from scipy.stats import norm
 

S0    = 100.0
K     = S0
r     = 0.04
T     = 1.0
kappa = 3.0
theta = 0.25
sigma = 0.75 * np.sqrt(2 * theta * kappa)
rho   = -0.2
X0    = 0.25
 
dt    = 1 / 365
N     = int(T / dt)
M     = 1_000_000
 
rng   = np.random.default_rng(seed=42)
sqrt_dt     = np.sqrt(dt)
sqrt_1mrho2 = np.sqrt(1 - rho**2)
disc        = np.exp(-r * T)
 
print(f"sigma = {sigma:.6f},  rho = {rho},  sqrt(theta) = {np.sqrt(theta):.4f}\n")
 
# Simulate 
logS  = np.full(M, np.log(S0), dtype=np.float64)
X     = np.full(M, X0,         dtype=np.float64)
 
# Accumulators for the three choices of Z
BT    = np.zeros(M)   # B_T
WT    = np.zeros(M)   # W_T  (independent of B)
MT    = np.zeros(M)   # M_T = rho*B_T + sqrt(1-rho^2)*W_T
 
for _ in range(N):
    z_B = rng.standard_normal(M)
    z_W = rng.standard_normal(M)
 
    X_pos  = np.maximum(X, 0.0)
    sqrtX  = np.sqrt(X_pos)
 
    X    += kappa * (theta - X_pos) * dt + sigma * sqrtX * sqrt_dt * z_B
 
    dlogS = (r - 0.5 * X_pos) * dt + sqrtX * sqrt_dt * (rho * z_B + sqrt_1mrho2 * z_W)
    logS += dlogS
 
    BT   += sqrt_dt * z_B
    WT   += sqrt_dt * z_W
    MT   += sqrt_dt * (rho * z_B + sqrt_1mrho2 * z_W)
 
S_T = np.exp(logS)
 
# Plain MC
payoff = disc * np.maximum(S_T - K, 0.0)
mu1    = payoff.mean()
se1    = payoff.std(ddof=1) / np.sqrt(M)
hw1    = 1.96 * se1
 
print("=" * 65)
print("Part 1 – Plain Monte Carlo")
print(f"  Price estimate : {mu1:.6f}")
print(f"  95% CI         : [{mu1 - hw1:.6f}, {mu1 + hw1:.6f}]")
print(f"  Half-width     : {hw1:.6f}")
print(f"  Std dev        : {payoff.std(ddof=1):.6f}\n")
 
# Black-Scholes call
def bs_call(S0, K, r, T, v):
    if v <= 0:
        return max(S0 - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * v**2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
 
# Control variate for each choice of Z 
# G_T = S0 * exp((r - v^2/2)*T + v*Z_T)
# E[disc*(G_T-K)^+] = BS(S0, K, r, T, v)
 
for label, ZT in [("B (variance BM)", BT), ("W (independent BM)", WT), ("M = rho*B+sqrt(1-rho^2)*W  [stock BM]", MT)]:
    print("=" * 65)
    print(f"Control variate: Z = {label}")
    print(f"{'v':>6}  {'CV Price':>12}  {'CI lo':>10}  {'CI hi':>10}  "
          f"{'HW':>10}  {'VarReduc':>9}  {'c_opt':>8}")
    print("-" * 82)
 
    best_hw = np.inf
    best_v  = None
 
    for m in range(1, 11):
        v = m * 0.05
 
        logG_T   = np.log(S0) + (r - 0.5 * v**2) * T + v * ZT
        G_T      = np.exp(logG_T)
        g_payoff = disc * np.maximum(G_T - K, 0.0)
        Eg       = bs_call(S0, K, r, T, v)
 
        cov_mat  = np.cov(payoff, g_payoff, ddof=1)
        c_opt    = cov_mat[0, 1] / cov_mat[1, 1]
 
        Y_cv     = payoff - c_opt * (g_payoff - Eg)
        mu_cv    = Y_cv.mean()
        se_cv    = Y_cv.std(ddof=1) / np.sqrt(M)
        hw       = 1.96 * se_cv
        var_red  = (se1 / se_cv) ** 2
 
        mark = " ◄ best" if hw < best_hw else ""
        if hw < best_hw:
            best_hw = hw
            best_v  = v
 
        print(f"{v:6.2f}  {mu_cv:12.6f}  {mu_cv-hw:10.6f}  {mu_cv+hw:10.6f}  "
              f"{hw:10.6f}  {var_red:9.2f}x  {c_opt:8.4f}{mark}")
 
    print(f"  → Best at v = {best_v:.2f}, half-width = {best_hw:.6f}  "
          f"(plain MC: {hw1:.6f}, improvement: {hw1/best_hw:.2f}x)\n")
 
# Summary
print("=" * 65)
print("Summary")
print(f"  Plain MC half-width : {hw1:.6f}")
print(f"  sqrt(theta) = {np.sqrt(theta):.4f}  (natural vol scale for G)")