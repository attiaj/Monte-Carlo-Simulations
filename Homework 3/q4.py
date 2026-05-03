import numpy as np
from scipy.stats import norm

# Parameters
S0      = 100.0
r       = 0.05
T       = 1.0
K       = 100.0
M       = 10**5
N       = 365
sigma_c = 0.5   # constant vol for GBM control variate

dt      = T / N
sqrt_dt = np.sqrt(dt)
disc    = np.exp(-r * T)
t_grid  = np.arange(N) * dt

np.random.seed(42)

def local_vol(t, S):
    return 0.5 * np.exp(-t) * (100.0 / S) ** 0.3

def bs_put(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

<<<<<<< Updated upstream
# Known expectation of control: BS put with constant vol sigma_c
E_C = bs_put(S0, K, r, sigma_c, T)
=======

# 1. Std Monte Carlo
payoffs = np.zeros(M)

for i in range(M):
    S = S0
    t = 0.0
>>>>>>> Stashed changes

def simulate_local_vol(Z):
    # Euler-Maruyama for local vol model
    S = np.full(Z.shape[0], S0, dtype=np.float64)
    for j in range(N):
<<<<<<< Updated upstream
        vol = local_vol(t_grid[j], S)
        S   = S + S * r * dt + S * vol * sqrt_dt * Z[:, j]
=======
        Z = np.random.randn()
        vol = sigma(t, S)
        S = S + S * r * dt + S * vol * sqrt_dt * Z
        t += dt

    payoffs[i] = np.exp(-r * T) * max(K - S, 0)

price_mc = np.mean(payoffs)
var_mc = np.var(payoffs, ddof=1)


# 2. Antithetic + simple control variate

payoffs_cv = np.zeros(M // 2)

# control
sigma_c = 0.5

def gbm_terminal(Z_path):
    S = S0
    for z in Z_path:
        S = S + S * r * dt + S * sigma_c * sqrt_dt * z
>>>>>>> Stashed changes
    return S

def simulate_gbm(Z):
    # GBM with constant vol sigma_c, used as control variate
    G = np.full(Z.shape[0], S0, dtype=np.float64)
    for j in range(N):
        G = G + G * r * dt + G * sigma_c * sqrt_dt * Z[:, j]
    return G

# 1. Standard Monte Carlo
Z_mc       = np.random.standard_normal((M, N))
S_T_mc     = simulate_local_vol(Z_mc)
payoffs_mc = disc * np.maximum(K - S_T_mc, 0.0)
price_mc   = np.mean(payoffs_mc)
var_mc     = np.var(payoffs_mc, ddof=1)
se_mc      = np.std(payoffs_mc, ddof=1) / np.sqrt(M)

print("Part 1: Standard Monte Carlo")
print(f"  Price    : {price_mc:.6f}")
print(f"  Variance : {var_mc:.4f}")
print(f"  95% CI   : ({price_mc - 1.96*se_mc:.6f}, {price_mc + 1.96*se_mc:.6f})\n")

# 2. Antithetic + Control Variate
M_half  = M // 2
M_pilot = 5000

# Pilot sample to estimate c_opt independently of the pricing sample
Z_pilot  = np.random.standard_normal((M_pilot, N))
Y_pilot  = 0.5 * (disc * np.maximum(K - simulate_local_vol( Z_pilot), 0.0)
                + disc * np.maximum(K - simulate_local_vol(-Z_pilot), 0.0))
C_pilot  = 0.5 * (disc * np.maximum(K - simulate_gbm( Z_pilot), 0.0)
                + disc * np.maximum(K - simulate_gbm(-Z_pilot), 0.0))

cov_mat = np.cov(Y_pilot, C_pilot, ddof=1)
c_opt   = cov_mat[0, 1] / cov_mat[1, 1]

<<<<<<< Updated upstream
# Pricing sample
Z_price = np.random.standard_normal((M_half, N))
Y = 0.5 * (disc * np.maximum(K - simulate_local_vol( Z_price), 0.0)
         + disc * np.maximum(K - simulate_local_vol(-Z_price), 0.0))
C = 0.5 * (disc * np.maximum(K - simulate_gbm( Z_price), 0.0)
         + disc * np.maximum(K - simulate_gbm(-Z_price), 0.0))

# Apply CV correction
Y_cv     = Y - c_opt * (C - E_C)
price_cv = np.mean(Y_cv)
var_cv   = np.var(Y_cv, ddof=1)
se_cv    = np.std(Y_cv, ddof=1) / np.sqrt(M_half)
=======
    # simple control coefficient
    if i == 0:
        cov = 0
        var_c = 0

    payoffs_cv[i] = payoff 
>>>>>>> Stashed changes

print("Part 2: Antithetic + Control Variate")
print(f"  c_opt    : {c_opt:.6f}")
print(f"  Price    : {price_cv:.6f}")
print(f"  Variance : {var_cv:.4f}")
print(f"  95% CI   : ({price_cv - 1.96*se_cv:.6f}, {price_cv + 1.96*se_cv:.6f})\n")

<<<<<<< Updated upstream
print("Summary")
print(f"  Standard MC      -- Price: {price_mc:.4f}  Var: {var_mc:.2f}")
print(f"  Antithetic + CV  -- Price: {price_cv:.4f}  Var: {var_cv:.2f}  VRF: {var_mc/var_cv:.2f}x")
=======


# Results

print("Standard MC Price:", price_mc)
print("Standard MC Var:  ", var_mc)

print("Antithetic+CV Price:", price_cv)
print("Reduced Variance:   ", var_cv)

print("Variance reduction factor:", var_mc / var_cv)

## Results printed
##Standard MC Price: 10.501233852319068
##Standard MC Var:   221.65065908147932
##Antithetic+CV Price: 10.498530297505825
##Reduced Variance:    55.997034547733904
##Variance reduction factor: 3.9582570911417863
>>>>>>> Stashed changes
