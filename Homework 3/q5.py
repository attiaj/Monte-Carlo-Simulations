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
 
# Accumulators for the choices
BT    = np.zeros(M)   # B_T
WT    = np.zeros(M)   # W_T 
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

## Results printed to see
## sigma = 0.918559,  rho = -0.2,  sqrt(theta) = 0.5000

## Part 1 – Plain Monte Carlo
##   Price estimate : 20.752528
##  95% CI         : [20.676328, 20.828728]
##   Half-width     : 0.076200
##   Std dev        : 38.877693

## Control variate: Z = B (variance BM)
##      v      CV Price       CI lo       CI hi          HW   VarReduc     c_opt

 ##  0.05     20.754829   20.678765   20.830892    0.076064       1.00x   -0.5582 ◄ best
##   0.10     20.754147   20.678079   20.830215    0.076068       1.00x   -0.3076
##   0.15     20.753876   20.677804   20.829947    0.076072       1.00x   -0.2086
##   0.20     20.753616   20.677541   20.829691    0.076075       1.00x   -0.1549
##   0.25     20.753373   20.677296   20.829451    0.076078       1.00x   -0.1213
##   0.30     20.753172   20.677091   20.829252    0.076080       1.00x   -0.0983
##   0.35     20.752978   20.676895   20.829061    0.076083       1.00x   -0.0816
##   0.40     20.752804   20.676718   20.828889    0.076086       1.00x   -0.0689
##   0.45     20.752625   20.676537   20.828714    0.076088       1.00x   -0.0589
##   0.50     20.752446   20.676355   20.828537    0.076091       1.00x   -0.0509
##   → Best at v = 0.05, half-width = 0.076064  (plain MC: 0.076200, improvement: 1.00x)

## Control variate: Z = W (independent BM)
##      v      CV Price       CI lo       CI hi          HW   VarReduc     c_opt

##   0.05     20.759424   20.716458   20.802389    0.042965       3.15x    7.6970 ◄ best
##   0.10     20.767298   20.730342   20.804253    0.036955       4.25x    4.5654 ◄ best
##  0.15     20.771957   20.737859   20.806056    0.034098       4.99x    3.2090 ◄ best
##   0.20     20.774434   20.742142   20.806726    0.032292       5.57x    2.4439 ◄ best
##   0.25     20.775764   20.744767   20.806760    0.030997       6.04x    1.9521 ◄ best
##   0.30     20.776548   20.746513   20.806582    0.030034       6.44x    1.6091 ◄ best
##   0.35     20.777145   20.747809   20.806480    0.029336       6.75x    1.3559 ◄ best
##   0.40     20.777331   20.748460   20.806202    0.028871       6.97x    1.1613 ◄ best
##   0.45     20.777274   20.748645   20.805903    0.028629       7.08x    1.0068 ◄ best
##   0.50     20.777353   20.748749   20.805957    0.028604       7.10x    0.8812 ◄ best
##   → Best at v = 0.50, half-width = 0.028604  (plain MC: 0.076200, improvement: 2.66x)

## Control variate: Z = M = rho*B+sqrt(1-rho^2)*W  [stock BM]
##      v      CV Price       CI lo       CI hi          HW   VarReduc     c_opt

##   0.05     20.765814   20.722293   20.809336    0.043521       3.07x    7.6493 ◄ best
##   0.10     20.772586   20.734636   20.810536    0.037950       4.03x    4.5264 ◄ best
##   0.15     20.775960   20.740437   20.811484    0.035524       4.60x    3.1743 ◄ best
##   0.20     20.776781   20.742639   20.810924    0.034142       4.98x    2.4121 ◄ best
##   0.25     20.777057   20.743785   20.810329    0.033272       5.25x    1.9225 ◄ best
##   0.30     20.777324   20.744593   20.810055    0.032731       5.42x    1.5811 ◄ best
##   0.35     20.777253   20.744808   20.809699    0.032445       5.52x    1.3295 ◄ best
##   0.40     20.777010   20.744631   20.809389    0.032379       5.54x    1.1361 ◄ best
##   0.45     20.776775   20.744263   20.809286    0.032512       5.49x    0.9828
##   0.50     20.776487   20.743658   20.809316    0.032829       5.39x    0.8583
##   → Best at v = 0.40, half-width = 0.032379  (plain MC: 0.076200, improvement: 2.35x)

## Summary
##   Plain MC half-width : 0.076200
##   sqrt(theta) = 0.5000  (natural vol scale for G)