import numpy as np
from scipy.stats import norm

# Parameters
S0    = 100.0
K     = 100.0
r     = 0.05
sigma = 0.4
T     = 1.0
N     = 5000
M     = 200000
dt    = T / N
disc  = np.exp(-r * dt)

np.random.seed(42)

t_grid = np.arange(N + 1) * dt

def bs_put_vec(S, tau):
    """Vectorized BS put. S: (M,) array, tau: scalar time to maturity."""
    if tau <= 1e-10:
        return np.maximum(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    return K*np.exp(-r*tau)*norm.cdf(-d2) - S*norm.cdf(-d1)

# European put price at t=0
P0 = bs_put_vec(np.array([S0]), T)[0]
print(f"European put price P(0, S0) = {P0:.4f}\n")

# Parts 1 & 2: stream through time, accumulate sup and stopping times
# Never store full (M, N+1) matrices — only current step quantities

print("Computing Parts 1 and 2 (streaming over time steps)...")

S       = np.full(M, S0, dtype=np.float64)   # current stock price
disc_t  = 1.0                                  # e^{-rt} at current t, starts at 1

# Accumulators for supremum (Parts 1)
sup_a = np.full(M, -np.inf)
sup_b = np.full(M, -np.inf)

# Accumulators for stopping time (Part 2)
first_exercise_a = np.full(M, N, dtype=int)   # default: exercise at T
first_exercise_b = np.full(M, N, dtype=int)
stopped_a        = np.zeros(M, dtype=bool)
stopped_b        = np.zeros(M, dtype=bool)

for i in range(N + 1):
    t_i    = t_grid[i]
    disc_t = np.exp(-r * t_i)

    # Discounted payoff G_t
    G_t = disc_t * np.maximum(K - S, 0.0)

    # Martingale (a): 
    M_a_t = disc_t * S - S0

    # Martingale (b): 
    tau   = T - t_i
    P_t   = bs_put_vec(S, tau)
    M_b_t = disc_t * P_t - P0

    # Update running supremum for Part 1
    sup_a = np.maximum(sup_a, G_t - M_a_t)
    sup_b = np.maximum(sup_b, G_t - M_b_t)

    # Step forward (except at last step)
    if i < N:
        Z = np.random.standard_normal(M)
        S = S * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

# Part 1 upper bounds
upper_a = np.mean(sup_a)
se_a    = np.std(sup_a, ddof=1) / np.sqrt(M)
upper_b = np.mean(sup_b)
se_b    = np.std(sup_b, ddof=1) / np.sqrt(M)

print("\nPart 1 Results:")
print(f"  (a) Discounted stock upper bound : {upper_a:.4f}  95% CI: ({upper_a - 1.96*se_a:.4f}, {upper_a + 1.96*se_a:.4f})")
print(f"  (b) European put upper bound     : {upper_b:.4f}  95% CI: ({upper_b - 1.96*se_b:.4f}, {upper_b + 1.96*se_b:.4f})")
print(f"\n  (b) is tighter since the European put value process is structurally closer to the true American put value process.")

# Part 2: second pass with known V0 values
print("\nComputing Part 2:")

np.random.seed(42)   # same seed = same paths
S = np.full(M, S0, dtype=np.float64)

# Track stopping time payoffs for each martingale
G_tau_a  = np.zeros(M)
G_tau_b  = np.zeros(M)
stopped_a = np.zeros(M, dtype=bool)
stopped_b = np.zeros(M, dtype=bool)

for i in range(N + 1):
    t_i    = t_grid[i]
    disc_t = np.exp(-r * t_i)

    G_t   = disc_t * np.maximum(K - S, 0.0)
    M_a_t = disc_t * S - S0
    tau   = T - t_i
    P_t   = bs_put_vec(S, tau)
    M_b_t = disc_t * P_t - P0

    # Exercise condition: G_t - M_t >= V0
    ex_a = (~stopped_a) & ((G_t - M_a_t) >= upper_a)
    ex_b = (~stopped_b) & ((G_t - M_b_t) >= upper_b)

    # Record payoff at first exercise
    G_tau_a[ex_a] = G_t[ex_a]
    G_tau_b[ex_b] = G_t[ex_b]
    stopped_a |= ex_a
    stopped_b |= ex_b

    if i < N:
        Z = np.random.standard_normal(M)
        S = S * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

# Paths never stopped: use terminal payoff
G_tau_a[~stopped_a] = np.exp(-r*T) * np.maximum(K - S, 0.0)[~stopped_a]
G_tau_b[~stopped_b] = np.exp(-r*T) * np.maximum(K - S, 0.0)[~stopped_b]

lower_a  = np.mean(G_tau_a)
se_la    = np.std(G_tau_a, ddof=1) / np.sqrt(M)
lower_b  = np.mean(G_tau_b)
se_lb    = np.std(G_tau_b, ddof=1) / np.sqrt(M)

print(f"\nEuropean put price (lower reference): P0 = {P0:.4f}")
print(f"\n  (a) Discounted stock martingale")
print(f"    V0 (upper bound)    : {upper_a:.4f}")
print(f"    E[G_tau] (lower bd) : {lower_a:.4f}  95% CI: ({lower_a - 1.96*se_la:.4f}, {lower_a + 1.96*se_la:.4f})")
print(f"    Interval            : [{lower_a:.4f}, {upper_a:.4f}]  width = {upper_a - lower_a:.4f}")
print(f"    Early exercise beats waiting: {lower_a > P0}  (E[G_tau]={lower_a:.4f} vs P0={P0:.4f})")
print(f"\n  (b) European put martingale")
print(f"    V0 (upper bound)    : {upper_b:.4f}")
print(f"    E[G_tau] (lower bd) : {lower_b:.4f}  95% CI: ({lower_b - 1.96*se_lb:.4f}, {lower_b + 1.96*se_lb:.4f})")
print(f"    Interval            : [{lower_b:.4f}, {upper_b:.4f}]  width = {upper_b - lower_b:.4f}")
print(f"    Early exercise beats waiting: {lower_b > P0}  (E[G_tau]={lower_b:.4f} vs P0={P0:.4f})")

# Part 3: LSMC in path batches
print("\nComputing Part 3: LSMC (batched)...")

BATCH = 10000   # number of paths per batch

# Run LSMC on each batch and average the prices
lsmc_prices = []

np.random.seed(123)   # separate seed for LSMC paths
for b_start in range(0, M, BATCH):
    b_size = min(BATCH, M - b_start)

    # Simulate batch of paths
    Z_b     = np.random.standard_normal((b_size, N))
    log_inc = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z_b
    logS    = np.zeros((b_size, N+1))
    logS[:, 0] = np.log(S0)
    for i in range(N):
        logS[:, i+1] = logS[:, i] + log_inc[:, i]
    S_batch = np.exp(logS)

    # Initialize value at terminal time
    V = np.zeros((b_size, N+1))
    V[:, N] = np.maximum(K - S_batch[:, N], 0.0)

    # Backward induction
    for t in range(N-1, 0, -1):
        V[:, t] = disc * V[:, t+1]
        itm = S_batch[:, t] < K
        if np.sum(itm) == 0:
            continue
        S_itm = S_batch[itm, t]
        psi   = np.column_stack([S_itm**k for k in range(7)])
        Q, _  = np.linalg.qr(psi)
        b_vec = Q.T @ (disc * V[itm, t+1])
        CV_hat = Q @ b_vec
        intrinsic    = np.maximum(K - S_itm, 0.0)
        V[itm, t]    = np.where(intrinsic > CV_hat, intrinsic, disc * V[itm, t+1])

    lsmc_prices.append(disc * np.mean(V[:, 1]))
    if (b_start // BATCH) % 5 == 0:
        print(f"  Batch {b_start//BATCH + 1}/{M//BATCH}...")

lsmc_price = np.mean(lsmc_prices)
lsmc_se    = np.std(lsmc_prices) / np.sqrt(len(lsmc_prices))

print(f"\nPart 3 Results:")
print(f"  LSMC price : {lsmc_price:.4f}  95% CI: ({lsmc_price - 1.96*lsmc_se:.4f}, {lsmc_price + 1.96*lsmc_se:.4f})")
print(f"  Lower bound (b) : {lower_b:.4f}")
print(f"  LSMC price      : {lsmc_price:.4f}")
print(f"  Upper bound (b) : {upper_b:.4f}")
print(f"  LSMC within duality bounds: {lower_b <= lsmc_price <= upper_b}")