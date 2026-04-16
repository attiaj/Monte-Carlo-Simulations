import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

# Parameters
S0    = 100
K     = 110
r     = 0.05
sigma = 0.3
T     = 1.0
m     = 12          # monthly monitoring
dt    = T / m       # 1/12
M0    = 1000        # pilot paths

# set seed to keep randomness consistent across results

np.random.seed(1001)

# Closed-form prices

# Black Scholes pricing
def bs_call(S, K, r, sigma, T):
    """Black-Scholes European call price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Asian call (G_T - K)^+ pricing
def geometric_asian_call(S0, K, r, sigma, T, m):
    """
    Closed-form price for a geometric average call with m+1 monitoring points
    (i=0,1,...,m), i.e. including S0.
    """
    n   = m + 1
    # Transform G_T into log(G_T)                         
    # Adjusted drift and vol for log(G_T)
    
    t_i       = np.arange(0, m + 1) * dt           # monitoring times: 0, 1/12, ..., 1
    mean_t    = np.mean(t_i)                        # = T*m/(2m) = T/2  (approx)

    # Compute E[log G_T] 
    mu_G  = np.log(S0) + (r - 0.5 * sigma**2) * mean_t

    # Compute Var[log G_T] using rules for variance of a sum = double sum of covariances
    # Var(log(S_ti)) = Var(sigma*B_t) = sigma^2*t_i
    # Cov(log(S_ti), log(S_tj)) = sigma^2*min(t_i, t_j)
    var_sum = 0.0
    for i in range(n):
        for j in range(n):
            var_sum += min(t_i[i], t_i[j])
    var_G = sigma**2 / n**2 * var_sum

    sig_G = np.sqrt(var_G)

    # Compute d1 and d2 similar to BS framework
    d1 = (mu_G - np.log(K) + var_G) / sig_G
    d2 = d1 - sig_G

    # Compute price using BS framework, with E[G_T] in place of E[C_T]
    price = np.exp(-r * T) * (np.exp(mu_G + 0.5 * var_G) * norm.cdf(d1)
                               - K * norm.cdf(d2))
    return price

# Known undiscounted expectations for the three control variates
E_C1 = S0 * np.exp(r * T)                           
E_C2 = bs_call(S0, K, r, sigma, T) * np.exp(r * T)  
E_C3 = geometric_asian_call(S0, K, r, sigma, T, m) * np.exp(r * T)  

def corr_and_b(PT, C):
    """Return (Pearson correlation, control coefficient b_hat)."""
    cov_matrix = np.cov(PT, C)          # 2x2 sample covariance matrix
    cov_PC = cov_matrix[0, 1]
    var_C  = cov_matrix[1, 1]
    var_P  = cov_matrix[0, 0]
    rho    = cov_PC / np.sqrt(var_P * var_C)
    b_hat  = cov_PC / var_C
    return rho, b_hat

def simulate_paths(M):
    """
    Simulate M GBM paths with monthly monitoring.
    Returns S_paths (M, m+1) and log_S (M, m+1).
    """
    Z = np.random.standard_normal((M, m))
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.zeros((M, m + 1))
    log_S[:, 0] = np.log(S0)
    for i in range(m):
        log_S[:, i + 1] = log_S[:, i] + log_increments[:, i]
    return np.exp(log_S), log_S

def compute_payoffs(S_paths, log_S):
    """
    Compute lookback payoff PT and the three control variates. (undiscounted)
    """
    PT  = np.maximum(np.max(S_paths, axis=1) - K, 0.0)
    C1  = S_paths[:, -1]
    C2  = np.maximum(S_paths[:, -1] - K, 0.0)
    G_T = np.exp(np.mean(log_S, axis=1))
    C3  = np.maximum(G_T - K, 0.0)
    return PT, C1, C2, C3

def part_1a():
    
    S_paths, log_S = simulate_paths(M0)
    PT, C1, C2, C3 = compute_payoffs(S_paths, log_S)

    rho1, b1 = corr_and_b(PT, C1)
    rho2, b2 = corr_and_b(PT, C2)
    rho3, b3 = corr_and_b(PT, C3)

    print("\nPart 1(a): Pilot results (M0 = 1000):\n")
    print(f"{'Control variate':<25} {'rho':>8} {'b_hat':>10}")
    print(f"{'(i)  S_T':<25} {rho1:>8.4f} {b1:>10.4f}")
    print(f"{'(ii) (S_T - K)^+':<25} {rho2:>8.4f} {b2:>10.4f}")
    print(f"{'(iii)(G_T - K)^+':<25} {rho3:>8.4f} {b3:>10.4f}")

    # Scatter plots 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    controls = [
        (C1, b1, rho1, r"$S_T$",          r"$C_1 = S_T$"),
        (C2, b2, rho2, r"$(S_T - K)^+$",  r"$C_2 = (S_T - K)^+$"),
        (C3, b3, rho3, r"$(G_T - K)^+$",  r"$C_3 = (G_T - K)^+$"),
    ]

    for ax, (C, b, rho, xlabel, title) in zip(axes, controls):
        ax.scatter(C, PT, alpha=0.3, s=8, color="steelblue", label="Simulated paths")

        # OLS regression line: P_T ≈ alpha + b * C
        alpha_hat = np.mean(PT) - b * np.mean(C)
        x_line    = np.linspace(C.min(), C.max(), 200)
        y_line    = alpha_hat + b * x_line
        ax.plot(x_line, y_line, color="crimson", linewidth=1.5,
                label=rf"OLS: slope $\hat{{b}}={b:.3f}$")

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(r"$P_T$ (lookback payoff)", fontsize=12)
        ax.set_title(f"{title}\n" + rf"$\hat{{\rho}} = {rho:.4f}$", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(r"Problem 3.1(a): Scatter plots of $P_T$ vs. control variates"
                "\n" + rf"($M_0 = {M0}$ pilot paths, $K={K}$, $\sigma={sigma}$, $r={r}$)",
                fontsize=13)
    plt.tight_layout()
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "prob3_1a_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()

    

    # Return pilot b_hats for part_1c
    return {"b1": b1, "b2": b2, "b3": b3}

def part_1b():

    M_values = [10**4, 10**5, 10**6]
    discount = np.exp(-r * T)

    print("\nPart 1(b): Standard MC vs CV MC (b re-estimated from pricing sample)\n")
    print(f"{'M':<10} {'Method':<22} {'Price':>8} {'Std Dev':>10} {'CI Lower':>10} {'CI Upper':>10} {'b_hat':>8}")

    for M in M_values:
        S_paths, log_S = simulate_paths(M)
        PT, C1, C2, C3 = compute_payoffs(S_paths, log_S)

        # Standard MC: discount and average
        discounted_PT = discount * PT
        price_mc      = np.mean(discounted_PT)
        std_mc        = np.std(discounted_PT, ddof=1) / np.sqrt(M)
        ci_lo_mc      = price_mc - 1.96 * std_mc
        ci_hi_mc      = price_mc + 1.96 * std_mc

        print(f"{M:<10} {'Standard MC':<22} {price_mc:>8.4f} {std_mc:>10.6f} "
              f"{ci_lo_mc:>10.4f} {ci_hi_mc:>10.4f} {'—':>8}")

        # CV MC: b_hat re-estimated from this same sample
        _, b_hat  = corr_and_b(PT, C2)
        PT_cv     = PT - b_hat * (C2 - E_C2)       # controlled undiscounted payoff
        disc_cv   = discount * PT_cv
        price_cv  = np.mean(disc_cv)
        std_cv    = np.std(disc_cv, ddof=1) / np.sqrt(M)
        ci_lo_cv  = price_cv - 1.96 * std_cv
        ci_hi_cv  = price_cv + 1.96 * std_cv

        print(f"{M:<10} {'CV MC (b re-est.)':<22} {price_cv:>8.4f} {std_cv:>10.6f} "
              f"{ci_lo_cv:>10.4f} {ci_hi_cv:>10.4f} {b_hat:>8.4f}")

    print()

def part_1c(pilot_b):
    
    b_fixed  = pilot_b["b2"]     # fixed pilot b_hat for C2, never changes
    M_values = [10**4, 10**5, 10**6]
    discount = np.exp(-r * T)
 
    print("\nPart 1(c): CV MC with fixed pilot b_hat from part_1a\n")
    print(f"Using fixed pilot b_hat = {b_fixed:.4f} (estimated from M0={M0} pilot paths)\n")
    print(f"{'M':<10} {'Method':<22} {'Price':>8} {'Std Dev':>10} {'CI Lower':>10} {'CI Upper':>10} {'b_hat':>8}")
 
    for M in M_values:
        S_paths, log_S = simulate_paths(M)
        PT, C1, C2, C3 = compute_payoffs(S_paths, log_S)
 
        # Standard MC (same as 1b, included for direct comparison)
        discounted_PT = discount * PT
        price_mc      = np.mean(discounted_PT)
        std_mc        = np.std(discounted_PT, ddof=1) / np.sqrt(M)
        ci_lo_mc      = price_mc - 1.96 * std_mc
        ci_hi_mc      = price_mc + 1.96 * std_mc
 
        print(f"{M:<10} {'Standard MC':<22} {price_mc:>8.4f} {std_mc:>10.6f} "
              f"{ci_lo_mc:>10.4f} {ci_hi_mc:>10.4f} {'—':>8}")
 
        # CV MC: b_fixed from pilot, not re-estimated — estimator is unbiased
        PT_cv    = PT - b_fixed * (C2 - E_C2)      # controlled undiscounted payoff
        disc_cv  = discount * PT_cv
        price_cv = np.mean(disc_cv)
        std_cv   = np.std(disc_cv, ddof=1) / np.sqrt(M)
        ci_lo_cv = price_cv - 1.96 * std_cv
        ci_hi_cv = price_cv + 1.96 * std_cv
 
        print(f"{M:<10} {'CV MC (b fixed)':<22} {price_cv:>8.4f} {std_cv:>10.6f} "
              f"{ci_lo_cv:>10.4f} {ci_hi_cv:>10.4f} {b_fixed:>8.4f}")

def part_2():

    M_values = [10**4, 10**5, 10**6]
    discount = np.exp(-r * T)
 
    print("\nPart 2: Antithetic Variates\n")
    print(f"{'M':<10} {'Method':<22} {'Price':>8} {'Std Dev':>10} {'CI Lower':>10} {'CI Upper':>10}")
 
    for M in M_values:
        # Standard MC for comparison 
        S_paths, log_S = simulate_paths(M)
        PT, _, _, _    = compute_payoffs(S_paths, log_S)
 
        discounted_PT = discount * PT
        price_mc      = np.mean(discounted_PT)
        std_mc        = np.std(discounted_PT, ddof=1) / np.sqrt(M)
        ci_lo_mc      = price_mc - 1.96 * std_mc
        ci_hi_mc      = price_mc + 1.96 * std_mc
 
        print(f"{M:<10} {'Standard MC':<22} {price_mc:>8.4f} {std_mc:>10.6f} "
              f"{ci_lo_mc:>10.4f} {ci_hi_mc:>10.4f}")
 
        # Antithetic MC
        # Draw M sets of normals — each row is one path's 12 increments
        Z = np.random.standard_normal((M, m))
 
        # Build original log paths using +Z
        log_inc          = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        log_S_orig       = np.zeros((M, m + 1))
        log_S_orig[:, 0] = np.log(S0)
        for i in range(m):
            log_S_orig[:, i + 1] = log_S_orig[:, i] + log_inc[:, i]
 
        # Build antithetic log paths using -Z (mirror image)
        log_inc_anti      = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * (-Z)
        log_S_anti        = np.zeros((M, m + 1))
        log_S_anti[:, 0]  = np.log(S0)
        for i in range(m):
            log_S_anti[:, i + 1] = log_S_anti[:, i] + log_inc_anti[:, i]
 
        # Compute lookback payoffs for both paths (undiscounted)
        PT_orig = np.maximum(np.max(np.exp(log_S_orig), axis=1) - K, 0.0)
        PT_anti = np.maximum(np.max(np.exp(log_S_anti), axis=1) - K, 0.0)
 
        # Antithetic estimator: average the pair then discount
        # Each Y_j = (PT_orig_j + PT_anti_j) / 2 is one sample
        Y        = (PT_orig + PT_anti) / 2.0
        disc_Y   = discount * Y
        price_av = np.mean(disc_Y)
        std_av   = np.std(disc_Y, ddof=1) / np.sqrt(M)
        ci_lo_av = price_av - 1.96 * std_av
        ci_hi_av = price_av + 1.96 * std_av
 
        # Empirical correlation between original and antithetic payoffs
        rho_anti = np.corrcoef(PT_orig, PT_anti)[0, 1]
 
        print(f"{M:<10} {'Antithetic MC':<22} {price_av:>8.4f} {std_av:>10.6f} "
              f"{ci_lo_av:>10.4f} {ci_hi_av:>10.4f}  rho(P,P~)={rho_anti:.4f}")

# Stratum boundaries and their CDF probabilities
STRATA_BOUNDS = [-np.inf, -0.8, -0.4, -0.1, 0.1, 0.4, 0.8, np.inf]
N_STRATA      = len(STRATA_BOUNDS) - 1
 
# CDF values at each boundary
CDF_LO = np.array([norm.cdf(b) for b in STRATA_BOUNDS[:-1]])  # Phi(left edge)
CDF_HI = np.array([norm.cdf(b) for b in STRATA_BOUNDS[1:]])   # Phi(right edge)
P_K    = CDF_HI - CDF_LO                                       # stratum probabilities
 
# Intermediate monitoring times: 1/12, 2/12, ..., 11/12 (excludes t=0 and t=T)
T_INTER = np.arange(1, m) * dt
 
def sample_B1_in_stratum(n_k, k):
    """
    Draw n_k samples of B_1 uniformly within stratum k using inverse CDF:
    """
    U = np.random.uniform(0, 1, n_k)
    return norm.ppf(U * (CDF_HI[k] - CDF_LO[k]) + CDF_LO[k])
 
def brownian_bridge(B1_samples):
    """
    Given terminal values B_T = B1_samples (shape (n,)), simulate intermediate
    Brownian motion values via the SEQUENTIAL bridge formula.
    """
    n  = len(B1_samples)
    t  = np.arange(0, m + 1) * dt     # t_0=0, t_1=1/12, ..., t_12=T
 
    B        = np.zeros((n, m + 1))
    B[:, m]  = B1_samples              # pin terminal value B_T = b
 
    # Step forward through t_1, ..., t_m-1 sequentially
    for i in range(1, m):
        s = t[i - 1]           # previous time
        u = t[i]               # current time
        # Conditional mean: 
        cond_mean = B[:, i-1] + (u - s) / (T - s) * (B1_samples - B[:, i-1])
        # Conditional std: bridge variance between s and T evaluated at u
        cond_std  = np.sqrt((u - s) * (T - u) / (T - s))
        B[:, i]   = cond_mean + cond_std * np.random.standard_normal(n)
 
    # Convert to log prices
    log_S = np.log(S0) + (r - 0.5*sigma**2)*t + sigma*B
 
    return log_S
 
 
def lookback_payoff_from_logS(log_S):
    """Undiscounted lookback payoff from log price paths."""
    return np.maximum(np.max(np.exp(log_S), axis=1) - K, 0.0)
 
def stratified_price(n_per_stratum):
    """
    Returns (price, std_error).
    """
    discount      = np.exp(-r * T)
    stratum_means = np.zeros(N_STRATA)
    stratum_vars  = np.zeros(N_STRATA)
 
    for k in range(N_STRATA):
        n_k   = int(n_per_stratum[k])
        B1    = sample_B1_in_stratum(n_k, k)
        log_S = brownian_bridge(B1)
        PT_k  = lookback_payoff_from_logS(log_S)
        stratum_means[k] = np.mean(PT_k)
        stratum_vars[k]  = np.var(PT_k, ddof=1) if n_k > 1 else 0.0
 
    price   = discount * np.sum(P_K * stratum_means)
    var_est = np.sum(P_K**2 * stratum_vars / np.array(n_per_stratum, dtype=float))
    return price, np.sqrt(var_est)
 
def part_3a():
    """
    Stratified sampling with PROPORTIONAL allocation: n_k = floor(M * p_k).
    """
    M_values = [10**4, 10**5, 10**6]
    discount  = np.exp(-r * T)
 
    print("\nPart 3(a): Stratified Sampling — Proportional Allocation\n")
    print("Stratum probabilities p_k: " + ", ".join(f"{p:.4f}" for p in P_K))
    print(f"\n{'M':<10} {'Method':<26} {'Price':>8} {'Std Dev':>10} {'CI Lower':>10} {'CI Upper':>10}")
 
    for M in M_values:
        # Standard MC for comparison
        S_paths, log_S = simulate_paths(M)
        PT, _, _, _    = compute_payoffs(S_paths, log_S)
        disc_PT  = discount * PT
        price_mc = np.mean(disc_PT)
        std_mc   = np.std(disc_PT, ddof=1) / np.sqrt(M)
        print(f"{M:<10} {'Standard MC':<26} {price_mc:>8.4f} {std_mc:>10.6f} "
              f"{price_mc - 1.96*std_mc:>10.4f} {price_mc + 1.96*std_mc:>10.4f}")
 
        # Proportional allocation
        n_prop = np.floor(M * P_K).astype(int)
        n_prop[np.argmax(P_K)] += M - np.sum(n_prop)   # assign leftover to largest stratum
 
        price_st, std_st = stratified_price(n_prop)
        print(f"{M:<10} {'Stratified (prop.)':<26} {price_st:>8.4f} {std_st:>10.6f} "
              f"{price_st - 1.96*std_st:>10.4f} {price_st + 1.96*std_st:>10.4f}")
 
def part_3b():
    """
    Stratified sampling with OPTIMAL allocation.
    n_k proportional to p_k * sigma_k, where sigma_k estimated from N0=1000 pilot paths.
    """
    N0       = 1000
    M_values = [10**4, 10**5, 10**6]
    discount = np.exp(-r * T)
 
    print("\nPart 3(b): Stratified Sampling — Optimal Allocation\n")
 
    # Pilot: estimate within-stratum std dev sigma_k
    sigma_k = np.zeros(N_STRATA)
    for k in range(N_STRATA):
        B1         = sample_B1_in_stratum(N0, k)
        log_S      = brownian_bridge(B1)
        PT_k       = lookback_payoff_from_logS(log_S)
        sigma_k[k] = np.std(PT_k, ddof=1)
 
    # print(f"Pilot sigma_k estimates (N0={N0} per stratum):")
    # for k in range(N_STRATA):
    #     lo = f"{STRATA_BOUNDS[k]:.1f}" if not np.isinf(STRATA_BOUNDS[k]) else "-inf"
    #     hi = f"{STRATA_BOUNDS[k+1]:.1f}" if not np.isinf(STRATA_BOUNDS[k+1]) else "+inf"
    #     print(f"  Stratum {k+1} ({lo}, {hi}): p_k={P_K[k]:.4f}  sigma_k={sigma_k[k]:.4f}")
 
    # Optimal weights: w_k = p_k * sigma_k / sum_j(p_j * sigma_j)
    w_k = P_K * sigma_k
    w_k = w_k / np.sum(w_k)
 
    print(f"\n{'M':<10} {'Method':<26} {'Price':>8} {'Std Dev':>10} {'CI Lower':>10} {'CI Upper':>10}")
 
    for M in M_values:
        # Standard MC for comparison
        S_paths, log_S = simulate_paths(M)
        PT, _, _, _    = compute_payoffs(S_paths, log_S)
        disc_PT  = discount * PT
        price_mc = np.mean(disc_PT)
        std_mc   = np.std(disc_PT, ddof=1) / np.sqrt(M)
        print(f"{M:<10} {'Standard MC':<26} {price_mc:>8.4f} {std_mc:>10.6f} "
              f"{price_mc - 1.96*std_mc:>10.4f} {price_mc + 1.96*std_mc:>10.4f}")
 
        # Optimal allocation
        n_opt = np.floor(M * w_k).astype(int)
        n_opt[np.argmax(w_k)] += M - np.sum(n_opt)
 
        price_st, std_st = stratified_price(n_opt)
        print(f"{M:<10} {'Stratified (optimal)':<26} {price_st:>8.4f} {std_st:>10.6f} "
              f"{price_st - 1.96*std_st:>10.4f} {price_st + 1.96*std_st:>10.4f}")

def main():

    pilot_b = part_1a()
    part_1b()
    part_1c(pilot_b)
    part_2()
    part_3a()
    part_3b()

if __name__ == "__main__":
    main()