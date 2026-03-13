"""
Variance Gamma Model - Monte Carlo Option Pricing
Problem #5

Model: S_t = S_0 * exp((r + w)*t + W_{gamma_t})
where:
  - W_t = mu*t + sigma*B_t  (Brownian motion with drift)
  - gamma_t is a Gamma process with shape a = 1/nu, scale b = nu
  - w = a * log(1 - mu*nu - sigma^2*nu/2)  (martingale correction)
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings('ignore')

# ── Helpers ──────────────────────────────────────────────────────────────────

def simulate_VG(S0, r, mu, sigma, nu, T, N_steps, N_paths, seed=42):
    """
    Simulate terminal (and intermediate) asset prices under the VG model.
    Uses the Gamma time-change representation.

    Returns array of shape (N_paths,) of S_T values.
    """
    rng = np.random.default_rng(seed)
    dt = T / N_steps

    # Gamma process parameters
    a = 1.0 / nu
    b = nu

    # Martingale correction
    w = a * np.log(1.0 - mu * nu - 0.5 * sigma**2 * nu)

    # Simulate increments of Gamma process: each increment ~ Gamma(a*dt, b)
    shape_inc = a * dt
    scale_inc = b
    dGamma = rng.gamma(shape=shape_inc, scale=scale_inc, size=(N_paths, N_steps))

    # Simulate increments of W_{gamma}: given dGamma_i, dW ~ N(mu*dGamma_i, sigma^2*dGamma_i)
    Z = rng.standard_normal(size=(N_paths, N_steps))
    dW = mu * dGamma + sigma * np.sqrt(dGamma) * Z

    # Log-price increments
    log_increments = (r + w) * dt + dW

    # Sum over time steps for terminal log-price
    log_ST = np.log(S0) + np.sum(log_increments, axis=1)
    return np.exp(log_ST)


def simulate_VG_path(S0, r, mu, sigma, nu, T1, T2, N_steps1, N_steps2, N_paths, seed=42):
    """
    Simulate paths up to T1 and T2 for the reset strike option.
    Returns (S_T1, S_T2) arrays of shape (N_paths,).
    """
    rng = np.random.default_rng(seed)
    a = 1.0 / nu
    b = nu
    w = a * np.log(1.0 - mu * nu - 0.5 * sigma**2 * nu)

    total_steps = N_steps1 + N_steps2
    dt1 = T1 / N_steps1
    dt2 = (T2 - T1) / N_steps2

    # Segment 1: 0 -> T1
    dGamma1 = rng.gamma(shape=a * dt1, scale=b, size=(N_paths, N_steps1))
    Z1 = rng.standard_normal(size=(N_paths, N_steps1))
    dW1 = mu * dGamma1 + sigma * np.sqrt(dGamma1) * Z1
    log_inc1 = (r + w) * dt1 + dW1
    log_ST1 = np.log(S0) + np.sum(log_inc1, axis=1)

    # Segment 2: T1 -> T2
    dGamma2 = rng.gamma(shape=a * dt2, scale=b, size=(N_paths, N_steps2))
    Z2 = rng.standard_normal(size=(N_paths, N_steps2))
    dW2 = mu * dGamma2 + sigma * np.sqrt(dGamma2) * Z2
    log_inc2 = (r + w) * dt2 + dW2
    log_ST2 = log_ST1 + np.sum(log_inc2, axis=1)

    return np.exp(log_ST1), np.exp(log_ST2)


def bs_call_price(S, K, r, T, sigma):
    """Black-Scholes call price."""
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, r, T, tol=1e-6):
    """Invert Black-Scholes to get implied volatility."""
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + tol:
        return np.nan
    try:
        iv = brentq(lambda v: bs_call_price(S, K, r, T, v) - price,
                    1e-6, 10.0, xtol=tol)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


# ── Problem 1: Implied Volatility Smile ──────────────────────────────────────

def problem1():
    print("=" * 60)
    print("PROBLEM 1: European Call Option Pricing & Implied Vol")
    print("=" * 60)

    S0   = 100
    T    = 1.0
    r    = 0.03
    N_MC = 20_000
    dt   = 1.0 / (24 * 365)
    N_steps = int(T / dt)          # 8760 steps
    strikes = np.arange(90, 121)

    mus    = [-0.2, 0.0, 0.2]
    sigmas = [0.1, 0.3, 0.5]
    nus    = [0.2, 0.4, 0.6]

    # ── Figure layout: 3x3 grid, one panel per (mu, sigma), curves = nu ──
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig.suptitle("VG Model – Implied Volatility Smile\n"
                 "Rows: σ ∈ {0.1, 0.3, 0.5}, Cols: μ ∈ {-0.2, 0, 0.2}",
                 fontsize=14)

    colors = {0.2: 'steelblue', 0.4: 'darkorange', 0.6: 'green'}

    for (i_s, sigma), (i_m, mu) in itertools.product(
            enumerate(sigmas), enumerate(mus)):
        ax = axes[i_s, i_m]

        for nu in nus:
            ST = simulate_VG(S0, r, mu, sigma, nu, T, N_steps, N_MC)
            discount = np.exp(-r * T)

            ivs = []
            for K in strikes:
                payoffs = np.maximum(ST - K, 0.0)
                price   = discount * np.mean(payoffs)
                iv = implied_vol(price, S0, K, r, T)
                ivs.append(iv * 100 if not np.isnan(iv) else np.nan)

            ax.plot(strikes, ivs, label=f'ν={nu}', color=colors[nu], linewidth=1.8)

        ax.set_title(f'μ={mu}, σ={sigma}', fontsize=10)
        ax.set_xlabel('Strike K')
        ax.set_ylabel('Impl. Vol (%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        moneyness = strikes / S0
        ax.axvline(S0, color='gray', linestyle='--', linewidth=0.8)

    plt.tight_layout()
    plt.show()
    #plt.savefig('/problem1_implied_vol.png', dpi=150, bbox_inches='tight')
    plt.close()
    #print("  → Saved: problem1_implied_vol.png")

    # ── Summary statistics for discussion ──
    print("\n  Sample implied vols at ATM (K=100) for selected params:")
    print(f"  {'mu':>6} {'sigma':>6} {'nu':>6} {'IV(%)':>8}")
    for mu, sigma, nu in [(0.0, 0.3, 0.4), (-0.2, 0.3, 0.4), (0.2, 0.3, 0.4),
                          (0.0, 0.1, 0.4), (0.0, 0.5, 0.4), (0.0, 0.3, 0.2),
                          (0.0, 0.3, 0.6)]:
        ST = simulate_VG(S0, r, mu, sigma, nu, T, N_steps, N_MC)
        price = np.exp(-r * T) * np.mean(np.maximum(ST - 100, 0.0))
        iv = implied_vol(price, S0, 100, r, T)
        print(f"  {mu:>6.1f} {sigma:>6.1f} {nu:>6.1f} {iv*100:>8.2f}")


# ── Problem 2: Reset Strike Put Option ───────────────────────────────────────

def problem2():
    print("\n" + "=" * 60)
    print("PROBLEM 2: Reset Strike Put Option")
    print("=" * 60)

    S0    = 100.0
    r     = 0.05
    mu    = 0.1
    sigma = 0.3
    nu    = 0.3       # sigma = 0.3 = nu
    T1    = 0.25
    T2    = 0.50
    K     = 100.0
    N_MC  = 20_000
    dt    = 1.0 / (24 * 365)

    N_steps1 = int(T1 / dt)          # steps from 0 to T1
    N_steps2 = int((T2 - T1) / dt)   # steps from T1 to T2

    print(f"\n  Parameters: S0={S0}, r={r}, μ={mu}, σ={sigma}, ν={nu}")
    print(f"  T1={T1}, T2={T2}, K={K}")
    print(f"  Simulating {N_MC:,} paths with Δt=1/(24×365)...")

    ST1, ST2 = simulate_VG_path(S0, r, mu, sigma, nu, T1, T2,
                                N_steps1, N_steps2, N_MC)

    # Payoff: (max(K, S_T1) - S_T2)^+
    reset_strike = np.maximum(K, ST1)
    payoffs = np.maximum(reset_strike - ST2, 0.0)
    discount = np.exp(-r * T2)
    price = discount * np.mean(payoffs)
    std_err = discount * np.std(payoffs) / np.sqrt(N_MC)
    ci_low  = price - 1.96 * std_err
    ci_high = price + 1.96 * std_err

    print(f"\n  Reset Strike Put Price : {price:.4f}")
    print(f"  Standard Error         : {std_err:.4f}")
    print(f"  95% CI                 : [{ci_low:.4f}, {ci_high:.4f}]")

    # ── Diagnostics plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Problem 2: Reset Strike Put – VG Model", fontsize=13)

    ax = axes[0]
    ax.hist(ST1, bins=80, alpha=0.6, color='steelblue', label=f'$S_{{T_1}}$')
    ax.hist(ST2, bins=80, alpha=0.6, color='darkorange', label=f'$S_{{T_2}}$')
    ax.axvline(K, color='red', linestyle='--', label=f'K={K}')
    ax.set_title('Distribution of $S_{T_1}$ and $S_{T_2}$')
    ax.set_xlabel('Asset Price')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(payoffs[payoffs > 0], bins=80, color='green', alpha=0.7)
    ax.set_title(f'Distribution of Positive Payoffs\n'
                 f'(% in-the-money: {100*np.mean(payoffs>0):.1f}%)')
    ax.set_xlabel('Payoff')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    #plt.savefig('/problem2_reset_put.png', dpi=150, bbox_inches='tight')
    plt.close()
    #print("  → Saved: problem2_reset_put.png")

    return price, std_err


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    problem1()
    price, se = problem2()

    print("\n" + "=" * 60)
    print("DISCUSSION NOTES")
    print("=" * 60)
    print("""
Problem 1 – Implied Volatility Smile:
  • σ (diffusion vol): Primary driver of the overall IV level. Higher σ
    raises the entire smile. When σ is small, the VG jump component
    dominates and the smile is steep; when σ is large, the smile flattens.

  • μ (drift of BM): Controls skewness. Negative μ tilts the distribution
    left → higher IV for low strikes (put skew). Positive μ tilts right
    → higher IV for high strikes. This is the main lever for controlling
    the risk-neutral skew.

  • ν (variance rate of Gamma clock): Controls kurtosis / curvature.
    Larger ν means more "time-change" variance → fatter tails → more
    pronounced smile (higher IV at wings relative to ATM). Small ν
    produces a nearly flat smile, converging toward Black-Scholes.

Problem 2 – Reset Strike Put:
  The option benefits when S_{T1} < K (strike resets to K) but S_{T2}
  is also low, or when S_{T1} > K and then S_{T2} falls below S_{T1}.
  The VG model introduces jump-like behaviour and skewness that raises
  the probability of large adverse moves, typically pricing this exotic
  higher than a comparable GBM model.
""")