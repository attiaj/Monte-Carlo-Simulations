import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tabulate import tabulate

np.random.seed(42)

S0    = 100.0
r     = 0.05
sigma = 0.3
T1    = 0.5
T2    = 1.0
tau   = T2 - T1          # = 0.5

K_VALUES = [3, 4, 5, 6, 7]

# Calculate price analytically using B-S formula
d1_an = (r * tau + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
d2_an = d1_an - sigma * np.sqrt(tau)
ANALYTICAL_PRICE = S0 * (norm.cdf(d1_an) - np.exp(-r * tau) * norm.cdf(d2_an))

print(f"\nAnalytical price = S0*[Phi(d1) - e^{{-r*tau}}*Phi(d2)]")
print(f"  d1 = {d1_an:.6f},  d2 = {d2_an:.6f}")
print(f"  Price = {ANALYTICAL_PRICE:.6f}\n")

def bs_call_cmc(s: np.ndarray) -> np.ndarray: # Calculate call price using B-S to reduce variance with CMC
    return s * np.exp(-r * T1) * (norm.cdf(d1_an) - np.exp(-r * tau) * norm.cdf(d2_an))

def standard_mc(M: int): # Standard Monte Carlo
    Z1 = np.random.randn(M)
    Z2 = np.random.randn(M)
    S_T1 = S0    * np.exp((r - 0.5*sigma**2)*T1  + sigma*np.sqrt(T1) *Z1)
    S_T2 = S_T1  * np.exp((r - 0.5*sigma**2)*tau + sigma*np.sqrt(tau)*Z2)
    payoff = np.exp(-r * T2) * np.maximum(S_T2 - S_T1, 0.0)
    est  = payoff.mean()
    se   = payoff.std(ddof=1) / np.sqrt(M)
    return est, se, payoff.var(ddof=1)

def conditional_mc(M: int): # Conditional Monte Carlo
    Z1   = np.random.randn(M)
    S_T1 = S0 * np.exp((r - 0.5*sigma**2)*T1 + sigma*np.sqrt(T1)*Z1)

    cv   = bs_call_cmc(S_T1)
    est  = cv.mean()
    se   = cv.std(ddof=1) / np.sqrt(M)
    return est, se, cv.var(ddof=1)

results_mc  = []
results_cmc = []

for k in K_VALUES: # Run simulations
    M = 10**k
    mc_est,  mc_se,  mc_var  = standard_mc(M)
    cmc_est, cmc_se, cmc_var = conditional_mc(M)
    results_mc.append( (k, M, mc_est,  mc_se,  mc_var))
    results_cmc.append((k, M, cmc_est, cmc_se, cmc_var))

def fmt_row(k, M, est, se, var):
    ci_lo = est - 1.96*se
    ci_hi = est + 1.96*se
    return [f"10^{k}", M, f"{est:.5f}",
            f"[{ci_lo:.5f}, {ci_hi:.5f}]",
            f"{se:.2e}", f"{var:.4e}"]

headers = ["M", "Samples", "Estimate", "95% CI", "Std Error", "Variance"]

print("=" * 80)
print(f"ANALYTICAL PRICE: {ANALYTICAL_PRICE:.6f}")
print("=" * 80)

print("\n── i) STANDARD MONTE CARLO ──")
print(tabulate([fmt_row(*r) for r in results_mc],  headers=headers, tablefmt="rounded_outline"))

print("\n── ii) CONDITIONAL MONTE CARLO ──")
print(tabulate([fmt_row(*r) for r in results_cmc], headers=headers, tablefmt="rounded_outline"))

print("\n── VARIANCE REDUCTION FACTOR (Var_MC / Var_CMC) ──")
vrf_rows = []
for (k, M, _, _, vmc), (_, _, _, _, vcmc) in zip(results_mc, results_cmc):
    vrf_rows.append([f"10^{k}", M, f"{vmc:.4e}", f"{vcmc:.4e}", f"{vmc/vcmc:.1f}x"])
print(tabulate(vrf_rows, headers=["M","Samples","Var(MC)","Var(CMC)","VRF"], tablefmt="rounded_outline"))

ks      = [r[0] for r in results_mc]
ms      = [r[1] for r in results_mc]
mc_ests  = np.array([r[2] for r in results_mc])
mc_ses   = np.array([r[3] for r in results_mc])
cmc_ests = np.array([r[2] for r in results_cmc])
cmc_ses  = np.array([r[3] for r in results_cmc])

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("Forward-Start Call Option: MC vs Conditional MC Convergence", fontsize=13, fontweight="bold")

for ax, ests, ses, label, color in [
    (axes[0], mc_ests,  mc_ses,  "Standard MC",    "#2196F3"),
    (axes[1], cmc_ests, cmc_ses, "Conditional MC",  "#4CAF50"),
]:
    ci_lo = ests - 1.96*ses
    ci_hi = ests + 1.96*ses
    x = range(len(ks))

    ax.fill_between(x, ci_lo, ci_hi, alpha=0.25, color=color, label="95% CI")
    ax.plot(x, ests, "o-", color=color, linewidth=2, markersize=7, label=label)
    ax.axhline(ANALYTICAL_PRICE, color="red", linestyle="--", linewidth=1.5, label=f"Analytical = {ANALYTICAL_PRICE:.4f}")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"$10^{k}$" for k in ks])
    ax.set_xlabel("Number of paths M", fontsize=11)
    ax.set_ylabel("Price estimate", fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Homework 3/forward_start_convergence.png", dpi=150, bbox_inches="tight")

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.loglog(ms, mc_ses,  "o-", color="#2196F3", linewidth=2, markersize=7, label="MC std error")
ax2.loglog(ms, cmc_ses, "s-", color="#4CAF50", linewidth=2, markersize=7, label="CMC std error")

ref_m = np.array(ms, dtype=float)
ax2.loglog(ref_m, mc_ses[0] * np.sqrt(ms[0]/ref_m), "k--", alpha=0.4, label=r"$O(1/\sqrt{M})$")

ax2.set_xlabel("M (log scale)", fontsize=11)
ax2.set_ylabel("Standard Error (log scale)", fontsize=11)
ax2.set_title("Std Error Convergence: MC vs Conditional MC", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("Homework 3/forward_start_stderr.png", dpi=150, bbox_inches="tight")