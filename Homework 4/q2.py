"""
Problem #2 — Portfolio Risk: Delta-Hedge Monte Carlo Analysis

Portfolio value at time t:
  V(t, x) = C(t, x) - Delta*x - delta_cash * e^(rt)

  Risk measures (all at level alpha = 0.05):
  rho_EL  = -E[Z]          Expected Loss
  rho_VaR = VaR_0.05(Z)    5th percentile of -Z  (= -5th pct of Z)
  rho_ES  = ES_0.05(Z)     mean of -Z given Z <= VaR quantile of Z
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
S0    = 100.0
K     = S0      
T     = 1.0
r     = 0.05
mu    = 0.10
sigma = 0.40
M     = 100_000
alpha = 0.05 
np.random.seed(42)

# Black-Scholes helpers 
def bs_call(S, K, r, sigma, tau):
    "tau = time to maturity"
    if tau <= 0:
        return np.maximum(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau)) # From Class
    d2 = d1 - sigma * np.sqrt(tau) # From Class
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

def bs_delta(S, K, r, sigma, tau):
    "BS delta = N(d1)"
    if tau <= 0:
        return (S > K).astype(float)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)

def bs_gamma(S, K, r, sigma, tau):
    """BS gamma = n(d1) / (S sigma sqrt(tau))."""
    if tau <= 0:
        return np.zeros_like(S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.pdf(d1) / (S * sigma * np.sqrt(tau))

# Risk-neutral pricing for deriv
C0         = bs_call(S0, K, r, sigma, T)
Delta      = bs_delta(S0, K, r, sigma, T)
Gamma      = bs_gamma(S0, K, r, sigma, T)
delta_cash = Delta * S0 - C0 

print("=" * 60)
print("  Black-Scholes initial quantities")
print("=" * 60)
print(f"  C(0, S0)       = {C0:.4f}")
print(f"  Delta          = {Delta:.4f}")
print(f"  Gamma          = {Gamma:.6f}")
print(f"  delta_cash     = {delta_cash:.4f}  (δ < 0 → borrow cash)")
print()

def expected_loss(Z):
    return -np.mean(Z)

def var_alpha(Z, a=alpha):
    # VaR level
    return -np.quantile(Z, a)

def es_alpha(Z, a=alpha):
    # Expected Shortfall
    q = np.quantile(Z, a)
    tail = Z[Z <= q]
    return -np.mean(tail)

# MC simulation 
# Draw standard normals once; reuse for both periods
W = np.random.randn(M)

results = {}

for i in [1, 2]:
    t   = T / i                              # evaluation time
    tau = T - t                              # time to maturity

    # Physical-world stock price at time t
    St  = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * W)
    dS  = St - S0

    # Now for questions
    # a) Linear approximation of dC
    dC_lin  = Delta * dS

    # b) Quadratic (2nd-order Taylor) approximation of dC
    dC_quad = Delta * dS + 0.5 * Gamma * dS**2

    # c) Exact Black-Scholes call price at (t, St)
    dC_exact = bs_call(St, K, r, sigma, tau) - C0

    # Change in risk-free bond position
    bond_gain = delta_cash * (np.exp(r * t) - 1)

    # Z_i for each model (Given Formula)
    # V(t,St) - V(0,S0) = [C(t,St) - Delta*St - delta_cash*e^{rt}]
    #                     - [C(0,S0) - Delta*S0 - delta_cash]
    #                   = dC - Delta*dS - delta_cash*(e^{rt}-1)
    for model, dC in [('linear', dC_lin), ('quadratic', dC_quad), ('exact', dC_exact)]:
        Z = dC - Delta * dS - bond_gain
        results[(i, model)] = Z

# Summary
header = f"{'i':>2}  {'Model':<12}  {'E[Z]':>9}  {'EL=−E[Z]':>9}  {'VaR5%':>9}  {'ES5%':>9}"
print("=" * 60)
print("  Risk measures  (α = 5%)")
print("=" * 60)
print(header)
print("-" * 60)
for i in [1, 2]:
    for model in ['linear', 'quadratic', 'exact']:
        Z  = results[(i, model)]
        EZ = np.mean(Z)
        EL = expected_loss(Z)
        VaR = var_alpha(Z)
        ES  = es_alpha(Z)
        print(f"  {i}  {model:<12}  {EZ:>9.4f}  {EL:>9.4f}  {VaR:>9.4f}  {ES:>9.4f}")
    print()

# Plot empirical CDFs
colors  = {'linear': '#185FA5', 'quadratic': '#0F6E56', 'exact': '#993C1D'}
lstyles = {'linear': '-',       'quadratic': '--',      'exact': ':'}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.suptitle("Empirical CDF of $Z_i$ — Delta-hedge portfolio P&L", fontsize=13, y=.98)

for ax, i in zip(axes, [1, 2]):
    for model in ['linear', 'quadratic', 'exact']:
        Z    = results[(i, model)]
        Zs   = np.sort(Z)
        cdf  = np.arange(1, M + 1) / M
        VaR  = var_alpha(Z)

        ax.plot(Zs, cdf,
                color=colors[model], linestyle=lstyles[model], linewidth=1.8,
                label=model.capitalize())

        # VaR (5th percentile of Z = left tail)
        q05 = np.quantile(Z, alpha)
        ax.axvline(q05, color=colors[model], linestyle=':', alpha=0.45, linewidth=1)

    ax.axhline(alpha, color='#888', linestyle='--', linewidth=0.9, label=f'α = {alpha}')
    ax.set_title(f"$Z_{i}$  (t = T/{i} = {T/i:.1f})", fontsize=11)
    ax.set_xlabel("$Z_i$", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(np.quantile(results[(i, 'exact')], 0.001),
                np.quantile(results[(i, 'exact')], 0.999))

plt.tight_layout()
plt.savefig("empirical_cdf.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to  empirical_cdf.png")
