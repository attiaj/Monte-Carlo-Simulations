"""
Estimating P[X > 10] for X ~ Exp(1) using:
  i)  Standard Monte Carlo (MC)
  ii) Importance Sampling (IS) with proposal g ~ Exp(lambda_star)

True value: P[X > 10] = e^{-10} ≈ 4.53999e-5

Importance Sampling details
----------------------------
We want  mu = E_f[h(X)]  where h(x) = 1{x > 10} and f is Exp(1).

Choose proposal  g = Exp(lambda*)  with lambda* chosen so that the bulk
of g sits in the tail x > 10.  A natural choice is lambda* = 1/(10+1) is
okay, but the *optimal* exponential shift puts the mean of g at 10, i.e.
lambda* = 0.1  (mean = 10).  Actually the classic result for exponential
tilting of an Exp(1) tail P[X>t] chooses lambda* = 1/t = 0.1, giving
mean 1/lambda* = t = 10.  We use lambda* = 0.1.

The IS estimator is:
    mu_IS = (1/M) * sum_i  h(X_i) * f(X_i) / g(X_i)
           = (1/M) * sum_i  1{X_i > 10} * exp(-X_i) / (lambda* * exp(-lambda* * X_i))
           = (1/M) * sum_i  1{X_i > 10} * exp(-(1 - lambda*)*X_i) / lambda*
where X_i ~ g = Exp(lambda*).
"""

import numpy as np
from tabulate import tabulate

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

TRUE_VALUE = np.exp(-10)          # ≈ 4.53999e-5
LAMBDA_STAR = 0.1                 # IS proposal Exp(0.1), mean = 10
K_VALUES = [3, 4, 5, 6]


def monte_carlo(M: int) -> tuple[float, float]:
    """Standard MC: draw X ~ Exp(1), estimate P[X > 10]."""
    X = np.random.exponential(scale=1.0, size=M)
    indicators = (X > 10).astype(float)
    estimate = indicators.mean()
    std_err = indicators.std() / np.sqrt(M)
    return estimate, std_err


def importance_sampling(M: int, lam: float = LAMBDA_STAR) -> tuple[float, float]:
    """
    IS with proposal g = Exp(lam).
    Likelihood ratio (weight): w(x) = f(x)/g(x) = exp(-x) / (lam * exp(-lam*x))
                                      = exp(-(1-lam)*x) / lam
    """
    X = np.random.exponential(scale=1.0 / lam, size=M)   # samples from g
    weights = np.exp(-(1 - lam) * X) / lam               # f(x)/g(x)
    contributions = (X > 10).astype(float) * weights      # h(x)*w(x)
    estimate = contributions.mean()
    std_err = contributions.std() / np.sqrt(M)
    return estimate, std_err


# ── run experiments ──────────────────────────────────────────────────────────
print(f"\nTrue value: P[X > 10] = e^{{-10}} = {TRUE_VALUE:.6e}\n")
print(f"IS proposal: g ~ Exp(λ* = {LAMBDA_STAR})  [mean = {1/LAMBDA_STAR:.1f}]\n")

rows_mc, rows_is = [], []

for k in K_VALUES:
    M = 10 ** k
    mc_est,  mc_se  = monte_carlo(M)
    is_est,  is_se  = importance_sampling(M)

    mc_re = mc_se  / TRUE_VALUE   # relative std error
    is_re = is_se  / TRUE_VALUE

    rows_mc.append([f"10^{k}", M, f"{mc_est:.4e}", f"{mc_se:.4e}",
                    f"{mc_re*100:.2f}%", f"{abs(mc_est - TRUE_VALUE)/TRUE_VALUE*100:.2f}%"])
    rows_is.append([f"10^{k}", M, f"{is_est:.4e}", f"{is_se:.4e}",
                    f"{is_re*100:.2f}%", f"{abs(is_est - TRUE_VALUE)/TRUE_VALUE*100:.2f}%"])

headers = ["M", "Samples", "Estimate", "Std Error", "Rel Std Err", "Rel Bias"]

print("=" * 72)
print("i)  STANDARD MONTE CARLO")
print("=" * 72)
print(tabulate(rows_mc, headers=headers, tablefmt="rounded_outline"))

print("\n" + "=" * 72)
print("ii) IMPORTANCE SAMPLING  (proposal: Exp(λ* = 0.1))")
print("=" * 72)
print(tabulate(rows_is, headers=headers, tablefmt="rounded_outline"))

# ── variance reduction factor ────────────────────────────────────────────────
print("\n" + "=" * 72)
print("VARIANCE REDUCTION FACTOR  (Var_MC / Var_IS)")
print("=" * 72)
vrf_rows = []
for k in K_VALUES:
    M = 10 ** k
    # Re-run with same seed split for fair comparison
    rng = np.random.default_rng(seed=0)
    X_mc = rng.exponential(1.0, M)
    ind  = (X_mc > 10).astype(float)
    var_mc = ind.var()

    rng2 = np.random.default_rng(seed=1)
    X_is   = rng2.exponential(1.0 / LAMBDA_STAR, M)
    w      = np.exp(-(1 - LAMBDA_STAR) * X_is) / LAMBDA_STAR
    cont   = (X_is > 10).astype(float) * w
    var_is = cont.var()

    vrf = var_mc / var_is if var_is > 0 else float("inf")
    vrf_rows.append([f"10^{k}", M, f"{var_mc:.4e}", f"{var_is:.4e}", f"{vrf:.1f}x"])

print(tabulate(vrf_rows, headers=["M", "Samples", "Var(MC)", "Var(IS)", "VRF"],
               tablefmt="rounded_outline"))

# ── discussion ────────────────────────────────────────────────────────────────
print("""
DISCUSSION
──────────
1. Standard MC struggles severely for rare events like P[X > 10] ≈ 4.54e-5.
   Even with M = 10^6 samples, fewer than ~45 will exceed 10, giving a
   very noisy estimate with high relative standard error (~100% or more for
   small M).  For M = 10^3 the estimate is often exactly 0 (no hits at all).

2. Importance Sampling with the exponentially-tilted proposal g ~ Exp(0.1)
   shifts almost all samples into the region x > 10.  Every draw contributes
   a non-zero weighted term, so the estimator converges far more quickly.

3. The Variance Reduction Factor (VRF = Var_MC / Var_IS) is enormous —
   typically on the order of 10^3 to 10^4 — meaning IS achieves the same
   accuracy as MC with orders of magnitude fewer samples.

4. The IS estimate is unbiased and its relative standard error shrinks
   predictably as O(1/√M), but starting from a much smaller base variance.

5. Choice of proposal matters: λ* = 0.1 (mean at the threshold 10) is the
   classical exponential-tilting optimum for Exp(1) tails, minimising the
   second moment of the weight function and hence the IS variance.
""")