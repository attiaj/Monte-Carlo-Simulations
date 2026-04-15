import numpy as np
from tabulate import tabulate

np.random.seed(42)

TRUE_VALUE = np.exp(-10)
LAMBDA_STAR = 0.1
K_VALUES = [3, 4, 5, 6]


def monte_carlo(M: int) -> tuple[float, float]: # Standard Monte Carlo
    X = np.random.exponential(scale=1.0, size=M) # Draw from Exp(1)
    indicators = (X > 10).astype(float)
    estimate = indicators.mean() # Proportion of draws > 10
    std_err = indicators.std() / np.sqrt(M)
    return estimate, std_err


def importance_sampling(M: int, lam: float = LAMBDA_STAR) -> tuple[float, float]: # Importance sampling
    X = np.random.exponential(scale=1.0 / lam, size=M)   # Samples from g, exponential helper function with mean 10
    weights = np.exp(-(1 - lam) * X) / lam               # f(x)/g(x)
    contributions = (X > 10).astype(float) * weights      # h(x)*w(x)
    estimate = contributions.mean()
    std_err = contributions.std() / np.sqrt(M)
    return estimate, std_err



print(f"\nTrue value: P[X > 10] = e^{{-10}} = {TRUE_VALUE:.6e}\n")
print(f"IS proposal: g ~ Exp(λ* = {LAMBDA_STAR})  [mean = {1/LAMBDA_STAR:.1f}]\n")

rows_mc, rows_is = [], []

for k in K_VALUES: # Run simulations
    M = 10 ** k
    mc_est,  mc_se  = monte_carlo(M)
    is_est,  is_se  = importance_sampling(M)

    mc_re = mc_se  / TRUE_VALUE   # Relative std error
    is_re = is_se  / TRUE_VALUE

    rows_mc.append([f"10^{k}", M, f"{mc_est:.4e}", f"{mc_se:.4e}", f"{mc_re*100:.2f}%", f"{abs(mc_est - TRUE_VALUE)/TRUE_VALUE*100:.2f}%"])
    rows_is.append([f"10^{k}", M, f"{is_est:.4e}", f"{is_se:.4e}", f"{is_re*100:.2f}%", f"{abs(is_est - TRUE_VALUE)/TRUE_VALUE*100:.2f}%"])

headers = ["M", "Samples", "Estimate", "Std Error", "Rel Std Err", "Rel Bias"]

print("=" * 72)
print("i)  STANDARD MONTE CARLO")
print("=" * 72)
print(tabulate(rows_mc, headers=headers, tablefmt="rounded_outline"))

print("\n" + "=" * 72)
print("ii) IMPORTANCE SAMPLING  (proposal: Exp(λ* = 0.1))")
print("=" * 72)
print(tabulate(rows_is, headers=headers, tablefmt="rounded_outline"))