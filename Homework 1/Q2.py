import numpy as np

np.random.seed(15)

N = 1000       # samples per replication
reps = 10**4  # number of independent means

# True mean for Uniform[0,1]
true_mean = 0.5

# Generate all data: shape (reps, N)
Z = np.random.uniform(0, 1, size=(reps, N))

# Compute sample means and sample std because for CLT you use sample not population
# ddof controls how denominator is calculated
# Z bar is an array of all replications
Z_bar = np.mean(Z, axis=1)
sN = np.std(Z, axis=1, ddof=1)

# Compute 95% confidence interval
lower = Z_bar - 1.96 * sN / np.sqrt(N)
upper = Z_bar + 1.96 * sN / np.sqrt(N)

# Using our hint to see if true mean is inside CI using boolean
inside = (lower <= true_mean) & (true_mean <= upper)

# Mean of indicators
fraction_inside = np.mean(inside)
print("Uniform[0,1]: Fraction of 95% CIs containing true mean =", fraction_inside)


# Part 2
# True mean for Exponential(1) with lambda=1; 1/lambda=1
true_mean_exp = 1

# Generate all data: shape (reps, N)
Z_exp = np.random.exponential(1, size=(reps, N))

# Compute sample means and sample std along axis=1
Z_bar_exp = np.mean(Z_exp, axis=1)       
sN_exp = np.std(Z_exp, axis=1, ddof=1) 

# Compute 95% confidence intervals the same way as Part 1
lower_exp = Z_bar_exp - 1.96 * sN_exp / np.sqrt(N)
upper_exp = Z_bar_exp + 1.96 * sN_exp / np.sqrt(N)

# Check if true mean is inside each CI using the array of Z bar
inside_exp = (lower_exp <= true_mean_exp) & (true_mean_exp <= upper_exp)

# Fraction of CIs containing true mean
fraction_inside_exp = np.mean(inside_exp)
print("Exponential(1): Fraction of 95% CIs containing true mean =", fraction_inside_exp)