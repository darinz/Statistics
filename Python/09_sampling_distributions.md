# Sampling and Sampling Distributions

## Overview

Sampling distributions are fundamental to statistical inference and form the bridge between probability theory and practical data analysis. They describe how sample statistics vary across different samples from the same population, providing the mathematical foundation for hypothesis testing, confidence intervals, and statistical modeling.

### The Importance of Sampling Distributions

Sampling distributions serve several critical functions:

1. **Statistical Inference**: Enable us to make probabilistic statements about population parameters
2. **Hypothesis Testing**: Provide the theoretical basis for testing statistical hypotheses
3. **Confidence Intervals**: Allow us to quantify uncertainty in our estimates
4. **Sample Size Planning**: Help determine appropriate sample sizes for desired precision

### Key Concepts

- **Population**: The complete set of individuals or objects of interest
- **Sample**: A subset of the population selected for study
- **Parameter**: A numerical characteristic of the population (e.g., population mean $\mu$)
- **Statistic**: A numerical characteristic of the sample (e.g., sample mean $\bar{X}$)
- **Sampling Distribution**: The probability distribution of a statistic across all possible samples

## Mathematical Foundations

### Expected Value and Variance of Sample Statistics

For a random sample of size $n$ from a population with mean $\mu$ and variance $\sigma^2$:

```math
\begin{align}
\text{Expected Value of Sample Mean:} & \quad E[\bar{X}] = \mu \\
\text{Variance of Sample Mean:} & \quad \text{Var}(\bar{X}) = \frac{\sigma^2}{n} \\
\text{Standard Error of Sample Mean:} & \quad \text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}
\end{align}
```

### Sampling Distribution Properties

The sampling distribution of the mean has these key properties:

```math
\begin{align}
\text{Mean:} & \quad \mu_{\bar{X}} = \mu \\
\text{Variance:} & \quad \sigma^2_{\bar{X}} = \frac{\sigma^2}{n} \\
\text{Standard Error:} & \quad \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}
\end{align}
```

### Central Limit Theorem

If $X_1, X_2, \ldots, X_n$ are independent and identically distributed random variables with mean $\mu$ and variance $\sigma^2$, then:

```math
\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i
```

As $n \to \infty$, the sampling distribution of $\bar{X}_n$ approaches a normal distribution:

```math
\bar{X}_n \xrightarrow{d} N\left(\mu, \frac{\sigma^2}{n}\right)
```

## Basic Sampling Concepts

### Simple Random Sampling

Simple random sampling ensures that every possible sample of size $n$ has an equal probability of being selected.

```python
# Load sample data for demonstration
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Load sample data (using iris dataset as equivalent to mtcars)
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Simple random sampling with mathematical precision
np.random.seed(123)
sample_size = 10
random_sample = data.sample(n=sample_size, random_state=123)

# View sample with mathematical context
print(f"Simple Random Sample (n = {sample_size}):")
print(random_sample)

# Calculate sample statistics with mathematical interpretation
sample_mean = random_sample['sepal length (cm)'].mean()
sample_sd = random_sample['sepal length (cm)'].std()
sample_se = sample_sd / np.sqrt(sample_size)

print(f"\nSample Statistics:")
print(f"Sample mean (x̄): {sample_mean:.4f}")
print(f"Sample standard deviation (s): {sample_sd:.4f}")
print(f"Standard error of the mean (SE): {sample_se:.4f}")

# Compare with population parameters
population_mean = data['sepal length (cm)'].mean()
population_sd = data['sepal length (cm)'].std()
population_se = population_sd / np.sqrt(sample_size)

print(f"\nPopulation Parameters:")
print(f"Population mean (μ): {population_mean:.4f}")
print(f"Population standard deviation (σ): {population_sd:.4f}")
print(f"Theoretical standard error: {population_se:.4f}")

# Calculate sampling error
sampling_error = sample_mean - population_mean
print(f"Sampling error (x̄ - μ): {sampling_error:.4f}")
```

### Systematic Sampling

Systematic sampling selects every $k$th element from the population, where $k = N/n$ and $N$ is the population size.

```python
# Systematic sampling with mathematical explanation
n = len(data)
k = n // sample_size  # Sampling interval
print(f"Systematic sampling interval (k = N/n): {k}")

# Perform systematic sampling
systematic_indices = np.arange(0, n, k)[:sample_size]
systematic_sample = data.iloc[systematic_indices]

print(f"\nSystematic Sample (n = {sample_size}):")
print(systematic_sample)

# Calculate systematic sample statistics
sys_mean = systematic_sample['sepal length (cm)'].mean()
sys_sd = systematic_sample['sepal length (cm)'].std()
sys_se = sys_sd / np.sqrt(sample_size)

print(f"\nSystematic Sample Statistics:")
print(f"Sample mean: {sys_mean:.4f}")
print(f"Sample standard deviation: {sys_sd:.4f}")
print(f"Standard error: {sys_se:.4f}")
```

### Stratified Sampling

Stratified sampling divides the population into homogeneous subgroups (strata) and samples from each stratum.

```python
# Stratified sampling by target class with mathematical precision
# Calculate sample size per stratum (proportional allocation)
strata_sizes = data.groupby('target').agg({
    'sepal length (cm)': ['count', 'mean', 'std']
}).round(4)
strata_sizes.columns = ['n', 'mean', 'std']

# Calculate proportions and sample sizes
total_n = len(data)
strata_sizes['proportion'] = strata_sizes['n'] / total_n
strata_sizes['sample_size'] = np.ceil(strata_sizes['n'] * sample_size / total_n).astype(int)

print("Stratified Sampling Allocation:")
print(strata_sizes)

# Perform stratified sampling
stratified_sample = data.groupby('target').apply(
    lambda x: x.sample(n=min(len(x), int(strata_sizes.loc[x.name, 'sample_size'])), random_state=123)
).reset_index(drop=True)

print(f"\nStratified Sample:")
print(stratified_sample)

# Calculate stratified sample statistics
strat_mean = stratified_sample['sepal length (cm)'].mean()
strat_sd = stratified_sample['sepal length (cm)'].std()
strat_se = strat_sd / np.sqrt(len(stratified_sample))

print(f"\nStratified Sample Statistics:")
print(f"Sample mean: {strat_mean:.4f}")
print(f"Sample standard deviation: {strat_sd:.4f}")
print(f"Standard error: {strat_se:.4f}")

# Compare sampling methods
print(f"\nComparison of Sampling Methods:")
print(f"Simple Random Sample mean: {sample_mean:.4f}")
print(f"Systematic Sample mean: {sys_mean:.4f}")
print(f"Stratified Sample mean: {strat_mean:.4f}")
print(f"Population mean: {population_mean:.4f}")
```

## Sampling Distribution of the Mean

### Mathematical Foundation

The sampling distribution of the mean is the probability distribution of sample means from all possible samples of size $n$ from the population.

For a population with mean $\mu$ and standard deviation $\sigma$:

```math
\begin{align}
\text{Mean of sampling distribution:} & \quad \mu_{\bar{X}} = \mu \\
\text{Standard error:} & \quad \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}} \\
\text{Shape:} & \quad \text{Approximately normal for large } n \text{ (CLT)}
\end{align}
```

### Simulating Sampling Distribution

```python
# Simulate sampling distribution of the mean with mathematical precision
np.random.seed(123)
population_mean = data['sepal length (cm)'].mean()
population_sd = data['sepal length (cm)'].std()
n_samples = 1000
sample_size = 10

print("Sampling Distribution Simulation:")
print(f"Population mean (μ): {population_mean:.4f}")
print(f"Population standard deviation (σ): {population_sd:.4f}")
print(f"Sample size (n): {sample_size}")
print(f"Number of samples: {n_samples}")

# Generate sampling distribution
sample_means = np.zeros(n_samples)
for i in range(n_samples):
    sample_data = np.random.choice(data['sepal length (cm)'], size=sample_size, replace=True)
    sample_means[i] = np.mean(sample_data)

# Analyze sampling distribution with mathematical interpretation
sampling_mean = np.mean(sample_means)
sampling_sd = np.std(sample_means)
theoretical_se = population_sd / np.sqrt(sample_size)

print(f"\nSampling Distribution Analysis:")
print(f"Mean of sampling distribution: {sampling_mean:.4f}")
print(f"Standard deviation of sampling distribution: {sampling_sd:.4f}")
print(f"Theoretical standard error (σ/√n): {theoretical_se:.4f}")
print(f"Empirical vs Theoretical SE ratio: {sampling_sd / theoretical_se:.4f}")

# Verify Central Limit Theorem properties
print(f"\nCentral Limit Theorem Verification:")
print(f"Sampling distribution mean ≈ Population mean: {abs(sampling_mean - population_mean) < 0.1}")
print(f"Sampling distribution SD ≈ σ/√n: {abs(sampling_sd - theoretical_se) < 0.1}")
```

### Visualizing Sampling Distribution

```python
# Comprehensive visualization of sampling distribution
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm

plt.figure(figsize=(15, 10))

# Histogram of sample means with normal overlay
plt.subplot(2, 2, 1)
plt.hist(sample_means, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
plt.xlabel('Sample Mean (x̄)')
plt.ylabel('Density')
plt.title('Sampling Distribution of Mean')

# Add theoretical normal curve
x = np.linspace(min(sample_means), max(sample_means), 100)
plt.plot(x, norm.pdf(x, population_mean, theoretical_se), 'r-', linewidth=2, label='Theoretical Normal')
plt.axvline(population_mean, color='red', linestyle='--', alpha=0.7)
plt.text(population_mean, plt.ylim()[1]*0.9, f'μ = {population_mean:.3f}', 
         color='red', fontweight='bold', ha='center')
plt.legend()

# Q-Q plot for normality assessment
plt.subplot(2, 2, 2)
stats.probplot(sample_means, dist="norm", plot=plt)
plt.title('Q-Q Plot of Sample Means')

# Box plot for distribution shape
plt.subplot(2, 2, 3)
plt.boxplot(sample_means, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.ylabel('Sample Mean')
plt.title('Box Plot of Sample Means')

# Density plot comparison
plt.subplot(2, 2, 4)
plt.hist(sample_means, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
plt.plot(x, norm.pdf(x, population_mean, theoretical_se), 'r--', linewidth=2, label='Theoretical Normal')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.title('Density Plot vs Normal')
plt.legend()

plt.tight_layout()
plt.show()

# Statistical normality test
shapiro_stat, shapiro_p = stats.shapiro(sample_means)
print(f"\nNormality Test for Sampling Distribution:")
print(f"Shapiro-Wilk test p-value: {shapiro_p:.6f}")
print(f"Interpretation: {'Sampling distribution appears normal' if shapiro_p > 0.05 else 'Sampling distribution may not be normal'}")
```

## Central Limit Theorem

### Mathematical Statement

The Central Limit Theorem states that for independent and identically distributed random variables $X_1, X_2, \ldots, X_n$ with mean $\mu$ and variance $\sigma^2$:

```math
\text{As } n \to \infty, \quad \frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0,1)
```

This means that regardless of the original population distribution, the sampling distribution of the mean approaches normality as sample size increases.

### Demonstrating CLT

```python
# Demonstrate Central Limit Theorem with different distributions
np.random.seed(123)

# Function to simulate sampling distribution with mathematical precision
def simulate_sampling_dist(population, sample_size, n_samples=1000):
    sample_means = np.zeros(n_samples)
    for i in range(n_samples):
        sample_data = np.random.choice(population, size=sample_size, replace=True)
        sample_means[i] = np.mean(sample_data)
    return sample_means

# Different population distributions with mathematical properties
n = 1000

# Uniform distribution: U(0,10)
uniform_pop = np.random.uniform(0, 10, n)
uniform_means = simulate_sampling_dist(uniform_pop, 30)
print("Uniform Distribution Properties:")
print(f"Population mean: {np.mean(uniform_pop):.4f} (theoretical: 5)")
print(f"Population variance: {np.var(uniform_pop):.4f} (theoretical: 8.33)")
print(f"Sampling distribution mean: {np.mean(uniform_means):.4f}")
print(f"Sampling distribution SE: {np.std(uniform_means):.4f}")
print(f"Theoretical SE: {np.sqrt(np.var(uniform_pop)/30):.4f}\n")

# Exponential distribution: Exp(0.5)
exp_pop = np.random.exponential(scale=2, size=n)  # scale = 1/rate
exp_means = simulate_sampling_dist(exp_pop, 30)
print("Exponential Distribution Properties:")
print(f"Population mean: {np.mean(exp_pop):.4f} (theoretical: 2)")
print(f"Population variance: {np.var(exp_pop):.4f} (theoretical: 4)")
print(f"Sampling distribution mean: {np.mean(exp_means):.4f}")
print(f"Sampling distribution SE: {np.std(exp_means):.4f}")
print(f"Theoretical SE: {np.sqrt(np.var(exp_pop)/30):.4f}\n")

# Skewed distribution: Gamma(2,1)
skewed_pop = np.random.gamma(shape=2, scale=1, size=n)
skewed_means = simulate_sampling_dist(skewed_pop, 30)
print("Gamma Distribution Properties:")
print(f"Population mean: {np.mean(skewed_pop):.4f} (theoretical: 2)")
print(f"Population variance: {np.var(skewed_pop):.4f} (theoretical: 2)")
print(f"Sampling distribution mean: {np.mean(skewed_means):.4f}")
print(f"Sampling distribution SE: {np.std(skewed_means):.4f}")
print(f"Theoretical SE: {np.sqrt(np.var(skewed_pop)/30):.4f}")
```

# Plot results with mathematical annotations
```python
plt.figure(figsize=(15, 12))

# Uniform distribution
plt.subplot(3, 2, 1)
plt.hist(uniform_pop, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
x_unif = np.linspace(0, 10, 100)
plt.plot(x_unif, stats.uniform.pdf(x_unif, 0, 10), 'r-', linewidth=2)
plt.xlabel('Value')
plt.title('Uniform Population U(0,10)')

plt.subplot(3, 2, 2)
plt.hist(uniform_means, bins=30, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
x_norm_unif = np.linspace(min(uniform_means), max(uniform_means), 100)
plt.plot(x_norm_unif, norm.pdf(x_norm_unif, np.mean(uniform_means), np.std(uniform_means)), 'r-', linewidth=2)
plt.xlabel('Sample Mean')
plt.title('Sampling Distribution (n=30)')

# Exponential distribution
plt.subplot(3, 2, 3)
plt.hist(exp_pop, bins=30, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
x_exp = np.linspace(0, max(exp_pop), 100)
plt.plot(x_exp, stats.expon.pdf(x_exp, scale=2), 'r-', linewidth=2)
plt.xlabel('Value')
plt.title('Exponential Population Exp(0.5)')

plt.subplot(3, 2, 4)
plt.hist(exp_means, bins=30, density=True, alpha=0.7, color='lightyellow', edgecolor='black')
x_norm_exp = np.linspace(min(exp_means), max(exp_means), 100)
plt.plot(x_norm_exp, norm.pdf(x_norm_exp, np.mean(exp_means), np.std(exp_means)), 'r-', linewidth=2)
plt.xlabel('Sample Mean')
plt.title('Sampling Distribution (n=30)')

# Skewed distribution
plt.subplot(3, 2, 5)
plt.hist(skewed_pop, bins=30, density=True, alpha=0.7, color='lightpink', edgecolor='black')
x_gamma = np.linspace(0, max(skewed_pop), 100)
plt.plot(x_gamma, stats.gamma.pdf(x_gamma, a=2, scale=1), 'r-', linewidth=2)
plt.xlabel('Value')
plt.title('Gamma Population Γ(2,1)')

plt.subplot(3, 2, 6)
plt.hist(skewed_means, bins=30, density=True, alpha=0.7, color='lightcyan', edgecolor='black')
x_norm_gamma = np.linspace(min(skewed_means), max(skewed_means), 100)
plt.plot(x_norm_gamma, norm.pdf(x_norm_gamma, np.mean(skewed_means), np.std(skewed_means)), 'r-', linewidth=2)
plt.xlabel('Sample Mean')
plt.title('Sampling Distribution (n=30)')

plt.tight_layout()
plt.show()
```

### Effect of Sample Size

```python
# Study effect of sample size on sampling distribution with mathematical precision
sample_sizes = [5, 10, 30, 50]
n_samples = 1000

print("Effect of Sample Size on Sampling Distribution:")
print("==============================================")

# Simulate for different sample sizes
sampling_distributions = {}
for size in sample_sizes:
    sample_means = np.zeros(n_samples)
    for i in range(n_samples):
        sample_data = np.random.choice(data['sepal length (cm)'], size=size, replace=True)
        sample_means[i] = np.mean(sample_data)
    sampling_distributions[str(size)] = sample_means
    
    # Calculate theoretical and empirical properties
    theoretical_se = population_sd / np.sqrt(size)
    empirical_se = np.std(sample_means)
    
    print(f"Sample size n = {size}:")
    print(f"  Theoretical SE (σ/√n): {theoretical_se:.4f}")
    print(f"  Empirical SE: {empirical_se:.4f}")
    print(f"  Ratio (Empirical/Theoretical): {empirical_se / theoretical_se:.4f}\n")
```

# Plot results with mathematical annotations
```python
plt.figure(figsize=(15, 10))

for i, size in enumerate(sample_sizes):
    plt.subplot(2, 2, i+1)
    means = sampling_distributions[str(size)]
    
    plt.hist(means, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    
    # Add theoretical normal curve
    x = np.linspace(min(means), max(means), 100)
    theoretical_mean = population_mean
    theoretical_se = population_sd / np.sqrt(size)
    plt.plot(x, norm.pdf(x, mean=theoretical_mean, scale=theoretical_se), 'r-', linewidth=2)
    
    # Add mathematical annotations
    plt.text(theoretical_mean, plt.ylim()[1]*0.9, f"SE = σ/√n", 
             color='red', fontweight='bold', ha='center')
    plt.title(f'Sample Size n = {size}')
    plt.xlabel('Sample Mean')

plt.tight_layout()
plt.show()

# Compare standard errors with mathematical precision
theoretical_ses = population_sd / np.sqrt(sample_sizes)
empirical_ses = [np.std(sampling_distributions[str(size)]) for size in sample_sizes]

comparison = pd.DataFrame({
    'Sample_Size': sample_sizes,
    'Theoretical_SE': theoretical_ses,
    'Empirical_SE': empirical_ses,
    'Ratio': np.array(empirical_ses) / theoretical_ses
})

print("Standard Error Comparison:")
print(comparison)

# Verify that SE decreases as 1/√n
print(f"\nVerification of SE ∝ 1/√n:")
for i in range(1, len(sample_sizes)):
    ratio = theoretical_ses[0] / theoretical_ses[i]
    expected_ratio = np.sqrt(sample_sizes[i] / sample_sizes[0])
    print(f"n = {sample_sizes[0]} to n = {sample_sizes[i]}: SE ratio = {ratio:.4f} (expected: {expected_ratio:.4f})")
```

## Sampling Distribution of the Proportion

### Mathematical Foundation

For a population with proportion $p$ of successes, the sampling distribution of the sample proportion $\hat{p}$ has these properties:

```math
\begin{align}
\text{Mean:} & \quad \mu_{\hat{p}} = p \\
\text{Variance:} & \quad \sigma^2_{\hat{p}} = \frac{p(1-p)}{n} \\
\text{Standard Error:} & \quad \sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}
\end{align}
```

For large samples ($np \geq 10$ and $n(1-p) \geq 10$), the sampling distribution is approximately normal:

```math
\hat{p} \sim N\left(p, \sqrt{\frac{p(1-p)}{n}}\right)
```

### Simulating Proportion Sampling

```python
# Simulate sampling distribution of proportion with mathematical precision
np.random.seed(123)
# Create a binary variable (e.g., sepal length > 5.5)
data['binary_var'] = (data['sepal length (cm)'] > 5.5).astype(int)
population_prop = data['binary_var'].mean()  # Proportion of "successes"
n_samples = 1000
sample_size = 20

print("Sampling Distribution of Proportion:")
print("==================================")
print(f"Population proportion (p): {population_prop:.4f}")
print(f"Sample size (n): {sample_size}")
print(f"Number of samples: {n_samples}")

# Verify normal approximation conditions
print(f"np = {sample_size * population_prop:.2f} (should be ≥ 10)")
print(f"n(1-p) = {sample_size * (1 - population_prop):.2f} (should be ≥ 10)")

# Generate sampling distribution of proportion
sample_proportions = np.zeros(n_samples)
for i in range(n_samples):
    sample_data = np.random.choice(data['binary_var'], size=sample_size, replace=True)
    sample_proportions[i] = np.mean(sample_data)

# Analyze sampling distribution with mathematical interpretation
sampling_prop_mean = np.mean(sample_proportions)
sampling_prop_sd = np.std(sample_proportions)
theoretical_prop_se = np.sqrt(population_prop * (1 - population_prop) / sample_size)

print(f"\nSampling Distribution Analysis:")
print(f"Mean of sampling distribution: {sampling_prop_mean:.4f}")
print(f"Standard deviation of sampling distribution: {sampling_prop_sd:.4f}")
print(f"Theoretical standard error √[p(1-p)/n]: {theoretical_prop_se:.4f}")
print(f"Empirical vs Theoretical SE ratio: {sampling_prop_sd / theoretical_prop_se:.4f}")

# Verify CLT for proportions
print(f"\nCentral Limit Theorem Verification for Proportions:")
print(f"Sampling distribution mean ≈ Population proportion: {abs(sampling_prop_mean - population_prop) < 0.01}")
print(f"Sampling distribution SD ≈ √[p(1-p)/n]: {abs(sampling_prop_sd - theoretical_prop_se) < 0.01}")

# Plot sampling distribution with mathematical annotations
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(sample_proportions, bins=30, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
plt.xlabel('Sample Proportion (p̂)')
plt.ylabel('Density')
plt.title('Sampling Distribution of Proportion')

# Add theoretical normal curve
x = np.linspace(min(sample_proportions), max(sample_proportions), 100)
plt.plot(x, norm.pdf(x, population_prop, theoretical_prop_se), 'r-', linewidth=2, label='Theoretical Normal')
plt.axvline(population_prop, color='red', linestyle='--', alpha=0.7)
plt.text(population_prop, plt.ylim()[1]*0.9, f'p = {population_prop:.3f}', 
         color='red', fontweight='bold', ha='center')
plt.legend()

# Q-Q plot for normality assessment
plt.subplot(1, 3, 2)
stats.probplot(sample_proportions, dist="norm", plot=plt)
plt.title('Q-Q Plot of Sample Proportions')

# Box plot for distribution shape
plt.subplot(1, 3, 3)
plt.boxplot(sample_proportions, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.ylabel('Sample Proportion')
plt.title('Box Plot of Sample Proportions')

plt.tight_layout()
plt.show()

# Statistical normality test
shapiro_stat_prop, shapiro_p_prop = stats.shapiro(sample_proportions)
print(f"\nNormality Test for Proportion Sampling Distribution:")
print(f"Shapiro-Wilk test p-value: {shapiro_p_prop:.6f}")
print(f"Interpretation: {'Sampling distribution appears normal' if shapiro_p_prop > 0.05 else 'Sampling distribution may not be normal'}")
```

## Sampling Distribution of the Variance

### Mathematical Foundation

For a sample of size $n$ from a normal population with variance $\sigma^2$, the sampling distribution of the sample variance $S^2$ follows a chi-square distribution:

```math
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)
```

This means:

```math
\begin{align}
\text{Mean of } S^2: & \quad E[S^2] = \sigma^2 \\
\text{Variance of } S^2: & \quad \text{Var}(S^2) = \frac{2\sigma^4}{n-1} \\
\text{Standard Error of } S^2: & \quad \text{SE}(S^2) = \sigma^2\sqrt{\frac{2}{n-1}}
\end{align}
```

### Simulating Variance Sampling

```python
# Simulate sampling distribution of variance with mathematical precision
np.random.seed(123)
population_var = data['sepal length (cm)'].var()
population_sd = data['sepal length (cm)'].std()
n_samples = 1000
sample_size = 10

print("Sampling Distribution of Variance:")
print("=================================")
print(f"Population variance (σ²): {population_var:.4f}")
print(f"Population standard deviation (σ): {population_sd:.4f}")
print(f"Sample size (n): {sample_size}")
print(f"Degrees of freedom (n-1): {sample_size - 1}")

# Generate sampling distribution of variance
sample_variances = np.zeros(n_samples)
for i in range(n_samples):
    sample_data = np.random.choice(data['sepal length (cm)'], size=sample_size, replace=True)
    sample_variances[i] = np.var(sample_data, ddof=1)  # ddof=1 for sample variance

# Analyze sampling distribution with mathematical interpretation
sampling_var_mean = np.mean(sample_variances)
sampling_var_sd = np.std(sample_variances)
theoretical_var_se = population_var * np.sqrt(2 / (sample_size - 1))

print(f"\nSampling Distribution Analysis:")
print(f"Mean of sampling distribution: {sampling_var_mean:.4f}")
print(f"Standard deviation of sampling distribution: {sampling_var_sd:.4f}")
print(f"Theoretical standard error σ²√[2/(n-1)]: {theoretical_var_se:.4f}")
print(f"Empirical vs Theoretical SE ratio: {sampling_var_sd / theoretical_var_se:.4f}")

# Verify chi-square distribution properties
print(f"\nChi-Square Distribution Verification:")
print(f"Sampling distribution mean ≈ Population variance: {abs(sampling_var_mean - population_var) < 1}")
print(f"Sampling distribution SD ≈ σ²√[2/(n-1)]: {abs(sampling_var_sd - theoretical_var_se) < 1}")
```

# Plot sampling distribution with mathematical annotations
```python
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(sample_variances, bins=30, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
plt.xlabel('Sample Variance (S²)')
plt.ylabel('Density')
plt.title('Sampling Distribution of Variance')

# Add mathematical annotations
plt.axvline(population_var, color='red', linestyle='--', alpha=0.7)
plt.text(population_var, plt.ylim()[1]*0.9, f'σ² = {population_var:.3f}', 
         color='red', fontweight='bold', ha='center')

# Q-Q plot for chi-square distribution
plt.subplot(1, 3, 2)
df = sample_size - 1
chi_square_quantiles = stats.chi2.ppf(np.linspace(0.01, 0.99, len(sample_variances)), df)
scaled_variances = (sample_variances * df) / population_var

plt.scatter(chi_square_quantiles, scaled_variances, alpha=0.6)
plt.plot([0, max(chi_square_quantiles)], [0, max(chi_square_quantiles)], 'r-', linewidth=2)
plt.xlabel('Theoretical Chi-Square Quantiles')
plt.ylabel('Sample Chi-Square Values')
plt.title('Q-Q Plot for Chi-Square Distribution')

# Box plot for distribution shape
plt.subplot(1, 3, 3)
plt.boxplot(sample_variances, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
plt.ylabel('Sample Variance')
plt.title('Box Plot of Sample Variances')

plt.tight_layout()
plt.show()

# Statistical test for chi-square distribution
ks_stat_var, ks_p_var = stats.kstest(scaled_variances, 'chi2', args=(df,))
print(f"\nChi-Square Distribution Test:")
print(f"Kolmogorov-Smirnov test p-value: {ks_p_var:.6f}")
print(f"Interpretation: {'Variance sampling distribution appears chi-square' if ks_p_var > 0.05 else 'Variance sampling distribution may not be chi-square'}")
```

## Bootstrap Sampling

### Mathematical Foundation

Bootstrap sampling is a resampling technique that estimates the sampling distribution of a statistic by repeatedly sampling with replacement from the original data.

For a sample $X_1, X_2, \ldots, X_n$, bootstrap samples are created by sampling with replacement:

```math
X^*_1, X^*_2, \ldots, X^*_n \sim \text{Uniform}(X_1, X_2, \ldots, X_n)
```

The bootstrap estimate of the standard error is:

```math
\text{SE}_{boot}(\hat{\theta}) = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}(\hat{\theta}^*_b - \bar{\hat{\theta}}^*)^2}
```

Where $\hat{\theta}^*_b$ is the statistic calculated from the $b$th bootstrap sample.

### Basic Bootstrap

```python
# Bootstrap sampling with mathematical precision
from scipy.stats import skew, kurtosis

# Bootstrap function for mean with mathematical interpretation
def boot_mean(data, indices):
    return np.mean(data[indices])

# Perform bootstrap with comprehensive analysis
np.random.seed(123)
n_bootstrap = 1000
bootstrap_means = np.zeros(n_bootstrap)

for i in range(n_bootstrap):
    # Sample with replacement
    bootstrap_indices = np.random.choice(len(data['sepal length (cm)']), 
                                       size=len(data['sepal length (cm)']), replace=True)
    bootstrap_means[i] = boot_mean(data['sepal length (cm)'], bootstrap_indices)

print("Bootstrap Analysis for Sample Mean:")
print("==================================")
print(f"Original sample mean: {data['sepal length (cm)'].mean():.4f}")
print(f"Bootstrap mean: {np.mean(bootstrap_means):.4f}")
print(f"Bootstrap standard error: {np.std(bootstrap_means):.4f}")
print(f"Theoretical standard error: {data['sepal length (cm)'].std() / np.sqrt(len(data)):.4f}")

# Bootstrap confidence interval with mathematical interpretation
bootstrap_ci_lower = np.percentile(bootstrap_means, 2.5)
bootstrap_ci_upper = np.percentile(bootstrap_means, 97.5)
print(f"\nBootstrap 95% Confidence Interval:")
print(f"Lower bound: {bootstrap_ci_lower:.4f}")
print(f"Upper bound: {bootstrap_ci_upper:.4f}")

# Plot bootstrap distribution with mathematical annotations
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(bootstrap_means, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
plt.xlabel('Bootstrap Mean (x̄*)')
plt.ylabel('Density')
plt.title('Bootstrap Distribution of Mean')

# Add confidence interval lines
plt.axvline(bootstrap_ci_lower, color='red', linestyle='--', alpha=0.7, label='95% CI')
plt.axvline(bootstrap_ci_upper, color='red', linestyle='--', alpha=0.7)
plt.axvline(data['sepal length (cm)'].mean(), color='green', linewidth=2, label='Original Mean')
plt.legend()

# Q-Q plot for bootstrap distribution
plt.subplot(1, 3, 2)
stats.probplot(bootstrap_means, dist="norm", plot=plt)
plt.title('Q-Q Plot of Bootstrap Means')

# Box plot for distribution shape
plt.subplot(1, 3, 3)
plt.boxplot(bootstrap_means, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.ylabel('Bootstrap Mean')
plt.title('Box Plot of Bootstrap Means')

plt.tight_layout()
plt.show()

# Statistical analysis of bootstrap distribution
print(f"\nBootstrap Distribution Analysis:")
print(f"Skewness: {skew(bootstrap_means):.4f}")
print(f"Kurtosis: {kurtosis(bootstrap_means):.4f}")
shapiro_stat_boot, shapiro_p_boot = stats.shapiro(bootstrap_means)
print(f"Normality test p-value: {shapiro_p_boot:.6f}")
```

### Bootstrap for Different Statistics

```python
# Bootstrap for median with mathematical interpretation
def boot_median(data, indices):
    return np.median(data[indices])

bootstrap_medians = np.zeros(n_bootstrap)
for i in range(n_bootstrap):
    bootstrap_indices = np.random.choice(len(data['sepal length (cm)']), 
                                       size=len(data['sepal length (cm)']), replace=True)
    bootstrap_medians[i] = boot_median(data['sepal length (cm)'], bootstrap_indices)

bootstrap_median_ci_lower = np.percentile(bootstrap_medians, 2.5)
bootstrap_median_ci_upper = np.percentile(bootstrap_medians, 97.5)

print("Bootstrap Analysis for Sample Median:")
print("====================================")
print(f"Original sample median: {data['sepal length (cm)'].median():.4f}")
print(f"Bootstrap median: {np.median(bootstrap_medians):.4f}")
print(f"Bootstrap standard error: {np.std(bootstrap_medians):.4f}")
print(f"95% Confidence interval: [{bootstrap_median_ci_lower:.4f}, {bootstrap_median_ci_upper:.4f}]\n")

# Bootstrap for standard deviation with mathematical interpretation
def boot_sd(data, indices):
    return np.std(data[indices])

bootstrap_sds = np.zeros(n_bootstrap)
for i in range(n_bootstrap):
    bootstrap_indices = np.random.choice(len(data['sepal length (cm)']), 
                                       size=len(data['sepal length (cm)']), replace=True)
    bootstrap_sds[i] = boot_sd(data['sepal length (cm)'], bootstrap_indices)

bootstrap_sd_ci_lower = np.percentile(bootstrap_sds, 2.5)
bootstrap_sd_ci_upper = np.percentile(bootstrap_sds, 97.5)

print("Bootstrap Analysis for Sample Standard Deviation:")
print("===============================================")
print(f"Original sample SD: {data['sepal length (cm)'].std():.4f}")
print(f"Bootstrap SD: {np.median(bootstrap_sds):.4f}")
print(f"Bootstrap standard error: {np.std(bootstrap_sds):.4f}")
print(f"95% Confidence interval: [{bootstrap_sd_ci_lower:.4f}, {bootstrap_sd_ci_upper:.4f}]\n")

# Compare bootstrap results with mathematical precision
comparison_table = pd.DataFrame({
    'Statistic': ['Mean', 'Median', 'Standard Deviation'],
    'Original_Value': [data['sepal length (cm)'].mean(), data['sepal length (cm)'].median(), data['sepal length (cm)'].std()],
    'Bootstrap_Mean': [np.mean(bootstrap_means), np.mean(bootstrap_medians), np.mean(bootstrap_sds)],
    'Bootstrap_SE': [np.std(bootstrap_means), np.std(bootstrap_medians), np.std(bootstrap_sds)],
    'CI_Lower': [bootstrap_ci_lower, bootstrap_median_ci_lower, bootstrap_sd_ci_lower],
    'CI_Upper': [bootstrap_ci_upper, bootstrap_median_ci_upper, bootstrap_sd_ci_upper]
})

print("Bootstrap Comparison Summary:")
print(comparison_table)

# Plot comparison of bootstrap distributions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(bootstrap_means, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(data['sepal length (cm)'].mean(), color='red', linewidth=2)
plt.xlabel('Value')
plt.title('Bootstrap: Mean')

plt.subplot(1, 3, 2)
plt.hist(bootstrap_medians, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(data['sepal length (cm)'].median(), color='red', linewidth=2)
plt.xlabel('Value')
plt.title('Bootstrap: Median')

plt.subplot(1, 3, 3)
plt.hist(bootstrap_sds, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
plt.axvline(data['sepal length (cm)'].std(), color='red', linewidth=2)
plt.xlabel('Value')
plt.title('Bootstrap: SD')

plt.tight_layout()
plt.show()
```

## Sampling Methods Comparison

### Mathematical Comparison

Different sampling methods have different statistical properties:

```math
\begin{align}
\text{Simple Random Sampling:} & \quad \text{Var}(\bar{X}_{SRS}) = \frac{\sigma^2}{n} \\
\text{Systematic Sampling:} & \quad \text{Var}(\bar{X}_{SYS}) \approx \frac{\sigma^2}{n} \text{ (if no periodicity)} \\
\text{Stratified Sampling:} & \quad \text{Var}(\bar{X}_{STR}) = \sum_{h=1}^{H} \frac{N_h^2}{N^2} \frac{\sigma_h^2}{n_h}
\end{align}
```

### Comparing Different Sampling Methods

```python
# Function to compare sampling methods with mathematical precision
def compare_sampling_methods(data, variable, sample_size, n_simulations=100):
    # Simple random sampling
    srs_means = np.zeros(n_simulations)
    srs_ses = np.zeros(n_simulations)
    
    for i in range(n_simulations):
        srs_sample = np.random.choice(data[variable], size=sample_size, replace=True)
        srs_means[i] = np.mean(srs_sample)
        srs_ses[i] = np.std(srs_sample) / np.sqrt(sample_size)
    
    # Systematic sampling
    sys_means = np.zeros(n_simulations)
    sys_ses = np.zeros(n_simulations)
    n = len(data[variable])
    k = n // sample_size
    
    for i in range(n_simulations):
        start = np.random.randint(0, k)
        indices = np.arange(start, n, k)[:sample_size]
        sys_sample = data[variable].iloc[indices]
        sys_means[i] = np.mean(sys_sample)
        sys_ses[i] = np.std(sys_sample) / np.sqrt(sample_size)
    
    # Stratified sampling (if grouping variable available)
    if 'target' in data.columns:
        strat_means = np.zeros(n_simulations)
        strat_ses = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            strat_sample = data.groupby('target').apply(
                lambda x: x.sample(n=min(len(x), sample_size//len(data['target'].unique())), random_state=i)
            )[variable]
            strat_means[i] = np.mean(strat_sample)
            strat_ses[i] = np.std(strat_sample) / np.sqrt(len(strat_sample))
    else:
        strat_means = None
        strat_ses = None
    
    # Calculate efficiency measures
    pop_mean = data[variable].mean()
    pop_var = data[variable].var()
    
    srs_efficiency = pop_var / np.var(srs_means)
    sys_efficiency = pop_var / np.var(sys_means)
    
    if strat_means is not None:
        strat_efficiency = pop_var / np.var(strat_means)
    else:
        strat_efficiency = None
    
    # Return comprehensive results
    results = {
        'SRS': {'means': srs_means, 'ses': srs_ses, 'efficiency': srs_efficiency},
        'Systematic': {'means': sys_means, 'ses': sys_ses, 'efficiency': sys_efficiency},
        'Stratified': {'means': strat_means, 'ses': strat_ses, 'efficiency': strat_efficiency}
    }
    
    return results

# Apply comparison with mathematical analysis
sampling_comparison = compare_sampling_methods(data, "sepal length (cm)", 10)

print("Sampling Methods Comparison:")
print("===========================")
print(f"Population mean: {data['sepal length (cm)'].mean():.4f}")
print(f"Population variance: {data['sepal length (cm)'].var():.4f}\n")

print("Simple Random Sampling:")
print(f"  Mean of sample means: {np.mean(sampling_comparison['SRS']['means']):.4f}")
print(f"  Variance of sample means: {np.var(sampling_comparison['SRS']['means']):.4f}")
print(f"  Efficiency: {sampling_comparison['SRS']['efficiency']:.4f}\n")

print("Systematic Sampling:")
print(f"  Mean of sample means: {np.mean(sampling_comparison['Systematic']['means']):.4f}")
print(f"  Variance of sample means: {np.var(sampling_comparison['Systematic']['means']):.4f}")
print(f"  Efficiency: {sampling_comparison['Systematic']['efficiency']:.4f}\n")

if sampling_comparison['Stratified']['means'] is not None:
    print("Stratified Sampling:")
    print(f"  Mean of sample means: {np.mean(sampling_comparison['Stratified']['means']):.4f}")
    print(f"  Variance of sample means: {np.var(sampling_comparison['Stratified']['means']):.4f}")
    print(f"  Efficiency: {sampling_comparison['Stratified']['efficiency']:.4f}\n")
```

# Plot comparison with mathematical annotations
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.hist(sampling_comparison['SRS']['means'], bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(data['sepal length (cm)'].mean(), color='red', linewidth=2)
plt.xlabel('Sample Mean')
plt.title('Simple Random Sampling')

plt.subplot(2, 2, 2)
plt.hist(sampling_comparison['Systematic']['means'], bins=20, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(data['sepal length (cm)'].mean(), color='red', linewidth=2)
plt.xlabel('Sample Mean')
plt.title('Systematic Sampling')

if sampling_comparison['Stratified']['means'] is not None:
    plt.subplot(2, 2, 3)
    plt.hist(sampling_comparison['Stratified']['means'], bins=20, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(data['sepal length (cm)'].mean(), color='red', linewidth=2)
    plt.xlabel('Sample Mean')
    plt.title('Stratified Sampling')

# Efficiency comparison
efficiencies = {
    'SRS': sampling_comparison['SRS']['efficiency'],
    'Systematic': sampling_comparison['Systematic']['efficiency']
}

if sampling_comparison['Stratified']['efficiency'] is not None:
    efficiencies['Stratified'] = sampling_comparison['Stratified']['efficiency']

plt.subplot(2, 2, 4)
plt.bar(efficiencies.keys(), efficiencies.values(), 
        color=['lightblue', 'lightgreen', 'lightcoral'][:len(efficiencies)])
plt.ylabel('Efficiency (Higher is Better)')
plt.title('Sampling Efficiency Comparison')

plt.tight_layout()
plt.show()

# Statistical comparison
print("Statistical Comparison of Sampling Methods:")
print("==========================================")

# Bias comparison
srs_bias = np.mean(sampling_comparison['SRS']['means']) - data['sepal length (cm)'].mean()
sys_bias = np.mean(sampling_comparison['Systematic']['means']) - data['sepal length (cm)'].mean()

print(f"SRS bias: {srs_bias:.4f}")
print(f"Systematic bias: {sys_bias:.4f}")

if sampling_comparison['Stratified']['means'] is not None:
    strat_bias = np.mean(sampling_comparison['Stratified']['means']) - data['sepal length (cm)'].mean()
    print(f"Stratified bias: {strat_bias:.4f}")

# Precision comparison
print(f"\nPrecision Comparison (Standard Error):")
print(f"SRS SE: {np.std(sampling_comparison['SRS']['means']):.4f}")
print(f"Systematic SE: {np.std(sampling_comparison['Systematic']['means']):.4f}")

if sampling_comparison['Stratified']['means'] is not None:
    print(f"Stratified SE: {np.std(sampling_comparison['Stratified']['means']):.4f}")
```
```

## Practical Examples

### Example 1: Quality Control Sampling with Mathematical Precision

```python
# Simulate quality control scenario with comprehensive analysis
np.random.seed(123)
production_batch = np.random.normal(mean=100, std=5, size=1000)  # Product weights

print("Quality Control Sampling Analysis:")
print("=================================")
print(f"Population mean (μ): {np.mean(production_batch):.4f}")
print(f"Population standard deviation (σ): {np.std(production_batch):.4f}")
print(f"Population size (N): {len(production_batch)}")

# Quality control sampling with mathematical interpretation
n_samples = 100
sample_size = 20

qc_means = np.zeros(n_samples)
qc_sds = np.zeros(n_samples)

for i in range(n_samples):
    sample_data = np.random.choice(production_batch, size=sample_size, replace=False)
    qc_means[i] = np.mean(sample_data)
    qc_sds[i] = np.std(sample_data)

# Calculate quality control limits with mathematical precision
population_mean = np.mean(production_batch)
population_sd = np.std(production_batch)
theoretical_se = population_sd / np.sqrt(sample_size)

ucl_mean = population_mean + 2 * theoretical_se
lcl_mean = population_mean - 2 * theoretical_se

print(f"\nQuality Control Limits (2σ limits):")
print(f"Upper Control Limit (UCL): {ucl_mean:.4f}")
print(f"Lower Control Limit (LCL): {lcl_mean:.4f}")
print(f"Theoretical standard error: {theoretical_se:.4f}")

# Analyze sampling distribution
print(f"\nSampling Distribution Analysis:")
print(f"Mean of sample means: {np.mean(qc_means):.4f}")
print(f"Standard deviation of sample means: {np.std(qc_means):.4f}")
print(f"Empirical vs Theoretical SE ratio: {np.std(qc_means) / theoretical_se:.4f}")

# Count out-of-control samples
out_of_control = np.sum((qc_means < lcl_mean) | (qc_means > ucl_mean))
print(f"Out-of-control samples: {out_of_control} ({out_of_control/n_samples*100:.1f}%)")
```

# Plot control chart with mathematical annotations
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(qc_means, 'b-', linewidth=1)
plt.axhline(y=population_mean, color='blue', linestyle='--', linewidth=2, label='Population Mean')
plt.axhline(y=ucl_mean, color='red', linestyle='--', linewidth=2, label='UCL')
plt.axhline(y=lcl_mean, color='red', linestyle='--', linewidth=2, label='LCL')
plt.xlabel('Sample Number')
plt.ylabel('Sample Mean (x̄)')
plt.title('Quality Control Chart (x̄ Chart)')
plt.legend()

# Add mathematical annotations
plt.text(n_samples/2, population_mean + 2*theoretical_se, 
         'UCL = μ + 2σ/√n', color='red', fontweight='bold', ha='center')
plt.text(n_samples/2, population_mean - 2*theoretical_se, 
         'LCL = μ - 2σ/√n', color='red', fontweight='bold', ha='center')

# Histogram of sample means with normal overlay
plt.subplot(1, 2, 2)
plt.hist(qc_means, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
x = np.linspace(min(qc_means), max(qc_means), 100)
plt.plot(x, norm.pdf(x, population_mean, theoretical_se), 'r-', linewidth=2, label='Theoretical Normal')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.title('Distribution of Sample Means')
plt.legend()

# Add mathematical annotations
plt.text(population_mean, plt.ylim()[1]*0.9, f'μ = {population_mean:.2f}', 
         color='red', fontweight='bold', ha='center')

plt.tight_layout()
plt.show()
```
```

### Example 2: Survey Sampling with Statistical Precision

```python
# Simulate survey data with comprehensive analysis
np.random.seed(123)
population_size = 10000
satisfaction_scores = np.random.normal(mean=7.5, std=1.2, size=population_size)

print("Survey Sampling Analysis:")
print("=======================")
print(f"Population size (N): {population_size}")
print(f"Population mean (μ): {np.mean(satisfaction_scores):.4f}")
print(f"Population standard deviation (σ): {np.std(satisfaction_scores):.4f}")

# Different sampling strategies with mathematical comparison
sample_size = 100

# Simple random sample with mathematical analysis
srs_sample = np.random.choice(satisfaction_scores, size=sample_size, replace=False)
srs_mean = np.mean(srs_sample)
srs_se = np.std(srs_sample) / np.sqrt(sample_size)
srs_ci_lower = srs_mean - 1.96 * srs_se
srs_ci_upper = srs_mean + 1.96 * srs_se

print(f"\nSimple Random Sampling Results:")
print(f"Sample mean (x̄): {srs_mean:.4f}")
print(f"Standard error (SE): {srs_se:.4f}")
print(f"95% Confidence interval: [{srs_ci_lower:.4f}, {srs_ci_upper:.4f}]")
print(f"Margin of error: ±{1.96 * srs_se:.4f}")

# Stratified sample with mathematical analysis
satisfaction_levels = pd.cut(satisfaction_scores, bins=3, labels=['Low', 'Medium', 'High'])
stratified_data = pd.DataFrame({'score': satisfaction_scores, 'level': satisfaction_levels})

# Calculate stratum sizes and proportions
stratum_info = stratified_data.groupby('level').agg({
    'score': ['count', 'mean', 'std']
}).round(4)
stratum_info.columns = ['n', 'stratum_mean', 'stratum_sd']
stratum_info['proportion'] = stratum_info['n'] / len(stratified_data)

print(f"\nStratum Information:")
print(stratum_info)

# Perform stratified sampling
stratified_sample = stratified_data.groupby('level').apply(
    lambda x: x.sample(n=min(len(x), sample_size//3), random_state=123)
)['score'].values

strat_mean = np.mean(stratified_sample)
strat_se = np.std(stratified_sample) / np.sqrt(len(stratified_sample))
strat_ci_lower = strat_mean - 1.96 * strat_se
strat_ci_upper = strat_mean + 1.96 * strat_se

print(f"\nStratified Sampling Results:")
print(f"Sample mean (x̄): {strat_mean:.4f}")
print(f"Standard error (SE): {strat_se:.4f}")
print(f"95% Confidence interval: [{strat_ci_lower:.4f}, {strat_ci_upper:.4f}]")
print(f"Margin of error: ±{1.96 * strat_se:.4f}")

# Compare sampling methods with mathematical precision
print(f"\nSampling Method Comparison:")
print(f"==========================")
print(f"Population mean: {np.mean(satisfaction_scores):.4f}")
print(f"SRS estimate: {srs_mean:.4f} (error: {srs_mean - np.mean(satisfaction_scores):.4f})")
print(f"Stratified estimate: {strat_mean:.4f} (error: {strat_mean - np.mean(satisfaction_scores):.4f})")
print(f"SRS precision (SE): {srs_se:.4f}")
print(f"Stratified precision (SE): {strat_se:.4f}")
print(f"Relative efficiency (SRS/Stratified): {srs_se / strat_se:.4f}")
```

# Plot comparison with mathematical annotations
plt.figure(figsize=(15, 10))

# Population distribution
plt.subplot(2, 2, 1)
plt.hist(satisfaction_scores, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(np.mean(satisfaction_scores), color='red', linewidth=2)
plt.xlabel('Satisfaction Score')
plt.title('Population Distribution')

# SRS sample distribution
plt.subplot(2, 2, 2)
plt.hist(srs_sample, bins=20, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(srs_mean, color='red', linewidth=2)
plt.xlabel('Satisfaction Score')
plt.title('Simple Random Sample')

# Stratified sample distribution
plt.subplot(2, 2, 3)
plt.hist(stratified_sample, bins=20, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
plt.axvline(strat_mean, color='red', linewidth=2)
plt.xlabel('Satisfaction Score')
plt.title('Stratified Sample')

# Confidence intervals comparison
ci_comparison = pd.DataFrame({
    'Method': ['SRS', 'Stratified'],
    'Estimate': [srs_mean, strat_mean],
    'SE': [srs_se, strat_se],
    'CI_Lower': [srs_ci_lower, strat_ci_lower],
    'CI_Upper': [srs_ci_upper, strat_ci_upper]
})

plt.subplot(2, 2, 4)
plt.errorbar([1, 2], ci_comparison['Estimate'], 
             yerr=[ci_comparison['Estimate'] - ci_comparison['CI_Lower'], 
                   ci_comparison['CI_Upper'] - ci_comparison['Estimate']],
             fmt='o', capsize=5, capthick=2, markersize=8,
             color=['blue', 'red'], label=['SRS', 'Stratified'])
plt.axhline(y=np.mean(satisfaction_scores), color='black', linestyle='--', alpha=0.7)
plt.xlabel('Sampling Method')
plt.ylabel('Satisfaction Score')
plt.title('Confidence Intervals Comparison')
plt.xticks([1, 2], ['SRS', 'Stratified'])
plt.legend()

plt.tight_layout()
plt.show()
```
```

## Best Practices

### Sample Size Determination with Mathematical Precision

```python
# Function to determine sample size for mean estimation with mathematical foundation
def determine_sample_size(population_sd, margin_of_error, confidence_level=0.95):
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    sample_size = int(np.ceil((z_score * population_sd / margin_of_error)**2))
    
    print("Sample Size Calculation for Mean Estimation:")
    print("===========================================")
    print(f"Population standard deviation (σ): {population_sd:.4f}")
    print(f"Margin of error (E): {margin_of_error}")
    print(f"Confidence level: {confidence_level}")
    print(f"Z-score (z_{{α/2}}): {z_score:.4f}")
    print(f"Required sample size (n): {sample_size}")
    print(f"Formula: n = (z_{{α/2}} × σ / E)²\n")
    
    return sample_size

# Function to determine sample size for proportion estimation
def determine_prop_sample_size(expected_prop, margin_of_error, confidence_level=0.95):
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    sample_size = int(np.ceil((z_score**2 * expected_prop * (1 - expected_prop)) / margin_of_error**2))
    
    print("Sample Size Calculation for Proportion Estimation:")
    print("================================================")
    print(f"Expected proportion (p): {expected_prop}")
    print(f"Margin of error (E): {margin_of_error}")
    print(f"Confidence level: {confidence_level}")
    print(f"Z-score (z_{{α/2}}): {z_score:.4f}")
    print(f"Required sample size (n): {sample_size}")
    print(f"Formula: n = (z_{{α/2}}² × p(1-p)) / E²\n")
    
    return sample_size

# Example calculations with mathematical interpretation
population_sd = data['sepal length (cm)'].std()
margin_of_error = 0.2  # cm units

required_sample_size = determine_sample_size(population_sd, margin_of_error)

# Sample size for proportion
expected_prop = 0.5  # Conservative estimate
margin_of_error_prop = 0.05  # 5 percentage points

required_prop_sample_size = determine_prop_sample_size(expected_prop, margin_of_error_prop)

# Finite population correction
def finite_pop_correction(sample_size, population_size):
    corrected_size = int(np.ceil(sample_size / (1 + sample_size / population_size)))
    
    print("Finite Population Correction:")
    print("============================")
    print(f"Original sample size: {sample_size}")
    print(f"Population size (N): {population_size}")
    print(f"Corrected sample size: {corrected_size}")
    print(f"Reduction: {sample_size - corrected_size} samples")
    print(f"Formula: n' = n / (1 + n/N)\n")
    
    return corrected_size

# Apply finite population correction
population_size_example = 1000
corrected_sample_size = finite_pop_correction(required_sample_size, population_size_example)
```
```

### Sampling Bias and Errors with Mathematical Analysis

```python
# Function to check for sampling bias with comprehensive analysis
def check_sampling_bias(population, sample_data, variable):
    pop_mean = population[variable].mean()
    pop_sd = population[variable].std()
    
    sample_mean = sample_data[variable].mean()
    sample_sd = sample_data[variable].std()
    
    # Calculate bias with mathematical interpretation
    bias = sample_mean - pop_mean
    relative_bias = (bias / pop_mean) * 100
    
    # Calculate efficiency with mathematical foundation
    theoretical_se = pop_sd / np.sqrt(len(sample_data[variable]))
    empirical_se = sample_sd / np.sqrt(len(sample_data[variable]))
    efficiency = (theoretical_se**2) / (empirical_se**2)
    
    # Calculate precision
    precision = 1 / (empirical_se**2)
    
    print("Sampling Bias Analysis:")
    print("======================")
    print(f"Population mean (μ): {pop_mean:.4f}")
    print(f"Sample mean (x̄): {sample_mean:.4f}")
    print(f"Absolute bias (x̄ - μ): {bias:.4f}")
    print(f"Relative bias (%): {relative_bias:.2f}%")
    print(f"Population standard deviation (σ): {pop_sd:.4f}")
    print(f"Sample standard deviation (s): {sample_sd:.4f}")
    print(f"Theoretical standard error (σ/√n): {theoretical_se:.4f}")
    print(f"Empirical standard error (s/√n): {empirical_se:.4f}")
    print(f"Efficiency (σ²/s²): {efficiency:.4f}")
    print(f"Precision (1/SE²): {precision:.4f}")
    
    # Statistical significance of bias
    t_statistic = bias / empirical_se
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=len(sample_data[variable]) - 1))
    
    print(f"t-statistic for bias test: {t_statistic:.4f}")
    print(f"p-value for bias test: {p_value:.6f}")
    print(f"Bias statistically significant: {p_value < 0.05}")
    
    return {
        'bias': bias,
        'relative_bias': relative_bias,
        'efficiency': efficiency,
        'precision': precision,
        't_statistic': t_statistic,
        'p_value': p_value
    }

# Example bias check with mathematical analysis
population_data = data
sample_data = data.sample(n=15, random_state=123)

bias_check = check_sampling_bias(population_data, sample_data, "sepal length (cm)")
```

# Visual analysis of bias
plt.figure(figsize=(15, 10))

# Population vs sample distribution
plt.subplot(2, 2, 1)
plt.hist(population_data['sepal length (cm)'], bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(population_data['sepal length (cm)'].mean(), color='red', linewidth=2)
plt.xlabel('Sepal Length (cm)')
plt.title('Population Distribution')

plt.subplot(2, 2, 2)
plt.hist(sample_data['sepal length (cm)'], bins=10, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(sample_data['sepal length (cm)'].mean(), color='red', linewidth=2)
plt.xlabel('Sepal Length (cm)')
plt.title('Sample Distribution')

# Bias visualization
bias_data = pd.DataFrame({
    'Type': ['Population', 'Sample'],
    'Mean': [population_data['sepal length (cm)'].mean(), sample_data['sepal length (cm)'].mean()],
    'SE': [population_data['sepal length (cm)'].std()/np.sqrt(len(population_data)), 
           sample_data['sepal length (cm)'].std()/np.sqrt(len(sample_data))]
})

plt.subplot(2, 2, 3)
plt.bar(bias_data['Type'], bias_data['Mean'], 
        color=['lightblue', 'lightgreen'])
plt.ylabel('Mean Sepal Length (cm)')
plt.title('Mean Comparison')

# Precision comparison
plt.subplot(2, 2, 4)
plt.bar(['Population', 'Sample'], 
        [1/bias_data['SE'].iloc[0]**2, 1/bias_data['SE'].iloc[1]**2],
        color=['lightblue', 'lightgreen'])
plt.ylabel('Precision (1/SE²)')
plt.title('Precision Comparison')

plt.tight_layout()
plt.show()
```
```

## Exercises

### Exercise 1: Sampling Distribution Simulation with Mathematical Verification
Simulate the sampling distribution of the mean for sample sizes 5, 10, 20, and 50. Compare the standard errors and verify that they follow the theoretical relationship $\text{SE} = \sigma/\sqrt{n}$.

### Exercise 2: Central Limit Theorem Demonstration with Mathematical Analysis
Demonstrate the Central Limit Theorem using different population distributions (uniform, exponential, chi-square). Verify that the sampling distribution approaches normality and that the mean and standard error match theoretical predictions.

### Exercise 3: Bootstrap Confidence Intervals with Mathematical Interpretation
Use bootstrap sampling to estimate confidence intervals for the median and standard deviation of the mtcars MPG data. Compare bootstrap intervals with theoretical intervals and interpret the differences.

### Exercise 4: Sampling Methods Comparison with Statistical Analysis
Compare simple random sampling, systematic sampling, and stratified sampling for estimating the mean MPG. Calculate efficiency measures and determine which method provides the most precise estimates.

### Exercise 5: Sample Size Determination with Mathematical Foundation
Determine the required sample size for estimating the mean MPG with a margin of error of ±1 MPG at 95% confidence. Verify your calculation using simulation.

### Exercise 6: Sampling Bias Analysis with Statistical Testing
Generate a biased sample and analyze the bias using statistical tests. Calculate the magnitude and significance of the bias and suggest methods to reduce it.

### Exercise 7: Advanced Bootstrap Applications
Use bootstrap sampling to estimate the sampling distribution of the correlation coefficient between MPG and weight in the mtcars dataset. Compare with parametric methods.

### Exercise 8: Stratified Sampling Optimization
Design an optimal stratified sampling plan for the mtcars dataset using different stratification variables. Compare the efficiency of different stratification schemes.

## Next Steps

In the next chapter, we'll learn about confidence intervals, which build upon our understanding of sampling distributions and provide the foundation for statistical inference and hypothesis testing.

---

**Key Takeaways:**
- Sampling distributions show how sample statistics vary across different samples
- Central Limit Theorem applies to sample means from any distribution with finite variance
- Bootstrap sampling is useful for non-parametric inference and complex statistics
- Sample size affects the precision of estimates according to the square root law
- Different sampling methods have different statistical properties and efficiency
- Always consider potential sampling bias and use appropriate statistical tests
- Use mathematical foundations to determine appropriate sample sizes and validate results
- Understanding sampling distributions is crucial for statistical inference and decision-making 