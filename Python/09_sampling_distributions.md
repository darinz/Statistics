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

**Python Code:** See `simple_random_sampling()` function in `09_sampling_distributions.py`

This function demonstrates simple random sampling with mathematical precision, showing how to:
- Load sample data and perform simple random sampling
- Calculate sample statistics (mean, standard deviation, standard error)
- Compare sample statistics with population parameters
- Calculate and interpret sampling error

### Systematic Sampling

Systematic sampling selects every $k$th element from the population, where $k = N/n$ and $N$ is the population size.

**Python Code:** See `systematic_sampling()` function in `09_sampling_distributions.py`

This function demonstrates systematic sampling with mathematical explanation, showing how to:
- Calculate the sampling interval k = N/n
- Perform systematic sampling by selecting every kth element
- Calculate and compare systematic sample statistics
- Interpret the mathematical properties of systematic sampling

### Stratified Sampling

Stratified sampling divides the population into homogeneous subgroups (strata) and samples from each stratum.

**Python Code:** See `stratified_sampling()` function in `09_sampling_distributions.py`

This function demonstrates stratified sampling by target class with mathematical precision, showing how to:
- Calculate sample size per stratum using proportional allocation
- Perform stratified sampling from each stratum
- Calculate stratified sample statistics
- Compare different sampling methods and their efficiency

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

**Python Code:** See `simulate_sampling_distribution()` function in `09_sampling_distributions.py`

This function simulates the sampling distribution of the mean with mathematical precision, showing how to:
- Generate multiple samples from the population
- Calculate the sampling distribution of sample means
- Compare empirical vs theoretical standard errors
- Verify Central Limit Theorem properties

### Visualizing Sampling Distribution

**Python Code:** See `visualize_sampling_distribution()` function in `09_sampling_distributions.py`

This function provides comprehensive visualization of the sampling distribution, including:
- Histogram with theoretical normal curve overlay
- Q-Q plot for normality assessment
- Box plot for distribution shape analysis
- Density plot comparison
- Statistical normality test using Shapiro-Wilk test

## Central Limit Theorem

### Mathematical Statement

The Central Limit Theorem states that for independent and identically distributed random variables $X_1, X_2, \ldots, X_n$ with mean $\mu$ and variance $\sigma^2$:

```math
\text{As } n \to \infty, \quad \frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0,1)
```

This means that regardless of the original population distribution, the sampling distribution of the mean approaches normality as sample size increases.

### Demonstrating CLT

**Python Code:** See `demonstrate_central_limit_theorem()` function in `09_sampling_distributions.py`

This function demonstrates the Central Limit Theorem with different distributions, showing how to:
- Simulate sampling distributions from uniform, exponential, and gamma populations
- Compare empirical vs theoretical properties for each distribution
- Verify that sampling distributions approach normality regardless of population shape
- Visualize the transformation from non-normal populations to normal sampling distributions
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

**Python Code:** See `effect_of_sample_size()` function in `09_sampling_distributions.py`

This function studies the effect of sample size on sampling distribution with mathematical precision, showing how to:
- Simulate sampling distributions for different sample sizes (5, 10, 30, 50)
- Compare theoretical vs empirical standard errors
- Verify that SE decreases as 1/√n
- Visualize how sample size affects the shape and precision of sampling distributions

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

**Python Code:** See `proportion_sampling()` function in `09_sampling_distributions.py`

This function simulates the sampling distribution of proportion with mathematical precision, showing how to:
- Create binary variables and calculate population proportions
- Verify normal approximation conditions (np ≥ 10, n(1-p) ≥ 10)
- Generate sampling distribution of proportions
- Compare empirical vs theoretical standard errors
- Visualize proportion sampling distributions with normality tests

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

**Python Code:** See `variance_sampling()` function in `09_sampling_distributions.py`

This function simulates the sampling distribution of variance with mathematical precision, showing how to:
- Generate sampling distribution of sample variances
- Compare empirical vs theoretical standard errors for variance
- Verify chi-square distribution properties
- Create Q-Q plots for chi-square distribution assessment
- Perform statistical tests for chi-square distribution fit

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

**Python Code:** See `bootstrap_sampling()` function in `09_sampling_distributions.py`

This function demonstrates bootstrap sampling with mathematical precision, showing how to:
- Perform bootstrap resampling with replacement
- Calculate bootstrap estimates of mean, standard error, and confidence intervals
- Visualize bootstrap distributions with Q-Q plots and box plots
- Analyze bootstrap distribution properties (skewness, kurtosis, normality)

### Bootstrap for Different Statistics

**Python Code:** See `bootstrap_different_statistics()` function in `09_sampling_distributions.py`

This function demonstrates bootstrap for different statistics with mathematical interpretation, showing how to:
- Apply bootstrap to median and standard deviation
- Calculate bootstrap confidence intervals for different statistics
- Compare bootstrap results across different statistics
- Visualize bootstrap distributions for mean, median, and standard deviation

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

**Python Code:** See `compare_sampling_methods()` function in `09_sampling_distributions.py`

This function compares different sampling methods with mathematical precision, showing how to:
- Implement and compare simple random, systematic, and stratified sampling
- Calculate efficiency measures for each sampling method
- Visualize sampling distributions and efficiency comparisons
- Analyze bias and precision differences between sampling methods

## Practical Examples

### Example 1: Quality Control Sampling with Mathematical Precision

**Python Code:** See `quality_control_example()` function in `09_sampling_distributions.py`

This function demonstrates quality control sampling with mathematical precision, showing how to:
- Simulate production batch data and quality control sampling
- Calculate control limits using 2σ limits (UCL = μ + 2σ/√n, LCL = μ - 2σ/√n)
- Analyze sampling distribution and compare empirical vs theoretical standard errors
- Create control charts and visualize out-of-control samples

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