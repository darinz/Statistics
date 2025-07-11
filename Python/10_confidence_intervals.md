# Confidence Intervals

## Overview

Confidence intervals (CIs) are one of the most important tools in statistical inference, providing a range of plausible values for population parameters based on sample data. Unlike point estimates that give a single value, confidence intervals acknowledge the inherent uncertainty in sampling and provide a range that likely contains the true population parameter.

### What is a Confidence Interval?

A confidence interval is an interval estimate that, with a specified level of confidence, contains the true population parameter. The confidence level (typically 90%, 95%, or 99%) represents the long-run frequency with which the method produces intervals that contain the true parameter.

### Key Concepts

1. **Point Estimate**: A single value estimate of a population parameter (e.g., sample mean)
2. **Interval Estimate**: A range of values that likely contains the population parameter
3. **Confidence Level**: The probability that the interval contains the true parameter (long-run interpretation)
4. **Margin of Error**: Half the width of the confidence interval
5. **Standard Error**: The standard deviation of the sampling distribution

### Mathematical Foundation

The general form of a confidence interval is:

```math
\text{Point Estimate} \pm \text{Critical Value} \times \text{Standard Error}
```

For a population mean with known standard deviation:

```math
\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
```

Where:
- $\bar{x}$ is the sample mean
- $z_{\alpha/2}$ is the critical value from the standard normal distribution
- $\sigma$ is the population standard deviation
- $n$ is the sample size
- $\alpha = 1 - \text{confidence level}$

### Interpretation

The correct interpretation of a 95% confidence interval is:
"We are 95% confident that the true population parameter lies within this interval."

**Important**: This does NOT mean there's a 95% probability that the parameter is in the interval. The parameter is fixed; the interval varies across samples.

### Properties of Confidence Intervals

1. **Width**: Wider intervals indicate more uncertainty
2. **Sample Size**: Larger samples produce narrower intervals
3. **Confidence Level**: Higher confidence levels produce wider intervals
4. **Population Variability**: More variable populations produce wider intervals

## Confidence Interval for the Mean

The mean is one of the most commonly estimated parameters. The method for constructing confidence intervals depends on whether we know the population standard deviation.

### Normal Distribution (Known Population Standard Deviation)

When the population standard deviation $\sigma$ is known, we use the standard normal distribution (z-distribution) to construct confidence intervals.

#### Mathematical Foundation

The confidence interval formula is:

```math
\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
```

Where:
- $\bar{x}$ is the sample mean
- $z_{\alpha/2}$ is the critical value from the standard normal distribution
- $\sigma$ is the population standard deviation
- $n$ is the sample size
- $\alpha = 1 - \text{confidence level}$

The standard error of the mean is:

```math
SE(\bar{x}) = \frac{\sigma}{\sqrt{n}}
```

#### Code Implementation

```python
# Load sample data
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from scipy import stats

# Load sample data (using iris dataset as equivalent to mtcars)
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Calculate confidence interval for mean sepal length
sample_mean = data['sepal length (cm)'].mean()
sample_sd = data['sepal length (cm)'].std()
n = len(data['sepal length (cm)'])

# For demonstration, assume we know population standard deviation
population_sd = 0.8  # Hypothetical known population SD
confidence_level = 0.95
alpha = 1 - confidence_level

# Calculate z-score (critical value)
z_score = stats.norm.ppf(1 - alpha/2)

# Calculate standard error
standard_error = population_sd / np.sqrt(n)

# Calculate margin of error
margin_of_error = z_score * standard_error

# Calculate confidence interval
ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

# Display results
print("=== Confidence Interval for Mean (Known Population SD) ===")
print(f"Sample mean: {sample_mean:.3f}")
print(f"Population SD: {population_sd}")
print(f"Sample size: {n}")
print(f"Confidence level: {confidence_level * 100}%")
print(f"Z-score: {z_score:.3f}")
print(f"Standard error: {standard_error:.3f}")
print(f"Margin of error: {margin_of_error:.3f}")
print(f"95% CI: {ci_lower:.3f} to {ci_upper:.3f}")

# Verify the calculation
print(f"\nVerification:")
print(f"CI width: {ci_upper - ci_lower:.3f}")
print(f"Expected width: {2 * margin_of_error:.3f}")
```

### t-Distribution (Unknown Population Standard Deviation)

In practice, we rarely know the population standard deviation. When $\sigma$ is unknown, we use the sample standard deviation $s$ and the t-distribution instead of the normal distribution.

#### Mathematical Foundation

The confidence interval formula becomes:

```math
\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}
```

Where:
- $\bar{x}$ is the sample mean
- $t_{\alpha/2, n-1}$ is the critical value from the t-distribution with $n-1$ degrees of freedom
- $s$ is the sample standard deviation
- $n$ is the sample size

The standard error is estimated as:

```math
SE(\bar{x}) = \frac{s}{\sqrt{n}}
```

#### Properties of the t-Distribution

1. **Degrees of Freedom**: The t-distribution has $n-1$ degrees of freedom
2. **Shape**: Similar to normal distribution but with heavier tails
3. **Convergence**: As $n \to \infty$, the t-distribution approaches the normal distribution
4. **Critical Values**: t-critical values are always larger than z-critical values for the same confidence level

#### Code Implementation

```python
# Calculate confidence interval using t-distribution
confidence_level = 0.95
alpha = 1 - confidence_level
df = n - 1  # degrees of freedom

# Calculate t-score (critical value)
t_score = stats.t.ppf(1 - alpha/2, df=df)

# Calculate standard error using sample SD
standard_error = sample_sd / np.sqrt(n)

# Calculate margin of error
margin_of_error = t_score * standard_error

# Calculate confidence interval
ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

# Display results
print("=== Confidence Interval for Mean (Unknown Population SD) ===")
print(f"Sample mean: {sample_mean:.3f}")
print(f"Sample SD: {sample_sd:.3f}")
print(f"Sample size: {n}")
print(f"Degrees of freedom: {df}")
print(f"Confidence level: {confidence_level * 100}%")
print(f"t-score: {t_score:.3f}")
print(f"Standard error: {standard_error:.3f}")
print(f"Margin of error: {margin_of_error:.3f}")
print(f"95% CI: {ci_lower:.3f} to {ci_upper:.3f}")

# Compare with z-distribution (for demonstration)
z_score = stats.norm.ppf(1 - alpha/2)
z_margin_of_error = z_score * standard_error
print(f"\nComparison with z-distribution:")
print(f"Z-score: {z_score:.3f}")
print(f"Z-based margin of error: {z_margin_of_error:.3f}")
print(f"Difference in margin of error: {margin_of_error - z_margin_of_error:.3f}")
```

### Using Built-in Functions

R provides convenient built-in functions for calculating confidence intervals. The `t.test()` function is particularly useful for confidence intervals of the mean.

#### Understanding the t.test() Function

The `t.test()` function performs a t-test and automatically calculates confidence intervals. It handles:
- Unknown population standard deviation (uses t-distribution)
- Degrees of freedom calculation
- Standard error estimation
- Critical value selection

#### Code Implementation

```python
# Using scipy.stats.ttest_1samp for confidence interval
t_stat, p_value = stats.ttest_1samp(data['sepal length (cm)'], 0)
# Note: scipy.stats.ttest_1samp doesn't return CI directly, so we'll use our manual calculation

# For demonstration, let's use statsmodels for a more complete t-test
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW

# Create descriptive statistics object
desc_stats = DescrStatsW(data['sepal length (cm)'])

# Get confidence interval
ci_from_test = desc_stats.tconfint_mean(alpha=0.05)
print("=== Using statsmodels for t-test ===")
print(f"95% CI from statsmodels: {ci_from_test[0]:.3f} to {ci_from_test[1]:.3f}")

# Extract other useful information
print(f"\nDetailed Information:")
print(f"Sample mean: {desc_stats.mean:.3f}")
print(f"Standard error: {desc_stats.std_mean:.3f}")
print(f"Degrees of freedom: {desc_stats.df_t:.0f}")
print(f"Confidence level: 95%")

# Verify our manual calculation
print(f"\nVerification with manual calculation:")
manual_ci_lower = sample_mean - margin_of_error
manual_ci_upper = sample_mean + margin_of_error
print(f"Manual CI: {manual_ci_lower:.3f} to {manual_ci_upper:.3f}")
print(f"statsmodels CI: {ci_from_test[0]:.3f} to {ci_from_test[1]:.3f}")
print(f"Difference: {abs(manual_ci_lower - ci_from_test[0]):.6f}")
```

## Confidence Interval for the Proportion

Proportions represent the fraction of a population that has a particular characteristic. Confidence intervals for proportions are essential in surveys, quality control, and many other applications.

### Mathematical Foundation

For a population proportion $p$, the sample proportion $\hat{p}$ is:

```math
\hat{p} = \frac{x}{n}
```

Where $x$ is the number of successes and $n$ is the sample size.

The standard error of the proportion is:

```math
SE(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
```

### Large Sample Approximation

When $n\hat{p} \geq 10$ and $n(1-\hat{p}) \geq 10$, we can use the normal approximation:

```math
\hat{p} \pm z_{\alpha/2} \cdot \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
```

#### Code Implementation

```python
# Calculate confidence interval for proportion
# Example: proportion of sepal length > 5.5 cm
data['long_sepal'] = (data['sepal length (cm)'] > 5.5).astype(int)
long_count = data['long_sepal'].sum()
total_count = len(data['long_sepal'])
sample_proportion = long_count / total_count

# Check large sample conditions
np = total_count * sample_proportion
nq = total_count * (1 - sample_proportion)

print("=== Confidence Interval for Proportion ===")
print(f"Long sepals (>5.5 cm): {long_count}")
print(f"Total flowers: {total_count}")
print(f"Sample proportion: {sample_proportion:.3f}")

# Check large sample approximation conditions
print(f"\nLarge Sample Approximation Check:")
print(f"n*p = {np:.1f} (should be ≥ 10): {'✓' if np >= 10 else '✗'}")
print(f"n*(1-p) = {nq:.1f} (should be ≥ 10): {'✓' if nq >= 10 else '✗'}")

# Calculate standard error
standard_error = np.sqrt(sample_proportion * (1 - sample_proportion) / total_count)

# Calculate margin of error
z_score = stats.norm.ppf(0.975)  # 95% confidence level
margin_of_error = z_score * standard_error

# Calculate confidence interval
ci_lower = sample_proportion - margin_of_error
ci_upper = sample_proportion + margin_of_error

print(f"\nNormal Approximation Results:")
print(f"Standard error: {standard_error:.4f}")
print(f"Z-score: {z_score:.3f}")
print(f"Margin of error: {margin_of_error:.4f}")
print(f"95% CI: {ci_lower:.4f} to {ci_upper:.4f}")

# Calculate confidence interval width
ci_width = ci_upper - ci_lower
print(f"CI width: {ci_width:.4f}")
```

### Exact Binomial Confidence Interval

When the large sample conditions are not met, or when we want the most accurate confidence interval, we use the exact binomial method. This method is based on the binomial distribution and doesn't rely on normal approximation.

#### Mathematical Foundation

The exact binomial confidence interval uses the Clopper-Pearson method, which is based on the relationship between the binomial distribution and the beta distribution. The confidence interval $(p_L, p_U)$ satisfies:

```math
P(X \geq x | p = p_L) = \frac{\alpha}{2}
```

```math
P(X \leq x | p = p_U) = \frac{\alpha}{2}
```

Where $X$ follows a binomial distribution with parameters $n$ and $p$.

#### Code Implementation

```python
# Using scipy.stats.binomtest for exact confidence interval
binom_result = stats.binomtest(long_count, total_count, p=0.5)
ci_from_binom = binom_result.proportions_ci(confidence_level=0.95)
print("=== Using scipy.stats.binomtest for exact confidence interval ===")
print(f"Exact 95% CI: {ci_from_binom[0]:.4f} to {ci_from_binom[1]:.4f}")

# Compare normal approximation vs exact method
print(f"\n=== Comparison: Normal vs Exact ===")
print(f"Normal approximation CI: {ci_lower:.4f} to {ci_upper:.4f}")
print(f"Exact binomial CI: {ci_from_binom[0]:.4f} to {ci_from_binom[1]:.4f}")

# Calculate differences
normal_width = ci_upper - ci_lower
exact_width = ci_from_binom[1] - ci_from_binom[0]
print(f"Normal CI width: {normal_width:.4f}")
print(f"Exact CI width: {exact_width:.4f}")
print(f"Width difference: {exact_width - normal_width:.4f}")

# Check if the intervals overlap significantly
overlap_lower = max(ci_lower, ci_from_binom[0])
overlap_upper = min(ci_upper, ci_from_binom[1])
overlap_width = max(0, overlap_upper - overlap_lower)
print(f"Overlap width: {overlap_width:.4f}")

# When to use each method
print(f"\n=== When to Use Each Method ===")
if np >= 10 and nq >= 10:
    print("✓ Large sample conditions met - both methods are reasonable")
    print("✓ Normal approximation is simpler and often sufficient")
    print("✓ Exact method is more accurate but computationally intensive")
else:
    print("✗ Large sample conditions not met")
    print("✓ Use exact binomial method")
    print("✗ Normal approximation may be unreliable")
```

## Confidence Interval for the Difference Between Two Means

Comparing two population means is a common statistical task. We can construct confidence intervals for the difference between two means using either independent or paired samples.

### Independent Samples

When the two samples are independent (e.g., different groups of subjects), we use the two-sample t-test approach.

#### Mathematical Foundation

For independent samples, the confidence interval for $\mu_1 - \mu_2$ is:

```math
(\bar{x}_1 - \bar{x}_2) \pm t_{\alpha/2, df} \cdot SE(\bar{x}_1 - \bar{x}_2)
```

Where the standard error depends on whether we assume equal variances:

**Equal Variances (Pooled):**
```math
SE(\bar{x}_1 - \bar{x}_2) = s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}
```

Where the pooled standard deviation is:
```math
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
```

**Unequal Variances (Welch's):**
```math
SE(\bar{x}_1 - \bar{x}_2) = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
```

The degrees of freedom for unequal variances (Welch's approximation):
```math
df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
```

#### Code Implementation

```python
# Compare sepal length between different iris species
# Group 1: Setosa (target = 0), Group 2: Versicolor (target = 1)
setosa_sepal = data[data['target'] == 0]['sepal length (cm)']
versicolor_sepal = data[data['target'] == 1]['sepal length (cm)']

# Calculate sample statistics
n1 = len(setosa_sepal)
n2 = len(versicolor_sepal)
mean1 = setosa_sepal.mean()
mean2 = versicolor_sepal.mean()
sd1 = setosa_sepal.std()
sd2 = versicolor_sepal.std()

print("=== Confidence Interval for Difference Between Two Means ===")
print(f"Group 1 (Setosa): n = {n1}, mean = {mean1:.3f}, SD = {sd1:.3f}")
print(f"Group 2 (Versicolor): n = {n2}, mean = {mean2:.3f}, SD = {sd2:.3f}")

# Test for equal variances (Levene's test)
from scipy.stats import levene
levene_stat, levene_p = levene(setosa_sepal, versicolor_sepal)
print(f"\nVariance Test (Levene's test):")
print(f"Levene statistic: {levene_stat:.3f}")
print(f"p-value: {levene_p:.4f}")
print(f"Equal variances assumption: {'✓' if levene_p > 0.05 else '✗'}")

# Calculate pooled standard deviation (equal variances)
pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))

# Calculate standard error
standard_error = pooled_sd * np.sqrt(1/n1 + 1/n2)

# Calculate degrees of freedom
df = n1 + n2 - 2

# Calculate t-score
t_score = stats.t.ppf(0.975, df=df)

# Calculate margin of error
margin_of_error = t_score * standard_error

# Calculate confidence interval
difference = mean2 - mean1
ci_lower = difference - margin_of_error
ci_upper = difference + margin_of_error

print(f"\nPooled Variance Results:")
print(f"Pooled SD: {pooled_sd:.3f}")
print(f"Standard error: {standard_error:.3f}")
print(f"Degrees of freedom: {df}")
print(f"t-score: {t_score:.3f}")
print(f"Margin of error: {margin_of_error:.3f}")
print(f"Difference (Versicolor - Setosa): {difference:.3f}")
print(f"95% CI for difference: {ci_lower:.3f} to {ci_upper:.3f}")

# Using built-in function (pooled variance)
t_stat, p_val = stats.ttest_ind(versicolor_sepal, setosa_sepal, equal_var=True)
print(f"\nIndependent t-test (equal variances):")
print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.6f}")

# Using built-in function (Welch's - unequal variances)
t_stat_welch, p_val_welch = stats.ttest_ind(versicolor_sepal, setosa_sepal, equal_var=False)
print(f"\nWelch's t-test (unequal variances):")
print(f"t-statistic: {t_stat_welch:.3f}, p-value: {p_val_welch:.6f}")

# Calculate confidence interval using statsmodels
from statsmodels.stats.weightstats import CompareMeans
cm = CompareMeans.from_data(versicolor_sepal, setosa_sepal)
ci_welch = cm.tconfint_diff(alpha=0.05, usevar='unequal')
print(f"95% CI for difference (Welch's): {ci_welch[0]:.3f} to {ci_welch[1]:.3f}")
```

### Paired Samples

When the two samples are paired (e.g., same subjects measured before and after treatment), we analyze the differences between pairs rather than the individual measurements.

#### Mathematical Foundation

For paired samples, we work with the differences $d_i = x_{i1} - x_{i2}$. The confidence interval for the mean difference $\mu_d$ is:

```math
\bar{d} \pm t_{\alpha/2, n-1} \cdot \frac{s_d}{\sqrt{n}}
```

Where:
- $\bar{d}$ is the mean of the differences
- $s_d$ is the standard deviation of the differences
- $n$ is the number of pairs
- $t_{\alpha/2, n-1}$ is the critical value from the t-distribution with $n-1$ degrees of freedom

The standard error of the mean difference is:

```math
SE(\bar{d}) = \frac{s_d}{\sqrt{n}}
```

#### Advantages of Paired Analysis

1. **Reduced Variability**: Paired analysis removes between-subject variability
2. **Increased Power**: More sensitive to detect treatment effects
3. **Control for Confounding**: Each subject serves as their own control

#### Code Implementation

```python
# Simulate paired data (before and after treatment)
np.random.seed(123)
before = np.random.normal(50, 10, 20)
after = before + np.random.normal(5, 3, 20)  # Treatment effect

# Calculate differences
differences = after - before

# Calculate confidence interval for mean difference
n_diff = len(differences)
mean_diff = np.mean(differences)
sd_diff = np.std(differences, ddof=1)

print("=== Paired Samples Confidence Interval ===")
print(f"Sample size (pairs): {n_diff}")
print(f"Mean difference: {mean_diff:.3f}")
print(f"SD of differences: {sd_diff:.3f}")

# Calculate standard error
standard_error = sd_diff / np.sqrt(n_diff)

# Calculate t-score
df = n_diff - 1
t_score = stats.t.ppf(0.975, df=df)

# Calculate margin of error
margin_of_error = t_score * standard_error

# Calculate confidence interval
ci_lower = mean_diff - margin_of_error
ci_upper = mean_diff + margin_of_error

print(f"\nPaired Analysis Results:")
print(f"Standard error: {standard_error:.3f}")
print(f"Degrees of freedom: {df}")
print(f"t-score: {t_score:.3f}")
print(f"Margin of error: {margin_of_error:.3f}")
print(f"95% CI for mean difference: {ci_lower:.3f} to {ci_upper:.3f}")

# Using built-in function
t_stat, p_val = stats.ttest_rel(after, before)
print(f"\nPaired t-test:")
print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.6f}")

# Calculate confidence interval using statsmodels
desc_stats_paired = DescrStatsW(differences)
ci_paired = desc_stats_paired.tconfint_mean(alpha=0.05)
print(f"95% CI for mean difference (statsmodels): {ci_paired[0]:.3f} to {ci_paired[1]:.3f}")

# Compare with independent samples analysis (incorrect for paired data)
t_stat_ind, p_val_ind = stats.ttest_ind(after, before)
print(f"\n=== Comparison: Paired vs Independent Analysis ===")
print(f"Paired CI width: {ci_upper - ci_lower:.3f}")
print(f"Independent CI width: {ci_paired[1] - ci_paired[0]:.3f}")

# Calculate independent CI width manually for comparison
pooled_sd_ind = np.sqrt(((len(after) - 1) * np.var(after, ddof=1) + 
                        (len(before) - 1) * np.var(before, ddof=1)) / 
                       (len(after) + len(before) - 2))
se_ind = pooled_sd_ind * np.sqrt(1/len(after) + 1/len(before))
t_ind = stats.t.ppf(0.975, df=len(after) + len(before) - 2)
ci_width_ind = 2 * t_ind * se_ind
print(f"Independent CI width (manual): {ci_width_ind:.3f}")
print(f"Width ratio (Independent/Paired): {ci_width_ind / (ci_upper - ci_lower):.2f}")
```

## Confidence Interval for the Variance

Confidence intervals for variance are useful when we need to estimate the variability of a population or compare the variability between groups.

### Mathematical Foundation

For a normal population, the sampling distribution of the sample variance follows a chi-square distribution:

```math
\frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}
```

Where:
- $s^2$ is the sample variance
- $\sigma^2$ is the population variance
- $n$ is the sample size
- $\chi^2_{n-1}$ is the chi-square distribution with $n-1$ degrees of freedom

The confidence interval for the population variance is:

```math
\left(\frac{(n-1)s^2}{\chi^2_{\alpha/2, n-1}}, \frac{(n-1)s^2}{\chi^2_{1-\alpha/2, n-1}}\right)
```

Where $\chi^2_{\alpha/2, n-1}$ and $\chi^2_{1-\alpha/2, n-1}$ are the critical values from the chi-square distribution.

### Chi-Square Based Confidence Interval

#### Code Implementation

```python
# Calculate confidence interval for variance
sample_variance = data['sepal length (cm)'].var()
n = len(data['sepal length (cm)'])
df = n - 1

print("=== Confidence Interval for Variance ===")
print(f"Sample variance: {sample_variance:.3f}")
print(f"Sample size: {n}")
print(f"Degrees of freedom: {df}")

# Calculate chi-square critical values
chi_lower = stats.chi2.ppf(0.025, df=df)
chi_upper = stats.chi2.ppf(0.975, df=df)

print(f"\nChi-square Critical Values:")
print(f"Lower critical value (2.5%): {chi_lower:.3f}")
print(f"Upper critical value (97.5%): {chi_upper:.3f}")

# Calculate confidence interval
ci_lower_var = (df * sample_variance) / chi_upper
ci_upper_var = (df * sample_variance) / chi_lower

print(f"\nVariance Confidence Interval:")
print(f"95% CI for variance: {ci_lower_var:.3f} to {ci_upper_var:.3f}")

# Confidence interval for standard deviation
ci_lower_sd = np.sqrt(ci_lower_var)
ci_upper_sd = np.sqrt(ci_upper_var)
print(f"95% CI for standard deviation: {ci_lower_sd:.3f} to {ci_upper_sd:.3f}")

# Verify the calculation
print(f"\nVerification:")
print(f"Sample SD: {data['sepal length (cm)'].std():.3f}")
print(f"CI width (variance): {ci_upper_var - ci_lower_var:.3f}")
print(f"CI width (SD): {ci_upper_sd - ci_lower_sd:.3f}")

# Check if the intervals are reasonable
print(f"\nReasonableness Check:")
if ci_lower_var > 0:
    print("✓ Lower bound is positive (reasonable)")
else:
    print("✗ Lower bound is negative (unreasonable)")

if ci_upper_var > sample_variance:
    print("✓ Upper bound is greater than sample variance")
else:
    print("✗ Upper bound is less than sample variance")
```

## Bootstrap Confidence Intervals

Bootstrap methods provide a non-parametric approach to constructing confidence intervals. They are particularly useful when the sampling distribution is unknown or when the assumptions of parametric methods are violated.

### Mathematical Foundation

The bootstrap method works by:
1. Resampling with replacement from the original sample
2. Computing the statistic of interest for each bootstrap sample
3. Using the distribution of bootstrap statistics to construct confidence intervals

For a sample $x_1, x_2, \ldots, x_n$, we create bootstrap samples $x_1^*, x_2^*, \ldots, x_n^*$ by sampling with replacement from the original data.

The bootstrap estimate of the standard error is:

```math
SE_{boot}(\hat{\theta}) = \sqrt{\frac{1}{B-1} \sum_{b=1}^B (\hat{\theta}_b^* - \bar{\theta}^*)^2}
```

Where:
- $\hat{\theta}_b^*$ is the estimate from the $b$-th bootstrap sample
- $\bar{\theta}^*$ is the mean of all bootstrap estimates
- $B$ is the number of bootstrap samples

### Types of Bootstrap Confidence Intervals

1. **Normal Bootstrap**: Assumes bootstrap distribution is normal
2. **Percentile Bootstrap**: Uses percentiles of bootstrap distribution
3. **Basic Bootstrap**: Adjusts for bias in the bootstrap distribution
4. **BCa Bootstrap**: Bias-corrected and accelerated method

### Bootstrap for Mean

#### Code Implementation

```python
# Bootstrap function for mean
def boot_mean(data, indices):
    return np.mean(data[indices])

# Perform bootstrap
np.random.seed(123)
n_bootstrap = 1000
bootstrap_means = []

for i in range(n_bootstrap):
    # Sample with replacement
    bootstrap_sample = np.random.choice(data['sepal length (cm)'], size=len(data['sepal length (cm)']), replace=True)
    bootstrap_means.append(np.mean(bootstrap_sample))

bootstrap_means = np.array(bootstrap_means)

print("=== Bootstrap Confidence Intervals ===")
print(f"Original sample mean: {data['sepal length (cm)'].mean():.3f}")
print(f"Bootstrap replications: {len(bootstrap_means)}")
print(f"Bootstrap mean: {np.mean(bootstrap_means):.3f}")
print(f"Bootstrap SE: {np.std(bootstrap_means):.3f}")

# Different types of bootstrap confidence intervals
# Normal bootstrap CI
bootstrap_se = np.std(bootstrap_means)
normal_ci_lower = data['sepal length (cm)'].mean() - 1.96 * bootstrap_se
normal_ci_upper = data['sepal length (cm)'].mean() + 1.96 * bootstrap_se

# Percentile bootstrap CI
percentile_ci_lower = np.percentile(bootstrap_means, 2.5)
percentile_ci_upper = np.percentile(bootstrap_means, 97.5)

# Basic bootstrap CI
basic_ci_lower = 2 * data['sepal length (cm)'].mean() - np.percentile(bootstrap_means, 97.5)
basic_ci_upper = 2 * data['sepal length (cm)'].mean() - np.percentile(bootstrap_means, 2.5)

print(f"\nBootstrap Confidence Intervals:")
print(f"Normal bootstrap CI: {normal_ci_lower:.3f} to {normal_ci_upper:.3f}")
print(f"Percentile bootstrap CI: {percentile_ci_lower:.3f} to {percentile_ci_upper:.3f}")
print(f"Basic bootstrap CI: {basic_ci_lower:.3f} to {basic_ci_upper:.3f}")

# Compare with parametric CI
parametric_ci = desc_stats.tconfint_mean(alpha=0.05)
print(f"Parametric t-test CI: {parametric_ci[0]:.3f} to {parametric_ci[1]:.3f}")

# Calculate bootstrap bias
bootstrap_bias = np.mean(bootstrap_means) - data['sepal length (cm)'].mean()
print(f"\nBootstrap Analysis:")
print(f"Bootstrap bias: {bootstrap_bias:.4f}")
print(f"Bootstrap variance: {np.var(bootstrap_means):.4f}")

# Plot bootstrap distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(bootstrap_means, bins=30, alpha=0.7, color='lightblue', edgecolor='white', density=True)
plt.axvline(percentile_ci_lower, color='red', linestyle='--', linewidth=2, label='95% CI')
plt.axvline(percentile_ci_upper, color='red', linestyle='--', linewidth=2)
plt.axvline(data['sepal length (cm)'].mean(), color='green', linewidth=2, label='Original Mean')
plt.axvline(np.mean(bootstrap_means), color='blue', linestyle=':', linewidth=2, label='Bootstrap Mean')
plt.xlabel('Bootstrap Mean')
plt.ylabel('Density')
plt.title('Bootstrap Distribution of Mean')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Bootstrap for Median

Bootstrap methods are particularly valuable for statistics like the median, which don't have simple parametric sampling distributions.

#### Mathematical Foundation

The median is a robust measure of central tendency that is less sensitive to outliers than the mean. The bootstrap provides a way to estimate the sampling distribution of the median without making distributional assumptions.

For a sample $x_1, x_2, \ldots, x_n$, the sample median is:
```math
\text{median} = \begin{cases}
x_{(n+1)/2} & \text{if } n \text{ is odd} \\
\frac{x_{n/2} + x_{n/2+1}}{2} & \text{if } n \text{ is even}
\end{cases}
```

Where $x_{(i)}$ denotes the $i$-th order statistic.

#### Code Implementation

```python
# Bootstrap function for median
def boot_median(data, indices):
    return np.median(data[indices])

# Perform bootstrap for median
np.random.seed(123)
bootstrap_medians = []

for i in range(n_bootstrap):
    # Sample with replacement
    bootstrap_sample = np.random.choice(data['sepal length (cm)'], size=len(data['sepal length (cm)']), replace=True)
    bootstrap_medians.append(np.median(bootstrap_sample))

bootstrap_medians = np.array(bootstrap_medians)
boot_median_ci_lower = np.percentile(bootstrap_medians, 2.5)
boot_median_ci_upper = np.percentile(bootstrap_medians, 97.5)

print("=== Bootstrap Confidence Interval for Median ===")
print(f"Original sample median: {np.median(data['sepal length (cm)']):.3f}")
print(f"Bootstrap median: {np.mean(bootstrap_medians):.3f}")
print(f"Bootstrap SE: {np.std(bootstrap_medians):.3f}")
print(f"Bootstrap CI for median: {boot_median_ci_lower:.3f} to {boot_median_ci_upper:.3f}")

# Compare with mean
print(f"\nComparison with Mean:")
print(f"Sample mean: {data['sepal length (cm)'].mean():.3f}")
print(f"Sample median: {np.median(data['sepal length (cm)']):.3f}")
print(f"Difference (mean - median): {data['sepal length (cm)'].mean() - np.median(data['sepal length (cm)']):.3f}")

# Check for skewness
if data['sepal length (cm)'].mean() > np.median(data['sepal length (cm)']):
    print("Distribution appears right-skewed (mean > median)")
elif data['sepal length (cm)'].mean() < np.median(data['sepal length (cm)']):
    print("Distribution appears left-skewed (mean < median)")
else:
    print("Distribution appears symmetric (mean ≈ median)")

# Plot bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_medians, bins=30, alpha=0.7, color='lightgreen', edgecolor='white', density=True)
plt.axvline(boot_median_ci_lower, color='red', linestyle='--', linewidth=2, label='95% CI')
plt.axvline(boot_median_ci_upper, color='red', linestyle='--', linewidth=2)
plt.axvline(np.median(data['sepal length (cm)']), color='green', linewidth=2, label='Original Median')
plt.axvline(np.mean(bootstrap_medians), color='blue', linestyle=':', linewidth=2, label='Bootstrap Median')
plt.xlabel('Bootstrap Median')
plt.ylabel('Density')
plt.title('Bootstrap Distribution of Median')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Compare bootstrap distributions of mean and median
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.hist(bootstrap_means, bins=30, alpha=0.7, color='lightblue', edgecolor='white', density=True)
ax1.set_title('Bootstrap Mean')
ax1.set_xlabel('Bootstrap Mean')
ax1.set_ylabel('Density')
ax1.grid(True, alpha=0.3)

ax2.hist(bootstrap_medians, bins=30, alpha=0.7, color='lightgreen', edgecolor='white', density=True)
ax2.set_title('Bootstrap Median')
ax2.set_xlabel('Bootstrap Median')
ax2.set_ylabel('Density')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Effect of Sample Size and Confidence Level

The width of confidence intervals is influenced by both sample size and confidence level. Understanding these relationships is crucial for study design and interpretation.

### Mathematical Foundation

The width of a confidence interval is determined by:

```math
\text{Width} = 2 \times \text{Critical Value} \times \text{Standard Error}
```

For the mean with unknown population standard deviation:
```math
\text{Width} = 2 \times t_{\alpha/2, n-1} \times \frac{s}{\sqrt{n}}
```

#### Sample Size Effect

As sample size increases:
- Standard error decreases: $SE \propto \frac{1}{\sqrt{n}}$
- Critical value decreases (t-distribution approaches normal)
- Overall effect: Width decreases approximately as $\frac{1}{\sqrt{n}}$

#### Confidence Level Effect

As confidence level increases:
- Critical value increases
- Width increases
- Trade-off: Higher confidence vs. wider intervals

### Sample Size Effect

#### Code Implementation

```python
# Function to calculate confidence interval width
def ci_width(sample_size, confidence_level=0.95):
    # Simulate sample
    np.random.seed(123)
    sample_data = np.random.choice(data['sepal length (cm)'], size=sample_size, replace=True)
    
    # Calculate confidence interval
    desc_stats_sample = DescrStatsW(sample_data)
    ci = desc_stats_sample.tconfint_mean(alpha=1-confidence_level)
    
    # Return width
    return ci[1] - ci[0]

# Test different sample sizes
sample_sizes = [5, 10, 20, 30, 50, 100]
ci_widths = [ci_width(size) for size in sample_sizes]

print("=== Effect of Sample Size on CI Width ===")
for i, size in enumerate(sample_sizes):
    print(f"Sample size: {size} → CI width: {ci_widths[i]:.3f}")

# Calculate theoretical relationship
theoretical_widths = [ci_widths[0] * np.sqrt(sample_sizes[0] / size) for size in sample_sizes]
print(f"\nTheoretical widths (based on 1/√n relationship):")
for i, size in enumerate(sample_sizes):
    print(f"Sample size: {size} → Theoretical width: {theoretical_widths[i]:.3f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, ci_widths, 'bo-', linewidth=2, markersize=8, label='Observed')
plt.plot(sample_sizes, theoretical_widths, 'r--', linewidth=2, label='Theoretical (1/√n)')
plt.xlabel('Sample Size')
plt.ylabel('CI Width')
plt.title('Effect of Sample Size on CI Width')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate efficiency gains
print(f"\nEfficiency Analysis:")
for i in range(1, len(sample_sizes)):
    efficiency_gain = (ci_widths[0] / ci_widths[i])**2
    print(f"Sample size {sample_sizes[i]} is {efficiency_gain:.1f} times more efficient than size {sample_sizes[0]}")
```

### Confidence Level Effect

#### Code Implementation

```python
# Function to calculate confidence interval for different levels
def ci_by_level(confidence_levels):
    widths = []
    critical_values = []
    
    for level in confidence_levels:
        desc_stats_level = DescrStatsW(data['sepal length (cm)'])
        ci = desc_stats_level.tconfint_mean(alpha=1-level)
        widths.append(ci[1] - ci[0])
        
        # Calculate critical value
        alpha = 1 - level
        critical_values.append(stats.t.ppf(1 - alpha/2, df=len(data['sepal length (cm)']) - 1))
    
    return widths, critical_values

# Test different confidence levels
confidence_levels = [0.80, 0.85, 0.90, 0.95, 0.99]
ci_widths_by_level, critical_values = ci_by_level(confidence_levels)

print("=== Effect of Confidence Level on CI Width ===")
for i, level in enumerate(confidence_levels):
    print(f"Confidence level: {level * 100}% → CI width: {ci_widths_by_level[i]:.3f} → Critical value: {critical_values[i]:.3f}")

# Calculate relative increases
print(f"\nRelative Increases:")
for i in range(1, len(confidence_levels)):
    relative_increase = (ci_widths_by_level[i] / ci_widths_by_level[0]) - 1
    print(f"From {confidence_levels[0] * 100}% to {confidence_levels[i] * 100}%: +{relative_increase * 100:.1f}% wider")

# Plot results
fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary axis for CI width
ax1.plot(confidence_levels, ci_widths_by_level, 'ro-', linewidth=2, markersize=8, label='CI Width')
ax1.set_xlabel('Confidence Level')
ax1.set_ylabel('CI Width', color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Secondary axis for critical values
ax2 = ax1.twinx()
ax2.plot(confidence_levels, critical_values, 'b--', linewidth=2, label='Critical Value')
ax2.set_ylabel('Critical Value', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Effect of Confidence Level on CI Width')
plt.grid(True, alpha=0.3)
plt.show()

# Trade-off analysis
print(f"\nTrade-off Analysis:")
print("Higher confidence levels provide more certainty but wider intervals.")
print("This creates a trade-off between precision and confidence.\n")

# Practical recommendations
print("Practical Recommendations:")
print("• 90% CI: Good balance for most applications")
print("• 95% CI: Standard choice for most research")
print("• 99% CI: Use when high confidence is critical")
print("• 80% CI: Use when precision is more important than confidence")
```

## Multiple Confidence Intervals

When constructing multiple confidence intervals, we need to account for the increased probability of making at least one Type I error (false positive). This is known as the multiple comparisons problem.

### Mathematical Foundation

When making $k$ independent confidence intervals, each with confidence level $1 - \alpha$, the probability that all intervals contain their true parameters is:

```math
P(\text{All intervals correct}) = (1 - \alpha)^k
```

The family-wise error rate (FWER) is:
```math
\text{FWER} = 1 - (1 - \alpha)^k
```

For small $\alpha$, this is approximately:
```math
\text{FWER} \approx k\alpha
```

### Bonferroni Correction

The Bonferroni correction adjusts the individual significance level to maintain the desired family-wise error rate:

```math
\alpha_{individual} = \frac{\alpha_{family}}{k}
```

### Simultaneous Confidence Intervals

#### Code Implementation

```python
# Bonferroni correction for multiple comparisons
# Example: confidence intervals for sepal length by iris species
species = data['target'].unique()
n_comparisons = len(species)
alpha_family = 0.05
alpha_individual = alpha_family / n_comparisons

print("=== Multiple Confidence Intervals ===")
print(f"Number of comparisons: {n_comparisons}")
print(f"Family-wise alpha: {alpha_family}")
print(f"Individual alpha: {alpha_individual:.4f}")
print(f"Individual confidence level: {(1 - alpha_individual) * 100:.2f}%")

# Calculate confidence intervals for each species
ci_results = {}
sample_sizes = {}
sample_means = {}

for sp in species:
    sp_data = data[data['target'] == sp]['sepal length (cm)']
    desc_stats_sp = DescrStatsW(sp_data)
    ci = desc_stats_sp.tconfint_mean(alpha=alpha_individual)
    ci_results[str(sp)] = ci
    sample_sizes[str(sp)] = len(sp_data)
    sample_means[str(sp)] = sp_data.mean()

# Display results
print(f"\nBonferroni-corrected confidence intervals (α = {alpha_family}):")
for sp in species:
    ci = ci_results[str(sp)]
    n = sample_sizes[str(sp)]
    mean_val = sample_means[str(sp)]
    species_name = ['Setosa', 'Versicolor', 'Virginica'][int(sp)]
    print(f"{species_name}: n = {n}, mean = {mean_val:.2f}, CI = {ci[0]:.2f} to {ci[1]:.2f}")

# Compare with uncorrected intervals
print(f"\nUncorrected confidence intervals (95%):")
for sp in species:
    sp_data = data[data['target'] == sp]['sepal length (cm)']
    desc_stats_uncorr = DescrStatsW(sp_data)
    ci = desc_stats_uncorr.tconfint_mean(alpha=0.05)
    species_name = ['Setosa', 'Versicolor', 'Virginica'][int(sp)]
    print(f"{species_name}: {ci[0]:.2f} to {ci[1]:.2f}")

# Calculate width differences
print(f"\nWidth Comparison:")
for sp in species:
    sp_data = data[data['target'] == sp]['sepal length (cm)']
    
    # Bonferroni-corrected
    desc_stats_bonf = DescrStatsW(sp_data)
    ci_bonf = desc_stats_bonf.tconfint_mean(alpha=alpha_individual)
    width_bonf = ci_bonf[1] - ci_bonf[0]
    
    # Uncorrected
    desc_stats_uncorr = DescrStatsW(sp_data)
    ci_uncorr = desc_stats_uncorr.tconfint_mean(alpha=0.05)
    width_uncorr = ci_uncorr[1] - ci_uncorr[0]
    
    width_ratio = width_bonf / width_uncorr
    species_name = ['Setosa', 'Versicolor', 'Virginica'][int(sp)]
    print(f"{species_name}: width ratio = {width_ratio:.2f}")

# Family-wise error rate calculation
print(f"\nFamily-wise Error Rate Analysis:")
print(f"Uncorrected FWER: {1 - (1 - 0.05)**n_comparisons:.4f}")
print(f"Bonferroni-corrected FWER: {1 - (1 - alpha_individual)**n_comparisons:.4f}")
print(f"Bonferroni upper bound: {n_comparisons * alpha_individual:.4f}")
```

## Practical Examples

### Example 1: Quality Control

Quality control is a common application of confidence intervals in manufacturing and production processes.

#### Mathematical Foundation

In quality control, we often want to estimate the mean of a production process and determine if it meets specifications. The confidence interval helps us understand the uncertainty in our estimate.

For a production process with target value $\mu_0$, we construct a confidence interval for the population mean $\mu$. If $\mu_0$ falls within the interval, we have evidence that the process is on target.

#### Code Implementation

```python
# Quality control example
np.random.seed(123)
production_batch = np.random.normal(100, 5, 100)

# Calculate confidence interval for mean weight
desc_stats_qc = DescrStatsW(production_batch)
ci_qc = desc_stats_qc.tconfint_mean(alpha=0.05)

print("=== Quality Control Example ===")
print(f"Sample size: {len(production_batch)}")
print(f"Production batch mean: {np.mean(production_batch):.2f}")
print(f"Production batch SD: {np.std(production_batch, ddof=1):.2f}")
print(f"95% CI for mean weight: {ci_qc[0]:.2f} to {ci_qc[1]:.2f}")

# Check if target weight (100) is in confidence interval
target_weight = 100
if ci_qc[0] <= target_weight <= ci_qc[1]:
    print("✓ Target weight is within the confidence interval.")
    print("  → Process appears to be on target.")
else:
    print("✗ Target weight is outside the confidence interval.")
    print("  → Process may need adjustment.")

# Calculate process capability
process_capability = abs(np.mean(production_batch) - target_weight) / np.std(production_batch, ddof=1)
print(f"\nProcess Capability Analysis:")
print(f"Process capability index: {process_capability:.3f}")

if process_capability < 0.5:
    print("✓ Process is well-controlled")
elif process_capability < 1.0:
    print("⚠ Process needs monitoring")
else:
    print("✗ Process needs immediate attention")

# Tolerance analysis
tolerance = 2 * np.std(production_batch, ddof=1)  # ±2 SD tolerance
print(f"Process tolerance (±2 SD): {np.mean(production_batch) - tolerance:.2f} to {np.mean(production_batch) + tolerance:.2f}")
```

### Example 2: Survey Results

Survey research is another important application of confidence intervals, particularly for proportions.

#### Mathematical Foundation

In survey research, we often want to estimate the proportion of a population that has a particular characteristic. The confidence interval for a proportion helps us understand the precision of our estimate.

For a population proportion $p$, the sample proportion $\hat{p}$ follows approximately a normal distribution when the sample size is large enough:

```math
\hat{p} \sim N\left(p, \sqrt{\frac{p(1-p)}{n}}\right)
```

#### Code Implementation

```python
# Survey example
np.random.seed(123)
survey_responses = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
# 1 = satisfied, 0 = not satisfied

# Calculate confidence interval for satisfaction proportion
satisfied_count = np.sum(survey_responses)
total_responses = len(survey_responses)
satisfaction_proportion = satisfied_count / total_responses

print("=== Survey Results Example ===")
print(f"Total responses: {total_responses}")
print(f"Satisfied responses: {satisfied_count}")
print(f"Satisfaction proportion: {satisfaction_proportion:.3f}")

# Check large sample conditions
np_check = total_responses * satisfaction_proportion
nq_check = total_responses * (1 - satisfaction_proportion)
print(f"Large sample check - n*p = {np_check:.1f}, n*(1-p) = {nq_check:.1f}")

# Calculate confidence interval
binom_result = stats.binomtest(satisfied_count, total_responses, p=0.5)
ci_survey = binom_result.proportions_ci(confidence_level=0.95)

print(f"95% CI for satisfaction proportion: {ci_survey[0]:.4f} to {ci_survey[1]:.4f}")

# Interpret the results
print(f"\nInterpretation:")
print("We are 95% confident that the true satisfaction proportion")
print(f"in the population lies between {ci_survey[0] * 100:.1f}% and {ci_survey[1] * 100:.1f}%.")

# Margin of error
margin_of_error = (ci_survey[1] - ci_survey[0]) / 2
print(f"Margin of error: ±{margin_of_error * 100:.1f}%")

# Sample size planning for future surveys
print(f"\nSample Size Planning:")
print("For a margin of error of ±5% at 95% confidence:")
required_n = np.ceil(1.96**2 * 0.5 * 0.5 / 0.05**2)
print(f"Required sample size: {int(required_n)}")

# Compare with different confidence levels
confidence_levels = [0.90, 0.95, 0.99]
for level in confidence_levels:
    binom_result_level = stats.binomtest(satisfied_count, total_responses, p=0.5)
    ci_level = binom_result_level.proportions_ci(confidence_level=level)
    print(f"Confidence level {level * 100}%: CI = {ci_level[0]:.4f} to {ci_level[1]:.4f}")
```

### Example 3: Treatment Effect

Clinical trials and experimental studies often use confidence intervals to estimate treatment effects.

#### Mathematical Foundation

In experimental studies, we want to estimate the difference between treatment and control groups. The confidence interval for the treatment effect helps us understand both the magnitude and precision of the effect.

For independent samples, the treatment effect $\delta = \mu_1 - \mu_2$ is estimated by $\bar{x}_1 - \bar{x}_2$ with standard error:

```math
SE(\bar{x}_1 - \bar{x}_2) = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
```

#### Code Implementation

```python
# Treatment effect example
np.random.seed(123)
control_group = np.random.normal(50, 10, 30)
treatment_group = np.random.normal(55, 10, 30)

print("=== Treatment Effect Example ===")
print(f"Control group: n = {len(control_group)}, mean = {np.mean(control_group):.2f}, SD = {np.std(control_group, ddof=1):.2f}")
print(f"Treatment group: n = {len(treatment_group)}, mean = {np.mean(treatment_group):.2f}, SD = {np.std(treatment_group, ddof=1):.2f}")

# Calculate confidence interval for treatment effect
t_stat, p_val = stats.ttest_ind(treatment_group, control_group)
cm_treatment = CompareMeans.from_data(treatment_group, control_group)
ci_treatment = cm_treatment.tconfint_diff(alpha=0.05, usevar='pooled')

treatment_effect = np.mean(treatment_group) - np.mean(control_group)
print(f"\nTreatment Effect Analysis:")
print(f"Point estimate of treatment effect: {treatment_effect:.2f}")
print(f"95% CI for treatment effect: {ci_treatment[0]:.2f} to {ci_treatment[1]:.2f}")

# Check if treatment effect is significant (CI doesn't include 0)
if ci_treatment[0] > 0 or ci_treatment[1] < 0:
    print("✓ Treatment effect is significant (CI doesn't include 0).")
else:
    print("✗ Treatment effect is not significant (CI includes 0).")

# Effect size calculation (Cohen's d)
pooled_sd = np.sqrt(((len(control_group) - 1) * np.var(control_group, ddof=1) + 
                     (len(treatment_group) - 1) * np.var(treatment_group, ddof=1)) / 
                    (len(control_group) + len(treatment_group) - 2))
cohens_d = treatment_effect / pooled_sd

print(f"\nEffect Size Analysis:")
print(f"Cohen's d: {cohens_d:.3f}")

if abs(cohens_d) < 0.2:
    print("Effect size: Small")
elif abs(cohens_d) < 0.5:
    print("Effect size: Medium")
elif abs(cohens_d) < 0.8:
    print("Effect size: Large")
else:
    print("Effect size: Very large")

# Power analysis
print(f"\nPower Analysis:")
print(f"Sample size per group: {len(control_group)}")
print(f"Effect size (Cohen's d): {cohens_d:.3f}")

# Approximate power calculation
power_approx = stats.norm.cdf(abs(cohens_d) * np.sqrt(len(control_group) / 2) - 1.96)
print(f"Approximate power: {power_approx:.3f}")

if power_approx < 0.8:
    print("⚠ Study may be underpowered")
else:
    print("✓ Study appears adequately powered")

# Clinical significance
print(f"\nClinical Significance:")
print(f"Treatment effect: {treatment_effect:.2f} units")
print(f"95% CI width: {ci_treatment[1] - ci_treatment[0]:.2f} units")
print(f"Precision (1/width): {1 / (ci_treatment[1] - ci_treatment[0]):.3f}")
```

## Best Practices

### Interpretation Guidelines

Proper interpretation of confidence intervals is crucial for statistical inference. Here are guidelines for correct interpretation.

#### Mathematical Foundation

The correct interpretation of a $(1-\alpha) \times 100\%$ confidence interval is:
"We are $(1-\alpha) \times 100\%$ confident that the true parameter lies within this interval."

This means that if we were to repeat the sampling process many times, approximately $(1-\alpha) \times 100\%$ of the intervals would contain the true parameter.

#### Code Implementation

```python
# Function to interpret confidence intervals
def interpret_ci(ci, parameter_name="parameter", confidence_level=0.95):
    print("=== CONFIDENCE INTERVAL INTERPRETATION ===")
    print(f"Confidence Level: {confidence_level * 100}%")
    print(f"Parameter: {parameter_name}")
    print(f"Interval: {ci[0]:.3f} to {ci[1]:.3f}\n")
    
    print("CORRECT INTERPRETATION:")
    print(f"We are {confidence_level * 100}% confident that the true {parameter_name}")
    print(f"lies between {ci[0]:.3f} and {ci[1]:.3f}.\n")
    
    print("This means that if we were to repeat this study many times,")
    print(f"approximately {confidence_level * 100}% of the confidence intervals")
    print(f"would contain the true {parameter_name}.\n")
    
    print("IMPORTANT NOTES:")
    print("✓ The confidence interval is about the method, not the parameter")
    print("✓ The parameter is fixed; the interval varies across samples")
    print("✓ A wider interval indicates more uncertainty")
    print("✓ A narrower interval indicates more precision")
    print("✓ The confidence level refers to the long-run frequency")
    
    # Calculate interval width
    width = ci[1] - ci[0]
    print(f"\nINTERVAL ANALYSIS:")
    print(f"Interval width: {width:.3f}")
    print(f"Margin of error: ±{width/2:.3f}")
    
    if width < 1:
        print("Precision: High (narrow interval)")
    elif width < 5:
        print("Precision: Moderate")
    else:
        print("Precision: Low (wide interval)")

# Example interpretation
example_ci = desc_stats.tconfint_mean(alpha=0.05)
interpret_ci(example_ci, "mean sepal length", 0.95)

# Compare different confidence levels
print(f"\n=== COMPARISON OF CONFIDENCE LEVELS ===")
confidence_levels = [0.90, 0.95, 0.99]
for level in confidence_levels:
    ci_level = desc_stats.tconfint_mean(alpha=1-level)
    print(f"Confidence level {level * 100}%: CI width = {ci_level[1] - ci_level[0]:.3f}")
```

### Common Mistakes to Avoid

Understanding common mistakes helps prevent misinterpretation and misuse of confidence intervals.

#### Mathematical Foundation

The key principle is that confidence intervals are about the method, not the parameter. The parameter is fixed, but the interval varies across samples.

#### Code Implementation

```python
# Function to demonstrate common mistakes
def demonstrate_mistakes():
    print("=== COMMON MISTAKES IN CONFIDENCE INTERVALS ===\n")
    
    print("1. ❌ Saying '95% probability that the parameter is in the interval'")
    print("   ✓ Correct: 'We are 95% confident that the interval contains the parameter'")
    print("   - The parameter is fixed, not random")
    print("   - The interval is random, not the parameter\n")
    
    print("2. ❌ Comparing confidence intervals for significance")
    print("   ✓ Correct: Use hypothesis tests for significance")
    print("   - Overlapping CIs don't necessarily mean no significant difference")
    print("   - Non-overlapping CIs don't necessarily mean significant difference\n")
    
    print("3. ❌ Using confidence intervals for individual predictions")
    print("   ✓ Correct: Use prediction intervals for individual predictions")
    print("   - CIs are for population parameters, not individual values")
    print("   - Prediction intervals are wider than confidence intervals\n")
    
    print("4. ❌ Ignoring multiple comparisons")
    print("   ✓ Correct: Use corrections like Bonferroni for multiple comparisons")
    print("   - Multiple CIs increase family-wise error rate")
    print("   - Each additional comparison increases the chance of false positives\n")
    
    print("5. ❌ Focusing only on whether 0 is in the interval")
    print("   ✓ Correct: Consider the practical significance of the interval")
    print("   - Statistical significance ≠ practical significance")
    print("   - Consider the context and magnitude of effects\n")
    
    print("6. ❌ Using the same confidence level for all analyses")
    print("   ✓ Correct: Choose confidence level based on the context")
    print("   - Higher confidence = wider intervals = less precision")
    print("   - Lower confidence = narrower intervals = less certainty\n")

demonstrate_mistakes()

# Demonstrate the overlap fallacy
print(f"\n=== DEMONSTRATION: CI OVERLAP FALLACY ===")

# Create two groups with overlapping CIs but significant difference
np.random.seed(123)
group1 = np.random.normal(10, 2, 20)
group2 = np.random.normal(12, 2, 20)

desc_stats1 = DescrStatsW(group1)
desc_stats2 = DescrStatsW(group2)
ci1 = desc_stats1.tconfint_mean(alpha=0.05)
ci2 = desc_stats2.tconfint_mean(alpha=0.05)
t_stat, p_val = stats.ttest_ind(group1, group2)

print(f"Group 1 CI: {ci1[0]:.2f} to {ci1[1]:.2f}")
print(f"Group 2 CI: {ci2[0]:.2f} to {ci2[1]:.2f}")
print(f"CIs overlap: {(ci1[1] > ci2[0] and ci2[1] > ci1[0])}")
print(f"t-test p-value: {p_val:.4f}")
print(f"Significant difference: {p_val < 0.05}")

if p_val < 0.05 and (ci1[1] > ci2[0] and ci2[1] > ci1[0]):
    print("⚠ This demonstrates the overlap fallacy!")
    print("   CIs overlap but groups are significantly different.")
```

## Exercises

### Exercise 1: Basic Confidence Intervals
Calculate 90%, 95%, and 99% confidence intervals for the mean MPG of cars with 6 cylinders.

**Hints:**
- Use `subset()` or logical indexing to get cars with 6 cylinders
- Use `t.test()` with different `conf.level` values
- Compare the widths of the intervals

### Exercise 2: Proportion Confidence Intervals
Calculate confidence intervals for the proportion of cars with manual transmission, using both normal approximation and exact binomial methods.

**Hints:**
- Check the large sample conditions for normal approximation
- Use `binom.test()` for exact method
- Compare the results and explain any differences

### Exercise 3: Difference Between Means
Calculate a confidence interval for the difference in MPG between cars with 4 and 8 cylinders.

**Hints:**
- Test for equal variances using `var.test()`
- Use both pooled and Welch's t-test
- Interpret the results in context

### Exercise 4: Bootstrap Confidence Intervals
Use bootstrap sampling to calculate confidence intervals for the median and standard deviation of the iris sepal length data.

**Hints:**
- Load the `iris` dataset: `data(iris)`
- Use the `boot` package
- Compare bootstrap CIs with parametric CIs

### Exercise 5: Sample Size Planning
Determine the sample size needed to estimate the mean MPG with a margin of error of ±1 MPG at 95% confidence.

**Hints:**
- Use the formula: $n = \left(\frac{z_{\alpha/2} \cdot s}{E}\right)^2$
- Estimate $s$ from the current data
- Consider different confidence levels

### Exercise 6: Multiple Comparisons
Calculate confidence intervals for the mean MPG by cylinder type (4, 6, 8) using Bonferroni correction.

**Hints:**
- Calculate the number of comparisons
- Adjust the individual confidence level
- Compare with uncorrected intervals

### Exercise 7: Effect Size and Power
For the treatment effect example in the chapter:
- Calculate Cohen's d effect size
- Determine if the study is adequately powered
- Suggest sample size for 80% power

### Exercise 8: Confidence Interval Width Analysis
Investigate how confidence interval width changes with:
- Sample size (use different subsamples)
- Confidence level (80%, 90%, 95%, 99%)
- Population variability (compare different variables)

### Exercise 9: Practical Applications
Choose a real dataset and:
- Calculate confidence intervals for relevant parameters
- Interpret the results in context
- Discuss practical implications

### Exercise 10: Advanced Bootstrap
Implement different bootstrap confidence interval methods:
- Normal bootstrap
- Percentile bootstrap
- Basic bootstrap
- BCa bootstrap (if available)

Compare the results and discuss when each method is appropriate.

---

**Solutions and Additional Resources:**
- Use `?t.test` for help with t-test functions
- Use `?binom.test` for help with binomial tests
- Use `?boot` for help with bootstrap functions
- Consider using `ggplot2` for visualization
- Practice with different datasets to build intuition

## Next Steps

In the next chapter, we'll learn about hypothesis testing, which complements confidence intervals for statistical inference.

---

**Key Takeaways:**
- Confidence intervals provide ranges of plausible parameter values
- Use t-distribution when population standard deviation is unknown
- Bootstrap methods are useful for non-parametric inference
- Sample size affects the width of confidence intervals
- Higher confidence levels result in wider intervals
- Always interpret confidence intervals correctly
- Consider multiple comparisons when making many inferences 