# Hypothesis Testing

## Overview

Hypothesis testing is a fundamental concept in statistics that allows us to make decisions about populations based on sample data. It provides a framework for testing claims about parameters or relationships in data through a systematic process of statistical inference.

### What is Hypothesis Testing?

Hypothesis testing is a statistical method used to determine whether there is enough evidence in a sample to support a particular claim about a population parameter. It involves making an initial assumption (the null hypothesis) and then determining whether the observed data provides sufficient evidence to reject this assumption.

### The Scientific Method and Hypothesis Testing

Hypothesis testing follows the scientific method:
1. **Observation**: Collect data from a sample
2. **Hypothesis**: Formulate null and alternative hypotheses
3. **Prediction**: Determine what we expect to observe if the null hypothesis is true
4. **Testing**: Calculate a test statistic and compare it to a critical value
5. **Conclusion**: Make a decision based on the evidence

### Key Components of Hypothesis Testing

1. **Null Hypothesis ($H_0$)**: The default assumption about the population parameter
2. **Alternative Hypothesis ($H_1$ or $H_a$)**: The claim we want to support
3. **Test Statistic**: A calculated value that measures the strength of evidence
4. **Significance Level ($\alpha$)**: The probability of rejecting $H_0$ when it's true
5. **P-value**: The probability of observing the test statistic or more extreme values
6. **Decision Rule**: Criteria for rejecting or failing to reject $H_0$

### Mathematical Foundation

The general framework for hypothesis testing involves:

```math
\text{Test Statistic} = \frac{\text{Sample Statistic} - \text{Parameter under } H_0}{\text{Standard Error}}
```

The decision rule is:
- Reject $H_0$ if $|T| > t_{\alpha/2, df}$ (two-tailed test)
- Reject $H_0$ if $T > t_{\alpha, df}$ (one-tailed test, upper)
- Reject $H_0$ if $T < -t_{\alpha, df}$ (one-tailed test, lower)

Where $T$ is the test statistic and $t_{\alpha, df}$ is the critical value from the appropriate distribution.

## Basic Concepts

### Null and Alternative Hypotheses

The foundation of hypothesis testing lies in formulating two competing hypotheses about a population parameter.

#### Mathematical Foundation

**Null Hypothesis ($H_0$)**: The default assumption that there is no effect, no difference, or no relationship. It represents the status quo or the claim we want to test against.

**Alternative Hypothesis ($H_1$ or $H_a$)**: The claim we want to support. It represents the research hypothesis or the effect we're trying to detect.

For a population parameter $\theta$, the hypotheses can be:

**Two-tailed test:**
```math
H_0: \theta = \theta_0 \quad \text{vs.} \quad H_1: \theta \neq \theta_0
```

**One-tailed test (upper):**
```math
H_0: \theta \leq \theta_0 \quad \text{vs.} \quad H_1: \theta > \theta_0
```

**One-tailed test (lower):**
```math
H_0: \theta \geq \theta_0 \quad \text{vs.} \quad H_1: \theta < \theta_0
```

#### Code Implementation

```python
# Example: Testing if a coin is fair
# H0: p = 0.5 (coin is fair)
# H1: p ≠ 0.5 (coin is not fair)

import numpy as np
from scipy import stats
import pandas as pd
from sklearn.datasets import load_iris

# Simulate coin flips
np.random.seed(123)
coin_flips = np.random.binomial(1, 0.6, 100)  # Biased coin
head_count = np.sum(coin_flips)
total_flips = len(coin_flips)
observed_proportion = head_count / total_flips

print("=== Coin Fairness Test ===")
print("Null hypothesis (H0): p = 0.5 (coin is fair)")
print("Alternative hypothesis (H1): p ≠ 0.5 (coin is not fair)")
print("Test type: Two-tailed\n")

print("Data Summary:")
print(f"Total flips: {total_flips}")
print(f"Heads observed: {head_count}")
print(f"Observed proportion of heads: {observed_proportion:.3f}")
print(f"Expected proportion under H0: 0.5")
print(f"Difference from expected: {observed_proportion - 0.5:.3f}")

# Calculate test statistic
expected_heads = total_flips * 0.5
standard_error = np.sqrt(total_flips * 0.5 * 0.5)
z_statistic = (head_count - expected_heads) / standard_error

print(f"\nTest Statistic Calculation:")
print(f"Expected heads under H0: {expected_heads}")
print(f"Standard error: {standard_error:.3f}")
print(f"Z-statistic: {z_statistic:.3f}")

# Calculate p-value
p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
print(f"P-value: {p_value:.4f}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Decision: Reject H0 (coin is not fair)")
else:
    print("Decision: Fail to reject H0 (insufficient evidence)")
```

### Type I and Type II Errors

Understanding the types of errors that can occur in hypothesis testing is crucial for interpreting results and making informed decisions.

#### Mathematical Foundation

**Type I Error ($\alpha$)**: The probability of rejecting $H_0$ when it's actually true.
```math
\alpha = P(\text{Reject } H_0 | H_0 \text{ is true})
```

**Type II Error ($\beta$)**: The probability of failing to reject $H_0$ when it's actually false.
```math
\beta = P(\text{Fail to reject } H_0 | H_1 \text{ is true})
```

**Power ($1 - \beta$)**: The probability of correctly rejecting $H_0$ when it's false.
```math
\text{Power} = 1 - \beta = P(\text{Reject } H_0 | H_1 \text{ is true})
```

The relationship between these concepts:
```math
\alpha + \text{Power} \leq 1
```

#### Decision Matrix

| Decision | $H_0$ True | $H_0$ False |
|----------|-------------|-------------|
| Reject $H_0$ | Type I Error ($\alpha$) | Correct Decision (Power) |
| Fail to reject $H_0$ | Correct Decision ($1-\alpha$) | Type II Error ($\beta$) |

#### Code Implementation

```python
# Understanding error types
print("=== Type I and Type II Errors ===")
print("Type I Error (α): Rejecting H0 when it's true")
print("Type II Error (β): Failing to reject H0 when it's false")
print("Power (1-β): Probability of correctly rejecting H0\n")

# Example: Setting significance level
alpha = 0.05
print(f"Significance level (α): {alpha}")
print("This means we're willing to make a Type I error 5% of the time\n")

# Demonstrate the trade-off between Type I and Type II errors
print("=== Error Trade-off Demonstration ===")

# Function to calculate power for different effect sizes
def calculate_power(effect_size, n, alpha=0.05):
    # For a two-sample t-test, approximate power calculation
    # This is a simplified version - in practice, use statsmodels or specialized packages
    from scipy.stats import norm
    
    # Critical value
    z_alpha = norm.ppf(1 - alpha/2)
    
    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n/2)
    
    # Power calculation
    power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    return power

# Test different scenarios
effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large
sample_sizes = [20, 50, 100]
alpha_levels = [0.01, 0.05, 0.10]

print("Power Analysis for Different Scenarios:")
print("Effect Size | Sample Size | α = 0.01 | α = 0.05 | α = 0.10")
print("-----------|-------------|-----------|-----------|-----------")

for d in effect_sizes:
    for n in sample_sizes:
        power_01 = calculate_power(d, n, 0.01)
        power_05 = calculate_power(d, n, 0.05)
        power_10 = calculate_power(d, n, 0.10)
        print(f"{d:.1f}        | {n}          | {power_01:.3f}     | {power_05:.3f}     | {power_10:.3f}")

print("\nKey Insights:")
print("• Lower α (Type I error) → Lower power (higher Type II error)")
print("• Larger effect size → Higher power")
print("• Larger sample size → Higher power")
print("• There's always a trade-off between Type I and Type II errors")
```

## One-Sample Tests

One-sample tests are used to compare a sample statistic to a hypothesized population parameter.

### One-Sample t-Test

The one-sample t-test is used to determine whether the mean of a single sample differs from a hypothesized population mean.

#### Mathematical Foundation

For a sample of size $n$ with mean $\bar{x}$ and standard deviation $s$, the test statistic is:

```math
t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}
```

Where $\mu_0$ is the hypothesized population mean under $H_0$.

The test statistic follows a t-distribution with $n-1$ degrees of freedom.

**Hypotheses:**
```math
H_0: \mu = \mu_0 \quad \text{vs.} \quad H_1: \mu \neq \mu_0
```

#### Assumptions

1. **Independence**: Observations are independent
2. **Normality**: The population is normally distributed (or sample size is large)
3. **Random Sampling**: Data comes from a random sample

#### Code Implementation

```python
# Load data
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Test if mean sepal length is different from 5.5
sepal_length = data['sepal length (cm)']
t_stat, p_value = stats.ttest_1samp(sepal_length, 5.5)

print("=== One-Sample t-Test ===")
print("Hypothesis: H0: μ = 5.5 vs. H1: μ ≠ 5.5")
print("Test type: Two-tailed\n")

# View results
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.6f}")

# Extract components
print(f"\nDetailed Results:")
print(f"Sample mean: {sepal_length.mean():.3f}")
print(f"Hypothesized mean: 5.5")
print(f"Test statistic (t): {t_stat:.3f}")
print(f"Degrees of freedom: {len(sepal_length) - 1}")
print(f"P-value: {p_value:.4f}")

# Calculate confidence interval
from statsmodels.stats.weightstats import DescrStatsW
desc_stats = DescrStatsW(sepal_length)
ci = desc_stats.tconfint_mean(alpha=0.05)
print(f"95% Confidence interval: {ci[0]:.3f} to {ci[1]:.3f}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Decision: Reject H0 (mean sepal length differs from 5.5)")
else:
    print("Decision: Fail to reject H0 (insufficient evidence)")

# Effect size calculation
def cohens_d_one_sample(data, mu0):
    """Calculate Cohen's d for one-sample t-test"""
    return (data.mean() - mu0) / data.std(ddof=1)

effect_size = cohens_d_one_sample(sepal_length, 5.5)
print(f"Effect size (Cohen's d): {effect_size:.3f}")

# Interpret effect size
if abs(effect_size) < 0.2:
    print("Effect size: Small")
elif abs(effect_size) < 0.5:
    print("Effect size: Medium")
elif abs(effect_size) < 0.8:
    print("Effect size: Large")
else:
    print("Effect size: Very large")

# Check assumptions
print(f"\n=== Assumption Checking ===")

# Normality test
shapiro_stat, shapiro_p = stats.shapiro(sepal_length)
print("Shapiro-Wilk normality test:")
print(f"W = {shapiro_stat:.3f}")
print(f"P-value = {shapiro_p:.4f}")

if shapiro_p < 0.05:
    print("Warning: Data may not be normally distributed")
else:
    print("✓ Data appears to be normally distributed")

# Sample size check
print(f"Sample size: {len(sepal_length)}")
if len(sepal_length) >= 30:
    print("✓ Sample size is adequate for t-test (Central Limit Theorem)")
else:
    print("⚠ Small sample size - normality assumption is important")
```

### One-Sample Proportion Test

The one-sample proportion test is used to determine whether a sample proportion differs from a hypothesized population proportion.

#### Mathematical Foundation

For a sample of size $n$ with $x$ successes, the sample proportion is $\hat{p} = x/n$.

**Large Sample Approximation (Normal):**
When $np \geq 10$ and $n(1-p) \geq 10$, the test statistic is:

```math
z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}
```

Where $p_0$ is the hypothesized population proportion under $H_0$.

**Exact Test (Binomial):**
For small samples, use the exact binomial test:

```math
P(X \geq x) = \sum_{k=x}^n \binom{n}{k} p_0^k (1-p_0)^{n-k}
```

**Hypotheses:**
```math
H_0: p = p_0 \quad \text{vs.} \quad H_1: p \neq p_0
```

#### Code Implementation

```python
# Test proportion of sepal length > 5.5 cm
data['long_sepal'] = (data['sepal length (cm)'] > 5.5).astype(int)
long_count = data['long_sepal'].sum()
total_flowers = len(data['long_sepal'])
long_proportion = long_count / total_flowers

print("=== One-Sample Proportion Test ===")
print("Hypothesis: H0: p = 0.5 vs. H1: p ≠ 0.5")
print("Test type: Two-tailed\n")

print("Data Summary:")
print(f"Total flowers: {total_flowers}")
print(f"Long sepals (>5.5 cm): {long_count}")
print(f"Sample proportion: {long_proportion:.3f}")
print(f"Hypothesized proportion: 0.5")

# Check large sample conditions
np_check = total_flowers * 0.5
nq_check = total_flowers * 0.5
print(f"\nLarge Sample Conditions:")
print(f"n*p = {np_check} (should be ≥ 10): {'✓' if np_check >= 10 else '✗'}")
print(f"n*(1-p) = {nq_check} (should be ≥ 10): {'✓' if nq_check >= 10 else '✗'}")

# Test if proportion is different from 0.5
binom_result = stats.binomtest(long_count, total_flowers, p=0.5)

print(f"\nExact Binomial Test Results:")
print(f"Test statistic: {binom_result.statistic}")
print(f"P-value: {binom_result.pvalue:.4f}")
ci = binom_result.proportions_ci(confidence_level=0.95)
print(f"95% Confidence interval: {ci[0]:.4f} to {ci[1]:.4f}")

# Decision
alpha = 0.05
if binom_result.pvalue < alpha:
    print("Decision: Reject H0 (proportion differs from 0.5)")
else:
    print("Decision: Fail to reject H0 (insufficient evidence)")

# Normal approximation for comparison
# Calculate z-statistic manually
z_stat = (long_proportion - 0.5) / np.sqrt(0.5 * 0.5 / total_flowers)
p_value_normal = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"\nNormal Approximation Results:")
print(f"Z-statistic: {z_stat:.3f}")
print(f"P-value: {p_value_normal:.4f}")

# Compare methods
print(f"\nMethod Comparison:")
print(f"Normal approximation p-value: {p_value_normal:.4f}")
print(f"Exact binomial p-value: {binom_result.pvalue:.4f}")
print(f"Difference: {abs(p_value_normal - binom_result.pvalue):.4f}")

if abs(p_value_normal - binom_result.pvalue) < 0.01:
    print("✓ Methods give similar results (large sample conditions met)")
else:
    print("⚠ Methods differ - use exact test for small samples")

# Effect size (difference from hypothesized proportion)
effect_size = abs(long_proportion - 0.5)
print(f"\nEffect Size Analysis:")
print(f"Absolute difference from 0.5: {effect_size:.3f}")

if effect_size < 0.1:
    print("Effect size: Small")
elif effect_size < 0.2:
    print("Effect size: Medium")
else:
    print("Effect size: Large")
```

### One-Sample Variance Test

The one-sample variance test (chi-square test for variance) is used to determine whether a sample variance differs from a hypothesized population variance.

#### Mathematical Foundation

For a sample of size $n$ with variance $s^2$, the test statistic is:

```math
\chi^2 = \frac{(n-1)s^2}{\sigma_0^2}
```

Where $\sigma_0^2$ is the hypothesized population variance under $H_0$.

The test statistic follows a chi-square distribution with $n-1$ degrees of freedom.

**Hypotheses:**
```math
H_0: \sigma^2 = \sigma_0^2 \quad \text{vs.} \quad H_1: \sigma^2 \neq \sigma_0^2
```

**Assumptions:**
1. **Normality**: The population is normally distributed
2. **Independence**: Observations are independent
3. **Random Sampling**: Data comes from a random sample

#### Code Implementation

```python
# Test if variance of sepal length is different from 0.5
sepal_variance = sepal_length.var()
n = len(sepal_length)
hypothesized_variance = 0.5

print("=== One-Sample Variance Test ===")
print("Hypothesis: H0: σ² = 0.5 vs. H1: σ² ≠ 0.5")
print("Test type: Two-tailed\n")

print("Data Summary:")
print(f"Sample size: {n}")
print(f"Sample variance: {sepal_variance:.3f}")
print(f"Hypothesized variance: {hypothesized_variance}")
print(f"Sample standard deviation: {sepal_length.std():.3f}")
print(f"Hypothesized standard deviation: {np.sqrt(hypothesized_variance):.3f}")

# Calculate test statistic
test_statistic = (n - 1) * sepal_variance / hypothesized_variance
df = n - 1

print(f"\nTest Statistic Calculation:")
print(f"Degrees of freedom: {df}")
print(f"Chi-square statistic: {test_statistic:.3f}")

# Calculate p-value (two-tailed)
p_value_lower = stats.chi2.cdf(test_statistic, df)
p_value_upper = 1 - stats.chi2.cdf(test_statistic, df)
p_value = 2 * min(p_value_lower, p_value_upper)

print(f"P-value (two-tailed): {p_value:.4f}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Decision: Reject H0 (variance differs from 0.5)")
else:
    print("Decision: Fail to reject H0 (insufficient evidence)")

# Calculate confidence interval for variance
chi_lower = stats.chi2.ppf(0.025, df)
chi_upper = stats.chi2.ppf(0.975, df)
ci_lower_var = (df * sepal_variance) / chi_upper
ci_upper_var = (df * sepal_variance) / chi_lower

print(f"\nConfidence Interval for Variance:")
print(f"95% CI for variance: {ci_lower_var:.3f} to {ci_upper_var:.3f}")

# Check if hypothesized variance is in confidence interval
if ci_lower_var <= hypothesized_variance <= ci_upper_var:
    print("✓ Hypothesized variance is within the confidence interval")
else:
    print("✗ Hypothesized variance is outside the confidence interval")

# Effect size (ratio of variances)
variance_ratio = sepal_variance / hypothesized_variance
print(f"\nEffect Size Analysis:")
print(f"Variance ratio (sample/hypothesized): {variance_ratio:.3f}")

if abs(variance_ratio - 1) < 0.2:
    print("Effect size: Small")
elif abs(variance_ratio - 1) < 0.5:
    print("Effect size: Medium")
else:
    print("Effect size: Large")

# Check normality assumption
print(f"\n=== Assumption Checking ===")
shapiro_stat, shapiro_p = stats.shapiro(sepal_length)
print("Shapiro-Wilk normality test:")
print(f"W = {shapiro_stat:.3f}")
print(f"P-value = {shapiro_p:.4f}")

if shapiro_p < 0.05:
    print("⚠ Warning: Data may not be normally distributed")
    print("   Chi-square test for variance is sensitive to non-normality")
    print("   Consider using nonparametric alternatives")
else:
    print("✓ Data appears to be normally distributed")

# Alternative: Levene's test for homogeneity of variance
print(f"\nAlternative: Levene's Test")
print("Note: Levene's test is typically used for comparing variances across groups")
```

## Two-Sample Tests

Two-sample tests are used to compare parameters between two independent or paired groups.

### Independent t-Test

The independent t-test (two-sample t-test) is used to compare the means of two independent groups.

#### Mathematical Foundation

For two independent samples with means $\bar{x}_1$ and $\bar{x}_2$, the test statistic is:

**Equal Variances (Pooled):**
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
```

Where the pooled standard deviation is:
```math
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
```

**Unequal Variances (Welch's):**
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
```

With degrees of freedom (Welch's approximation):
```math
df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
```

**Hypotheses:**
```math
H_0: \mu_1 = \mu_2 \quad \text{vs.} \quad H_1: \mu_1 \neq \mu_2
```

#### Assumptions

1. **Independence**: Observations are independent within and between groups
2. **Normality**: Both populations are normally distributed (or sample sizes are large)
3. **Equal Variances**: Population variances are equal (for pooled t-test)
4. **Random Sampling**: Data comes from random samples

#### Code Implementation

```python
# Compare sepal length between different iris species
# Group 1: Setosa (target = 0), Group 2: Versicolor (target = 1)
setosa_sepal = data[data['target'] == 0]['sepal length (cm)']
versicolor_sepal = data[data['target'] == 1]['sepal length (cm)']

print("=== Independent t-Test ===")
print("Hypothesis: H0: μ_setosa = μ_versicolor vs. H1: μ_setosa ≠ μ_versicolor")
print("Test type: Two-tailed\n")

print("Data Summary:")
print(f"Setosa: n = {len(setosa_sepal)}, mean = {setosa_sepal.mean():.3f}, SD = {setosa_sepal.std():.3f}")
print(f"Versicolor: n = {len(versicolor_sepal)}, mean = {versicolor_sepal.mean():.3f}, SD = {versicolor_sepal.std():.3f}")

# Check assumptions
print(f"\n=== Assumption Checking ===")

# Normality tests
shapiro_setosa_stat, shapiro_setosa_p = stats.shapiro(setosa_sepal)
shapiro_versicolor_stat, shapiro_versicolor_p = stats.shapiro(versicolor_sepal)

print("Normality Tests:")
print(f"Setosa - W = {shapiro_setosa_stat:.3f}, p = {shapiro_setosa_p:.4f}")
print(f"Versicolor - W = {shapiro_versicolor_stat:.3f}, p = {shapiro_versicolor_p:.4f}")

# Variance ratio test
variance_ratio = setosa_sepal.var() / versicolor_sepal.var()
print(f"Variance ratio: {variance_ratio:.3f}")

if variance_ratio > 2 or variance_ratio < 0.5:
    print("⚠ Variance ratio suggests unequal variances - use Welch's t-test")
else:
    print("✓ Variances appear approximately equal")

# Levene's test for equality of variances
levene_stat, levene_p = stats.levene(setosa_sepal, versicolor_sepal)
print(f"Levene's test for equality of variances: F = {levene_stat:.3f}, p = {levene_p:.4f}")

if levene_p < 0.05:
    print("✓ Variances are significantly different - use Welch's t-test")
else:
    print("✓ Variances are not significantly different - pooled t-test is appropriate")

# Perform t-tests
print(f"\n=== Test Results ===")

# Pooled t-test (equal variances)
pooled_t_stat, pooled_p = stats.ttest_ind(setosa_sepal, versicolor_sepal, equal_var=True)
print("Pooled t-test (equal variances):")
print(f"t = {pooled_t_stat:.3f}, p = {pooled_p:.4f}")

# Welch's t-test (unequal variances)
welch_t_stat, welch_p = stats.ttest_ind(setosa_sepal, versicolor_sepal, equal_var=False)
print("Welch's t-test (unequal variances):")
print(f"t = {welch_t_stat:.3f}, p = {welch_p:.4f}")

# Decision
alpha = 0.05
if welch_p < alpha:
    print("Decision: Reject H0 (means differ significantly)")
else:
    print("Decision: Fail to reject H0 (insufficient evidence)")

# Effect size
def cohens_d_two_sample(x1, x2):
    """Calculate Cohen's d for two independent samples"""
    pooled_std = np.sqrt(((len(x1) - 1) * x1.var() + (len(x2) - 1) * x2.var()) / (len(x1) + len(x2) - 2))
    return (x1.mean() - x2.mean()) / pooled_std

effect_size = cohens_d_two_sample(setosa_sepal, versicolor_sepal)
print(f"\nEffect Size (Cohen's d): {abs(effect_size):.3f}")

if abs(effect_size) < 0.2:
    print("Effect size: Small")
elif abs(effect_size) < 0.5:
    print("Effect size: Medium")
elif abs(effect_size) < 0.8:
    print("Effect size: Large")
else:
    print("Effect size: Very large")

# Confidence interval using statsmodels
from statsmodels.stats.weightstats import CompareMeans
cm = CompareMeans.from_data(setosa_sepal, versicolor_sepal)
ci = cm.tconfint_diff(alpha=0.05, usevar='unequal')
print(f"95% CI for difference: {ci[0]:.3f} to {ci[1]:.3f}")

# Practical significance
mean_diff = setosa_sepal.mean() - versicolor_sepal.mean()
print(f"Mean difference (setosa - versicolor): {mean_diff:.3f}")

if abs(mean_diff) > 0.5:
    print("✓ Practical significance: Large difference in sepal length")
elif abs(mean_diff) > 0.2:
    print("⚠ Practical significance: Moderate difference in sepal length")
else:
    print("✗ Practical significance: Small difference in sepal length")
```

### Paired t-Test

The paired t-test is used to compare the means of two related groups, such as before/after measurements on the same subjects.

#### Mathematical Foundation

For paired observations $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$, we work with the differences $d_i = x_i - y_i$.

The test statistic is:
```math
t = \frac{\bar{d}}{s_d/\sqrt{n}}
```

Where:
- $\bar{d}$ is the mean of the differences
- $s_d$ is the standard deviation of the differences
- $n$ is the number of pairs

The test statistic follows a t-distribution with $n-1$ degrees of freedom.

**Hypotheses:**
```math
H_0: \mu_d = 0 \quad \text{vs.} \quad H_1: \mu_d \neq 0
```

#### Advantages of Paired Tests

1. **Reduced Variability**: Removes between-subject variability
2. **Increased Power**: More sensitive to detect treatment effects
3. **Control for Confounding**: Each subject serves as their own control

#### Code Implementation

```python
# Simulate paired data (before/after treatment)
np.random.seed(123)
before = np.random.normal(100, 15, 20)
after = before + np.random.normal(5, 10, 20)

print("=== Paired t-Test ===")
print("Hypothesis: H0: μ_d = 0 vs. H1: μ_d ≠ 0")
print("Test type: Two-tailed\n")

print("Data Summary:")
print(f"Sample size (pairs): {len(before)}")
print(f"Before treatment - mean = {before.mean():.2f}, SD = {before.std():.2f}")
print(f"After treatment - mean = {after.mean():.2f}, SD = {after.std():.2f}")

# Calculate differences
differences = after - before
print(f"Differences - mean = {differences.mean():.2f}, SD = {differences.std():.2f}")

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(after, before)

print(f"\n=== Test Results ===")
print(f"t-statistic: {t_stat:.3f}")
print(f"Degrees of freedom: {len(before) - 1}")
print(f"P-value: {p_value:.4f}")

# Calculate confidence interval
desc_stats = DescrStatsW(differences)
ci = desc_stats.tconfint_mean(alpha=0.05)
print(f"95% CI for mean difference: {ci[0]:.2f} to {ci[1]:.2f}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Decision: Reject H0 (treatment has significant effect)")
else:
    print("Decision: Fail to reject H0 (insufficient evidence)")

# Effect size for paired samples
def cohens_d_paired(differences):
    """Calculate Cohen's d for paired samples"""
    return differences.mean() / differences.std(ddof=1)

effect_size = cohens_d_paired(differences)
print(f"\nEffect Size (Cohen's d for paired samples): {abs(effect_size):.3f}")

if abs(effect_size) < 0.2:
    print("Effect size: Small")
elif abs(effect_size) < 0.5:
    print("Effect size: Medium")
elif abs(effect_size) < 0.8:
    print("Effect size: Large")
else:
    print("Effect size: Very large")

# Compare with independent t-test (incorrect for paired data)
independent_t_stat, independent_p = stats.ttest_ind(after, before)
print(f"\n=== Comparison with Independent t-Test ===")
print(f"Independent t-test p-value: {independent_p:.4f}")
print(f"Paired t-test p-value: {p_value:.4f}")

if p_value < independent_p:
    print("✓ Paired test is more powerful (as expected)")
else:
    print("⚠ Paired test is not more powerful (unusual)")

# Check normality of differences
print(f"\n=== Assumption Checking ===")
shapiro_diff_stat, shapiro_diff_p = stats.shapiro(differences)
print("Shapiro-Wilk test for differences:")
print(f"W = {shapiro_diff_stat:.3f}, p = {shapiro_diff_p:.4f}")

if shapiro_diff_p < 0.05:
    print("⚠ Differences may not be normally distributed")
    print("   Consider using Wilcoxon signed-rank test")
else:
    print("✓ Differences appear to be normally distributed")

# Correlation between before and after
correlation = np.corrcoef(before, after)[0, 1]
print(f"Correlation between before and after: {correlation:.3f}")

if correlation > 0.5:
    print("✓ Strong correlation - paired test is appropriate")
elif correlation > 0.3:
    print("⚠ Moderate correlation - paired test may still be beneficial")
else:
    print("✗ Weak correlation - independent test might be more appropriate")

# Practical significance
mean_diff = differences.mean()
print(f"\nPractical Significance:")
print(f"Mean treatment effect: {mean_diff:.2f} units")

if abs(mean_diff) > 10:
    print("✓ Large practical effect")
elif abs(mean_diff) > 5:
    print("⚠ Moderate practical effect")
else:
    print("✗ Small practical effect")
```

### Two-Sample Proportion Test

The two-sample proportion test is used to compare proportions between two independent groups.

#### Mathematical Foundation

For two independent samples with proportions $\hat{p}_1$ and $\hat{p}_2$, the test statistic is:

**Large Sample Approximation:**
```math
z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}
```

Where the pooled proportion is:
```math
\hat{p} = \frac{x_1 + x_2}{n_1 + n_2}
```

**Hypotheses:**
```math
H_0: p_1 = p_2 \quad \text{vs.} \quad H_1: p_1 \neq p_2
```

#### Assumptions

1. **Independence**: Observations are independent within and between groups
2. **Large Sample**: $n_1p_1 \geq 10$, $n_1(1-p_1) \geq 10$, $n_2p_2 \geq 10$, $n_2(1-p_2) \geq 10$
3. **Random Sampling**: Data comes from random samples

#### Code Implementation

```python
# Compare proportions between two groups
group1_success = 15
group1_total = 50
group2_success = 25
group2_total = 60

print("=== Two-Sample Proportion Test ===")
print("Hypothesis: H0: p1 = p2 vs. H1: p1 ≠ p2")
print("Test type: Two-tailed\n")

print("Data Summary:")
print(f"Group 1: Successes = {group1_success}, Total = {group1_total}, Proportion = {group1_success/group1_total:.3f}")
print(f"Group 2: Successes = {group2_success}, Total = {group2_total}, Proportion = {group2_success/group2_total:.3f}")

# Check large sample conditions
p1 = group1_success / group1_total
p2 = group2_success / group2_total

print(f"\nLarge Sample Conditions:")
print(f"Group 1 - n1*p1 = {group1_total * p1}, n1*(1-p1) = {group1_total * (1-p1)}")
print(f"Group 2 - n2*p2 = {group2_total * p2}, n2*(1-p2) = {group2_total * (1-p2)}")

large_sample_ok = ((group1_total * p1 >= 10) and (group1_total * (1-p1) >= 10) and
                   (group2_total * p2 >= 10) and (group2_total * (1-p2) >= 10))

if large_sample_ok:
    print("✓ Large sample conditions are met")
else:
    print("⚠ Large sample conditions may not be met")

# Perform proportion test using chi-square test
from scipy.stats import chi2_contingency
contingency_table = np.array([[group1_success, group1_total - group1_success],
                              [group2_success, group2_total - group2_success]])

chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"\n=== Test Results ===")
print(f"Chi-square statistic: {chi2_stat:.3f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_value:.4f}")

# Calculate confidence interval for difference in proportions
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
z_stat, p_val = proportions_ztest([group1_success, group2_success], 
                                  [group1_total, group2_total])
ci = confint_proportions_2indep(group1_success, group1_total, 
                               group2_success, group2_total, 
                               method='wald')
print(f"95% CI for difference: {ci[0]:.4f} to {ci[1]:.4f}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Decision: Reject H0 (proportions differ significantly)")
else:
    print("Decision: Fail to reject H0 (insufficient evidence)")

# Effect size (difference in proportions)
prop_diff = p2 - p1
print(f"\nEffect Size Analysis:")
print(f"Difference in proportions (p2 - p1): {prop_diff:.3f}")

if abs(prop_diff) < 0.1:
    print("Effect size: Small")
elif abs(prop_diff) < 0.2:
    print("Effect size: Medium")
else:
    print("Effect size: Large")

# Risk ratio and odds ratio
risk_ratio = p2 / p1
odds1 = p1 / (1 - p1)
odds2 = p2 / (1 - p2)
odds_ratio = odds2 / odds1

print(f"\nAdditional Measures:")
print(f"Risk ratio (p2/p1): {risk_ratio:.3f}")
print(f"Odds ratio: {odds_ratio:.3f}")

# Practical significance
print(f"\nPractical Significance:")
if abs(prop_diff) > 0.2:
    print("✓ Large practical difference")
elif abs(prop_diff) > 0.1:
    print("⚠ Moderate practical difference")
else:
    print("✗ Small practical difference")

# Fisher's exact test for small samples
if not large_sample_ok:
    print(f"\n=== Fisher's Exact Test (for small samples) ===")
    
    fisher_stat, fisher_p = stats.fisher_exact(contingency_table)
    print(f"Fisher's exact test p-value: {fisher_p:.4f}")
    
    if fisher_p < alpha:
        print("Decision: Reject H0 (proportions differ significantly)")
    else:
        print("Decision: Fail to reject H0 (insufficient evidence)")
```

## Nonparametric Tests

Nonparametric tests make fewer assumptions about the underlying population distribution and are often used when parametric assumptions are violated.

### Wilcoxon Rank-Sum Test (Mann-Whitney U)

The Wilcoxon rank-sum test is a nonparametric alternative to the independent t-test that tests whether two independent samples come from populations with the same distribution.

#### Mathematical Foundation

The test statistic is based on the sum of ranks. For two samples of sizes $n_1$ and $n_2$:

```math
W = \sum_{i=1}^{n_1} R_i
```

Where $R_i$ are the ranks of the first sample when all observations are ranked together.

The test statistic follows approximately a normal distribution:
```math
Z = \frac{W - \mu_W}{\sigma_W}
```

Where:
```math
\mu_W = \frac{n_1(n_1 + n_2 + 1)}{2}
```

```math
\sigma_W = \sqrt{\frac{n_1 n_2(n_1 + n_2 + 1)}{12}}
```

**Hypotheses:**
```math
H_0: \text{The two populations have the same distribution}
```

```math
H_1: \text{The two populations have different distributions}
```

#### Assumptions

1. **Independence**: Observations are independent within and between groups
2. **Random Sampling**: Data comes from random samples
3. **Ordinal Data**: Data can be ranked (no normality assumption)

#### Code Implementation

```python
# Nonparametric alternative to independent t-test
wilcox_stat, wilcox_p = stats.mannwhitneyu(setosa_sepal, versicolor_sepal, alternative='two-sided')

print("=== Wilcoxon Rank-Sum Test ===")
print("Hypothesis: H0: Same distribution vs. H1: Different distributions")
print("Test type: Two-tailed\n")

print("Data Summary:")
print(f"Setosa: n = {len(setosa_sepal)}, median = {np.median(setosa_sepal):.3f}")
print(f"Versicolor: n = {len(versicolor_sepal)}, median = {np.median(versicolor_sepal):.3f}")

print(f"\n=== Test Results ===")
print(f"U-statistic: {wilcox_stat:.3f}")
print(f"P-value: {wilcox_p:.4f}")

# Decision
alpha = 0.05
if wilcox_p < alpha:
    print("Decision: Reject H0 (distributions differ significantly)")
else:
    print("Decision: Fail to reject H0 (insufficient evidence)")

# Compare with parametric t-test
print(f"\n=== Comparison with Parametric Test ===")
t_test_result = stats.ttest_ind(setosa_sepal, versicolor_sepal)
print(f"t-test p-value: {t_test_result.pvalue:.4f}")
print(f"Wilcoxon p-value: {wilcox_p:.4f}")

if abs(t_test_result.pvalue - wilcox_p) < 0.01:
    print("✓ Both tests give similar results")
else:
    print("⚠ Tests give different results - check assumptions")

# Effect size (rank-biserial correlation)
def rank_biserial_correlation(x1, x2):
    """Calculate rank-biserial correlation"""
    # Combine and rank all data
    combined = np.concatenate([x1, x2])
    ranks = stats.rankdata(combined)
    
    # Get ranks for first group
    n1 = len(x1)
    n2 = len(x2)
    ranks1 = ranks[:n1]
    
    # Calculate U statistic
    U = np.sum(ranks1) - n1 * (n1 + 1) / 2
    
    # Calculate rank-biserial correlation
    r_rb = 1 - (2 * U) / (n1 * n2)
    return r_rb

rank_biserial = rank_biserial_correlation(setosa_sepal, versicolor_sepal)
print(f"\nEffect Size (Rank-biserial correlation): {abs(rank_biserial):.3f}")

if abs(rank_biserial) < 0.1:
    print("Effect size: Small")
elif abs(rank_biserial) < 0.3:
    print("Effect size: Medium")
elif abs(rank_biserial) < 0.5:
    print("Effect size: Large")
else:
    print("Effect size: Very large")

# Check when to use nonparametric test
print(f"\n=== When to Use Nonparametric Tests ===")

# Normality check
shapiro_setosa_stat, shapiro_setosa_p = stats.shapiro(setosa_sepal)
shapiro_versicolor_stat, shapiro_versicolor_p = stats.shapiro(versicolor_sepal)

print("Normality tests:")
print(f"Setosa - p = {shapiro_setosa_p:.4f}")
print(f"Versicolor - p = {shapiro_versicolor_p:.4f}")

if shapiro_setosa_p < 0.05 or shapiro_versicolor_p < 0.05:
    print("✓ Nonparametric test is appropriate (data not normal)")
else:
    print("✓ Both parametric and nonparametric tests are valid")

# Sample size consideration
total_n = len(setosa_sepal) + len(versicolor_sepal)
if total_n < 30:
    print("✓ Nonparametric test is recommended for small samples")
else:
    print("✓ Parametric test is generally robust for large samples")
```

### Wilcoxon Signed-Rank Test

The Wilcoxon signed-rank test is a nonparametric alternative to the paired t-test that tests whether the median of the differences is zero.

#### Mathematical Foundation

For paired differences $d_i = x_i - y_i$, the test statistic is:

```math
W = \sum_{i=1}^{n} \text{sgn}(d_i) R_i
```

Where:
- $\text{sgn}(d_i)$ is the sign of the difference
- $R_i$ is the rank of $|d_i|$ when all absolute differences are ranked

The test statistic follows approximately a normal distribution:
```math
Z = \frac{W}{\sqrt{\frac{n(n+1)(2n+1)}{6}}}
```

**Hypotheses:**
```math
H_0: \text{The median of differences is zero}
```

```math
H_1: \text{The median of differences is not zero}
```

#### Code Implementation

```python
# Nonparametric alternative to paired t-test
wilcox_paired_stat, wilcox_paired_p = stats.wilcoxon(after, before)

print("=== Wilcoxon Signed-Rank Test ===")
print("Hypothesis: H0: Median of differences = 0 vs. H1: Median of differences ≠ 0")
print("Test type: Two-tailed\n")

print("Data Summary:")
print(f"Sample size (pairs): {len(before)}")
print(f"Before treatment - median = {np.median(before):.2f}")
print(f"After treatment - median = {np.median(after):.2f}")

# Calculate differences
differences = after - before
print(f"Differences - median = {np.median(differences):.2f}")

print(f"\n=== Test Results ===")
print(f"W-statistic: {wilcox_paired_stat:.3f}")
print(f"P-value: {wilcox_paired_p:.4f}")

# Decision
alpha = 0.05
if wilcox_paired_p < alpha:
    print("Decision: Reject H0 (median difference is significant)")
else:
    print("Decision: Fail to reject H0 (insufficient evidence)")

# Compare with parametric paired t-test
print(f"\n=== Comparison with Parametric Test ===")
paired_t_test = stats.ttest_rel(after, before)
print(f"Paired t-test p-value: {paired_t_test.pvalue:.4f}")
print(f"Wilcoxon signed-rank p-value: {wilcox_paired_p:.4f}")

if abs(paired_t_test.pvalue - wilcox_paired_p) < 0.01:
    print("✓ Both tests give similar results")
else:
    print("⚠ Tests give different results - check assumptions")

# Effect size (rank-biserial correlation for paired data)
def rank_biserial_paired(differences):
    """Calculate rank-biserial correlation for paired data"""
    # Rank the absolute differences
    abs_diffs = np.abs(differences)
    ranks = stats.rankdata(abs_diffs)
    
    # Calculate signed ranks
    signed_ranks = np.where(differences > 0, ranks, -ranks)
    
    # Calculate rank-biserial correlation
    n = len(differences)
    r_rb = np.sum(signed_ranks) / (n * (n + 1) / 2)
    return r_rb

rank_biserial_paired_val = rank_biserial_paired(differences)
print(f"\nEffect Size (Rank-biserial correlation): {abs(rank_biserial_paired_val):.3f}")

if abs(rank_biserial_paired_val) < 0.1:
    print("Effect size: Small")
elif abs(rank_biserial_paired_val) < 0.3:
    print("Effect size: Medium")
elif abs(rank_biserial_paired_val) < 0.5:
    print("Effect size: Large")
else:
    print("Effect size: Very large")

# Check normality of differences
print(f"\n=== Assumption Checking ===")
shapiro_diff_stat, shapiro_diff_p = stats.shapiro(differences)
print("Shapiro-Wilk test for differences:")
print(f"W = {shapiro_diff_stat:.3f}, p = {shapiro_diff_p:.4f}")

if shapiro_diff_p < 0.05:
    print("✓ Nonparametric test is appropriate (differences not normal)")
else:
    print("✓ Both parametric and nonparametric tests are valid")

# Practical significance
median_diff = np.median(differences)
print(f"\nPractical Significance:")
print(f"Median treatment effect: {median_diff:.2f} units")

if abs(median_diff) > 10:
    print("✓ Large practical effect")
elif abs(median_diff) > 5:
    print("⚠ Moderate practical effect")
else:
    print("✗ Small practical effect")

# Number of positive vs negative differences
positive_diff = np.sum(differences > 0)
negative_diff = np.sum(differences < 0)
zero_diff = np.sum(differences == 0)

print(f"\nDifference Analysis:")
print(f"Positive differences: {positive_diff}")
print(f"Negative differences: {negative_diff}")
print(f"Zero differences: {zero_diff}")

if positive_diff > negative_diff:
    print("✓ More positive differences (treatment appears effective)")
elif negative_diff > positive_diff:
    print("✗ More negative differences (treatment appears harmful)")
else:
    print("⚠ Equal positive and negative differences")
```

### Kruskal-Wallis Test

The Kruskal-Wallis test is a nonparametric alternative to one-way ANOVA that tests whether multiple independent groups have the same distribution.

#### Mathematical Foundation

For $k$ groups with sample sizes $n_1, n_2, \ldots, n_k$, the test statistic is:

```math
H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)
```

Where:
- $N = n_1 + n_2 + \ldots + n_k$ is the total sample size
- $R_i$ is the sum of ranks for group $i$

The test statistic follows approximately a chi-square distribution with $k-1$ degrees of freedom.

**Hypotheses:**
```math
H_0: \text{All groups have the same distribution}
```

```math
H_1: \text{At least one group has a different distribution}
```

#### Code Implementation

```python
# Nonparametric alternative to one-way ANOVA
# Group by target (iris species)
virginica_sepal = data[data['target'] == 2]['sepal length (cm)']
groups = [setosa_sepal, versicolor_sepal, virginica_sepal]
group_names = ['Setosa', 'Versicolor', 'Virginica']

kruskal_stat, kruskal_p = stats.kruskal(*groups)

print("=== Kruskal-Wallis Test ===")
print("Hypothesis: H0: All groups have same distribution vs. H1: At least one group differs")
print("Test type: Two-tailed\n")

# Data summary by group
print("Data Summary by Species:")
for i, (name, group_data) in enumerate(zip(group_names, groups)):
    print(f"{name}: n = {len(group_data)}, median = {np.median(group_data):.2f}, mean = {group_data.mean():.2f}")

print(f"\n=== Test Results ===")
print(f"H-statistic: {kruskal_stat:.3f}")
print(f"Degrees of freedom: {len(groups) - 1}")
print(f"P-value: {kruskal_p:.4f}")

# Decision
alpha = 0.05
if kruskal_p < alpha:
    print("Decision: Reject H0 (at least one group differs)")
else:
    print("Decision: Fail to reject H0 (insufficient evidence)")

# Effect size (epsilon-squared)
def epsilon_squared(h_stat, n, k):
    """Calculate epsilon-squared effect size for Kruskal-Wallis test"""
    return (h_stat - k + 1) / (n - k)

n_total = sum(len(g) for g in groups)
k_groups = len(groups)
epsilon_sq = epsilon_squared(kruskal_stat, n_total, k_groups)
print(f"\nEffect Size (Epsilon-squared): {epsilon_sq:.3f}")

if epsilon_sq < 0.06:
    print("Effect size: Small")
elif epsilon_sq < 0.14:
    print("Effect size: Medium")
else:
    print("Effect size: Large")

# Compare with parametric ANOVA
print(f"\n=== Comparison with Parametric ANOVA ===")
from scipy.stats import f_oneway
f_stat, f_p = f_oneway(*groups)
print(f"F-statistic: {f_stat:.3f}")
print(f"ANOVA p-value: {f_p:.4f}")
print(f"Kruskal-Wallis p-value: {kruskal_p:.4f}")

if abs(f_p - kruskal_p) < 0.01:
    print("✓ Both tests give similar results")
else:
    print("⚠ Tests give different results - check assumptions")

# Post-hoc analysis (if significant)
if kruskal_p < alpha:
    print(f"\n=== Post-hoc Analysis ===")
    print("Since the overall test is significant, we need post-hoc tests")
    print("to determine which specific groups differ.")
    
    # Pairwise Mann-Whitney U tests with Bonferroni correction
    from itertools import combinations
    n_comparisons = len(list(combinations(range(len(groups)), 2)))
    alpha_corrected = alpha / n_comparisons
    
    print(f"Pairwise Mann-Whitney U tests with Bonferroni correction (α = {alpha_corrected:.4f}):")
    for i, j in combinations(range(len(groups)), 2):
        u_stat, u_p = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
        significant = "***" if u_p < alpha_corrected else ""
        print(f"{group_names[i]} vs {group_names[j]}: U = {u_stat:.3f}, p = {u_p:.4f} {significant}")

# Check assumptions
print(f"\n=== Assumption Checking ===")

# Normality check for each group
print("Normality tests by group:")
for name, group_data in zip(group_names, groups):
    shapiro_stat, shapiro_p = stats.shapiro(group_data)
    print(f"{name} - W = {shapiro_stat:.3f}, p = {shapiro_p:.4f}")

# Homogeneity of variance
print(f"\nHomogeneity of variance test:")
levene_stat, levene_p = stats.levene(*groups)
print(f"Levene's test - p = {levene_p:.4f}")

if levene_p < 0.05:
    print("✓ Nonparametric test is appropriate (variances not equal)")
else:
    print("✓ Both parametric and nonparametric tests are valid")

# Sample size consideration
total_n = sum(len(g) for g in groups)
if total_n < 30:
    print("✓ Nonparametric test is recommended for small samples")
else:
    print("✓ Parametric test is generally robust for large samples")

# Practical significance
print(f"\nPractical Significance:")
overall_median = np.median(np.concatenate(groups))
print(f"Overall median sepal length: {overall_median:.2f}")

for name, group_data in zip(group_names, groups):
    group_median = np.median(group_data)
    print(f"{name} median: {group_median:.2f} (difference from overall: {group_median - overall_median:.2f})")
```

## Multiple Comparison Tests

### One-Way ANOVA

```python
# Test if mean sepal length differs by iris species
from scipy.stats import f_oneway
f_stat, f_p = f_oneway(setosa_sepal, versicolor_sepal, virginica_sepal)

print("One-Way ANOVA Results:")
print(f"F-statistic: {f_stat:.3f}")
print(f"p-value: {f_p:.6f}")

# Post-hoc tests using Tukey's HSD
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(np.concatenate([setosa_sepal, versicolor_sepal, virginica_sepal]),
                          np.repeat(['Setosa', 'Versicolor', 'Virginica'], 
                                   [len(setosa_sepal), len(versicolor_sepal), len(virginica_sepal)]))
print("\nTukey's HSD Post-hoc Test:")
print(tukey)
```

### Chi-Square Test

```python
# Test independence between sepal length category and species
data['sepal_category'] = pd.cut(data['sepal length (cm)'], 
                                bins=[0, 5.5, 6.5, 10], 
                                labels=['Short', 'Medium', 'Long'])
contingency_table = pd.crosstab(data['sepal_category'], data['target'])
chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)

print("Chi-Square Test Results:")
print(f"Chi-square statistic: {chi2_stat:.3f}")
print(f"p-value: {chi2_p:.6f}")
print(f"Degrees of freedom: {dof}")
print("\nContingency Table:")
print(contingency_table)
```

## Effect Size

### Cohen's d

```python
# Calculate Cohen's d for t-test
def cohens_d_two_sample(x1, x2):
    """Calculate Cohen's d for two independent samples"""
    pooled_std = np.sqrt(((len(x1) - 1) * x1.var() + (len(x2) - 1) * x2.var()) / (len(x1) + len(x2) - 2))
    return (x1.mean() - x2.mean()) / pooled_std

effect_size = cohens_d_two_sample(setosa_sepal, versicolor_sepal)
print(f"Cohen's d: {effect_size:.3f}")

# Interpret effect size
print("Effect size interpretation:")
print("d = 0.2: Small effect")
print("d = 0.5: Medium effect")
print("d = 0.8: Large effect")
```

### Eta-squared

```python
# Calculate eta-squared for ANOVA
def eta_squared(ss_between, ss_total):
    """Calculate eta-squared effect size for ANOVA"""
    return ss_between / ss_total

# For demonstration, we'll calculate it manually
# In practice, you'd get these from the ANOVA results
print("Eta-squared calculation:")
print("η² = SS_between / SS_total")
print("This measures the proportion of variance explained by the factor")
```

## Power Analysis

```python
# Power analysis for t-test
from statsmodels.stats.power import TTestPower

# Power analysis for t-test
power_analysis = TTestPower()
sample_size = power_analysis.solve_power(effect_size=0.5, alpha=0.05, power=0.8)
print(f"Sample size needed per group: {sample_size:.0f}")

# Power for different effect sizes
effect_sizes = [0.2, 0.5, 0.8]
sample_sizes = [20, 50, 100]

print("\nPower Analysis for Different Scenarios:")
print("Effect Size | Sample Size | Power")
print("-----------|-------------|-------")
for d in effect_sizes:
    for n in sample_sizes:
        power = power_analysis.power(effect_size=d, nobs=n, alpha=0.05)
        print(f"{d:.1f}        | {n}          | {power:.3f}")
```

## Multiple Testing Correction

### Bonferroni Correction

```python
# Multiple p-values
p_values = [0.01, 0.03, 0.05, 0.08, 0.12]

# Bonferroni correction
from statsmodels.stats.multitest import multipletests
alpha = 0.05
rejected, bonferroni_corrected, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')

print(f"Original p-values: {p_values}")
print(f"Bonferroni corrected: {bonferroni_corrected}")
print(f"Significant at α = 0.05: {rejected}")
```

### False Discovery Rate

```python
# FDR correction
rejected_fdr, fdr_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

print(f"FDR corrected: {fdr_corrected}")
print(f"Significant at α = 0.05: {rejected_fdr}")
```

## Practical Examples

### Example 1: Drug Efficacy Study

```python
# Simulate drug efficacy data
np.random.seed(123)
placebo = np.random.normal(50, 10, 30)
treatment = np.random.normal(55, 10, 30)

# Test if treatment is effective
t_stat, p_value = stats.ttest_ind(treatment, placebo, alternative='greater')

print("Drug Efficacy Study Results:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

# Calculate effect size
effect_size = cohens_d_two_sample(treatment, placebo)
print(f"Effect size (Cohen's d): {effect_size:.3f}")
```

### Example 2: Customer Satisfaction Survey

```python
# Simulate satisfaction scores
np.random.seed(123)
store_a = np.random.normal(7.5, 1.2, 50)
store_b = np.random.normal(7.8, 1.1, 50)

# Test if satisfaction differs
t_stat, p_value = stats.ttest_ind(store_a, store_b)

print("Customer Satisfaction Survey Results:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

# Check normality
shapiro_a_stat, shapiro_a_p = stats.shapiro(store_a)
shapiro_b_stat, shapiro_b_p = stats.shapiro(store_b)
print(f"\nNormality tests:")
print(f"Store A - W = {shapiro_a_stat:.3f}, p = {shapiro_a_p:.4f}")
print(f"Store B - W = {shapiro_b_stat:.3f}, p = {shapiro_b_p:.4f}")
```

### Example 3: A/B Testing

```python
# Simulate A/B test data
np.random.seed(123)
version_a_conversions = np.random.binomial(1, 0.05, 1000)
version_b_conversions = np.random.binomial(1, 0.06, 1000)

# Test conversion rates
contingency_table = np.array([[np.sum(version_a_conversions), len(version_a_conversions) - np.sum(version_a_conversions)],
                              [np.sum(version_b_conversions), len(version_b_conversions) - np.sum(version_b_conversions)]])
chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)

print("A/B Testing Results:")
print(f"Chi-square statistic: {chi2_stat:.3f}")
print(f"p-value: {chi2_p:.4f}")
print(f"Degrees of freedom: {dof}")
```

## Best Practices

### Assumption Checking

```python
# Function to check t-test assumptions
def check_t_test_assumptions(x, y=None):
    print("Normality test (Shapiro-Wilk):")
    if y is None:
        shapiro_stat, shapiro_p = stats.shapiro(x)
        print(f"W = {shapiro_stat:.3f}, p = {shapiro_p:.4f}")
    else:
        shapiro_x_stat, shapiro_x_p = stats.shapiro(x)
        shapiro_y_stat, shapiro_y_p = stats.shapiro(y)
        print(f"Group 1 - W = {shapiro_x_stat:.3f}, p = {shapiro_x_p:.4f}")
        print(f"Group 2 - W = {shapiro_y_stat:.3f}, p = {shapiro_y_p:.4f}")
    
    if y is not None:
        variance_ratio = x.var() / y.var()
        print(f"\nVariance ratio: {variance_ratio:.3f}")
        print("If ratio > 2, consider Welch's t-test")

# Apply to our data
check_t_test_assumptions(setosa_sepal, versicolor_sepal)
```

### Reporting Results

```python
# Function to format test results
def format_test_results(test_statistic, p_value, test_name):
    print(test_name)
    print(f"Test statistic: {test_statistic:.3f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.001:
        print("Significance: p < 0.001")
    elif p_value < 0.01:
        print("Significance: p < 0.01")
    elif p_value < 0.05:
        print("Significance: p < 0.05")
    else:
        print("Significance: p >= 0.05")

# Apply to t-test
t_stat, t_p = stats.ttest_ind(setosa_sepal, versicolor_sepal)
format_test_results(t_stat, t_p, "Independent t-test")
```

## Common Mistakes to Avoid

```python
# 1. Multiple testing without correction
print("Mistake: Running many tests without correction")
print("Solution: Use Bonferroni, FDR, or other corrections\n")

# 2. Ignoring effect size
print("Mistake: Only reporting p-values")
print("Solution: Always report effect sizes\n")

# 3. Data dredging
print("Mistake: Testing many hypotheses without pre-specification")
print("Solution: Pre-specify hypotheses and analysis plan\n")

# 4. Ignoring assumptions
print("Mistake: Not checking test assumptions")
print("Solution: Always verify assumptions before testing")
```

## Exercises

### Exercise 1: Basic Hypothesis Testing
Test if the mean MPG in the mtcars dataset is different from 20 using a t-test.

### Exercise 2: Two-Sample Comparison
Compare the MPG between automatic and manual transmissions using both parametric and nonparametric tests.

### Exercise 3: Multiple Testing
Perform multiple t-tests and apply correction methods to control for multiple comparisons.

### Exercise 4: Power Analysis
Calculate the required sample size for detecting a medium effect size with 80% power.

### Exercise 5: Real-world Application
Design and conduct a hypothesis test for a real-world scenario of your choice.

## Next Steps

In the next chapter, we'll learn about confidence intervals and their relationship with hypothesis testing.

---

**Key Takeaways:**
- Always state null and alternative hypotheses clearly
- Choose appropriate tests based on data type and assumptions
- Report both p-values and effect sizes
- Check assumptions before conducting tests
- Use correction methods for multiple testing
- Consider power analysis for study design
- Interpret results in context of practical significance
- Avoid common pitfalls like data dredging 