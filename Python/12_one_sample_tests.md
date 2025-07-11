# One-Sample Tests

## Overview

One-sample tests are fundamental statistical procedures used to determine whether a sample mean differs significantly from a hypothesized population mean. These tests form the cornerstone of statistical inference and are widely used across research disciplines including psychology, medicine, education, and business analytics.

### When to Use One-Sample Tests

One-sample tests are appropriate when you want to:
- Compare a sample mean to a known or hypothesized population value
- Test whether a treatment has a significant effect compared to a baseline
- Validate whether a process meets quality control standards
- Assess whether educational interventions improve performance above a threshold

### Key Concepts

**Null Hypothesis (H₀):** The population mean equals the hypothesized value
**Alternative Hypothesis (H₁):** The population mean differs from the hypothesized value

The test determines whether observed differences are statistically significant or due to random sampling variation.

### Mathematical Foundation

The one-sample t-test is based on the t-distribution, which accounts for the uncertainty in estimating the population standard deviation from sample data.

For a sample of size $n$ with mean $\bar{x}$ and standard deviation $s$, the t-statistic is:

```math
t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}
```

where $\mu_0$ is the hypothesized population mean.

The degrees of freedom are $df = n - 1$, and the p-value is calculated from the t-distribution with these degrees of freedom.

## One-Sample t-Test

The one-sample t-test is the most commonly used parametric test for comparing a sample mean to a hypothesized population mean. It's robust and works well even with moderate violations of normality assumptions, especially for larger sample sizes.

### Mathematical Foundation

The t-test is based on the t-distribution, which was developed by William Gosset (publishing under the pseudonym "Student") in 1908. The t-distribution accounts for the uncertainty in estimating the population standard deviation from sample data.

**Test Statistic:**
```math
t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}
```

**Degrees of Freedom:**
```math
df = n - 1
```

**Confidence Interval:**
```math
CI = \bar{x} \pm t_{\alpha/2, df} \cdot \frac{s}{\sqrt{n}}
```

where:
- $\bar{x}$ = sample mean
- $\mu_0$ = hypothesized population mean
- $s$ = sample standard deviation
- $n$ = sample size
- $t_{\alpha/2, df}$ = critical t-value for confidence level $\alpha$

### Basic One-Sample t-Test

```python
# Load sample data
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris
from statsmodels.stats.weightstats import DescrStatsW

# Load iris dataset as example
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Test if the mean sepal length is significantly different from 5.5
sepal_length = data['sepal length (cm)']
t_statistic, p_value = stats.ttest_1samp(sepal_length, 5.5)

# Calculate confidence interval
desc_stats = DescrStatsW(sepal_length)
confidence_interval = desc_stats.tconfint_mean(alpha=0.05)
sample_mean = sepal_length.mean()
hypothesized_mean = 5.5

print("Test Results:")
print(f"Sample mean: {sample_mean:.3f}")
print(f"Hypothesized mean: {hypothesized_mean}")
print(f"t-statistic: {t_statistic:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"95% Confidence Interval: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")

# Manual calculation for understanding
n = len(sepal_length)
sample_sd = sepal_length.std(ddof=1)
standard_error = sample_sd / np.sqrt(n)
manual_t = (sample_mean - hypothesized_mean) / standard_error

print(f"\nManual Calculation Verification:")
print(f"Sample SD: {sample_sd:.3f}")
print(f"Standard Error: {standard_error:.3f}")
print(f"Manual t-statistic: {manual_t:.3f}")
print(f"Degrees of freedom: {n - 1}")
```

### One-Sample t-Test with Different Alternatives

The choice of alternative hypothesis depends on your research question and prior knowledge about the direction of the effect.

**Types of Alternative Hypotheses:**

1. **Two-tailed (two.sided):** Tests for any difference from the hypothesized value
   - H₀: $\mu = \mu_0$
   - H₁: $\mu \neq \mu_0$

2. **One-tailed greater:** Tests if the population mean is greater than hypothesized value
   - H₀: $\mu = \mu_0$
   - H₁: $\mu > \mu_0$

3. **One-tailed less:** Tests if the population mean is less than hypothesized value
   - H₀: $\mu = \mu_0$
   - H₁: $\mu < \mu_0$

**Mathematical Differences:**
- Two-tailed p-value: $P(|T| > |t_{obs}|)$
- One-tailed greater: $P(T > t_{obs})$
- One-tailed less: $P(T < t_{obs})$

```python
# Two-tailed test (default)
two_tailed_stat, two_tailed_p = stats.ttest_1samp(sepal_length, 5.5)
print("Two-tailed test:")
print(f"t-statistic: {two_tailed_stat:.3f}, p-value: {two_tailed_p:.4f}")

# One-tailed test (greater than)
greater_stat, greater_p = stats.ttest_1samp(sepal_length, 5.5, alternative='greater')
print("\nOne-tailed test (greater than):")
print(f"t-statistic: {greater_stat:.3f}, p-value: {greater_p:.4f}")

# One-tailed test (less than)
less_stat, less_p = stats.ttest_1samp(sepal_length, 5.5, alternative='less')
print("\nOne-tailed test (less than):")
print(f"t-statistic: {less_stat:.3f}, p-value: {less_p:.4f}")

# Compare p-values
print(f"\nP-values comparison:")
print(f"Two-tailed: {two_tailed_p:.4f}")
print(f"Greater than: {greater_p:.4f}")
print(f"Less than: {less_p:.4f}")

# Relationship between p-values
print(f"\nP-value relationships:")
print(f"Two-tailed p-value ≈ 2 × min(one-tailed p-values)")
print(f"Greater p-value + Less p-value = 1")

# Critical values comparison
alpha = 0.05
df = len(sepal_length) - 1
critical_two_tailed = stats.t.ppf(1 - alpha/2, df)
critical_one_tailed = stats.t.ppf(1 - alpha, df)

print(f"\nCritical values (α = 0.05):")
print(f"Two-tailed critical t: {critical_two_tailed:.3f}")
print(f"One-tailed critical t: {critical_one_tailed:.3f}")
```

### Effect Size for One-Sample t-Test

Effect size measures the magnitude of the difference between the sample mean and hypothesized population mean, independent of sample size. This is crucial for understanding practical significance beyond statistical significance.

**Mathematical Foundation:**

**Cohen's d (Standardized Mean Difference):**
```math
d = \frac{\bar{x} - \mu_0}{s}
```

**Hedges' g (Unbiased Estimator):**
```math
g = d \cdot \left(1 - \frac{3}{4(n-1) - 1}\right)
```

**Confidence Interval for Effect Size:**
```math
CI_d = d \pm t_{\alpha/2, df} \cdot SE_d
```

where $SE_d = \sqrt{\frac{1}{n} + \frac{d^2}{2n}}$

**Interpretation Guidelines:**
- Small effect: $|d| < 0.2$
- Medium effect: $0.2 \leq |d| < 0.5$
- Large effect: $0.5 \leq |d| < 0.8$
- Very large effect: $|d| \geq 0.8$

```python
# Calculate Cohen's d effect size
def calculate_cohens_d(sample_data, hypothesized_mean):
    sample_mean = sample_data.mean()
    sample_sd = sample_data.std(ddof=1)
    n = len(sample_data)
    
    # Cohen's d
    cohens_d = (sample_mean - hypothesized_mean) / sample_sd
    
    # Hedges' g (unbiased estimator)
    hedges_g = cohens_d * (1 - 3 / (4 * (n - 1) - 1))
    
    # Standard error of effect size
    se_d = np.sqrt(1/n + cohens_d**2/(2*n))
    
    # Confidence interval for effect size
    df = n - 1
    t_critical = stats.t.ppf(0.975, df)
    ci_lower = cohens_d - t_critical * se_d
    ci_upper = cohens_d + t_critical * se_d
    
    return {
        'cohens_d': cohens_d,
        'hedges_g': hedges_g,
        'se_d': se_d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'sample_mean': sample_mean,
        'sample_sd': sample_sd,
        'n': n
    }

# Apply to sepal length data
sepal_effect_size = calculate_cohens_d(sepal_length, 5.5)

print("Effect Size Analysis:")
print(f"Cohen's d: {sepal_effect_size['cohens_d']:.3f}")
print(f"Hedges' g: {sepal_effect_size['hedges_g']:.3f}")
print(f"Standard Error of d: {sepal_effect_size['se_d']:.3f}")
print(f"95% CI for d: [{sepal_effect_size['ci_lower']:.3f}, {sepal_effect_size['ci_upper']:.3f}]")

# Interpret effect size
def interpret_effect_size(d):
    if abs(d) < 0.2:
        return "Small effect"
    elif abs(d) < 0.5:
        return "Medium effect"
    elif abs(d) < 0.8:
        return "Large effect"
    else:
        return "Very large effect"

print(f"Effect size interpretation: {interpret_effect_size(sepal_effect_size['cohens_d'])}")

# Power analysis based on effect size
from statsmodels.stats.power import TTestPower
power_analysis = TTestPower()
power = power_analysis.power(effect_size=sepal_effect_size['cohens_d'], 
                           nobs=sepal_effect_size['n'], alpha=0.05)
print(f"Power for current effect size: {power:.3f}")
```

## One-Sample z-Test

The z-test is used when the population standard deviation is known, which is rare in practice but theoretically important. It's more powerful than the t-test when the population standard deviation is known with certainty.

### Mathematical Foundation

**Test Statistic:**
```math
z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}
```

**Confidence Interval:**
```math
CI = \bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
```

**P-value Calculations:**
- Two-tailed: $P(|Z| > |z_{obs}|) = 2(1 - \Phi(|z_{obs}|))$
- One-tailed greater: $P(Z > z_{obs}) = 1 - \Phi(z_{obs})$
- One-tailed less: $P(Z < z_{obs}) = \Phi(z_{obs})$

where $\Phi(z)$ is the standard normal cumulative distribution function.

**When to Use z-test vs t-test:**
- Use z-test when population standard deviation ($\sigma$) is known
- Use t-test when population standard deviation is unknown (estimated from sample)
- For large samples ($n > 30$), t-test approximates z-test due to Central Limit Theorem

### Basic One-Sample z-Test

```python
# Function to perform one-sample z-test
def one_sample_z_test(sample_data, hypothesized_mean, population_sd, alpha=0.05):
    sample_mean = sample_data.mean()
    n = len(sample_data)
    
    # Calculate z-statistic
    z_statistic = (sample_mean - hypothesized_mean) / (population_sd / np.sqrt(n))
    
    # Calculate p-value
    p_value_two_tailed = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    p_value_greater = 1 - stats.norm.cdf(z_statistic)
    p_value_less = stats.norm.cdf(z_statistic)
    
    # Calculate confidence interval
    margin_of_error = stats.norm.ppf(1 - alpha/2) * (population_sd / np.sqrt(n))
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    # Calculate effect size (Cohen's d using population SD)
    cohens_d = (sample_mean - hypothesized_mean) / population_sd
    
    return {
        'z_statistic': z_statistic,
        'p_value_two_tailed': p_value_two_tailed,
        'p_value_greater': p_value_greater,
        'p_value_less': p_value_less,
        'sample_mean': sample_mean,
        'hypothesized_mean': hypothesized_mean,
        'confidence_interval': [ci_lower, ci_upper],
        'margin_of_error': margin_of_error,
        'cohens_d': cohens_d,
        'n': n
    }

# Example: Test if sepal length mean is different from 5.5 (assuming known population SD = 0.5)
sepal_z_test = one_sample_z_test(sepal_length, 5.5, 0.5)

print("One-Sample Z-Test Results:")
print(f"Sample mean: {sepal_z_test['sample_mean']:.3f}")
print(f"Hypothesized mean: {sepal_z_test['hypothesized_mean']}")
print(f"Population SD: 0.5")
print(f"z-statistic: {sepal_z_test['z_statistic']:.3f}")
print(f"Two-tailed p-value: {sepal_z_test['p_value_two_tailed']:.4f}")
print(f"95% Confidence Interval: [{sepal_z_test['confidence_interval'][0]:.3f}, {sepal_z_test['confidence_interval'][1]:.3f}]")
print(f"Effect size (Cohen's d): {sepal_z_test['cohens_d']:.3f}")

# Compare with t-test results
sepal_t_stat, sepal_t_p = stats.ttest_1samp(sepal_length, 5.5)
print(f"\nComparison with t-test:")
print(f"t-statistic: {sepal_t_stat:.3f}")
print(f"t-test p-value: {sepal_t_p:.4f}")
print(f"z-test p-value: {sepal_z_test['p_value_two_tailed']:.4f}")
```

## Nonparametric One-Sample Tests

Nonparametric tests make fewer assumptions about the underlying population distribution and are robust to violations of normality. They're particularly useful for small samples or when data is skewed or contains outliers.

### Wilcoxon Signed-Rank Test

The Wilcoxon signed-rank test is the most commonly used nonparametric alternative to the one-sample t-test. It tests whether the median differs from a hypothesized value by ranking the absolute differences.

**Mathematical Foundation:**

1. Calculate differences: $d_i = x_i - \mu_0$
2. Rank absolute differences: $R_i = rank(|d_i|)$
3. Assign signs: $s_i = sign(d_i)$
4. Calculate test statistic: $W = \sum_{i=1}^{n} s_i \cdot R_i$

**For large samples ($n > 20$), W is approximately normal:**
```math
Z = \frac{W}{\sqrt{\frac{n(n+1)(2n+1)}{6}}}
```

**Effect Size (r):**
```math
r = \frac{Z}{\sqrt{n}}
```

```python
# Wilcoxon signed-rank test (nonparametric alternative to t-test)
wilcox_stat, wilcox_p = stats.wilcoxon(sepal_length - 5.5)
print(f"Wilcoxon signed-rank test:")
print(f"W-statistic: {wilcox_stat:.3f}")
print(f"p-value: {wilcox_p:.4f}")

# Manual calculation for understanding
differences = sepal_length - 5.5
abs_differences = np.abs(differences)
ranks = stats.rankdata(abs_differences)
signed_ranks = np.sign(differences) * ranks
W_statistic = np.sum(signed_ranks)

print(f"\nManual Wilcoxon Calculation:")
print(f"W statistic: {W_statistic:.3f}")
print(f"SciPy W statistic: {wilcox_stat:.3f}")

# Compare with t-test results
print(f"\nComparison of t-test and Wilcoxon test:")
print(f"t-test p-value: {sepal_t_p:.4f}")
print(f"Wilcoxon p-value: {wilcox_p:.4f}")

# Effect size for Wilcoxon test
wilcox_effect_size = abs(stats.norm.ppf(wilcox_p / 2)) / np.sqrt(len(sepal_length))
print(f"Wilcoxon effect size (r): {wilcox_effect_size:.3f}")

# Interpret Wilcoxon effect size
def interpret_wilcox_effect(r):
    if abs(r) < 0.1:
        return "Small effect"
    elif abs(r) < 0.3:
        return "Medium effect"
    elif abs(r) < 0.5:
        return "Large effect"
    else:
        return "Very large effect"

print(f"Wilcoxon effect interpretation: {interpret_wilcox_effect(wilcox_effect_size)}")
```

### Sign Test

The sign test is the simplest nonparametric test that only considers whether observations are above or below the hypothesized median. It's less powerful than the Wilcoxon test but makes the fewest assumptions.

**Mathematical Foundation:**

The sign test is based on the binomial distribution with $p = 0.5$ under the null hypothesis.

**Test Statistic:**
- $S^+$ = number of positive differences
- $S^-$ = number of negative differences
- $n = S^+ + S^-$ (excluding ties)

**Under H₀:** $S^+ \sim Binomial(n, 0.5)$

**P-value Calculations:**
- Two-tailed: $P = 2 \cdot P(S^+ \leq \min(S^+, S^-))$
- One-tailed greater: $P = P(S^+ \geq S^+_{obs})$
- One-tailed less: $P = P(S^+ \leq S^+_{obs})$

**For large samples ($n > 20$), normal approximation:**
```math
Z = \frac{S^+ - n/2}{\sqrt{n/4}}
```

```python
# Function to perform sign test
def sign_test(sample_data, hypothesized_median):
    differences = sample_data - hypothesized_median
    positive_signs = np.sum(differences > 0)
    negative_signs = np.sum(differences < 0)
    n = positive_signs + negative_signs
    
    # Binomial test
    p_value_two_tailed = 2 * stats.binom.cdf(min(positive_signs, negative_signs), n, 0.5)
    p_value_greater = 1 - stats.binom.cdf(positive_signs - 1, n, 0.5)
    p_value_less = stats.binom.cdf(positive_signs, n, 0.5)
    
    # Normal approximation for large samples
    if n > 20:
        z_statistic = (positive_signs - n/2) / np.sqrt(n/4)
        p_value_normal = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    else:
        z_statistic = np.nan
        p_value_normal = np.nan
    
    return {
        'positive_signs': positive_signs,
        'negative_signs': negative_signs,
        'n': n,
        'p_value_two_tailed': p_value_two_tailed,
        'p_value_greater': p_value_greater,
        'p_value_less': p_value_less,
        'z_statistic': z_statistic,
        'p_value_normal': p_value_normal
    }

# Apply sign test to sepal length data
sepal_sign_test = sign_test(sepal_length, 5.5)

print("Sign Test Results:")
print(f"Positive signs: {sepal_sign_test['positive_signs']}")
print(f"Negative signs: {sepal_sign_test['negative_signs']}")
print(f"Total observations (excluding ties): {sepal_sign_test['n']}")
print(f"Two-tailed p-value (exact): {sepal_sign_test['p_value_two_tailed']:.4f}")

if not np.isnan(sepal_sign_test['z_statistic']):
    print(f"z-statistic (normal approx): {sepal_sign_test['z_statistic']:.3f}")
    print(f"Two-tailed p-value (normal): {sepal_sign_test['p_value_normal']:.4f}")

# Compare with other tests
print(f"\nComparison with other tests:")
print(f"t-test p-value: {sepal_t_p:.4f}")
print(f"Wilcoxon p-value: {wilcox_p:.4f}")
print(f"Sign test p-value: {sepal_sign_test['p_value_two_tailed']:.4f}")
```

## Power Analysis

Power analysis helps determine the probability of detecting a true effect and the sample size needed for adequate statistical power. This is crucial for study design and interpreting results.

### Mathematical Foundation

**Power Definition:**
```math
Power = P(\text{Reject } H_0 | H_1 \text{ is true}) = 1 - \beta
```

**For one-sample t-test:**
```math
Power = P\left(|T| > t_{\alpha/2, df} \mid \mu = \mu_0 + \delta\right)
```

where $\delta$ is the true effect size and $t_{\alpha/2, df}$ is the critical t-value.

**Effect Size and Power Relationship:**
- Larger effect sizes require smaller sample sizes for the same power
- Smaller effect sizes require larger sample sizes for adequate power
- Power increases with sample size for fixed effect sizes

### Power Analysis for One-Sample t-Test

```python
# Comprehensive power analysis function
def power_analysis(sample_size, effect_size, alpha=0.05):
    # Calculate power for current sample size
    power_result = TTestPower()
    power = power_result.power(effect_size=effect_size, nobs=sample_size, alpha=alpha)
    
    # Calculate required sample size for 80% power
    sample_size_80 = power_result.solve_power(effect_size=effect_size, alpha=alpha, power=0.8)
    
    # Calculate required sample size for 90% power
    sample_size_90 = power_result.solve_power(effect_size=effect_size, alpha=alpha, power=0.9)
    
    # Calculate power for different effect sizes
    small_effect_power = power_result.power(effect_size=0.2, nobs=sample_size, alpha=alpha)
    medium_effect_power = power_result.power(effect_size=0.5, nobs=sample_size, alpha=alpha)
    large_effect_power = power_result.power(effect_size=0.8, nobs=sample_size, alpha=alpha)
    
    return {
        'power': power,
        'required_n_80': sample_size_80,
        'required_n_90': sample_size_90,
        'effect_size': effect_size,
        'alpha': alpha,
        'small_effect_power': small_effect_power,
        'medium_effect_power': medium_effect_power,
        'large_effect_power': large_effect_power
    }

# Apply to sepal length data
sepal_power = power_analysis(len(sepal_length), sepal_effect_size['cohens_d'])

print("Power Analysis Results:")
print(f"Current effect size: {sepal_power['effect_size']:.3f}")
print(f"Current power: {sepal_power['power']:.3f}")
print(f"Required sample size for 80% power: {int(np.ceil(sepal_power['required_n_80']))}")
print(f"Required sample size for 90% power: {int(np.ceil(sepal_power['required_n_90']))}")

print(f"\nPower for different effect sizes:")
print(f"Small effect (d = 0.2): {sepal_power['small_effect_power']:.3f}")
print(f"Medium effect (d = 0.5): {sepal_power['medium_effect_power']:.3f}")
print(f"Large effect (d = 0.8): {sepal_power['large_effect_power']:.3f}")

# Power curve analysis
effect_sizes = np.arange(0.1, 1.1, 0.1)
power_curve = [power_result.power(effect_size=d, nobs=len(sepal_length), alpha=0.05) 
               for d in effect_sizes]

print(f"\nPower curve (effect size vs power):")
for i, d in enumerate(effect_sizes):
    print(f"d = {d:.1f}: Power = {power_curve[i]:.3f}")
```

## Assumption Checking

Proper assumption checking is crucial for valid statistical inference. Violations of assumptions can lead to incorrect conclusions and inflated Type I error rates.

### Key Assumptions for One-Sample Tests

1. **Independence:** Observations are independent of each other
2. **Normality:** Data follows a normal distribution (for parametric tests)
3. **Random Sampling:** Data represents a random sample from the population
4. **No Outliers:** Extreme values don't unduly influence results

### Normality Test

The normality assumption is critical for parametric tests like the t-test. Several methods can assess normality:

**Mathematical Tests:**
- **Shapiro-Wilk:** Most powerful test for normality
- **Anderson-Darling:** Good for detecting departures from normality
- **Kolmogorov-Smirnov:** Tests against a specified distribution

**Graphical Methods:**
- **Q-Q plots:** Compare sample quantiles to theoretical normal quantiles
- **Histograms:** Visual assessment of distribution shape
- **Box plots:** Detect skewness and outliers

```python
# Comprehensive normality checking function
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, anderson, kstest
from scipy.stats import skew, kurtosis

def check_normality(data):
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = shapiro(data)
    
    # Anderson-Darling test
    anderson_result = anderson(data)
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
    
    # Descriptive statistics for normality
    data_skewness = skew(data)
    data_kurtosis = kurtosis(data)
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[0])
    axes[0].set_title(f"Q-Q Plot for Normality Check\nShapiro-Wilk p = {shapiro_p:.4f}")
    
    # Histogram with normal curve
    axes[1].hist(data, bins=15, density=True, alpha=0.7, color='steelblue')
    x = np.linspace(data.min(), data.max(), 100)
    axes[1].plot(x, stats.norm.pdf(x, data.mean(), data.std()), 'r-', linewidth=2)
    axes[1].set_title(f"Histogram with Normal Curve\nSkewness = {data_skewness:.3f}, Kurtosis = {data_kurtosis:.3f}")
    
    # Box plot
    axes[2].boxplot(data, patch_artist=True, boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[2].set_title("Box Plot for Outlier Detection")
    
    plt.tight_layout()
    plt.show()
    
    print("=== NORMALITY ASSESSMENT ===")
    print(f"Sample size: {len(data)}")
    print(f"Mean: {data.mean():.3f}")
    print(f"SD: {data.std():.3f}")
    print(f"Skewness: {data_skewness:.3f}")
    print(f"Kurtosis: {data_kurtosis:.3f}\n")
    
    print("Normality Tests:")
    print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
    print(f"Anderson-Darling statistic: {anderson_result.statistic:.4f}")
    print(f"Kolmogorov-Smirnov p-value: {ks_p:.4f}\n")
    
    # Interpretation
    print("Interpretation:")
    if shapiro_p >= 0.05:
        print("✓ Data appears to be normally distributed")
    else:
        print("✗ Data is not normally distributed")
    
    if abs(data_skewness) < 1:
        print("✓ Skewness is acceptable")
    else:
        print("✗ Data is significantly skewed")
    
    if abs(data_kurtosis - 3) < 2:
        print("✓ Kurtosis is acceptable")
    else:
        print("✗ Data has unusual kurtosis")
    
    return {
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'anderson_stat': anderson_result.statistic,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'skewness': data_skewness,
        'kurtosis': data_kurtosis
    }

# Check normality of sepal length data
sepal_normality = check_normality(sepal_length)
```

### Outlier Detection

Outliers can significantly influence test results and should be identified and handled appropriately. Different methods have different sensitivities to outliers.

**Mathematical Methods:**

1. **IQR Method (Tukey's Fences):**
```math
\text{Lower bound} = Q_1 - 1.5 \times IQR
\text{Upper bound} = Q_3 + 1.5 \times IQR
```

2. **Z-score Method:**
```math
z_i = \frac{x_i - \bar{x}}{s}
```
   Values with $|z_i| > 3$ are considered outliers.

3. **Modified Z-score Method (Robust):**
```math
M_i = \frac{0.6745(x_i - \text{median})}{\text{MAD}}
```
   Values with $|M_i| > 3.5$ are considered outliers.

4. **Mahalanobis Distance (Multivariate):**
```math
D^2 = (x - \mu)^T \Sigma^{-1} (x - \mu)
```

```python
# Comprehensive outlier detection function
def detect_outliers(data):
    # IQR method
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound_iqr = q1 - 1.5 * iqr
    upper_bound_iqr = q3 + 1.5 * iqr
    
    outliers_iqr = (data < lower_bound_iqr) | (data > upper_bound_iqr)
    
    # Z-score method
    z_scores = np.abs((data - data.mean()) / data.std())
    outliers_z = z_scores > 3
    
    # Modified Z-score method (robust)
    median_val = np.median(data)
    mad_val = stats.median_abs_deviation(data)
    modified_z_scores = np.abs((data - median_val) / mad_val)
    outliers_modified_z = modified_z_scores > 3.5
    
    # Extreme IQR method (more conservative)
    lower_bound_extreme = q1 - 3 * iqr
    upper_bound_extreme = q3 + 3 * iqr
    outliers_extreme_iqr = (data < lower_bound_extreme) | (data > upper_bound_extreme)
    
    # Percentile method
    outliers_percentile = (data < np.percentile(data, 1)) | (data > np.percentile(data, 99))
    
    return {
        'outliers_iqr': np.where(outliers_iqr)[0],
        'outliers_z': np.where(outliers_z)[0],
        'outliers_modified_z': np.where(outliers_modified_z)[0],
        'outliers_extreme_iqr': np.where(outliers_extreme_iqr)[0],
        'outliers_percentile': np.where(outliers_percentile)[0],
        'bounds_iqr': [lower_bound_iqr, upper_bound_iqr],
        'bounds_extreme_iqr': [lower_bound_extreme, upper_bound_extreme],
        'z_scores': z_scores,
        'modified_z_scores': modified_z_scores,
        'q1': q1,
        'q3': q3,
        'iqr': iqr
    }

# Detect outliers in sepal length data
sepal_outliers = detect_outliers(sepal_length)

print("=== OUTLIER DETECTION ===")
print(f"Sample size: {len(sepal_length)}")
print(f"Range: [{sepal_length.min():.2f}, {sepal_length.max():.2f}]")
print(f"Q1: {sepal_outliers['q1']:.2f}")
print(f"Q3: {sepal_outliers['q3']:.2f}")
print(f"IQR: {sepal_outliers['iqr']:.2f}\n")

print("Outlier Detection Results:")
print(f"IQR method outliers: {len(sepal_outliers['outliers_iqr'])}")
print(f"Z-score method outliers: {len(sepal_outliers['outliers_z'])}")
print(f"Modified Z-score outliers: {len(sepal_outliers['outliers_modified_z'])}")
print(f"Extreme IQR outliers: {len(sepal_outliers['outliers_extreme_iqr'])}")
print(f"Percentile outliers: {len(sepal_outliers['outliers_percentile'])}\n")

# Show outlier values
if len(sepal_outliers['outliers_iqr']) > 0:
    print(f"IQR outliers: {sepal_length.iloc[sepal_outliers['outliers_iqr']].values}")
if len(sepal_outliers['outliers_z']) > 0:
    print(f"Z-score outliers: {sepal_length.iloc[sepal_outliers['outliers_z']].values}")

# Impact analysis
print(f"\nImpact Analysis:")
original_mean = sepal_length.mean()
original_sd = sepal_length.std()

# Remove IQR outliers
clean_data_iqr = sepal_length[~sepal_outliers['outliers_iqr']]
if len(clean_data_iqr) < len(sepal_length):
    print(f"Mean without IQR outliers: {clean_data_iqr.mean():.3f} (original: {original_mean:.3f})")
    print(f"SD without IQR outliers: {clean_data_iqr.std():.3f} (original: {original_sd:.3f})")
```

## Practical Examples

Real-world applications of one-sample tests demonstrate their importance across various fields. These examples show how to apply the concepts in practice.

### Example 1: Quality Control

Quality control is a common application where one-sample tests are used to ensure products meet specifications.

**Scenario:** A manufacturing plant produces widgets with a target weight of 100 grams. A sample of 50 widgets is tested to ensure the production process is working correctly.

```python
# Simulate quality control data
np.random.seed(123)
n_products = 50
product_weights = np.random.normal(100, 5, n_products)

# Comprehensive quality control analysis
print("=== QUALITY CONTROL ANALYSIS ===")
print(f"Target weight: 100 grams")
print(f"Sample size: {n_products}")
print(f"Sample mean: {product_weights.mean():.2f} grams")
print(f"Sample SD: {product_weights.std():.2f} grams")

# Test if mean weight is 100 grams
weight_stat, weight_p = stats.ttest_1samp(product_weights, 100)
print(f"\nt-test results:")
print(f"t-statistic: {weight_stat:.3f}, p-value: {weight_p:.4f}")

# Calculate effect size
weight_effect = calculate_cohens_d(product_weights, 100)

print(f"\nQuality Control Results:")
print(f"Effect size: {weight_effect['cohens_d']:.3f}")
print(f"Interpretation: {interpret_effect_size(weight_effect['cohens_d'])}")

# Tolerance analysis
tolerance_limits = [95, 105]  # ±5 grams tolerance
within_tolerance = np.sum((product_weights >= tolerance_limits[0]) & 
                         (product_weights <= tolerance_limits[1]))
tolerance_percentage = within_tolerance / n_products * 100

print(f"\nTolerance Analysis:")
print(f"Within ±5g tolerance: {within_tolerance}/{n_products} ({tolerance_percentage:.1f}%)")

# Process capability
process_capability = (tolerance_limits[1] - tolerance_limits[0]) / (6 * product_weights.std())
print(f"Process capability (Cpk): {process_capability:.3f}")

# Decision making
alpha = 0.05
if weight_p < alpha:
    print(f"\nDECISION: Process adjustment needed (p < {alpha})")
else:
    print(f"\nDECISION: Process is in control (p >= {alpha})")
```

### Example 2: Educational Assessment

Educational assessment uses one-sample tests to evaluate whether students meet learning objectives or pass thresholds.

**Scenario:** A teacher wants to determine if her class of 30 students has achieved a passing score (70%) on a standardized test, with the goal of demonstrating that the class performs above the minimum threshold.

```python
# Simulate test scores
np.random.seed(123)
n_students = 30
test_scores = np.random.normal(75, 10, n_students)

# Comprehensive educational assessment
print("=== EDUCATIONAL ASSESSMENT ===")
print(f"Passing threshold: 70%")
print(f"Sample size: {n_students} students")
print(f"Sample mean: {test_scores.mean():.1f}%")
print(f"Sample SD: {test_scores.std():.1f}%")
print(f"Range: [{test_scores.min():.1f}, {test_scores.max():.1f}]%")

# Descriptive statistics
passing_students = np.sum(test_scores >= 70)
passing_rate = passing_students / n_students * 100

print(f"\nDescriptive Statistics:")
print(f"Students passing (≥70%): {passing_students}/{n_students} ({passing_rate:.1f}%)")
print(f"Students below threshold: {n_students - passing_students}")

# Test if mean score is above 70 (passing threshold)
passing_stat, passing_p = stats.ttest_1samp(test_scores, 70, alternative='greater')
print(f"\nt-test results (one-tailed greater):")
print(f"t-statistic: {passing_stat:.3f}, p-value: {passing_p:.4f}")

# Calculate confidence interval
ci_stat, ci_p, ci_lower, ci_upper = stats.t.interval(0.95, len(test_scores)-1, 
                                                     loc=test_scores.mean(), 
                                                     scale=stats.sem(test_scores))
print(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]%")

# Effect size
score_effect = calculate_cohens_d(test_scores, 70)
print(f"Effect size: {score_effect['cohens_d']:.3f}")
print(f"Effect interpretation: {interpret_effect_size(score_effect['cohens_d'])}")

# Power analysis
power_analysis = TTestPower()
power = power_analysis.power(effect_size=score_effect['cohens_d'], 
                           nobs=n_students, alpha=0.05)
print(f"Power for detecting this effect: {power:.3f}")

# Educational interpretation
alpha = 0.05
if passing_p < alpha:
    print(f"\nEDUCATIONAL CONCLUSION:")
    print(f"✓ The class significantly exceeds the passing threshold (p < {alpha})")
    print(f"✓ The intervention/teaching method appears effective")
    print(f"✓ Consider advancing to more challenging material")
else:
    print(f"\nEDUCATIONAL CONCLUSION:")
    print(f"✗ The class does not significantly exceed the passing threshold (p >= {alpha})")
    print(f"✗ Additional instruction may be needed")
    print(f"✗ Consider reviewing foundational concepts")

# Performance categories
excellent = np.sum(test_scores >= 90)
good = np.sum((test_scores >= 80) & (test_scores < 90))
satisfactory = np.sum((test_scores >= 70) & (test_scores < 80))
needs_improvement = np.sum(test_scores < 70)

print(f"\nPerformance Distribution:")
print(f"Excellent (≥90%): {excellent} students")
print(f"Good (80-89%): {good} students")
print(f"Satisfactory (70-79%): {satisfactory} students")
print(f"Needs Improvement (<70%): {needs_improvement} students")
```

### Example 3: Medical Research

Medical research frequently uses one-sample tests to compare patient outcomes against established norms or baseline values.

**Scenario:** A researcher wants to determine if patients with a specific condition have elevated systolic blood pressure compared to the normal range of 120 mmHg.

```python
# Simulate blood pressure data
np.random.seed(123)
n_patients = 25
systolic_bp = np.random.normal(130, 15, n_patients)

# Comprehensive medical analysis
print("=== MEDICAL RESEARCH ANALYSIS ===")
print(f"Normal systolic BP: 120 mmHg")
print(f"Sample size: {n_patients} patients")
print(f"Sample mean: {systolic_bp.mean():.1f} mmHg")
print(f"Sample SD: {systolic_bp.std():.1f} mmHg")
print(f"Range: [{systolic_bp.min():.1f}, {systolic_bp.max():.1f}] mmHg")

# Clinical categories
normal_bp = np.sum(systolic_bp < 120)
elevated_bp = np.sum((systolic_bp >= 120) & (systolic_bp < 130))
stage1_hypertension = np.sum((systolic_bp >= 130) & (systolic_bp < 140))
stage2_hypertension = np.sum(systolic_bp >= 140)

print(f"\nClinical Classification:")
print(f"Normal (<120): {normal_bp} patients")
print(f"Elevated (120-129): {elevated_bp} patients")
print(f"Stage 1 Hypertension (130-139): {stage1_hypertension} patients")
print(f"Stage 2 Hypertension (≥140): {stage2_hypertension} patients")

# Test if mean systolic BP is different from 120 (normal)
bp_stat, bp_p = stats.ttest_1samp(systolic_bp, 120)
print(f"\nt-test results:")
print(f"t-statistic: {bp_stat:.3f}, p-value: {bp_p:.4f}")

# Nonparametric alternative
bp_wilcox_stat, bp_wilcox_p = stats.wilcoxon(systolic_bp - 120)
print(f"\nWilcoxon signed-rank test:")
print(f"W-statistic: {bp_wilcox_stat:.3f}, p-value: {bp_wilcox_p:.4f}")

# Effect size
bp_effect = calculate_cohens_d(systolic_bp, 120)
print(f"Effect size: {bp_effect['cohens_d']:.3f}")
print(f"Effect interpretation: {interpret_effect_size(bp_effect['cohens_d'])}")

# Compare parametric and nonparametric results
print(f"\nTest Comparison:")
print(f"t-test p-value: {bp_p:.4f}")
print(f"Wilcoxon p-value: {bp_wilcox_p:.4f}")

# Clinical interpretation
alpha = 0.05
if bp_p < alpha:
    print(f"\nCLINICAL CONCLUSION:")
    print(f"✓ Patients have significantly elevated BP compared to normal (p < {alpha})")
    print(f"✓ Clinical intervention may be warranted")
    print(f"✓ Consider lifestyle modifications or medication")
else:
    print(f"\nCLINICAL CONCLUSION:")
    print(f"✗ No significant elevation in BP compared to normal (p >= {alpha})")
    print(f"✗ Continue monitoring as standard")

# Risk assessment
high_risk_patients = np.sum(systolic_bp >= 140)
risk_percentage = high_risk_patients / n_patients * 100

print(f"\nRisk Assessment:")
print(f"High-risk patients (≥140 mmHg): {high_risk_patients}/{n_patients} ({risk_percentage:.1f}%)")

# Confidence interval for clinical decision making
bp_ci_lower, bp_ci_upper = stats.t.interval(0.95, len(systolic_bp)-1, 
                                           loc=systolic_bp.mean(), 
                                           scale=stats.sem(systolic_bp))
print(f"95% CI for mean BP: [{bp_ci_lower:.1f}, {bp_ci_upper:.1f}] mmHg")

# Power analysis for future studies
bp_power_analysis = TTestPower()
bp_power = bp_power_analysis.power(effect_size=bp_effect['cohens_d'], 
                                  nobs=n_patients, alpha=0.05)
print(f"Power for detecting this effect: {bp_power:.3f}")

# Sample size recommendation for future studies
if bp_power < 0.8:
    recommended_n = bp_power_analysis.solve_power(effect_size=bp_effect['cohens_d'], 
                                                 alpha=0.05, power=0.8)
    print(f"Recommended sample size for 80% power: {int(np.ceil(recommended_n))}")
```

## Advanced Topics

Advanced methods provide robust alternatives and enhanced inference capabilities for one-sample tests.

### Bootstrap Confidence Intervals

Bootstrap methods provide nonparametric confidence intervals that don't rely on distributional assumptions. They're particularly useful when data doesn't meet normality assumptions.

**Mathematical Foundation:**

The bootstrap estimates the sampling distribution by resampling with replacement from the original data.

**Bootstrap Algorithm:**
1. Draw $B$ bootstrap samples of size $n$ with replacement
2. Calculate statistic of interest for each bootstrap sample
3. Use empirical distribution of bootstrap statistics for inference

**Bootstrap Confidence Interval:**
```math
CI_{1-\alpha} = [\hat{\theta}_{\alpha/2}, \hat{\theta}_{1-\alpha/2}]
```

where $\hat{\theta}_p$ is the $p$-th percentile of bootstrap statistics.

```python
# Bootstrap function for mean
def boot_mean(data, indices):
    return np.mean(data[indices])

# Comprehensive bootstrap analysis
from scipy.stats import bootstrap

# Perform bootstrap analysis
bootstrap_result = bootstrap((sepal_length,), np.mean, n_resamples=10000, 
                           confidence_level=0.95, method='percentile')

# Calculate different types of bootstrap CIs
bootstrap_means = []
for _ in range(10000):
    bootstrap_sample = np.random.choice(sepal_length, size=len(sepal_length), replace=True)
    bootstrap_means.append(np.mean(bootstrap_sample))

bootstrap_means = np.array(bootstrap_means)
percentile_ci = np.percentile(bootstrap_means, [2.5, 97.5])
normal_ci = [np.mean(bootstrap_means) - 1.96 * np.std(bootstrap_means),
             np.mean(bootstrap_means) + 1.96 * np.std(bootstrap_means)]

print("=== BOOTSTRAP ANALYSIS ===")
print(f"Number of bootstrap samples: 10,000")
print(f"Original sample mean: {sepal_length.mean():.3f}")
print(f"Bootstrap mean: {bootstrap_means.mean():.3f}")
print(f"Bootstrap SD: {bootstrap_means.std():.3f}")

print(f"\nBootstrap Confidence Intervals:")
print(f"Percentile 95% CI: [{percentile_ci[0]:.3f}, {percentile_ci[1]:.3f}]")
print(f"Normal 95% CI: [{normal_ci[0]:.3f}, {normal_ci[1]:.3f}]")

# Compare with parametric methods
t_ci_lower, t_ci_upper = stats.t.interval(0.95, len(sepal_length)-1, 
                                         loc=sepal_length.mean(), 
                                         scale=stats.sem(sepal_length))
print(f"\nComparison with Parametric Methods:")
print(f"t-test 95% CI: [{t_ci_lower:.3f}, {t_ci_upper:.3f}]")
print(f"Bootstrap percentile 95% CI: [{percentile_ci[0]:.3f}, {percentile_ci[1]:.3f}]")

# Bootstrap bias and standard error
bootstrap_bias = bootstrap_means.mean() - sepal_length.mean()
bootstrap_se = bootstrap_means.std()

print(f"\nBootstrap Diagnostics:")
print(f"Bias: {bootstrap_bias:.4f}")
print(f"Standard Error: {bootstrap_se:.4f}")
print(f"Bias-corrected mean: {sepal_length.mean() - bootstrap_bias:.3f}")
```

### Robust One-Sample Tests

Robust methods provide resistance to outliers and violations of distributional assumptions while maintaining good statistical properties.

**Mathematical Foundation:**

**Trimmed Mean:**
```math
\bar{x}_\alpha = \frac{1}{n-2k} \sum_{i=k+1}^{n-k} x_{(i)}
```

where $k = \lfloor n\alpha \rfloor$ and $x_{(i)}$ are the ordered observations.

**Winsorized Variance:**
```math
s_w^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x}_w)^2
```

where $\bar{x}_w$ is the Winsorized mean.

**Robust t-statistic:**
```math
t_{robust} = \frac{\bar{x}_\alpha - \mu_0}{s_w/\sqrt{n}}
```

```python
# Comprehensive robust testing function
from scipy.stats import trim_mean
from sklearn.robust import HuberRegressor

def robust_t_test(data, mu, trim=0.1):
    trimmed_data = data.dropna()
    n = len(trimmed_data)
    k = int(n * trim)
    
    if k > 0:
        sorted_data = np.sort(trimmed_data)
        trimmed_values = sorted_data[k:(n - k)]
    else:
        trimmed_values = trimmed_data
    
    trimmed_mean_val = np.mean(trimmed_values)
    trimmed_var = np.var(trimmed_values, ddof=1)
    trimmed_se = np.sqrt(trimmed_var / (n - 2 * k))
    
    t_statistic = (trimmed_mean_val - mu) / trimmed_se
    df = n - 2 * k - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
    
    # Winsorized statistics
    winsorized_data = data.copy()
    if k > 0:
        winsorized_data[winsorized_data < sorted_data[k]] = sorted_data[k]
        winsorized_data[winsorized_data > sorted_data[n - k - 1]] = sorted_data[n - k - 1]
    winsorized_mean = winsorized_data.mean()
    winsorized_var = winsorized_data.var(ddof=1)
    
    # M-estimator (Huber's method)
    huber_reg = HuberRegressor(fit_intercept=True)
    huber_reg.fit(np.ones((len(data), 1)), data)
    huber_estimate = huber_reg.intercept_
    
    return {
        'trimmed_mean': trimmed_mean_val,
        'trimmed_var': trimmed_var,
        'trimmed_se': trimmed_se,
        't_statistic': t_statistic,
        'p_value': p_value,
        'df': df,
        'winsorized_mean': winsorized_mean,
        'winsorized_var': winsorized_var,
        'huber_estimate': huber_estimate,
        'trim_proportion': trim,
        'n_trimmed': 2 * k
    }

# Apply robust t-test to sepal length data
sepal_robust = robust_t_test(sepal_length, 5.5, trim=0.1)

print("=== ROBUST ONE-SAMPLE TEST ===")
print(f"Trim proportion: {sepal_robust['trim_proportion']}")
print(f"Number of observations trimmed: {sepal_robust['n_trimmed']}\n")

print("Robust Statistics:")
print(f"Trimmed mean: {sepal_robust['trimmed_mean']:.3f}")
print(f"Winsorized mean: {sepal_robust['winsorized_mean']:.3f}")
print(f"Huber M-estimate: {sepal_robust['huber_estimate']:.3f}")
print(f"Original mean: {sepal_length.mean():.3f}\n")

print("Test Results:")
print(f"Robust t-statistic: {sepal_robust['t_statistic']:.3f}")
print(f"Degrees of freedom: {sepal_robust['df']}")
print(f"p-value: {sepal_robust['p_value']:.4f}")

# Compare with standard t-test
standard_t_stat, standard_t_p = stats.ttest_1samp(sepal_length, 5.5)
print(f"\nComparison with Standard t-test:")
print(f"Standard t-statistic: {standard_t_stat:.3f}")
print(f"Standard p-value: {standard_t_p:.4f}")
print(f"Robust t-statistic: {sepal_robust['t_statistic']:.3f}")
print(f"Robust p-value: {sepal_robust['p_value']:.4f}")

# Effect size for robust test
robust_effect_size = (sepal_robust['trimmed_mean'] - 5.5) / np.sqrt(sepal_robust['trimmed_var'])
print(f"\nRobust Effect Size:")
print(f"Cohen's d (robust): {robust_effect_size:.3f}")
print(f"Interpretation: {interpret_effect_size(robust_effect_size)}")
```

## Best Practices

Following best practices ensures valid statistical inference and meaningful results. These guidelines help researchers make appropriate choices and avoid common pitfalls.

### Test Selection Guidelines

The choice of test depends on data characteristics, sample size, and research objectives. This decision tree helps select the most appropriate method.

**Decision Framework:**

1. **Sample Size Consideration:**
   - $n < 15$: Nonparametric tests recommended
   - $15 \leq n < 30$: Check normality carefully
   - $n \geq 30$: Central Limit Theorem applies

2. **Distribution Assessment:**
   - Normal data: Parametric tests preferred
   - Non-normal data: Nonparametric or robust methods

3. **Outlier Sensitivity:**
   - Outliers present: Robust methods recommended
   - Clean data: Standard methods appropriate

```python
# Comprehensive test selection function
def choose_one_sample_test(data, hypothesized_value):
    print("=== ONE-SAMPLE TEST SELECTION GUIDE ===")
    
    # Basic information
    n = len(data)
    print(f"Sample size: {n}")
    print(f"Hypothesized value: {hypothesized_value}")
    print(f"Sample mean: {data.mean():.3f}")
    print(f"Sample SD: {data.std():.3f}\n")
    
    # Normality assessment
    shapiro_stat, shapiro_p = shapiro(data)
    print("=== NORMALITY ASSESSMENT ===")
    print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
    
    # Skewness and kurtosis
    data_skewness = skew(data)
    data_kurtosis = kurtosis(data)
    print(f"Skewness: {data_skewness:.3f}")
    print(f"Kurtosis: {data_kurtosis:.3f}")
    
    # Outlier assessment
    outliers = detect_outliers(data)
    n_outliers_iqr = len(outliers['outliers_iqr'])
    n_outliers_z = len(outliers['outliers_z'])
    
    print(f"\n=== OUTLIER ASSESSMENT ===")
    print(f"IQR method outliers: {n_outliers_iqr}")
    print(f"Z-score method outliers: {n_outliers_z}")
    
    # Effect size
    effect_size = calculate_cohens_d(data, hypothesized_value)
    print(f"\n=== EFFECT SIZE ===")
    print(f"Cohen's d: {effect_size['cohens_d']:.3f}")
    print(f"Interpretation: {interpret_effect_size(effect_size['cohens_d'])}")
    
    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    
    # Sample size recommendations
    if n < 15:
        print("⚠️  Small sample size (n < 15):")
        print("   - Nonparametric tests strongly recommended")
        print("   - Consider collecting more data if possible")
        print("   - Use Wilcoxon signed-rank test")
    elif n < 30:
        print("⚠️  Moderate sample size (15 ≤ n < 30):")
        print("   - Check normality carefully")
        if shapiro_p >= 0.05:
            print("   - Data appears normal: t-test appropriate")
        else:
            print("   - Data not normal: Use Wilcoxon test")
    else:
        print("✓ Large sample size (n ≥ 30):")
        print("   - Central Limit Theorem applies")
        print("   - t-test is robust to normality violations")
    
    # Normality recommendations
    if shapiro_p < 0.05:
        print("✗ Normality assumption violated:")
        print("   - Consider nonparametric alternatives")
        print("   - Robust methods may be appropriate")
    else:
        print("✓ Normality assumption met:")
        print("   - Parametric tests are appropriate")
    
    # Outlier recommendations
    if n_outliers_iqr > 0:
        print("⚠️  Outliers detected:")
        print("   - Consider robust methods")
        print("   - Trimmed t-test recommended")
        print("   - Investigate outlier causes")
    else:
        print("✓ No significant outliers detected")
    
    # Final recommendation
    print(f"\n=== FINAL RECOMMENDATION ===")
    if n >= 30:
        print("Primary: One-sample t-test")
        print("Alternative: Bootstrap confidence intervals")
    elif shapiro_p >= 0.05 and n_outliers_iqr == 0:
        print("Primary: One-sample t-test")
        print("Alternative: Wilcoxon signed-rank test")
    else:
        print("Primary: Wilcoxon signed-rank test")
        print("Alternative: Robust t-test (trimmed mean)")
    
    # Power consideration
    power_analysis = TTestPower()
    power = power_analysis.power(effect_size=effect_size['cohens_d'], 
                               nobs=n, alpha=0.05)
    
    print(f"\n=== POWER ANALYSIS ===")
    print(f"Current power: {power:.3f}")
    if power < 0.8:
        recommended_n = power_analysis.solve_power(effect_size=effect_size['cohens_d'], 
                                                 alpha=0.05, power=0.8)
        print(f"Recommended sample size for 80% power: {int(np.ceil(recommended_n))}")

# Apply to sepal length data
choose_one_sample_test(sepal_length, 5.5)
```

### Reporting Guidelines

Proper reporting of statistical results is essential for transparency and reproducibility. These guidelines ensure comprehensive and clear communication of findings.

**Essential Elements for Reporting:**

1. **Descriptive Statistics:** Mean, standard deviation, sample size
2. **Test Statistics:** t-value, degrees of freedom, p-value
3. **Effect Size:** Cohen's d with interpretation
4. **Confidence Intervals:** 95% CI for the mean difference
5. **Assumption Checks:** Normality tests, outlier assessment
6. **Practical Significance:** Interpretation of results

**APA Style Reporting:**
- t-test: $t(df) = t\text{-value}, p = p\text{-value}$
- Effect size: $d = \text{value}$
- Confidence interval: 95% CI [lower, upper]

```python
# Comprehensive reporting function
def generate_test_report(test_result, data, hypothesized_value, test_type="t-test"):
    print("=== COMPREHENSIVE ONE-SAMPLE TEST REPORT ===\n")
    
    # Basic information
    print("RESEARCH CONTEXT:")
    print(f"Test type: {test_type}")
    print(f"Sample size: {len(data)}")
    print(f"Hypothesized value: {hypothesized_value}")
    print(f"Alpha level: 0.05\n")
    
    # Descriptive statistics
    print("DESCRIPTIVE STATISTICS:")
    print(f"Sample mean: {data.mean():.3f}")
    print(f"Sample SD: {data.std():.3f}")
    print(f"Sample median: {data.median():.3f}")
    print(f"Range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"IQR: {data.quantile(0.75) - data.quantile(0.25):.3f}\n")
    
    # Assumption checks
    print("ASSUMPTION CHECKS:")
    shapiro_stat, shapiro_p = shapiro(data)
    print(f"Normality (Shapiro-Wilk): W = {shapiro_stat:.3f}, p = {shapiro_p:.4f}")
    
    outliers = detect_outliers(data)
    print(f"Outliers (IQR method): {len(outliers['outliers_iqr'])}")
    print(f"Outliers (Z-score method): {len(outliers['outliers_z'])}\n")
    
    # Test results
    if test_type == "t-test":
        print("T-TEST RESULTS:")
        print(f"t-statistic: {test_result[0]:.3f}")
        print(f"p-value: {test_result[1]:.4f}")
        
        # Calculate confidence interval
        ci_lower, ci_upper = stats.t.interval(0.95, len(data)-1, 
                                             loc=data.mean(), 
                                             scale=stats.sem(data))
        print(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"Standard Error: {stats.sem(data):.3f}\n")
    elif test_type == "wilcoxon":
        print("WILCOXON SIGNED-RANK TEST RESULTS:")
        print(f"W-statistic: {test_result[0]:.3f}")
        print(f"p-value: {test_result[1]:.4f}\n")
    
    # Effect size
    effect_size = calculate_cohens_d(data, hypothesized_value)
    print("EFFECT SIZE:")
    print(f"Cohen's d: {effect_size['cohens_d']:.3f}")
    print(f"Interpretation: {interpret_effect_size(effect_size['cohens_d'])}")
    print(f"95% CI for effect size: [{effect_size['ci_lower']:.3f}, {effect_size['ci_upper']:.3f}]\n")
    
    # Power analysis
    power_analysis = TTestPower()
    power = power_analysis.power(effect_size=effect_size['cohens_d'], 
                               nobs=len(data), alpha=0.05)
    print("POWER ANALYSIS:")
    print(f"Observed power: {power:.3f}")
    if power < 0.8:
        recommended_n = power_analysis.solve_power(effect_size=effect_size['cohens_d'], 
                                                 alpha=0.05, power=0.8)
        print(f"Recommended sample size for 80% power: {int(np.ceil(recommended_n))}")
    print()
    
    # Statistical decision
    alpha = 0.05
    print("STATISTICAL DECISION:")
    if test_result[1] < alpha:
        print(f"✓ Reject the null hypothesis (p < {alpha})")
        print(f"✓ There is significant evidence that the population mean differs from {hypothesized_value}")
    else:
        print(f"✗ Fail to reject the null hypothesis (p >= {alpha})")
        print(f"✗ There is insufficient evidence that the population mean differs from {hypothesized_value}")
    print()
    
    # Practical interpretation
    print("PRACTICAL INTERPRETATION:")
    mean_diff = data.mean() - hypothesized_value
    if abs(effect_size['cohens_d']) < 0.2:
        print("The effect is small and may not be practically meaningful.")
    elif abs(effect_size['cohens_d']) < 0.5:
        print("The effect is moderate and may be practically meaningful.")
    else:
        print("The effect is large and likely practically meaningful.")
    
    # APA style reporting
    print(f"\nAPA STYLE REPORTING:")
    if test_type == "t-test":
        print(f"A one-sample t-test was conducted to compare the sample mean (M = {data.mean():.2f}, SD = {data.std():.2f}) to the hypothesized value of {hypothesized_value}. ", end="")
        
        if test_result[1] < alpha:
            print(f"The test was significant, t({len(data)-1}) = {test_result[0]:.2f}, p = {test_result[1]:.3f}, d = {effect_size['cohens_d']:.2f}. ", end="")
            print(f"The 95% confidence interval for the mean difference was [{ci_lower:.2f}, {ci_upper:.2f}].")
        else:
            print(f"The test was not significant, t({len(data)-1}) = {test_result[0]:.2f}, p = {test_result[1]:.3f}, d = {effect_size['cohens_d']:.2f}.")

# Generate comprehensive report for sepal length t-test
sepal_t_result = stats.ttest_1samp(sepal_length, 5.5)
generate_test_report(sepal_t_result, sepal_length, 5.5, "t-test")
```

## Exercises

These exercises provide hands-on practice with one-sample tests, covering various scenarios and applications. Work through them systematically to build your understanding.

### Exercise 1: Basic One-Sample t-Test

**Scenario:** A car manufacturer claims their vehicles have an average weight of 3.0 thousand pounds. Test this claim using the mtcars dataset.

**Tasks:**
1. Perform a one-sample t-test to compare the mean weight to 3.0
2. Calculate and interpret the effect size
3. Report the results in APA style
4. Provide a practical interpretation

```python
# Your code here
# Hint: Use stats.ttest_1samp() function with popmean = 3.0
# Hint: Use the calculate_cohens_d() function for effect size
```

### Exercise 2: Effect Size Analysis

**Scenario:** Compare effect sizes across different variables in the mtcars dataset to understand which variables show the strongest deviations from hypothesized values.

**Tasks:**
1. Test MPG against hypothesized mean of 20
2. Test horsepower against hypothesized mean of 100
3. Test displacement against hypothesized mean of 200
4. Compare effect sizes and interpret practical significance
5. Create a summary table of results

```python
# Your code here
# Hint: Test multiple variables and store results in a pandas DataFrame
# Hint: Use a loop or list comprehension to test multiple variables efficiently
```

### Exercise 3: Nonparametric Alternatives

**Scenario:** Investigate whether nonparametric tests provide different conclusions than parametric tests for skewed data.

**Tasks:**
1. Generate skewed data using rgamma() or rchisq()
2. Perform both t-test and Wilcoxon signed-rank test
3. Compare p-values and conclusions
4. Assess normality of the generated data
5. Explain when each test is more appropriate

```python
# Your code here
# Hint: skewed_data = np.random.gamma(2, 2, 30)
# Hint: Use shapiro() to check normality
# Hint: Compare stats.ttest_1samp() and stats.wilcoxon() results
```

### Exercise 4: Power Analysis

**Scenario:** Design a study to detect a medium effect size (d = 0.5) with adequate power.

**Tasks:**
1. Calculate required sample size for 80% power
2. Calculate required sample size for 90% power
3. Create a power curve for different effect sizes
4. Simulate data and verify power calculations
5. Discuss implications for study design

```python
# Your code here
# Hint: Use TTestPower() with different power levels
# Hint: Use np.arange() to create a range of effect sizes
# Hint: Use a loop to simulate multiple studies
```

### Exercise 5: Assumption Checking

**Scenario:** Perform comprehensive assumption checking for a dataset and recommend appropriate statistical methods.

**Tasks:**
1. Load a dataset (e.g., iris$Sepal.Length)
2. Check normality using multiple methods
3. Detect outliers using different criteria
4. Assess sample size adequacy
5. Recommend appropriate test based on findings
6. Perform the recommended test and interpret results

```python
# Your code here
# Hint: Use the check_normality() and detect_outliers() functions
# Hint: Use the choose_one_sample_test() function for recommendations
```

### Exercise 6: Bootstrap Confidence Intervals

**Scenario:** Compare bootstrap confidence intervals with parametric confidence intervals for non-normal data.

**Tasks:**
1. Generate non-normal data (e.g., exponential distribution)
2. Calculate parametric confidence interval using t.test()
3. Calculate bootstrap confidence interval using boot()
4. Compare the two methods
5. Discuss advantages and limitations of each approach

```python
# Your code here
# Hint: non_normal_data = np.random.exponential(2, 50)
# Hint: Use bootstrap() function with appropriate statistic function
# Hint: Use np.percentile() to get different types of bootstrap CIs
```

### Exercise 7: Robust Methods

**Scenario:** Compare robust methods with standard methods when outliers are present.

**Tasks:**
1. Create a dataset with outliers
2. Perform standard t-test
3. Perform robust t-test (trimmed mean)
4. Perform Wilcoxon signed-rank test
5. Compare results and discuss implications
6. Recommend the best approach for this data

```python
# Your code here
# Hint: Add outliers to existing data: data_with_outliers = np.append(data, [100, -50])
# Hint: Use the robust_t_test() function
# Hint: Compare effect sizes across methods
```

### Exercise 8: Real-World Application

**Scenario:** Analyze a real-world dataset and provide comprehensive statistical reporting.

**Tasks:**
1. Choose a variable from a dataset (e.g., mtcars$mpg, iris$Sepal.Length)
2. Formulate a research question
3. Perform comprehensive analysis including:
   - Descriptive statistics
   - Assumption checking
   - Appropriate statistical test
   - Effect size calculation
   - Power analysis
   - APA style reporting
4. Provide practical interpretation and recommendations

```python
# Your code here
# Hint: Use the generate_test_report() function
# Hint: Consider the context and implications of your findings
# Hint: Provide actionable recommendations based on results
```

### Exercise Solutions and Hints

**Exercise 1 Solution:**
```python
# Test weight against 3.0
weight_stat, weight_p = stats.ttest_1samp(data['sepal width (cm)'], 3.0)
print(f"t-statistic: {weight_stat:.3f}, p-value: {weight_p:.4f}")

# Effect size
weight_effect = calculate_cohens_d(data['sepal width (cm)'], 3.0)
print(f"Effect size: {weight_effect['cohens_d']:.3f}")
```

**Exercise 2 Solution:**
```python
# Test multiple variables
variables = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']
hypothesized_values = [5.5, 3.0, 3.8]

results = pd.DataFrame({
    'Variable': variables,
    'Sample_Mean': [data[var].mean() for var in variables],
    'Hypothesized': hypothesized_values,
    't_statistic': [None] * len(variables),
    'p_value': [None] * len(variables),
    'effect_size': [None] * len(variables)
})

for i, var in enumerate(variables):
    test_result = stats.ttest_1samp(data[var], hypothesized_values[i])
    effect_size = calculate_cohens_d(data[var], hypothesized_values[i])
    
    results.loc[i, 't_statistic'] = test_result[0]
    results.loc[i, 'p_value'] = test_result[1]
    results.loc[i, 'effect_size'] = effect_size['cohens_d']

print(results)
```

**Exercise 3 Solution:**
```python
# Generate skewed data
np.random.seed(123)
skewed_data = np.random.gamma(2, 2, 30)

# Compare tests
t_result = stats.ttest_1samp(skewed_data, 4)
wilcox_result = stats.wilcoxon(skewed_data - 4)

print(f"t-test p-value: {t_result[1]:.4f}")
print(f"Wilcoxon p-value: {wilcox_result[1]:.4f}")
```

**Exercise 4 Solution:**
```python
# Power analysis
power_analysis = TTestPower()
sample_size_80 = power_analysis.solve_power(effect_size=0.5, alpha=0.05, power=0.8)
sample_size_90 = power_analysis.solve_power(effect_size=0.5, alpha=0.05, power=0.9)

print(f"Required n for 80% power: {int(np.ceil(sample_size_80))}")
print(f"Required n for 90% power: {int(np.ceil(sample_size_90))}")
```

**Exercise 5 Solution:**
```python
# Comprehensive assumption checking
data_var = data['sepal length (cm)']
choose_one_sample_test(data_var, 5.5)
```

These exercises provide comprehensive practice with one-sample tests and help develop critical thinking about statistical methodology.

## Next Steps

In the next chapter, we'll learn about two-sample tests for comparing means between groups.

---

**Key Takeaways:**
- One-sample tests compare a sample mean to a hypothesized population mean
- t-test is appropriate for normally distributed data or large samples
- Nonparametric alternatives exist for non-normal data
- Always check assumptions before interpreting results
- Effect sizes provide important information about practical significance
- Power analysis helps determine appropriate sample sizes
- Bootstrap methods provide robust confidence intervals 