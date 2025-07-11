# Two-Sample Tests

## Overview

Two-sample tests are fundamental statistical procedures used to compare means or distributions between two groups. These tests form the backbone of experimental design and observational studies, allowing researchers to determine if observed differences between groups are statistically significant or due to random variation.

### When to Use Two-Sample Tests

Two-sample tests are appropriate when you want to:
- Compare treatment and control groups in experiments
- Assess differences between two populations
- Evaluate the effectiveness of interventions
- Test hypotheses about group differences
- Analyze before-and-after measurements

### Types of Two-Sample Tests

**Independent Samples:** Groups are unrelated (e.g., treatment vs control)
**Paired Samples:** Groups are related (e.g., before vs after, matched pairs)

### Key Concepts

**Null Hypothesis (H₀):** The population means are equal ($\mu_1 = \mu_2$)
**Alternative Hypothesis (H₁):** The population means are different ($\mu_1 \neq \mu_2$)

The test determines whether observed differences are statistically significant or due to random sampling variation.

### Mathematical Foundation

Two-sample tests are based on the sampling distribution of the difference between sample means:

```math
\bar{X}_1 - \bar{X}_2 \sim N\left(\mu_1 - \mu_2, \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}\right)
```

For large samples, the Central Limit Theorem ensures this distribution is approximately normal regardless of the underlying population distributions.

## Independent Samples t-Test

The independent samples t-test is the most commonly used parametric test for comparing means between two unrelated groups. It's robust and works well even with moderate violations of normality assumptions, especially for larger sample sizes.

### Mathematical Foundation

The independent samples t-test is based on the t-distribution and accounts for the uncertainty in estimating population standard deviations from sample data.

**Test Statistic:**
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
```

**Degrees of Freedom (Welch's approximation):**
```math
df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
```

**Pooled Standard Deviation (for equal variances):**
```math
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
```

**Confidence Interval:**
```math
CI = (\bar{x}_1 - \bar{x}_2) \pm t_{\alpha/2, df} \cdot \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
```

where:
- $\bar{x}_1, \bar{x}_2$ = sample means
- $s_1^2, s_2^2$ = sample variances
- $n_1, n_2$ = sample sizes
- $t_{\alpha/2, df}$ = critical t-value for confidence level $\alpha$

### Basic Independent Samples t-Test

The `independent_samples_t_test()` function demonstrates a comprehensive independent samples t-test using the Iris dataset. It shows manual calculations for Welch's t-test including the t-statistic, degrees of freedom, and confidence intervals, then compares with SciPy's implementation.

**Key features:**
- Manual calculation of Welch's t-test formula
- Confidence interval computation
- Verification with SciPy's ttest_ind function
- Real dataset example (Iris species comparison)
- Detailed output showing all intermediate calculations

### Equal vs Unequal Variances

The choice between equal and unequal variance t-tests depends on whether the population variances are assumed to be equal. This assumption significantly affects the test statistic and degrees of freedom.

**Mathematical Differences:**

**Equal Variances (Pooled t-test):**
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
```

where $s_p$ is the pooled standard deviation.

**Unequal Variances (Welch's t-test):**
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
```

**Degrees of Freedom:**
- Equal variances: $df = n_1 + n_2 - 2$
- Unequal variances: $df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}$

**When to Use Each:**
- **Equal variances:** When population variances are known to be equal or similar
- **Unequal variances:** When variances may differ (more conservative approach)

The `assumption_checking()` function demonstrates comprehensive assumption checking for two-sample tests, including Levene's test for homogeneity of variance and comparison between equal and unequal variance t-tests.

**Key features:**
- Levene's test for homogeneity of variance
- Manual F-test calculation
- Comparison of pooled vs Welch's t-test
- Confidence interval width comparison
- Decision-making based on assumption test results

### Effect Size for Independent Samples

Effect size measures the magnitude of the difference between groups, independent of sample size. This is crucial for understanding practical significance beyond statistical significance.

**Mathematical Foundation:**

**Cohen's d (Standardized Mean Difference):**
```math
d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}
```

where $s_p$ is the pooled standard deviation.

**Pooled Standard Deviation:**
```math
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
```

**Hedges' g (Unbiased Estimator):**
```math
g = d \cdot \left(1 - \frac{3}{4(n_1 + n_2) - 9}\right)
```

**Confidence Interval for Effect Size:**
```math
CI_d = d \pm t_{\alpha/2, df} \cdot SE_d
```

where $SE_d = \sqrt{\frac{n_1 + n_2}{n_1 n_2} + \frac{d^2}{2(n_1 + n_2)}}$

**Interpretation Guidelines:**
- Small effect: $|d| < 0.2$
- Medium effect: $0.2 \leq |d| < 0.5$
- Large effect: $0.5 \leq |d| < 0.8$
- Very large effect: $|d| \geq 0.8$

The `effect_size_calculations()` function demonstrates various effect size measures for two-sample comparisons, including Cohen's d, Hedges' g, and confidence intervals for effect sizes.

**Key features:**
- Cohen's d calculation with pooled standard deviation
- Hedges' g (bias-corrected effect size)
- Confidence intervals for effect sizes
- Effect size interpretation guidelines
- Power analysis based on effect size
- Sample size determination for desired power

## Paired Samples t-Test

The paired samples t-test is used when observations in the two groups are related or matched. This design increases statistical power by controlling for individual differences and reducing error variance.

### Mathematical Foundation

The paired t-test analyzes the differences between paired observations rather than the raw scores.

**Test Statistic:**
```math
t = \frac{\bar{d}}{s_d/\sqrt{n}}
```

where:
- $\bar{d} = \frac{1}{n}\sum_{i=1}^n d_i$ (mean of differences)
- $d_i = x_{1i} - x_{2i}$ (difference for pair $i$)
- $s_d = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (d_i - \bar{d})^2}$ (standard deviation of differences)

**Degrees of Freedom:**
```math
df = n - 1
```

**Confidence Interval:**
```math
CI = \bar{d} \pm t_{\alpha/2, n-1} \cdot \frac{s_d}{\sqrt{n}}
```

**Effect Size for Paired Data:**
```math
d = \frac{\bar{d}}{s_d}
```

### Basic Paired Samples t-Test

The `paired_samples_t_test()` function demonstrates paired t-test calculations using simulated before/after treatment data. It includes manual calculations of the t-statistic, confidence intervals, and effect size (Cohen's d for paired samples).

**Key features:**
- Manual calculation of paired t-test
- Confidence interval for mean difference
- Effect size calculation (Cohen's d)
- Comparison with SciPy's ttest_rel function
- Power comparison between paired and independent designs

### Paired Data Analysis

Paired data analysis provides insights into the relationship between measurements and the distribution of differences, which is crucial for understanding the effectiveness of interventions.

**Key Concepts:**

**Correlation in Paired Data:**
```math
r = \frac{\sum_{i=1}^n (x_{1i} - \bar{x}_1)(x_{2i} - \bar{x}_2)}{\sqrt{\sum_{i=1}^n (x_{1i} - \bar{x}_1)^2 \sum_{i=1}^n (x_{2i} - \bar{x}_2)^2}}
```

**Variance Reduction:**
```math
\text{Var}(\bar{d}) = \frac{\sigma_d^2}{n} = \frac{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}{n}
```

where $\rho$ is the correlation between paired observations.

**Power Advantage:**
The paired design increases power by reducing error variance through the correlation between measurements.

```python
# Create paired data frame
paired_data = pd.DataFrame({
    'subject': range(1, n_subjects + 1),
    'before': before_scores,
    'after': after_scores,
    'difference': after_scores - before_scores
})

# Comprehensive paired data analysis
print("=== PAIRED DATA ANALYSIS ===")
print(f"Sample size: {n_subjects}")
print(f"Mean difference: {paired_data['difference'].mean():.3f}")
print(f"SD of differences: {paired_data['difference'].std():.3f}")
print(f"SE of mean difference: {paired_data['difference'].std()/np.sqrt(n_subjects):.3f}")

# Correlation analysis
correlation = paired_data['before'].corr(paired_data['after'])
print(f"Correlation between before and after: {correlation:.3f}")

# Variance analysis
var_before = paired_data['before'].var()
var_after = paired_data['after'].var()
var_diff = paired_data['difference'].var()

print("\nVariance Analysis:")
print(f"Variance (before): {var_before:.3f}")
print(f"Variance (after): {var_after:.3f}")
print(f"Variance (differences): {var_diff:.3f}")
print(f"Theoretical variance (if independent): {var_before + var_after:.3f}")
print(f"Variance reduction: {var_before + var_after - var_diff:.3f}")

# Effect of correlation on power
theoretical_var_independent = var_before + var_after
actual_var_paired = var_diff
power_ratio = theoretical_var_independent / actual_var_paired

print("\nPower Analysis:")
print(f"Variance ratio (independent/paired): {power_ratio:.3f}")
print(f"Effective sample size increase: {power_ratio:.1f} times")

# Individual change analysis
positive_changes = (paired_data['difference'] > 0).sum()
negative_changes = (paired_data['difference'] < 0).sum()
no_change = (paired_data['difference'] == 0).sum()

print("\nIndividual Change Analysis:")
print(f"Subjects with improvement: {positive_changes} ({positive_changes/n_subjects*100:.1f}%)")
print(f"Subjects with decline: {negative_changes} ({negative_changes/n_subjects*100:.1f}%)")
print(f"Subjects with no change: {no_change} ({no_change/n_subjects*100:.1f}%)")

# Normality of differences
shapiro_diff = stats.shapiro(paired_data['difference'])
print("\nNormality of Differences:")
print(f"Shapiro-Wilk p-value: {shapiro_diff.pvalue:.4f}")
if shapiro_diff.pvalue < 0.05:
    print("Differences are not normally distributed - consider nonparametric test")
else:
    print("Differences appear normally distributed - parametric test appropriate")

# Visualize paired data
import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before vs After plot
ax1.scatter(paired_data['before'], paired_data['after'], alpha=0.7)
ax1.plot([paired_data['before'].min(), paired_data['before'].max()], 
         [paired_data['before'].min(), paired_data['before'].max()], 
         'r--', label='y=x')
ax1.set_xlabel('Before Scores')
ax1.set_ylabel('After Scores')
ax1.set_title(f'Before vs After Scores\nr = {correlation:.3f}')
ax1.legend()

# Difference plot
ax2.hist(paired_data['difference'], bins=10, alpha=0.7, color='steelblue', edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', label='No change')
ax2.axvline(paired_data['difference'].mean(), color='blue', linewidth=2, label='Mean difference')
ax2.set_xlabel('Difference (After - Before)')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Distribution of Differences\nMean = {paired_data["difference"].mean():.2f}')
ax2.legend()

plt.tight_layout()
plt.show()
```

## Nonparametric Two-Sample Tests

Nonparametric tests make fewer assumptions about the underlying population distributions and are robust to violations of normality. They're particularly useful for small samples or when data is skewed or contains outliers.

### Mann-Whitney U Test (Wilcoxon Rank-Sum)

The Mann-Whitney U test is the most commonly used nonparametric alternative to the independent samples t-test. It tests whether the distributions of two groups differ by analyzing the ranks of the combined data.

**Mathematical Foundation:**

1. **Combine and rank all observations:**
   ```math
   R_i = \text{rank}(x_i) \text{ in combined sample}
   ```

2. **Calculate rank sums:**
   ```math
   W_1 = \sum_{i=1}^{n_1} R_i \text{ (Group 1)}
   ```
   ```math
   W_2 = \sum_{i=1}^{n_2} R_i \text{ (Group 2)}
   ```

3. **Calculate U statistics:**
   ```math
   U_1 = W_1 - \frac{n_1(n_1 + 1)}{2}
   ```
   ```math
   U_2 = W_2 - \frac{n_2(n_2 + 1)}{2}
   ```

4. **Test statistic:**
   ```math
   U = \min(U_1, U_2)
   ```

**For large samples ($n_1, n_2 > 20$), U is approximately normal:**
```math
Z = \frac{U - \mu_U}{\sigma_U}
```

where:
```math
\mu_U = \frac{n_1 n_2}{2}
```
```math
\sigma_U = \sqrt{\frac{n_1 n_2(n_1 + n_2 + 1)}{12}}
```

**Effect Size (r):**
```math
r = \frac{Z}{\sqrt{n_1 + n_2}}
```

```python
# Mann-Whitney U test for independent samples
wilcox_test = stats.mannwhitneyu(species_0_sepal_length, species_1_sepal_length, alternative='two-sided')

# Manual calculation for understanding
combined_data = np.concatenate([species_0_sepal_length, species_1_sepal_length])
ranks = stats.rankdata(combined_data)
n1 = len(species_0_sepal_length)
n2 = len(species_1_sepal_length)

# Rank sums
ranks_group1 = ranks[:n1]
ranks_group2 = ranks[n1:n1+n2]
W1 = ranks_group1.sum()
W2 = ranks_group2.sum()

# U statistics
U1 = W1 - (n1 * (n1 + 1)) / 2
U2 = W2 - (n2 * (n2 + 1)) / 2
U_statistic = min(U1, U2)

print("\nManual Mann-Whitney Calculation:")
print(f"Rank sum (Group 1): {W1}")
print(f"Rank sum (Group 2): {W2}")
print(f"U1: {U1}")
print(f"U2: {U2}")
print(f"U statistic: {U_statistic}")
print(f"SciPy U statistic: {wilcox_test.statistic}")

# Compare with t-test results
print("\nComparison of parametric and nonparametric tests:")
print(f"t-test p-value: {p_value:.4f}")
print(f"Wilcoxon p-value: {wilcox_test.pvalue:.4f}")

# Effect size for Wilcoxon test
wilcox_effect_size = abs(stats.norm.ppf(wilcox_test.pvalue / 2)) / np.sqrt(n1 + n2)
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

# Power comparison
wilcox_power = power_analysis.power(effect_size=wilcox_effect_size, 
                                   nobs1=n1, 
                                   nobs2=n2, 
                                   alpha=0.05)
t_power = power_analysis.power(effect_size=species_effect['cohens_d'], 
                              nobs1=n1, 
                              nobs2=n2, 
                              alpha=0.05)

print("\nPower Comparison:")
print(f"t-test power: {t_power:.3f}")
print(f"Wilcoxon power: {wilcox_power:.3f}")
```

### Wilcoxon Signed-Rank Test for Paired Data

The Wilcoxon signed-rank test is the nonparametric alternative to the paired t-test. It analyzes the ranks of the absolute differences between paired observations.

**Mathematical Foundation:**

1. **Calculate differences:**
   ```math
   d_i = x_{1i} - x_{2i}
   ```

2. **Rank absolute differences:**
   ```math
   R_i = \text{rank}(|d_i|)
   ```

3. **Assign signs:**
   ```math
   s_i = \text{sign}(d_i)
   ```

4. **Calculate test statistic:**
   ```math
   W = \sum_{i=1}^{n} s_i \cdot R_i
   ```

**For large samples ($n > 20$), W is approximately normal:**
```math
Z = \frac{W}{\sqrt{\frac{n(n+1)(2n+1)}{6}}}
```

**Effect Size (r):**
```math
r = \frac{Z}{\sqrt{n}}
```

```python
# Wilcoxon signed-rank test for paired samples
paired_wilcox = stats.wilcoxon(before_scores, after_scores)
print(f"Wilcoxon signed-rank test: statistic={paired_wilcox.statistic:.3f}, p-value={paired_wilcox.pvalue:.4f}")

# Manual calculation for understanding
differences = before_scores - after_scores
abs_differences = np.abs(differences)
ranks = stats.rankdata(abs_differences)
signed_ranks = np.sign(differences) * ranks
W_statistic = signed_ranks.sum()

print("\nManual Wilcoxon Signed-Rank Calculation:")
print(f"W statistic: {W_statistic}")
print(f"SciPy W statistic: {paired_wilcox.statistic}")

# Compare with paired t-test
print("\nComparison of paired tests:")
print(f"Paired t-test p-value: {paired_test.pvalue:.4f}")
print(f"Paired Wilcoxon p-value: {paired_wilcox.pvalue:.4f}")

# Effect size for paired Wilcoxon
paired_wilcox_effect = abs(stats.norm.ppf(paired_wilcox.pvalue / 2)) / np.sqrt(n_subjects)
print(f"Paired Wilcoxon effect size (r): {paired_wilcox_effect:.3f}")
print(f"Effect interpretation: {interpret_wilcox_effect(paired_wilcox_effect)}")

# Power comparison for paired tests
paired_wilcox_power = power_analysis.power(effect_size=paired_wilcox_effect, 
                                          nobs=n_subjects, 
                                          alpha=0.05)

print("\nPaired Tests Power Comparison:")
print(f"Paired t-test power: {paired_power:.3f}")
print(f"Paired Wilcoxon power: {paired_wilcox_power:.3f}")

# Assumption checking for paired data
print("\nAssumption Checking for Paired Data:")
print(f"Normality of differences (Shapiro-Wilk): {shapiro_diff.pvalue:.4f}")
if shapiro_diff.pvalue < 0.05:
    print("RECOMMENDATION: Use Wilcoxon signed-rank test")
else:
    print("RECOMMENDATION: Paired t-test is appropriate")
```

## Assumption Checking

Proper assumption checking is crucial for valid statistical inference in two-sample tests. Violations of assumptions can lead to incorrect conclusions and inflated Type I error rates.

### Key Assumptions for Two-Sample Tests

1. **Independence:** Observations within and between groups are independent
2. **Normality:** Data follows normal distributions (for parametric tests)
3. **Homogeneity of Variance:** Population variances are equal (for pooled t-test)
4. **Random Sampling:** Data represents random samples from populations
5. **No Outliers:** Extreme values don't unduly influence results

### Normality Tests

The normality assumption is critical for parametric tests. Several methods can assess normality for each group.

**Mathematical Tests:**
- **Shapiro-Wilk:** Most powerful test for normality
- **Anderson-Darling:** Good for detecting departures from normality
- **Kolmogorov-Smirnov:** Tests against a specified distribution

**Graphical Methods:**
- **Q-Q plots:** Compare sample quantiles to theoretical normal quantiles
- **Histograms:** Visual assessment of distribution shape
- **Box plots:** Detect skewness and outliers

```python
# Comprehensive normality checking function for two groups
def check_normality_groups(group1, group2, group_names=["Group 1", "Group 2"]):
    print("=== COMPREHENSIVE NORMALITY ASSESSMENT ===")
    
    # Basic information
    n1 = len(group1)
    n2 = len(group2)
    print(f"Sample sizes: {n1} and {n2}\n")
    
    # Shapiro-Wilk tests
    shapiro1 = stats.shapiro(group1)
    shapiro2 = stats.shapiro(group2)
    
    print("Shapiro-Wilk Tests:")
    print(f"{group_names[0]} p-value: {shapiro1.pvalue:.4f}")
    print(f"{group_names[1]} p-value: {shapiro2.pvalue:.4f}\n")
    
    # Anderson-Darling tests
    ad1 = stats.anderson(group1)
    ad2 = stats.anderson(group2)
    
    print("Anderson-Darling Tests:")
    print(f"{group_names[0]} statistic: {ad1.statistic:.3f}")
    print(f"{group_names[1]} statistic: {ad2.statistic:.3f}\n")
    
    # Descriptive statistics
    print("Descriptive Statistics:")
    print(f"{group_names[0]} mean: {group1.mean():.3f} SD: {group1.std():.3f}")
    print(f"{group_names[1]} mean: {group2.mean():.3f} SD: {group2.std():.3f}")
    
    # Skewness and kurtosis
    from scipy.stats import skew, kurtosis
    skewness1 = skew(group1)
    kurtosis1 = kurtosis(group1)
    skewness2 = skew(group2)
    kurtosis2 = kurtosis(group2)
    
    print(f"\nSkewness and Kurtosis:")
    print(f"{group_names[0]} skewness: {skewness1:.3f} kurtosis: {kurtosis1:.3f}")
    print(f"{group_names[1]} skewness: {skewness2:.3f} kurtosis: {kurtosis2:.3f}\n")
    
    # Q-Q plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    stats.probplot(group1, dist="norm", plot=ax1)
    ax1.set_title(f"Q-Q Plot: {group_names[0]}")
    
    stats.probplot(group2, dist="norm", plot=ax2)
    ax2.set_title(f"Q-Q Plot: {group_names[1]}")
    
    plt.tight_layout()
    plt.show()
    
    # Comprehensive recommendations
    print("=== RECOMMENDATIONS ===")
    
    # Sample size considerations
    if n1 >= 30 and n2 >= 30:
        print("✓ Large sample sizes: Central Limit Theorem applies")
        print("✓ Parametric tests are robust to moderate normality violations")
    elif n1 >= 15 and n2 >= 15:
        print("⚠️  Moderate sample sizes: Check normality carefully")
    else:
        print("⚠️  Small sample sizes: Nonparametric tests recommended")
    
    # Normality assessment
    if shapiro1.pvalue >= 0.05 and shapiro2.pvalue >= 0.05:
        print("✓ Both groups appear normally distributed")
        print("✓ Parametric tests are appropriate")
    else:
        print("✗ At least one group is not normally distributed")
        if n1 < 30 or n2 < 30:
            print("✗ Consider nonparametric alternatives")
        else:
            print("⚠️  Parametric tests may still be appropriate due to large sample sizes")
    
    # Skewness assessment
    if abs(skewness1) < 1 and abs(skewness2) < 1:
        print("✓ Both groups have acceptable skewness")
    else:
        print("✗ At least one group is significantly skewed")
    
    # Kurtosis assessment
    if abs(kurtosis1 - 3) < 2 and abs(kurtosis2 - 3) < 2:
        print("✓ Both groups have acceptable kurtosis")
    else:
        print("✗ At least one group has unusual kurtosis")
    
    return {
        'shapiro1': shapiro1,
        'shapiro2': shapiro2,
        'ad1': ad1,
        'ad2': ad2,
        'skewness1': skewness1,
        'skewness2': skewness2,
        'kurtosis1': kurtosis1,
        'kurtosis2': kurtosis2
    }

# Check normality for species groups
normality_results = check_normality_groups(species_0_sepal_length, species_1_sepal_length, 
                                          ["Species 0", "Species 1"])
```

### Homogeneity of Variance

The homogeneity of variance assumption is crucial for choosing between pooled and Welch's t-tests. Violations can lead to incorrect Type I error rates and reduced power.

**Mathematical Foundation:**

**F-test for Equality of Variances:**
```math
F = \frac{s_1^2}{s_2^2} \sim F_{n_1-1, n_2-1}
```

**Levene's Test (More Robust):**
```math
W = \frac{(N-k)}{(k-1)} \cdot \frac{\sum_{i=1}^k n_i(\bar{Z}_i - \bar{Z})^2}{\sum_{i=1}^k \sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_i)^2}
```

where $Z_{ij} = |X_{ij} - \bar{X}_i|$ and $\bar{Z}_i$ is the mean of $Z_{ij}$ for group $i$.

**Variance Ratio Guidelines:**
- Ratio < 2: Variances considered equal
- Ratio 2-4: Moderate difference, use Welch's test
- Ratio > 4: Large difference, use Welch's test

```python
# Comprehensive homogeneity of variance testing function
def check_homogeneity(group1, group2, group_names=["Group 1", "Group 2"]):
    print("=== HOMOGENEITY OF VARIANCE ASSESSMENT ===")
    
    # Basic statistics
    n1 = len(group1)
    n2 = len(group2)
    var1 = group1.var()
    var2 = group2.var()
    ratio = max(var1, var2) / min(var1, var2)
    
    print(f"Sample sizes: {n1} and {n2}")
    print(f"Variances: {var1:.3f} and {var2:.3f}")
    print(f"Variance ratio: {ratio:.3f}\n")
    
    # F-test for equality of variances
    f_stat, f_p_value = stats.levene(group1, group2)
    print("F-test for Equality of Variances:")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"p-value: {f_p_value:.4f}\n")
    
    # Levene's test (more robust)
    levene_stat, levene_p_value = stats.levene(group1, group2)
    print("Levene's Test (Robust):")
    print(f"F-statistic: {levene_stat:.3f}")
    print(f"p-value: {levene_p_value:.4f}\n")
    
    # Brown-Forsythe test (median-based)
    # For Brown-Forsythe, we use median instead of mean
    from scipy.stats import median_abs_deviation
    mad1 = median_abs_deviation(group1)
    mad2 = median_abs_deviation(group2)
    bf_stat = (mad1**2) / (mad2**2)
    bf_p_value = 2 * (1 - stats.f.cdf(bf_stat, n1-1, n2-1))
    
    print("Brown-Forsythe Test (Median-based):")
    print(f"F-statistic: {bf_stat:.3f}")
    print(f"p-value: {bf_p_value:.4f}\n")
    
    # Comprehensive recommendations
    print("=== RECOMMENDATIONS ===")
    
    # Variance ratio assessment
    if ratio < 2:
        print("✓ Variance ratio < 2: Variances appear equal")
    elif ratio < 4:
        print("⚠️  Variance ratio 2-4: Moderate difference in variances")
    else:
        print("✗ Variance ratio > 4: Large difference in variances")
    
    # F-test assessment
    if f_p_value >= 0.05:
        print(f"✓ F-test: Variances appear equal (p >= {0.05})")
    else:
        print(f"✗ F-test: Variances are significantly different (p < {0.05})")
    
    # Levene's test assessment
    if levene_p_value >= 0.05:
        print(f"✓ Levene's test: Variances appear equal (p >= {0.05})")
    else:
        print(f"✗ Levene's test: Variances are significantly different (p < {0.05})")
    
    # Final recommendation
    print("\nFINAL RECOMMENDATION:")
    if f_p_value >= 0.05 and levene_p_value >= 0.05 and ratio < 2:
        print("✓ Use pooled t-test (equal variances)")
        print("✓ Standard t-test with equal_var = True")
    else:
        print("✗ Use Welch's t-test (unequal variances)")
        print("✗ Standard t-test with equal_var = False")
    
    # Impact on results
    t_pooled = stats.ttest_ind(group1, group2, equal_var=True)
    t_welch = stats.ttest_ind(group1, group2, equal_var=False)
    
    print("\nImpact on Results:")
    print(f"Pooled t-test p-value: {t_pooled.pvalue:.4f}")
    print(f"Welch's t-test p-value: {t_welch.pvalue:.4f}")
    
    return {
        'f_test': (f_stat, f_p_value),
        'levene_test': (levene_stat, levene_p_value),
        'bf_test': (bf_stat, bf_p_value),
        'variance_ratio': ratio,
        't_pooled': t_pooled,
        't_welch': t_welch
    }

# Check homogeneity for species groups
homogeneity_results = check_homogeneity(species_0_sepal_length, species_1_sepal_length, 
                                       ["Species 0", "Species 1"])
```

## Power Analysis

Power analysis helps determine the probability of detecting a true effect and the sample size needed for adequate statistical power in two-sample designs. This is crucial for study design and interpreting results.

### Mathematical Foundation

**Power Definition:**
```math
Power = P(\text{Reject } H_0 | H_1 \text{ is true}) = 1 - \beta
```

**For two-sample t-test:**
```math
Power = P\left(|T| > t_{\alpha/2, df} \mid \mu_1 - \mu_2 = \delta\right)
```

where $\delta$ is the true effect size and $t_{\alpha/2, df}$ is the critical t-value.

**Effect Size and Power Relationship:**
- Larger effect sizes require smaller sample sizes for the same power
- Smaller effect sizes require larger sample sizes for adequate power
- Power increases with sample size for fixed effect sizes
- Unequal sample sizes reduce power compared to equal sample sizes

### Power Analysis for Two-Sample t-Test

```python
# Comprehensive power analysis function for two-sample designs
def power_analysis_two_sample(n1, n2, effect_size, alpha=0.05):
    # Calculate power for current sample sizes
    power_result = power_analysis.power(effect_size=effect_size, 
                                       nobs1=n1, 
                                       nobs2=n2, 
                                       alpha=alpha)
    
    # Calculate required sample sizes for different power levels
    sample_size_80 = power_analysis.solve_power(effect_size=effect_size, 
                                               alpha=alpha, 
                                               power=0.8, 
                                               ratio=n2/n1)
    sample_size_90 = power_analysis.solve_power(effect_size=effect_size, 
                                               alpha=alpha, 
                                               power=0.9, 
                                               ratio=n2/n1)
    
    # Calculate power for different effect sizes
    small_effect_power = power_analysis.power(effect_size=0.2, 
                                             nobs1=n1, 
                                             nobs2=n2, 
                                             alpha=alpha)
    medium_effect_power = power_analysis.power(effect_size=0.5, 
                                              nobs1=n1, 
                                              nobs2=n2, 
                                              alpha=alpha)
    large_effect_power = power_analysis.power(effect_size=0.8, 
                                             nobs1=n1, 
                                             nobs2=n2, 
                                             alpha=alpha)
    
    # Calculate power for equal sample sizes
    n_equal = (n1 + n2) / 2
    power_equal = power_analysis.power(effect_size=effect_size, 
                                      nobs=n_equal, 
                                      alpha=alpha)
    
    return {
        'power': power_result,
        'required_n_80': int(np.ceil(sample_size_80)),
        'required_n_90': int(np.ceil(sample_size_90)),
        'effect_size': effect_size,
        'alpha': alpha,
        'small_effect_power': small_effect_power,
        'medium_effect_power': medium_effect_power,
        'large_effect_power': large_effect_power,
        'power_equal': power_equal,
        'n1': n1,
        'n2': n2
    }

# Apply to species comparison
species_power = power_analysis_two_sample(
    len(species_0_sepal_length), 
    len(species_1_sepal_length), 
    species_effect['cohens_d']
)

print("=== POWER ANALYSIS FOR TWO-SAMPLE DESIGN ===")
print(f"Current sample sizes: {species_power['n1']} and {species_power['n2']}")
print(f"Effect size (Cohen's d): {species_power['effect_size']:.3f}")
print(f"Current power: {species_power['power']:.3f}")

print("\nRequired Sample Sizes:")
print(f"For 80% power: {species_power['required_n_80']} per group")
print(f"For 90% power: {species_power['required_n_90']} per group")

print("\nPower for Different Effect Sizes:")
print(f"Small effect (d = 0.2): {species_power['small_effect_power']:.3f}")
print(f"Medium effect (d = 0.5): {species_power['medium_effect_power']:.3f}")
print(f"Large effect (d = 0.8): {species_power['large_effect_power']:.3f}")

print("\nSample Size Efficiency:")
print(f"Current power (unequal n): {species_power['power']:.3f}")
print(f"Power with equal n: {species_power['power_equal']:.3f}")
print(f"Power loss due to unequal n: {species_power['power_equal'] - species_power['power']:.3f}")

# Power curve analysis
effect_sizes = np.arange(0.1, 1.1, 0.1)
power_curve = [power_analysis.power(effect_size=d, 
                                   nobs1=species_power['n1'], 
                                   nobs2=species_power['n2'], 
                                   alpha=0.05) for d in effect_sizes]

print("\nPower Curve (Effect Size vs Power):")
for i, d in enumerate(effect_sizes):
    print(f"d = {d:.1f}: Power = {power_curve[i]:.3f}")

# Sample size recommendations
print("\nSample Size Recommendations:")
if species_power['power'] < 0.8:
    print("⚠️  Current power is below 80%")
    print(f"📈 Consider increasing sample size to {species_power['required_n_80']} per group")
else:
    print("✓ Current power is adequate (≥80%)")

if abs(species_power['n1'] - species_power['n2']) > 0:
    print("⚠️  Unequal sample sizes detected")
    print("📈 Consider equal sample sizes for maximum power")
else:
    print("✓ Equal sample sizes (optimal for power)")
```

## Practical Examples

Real-world applications of two-sample tests demonstrate their importance across various fields. These examples show how to apply the concepts in practice with comprehensive analysis.

### Example 1: Clinical Trial

Clinical trials are a common application where two-sample tests are used to evaluate treatment effectiveness compared to control conditions.

**Scenario:** A pharmaceutical company conducts a clinical trial to test a new medication for reducing blood pressure. Patients are randomly assigned to either the treatment group (new medication) or control group (placebo).

```python
# Simulate clinical trial data
np.random.seed(123)
n_treatment = 25
n_control = 25

# Generate treatment and control group data
treatment_scores = np.random.normal(85, 12, n_treatment)
control_scores = np.random.normal(75, 10, n_control)

# Comprehensive clinical trial analysis
print("=== CLINICAL TRIAL ANALYSIS ===")
print(f"Treatment group n: {n_treatment}")
print(f"Control group n: {n_control}")
print(f"Treatment mean: {treatment_scores.mean():.2f}")
print(f"Control mean: {control_scores.mean():.2f}")
print(f"Treatment SD: {treatment_scores.std():.2f}")
print(f"Control SD: {control_scores.std():.2f}")

# Perform comprehensive t-test analysis
clinical_t_test = stats.ttest_ind(treatment_scores, control_scores)
print(f"\nt-test results: statistic={clinical_t_test.statistic:.3f}, p-value={clinical_t_test.pvalue:.4f}")

# Calculate effect size
clinical_effect = calculate_cohens_d_independent(pd.Series(treatment_scores), pd.Series(control_scores))

print("\nClinical Trial Results:")
print(f"Mean difference (Treatment - Control): {treatment_scores.mean() - control_scores.mean():.2f}")
print(f"Effect size (Cohen's d): {clinical_effect['cohens_d']:.3f}")
print(f"Effect interpretation: {interpret_effect_size(clinical_effect['cohens_d'])}")

# Power analysis
clinical_power = power_analysis_two_sample(n_treatment, n_control, clinical_effect['cohens_d'])
print(f"Power for detecting this effect: {clinical_power['power']:.3f}")

# Clinical significance assessment
mean_diff = treatment_scores.mean() - control_scores.mean()
# Calculate confidence interval
se_diff = np.sqrt(treatment_scores.var()/n_treatment + control_scores.var()/n_control)
t_critical = stats.t.ppf(0.975, clinical_t_test.df)
ci_lower = mean_diff - t_critical * se_diff
ci_upper = mean_diff + t_critical * se_diff
ci_width = ci_upper - ci_lower

print("\nClinical Significance Assessment:")
print(f"95% CI for mean difference: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"CI width: {ci_width:.2f}")

# Clinical interpretation
alpha = 0.05
if clinical_t_test.pvalue < alpha:
    print("\nCLINICAL CONCLUSION:")
    print(f"✓ Treatment shows significant improvement (p < {alpha})")
    print(f"✓ Mean improvement: {mean_diff:.1f} points")
    if clinical_effect['cohens_d'] >= 0.5:
        print("✓ Effect size is clinically meaningful")
    else:
        print("⚠️  Effect size may not be clinically meaningful")
else:
    print("\nCLINICAL CONCLUSION:")
    print(f"✗ No significant treatment effect (p >= {alpha})")
    print("✗ Consider larger sample size or different treatment")

# Number needed to treat (NNT) calculation
# Assuming higher scores are better
improvement_rate_treatment = (treatment_scores > control_scores.mean()).sum() / n_treatment
improvement_rate_control = (control_scores > control_scores.mean()).sum() / n_control
nnt = 1 / (improvement_rate_treatment - improvement_rate_control)

print("\nNumber Needed to Treat (NNT):")
print(f"NNT: {nnt:.1f} patients")
print(f"Interpretation: Treat {int(round(nnt, 0))} patients to see 1 additional benefit")

# Safety analysis (outliers)
treatment_outliers = (np.abs(stats.zscore(treatment_scores)) > 3).sum()
control_outliers = (np.abs(stats.zscore(control_scores)) > 3).sum()

print("\nSafety Analysis:")
print(f"Treatment outliers: {treatment_outliers}")
print(f"Control outliers: {control_outliers}")
```

### Example 2: Educational Research

Educational research frequently uses two-sample tests to evaluate the effectiveness of teaching interventions, comparing experimental and control groups.

**Scenario:** A researcher wants to evaluate the effectiveness of a new teaching method on student performance. Students are randomly assigned to either the experimental group (new method) or control group (traditional method).

```python
# Simulate educational intervention data
np.random.seed(123)
n_experimental = 30
n_control = 30

# Generate pre-test and post-test data
experimental_pre = np.random.normal(70, 15, n_experimental)
experimental_post = experimental_pre + np.random.normal(8, 6, n_experimental)

control_pre = np.random.normal(72, 14, n_control)
control_post = control_pre + np.random.normal(2, 5, n_control)

# Comprehensive educational analysis
print("=== EDUCATIONAL INTERVENTION ANALYSIS ===")
print(f"Experimental group n: {n_experimental}")
print(f"Control group n: {n_control}")

# Pre-test analysis
print("\nPre-test Analysis:")
print(f"Experimental pre-test mean: {experimental_pre.mean():.2f}")
print(f"Control pre-test mean: {control_pre.mean():.2f}")
pre_test_diff = experimental_pre.mean() - control_pre.mean()
print(f"Pre-test difference: {pre_test_diff:.2f}")

# Check for baseline differences
pre_test_t = stats.ttest_ind(experimental_pre, control_pre)
print(f"Pre-test t-test p-value: {pre_test_t.pvalue:.4f}")
if pre_test_t.pvalue < 0.05:
    print("⚠️  Significant baseline differences detected")
else:
    print("✓ No significant baseline differences")

# Analyze gain scores
experimental_gains = experimental_post - experimental_pre
control_gains = control_post - control_pre

print("\nGain Score Analysis:")
print(f"Experimental gain mean: {experimental_gains.mean():.2f}")
print(f"Control gain mean: {control_gains.mean():.2f}")
print(f"Experimental gain SD: {experimental_gains.std():.2f}")
print(f"Control gain SD: {control_gains.std():.2f}")

# Independent samples t-test on gains
gains_test = stats.ttest_ind(experimental_gains, control_gains)
print(f"\nGains t-test: statistic={gains_test.statistic:.3f}, p-value={gains_test.pvalue:.4f}")

# Effect size for gains
gains_effect = calculate_cohens_d_independent(pd.Series(experimental_gains), pd.Series(control_gains))

print("\nEducational Intervention Results:")
print(f"Gain difference (Experimental - Control): {experimental_gains.mean() - control_gains.mean():.2f}")
print(f"Effect size (Cohen's d): {gains_effect['cohens_d']:.3f}")
print(f"Effect interpretation: {interpret_effect_size(gains_effect['cohens_d'])}")

# Power analysis
edu_power = power_analysis_two_sample(n_experimental, n_control, gains_effect['cohens_d'])
print(f"Power for detecting this effect: {edu_power['power']:.3f}")

# Educational significance assessment
gain_diff = experimental_gains.mean() - control_gains.mean()
# Calculate confidence interval for gains
se_gains = np.sqrt(experimental_gains.var()/n_experimental + control_gains.var()/n_control)
t_critical_gains = stats.t.ppf(0.975, gains_test.df)
gain_ci_lower = gain_diff - t_critical_gains * se_gains
gain_ci_upper = gain_diff + t_critical_gains * se_gains

print("\nEducational Significance Assessment:")
print(f"95% CI for gain difference: [{gain_ci_lower:.2f}, {gain_ci_upper:.2f}]")

# Practical significance (minimum important difference)
# Assume 5 points is educationally meaningful
mid = 5
if abs(gain_diff) >= mid:
    print(f"✓ Effect exceeds minimum important difference ({mid} points)")
else:
    print(f"⚠️  Effect below minimum important difference ({mid} points)")

# Educational interpretation
alpha = 0.05
if gains_test.pvalue < alpha:
    print("\nEDUCATIONAL CONCLUSION:")
    print(f"✓ New teaching method shows significant improvement (p < {alpha})")
    print(f"✓ Average improvement: {gain_diff:.1f} points")
    
    # Effect size interpretation for education
    if gains_effect['cohens_d'] >= 0.5:
        print("✓ Large practical effect on learning")
    elif gains_effect['cohens_d'] >= 0.3:
        print("✓ Moderate practical effect on learning")
    else:
        print("⚠️  Small practical effect on learning")
else:
    print("\nEDUCATIONAL CONCLUSION:")
    print(f"✗ No significant improvement with new method (p >= {alpha})")
    print("✗ Consider revising the intervention or increasing sample size")

# Individual student analysis
experimental_improvers = (experimental_gains > 0).sum()
control_improvers = (control_gains > 0).sum()

print("\nIndividual Student Analysis:")
print(f"Experimental students improving: {experimental_improvers}/{n_experimental} ({experimental_improvers/n_experimental*100:.1f}%)")
print(f"Control students improving: {control_improvers}/{n_control} ({control_improvers/n_control*100:.1f}%)")

# Cost-effectiveness analysis
# Assume new method costs $100 more per student
cost_per_student = 100
total_cost = n_experimental * cost_per_student
cost_per_point = total_cost / (gain_diff * n_experimental)

print("\nCost-Effectiveness Analysis:")
print(f"Additional cost per student: ${cost_per_student}")
print(f"Total additional cost: ${total_cost}")
print(f"Cost per point of improvement: ${cost_per_point:.2f}")
```

### Example 3: Quality Control

Quality control applications use two-sample tests to compare production processes, machine performance, and product specifications.

**Scenario:** A manufacturing plant wants to compare the output quality of two production machines to determine if they produce significantly different results.

```python
# Simulate quality control data
np.random.seed(123)
n_machine1 = 20
n_machine2 = 20

# Generate production data
machine1_output = np.random.normal(100, 5, n_machine1)
machine2_output = np.random.normal(98, 6, n_machine2)

# Comprehensive quality control analysis
print("=== QUALITY CONTROL ANALYSIS ===")
print(f"Machine 1 n: {n_machine1}")
print(f"Machine 2 n: {n_machine2}")
print(f"Machine 1 mean: {machine1_output.mean():.2f}")
print(f"Machine 2 mean: {machine2_output.mean():.2f}")
print(f"Machine 1 SD: {machine1_output.std():.2f}")
print(f"Machine 2 SD: {machine2_output.std():.2f}")

# Specification limits (assume 95-105 is acceptable)
spec_lower = 95
spec_upper = 105

print(f"\nSpecification Analysis:")
print(f"Specification limits: {spec_lower} - {spec_upper}")

# Calculate process capability
machine1_capable = ((machine1_output >= spec_lower) & (machine1_output <= spec_upper)).sum()
machine2_capable = ((machine2_output >= spec_lower) & (machine2_output <= spec_upper)).sum()

print(f"Machine 1 within specs: {machine1_capable}/{n_machine1} ({machine1_capable/n_machine1*100:.1f}%)")
print(f"Machine 2 within specs: {machine2_capable}/{n_machine2} ({machine2_capable/n_machine2*100:.1f}%)")

# Perform quality control test
quality_test = stats.ttest_ind(machine1_output, machine2_output)
print(f"\nQuality t-test: statistic={quality_test.statistic:.3f}, p-value={quality_test.pvalue:.4f}")

# Nonparametric alternative
quality_wilcox = stats.mannwhitneyu(machine1_output, machine2_output, alternative='two-sided')
print(f"Quality Wilcoxon: statistic={quality_wilcox.statistic:.3f}, p-value={quality_wilcox.pvalue:.4f}")

# Effect size
quality_effect = calculate_cohens_d_independent(pd.Series(machine1_output), pd.Series(machine2_output))

print("\nQuality Control Results:")
print(f"Mean difference (Machine 1 - Machine 2): {machine1_output.mean() - machine2_output.mean():.2f}")
print(f"Effect size (Cohen's d): {quality_effect['cohens_d']:.3f}")
print(f"Effect interpretation: {interpret_effect_size(quality_effect['cohens_d'])}")

# Compare parametric and nonparametric results
print("\nTest Comparison:")
print(f"t-test p-value: {quality_test.pvalue:.4f}")
print(f"Wilcoxon p-value: {quality_wilcox.pvalue:.4f}")

# Process capability indices
machine1_cpk = min((machine1_output.mean() - spec_lower) / (3 * machine1_output.std()),
                   (spec_upper - machine1_output.mean()) / (3 * machine1_output.std()))
machine2_cpk = min((machine2_output.mean() - spec_lower) / (3 * machine2_output.std()),
                   (spec_upper - machine2_output.mean()) / (3 * machine2_output.std()))

print("\nProcess Capability Analysis:")
print(f"Machine 1 Cpk: {machine1_cpk:.3f}")
print(f"Machine 2 Cpk: {machine2_cpk:.3f}")

# Cpk interpretation
def interpret_cpk(cpk):
    if cpk >= 1.33:
        return "Excellent capability"
    elif cpk >= 1.0:
        return "Adequate capability"
    elif cpk >= 0.67:
        return "Marginal capability"
    else:
        return "Poor capability"

print(f"Machine 1 capability: {interpret_cpk(machine1_cpk)}")
print(f"Machine 2 capability: {interpret_cpk(machine2_cpk)}")

# Quality control interpretation
alpha = 0.05
mean_diff = machine1_output.mean() - machine2_output.mean()

if quality_test.pvalue < alpha:
    print("\nQUALITY CONTROL CONCLUSION:")
    print(f"✓ Machines produce significantly different outputs (p < {alpha})")
    print(f"✓ Mean difference: {mean_diff:.2f} units")
    
    # Practical significance
    if abs(mean_diff) > 2:
        print("⚠️  Difference exceeds practical tolerance (2 units)")
        print("⚠️  Consider machine calibration or maintenance")
    else:
        print("✓ Difference within practical tolerance")
    
    # Which machine is better
    if mean_diff > 0:
        print("✓ Machine 1 produces higher quality output")
    else:
        print("✓ Machine 2 produces higher quality output")
else:
    print("\nQUALITY CONTROL CONCLUSION:")
    print(f"✗ No significant difference between machines (p >= {alpha})")
    print("✓ Both machines perform similarly")

# Economic analysis
# Assume cost of poor quality is $10 per unit outside specs
cost_per_defect = 10
machine1_defects = n_machine1 - machine1_capable
machine2_defects = n_machine2 - machine2_capable

machine1_cost = machine1_defects * cost_per_defect
machine2_cost = machine2_defects * cost_per_defect

print("\nEconomic Analysis:")
print(f"Machine 1 defect cost: ${machine1_cost}")
print(f"Machine 2 defect cost: ${machine2_cost}")
print(f"Cost difference: ${abs(machine1_cost - machine2_cost)}")

# Recommendations
print("\nRECOMMENDATIONS:")
if quality_test.pvalue < alpha:
    if machine1_cpk > machine2_cpk:
        print("✓ Prefer Machine 1 for production")
    else:
        print("✓ Prefer Machine 2 for production")
else:
    print("✓ Both machines are equivalent")
    print("✓ Choose based on other factors (speed, cost, availability)")
```

## Advanced Topics

Advanced techniques extend the basic two-sample tests to handle complex scenarios, provide robust alternatives, and offer deeper insights into the data.

### Bootstrap Confidence Intervals

Bootstrap methods provide nonparametric confidence intervals that don't rely on distributional assumptions. They're particularly useful when data doesn't meet normality assumptions or when sample sizes are small.

**Mathematical Foundation:**

The bootstrap estimates the sampling distribution by resampling with replacement from the observed data:

```math
\hat{F}_n(x) = \frac{1}{n} \sum_{i=1}^n I(X_i \leq x)
```

**Bootstrap Confidence Interval:**
```math
CI_{1-\alpha} = [\hat{\theta}_{\alpha/2}^*, \hat{\theta}_{1-\alpha/2}^*]
```

where $\hat{\theta}_\alpha^*$ is the $\alpha$-th percentile of bootstrap estimates.

```python
# Bootstrap function for mean difference
def boot_mean_diff(data, indices):
    d = data.iloc[indices]
    group1 = d[d['group'] == "Group1"]['value']
    group2 = d[d['group'] == "Group2"]['value']
    return group1.mean() - group2.mean()

# Create data frame for bootstrap
boot_data = pd.DataFrame({
    'value': np.concatenate([species_0_sepal_length, species_1_sepal_length]),
    'group': ["Group1"] * len(species_0_sepal_length) + ["Group2"] * len(species_1_sepal_length)
})

# Comprehensive bootstrap analysis
print("=== BOOTSTRAP ANALYSIS ===")
print(f"Original sample sizes: {len(species_0_sepal_length)} and {len(species_1_sepal_length)}")
print("Bootstrap replications: 1000")

# Manual bootstrap implementation
n_bootstrap = 1000
bootstrap_estimates = []

for i in range(n_bootstrap):
    # Resample with replacement
    indices = np.random.choice(len(boot_data), len(boot_data), replace=True)
    bootstrap_estimate = boot_mean_diff(boot_data, indices)
    bootstrap_estimates.append(bootstrap_estimate)

bootstrap_estimates = np.array(bootstrap_estimates)
original_estimate = species_0_sepal_length.mean() - species_1_sepal_length.mean()

print("\nBootstrap Results:")
print(f"Original mean difference: {original_estimate:.3f}")
print(f"Bootstrap mean: {bootstrap_estimates.mean():.3f}")
print(f"Bootstrap SE: {bootstrap_estimates.std():.3f}")

# Bootstrap confidence intervals
boot_ci_perc_lower = np.percentile(bootstrap_estimates, 2.5)
boot_ci_perc_upper = np.percentile(bootstrap_estimates, 97.5)

print("\nBootstrap Confidence Intervals:")
print(f"Percentile 95% CI: [{boot_ci_perc_lower:.3f}, {boot_ci_perc_upper:.3f}]")

# Compare with parametric CI
t_ci_lower, t_ci_upper = ci_lower, ci_upper
print(f"t-test 95% CI: [{t_ci_lower:.3f}, {t_ci_upper:.3f}]")

# Bootstrap bias and acceleration
bias = bootstrap_estimates.mean() - original_estimate
print("\nBootstrap Diagnostics:")
print(f"Bias: {bias:.4f}")
print(f"Bias-corrected estimate: {original_estimate - bias:.3f}")

# Bootstrap distribution analysis
print("\nBootstrap Distribution:")
print(f"2.5th percentile: {np.percentile(bootstrap_estimates, 2.5):.3f}")
print(f"50th percentile (median): {np.percentile(bootstrap_estimates, 50):.3f}")
print(f"97.5th percentile: {np.percentile(bootstrap_estimates, 97.5):.3f}")

# Compare CI widths
t_ci_width = t_ci_upper - t_ci_lower
boot_ci_width = boot_ci_perc_upper - boot_ci_perc_lower

print("\nConfidence Interval Comparison:")
print(f"t-test CI width: {t_ci_width:.3f}")
print(f"Bootstrap CI width: {boot_ci_width:.3f}")
print(f"Width ratio (Bootstrap/t-test): {boot_ci_width/t_ci_width:.3f}")

# Bootstrap histogram
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_estimates, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
plt.axvline(original_estimate, color='red', linewidth=2, label='Original estimate')
plt.axvline(boot_ci_perc_lower, color='blue', linestyle='--', label='95% CI lower')
plt.axvline(boot_ci_perc_upper, color='blue', linestyle='--', label='95% CI upper')
plt.xlabel('Mean Difference')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution of Mean Difference')
plt.legend()
plt.show()
```

### Robust Two-Sample Tests

Robust methods provide alternatives to traditional t-tests that are less sensitive to outliers and violations of normality assumptions. These methods use trimmed means, winsorized data, or other robust estimators.

**Mathematical Foundation:**

**Yuen's t-test (Trimmed Means):**
```math
t_Y = \frac{\bar{X}_{t1} - \bar{X}_{t2}}{\sqrt{\frac{s_{w1}^2}{h_1} + \frac{s_{w2}^2}{h_2}}}
```

where:
- $\bar{X}_{ti}$ = trimmed mean for group $i$
- $s_{wi}^2$ = winsorized variance for group $i$
- $h_i = n_i - 2g_i$ = effective sample size after trimming

**Winsorized Variance:**
```math
s_w^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X}_t)^2
```

**M-estimators:**
```math
\psi(x) = \begin{cases}
x & \text{if } |x| \leq c \\
c \cdot \text{sign}(x) & \text{if } |x| > c
\end{cases}
```

```python
# Comprehensive robust testing function
def robust_two_sample_test(group1, group2, group_names=["Group 1", "Group 2"]):
    print("=== ROBUST TWO-SAMPLE ANALYSIS ===")
    
    # Basic statistics
    n1 = len(group1)
    n2 = len(group2)
    
    print(f"Sample sizes: {n1} and {n2}")
    print(f"Original means: {group1.mean():.3f} and {group2.mean():.3f}")
    print(f"Original SDs: {group1.std():.3f} and {group2.std():.3f}")
    
    # Yuen's t-test for trimmed means
    def yuen_test(group1, group2, trim=0.1):
        # Trim the data
        n1 = len(group1)
        n2 = len(group2)
        k1 = int(n1 * trim)
        k2 = int(n2 * trim)
        
        # Sort and trim
        sorted1 = np.sort(group1)
        sorted2 = np.sort(group2)
        
        trimmed1 = sorted1[k1:(n1 - k1)]
        trimmed2 = sorted2[k2:(n2 - k2)]
        
        # Calculate trimmed statistics
        mean1 = trimmed1.mean()
        mean2 = trimmed2.mean()
        var1 = trimmed1.var()
        var2 = trimmed2.var()
        
        # Calculate test statistic
        se = np.sqrt(var1 / (n1 - 2 * k1) + var2 / (n2 - 2 * k2))
        t_stat = (mean1 - mean2) / se
        
        # Degrees of freedom
        df = (var1 / (n1 - 2 * k1) + var2 / (n2 - 2 * k2))**2 / \
             ((var1 / (n1 - 2 * k1))**2 / (n1 - 2 * k1 - 1) + 
              (var2 / (n2 - 2 * k2))**2 / (n2 - 2 * k2 - 1))
        
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'df': df,
            'trimmed_means': [mean1, mean2],
            'trimmed_sds': [np.sqrt(var1), np.sqrt(var2)],
            'trim_proportion': trim
        }
    
    # Apply Yuen's test with different trim levels
    yuen_10 = yuen_test(group1, group2, trim=0.1)
    yuen_20 = yuen_test(group1, group2, trim=0.2)
    
    print("\nYuen's t-test Results:")
    print(f"10% trimmed means: {yuen_10['trimmed_means']}")
    print(f"20% trimmed means: {yuen_20['trimmed_means']}")
    
    print("\n10% Trim Results:")
    print(f"t-statistic: {yuen_10['t_statistic']:.3f}")
    print(f"p-value: {yuen_10['p_value']:.4f}")
    print(f"df: {yuen_10['df']:.1f}")
    
    print("\n20% Trim Results:")
    print(f"t-statistic: {yuen_20['t_statistic']:.3f}")
    print(f"p-value: {yuen_20['p_value']:.4f}")
    print(f"df: {yuen_20['df']:.1f}")
    
    # Compare with standard t-test
    standard_t = stats.ttest_ind(group1, group2)
    
    print("\nComparison with Standard t-test:")
    print(f"Standard t-test p-value: {standard_t.pvalue:.4f}")
    print(f"Yuen 10% p-value: {yuen_10['p_value']:.4f}")
    print(f"Yuen 20% p-value: {yuen_20['p_value']:.4f}")
    
    # Outlier detection
    outliers1 = (np.abs(stats.zscore(group1)) > 2).sum()
    outliers2 = (np.abs(stats.zscore(group2)) > 2).sum()
    
    print("\nOutlier Analysis:")
    print(f"{group_names[0]} outliers (|z| > 2): {outliers1}")
    print(f"{group_names[1]} outliers (|z| > 2): {outliers2}")
    
    # Robust effect size
    robust_effect_10 = (yuen_10['trimmed_means'][0] - yuen_10['trimmed_means'][1]) / \
                       np.sqrt((yuen_10['trimmed_sds'][0]**2 + yuen_10['trimmed_sds'][1]**2) / 2)
    
    print(f"\nRobust Effect Size (10% trim): {robust_effect_10:.3f}")
    print(f"Effect interpretation: {interpret_effect_size(robust_effect_10)}")
    
    # Recommendations
    print("\nRobust Analysis Recommendations:")
    if outliers1 > 0 or outliers2 > 0:
        print("⚠️  Outliers detected - robust methods recommended")
        if abs(yuen_10['p_value'] - standard_t.pvalue) > 0.01:
            print("⚠️  Results differ from standard t-test")
            print("✓ Use robust methods for inference")
        else:
            print("✓ Results consistent with standard t-test")
    else:
        print("✓ No outliers detected")
        print("✓ Standard t-test is appropriate")
    
    return {
        'yuen_10': yuen_10,
        'yuen_20': yuen_20,
        'standard_t': standard_t,
        'outliers1': outliers1,
        'outliers2': outliers2,
        'robust_effect': robust_effect_10
    }

# Apply robust analysis
robust_results = robust_two_sample_test(species_0_sepal_length, species_1_sepal_length, 
                                       ["Species 0", "Species 1"])
```

## Best Practices

Following best practices ensures valid statistical inference and meaningful interpretation of two-sample test results. These guidelines help avoid common pitfalls and improve the quality of statistical analysis.

### Test Selection Guidelines

Proper test selection is crucial for valid statistical inference. The choice depends on data characteristics, sample sizes, and research design.

**Decision Tree for Two-Sample Tests:**

1. **Independent vs Paired:** Are observations related?
2. **Sample Size:** Large ($n \geq 30$) vs Small ($n < 30$)
3. **Normality:** Check distribution assumptions
4. **Variance Equality:** For independent samples
5. **Outliers:** Assess influence of extreme values

```python
# Comprehensive test selection function
def choose_two_sample_test(group1, group2, paired=False, alpha=0.05):
    print("=== COMPREHENSIVE TWO-SAMPLE TEST SELECTION ===")
    
    n1 = len(group1)
    n2 = len(group2)
    
    print(f"Sample sizes: {n1} and {n2}")
    print(f"Design: {'Paired samples' if paired else 'Independent samples'}\n")
    
    if paired:
        print("PAIRED SAMPLES ANALYSIS:")
        
        # Check normality of differences
        differences = group1 - group2
        shapiro_diff = stats.shapiro(differences)
        print(f"Normality of differences p-value: {shapiro_diff.pvalue:.4f}")
        
        # Sample size considerations
        if n1 >= 30:
            print("✓ Large sample size: Central Limit Theorem applies")
            print("✓ Parametric tests are robust")
        else:
            print("⚠️  Small sample size: Check normality carefully")
        
        # Test recommendation
        if shapiro_diff.pvalue >= alpha or n1 >= 30:
            print("RECOMMENDATION: Paired t-test")
            print("REASON: Normal differences or large sample size")
        else:
            print("RECOMMENDATION: Wilcoxon signed-rank test")
            print("REASON: Non-normal differences in small sample")
        
    else:
        print("INDEPENDENT SAMPLES ANALYSIS:")
        
        # Check normality
        shapiro1 = stats.shapiro(group1)
        shapiro2 = stats.shapiro(group2)
        print(f"Group 1 normality p-value: {shapiro1.pvalue:.4f}")
        print(f"Group 2 normality p-value: {shapiro2.pvalue:.4f}")
        
        # Check homogeneity of variance
        var_stat, var_p_value = stats.levene(group1, group2)
        print(f"Homogeneity of variance p-value: {var_p_value:.4f}")
        
        # Sample size considerations
        if n1 >= 30 and n2 >= 30:
            print("✓ Large sample sizes: Central Limit Theorem applies")
            print("✓ Parametric tests are robust to moderate violations")
        elif n1 >= 15 and n2 >= 15:
            print("⚠️  Moderate sample sizes: Check assumptions carefully")
        else:
            print("⚠️  Small sample sizes: Nonparametric tests recommended")
        
        # Test recommendation logic
        print("\nTEST SELECTION LOGIC:")
        
        # Normality assessment
        if shapiro1.pvalue >= alpha and shapiro2.pvalue >= alpha:
            print("✓ Both groups appear normally distributed")
            normality_ok = True
        elif n1 >= 30 and n2 >= 30:
            print("✓ Large sample sizes compensate for non-normality")
            normality_ok = True
        else:
            print("✗ Non-normal distributions in small samples")
            normality_ok = False
        
        # Variance assessment
        if var_p_value >= alpha:
            print("✓ Variances appear equal")
            variance_ok = True
        else:
            print("✗ Variances are significantly different")
            variance_ok = False
        
        # Final recommendation
        print("\nFINAL RECOMMENDATION:")
        if normality_ok:
            if variance_ok:
                print("✓ Standard t-test (equal variances)")
                print("REASON: Normal distributions and equal variances")
            else:
                print("✓ Welch's t-test (unequal variances)")
                print("REASON: Normal distributions but unequal variances")
        else:
            print("✓ Mann-Whitney U test")
            print("REASON: Non-normal distributions or small samples")
    
    # Effect size calculation
    if paired:
        effect_size = abs(differences.mean()) / differences.std()
    else:
        effect_size = calculate_cohens_d_independent(group1, group2)['cohens_d']
    
    print("\nEFFECT SIZE ANALYSIS:")
    print(f"Cohen's d: {effect_size:.3f}")
    print(f"Interpretation: {interpret_effect_size(effect_size)}")
    
    # Power considerations
    if paired:
        power_est = power_analysis.power(effect_size=effect_size, 
                                        nobs=n1, 
                                        alpha=alpha)
    else:
        power_est = power_analysis.power(effect_size=effect_size, 
                                        nobs1=n1, 
                                        nobs2=n2, 
                                        alpha=alpha)
    
    print(f"Estimated power: {power_est:.3f}")
    
    if power_est < 0.8:
        print("⚠️  Power below 80% - consider larger sample size")
    else:
        print("✓ Adequate power for detecting effects")
    
    return {
        'paired': paired,
        'n1': n1,
        'n2': n2,
        'effect_size': effect_size,
        'power': power_est,
        'normality_ok': shapiro_diff.pvalue >= alpha if paired else (shapiro1.pvalue >= alpha and shapiro2.pvalue >= alpha),
        'variance_ok': None if paired else var_p_value >= alpha
    }

# Apply comprehensive test selection
test_selection = choose_two_sample_test(species_0_sepal_length, species_1_sepal_length, paired=False)
```

### Reporting Guidelines

Proper reporting of two-sample test results is essential for transparency, reproducibility, and scientific communication. Following standardized reporting guidelines ensures that results are clear, complete, and interpretable.

**Essential Elements for Reporting:**

1. **Study Design:** Independent vs paired samples
2. **Descriptive Statistics:** Means, standard deviations, sample sizes
3. **Test Results:** Test statistic, degrees of freedom, p-value
4. **Effect Size:** Cohen's d and interpretation
5. **Confidence Intervals:** Precision of estimates
6. **Assumption Checks:** Normality, homogeneity of variance
7. **Practical Significance:** Clinical or practical relevance

```python
# Comprehensive reporting function for two-sample tests
def generate_two_sample_report(test_result, group1, group2, test_type="t-test", 
                              paired=False, alpha=0.05):
    print("=== COMPREHENSIVE TWO-SAMPLE TEST REPORT ===\n")
    
    # Basic information
    n1 = len(group1)
    n2 = len(group2)
    mean1 = group1.mean()
    mean2 = group2.mean()
    sd1 = group1.std()
    sd2 = group2.std()
    
    print("STUDY DESIGN:")
    print(f"Test type: {test_type}")
    print(f"Design: {'Paired samples' if paired else 'Independent samples'}")
    print(f"Group 1 sample size: {n1}")
    print(f"Group 2 sample size: {n2}")
    print(f"Total sample size: {n1 + n2}")
    print(f"Significance level (α): {alpha}\n")
    
    print("DESCRIPTIVE STATISTICS:")
    print(f"Group 1: M = {mean1:.2f}, SD = {sd1:.2f}")
    print(f"Group 2: M = {mean2:.2f}, SD = {sd2:.2f}")
    
    if paired:
        differences = group1 - group2
        mean_diff = differences.mean()
        sd_diff = differences.std()
        print(f"Differences: M = {mean_diff:.2f}, SD = {sd_diff:.2f}")
    else:
        mean_diff = mean1 - mean2
        print(f"Mean difference (Group 1 - Group 2): {mean_diff:.2f}")
    print()
    
    # Test results
    if test_type == "t-test":
        print("T-TEST RESULTS:")
        print(f"t({test_result.df:.1f}) = {test_result.statistic:.3f}")
        print(f"p-value: {test_result.pvalue:.4f}")
        
        # Calculate confidence interval
        se_diff = np.sqrt(group1.var()/n1 + group2.var()/n2)
        t_critical = stats.t.ppf(0.975, test_result.df)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        # Effect size
        if paired:
            effect_size = abs(mean_diff) / sd_diff
        else:
            effect_size = calculate_cohens_d_independent(group1, group2)['cohens_d']
        
        print(f"Cohen's d = {effect_size:.3f}")
        print(f"Effect size interpretation: {interpret_effect_size(effect_size)}\n")
        
    elif test_type == "wilcoxon":
        print("WILCOXON TEST RESULTS:")
        print(f"W = {test_result.statistic}")
        print(f"p-value: {test_result.pvalue:.4f}")
        
        # Effect size for nonparametric test
        wilcox_effect = abs(stats.norm.ppf(test_result.pvalue / 2)) / np.sqrt(n1 + n2)
        print(f"Effect size (r) = {wilcox_effect:.3f}")
        print(f"Effect size interpretation: {interpret_wilcox_effect(wilcox_effect)}\n")
    
    # Assumption checks
    print("ASSUMPTION CHECKS:")
    
    if paired:
        differences = group1 - group2
        shapiro_diff = stats.shapiro(differences)
        print(f"Normality of differences: Shapiro-Wilk p = {shapiro_diff.pvalue:.4f}")
        if shapiro_diff.pvalue < alpha:
            print("⚠️  Differences are not normally distributed")
        else:
            print("✓ Differences appear normally distributed")
    else:
        shapiro1 = stats.shapiro(group1)
        shapiro2 = stats.shapiro(group2)
        print(f"Group 1 normality: Shapiro-Wilk p = {shapiro1.pvalue:.4f}")
        print(f"Group 2 normality: Shapiro-Wilk p = {shapiro2.pvalue:.4f}")
        
        var_stat, var_p_value = stats.levene(group1, group2)
        print(f"Homogeneity of variance: F-test p = {var_p_value:.4f}")
        
        if shapiro1.pvalue < alpha or shapiro2.pvalue < alpha:
            print("⚠️  At least one group is not normally distributed")
        else:
            print("✓ Both groups appear normally distributed")
        
        if var_p_value < alpha:
            print("⚠️  Variances are significantly different")
        else:
            print("✓ Variances appear equal")
    print()
    
    # Power analysis
    if test_type == "t-test":
        if paired:
            power_est = power_analysis.power(effect_size=effect_size, 
                                            nobs=n1, 
                                            alpha=alpha)
        else:
            power_est = power_analysis.power(effect_size=effect_size, 
                                            nobs1=n1, 
                                            nobs2=n2, 
                                            alpha=alpha)
        print("POWER ANALYSIS:")
        print(f"Estimated power = {power_est:.3f}")
        if power_est < 0.8:
            print("⚠️  Power below 80% - results should be interpreted cautiously")
        else:
            print("✓ Adequate power for detecting effects")
        print()
    
    # Statistical conclusion
    print("STATISTICAL CONCLUSION:")
    if test_result.pvalue < alpha:
        print(f"✓ Reject the null hypothesis (p < {alpha})")
        print("✓ There is significant evidence of a difference between groups")
    else:
        print(f"✗ Fail to reject the null hypothesis (p >= {alpha})")
        print("✗ There is insufficient evidence of a difference between groups")
    print()
    
    # Practical significance
    print("PRACTICAL SIGNIFICANCE:")
    if test_type == "t-test":
        if abs(effect_size) >= 0.8:
            print("✓ Large practical effect")
        elif abs(effect_size) >= 0.5:
            print("✓ Medium practical effect")
        elif abs(effect_size) >= 0.2:
            print("✓ Small practical effect")
        else:
            print("⚠️  Very small practical effect")
    
    # APA style reporting
    print("\nAPA STYLE REPORTING:")
    if test_type == "t-test":
        if test_result.pvalue < 0.001:
            p_report = "p < .001"
        else:
            p_report = f"p = {test_result.pvalue:.3f}"
        
        design_type = "paired" if paired else "independent"
        condition_type = "the two conditions" if paired else "the two groups"
        significant_text = "" if test_result.pvalue < alpha else "no "
        
        print(f"A {design_type} samples t-test was conducted to compare {condition_type}.")
        print(f"There was {significant_text}significant difference between groups, "
              f"t({test_result.df:.1f}) = {test_result.statistic:.3f}, {p_report}, "
              f"d = {effect_size:.3f}.")
    
    return {
        'test_type': test_type,
        'paired': paired,
        'n1': n1,
        'n2': n2,
        'mean1': mean1,
        'mean2': mean2,
        'sd1': sd1,
        'sd2': sd2,
        'test_statistic': test_result.statistic,
        'p_value': test_result.pvalue,
        'effect_size': effect_size if test_type == "t-test" else wilcox_effect,
        'significant': test_result.pvalue < alpha
    }

# Generate comprehensive report for species comparison
species_t_test = stats.ttest_ind(species_0_sepal_length, species_1_sepal_length)
species_report = generate_two_sample_report(species_t_test, species_0_sepal_length, species_1_sepal_length, 
                                           "t-test", paired=False)
```

## Exercises

These exercises provide hands-on practice with two-sample tests, helping you develop proficiency in statistical analysis and interpretation.

### Exercise 1: Independent Samples t-Test

**Objective:** Compare the sepal length of different iris species using independent samples t-tests.

**Data:** Use the iris dataset from sklearn.

**Tasks:**
1. Create two groups: species 0 vs species 1
2. Perform descriptive statistics for both groups
3. Check assumptions (normality, homogeneity of variance)
4. Conduct independent samples t-test
5. Calculate and interpret effect size
6. Generate a comprehensive report

**Hints:**
- Use `iris.target == 0` and `iris.target == 1` to create groups
- Remember to handle unequal sample sizes
- Consider both pooled and Welch's t-tests

**Expected Learning Outcomes:**
- Understanding when to use pooled vs Welch's t-test
- Interpreting effect sizes in context
- Recognizing the importance of assumption checking

### Exercise 2: Paired Samples Analysis

**Objective:** Create a paired dataset and perform both parametric and nonparametric paired tests.

**Scenario:** Simulate before-and-after data for a weight loss program.

**Tasks:**
1. Generate paired data (before and after weights)
2. Calculate and analyze differences
3. Check normality of differences
4. Perform paired t-test and Wilcoxon signed-rank test
5. Compare results and interpret differences
6. Calculate paired effect size

**Hints:**
- Use `np.random.normal()` to generate realistic weight data
- Ensure positive correlation between before and after scores
- Consider the power advantage of paired designs

**Expected Learning Outcomes:**
- Understanding the power advantage of paired designs
- Recognizing when paired vs independent tests are appropriate
- Interpreting correlation in paired data

### Exercise 3: Effect Size Analysis

**Objective:** Calculate and interpret effect sizes for various two-sample comparisons.

**Data:** Use multiple variables from the iris dataset.

**Tasks:**
1. Compare multiple variables between species
2. Calculate Cohen's d for each comparison
3. Create a table of effect sizes and interpretations
4. Identify which comparisons show the largest effects
5. Discuss practical significance vs statistical significance

**Variables to Compare:**
- Sepal length
- Sepal width
- Petal length
- Petal width

**Hints:**
- Use a loop or list comprehension for efficiency
- Consider creating a function for effect size calculation
- Think about which effects are most practically meaningful

**Expected Learning Outcomes:**
- Understanding effect size interpretation
- Recognizing the difference between statistical and practical significance
- Developing intuition for meaningful effect sizes

### Exercise 4: Assumption Checking

**Objective:** Perform comprehensive assumption checking for two-sample tests.

**Data:** Create datasets with various characteristics (normal, skewed, with outliers).

**Tasks:**
1. Generate three datasets:
   - Normal distributions with equal variances
   - Normal distributions with unequal variances
   - Non-normal distributions with outliers
2. For each dataset, perform comprehensive assumption checking
3. Recommend appropriate tests based on findings
4. Compare results across different test approaches
5. Discuss the robustness of different methods

**Hints:**
- Use `np.random.normal()`, `np.random.gamma()`, and concatenate with outliers
- Test normality with multiple methods
- Consider sample size effects on assumption violations

**Expected Learning Outcomes:**
- Understanding when assumption violations matter
- Recognizing the robustness of different tests
- Developing judgment for test selection

### Exercise 5: Power Analysis

**Objective:** Conduct power analysis for two-sample designs.

**Scenario:** Design a study to compare two teaching methods.

**Tasks:**
1. Determine required sample sizes for different effect sizes (0.2, 0.5, 0.8)
2. Calculate power for different sample sizes (20, 50, 100 per group)
3. Create power curves
4. Consider cost-benefit analysis of sample sizes
5. Make recommendations for study design

**Hints:**
- Use the `statsmodels.stats.power` module
- Consider practical constraints (time, cost, availability)
- Think about minimum important differences

**Expected Learning Outcomes:**
- Understanding the relationship between effect size, sample size, and power
- Developing intuition for study design
- Making informed decisions about sample size

### Exercise 6: Real-World Application

**Objective:** Apply two-sample tests to a real-world scenario.

**Scenario:** Analyze customer satisfaction data for two different service providers.

**Tasks:**
1. Create realistic customer satisfaction data
2. Perform comprehensive analysis including:
   - Descriptive statistics
   - Assumption checking
   - Appropriate statistical tests
   - Effect size calculation
   - Practical interpretation
3. Write a professional report
4. Make business recommendations

**Hints:**
- Consider the business context
- Think about practical significance
- Include confidence intervals
- Consider multiple outcome measures

**Expected Learning Outcomes:**
- Applying statistical concepts to real problems
- Communicating results to non-statisticians
- Making data-driven recommendations

### Exercise 7: Advanced Topics

**Objective:** Explore advanced two-sample testing methods.

**Tasks:**
1. Implement bootstrap confidence intervals
2. Perform Yuen's t-test for trimmed means
3. Compare results with standard methods
4. Discuss when advanced methods are beneficial
5. Analyze the trade-offs between methods

**Hints:**
- Use manual bootstrap implementation or `scipy.stats.bootstrap`
- Consider different trim levels
- Compare computational efficiency

**Expected Learning Outcomes:**
- Understanding when advanced methods are needed
- Recognizing the limitations of standard methods
- Developing expertise in robust statistics

### Solutions and Additional Resources

**For each exercise:**
- Start with small datasets to verify your approach
- Use the functions developed in this chapter
- Check your results with built-in Python functions
- Consider multiple approaches to the same problem

**Common Mistakes to Avoid:**
- Using independent tests for paired data
- Ignoring assumption violations in small samples
- Focusing only on p-values without effect sizes
- Not considering practical significance

**Next Steps:**
- Practice with your own datasets
- Explore related topics (ANOVA, regression)
- Learn about multiple comparison corrections
- Study experimental design principles

## Next Steps

In the next chapter, we'll learn about one-way ANOVA for comparing means across multiple groups.

---

**Key Takeaways:**
- Two-sample tests compare means or distributions between two groups
- Independent samples tests are for unrelated groups
- Paired samples tests are for related observations
- Always check assumptions before choosing a test
- Effect sizes provide important information about practical significance
- Nonparametric alternatives exist for non-normal data
- Power analysis helps determine appropriate sample sizes 