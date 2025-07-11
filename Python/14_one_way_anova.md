# One-Way ANOVA

## Overview

One-way Analysis of Variance (ANOVA) is a powerful statistical technique used to compare means across three or more groups simultaneously. It extends the two-sample t-test to multiple groups and is fundamental to experimental design and observational studies.

### When to Use One-Way ANOVA

One-way ANOVA is appropriate when you want to:
- Compare means across three or more independent groups
- Test the null hypothesis that all group means are equal
- Determine if there are significant differences between any groups
- Analyze experimental data with one categorical independent variable
- Compare multiple treatments, conditions, or categories

### Key Concepts

**Null Hypothesis (H₀):** All population means are equal ($\mu_1 = \mu_2 = \mu_3 = ... = \mu_k$)
**Alternative Hypothesis (H₁):** At least one population mean differs from the others

The test determines whether observed differences between group means are statistically significant or due to random sampling variation.

### Mathematical Foundation

One-way ANOVA is based on partitioning the total variance in the data into two components:

**Total Sum of Squares (SST):**
```math
SST = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_{..})^2
```

**Between-Groups Sum of Squares (SSB):**
```math
SSB = \sum_{i=1}^{k} n_i(\bar{X}_{i.} - \bar{X}_{..})^2
```

**Within-Groups Sum of Squares (SSW):**
```math
SSW = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_{i.})^2
```

where:
- $X_{ij}$ = observation $j$ in group $i$
- $\bar{X}_{i.}$ = mean of group $i$
- $\bar{X}_{..}$ = overall mean
- $n_i$ = sample size of group $i$
- $k$ = number of groups

**F-Statistic:**
```math
F = \frac{MSB}{MSW} = \frac{SSB/(k-1)}{SSW/(N-k)}
```

where $N = \sum_{i=1}^{k} n_i$ is the total sample size.

The F-statistic follows an F-distribution with $(k-1, N-k)$ degrees of freedom under the null hypothesis.

## Basic One-Way ANOVA

The basic one-way ANOVA procedure involves calculating the F-statistic by comparing the variance between groups to the variance within groups. Understanding the manual calculation helps clarify the underlying principles.

### Mathematical Foundation

**Variance Partitioning:**
The total variance in the data is partitioned into two components:
1. **Between-groups variance:** Measures differences between group means
2. **Within-groups variance:** Measures variability within each group

**Expected Values Under Null Hypothesis:**
- $E(MSB) = \sigma^2 + \frac{\sum_{i=1}^{k} n_i(\mu_i - \bar{\mu})^2}{k-1}$
- $E(MSW) = \sigma^2$

where $\bar{\mu} = \frac{\sum_{i=1}^{k} n_i\mu_i}{N}$ is the weighted grand mean.

Under the null hypothesis ($\mu_1 = \mu_2 = ... = \mu_k$), both expected values equal $\sigma^2$, so $E(F) = 1$.

**F-Distribution Properties:**
- $F \sim F(k-1, N-k)$ under $H_0$
- $F \geq 0$ (always positive)
- Reject $H_0$ if $F > F_{\alpha, k-1, N-k}$

### Manual Calculation

```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f
import matplotlib.pyplot as plt
import seaborn as sns

# Load sample data (using seaborn's built-in dataset)
import seaborn as sns
mtcars = sns.load_dataset('mpg')

# Create groups based on number of cylinders
mtcars['cyl_factor'] = pd.Categorical(mtcars['cylinders'], 
                                     categories=[4, 6, 8], 
                                     ordered=True)
mtcars['cyl_factor'] = mtcars['cyl_factor'].map({4: '4-cylinder', 6: '6-cylinder', 8: '8-cylinder'})

# Extract MPG data for each cylinder group
mpg_4cyl = mtcars[mtcars['cylinders'] == 4]['mpg'].values
mpg_6cyl = mtcars[mtcars['cylinders'] == 6]['mpg'].values
mpg_8cyl = mtcars[mtcars['cylinders'] == 8]['mpg'].values

# Comprehensive manual ANOVA calculation
def manual_anova(group_data):
    """
    Perform manual one-way ANOVA calculation
    
    Parameters:
    group_data: list of arrays, each containing data for one group
    
    Returns:
    dict: ANOVA results
    """
    # Basic information
    k = len(group_data)
    n_per_group = [len(group) for group in group_data]
    total_n = sum(n_per_group)
    
    print("=== MANUAL ANOVA CALCULATION ===")
    print(f"Number of groups (k): {k}")
    print(f"Sample sizes per group: {n_per_group}")
    print(f"Total sample size (N): {total_n}\n")
    
    # Calculate group means and overall mean
    group_means = [np.mean(group) for group in group_data]
    overall_mean = np.mean(np.concatenate(group_data))
    
    print(f"Group means: {[round(m, 3) for m in group_means]}")
    print(f"Overall mean: {round(overall_mean, 3)}\n")
    
    # Calculate Sum of Squares
    # Between-groups SS
    ss_between = sum(n_per_group[i] * (group_means[i] - overall_mean)**2 for i in range(k))
    
    # Within-groups SS
    ss_within = sum(sum((group_data[i] - group_means[i])**2) for i in range(k))
    
    # Total SS
    ss_total = ss_between + ss_within
    
    print("Sum of Squares:")
    print(f"Between-groups SS: {round(ss_between, 3)}")
    print(f"Within-groups SS: {round(ss_within, 3)}")
    print(f"Total SS: {round(ss_total, 3)}")
    print(f"Verification (SSB + SSW = SST): {round(ss_between + ss_within, 3)} = {round(ss_total, 3)}\n")
    
    # Degrees of freedom
    df_between = k - 1
    df_within = total_n - k
    df_total = total_n - 1
    
    print("Degrees of Freedom:")
    print(f"Between-groups df: {df_between}")
    print(f"Within-groups df: {df_within}")
    print(f"Total df: {df_total}")
    print(f"Verification (dfB + dfW = dfT): {df_between + df_within} = {df_total}\n")
    
    # Mean Squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    print("Mean Squares:")
    print(f"Between-groups MS: {round(ms_between, 3)}")
    print(f"Within-groups MS: {round(ms_within, 3)}\n")
    
    # F-statistic
    f_statistic = ms_between / ms_within
    
    # p-value
    p_value = 1 - f.cdf(f_statistic, df_between, df_within)
    
    # Critical F-value
    f_critical = f.ppf(0.95, df_between, df_within)
    
    print("F-Test Results:")
    print(f"F-statistic: {round(f_statistic, 3)}")
    print(f"Critical F-value (α = 0.05): {round(f_critical, 3)}")
    print(f"p-value: {round(p_value, 4)}")
    print(f"Significant: {p_value < 0.05}\n")
    
    # Effect size (eta-squared)
    eta_squared = ss_between / ss_total
    
    # Partial eta-squared (same as eta-squared for one-way ANOVA)
    partial_eta_squared = eta_squared
    
    # Omega-squared (unbiased estimator)
    omega_squared = (ss_between - (df_between * ms_within)) / (ss_total + ms_within)
    
    print("Effect Sizes:")
    print(f"Eta-squared: {round(eta_squared, 3)}")
    print(f"Partial eta-squared: {round(partial_eta_squared, 3)}")
    print(f"Omega-squared: {round(omega_squared, 3)}\n")
    
    # ANOVA table
    print("ANOVA Table:")
    print("Source\t\tSS\t\t\tDF\tMS\t\t\tF\t\tp-value")
    print(f"Between\t{round(ss_between, 3)}\t\t{df_between}\t{round(ms_between, 3)}\t\t{round(f_statistic, 3)}\t\t{round(p_value, 4)}")
    print(f"Within\t\t{round(ss_within, 3)}\t\t{df_within}\t{round(ms_within, 3)}")
    print(f"Total\t\t{round(ss_total, 3)}\t\t{df_total}\n")
    
    return {
        'ss_between': ss_between,
        'ss_within': ss_within,
        'ss_total': ss_total,
        'df_between': df_between,
        'df_within': df_within,
        'df_total': df_total,
        'ms_between': ms_between,
        'ms_within': ms_within,
        'f_statistic': f_statistic,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'partial_eta_squared': partial_eta_squared,
        'omega_squared': omega_squared,
        'group_means': group_means,
        'overall_mean': overall_mean,
        'f_critical': f_critical
    }

# Apply manual calculation
anova_result = manual_anova([mpg_4cyl, mpg_6cyl, mpg_8cyl])

# Verify with Python's built-in function
print("=== VERIFICATION WITH PYTHON'S BUILT-IN FUNCTION ===")
```

### Using Python's Built-in ANOVA

Python's statistical libraries provide efficient and comprehensive ANOVA analysis. The `scipy.stats.f_oneway()` function and `statsmodels` use the same mathematical principles as manual calculation but with optimized algorithms.

**Model Specification:**
The ANOVA model can be written as:
```math
Y_{ij} = \mu + \alpha_i + \epsilon_{ij}
```

where:
- $Y_{ij}$ = observation $j$ in group $i$
- $\mu$ = overall mean
- $\alpha_i$ = effect of group $i$ (deviation from overall mean)
- $\epsilon_{ij}$ = random error term

**Assumptions:**
1. **Independence:** Observations are independent
2. **Normality:** Error terms are normally distributed
3. **Homoscedasticity:** Error terms have constant variance
4. **Linearity:** Effects are additive

```python
# Perform ANOVA using scipy's f_oneway function
from scipy.stats import f_oneway

f_statistic, p_value = f_oneway(mpg_4cyl, mpg_6cyl, mpg_8cyl)

print("=== PYTHON BUILT-IN ANOVA RESULTS ===")
print(f"F-statistic: {round(f_statistic, 3)}")
print(f"p-value: {round(p_value, 4)}")

# Using statsmodels for more comprehensive ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Fit the model
model = ols('mpg ~ cyl_factor', data=mtcars).fit()
anova_table = anova_lm(model, typ=2)

print("\n=== STATSMODELS ANOVA TABLE ===")
print(anova_table)

# Extract key statistics
ss_between = anova_table.loc['cyl_factor', 'sum_sq']
ss_within = anova_table.loc['Residual', 'sum_sq']
df_between = anova_table.loc['cyl_factor', 'df']
df_within = anova_table.loc['Residual', 'df']
ms_between = anova_table.loc['cyl_factor', 'mean_sq']
ms_within = anova_table.loc['Residual', 'mean_sq']

print(f"\nDegrees of freedom: {df_between}, {df_within}")
print(f"Sum of Squares (Between): {round(ss_between, 3)}")
print(f"Sum of Squares (Within): {round(ss_within, 3)}")
print(f"Mean Square (Between): {round(ms_between, 3)}")
print(f"Mean Square (Within): {round(ms_within, 3)}")

# Verify manual calculation matches Python results
print("\n=== VERIFICATION ===")
print(f"Manual F-statistic: {round(anova_result['f_statistic'], 3)}")
print(f"Python F-statistic: {round(f_statistic, 3)}")
print(f"Match: {abs(anova_result['f_statistic'] - f_statistic) < 0.001}")

print(f"\nManual p-value: {round(anova_result['p_value'], 4)}")
print(f"Python p-value: {round(p_value, 4)}")
print(f"Match: {abs(anova_result['p_value'] - p_value) < 0.0001}")

# Model diagnostics
print("\n=== MODEL DIAGNOSTICS ===")
print(f"Number of observations: {len(mtcars)}")
print(f"Number of groups: {len(mtcars['cyl_factor'].unique())}")

# Residual analysis
residuals_model = model.resid
fitted_values = model.fittedvalues

print("\nResidual Analysis:")
print(f"Mean of residuals: {round(np.mean(residuals_model), 6)} (should be ~0)")
print(f"SD of residuals: {round(np.std(residuals_model), 3)}")
print(f"Min residual: {round(np.min(residuals_model), 3)}")
print(f"Max residual: {round(np.max(residuals_model), 3)}")

# Model fit statistics
print("\nModel Fit:")
r_squared = ss_between / (ss_between + ss_within)
adj_r_squared = 1 - (ss_within / df_within) / ((ss_between + ss_within) / (df_between + df_within))
print(f"R-squared: {round(r_squared, 3)}")
print(f"Adjusted R-squared: {round(adj_r_squared, 3)}")

# Confidence intervals for group means
print("\n=== CONFIDENCE INTERVALS FOR GROUP MEANS ===")
for group in mtcars['cyl_factor'].unique():
    group_data = mtcars[mtcars['cyl_factor'] == group]['mpg']
    n = len(group_data)
    mean_val = np.mean(group_data)
    se = np.std(group_data) / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, n - 1)
    ci_lower = mean_val - t_critical * se
    ci_upper = mean_val + t_critical * se
    print(f"{group}: {round(mean_val, 2)} [{round(ci_lower, 2)}, {round(ci_upper, 2)}]")
```

## Descriptive Statistics

Descriptive statistics provide essential information about the data distribution and help assess ANOVA assumptions. They also aid in interpreting the practical significance of results.

### Mathematical Foundation

**Group Statistics:**
For each group $i$:
- **Mean:** $\bar{X}_i = \frac{1}{n_i}\sum_{j=1}^{n_i} X_{ij}$
- **Variance:** $s_i^2 = \frac{1}{n_i-1}\sum_{j=1}^{n_i} (X_{ij} - \bar{X}_i)^2$
- **Standard Error:** $SE_i = \frac{s_i}{\sqrt{n_i}}$

**Overall Statistics:**
- **Grand Mean:** $\bar{X}_{..} = \frac{1}{N}\sum_{i=1}^{k}\sum_{j=1}^{n_i} X_{ij}$
- **Pooled Variance:** $s_p^2 = \frac{\sum_{i=1}^{k} (n_i-1)s_i^2}{\sum_{i=1}^{k} (n_i-1)}$

**Effect Size Indicators:**
- **Coefficient of Variation:** $CV_i = \frac{s_i}{\bar{X}_i}$
- **Standardized Mean Difference:** $d_i = \frac{\bar{X}_i - \bar{X}_{..}}{s_p}$

### Group Comparisons

```python
# Comprehensive descriptive statistics for each group
from scipy import stats
from scipy.stats import skew, kurtosis

group_stats = mtcars.groupby('cyl_factor')['mpg'].agg([
    'count', 'mean', 'std', 'median', 'min', 'max',
    lambda x: np.percentile(x, 25),  # q25
    lambda x: np.percentile(x, 75),  # q75
]).rename(columns={
    '<lambda_0>': 'q25',
    '<lambda_1>': 'q75'
})

# Calculate additional statistics
group_stats['iqr'] = group_stats['q75'] - group_stats['q25']
group_stats['se'] = group_stats['std'] / np.sqrt(group_stats['count'])
group_stats['cv'] = group_stats['std'] / group_stats['mean']

# Calculate skewness and kurtosis for each group
skewness_values = []
kurtosis_values = []

for group in mtcars['cyl_factor'].unique():
    group_data = mtcars[mtcars['cyl_factor'] == group]['mpg']
    skewness_values.append(skew(group_data))
    kurtosis_values.append(kurtosis(group_data))

group_stats['skewness'] = skewness_values
group_stats['kurtosis'] = kurtosis_values

print("=== COMPREHENSIVE GROUP STATISTICS ===")
print(group_stats)

# Overall statistics
overall_stats = {
    'n': len(mtcars),
    'mean': np.mean(mtcars['mpg']),
    'sd': np.std(mtcars['mpg']),
    'median': np.median(mtcars['mpg']),
    'min': np.min(mtcars['mpg']),
    'max': np.max(mtcars['mpg']),
    'q25': np.percentile(mtcars['mpg'], 25),
    'q75': np.percentile(mtcars['mpg'], 75),
    'iqr': np.percentile(mtcars['mpg'], 75) - np.percentile(mtcars['mpg'], 25),
    'cv': np.std(mtcars['mpg']) / np.mean(mtcars['mpg']),
    'skewness': skew(mtcars['mpg']),
    'kurtosis': kurtosis(mtcars['mpg'])
}

print("\n=== OVERALL STATISTICS ===")
for key, value in overall_stats.items():
    print(f"{key}: {round(value, 3)}")

# Pooled variance calculation
def pooled_variance(group_data):
    """Calculate pooled variance across groups"""
    n_per_group = [len(group) for group in group_data]
    var_per_group = [np.var(group, ddof=1) for group in group_data]
    
    numerator = sum((n_per_group[i] - 1) * var_per_group[i] for i in range(len(group_data)))
    denominator = sum(n_per_group[i] - 1 for i in range(len(group_data)))
    
    return numerator / denominator

pooled_var = pooled_variance([mpg_4cyl, mpg_6cyl, mpg_8cyl])
pooled_sd = np.sqrt(pooled_var)

print(f"\n=== POOLED STATISTICS ===")
print(f"Pooled variance: {round(pooled_var, 3)}")
print(f"Pooled standard deviation: {round(pooled_sd, 3)}")

# Effect size calculations for each group
group_means = group_stats['mean'].values
overall_mean = overall_stats['mean']

effect_sizes = (group_means - overall_mean) / pooled_sd
effect_size_dict = dict(zip(group_stats.index, effect_sizes))

print(f"\n=== EFFECT SIZE INDICATORS ===")
print("Standardized mean differences (Cohen's d relative to overall mean):")
for group, effect_size in effect_size_dict.items():
    print(f"{group}: d = {round(effect_size, 3)}")

# Variance ratio analysis
variances = group_stats['std']**2
max_var = np.max(variances)
min_var = np.min(variances)
var_ratio = max_var / min_var

print(f"\n=== VARIANCE ANALYSIS ===")
print(f"Group variances: {[round(v, 3) for v in variances]}")
print(f"Variance ratio (max/min): {round(var_ratio, 3)}")
if var_ratio > 4:
    print("⚠️  Large variance ratio - consider assumption violations")
elif var_ratio > 2:
    print("⚠️  Moderate variance ratio - check homogeneity assumption")
else:
    print("✓ Variance ratio acceptable")

# Sample size analysis
n_per_group = group_stats['count'].values
balanced_design = len(np.unique(n_per_group)) == 1

print(f"\n=== SAMPLE SIZE ANALYSIS ===")
print(f"Sample sizes per group: {n_per_group}")
print(f"Design balanced: {balanced_design}")
if not balanced_design:
    print("⚠️  Unbalanced design - consider Type III SS for interactions")
else:
    print("✓ Balanced design")

# Power analysis based on sample sizes
min_n = np.min(n_per_group)
print(f"Minimum sample size per group: {min_n}")
if min_n < 10:
    print("⚠️  Small sample sizes - consider nonparametric alternatives")
elif min_n < 30:
    print("⚠️  Moderate sample sizes - check normality carefully")
else:
    print("✓ Adequate sample sizes for parametric tests")
```

### Visualization

Visualization is crucial for understanding data distributions, identifying patterns, and assessing ANOVA assumptions. Different plots provide complementary information about the data structure.

**Key Visualization Goals:**
1. **Distribution Comparison:** Assess normality and homogeneity
2. **Central Tendency:** Compare group means and medians
3. **Variability:** Examine spread and outliers
4. **Effect Size:** Visualize practical significance

```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comprehensive ANOVA Visualization', fontsize=16, fontweight='bold')

# 1. Enhanced box plot with statistics
ax1 = axes[0, 0]
sns.boxplot(data=mtcars, x='cyl_factor', y='mpg', ax=ax1)
# Add mean points
means = mtcars.groupby('cyl_factor')['mpg'].mean()
ax1.plot(range(len(means)), means, 'o', color='red', markersize=8, label='Mean')
ax1.set_title('MPG by Number of Cylinders\nBoxes show IQR, lines show medians, red dots show means')
ax1.set_xlabel('Cylinders')
ax1.set_ylabel('MPG')
ax1.legend()

# 2. Violin plot with box plot overlay
ax2 = axes[0, 1]
sns.violinplot(data=mtcars, x='cyl_factor', y='mpg', ax=ax2, inner='box')
ax2.set_title('MPG Distribution by Cylinders\nViolin shows density, box shows quartiles')
ax2.set_xlabel('Cylinders')
ax2.set_ylabel('MPG')

# 3. Histogram by group with density curves
ax3 = axes[0, 2]
for group in mtcars['cyl_factor'].unique():
    group_data = mtcars[mtcars['cyl_factor'] == group]['mpg']
    ax3.hist(group_data, alpha=0.7, label=group, bins=8, density=True)
    # Add density curve
    x_range = np.linspace(group_data.min(), group_data.max(), 100)
    density = stats.gaussian_kde(group_data)(x_range)
    ax3.plot(x_range, density, linewidth=2)
ax3.set_title('MPG Distribution by Cylinders\nHistograms with density curves')
ax3.set_xlabel('MPG')
ax3.set_ylabel('Density')
ax3.legend()

# 4. Q-Q plots for normality assessment
ax4 = axes[1, 0]
for i, group in enumerate(mtcars['cyl_factor'].unique()):
    group_data = mtcars[mtcars['cyl_factor'] == group]['mpg']
    stats.probplot(group_data, dist="norm", plot=ax4)
    ax4.set_title(f'Q-Q Plot for {group}\nPoints should follow the line for normal distributions')
    break  # Show only first group for clarity

# 5. Residuals vs fitted plot
ax5 = axes[1, 1]
ax5.scatter(fitted_values, residuals_model, alpha=0.7, c=mtcars['cyl_factor'].astype('category').cat.codes)
ax5.axhline(y=0, color='red', linestyle='--')
# Add trend line
z = np.polyfit(fitted_values, residuals_model, 1)
p = np.poly1d(z)
ax5.plot(fitted_values, p(fitted_values), "b--", alpha=0.8)
ax5.set_title('Residuals vs Fitted Values\nShould show no pattern for valid ANOVA')
ax5.set_xlabel('Fitted Values')
ax5.set_ylabel('Residuals')

# 6. Mean comparison plot with confidence intervals
ax6 = axes[1, 2]
x_pos = np.arange(len(group_stats))
bars = ax6.bar(x_pos, group_stats['mean'], alpha=0.7, 
               yerr=1.96 * group_stats['se'], capsize=5)
ax6.set_title('Group Means with 95% Confidence Intervals\nBars show means, error bars show 95% CI')
ax6.set_xlabel('Cylinders')
ax6.set_ylabel('Mean MPG')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(group_stats.index)

plt.tight_layout()
plt.show()

# Additional diagnostic plots
fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle('Additional Diagnostic Plots', fontsize=16, fontweight='bold')

# 7. Scale-location plot for homoscedasticity
sqrt_abs_resid = np.sqrt(np.abs(residuals_model))
ax7.scatter(fitted_values, sqrt_abs_resid, alpha=0.7, c=mtcars['cyl_factor'].astype('category').cat.codes)
# Add trend line
z = np.polyfit(fitted_values, sqrt_abs_resid, 1)
p = np.poly1d(z)
ax7.plot(fitted_values, p(fitted_values), "b--", alpha=0.8)
ax7.set_title('Scale-Location Plot\nShould show constant spread for homoscedasticity')
ax7.set_xlabel('Fitted Values')
ax7.set_ylabel('√|Residuals|')

# 8. Leverage plot (simplified version)
# Calculate leverage manually
X = pd.get_dummies(mtcars['cyl_factor'], drop_first=True)
X = sm.add_constant(X)
leverage = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)

ax8.scatter(leverage, residuals_model, alpha=0.7, c=mtcars['cyl_factor'].astype('category').cat.codes)
ax8.axhline(y=0, color='red', linestyle='--')
ax8.set_title('Leverage Plot\nHigh leverage points may be influential')
ax8.set_xlabel('Leverage')
ax8.set_ylabel('Residuals')

plt.tight_layout()
plt.show()

# Summary of visual findings
print("=== VISUAL ANALYSIS SUMMARY ===")
print("1. Box plots: Show group differences and outliers")
print("2. Violin plots: Show distribution shapes and density")
print("3. Histograms: Show data distribution and normality")
print("4. Q-Q plots: Assess normality assumption")
print("5. Residual plots: Check model assumptions")
print("6. Mean plots: Show effect sizes with uncertainty")
print("7. Scale-location: Check homoscedasticity")
print("8. Leverage: Identify influential observations")
```

## Effect Size

Effect size measures the magnitude of the relationship between the independent variable and the dependent variable, independent of sample size. This is crucial for understanding practical significance beyond statistical significance.

### Mathematical Foundation

**Eta-Squared (η²):**
```math
\eta^2 = \frac{SS_{between}}{SS_{total}} = \frac{SS_{between}}{SS_{between} + SS_{within}}
```

**Partial Eta-Squared (η²p):**
```math
\eta_p^2 = \frac{SS_{between}}{SS_{between} + SS_{within}}
```

For one-way ANOVA, partial eta-squared equals eta-squared.

**Omega-Squared (ω²):**
```math
\omega^2 = \frac{SS_{between} - (k-1)MS_{within}}{SS_{total} + MS_{within}}
```

**Cohen's f:**
```math
f = \sqrt{\frac{\eta^2}{1 - \eta^2}}
```

**Interpretation Guidelines:**
- **Eta-squared:** Small (0.01), Medium (0.06), Large (0.14)
- **Omega-squared:** Small (0.01), Medium (0.06), Large (0.14)
- **Cohen's f:** Small (0.10), Medium (0.25), Large (0.40)

### Comprehensive Effect Size Analysis

```python
# Comprehensive effect size calculation function
def calculate_anova_effect_sizes(anova_result):
    """Calculate comprehensive effect size measures for ANOVA"""
    # Extract components
    ss_between = anova_result['ss_between']
    ss_within = anova_result['ss_within']
    ss_total = anova_result['ss_total']
    df_between = anova_result['df_between']
    ms_within = anova_result['ms_within']
    
    # Eta-squared
    eta_squared = ss_between / ss_total
    
    # Partial eta-squared (same as eta-squared for one-way ANOVA)
    partial_eta_squared = eta_squared
    
    # Omega-squared (unbiased estimator)
    omega_squared = (ss_between - (df_between * ms_within)) / (ss_total + ms_within)
    
    # Cohen's f
    cohens_f = np.sqrt(eta_squared / (1 - eta_squared))
    
    # Epsilon-squared (for nonparametric ANOVA)
    epsilon_squared = (ss_between - (df_between * ms_within)) / ss_total
    
    # Confidence interval for eta-squared
    f_stat = anova_result['f_statistic']
    df1 = df_between
    df2 = anova_result['df_within']
    
    # Noncentrality parameter
    lambda_param = f_stat * df1
    
    # Confidence interval using noncentral F distribution
    ci_lower = 1 - 1 / (1 + lambda_param / f.ppf(0.975, df1, df2))
    ci_upper = 1 - 1 / (1 + lambda_param / f.ppf(0.025, df1, df2))
    
    return {
        'eta_squared': eta_squared,
        'partial_eta_squared': partial_eta_squared,
        'omega_squared': omega_squared,
        'cohens_f': cohens_f,
        'epsilon_squared': epsilon_squared,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

# Apply to our ANOVA results
effect_sizes = calculate_anova_effect_sizes(anova_result)

print("=== COMPREHENSIVE EFFECT SIZE ANALYSIS ===")
print(f"Eta-squared (η²): {round(effect_sizes['eta_squared'], 3)}")
print(f"Partial eta-squared (η²p): {round(effect_sizes['partial_eta_squared'], 3)}")
print(f"Omega-squared (ω²): {round(effect_sizes['omega_squared'], 3)}")
print(f"Cohen's f: {round(effect_sizes['cohens_f'], 3)}")
print(f"Epsilon-squared (ε²): {round(effect_sizes['epsilon_squared'], 3)}")
print(f"95% CI for η²: [{round(effect_sizes['ci_lower'], 3)}, {round(effect_sizes['ci_upper'], 3)}]")

# Enhanced effect size interpretation
def interpret_effect_size_comprehensive(eta_sq, omega_sq, cohens_f):
    """Provide comprehensive interpretation of effect sizes"""
    print("\n=== EFFECT SIZE INTERPRETATION ===")
    
    # Eta-squared interpretation
    print(f"Eta-squared (η²): {round(eta_sq, 3)}")
    if eta_sq < 0.01:
        print("  Interpretation: Negligible effect")
    elif eta_sq < 0.06:
        print("  Interpretation: Small effect")
    elif eta_sq < 0.14:
        print("  Interpretation: Medium effect")
    else:
        print("  Interpretation: Large effect")
    
    # Omega-squared interpretation
    print(f"\nOmega-squared (ω²): {round(omega_sq, 3)}")
    if omega_sq < 0.01:
        print("  Interpretation: Negligible effect")
    elif omega_sq < 0.06:
        print("  Interpretation: Small effect")
    elif omega_sq < 0.14:
        print("  Interpretation: Medium effect")
    else:
        print("  Interpretation: Large effect")
    
    # Cohen's f interpretation
    print(f"\nCohen's f: {round(cohens_f, 3)}")
    if cohens_f < 0.10:
        print("  Interpretation: Small effect")
    elif cohens_f < 0.25:
        print("  Interpretation: Medium effect")
    elif cohens_f < 0.40:
        print("  Interpretation: Large effect")
    else:
        print("  Interpretation: Very large effect")
    
    # Practical significance assessment
    print("\n=== PRACTICAL SIGNIFICANCE ASSESSMENT ===")
    if eta_sq >= 0.14:
        print("✓ Large practical effect - results are practically meaningful")
    elif eta_sq >= 0.06:
        print("⚠️  Medium practical effect - consider context for interpretation")
    elif eta_sq >= 0.01:
        print("⚠️  Small practical effect - may not be practically meaningful")
    else:
        print("✗ Negligible practical effect - results may not be practically useful")

# Apply comprehensive interpretation
interpret_effect_size_comprehensive(effect_sizes['eta_squared'], 
                                   effect_sizes['omega_squared'], 
                                   effect_sizes['cohens_f'])

# Effect size comparison with other measures
print("\n=== EFFECT SIZE COMPARISON ===")
print("Comparison of different effect size measures:")
print(f"Eta-squared (biased): {round(effect_sizes['eta_squared'], 3)}")
print(f"Omega-squared (unbiased): {round(effect_sizes['omega_squared'], 3)}")
print(f"Difference (bias): {round(effect_sizes['eta_squared'] - effect_sizes['omega_squared'], 3)}")

if abs(effect_sizes['eta_squared'] - effect_sizes['omega_squared']) < 0.01:
    print("✓ Bias is minimal - eta-squared is acceptable")
else:
    print("⚠️  Notable bias - prefer omega-squared for small samples")

# Power analysis based on effect size
# Note: Python doesn't have a direct equivalent to R's pwr package
# We'll implement a simplified version
def power_analysis_anova(k, n_per_group, f_effect_size, alpha=0.05):
    """Calculate power for one-way ANOVA"""
    df1 = k - 1
    df2 = k * (n_per_group - 1)
    lambda_param = f_effect_size**2 * k * n_per_group
    
    # Calculate power using noncentral F distribution
    f_critical = f.ppf(1 - alpha, df1, df2)
    power = 1 - f.cdf(f_critical, df1, df2, lambda_param)
    return power

def estimate_sample_size_for_power(k, f_effect_size, alpha, power):
    """Estimate required sample size for desired power (simplified)"""
    # This is a simplified estimation - in practice use proper power analysis libraries
    # For demonstration purposes, we'll use a rough approximation
    df1 = k - 1
    
    # Rough estimation based on effect size and power
    if f_effect_size < 0.1:
        base_n = 50
    elif f_effect_size < 0.25:
        base_n = 20
    elif f_effect_size < 0.4:
        base_n = 10
    else:
        base_n = 5
    
    # Adjust for power level
    if power >= 0.95:
        base_n = int(base_n * 1.5)
    elif power >= 0.9:
        base_n = int(base_n * 1.2)
    elif power < 0.8:
        base_n = int(base_n * 0.8)
    
    return max(base_n, 5)  # Minimum of 5 per group

f_effect_size = effect_sizes['cohens_f']
n_per_group = min(mtcars.groupby('cyl_factor').size())
current_power = power_analysis_anova(k=3, n_per_group=n_per_group, 
                                   f_effect_size=f_effect_size)

print(f"\n=== POWER ANALYSIS ===")
print(f"Effect size f: {round(f_effect_size, 3)}")
print(f"Current power: {round(current_power, 3)}")

if current_power < 0.8:
    # Estimate required sample size (simplified)
    print("Required sample size per group for 80% power: Estimate needed")
else:
    print("✓ Adequate power for detecting this effect size")
```

## Post Hoc Tests

When ANOVA reveals significant differences between groups, post hoc tests identify which specific group pairs differ significantly. These tests control for multiple comparisons to maintain the family-wise error rate.

### Mathematical Foundation

**Multiple Comparison Problem:**
With $k$ groups, there are $\binom{k}{2} = \frac{k(k-1)}{2}$ pairwise comparisons. If each test uses $\alpha = 0.05$, the family-wise error rate becomes:
```math
\alpha_{FW} = 1 - (1 - \alpha)^m
```
where $m$ is the number of comparisons.

**Tukey's HSD (Honestly Significant Difference):**
```math
HSD = q_{\alpha,k,N-k} \sqrt{\frac{MS_{within}}{n}}
```
where $q_{\alpha,k,N-k}$ is the critical value from the studentized range distribution.

**Bonferroni Correction:**
```math
\alpha_{adjusted} = \frac{\alpha}{m}
```

**Scheffe's Test:**
```math
S = \sqrt{(k-1)F_{\alpha,k-1,N-k}}
```

### Comprehensive Post Hoc Analysis

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Comprehensive post hoc analysis function
def comprehensive_posthoc(data, group_var, response_var, alpha=0.05):
    """Perform comprehensive post hoc analysis with multiple methods"""
    print("=== COMPREHENSIVE POST HOC ANALYSIS ===")
    
    # Get group information
    groups = data[group_var].unique()
    k = len(groups)
    m = k * (k - 1) / 2  # number of pairwise comparisons
    
    print(f"Number of groups (k): {k}")
    print(f"Number of pairwise comparisons (m): {m}")
    print(f"Family-wise error rate without correction: {round(1 - (1 - alpha)**m, 4)}")
    
    # 1. Tukey's HSD Test
    print("\n=== TUKEY'S HSD TEST ===")
    mc = MultiComparison(data[response_var], data[group_var])
    tukey_result = mc.tukeyhsd(alpha=alpha)
    print(tukey_result)
    
    # Extract significant differences
    significant_tukey = tukey_result.pvalues < alpha
    
    if np.any(significant_tukey):
        print(f"\nSignificant pairwise differences (Tukey's HSD, p < {alpha}):")
        for i, (group1, group2) in enumerate(tukey_result.groupsunique):
            if significant_tukey[i]:
                diff = tukey_result.meandiffs[i]
                p_val = tukey_result.pvalues[i]
                print(f"{group1} vs {group2}: diff = {round(diff, 3)}, p = {round(p_val, 4)}")
    else:
        print("No significant pairwise differences found with Tukey's HSD.")
    
    # 2. Bonferroni Correction
    print("\n=== BONFERRONI CORRECTION ===")
    # Perform all pairwise t-tests
    p_values = []
    comparisons = []
    
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            group1_data = data[data[group_var] == groups[i]][response_var]
            group2_data = data[data[group_var] == groups[j]][response_var]
            _, p_val = ttest_ind(group1_data, group2_data)
            p_values.append(p_val)
            comparisons.append(f"{groups[i]} vs {groups[j]}")
    
    # Apply Bonferroni correction
    bonferroni_pvals = multipletests(p_values, method='bonferroni')[1]
    
    print("Bonferroni-corrected p-values:")
    for comp, p_val in zip(comparisons, bonferroni_pvals):
        print(f"{comp}: p = {round(p_val, 4)}")
    
    # Extract significant Bonferroni results
    significant_bonferroni = bonferroni_pvals < alpha
    if np.any(significant_bonferroni):
        print(f"\nSignificant differences (Bonferroni-corrected p < {alpha}):")
        for i, comp in enumerate(comparisons):
            if significant_bonferroni[i]:
                print(f"{comp}: p = {round(bonferroni_pvals[i], 4)}")
    else:
        print(f"No significant differences found with Bonferroni correction.")
    
    # 3. Scheffe's Test
    print("\n=== SCHEFFE'S TEST ===")
    scheffe_result = scheffe_test(anova_result, [mpg_4cyl, mpg_6cyl, mpg_8cyl], alpha)
    
    print(f"Scheffe's critical value: {round(scheffe_result['scheffe_critical'], 3)}")
    
    for comp in scheffe_result['comparisons']:
        group_names = ["4-cylinder", "6-cylinder", "8-cylinder"]
        print(f"{group_names[comp['group1']]} vs {group_names[comp['group2']]}:")
        print(f"  Mean difference: {round(comp['mean_diff'], 3)}")
        print(f"  Test statistic: {round(comp['test_stat'], 3)}")
        print(f"  p-value: {round(comp['p_value'], 4)}")
        print(f"  Significant: {comp['significant']}")
    
    # 4. Holm's Method
    print("\n=== HOLM'S METHOD ===")
    holm_pvals = multipletests(p_values, method='holm')[1]
    
    print("Holm-corrected p-values:")
    for comp, p_val in zip(comparisons, holm_pvals):
        print(f"{comp}: p = {round(p_val, 4)}")
    
    # 5. False Discovery Rate (FDR)
    print("\n=== FALSE DISCOVERY RATE (FDR) ===")
    fdr_pvals = multipletests(p_values, method='fdr_bh')[1]
    
    print("FDR-corrected p-values:")
    for comp, p_val in zip(comparisons, fdr_pvals):
        print(f"{comp}: p = {round(p_val, 4)}")
    
    # Comparison of methods
    print("\n=== METHOD COMPARISON ===")
    print("Method comparison summary:")
    print("- Tukey's HSD: Controls FWER, most powerful for balanced designs")
    print("- Bonferroni: Most conservative, controls FWER")
    print("- Holm's method: Less conservative than Bonferroni, controls FWER")
    print("- FDR: Controls false discovery rate, less conservative")
    print("- Scheffe's: Most conservative, allows any contrast")
    
    return {
        'tukey': tukey_result,
        'bonferroni': {'p_values': bonferroni_pvals, 'comparisons': comparisons},
        'scheffe': scheffe_result,
        'holm': {'p_values': holm_pvals, 'comparisons': comparisons},
        'fdr': {'p_values': fdr_pvals, 'comparisons': comparisons}
    }

# Apply comprehensive post hoc analysis
posthoc_results = comprehensive_posthoc(mtcars, "cyl_factor", "mpg")

# Visualization of post hoc results
print("\n=== POST HOC VISUALIZATION ===")

# Tukey's HSD plot
plt.figure(figsize=(10, 6))
tukey_result.plot_simultaneous()
plt.title("Tukey's HSD Test Results")
plt.show()

# Create a summary table of all methods
def create_posthoc_summary(posthoc_results, alpha=0.05):
    """Create summary table of post hoc results"""
    print("=== POST HOC SUMMARY TABLE ===")
    
    # Extract significant pairs from each method
    tukey_sig = np.sum(posthoc_results['tukey'].pvalues < alpha)
    bonferroni_sig = np.sum(posthoc_results['bonferroni']['p_values'] < alpha)
    holm_sig = np.sum(posthoc_results['holm']['p_values'] < alpha)
    fdr_sig = np.sum(posthoc_results['fdr']['p_values'] < alpha)
    
    print("Method\t\tSignificant Pairs")
    print(f"Tukey's HSD\t{tukey_sig}")
    print(f"Bonferroni\t{bonferroni_sig}")
    print(f"Holm's\t\t{holm_sig}")
    print(f"FDR\t\t{fdr_sig}")
    
    # Agreement analysis
    print("\n=== AGREEMENT ANALYSIS ===")
    print("Methods that found the most significant differences:")
    method_counts = {
        "Tukey": tukey_sig,
        "Bonferroni": bonferroni_sig,
        "Holm": holm_sig,
        "FDR": fdr_sig
    }
    
    for method in sorted(method_counts.keys(), key=lambda x: method_counts[x], reverse=True):
        print(f"{method}: {method_counts[method]} significant pairs")

create_posthoc_summary(posthoc_results)
```

### Bonferroni Correction

```python
# Perform pairwise t-tests with Bonferroni correction
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Get all pairwise combinations
groups = mtcars['cyl_factor'].unique()
p_values = []
comparisons = []

for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        group1_data = mtcars[mtcars['cyl_factor'] == groups[i]]['mpg']
        group2_data = mtcars[mtcars['cyl_factor'] == groups[j]]['mpg']
        _, p_val = ttest_ind(group1_data, group2_data)
        p_values.append(p_val)
        comparisons.append(f"{groups[i]} vs {groups[j]}")

# Apply Bonferroni correction
bonferroni_pvals = multipletests(p_values, method='bonferroni')[1]

print("Bonferroni-corrected p-values:")
for comp, p_val in zip(comparisons, bonferroni_pvals):
    print(f"{comp}: p = {round(p_val, 4)}")

# Extract significant pairs
significant_bonferroni = bonferroni_pvals < 0.05

if np.any(significant_bonferroni):
    print("\nSignificant differences (Bonferroni-corrected p < 0.05):")
    for i, comp in enumerate(comparisons):
        if significant_bonferroni[i]:
            print(f"{comp}: p = {round(bonferroni_pvals[i], 4)}")
else:
    print("\nNo significant differences found with Bonferroni correction.")
```

### Scheffe's Test

```python
# Function to perform Scheffe's test
def scheffe_test(anova_result, group_data, alpha=0.05):
    """Perform Scheffe's test for multiple comparisons"""
    k = len(group_data)
    n_per_group = [len(group) for group in group_data]
    total_n = sum(n_per_group)
    
    # Calculate critical value
    f_critical = f.ppf(1 - alpha, k - 1, total_n - k)
    scheffe_critical = np.sqrt((k - 1) * f_critical)
    
    # Calculate group means
    group_means = [np.mean(group) for group in group_data]
    
    # Perform all pairwise comparisons
    comparisons = []
    pair_count = 0
    
    for i in range(k-1):
        for j in range(i+1, k):
            mean_diff = group_means[i] - group_means[j]
            
            # Standard error for the difference
            se_diff = np.sqrt(anova_result['ms_within'] * (1/n_per_group[i] + 1/n_per_group[j]))
            
            # Test statistic
            test_stat = abs(mean_diff) / se_diff
            
            # p-value
            p_value = 1 - f.cdf(test_stat**2 / (k - 1), k - 1, total_n - k)
            
            comparisons.append({
                'group1': i,
                'group2': j,
                'mean_diff': mean_diff,
                'test_stat': test_stat,
                'p_value': p_value,
                'significant': test_stat > scheffe_critical
            })
            
            pair_count += 1
    
    return {
        'comparisons': comparisons,
        'scheffe_critical': scheffe_critical,
        'alpha': alpha
    }

# Apply Scheffe's test
scheffe_result = scheffe_test(anova_result, [mpg_4cyl, mpg_6cyl, mpg_8cyl])

print("Scheffe's Test Results:")
print(f"Critical value: {round(scheffe_result['scheffe_critical'], 3)}")

for comp in scheffe_result['comparisons']:
    group_names = ["4-cylinder", "6-cylinder", "8-cylinder"]
    print(f"{group_names[comp['group1']]} vs {group_names[comp['group2']]}:")
    print(f"  Mean difference: {round(comp['mean_diff'], 3)}")
    print(f"  Test statistic: {round(comp['test_stat'], 3)}")
    print(f"  p-value: {round(comp['p_value'], 4)}")
    print(f"  Significant: {comp['significant']}")
```

## Assumption Checking

ANOVA relies on several key assumptions. Violations can affect the validity of results and may require alternative approaches. Comprehensive assumption checking is essential for robust statistical inference.

### Mathematical Foundation

**ANOVA Assumptions:**

1. **Independence:** Observations are independent within and between groups
2. **Normality:** Error terms follow a normal distribution
3. **Homoscedasticity:** Error terms have constant variance across groups
4. **Linearity:** Effects are additive

**Test Statistics:**
- **Shapiro-Wilk:** $W = \frac{(\sum_{i=1}^{n} a_i x_{(i)})^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$
- **Levene's Test:** $W = \frac{(N-k)\sum_{i=1}^{k} n_i(\bar{Z}_i - \bar{Z})^2}{(k-1)\sum_{i=1}^{k}\sum_{j=1}^{n_i}(Z_{ij} - \bar{Z}_i)^2}$
- **Bartlett's Test:** $\chi^2 = \frac{(N-k)\ln(s_p^2) - \sum_{i=1}^{k}(n_i-1)\ln(s_i^2)}{1 + \frac{1}{3(k-1)}(\sum_{i=1}^{k}\frac{1}{n_i-1} - \frac{1}{N-k})}$

### Comprehensive Assumption Checking

```python
from scipy.stats import shapiro, anderson, ks_2samp, levene, bartlett, fligner
from scipy.stats import jarque_bera
import warnings
warnings.filterwarnings('ignore')

# Comprehensive assumption checking function
def comprehensive_assumption_check(data, group_var, response_var, alpha=0.05):
    """Perform comprehensive assumption checking for ANOVA"""
    print("=== COMPREHENSIVE ANOVA ASSUMPTION CHECKING ===")
    
    # Get basic information
    groups = data[group_var].unique()
    k = len(groups)
    n_per_group = [sum(data[group_var] == group) for group in groups]
    total_n = sum(n_per_group)
    
    print(f"Number of groups: {k}")
    print(f"Sample sizes per group: {n_per_group}")
    print(f"Total sample size: {total_n}")
    
    # 1. Independence Assessment
    print("\n=== 1. INDEPENDENCE ASSESSMENT ===")
    
    # Check for balanced design
    balanced_design = len(set(n_per_group)) == 1
    print(f"Balanced design: {balanced_design}")
    
    # Check for random sampling (simulation-based)
    np.random.seed(123)
    def independence_test(data, group_var, response_var):
        # Create a simple test for independence
        model = ols(f'{response_var} ~ {group_var}', data=data).fit()
        residuals = model.resid
        
        # Test for autocorrelation in residuals
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(residuals, lags=1, return_df=True)
        lag1_corr = lb_result['lb_pvalue'].iloc[0]  # Simplified approach
        
        return {
            'lag1_correlation': lag1_corr,
            'independent': lag1_corr > 0.05
        }
    
    independence_result = independence_test(data, group_var, response_var)
    print(f"Lag-1 autocorrelation: {round(independence_result['lag1_correlation'], 3)}")
    print(f"Independence assumption met: {independence_result['independent']}")
    
    # 2. Normality Assessment
    print("\n=== 2. NORMALITY ASSESSMENT ===")
    
    # Shapiro-Wilk test for each group
    normality_results = {}
    all_normal = True
    
    for group in groups:
        group_data = data[data[group_var] == group][response_var]
        shapiro_result = shapiro(group_data)
        normality_results[str(group)] = shapiro_result
        
        print(f"Group {group} Shapiro-Wilk:")
        print(f"  W = {round(shapiro_result.statistic, 4)}")
        print(f"  p-value = {round(shapiro_result.pvalue, 4)}")
        print(f"  Normal: {shapiro_result.pvalue >= alpha}")
        
        if shapiro_result.pvalue < alpha:
            all_normal = False
    
    # Overall normality test on residuals
    model = ols(f'{response_var} ~ {group_var}', data=data).fit()
    residuals = model.resid
    overall_shapiro = shapiro(residuals)
    
    print(f"\nOverall residuals Shapiro-Wilk:")
    print(f"  W = {round(overall_shapiro.statistic, 4)}")
    print(f"  p-value = {round(overall_shapiro.pvalue, 4)}")
    print(f"  Normal: {overall_shapiro.pvalue >= alpha}")
    
    # Additional normality tests
    # Anderson-Darling test
    ad_result = anderson(residuals)
    print(f"\nAnderson-Darling test:")
    print(f"  A = {round(ad_result.statistic, 4)}")
    # Note: Anderson-Darling doesn't provide p-value directly in scipy
    print(f"  Critical values: {ad_result.critical_values}")
    
    # Kolmogorov-Smirnov test
    ks_result = ks_2samp(residuals, np.random.normal(np.mean(residuals), np.std(residuals), len(residuals)))
    print(f"\nKolmogorov-Smirnov test:")
    print(f"  D = {round(ks_result.statistic, 4)}")
    print(f"  p-value = {round(ks_result.pvalue, 4)}")
    print(f"  Normal: {ks_result.pvalue >= alpha}")
    
    # 3. Homoscedasticity Assessment
    print("\n=== 3. HOMOSCEDASTICITY ASSESSMENT ===")
    
    # Levene's test
    group_data_list = [data[data[group_var] == group][response_var].values for group in groups]
    levene_result = levene(*group_data_list)
    print(f"Levene's test:")
    print(f"  W = {round(levene_result.statistic, 4)}")
    print(f"  p-value = {round(levene_result.pvalue, 4)}")
    print(f"  Equal variances: {levene_result.pvalue >= alpha}")
    
    # Bartlett's test
    bartlett_result = bartlett(*group_data_list)
    print(f"\nBartlett's test:")
    print(f"  Chi-squared = {round(bartlett_result.statistic, 4)}")
    print(f"  p-value = {round(bartlett_result.pvalue, 4)}")
    print(f"  Equal variances: {bartlett_result.pvalue >= alpha}")
    
    # Fligner-Killeen test (more robust)
    fligner_result = fligner(*group_data_list)
    print(f"\nFligner-Killeen test:")
    print(f"  Chi-squared = {round(fligner_result.statistic, 4)}")
    print(f"  p-value = {round(fligner_result.pvalue, 4)}")
    print(f"  Equal variances: {fligner_result.pvalue >= alpha}")
    
    # Variance ratio analysis
    group_variances = [np.var(data[data[group_var] == group][response_var]) for group in groups]
    max_var = max(group_variances)
    min_var = min(group_variances)
    var_ratio = max_var / min_var
    
    print(f"\nVariance analysis:")
    print(f"  Group variances: {[round(v, 3) for v in group_variances]}")
    print(f"  Variance ratio (max/min): {round(var_ratio, 3)}")
    print(f"  Acceptable ratio (< 4): {var_ratio < 4}")
    
    # 4. Linearity Assessment
    print("\n=== 4. LINEARITY ASSESSMENT ===")
    print("Linearity assumption is inherent in one-way ANOVA design.")
    print("No interaction effects to test.")
    
    # 5. Outlier Detection
    print("\n=== 5. OUTLIER DETECTION ===")
    
    outliers_by_group = {}
    total_outliers = 0
    
    for group in groups:
        group_data = data[data[group_var] == group][response_var]
        
        # IQR method
        q1 = np.percentile(group_data, 25)
        q3 = np.percentile(group_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (group_data < lower_bound) | (group_data > upper_bound)
        outlier_indices = np.where(outliers)[0]
        outliers_by_group[str(group)] = outlier_indices
        
        print(f"Group {group} outliers (IQR method):")
        print(f"  Number of outliers: {len(outlier_indices)}")
        if len(outlier_indices) > 0:
            print(f"  Outlier values: {[round(group_data.iloc[i], 3) for i in outlier_indices]}")
    
        total_outliers += len(outlier_indices)
    
    # 6. Comprehensive Assessment
    print("\n=== 6. COMPREHENSIVE ASSESSMENT ===")
    
    # Count assumption violations
    violations = 0
    violation_details = []
    
    if not independence_result['independent']:
        violations += 1
        violation_details.append("Independence")
    
    if not all_normal or overall_shapiro.pvalue < alpha:
        violations += 1
        violation_details.append("Normality")
    
    if levene_result.pvalue < alpha or var_ratio >= 4:
        violations += 1
        violation_details.append("Homoscedasticity")
    
    if total_outliers > total_n * 0.1:
        violations += 1
        violation_details.append("Outliers")
    
    print(f"Total assumption violations: {violations}")
    if violations > 0:
        print(f"Violated assumptions: {', '.join(violation_details)}")
  
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    
    if violations == 0:
        print("✓ All assumptions met - standard ANOVA is appropriate")
    else:
        print("⚠️  Some assumptions violated - consider alternatives:")
        
        if "Normality" in violation_details:
            print("- Use Kruskal-Wallis test for non-normal data")
            print("- Consider data transformation")
        
        if "Homoscedasticity" in violation_details:
            print("- Use Welch's ANOVA for unequal variances")
            print("- Consider robust ANOVA methods")
        
        if "Outliers" in violation_details:
            print("- Investigate outliers for data entry errors")
            print("- Consider robust statistical methods")
            print("- Report results with and without outliers")
        
        if "Independence" in violation_details:
            print("- Check study design for independence violations")
            print("- Consider mixed-effects models if appropriate")
    
    return {
        'independence': independence_result,
        'normality': {'group_tests': normality_results, 'overall': overall_shapiro, 'ad': ad_result, 'ks': ks_result},
        'homoscedasticity': {'levene': levene_result, 'bartlett': bartlett_result, 'fligner': fligner_result, 'var_ratio': var_ratio},
        'outliers': outliers_by_group,
        'violations': violations,
        'violation_details': violation_details
    }

# Apply comprehensive assumption checking
assumption_results = comprehensive_assumption_check(mtcars, "cyl_factor", "mpg")
```

### Homogeneity of Variance

```python
# Function to test homogeneity of variance
def check_homogeneity_anova(data, group_var, response_var):
    """Test homogeneity of variance using multiple methods"""
    print("=== HOMOGENEITY OF VARIANCE TESTS ===")
    
    # Levene's test
    from scipy.stats import levene
    group_data_list = [data[data[group_var] == group][response_var].values for group in data[group_var].unique()]
    levene_result = levene(*group_data_list)
    print(f"Levene's test p-value: {round(levene_result.pvalue, 4)}")
    
    # Bartlett's test
    from scipy.stats import bartlett
    bartlett_result = bartlett(*group_data_list)
    print(f"Bartlett's test p-value: {round(bartlett_result.pvalue, 4)}")
    
    # Fligner-Killeen test (more robust)
    from scipy.stats import fligner
    fligner_result = fligner(*group_data_list)
    print(f"Fligner-Killeen test p-value: {round(fligner_result.pvalue, 4)}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if levene_result.pvalue >= 0.05:
        print("- Variances appear equal across groups")
        print("- Standard ANOVA is appropriate")
    else:
        print("- Variances are significantly different")
        print("- Consider Welch's ANOVA or nonparametric alternatives")
    
    return {
        'levene': levene_result,
        'bartlett': bartlett_result,
        'fligner': fligner_result
    }

# Check homogeneity for cylinder groups
homogeneity_results = check_homogeneity_anova(mtcars, "cyl_factor", "mpg")
```

### Independence and Random Sampling

```python
# Function to check independence assumption
def check_independence_anova(data, group_var, response_var):
    """Check independence assumption for ANOVA"""
    print("=== INDEPENDENCE ASSESSMENT ===")
    
    # Check for balanced design
    group_counts = data[group_var].value_counts()
    print("Group sample sizes:")
    print(group_counts)
    
    # Check for equal sample sizes
    n_groups = len(group_counts)
    equal_sizes = len(group_counts.unique()) == 1
    
    if equal_sizes:
        print("Design is balanced (equal sample sizes)")
    else:
        print("Design is unbalanced (unequal sample sizes)")
    
    # Check for outliers
    outliers_by_group = {}
    for group in data[group_var].unique():
        group_data = data[data[group_var] == group][response_var]
        
        # IQR method
        q1 = np.percentile(group_data, 25)
        q3 = np.percentile(group_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (group_data < lower_bound) | (group_data > upper_bound)
        outliers_by_group[str(group)] = np.where(outliers)[0]
    
    print("Outliers by group (IQR method):")
    for group in outliers_by_group.keys():
        n_outliers = len(outliers_by_group[group])
        print(f"Group {group}: {n_outliers} outliers")
    
    return {
        'group_counts': group_counts,
        'balanced': equal_sizes,
        'outliers': outliers_by_group
    }

# Check independence for cylinder groups
independence_results = check_independence_anova(mtcars, "cyl_factor", "mpg")
```

## Nonparametric Alternatives

When ANOVA assumptions are violated, nonparametric alternatives provide robust statistical inference without requiring normality or equal variances. These methods are based on ranks rather than actual values.

### Mathematical Foundation

**Kruskal-Wallis Test:**
The test statistic is:
```math
H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)
```

where:
- $R_i$ = sum of ranks for group $i$
- $n_i$ = sample size of group $i$
- $N$ = total sample size
- $k$ = number of groups

Under the null hypothesis, $H$ follows a chi-squared distribution with $k-1$ degrees of freedom.

**Effect Size (Epsilon-squared):**
```math
\varepsilon^2 = \frac{H - (k-1)}{N - k}
```

**Mann-Whitney U Test (for pairwise comparisons):**
```math
U = n_1n_2 + \frac{n_1(n_1+1)}{2} - R_1
```

### Comprehensive Nonparametric Analysis

```python
from scipy.stats import kruskal, mannwhitneyu, rankdata
from statsmodels.stats.multitest import multipletests

# Comprehensive nonparametric analysis function
def comprehensive_nonparametric(data, group_var, response_var, alpha=0.05):
    """Perform comprehensive nonparametric analysis"""
    print("=== COMPREHENSIVE NONPARAMETRIC ANALYSIS ===")
    
    # Get basic information
    groups = data[group_var].unique()
    k = len(groups)
    n_per_group = [sum(data[group_var] == group) for group in groups]
    total_n = sum(n_per_group)
    
    print(f"Number of groups: {k}")
    print(f"Sample sizes per group: {n_per_group}")
    print(f"Total sample size: {total_n}")
    
    # 1. Kruskal-Wallis Test
    print("\n=== 1. KRUSKAL-WALLIS TEST ===")
    group_data_list = [data[data[group_var] == group][response_var].values for group in groups]
    kruskal_result = kruskal(*group_data_list)
    print(f"H-statistic: {round(kruskal_result.statistic, 3)}")
    print(f"p-value: {round(kruskal_result.pvalue, 4)}")
    
    # Calculate effect size
    h_statistic = kruskal_result.statistic
    epsilon_squared = (h_statistic - (k - 1)) / (total_n - k)
    
    print(f"\nEffect size analysis:")
    print(f"H-statistic: {round(h_statistic, 3)}")
    print(f"Epsilon-squared (ε²): {round(epsilon_squared, 3)}")
    print(f"p-value: {round(kruskal_result.pvalue, 4)}")
    print(f"Significant: {kruskal_result.pvalue < alpha}")
    
    # 2. Rank-based descriptive statistics
    print("\n=== 2. RANK-BASED DESCRIPTIVE STATISTICS ===")
    
    # Calculate ranks for the entire dataset
    data_copy = data.copy()
    data_copy['ranks'] = rankdata(data_copy[response_var])
    
    rank_stats = data_copy.groupby(group_var)['ranks'].agg([
        'count', 'mean', 'median', 'min', 'max', 'sum'
    ]).rename(columns={
        'count': 'n',
        'mean': 'mean_rank',
        'median': 'median_rank',
        'min': 'min_rank',
        'max': 'max_rank',
        'sum': 'rank_sum'
    })
    
    print(rank_stats)
    
    # 3. Pairwise Mann-Whitney U tests
    print("\n=== 3. PAIRWISE MANN-WHITNEY U TESTS ===")
    
    # Perform all pairwise comparisons
    pairwise_results = []
    pair_count = 0
    
    for i in range(k-1):
        for j in range(i+1, k):
            group1 = groups[i]
            group2 = groups[j]
            
            data1 = data[data[group_var] == group1][response_var]
            data2 = data[data[group_var] == group2][response_var]
            
            # Mann-Whitney U test
            mw_result = mannwhitneyu(data1, data2, alternative='two-sided')
            
            # Calculate effect size (r)
            from scipy.stats import norm
            z_stat = norm.ppf(mw_result.pvalue / 2)
            n1 = len(data1)
            n2 = len(data2)
            r_effect_size = abs(z_stat) / np.sqrt(n1 + n2)
            
            pairwise_results.append({
                'group1': group1,
                'group2': group2,
                'u_statistic': mw_result.statistic,
                'p_value': mw_result.pvalue,
                'effect_size_r': r_effect_size,
                'significant': mw_result.pvalue < alpha
            })
            
            print(f"Comparison: {group1} vs {group2}")
            print(f"  U-statistic: {round(mw_result.statistic, 3)}")
            print(f"  p-value: {round(mw_result.pvalue, 4)}")
            print(f"  Effect size (r): {round(r_effect_size, 3)}")
            print(f"  Significant: {mw_result.pvalue < alpha}")
            
            pair_count += 1
    
    # 4. Dunn's test (simplified version using multiple comparisons)
    print("\n=== 4. DUNN'S TEST (SIMPLIFIED) ===")
    
    # Get p-values from Mann-Whitney tests
    p_values = [result['p_value'] for result in pairwise_results]
    comparisons = [f"{result['group1']} vs {result['group2']}" for result in pairwise_results]
    
    # Apply multiple corrections
    bonferroni_pvals = multipletests(p_values, method='bonferroni')[1]
    holm_pvals = multipletests(p_values, method='holm')[1]
    fdr_pvals = multipletests(p_values, method='fdr_bh')[1]
    
    print("Dunn's test with Bonferroni correction:")
    for comp, p_val in zip(comparisons, bonferroni_pvals):
        print(f"  {comp}: p = {round(p_val, 4)}")
    
    print("\nDunn's test with Holm correction:")
    for comp, p_val in zip(comparisons, holm_pvals):
        print(f"  {comp}: p = {round(p_val, 4)}")
    
    print("\nDunn's test with FDR correction:")
    for comp, p_val in zip(comparisons, fdr_pvals):
        print(f"  {comp}: p = {round(p_val, 4)}")
    
    # 5. Comparison with parametric ANOVA
    print("\n=== 5. COMPARISON WITH PARAMETRIC ANOVA ===")
    
    # Perform parametric ANOVA for comparison
    f_stat, anova_p_value = f_oneway(*group_data_list)
    
    print(f"Parametric ANOVA results:")
    print(f"  F-statistic: {round(f_stat, 3)}")
    print(f"  p-value: {round(anova_p_value, 4)}")
    print(f"  Significant: {anova_p_value < alpha}")
    
    print(f"\nNonparametric Kruskal-Wallis results:")
    print(f"  H-statistic: {round(h_statistic, 3)}")
    print(f"  p-value: {round(kruskal_result.pvalue, 4)}")
    print(f"  Significant: {kruskal_result.pvalue < alpha}")
    
    # Agreement analysis
    print(f"\n=== 6. AGREEMENT ANALYSIS ===")
    anova_sig = anova_p_value < alpha
    kruskal_sig = kruskal_result.pvalue < alpha
    
    print(f"Agreement between parametric and nonparametric tests:")
    print(f"  ANOVA significant: {anova_sig}")
    print(f"  Kruskal-Wallis significant: {kruskal_sig}")
    print(f"  Agreement: {anova_sig == kruskal_sig}")
    
    if anova_sig == kruskal_sig:
        print("  ✓ Both tests agree on significance")
    else:
        print("  ⚠️  Tests disagree - check assumptions carefully")
    
    # 6. Effect size comparison
    print(f"\n=== 7. EFFECT SIZE COMPARISON ===")
    
    # Calculate eta-squared for ANOVA (simplified)
    ss_between = sum(n * (np.mean(group_data) - np.mean(np.concatenate(group_data_list)))**2 
                    for n, group_data in zip(n_per_group, group_data_list))
    ss_total = sum((x - np.mean(np.concatenate(group_data_list)))**2 
                   for group_data in group_data_list for x in group_data)
    eta_squared = ss_between / ss_total
    
    print(f"Effect size comparison:")
    print(f"  ANOVA eta-squared (η²): {round(eta_squared, 3)}")
    print(f"  Kruskal-Wallis epsilon-squared (ε²): {round(epsilon_squared, 3)}")
    print(f"  Difference: {round(abs(eta_squared - epsilon_squared), 3)}")
    
    # 7. Recommendations
    print(f"\n=== 8. RECOMMENDATIONS ===")
    
    if kruskal_result.pvalue < alpha:
        print("✓ Kruskal-Wallis test is significant")
        print("  - There are significant differences between groups")
        print("  - Use Dunn's test for pairwise comparisons")
    else:
        print("✗ Kruskal-Wallis test is not significant")
        print("  - No evidence of group differences")
    
    if abs(eta_squared - epsilon_squared) < 0.05:
        print("✓ Effect sizes are similar")
        print("  - Both parametric and nonparametric approaches are valid")
    else:
        print("⚠️  Effect sizes differ substantially")
        print("  - Consider which approach better fits your data")
    
    return {
        'kruskal_wallis': kruskal_result,
        'epsilon_squared': epsilon_squared,
        'rank_stats': rank_stats,
        'pairwise_tests': pairwise_results,
        'dunn_bonferroni': {'p_values': bonferroni_pvals, 'comparisons': comparisons},
        'dunn_holm': {'p_values': holm_pvals, 'comparisons': comparisons},
        'dunn_fdr': {'p_values': fdr_pvals, 'comparisons': comparisons},
        'anova_comparison': {'f_stat': f_stat, 'p_value': anova_p_value, 'eta_squared': eta_squared},
        'agreement': (anova_sig == kruskal_sig)
    }

# Apply comprehensive nonparametric analysis
nonparametric_results = comprehensive_nonparametric(mtcars, "cyl_factor", "mpg")

# Additional robust methods
print("\n=== ADDITIONAL ROBUST METHODS ===")

# Bootstrap confidence intervals for group differences
def bootstrap_group_diff(data, group1, group2, group_var, response_var, n_bootstrap=1000):
    """Bootstrap function to estimate confidence intervals for group differences"""
    np.random.seed(123)
    differences = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        boot_indices = np.random.choice(len(data), size=len(data), replace=True)
        boot_data = data.iloc[boot_indices]
        
        # Calculate means for each group
        mean1 = boot_data[boot_data[group_var] == group1][response_var].mean()
        mean2 = boot_data[boot_data[group_var] == group2][response_var].mean()
        
        differences.append(mean1 - mean2)
    
    # Calculate confidence intervals
    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)
    
    return {
        'differences': differences,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_diff': np.mean(differences)
    }

# Example bootstrap for 4-cylinder vs 8-cylinder
boot_result = bootstrap_group_diff(mtcars, "4-cylinder", "8-cylinder", "cyl_factor", "mpg")

print("Bootstrap 95% CI for 4-cylinder vs 8-cylinder difference:")
print(f"  Lower: {round(boot_result['ci_lower'], 3)}")
print(f"  Upper: {round(boot_result['ci_upper'], 3)}")
print(f"  Mean difference: {round(boot_result['mean_diff'], 3)}")
```

### Post Hoc Tests for Nonparametric ANOVA

```python
# Dunn's test for Kruskal-Wallis (simplified implementation)
# Note: Python doesn't have a direct equivalent to R's dunn.test package
# We'll implement a simplified version using Mann-Whitney U tests with corrections

def dunn_test_simplified(data, group_var, response_var, alpha=0.05):
    """Simplified Dunn's test using Mann-Whitney U with multiple corrections"""
    print("=== DUNN'S TEST (SIMPLIFIED) ===")
    
    groups = data[group_var].unique()
    k = len(groups)
    
    # Perform all pairwise Mann-Whitney U tests
    p_values = []
    comparisons = []
    
    for i in range(k-1):
        for j in range(i+1, k):
            group1_data = data[data[group_var] == groups[i]][response_var]
            group2_data = data[data[group_var] == groups[j]][response_var]
            
            _, p_val = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            p_values.append(p_val)
            comparisons.append(f"{groups[i]} vs {groups[j]}")
    
    # Apply Bonferroni correction
    from statsmodels.stats.multitest import multipletests
    bonferroni_pvals = multipletests(p_values, method='bonferroni')[1]
    
    print("Dunn's test results (Bonferroni-corrected):")
    for comp, p_val in zip(comparisons, bonferroni_pvals):
        print(f"{comp}: p = {round(p_val, 4)}")
    
    # Extract significant pairs
    significant_dunn = bonferroni_pvals < alpha
    
    if np.any(significant_dunn):
        print(f"\nSignificant differences (Dunn's test with Bonferroni correction):")
        for i, comp in enumerate(comparisons):
            if significant_dunn[i]:
                print(f"{comp}: p = {round(bonferroni_pvals[i], 4)}")
    else:
        print("No significant differences found with Dunn's test.")
    
    return {
        'comparisons': comparisons,
        'p_values': bonferroni_pvals,
        'significant': significant_dunn
    }

# Apply Dunn's test
dunn_result = dunn_test_simplified(mtcars, "cyl_factor", "mpg")
```

## Power Analysis

Power analysis helps determine the probability of detecting a true effect and guides sample size planning. It's essential for study design and interpreting results.

### Mathematical Foundation

**Power Definition:**
Power = $P(\text{Reject } H_0 | H_1 \text{ is true}) = 1 - \beta$

**Effect Size Measures:**
- **Cohen's f:** $f = \sqrt{\frac{\eta^2}{1 - \eta^2}}$
- **Eta-squared:** $\eta^2 = \frac{SS_{between}}{SS_{total}}$

**Noncentrality Parameter:**
```math
\lambda = n \sum_{i=1}^{k} \frac{(\mu_i - \bar{\mu})^2}{\sigma^2}
```

**Power Calculation:**
Power depends on the noncentral F-distribution:
```math
\text{Power} = P(F > F_{\alpha, k-1, N-k} | \lambda)
```

### Comprehensive Power Analysis

```python
# Comprehensive power analysis function
def comprehensive_power_analysis(data, group_var, response_var, alpha=0.05):
    """Perform comprehensive power analysis for ANOVA"""
    print("=== COMPREHENSIVE POWER ANALYSIS ===")
    
    # Get basic information
    groups = data[group_var].unique()
    k = len(groups)
    n_per_group = [sum(data[group_var] == group) for group in groups]
    total_n = sum(n_per_group)
    
    print("Study design:")
    print(f"  Number of groups (k): {k}")
    print(f"  Sample sizes per group: {n_per_group}")
    print(f"  Total sample size (N): {total_n}")
    print(f"  Significance level (α): {alpha}\n")
    
    # 1. Calculate observed effect size
    print("=== 1. OBSERVED EFFECT SIZE ===")
    
    # Perform ANOVA to get effect size
    model = ols(f'{response_var} ~ {group_var}', data=data).fit()
    anova_table = anova_lm(model, typ=2)
    
    ss_between = anova_table.loc[group_var, 'sum_sq']
    ss_total = anova_table['sum_sq'].sum()
    eta_squared = ss_between / ss_total
    
    # Calculate Cohen's f
    f_effect_size = np.sqrt(eta_squared / (1 - eta_squared))
    
    print("Observed effect sizes:")
    print(f"  Eta-squared (η²): {round(eta_squared, 3)}")
    print(f"  Cohen's f: {round(f_effect_size, 3)}")
    
    # Interpret effect size
    if f_effect_size < 0.10:
        effect_interpretation = "Small"
    elif f_effect_size < 0.25:
        effect_interpretation = "Medium"
    elif f_effect_size < 0.40:
        effect_interpretation = "Large"
    else:
        effect_interpretation = "Very large"
    
    print(f"  Interpretation: {effect_interpretation} effect\n")
    
    # 2. Current power analysis
    print("=== 2. CURRENT POWER ANALYSIS ===")
    
    # Calculate current power using our function
    current_power = power_analysis_anova(k, min(n_per_group), f_effect_size, alpha)
    
    print("Current power analysis:")
    print(f"  Effect size f: {round(f_effect_size, 3)}")
    print(f"  Sample size per group: {min(n_per_group)}")
    print(f"  Current power: {round(current_power, 3)}")
    print(f"  Type II error rate (β): {round(1 - current_power, 3)}\n")
    
    # 3. Sample size planning
    print("=== 3. SAMPLE SIZE PLANNING ===")
    
    # Calculate required sample sizes for different power levels
    power_levels = [0.8, 0.85, 0.9, 0.95]
    sample_size_results = {}
    
    print("Required sample sizes per group:")
    for power_level in power_levels:
        # Estimate required sample size (simplified approach)
        # In practice, you'd use a proper power analysis library
        required_n = estimate_sample_size_for_power(k, f_effect_size, alpha, power_level)
        sample_size_results[str(power_level)] = required_n
        print(f"  Power {power_level}: {required_n} per group")
    
    # 4. Effect size sensitivity analysis
    print("\n=== 4. EFFECT SIZE SENSITIVITY ANALYSIS ===")
    
    # Test different effect sizes
    effect_sizes_to_test = [0.1, 0.25, 0.4, 0.6]
    current_n = min(n_per_group)
    
    print(f"Power for different effect sizes (n = {current_n} per group):")
    for f_test in effect_sizes_to_test:
        power_test = power_analysis_anova(k, current_n, f_test, alpha)
        print(f"  f = {f_test}: {round(power_test, 3)}")
    
    # 5. Power curve analysis
    print("\n=== 5. POWER CURVE ANALYSIS ===")
    
    # Generate power curve data
    sample_sizes = list(range(5, 51, 5))
    power_curve = [power_analysis_anova(k, n, f_effect_size, alpha) for n in sample_sizes]
    
    print(f"Power curve (effect size f = {round(f_effect_size, 3)}):")
    for i, n in enumerate(sample_sizes):
        print(f"  n = {n} per group: power = {round(power_curve[i], 3)}")
    
    # 6. Multiple comparison correction impact
    print("\n=== 6. MULTIPLE COMPARISON IMPACT ===")
    
    # Number of pairwise comparisons
    m = k * (k - 1) / 2
    
    # Bonferroni correction
    alpha_bonferroni = alpha / m
    power_bonferroni = power_analysis_anova(k, min(n_per_group), f_effect_size, alpha_bonferroni)
    
    print("Multiple comparison impact:")
    print(f"  Number of pairwise comparisons: {m}")
    print(f"  Bonferroni α: {round(alpha_bonferroni, 4)}")
    print(f"  Power with Bonferroni correction: {round(power_bonferroni, 3)}")
    print(f"  Power loss: {round(current_power - power_bonferroni, 3)}\n")
    
    # 7. Recommendations
    print("=== 7. RECOMMENDATIONS ===")
    
    if current_power >= 0.8:
        print("✓ Current power is adequate (≥ 0.8)")
        print("  - Study has sufficient power to detect the observed effect")
    elif current_power >= 0.6:
        print("⚠️  Current power is moderate (0.6-0.8)")
        print("  - Consider increasing sample size for better power")
    else:
        print("✗ Current power is low (< 0.6)")
        print("  - Study may be underpowered")
        print("  - Consider larger sample size or different design")
    
    # Sample size recommendations
    if current_power < 0.8:
        required_n_80 = sample_size_results["0.8"]
        print("\nSample size recommendations:")
        print(f"  For 80% power: {required_n_80} per group")
        print(f"  Total sample size needed: {required_n_80 * k}")
        print(f"  Additional participants needed: {(required_n_80 * k) - total_n}")
    
    # Effect size recommendations
    print("\nEffect size considerations:")
    if f_effect_size < 0.1:
        print("  - Very small effect size detected")
        print("  - Consider if this effect is practically meaningful")
        print("  - Large sample sizes may be needed for adequate power")
    elif f_effect_size > 0.4:
        print("  - Large effect size detected")
        print("  - Current sample size likely provides adequate power")
        print("  - Effect is likely to be practically significant")
    
    return {
        'observed_effect': {'eta_squared': eta_squared, 'f': f_effect_size},
        'current_power': current_power,
        'sample_size_planning': sample_size_results,
        'power_curve': {'n': sample_sizes, 'power': power_curve},
        'multiple_comparison_impact': {
            'comparisons': m,
            'bonferroni_alpha': alpha_bonferroni,
            'bonferroni_power': power_bonferroni
        }
    }

# Apply comprehensive power analysis
power_results = comprehensive_power_analysis(mtcars, "cyl_factor", "mpg")

# Additional power analysis tools
print("\n=== ADDITIONAL POWER ANALYSIS TOOLS ===")

# Monte Carlo power simulation
def monte_carlo_power(k, n_per_group, effect_size, alpha=0.05, n_sim=1000):
    """Monte Carlo simulation for power analysis"""
    np.random.seed(123)
    
    # Simulate data and test power
    significant_tests = 0
    
    for i in range(n_sim):
        # Generate data with specified effect size
        group_means = [0, effect_size, effect_size * 2]  # Example means
        data_sim = []
        groups = []
        
        for j in range(k):
            data_sim.extend(np.random.normal(group_means[j], 1, n_per_group))
            groups.extend([j] * n_per_group)
        
        # Create DataFrame
        df_sim = pd.DataFrame({
            'y': data_sim,
            'group': pd.Categorical(groups)
        })
        
        # Perform ANOVA
        model = ols('y ~ group', data=df_sim).fit()
        anova_table = anova_lm(model, typ=2)
        p_value = anova_table.loc['group', 'PR(>F)']
        
        if p_value < alpha:
            significant_tests += 1
    
    return significant_tests / n_sim

# Example Monte Carlo power calculation
mc_power = monte_carlo_power(k=3, n_per_group=10, effect_size=0.5)
theoretical_power = power_analysis_anova(k=3, n_per_group=10, f_effect_size=0.5)
print("Monte Carlo power simulation (n = 1000):")
print(f"  Simulated power: {round(mc_power, 3)}")
print(f"  Theoretical power: {round(theoretical_power, 3)}")
```

## Practical Examples

Real-world applications demonstrate how one-way ANOVA is used across different fields. These examples show the complete analysis workflow from data preparation to interpretation.

### Example 1: Educational Research

**Research Question:** Do different teaching methods affect student performance on standardized tests?

**Study Design:** Randomized controlled trial with three teaching methods (Traditional, Interactive, Technology-based)

```python
# Simulate comprehensive educational intervention data
np.random.seed(123)
n_per_group = 25

# Generate realistic data for three teaching methods
# Method A: Traditional lecture-based (baseline)
# Method B: Interactive group learning (expected improvement)
# Method C: Technology-enhanced learning (moderate improvement)

method_a_scores = np.random.normal(mean=72, std=8, size=n_per_group)
method_b_scores = np.random.normal(mean=78, std=9, size=n_per_group)
method_c_scores = np.random.normal(mean=75, std=7, size=n_per_group)

# Create comprehensive data frame
education_data = pd.DataFrame({
    'score': np.concatenate([method_a_scores, method_b_scores, method_c_scores]),
    'method': pd.Categorical(['Traditional'] * n_per_group + ['Interactive'] * n_per_group + ['Technology'] * n_per_group),
    'student_id': range(1, n_per_group * 3 + 1),
    'study_time': np.concatenate([
        np.random.normal(5, 1, n_per_group),
        np.random.normal(6, 1, n_per_group),
        np.random.normal(5.5, 1, n_per_group)
    ])
})

# Add some realistic variation
education_data['score'] += np.random.normal(0, 2, len(education_data))

print("=== EDUCATIONAL RESEARCH EXAMPLE ===")
print("Research Question: Do different teaching methods affect student performance?")
print(f"Sample size per group: {n_per_group}")
print(f"Total participants: {len(education_data)}\n")

# 1. Descriptive Statistics
print("=== 1. DESCRIPTIVE STATISTICS ===")
desc_stats = education_data.groupby('method')['score'].agg([
    'count', 'mean', 'std', 'median', 'min', 'max'
]).rename(columns={
    'count': 'n',
    'mean': 'mean_score',
    'std': 'sd_score',
    'median': 'median_score',
    'min': 'min_score',
    'max': 'max_score'
})

desc_stats['se_score'] = desc_stats['sd_score'] / np.sqrt(desc_stats['n'])
print(desc_stats)

# 2. Assumption Checking
print("\n=== 2. ASSUMPTION CHECKING ===")
education_assumptions = comprehensive_assumption_check(education_data, "method", "score")

# 3. One-Way ANOVA
print("\n=== 3. ONE-WAY ANOVA ===")
education_model = ols('score ~ method', data=education_data).fit()
education_anova = anova_lm(education_model, typ=2)
print(education_anova)

# Extract key statistics
f_stat = education_anova.loc['method', 'F']
p_value = education_anova.loc['method', 'PR(>F)']
eta_squared = education_anova.loc['method', 'sum_sq'] / education_anova['sum_sq'].sum()

# 4. Effect Size Analysis
print("\n=== 4. EFFECT SIZE ANALYSIS ===")
f_effect_size = np.sqrt(eta_squared / (1 - eta_squared))

print("Effect size measures:")
print(f"  Eta-squared (η²): {round(eta_squared, 3)}")
print(f"  Cohen's f: {round(f_effect_size, 3)}")

# Interpret effect size
if eta_squared < 0.06:
    effect_interpretation = "Small"
elif eta_squared < 0.14:
    effect_interpretation = "Medium"
else:
    effect_interpretation = "Large"
print(f"  Interpretation: {effect_interpretation} effect")

# 5. Post Hoc Analysis
print("\n=== 5. POST HOC ANALYSIS ===")
mc = MultiComparison(education_data['score'], education_data['method'])
education_tukey = mc.tukeyhsd(alpha=0.05)
print(education_tukey)

# Extract significant differences
significant_pairs = education_tukey.pvalues < 0.05
if np.any(significant_pairs):
    print("\nSignificant pairwise differences:")
    for i, (group1, group2) in enumerate(education_tukey.groupsunique):
        if significant_pairs[i]:
            diff = education_tukey.meandiffs[i]
            p_adj = education_tukey.pvalues[i]
            print(f"  {group1} vs {group2}: diff = {round(diff, 2)}, p = {round(p_adj, 4)}")
else:
    print("No significant pairwise differences found.")

# 6. Power Analysis
print("\n=== 6. POWER ANALYSIS ===")
education_power = comprehensive_power_analysis(education_data, "method", "score")

# 7. Visualization
print("\n=== 7. VISUALIZATION ===")

# Create comprehensive visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Box plot with means
sns.boxplot(data=education_data, x='method', y='score', ax=ax1)
means = education_data.groupby('method')['score'].mean()
ax1.plot(range(len(means)), means, 'o', color='red', markersize=8, label='Mean')
ax1.set_title('Student Performance by Teaching Method\nBoxes show IQR, lines show medians, red dots show means')
ax1.set_xlabel('Teaching Method')
ax1.set_ylabel('Test Score')
ax1.legend()

# Violin plot with box plot overlay
sns.violinplot(data=education_data, x='method', y='score', ax=ax2, inner='box')
ax2.set_title('Distribution of Scores by Teaching Method')
ax2.set_xlabel('Teaching Method')
ax2.set_ylabel('Test Score')

plt.tight_layout()
plt.show()

# 8. Comprehensive Results Summary
print("\n=== 8. COMPREHENSIVE RESULTS SUMMARY ===")
print("Educational Research Results:")
print(f"  F-statistic: {round(f_stat, 3)}")
print(f"  p-value: {round(p_value, 4)}")
print(f"  Effect size (η²): {round(eta_squared, 3)}")
print(f"  Effect size interpretation: {effect_interpretation}")
print(f"  Power: {round(education_power['current_power'], 3)}")

# 9. Practical Interpretation
print("\n=== 9. PRACTICAL INTERPRETATION ===")
if p_value < 0.05:
    print("✓ Statistically significant differences found between teaching methods")
    print("  - Teaching method significantly affects student performance")
    
    if eta_squared >= 0.14:
        print("  - Large practical effect: teaching method has substantial impact")
    elif eta_squared >= 0.06:
        print("  - Medium practical effect: teaching method has moderate impact")
    else:
        print("  - Small practical effect: teaching method has limited impact")
    
    # Identify best method
    best_method = desc_stats['mean_score'].idxmax()
    print(f"  - Best performing method: {best_method}")
    
else:
    print("✗ No statistically significant differences found")
    print("  - Teaching method does not significantly affect performance")
    print("  - Consider other factors that may influence student success")

# 10. Recommendations
print("\n=== 10. RECOMMENDATIONS ===")
if p_value < 0.05 and eta_squared >= 0.06:
    print("✓ Evidence supports the effectiveness of different teaching methods")
    print("  - Consider implementing the best-performing method")
    print("  - Conduct follow-up studies to confirm findings")
    print("  - Consider cost-benefit analysis of different methods")
else:
    print("⚠️  Limited evidence for teaching method effectiveness")
    print("  - Consider larger sample sizes for future studies")
    print("  - Investigate other factors affecting student performance")
    print("  - Consider qualitative research to understand student experiences")
```

### Example 2: Clinical Trial

**Research Question:** Do different drug treatments reduce blood pressure in hypertensive patients?

**Study Design:** Randomized controlled trial with three treatment groups (Placebo, Drug A, Drug B)

```python
# Simulate comprehensive clinical trial data
np.random.seed(456)
n_per_treatment = 30

# Generate realistic clinical data
# Placebo: minimal effect (baseline)
# Drug A: moderate blood pressure reduction
# Drug B: strong blood pressure reduction

placebo_bp = np.random.normal(loc=145, scale=12, size=n_per_treatment)
drug_a_bp = np.random.normal(loc=135, scale=11, size=n_per_treatment)
drug_b_bp = np.random.normal(loc=125, scale=10, size=n_per_treatment)

# Create comprehensive data frame
clinical_data = pd.DataFrame({
    'blood_pressure': np.concatenate([placebo_bp, drug_a_bp, drug_b_bp]),
    'treatment': pd.Categorical(['Placebo'] * n_per_treatment + ['Drug A'] * n_per_treatment + ['Drug B'] * n_per_treatment),
    'patient_id': range(1, n_per_treatment * 3 + 1),
    'age': np.concatenate([
        np.random.normal(55, 8, n_per_treatment),
        np.random.normal(57, 9, n_per_treatment),
        np.random.normal(54, 7, n_per_treatment)
    ]),
    'baseline_bp': np.concatenate([
        np.random.normal(150, 15, n_per_treatment),
        np.random.normal(148, 14, n_per_treatment),
        np.random.normal(152, 16, n_per_treatment)
    ])
})

# Calculate change from baseline
clinical_data['bp_change'] = clinical_data['baseline_bp'] - clinical_data['blood_pressure']

print("=== CLINICAL TRIAL EXAMPLE ===")
print("Research Question: Do different drug treatments reduce blood pressure?")
print(f"Sample size per group: {n_per_treatment}")
print(f"Total participants: {len(clinical_data)}\n")

# 1. Descriptive Statistics
print("=== 1. DESCRIPTIVE STATISTICS ===")
clinical_desc = clinical_data.groupby('treatment').agg(
    n=('blood_pressure', 'count'),
    mean_bp=('blood_pressure', 'mean'),
    sd_bp=('blood_pressure', 'std'),
    mean_change=('bp_change', 'mean'),
    sd_change=('bp_change', 'std')
)
clinical_desc['se_bp'] = clinical_desc['sd_bp'] / np.sqrt(clinical_desc['n'])
print(clinical_desc)

# 2. Assumption Checking
print("\n=== 2. ASSUMPTION CHECKING ===")
clinical_assumptions = comprehensive_assumption_check(clinical_data, "treatment", "blood_pressure")

# 3. One-Way ANOVA
print("\n=== 3. ONE-WAY ANOVA ===")
clinical_model = ols('blood_pressure ~ treatment', data=clinical_data).fit()
clinical_anova = anova_lm(clinical_model, typ=2)
print(clinical_anova)

# Extract key statistics
clinical_f_stat = clinical_anova.loc['treatment', 'F']
clinical_p_value = clinical_anova.loc['treatment', 'PR(>F)']
clinical_eta_squared = clinical_anova.loc['treatment', 'sum_sq'] / clinical_anova['sum_sq'].sum()

# 4. Effect Size and Power
print("\n=== 4. EFFECT SIZE AND POWER ===")
clinical_f_effect = np.sqrt(clinical_eta_squared / (1 - clinical_eta_squared))
clinical_power = comprehensive_power_analysis(clinical_data, "treatment", "blood_pressure")
print(f"Effect size (η²): {round(clinical_eta_squared, 3)}")
print(f"Cohen's f: {round(clinical_f_effect, 3)}")
print(f"Power: {round(clinical_power['current_power'], 3)}")

# 5. Post Hoc Analysis
print("\n=== 5. POST HOC ANALYSIS ===")
mc = MultiComparison(clinical_data['blood_pressure'], clinical_data['treatment'])
clinical_tukey = mc.tukeyhsd(alpha=0.05)
print(clinical_tukey)

# 6. Nonparametric Alternative
print("\n=== 6. NONPARAMETRIC ALTERNATIVE ===")
groups = [clinical_data[clinical_data['treatment'] == t]['blood_pressure'] for t in clinical_data['treatment'].unique()]
clinical_kruskal = stats.kruskal(*groups)
print(clinical_kruskal)

# 7. Clinical Significance
print("\n=== 7. CLINICAL SIGNIFICANCE ===")
def effect_category(mean_change):
    if mean_change >= 10:
        return "Large clinical effect"
    elif mean_change >= 5:
        return "Moderate clinical effect"
    elif mean_change >= 2:
        return "Small clinical effect"
    else:
        return "No clinical effect"
clinical_desc['clinically_meaningful'] = clinical_desc['mean_change'] >= 5
clinical_desc['effect_category'] = clinical_desc['mean_change'].apply(effect_category)
print(clinical_desc[['mean_change', 'clinically_meaningful', 'effect_category']])

# 8. Safety Analysis
print("\n=== 8. SAFETY ANALYSIS ===")
clinical_data['hypotension'] = clinical_data['blood_pressure'] < 90
safety_summary = clinical_data.groupby('treatment').agg(
    n_hypotension=('hypotension', 'sum'),
    pct_hypotension=('hypotension', 'mean')
)
safety_summary['pct_hypotension'] *= 100
print("Safety analysis (hypotension < 90 mmHg):")
print(safety_summary)

# 9. Comprehensive Results
print("\n=== 9. COMPREHENSIVE RESULTS ===")
print("Clinical Trial Results:")
print(f"  F-statistic: {round(clinical_f_stat, 3)}")
print(f"  p-value: {round(clinical_p_value, 4)}")
print(f"  Effect size (η²): {round(clinical_eta_squared, 3)}")
print(f"  Power: {round(clinical_power['current_power'], 3)}")

# 10. Clinical Recommendations
print("\n=== 10. CLINICAL RECOMMENDATIONS ===")
if clinical_p_value < 0.05:
    best_treatment = clinical_desc['mean_bp'].idxmin()
    print("✓ Statistically significant treatment effects found")
    print(f"  - Best treatment: {best_treatment}")
    print("  - Consider safety profile when choosing treatment")
    print("  - Monitor for adverse effects")
else:
    print("✗ No statistically significant treatment effects")
    print("  - Consider larger sample size or different endpoints")
    print("  - Investigate patient compliance and adherence")
```

### Example 3: Quality Control

**Research Question:** Do different manufacturing machines produce products with consistent quality?

**Study Design:** Quality control study comparing three production machines

```python
# Simulate comprehensive quality control data
np.random.seed(789)
n_per_machine = 20

# Generate realistic quality data
# Machine A: High precision, consistent output
# Machine B: Moderate precision, some variation
# Machine C: Lower precision, more variation

machine_a_output = np.random.normal(loc=100, scale=2, size=n_per_machine)
machine_b_output = np.random.normal(loc=98, scale=3.5, size=n_per_machine)
machine_c_output = np.random.normal(loc=102, scale=4.5, size=n_per_machine)

# Create comprehensive data frame
quality_data = pd.DataFrame({
    'output': np.concatenate([machine_a_output, machine_b_output, machine_c_output]),
    'machine': pd.Categorical(['Machine A'] * n_per_machine + ['Machine B'] * n_per_machine + ['Machine C'] * n_per_machine),
    'batch_id': range(1, n_per_machine * 3 + 1),
    'temperature': np.concatenate([
        np.random.normal(25, 2, n_per_machine),
        np.random.normal(26, 3, n_per_machine),
        np.random.normal(24, 2.5, n_per_machine)
    ]),
    'humidity': np.concatenate([
        np.random.normal(50, 5, n_per_machine),
        np.random.normal(52, 6, n_per_machine),
        np.random.normal(48, 4, n_per_machine)
    ])
})

# Add quality control specifications
quality_data['within_spec'] = np.abs(quality_data['output'] - 100) <= 5
quality_data['defect_rate'] = np.where(quality_data['within_spec'], 0, 1)

print("=== QUALITY CONTROL EXAMPLE ===")
print("Research Question: Do different machines produce consistent quality?")
print(f"Sample size per machine: {n_per_machine}")
print(f"Total measurements: {len(quality_data)}\n")

# 1. Descriptive Statistics
print("=== 1. DESCRIPTIVE STATISTICS ===")
quality_desc = quality_data.groupby('machine').agg(
    n=('output', 'count'),
    mean_output=('output', 'mean'),
    sd_output=('output', 'std'),
    cv=('output', lambda x: np.std(x) / np.mean(x) * 100),
    defect_rate=('defect_rate', 'mean'),
    within_spec_rate=('within_spec', 'mean')
)
quality_desc['defect_rate'] *= 100
quality_desc['within_spec_rate'] *= 100
print(quality_desc)

# 2. Assumption Checking
print("\n=== 2. ASSUMPTION CHECKING ===")
quality_assumptions = comprehensive_assumption_check(quality_data, "machine", "output")

# 3. One-Way ANOVA
print("\n=== 3. ONE-WAY ANOVA ===")
quality_model = ols('output ~ machine', data=quality_data).fit()
quality_anova = anova_lm(quality_model, typ=2)
print(quality_anova)

# Extract key statistics
quality_f_stat = quality_anova.loc['machine', 'F']
quality_p_value = quality_anova.loc['machine', 'PR(>F)']
quality_eta_squared = quality_anova.loc['machine', 'sum_sq'] / quality_anova['sum_sq'].sum()

# 4. Post Hoc Analysis
print("\n=== 4. POST HOC ANALYSIS ===")
mc = MultiComparison(quality_data['output'], quality_data['machine'])
quality_tukey = mc.tukeyhsd(alpha=0.05)
print(quality_tukey)

# 5. Quality Control Analysis
print("\n=== 5. QUALITY CONTROL ANALYSIS ===")
quality_data['deviation_from_target'] = quality_data['output'] - 100
capability_analysis = quality_data.groupby('machine').agg(
    mean_deviation=('deviation_from_target', 'mean'),
    sd_deviation=('deviation_from_target', 'std')
)
capability_analysis['process_capability'] = 6 / (6 * capability_analysis['sd_deviation'])  # Cp index
capability_analysis['process_capability_centered'] = (6 - np.abs(capability_analysis['mean_deviation'])) / (6 * capability_analysis['sd_deviation'])  # Cpk index
print("Process capability analysis:")
print(capability_analysis)

# 6. Economic Impact Analysis
print("\n=== 6. ECONOMIC IMPACT ANALYSIS ===")
quality_data['cost_per_unit'] = np.where(quality_data['within_spec'], 10, 25)
economic_analysis = quality_data.groupby('machine').agg(
    total_cost=('cost_per_unit', 'sum'),
    avg_cost_per_unit=('cost_per_unit', 'mean'),
    defect_cost=('cost_per_unit', lambda x: x[~quality_data.loc[x.index, 'within_spec']].sum()),
    efficiency_score=('defect_rate', lambda x: (len(x) - x.sum()) / len(x) * 100)
)
print("Economic analysis:")
print(economic_analysis)

# 7. Visualization
print("\n=== 7. VISUALIZATION ===")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Box plot with target and spec limits
sns.boxplot(data=quality_data, x='machine', y='output', ax=ax1)
ax1.axhline(100, linestyle='--', color='red', label='Target')
ax1.axhline(95, linestyle=':', color='orange', label='Spec Limit')
ax1.axhline(105, linestyle=':', color='orange')
means = quality_data.groupby('machine')['output'].mean()
ax1.plot(range(len(means)), means, 'o', color='blue', markersize=8, label='Mean')
ax1.set_title('Output Quality by Machine\nRed line = target, Orange lines = specification limits')
ax1.set_xlabel('Machine')
ax1.set_ylabel('Output Quality')
ax1.legend()

# Box plot of deviation from target
sns.boxplot(data=quality_data, x='machine', y='deviation_from_target', ax=ax2)
ax2.axhline(0, linestyle='--', color='red', label='Target')
ax2.set_title('Deviation from Target by Machine')
ax2.set_xlabel('Machine')
ax2.set_ylabel('Deviation from Target (100)')
ax2.legend()

plt.tight_layout()
plt.show()

# 8. Quality Control Recommendations
print("\n=== 8. QUALITY CONTROL RECOMMENDATIONS ===")
best_machine = quality_desc['within_spec_rate'].idxmax()
worst_machine = quality_desc['within_spec_rate'].idxmin()
print("Quality control recommendations:")
print(f"  - Best performing machine: {best_machine}")
print(f"  - Machine needing improvement: {worst_machine}")

if quality_p_value < 0.05:
    print("  - Significant differences in machine performance detected")
    print("  - Consider machine maintenance or replacement for underperforming units")
    print("  - Implement quality control procedures for all machines")
else:
    print("  - No significant differences in machine performance")
    print("  - All machines appear to be operating within acceptable parameters")

# 9. Process Improvement Suggestions
print("\n=== 9. PROCESS IMPROVEMENT SUGGESTIONS ===")
print("Process improvement recommendations:")
for machine, defect_rate in zip(quality_desc.index, quality_desc['defect_rate']):
    if defect_rate > 10:
        print(f"  - {machine}: High defect rate ({round(defect_rate, 1)}%) - needs immediate attention")
    elif defect_rate > 5:
        print(f"  - {machine}: Moderate defect rate ({round(defect_rate, 1)}%) - monitor closely")
    else:
        print(f"  - {machine}: Low defect rate ({round(defect_rate, 1)}%) - performing well")
```

## Best Practices

Following best practices ensures robust statistical analysis and valid conclusions. These guidelines help researchers make appropriate methodological choices and interpret results correctly.

### Comprehensive Test Selection Guidelines

```python
# Comprehensive function to help choose appropriate ANOVA test
def comprehensive_test_selection(data, group_var, response_var, alpha=0.05):
    """Comprehensive function to help choose appropriate ANOVA test"""
    print("=== COMPREHENSIVE ANOVA TEST SELECTION ===")
    
    # Get basic information
    groups = data[group_var].unique()
    k = len(groups)
    n_per_group = [sum(data[group_var] == group) for group in groups]
    total_n = sum(n_per_group)
    
    print("Study characteristics:")
    print(f"  Number of groups: {k}")
    print(f"  Sample sizes per group: {n_per_group}")
    print(f"  Total sample size: {total_n}")
    print(f"  Significance level: {alpha}\n")
    
    # 1. Check all assumptions comprehensively
    print("=== 1. ASSUMPTION ASSESSMENT ===")
    
    assumption_results = comprehensive_assumption_check(data, group_var, response_var, alpha)
    
    # 2. Evaluate assumption violations
    print("\n=== 2. ASSUMPTION VIOLATION EVALUATION ===")
    
    violations = assumption_results['violations']
    violation_details = assumption_results['violation_details']
    
    print(f"Total assumption violations: {violations}")
    if violations > 0:
        print(f"Violated assumptions: {', '.join(violation_details)}")
    
    # 3. Sample size considerations
    print("\n=== 3. SAMPLE SIZE CONSIDERATIONS ===")
    
    min_n = min(n_per_group)
    max_n = max(n_per_group)
    balanced = len(set(n_per_group)) == 1
    
    print("Sample size analysis:")
    print(f"  Minimum sample size per group: {min_n}")
    print(f"  Maximum sample size per group: {max_n}")
    print(f"  Balanced design: {balanced}")
    
    if min_n < 10:
        print("  ⚠️  Small sample sizes - consider nonparametric alternatives")
    elif min_n < 30:
        print("  ⚠️  Moderate sample sizes - check normality carefully")
    else:
        print("  ✓ Adequate sample sizes for parametric tests")
    
    # 4. Effect size considerations
    print("\n=== 4. EFFECT SIZE CONSIDERATIONS ===")
    
    # Perform ANOVA to get effect size
    model = ols(f'{response_var} ~ {group_var}', data=data).fit()
    anova_table = anova_lm(model, typ=2)
    eta_squared = anova_table.loc[group_var, 'sum_sq'] / anova_table['sum_sq'].sum()
    
    print(f"Observed effect size (η²): {round(eta_squared, 3)}")
    
    if eta_squared < 0.01:
        print("  ⚠️  Very small effect size - consider practical significance")
    elif eta_squared < 0.06:
        print("  ⚠️  Small effect size - may not be practically meaningful")
    elif eta_squared < 0.14:
        print("  ✓ Medium effect size - likely practically meaningful")
    else:
        print("  ✓ Large effect size - clearly practically meaningful")
    
    # 5. Decision tree for test selection
    print("\n=== 5. TEST SELECTION DECISION TREE ===")
    
    # Check specific assumptions
    normality_ok = all(assumption_results['normality']['group_tests'][str(g)]['p_value'] >= alpha 
                      for g in groups)
    homogeneity_ok = assumption_results['homoscedasticity']['levene']['p_value'] >= alpha
    independence_ok = assumption_results['independence']['independent']
    outliers_ok = len([item for sublist in assumption_results['outliers'].values() 
                      for item in sublist]) <= total_n * 0.1
    
    print("Assumption status:")
    print(f"  Normality: {normality_ok}")
    print(f"  Homogeneity of variance: {homogeneity_ok}")
    print(f"  Independence: {independence_ok}")
    print(f"  Outliers acceptable: {outliers_ok}")
    
    # Decision logic
    print("\n=== 6. RECOMMENDED APPROACH ===")
    
    if normality_ok and homogeneity_ok and independence_ok and outliers_ok:
        print("✓ RECOMMENDATION: Standard one-way ANOVA")
        print("  - All assumptions are met")
        print("  - Most powerful parametric approach")
        print("  - Use Tukey's HSD for post hoc tests")
    elif normality_ok and not homogeneity_ok and independence_ok:
        print("✓ RECOMMENDATION: Welch's ANOVA")
        print("  - Data is normal but variances are unequal")
        print("  - More robust to variance heterogeneity")
        print("  - Use Games-Howell for post hoc tests")
    elif not normality_ok and min_n >= 15:
        print("✓ RECOMMENDATION: Kruskal-Wallis test")
        print("  - Data is not normally distributed")
        print("  - Nonparametric alternative")
        print("  - Use Dunn's test for post hoc comparisons")
    elif min_n < 15:
        print("✓ RECOMMENDATION: Kruskal-Wallis test")
        print("  - Small sample sizes")
        print("  - Nonparametric approach is more robust")
        print("  - Use Dunn's test for post hoc comparisons")
    else:
        print("⚠️  RECOMMENDATION: Multiple approaches")
        print("  - Complex assumption violations")
        print("  - Consider both parametric and nonparametric approaches")
        print("  - Report results from both methods")
    
    # 7. Additional considerations
    print("\n=== 7. ADDITIONAL CONSIDERATIONS ===")
    
    # Power analysis
    f_effect_size = np.sqrt(eta_squared / (1 - eta_squared))
    current_power = power_analysis_anova(k, min_n, f_effect_size, alpha)
    
    print("Power analysis:")
    print(f"  Current power: {round(current_power, 3)}")
    
    if current_power < 0.8:
        print("  ⚠️  Low power - consider larger sample size")
        required_n = estimate_sample_size_for_power(k, f_effect_size, alpha, 0.8)
        print(f"  Required sample size per group for 80% power: {required_n}")
    else:
        print("  ✓ Adequate power for detecting effects")
    
    # Multiple comparison considerations
    m = k * (k - 1) / 2
    print("\nMultiple comparison considerations:")
    print(f"  Number of pairwise comparisons: {m}")
    print(f"  Family-wise error rate without correction: {round(1 - (1 - alpha)**m, 4)}")
    
    if m > 6:
        print("  ⚠️  Many comparisons - use conservative correction methods")
    else:
        print("  ✓ Reasonable number of comparisons")
    
    # 8. Reporting recommendations
    print("\n=== 8. REPORTING RECOMMENDATIONS ===")
    
    print("Essential elements to report:")
    print("  - Descriptive statistics for each group")
    print("  - Assumption checking results")
    print("  - Test statistic and p-value")
    print("  - Effect size measures")
    print("  - Post hoc test results (if significant)")
    print("  - Power analysis results")
    print("  - Practical significance assessment")
    
    return {
        'assumptions': assumption_results,
        'effect_size': eta_squared,
        'power': current_power,
        'violations': violations,
        'violation_details': violation_details,
        'recommendation': ("Standard ANOVA" if normality_ok and homogeneity_ok and independence_ok and outliers_ok
                          else "Welch's ANOVA" if normality_ok and not homogeneity_ok and independence_ok
                          else "Kruskal-Wallis")
    }

# Apply comprehensive test selection
test_selection_results = comprehensive_test_selection(mtcars, "cyl_factor", "mpg")
```

### Comprehensive Reporting Guidelines

Proper reporting of ANOVA results is essential for transparency and reproducibility. This section provides a comprehensive template for reporting one-way ANOVA analyses.

```python
# Comprehensive function to generate detailed ANOVA report
def generate_comprehensive_anova_report(data, group_var, response_var, alpha=0.05):
    """Generate comprehensive ANOVA report"""
    from datetime import datetime
    
    print("=== COMPREHENSIVE ONE-WAY ANOVA REPORT ===")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Study Information
    print("=== STUDY INFORMATION ===")
    groups = data[group_var].unique()
    k = len(groups)
    n_per_group = [sum(data[group_var] == group) for group in groups]
    total_n = sum(n_per_group)
    
    print("Research Design: One-way ANOVA")
    print(f"Number of groups: {k}")
    print(f"Group names: {', '.join(groups)}")
    print(f"Sample sizes per group: {', '.join(map(str, n_per_group))}")
    print(f"Total sample size: {total_n}")
    print(f"Significance level (α): {alpha}\n")
    
    # 1. Descriptive Statistics
    print("=== 1. DESCRIPTIVE STATISTICS ===")
    
    desc_stats = data.groupby(group_var)[response_var].agg([
        'count', 'mean', 'std', 'median', 'min', 'max'
    ]).rename(columns={
        'count': 'n',
        'mean': 'mean_val',
        'std': 'sd',
        'median': 'median_val',
        'min': 'min_val',
        'max': 'max_val'
    })
    desc_stats['se'] = desc_stats['sd'] / np.sqrt(desc_stats['n'])
    
    print(desc_stats)
    
    # Overall statistics
    overall_stats = {
        'n': len(data),
        'mean': data[response_var].mean(),
        'sd': data[response_var].std(),
        'median': data[response_var].median()
    }
    
    print("\nOverall statistics:")
    for key, value in overall_stats.items():
        print(f"  {key}: {round(value, 3)}")
    print()
    
    # 2. Assumption Checking
    print("=== 2. ASSUMPTION CHECKING ===")
    
    assumption_results = comprehensive_assumption_check(data, group_var, response_var, alpha)
    
    print(f"Assumption violations: {assumption_results['violations']}")
    if assumption_results['violations'] > 0:
        print(f"Violated assumptions: {', '.join(assumption_results['violation_details'])}")
    else:
        print("All assumptions met ✓")
    print()
    
    # 3. Primary Analysis
    print("=== 3. PRIMARY ANALYSIS ===")
    
    # Perform ANOVA
    model = ols(f'{response_var} ~ {group_var}', data=data).fit()
    anova_table = anova_lm(model, typ=2)
    
    # Extract key statistics
    f_stat = anova_table.loc[group_var, 'F']
    p_value = anova_table.loc[group_var, 'PR(>F)']
    df_between = anova_table.loc[group_var, 'df']
    df_within = anova_table.loc['Residual', 'df']
    ss_between = anova_table.loc[group_var, 'sum_sq']
    ss_within = anova_table.loc['Residual', 'sum_sq']
    ms_between = anova_table.loc[group_var, 'mean_sq']
    ms_within = anova_table.loc['Residual', 'mean_sq']
    
    # Critical F-value
    f_critical = f.ppf(1 - alpha, df_between, df_within)
    
    print("ANOVA Results:")
    print(f"  F-statistic: {round(f_stat, 3)}")
    print(f"  Degrees of freedom: {df_between}, {df_within}")
    print(f"  p-value: {round(p_value, 4)}")
    print(f"  Critical F-value: {round(f_critical, 3)}")
    print(f"  Significant: {p_value < alpha}\n")
    
    # 4. Effect Size Analysis
    print("=== 4. EFFECT SIZE ANALYSIS ===")
    
    # Calculate effect sizes
    ss_total = ss_between + ss_within
    eta_squared = ss_between / ss_total
    omega_squared = (ss_between - (df_between * ms_within)) / (ss_total + ms_within)
    cohens_f = np.sqrt(eta_squared / (1 - eta_squared))
    
    print("Effect size measures:")
    print(f"  Eta-squared (η²): {round(eta_squared, 3)}")
    print(f"  Omega-squared (ω²): {round(omega_squared, 3)}")
    print(f"  Cohen's f: {round(cohens_f, 3)}")
    
    # Interpret effect size
    if eta_squared < 0.01:
        effect_interpretation = "Negligible"
    elif eta_squared < 0.06:
        effect_interpretation = "Small"
    elif eta_squared < 0.14:
        effect_interpretation = "Medium"
    else:
        effect_interpretation = "Large"
    
    print(f"  Effect size interpretation: {effect_interpretation}\n")
    
    # 5. Post Hoc Analysis
    print("=== 5. POST HOC ANALYSIS ===")
    
    if p_value < alpha:
        print("Post hoc tests performed (ANOVA was significant):")
        
        # Tukey's HSD
        mc = MultiComparison(data[response_var], data[group_var])
        tukey_result = mc.tukeyhsd(alpha=alpha)
        
        significant_pairs = tukey_result.pvalues < alpha
        if np.any(significant_pairs):
            print("Significant pairwise differences (Tukey's HSD):")
            for i, (group1, group2) in enumerate(tukey_result.groupsunique):
                if significant_pairs[i]:
                    diff = tukey_result.meandiffs[i]
                    p_adj = tukey_result.pvalues[i]
                    print(f"  {group1} vs {group2}: diff = {round(diff, 3)}, p = {round(p_adj, 4)}")
        else:
            print("No significant pairwise differences found with Tukey's HSD.")
    else:
        print("Post hoc tests not performed (ANOVA was not significant).")
    print()
    
    # 6. Power Analysis
    print("=== 6. POWER ANALYSIS ===")
    
    current_power = power_analysis_anova(k, min(n_per_group), cohens_f, alpha)
    
    print("Power analysis:")
    print(f"  Current power: {round(current_power, 3)}")
    print(f"  Type II error rate (β): {round(1 - current_power, 3)}")
    
    if current_power < 0.8:
        required_n = estimate_sample_size_for_power(k, cohens_f, alpha, 0.8)
        print(f"  Required sample size per group for 80% power: {required_n}")
    print()
    
    # 7. Nonparametric Alternative
    print("=== 7. NONPARAMETRIC ALTERNATIVE ===")
    
    groups_data = [data[data[group_var] == group][response_var] for group in groups]
    kruskal_result = stats.kruskal(*groups_data)
    
    print("Kruskal-Wallis test results:")
    print(f"  H-statistic: {round(kruskal_result.statistic, 3)}")
    print(f"  p-value: {round(kruskal_result.pvalue, 4)}")
    print(f"  Significant: {kruskal_result.pvalue < alpha}")
    
    # Agreement between parametric and nonparametric tests
    agreement = (p_value < alpha) == (kruskal_result.pvalue < alpha)
    print(f"  Agreement with parametric ANOVA: {agreement}\n")
    
    # 8. Confidence Intervals
    print("=== 8. CONFIDENCE INTERVALS ===")
    
    # Confidence intervals for group means
    print("95% Confidence intervals for group means:")
    for group in groups:
        group_data = data[data[group_var] == group][response_var]
        n = len(group_data)
        mean_val = group_data.mean()
        se = group_data.std() / np.sqrt(n)
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        ci_lower = mean_val - t_critical * se
        ci_upper = mean_val + t_critical * se
        print(f"  {group}: {round(mean_val, 2)} [{round(ci_lower, 2)}, {round(ci_upper, 2)}]")
    print()
    
    # 9. Practical Significance
    print("=== 9. PRACTICAL SIGNIFICANCE ===")
    
    print("Practical significance assessment:")
    if p_value < alpha:
        print("  ✓ Statistically significant differences detected")
        
        if eta_squared >= 0.14:
            print("  ✓ Large practical effect - results are practically meaningful")
        elif eta_squared >= 0.06:
            print("  ⚠️  Medium practical effect - consider context for interpretation")
        else:
            print("  ⚠️  Small practical effect - may not be practically meaningful")
        
        # Identify best performing group
        best_group = desc_stats['mean_val'].idxmax()
        print(f"  - Best performing group: {best_group}")
        
    else:
        print("  ✗ No statistically significant differences detected")
        print("  - Consider power analysis and effect size")
        print("  - May need larger sample size or different design")
    print()
    
    # 10. Conclusions and Recommendations
    print("=== 10. CONCLUSIONS AND RECOMMENDATIONS ===")
    
    print("Primary conclusion:")
    if p_value < alpha:
        print(f"  Reject the null hypothesis (p < {alpha})")
        print("  There are significant differences between group means")
    else:
        print(f"  Fail to reject the null hypothesis (p >= {alpha})")
        print("  There is insufficient evidence of differences between group means")
    
    print("\nRecommendations:")
    if p_value < alpha and eta_squared >= 0.06:
        print("  - Results support the effectiveness of the intervention/factor")
        print("  - Consider implementing the best-performing condition")
        print("  - Conduct follow-up studies to confirm findings")
    elif p_value < alpha and eta_squared < 0.06:
        print("  - Statistically significant but small practical effect")
        print("  - Consider whether the effect is meaningful in practice")
        print("  - May need larger sample size for adequate power")
    else:
        print("  - No significant differences detected")
        print("  - Consider increasing sample size for future studies")
        print("  - Investigate other factors that may influence the outcome")
    
    # 11. Limitations
    print("\n=== 11. LIMITATIONS ===")
    
    print("Study limitations:")
    if current_power < 0.8:
        print("  - Low statistical power may have missed true effects")
    if assumption_results['violations'] > 0:
        print("  - Some ANOVA assumptions were violated")
    if min(n_per_group) < 30:
        print("  - Small sample sizes may affect robustness")
    if len(set(n_per_group)) != 1:
        print("  - Unbalanced design may affect power")
    
    print("\n=== END OF REPORT ===")
    
    return {
        'descriptive_stats': desc_stats,
        'anova_results': {'f_stat': f_stat, 'p_value': p_value, 'df': [df_between, df_within]},
        'effect_sizes': {'eta_squared': eta_squared, 'omega_squared': omega_squared, 'cohens_f': cohens_f},
        'power': current_power,
        'assumptions': assumption_results,
        'kruskal_wallis': kruskal_result,
        'significant': p_value < alpha
    }

# Generate comprehensive report
comprehensive_report = generate_comprehensive_anova_report(mtcars, "cyl_factor", "mpg")
```

## Exercises

These exercises provide hands-on practice with one-way ANOVA concepts and techniques. Each exercise builds upon the previous ones and includes comprehensive analysis workflows.

### Exercise 1: Basic One-Way ANOVA with Transmission Types

**Objective:** Perform a complete one-way ANOVA analysis comparing MPG across different transmission types.

**Dataset:** Use the mtcars dataset and create a transmission factor variable.

**Tasks:**
1. Create a transmission factor variable (automatic vs manual)
2. Perform descriptive statistics for each transmission type
3. Conduct one-way ANOVA
4. Calculate and interpret effect sizes
5. Perform post hoc tests if significant
6. Create appropriate visualizations

**Expected Learning Outcomes:**
- Understand the complete ANOVA workflow
- Interpret ANOVA results correctly
- Calculate and interpret effect sizes
- Perform post hoc analysis appropriately

### Exercise 2: Comprehensive Assumption Checking

**Objective:** Conduct thorough assumption checking and recommend appropriate statistical approaches.

**Dataset:** Use the built-in `PlantGrowth` dataset or create simulated data with known violations.

**Tasks:**
1. Check normality for each group using multiple tests
2. Test homogeneity of variance using different methods
3. Assess independence and identify outliers
4. Evaluate the impact of assumption violations
5. Recommend appropriate alternative approaches
6. Compare parametric and nonparametric results

**Expected Learning Outcomes:**
- Master assumption checking procedures
- Understand when to use alternative methods
- Interpret assumption violation impacts
- Make informed methodological decisions

### Exercise 3: Advanced Post Hoc Analysis

**Objective:** Compare multiple post hoc methods and understand their differences.

**Dataset:** Create simulated data with 4 groups and known differences.

**Tasks:**
1. Perform ANOVA and confirm significance
2. Apply Tukey's HSD test
3. Apply Bonferroni correction
4. Apply Holm's method
5. Apply FDR correction
6. Compare results across methods
7. Create visualization of post hoc results

**Expected Learning Outcomes:**
- Understand different multiple comparison methods
- Compare conservative vs. liberal approaches
- Interpret post hoc results correctly
- Choose appropriate methods for different scenarios

### Exercise 4: Effect Size Analysis and Interpretation

**Objective:** Calculate and interpret various effect size measures comprehensively.

**Dataset:** Use the `ToothGrowth` dataset or create data with different effect sizes.

**Tasks:**
1. Calculate eta-squared, omega-squared, and Cohen's f
2. Compute confidence intervals for effect sizes
3. Compare effect sizes across different datasets
4. Assess practical significance
5. Create effect size visualizations
6. Provide comprehensive interpretation

**Expected Learning Outcomes:**
- Master effect size calculations
- Understand effect size interpretation
- Distinguish statistical vs. practical significance
- Communicate effect sizes effectively

### Exercise 5: Power Analysis and Sample Size Planning

**Objective:** Conduct comprehensive power analysis for ANOVA designs.

**Dataset:** Use existing data or create scenarios with different effect sizes.

**Tasks:**
1. Calculate current power for existing data
2. Determine required sample sizes for different power levels
3. Create power curves for different effect sizes
4. Assess the impact of multiple comparisons on power
5. Plan sample sizes for future studies
6. Consider cost-benefit analysis

**Expected Learning Outcomes:**
- Master power analysis techniques
- Plan sample sizes effectively
- Understand power trade-offs
- Design efficient studies


### Exercise 6: Real-World Application

**Objective:** Apply one-way ANOVA to a real-world research question.

**Research Question:** Do different study environments (library, coffee shop, home) affect student concentration levels?

**Tasks:**
1. Design a study to address the research question
2. Simulate realistic data based on the design
3. Perform complete ANOVA analysis
4. Check all assumptions thoroughly
5. Calculate effect sizes and power
6. Provide practical recommendations
7. Write a comprehensive report

**Expected Learning Outcomes:**
- Apply ANOVA to real research questions
- Design appropriate studies
- Conduct complete statistical analysis
- Communicate results effectively

### Exercise 7: Robust Methods and Alternatives

**Objective:** Explore robust alternatives to traditional ANOVA.

**Dataset:** Create data with various assumption violations.

**Tasks:**
1. Create data with normality violations
2. Create data with variance heterogeneity
3. Apply traditional ANOVA
4. Apply Welch's ANOVA
5. Apply Kruskal-Wallis test
6. Apply bootstrap methods
7. Compare results across methods

**Expected Learning Outcomes:**
- Understand robust alternatives
- Choose appropriate methods for violated assumptions
- Compare parametric and nonparametric approaches
- Apply modern statistical techniques


### Exercise 8: Advanced Visualization and Reporting

**Objective:** Create comprehensive visualizations and reports for ANOVA results.

**Dataset:** Use any of the previous datasets or create new data.

**Tasks:**
1. Create publication-quality visualizations
2. Generate comprehensive statistical reports
3. Create interactive visualizations (if possible)
4. Design presentation materials
5. Write statistical interpretation
6. Provide practical recommendations

**Expected Learning Outcomes:**
- Create effective visualizations
- Write comprehensive reports
- Communicate statistical results clearly
- Provide actionable recommendations


### Exercise Hints and Solutions

**General Tips:**
- Always start with descriptive statistics
- Check assumptions before interpreting results
- Consider both statistical and practical significance
- Use appropriate visualizations for your data
- Report results comprehensively and transparently

**Common Pitfalls to Avoid:**
- Ignoring assumption violations
- Focusing only on p-values
- Neglecting effect size interpretation
- Performing post hoc tests without significant ANOVA
- Not considering practical significance

**Advanced Considerations:**
- Multiple comparison corrections
- Power analysis for study design
- Robust alternatives for assumption violations
- Effect size confidence intervals
- Practical significance assessment

### Expected Learning Progression

**Beginner Level (Exercises 1-3):**
- Basic ANOVA workflow
- Assumption checking
- Post hoc analysis

**Intermediate Level (Exercises 4-6):**
- Effect size analysis
- Power analysis
- Real-world applications

**Advanced Level (Exercises 7-8):**
- Robust methods
- Advanced visualization
- Comprehensive reporting

Each exercise builds upon previous knowledge and introduces new concepts progressively.

## Next Steps

In the next chapter, we'll learn about two-way ANOVA for analyzing the effects of two independent variables.

---

**Key Takeaways:**
- One-way ANOVA compares means across three or more groups
- Always check assumptions before interpreting results
- Effect sizes provide important information about practical significance
- Post hoc tests are necessary when ANOVA is significant
- Nonparametric alternatives exist for non-normal data
- Power analysis helps determine appropriate sample sizes
- Proper reporting includes descriptive statistics, test results, and effect sizes 