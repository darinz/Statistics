# Repeated Measures ANOVA

## Overview

Repeated Measures Analysis of Variance (ANOVA) is a powerful statistical technique used to analyze data where the same subjects are measured multiple times under different conditions. This within-subjects design is more powerful than between-subjects designs because it controls for individual differences and reduces error variance, leading to increased statistical power and more precise effect estimates.

### Key Concepts

**Within-Subjects Design**: A research design where each participant is measured under all experimental conditions, allowing for direct comparison of conditions within the same individuals.

**Individual Differences**: The natural variation between participants that is controlled for in repeated measures designs, reducing error variance.

**Sphericity**: The assumption that the variances of the differences between all pairs of conditions are equal, which is crucial for the validity of repeated measures ANOVA.

**Compound Symmetry**: A stronger assumption than sphericity, requiring equal variances and equal covariances between all conditions.

### Mathematical Foundation

The repeated measures ANOVA model can be expressed as:

```math
Y_{ij} = \mu + \alpha_i + \pi_j + \epsilon_{ij}
```

Where:
- $`Y_{ij}`$ is the observed value for the jth subject in the ith condition
- $`\mu`$ is the overall population mean
- $`\alpha_i`$ is the effect of the ith condition (treatment effect)
- $`\pi_j`$ is the effect of the jth subject (individual differences)
- $`\epsilon_{ij}`$ is the random error term

### Sum of Squares Decomposition

The total variability in repeated measures data is partitioned into three components:

```math
SS_{Total} = SS_{Between Subjects} + SS_{Within Subjects}
```

Where:
- $`SS_{Total} = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{..})^2`$
- $`SS_{Between Subjects} = k\sum_{j=1}^{n}(\bar{Y}_{.j} - \bar{Y}_{..})^2`$ (individual differences)
- $`SS_{Within Subjects} = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{.j})^2`$ (within-subject variation)

The within-subjects sum of squares is further partitioned into:

```math
SS_{Within Subjects} = SS_{Conditions} + SS_{Error}
```

Where:
- $`SS_{Conditions} = n\sum_{i=1}^{k}(\bar{Y}_{i.} - \bar{Y}_{..})^2`$ (treatment effect)
- $`SS_{Error} = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{i.} - \bar{Y}_{.j} + \bar{Y}_{..})^2`$ (residual error)

### Degrees of Freedom

```math
df_{Total} = nk - 1
df_{Between Subjects} = n - 1
df_{Within Subjects} = n(k - 1)
df_{Conditions} = k - 1
df_{Error} = (n - 1)(k - 1)
```

Where $`n`$ is the number of subjects and $`k`$ is the number of conditions.

### F-Statistic

```math
F = \frac{MS_{Conditions}}{MS_{Error}} = \frac{SS_{Conditions}/df_{Conditions}}{SS_{Error}/df_{Error}}
```

### Effect Size Measures

**Partial Eta-Squared**:
```math
\eta_p^2 = \frac{SS_{Conditions}}{SS_{Conditions} + SS_{Error}}
```

**Eta-Squared**:
```math
\eta^2 = \frac{SS_{Conditions}}{SS_{Total}}
```

**Omega-Squared** (unbiased estimate):
```math
\omega^2 = \frac{SS_{Conditions} - df_{Conditions} \times MS_{Error}}{SS_{Total} + MS_{Error}}
```

### Sphericity Assumption

The sphericity assumption requires that the variance of the differences between any two conditions is constant:

```math
Var(Y_{i} - Y_{j}) = \sigma^2 \text{ for all } i \neq j
```

This can be tested using Mauchly's test of sphericity, which examines the correlation matrix of the conditions.

## Basic Repeated Measures ANOVA

### Manual Calculation

The manual calculation helps understand the underlying mathematics and provides insight into how the statistical software performs the analysis.

```python
# Simulate repeated measures data
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

np.random.seed(123)
n_subjects = 20
n_conditions = 3

# Generate data for 3 conditions (e.g., pre-test, post-test, follow-up)
subject_id = np.repeat(range(1, n_subjects + 1), n_conditions)
condition = np.tile(["Pre", "Post", "Follow-up"], n_subjects)

# Generate scores with subject effects and condition effects
scores = np.zeros(len(subject_id))
for i in range(len(subject_id)):
    subject_effect = np.random.normal(0, 10)  # Individual differences
    if condition[i] == "Pre":
        scores[i] = 70 + subject_effect + np.random.normal(0, 5)
    elif condition[i] == "Post":
        scores[i] = 75 + subject_effect + np.random.normal(0, 5)
    else:
        scores[i] = 78 + subject_effect + np.random.normal(0, 5)

# Create data frame
repeated_data = pd.DataFrame({
    'subject': pd.Categorical(subject_id),
    'condition': pd.Categorical(condition, categories=["Pre", "Post", "Follow-up"], ordered=True),
    'score': scores
})

# Manual repeated measures ANOVA calculation
def manual_repeated_anova(data, subject_var, condition_var, response_var):
    # Calculate overall mean
    overall_mean = data[response_var].mean()
    
    # Calculate condition means
    condition_means = data.groupby(condition_var)[response_var].mean()
    
    # Calculate subject means
    subject_means = data.groupby(subject_var)[response_var].mean()
    
    # Calculate sample sizes
    n_subjects = len(data[subject_var].unique())
    n_conditions = len(data[condition_var].unique())
    n_total = len(data)
    
    # Calculate Sum of Squares
    # SS Total
    ss_total = ((data[response_var] - overall_mean) ** 2).sum()
    
    # SS Between Subjects (individual differences)
    ss_between_subjects = n_conditions * ((subject_means - overall_mean) ** 2).sum()
    
    # SS Within Subjects
    ss_within_subjects = ss_total - ss_between_subjects
    
    # SS Conditions (Treatment effect)
    ss_conditions = n_subjects * ((condition_means - overall_mean) ** 2).sum()
    
    # SS Error (Residual)
    ss_error = ss_within_subjects - ss_conditions
    
    # Degrees of freedom
    df_between_subjects = n_subjects - 1
    df_within_subjects = n_subjects * (n_conditions - 1)
    df_conditions = n_conditions - 1
    df_error = (n_subjects - 1) * (n_conditions - 1)
    
    # Mean Squares
    ms_conditions = ss_conditions / df_conditions
    ms_error = ss_error / df_error
    
    # F-statistic
    f_statistic = ms_conditions / ms_error
    
    # p-value
    p_value = 1 - stats.f.cdf(f_statistic, df_conditions, df_error)
    
    # Effect size (partial eta-squared)
    partial_eta2 = ss_conditions / (ss_conditions + ss_error)
    
    # Calculate sphericity measures
    # Create wide format for correlation analysis
    wide_data = data.pivot(index=subject_var, columns=condition_var, values=response_var)
    
    # Remove subject column for correlation matrix
    cor_matrix = wide_data.corr()
    
    # Greenhouse-Geisser epsilon
    gg_epsilon = 1 / (n_conditions - 1)
    
    # Huynh-Feldt epsilon (approximation)
    hf_epsilon = min(1, gg_epsilon * (n_subjects - 1) / (n_subjects - 1 - (n_conditions - 1)))
    
    return {
        'ss_total': ss_total,
        'ss_between_subjects': ss_between_subjects,
        'ss_within_subjects': ss_within_subjects,
        'ss_conditions': ss_conditions,
        'ss_error': ss_error,
        'df_conditions': df_conditions,
        'df_error': df_error,
        'ms_conditions': ms_conditions,
        'ms_error': ms_error,
        'f_statistic': f_statistic,
        'p_value': p_value,
        'partial_eta2': partial_eta2,
        'condition_means': condition_means,
        'subject_means': subject_means,
        'correlation_matrix': cor_matrix,
        'gg_epsilon': gg_epsilon,
        'hf_epsilon': hf_epsilon
    }

# Apply manual calculation
anova_result = manual_repeated_anova(repeated_data, "subject", "condition", "score")

print("Manual Repeated Measures ANOVA Results:")
print(f"F-statistic: {anova_result['f_statistic']:.3f}")
print(f"p-value: {anova_result['p_value']:.4f}")
print(f"Partial η²: {anova_result['partial_eta2']:.3f}")
print(f"Condition means: {anova_result['condition_means'].round(2).to_dict()}")
print(f"Greenhouse-Geisser ε: {anova_result['gg_epsilon']:.3f}")
print(f"Huynh-Feldt ε: {anova_result['hf_epsilon']:.3f}")
```

### Using Python's Built-in Repeated Measures ANOVA

Python's `statsmodels` provides a convenient way to perform repeated measures ANOVA.

```python
# Perform repeated measures ANOVA using statsmodels
repeated_model = AnovaRM(repeated_data, 'score', 'subject', within=['condition']).fit()
print(repeated_model.anova_table)

# Extract key statistics
f_statistic = repeated_model.anova_table.loc['condition', 'F Value']
p_value = repeated_model.anova_table.loc['condition', 'Pr > F']
df_conditions = repeated_model.anova_table.loc['condition', 'Num DF']
df_error = repeated_model.anova_table.loc['condition', 'Den DF']

print("Repeated Measures ANOVA Results:")
print(f"F-statistic: {f_statistic:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {df_conditions}, {df_error}")
```

## Descriptive Statistics

### Understanding Repeated Measures Descriptive Statistics

In repeated measures designs, we need to understand several types of statistics:

1. **Condition Means**: The average performance across all subjects for each condition
2. **Subject Means**: The average performance across all conditions for each subject
3. **Individual Differences**: The variation between subjects
4. **Within-Subject Variation**: The variation within each subject across conditions

### Mathematical Definitions

**Condition Mean**: $`\bar{Y}_{i.} = \frac{1}{n}\sum_{j=1}^{n}Y_{ij}`$

**Subject Mean**: $`\bar{Y}_{.j} = \frac{1}{k}\sum_{i=1}^{k}Y_{ij}`$

**Grand Mean**: $`\bar{Y}_{..} = \frac{1}{nk}\sum_{i=1}^{k}\sum_{j=1}^{n}Y_{ij}`$

**Individual Differences**: $`SS_{Between Subjects} = k\sum_{j=1}^{n}(\bar{Y}_{.j} - \bar{Y}_{..})^2`$

**Within-Subject Variation**: $`SS_{Within Subjects} = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{.j})^2`$

### Condition and Subject Statistics

```python
# Calculate comprehensive descriptive statistics

# Condition means with confidence intervals
condition_stats = repeated_data.groupby('condition').agg({
    'score': ['count', 'mean', 'std', 'min', 'max']
}).round(3)
condition_stats.columns = ['n', 'mean', 'sd', 'min', 'max']
condition_stats = condition_stats.reset_index()

# Calculate standard error and confidence intervals
condition_stats['se'] = condition_stats['sd'] / np.sqrt(condition_stats['n'])
condition_stats['ci_lower'] = condition_stats['mean'] - stats.t.ppf(0.975, condition_stats['n']-1) * condition_stats['se']
condition_stats['ci_upper'] = condition_stats['mean'] + stats.t.ppf(0.975, condition_stats['n']-1) * condition_stats['se']

print("Condition Statistics:")
print(condition_stats)

# Subject means with individual differences analysis
subject_stats = repeated_data.groupby('subject').agg({
    'score': ['count', 'mean', 'std', 'min', 'max']
}).round(3)
subject_stats.columns = ['n', 'mean', 'sd', 'min', 'max']
subject_stats = subject_stats.reset_index()
subject_stats['range'] = subject_stats['max'] - subject_stats['min']

print("\nSubject Statistics Summary:")
print(f"Mean subject score: {subject_stats['mean'].mean():.2f}")
print(f"SD of subject scores: {subject_stats['mean'].std():.2f}")
print(f"Range of subject scores: {subject_stats['mean'].min():.2f} to {subject_stats['mean'].max():.2f}")
print(f"Mean within-subject range: {subject_stats['range'].mean():.2f}")
print(f"SD of within-subject ranges: {subject_stats['range'].std():.2f}")

# Calculate individual differences
grand_mean = repeated_data['score'].mean()
individual_differences = subject_stats['mean'] - grand_mean

print("\nIndividual Differences Analysis:")
print(f"Mean individual difference: {individual_differences.mean():.2f}")
print(f"SD of individual differences: {individual_differences.std():.2f}")
print(f"Range of individual differences: {individual_differences.min():.2f} to {individual_differences.max():.2f}")

# Calculate condition differences
condition_differences = condition_stats['mean'] - grand_mean
print("\nCondition Differences from Grand Mean:")
for i, row in condition_stats.iterrows():
    print(f"{row['condition']}: {condition_differences.iloc[i]:.2f}")
```

### Understanding Individual Differences

```python
# Analyze individual differences in detail
def analyze_individual_differences(data, subject_var, condition_var, response_var):
    # Calculate subject means
    subject_means = data.groupby(subject_var)[response_var].mean().reset_index()
    
    # Calculate grand mean
    grand_mean = data[response_var].mean()
    
    # Individual differences
    subject_means['individual_diff'] = subject_means[response_var] - grand_mean
    
    # Categorize subjects by performance level
    sd_diff = subject_means['individual_diff'].std()
    subject_means['performance_level'] = pd.cut(
        subject_means['individual_diff'],
        bins=[-np.inf, -sd_diff, sd_diff, np.inf],
        labels=['Low', 'Average', 'High']
    )
    
    # Calculate within-subject variability
    within_subject_var = data.groupby(subject_var).agg({
        response_var: ['std', lambda x: x.max() - x.min()]
    }).round(3)
    within_subject_var.columns = ['within_sd', 'within_range']
    within_subject_var = within_subject_var.reset_index()
    
    # Combine results
    results = pd.merge(subject_means, within_subject_var, on=subject_var)
    
    print("INDIVIDUAL DIFFERENCES ANALYSIS:")
    print("=" * 35)
    print(f"Grand mean: {grand_mean:.2f}")
    print(f"SD of individual differences: {results['individual_diff'].std():.2f}")
    print(f"Range of individual differences: {results['individual_diff'].min():.2f} to {results['individual_diff'].max():.2f}\n")
    
    print("PERFORMANCE LEVEL DISTRIBUTION:")
    print("=" * 32)
    print(results['performance_level'].value_counts())
    
    print("\nWITHIN-SUBJECT VARIABILITY:")
    print("=" * 26)
    print(f"Mean within-subject SD: {results['within_sd'].mean():.2f}")
    print(f"SD of within-subject SDs: {results['within_sd'].std():.2f}")
    print(f"Mean within-subject range: {results['within_range'].mean():.2f}")
    
    return results

# Apply individual differences analysis
individual_analysis = analyze_individual_differences(repeated_data, "subject", "condition", "score")
```

### Visualization

Visualization is crucial for understanding repeated measures data, as it helps identify patterns, individual differences, and potential violations of assumptions.

#### Individual Subject Profiles

The individual subject profiles plot is the most important visualization for repeated measures data as it shows how each subject changes across conditions.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Enhanced individual subject profiles
plt.figure(figsize=(10, 6))
for subject in repeated_data['subject'].unique():
    subject_data = repeated_data[repeated_data['subject'] == subject]
    plt.plot(subject_data['condition'], subject_data['score'], 
             alpha=0.4, linewidth=0.8, marker='o', markersize=4)

# Add group mean line
group_means = repeated_data.groupby('condition')['score'].mean()
plt.plot(group_means.index, group_means.values, 'r-', linewidth=2.5, marker='s', 
         markersize=8, label='Group Mean')

# Add confidence intervals
for condition in repeated_data['condition'].unique():
    cond_data = repeated_data[repeated_data['condition'] == condition]['score']
    mean_val = cond_data.mean()
    se_val = cond_data.std() / np.sqrt(len(cond_data))
    plt.errorbar(condition, mean_val, yerr=se_val, color='red', 
                capsize=5, capthick=2, linewidth=1)

plt.title('Individual Subject Profiles', fontsize=14, fontweight='bold')
plt.xlabel('Condition')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Interpret individual profiles
print("Individual Profiles Interpretation:")
print("- Parallel lines indicate consistent individual differences")
print("- Non-parallel lines indicate individual differences in treatment effects")
print("- Steep slopes indicate strong treatment effects")
print("- Flat lines indicate no treatment effect\n")
```

#### Enhanced Distribution Plots

```python
# Enhanced box plot with individual points
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Box plot with individual points
sns.boxplot(data=repeated_data, x='condition', y='score', ax=ax1, alpha=0.7)
sns.stripplot(data=repeated_data, x='condition', y='score', ax=ax1, 
              alpha=0.6, size=3, color='black', jitter=0.2)

# Add mean points
group_means = repeated_data.groupby('condition')['score'].mean()
ax1.plot(range(len(group_means)), group_means.values, 'ro', markersize=8, label='Mean')
ax1.set_title('Score Distribution by Condition', fontweight='bold')
ax1.set_xlabel('Condition')
ax1.set_ylabel('Score')

# Enhanced violin plot
sns.violinplot(data=repeated_data, x='condition', y='score', ax=ax2, alpha=0.7)
sns.boxplot(data=repeated_data, x='condition', y='score', ax=ax2, 
            width=0.2, alpha=0.8, color='white')

# Add mean points
ax2.plot(range(len(group_means)), group_means.values, 'ro', markersize=8, label='Mean')
ax2.set_title('Score Distribution by Condition', fontweight='bold')
ax2.set_xlabel('Condition')
ax2.set_ylabel('Score')

plt.tight_layout()
plt.show()
```

#### Individual Differences Visualization

```python
# Individual differences plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Individual differences plot
individual_analysis_sorted = individual_analysis.sort_values('individual_diff')
ax1.bar(range(len(individual_analysis_sorted)), individual_analysis_sorted['individual_diff'], 
        color='steelblue', alpha=0.7)
ax1.axhline(y=0, color='red', linestyle='--')
ax1.axhline(y=individual_analysis['individual_diff'].std(), color='orange', linestyle=':')
ax1.axhline(y=-individual_analysis['individual_diff'].std(), color='orange', linestyle=':')
ax1.set_title('Individual Differences from Grand Mean', fontweight='bold')
ax1.set_xlabel('Subject')
ax1.set_ylabel('Individual Difference')
ax1.set_xticks(range(len(individual_analysis_sorted)))
ax1.set_xticklabels(individual_analysis_sorted['subject'], rotation=45)

# Within-subject variability plot
individual_analysis_sorted_sd = individual_analysis.sort_values('within_sd')
ax2.bar(range(len(individual_analysis_sorted_sd)), individual_analysis_sorted_sd['within_sd'], 
        color='darkgreen', alpha=0.7)
ax2.axhline(y=individual_analysis['within_sd'].mean(), color='red', linestyle='--')
ax2.set_title('Within-Subject Variability', fontweight='bold')
ax2.set_xlabel('Subject')
ax2.set_ylabel('Within-Subject Standard Deviation')
ax2.set_xticks(range(len(individual_analysis_sorted_sd)))
ax2.set_xticklabels(individual_analysis_sorted_sd['subject'], rotation=45)

plt.tight_layout()
plt.show()
```

#### Correlation Matrix Visualization

```python
# Correlation matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(anova_result['correlation_matrix'], annot=True, fmt='.2f', cmap='RdYlBu_r',
            cbar_kws={'label': 'Correlation'}, square=True)
plt.title('Condition Correlation Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Condition')
plt.ylabel('Condition')
plt.show()

# Combine all plots in a comprehensive figure
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))

# Individual profiles
for subject in repeated_data['subject'].unique():
    subject_data = repeated_data[repeated_data['subject'] == subject]
    ax1.plot(subject_data['condition'], subject_data['score'], 
             alpha=0.4, linewidth=0.8, marker='o', markersize=4)
group_means = repeated_data.groupby('condition')['score'].mean()
ax1.plot(group_means.index, group_means.values, 'r-', linewidth=2.5, marker='s', 
         markersize=8, label='Group Mean')
ax1.set_title('Individual Subject Profiles', fontweight='bold')
ax1.legend()

# Box plot
sns.boxplot(data=repeated_data, x='condition', y='score', ax=ax2, alpha=0.7)
sns.stripplot(data=repeated_data, x='condition', y='score', ax=ax2, 
              alpha=0.6, size=3, color='black', jitter=0.2)
ax2.set_title('Score Distribution by Condition', fontweight='bold')

# Violin plot
sns.violinplot(data=repeated_data, x='condition', y='score', ax=ax3, alpha=0.7)
sns.boxplot(data=repeated_data, x='condition', y='score', ax=ax3, 
            width=0.2, alpha=0.8, color='white')
ax3.set_title('Score Distribution by Condition', fontweight='bold')

# Individual differences
individual_analysis_sorted = individual_analysis.sort_values('individual_diff')
ax4.bar(range(len(individual_analysis_sorted)), individual_analysis_sorted['individual_diff'], 
        color='steelblue', alpha=0.7)
ax4.axhline(y=0, color='red', linestyle='--')
ax4.set_title('Individual Differences from Grand Mean', fontweight='bold')
ax4.set_xlabel('Subject')
ax4.set_ylabel('Individual Difference')

# Within-subject variability
individual_analysis_sorted_sd = individual_analysis.sort_values('within_sd')
ax5.bar(range(len(individual_analysis_sorted_sd)), individual_analysis_sorted_sd['within_sd'], 
        color='darkgreen', alpha=0.7)
ax5.axhline(y=individual_analysis['within_sd'].mean(), color='red', linestyle='--')
ax5.set_title('Within-Subject Variability', fontweight='bold')
ax5.set_xlabel('Subject')
ax5.set_ylabel('Within-Subject Standard Deviation')

# Correlation matrix
sns.heatmap(anova_result['correlation_matrix'], annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax6)
ax6.set_title('Condition Correlation Matrix', fontweight='bold')

plt.tight_layout()
plt.show()
```

#### Trend Analysis Visualization

```python
# Trend analysis plot
plt.figure(figsize=(10, 6))

# Plot individual subject lines
for subject in repeated_data['subject'].unique():
    subject_data = repeated_data[repeated_data['subject'] == subject]
    condition_numeric = pd.Categorical(subject_data['condition']).codes
    plt.plot(condition_numeric, subject_data['score'], 
             alpha=0.3, linewidth=0.8, marker='o', markersize=4)

# Plot group mean line
condition_numeric_all = pd.Categorical(repeated_data['condition']).codes
group_means = repeated_data.groupby('condition')['score'].mean()
condition_numeric_means = pd.Categorical(group_means.index).codes
plt.plot(condition_numeric_means, group_means.values, 'r-', linewidth=2.5, marker='s', 
         markersize=8, label='Group Mean')

# Add confidence intervals
for i, condition in enumerate(repeated_data['condition'].unique()):
    cond_data = repeated_data[repeated_data['condition'] == condition]['score']
    mean_val = cond_data.mean()
    se_val = cond_data.std() / np.sqrt(len(cond_data))
    plt.errorbar(i, mean_val, yerr=se_val, color='red', 
                capsize=5, capthick=2, linewidth=1)

# Add trend line
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(condition_numeric_means, group_means.values)
trend_x = np.array([0, 1, 2])
trend_y = slope * trend_x + intercept
plt.plot(trend_x, trend_y, 'b--', linewidth=2, label=f'Trend (r={r_value:.3f})')

plt.title('Trend Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Condition (Numeric)')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Effect Size Analysis

Effect sizes are crucial for understanding the practical significance of repeated measures ANOVA results, as they provide information about the magnitude of effects independent of sample size.

### Understanding Effect Size Measures in Repeated Measures

#### Partial Eta-Squared ($`\eta_p^2`$)

Partial eta-squared is the most commonly used effect size measure in repeated measures ANOVA. It represents the proportion of variance in the dependent variable that is explained by the condition effect, controlling for individual differences.

```math
\eta_p^2 = \frac{SS_{Conditions}}{SS_{Conditions} + SS_{Error}}
```

**Advantages:**
- Controls for individual differences
- Ranges from 0 to 1
- Easy to interpret
- Most commonly reported in literature

**Disadvantages:**
- Can be biased upward in small samples
- Values can be larger than in between-subjects designs

#### Eta-Squared ($`\eta^2`$)

Eta-squared represents the proportion of total variance explained by the condition effect.

```math
\eta^2 = \frac{SS_{Conditions}}{SS_{Total}}
```

**Advantages:**
- Intuitive interpretation
- Values sum to 1 across all effects

**Disadvantages:**
- Does not control for individual differences
- Can be misleading in repeated measures designs

#### Omega-Squared ($`\omega^2`$)

Omega-squared is an unbiased estimate of the population effect size.

```math
\omega^2 = \frac{SS_{Conditions} - df_{Conditions} \times MS_{Error}}{SS_{Total} + MS_{Error}}
```

**Advantages:**
- Unbiased estimate
- More conservative than eta-squared

**Disadvantages:**
- Can be negative for small effects
- Less commonly reported

### Comprehensive Effect Size Calculation

```python
# Calculate comprehensive effect sizes for repeated measures ANOVA
def calculate_repeated_effect_sizes(anova_result, data, subject_var, condition_var, response_var):
    # Extract Sum of Squares
    ss_conditions = anova_result['ss_conditions']
    ss_error = anova_result['ss_error']
    ss_total = anova_result['ss_total']
    ss_between_subjects = anova_result['ss_between_subjects']
    ss_within_subjects = anova_result['ss_within_subjects']
    
    # Degrees of freedom
    df_conditions = anova_result['df_conditions']
    df_error = anova_result['df_error']
    df_between_subjects = anova_result['df_between_subjects']
    df_within_subjects = anova_result['df_within_subjects']
    
    # Mean Squares
    ms_error = anova_result['ms_error']
    
    # Partial eta-squared
    partial_eta2 = ss_conditions / (ss_conditions + ss_error)
    
    # Eta-squared (total)
    eta2 = ss_conditions / ss_total
    
    # Omega-squared (unbiased)
    omega2 = (ss_conditions - df_conditions * ms_error) / (ss_total + ms_error)
    
    # Cohen's f (for power analysis)
    cohens_f = np.sqrt(partial_eta2 / (1 - partial_eta2))
    
    # Individual differences effect size
    individual_eta2 = ss_between_subjects / ss_total
    individual_partial_eta2 = ss_between_subjects / (ss_between_subjects + ss_error)
    
    # Bootstrap confidence intervals for partial eta-squared
    def bootstrap_effect_size(data, subject_var, condition_var, response_var):
        # Resample data
        indices = np.random.choice(len(data), len(data), replace=True)
        d = data.iloc[indices]
        
        # Fit model
        model = AnovaRM(d, response_var, subject_var, within=[condition_var]).fit()
        
        ss_conditions = model.anova_table.loc[condition_var, 'Sum Sq']
        ss_error = model.anova_table.loc['Residual', 'Sum Sq']
        
        partial_eta2 = ss_conditions / (ss_conditions + ss_error)
        return partial_eta2
    
    # Bootstrap for confidence intervals
    bootstrap_results = []
    for _ in range(1000):
        bootstrap_results.append(bootstrap_effect_size(data, subject_var, condition_var, response_var))
    
    bootstrap_results = np.array(bootstrap_results)
    ci_partial_eta2 = np.percentile(bootstrap_results, [2.5, 97.5])
    
    # Calculate condition-specific effect sizes
    condition_means = anova_result['condition_means']
    grand_mean = data[response_var].mean()
    
    # Standardized mean differences (Cohen's d) for each condition
    condition_effects = (condition_means - grand_mean) / np.sqrt(ms_error)
    
    return {
        'partial_eta2': partial_eta2,
        'eta2': eta2,
        'omega2': omega2,
        'cohens_f': cohens_f,
        'individual_eta2': individual_eta2,
        'individual_partial_eta2': individual_partial_eta2,
        'condition_effects': condition_effects,
        'ci_partial_eta2': ci_partial_eta2,
        'bootstrap_results': bootstrap_results
    }

# Apply comprehensive effect size calculation
effect_sizes = calculate_repeated_effect_sizes(anova_result, repeated_data, "subject", "condition", "score")
```

# Display comprehensive effect size results
print("=== COMPREHENSIVE EFFECT SIZE ANALYSIS ===\n")

print("CONDITION EFFECT SIZES:")
print("=" * 22)
print(f"Partial η²: {effect_sizes['partial_eta2']:.4f}")
print(f"η²: {effect_sizes['eta2']:.4f}")
print(f"ω²: {effect_sizes['omega2']:.4f}")
print(f"Cohen's f: {effect_sizes['cohens_f']:.4f}\n")

print("INDIVIDUAL DIFFERENCES EFFECT SIZES:")
print("=" * 33)
print(f"Individual differences η²: {effect_sizes['individual_eta2']:.4f}")
print(f"Individual differences partial η²: {effect_sizes['individual_partial_eta2']:.4f}\n")

print("CONDITION-SPECIFIC EFFECTS (Cohen's d):")
print("=" * 35)
for condition, effect in effect_sizes['condition_effects'].items():
    print(f"{condition}: {effect:.3f}")
```
```

### Effect Size Interpretation

```python
# Enhanced effect size interpretation
def interpret_effect_size(eta_sq, measure="partial_eta2"):
    if measure == "partial_eta2" or measure == "eta2":
        if eta_sq < 0.01:
            return "Negligible effect (< 1% of variance explained)"
        elif eta_sq < 0.06:
            return "Small effect (1-6% of variance explained)"
        elif eta_sq < 0.14:
            return "Medium effect (6-14% of variance explained)"
        else:
            return "Large effect (> 14% of variance explained)"
    elif measure == "f":
        if eta_sq < 0.10:
            return "Small effect"
        elif eta_sq < 0.25:
            return "Medium effect"
        elif eta_sq < 0.40:
            return "Large effect"
        else:
            return "Very large effect"
    elif measure == "d":
        if abs(eta_sq) < 0.20:
            return "Negligible effect"
        elif abs(eta_sq) < 0.50:
            return "Small effect"
        elif abs(eta_sq) < 0.80:
            return "Medium effect"
        else:
            return "Large effect"

print("EFFECT SIZE INTERPRETATION:")
print("=" * 25)
print(f"Condition effect: {interpret_effect_size(effect_sizes['partial_eta2'])}")
print(f"Individual differences: {interpret_effect_size(effect_sizes['individual_eta2'])}\n")

# Practical significance assessment
print("PRACTICAL SIGNIFICANCE ASSESSMENT:")
print("=" * 32)
if effect_sizes['partial_eta2'] > 0.06:
    print("✓ Condition effect is practically significant")
else:
    print("✗ Condition effect may not be practically significant")

if effect_sizes['individual_partial_eta2'] > 0.06:
    print("✓ Individual differences are practically significant")
else:
    print("✗ Individual differences may not be practically significant")
```

### Effect Size Visualization

```python
# Create effect size comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Effect size comparison plot
effect_size_data = pd.DataFrame({
    'Effect': ['Condition Effect', 'Individual Differences'] * 3,
    'Measure': ['Partial η²', 'Partial η²', 'η²', 'η²', 'ω²', 'ω²'],
    'Value': [effect_sizes['partial_eta2'], effect_sizes['individual_partial_eta2'],
              effect_sizes['eta2'], effect_sizes['individual_eta2'],
              effect_sizes['omega2'], effect_sizes['individual_partial_eta2']]
})

sns.barplot(data=effect_size_data, x='Effect', y='Value', hue='Measure', ax=ax1, alpha=0.8)
ax1.axhline(y=0.01, color='red', linestyle='--', alpha=0.7)
ax1.axhline(y=0.06, color='red', linestyle='--', alpha=0.7)
ax1.axhline(y=0.14, color='red', linestyle='--', alpha=0.7)
ax1.text(0.5, 0.01, 'Negligible', color='red', ha='left')
ax1.text(0.5, 0.06, 'Small', color='red', ha='left')
ax1.text(0.5, 0.14, 'Medium', color='red', ha='left')
ax1.set_title('Effect Size Comparison', fontweight='bold')
ax1.set_ylabel('Effect Size')

# Condition-specific effects plot
condition_effects_data = pd.DataFrame({
    'Condition': list(effect_sizes['condition_effects'].keys()),
    'Effect_Size': list(effect_sizes['condition_effects'].values())
})

sns.barplot(data=condition_effects_data, x='Condition', y='Effect_Size', ax=ax2, alpha=0.7)
ax2.axhline(y=-0.2, color='red', linestyle='--', alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.7)
ax2.text(0.5, -0.2, 'Small', color='red', ha='left')
ax2.text(0.5, 0.2, 'Small', color='red', ha='left')
ax2.set_title('Condition-Specific Effect Sizes (Cohen\'s d)', fontweight='bold')
ax2.set_ylabel('Effect Size (Cohen\'s d)')

plt.tight_layout()
plt.show()
```

## Post Hoc Tests

### Pairwise Comparisons

```python
# Perform pairwise comparisons
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_rel

# Get estimated marginal means
emm = repeated_data.groupby('condition')['score'].mean()
print("Estimated Marginal Means:")
print(emm)

# Pairwise comparisons with Tukey's HSD
# First, we need to create a wide format for the test
wide_data = repeated_data.pivot(index='subject', columns='condition', values='score')
tukey_result = pairwise_tukeyhsd(repeated_data['score'], repeated_data['condition'])

print("\nTukey's HSD Results:")
print(tukey_result)

# Extract significant pairs
significant_pairs = tukey_result.pvalues < 0.05
if np.any(significant_pairs):
    print("\nSignificant pairwise differences (Tukey's HSD, p < 0.05):")
    significant_indices = np.where(significant_pairs)[0]
    for idx in significant_indices:
        group1 = tukey_result.groupsunique[tukey_result.pvalues < 0.05][0]
        group2 = tukey_result.groupsunique[tukey_result.pvalues < 0.05][1]
        p_val = tukey_result.pvalues[idx]
        print(f"{group1} vs {group2}: p = {p_val:.4f}")
else:
    print("\nNo significant pairwise differences found.")

# Pairwise t-tests with Bonferroni correction
conditions = repeated_data['condition'].unique()
n_comparisons = len(conditions) * (len(conditions) - 1) // 2
bonferroni_alpha = 0.05 / n_comparisons

print(f"\nBonferroni-corrected alpha: {bonferroni_alpha:.4f}")

for i in range(len(conditions)):
    for j in range(i+1, len(conditions)):
        cond1 = conditions[i]
        cond2 = conditions[j]
        
        data1 = repeated_data[repeated_data['condition'] == cond1]['score']
        data2 = repeated_data[repeated_data['condition'] == cond2]['score']
        
        t_stat, p_val = ttest_rel(data1, data2)
        
        print(f"{cond1} vs {cond2}: t = {t_stat:.3f}, p = {p_val:.4f}")
        if p_val < bonferroni_alpha:
            print(f"  ✓ Significant after Bonferroni correction")
        else:
            print(f"  ✗ Not significant after Bonferroni correction")
```

### Trend Analysis

```python
# Polynomial contrasts for trend analysis
from scipy import stats

# Create polynomial contrasts
conditions = repeated_data['condition'].unique()
n_conditions = len(conditions)

# Linear trend
linear_contrast = np.array([-1, 0, 1])  # For 3 conditions: linear trend
quadratic_contrast = np.array([1, -2, 1])  # For 3 conditions: quadratic trend

# Calculate trend effects
def calculate_trend_effect(data, contrast):
    condition_means = data.groupby('condition')['score'].mean().values
    trend_effect = np.sum(condition_means * contrast)
    return trend_effect

linear_effect = calculate_trend_effect(repeated_data, linear_contrast)
quadratic_effect = calculate_trend_effect(repeated_data, quadratic_contrast)

# Calculate F-statistics for trends
def calculate_trend_f(data, contrast):
    condition_means = data.groupby('condition')['score'].mean().values
    n_subjects = len(data['subject'].unique())
    
    # Calculate sum of squares for trend
    trend_effect = np.sum(condition_means * contrast)
    ss_trend = n_subjects * (trend_effect ** 2) / np.sum(contrast ** 2)
    
    # Get error sum of squares from ANOVA result
    ss_error = anova_result['ss_error']
    df_error = anova_result['df_error']
    
    # Calculate F-statistic
    f_stat = ss_trend / (ss_error / df_error)
    p_val = 1 - stats.f.cdf(f_stat, 1, df_error)
    
    return f_stat, p_val

linear_f, linear_p = calculate_trend_f(repeated_data, linear_contrast)
quadratic_f, quadratic_p = calculate_trend_f(repeated_data, quadratic_contrast)

print("Trend Analysis Results:")
print(f"Linear trend F-statistic: {linear_f:.3f}")
print(f"Linear trend p-value: {linear_p:.4f}")
print(f"Quadratic trend F-statistic: {quadratic_f:.3f}")
print(f"Quadratic trend p-value: {quadratic_p:.4f}")

# Alternative: Using polynomial regression
condition_numeric = pd.Categorical(repeated_data['condition']).codes
repeated_data['condition_numeric'] = condition_numeric

# Linear trend
linear_model = ols('score ~ condition_numeric', data=repeated_data).fit()
linear_f_alt = linear_model.fvalue
linear_p_alt = linear_model.f_pvalue

# Quadratic trend
repeated_data['condition_numeric_sq'] = condition_numeric ** 2
quadratic_model = ols('score ~ condition_numeric + condition_numeric_sq', data=repeated_data).fit()
quadratic_f_alt = quadratic_model.fvalue
quadratic_p_alt = quadratic_model.f_pvalue

print(f"\nAlternative Trend Analysis (Regression):")
print(f"Linear trend F-statistic: {linear_f_alt:.3f}")
print(f"Linear trend p-value: {linear_p_alt:.4f}")
print(f"Quadratic trend F-statistic: {quadratic_f_alt:.3f}")
print(f"Quadratic trend p-value: {quadratic_p_alt:.4f}")
```

## Assumption Checking

Repeated measures ANOVA relies on several key assumptions that must be verified before interpreting results. Violations of these assumptions can lead to incorrect conclusions and inflated Type I error rates.

### Key Assumptions

1. **Sphericity**: The variances of the differences between all pairs of conditions are equal
2. **Normality**: Residuals should be normally distributed
3. **Independence**: Observations should be independent (within-subjects correlation is expected)
4. **Linearity**: The relationship between conditions and the dependent variable should be linear

### Comprehensive Sphericity Testing

The sphericity assumption is the most critical assumption for repeated measures ANOVA. It requires that the variance of the differences between any two conditions is constant.

```python
# Enhanced function to check sphericity assumption
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def check_sphericity(data, subject_var, condition_var, response_var, alpha=0.05):
    print("=== COMPREHENSIVE SPHERICITY TESTING ===\n")
    # Create wide format for analysis
    wide_data = data.pivot(index=subject_var, columns=condition_var, values=response_var)
    cor_matrix = wide_data.corr()
    print("CORRELATION MATRIX OF CONDITIONS:")
    print("=" * 35)
    print(np.round(cor_matrix, 3))
    
    # Calculate condition variances
    condition_vars = wide_data.var(skipna=True)
    print("\nCONDITION VARIANCES:")
    print("=" * 20)
    for cond, var in condition_vars.items():
        print(f"{cond}: {var:.2f}")
    
    # Calculate variance ratios
    max_var = condition_vars.max()
    min_var = condition_vars.min()
    var_ratio = max_var / min_var
    print(f"\nVARIANCE RATIO (max/min): {var_ratio:.2f}")
    
    # Calculate difference variances
    n_conditions = cor_matrix.shape[0]
    diff_vars = []
    diff_names = []
    columns = wide_data.columns
    for i in range(n_conditions-1):
        for j in range(i+1, n_conditions):
            diff = wide_data.iloc[:, i] - wide_data.iloc[:, j]
            diff_vars.append(np.nanvar(diff, ddof=1))
            diff_names.append(f"{columns[i]} - {columns[j]}")
    print("\nDIFFERENCE VARIANCES:")
    print("=" * 20)
    for name, var in zip(diff_names, diff_vars):
        print(f"{name}: {var:.2f}")
    diff_var_ratio = np.nanmax(diff_vars) / np.nanmin(diff_vars)
    print(f"\nDIFFERENCE VARIANCE RATIO (max/min): {diff_var_ratio:.2f}")
    
    # Calculate sphericity measures
    n_subjects = wide_data.shape[0]
    gg_epsilon = 1 / (n_conditions - 1)
    hf_epsilon = min(1, gg_epsilon * (n_subjects - 1) / (n_subjects - 1 - (n_conditions - 1)))
    lb_epsilon = 1 / (n_conditions - 1)
    print("\nSPHERICITY CORRECTIONS:")
    print("=" * 22)
    print(f"Greenhouse-Geisser ε: {gg_epsilon:.3f}")
    print(f"Huynh-Feldt ε: {hf_epsilon:.3f}")
    print(f"Lower-bound ε: {lb_epsilon:.3f}")
    
    # Sphericity assessment
    print("\nSPHERICITY ASSESSMENT:")
    print("=" * 21)
    sphericity_indicators = 0
    total_indicators = 0
    # Check correlation matrix
    if np.all(cor_matrix.values[np.triu_indices(n_conditions, 1)] > 0.3):
        print("✓ Correlations are reasonably high")
        sphericity_indicators += 1
    else:
        print("✗ Some correlations are low")
    total_indicators += 1
    # Check variance ratio
    if var_ratio <= 4:
        print("✓ Condition variances are reasonably equal")
        sphericity_indicators += 1
    else:
        print("✗ Condition variances are unequal")
    total_indicators += 1
    # Check difference variance ratio
    if diff_var_ratio <= 4:
        print("✓ Difference variances are reasonably equal")
        sphericity_indicators += 1
    else:
        print("✗ Difference variances are unequal")
    total_indicators += 1
    # Check epsilon values
    if gg_epsilon >= 0.75:
        print("✓ Greenhouse-Geisser epsilon suggests sphericity")
        sphericity_indicators += 1
    else:
        print("✗ Greenhouse-Geisser epsilon suggests sphericity violation")
    total_indicators += 1
    print(f"\nSphericity indicators met: {sphericity_indicators} out of {total_indicators}")
    print("\nRECOMMENDATIONS:")
    print("=" * 15)
    if sphericity_indicators >= 3:
        print("✓ Sphericity assumption appears to be met")
        print("✓ Standard repeated measures ANOVA is appropriate")
    elif sphericity_indicators >= 2:
        print("⚠ Sphericity assumption may be questionable")
        print("⚠ Consider using Greenhouse-Geisser or Huynh-Feldt corrections")
    else:
        print("✗ Sphericity assumption appears to be violated")
        print("✗ Use Greenhouse-Geisser or Huynh-Feldt corrections")
        print("✗ Consider nonparametric alternatives")
    # Create diagnostic plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # Correlation heatmap
    sns.heatmap(cor_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax1)
    ax1.set_title('Condition Correlation Matrix', fontweight='bold')
    # Difference variances plot
    diff_var_data = pd.DataFrame({'Comparison': diff_names, 'Variance': diff_vars})
    sns.barplot(data=diff_var_data, x='Comparison', y='Variance', ax=ax2, alpha=0.7)
    ax2.axhline(np.mean(diff_vars), color='red', linestyle='--')
    ax2.set_title('Variances of Condition Differences', fontweight='bold')
    ax2.set_xlabel('Condition Comparison')
    ax2.set_ylabel('Variance')
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    return {
        'correlation_matrix': cor_matrix,
        'condition_vars': condition_vars,
        'diff_vars': diff_vars,
        'diff_names': diff_names,
        'gg_epsilon': gg_epsilon,
        'hf_epsilon': hf_epsilon,
        'lb_epsilon': lb_epsilon,
        'sphericity_indicators': sphericity_indicators,
        'total_indicators': total_indicators
    }

# Check sphericity with enhanced function
sphericity_results = check_sphericity(repeated_data, "subject", "condition", "score")
```

### Comprehensive Normality Testing

```python
# Enhanced function to check normality for repeated measures
from scipy.stats import shapiro, kstest, anderson, norm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def check_normality_repeated(data, subject_var, condition_var, response_var, alpha=0.05):
    print("=== COMPREHENSIVE NORMALITY TESTS FOR REPEATED MEASURES ===\n")
    # Calculate residuals as deviations from condition means
    condition_means = data.groupby(condition_var)[response_var].transform('mean')
    residuals = data[response_var] - condition_means
    print("NORMALITY TESTS ON RESIDUALS:")
    print("=" * 35)
    tests = {}
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = shapiro(residuals)
    tests['shapiro'] = {'statistic': shapiro_stat, 'pvalue': shapiro_p}
    print("Shapiro-Wilk test:")
    print(f"  W = {shapiro_stat:.4f}")
    print(f"  p-value = {shapiro_p:.4f}")
    print(f"  Decision: {'✓ Normal' if shapiro_p >= alpha else '✗ Non-normal'}\n")
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
    tests['ks'] = {'statistic': ks_stat, 'pvalue': ks_p}
    print("Kolmogorov-Smirnov test:")
    print(f"  D = {ks_stat:.4f}")
    print(f"  p-value = {ks_p:.4f}")
    print(f"  Decision: {'✓ Normal' if ks_p >= alpha else '✗ Non-normal'}\n")
    # Anderson-Darling test
    ad_result = anderson(residuals)
    ad_stat = ad_result.statistic
    ad_crit = ad_result.critical_values[2]  # 5% level
    ad_sig = ad_result.significance_level[2] / 100
    print("Anderson-Darling test:")
    print(f"  A = {ad_stat:.4f}")
    print(f"  5% critical value = {ad_crit:.4f}")
    print(f"  Decision: {'✓ Normal' if ad_stat < ad_crit else '✗ Non-normal'}\n")
    tests['ad'] = {'statistic': ad_stat, 'critical': ad_crit, 'significance_level': ad_sig}
    # Summary of normality tests
    p_values = [shapiro_p, ks_p, 1 if ad_stat < ad_crit else 0]
    test_names = ["Shapiro-Wilk", "Kolmogorov-Smirnov", "Anderson-Darling"]
    normal_tests = sum([p >= alpha for p in p_values[:2]]) + (1 if ad_stat < ad_crit else 0)
    total_tests = 3
    print("NORMALITY TEST SUMMARY:")
    print("=" * 25)
    print(f"Tests supporting normality: {normal_tests} out of {total_tests}")
    if normal_tests >= 3:
        print("✓ Normality assumption appears to be met")
    elif normal_tests >= 2:
        print("⚠ Normality assumption may be questionable")
    else:
        print("✗ Normality assumption appears to be violated")
    # Check normality within each condition
    print("\nNORMALITY WITHIN CONDITIONS:")
    print("=" * 27)
    conditions = data[condition_var].unique()
    condition_normality = {}
    for cond in conditions:
        cond_data = data[data[condition_var] == cond][response_var]
        cond_residuals = cond_data - cond_data.mean()
        shapiro_cond_stat, shapiro_cond_p = shapiro(cond_residuals)
        condition_normality[cond] = {'statistic': shapiro_cond_stat, 'pvalue': shapiro_cond_p}
        print(f"{cond} Shapiro-Wilk p-value: {shapiro_cond_p:.4f} {'✓ Normal' if shapiro_cond_p >= alpha else '✗ Non-normal'}")
    # Create diagnostic plots
    import scipy.stats as stats
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax1)
    ax1.set_title("Q-Q Plot of Residuals")
    # Histogram with normal curve
    ax2.hist(residuals, bins=15, density=True, alpha=0.7, color='steelblue')
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x, norm.pdf(x, np.mean(residuals), np.std(residuals)), 'r-', linewidth=2)
    ax2.set_title("Histogram of Residuals with Normal Curve")
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Density")
    # Residuals vs fitted values (here, fitted = condition means)
    fitted_values = condition_means
    ax3.scatter(fitted_values, residuals, alpha=0.7)
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_title("Residuals vs Fitted Values")
    ax3.set_xlabel("Fitted Values")
    ax3.set_ylabel("Residuals")
    plt.tight_layout()
    plt.show()
    # Recommendations based on results
    print("\nRECOMMENDATIONS:")
    print("=" * 15)
    if normal_tests >= 3:
        print("✓ Repeated measures ANOVA is appropriate for these data")
        print("✓ All normality assumptions appear to be met")
    elif normal_tests >= 2:
        print("⚠ Consider robust alternatives or data transformation")
        print("⚠ Repeated measures ANOVA may still be appropriate with caution")
    else:
        print("✗ Consider nonparametric alternatives:")
        print("  - Friedman test")
        print("  - Wilcoxon signed-rank test for pairwise comparisons")
        print("  - Permutation tests")
        print("  - Data transformation (log, square root, etc.)")
    return {
        'tests': tests,
        'p_values': p_values,
        'test_names': test_names,
        'normal_tests': normal_tests,
        'total_tests': total_tests,
        'condition_normality': condition_normality
    }

# Check normality with enhanced function
normality_results = check_normality_repeated(repeated_data, "subject", "condition", "score")
```

## Mixed ANOVA

### Between-Subjects and Within-Subjects Factors

```python
# Simulate mixed ANOVA data
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols

np.random.seed(123)
n_per_group = 15

# Generate data for 2x3 mixed design
# Between-subjects factor: Group (A, B)
# Within-subjects factor: Time (Pre, Post, Follow-up)
group = ["Group A"] * (n_per_group * 3) + ["Group B"] * (n_per_group * 3)
subject_id = np.repeat(np.arange(1, n_per_group * 2 + 1), 3)
time = np.tile(["Pre", "Post", "Follow-up"], n_per_group * 2)

# Generate scores with group and time effects
scores = []
for i in range(len(group)):
    subject_effect = np.random.normal(0, 8)
    if group[i] == "Group A":
        if time[i] == "Pre":
            scores.append(70 + subject_effect + np.random.normal(0, 4))
        elif time[i] == "Post":
            scores.append(75 + subject_effect + np.random.normal(0, 4))
        else:
            scores.append(78 + subject_effect + np.random.normal(0, 4))
    else:
        if time[i] == "Pre":
            scores.append(72 + subject_effect + np.random.normal(0, 4))
        elif time[i] == "Post":
            scores.append(80 + subject_effect + np.random.normal(0, 4))
        else:
            scores.append(85 + subject_effect + np.random.normal(0, 4))

# Create data frame
mixed_data = pd.DataFrame({
    'subject': pd.Categorical(subject_id),
    'group': pd.Categorical(group),
    'time': pd.Categorical(time, categories=["Pre", "Post", "Follow-up"], ordered=True),
    'score': scores
})

# Perform mixed ANOVA (using AnovaRM for within, OLS for between)
# For a full mixed model, use statsmodels MixedLM or pingouin.mixed_anova
from statsmodels.stats.anova import AnovaRM
mixed_model = AnovaRM(mixed_data, 'score', 'subject', within=['time'], between=['group']).fit()
print(mixed_model.summary())

# Extract results
df = mixed_model.anova_table
print("\nMixed ANOVA Results:")
for effect in df.index:
    print(f"{effect}: F = {df.loc[effect, 'F Value']:.3f}, p = {df.loc[effect, 'Pr > F']:.4f}")
```

## Nonparametric Alternatives

### Friedman Test

```python
# Perform Friedman test (nonparametric alternative)
from scipy.stats import friedmanchisquare

# Pivot data to wide format for Friedman test
wide_data = repeated_data.pivot(index='subject', columns='condition', values='score')
stat, p = friedmanchisquare(*[wide_data[col].dropna() for col in wide_data.columns])
print("Friedman test results:")
print(f"Chi-squared: {stat:.3f}, p-value: {p:.4f}")

# Compare with parametric results
print("Comparison of parametric and nonparametric tests:")
print(f"Repeated measures ANOVA F-statistic: {anova_result['f_statistic']:.3f}")
print(f"Repeated measures ANOVA p-value: {anova_result['p_value']:.4f}")
print(f"Friedman chi-squared: {stat:.3f}")
print(f"Friedman p-value: {p:.4f}")
```

### Wilcoxon Signed-Rank Test for Pairwise Comparisons

```python
# Pairwise Wilcoxon signed-rank tests
from scipy.stats import wilcoxon

conditions = repeated_data['condition'].unique()
n_conditions = len(conditions)

print("Pairwise Wilcoxon Signed-Rank Tests:")
for i in range(n_conditions-1):
    for j in range(i+1, n_conditions):
        cond1 = conditions[i]
        cond2 = conditions[j]
        # Extract paired data
        data1 = repeated_data[repeated_data['condition'] == cond1].sort_values('subject')['score']
        data2 = repeated_data[repeated_data['condition'] == cond2].sort_values('subject')['score']
        # Only keep subjects present in both conditions
        paired_subjects = np.intersect1d(
            repeated_data[repeated_data['condition'] == cond1]['subject'],
            repeated_data[repeated_data['condition'] == cond2]['subject']
        )
        data1 = repeated_data[(repeated_data['condition'] == cond1) & (repeated_data['subject'].isin(paired_subjects))].sort_values('subject')['score']
        data2 = repeated_data[(repeated_data['condition'] == cond2) & (repeated_data['subject'].isin(paired_subjects))].sort_values('subject')['score']
        # Perform Wilcoxon signed-rank test
        stat, p_val = wilcoxon(data1, data2)
        print(f"{cond1} vs {cond2}:")
        print(f"  V-statistic: {stat}")
        print(f"  p-value: {p_val:.4f}")
        if p_val < 0.05:
            print("  Significant")
        else:
            print("  Non-significant")
        print()
```

## Power Analysis

### Power Analysis for Repeated Measures ANOVA

```python
from statsmodels.stats.power import FTestAnovaPower
import numpy as np

# Power analysis for repeated measures ANOVA
def power_analysis_repeated(n_subjects, n_conditions, effect_size, alpha=0.05):
    # Calculate power using F-test
    power = FTestAnovaPower().power(effect_size=effect_size, nobs=n_subjects, alpha=alpha, k_groups=n_conditions)
    # Calculate required sample size for 80% power
    required_n = FTestAnovaPower().solve_power(effect_size=effect_size, power=0.8, alpha=alpha, k_groups=n_conditions)
    return {
        'power': power,
        'required_n': int(np.ceil(required_n)),
        'effect_size': effect_size,
        'alpha': alpha,
        'n_conditions': n_conditions
    }

# Apply power analysis
f_effect_size = effect_sizes['cohens_f']
repeated_power = power_analysis_repeated(
    n_subjects=len(repeated_data['subject'].unique()),
    n_conditions=len(repeated_data['condition'].unique()),
    effect_size=f_effect_size
)

print("Power Analysis Results:")
print(f"Effect size f: {f_effect_size:.3f}")
print(f"Current power: {repeated_power['power']:.3f}")
print(f"Required sample size for 80% power: {repeated_power['required_n']}")
```

## Practical Examples

### Example 1: Clinical Trial

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM

np.random.seed(123)
n_patients = 25

# Generate data for 4 time points
patient_id = np.repeat(np.arange(1, n_patients + 1), 4)
time_point = np.tile(["Baseline", "Week 4", "Week 8", "Week 12"], n_patients)

# Generate symptom scores with individual differences and time effects
scores = []
for i in range(len(patient_id)):
    patient_effect = np.random.normal(0, 15)
    if time_point[i] == "Baseline":
        scores.append(80 + patient_effect + np.random.normal(0, 8))
    elif time_point[i] == "Week 4":
        scores.append(70 + patient_effect + np.random.normal(0, 8))
    elif time_point[i] == "Week 8":
        scores.append(60 + patient_effect + np.random.normal(0, 8))
    else:
        scores.append(55 + patient_effect + np.random.normal(0, 8))

clinical_data = pd.DataFrame({
    'patient': patient_id,
    'time': pd.Categorical(time_point, categories=["Baseline", "Week 4", "Week 8", "Week 12"], ordered=True),
    'symptom_score': scores
})

# Perform repeated measures ANOVA
anova = AnovaRM(clinical_data, 'symptom_score', 'patient', within=['time']).fit()
print(anova.summary())

# Visualize results
plt.figure(figsize=(8, 5))
for pid in clinical_data['patient'].unique():
    plt.plot(clinical_data[clinical_data['patient'] == pid]['time'],
             clinical_data[clinical_data['patient'] == pid]['symptom_score'],
             alpha=0.3, color='gray')
sns.pointplot(data=clinical_data, x='time', y='symptom_score',
              color='red', errorbar='se', capsize=.1, markers='o', linestyles='-',
              label='Mean')
plt.title('Symptom Scores Over Time')
plt.xlabel('Time Point')
plt.ylabel('Symptom Score')
plt.tight_layout()
plt.show()
```

### Example 2: Learning Study

```python
np.random.seed(123)
n_students = 30

# Generate data for 3 learning sessions
student_id = np.repeat(np.arange(1, n_students + 1), 3)
session = np.tile(["Session 1", "Session 2", "Session 3"], n_students)

# Generate performance scores with individual differences and session effects
scores = []
for i in range(len(student_id)):
    student_effect = np.random.normal(0, 12)
    if session[i] == "Session 1":
        scores.append(65 + student_effect + np.random.normal(0, 6))
    elif session[i] == "Session 2":
        scores.append(75 + student_effect + np.random.normal(0, 6))
    else:
        scores.append(82 + student_effect + np.random.normal(0, 6))

learning_data = pd.DataFrame({
    'student': student_id,
    'session': pd.Categorical(session, categories=["Session 1", "Session 2", "Session 3"], ordered=True),
    'performance': scores
})

# Perform repeated measures ANOVA
anova = AnovaRM(learning_data, 'performance', 'student', within=['session']).fit()
print(anova.summary())

# Visualize results
plt.figure(figsize=(8, 5))
for sid in learning_data['student'].unique():
    plt.plot(learning_data[learning_data['student'] == sid]['session'],
             learning_data[learning_data['student'] == sid]['performance'],
             alpha=0.3, color='gray')
sns.pointplot(data=learning_data, x='session', y='performance',
              color='blue', errorbar='se', capsize=.1, markers='o', linestyles='-',
              label='Mean')
plt.title('Performance Over Learning Sessions')
plt.xlabel('Session')
plt.ylabel('Performance Score')
plt.tight_layout()
plt.show()
```

### Example 3: Exercise Study

```python
np.random.seed(123)
n_participants = 20

# Generate data for 2x3 mixed design
exercise_type = np.repeat(["Aerobic", "Strength"], n_participants * 3)
participant_id = np.repeat(np.arange(1, n_participants * 2 + 1), 3)
time_point = np.tile(["Pre", "Mid", "Post"], n_participants * 2)

# Generate fitness scores with individual differences, exercise, and time effects
scores = []
for i in range(len(exercise_type)):
    participant_effect = np.random.normal(0, 10)
    if exercise_type[i] == "Aerobic":
        if time_point[i] == "Pre":
            scores.append(60 + participant_effect + np.random.normal(0, 5))
        elif time_point[i] == "Mid":
            scores.append(70 + participant_effect + np.random.normal(0, 5))
        else:
            scores.append(75 + participant_effect + np.random.normal(0, 5))
    else:
        if time_point[i] == "Pre":
            scores.append(55 + participant_effect + np.random.normal(0, 5))
        elif time_point[i] == "Mid":
            scores.append(65 + participant_effect + np.random.normal(0, 5))
        else:
            scores.append(80 + participant_effect + np.random.normal(0, 5))

exercise_data = pd.DataFrame({
    'participant': participant_id,
    'exercise_type': exercise_type,
    'time': pd.Categorical(time_point, categories=["Pre", "Mid", "Post"], ordered=True),
    'fitness_score': scores
})

# Perform mixed ANOVA (using AnovaRM for within, OLS for between)
from statsmodels.stats.anova import AnovaRM
mixed_model = AnovaRM(exercise_data, 'fitness_score', 'participant', within=['time'], between=['exercise_type']).fit()
print(mixed_model.summary())

# Visualize interaction
plt.figure(figsize=(8, 5))
sns.pointplot(data=exercise_data, x='time', y='fitness_score', hue='exercise_type',
              errorbar='se', capsize=.1, markers='o', linestyles='-', dodge=True)
plt.title('Fitness Scores by Exercise Type and Time')
plt.xlabel('Time Point')
plt.ylabel('Fitness Score')
plt.tight_layout()
plt.show()
```

## Best Practices

### Test Selection Guidelines

```python
# Function to help choose appropriate repeated measures test

def choose_repeated_test(data, subject_var, condition_var, response_var, alpha=0.05):
    print("=== REPEATED MEASURES TEST SELECTION ===\n")
    # Check normality
    normality_results = check_normality_repeated(data, subject_var, condition_var, response_var, alpha=alpha)
    # Check sphericity
    sphericity_results = check_sphericity(data, subject_var, condition_var, response_var, alpha=alpha)
    # Sample size
    n_subjects = data[subject_var].nunique()
    n_conditions = data[condition_var].nunique()
    print(f"Sample size: {n_subjects} subjects, {n_conditions} conditions\n")
    print("FINAL RECOMMENDATION:")
    # Decision logic
    normal_residuals = normality_results['tests']['shapiro']['pvalue'] >= alpha
    sphericity_ok = sphericity_results['gg_epsilon'] >= 0.75
    if normal_residuals and sphericity_ok:
        print("Use standard repeated measures ANOVA\nAll assumptions are met")
    elif normal_residuals and not sphericity_ok:
        print("Use repeated measures ANOVA with Greenhouse-Geisser correction\nResiduals are normal but sphericity is violated")
    else:
        print("Use Friedman test (nonparametric alternative)\nResiduals are not normal")
    return {
        'normality': normality_results,
        'sphericity': sphericity_results,
        'n_subjects': n_subjects,
        'n_conditions': n_conditions
    }

# Example usage:
# test_selection = choose_repeated_test(repeated_data, "subject", "condition", "score")
```

### Reporting Guidelines

```python
# Function to generate comprehensive repeated measures ANOVA report

def generate_repeated_report(anova_result, data, subject_var, condition_var, response_var, effect_sizes=None, alpha=0.05):
    print("=== REPEATED MEASURES ANOVA REPORT ===\n")
    # Descriptive statistics
    desc_stats = data.groupby(condition_var)[response_var].agg(['count', 'mean', 'std'])
    print("DESCRIPTIVE STATISTICS:")
    print(desc_stats)
    print()
    # ANOVA results
    print("REPEATED MEASURES ANOVA RESULTS:")
    try:
        f_stat = anova_result.anova_table['F Value'][0]
        p_value = anova_result.anova_table['Pr > F'][0]
        df_conditions = anova_result.anova_table['Num DF'][0]
        df_error = anova_result.anova_table['Den DF'][0]
    except Exception:
        # fallback for different statsmodels versions
        f_stat = anova_result.anova_table.iloc[0]['F Value']
        p_value = anova_result.anova_table.iloc[0]['Pr > F']
        df_conditions = anova_result.anova_table.iloc[0]['Num DF']
        df_error = anova_result.anova_table.iloc[0]['Den DF']
    print(f"F-statistic: {f_stat:.3f}")
    print(f"Degrees of freedom: {df_conditions}, {df_error}")
    print(f"p-value: {p_value:.4f}")
    # Effect size
    if effect_sizes is not None and 'partial_eta2' in effect_sizes:
        partial_eta2 = effect_sizes['partial_eta2']
        print(f"Effect size (partial η²): {partial_eta2:.3f}")
        print(f"Interpretation: {interpret_effect_size(partial_eta2)}\n")
    # Post hoc results (if available)
    # (Assume Tukey's HSD or pairwise t-tests have been run separately)
    print("(Add post hoc results here if available)")
    # Conclusion
    print("\nCONCLUSION:")
    if p_value < alpha:
        print(f"Reject the null hypothesis (p < {alpha})")
        print("There are significant differences between conditions")
    else:
        print(f"Fail to reject the null hypothesis (p >= {alpha})")
        print("There is insufficient evidence of differences between conditions")

# Example usage:
# generate_repeated_report(anova, repeated_data, "subject", "condition", "score", effect_sizes)
```

## Comprehensive Exercises

### Exercise 1: Basic Repeated Measures ANOVA Analysis

**Objective**: Perform a complete repeated measures ANOVA analysis from data preparation to interpretation.

**Scenario**: A researcher is studying the effectiveness of a new learning intervention. 15 students are tested on their problem-solving skills at three time points: before the intervention (baseline), immediately after (post-test), and 3 months later (follow-up).

**Tasks**:
1. Generate realistic data for this scenario
2. Perform descriptive statistics and visualization
3. Conduct repeated measures ANOVA with manual calculations
4. Check assumptions thoroughly
5. Calculate and interpret effect sizes
6. Perform post hoc analysis if needed
7. Write a comprehensive report

**Requirements**:
- Use 15 subjects and 3 time points
- Include individual differences and learning effects
- Generate data with realistic effect sizes
- Provide detailed interpretation of results

**Expected Learning Outcomes**:
- Understanding of repeated measures ANOVA workflow
- Ability to interpret individual differences
- Knowledge of assumption checking procedures
- Skills in effect size calculation and interpretation

**Solution Framework**:
```r
# Exercise 1 Solution Framework
set.seed(123)
n_subjects <- 15
n_timepoints <- 3

# Generate data with learning effects
# ... (students will implement this)

# Perform comprehensive analysis
# ... (students will implement this)

# Check assumptions
# ... (students will implement this)

# Calculate effect sizes
# ... (students will implement this)

# Write report
# ... (students will implement this)
```

### Exercise 2: Mixed ANOVA with Interaction Effects

**Objective**: Analyze a mixed ANOVA design with both between-subjects and within-subjects factors.

**Scenario**: A clinical trial comparing two treatments (A and B) for anxiety reduction. Patients are randomly assigned to treatments and measured at baseline, week 4, and week 8.

**Tasks**:
1. Create a 2×3 mixed design dataset
2. Analyze main effects and interaction effects
3. Perform simple effects analysis
4. Create interaction plots
5. Conduct post hoc tests for significant effects
6. Report results comprehensively

**Requirements**:
- 20 patients per treatment group
- 3 measurement time points
- Include treatment × time interaction
- Realistic anxiety scores (0-100 scale)

**Expected Learning Outcomes**:
- Understanding of mixed ANOVA designs
- Ability to interpret interaction effects
- Skills in simple effects analysis
- Knowledge of appropriate post hoc procedures

### Exercise 3: Advanced Assumption Checking and Remedies

**Objective**: Practice comprehensive assumption checking and implement appropriate remedies for violations.

**Scenario**: You have repeated measures data that may violate sphericity and normality assumptions.

**Tasks**:
1. Generate data with known assumption violations
2. Perform comprehensive assumption checking
3. Apply appropriate corrections (Greenhouse-Geisser, Huynh-Feldt)
4. Compare results with and without corrections
5. Implement nonparametric alternatives
6. Compare parametric and nonparametric results

**Requirements**:
- Create data with sphericity violations
- Create data with normality violations
- Apply multiple correction methods
- Use Friedman test and Wilcoxon tests
- Provide recommendations for each scenario

**Expected Learning Outcomes**:
- Understanding of assumption violations
- Knowledge of correction methods
- Skills in nonparametric alternatives
- Ability to make informed decisions about analysis methods

### Exercise 4: Effect Size Analysis and Power

**Objective**: Conduct comprehensive effect size analysis and power calculations.

**Scenario**: You need to design a study to detect different effect sizes with adequate power.

**Tasks**:
1. Calculate effect sizes for existing data
2. Perform power analysis for different scenarios
3. Determine required sample sizes
4. Analyze effect size confidence intervals
5. Compare different effect size measures
6. Make recommendations for study design

**Requirements**:
- Use multiple effect size measures
- Calculate bootstrap confidence intervals
- Perform power analysis for different effect sizes
- Consider practical significance
- Provide sample size recommendations

**Expected Learning Outcomes**:
- Understanding of effect size interpretation
- Knowledge of power analysis procedures
- Skills in sample size determination
- Ability to assess practical significance

### Exercise 5: Real-World Application

**Objective**: Apply repeated measures ANOVA to a real-world research scenario.

**Scenario**: Choose one of the following:
- **Educational Research**: Student performance across multiple exams
- **Clinical Research**: Patient outcomes over treatment period
- **Sports Science**: Athlete performance across training phases
- **Psychology**: Cognitive task performance under different conditions

**Tasks**:
1. Design a realistic research study
2. Generate appropriate data
3. Perform complete analysis workflow
4. Create publication-ready visualizations
5. Write a research report
6. Address potential limitations

**Requirements**:
- Realistic research design
- Appropriate sample sizes
- Meaningful dependent variables
- Comprehensive analysis
- Professional reporting

**Expected Learning Outcomes**:
- Application of repeated measures ANOVA to real problems
- Skills in research design
- Ability to communicate results effectively
- Understanding of research limitations

### Exercise 6: Advanced Topics

**Objective**: Explore advanced topics in repeated measures analysis.

**Tasks**:
1. **Unbalanced Designs**: Handle missing data and unbalanced designs
2. **Robust Methods**: Implement robust repeated measures ANOVA
3. **Multivariate Approaches**: Use MANOVA for multiple dependent variables
4. **Trend Analysis**: Perform polynomial contrasts and trend analysis
5. **Bootstrap Methods**: Use bootstrap for confidence intervals and hypothesis testing

**Requirements**:
- Implement multiple advanced techniques
- Compare results across methods
- Understand when to use each approach
- Provide practical recommendations

**Expected Learning Outcomes**:
- Knowledge of advanced repeated measures techniques
- Understanding of robust methods
- Skills in multivariate analysis
- Ability to choose appropriate methods for different situations

## Best Practices and Guidelines

### Test Selection Guidelines

```r
# Function to help choose appropriate repeated measures test
choose_repeated_test <- function(data, subject_var, condition_var, response_var, alpha = 0.05) {
  cat("=== REPEATED MEASURES TEST SELECTION GUIDE ===\n\n")
  
  # Check sample size
  n_subjects <- length(unique(data[[subject_var]]))
  n_conditions <- length(unique(data[[condition_var]]))
  
  cat("SAMPLE SIZE ASSESSMENT:\n")
  cat("=" * 24, "\n")
  cat("Number of subjects:", n_subjects, "\n")
  cat("Number of conditions:", n_conditions, "\n")
  
  if (n_subjects < 10) {
    cat("⚠ Small sample size - consider nonparametric alternatives\n")
  } else if (n_subjects < 20) {
    cat("⚠ Moderate sample size - use with caution\n")
  } else {
    cat("✓ Adequate sample size for parametric tests\n")
  }
  
  # Check normality
  normality_results <- check_normality_repeated(data, subject_var, condition_var, response_var, alpha)
  
  # Check sphericity
  sphericity_results <- check_sphericity(data, subject_var, condition_var, response_var, alpha)
  
  # Decision matrix
  cat("\nDECISION MATRIX:\n")
  cat("=" * 16, "\n")
  
  normal_data <- normality_results$normal_tests >= 3
  sphericity_ok <- sphericity_results$sphericity_indicators >= 3
  
  if (normal_data && sphericity_ok) {
    cat("✓ RECOMMENDATION: Standard repeated measures ANOVA\n")
    cat("  - All assumptions met\n")
    cat("  - Most powerful test available\n")
  } else if (normal_data && !sphericity_ok) {
    cat("✓ RECOMMENDATION: Repeated measures ANOVA with corrections\n")
    cat("  - Use Greenhouse-Geisser or Huynh-Feldt corrections\n")
    cat("  - Data are normal but sphericity violated\n")
  } else if (!normal_data && sphericity_ok) {
    cat("✓ RECOMMENDATION: Robust repeated measures ANOVA\n")
    cat("  - Consider Yuen's t-test or trimmed means\n")
    cat("  - Sphericity met but normality violated\n")
  } else {
    cat("✓ RECOMMENDATION: Nonparametric alternatives\n")
    cat("  - Use Friedman test\n")
    cat("  - Consider permutation tests\n")
    cat("  - Both normality and sphericity violated\n")
  }
  
  # Additional considerations
  cat("\nADDITIONAL CONSIDERATIONS:\n")
  cat("=" * 25, "\n")
  
  if (n_conditions == 2) {
    cat("• For 2 conditions, consider paired t-test\n")
    cat("• Simpler and equivalent to repeated measures ANOVA\n")
  }
  
  if (n_conditions > 5) {
    cat("• Many conditions - consider trend analysis\n")
    cat("• Polynomial contrasts may be more informative\n")
  }
  
  if (n_subjects > 50) {
    cat("• Large sample - parametric tests are robust\n")
    cat("• Minor assumption violations may be acceptable\n")
  }
  
  return(list(
    n_subjects = n_subjects,
    n_conditions = n_conditions,
    normality_results = normality_results,
    sphericity_results = sphericity_results,
    recommendation = ifelse(normal_data && sphericity_ok, "Standard ANOVA",
                           ifelse(normal_data && !sphericity_ok, "ANOVA with corrections",
                                  ifelse(!normal_data && sphericity_ok, "Robust ANOVA", "Nonparametric")))
  ))
}

# Apply test selection guide
test_selection <- choose_repeated_test(repeated_data, "subject", "condition", "score")
```

### Data Preparation Best Practices

```r
# Function to prepare data for repeated measures analysis
prepare_repeated_data <- function(data, subject_var, condition_var, response_var) {
  cat("=== DATA PREPARATION CHECKLIST ===\n\n")
  
  # Check for missing data
  missing_data <- sum(is.na(data[[response_var]]))
  total_obs <- nrow(data)
  missing_pct <- (missing_data / total_obs) * 100
  
  cat("MISSING DATA ASSESSMENT:\n")
  cat("=" * 24, "\n")
  cat("Missing observations:", missing_data, "out of", total_obs, "(", round(missing_pct, 1), "%)\n")
  
  if (missing_pct > 5) {
    cat("⚠ High missing data - consider imputation or exclusion\n")
  } else if (missing_pct > 0) {
    cat("⚠ Some missing data - check for patterns\n")
  } else {
    cat("✓ No missing data\n")
  }
  
  # Check data structure
  cat("\nDATA STRUCTURE CHECK:\n")
  cat("=" * 20, "\n")
  
  # Check for balanced design
  subject_counts <- table(data[[subject_var]])
  condition_counts <- table(data[[condition_var]])
  
  if (length(unique(subject_counts)) == 1) {
    cat("✓ Balanced design (all subjects have same number of observations)\n")
  } else {
    cat("⚠ Unbalanced design - some subjects have different numbers of observations\n")
  }
  
  # Check for outliers
  outliers <- boxplot.stats(data[[response_var]])$out
  cat("Number of outliers:", length(outliers), "\n")
  
  if (length(outliers) > 0) {
    cat("⚠ Outliers detected - consider robust methods or data transformation\n")
  } else {
    cat("✓ No outliers detected\n")
  }
  
  # Data quality recommendations
  cat("\nDATA QUALITY RECOMMENDATIONS:\n")
  cat("=" * 29, "\n")
  
  if (missing_pct > 10) {
    cat("• Consider multiple imputation\n")
    cat("• Check for missing data patterns\n")
    cat("• Consider excluding subjects with too much missing data\n")
  }
  
  if (length(unique(subject_counts)) > 1) {
    cat("• Consider mixed-effects models for unbalanced data\n")
    cat("• Check if missing data is missing at random\n")
  }
  
  if (length(outliers) > 0) {
    cat("• Consider robust statistical methods\n")
    cat("• Check if outliers are data entry errors\n")
    cat("• Consider data transformation\n")
  }
  
  # Create cleaned dataset
  cleaned_data <- data[!is.na(data[[response_var]]), ]
  
  return(list(
    original_data = data,
    cleaned_data = cleaned_data,
    missing_data = missing_data,
    missing_pct = missing_pct,
    outliers = outliers,
    subject_counts = subject_counts,
    condition_counts = condition_counts
  ))
}

# Apply data preparation
data_prep <- prepare_repeated_data(repeated_data, "subject", "condition", "score")
```

### Reporting Guidelines

```r
# Function to generate comprehensive repeated measures ANOVA report
generate_comprehensive_report <- function(anova_result, data, subject_var, condition_var, response_var, 
                                        effect_sizes, normality_results, sphericity_results) {
  cat("=== COMPREHENSIVE REPEATED MEASURES ANOVA REPORT ===\n\n")
  
  # Study description
  cat("STUDY DESCRIPTION:\n")
  cat("=" * 19, "\n")
  cat("Design: Repeated measures ANOVA\n")
  cat("Subjects:", length(unique(data[[subject_var]])), "\n")
  cat("Conditions:", length(unique(data[[condition_var]])), "\n")
  cat("Total observations:", nrow(data), "\n\n")
  
  # Descriptive statistics
  desc_stats <- data %>%
    group_by(!!sym(condition_var)) %>%
    summarise(
      n = n(),
      mean = mean(!!sym(response_var), na.rm = TRUE),
      sd = sd(!!sym(response_var), na.rm = TRUE),
      se = sd / sqrt(n),
      ci_lower = mean - qt(0.975, n-1) * se,
      ci_upper = mean + qt(0.975, n-1) * se
    )
  
  cat("DESCRIPTIVE STATISTICS:\n")
  cat("=" * 23, "\n")
  print(desc_stats)
  cat("\n")
  
  # Assumption checking summary
  cat("ASSUMPTION CHECKING:\n")
  cat("=" * 19, "\n")
  
  # Normality
  normal_tests <- normality_results$normal_tests
  total_tests <- normality_results$total_tests
  cat("Normality:", normal_tests, "out of", total_tests, "tests passed\n")
  
  # Sphericity
  sphericity_indicators <- sphericity_results$sphericity_indicators
  total_indicators <- sphericity_results$total_indicators
  cat("Sphericity:", sphericity_indicators, "out of", total_indicators, "indicators met\n")
  
  # ANOVA results
  anova_summary <- summary(anova_result)
  f_stat <- anova_summary[[2]][[1]]$`F value`[1]
  p_value <- anova_summary[[2]][[1]]$`Pr(>F)`[1]
  df_conditions <- anova_summary[[2]][[1]]$Df[1]
  df_error <- anova_summary[[2]][[1]]$Df[2]
  
  cat("\nREPEATED MEASURES ANOVA RESULTS:\n")
  cat("=" * 32, "\n")
  cat("F-statistic:", round(f_stat, 3), "\n")
  cat("Degrees of freedom:", df_conditions, ",", df_error, "\n")
  cat("p-value:", round(p_value, 4), "\n")
  
  # Effect sizes
  cat("\nEFFECT SIZES:\n")
  cat("=" * 12, "\n")
  cat("Partial η²:", round(effect_sizes$partial_eta2, 4), "\n")
  cat("η²:", round(effect_sizes$eta2, 4), "\n")
  cat("ω²:", round(effect_sizes$omega2, 4), "\n")
  cat("Cohen's f:", round(effect_sizes$cohens_f, 4), "\n")
  
  # Interpretation
  cat("\nINTERPRETATION:\n")
  cat("=" * 14, "\n")
  
  alpha <- 0.05
  if (p_value < alpha) {
    cat("✓ Significant main effect of condition (p <", alpha, ")\n")
    cat("✓ There are significant differences between conditions\n")
  } else {
    cat("✗ No significant main effect of condition (p >=", alpha, ")\n")
    cat("✗ There is insufficient evidence of differences between conditions\n")
  }
  
  # Effect size interpretation
  if (effect_sizes$partial_eta2 > 0.14) {
    cat("✓ Large effect size\n")
  } else if (effect_sizes$partial_eta2 > 0.06) {
    cat("✓ Medium effect size\n")
  } else if (effect_sizes$partial_eta2 > 0.01) {
    cat("✓ Small effect size\n")
  } else {
    cat("✓ Negligible effect size\n")
  }
  
  # Recommendations
  cat("\nRECOMMENDATIONS:\n")
  cat("=" * 15, "\n")
  
  if (normal_tests >= 3 && sphericity_indicators >= 3) {
    cat("✓ Results are reliable and valid\n")
    cat("✓ Standard repeated measures ANOVA is appropriate\n")
  } else if (normal_tests >= 2 && sphericity_indicators >= 2) {
    cat("⚠ Results should be interpreted with caution\n")
    cat("⚠ Consider robust alternatives for future studies\n")
  } else {
    cat("✗ Results may not be reliable\n")
    cat("✗ Consider nonparametric alternatives\n")
  }
  
  # Post hoc recommendations
  if (p_value < alpha) {
    cat("✓ Perform post hoc tests to identify specific differences\n")
    cat("✓ Consider trend analysis for ordered conditions\n")
  }
  
  return(list(
    descriptive_stats = desc_stats,
    anova_results = list(f_stat = f_stat, p_value = p_value, df = c(df_conditions, df_error)),
    effect_sizes = effect_sizes,
    interpretation = list(significant = p_value < alpha, effect_size = effect_sizes$partial_eta2)
  ))
}

# Generate comprehensive report
comprehensive_report <- generate_comprehensive_report(
  repeated_model, repeated_data, "subject", "condition", "score",
  effect_sizes, normality_results, sphericity_results
)
```

## Next Steps

In the next chapter, we'll learn about correlation analysis for examining relationships between variables.

---

**Key Takeaways:**
- Repeated measures ANOVA is more powerful than between-subjects designs
- Always check sphericity and normality assumptions
- Effect sizes provide important information about practical significance
- Post hoc tests are necessary when the main effect is significant
- Nonparametric alternatives exist for non-normal data
- Mixed ANOVA combines between-subjects and within-subjects factors
- Proper reporting includes descriptive statistics, test results, and effect sizes 