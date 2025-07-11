# Two-Way ANOVA

## Overview

Two-way Analysis of Variance (ANOVA) is a powerful statistical technique used to analyze the effects of two independent variables (factors) on a dependent variable, including their interaction effects. This design is more sophisticated than one-way ANOVA as it allows researchers to examine both main effects and interaction effects simultaneously, providing a more complete understanding of the relationships between variables.

### Key Concepts

**Main Effects**: The individual effect of each factor on the dependent variable, ignoring the other factor.

**Interaction Effects**: The combined effect of both factors that cannot be explained by their individual main effects alone. An interaction occurs when the effect of one factor depends on the level of the other factor.

**Factorial Design**: A research design where all combinations of factor levels are included in the study.

### Mathematical Foundation

The two-way ANOVA model can be expressed as:

```math
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}
```

Where:
- $`Y_{ijk}`$ is the observed value for the kth observation in the ith level of factor A and jth level of factor B
- $`\mu`$ is the overall population mean
- $`\alpha_i`$ is the effect of the ith level of factor A (main effect of factor A)
- $`\beta_j`$ is the effect of the jth level of factor B (main effect of factor B)
- $`(\alpha\beta)_{ij}`$ is the interaction effect between the ith level of factor A and jth level of factor B
- $`\epsilon_{ijk}`$ is the random error term

### Sum of Squares Decomposition

The total variability in the data is partitioned into four components:

```math
SS_{Total} = SS_A + SS_B + SS_{AB} + SS_{Error}
```

Where:
- $`SS_{Total} = \sum_{i=1}^{a}\sum_{j=1}^{b}\sum_{k=1}^{n_{ij}}(Y_{ijk} - \bar{Y}_{...})^2`$
- $`SS_A = \sum_{i=1}^{a}bn_i(\bar{Y}_{i..} - \bar{Y}_{...})^2`$ (Factor A main effect)
- $`SS_B = \sum_{j=1}^{b}an_j(\bar{Y}_{.j.} - \bar{Y}_{...})^2`$ (Factor B main effect)
- $`SS_{AB} = \sum_{i=1}^{a}\sum_{j=1}^{b}n_{ij}(\bar{Y}_{ij.} - \bar{Y}_{i..} - \bar{Y}_{.j.} + \bar{Y}_{...})^2`$ (Interaction effect)
- $`SS_{Error} = \sum_{i=1}^{a}\sum_{j=1}^{b}\sum_{k=1}^{n_{ij}}(Y_{ijk} - \bar{Y}_{ij.})^2`$ (Error term)

### Degrees of Freedom

```math
df_A = a - 1
df_B = b - 1
df_{AB} = (a - 1)(b - 1)
df_{Error} = N - ab
df_{Total} = N - 1
```

Where $`a`$ and $`b`$ are the number of levels for factors A and B respectively, and $`N`$ is the total sample size.

### F-Statistics

```math
F_A = \frac{MS_A}{MS_{Error}} = \frac{SS_A/df_A}{SS_{Error}/df_{Error}}
```

```math
F_B = \frac{MS_B}{MS_{Error}} = \frac{SS_B/df_B}{SS_{Error}/df_{Error}}
```

```math
F_{AB} = \frac{MS_{AB}}{MS_{Error}} = \frac{SS_{AB}/df_{AB}}{SS_{Error}/df_{Error}}
```

### Effect Size Measures

**Partial Eta-Squared** (most commonly used):
```math
\eta_p^2 = \frac{SS_{Effect}}{SS_{Effect} + SS_{Error}}
```

**Eta-Squared**:
```math
\eta^2 = \frac{SS_{Effect}}{SS_{Total}}
```

**Omega-Squared** (unbiased estimate):
```math
\omega^2 = \frac{SS_{Effect} - df_{Effect} \times MS_{Error}}{SS_{Total} + MS_{Error}}
```

## Basic Two-Way ANOVA

### Manual Calculation

The manual calculation helps understand the underlying mathematics and provides insight into how the statistical software performs the analysis.

```python
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import f

# Load sample data (using seaborn's mpg dataset as a proxy for mtcars)
import seaborn as sns
mtcars = sns.load_dataset('mpg').dropna(subset=['mpg', 'cylinders', 'origin'])

# Create factors for analysis
mtcars['cyl_factor'] = pd.Categorical(mtcars['cylinders'], categories=[4, 6, 8], ordered=True)
mtcars['cyl_factor'] = mtcars['cyl_factor'].map({4: '4-cylinder', 6: '6-cylinder', 8: '8-cylinder'})
mtcars['am_factor'] = pd.Categorical(mtcars['origin'], categories=['usa', 'europe', 'japan'])  # Using 'origin' as a proxy for transmission

# Manual two-way ANOVA calculation
# (This is a didactic implementation; for real analysis, use statsmodels as shown below)
def manual_two_way_anova(data, factor1, factor2, response):
    # Drop missing
    data = data.dropna(subset=[factor1, factor2, response])
    overall_mean = data[response].mean()
    factor1_levels = data[factor1].unique()
    factor2_levels = data[factor2].unique()
    n_total = len(data)
    n_factor1_levels = len(factor1_levels)
    n_factor2_levels = len(factor2_levels)

    # Cell means
    cell_means = data.pivot_table(index=factor1, columns=factor2, values=response, aggfunc='mean')
    factor1_means = data.groupby(factor1)[response].mean()
    factor2_means = data.groupby(factor2)[response].mean()

    # SS Total
    ss_total = ((data[response] - overall_mean) ** 2).sum()
    # SS Factor 1
    ss_factor1 = sum(data[factor1].value_counts()[lvl] * (factor1_means[lvl] - overall_mean) ** 2 for lvl in factor1_levels)
    # SS Factor 2
    ss_factor2 = sum(data[factor2].value_counts()[lvl] * (factor2_means[lvl] - overall_mean) ** 2 for lvl in factor2_levels)
    # SS Interaction
    ss_interaction = 0
    for i in factor1_levels:
        for j in factor2_levels:
            n_cell = len(data[(data[factor1] == i) & (data[factor2] == j)])
            if n_cell == 0:
                continue
            cell_mean = cell_means.loc[i, j]
            interaction_effect = cell_mean - factor1_means[i] - factor2_means[j] + overall_mean
            ss_interaction += n_cell * interaction_effect ** 2
    # SS Error
    ss_error = ss_total - ss_factor1 - ss_factor2 - ss_interaction

    # Degrees of freedom
    df_factor1 = n_factor1_levels - 1
    df_factor2 = n_factor2_levels - 1
    df_interaction = (n_factor1_levels - 1) * (n_factor2_levels - 1)
    df_error = n_total - n_factor1_levels * n_factor2_levels
    df_total = n_total - 1

    # Mean Squares
    ms_factor1 = ss_factor1 / df_factor1
    ms_factor2 = ss_factor2 / df_factor2
    ms_interaction = ss_interaction / df_interaction
    ms_error = ss_error / df_error

    # F-statistics
    f_factor1 = ms_factor1 / ms_error
    f_factor2 = ms_factor2 / ms_error
    f_interaction = ms_interaction / ms_error

    # p-values
    p_factor1 = 1 - f.cdf(f_factor1, df_factor1, df_error)
    p_factor2 = 1 - f.cdf(f_factor2, df_factor2, df_error)
    p_interaction = 1 - f.cdf(f_interaction, df_interaction, df_error)

    # Effect sizes (partial eta-squared)
    partial_eta2_factor1 = ss_factor1 / (ss_factor1 + ss_error)
    partial_eta2_factor2 = ss_factor2 / (ss_factor2 + ss_error)
    partial_eta2_interaction = ss_interaction / (ss_interaction + ss_error)

    return {
        'ss_factor1': ss_factor1,
        'ss_factor2': ss_factor2,
        'ss_interaction': ss_interaction,
        'ss_error': ss_error,
        'ss_total': ss_total,
        'df_factor1': df_factor1,
        'df_factor2': df_factor2,
        'df_interaction': df_interaction,
        'df_error': df_error,
        'df_total': df_total,
        'ms_factor1': ms_factor1,
        'ms_factor2': ms_factor2,
        'ms_interaction': ms_interaction,
        'ms_error': ms_error,
        'f_factor1': f_factor1,
        'f_factor2': f_factor2,
        'f_interaction': f_interaction,
        'p_factor1': p_factor1,
        'p_factor2': p_factor2,
        'p_interaction': p_interaction,
        'partial_eta2_factor1': partial_eta2_factor1,
        'partial_eta2_factor2': partial_eta2_factor2,
        'partial_eta2_interaction': partial_eta2_interaction,
        'cell_means': cell_means,
        'factor1_means': factor1_means,
        'factor2_means': factor2_means
    }

# Apply manual calculation
anova_result = manual_two_way_anova(mtcars, 'cyl_factor', 'am_factor', 'mpg')

print("Manual Two-Way ANOVA Results:")
print(f"Factor 1 (Cylinders) F-statistic: {anova_result['f_factor1']:.3f}")
print(f"Factor 1 p-value: {anova_result['p_factor1']:.4f}")
print(f"Factor 2 (Transmission) F-statistic: {anova_result['f_factor2']:.3f}")
print(f"Factor 2 p-value: {anova_result['p_factor2']:.4f}")
print(f"Interaction F-statistic: {anova_result['f_interaction']:.3f}")
print(f"Interaction p-value: {anova_result['p_interaction']:.4f}")
```

### Using Python's Built-in Two-Way ANOVA

Python's `statsmodels` provides a convenient way to perform two-way ANOVA with automatic calculation of all statistics.

```python
# Perform two-way ANOVA using statsmodels
model = ols('mpg ~ C(cyl_factor) * C(am_factor)', data=mtcars).fit()
two_way_anova = anova_lm(model, typ=2)
print(two_way_anova)

# Extract key statistics
f_cylinders = two_way_anova.loc['C(cyl_factor)', 'F']
f_transmission = two_way_anova.loc['C(am_factor)', 'F']
f_interaction = two_way_anova.loc['C(cyl_factor):C(am_factor)', 'F']
p_cylinders = two_way_anova.loc['C(cyl_factor)', 'PR(>F)']
p_transmission = two_way_anova.loc['C(am_factor)', 'PR(>F)']
p_interaction = two_way_anova.loc['C(cyl_factor):C(am_factor)', 'PR(>F)']

print("Two-Way ANOVA Results:")
print(f"Cylinders F-statistic: {f_cylinders:.3f}")
print(f"Cylinders p-value: {p_cylinders:.4f}")
print(f"Transmission F-statistic: {f_transmission:.3f}")
print(f"Transmission p-value: {p_transmission:.4f}")
print(f"Interaction F-statistic: {f_interaction:.3f}")
print(f"Interaction p-value: {p_interaction:.4f}")
```

## Descriptive Statistics

### Understanding Cell Means and Marginal Means

In two-way ANOVA, we need to understand three types of means:

1. **Cell Means**: The mean of the dependent variable for each combination of factor levels
2. **Marginal Means**: The mean of the dependent variable for each level of a factor, averaged across all levels of the other factor
3. **Grand Mean**: The overall mean of the dependent variable across all observations

### Mathematical Definitions

**Cell Mean**: $`\bar{Y}_{ij.} = \frac{1}{n_{ij}}\sum_{k=1}^{n_{ij}}Y_{ijk}`$

**Marginal Mean for Factor A**: $`\bar{Y}_{i..} = \frac{1}{bn_i}\sum_{j=1}^{b}\sum_{k=1}^{n_{ij}}Y_{ijk}`$

**Marginal Mean for Factor B**: $`\bar{Y}_{.j.} = \frac{1}{an_j}\sum_{i=1}^{a}\sum_{k=1}^{n_{ij}}Y_{ijk}`$

**Grand Mean**: $`\bar{Y}_{...} = \frac{1}{N}\sum_{i=1}^{a}\sum_{j=1}^{b}\sum_{k=1}^{n_{ij}}Y_{ijk}`$

### Cell Means and Marginal Means

```python
# Calculate descriptive statistics
import pandas as pd
import numpy as np
from scipy import stats

# Cell means - these represent the mean for each combination of factor levels
cell_means = mtcars.groupby(['cyl_factor', 'am_factor']).agg({
    'mpg': ['count', 'mean', 'std']
}).round(3)
cell_means.columns = ['n', 'mean', 'sd']
cell_means = cell_means.reset_index()

# Calculate standard error and confidence intervals
cell_means['se'] = cell_means['sd'] / np.sqrt(cell_means['n'])
cell_means['ci_lower'] = cell_means['mean'] - stats.t.ppf(0.975, cell_means['n']-1) * cell_means['se']
cell_means['ci_upper'] = cell_means['mean'] + stats.t.ppf(0.975, cell_means['n']-1) * cell_means['se']

print("Cell Means (Mean MPG for each combination):")
print(cell_means)

# Marginal means for Factor 1 (Cylinders) - averaged across transmission types
marginal_cylinders = mtcars.groupby('cyl_factor').agg({
    'mpg': ['count', 'mean', 'std']
}).round(3)
marginal_cylinders.columns = ['n', 'mean', 'sd']
marginal_cylinders = marginal_cylinders.reset_index()

marginal_cylinders['se'] = marginal_cylinders['sd'] / np.sqrt(marginal_cylinders['n'])
marginal_cylinders['ci_lower'] = marginal_cylinders['mean'] - stats.t.ppf(0.975, marginal_cylinders['n']-1) * marginal_cylinders['se']
marginal_cylinders['ci_upper'] = marginal_cylinders['mean'] + stats.t.ppf(0.975, marginal_cylinders['n']-1) * marginal_cylinders['se']

print("\nMarginal Means - Cylinders (averaged across transmission types):")
print(marginal_cylinders)

# Marginal means for Factor 2 (Transmission) - averaged across cylinder types
marginal_transmission = mtcars.groupby('am_factor').agg({
    'mpg': ['count', 'mean', 'std']
}).round(3)
marginal_transmission.columns = ['n', 'mean', 'sd']
marginal_transmission = marginal_transmission.reset_index()

marginal_transmission['se'] = marginal_transmission['sd'] / np.sqrt(marginal_transmission['n'])
marginal_transmission['ci_lower'] = marginal_transmission['mean'] - stats.t.ppf(0.975, marginal_transmission['n']-1) * marginal_transmission['se']
marginal_transmission['ci_upper'] = marginal_transmission['mean'] + stats.t.ppf(0.975, marginal_transmission['n']-1) * marginal_transmission['se']

print("\nMarginal Means - Transmission (averaged across cylinder types):")
print(marginal_transmission)

# Grand mean
grand_mean = mtcars['mpg'].mean()
print(f"\nGrand Mean (overall average MPG): {grand_mean:.2f}")
```

### Understanding Interaction Effects

Interaction effects can be calculated as the difference between the observed cell mean and what would be expected based on the main effects alone:

```math
(\alpha\beta)_{ij} = \bar{Y}_{ij.} - \bar{Y}_{i..} - \bar{Y}_{.j.} + \bar{Y}_{...}
```

```python
# Calculate interaction effects
def interaction_effects(cell_means, marginal_factor1, marginal_factor2, grand_mean):
    # Create a matrix of interaction effects
    interaction_matrix = np.zeros((len(marginal_factor1), len(marginal_factor2)))
    
    for i in range(len(marginal_factor1)):
        for j in range(len(marginal_factor2)):
            # Find the cell mean for this combination
            cell_data = cell_means[
                (cell_means['cyl_factor'] == marginal_factor1.iloc[i]['cyl_factor']) & 
                (cell_means['am_factor'] == marginal_factor2.iloc[j]['am_factor'])
            ]
            
            if len(cell_data) > 0:
                cell_mean = cell_data.iloc[0]['mean']
                factor1_mean = marginal_factor1.iloc[i]['mean']
                factor2_mean = marginal_factor2.iloc[j]['mean']
                
                interaction_effect = cell_mean - factor1_mean - factor2_mean + grand_mean
                interaction_matrix[i, j] = interaction_effect
    
    return interaction_matrix

# Calculate and display interaction effects
interaction_matrix = interaction_effects(cell_means, marginal_cylinders, marginal_transmission, grand_mean)
print("Interaction Effects Matrix:")
print(np.round(interaction_matrix, 3))

# Interpret interaction effects
print("\nInteraction Effect Interpretation:")
for i in range(interaction_matrix.shape[0]):
    for j in range(interaction_matrix.shape[1]):
        effect = interaction_matrix[i, j]
        if abs(effect) > 0.5:
            factor1_name = marginal_cylinders.iloc[i]['cyl_factor']
            factor2_name = marginal_transmission.iloc[j]['am_factor']
            print(f"Strong interaction between {factor1_name} and {factor2_name}: {effect:.2f}")
```

### Visualization

Visualization is crucial for understanding two-way ANOVA results, especially for detecting interaction effects and patterns in the data.

#### Interaction Plot

The interaction plot is the most important visualization for two-way ANOVA as it shows how the effect of one factor changes across the levels of the other factor.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Interaction plot with confidence intervals
plt.figure(figsize=(10, 6))
sns.pointplot(data=mtcars, x='cyl_factor', y='mpg', hue='am_factor', 
              capsize=0.1, markers=['o', 's', '^'], markersize=8)
plt.title('Interaction Plot: MPG by Cylinders and Transmission', fontsize=14, fontweight='bold')
plt.xlabel('Number of Cylinders')
plt.ylabel('MPG')
plt.legend(title='Transmission')
plt.grid(True, alpha=0.3)
plt.show()

# Interpret interaction plot
print("Interaction Plot Interpretation:")
print("- Parallel lines indicate no interaction")
print("- Non-parallel lines indicate interaction")
print("- Crossing lines indicate strong interaction\n")
```

#### Box Plot with Individual Points

Box plots show the distribution of data within each cell, while individual points show the actual data distribution.

```python
# Box plot with individual points
plt.figure(figsize=(10, 6))
sns.boxplot(data=mtcars, x='cyl_factor', y='mpg', hue='am_factor', alpha=0.7)
sns.stripplot(data=mtcars, x='cyl_factor', y='mpg', hue='am_factor', 
              dodge=True, alpha=0.6, size=3, legend=False)
plt.title('MPG Distribution by Cylinders and Transmission', fontsize=14, fontweight='bold')
plt.xlabel('Number of Cylinders')
plt.ylabel('MPG')
plt.legend(title='Transmission')
plt.grid(True, alpha=0.3)
plt.show()
```

#### Heatmap of Cell Means

A heatmap provides a visual representation of cell means, making it easy to identify patterns and interactions.

```python
# Heatmap of cell means with enhanced styling
plt.figure(figsize=(8, 6))
pivot_means = cell_means.pivot(index='am_factor', columns='cyl_factor', values='mean')
sns.heatmap(pivot_means, annot=True, fmt='.1f', cmap='RdYlBu_r', 
            cbar_kws={'label': 'Mean MPG'}, square=True)
plt.title('Cell Means Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Cylinders')
plt.ylabel('Transmission')
plt.show()
```

#### Marginal Effects Plot

This plot shows the main effects of each factor independently.

```python
# Marginal effects plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Marginal effects for cylinders
ax1.errorbar(marginal_cylinders['cyl_factor'], marginal_cylinders['mean'], 
             yerr=marginal_cylinders['se'], marker='o', capsize=5, capthick=2)
ax1.set_title('Marginal Effects - Cylinders', fontweight='bold')
ax1.set_xlabel('Cylinders')
ax1.set_ylabel('Mean MPG')
ax1.grid(True, alpha=0.3)

# Marginal effects for transmission
ax2.errorbar(marginal_transmission['am_factor'], marginal_transmission['mean'], 
             yerr=marginal_transmission['se'], marker='s', capsize=5, capthick=2)
ax2.set_title('Marginal Effects - Transmission', fontweight='bold')
ax2.set_xlabel('Transmission')
ax2.set_ylabel('Mean MPG')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Combine all plots in a single figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Interaction plot
sns.pointplot(data=mtcars, x='cyl_factor', y='mpg', hue='am_factor', ax=ax1)
ax1.set_title('Interaction Plot', fontweight='bold')

# Box plot
sns.boxplot(data=mtcars, x='cyl_factor', y='mpg', hue='am_factor', ax=ax2)
sns.stripplot(data=mtcars, x='cyl_factor', y='mpg', hue='am_factor', 
              dodge=True, alpha=0.6, size=2, legend=False, ax=ax2)
ax2.set_title('Box Plot with Individual Points', fontweight='bold')

# Heatmap
pivot_means = cell_means.pivot(index='am_factor', columns='cyl_factor', values='mean')
sns.heatmap(pivot_means, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax3)
ax3.set_title('Cell Means Heatmap', fontweight='bold')

# Marginal effects
ax4.errorbar(marginal_cylinders['cyl_factor'], marginal_cylinders['mean'], 
             yerr=marginal_cylinders['se'], marker='o', capsize=5, capthick=2)
ax4.set_title('Marginal Effects - Cylinders', fontweight='bold')
ax4.set_xlabel('Cylinders')
ax4.set_ylabel('Mean MPG')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Residuals Plot

Checking residuals is important for assumption validation.

```python
# Residuals plot
model_residuals = model.resid
fitted_values = model.fittedvalues

plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, model_residuals, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
```

## Effect Size Analysis

Effect sizes are crucial for understanding the practical significance of statistical results, as they provide information about the magnitude of effects independent of sample size.

### Understanding Effect Size Measures

#### Partial Eta-Squared ($`\eta_p^2`$)

Partial eta-squared is the most commonly used effect size measure in ANOVA. It represents the proportion of variance in the dependent variable that is explained by a specific effect, controlling for other effects in the model.

```math
\eta_p^2 = \frac{SS_{Effect}}{SS_{Effect} + SS_{Error}}
```

**Advantages:**
- Controls for other effects in the model
- Ranges from 0 to 1
- Easy to interpret

**Disadvantages:**
- Can be biased upward in small samples
- Values can sum to more than 1

#### Eta-Squared ($`\eta^2`$)

Eta-squared represents the proportion of total variance explained by an effect.

```math
\eta^2 = \frac{SS_{Effect}}{SS_{Total}}
```

**Advantages:**
- Values sum to 1 across all effects
- Intuitive interpretation

**Disadvantages:**
- Does not control for other effects
- Can be misleading in factorial designs

#### Omega-Squared ($`\omega^2`$)

Omega-squared is an unbiased estimate of the population effect size.

```math
\omega^2 = \frac{SS_{Effect} - df_{Effect} \times MS_{Error}}{SS_{Total} + MS_{Error}}
```

**Advantages:**
- Unbiased estimate
- More conservative than eta-squared

**Disadvantages:**
- Can be negative for small effects
- Less commonly reported

### Comprehensive Effect Size Calculation

```python
# Calculate effect sizes with confidence intervals
def calculate_two_way_effect_sizes(anova_result, data, factor1, factor2, response):
    # Extract Sum of Squares
    ss_factor1 = anova_result['ss_factor1']
    ss_factor2 = anova_result['ss_factor2']
    ss_interaction = anova_result['ss_interaction']
    ss_error = anova_result['ss_error']
    ss_total = anova_result['ss_total']
    
    # Degrees of freedom
    df_factor1 = anova_result['df_factor1']
    df_factor2 = anova_result['df_factor2']
    df_interaction = anova_result['df_interaction']
    df_error = anova_result['df_error']
    
    # Mean Squares
    ms_error = anova_result['ms_error']
    
    # Partial eta-squared
    partial_eta2_factor1 = ss_factor1 / (ss_factor1 + ss_error)
    partial_eta2_factor2 = ss_factor2 / (ss_factor2 + ss_error)
    partial_eta2_interaction = ss_interaction / (ss_interaction + ss_error)
    
    # Eta-squared (total)
    eta2_factor1 = ss_factor1 / ss_total
    eta2_factor2 = ss_factor2 / ss_total
    eta2_interaction = ss_interaction / ss_total
    
    # Omega-squared (unbiased)
    omega2_factor1 = (ss_factor1 - df_factor1 * ms_error) / (ss_total + ms_error)
    omega2_factor2 = (ss_factor2 - df_factor2 * ms_error) / (ss_total + ms_error)
    omega2_interaction = (ss_interaction - df_interaction * ms_error) / (ss_total + ms_error)
    
    # Cohen's f (for power analysis)
    f_factor1 = np.sqrt(partial_eta2_factor1 / (1 - partial_eta2_factor1))
    f_factor2 = np.sqrt(partial_eta2_factor2 / (1 - partial_eta2_factor2))
    f_interaction = np.sqrt(partial_eta2_interaction / (1 - partial_eta2_interaction))
    
    # Bootstrap confidence intervals for partial eta-squared
    def bootstrap_effect_size(data, indices, factor1, factor2, response):
        d = data.iloc[indices]
        model = ols(f'{response} ~ C({factor1}) * C({factor2})', data=d).fit()
        anova_table = anova_lm(model, typ=2)
        
        ss_factor1 = anova_table.loc[f'C({factor1})', 'sum_sq']
        ss_factor2 = anova_table.loc[f'C({factor2})', 'sum_sq']
        ss_interaction = anova_table.loc[f'C({factor1}):C({factor2})', 'sum_sq']
        ss_error = anova_table.loc['Residual', 'sum_sq']
        
        partial_eta2_factor1 = ss_factor1 / (ss_factor1 + ss_error)
        partial_eta2_factor2 = ss_factor2 / (ss_factor2 + ss_error)
        partial_eta2_interaction = ss_interaction / (ss_interaction + ss_error)
        
        return [partial_eta2_factor1, partial_eta2_factor2, partial_eta2_interaction]
    
    # Bootstrap for confidence intervals
    from scipy.stats import bootstrap
    bootstrap_result = bootstrap((data,), bootstrap_effect_size, n_resamples=1000,
                               args=(factor1, factor2, response))
    
    ci_factor1 = (np.percentile(bootstrap_result.bootstrap_distribution[:, 0], 2.5),
                  np.percentile(bootstrap_result.bootstrap_distribution[:, 0], 97.5))
    ci_factor2 = (np.percentile(bootstrap_result.bootstrap_distribution[:, 1], 2.5),
                  np.percentile(bootstrap_result.bootstrap_distribution[:, 1], 97.5))
    ci_interaction = (np.percentile(bootstrap_result.bootstrap_distribution[:, 2], 2.5),
                      np.percentile(bootstrap_result.bootstrap_distribution[:, 2], 97.5))
    
    return {
        'partial_eta2_factor1': partial_eta2_factor1,
        'partial_eta2_factor2': partial_eta2_factor2,
        'partial_eta2_interaction': partial_eta2_interaction,
        'eta2_factor1': eta2_factor1,
        'eta2_factor2': eta2_factor2,
        'eta2_interaction': eta2_interaction,
        'omega2_factor1': omega2_factor1,
        'omega2_factor2': omega2_factor2,
        'omega2_interaction': omega2_interaction,
        'f_factor1': f_factor1,
        'f_factor2': f_factor2,
        'f_interaction': f_interaction,
        'ci_factor1': ci_factor1,
        'ci_factor2': ci_factor2,
        'ci_interaction': ci_interaction
    }

# Apply to our results
effect_sizes = calculate_two_way_effect_sizes(anova_result, mtcars, "cyl_factor", "am_factor", "mpg")

# Display comprehensive effect size results
print("=== COMPREHENSIVE EFFECT SIZE ANALYSIS ===")
print()

print("PARTIAL ETA-SQUARED (η²p):")
print(f"Factor 1 (Cylinders): {effect_sizes['partial_eta2_factor1']:.4f}")
print(f"Factor 2 (Transmission): {effect_sizes['partial_eta2_factor2']:.4f}")
print(f"Interaction: {effect_sizes['partial_eta2_interaction']:.4f}")
print()

print("ETA-SQUARED (η²):")
print(f"Factor 1 (Cylinders): {effect_sizes['eta2_factor1']:.4f}")
print(f"Factor 2 (Transmission): {effect_sizes['eta2_factor2']:.4f}")
print(f"Interaction: {effect_sizes['eta2_interaction']:.4f}")
print()

print("OMEGA-SQUARED (ω²):")
print(f"Factor 1 (Cylinders): {effect_sizes['omega2_factor1']:.4f}")
print(f"Factor 2 (Transmission): {effect_sizes['omega2_factor2']:.4f}")
print(f"Interaction: {effect_sizes['omega2_interaction']:.4f}")
print()

print("COHEN'S f (for power analysis):")
print(f"Factor 1 (Cylinders): {effect_sizes['f_factor1']:.4f}")
print(f"Factor 2 (Transmission): {effect_sizes['f_factor2']:.4f}")
print(f"Interaction: {effect_sizes['f_interaction']:.4f}")
print()
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

print("EFFECT SIZE INTERPRETATION:")
print(f"Factor 1 (Cylinders): {interpret_effect_size(effect_sizes['partial_eta2_factor1'])}")
print(f"Factor 2 (Transmission): {interpret_effect_size(effect_sizes['partial_eta2_factor2'])}")
print(f"Interaction: {interpret_effect_size(effect_sizes['partial_eta2_interaction'])}")
print()

# Practical significance assessment
print("PRACTICAL SIGNIFICANCE ASSESSMENT:")
if effect_sizes['partial_eta2_factor1'] > 0.06:
    print("- Cylinder count has a practically significant effect on MPG")
if effect_sizes['partial_eta2_factor2'] > 0.06:
    print("- Transmission type has a practically significant effect on MPG")
if effect_sizes['partial_eta2_interaction'] > 0.06:
    print("- There is a practically significant interaction effect")
```

### Effect Size Visualization

```python
# Create effect size comparison plot
effect_size_data = pd.DataFrame({
    'Effect': ['Factor 1 (Cylinders)', 'Factor 2 (Transmission)', 'Interaction'] * 3,
    'Measure': ['Partial η²'] * 3 + ['η²'] * 3 + ['ω²'] * 3,
    'Value': [effect_sizes['partial_eta2_factor1'], effect_sizes['partial_eta2_factor2'], effect_sizes['partial_eta2_interaction'],
              effect_sizes['eta2_factor1'], effect_sizes['eta2_factor2'], effect_sizes['eta2_interaction'],
              effect_sizes['omega2_factor1'], effect_sizes['omega2_factor2'], effect_sizes['omega2_interaction']]
})

plt.figure(figsize=(10, 6))
sns.barplot(data=effect_size_data, x='Effect', y='Value', hue='Measure', alpha=0.8)
plt.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Negligible')
plt.axhline(y=0.06, color='red', linestyle='--', alpha=0.7, label='Small')
plt.axhline(y=0.14, color='red', linestyle='--', alpha=0.7, label='Medium')
plt.title('Effect Size Comparison\nDifferent measures of effect size for each factor and interaction', 
          fontsize=14, fontweight='bold')
plt.xlabel('Effect')
plt.ylabel('Effect Size')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Measure')
plt.tight_layout()
plt.show()
```

## Simple Effects Analysis

When a significant interaction is found in two-way ANOVA, the main effects cannot be interpreted independently. Instead, we must examine **simple effects** - the effect of one factor at each level of the other factor.

### Understanding Simple Effects

Simple effects analysis answers questions like:
- "Does the effect of teaching method differ across class sizes?"
- "Is there a difference between automatic and manual transmission for 4-cylinder engines?"
- "Do the effects of drug treatment vary by dosage level?"

### Mathematical Foundation

The simple effect of Factor A at level j of Factor B is:

```math
SS_{A|B_j} = \sum_{i=1}^{a} n_{ij}(\bar{Y}_{ij.} - \bar{Y}_{.j.})^2
```

With degrees of freedom: $`df_{A|B_j} = a - 1`$

The F-statistic for the simple effect is:

```math
F_{A|B_j} = \frac{MS_{A|B_j}}{MS_{Error}} = \frac{SS_{A|B_j}/df_{A|B_j}}{SS_{Error}/df_{Error}}
```

### Comprehensive Simple Effects Analysis

```python
# Enhanced function to perform simple effects analysis with effect sizes
def simple_effects_analysis(data, factor1, factor2, response, alpha=0.05):
    print("=== COMPREHENSIVE SIMPLE EFFECTS ANALYSIS ===")
    print()
    
    # Store results for summary
    results = {}
    
    # Simple effects of Factor 1 at each level of Factor 2
    print(f"SIMPLE EFFECTS OF {factor1} AT EACH LEVEL OF {factor2}:")
    print("=" * 50)
    
    factor2_levels = data[factor2].unique()
    
    for level in factor2_levels:
        subset_data = data[data[factor2] == level]
        
        if len(subset_data[factor1].unique()) > 1:
            # Perform one-way ANOVA
            simple_model = ols(f'{response} ~ C({factor1})', data=subset_data).fit()
            simple_anova = anova_lm(simple_model, typ=2)
            
            f_stat = simple_anova.loc[f'C({factor1})', 'F']
            p_value = simple_anova.loc[f'C({factor1})', 'PR(>F)']
            df_factor = simple_anova.loc[f'C({factor1})', 'df']
            df_error = simple_anova.loc['Residual', 'df']
            
            # Calculate effect size
            ss_factor = simple_anova.loc[f'C({factor1})', 'sum_sq']
            ss_error = simple_anova.loc['Residual', 'sum_sq']
            partial_eta2 = ss_factor / (ss_factor + ss_error)
            
            # Calculate descriptive statistics
            desc_stats = subset_data.groupby(factor1).agg({
                response: ['count', 'mean', 'std']
            }).round(3)
            desc_stats.columns = ['n', 'mean', 'sd']
            desc_stats = desc_stats.reset_index()
            desc_stats['se'] = desc_stats['sd'] / np.sqrt(desc_stats['n'])
            
            print(f"At {factor2} = {level}:")
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Partial η²: {partial_eta2:.4f}")
            print(f"  Degrees of freedom: {df_factor}, {df_error}")
            
            if p_value < alpha:
                print(f"  ✓ Significant effect (p < {alpha})")
            else:
                print(f"  ✗ Non-significant effect (p ≥ {alpha})")
            
            print("  Descriptive statistics:")
            for _, row in desc_stats.iterrows():
                print(f"    {row[factor1]}: M = {row['mean']:.2f}, SD = {row['sd']:.2f}, n = {row['n']}")
            
            # Store results
            results[f"{factor1}_at_{factor2}_{level}"] = {
                'f_stat': f_stat,
                'p_value': p_value,
                'partial_eta2': partial_eta2,
                'df_factor': df_factor,
                'df_error': df_error,
                'desc_stats': desc_stats,
                'significant': p_value < alpha
            }
            
            print()
    
    # Simple effects of Factor 2 at each level of Factor 1
    print(f"SIMPLE EFFECTS OF {factor2} AT EACH LEVEL OF {factor1}:")
    print("=" * 50)
    
    factor1_levels = data[factor1].unique()
    
    for level in factor1_levels:
        subset_data = data[data[factor1] == level]
        
        if len(subset_data[factor2].unique()) > 1:
            # Perform one-way ANOVA
            simple_model = ols(f'{response} ~ C({factor2})', data=subset_data).fit()
            simple_anova = anova_lm(simple_model, typ=2)
            
            f_stat = simple_anova.loc[f'C({factor2})', 'F']
            p_value = simple_anova.loc[f'C({factor2})', 'PR(>F)']
            df_factor = simple_anova.loc[f'C({factor2})', 'df']
            df_error = simple_anova.loc['Residual', 'df']
            
            # Calculate effect size
            ss_factor = simple_anova.loc[f'C({factor2})', 'sum_sq']
            ss_error = simple_anova.loc['Residual', 'sum_sq']
            partial_eta2 = ss_factor / (ss_factor + ss_error)
            
            # Calculate descriptive statistics
            desc_stats = subset_data.groupby(factor2).agg({
                response: ['count', 'mean', 'std']
            }).round(3)
            desc_stats.columns = ['n', 'mean', 'sd']
            desc_stats = desc_stats.reset_index()
            desc_stats['se'] = desc_stats['sd'] / np.sqrt(desc_stats['n'])
            
            print(f"At {factor1} = {level}:")
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Partial η²: {partial_eta2:.4f}")
            print(f"  Degrees of freedom: {df_factor}, {df_error}")
            
            if p_value < alpha:
                print(f"  ✓ Significant effect (p < {alpha})")
            else:
                print(f"  ✗ Non-significant effect (p ≥ {alpha})")
            
            print("  Descriptive statistics:")
            for _, row in desc_stats.iterrows():
                print(f"    {row[factor2]}: M = {row['mean']:.2f}, SD = {row['sd']:.2f}, n = {row['n']}")
            
            # Store results
            results[f"{factor2}_at_{factor1}_{level}"] = {
                'f_stat': f_stat,
                'p_value': p_value,
                'partial_eta2': partial_eta2,
                'df_factor': df_factor,
                'df_error': df_error,
                'desc_stats': desc_stats,
                'significant': p_value < alpha
            }
            
            print()
    
    # Summary of significant effects
    print("SUMMARY OF SIGNIFICANT SIMPLE EFFECTS:")
    print("=" * 40)
    
    significant_effects = 0
    for effect_name, result in results.items():
        if result['significant']:
            print(f"✓ {effect_name} (p = {result['p_value']:.4f})")
            significant_effects += 1
    
    if significant_effects == 0:
        print("No significant simple effects found.")
    else:
        print(f"Total significant simple effects: {significant_effects}")
    
    return results

# Apply enhanced simple effects analysis
simple_effects_results = simple_effects_analysis(mtcars, "cyl_factor", "am_factor", "mpg")
```

### Simple Effects Visualization

```python
# Create visualization for simple effects
def plot_simple_effects(data, factor1, factor2, response):
    # Create plots for simple effects of factor1 at each level of factor2
    factor2_levels = data[factor2].unique()
    
    fig, axes = plt.subplots(1, len(factor2_levels), figsize=(5*len(factor2_levels), 5))
    if len(factor2_levels) == 1:
        axes = [axes]
    
    for i, level in enumerate(factor2_levels):
        subset_data = data[data[factor2] == level]
        
        # Box plot with individual points
        sns.boxplot(data=subset_data, x=factor1, y=response, ax=axes[i], alpha=0.7, color='lightblue')
        sns.stripplot(data=subset_data, x=factor1, y=response, ax=axes[i], alpha=0.6, size=3, color='black')
        
        # Add mean points
        means = subset_data.groupby(factor1)[response].mean()
        for j, (group, mean_val) in enumerate(means.items()):
            axes[i].plot(j, mean_val, 'ro', markersize=8, markeredgecolor='red', markeredgewidth=2)
        
        axes[i].set_title(f'Simple Effect of {factor1} at {factor2} = {level}\nOne-way ANOVA within {factor2} = {level}', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel(factor1)
        axes[i].set_ylabel(response)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Create simple effects plots
plot_simple_effects(mtcars, "cyl_factor", "am_factor", "mpg")
```

### Post Hoc Tests for Simple Effects

```python
# Post hoc tests for significant simple effects
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Function to perform post hoc tests for simple effects
def simple_effects_posthoc(simple_effects_results, data, factor1, factor2, response):
    print("=== POST HOC TESTS FOR SIGNIFICANT SIMPLE EFFECTS ===")
    print()
    
    for effect_name, result in simple_effects_results.items():
        if result['significant']:
            print(f"Post hoc tests for: {effect_name}")
            print("-" * 40)
            
            # Extract factor information from effect name
            if factor1 + "_at_" in effect_name:
                # Simple effect of factor1 at a level of factor2
                level = effect_name.replace(f"{factor1}_at_{factor2}_", "")
                subset_data = data[data[factor2] == level]
                
                if len(subset_data[factor1].unique()) > 2:
                    # Perform Tukey's HSD
                    tukey_result = pairwise_tukeyhsd(subset_data[response], subset_data[factor1])
                    
                    print("Tukey's HSD results:")
                    print(tukey_result)
                    
                    # Extract significant pairwise comparisons
                    significant_comparisons = 0
                    for i, row in tukey_result.pvalues.items():
                        if row < 0.05:
                            print(f"  {i}: p = {row:.4f}")
                            significant_comparisons += 1
                    
                    if significant_comparisons == 0:
                        print("  No significant pairwise differences found.")
                else:
                    print("Only 2 levels - no post hoc tests needed.")
            
            print()

# Apply post hoc tests
simple_effects_posthoc(simple_effects_results, mtcars, "cyl_factor", "am_factor", "mpg")
```

## Post Hoc Tests

### Post Hoc for Main Effects

```python
# Post hoc tests for significant main effects
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Post hoc for Factor 1 (Cylinders) if significant
if p_cylinders < 0.05:
    print("Post hoc tests for Cylinders (Factor 1):")
    
    # Tukey's HSD for cylinders
    cyl_posthoc = pairwise_tukeyhsd(mtcars['mpg'], mtcars['cyl_factor'])
    print(cyl_posthoc)
    
    # Extract significant pairs
    print("Significant pairwise differences (p < 0.05):")
    for i, row in cyl_posthoc.pvalues.items():
        if row < 0.05:
            print(f"{i}: p = {row:.4f}")
    print()

# Post hoc for Factor 2 (Transmission) if significant
if p_transmission < 0.05:
    print("Post hoc tests for Transmission (Factor 2):")
    
    # Since transmission has only 2 levels, no post hoc needed
    print("Transmission has only 2 levels - no post hoc tests needed")
    print("Automatic vs Manual transmission difference is already tested")
    print()
```

### Post Hoc for Interaction Effects

```python
# Post hoc tests for interaction effects
if p_interaction < 0.05:
    print("Post hoc tests for Interaction Effects:")
    
    # Create interaction factor for cell comparisons
    mtcars['interaction_factor'] = mtcars['cyl_factor'] + ':' + mtcars['am_factor']
    
    # Pairwise comparisons for all cells
    cell_posthoc = pairwise_tukeyhsd(mtcars['mpg'], mtcars['interaction_factor'])
    print(cell_posthoc)
    
    # Extract significant cell comparisons
    print("Significant cell differences (p < 0.05):")
    for i, row in cell_posthoc.pvalues.items():
        if row < 0.05:
            print(f"{i}: p = {row:.4f}")
    print()
else:
    print("No post hoc tests needed for interaction (not significant)")
    print()
```

## Assumption Checking

Two-way ANOVA relies on several key assumptions that must be verified before interpreting results. Violations of these assumptions can lead to incorrect conclusions.

### Key Assumptions

1. **Normality**: Residuals should be normally distributed
2. **Homogeneity of Variance**: Variances should be equal across all groups
3. **Independence**: Observations should be independent
4. **Linearity**: The relationship between factors and the dependent variable should be linear

### Comprehensive Normality Testing

```python
# Enhanced function to check normality for two-way ANOVA
from scipy.stats import shapiro, ks_1samp, anderson
from scipy.stats import norm

def check_normality_two_way(data, factor1, factor2, response, alpha=0.05):
    print("=== COMPREHENSIVE NORMALITY TESTS FOR TWO-WAY ANOVA ===")
    print()
    
    # Fit the model
    model = ols(f'{response} ~ C({factor1}) * C({factor2})', data=data).fit()
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    print("NORMALITY TESTS ON RESIDUALS:")
    print("=" * 40)
    
    # Multiple normality tests
    tests = {}
    
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = shapiro(residuals)
    tests['shapiro'] = {'statistic': shapiro_stat, 'pvalue': shapiro_p}
    print("Shapiro-Wilk test:")
    print(f"  W = {shapiro_stat:.4f}")
    print(f"  p-value = {shapiro_p:.4f}")
    print(f"  Decision: {'✓ Normal' if shapiro_p >= alpha else '✗ Non-normal'}")
    print()
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = ks_1samp(residuals, norm.cdf, args=(np.mean(residuals), np.std(residuals)))
    tests['ks'] = {'statistic': ks_stat, 'pvalue': ks_p}
    print("Kolmogorov-Smirnov test:")
    print(f"  D = {ks_stat:.4f}")
    print(f"  p-value = {ks_p:.4f}")
    print(f"  Decision: {'✓ Normal' if ks_p >= alpha else '✗ Non-normal'}")
    print()
    
    # Anderson-Darling test
    ad_result = anderson(residuals)
    tests['ad'] = {'statistic': ad_result.statistic, 'pvalue': ad_result.significance_level[2]/100}
    print("Anderson-Darling test:")
    print(f"  A = {ad_result.statistic:.4f}")
    print(f"  p-value = {ad_result.significance_level[2]/100:.4f}")
    print(f"  Decision: {'✓ Normal' if ad_result.significance_level[2]/100 >= alpha else '✗ Non-normal'}")
    print()
    
    # Summary of normality tests
    print("NORMALITY TEST SUMMARY:")
    print("=" * 25)
    
    p_values = [shapiro_p, ks_p, ad_result.significance_level[2]/100]
    test_names = ["Shapiro-Wilk", "Kolmogorov-Smirnov", "Anderson-Darling"]
    
    normal_tests = sum(p >= alpha for p in p_values)
    total_tests = len(p_values)
    
    print(f"Tests supporting normality: {normal_tests} out of {total_tests}")
    
    if normal_tests >= 3:
        print("✓ Normality assumption appears to be met")
    elif normal_tests >= 2:
        print("⚠ Normality assumption may be questionable")
    else:
        print("✗ Normality assumption appears to be violated")
    
    # Create diagnostic plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Q-Q plot
    from scipy.stats import probplot
    probplot(residuals, dist="norm", plot=ax1)
    ax1.set_title("Q-Q Plot of Residuals\nPoints should fall along the red line", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram with normal curve
    ax2.hist(residuals, bins=15, density=True, alpha=0.7, color='steelblue')
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x, norm.pdf(x, np.mean(residuals), np.std(residuals)), 'r-', linewidth=2)
    ax2.set_title("Histogram of Residuals with Normal Curve\nHistogram should approximate the red normal curve", fontweight='bold')
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Density")
    ax2.grid(True, alpha=0.3)
    
    # Residuals vs fitted values
    ax3.scatter(fitted_values, residuals, alpha=0.7)
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_title("Residuals vs Fitted Values\nCheck for homoscedasticity and linearity", fontweight='bold')
    ax3.set_xlabel("Fitted Values")
    ax3.set_ylabel("Residuals")
    ax3.grid(True, alpha=0.3)
    
    # Remove the fourth subplot
    ax4.remove()
    
    plt.tight_layout()
    plt.show()
    
    # Recommendations based on results
    print("\nRECOMMENDATIONS:")
    print("=" * 15)
    
    if normal_tests >= 3:
        print("✓ Two-way ANOVA is appropriate for these data")
        print("✓ All normality assumptions appear to be met")
    elif normal_tests >= 2:
        print("⚠ Consider robust alternatives or data transformation")
        print("⚠ Two-way ANOVA may still be appropriate with caution")
    else:
        print("✗ Consider nonparametric alternatives:")
        print("  - Kruskal-Wallis test for main effects")
        print("  - Permutation tests")
        print("  - Data transformation (log, square root, etc.)")
    
    return {
        'tests': tests,
        'p_values': p_values,
        'test_names': test_names,
        'normal_tests': normal_tests,
        'total_tests': total_tests
    }

# Check normality with enhanced function
normality_results = check_normality_two_way(mtcars, "cyl_factor", "am_factor", "mpg")
```

### Comprehensive Homogeneity of Variance Testing

```python
# Enhanced function to test homogeneity of variance for two-way ANOVA
from scipy.stats import levene, bartlett, fligner
from scipy import stats

def check_homogeneity_two_way(data, factor1, factor2, response, alpha=0.05):
    print("=== COMPREHENSIVE HOMOGENEITY OF VARIANCE TESTS ===")
    print()
    
    # Create interaction factor for testing
    data['interaction_factor'] = data[factor1].astype(str) + ':' + data[factor2].astype(str)
    
    print("HOMOGENEITY OF VARIANCE TESTS:")
    print("=" * 35)
    
    # Multiple homogeneity tests
    tests = {}
    
    # Prepare groups for testing
    groups = [group[response].values for name, group in data.groupby('interaction_factor')]
    
    # Levene's test (most robust)
    levene_stat, levene_p = levene(*groups)
    tests['levene'] = {'statistic': levene_stat, 'pvalue': levene_p}
    print("Levene's test (robust to non-normality):")
    print(f"  F = {levene_stat:.4f}")
    print(f"  p-value = {levene_p:.4f}")
    print(f"  Decision: {'✓ Equal variances' if levene_p >= alpha else '✗ Unequal variances'}")
    print()
    
    # Bartlett's test (sensitive to non-normality)
    bartlett_stat, bartlett_p = bartlett(*groups)
    tests['bartlett'] = {'statistic': bartlett_stat, 'pvalue': bartlett_p}
    print("Bartlett's test (sensitive to non-normality):")
    print(f"  K-squared = {bartlett_stat:.4f}")
    print(f"  p-value = {bartlett_p:.4f}")
    print(f"  Decision: {'✓ Equal variances' if bartlett_p >= alpha else '✗ Unequal variances'}")
    print()
    
    # Fligner-Killeen test (robust)
    fligner_stat, fligner_p = fligner(*groups)
    tests['fligner'] = {'statistic': fligner_stat, 'pvalue': fligner_p}
    print("Fligner-Killeen test (robust):")
    print(f"  chi-squared = {fligner_stat:.4f}")
    print(f"  p-value = {fligner_p:.4f}")
    print(f"  Decision: {'✓ Equal variances' if fligner_p >= alpha else '✗ Unequal variances'}")
    print()
    
    # Summary of homogeneity tests
    print("HOMOGENEITY TEST SUMMARY:")
    print("=" * 28)
    
    p_values = [levene_p, bartlett_p, fligner_p]
    test_names = ["Levene's", "Bartlett's", "Fligner-Killeen"]
    
    equal_var_tests = sum(p >= alpha for p in p_values)
    total_tests = len(p_values)
    
    print(f"Tests supporting equal variances: {equal_var_tests} out of {total_tests}")
    
    if equal_var_tests >= 3:
        print("✓ Homogeneity of variance assumption appears to be met")
    elif equal_var_tests >= 2:
        print("⚠ Homogeneity of variance assumption may be questionable")
    else:
        print("✗ Homogeneity of variance assumption appears to be violated")
    
    # Calculate and display group variances
    print("\nGROUP VARIANCES:")
    print("=" * 15)
    
    group_vars = data.groupby([factor1, factor2]).agg({
        response: ['count', 'var', 'std']
    }).round(3)
    group_vars.columns = ['n', 'variance', 'sd']
    group_vars = group_vars.reset_index()
    
    print(group_vars)
    
    # Calculate variance ratio (largest/smallest)
    max_var = group_vars['variance'].max()
    min_var = group_vars['variance'].min()
    var_ratio = max_var / min_var
    
    print(f"\nVariance ratio (largest/smallest): {var_ratio:.2f}")
    
    if var_ratio <= 4:
        print("✓ Variance ratio is acceptable (≤ 4)")
    elif var_ratio <= 10:
        print("⚠ Variance ratio is questionable (4-10)")
    else:
        print("✗ Variance ratio is too large (> 10)")
    
    # Create diagnostic plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot to visualize variance differences
    sns.boxplot(data=data, x='interaction_factor', y=response, hue=factor1, ax=ax1, alpha=0.7)
    ax1.set_title("Box Plot by Group\nCheck for differences in spread (variance)", fontweight='bold')
    ax1.set_xlabel("Group")
    ax1.set_ylabel(response)
    ax1.tick_params(axis='x', rotation=45)
    
    # Variance vs mean plot
    group_means = data.groupby('interaction_factor')[response].mean()
    group_vars_dict = dict(zip(group_vars[factor1] + ':' + group_vars[factor2], group_vars['variance']))
    
    means = [group_means[group] for group in group_means.index]
    variances = [group_vars_dict[group] for group in group_means.index]
    
    ax2.scatter(means, variances, s=50, alpha=0.7)
    ax2.set_title("Variance vs Mean Plot\nCheck for heteroscedasticity patterns", fontweight='bold')
    ax2.set_xlabel("Group Mean")
    ax2.set_ylabel("Group Variance")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Recommendations based on results
    print("\nRECOMMENDATIONS:")
    print("=" * 15)
    
    if equal_var_tests >= 3 and var_ratio <= 4:
        print("✓ Standard two-way ANOVA is appropriate")
        print("✓ All homogeneity assumptions appear to be met")
    elif equal_var_tests >= 2 and var_ratio <= 10:
        print("⚠ Consider robust alternatives:")
        print("  - Welch's ANOVA (if available for two-way)")
        print("  - Data transformation (log, square root, etc.)")
        print("  - Two-way ANOVA may still be appropriate with caution")
    else:
        print("✗ Consider alternatives:")
        print("  - Robust statistical methods")
        print("  - Nonparametric alternatives")
        print("  - Data transformation")
        print("  - Bootstrapping methods")
    
    return {
        'tests': tests,
        'p_values': p_values,
        'test_names': test_names,
        'equal_var_tests': equal_var_tests,
        'total_tests': total_tests,
        'group_vars': group_vars,
        'var_ratio': var_ratio
    }

# Check homogeneity with enhanced function
homogeneity_results = check_homogeneity_two_way(mtcars, "cyl_factor", "am_factor", "mpg")
```

## Practical Examples

### Example 1: Educational Research

```python
# Simulate educational intervention data
np.random.seed(123)
n_per_cell = 15

# Generate data for 2x3 factorial design
# Factor 1: Teaching Method (A, B)
# Factor 2: Class Size (Small, Medium, Large)

teaching_method = ["Method A"] * (n_per_cell * 3) + ["Method B"] * (n_per_cell * 3)
class_size = ["Small"] * n_per_cell + ["Medium"] * n_per_cell + ["Large"] * n_per_cell
class_size = class_size * 2

# Generate scores with interaction effects
scores = []
for i in range(len(teaching_method)):
    if teaching_method[i] == "Method A":
        if class_size[i] == "Small":
            scores.append(np.random.normal(85, 8))
        elif class_size[i] == "Medium":
            scores.append(np.random.normal(80, 10))
        else:
            scores.append(np.random.normal(75, 12))
    else:
        if class_size[i] == "Small":
            scores.append(np.random.normal(82, 9))
        elif class_size[i] == "Medium":
            scores.append(np.random.normal(85, 8))
        else:
            scores.append(np.random.normal(88, 7))

# Create data frame
education_data = pd.DataFrame({
    'score': scores,
    'method': teaching_method,
    'class_size': class_size
})

# Perform two-way ANOVA
education_model = ols('score ~ C(method) * C(class_size)', data=education_data).fit()
education_anova = anova_lm(education_model, typ=2)
print(education_anova)

# Visualize interaction
plt.figure(figsize=(10, 6))
sns.pointplot(data=education_data, x='class_size', y='score', hue='method', 
              markers=['o', 's'], markersize=8)
plt.title('Interaction Plot: Score by Teaching Method and Class Size', fontweight='bold')
plt.xlabel('Class Size')
plt.ylabel('Score')
plt.legend(title='Teaching Method')
plt.grid(True, alpha=0.3)
plt.show()
```

### Example 2: Clinical Trial

```python
# Simulate clinical trial data
np.random.seed(123)
n_per_cell = 20

# Generate data for 2x2 factorial design
# Factor 1: Treatment (Drug A, Drug B)
# Factor 2: Dosage (Low, High)

treatment = ["Drug A"] * (n_per_cell * 2) + ["Drug B"] * (n_per_cell * 2)
dosage = ["Low"] * n_per_cell + ["High"] * n_per_cell
dosage = dosage * 2

# Generate outcomes with interaction
outcomes = []
for i in range(len(treatment)):
    if treatment[i] == "Drug A":
        if dosage[i] == "Low":
            outcomes.append(np.random.normal(60, 12))
        else:
            outcomes.append(np.random.normal(75, 10))
    else:
        if dosage[i] == "Low":
            outcomes.append(np.random.normal(65, 11))
        else:
            outcomes.append(np.random.normal(85, 9))

# Create data frame
clinical_data = pd.DataFrame({
    'outcome': outcomes,
    'treatment': treatment,
    'dosage': dosage
})

# Perform two-way ANOVA
clinical_model = ols('outcome ~ C(treatment) * C(dosage)', data=clinical_data).fit()
clinical_anova = anova_lm(clinical_model, typ=2)
print(clinical_anova)

# Simple effects analysis
simple_effects_analysis(clinical_data, "treatment", "dosage", "outcome")
```

### Example 3: Manufacturing Quality

```python
# Simulate manufacturing quality data
np.random.seed(123)
n_per_cell = 12

# Generate data for 3x2 factorial design
# Factor 1: Machine Type (A, B, C)
# Factor 2: Shift (Day, Night)

machine = ["Machine A"] * (n_per_cell * 2) + ["Machine B"] * (n_per_cell * 2) + ["Machine C"] * (n_per_cell * 2)
shift = ["Day"] * n_per_cell + ["Night"] * n_per_cell
shift = shift * 3

# Generate quality scores
quality_scores = []
for i in range(len(machine)):
    if machine[i] == "Machine A":
        if shift[i] == "Day":
            quality_scores.append(np.random.normal(95, 3))
        else:
            quality_scores.append(np.random.normal(92, 4))
    elif machine[i] == "Machine B":
        if shift[i] == "Day":
            quality_scores.append(np.random.normal(88, 5))
        else:
            quality_scores.append(np.random.normal(85, 6))
    else:
        if shift[i] == "Day":
            quality_scores.append(np.random.normal(90, 4))
        else:
            quality_scores.append(np.random.normal(87, 5))

# Create data frame
quality_data = pd.DataFrame({
    'quality': quality_scores,
    'machine': machine,
    'shift': shift
})

# Perform two-way ANOVA
quality_model = ols('quality ~ C(machine) * C(shift)', data=quality_data).fit()
quality_anova = anova_lm(quality_model, typ=2)
print(quality_anova)

# Visualize results
plt.figure(figsize=(10, 6))
sns.boxplot(data=quality_data, x='machine', y='quality', hue='shift', alpha=0.7)
plt.title('Quality Scores by Machine and Shift', fontweight='bold')
plt.xlabel('Machine')
plt.ylabel('Quality Score')
plt.legend(title='Shift')
plt.grid(True, alpha=0.3)
plt.show()
```

## Best Practices and Guidelines

Following best practices ensures reliable, reproducible, and interpretable two-way ANOVA results. This section provides comprehensive guidelines for conducting and reporting two-way ANOVA analyses.

### Test Selection Guidelines

```python
# Comprehensive function to help choose appropriate two-way ANOVA test
def choose_two_way_test(data, factor1, factor2, response, alpha=0.05):
    print("=== COMPREHENSIVE TWO-WAY ANOVA TEST SELECTION ===")
    print()
    
    # Check normality
    print("STEP 1: NORMALITY ASSESSMENT")
    print("=" * 30)
    normality_results = check_normality_two_way(data, factor1, factor2, response, alpha)
    
    # Check homogeneity
    print("\nSTEP 2: HOMOGENEITY ASSESSMENT")
    print("=" * 32)
    homogeneity_results = check_homogeneity_two_way(data, factor1, factor2, response, alpha)
    
    # Check sample sizes
    print("\nSTEP 3: SAMPLE SIZE ASSESSMENT")
    print("=" * 28)
    cell_counts = pd.crosstab(data[factor1], data[factor2])
    print("Cell sample sizes:")
    print(cell_counts)
    
    # Check for balanced design
    balanced = len(cell_counts.values.flatten()) == len(set(cell_counts.values.flatten()))
    print(f"Balanced design: {balanced}")
    
    # Calculate total sample size
    total_n = cell_counts.sum().sum()
    print(f"Total sample size: {total_n}")
    
    # Check minimum cell size
    min_cell_size = cell_counts.min().min()
    print(f"Minimum cell size: {min_cell_size}")
    
    # Sample size recommendations
    if min_cell_size < 5:
        print("⚠ Warning: Very small cell sizes (< 5)")
    elif min_cell_size < 10:
        print("⚠ Warning: Small cell sizes (< 10)")
    elif min_cell_size >= 20:
        print("✓ Adequate cell sizes (≥ 20)")
    
    # Check for missing data
    missing_data = data[response].isna().sum()
    if missing_data > 0:
        print(f"Missing data points: {missing_data}")
    else:
        print("✓ No missing data")
    
    print("\nSTEP 4: FINAL RECOMMENDATION")
    print("=" * 25)
    
    # Decision logic based on comprehensive assessment
    normal_tests_passed = normality_results['normal_tests'] >= 3
    homogeneity_tests_passed = homogeneity_results['equal_var_tests'] >= 3
    adequate_sample_size = min_cell_size >= 10
    
    if normal_tests_passed and homogeneity_tests_passed and adequate_sample_size:
        print("✓ RECOMMENDATION: Standard Two-Way ANOVA")
        print("  - All assumptions appear to be met")
        print("  - Sample sizes are adequate")
        print("  - Results should be reliable")
    elif normal_tests_passed and not homogeneity_tests_passed and adequate_sample_size:
        print("⚠ RECOMMENDATION: Robust Alternatives")
        print("  - Normality assumption met")
        print("  - Homogeneity assumption violated")
        print("  - Consider: Welch's ANOVA, data transformation, or robust methods")
    elif not normal_tests_passed and adequate_sample_size:
        print("⚠ RECOMMENDATION: Nonparametric or Robust Methods")
        print("  - Normality assumption violated")
        print("  - Consider: Kruskal-Wallis, permutation tests, or robust ANOVA")
    elif not adequate_sample_size:
        print("⚠ RECOMMENDATION: Caution Required")
        print("  - Sample sizes are small")
        print("  - Consider: Larger sample size, nonparametric methods, or bootstrapping")
    
    # Additional considerations
    print("\nADDITIONAL CONSIDERATIONS:")
    print("=" * 25)
    
    if not balanced:
        print("- Unbalanced design detected")
        print("- Consider Type III Sum of Squares")
        print("- Be cautious with interaction interpretation")
    
    if missing_data > 0:
        print("- Missing data present")
        print("- Consider multiple imputation or complete case analysis")
    
    # Power analysis recommendation
    if total_n < 50:
        print("- Small total sample size")
        print("- Consider power analysis for future studies")
    
    return {
        'normality': normality_results,
        'homogeneity': homogeneity_results,
        'cell_counts': cell_counts,
        'balanced': balanced,
        'total_n': total_n,
        'min_cell_size': min_cell_size,
        'missing_data': missing_data,
        'recommendation': {
            'normal_tests_passed': normal_tests_passed,
            'homogeneity_tests_passed': homogeneity_tests_passed,
            'adequate_sample_size': adequate_sample_size
        }
    }

# Apply comprehensive test selection
test_selection = choose_two_way_test(mtcars, "cyl_factor", "am_factor", "mpg")
```

### Data Preparation Best Practices

```python
# Function to prepare data for two-way ANOVA
def prepare_data_for_two_way_anova(data, factor1, factor2, response):
    print("=== DATA PREPARATION FOR TWO-WAY ANOVA ===")
    print()
    
    # Check data structure
    print("DATA STRUCTURE CHECK:")
    print("=" * 20)
    print(f"Number of observations: {len(data)}")
    print(f"Number of variables: {len(data.columns)}")
    print(f"Factor 1 levels: {len(data[factor1].unique())}")
    print(f"Factor 2 levels: {len(data[factor2].unique())}")
    print()
    
    # Check for missing values
    missing_summary = {
        'factor1_missing': data[factor1].isna().sum(),
        'factor2_missing': data[factor2].isna().sum(),
        'response_missing': data[response].isna().sum()
    }
    
    print("MISSING DATA CHECK:")
    print("=" * 18)
    print(f"Factor 1 missing: {missing_summary['factor1_missing']}")
    print(f"Factor 2 missing: {missing_summary['factor2_missing']}")
    print(f"Response missing: {missing_summary['response_missing']}")
    print()
    
    # Create clean dataset
    clean_data = data.dropna(subset=[factor1, factor2, response])
    
    print("CLEAN DATA SUMMARY:")
    print("=" * 18)
    print(f"Observations removed: {len(data) - len(clean_data)}")
    print(f"Remaining observations: {len(clean_data)}")
    print()
    
    # Ensure factors are properly coded
    clean_data[factor1] = clean_data[factor1].astype('category')
    clean_data[factor2] = clean_data[factor2].astype('category')
    
    # Check factor levels
    print("FACTOR LEVELS:")
    print("=" * 14)
    print(f"Factor 1 levels: {list(clean_data[factor1].cat.categories)}")
    print(f"Factor 2 levels: {list(clean_data[factor2].cat.categories)}")
    print()
    
    # Check for empty cells
    cell_counts = pd.crosstab(clean_data[factor1], clean_data[factor2])
    empty_cells = (cell_counts == 0).sum().sum()
    
    if empty_cells > 0:
        print("⚠ WARNING: Empty cells detected")
        print(f"Empty cells: {empty_cells}")
        print("This may cause issues with the analysis")
        print()
    else:
        print("✓ No empty cells detected")
        print()
    
    # Check for outliers
    print("OUTLIER DETECTION:")
    print("=" * 18)
    
    # Boxplot method
    Q1 = clean_data[response].quantile(0.25)
    Q3 = clean_data[response].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = clean_data[(clean_data[response] < lower_bound) | (clean_data[response] > upper_bound)][response]
    
    if len(outliers) > 0:
        print(f"Potential outliers detected: {len(outliers)}")
        print(f"Outlier values: {outliers.values.round(2)}")
        print("Consider investigating these values")
        print()
    else:
        print("✓ No outliers detected by boxplot method")
        print()
    
    return {
        'original_data': data,
        'clean_data': clean_data,
        'missing_summary': missing_summary,
        'cell_counts': cell_counts,
        'empty_cells': empty_cells,
        'outliers': outliers.values if len(outliers) > 0 else []
    }

# Prepare data
data_prep = prepare_data_for_two_way_anova(mtcars, "cyl_factor", "am_factor", "mpg")
```

### Reporting Guidelines

```python
# Function to generate comprehensive two-way ANOVA report
def generate_two_way_report(anova_result, data, factor1, factor2, response, 
                           normality_results=None, homogeneity_results=None):
    print("=== COMPREHENSIVE TWO-WAY ANOVA REPORT ===")
    print()
    
    # 1. Descriptive Statistics
    print("1. DESCRIPTIVE STATISTICS")
    print("=" * 25)
    
    desc_stats = data.groupby([factor1, factor2]).agg({
        response: ['count', 'mean', 'std', 'min', 'max']
    }).round(3)
    desc_stats.columns = ['n', 'mean', 'sd', 'min', 'max']
    desc_stats['se'] = desc_stats['sd'] / np.sqrt(desc_stats['n'])
    desc_stats = desc_stats.reset_index()
    
    print(desc_stats)
    
    # Marginal means
    marginal_factor1 = data.groupby(factor1).agg({
        response: ['count', 'mean', 'std']
    }).round(3)
    marginal_factor1.columns = ['n', 'mean', 'sd']
    marginal_factor1 = marginal_factor1.reset_index()
    
    marginal_factor2 = data.groupby(factor2).agg({
        response: ['count', 'mean', 'std']
    }).round(3)
    marginal_factor2.columns = ['n', 'mean', 'sd']
    marginal_factor2 = marginal_factor2.reset_index()
    
    print(f"\nMarginal means for {factor1}:")
    print(marginal_factor1)
    
    print(f"\nMarginal means for {factor2}:")
    print(marginal_factor2)
    
    # 2. ANOVA Results
    print("\n2. TWO-WAY ANOVA RESULTS")
    print("=" * 25)
    
    print(anova_result)
    
    # Extract key statistics
    f_factor1 = anova_result.loc[f'C({factor1})', 'F']
    f_factor2 = anova_result.loc[f'C({factor2})', 'F']
    f_interaction = anova_result.loc[f'C({factor1}):C({factor2})', 'F']
    p_factor1 = anova_result.loc[f'C({factor1})', 'PR(>F)']
    p_factor2 = anova_result.loc[f'C({factor2})', 'PR(>F)']
    p_interaction = anova_result.loc[f'C({factor1}):C({factor2})', 'PR(>F)']
    
    # 3. Effect Sizes
    print("\n3. EFFECT SIZES")
    print("=" * 15)
    
    ss_factor1 = anova_result.loc[f'C({factor1})', 'sum_sq']
    ss_factor2 = anova_result.loc[f'C({factor2})', 'sum_sq']
    ss_interaction = anova_result.loc[f'C({factor1}):C({factor2})', 'sum_sq']
    ss_error = anova_result.loc['Residual', 'sum_sq']
    ss_total = anova_result['sum_sq'].sum()
    
    partial_eta2_factor1 = ss_factor1 / (ss_factor1 + ss_error)
    partial_eta2_factor2 = ss_factor2 / (ss_factor2 + ss_error)
    partial_eta2_interaction = ss_interaction / (ss_interaction + ss_error)
    
    print("Partial η² values:")
    print(f"  {factor1}: {partial_eta2_factor1:.4f}")
    print(f"  {factor2}: {partial_eta2_factor2:.4f}")
    print(f"  Interaction: {partial_eta2_interaction:.4f}")
    
    # 4. Assumption Checking Summary
    if normality_results is not None and homogeneity_results is not None:
        print("\n4. ASSUMPTION CHECKING SUMMARY")
        print("=" * 30)
        
        print(f"Normality tests passed: {normality_results['normal_tests']} / {normality_results['total_tests']}")
        print(f"Homogeneity tests passed: {homogeneity_results['equal_var_tests']} / {homogeneity_results['total_tests']}")
        
        if (normality_results['normal_tests'] >= 3 and 
            homogeneity_results['equal_var_tests'] >= 3):
            print("✓ All assumptions appear to be met")
        else:
            print("⚠ Some assumptions may be violated")
    
    # 5. Interpretation
    print("\n5. INTERPRETATION")
    print("=" * 15)
    
    alpha = 0.05
    
    # Main effects
    if p_factor1 < alpha:
        print(f"✓ Significant main effect of {factor1} (p = {p_factor1:.4f})")
        if partial_eta2_factor1 >= 0.14:
            print(f"  Large effect size (η²p = {partial_eta2_factor1:.3f})")
        elif partial_eta2_factor1 >= 0.06:
            print(f"  Medium effect size (η²p = {partial_eta2_factor1:.3f})")
        else:
            print(f"  Small effect size (η²p = {partial_eta2_factor1:.3f})")
    else:
        print(f"✗ No significant main effect of {factor1} (p = {p_factor1:.4f})")
    
    if p_factor2 < alpha:
        print(f"✓ Significant main effect of {factor2} (p = {p_factor2:.4f})")
        if partial_eta2_factor2 >= 0.14:
            print(f"  Large effect size (η²p = {partial_eta2_factor2:.3f})")
        elif partial_eta2_factor2 >= 0.06:
            print(f"  Medium effect size (η²p = {partial_eta2_factor2:.3f})")
        else:
            print(f"  Small effect size (η²p = {partial_eta2_factor2:.3f})")
    else:
        print(f"✗ No significant main effect of {factor2} (p = {p_factor2:.4f})")
    
    # Interaction effect
    if p_interaction < alpha:
        print(f"✓ Significant interaction effect (p = {p_interaction:.4f})")
        if partial_eta2_interaction >= 0.14:
            print(f"  Large interaction effect (η²p = {partial_eta2_interaction:.3f})")
        elif partial_eta2_interaction >= 0.06:
            print(f"  Medium interaction effect (η²p = {partial_eta2_interaction:.3f})")
        else:
            print(f"  Small interaction effect (η²p = {partial_eta2_interaction:.3f})")
        print("  → Simple effects analysis recommended")
    else:
        print(f"✗ No significant interaction effect (p = {p_interaction:.4f})")
        print("  → Main effects can be interpreted independently")
    
    # 6. Recommendations
    print("\n6. RECOMMENDATIONS")
    print("=" * 17)
    
    if p_interaction < alpha:
        print("- Conduct simple effects analysis")
        print("- Perform post hoc tests for significant simple effects")
        print("- Focus interpretation on interaction effects")
    else:
        if p_factor1 < alpha or p_factor2 < alpha:
            print("- Perform post hoc tests for significant main effects")
            print("- Consider planned comparisons if theoretically justified")
    
    print("- Report effect sizes alongside p-values")
    print("- Consider practical significance of results")
    print("- Validate findings with additional analyses if needed")
    
    return {
        'descriptive_stats': desc_stats,
        'marginal_factor1': marginal_factor1,
        'marginal_factor2': marginal_factor2,
        'anova_results': anova_result,
        'effect_sizes': {
            'partial_eta2_factor1': partial_eta2_factor1,
            'partial_eta2_factor2': partial_eta2_factor2,
            'partial_eta2_interaction': partial_eta2_interaction
        },
        'interpretation': {
            'factor1_significant': p_factor1 < alpha,
            'factor2_significant': p_factor2 < alpha,
            'interaction_significant': p_interaction < alpha
        }
    }

# Generate comprehensive report
comprehensive_report = generate_two_way_report(two_way_anova, mtcars, "cyl_factor", "am_factor", "mpg",
                                              normality_results, homogeneity_results)
```

### Reporting Guidelines

```r
# Function to generate comprehensive two-way ANOVA report
generate_two_way_report <- function(anova_result, data, factor1, factor2, response) {
  cat("=== TWO-WAY ANOVA REPORT ===\n\n")
  
  # Descriptive statistics
  desc_stats <- data %>%
    group_by(!!sym(factor1), !!sym(factor2)) %>%
    summarise(
      n = n(),
      mean = mean(!!sym(response), na.rm = TRUE),
      sd = sd(!!sym(response), na.rm = TRUE)
    )
  
  cat("DESCRIPTIVE STATISTICS:\n")
  print(desc_stats)
  cat("\n")
  
  # ANOVA results
  anova_summary <- summary(anova_result)
  f_factor1 <- anova_summary[[1]]$`F value`[1]
  f_factor2 <- anova_summary[[1]]$`F value`[2]
  f_interaction <- anova_summary[[1]]$`F value`[3]
  p_factor1 <- anova_summary[[1]]$`Pr(>F)`[1]
  p_factor2 <- anova_summary[[1]]$`Pr(>F)`[2]
  p_interaction <- anova_summary[[1]]$`Pr(>F)`[3]
  
  cat("ANOVA RESULTS:\n")
  cat("Factor 1 (", factor1, ") F-statistic:", round(f_factor1, 3), "\n")
  cat("Factor 1 p-value:", round(p_factor1, 4), "\n")
  cat("Factor 2 (", factor2, ") F-statistic:", round(f_factor2, 3), "\n")
  cat("Factor 2 p-value:", round(p_factor2, 4), "\n")
  cat("Interaction F-statistic:", round(f_interaction, 3), "\n")
  cat("Interaction p-value:", round(p_interaction, 4), "\n\n")
  
  # Effect sizes
  ss_factor1 <- anova_summary[[1]]$`Sum Sq`[1]
  ss_factor2 <- anova_summary[[1]]$`Sum Sq`[2]
  ss_interaction <- anova_summary[[1]]$`Sum Sq`[3]
  ss_error <- anova_summary[[1]]$`Sum Sq`[4]
  
  partial_eta2_factor1 <- ss_factor1 / (ss_factor1 + ss_error)
  partial_eta2_factor2 <- ss_factor2 / (ss_factor2 + ss_error)
  partial_eta2_interaction <- ss_interaction / (ss_interaction + ss_error)
  
  cat("EFFECT SIZES (Partial η²):\n")
  cat("Factor 1:", round(partial_eta2_factor1, 3), "\n")
  cat("Factor 2:", round(partial_eta2_factor2, 3), "\n")
  cat("Interaction:", round(partial_eta2_interaction, 3), "\n\n")
  
  # Interpretation
  alpha <- 0.05
  cat("INTERPRETATION:\n")
  
  if (p_factor1 < alpha) {
    cat("- Significant main effect of", factor1, "\n")
  } else {
    cat("- No significant main effect of", factor1, "\n")
  }
  
  if (p_factor2 < alpha) {
    cat("- Significant main effect of", factor2, "\n")
  } else {
    cat("- No significant main effect of", factor2, "\n")
  }
  
  if (p_interaction < alpha) {
    cat("- Significant interaction effect\n")
    cat("- Simple effects analysis recommended\n")
  } else {
    cat("- No significant interaction effect\n")
    cat("- Main effects can be interpreted independently\n")
  }
}

# Generate report for cylinder and transmission ANOVA
generate_two_way_report(two_way_model, mtcars, "cyl_factor", "am_factor", "mpg")
```

## Comprehensive Exercises

The following exercises are designed to help you master two-way ANOVA concepts, from basic applications to advanced analyses.

### Exercise 1: Basic Two-Way ANOVA Analysis

**Objective**: Perform a complete two-way ANOVA analysis on the mtcars dataset.

**Task**: Analyze the effects of cylinder count (4, 6, 8) and transmission type (automatic, manual) on horsepower.

**Requirements**:
1. Create appropriate factor variables
2. Perform manual calculations for all components (SS, df, MS, F, p-values)
3. Use R's built-in `aov()` function
4. Calculate descriptive statistics for all cells
5. Create interaction plots and box plots
6. Calculate effect sizes (partial η², η², ω²)
7. Interpret results comprehensively

**Expected Learning Outcomes**:
- Understanding of two-way ANOVA mathematical foundations
- Ability to perform both manual and automated calculations
- Skills in creating informative visualizations
- Competence in effect size interpretation

**Hints**:
- Use `factor()` to create categorical variables
- Remember to check assumptions before interpretation
- Consider the practical significance of results

### Exercise 2: Interaction Effects Analysis

**Objective**: Create and analyze a dataset with significant interaction effects.

**Task**: Generate a simulated dataset with a 2×3 factorial design where:
- Factor A has 2 levels
- Factor B has 3 levels
- There is a significant interaction effect
- Sample size is 20 per cell

**Requirements**:
1. Generate data with known interaction effects
2. Perform two-way ANOVA
3. Conduct simple effects analysis
4. Create interaction plots
5. Perform post hoc tests for significant simple effects
6. Calculate and interpret effect sizes for all effects

**Expected Learning Outcomes**:
- Understanding of interaction effects
- Ability to perform simple effects analysis
- Skills in interpreting complex factorial designs
- Competence in post hoc testing for interactions

**Hints**:
- Use different means for different cells to create interactions
- Consider using `rnorm()` with different parameters for each cell
- Remember that interactions can be ordinal or disordinal

### Exercise 3: Comprehensive Assumption Checking

**Objective**: Develop expertise in assumption checking and alternative methods.

**Task**: Using the mtcars dataset, perform comprehensive assumption checking and recommend appropriate analyses.

**Requirements**:
1. Test normality using multiple methods (Shapiro-Wilk, KS, Anderson-Darling, Lilliefors)
2. Test homogeneity of variance using multiple methods (Levene's, Bartlett's, Fligner-Killeen, Brown-Forsythe)
3. Create diagnostic plots (Q-Q plots, residual plots, variance plots)
4. If assumptions are violated, perform appropriate alternatives:
   - Data transformations (log, square root, etc.)
   - Nonparametric alternatives
   - Robust methods
5. Compare results between parametric and nonparametric approaches

**Expected Learning Outcomes**:
- Comprehensive understanding of ANOVA assumptions
- Ability to choose appropriate alternative methods
- Skills in diagnostic plotting and interpretation
- Competence in data transformation

**Hints**:
- Use `shapiro.test()`, `ks.test()`, `ad.test()`, `lillie.test()`
- For homogeneity: `leveneTest()`, `bartlett.test()`, `fligner.test()`
- Consider `kruskal.test()` for nonparametric alternatives

### Exercise 4: Advanced Effect Size Analysis

**Objective**: Master effect size calculations and interpretation.

**Task**: Perform comprehensive effect size analysis for a two-way ANOVA.

**Requirements**:
1. Calculate all effect size measures (partial η², η², ω², Cohen's f)
2. Compute bootstrap confidence intervals for effect sizes
3. Create effect size comparison plots
4. Assess practical significance
5. Perform power analysis
6. Compare effect sizes across different datasets

**Expected Learning Outcomes**:
- Deep understanding of effect size measures
- Ability to interpret practical significance
- Skills in power analysis
- Competence in bootstrap methods

**Hints**:
- Use the `boot` package for bootstrap confidence intervals
- Consider using `pwr` package for power analysis
- Remember that effect sizes are sample size independent

### Exercise 5: Real-World Application

**Objective**: Apply two-way ANOVA to a real-world research scenario.

**Task**: Design and analyze a research study using two-way ANOVA.

**Scenario Options**:
- **Educational Research**: Teaching method × Class size on student performance
- **Clinical Trial**: Drug treatment × Dosage level on patient outcomes
- **Manufacturing**: Machine type × Shift on product quality
- **Marketing**: Advertisement type × Target audience on purchase behavior

**Requirements**:
1. Design the study with appropriate sample sizes
2. Generate realistic data based on the scenario
3. Perform complete two-way ANOVA analysis
4. Create publication-ready visualizations
5. Write a comprehensive results section
6. Discuss practical implications

**Expected Learning Outcomes**:
- Ability to design factorial research studies
- Skills in data generation and analysis
- Competence in results presentation
- Understanding of practical applications

**Hints**:
- Consider effect sizes when determining sample sizes
- Use realistic means and standard deviations
- Include interaction effects that make sense for the scenario

### Exercise 6: Advanced Topics

**Objective**: Explore advanced two-way ANOVA topics.

**Task**: Investigate one or more advanced topics:

**Options**:
1. **Unbalanced Designs**: Analyze data with unequal cell sizes
2. **Mixed Models**: Use `lme4` package for mixed-effects ANOVA
3. **Robust ANOVA**: Implement robust alternatives using `WRS2` package
4. **Bootstrap ANOVA**: Perform bootstrap-based ANOVA
5. **Bayesian ANOVA**: Use `BayesFactor` package for Bayesian ANOVA

**Requirements**:
1. Implement the chosen advanced method
2. Compare results with standard two-way ANOVA
3. Discuss advantages and limitations
4. Create appropriate visualizations
5. Provide recommendations for when to use each method

**Expected Learning Outcomes**:
- Understanding of advanced ANOVA methods
- Ability to choose appropriate methods for different situations
- Skills in implementing specialized packages
- Competence in method comparison

**Hints**:
- For unbalanced designs, consider Type I vs Type III SS
- For mixed models, understand random vs fixed effects
- For robust methods, understand when they're most useful

### Exercise Solutions Framework

For each exercise, follow this systematic approach:

1. **Data Preparation**:
   - Check data structure and quality
   - Create appropriate factor variables
   - Handle missing values appropriately

2. **Exploratory Analysis**:
   - Calculate descriptive statistics
   - Create initial visualizations
   - Identify potential issues

3. **Assumption Checking**:
   - Test normality and homogeneity
   - Create diagnostic plots
   - Decide on appropriate analysis method

4. **Statistical Analysis**:
   - Perform the chosen analysis
   - Calculate effect sizes
   - Conduct post hoc tests if needed

5. **Results Interpretation**:
   - Interpret statistical significance
   - Assess practical significance
   - Consider limitations and assumptions

6. **Reporting**:
   - Create publication-ready tables and figures
   - Write clear interpretations
   - Provide recommendations

**Learning Progression**:
- Start with Exercise 1 to build foundational skills
- Progress through exercises 2-4 to develop advanced competencies
- Complete Exercise 5 to apply skills to real-world scenarios
- Attempt Exercise 6 to explore cutting-edge methods

**Assessment Criteria**:
- Correct implementation of statistical methods
- Appropriate interpretation of results
- Quality of visualizations and reporting
- Understanding of underlying concepts
- Ability to make practical recommendations

## Next Steps

In the next chapter, we'll learn about repeated measures ANOVA for analyzing within-subject designs.

---

**Key Takeaways:**
- Two-way ANOVA analyzes effects of two independent variables and their interaction
- Always check assumptions before interpreting results
- Effect sizes provide important information about practical significance
- Simple effects analysis is needed when interactions are significant
- Post hoc tests are necessary for significant main effects
- Proper reporting includes descriptive statistics, test results, and effect sizes
- Interaction effects can modify the interpretation of main effects 