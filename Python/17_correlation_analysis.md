# Correlation Analysis

## Overview

Correlation analysis examines the strength and direction of relationships between two or more quantitative variables. It is foundational for understanding associations, prediction, and causality in statistics and data science.

### Key Concepts

- **Covariance**: Measures the joint variability of two variables.
- **Correlation**: Standardizes covariance to a scale from -1 to 1, indicating both strength and direction.
- **Types of Correlation**: Pearson (linear), Spearman (monotonic, rank-based), Kendall (ordinal, rank-based).
- **Interpretation**: Correlation does not imply causation.

## Mathematical Foundations

### Covariance

Covariance quantifies how two variables change together:

```math
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})
```

- $`\text{Cov}(X, Y) > 0`$: Variables tend to increase together
- $`\text{Cov}(X, Y) < 0`$: One increases as the other decreases
- $`\text{Cov}(X, Y) = 0`$: No linear relationship

### Pearson Correlation Coefficient ($`r`$)

Pearson's $`r`$ measures the strength and direction of a linear relationship:

```math
r = \frac{\text{Cov}(X, Y)}{s_X s_Y} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}
```

- $`r = 1`$: Perfect positive linear relationship
- $`r = -1`$: Perfect negative linear relationship
- $`r = 0`$: No linear relationship

### Spearman's Rank Correlation ($`\rho`$)

Spearman's $`\rho`$ assesses monotonic relationships using ranks:

```math
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
```

Where $`d_i`$ is the difference between the ranks of $`x_i`$ and $`y_i`$.

### Kendall's Tau ($`\tau`$)

Kendall's $`\tau`$ measures ordinal association:

```math
\tau = \frac{(\text{number of concordant pairs}) - (\text{number of discordant pairs})}{\frac{1}{2} n(n-1)}
```

## When to Use Each Correlation

| Correlation | Data Type | Assumptions | Use Case |
|------------|-----------|-------------|----------|
| Pearson    | Interval/Ratio | Linear, normality | Linear relationships |
| Spearman   | Ordinal/Interval | Monotonic | Nonlinear monotonic, outliers |
| Kendall    | Ordinal | Monotonic | Small samples, many ties |

## Calculating Correlation in Python

### Pearson Correlation

```python
import numpy as np
import pandas as pd
from scipy import stats

# Simulate data
np.random.seed(42)
x = np.random.normal(size=100)
y = 0.7 * x + np.random.normal(scale=0.5, size=100)

# Pearson correlation
cor_pearson, _ = stats.pearsonr(x, y)
print(f"Pearson correlation: {cor_pearson:.3f}")

# Test significance
cor_test = stats.pearsonr(x, y)
print(f"Pearson r: {cor_test[0]:.3f}, p-value: {cor_test[1]:.4f}")
```

### Spearman and Kendall Correlation

```python
# Spearman correlation
cor_spearman, p_spearman = stats.spearmanr(x, y)
print(f"Spearman correlation: {cor_spearman:.3f}")
print(f"Spearman p-value: {p_spearman:.4f}")

# Kendall correlation
cor_kendall, p_kendall = stats.kendalltau(x, y)
print(f"Kendall correlation: {cor_kendall:.3f}")
print(f"Kendall p-value: {p_kendall:.4f}")
```

### Correlation Matrix

```python
# Multiple variables
np.random.seed(123)
data = pd.DataFrame({
    'A': np.random.normal(size=100),
    'B': np.random.normal(size=100),
    'C': np.random.normal(size=100)
})

# Correlation matrix
cor_matrix = data.corr()
print(cor_matrix)

# Significance matrix
from scipy.stats import pearsonr
p_matrix = pd.DataFrame(np.ones((data.shape[1], data.shape[1])), columns=data.columns, index=data.columns)
for i in data.columns:
    for j in data.columns:
        if i != j:
            _, p = pearsonr(data[i], data[j])
            p_matrix.loc[i, j] = p
        else:
            p_matrix.loc[i, j] = np.nan
print("\nP-values matrix:")
print(p_matrix)
```

## Visualization

### Scatter Plot with Correlation

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot with regression line
plot_data = pd.DataFrame({'x': x, 'y': y})
plt.figure(figsize=(8, 6))
sns.regplot(x='x', y='y', data=plot_data, ci=95, line_kws={'color': 'blue'})
plt.title(f"Scatter Plot with Regression Line\nPearson r = {cor_pearson:.2f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.show()
```

### Correlation Matrix Heatmap

```python
cor_melt = cor_matrix.reset_index().melt(id_vars='index')
cor_melt.columns = ['Var1', 'Var2', 'value']
plt.figure(figsize=(6, 5))
sns.heatmap(cor_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', square=True)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()
```

## Assumption Checking

### Pearson Correlation Assumptions
- Linearity: Relationship between variables is linear
- Normality: Both variables are normally distributed
- Homoscedasticity: Constant variance of residuals
- No significant outliers

#### Checking Linearity and Outliers

```python
# Scatter plot for linearity and outliers
plt.figure(figsize=(7, 5))
plt.scatter(x, y, alpha=0.7)
slope, intercept = np.polyfit(x, y, 1)
plt.plot(x, slope * x + intercept, color='red')
plt.title("Scatter Plot for Linearity and Outliers")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
```

#### Checking Normality

```python
# Q-Q plots
import scipy.stats as stats
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
stats.probplot(x, dist="norm", plot=axes[0])
axes[0].set_title("Q-Q Plot for x")
stats.probplot(y, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot for y")
plt.tight_layout()
plt.show()

# Shapiro-Wilk test
print("Shapiro-Wilk test for x:", stats.shapiro(x))
print("Shapiro-Wilk test for y:", stats.shapiro(y))
```

#### Checking Homoscedasticity

```python
# Residuals vs fitted
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x.reshape(-1, 1), y)
fitted = model.predict(x.reshape(-1, 1))
residuals = y - fitted
plt.figure(figsize=(7, 5))
plt.scatter(fitted, residuals)
plt.axhline(0, color='red')
plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()
```

### Robust Alternatives
- Use Spearman or Kendall correlation if assumptions are violated
- Consider robust correlation methods (e.g., biweight midcorrelation)

## Effect Size and Confidence Intervals

### Interpreting Correlation Coefficient

| $`|r|`$      | Strength         |
|----------|------------------|
| 0.00-0.10 | Negligible       |
| 0.10-0.30 | Small            |
| 0.30-0.50 | Moderate         |
| 0.50-0.70 | Large            |
| 0.70-0.90 | Very large       |
| 0.90-1.00 | Nearly perfect   |

### Confidence Interval for $`r`$

```python
# Confidence interval for correlation (using Fisher's z)
def correlation_ci(r, n, alpha=0.05):
    from scipy.stats import norm
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha/2)
    ci_lower = np.tanh(z - z_crit * se)
    ci_upper = np.tanh(z + z_crit * se)
    return ci_lower, ci_upper

ci_low, ci_up = correlation_ci(cor_pearson, len(x))
print(f"95% CI for r: [{ci_low:.2f}, {ci_up:.2f}]")
```

### Fisher's z-Transformation

To compare correlations or compute confidence intervals:

```math
z = \frac{1}{2} \ln\left(\frac{1 + r}{1 - r}\right)
```

The standard error of $`z`$ is $`\frac{1}{\sqrt{n-3}}`$.

## Practical Examples

### Example 1: Height and Weight

```python
# Simulate height and weight data
np.random.seed(1)
height = np.random.normal(170, 10, 100)
weight = 0.5 * height + np.random.normal(0, 8, 100)

# Pearson correlation
cor_hw, p_hw = stats.pearsonr(height, weight)
print(f"Pearson correlation: {cor_hw:.3f}, p-value: {p_hw:.4f}")
```

### Example 2: Nonlinear Relationship

```python
# Simulate nonlinear data
np.random.seed(2)
x_nl = np.random.normal(size=100)
y_nl = x_nl ** 2 + np.random.normal(size=100)

# Pearson vs Spearman
cor_pearson_nl, _ = stats.pearsonr(x_nl, y_nl)
cor_spearman_nl, _ = stats.spearmanr(x_nl, y_nl)
print(f"Pearson: {cor_pearson_nl:.3f}, Spearman: {cor_spearman_nl:.3f}")
```

## Best Practices

- Always visualize data before interpreting correlation
- Check assumptions for Pearson correlation
- Use Spearman or Kendall for non-normal or ordinal data
- Report effect size and confidence intervals
- Correlation does not imply causation
- Be cautious of outliers and influential points
- Use robust methods for non-normal data

## Reporting Guidelines

- Report the type of correlation, value, confidence interval, and p-value
- Example: "There was a large, positive correlation between X and Y, $`r = 0.65, 95\%\ CI [0.50, 0.77], p < .001`$."

## Exercises

### Exercise 1: Pearson Correlation
Simulate two variables with a linear relationship. Calculate and interpret the Pearson correlation, check assumptions, and visualize the data.

### Exercise 2: Spearman Correlation
Simulate two variables with a monotonic but nonlinear relationship. Calculate and interpret the Spearman correlation.

### Exercise 3: Correlation Matrix
Simulate a dataset with at least four variables. Compute and visualize the correlation matrix. Interpret the strongest and weakest relationships.

### Exercise 4: Robust Correlation
Simulate data with outliers. Compare Pearson, Spearman, and robust correlation methods. Discuss the impact of outliers.

### Exercise 5: Real-World Application
Find a real dataset (e.g., from R's datasets package). Perform a comprehensive correlation analysis, including visualization, assumption checking, and reporting.

---

**Key Takeaways:**
- Correlation quantifies the strength and direction of association
- Pearson for linear, normal data; Spearman/Kendall for ranks or non-normal data
- Always check assumptions and visualize
- Correlation ≠ causation
- Report effect size and confidence intervals 