# Measures of Variability

## Overview

Measures of variability describe how spread out or dispersed the data is around the central tendency. They complement measures of central tendency to give us a complete picture of our data distribution. Understanding variability is crucial for statistical inference, quality control, and decision-making.

### Why Variability Matters

- **Data Quality**: High variability may indicate data quality issues or heterogeneity
- **Statistical Inference**: Variability affects confidence intervals and hypothesis tests
- **Process Control**: Low variability often indicates stable, predictable processes
- **Comparison**: Allows comparison of consistency across different groups
- **Risk Assessment**: Variability is directly related to uncertainty and risk
- **Model Selection**: Influences choice of statistical models and methods

### Types of Variability Measures

1. **Range**: Simple measure of spread (max - min)
2. **Interquartile Range (IQR)**: Robust measure of middle 50% spread
3. **Variance**: Average squared deviation from mean
4. **Standard Deviation**: Square root of variance, same units as data
5. **Coefficient of Variation**: Relative measure of variability
6. **Mean Absolute Deviation (MAD)**: Average absolute deviation
7. **Percentiles**: Position-based measures of spread

## Range

The range is the difference between the maximum and minimum values in a dataset. It's the simplest measure of variability but is highly sensitive to outliers.

### Mathematical Foundation

**Range Formula**:
```math
\text{Range} = x_{\max} - x_{\min}
```

**Properties of Range**:
- Easy to calculate and understand
- Sensitive to outliers
- Depends only on two values
- Not affected by the distribution shape
- Always non-negative

### Basic Range Calculation

```python
import numpy as np
import pandas as pd

# Create sample data
data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Calculate range
data_range = np.ptp(data)  # peak to peak (max - min)
print(f"Range: {data_range}")
print(f"Min: {np.min(data)}")
print(f"Max: {np.max(data)}")

# Using built-in function
data_range_alt = np.max(data) - np.min(data)
print(f"Range (alternative): {data_range_alt}")

# Understanding the effect of outliers
data_with_outlier = data + [100]
print(f"Original range: {np.max(data) - np.min(data)}")
print(f"Range with outlier: {np.max(data_with_outlier) - np.min(data_with_outlier)}")

# Range as percentage of mean
range_percent = ((np.max(data) - np.min(data)) / np.mean(data)) * 100
print(f"Range as % of mean: {range_percent:.2f}%")
```

### Range by Group

```python
# Load data
from sklearn.datasets import fetch_openml
mtcars = fetch_openml(name='mtcars', as_frame=True).frame

# Calculate range by cylinders
range_by_cyl = mtcars.groupby('cyl').agg({
    'mpg': ['max', 'min', 'mean']
}).round(3)
range_by_cyl.columns = ['max_mpg', 'min_mpg', 'mean_mpg']
range_by_cyl['range_mpg'] = range_by_cyl['max_mpg'] - range_by_cyl['min_mpg']
range_by_cyl['range_percent'] = (range_by_cyl['range_mpg'] / range_by_cyl['mean_mpg']) * 100
print(range_by_cyl)

# Visualize ranges by group
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.barplot(data=range_by_cyl.reset_index(), x='cyl', y='range_mpg', color='steelblue', alpha=0.7)
plt.title('MPG Range by Number of Cylinders')
plt.xlabel('Number of Cylinders')
plt.ylabel('Range (Max - Min)')
plt.show()
```

## Interquartile Range (IQR)

The IQR is the difference between the 75th percentile (Q3) and the 25th percentile (Q1). It's a robust measure that is not affected by outliers.

### Mathematical Foundation

**IQR Formula**:
```math
\text{IQR} = Q_3 - Q_1
```

where $Q_1$ is the 25th percentile and $Q_3$ is the 75th percentile.

**Properties of IQR**:
- Robust to outliers
- Represents the middle 50% of data
- Used in box plots and outlier detection
- Not affected by extreme values
- Good for skewed distributions

### Basic IQR Calculation

```python
from scipy.stats import iqr

# Calculate IQR
data_iqr = iqr(data)
print(f"IQR: {data_iqr}")

# Manual calculation
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
manual_iqr = q3 - q1
print(f"Manual IQR: {manual_iqr}")

# Verify with built-in function
print(f"Built-in IQR: {iqr(data)}")

# Understanding quartiles
print(f"Q1 (25th percentile): {q1}")
print(f"Q2 (50th percentile/median): {np.percentile(data, 50)}")
print(f"Q3 (75th percentile): {q3}")

# IQR as percentage of median
iqr_percent = (data_iqr / np.median(data)) * 100
print(f"IQR as % of median: {iqr_percent:.2f}%")
```

### IQR by Group

```python
# IQR by cylinders
iqr_by_cyl = mtcars.groupby('cyl').agg({
    'mpg': ['count', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
}).round(3)
iqr_by_cyl.columns = ['count', 'median_mpg', 'q1_mpg', 'q3_mpg']
iqr_by_cyl['iqr_mpg'] = iqr_by_cyl['q3_mpg'] - iqr_by_cyl['q1_mpg']
iqr_by_cyl['iqr_percent'] = (iqr_by_cyl['iqr_mpg'] / iqr_by_cyl['median_mpg']) * 100
print(iqr_by_cyl)

# Compare IQR vs Range
comparison_by_cyl = mtcars.groupby('cyl').agg({
    'mpg': ['min', 'max', lambda x: iqr(x)]
}).round(3)
comparison_by_cyl.columns = ['min_mpg', 'max_mpg', 'iqr_mpg']
comparison_by_cyl['range_mpg'] = comparison_by_cyl['max_mpg'] - comparison_by_cyl['min_mpg']
comparison_by_cyl['ratio'] = comparison_by_cyl['range_mpg'] / comparison_by_cyl['iqr_mpg']
print(comparison_by_cyl)
```

## Variance

Variance measures the average squared deviation from the mean. It's the foundation for many statistical methods.

### Mathematical Foundation

**Population Variance**:
```math
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2
```

**Sample Variance**:
```math
s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2
```

**Properties of Variance**:
- Always non-negative
- Sensitive to outliers
- Units are squared
- Used in statistical inference
- Minimized by the mean

### Population vs Sample Variance

```python
# Population variance (n denominator)
def population_variance(x):
    n = len(x)
    mean_x = np.mean(x)
    return np.sum((x - mean_x)**2) / n

# Sample variance (n-1 denominator)
def sample_variance(x):
    n = len(x)
    mean_x = np.mean(x)
    return np.sum((x - mean_x)**2) / (n - 1)

# Compare with built-in function
print(f"Population variance: {population_variance(data)}")
print(f"Sample variance: {sample_variance(data)}")
print(f"Built-in var(): {np.var(data, ddof=1)}")

# Understanding the difference
n = len(data)
print(f"Sample size (n): {n}")
print(f"Degrees of freedom (n-1): {n-1}")
print(f"Ratio n/(n-1): {n/(n-1):.3f}")
print(f"Population variance * (n/(n-1)) = Sample variance: {population_variance(data) * (n/(n-1))}")
```

### Variance by Group

```python
# Variance by cylinders
variance_by_cyl = mtcars.groupby('cyl').agg({
    'mpg': ['var', 'mean', 'std']
}).round(3)
variance_by_cyl.columns = ['variance_mpg', 'mean_mpg', 'sd_mpg']
variance_by_cyl['cv_mpg'] = (variance_by_cyl['sd_mpg'] / variance_by_cyl['mean_mpg']) * 100
print(variance_by_cyl)

# Visualize variance by group
plt.figure(figsize=(8, 6))
sns.barplot(data=variance_by_cyl.reset_index(), x='cyl', y='variance_mpg', color='lightgreen', alpha=0.7)
plt.title('MPG Variance by Number of Cylinders')
plt.xlabel('Number of Cylinders')
plt.ylabel('Variance')
plt.show()
```

## Standard Deviation

Standard deviation is the square root of variance and is in the same units as the original data. It's the most commonly used measure of variability.

### Mathematical Foundation

**Population Standard Deviation**:
```math
\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2}
```

**Sample Standard Deviation**:
```math
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2}
```

**Properties of Standard Deviation**:
- Same units as original data
- Sensitive to outliers
- Used in normal distribution properties
- Basis for confidence intervals
- Used in z-scores and standardization

### Basic Standard Deviation

```python
# Calculate standard deviation
data_sd = np.std(data, ddof=1)
print(f"Standard deviation: {data_sd}")

# Manual calculation
manual_sd = np.sqrt(np.var(data, ddof=1))
print(f"Manual SD: {manual_sd}")

# Population standard deviation
def population_sd(x):
    return np.sqrt(population_variance(x))

print(f"Population SD: {population_sd(data)}")

# Understanding the empirical rule
mean_data = np.mean(data)
sd_data = np.std(data, ddof=1)
print(f"Mean: {mean_data}")
print(f"Standard deviation: {sd_data}")
print(f"68% of data within: {mean_data - sd_data:.2f} to {mean_data + sd_data:.2f}")
print(f"95% of data within: {mean_data - 2*sd_data:.2f} to {mean_data + 2*sd_data:.2f}")
print(f"99.7% of data within: {mean_data - 3*sd_data:.2f} to {mean_data + 3*sd_data:.2f}")
```

### Standard Deviation by Group

```python
# SD by cylinders
sd_by_cyl = mtcars.groupby('cyl').agg({
    'mpg': ['std', 'mean', 'count']
}).round(3)
sd_by_cyl.columns = ['sd_mpg', 'mean_mpg', 'count']
sd_by_cyl['cv_mpg'] = sd_by_cyl['sd_mpg'] / sd_by_cyl['mean_mpg']  # Coefficient of variation
sd_by_cyl['se_mpg'] = sd_by_cyl['sd_mpg'] / np.sqrt(sd_by_cyl['count'])  # Standard error
print(sd_by_cyl)

# Visualize mean ± SD by group
plt.figure(figsize=(8, 6))
plt.errorbar(x=sd_by_cyl.index, y=sd_by_cyl['mean_mpg'], 
             yerr=sd_by_cyl['sd_mpg'], fmt='o', color='red', capsize=5)
plt.title('Mean ± SD by Cylinders')
plt.xlabel('Number of Cylinders')
plt.ylabel('MPG')
plt.grid(True, alpha=0.3)
plt.show()
```

## Coefficient of Variation

The coefficient of variation (CV) is the ratio of standard deviation to mean, expressed as a percentage. It allows comparison of variability across different scales.

### Mathematical Foundation

**Coefficient of Variation Formula**:
```math
CV = \frac{s}{\bar{x}} \times 100\%
```

**Properties of CV**:
- Dimensionless measure
- Allows comparison across different scales
- Sensitive to small means
- Used in quality control
- Good for positive data only

```python
# Calculate CV
def cv(x):
    return (np.std(x, ddof=1) / np.mean(x)) * 100

# Apply to different variables
print(f"CV for MPG: {cv(mtcars['mpg']):.2f}%")
print(f"CV for Weight: {cv(mtcars['wt']):.2f}%")
print(f"CV for Horsepower: {cv(mtcars['hp']):.2f}%")

# Compare variability across different scales
cv_comparison = pd.DataFrame({
    'Variable': ['MPG', 'Weight', 'Horsepower'],
    'Mean': [mtcars['mpg'].mean(), mtcars['wt'].mean(), mtcars['hp'].mean()],
    'SD': [mtcars['mpg'].std(ddof=1), mtcars['wt'].std(ddof=1), mtcars['hp'].std(ddof=1)],
    'CV': [cv(mtcars['mpg']), cv(mtcars['wt']), cv(mtcars['hp'])]
})
print(cv_comparison)

# Visualize CV comparison
plt.figure(figsize=(8, 6))
sns.barplot(data=cv_comparison, x='Variable', y='CV', color='orange', alpha=0.7)
plt.title('Coefficient of Variation Comparison')
plt.xlabel('Variable')
plt.ylabel('CV (%)')
plt.show()

# Understanding CV interpretation
print("CV Interpretation:")
print("CV < 15%: Low variability")
print("CV 15-35%: Moderate variability")
print("CV > 35%: High variability")
```

## Mean Absolute Deviation (MAD)

MAD is the average of the absolute deviations from the mean. It's more robust than standard deviation but less commonly used.

### Mathematical Foundation

**MAD Formula**:
```math
\text{MAD} = \frac{1}{n}\sum_{i=1}^{n} |x_i - \bar{x}|
```

**Properties of MAD**:
- Same units as original data
- More robust than standard deviation
- Less sensitive to outliers
- Not used in normal distribution properties
- Computationally simple

```python
from scipy.stats import median_abs_deviation

# Calculate MAD
def mad_calculation(x):
    return np.mean(np.abs(x - np.mean(x)))

# Compare with built-in function
print(f"Manual MAD: {mad_calculation(data)}")
print(f"Built-in MAD: {median_abs_deviation(data)}")

# Note: scipy's median_abs_deviation uses median by default
print(f"MAD around median: {median_abs_deviation(data)}")
print(f"MAD around mean: {median_abs_deviation(data, center=np.mean(data))}")

# Compare MAD vs SD
print(f"Standard deviation: {np.std(data, ddof=1)}")
print(f"MAD around mean: {median_abs_deviation(data, center=np.mean(data))}")
print(f"Ratio SD/MAD: {np.std(data, ddof=1) / median_abs_deviation(data, center=np.mean(data)):.3f}")

# For normal distribution, SD ≈ 1.253 × MAD
print("Expected ratio for normal distribution: 1.253")
```

## Percentiles and Quantiles

Percentiles divide the data into 100 equal parts, while quantiles divide into any number of parts.

### Mathematical Foundation

**Percentile Definition**:
The $p$th percentile is the value below which $p\%$ of the data falls.

**Common Percentiles**:
- 25th percentile (Q1): First quartile
- 50th percentile: Median
- 75th percentile (Q3): Third quartile
- 90th percentile: 90% of data below this value

```python
# Calculate percentiles
percentiles = np.percentile(mtcars['mpg'], [10, 25, 50, 75, 90])
print("Percentiles:")
for p, val in zip([10, 25, 50, 75, 90], percentiles):
    print(f"{p}th percentile: {val:.2f}")

# Calculate deciles (10th, 20th, ..., 90th percentiles)
deciles = np.percentile(mtcars['mpg'], np.arange(10, 100, 10))
print("\nDeciles:")
for i, val in enumerate(deciles, 1):
    print(f"{i*10}th percentile: {val:.2f}")

# Five-number summary
five_number = [np.min(mtcars['mpg']), 
               np.percentile(mtcars['mpg'], 25),
               np.median(mtcars['mpg']),
               np.percentile(mtcars['mpg'], 75),
               np.max(mtcars['mpg'])]
print("\nFive-number summary:")
print(f"Min: {five_number[0]:.2f}")
print(f"Q1: {five_number[1]:.2f}")
print(f"Median: {five_number[2]:.2f}")
print(f"Q3: {five_number[3]:.2f}")
print(f"Max: {five_number[4]:.2f}")

# Percentile ranks
def percentile_rank(x, value):
    return (np.sum(x <= value) / len(x)) * 100

print(f"\nPercentile rank of 20 MPG: {percentile_rank(mtcars['mpg'], 20):.1f}%")
print(f"Percentile rank of 25 MPG: {percentile_rank(mtcars['mpg'], 25):.1f}%")
```

## Robust Measures of Variability

### Median Absolute Deviation (MAD)

MAD around the median is a robust measure of variability.

```python
# MAD around median
mad_median = median_abs_deviation(mtcars['mpg'])
print(f"MAD around median: {mad_median:.3f}")

# MAD around mean
mad_mean = median_abs_deviation(mtcars['mpg'], center=np.mean(mtcars['mpg']))
print(f"MAD around mean: {mad_mean:.3f}")

# Compare MAD with IQR
print(f"IQR: {iqr(mtcars['mpg']):.3f}")
print(f"MAD around median: {mad_median:.3f}")
print(f"Ratio IQR/MAD: {iqr(mtcars['mpg']) / mad_median:.3f}")

# For normal distribution, IQR ≈ 1.349 × MAD
print("Expected ratio for normal distribution: 1.349")
```

### Quartile Coefficient of Dispersion

The quartile coefficient of dispersion is a robust relative measure of variability.

```python
# Quartile coefficient of dispersion
def qcd(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    return (q3 - q1) / (q3 + q1)

print(f"Quartile coefficient of dispersion: {qcd(mtcars['mpg']):.3f}")

# Compare with CV
print(f"Coefficient of variation: {cv(mtcars['mpg']):.2f}%")
print(f"Quartile coefficient of dispersion: {qcd(mtcars['mpg']):.3f}")
```

## Comparing Variability Measures

```python
# Comprehensive variability summary
def variability_summary(x, variable_name="Variable"):
    print(f"=== VARIABILITY SUMMARY FOR {variable_name} ===")
    print(f"Data length: {len(x)}")
    print(f"Missing values: {np.sum(np.isnan(x))}\n")
    
    print("Basic Measures:")
    print(f"Range: {np.max(x) - np.min(x):.3f}")
    print(f"IQR: {iqr(x):.3f}")
    print(f"Variance: {np.var(x, ddof=1):.3f}")
    print(f"Standard Deviation: {np.std(x, ddof=1):.3f}")
    print(f"Coefficient of Variation: {(np.std(x, ddof=1) / np.mean(x)) * 100:.2f}%")
    
    print("\nRobust Measures:")
    print(f"MAD (around median): {median_abs_deviation(x):.3f}")
    print(f"MAD (around mean): {median_abs_deviation(x, center=np.mean(x)):.3f}")
    print(f"Quartile coefficient of dispersion: {qcd(x):.3f}")
    
    print("\nPercentiles:")
    percentiles = np.percentile(x, [5, 25, 50, 75, 95])
    print(f"5th percentile: {percentiles[0]:.3f}")
    print(f"25th percentile: {percentiles[1]:.3f}")
    print(f"50th percentile (median): {percentiles[2]:.3f}")
    print(f"75th percentile: {percentiles[3]:.3f}")
    print(f"95th percentile: {percentiles[4]:.3f}")

# Apply to different variables
variability_summary(mtcars['mpg'], "MPG")
print()
variability_summary(mtcars['wt'], "Weight")
```

## Outlier Detection Using Variability Measures

```python
# Function to detect outliers using IQR method
def detect_outliers_iqr(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr_val = q3 - q1
    lower_bound = q1 - 1.5 * iqr_val
    upper_bound = q3 + 1.5 * iqr_val
    
    outliers = (x < lower_bound) | (x > upper_bound)
    return {
        'outliers': x[outliers],
        'outlier_indices': np.where(outliers)[0],
        'bounds': [lower_bound, upper_bound],
        'n_outliers': np.sum(outliers)
    }

# Function to detect outliers using z-score method
def detect_outliers_zscore(x, threshold=3):
    z_scores = np.abs((x - np.mean(x)) / np.std(x, ddof=1))
    outliers = z_scores > threshold
    
    return {
        'outliers': x[outliers],
        'outlier_indices': np.where(outliers)[0],
        'z_scores': z_scores,
        'n_outliers': np.sum(outliers)
    }

# Detect outliers in MPG
mpg_outliers_iqr = detect_outliers_iqr(mtcars['mpg'])
mpg_outliers_zscore = detect_outliers_zscore(mtcars['mpg'])

print("=== OUTLIER DETECTION FOR MPG ===")
print(f"IQR method outliers: {mpg_outliers_iqr['outliers'].tolist()}")
print(f"IQR method count: {mpg_outliers_iqr['n_outliers']}")
print(f"Z-score method outliers: {mpg_outliers_zscore['outliers'].tolist()}")
print(f"Z-score method count: {mpg_outliers_zscore['n_outliers']}")
```

## Practical Examples

### Example 1: Comparing Variability Across Groups

```python
# Compare MPG variability across transmission types
auto_mpg = mtcars[mtcars['am'] == 0]['mpg']
manual_mpg = mtcars[mtcars['am'] == 1]['mpg']

print("=== MPG VARIABILITY BY TRANSMISSION ===")
print("Automatic transmission:")
variability_summary(auto_mpg, "Auto MPG")

print("\nManual transmission:")
variability_summary(manual_mpg, "Manual MPG")

# Test for equal variances
from scipy.stats import levene
var_test = levene(auto_mpg, manual_mpg)
print(f"\nLevene's test for equal variances:")
print(f"Statistic: {var_test.statistic:.3f}")
print(f"p-value: {var_test.pvalue:.4f}")

# Visualize variability comparison
transmission_data = pd.DataFrame({
    'transmission': ['Automatic'] * len(auto_mpg) + ['Manual'] * len(manual_mpg),
    'mpg': list(auto_mpg) + list(manual_mpg)
})

plt.figure(figsize=(8, 6))
sns.boxplot(data=transmission_data, x='transmission', y='mpg', color='lightblue', alpha=0.7)
plt.title('MPG Variability by Transmission Type')
plt.xlabel('Transmission')
plt.ylabel('MPG')
plt.show()
```

### Example 2: Time Series Variability

```python
# Simulate time series data
np.random.seed(123)
time_series = np.cumsum(np.random.normal(0, 1, 100))

# Calculate rolling standard deviation
def rolling_sd(x, window=10):
    n = len(x)
    result = np.zeros(n - window + 1)
    
    for i in range(n - window + 1):
        result[i] = np.std(x[i:(i + window)], ddof=1)
    
    return result

# Apply rolling SD
rolling_volatility = rolling_sd(time_series, window=10)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(rolling_volatility, 'b-')
plt.title('Rolling Standard Deviation')
plt.xlabel('Time')
plt.ylabel('Standard Deviation')
plt.grid(True, alpha=0.3)
plt.show()

# Calculate overall variability measures
print("=== TIME SERIES VARIABILITY ===")
print(f"Overall SD: {np.std(time_series, ddof=1):.3f}")
print(f"Mean rolling SD: {np.mean(rolling_volatility):.3f}")
print(f"SD of rolling SD: {np.std(rolling_volatility, ddof=1):.3f}")
```

### Example 3: Financial Data Analysis

```python
# Simulate stock returns
np.random.seed(123)
returns = np.random.normal(0.001, 0.02, 252)  # Daily returns

# Calculate volatility measures
print("=== STOCK RETURN VOLATILITY ===")
print(f"Daily volatility (SD): {np.std(returns, ddof=1):.4f}")
print(f"Annualized volatility: {np.std(returns, ddof=1) * np.sqrt(252):.4f}")
print(f"Downside deviation: {np.std(returns[returns < 0], ddof=1):.4f}")

# Value at Risk (VaR)
var_95 = np.percentile(returns, 5)
print(f"95% VaR: {var_95:.4f}")

# Expected Shortfall (Conditional VaR)
es_95 = np.mean(returns[returns <= var_95])
print(f"95% Expected Shortfall: {es_95:.4f}")

# Rolling volatility
rolling_vol = rolling_sd(returns, window=20)
print(f"Mean 20-day rolling volatility: {np.mean(rolling_vol):.4f}")
print(f"Volatility of volatility: {np.std(rolling_vol, ddof=1):.4f}")
```

## Visualization of Variability

```python
# Box plots to show variability
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
sns.boxplot(data=mtcars, x='cyl', y='mpg', color='lightblue')
plt.title('MPG Variability by Cylinders')
plt.xlabel('Number of Cylinders')
plt.ylabel('Miles per Gallon')

# Violin plots for density and variability
plt.subplot(2, 2, 2)
sns.violinplot(data=mtcars, x='cyl', y='mpg', color='lightgreen', alpha=0.7)
sns.boxplot(data=mtcars, x='cyl', y='mpg', width=0.2, color='white')
plt.title('MPG Distribution by Cylinders')
plt.xlabel('Number of Cylinders')
plt.ylabel('Miles per Gallon')

# Histogram with variability measures
plt.subplot(2, 2, 3)
plt.hist(mtcars['mpg'], bins=15, color='lightblue', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(mtcars['mpg']), color='red', linestyle='--', label='Mean')
plt.axvline(np.mean(mtcars['mpg']) + np.std(mtcars['mpg'], ddof=1), color='blue', linestyle=':', label='+1 SD')
plt.axvline(np.mean(mtcars['mpg']) - np.std(mtcars['mpg'], ddof=1), color='blue', linestyle=':', label='-1 SD')
plt.title('MPG Distribution with Mean and ±1 SD')
plt.xlabel('Miles per Gallon')
plt.ylabel('Frequency')
plt.legend()

# Scatter plot with error bars
plt.subplot(2, 2, 4)
plt.scatter(mtcars['wt'], mtcars['mpg'], alpha=0.7)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('MPG vs Weight')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Best Practices

### When to Use Each Measure

```python
# Guidelines for choosing variability measures
print("GUIDELINES FOR CHOOSING VARIABILITY MEASURES:")
print("1. Standard Deviation: Use for symmetric, normal distributions")
print("2. IQR: Use for skewed distributions or when outliers are present")
print("3. Coefficient of Variation: Use to compare variability across different scales")
print("4. MAD: Use when you want robustness to outliers")
print("5. Range: Use for quick assessment, but sensitive to outliers")
print("6. Percentiles: Use for non-parametric analysis")

# Decision tree function
def choose_variability_measure(x):
    print("=== VARIABILITY MEASURE DECISION TREE ===")
    
    # Check for missing values
    if np.sum(np.isnan(x)) > 0:
        print("Data contains missing values - use nan functions")
    
    # Check for outliers
    outliers_iqr = detect_outliers_iqr(x)
    outliers_zscore = detect_outliers_zscore(x)
    
    if outliers_iqr['n_outliers'] > 0 or outliers_zscore['n_outliers'] > 0:
        print("Outliers detected")
        print("Recommendation: Use IQR or MAD (robust measures)")
        return "IQR"
    
    # Check for skewness
    mean_val = np.mean(x)
    median_val = np.median(x)
    skewness_indicator = abs(mean_val - median_val) / mean_val
    
    if skewness_indicator > 0.1:
        print("Skewed distribution detected")
        print("Recommendation: Use IQR or MAD")
        return "IQR"
    else:
        print("Symmetric distribution")
        print("Recommendation: Use Standard Deviation")
        return "SD"

# Test the decision function
choose_variability_measure(mtcars['mpg'])
```

### Handling Missing Data

```python
# Function to handle missing data in variability calculations
def robust_variability(x):
    # Remove missing values
    x_clean = x[~np.isnan(x)]
    
    if len(x_clean) == 0:
        return np.nan
    
    # Calculate measures
    result = {
        'range': np.max(x_clean) - np.min(x_clean),
        'iqr': iqr(x_clean),
        'variance': np.var(x_clean, ddof=1),
        'sd': np.std(x_clean, ddof=1),
        'cv': (np.std(x_clean, ddof=1) / np.mean(x_clean)) * 100,
        'mad': median_abs_deviation(x_clean),
        'qcd': qcd(x_clean),
        'n': len(x_clean),
        'missing': np.sum(np.isnan(x)),
        'missing_percent': np.sum(np.isnan(x)) / len(x) * 100
    }
    
    return result

# Test with missing data
data_with_na = [1, 2, np.nan, 4, 5, np.nan, 7, 8]
result = robust_variability(data_with_na)
for key, value in result.items():
    print(f"{key}: {value}")
```

## Exercises

### Exercise 1: Basic Variability Calculations
Calculate range, IQR, variance, and standard deviation for the following dataset: [12, 15, 18, 22, 25, 28, 30, 35, 40, 45]

```python
# Your solution here
exercise_data = [12, 15, 18, 22, 25, 28, 30, 35, 40, 45]

print(f"Dataset: {exercise_data}")
print(f"Range: {np.max(exercise_data) - np.min(exercise_data)}")
print(f"IQR: {iqr(exercise_data)}")
print(f"Variance: {np.var(exercise_data, ddof=1):.3f}")
print(f"Standard Deviation: {np.std(exercise_data, ddof=1):.3f}")
print(f"Coefficient of Variation: {(np.std(exercise_data, ddof=1) / np.mean(exercise_data)) * 100:.2f}%")
```

### Exercise 2: Comparing Variability
Compare the variability of different variables in the mtcars dataset using appropriate measures.

```python
# Your solution here
# Compare variability across different variables
variables = ['mpg', 'wt', 'hp', 'disp']
comparison_df = pd.DataFrame({
    'Variable': variables,
    'Mean': [mtcars[var].mean() for var in variables],
    'SD': [mtcars[var].std(ddof=1) for var in variables],
    'CV': [cv(mtcars[var]) for var in variables],
    'IQR': [iqr(mtcars[var]) for var in variables],
    'Range': [mtcars[var].max() - mtcars[var].min() for var in variables]
})

print(comparison_df)

# Visualize CV comparison
plt.figure(figsize=(8, 6))
sns.barplot(data=comparison_df, x='Variable', y='CV', color='steelblue', alpha=0.7)
plt.title('Coefficient of Variation Comparison')
plt.xlabel('Variable')
plt.ylabel('CV (%)')
plt.show()
```

### Exercise 3: Outlier Detection
Use the IQR method to detect outliers in the iris dataset for each species.

```python
# Your solution here
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]

# Detect outliers by species
species_list = iris_df['species'].unique()
for species in species_list:
    species_data = iris_df[iris_df['species'] == species]['sepal length (cm)']
    outliers = detect_outliers_iqr(species_data)
    
    print(f"=== {species} ===")
    print(f"Outliers: {outliers['outliers'].tolist()}")
    print(f"Number of outliers: {outliers['n_outliers']}")
    print(f"Total observations: {len(species_data)}")
    print(f"Outlier percentage: {(outliers['n_outliers'] / len(species_data)) * 100:.1f}%\n")
```

### Exercise 4: Robust Measures
Compare standard deviation and MAD for datasets with and without outliers.

```python
# Your solution here
# Create datasets
clean_data = np.random.normal(10, 2, 100)
outlier_data = np.concatenate([clean_data, [50, 60, 70]])

# Compare measures
print("=== CLEAN DATA ===")
print(f"SD: {np.std(clean_data, ddof=1):.3f}")
print(f"MAD: {median_abs_deviation(clean_data):.3f}")
print(f"Ratio SD/MAD: {np.std(clean_data, ddof=1) / median_abs_deviation(clean_data):.3f}")

print("\n=== DATA WITH OUTLIERS ===")
print(f"SD: {np.std(outlier_data, ddof=1):.3f}")
print(f"MAD: {median_abs_deviation(outlier_data):.3f}")
print(f"Ratio SD/MAD: {np.std(outlier_data, ddof=1) / median_abs_deviation(outlier_data):.3f}")

print(f"\nChange in SD: {np.std(outlier_data, ddof=1) - np.std(clean_data, ddof=1):.3f}")
print(f"Change in MAD: {median_abs_deviation(outlier_data) - median_abs_deviation(clean_data):.3f}")
```

### Exercise 5: Real-world Application
Analyze the variability of a real dataset of your choice and interpret the results.

```python
# Your solution here
# Use built-in airquality dataset
from sklearn.datasets import fetch_openml
try:
    airquality = fetch_openml(name='airquality', as_frame=True).frame
except:
    # If airquality not available, use a different dataset
    airquality = pd.DataFrame({
        'Ozone': np.random.normal(30, 15, 100),
        'Month': np.random.randint(1, 13, 100)
    })

# Comprehensive variability analysis
print("=== AIR QUALITY VARIABILITY ANALYSIS ===")

# Analyze Ozone
variability_summary(airquality['Ozone'], "Ozone")

# Compare variability by month
monthly_summary = airquality.groupby('Month').agg({
    'Ozone': ['count', 'mean', 'std', lambda x: cv(x), lambda x: iqr(x)]
}).round(3)
monthly_summary.columns = ['n', 'mean_ozone', 'sd_ozone', 'cv_ozone', 'iqr_ozone']
print(monthly_summary)

# Visualize variability by month
plt.figure(figsize=(10, 6))
sns.boxplot(data=airquality, x='Month', y='Ozone', color='lightgreen', alpha=0.7)
plt.title('Ozone Variability by Month')
plt.xlabel('Month')
plt.ylabel('Ozone')
plt.show()
```

## Next Steps

In the next chapter, we'll learn about data visualization techniques to complement our understanding of central tendency and variability. We'll cover:

- **Histograms and Density Plots**: Visualizing distributions
- **Box Plots and Violin Plots**: Showing variability and shape
- **Scatter Plots**: Exploring relationships between variables
- **Q-Q Plots**: Assessing normality
- **Time Series Plots**: Visualizing temporal patterns

---

**Key Takeaways:**
- Standard deviation is most common but sensitive to outliers
- IQR is robust and good for skewed distributions
- Coefficient of variation allows comparison across different scales
- MAD is robust to outliers
- Always consider the data distribution when choosing measures
- Use multiple measures for comprehensive understanding
- Visualize variability alongside central tendency
- Outliers can significantly affect variance and standard deviation
- Robust measures are preferred for non-normal distributions
- Variability measures are essential for statistical inference 