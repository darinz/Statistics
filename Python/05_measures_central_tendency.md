# Measures of Central Tendency

## Overview

Measures of central tendency describe the center or typical value of a dataset. They help us understand what the "average" or "typical" value is in our data. These measures are fundamental to statistical analysis and provide the foundation for understanding data distributions.

### Why Central Tendency Matters

- **Data Summarization**: Provides a single value that represents the entire dataset
- **Comparison**: Allows comparison between different groups or datasets
- **Decision Making**: Helps in making informed decisions based on typical values
- **Statistical Inference**: Forms the basis for more advanced statistical methods
- **Communication**: Simplifies complex data for stakeholders

### Types of Central Tendency Measures

1. **Mean**: Arithmetic average, most common measure
2. **Median**: Middle value, robust to outliers
3. **Mode**: Most frequent value, useful for categorical data
4. **Geometric Mean**: For rates and ratios
5. **Harmonic Mean**: For rates and speeds
6. **Trimmed Mean**: Robust version of mean
7. **Winsorized Mean**: Another robust alternative

## Mean (Arithmetic Mean)

The arithmetic mean is the sum of all values divided by the number of values. It's the most commonly used measure of central tendency.

### Mathematical Foundation

**Population Mean**:
```math
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i
```

**Sample Mean**:
```math
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
```

**Properties of the Mean**:
- The sum of deviations from the mean equals zero: $\sum_{i=1}^{n} (x_i - \bar{x}) = 0$
- The mean minimizes the sum of squared deviations: $\sum_{i=1}^{n} (x_i - \bar{x})^2$ is minimized
- The mean is affected by every value in the dataset
- The mean is sensitive to outliers

### Basic Mean Calculation

```python
import numpy as np
import pandas as pd

# Create sample data
data = [2, 4, 6, 8, 10]

# Calculate mean
mean_value = np.mean(data)
print(f"Mean: {mean_value}")

# Verify calculation manually
manual_mean = sum(data) / len(data)
print(f"Manual calculation: {manual_mean}")

# Mean with missing values
data_with_na = [2, 4, np.nan, 8, 10]
mean_with_na = np.nanmean(data_with_na)
print(f"Mean with NaN removed: {mean_with_na}")

# Understanding the effect of outliers
data_with_outlier = [2, 4, 6, 8, 10, 100]
print(f"Original mean: {np.mean(data)}")
print(f"Mean with outlier: {np.mean(data_with_outlier)}")
```

### Mean by Group

```python
# Load data
from sklearn.datasets import fetch_openml
mtcars = fetch_openml(name='mtcars', as_frame=True).frame

# Calculate mean by group
# Mean MPG by number of cylinders
mpg_by_cyl = mtcars.groupby('cyl').agg({
    'mpg': ['count', 'mean', 'std']
}).round(3)
mpg_by_cyl.columns = ['count', 'mean_mpg', 'sd_mpg']
mpg_by_cyl['se_mpg'] = mpg_by_cyl['sd_mpg'] / np.sqrt(mpg_by_cyl['count'])
print(mpg_by_cyl)

# Visualize means by group
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.barplot(data=mpg_by_cyl.reset_index(), x='cyl', y='mean_mpg', color='steelblue', alpha=0.7)
plt.errorbar(x=range(len(mpg_by_cyl)), y=mpg_by_cyl['mean_mpg'], 
             yerr=mpg_by_cyl['se_mpg'], fmt='none', color='black', capsize=5)
plt.title('Mean MPG by Number of Cylinders')
plt.xlabel('Number of Cylinders')
plt.ylabel('Mean MPG')
plt.show()
```

### Weighted Mean

The weighted mean is useful when different observations have different importance or reliability.

**Mathematical Formula**:
```math
\bar{x}_w = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}
```

```python
# Create data with weights
values = [85, 92, 78, 96, 88]
weights = [0.2, 0.3, 0.1, 0.25, 0.15]

# Calculate weighted mean
weighted_mean = np.average(values, weights=weights)
print(f"Weighted mean: {weighted_mean}")

# Verify calculation manually
manual_weighted_mean = sum(v * w for v, w in zip(values, weights)) / sum(weights)
print(f"Manual calculation: {manual_weighted_mean}")

# Example: Course grades with different weights
course_grades = pd.DataFrame({
    'assignment': ['Homework', 'Midterm', 'Final', 'Participation'],
    'grade': [85, 78, 92, 95],
    'weight': [0.3, 0.25, 0.35, 0.1]
})

course_weighted_mean = np.average(course_grades['grade'], weights=course_grades['weight'])
print(f"Course weighted mean: {course_weighted_mean}")
```

## Median

The median is the middle value when data is ordered from smallest to largest. It's robust to outliers and represents the 50th percentile.

### Mathematical Foundation

**For odd number of observations**:
```math
\text{Median} = x_{\left(\frac{n+1}{2}\right)}
```

**For even number of observations**:
```math
\text{Median} = \frac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)}}{2}
```

**Properties of the Median**:
- Resistant to outliers
- Minimizes the sum of absolute deviations: $\sum_{i=1}^{n} |x_i - \text{median}|$ is minimized
- Represents the 50th percentile
- Not affected by extreme values

### Basic Median Calculation

```python
# Calculate median
data = [1, 3, 5, 7, 9]
median_value = np.median(data)
print(f"Median: {median_value}")

# Median with even number of observations
data_even = [1, 3, 5, 7, 9, 11]
median_even = np.median(data_even)
print(f"Median (even n): {median_even}")

# Verify calculation for odd n
sorted_data = sorted(data)
n = len(data)
manual_median_odd = sorted_data[(n + 1) // 2 - 1]  # Python uses 0-based indexing
print(f"Manual median (odd): {manual_median_odd}")

# Verify calculation for even n
sorted_even = sorted(data_even)
n_even = len(data_even)
manual_median_even = (sorted_even[n_even//2 - 1] + sorted_even[n_even//2]) / 2
print(f"Manual median (even): {manual_median_even}")

# Median with missing values
data_with_na = [1, 3, np.nan, 7, 9]
median_with_na = np.nanmedian(data_with_na)
print(f"Median with NaN removed: {median_with_na}")
```

### Median by Group

```python
# Median MPG by cylinders
median_by_cyl = mtcars.groupby('cyl').agg({
    'mpg': ['count', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
}).round(3)
median_by_cyl.columns = ['count', 'median_mpg', 'q25', 'q75']
print(median_by_cyl)

# Compare mean vs median by group
comparison_by_cyl = mtcars.groupby('cyl').agg({
    'mpg': ['mean', 'median']
}).round(3)
comparison_by_cyl.columns = ['mean_mpg', 'median_mpg']
comparison_by_cyl['difference'] = comparison_by_cyl['mean_mpg'] - comparison_by_cyl['median_mpg']
comparison_by_cyl['count'] = mtcars.groupby('cyl').size()
print(comparison_by_cyl)

# Visualize mean vs median
plt.figure(figsize=(8, 6))
x_pos = range(len(comparison_by_cyl))
plt.plot(x_pos, comparison_by_cyl['mean_mpg'], 'ro-', label='Mean', markersize=8)
plt.plot(x_pos, comparison_by_cyl['median_mpg'], 'bs-', label='Median', markersize=8)
plt.xticks(x_pos, comparison_by_cyl.index)
plt.title('Mean vs Median MPG by Cylinders')
plt.xlabel('Number of Cylinders')
plt.ylabel('MPG')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Mode

The mode is the most frequently occurring value in a dataset. It's the only measure of central tendency that can be used with categorical data.

### Mathematical Foundation

**Mode Definition**:
```math
\text{Mode} = \arg\max_{x} f(x)
```

where $f(x)$ is the frequency of value $x$.

**Properties of the Mode**:
- Can be used with any data type (numeric, categorical, ordinal)
- May not be unique (bimodal, multimodal)
- May not exist (uniform distribution)
- Not affected by extreme values
- Represents the most common value

### Finding the Mode

```python
from scipy.stats import mode

# Function to find mode
def find_mode(x):
    return mode(x, keepdims=False)[0]

# Test the function
data = [1, 2, 2, 3, 4, 4, 4, 5]
mode_value = find_mode(data)
print(f"Mode: {mode_value}")

# Using value_counts to see frequency
freq_table = pd.Series(data).value_counts()
print(freq_table)

# Find all modes (for multimodal data)
def find_all_modes(x):
    freq_table = pd.Series(x).value_counts()
    max_freq = freq_table.max()
    modes = freq_table[freq_table == max_freq].index.tolist()
    return modes

# Test with multimodal data
multimodal_data = [1, 2, 2, 3, 4, 4, 5]
all_modes = find_all_modes(multimodal_data)
print(f"All modes: {all_modes}")
```

### Mode for Categorical Data

```python
# Create categorical data
colors = ['red', 'blue', 'red', 'green', 'blue', 'red', 'yellow']
mode_color = find_mode(colors)
print(f"Mode color: {mode_color}")

# Frequency table
color_freq = pd.Series(colors).value_counts()
print(color_freq)

# Visualize frequency distribution
plt.figure(figsize=(8, 6))
color_freq.plot(kind='bar', color='steelblue', alpha=0.7)
plt.title('Frequency Distribution of Colors')
plt.xlabel('Color')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Geometric Mean

The geometric mean is useful for data that represents rates of change, ratios, or multiplicative relationships.

### Mathematical Foundation

**Geometric Mean Formula**:
```math
GM = \sqrt[n]{x_1 \times x_2 \times \cdots \times x_n} = \left(\prod_{i=1}^{n} x_i\right)^{\frac{1}{n}}
```

**Logarithmic Form**:
```math
GM = \exp\left(\frac{1}{n}\sum_{i=1}^{n} \ln(x_i)\right)
```

**Properties of Geometric Mean**:
- Always less than or equal to arithmetic mean
- Useful for rates of change and ratios
- Sensitive to zeros and negative values
- Multiplicative equivalent of arithmetic mean

```python
from scipy.stats import gmean

# Function to calculate geometric mean
def geometric_mean(x):
    x = np.array(x)
    x = x[~np.isnan(x)]  # Remove NaN values
    if np.any(x <= 0):
        print("Warning: Geometric mean requires positive values")
        return np.nan
    return gmean(x)

# Example: Growth rates
growth_rates = [1.05, 1.08, 1.12, 1.06]
gm_growth = geometric_mean(growth_rates)
print(f"Geometric mean of growth rates: {gm_growth}")

# Verify with built-in function
manual_gm = np.exp(np.mean(np.log(growth_rates)))
print(f"Manual calculation: {manual_gm}")

# Compare with arithmetic mean
am_growth = np.mean(growth_rates)
print(f"Arithmetic mean: {am_growth}")
print(f"Geometric mean: {gm_growth}")

# Example: Investment returns
returns = [0.05, 0.08, -0.02, 0.12, 0.06]
# Convert to growth factors
growth_factors = [1 + r for r in returns]
gm_return = geometric_mean(growth_factors) - 1
print(f"Geometric mean return: {gm_return}")
```

## Harmonic Mean

The harmonic mean is useful for rates, speeds, and situations involving reciprocals.

### Mathematical Foundation

**Harmonic Mean Formula**:
```math
HM = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}
```

**Properties of Harmonic Mean**:
- Always less than or equal to geometric mean
- Useful for rates and speeds
- Sensitive to small values
- Reciprocal of arithmetic mean of reciprocals

```python
from scipy.stats import hmean

# Function to calculate harmonic mean
def harmonic_mean(x):
    x = np.array(x)
    x = x[~np.isnan(x)]  # Remove NaN values
    if np.any(x <= 0):
        print("Warning: Harmonic mean requires positive values")
        return np.nan
    return hmean(x)

# Example: Average speed
speeds = [60, 40, 80]  # km/h
hm_speed = harmonic_mean(speeds)
print(f"Harmonic mean speed: {hm_speed} km/h")

# Verify calculation
manual_hm = 3 / (1/60 + 1/40 + 1/80)
print(f"Manual calculation: {manual_hm} km/h")

# Compare with arithmetic mean
am_speed = np.mean(speeds)
print(f"Arithmetic mean speed: {am_speed} km/h")
print(f"Harmonic mean speed: {hm_speed} km/h")

# Example: Parallel resistors
resistors = [10, 20, 30]  # ohms
hm_resistance = harmonic_mean(resistors)
print(f"Equivalent parallel resistance: {hm_resistance} ohms")
```

## Trimmed Mean

The trimmed mean removes a percentage of extreme values before calculating the mean, making it more robust to outliers.

### Mathematical Foundation

**Trimmed Mean Formula**:
```math
\bar{x}_{\alpha} = \frac{1}{n(1-2\alpha)}\sum_{i=\alpha n + 1}^{(1-\alpha)n} x_{(i)}
```

where $\alpha$ is the trimming proportion and $x_{(i)}$ are the ordered values.

**Properties of Trimmed Mean**:
- More robust than arithmetic mean
- Less sensitive to outliers
- Maintains some efficiency for normal data
- Common trimming levels: 5%, 10%, 20%

```python
from scipy.stats import trim_mean

# Calculate trimmed mean
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]

# 10% trimmed mean (removes 10% from each end)
trimmed_mean_val = trim_mean(data, 0.1)
print(f"10% trimmed mean: {trimmed_mean_val}")

# Compare with regular mean
regular_mean = np.mean(data)
print(f"Regular mean: {regular_mean}")

# Manual calculation of 10% trimmed mean
sorted_data = sorted(data)
n = len(data)
trim_n = round(n * 0.1)
trimmed_data = sorted_data[trim_n:(n - trim_n)]
manual_trimmed_mean = np.mean(trimmed_data)
print(f"Manual trimmed mean: {manual_trimmed_mean}")

# Compare different trimming levels
trim_levels = [0, 0.05, 0.1, 0.2]
trimmed_means = [trim_mean(data, level) for level in trim_levels]
for level, mean_val in zip(trim_levels, trimmed_means):
    print(f"{level*100}% trimmed mean: {mean_val}")
```

## Winsorized Mean

The winsorized mean replaces extreme values with less extreme values rather than removing them.

### Mathematical Foundation

**Winsorized Mean Formula**:
```math
\bar{x}_w = \frac{1}{n}\left(k \cdot x_{(k+1)} + \sum_{i=k+1}^{n-k} x_{(i)} + k \cdot x_{(n-k)}\right)
```

where $k$ is the number of values to be winsorized from each end.

```python
from scipy.stats.mstats import winsorize

# Function to calculate winsorized mean
def winsorized_mean(x, k=1):
    x_winsorized = winsorize(x, limits=(k/len(x), k/len(x)))
    return np.mean(x_winsorized)

# Example
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
winsorized_mean_val = winsorized_mean(data, k=1)
print(f"Winsorized mean (k=1): {winsorized_mean_val}")

# Compare different robust measures
print(f"Original data: {data}")
print(f"Arithmetic mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"10% trimmed mean: {trim_mean(data, 0.1)}")
print(f"Winsorized mean (k=1): {winsorized_mean_val}")
```

## Comparing Measures of Central Tendency

```python
# Create data with outliers
data_with_outliers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]

# Calculate different measures
mean_val = np.mean(data_with_outliers)
median_val = np.median(data_with_outliers)
mode_val = find_mode(data_with_outliers)
geom_mean = geometric_mean(data_with_outliers)
harm_mean = harmonic_mean(data_with_outliers)
trimmed_mean_val = trim_mean(data_with_outliers, 0.1)
winsorized_mean_val = winsorized_mean(data_with_outliers, k=1)

# Compare results
print("Measures of Central Tendency:")
print(f"Mean: {mean_val}")
print(f"Median: {median_val}")
print(f"Mode: {mode_val}")
print(f"Geometric Mean: {geom_mean}")
print(f"Harmonic Mean: {harm_mean}")
print(f"10% Trimmed Mean: {trimmed_mean_val}")
print(f"Winsorized Mean: {winsorized_mean_val}")

# Create comprehensive summary function
def central_tendency_summary(x):
    print("=== CENTRAL TENDENCY SUMMARY ===")
    print(f"Data length: {len(x)}")
    print(f"Missing values: {np.sum(np.isnan(x))}\n")
    
    print("Measures of Central Tendency:")
    print(f"Mean: {np.nanmean(x)}")
    print(f"Median: {np.nanmedian(x)}")
    print(f"Mode: {find_mode(x[~np.isnan(x)])}")
    print(f"Geometric Mean: {geometric_mean(x)}")
    print(f"Harmonic Mean: {harmonic_mean(x)}")
    print(f"10% Trimmed Mean: {trim_mean(x, 0.1)}")
    print(f"Winsorized Mean: {winsorized_mean(x, k=1)}\n")
    
    # Compare mean and median
    mean_val = np.nanmean(x)
    median_val = np.nanmedian(x)
    print(f"Mean - Median: {mean_val - median_val}")
    
    if abs(mean_val - median_val) > 0.1 * mean_val:
        print("Note: Large difference suggests skewed distribution")

# Apply to mtcars MPG
central_tendency_summary(mtcars['mpg'].values)
```

## Mathematical Relationships

### Inequality Relationships

For positive data, the following relationship holds:
```math
\text{Harmonic Mean} \leq \text{Geometric Mean} \leq \text{Arithmetic Mean}
```

This is known as the **Arithmetic Mean-Geometric Mean-Harmonic Mean Inequality**.

```python
# Demonstrate the inequality
positive_data = [2, 4, 8, 16]

hm = harmonic_mean(positive_data)
gm = geometric_mean(positive_data)
am = np.mean(positive_data)

print(f"Harmonic Mean: {hm}")
print(f"Geometric Mean: {gm}")
print(f"Arithmetic Mean: {am}")

print(f"Inequality holds: {hm <= gm and gm <= am}")
```

## Practical Examples

### Example 1: Student Test Scores

```python
# Create student test scores
test_scores = [65, 72, 78, 85, 88, 90, 92, 95, 98, 100]

# Calculate various measures
print("Test Score Analysis:")
central_tendency_summary(test_scores)

# Visualize the distribution
plt.figure(figsize=(10, 6))
plt.hist(test_scores, bins=10, color='lightblue', edgecolor='white', alpha=0.7)
plt.axvline(np.mean(test_scores), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(np.median(test_scores), color='blue', linestyle='--', linewidth=2, label='Median')
plt.title('Distribution of Test Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Analyze by grade level
def grade_analysis(scores):
    print("Grade Level Analysis:")
    scores = np.array(scores)
    print(f"A (90+): {np.mean(scores[scores >= 90])}")
    print(f"B (80-89): {np.mean(scores[(scores >= 80) & (scores < 90)])}")
    print(f"C (70-79): {np.mean(scores[(scores >= 70) & (scores < 80)])}")
    print(f"D (60-69): {np.mean(scores[(scores >= 60) & (scores < 70)])}")

grade_analysis(test_scores)
```

### Example 2: Income Data

```python
# Simulate income data (right-skewed)
np.random.seed(123)
income = np.random.lognormal(mean=10, sigma=0.5, size=1000)

# Calculate measures
print("Income Analysis:")
central_tendency_summary(income)

# Compare mean and median for skewed data
print(f"Mean vs Median difference: {np.mean(income) - np.median(income)}")
print(f"Skewness indicator: {'Right-skewed' if np.mean(income) > np.median(income) else 'Left-skewed'}")

# Visualize with log scale
plt.figure(figsize=(10, 6))
plt.hist(income, bins=50, color='lightgreen', edgecolor='white', alpha=0.7)
plt.axvline(np.mean(income), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(np.median(income), color='blue', linestyle='--', linewidth=2, label='Median')
plt.title('Distribution of Income (Log Scale)')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Income percentiles
percentiles = np.percentile(income, [25, 50, 75, 90, 95, 99])
print("Income Percentiles:")
for p, val in zip([25, 50, 75, 90, 95, 99], percentiles):
    print(f"{p}th percentile: {val:.2f}")
```

### Example 3: Temperature Data

```python
# Create temperature data
temperatures = [15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42]

# Calculate measures
print("Temperature Analysis:")
central_tendency_summary(temperatures)

# Seasonal analysis
winter_temps = temperatures[:3]
summer_temps = temperatures[-3:]

print("\nWinter temperatures:")
central_tendency_summary(winter_temps)

print("\nSummer temperatures:")
central_tendency_summary(summer_temps)

# Temperature range analysis
temp_range = max(temperatures) - min(temperatures)
print(f"Temperature range: {temp_range} degrees")
print(f"Mean temperature: {np.mean(temperatures)} degrees")
print(f"Temperature variability: {np.std(temperatures)} degrees")
```

## Using Built-in Functions

```python
# Comprehensive summary
from scipy.stats import describe
desc_stats = describe(mtcars['mpg'].dropna())
print("Descriptive Statistics:")
print(f"nobs: {desc_stats.nobs}")
print(f"minmax: {desc_stats.minmax}")
print(f"mean: {desc_stats.mean}")
print(f"variance: {desc_stats.variance}")
print(f"skewness: {desc_stats.skewness}")
print(f"kurtosis: {desc_stats.kurtosis}")

# Summary by group
summary_by_cyl = mtcars.groupby('cyl').agg({
    'mpg': ['count', 'mean', 'median', 'std', 'min', 'max', 
             lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
}).round(3)
summary_by_cyl.columns = ['n', 'mean_mpg', 'median_mpg', 'sd_mpg', 'min_mpg', 'max_mpg', 'q25', 'q75']
print(summary_by_cyl)

# Robust summary statistics
from scipy.stats import median_abs_deviation
mad = median_abs_deviation(mtcars['mpg'].dropna())
print(f"Median Absolute Deviation: {mad}")
```

## Best Practices

### When to Use Each Measure

```python
# Guidelines for choosing measures
print("Guidelines for Measures of Central Tendency:")
print("1. Mean: Use for symmetric, continuous data")
print("2. Median: Use for skewed data or data with outliers")
print("3. Mode: Use for categorical data or discrete data")
print("4. Geometric Mean: Use for rates, ratios, growth rates")
print("5. Harmonic Mean: Use for rates, speeds, ratios")
print("6. Trimmed Mean: Use when outliers are present")
print("7. Winsorized Mean: Use when outliers are present but you want to keep all data")

# Decision tree function
def choose_central_tendency(x):
    print("=== CENTRAL TENDENCY DECISION TREE ===")
    
    # Check for missing values
    if np.sum(np.isnan(x)) > 0:
        print("Data contains missing values - use nan functions")
    
    # Check data type
    if hasattr(x, 'dtype') and x.dtype == 'object':
        print("Recommendation: Use MODE (categorical data)")
        return "Mode"
    
    # Check for outliers
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    outliers = np.sum((x < (q1 - 1.5 * iqr)) | (x > (q3 + 1.5 * iqr)))
    
    if outliers > 0:
        print(f"Outliers detected: {outliers}")
        print("Recommendation: Use MEDIAN or TRIMMED MEAN")
        return "Median"
    
    # Check for skewness
    mean_val = np.mean(x)
    median_val = np.median(x)
    skewness_indicator = abs(mean_val - median_val) / mean_val
    
    if skewness_indicator > 0.1:
        print("Skewed distribution detected")
        print("Recommendation: Use MEDIAN")
        return "Median"
    else:
        print("Symmetric distribution")
        print("Recommendation: Use MEAN")
        return "Mean"

# Test the decision function
choose_central_tendency(mtcars['mpg'].values)
```

### Handling Missing Data

```python
# Function to handle missing data
def robust_central_tendency(x):
    # Remove missing values
    x_clean = x[~np.isnan(x)]
    
    if len(x_clean) == 0:
        return np.nan
    
    # Calculate measures
    result = {
        'mean': np.mean(x_clean),
        'median': np.median(x_clean),
        'mode': find_mode(x_clean),
        'geometric_mean': geometric_mean(x_clean),
        'harmonic_mean': harmonic_mean(x_clean),
        'trimmed_mean': trim_mean(x_clean, 0.1),
        'winsorized_mean': winsorized_mean(x_clean, k=1),
        'n': len(x_clean),
        'missing': np.sum(np.isnan(x)),
        'missing_percent': np.sum(np.isnan(x)) / len(x) * 100
    }
    
    return result

# Test with missing data
data_with_missing = [1, 2, np.nan, 4, 5, np.nan, 7, 8]
result = robust_central_tendency(data_with_missing)
for key, value in result.items():
    print(f"{key}: {value}")
```

## Exercises

### Exercise 1: Basic Calculations
Create a dataset with 10 values and calculate the mean, median, and mode. Compare the results.

```python
# Your solution here
# Create dataset
my_data = [3, 7, 2, 9, 5, 7, 8, 4, 6, 7]

# Calculate measures
print(f"Dataset: {my_data}")
print(f"Mean: {np.mean(my_data)}")
print(f"Median: {np.median(my_data)}")
print(f"Mode: {find_mode(my_data)}")

# Compare results
print(f"Mean - Median: {np.mean(my_data) - np.median(my_data)}")
```

### Exercise 2: Outlier Impact
Create a dataset, then add an outlier and observe how it affects the mean vs. median.

```python
# Your solution here
# Original dataset
original_data = [10, 12, 15, 18, 20, 22, 25, 28, 30, 32]

# Add outlier
data_with_outlier = original_data + [100]

# Compare measures
print("Original data measures:")
print(f"Mean: {np.mean(original_data)}")
print(f"Median: {np.median(original_data)}")

print("Data with outlier measures:")
print(f"Mean: {np.mean(data_with_outlier)}")
print(f"Median: {np.median(data_with_outlier)}")

print(f"Change in mean: {np.mean(data_with_outlier) - np.mean(original_data)}")
print(f"Change in median: {np.median(data_with_outlier) - np.median(original_data)}")
```

### Exercise 3: Group Analysis
Using the `iris` dataset, calculate the mean and median of sepal length by species.

```python
# Your solution here
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]

# Group analysis
iris_summary = iris_df.groupby('species').agg({
    'sepal length (cm)': ['count', 'mean', 'median', 'std']
}).round(3)
iris_summary.columns = ['n', 'mean_sepal_length', 'median_sepal_length', 'sd_sepal_length']
print(iris_summary)

# Visualize
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_df, x='species', y='sepal length (cm)', color='lightblue', alpha=0.7)
sns.pointplot(data=iris_df, x='species', y='sepal length (cm)', color='red', markers='s', scale=0.7)
plt.title('Sepal Length by Species\nBox shows median, red point shows mean')
plt.show()
```

### Exercise 4: Robust Measures
Create a dataset with outliers and compare the regular mean, trimmed mean, and winsorized mean.

```python
# Your solution here
# Create dataset with outliers
robust_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100]

# Compare robust measures
print(f"Dataset: {robust_data}")
print(f"Regular mean: {np.mean(robust_data)}")
print(f"Median: {np.median(robust_data)}")
print(f"10% trimmed mean: {trim_mean(robust_data, 0.1)}")
print(f"20% trimmed mean: {trim_mean(robust_data, 0.2)}")
print(f"Winsorized mean (k=1): {winsorized_mean(robust_data, k=1)}")
print(f"Winsorized mean (k=2): {winsorized_mean(robust_data, k=2)}")
```

### Exercise 5: Real-world Application
Find a real dataset (or use built-in data) and perform a comprehensive central tendency analysis.

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

# Comprehensive analysis of Ozone
print("Ozone Analysis:")
central_tendency_summary(airquality['Ozone'].values)

# Analysis by month
monthly_summary = airquality.groupby('Month').agg({
    'Ozone': ['count', 'mean', 'median', 'std']
}).round(3)
monthly_summary.columns = ['n', 'mean_ozone', 'median_ozone', 'sd_ozone']
print(monthly_summary)

# Visualize
plt.figure(figsize=(12, 6))
sns.boxplot(data=airquality, x='Month', y='Ozone', color='lightgreen', alpha=0.7)
sns.pointplot(data=airquality, x='Month', y='Ozone', color='red', markers='s', scale=0.7)
plt.title('Ozone Levels by Month')
plt.xlabel('Month')
plt.ylabel('Ozone')
plt.show()
```

## Next Steps

In the next chapter, we'll learn about measures of variability, which complement central tendency measures to give us a complete picture of our data. We'll cover:

- **Range and Interquartile Range**: Measures of spread
- **Variance and Standard Deviation**: Most common measures of variability
- **Coefficient of Variation**: Relative measure of variability
- **Skewness and Kurtosis**: Measures of distribution shape
- **Robust Measures of Variability**: Median absolute deviation

---

**Key Takeaways:**
- Mean is sensitive to outliers, median is robust
- Mode is best for categorical data
- Geometric mean for rates and ratios
- Harmonic mean for rates and speeds
- Always consider the data type and distribution
- Use multiple measures for comprehensive analysis
- Handle missing data appropriately
- Consider the context when choosing measures
- Outliers can significantly affect the mean
- Different measures serve different purposes 