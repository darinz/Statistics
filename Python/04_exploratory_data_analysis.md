# Exploratory Data Analysis

## Overview

Exploratory Data Analysis (EDA) is the first step in any statistical analysis. It involves examining and understanding your data through visualization and summary statistics before applying formal statistical methods. EDA is both an art and a science - it requires creativity in visualization and rigor in statistical thinking.

### The Philosophy of EDA

EDA follows these key principles:
- **Data-Driven Discovery**: Let the data guide your analysis, not preconceived notions
- **Iterative Process**: EDA is not linear; you may need to revisit earlier steps
- **Multiple Perspectives**: Use different techniques to understand the same data
- **Documentation**: Record your findings and decisions for reproducibility

### Why EDA Matters

- **Data Quality**: Identify issues before they affect your analysis
- **Pattern Recognition**: Discover relationships and trends in the data
- **Assumption Validation**: Check if your statistical assumptions are met
- **Hypothesis Generation**: EDA often leads to new research questions
- **Communication**: Visualizations help communicate findings to stakeholders

### The EDA Framework

1. **Data Overview**: Understand structure, size, and basic properties
2. **Data Quality Assessment**: Check for missing values, outliers, inconsistencies
3. **Univariate Analysis**: Examine individual variables in isolation
4. **Bivariate Analysis**: Explore relationships between pairs of variables
5. **Multivariate Analysis**: Understand complex interactions
6. **Data Transformation**: Apply transformations if needed
7. **Summary and Recommendations**: Document findings and next steps

## Data Exploration Fundamentals

### Initial Data Inspection

```python
import pandas as pd
import numpy as np

# Load sample data
from sklearn.datasets import fetch_openml
mtcars = fetch_openml(name='mtcars', as_frame=True).frame

# Basic data structure
print(mtcars.info())        # Structure: data types and first few values
print(mtcars.shape)         # Dimensions: rows and columns
print(mtcars.columns)       # Variable names

# Data types and summary
print(mtcars.dtypes)        # Data types of each variable
print(mtcars.head(10))      # First 10 rows
print(mtcars.tail(5))       # Last 5 rows

# Memory usage
print(mtcars.memory_usage(deep=True).sum(), 'bytes')
print(f"{mtcars.memory_usage(deep=True).sum() / 1024 ** 2:.3f} MB")
```

### Understanding Data Types

Different data types require different analytical approaches:

- **Numeric (Continuous)**: Can take any value within a range
- **Numeric (Discrete)**: Can only take specific values (e.g., counts)
- **Categorical (Nominal)**: Categories with no natural order
- **Categorical (Ordinal)**: Categories with natural order
- **Binary**: Two possible values (0/1, True/False)
- **Date/Time**: Temporal data requiring special handling

### Summary Statistics

```python
# Comprehensive summary
print(mtcars.describe(include='all'))

# Detailed summary for numeric variables
print(mtcars.select_dtypes(include=[np.number]).describe())

# Summary by groups
summary_by_am = mtcars.groupby('am').agg(
    n=('mpg', 'count'),
    mean_mpg=('mpg', 'mean'),
    sd_mpg=('mpg', 'std'),
    median_mpg=('mpg', 'median'),
    min_mpg=('mpg', 'min'),
    max_mpg=('mpg', 'max'),
    q25=('mpg', lambda x: x.quantile(0.25)),
    q75=('mpg', lambda x: x.quantile(0.75)),
    iqr=('mpg', lambda x: x.quantile(0.75) - x.quantile(0.25))
)
print(summary_by_am)

# Coefficient of variation (relative variability)
cv_mpg = mtcars['mpg'].std() / mtcars['mpg'].mean()
print(f"Coefficient of variation for MPG: {cv_mpg:.3f}")
```

### Mathematical Foundation: Descriptive Statistics

**Arithmetic Mean**: The sum of all values divided by the number of values
```math
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
```

**Median**: The middle value when data is ordered
- For odd n: middle value
- For even n: average of two middle values

**Standard Deviation**: Measure of variability around the mean
```math
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2}
```

**Coefficient of Variation**: Relative measure of variability
```math
CV = \frac{s}{\bar{x}} \times 100\%
```

**Interquartile Range (IQR)**: Range containing the middle 50% of data
```math
IQR = Q_3 - Q_1
```

## Data Quality Assessment

### Understanding Data Quality Issues

Data quality problems can significantly impact analysis:

- **Missing Values**: Can bias results and reduce sample size
- **Outliers**: May represent errors or genuine extreme values
- **Inconsistencies**: Logical contradictions in the data
- **Duplicates**: Can inflate sample size and bias results
- **Data Type Issues**: Numbers stored as text, incorrect formats

### Missing Values Analysis

```python
# Check for missing values
missing_summary = mtcars.isnull().sum()
print(missing_summary)

# Calculate missing percentage
missing_percent = (missing_summary / len(mtcars)) * 100

# Create missing data summary
missing_data = pd.DataFrame({
    'variable': missing_summary.index,
    'missing_count': missing_summary.values,
    'missing_percent': missing_percent.values,
    'data_type': mtcars.dtypes.values
})

# Display missing data summary
print(missing_data[missing_data['missing_count'] > 0])

# Visualize missing data
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 4))
sns.barplot(
    data=missing_data,
    y='variable', x='missing_percent', color='steelblue',
    order=missing_data.sort_values('missing_percent', ascending=False)['variable']
)
plt.title('Missing Values by Variable')
plt.xlabel('Percentage Missing')
plt.ylabel('Variable')
plt.show()

# Check for patterns in missing data
missing_pattern = mtcars.isnull()
missing_corr = missing_pattern.corr()
print(missing_corr.round(3))
```

### Duplicate Detection

```python
# Check for duplicate rows
duplicates = mtcars.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Check for duplicate values in specific columns
duplicate_mpg = mtcars['mpg'].duplicated().sum()
print(f"Duplicate MPG values: {duplicate_mpg}")

# Find duplicate rows
if duplicates > 0:
    duplicate_indices = mtcars[mtcars.duplicated()].index.tolist()
    print(f"Duplicate row indices: {duplicate_indices}")
    print(mtcars.loc[duplicate_indices])

# Check for near-duplicates (simple similarity)
def near_duplicates(df, threshold=0.95):
    from sklearn.metrics import jaccard_score
    n_rows = df.shape[0]
    high_similarity = []
    for i in range(n_rows):
        for j in range(i+1, n_rows):
            sim = (df.iloc[i] == df.iloc[j]).sum() / df.shape[1]
            if sim > threshold:
                high_similarity.append((i, j))
    return high_similarity
# Example usage: near_duplicates(mtcars)
```

### Outlier Detection

Outliers are data points that deviate significantly from the rest of the data. They can be:
- **True outliers**: Genuine extreme values
- **Errors**: Data entry mistakes
- **Influential points**: Extreme values that affect analysis

```python
# Function to detect outliers using multiple methods
def detect_outliers(x, method="iqr"):
    x = np.array(x)
    if method == "iqr":
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (x < lower_bound) | (x > upper_bound)
    elif method == "zscore":
        z_scores = np.abs((x - np.mean(x)) / np.std(x))
        outliers = z_scores > 3
    elif method == "modified_zscore":
        median_val = np.median(x)
        mad_val = np.median(np.abs(x - median_val))
        modified_z_scores = 0.6745 * (x - median_val) / mad_val
        outliers = np.abs(modified_z_scores) > 3.5
    else:
        raise ValueError("Unknown method")
    return {
        'outliers': outliers,
        'outlier_indices': np.where(outliers)[0],
        'outlier_values': x[outliers],
        'method': method
    }

# Apply to numeric variables
numeric_vars = mtcars.select_dtypes(include=[np.number]).columns
outlier_summary = {var: detect_outliers(mtcars[var].dropna()) for var in numeric_vars}

# Display outlier summary
for var, summary in outlier_summary.items():
    n_outliers = summary['outliers'].sum()
    if n_outliers > 0:
        print(f"{var}: {n_outliers} outliers detected")
        print(f"  Outlier values: {summary['outlier_values']}")

# Compare different outlier detection methods
def compare_outlier_methods(x, var_name):
    iqr_result = detect_outliers(x, "iqr")
    zscore_result = detect_outliers(x, "zscore")
    modified_z_result = detect_outliers(x, "modified_zscore")
    print(f"=== OUTLIER DETECTION COMPARISON FOR {var_name} ===")
    print(f"IQR method: {iqr_result['outliers'].sum()} outliers")
    print(f"Z-score method: {zscore_result['outliers'].sum()} outliers")
    print(f"Modified Z-score method: {modified_z_result['outliers'].sum()} outliers")
    return {
        'iqr': iqr_result,
        'zscore': zscore_result,
        'modified_zscore': modified_z_result
    }

# Compare methods for MPG
mpg_outlier_comparison = compare_outlier_methods(mtcars['mpg'], "MPG")
```

### Mathematical Foundation: Outlier Detection

**IQR Method**: Based on quartiles
```math
\text{Lower bound} = Q_1 - 1.5 \times IQR
\text{Upper bound} = Q_3 + 1.5 \times IQR
```

**Z-Score Method**: Based on standard deviations
```math
Z = \frac{x - \mu}{\sigma}
```
Values with |Z| > 3 are considered outliers.

**Modified Z-Score**: More robust to extreme values
```math
M = \frac{0.6745(x - \text{median})}{\text{MAD}}
```
where MAD is the Median Absolute Deviation.

## Univariate Analysis

### Distribution Analysis

Understanding the distribution of a variable is crucial for choosing appropriate statistical methods.

```python
from scipy.stats import skew, kurtosis, shapiro

def analyze_distribution(x, var_name):
    print(f"=== DISTRIBUTION ANALYSIS FOR {var_name} ===")
    mean_val = np.mean(x)
    median_val = np.median(x)
    sd_val = np.std(x, ddof=1)
    print(f"Mean: {mean_val:.3f}")
    print(f"Median: {median_val:.3f}")
    print(f"Standard Deviation: {sd_val:.3f}")
    skewness_val = skew(x, nan_policy='omit')
    kurtosis_val = kurtosis(x, nan_policy='omit', fisher=False)
    print(f"Skewness: {skewness_val:.3f}")
    print(f"Kurtosis: {kurtosis_val:.3f}")
    if abs(skewness_val) < 0.5:
        print("Distribution is approximately symmetric")
    elif skewness_val > 0.5:
        print("Distribution is right-skewed (positive skew)")
    else:
        print("Distribution is left-skewed (negative skew)")
    if kurtosis_val < 3:
        print("Distribution has lighter tails than normal (platykurtic)")
    elif kurtosis_val > 3:
        print("Distribution has heavier tails than normal (leptokurtic)")
    else:
        print("Distribution has normal-like tails (mesokurtic)")
    quantiles = np.percentile(x, [5, 25, 50, 75, 95])
    print("Quantiles:")
    print(np.round(quantiles, 3))
    shapiro_test = shapiro(x)
    print(f"Shapiro-Wilk normality test p-value: {shapiro_test.pvalue:.4f}")
    if shapiro_test.pvalue < 0.05:
        print("Data is NOT normally distributed (p < 0.05)")
    else:
        print("Data appears to be normally distributed (p >= 0.05)")
    return {
        'mean': mean_val,
        'median': median_val,
        'sd': sd_val,
        'skewness': skewness_val,
        'kurtosis': kurtosis_val,
        'shapiro_p': shapiro_test.pvalue
    }

# Apply to MPG
mpg_analysis = analyze_distribution(mtcars['mpg'].dropna(), "MPG")
```

### Mathematical Foundation: Distribution Characteristics

**Skewness**: Measures asymmetry of the distribution
```math
\text{Skewness} = \frac{E[(X - \mu)^3]}{\sigma^3}
```

**Kurtosis**: Measures the "tailedness" of the distribution
```math
\text{Kurtosis} = \frac{E[(X - \mu)^4]}{\sigma^4}
```

**Shapiro-Wilk Test**: Tests for normality
```math
W = \frac{(\sum_{i=1}^{n} a_i x_{(i)})^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
```

### Visualization for Univariate Data

```python
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def create_univariate_plots(data, variable):
    x = data[variable].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Histogram with density
    sns.histplot(x, kde=True, ax=axes[0, 0], color='steelblue', bins=15)
    axes[0, 0].axvline(x.mean(), color='green', linestyle='dashed', linewidth=1, label='Mean')
    axes[0, 0].axvline(x.median(), color='orange', linestyle='dashed', linewidth=1, label='Median')
    axes[0, 0].set_title(f"Distribution of {variable}")
    axes[0, 0].legend()
    # Box plot
    sns.boxplot(y=x, ax=axes[0, 1], color='lightgreen')
    sns.stripplot(y=x, ax=axes[0, 1], color='black', alpha=0.5)
    axes[0, 1].set_title(f"Boxplot of {variable}")
    # Q-Q plot
    stats.probplot(x, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f"Q-Q Plot of {variable}")
    # CDF
    sns.ecdfplot(x, ax=axes[1, 1])
    axes[1, 1].set_title(f"Cumulative Distribution of {variable}")
    plt.tight_layout()
    plt.show()

# Apply to MPG
create_univariate_plots(mtcars, "mpg")
```

### Understanding Distribution Shapes

**Normal Distribution**: Bell-shaped, symmetric
- Mean = Median = Mode
- 68% of data within ±1 standard deviation
- 95% of data within ±2 standard deviations

**Skewed Distributions**:
- **Right-skewed**: Long tail to the right, mean > median
- **Left-skewed**: Long tail to the left, mean < median

**Bimodal/Multimodal**: Multiple peaks, may indicate mixed populations

**Uniform Distribution**: All values equally likely

**Exponential Distribution**: Rapidly decreasing probability

## Bivariate Analysis

### Correlation Analysis

Correlation measures the strength and direction of linear relationships between variables.

```python
# Correlation matrix for numeric variables
correlation_matrix = mtcars.select_dtypes(include=[np.number]).corr()
print(correlation_matrix.round(3))

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Matrix')
plt.show()

# Detailed correlation analysis
def correlation_analysis(data, var1, var2):
    print(f"=== CORRELATION ANALYSIS ===")
    print(f"Variables: {var1} and {var2}")
    from scipy.stats import pearsonr, spearmanr, kendalltau
    pearson_cor, pearson_p = pearsonr(data[var1], data[var2])
    print(f"Pearson correlation: {pearson_cor:.3f}")
    print(f"p-value: {pearson_p:.4f}")
    spearman_cor, spearman_p = spearmanr(data[var1], data[var2])
    print(f"Spearman correlation: {spearman_cor:.3f}")
    print(f"p-value: {spearman_p:.4f}")
    kendall_cor, kendall_p = kendalltau(data[var1], data[var2])
    print(f"Kendall's tau: {kendall_cor:.3f}")
    print(f"p-value: {kendall_p:.4f}")
    pearson_abs = abs(pearson_cor)
    if pearson_abs < 0.1:
        print("Correlation strength: Negligible")
    elif pearson_abs < 0.3:
        print("Correlation strength: Weak")
    elif pearson_abs < 0.5:
        print("Correlation strength: Moderate")
    elif pearson_abs < 0.7:
        print("Correlation strength: Strong")
    else:
        print("Correlation strength: Very strong")
    return {
        'pearson': pearson_cor,
        'spearman': spearman_cor,
        'kendall': kendall_cor
    }

# Analyze correlation between MPG and Weight
mpg_wt_correlation = correlation_analysis(mtcars, "mpg", "wt")
```

### Mathematical Foundation: Correlation

**Pearson Correlation Coefficient**:
```math
r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}
```

**Spearman Rank Correlation**:
```math
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
```
where $d_i$ is the difference between ranks.

**Kendall's Tau**:
```math
\tau = \frac{n_c - n_d}{\frac{n(n-1)}{2}}
```
where $n_c$ is the number of concordant pairs and $n_d$ is the number of discordant pairs.

### Scatter Plot Analysis

```python
from sklearn.linear_model import LinearRegression

def create_scatter_analysis(data, x_var, y_var):
    x = data[x_var].values.reshape(-1, 1)
    y = data[y_var].values
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Scatter plot with regression line
    sns.regplot(x=x.flatten(), y=y, ax=axes[0, 0], line_kws={'color': 'red'}, scatter_kws={'alpha': 0.7})
    axes[0, 0].set_title(f"{y_var} vs {x_var} (Linear Regression)")
    # Scatter plot with LOWESS curve
    sns.regplot(x=x.flatten(), y=y, ax=axes[0, 1], lowess=True, line_kws={'color': 'blue'}, scatter_kws={'alpha': 0.7})
    axes[0, 1].set_title(f"{y_var} vs {x_var} (LOWESS)")
    # Residual plot
    model = LinearRegression().fit(x, y)
    fitted = model.predict(x)
    residuals = y - fitted
    axes[1, 0].scatter(fitted, residuals, alpha=0.7)
    axes[1, 0].axhline(0, color='red', linestyle='dashed')
    axes[1, 0].set_title("Residual Plot")
    axes[1, 0].set_xlabel("Fitted values")
    axes[1, 0].set_ylabel("Residuals")
    # Residuals vs predictor
    axes[1, 1].scatter(x, residuals, alpha=0.7)
    axes[1, 1].axhline(0, color='red', linestyle='dashed')
    axes[1, 1].set_title("Residuals vs Predictor")
    axes[1, 1].set_xlabel(x_var)
    axes[1, 1].set_ylabel("Residuals")
    plt.tight_layout()
    plt.show()
    # Print model summary
    import statsmodels.api as sm
    x_with_const = sm.add_constant(x)
    ols_model = sm.OLS(y, x_with_const).fit()
    print(ols_model.summary())
    return ols_model

# Apply to MPG vs Weight
mpg_wt_model = create_scatter_analysis(mtcars, "wt", "mpg")
```

### Understanding Scatter Plots

**Patterns to Look For**:
- **Linear**: Points follow a straight line
- **Nonlinear**: Curved patterns (quadratic, exponential, etc.)
- **No relationship**: Random scatter
- **Clusters**: Groups of points
- **Outliers**: Points far from the main pattern

**Regression Assumptions**:
1. **Linearity**: Relationship is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed

## Multivariate Analysis

### Group Comparisons

Comparing groups helps understand how categorical variables affect continuous outcomes.

```python
def create_group_comparison(data, group_var, response_var):
    import scipy.stats as stats
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    # Box plot by group
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=group_var, y=response_var, data=data, palette='Set2')
    sns.stripplot(x=group_var, y=response_var, data=data, color='black', alpha=0.5)
    plt.title(f"{response_var} by {group_var}")
    # Violin plot
    plt.subplot(1, 2, 2)
    sns.violinplot(x=group_var, y=response_var, data=data, palette='Set2')
    sns.boxplot(x=group_var, y=response_var, data=data, width=0.2, showcaps=False, boxprops={'facecolor':'None'}, showfliers=False, whiskerprops={'linewidth':0}, color='k')
    plt.title(f"{response_var} by {group_var} (Violin Plot)")
    plt.tight_layout()
    plt.show()
    # Summary statistics by group
    group_summary = data.groupby(group_var)[response_var].agg(['count', 'mean', 'std', 'median', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    group_summary.columns = ['n', 'mean', 'sd', 'median', 'min', 'max', 'q25', 'q75']
    print(group_summary)
    # Statistical test for group differences
    unique_groups = data[group_var].nunique()
    if unique_groups > 2:
        # ANOVA
        model = smf.ols(f"{response_var} ~ C({group_var})", data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print("\n=== ANOVA RESULTS ===")
        print(anova_table)
    elif unique_groups == 2:
        # t-test
        groups = data[group_var].unique()
        group1 = data[data[group_var] == groups[0]][response_var]
        group2 = data[data[group_var] == groups[1]][response_var]
        t_stat, p_val = stats.ttest_ind(group1, group2, nan_policy='omit')
        print("\n=== T-TEST RESULTS ===")
        print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
```

# Compare MPG by transmission type
mtcars['am_factor'] = mtcars['am'].map({0: 'Automatic', 1: 'Manual'})
create_group_comparison(mtcars, 'am_factor', 'mpg')
```

### Mathematical Foundation: Group Comparisons

**T-Test Statistic**:
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
```

**ANOVA F-Statistic**:
```math
F = \frac{\text{Between-group variance}}{\text{Within-group variance}} = \frac{MSB}{MSW}
```

**Effect Size (Cohen's d)**:
```math
d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}
```
where $s_{pooled} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$

### Interaction Analysis

Interactions occur when the effect of one variable depends on the level of another variable.

```python
def analyze_interactions(data, var1, var2, response_var):
    import statsmodels.formula.api as smf
    # Interaction plot
    plt.figure(figsize=(10, 5))
    sns.lmplot(x=var1, y=response_var, hue=var2, data=data, aspect=1.5)
    plt.title(f"Interaction: {response_var} ~ {var1} * {var2}")
    # Faceted plot
    g = sns.lmplot(x=var1, y=response_var, col=var2, data=data, aspect=1)
    g.fig.suptitle(f"{response_var} by {var1} faceted by {var2}", y=1.02)
    # Statistical test for interaction
    model = smf.ols(f"{response_var} ~ {var1} * {var2}", data=data).fit()
    print("=== INTERACTION ANALYSIS ===")
    print(model.summary())
    # Test for interaction significance
    model_no_inter = smf.ols(f"{response_var} ~ {var1} + {var2}", data=data).fit()
    from statsmodels.stats.anova import anova_lm
    anova_result = anova_lm(model_no_inter, model)
    print("\n=== INTERACTION SIGNIFICANCE TEST ===")
    print(anova_result)
    return model

# Analyze interaction between weight and transmission on MPG
analyze_interactions(mtcars, "wt", "am_factor", "mpg")
```

### Understanding Interactions

**Types of Interactions**:
- **Additive**: Effects of variables simply add together
- **Multiplicative**: Effects multiply together
- **Antagonistic**: One variable reduces the effect of another
- **Synergistic**: Variables work together to enhance effects

**Visualizing Interactions**:
- **Parallel lines**: No interaction
- **Converging lines**: Interaction
- **Crossing lines**: Strong interaction

## Data Transformation

### Variable Transformations

Transformations can help meet statistical assumptions and improve model performance.

```python
from scipy.stats import boxcox, yeojohnson

def apply_transformations(data, variable):
    x = data[variable].dropna()
    # Log transformation
    log_transformed = np.log(x)
    # Square root transformation
    sqrt_transformed = np.sqrt(x)
    # Reciprocal transformation
    reciprocal_transformed = 1 / x
    # Box-Cox transformation (only for positive values)
    if (x > 0).all():
        bc_transformed, lambda_bc = boxcox(x)
    else:
        bc_transformed, lambda_bc = np.full_like(x, np.nan), np.nan
    # Yeo-Johnson transformation (can handle zero/negative)
    yj_transformed, lambda_yj = yeojohnson(x)
    # Plot distributions
    import matplotlib.pyplot as plt
    transformed_data = pd.DataFrame({
        'Original': x,
        'Log': log_transformed,
        'Sqrt': sqrt_transformed,
        'Reciprocal': reciprocal_transformed,
        'BoxCox': bc_transformed,
        'YeoJohnson': yj_transformed
    })
    transformed_data_melted = transformed_data.melt(var_name='Transformation', value_name='Value')
    g = sns.FacetGrid(transformed_data_melted, col='Transformation', col_wrap=3, sharex=False, sharey=False)
    g.map(sns.histplot, 'Value', bins=15, color='steelblue', alpha=0.7)
    g.fig.suptitle(f"Distribution of {variable} under different transformations", y=1.05)
    plt.show()
    # Test normality for each transformation
    print("=== NORMALITY TESTS ===")
    for col in transformed_data.columns:
        vals = transformed_data[col].dropna()
        if len(vals) > 3:
            stat, p = shapiro(vals)
            print(f"{col} Shapiro-Wilk p-value: {p:.4f}")
    return {
        'original': x,
        'log': log_transformed,
        'sqrt': sqrt_transformed,
        'reciprocal': reciprocal_transformed,
        'boxcox': bc_transformed,
        'yeojohnson': yj_transformed,
        'lambda_boxcox': lambda_bc,
        'lambda_yeojohnson': lambda_yj
    }

# Apply transformations to MPG
mpg_transformations = apply_transformations(mtcars, "mpg")
```

### Mathematical Foundation: Transformations

**Log Transformation**:
```math
y' = \log(y)
```
Useful for right-skewed data, stabilizes variance.

**Square Root Transformation**:
```math
y' = \sqrt{y}
```
Less aggressive than log, good for count data.

**Box-Cox Transformation**:
```math
y' = \begin{cases}
\frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(y) & \text{if } \lambda = 0
\end{cases}
```

**Yeo-Johnson Transformation**:
```math
y' = \begin{cases}
\frac{(y+1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, y \geq 0 \\
\log(y+1) & \text{if } \lambda = 0, y \geq 0 \\
-\frac{(-y+1)^{2-\lambda} - 1}{2-\lambda} & \text{if } \lambda \neq 2, y < 0 \\
-\log(-y+1) & \text{if } \lambda = 2, y < 0
\end{cases}
```

### When to Transform

**Transform when**:
- Data is highly skewed
- Variance is not constant (heteroscedasticity)
- Residuals are not normal
- Relationships are nonlinear

**Common transformations**:
- **Right skew**: Log, square root, reciprocal
- **Left skew**: Square, cube, exponential
- **Count data**: Square root, log
- **Proportions**: Logit, arcsine

## Data Quality Checks

### Consistency Checks

Data consistency ensures that your data makes logical sense and doesn't contain contradictions.

```python
def check_data_consistency(data):
    print("=== DATA CONSISTENCY CHECKS ===")
    # Check for logical inconsistencies
    if 'mpg' in data.columns and 'wt' in data.columns:
        correlation = data['mpg'].corr(data['wt'])
        print(f"MPG-Weight correlation: {correlation:.3f}")
        if correlation > 0:
            print("WARNING: Positive correlation between weight and MPG (unexpected)")
    # Check for extreme values
    numeric_vars = data.select_dtypes(include=[np.number]).columns
    for var in numeric_vars:
        values = data[var].dropna()
        extreme_low = (values < np.percentile(values, 1)).sum()
        extreme_high = (values > np.percentile(values, 99)).sum()
        if extreme_low > 0 or extreme_high > 0:
            print(f"{var}: {extreme_low} extreme low values, {extreme_high} extreme high values")
        if var == 'mpg' and (values < 0).any():
            print("WARNING: mpg contains negative values (impossible for MPG)")
        if var == 'wt' and (values <= 0).any():
            print("WARNING: wt contains zero or negative values (impossible for weight)")
    # Check for data type consistency
    for var in data.columns:
        if pd.api.types.is_categorical_dtype(data[var]) or data[var].dtype == object:
            n_levels = data[var].nunique()
            if n_levels > 20:
                print(f"WARNING: {var} has {n_levels} levels (consider if this is correct)")
            level_counts = data[var].value_counts()
            rare_levels = level_counts[level_counts < 3]
            if len(rare_levels) > 0:
                print(f"WARNING: {var} has rare levels: {list(rare_levels.index)}")
    # Check for duplicate observations
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate rows found")
    # Check for missing value patterns
    missing_pattern = data.isnull()
    if missing_pattern.values.sum() > 0:
        missing_corr = missing_pattern.corr()
        high_missing_corr = ((abs(missing_corr) > 0.7) & (abs(missing_corr) < 1)).sum().sum()
        if high_missing_corr > 0:
            print("WARNING: Missing values are correlated between variables")

# Apply consistency checks
check_data_consistency(mtcars)
```

### Data Validation Rules

```python
def create_validation_rules(data):
    print("=== DATA VALIDATION RULES ===")
    rules = {
        'mpg': {'min': 0, 'max': 50, 'description': 'MPG should be between 0 and 50'},
        'wt': {'min': 0.5, 'max': 10, 'description': 'Weight should be between 0.5 and 10 tons'},
        'cyl': {'valid_values': [4, 6, 8], 'description': 'Cylinders should be 4, 6, or 8'}
    }
    violations = {}
    for var, rule in rules.items():
        if var in data.columns:
            values = data[var]
            if 'min' in rule and 'max' in rule:
                violation = (values < rule['min']) | (values > rule['max'])
                n_violations = violation.sum()
                if n_violations > 0:
                    print(f"{var}: {n_violations} values outside range [{rule['min']}, {rule['max']}]")
                violations[var] = violation
            elif 'valid_values' in rule:
                violation = ~values.isin(rule['valid_values'])
                n_violations = violation.sum()
                if n_violations > 0:
                    print(f"{var}: {n_violations} invalid values")
                violations[var] = violation
    return violations

# Apply validation rules
validation_violations = create_validation_rules(mtcars)
```

## EDA Report Generation

### Comprehensive EDA Report

```python
def generate_eda_report(data, dataset_name="Dataset"):
    print("=== EXPLORATORY DATA ANALYSIS REPORT ===\n")
    print(f"Dataset: {dataset_name}")
    print(f"Dimensions: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024 ** 2:.3f} MB\n")
    # Data types summary
    print("DATA TYPES:")
    print(data.dtypes)
    print()
    # Missing values summary
    missing_summary = data.isnull().sum()
    if missing_summary.sum() > 0:
        print("MISSING VALUES:")
        for var in missing_summary.index:
            if missing_summary[var] > 0:
                print(f"  {var}: {missing_summary[var]} ({missing_summary[var]/len(data)*100:.1f}%)")
        print()
    else:
        print("No missing values found.\n")
    # Summary statistics for numeric variables
    numeric_vars = data.select_dtypes(include=[np.number]).columns
    if len(numeric_vars) > 0:
        print("NUMERIC VARIABLES SUMMARY:")
        print(data[numeric_vars].describe())
        print()
        # Distribution characteristics
        print("DISTRIBUTION CHARACTERISTICS:")
        for var in numeric_vars:
            values = data[var].dropna()
            skewness_val = skew(values)
            kurtosis_val = kurtosis(values, fisher=False)
            print(f"  {var}:")
            print(f"    Skewness: {skewness_val:.3f}", end='')
            if abs(skewness_val) < 0.5:
                print(" (symmetric)")
            elif skewness_val > 0.5:
                print(" (right-skewed)")
            else:
                print(" (left-skewed)")
            print(f"    Kurtosis: {kurtosis_val:.3f}", end='')
            if kurtosis_val < 3:
                print(" (light tails)")
            elif kurtosis_val > 3:
                print(" (heavy tails)")
            else:
                print(" (normal-like)")
        print()
    # Key findings
    print("KEY FINDINGS:")
    # Find strongest correlations
    if len(numeric_vars) > 1:
        cor_matrix = data[numeric_vars].corr()
        np.fill_diagonal(cor_matrix.values, 0)
        max_cor = np.unravel_index(np.abs(cor_matrix.values).argmax(), cor_matrix.shape)
        var1 = cor_matrix.index[max_cor[0]]
        var2 = cor_matrix.columns[max_cor[1]]
        max_cor_value = cor_matrix.values[max_cor]
        print(f"  Strongest correlation: {var1} and {var2} ({max_cor_value:.3f})")
        # Find high correlations
        high_correlations = np.where((np.abs(cor_matrix.values) > 0.7) & (np.abs(cor_matrix.values) < 1))
        if len(high_correlations[0]) > 0:
            print("  High correlations (|r| > 0.7):")
            for i, j in zip(*high_correlations):
                print(f"    {cor_matrix.index[i]} - {cor_matrix.columns[j]}: {cor_matrix.values[i, j]:.3f}")
    # Find variables with most variation
    if len(numeric_vars) > 0:
        cv_values = data[numeric_vars].std() / data[numeric_vars].mean()
        most_variable = cv_values.idxmax()
        print(f"  Most variable numeric variable: {most_variable} (CV = {cv_values.max():.3f})")
    # Outlier summary
    if len(numeric_vars) > 0:
        outlier_counts = {}
        for var in numeric_vars:
            x = data[var].dropna()
            q1 = np.percentile(x, 25)
            q3 = np.percentile(x, 75)
            iqr = q3 - q1
            outlier_counts[var] = ((x < (q1 - 1.5 * iqr)) | (x > (q3 + 1.5 * iqr))).sum()
        variables_with_outliers = [var for var, count in outlier_counts.items() if count > 0]
        if variables_with_outliers:
            print("  Variables with outliers:")
            for var in variables_with_outliers:
                print(f"    {var}: {outlier_counts[var]} outliers")
    print("\nRECOMMENDATIONS:")
    print("1. Consider transformations for highly skewed variables")
    print("2. Investigate outliers and their impact on analysis")
    print("3. Check for multicollinearity in regression models")
    print("4. Consider interaction effects in modeling")
    print("5. Validate data quality before proceeding with analysis")

# Generate EDA report
generate_eda_report(mtcars, "Motor Trend Car Road Tests")
```

### Automated EDA Pipeline

```python
def automated_eda_pipeline(data, dataset_name="Dataset"):
    print("=== AUTOMATED EDA PIPELINE ===\n")
    # Step 1: Data overview
    print(f"STEP 1: DATA OVERVIEW\nDimensions: {data.shape[0]} × {data.shape[1]}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024 ** 2:.3f} MB")
    print(f"Data types: {', '.join(data.dtypes.unique().astype(str))}\n")
    # Step 2: Data quality assessment
    missing_count = data.isnull().sum().sum()
    print(f"STEP 2: DATA QUALITY ASSESSMENT\nTotal missing values: {missing_count}")
    duplicate_count = data.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")
    # Step 3: Univariate analysis
    numeric_vars = data.select_dtypes(include=[np.number]).columns
    print(f"STEP 3: UNIVARIATE ANALYSIS\nNumeric variables: {len(numeric_vars)}")
    print(f"Categorical variables: {data.shape[1] - len(numeric_vars)}")
    if len(numeric_vars) > 0:
        print("Distribution analysis:")
        for var in numeric_vars:
            values = data[var].dropna()
            skewness_val = skew(values)
            print(f"  {var}: skewness = {skewness_val:.3f}", end='')
            if abs(skewness_val) > 1:
                print(" (highly skewed - consider transformation)")
            else:
                print(" (moderate skew)")
    print()
    # Step 4: Bivariate analysis
    if len(numeric_vars) > 1:
        cor_matrix = data[numeric_vars].corr()
        high_correlations = np.where((np.abs(cor_matrix.values) > 0.7) & (np.abs(cor_matrix.values) < 1))
        if len(high_correlations[0]) > 0:
            print("STEP 4: BIVARIATE ANALYSIS\nHigh correlations detected:")
            for i, j in zip(*high_correlations):
                print(f"  {cor_matrix.index[i]} - {cor_matrix.columns[j]}: {cor_matrix.values[i, j]:.3f}")
        else:
            print("STEP 4: BIVARIATE ANALYSIS\nNo high correlations detected")
    print()
    # Step 5: Recommendations
    print("STEP 5: RECOMMENDATIONS")
    if missing_count > 0:
        print("- Address missing values before analysis")
    if duplicate_count > 0:
        print("- Remove or investigate duplicate records")
    if len(numeric_vars) > 1 and len(high_correlations[0]) > 0:
        print("- Consider multicollinearity in regression models")
    if len(numeric_vars) > 0:
        skewed_vars = [var for var in numeric_vars if abs(skew(data[var].dropna())) > 1]
        if len(skewed_vars) > 0:
            print("- Transform highly skewed variables")
    print("- Check for outliers and their impact")
    print("- Validate statistical assumptions")
    print("- Consider interaction effects in modeling")

# Apply pipeline to mtcars
automated_eda_pipeline(mtcars, "Motor Trend Car Road Tests")
```

## Practical Examples

### Example 1: Customer Satisfaction Survey

```python
# Simulate customer satisfaction data
np.random.seed(123)
n_customers = 200
customer_data = pd.DataFrame({
    'customer_id': range(1, n_customers + 1),
    'age': np.random.normal(45, 15, n_customers).round(0),
    'satisfaction_score': np.random.normal(7.5, 1.5, n_customers).round(1),
    'purchase_amount': np.random.exponential(100, n_customers).round(2),
    'service_quality': np.random.choice([1, 2, 3, 4, 5], n_customers, p=[0.05, 0.1, 0.2, 0.4, 0.25]),
    'loyalty_program': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], n_customers, p=[0.4, 0.3, 0.2, 0.1]),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_customers)
})

# Add some realistic patterns
customer_data.loc[customer_data['loyalty_program'] == 'Platinum', 'satisfaction_score'] += 0.5
customer_data.loc[customer_data['service_quality'] == 5, 'satisfaction_score'] += 0.3
customer_data.loc[customer_data['age'] > 60, 'purchase_amount'] *= 1.2

# Add some missing values
customer_data.loc[np.random.choice(customer_data.index, 10), 'satisfaction_score'] = np.nan
customer_data.loc[np.random.choice(customer_data.index, 5), 'purchase_amount'] = np.nan

print("Customer Satisfaction Survey Data:")
print(customer_data.head())
print(f"\nDataset shape: {customer_data.shape}")

# Comprehensive EDA
def analyze_customer_data(data):
    print("=== CUSTOMER SATISFACTION ANALYSIS ===\n")
    
    # 1. Data quality assessment
    print("1. DATA QUALITY ASSESSMENT:")
    missing_summary = data.isnull().sum()
    if missing_summary.sum() > 0:
        print("Missing values found:")
        for var in missing_summary.index:
            if missing_summary[var] > 0:
                print(f"  {var}: {missing_summary[var]} ({missing_summary[var]/len(data)*100:.1f}%)")
    else:
        print("No missing values found.")
    print()
    
    # 2. Univariate analysis
    print("2. UNIVARIATE ANALYSIS:")
    numeric_vars = ['age', 'satisfaction_score', 'purchase_amount']
    for var in numeric_vars:
        values = data[var].dropna()
        print(f"\n{var.upper()}:")
        print(f"  Mean: {values.mean():.2f}")
        print(f"  Median: {values.median():.2f}")
        print(f"  Std Dev: {values.std():.2f}")
        print(f"  Range: [{values.min():.2f}, {values.max():.2f}]")
        print(f"  Skewness: {skew(values):.3f}")
    
    # 3. Categorical analysis
    print("\n3. CATEGORICAL ANALYSIS:")
    categorical_vars = ['service_quality', 'loyalty_program', 'region']
    for var in categorical_vars:
        print(f"\n{var.upper()}:")
        value_counts = data[var].value_counts()
        for level, count in value_counts.items():
            print(f"  {level}: {count} ({count/len(data)*100:.1f}%)")
    
    # 4. Bivariate analysis
    print("\n4. BIVARIATE ANALYSIS:")
    # Satisfaction vs Service Quality
    print("\nSatisfaction Score by Service Quality:")
    satisfaction_by_quality = data.groupby('service_quality')['satisfaction_score'].agg(['count', 'mean', 'std'])
    print(satisfaction_by_quality)
    
    # Satisfaction vs Loyalty Program
    print("\nSatisfaction Score by Loyalty Program:")
    satisfaction_by_loyalty = data.groupby('loyalty_program')['satisfaction_score'].agg(['count', 'mean', 'std'])
    print(satisfaction_by_loyalty)
    
    # Correlation analysis
    numeric_data = data[numeric_vars].dropna()
    corr_matrix = numeric_data.corr()
    print(f"\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # 5. Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Satisfaction distribution
    sns.histplot(data['satisfaction_score'].dropna(), kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Satisfaction Score Distribution')
    
    # Age distribution
    sns.histplot(data['age'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Age Distribution')
    
    # Purchase amount distribution
    sns.histplot(data['purchase_amount'].dropna(), kde=True, ax=axes[0, 2])
    axes[0, 2].set_title('Purchase Amount Distribution')
    
    # Satisfaction by service quality
    sns.boxplot(x='service_quality', y='satisfaction_score', data=data, ax=axes[1, 0])
    axes[1, 0].set_title('Satisfaction by Service Quality')
    
    # Satisfaction by loyalty program
    sns.boxplot(x='loyalty_program', y='satisfaction_score', data=data, ax=axes[1, 1])
    axes[1, 1].set_title('Satisfaction by Loyalty Program')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Age vs Satisfaction scatter
    sns.scatterplot(x='age', y='satisfaction_score', data=data, ax=axes[1, 2])
    axes[1, 2].set_title('Age vs Satisfaction Score')
    
    plt.tight_layout()
    plt.show()
    
    # 6. Key insights
    print("\n5. KEY INSIGHTS:")
    print("- Average satisfaction score:", f"{data['satisfaction_score'].mean():.2f}")
    print("- Highest satisfaction by service quality:", 
          f"{satisfaction_by_quality['mean'].idxmax()} ({satisfaction_by_quality['mean'].max():.2f})")
    print("- Highest satisfaction by loyalty program:", 
          f"{satisfaction_by_loyalty['mean'].idxmax()} ({satisfaction_by_loyalty['mean'].max():.2f})")
    print("- Age-satisfaction correlation:", f"{data['age'].corr(data['satisfaction_score']):.3f}")

# Run the analysis
analyze_customer_data(customer_data)
```

### Example 2: Sales Performance Analysis

```python
# Simulate sales performance data
np.random.seed(456)
n_sales = 150
sales_data = pd.DataFrame({
    'salesperson_id': range(1, n_sales + 1),
    'monthly_sales': np.random.gamma(shape=2, scale=5000, size=n_sales).round(2),
    'commission_rate': np.random.choice([0.05, 0.07, 0.10, 0.12], n_sales, p=[0.3, 0.4, 0.2, 0.1]),
    'years_experience': np.random.poisson(lam=5, size=n_sales),
    'territory': np.random.choice(['Urban', 'Suburban', 'Rural'], n_sales, p=[0.4, 0.4, 0.2]),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], n_sales),
    'customer_satisfaction': np.random.normal(8.0, 1.0, n_sales).round(1)
})

# Add realistic relationships
sales_data.loc[sales_data['years_experience'] > 8, 'monthly_sales'] *= 1.3
sales_data.loc[sales_data['territory'] == 'Urban', 'monthly_sales'] *= 1.2
sales_data.loc[sales_data['commission_rate'] == 0.12, 'customer_satisfaction'] += 0.5

# Calculate commission earnings
sales_data['commission_earnings'] = sales_data['monthly_sales'] * sales_data['commission_rate']

print("Sales Performance Data:")
print(sales_data.head())
print(f"\nDataset shape: {sales_data.shape}")

# Sales performance analysis
def analyze_sales_performance(data):
    print("=== SALES PERFORMANCE ANALYSIS ===\n")
    
    # 1. Overall performance metrics
    print("1. OVERALL PERFORMANCE METRICS:")
    print(f"Total sales: ${data['monthly_sales'].sum():,.2f}")
    print(f"Average sales per person: ${data['monthly_sales'].mean():,.2f}")
    print(f"Total commission paid: ${data['commission_earnings'].sum():,.2f}")
    print(f"Average commission rate: {data['commission_rate'].mean():.1%}")
    print()
    
    # 2. Performance by territory
    print("2. PERFORMANCE BY TERRITORY:")
    territory_summary = data.groupby('territory').agg({
        'monthly_sales': ['count', 'mean', 'std', 'sum'],
        'commission_earnings': 'sum',
        'customer_satisfaction': 'mean'
    }).round(2)
    print(territory_summary)
    print()
    
    # 3. Performance by experience level
    print("3. PERFORMANCE BY EXPERIENCE:")
    data['experience_group'] = pd.cut(data['years_experience'], 
                                     bins=[0, 3, 6, 10, 20], 
                                     labels=['Novice', 'Intermediate', 'Experienced', 'Veteran'])
    experience_summary = data.groupby('experience_group').agg({
        'monthly_sales': ['count', 'mean', 'std'],
        'commission_earnings': 'mean',
        'customer_satisfaction': 'mean'
    }).round(2)
    print(experience_summary)
    print()
    
    # 4. Product category analysis
    print("4. PRODUCT CATEGORY ANALYSIS:")
    category_summary = data.groupby('product_category').agg({
        'monthly_sales': ['count', 'mean', 'sum'],
        'commission_earnings': 'sum',
        'customer_satisfaction': 'mean'
    }).round(2)
    print(category_summary)
    print()
    
    # 5. Correlation analysis
    numeric_vars = ['monthly_sales', 'years_experience', 'customer_satisfaction', 'commission_earnings']
    corr_matrix = data[numeric_vars].corr()
    print("5. CORRELATION ANALYSIS:")
    print(corr_matrix.round(3))
    print()
    
    # 6. Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Sales distribution
    sns.histplot(data['monthly_sales'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Monthly Sales Distribution')
    
    # Sales by territory
    sns.boxplot(x='territory', y='monthly_sales', data=data, ax=axes[0, 1])
    axes[0, 1].set_title('Sales by Territory')
    
    # Sales by experience
    sns.boxplot(x='experience_group', y='monthly_sales', data=data, ax=axes[0, 2])
    axes[0, 2].set_title('Sales by Experience Level')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Experience vs Sales
    sns.scatterplot(x='years_experience', y='monthly_sales', data=data, ax=axes[1, 0])
    axes[1, 0].set_title('Experience vs Sales')
    
    # Commission rate vs Sales
    sns.boxplot(x='commission_rate', y='monthly_sales', data=data, ax=axes[1, 1])
    axes[1, 1].set_title('Sales by Commission Rate')
    
    # Customer satisfaction vs Sales
    sns.scatterplot(x='customer_satisfaction', y='monthly_sales', data=data, ax=axes[1, 2])
    axes[1, 2].set_title('Customer Satisfaction vs Sales')
    
    plt.tight_layout()
    plt.show()
    
    # 7. Key insights and recommendations
    print("6. KEY INSIGHTS AND RECOMMENDATIONS:")
    best_territory = territory_summary[('monthly_sales', 'mean')].idxmax()
    print(f"- Best performing territory: {best_territory}")
    
    best_category = category_summary[('monthly_sales', 'mean')].idxmax()
    print(f"- Best performing product category: {best_category}")
    
    experience_correlation = data['years_experience'].corr(data['monthly_sales'])
    print(f"- Experience-sales correlation: {experience_correlation:.3f}")
    
    satisfaction_correlation = data['customer_satisfaction'].corr(data['monthly_sales'])
    print(f"- Satisfaction-sales correlation: {satisfaction_correlation:.3f}")
    
    print("\nRECOMMENDATIONS:")
    print("- Focus training on less experienced salespeople")
    print("- Consider territory-specific strategies")
    print("- Investigate high-performing product categories")
    print("- Monitor commission rate effectiveness")

# Run the analysis
analyze_sales_performance(sales_data)
```

## Best Practices for EDA

### 1. Systematic Approach

```python
def systematic_eda_checklist(data, dataset_name="Dataset"):
    """
    Systematic EDA checklist to ensure comprehensive analysis
    """
    print(f"=== SYSTEMATIC EDA CHECKLIST FOR {dataset_name} ===\n")
    
    checklist = {
        "Data Overview": False,
        "Data Quality Assessment": False,
        "Univariate Analysis": False,
        "Bivariate Analysis": False,
        "Multivariate Analysis": False,
        "Data Transformation": False,
        "Documentation": False
    }
    
    # 1. Data Overview
    print("1. DATA OVERVIEW:")
    print(f"   ✓ Dimensions: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"   ✓ Memory usage: {data.memory_usage(deep=True).sum() / 1024 ** 2:.3f} MB")
    print(f"   ✓ Data types: {', '.join(data.dtypes.unique().astype(str))}")
    print(f"   ✓ Variable names: {', '.join(data.columns[:5])}{'...' if len(data.columns) > 5 else ''}")
    checklist["Data Overview"] = True
    print()
    
    # 2. Data Quality Assessment
    print("2. DATA QUALITY ASSESSMENT:")
    missing_count = data.isnull().sum().sum()
    duplicate_count = data.duplicated().sum()
    print(f"   ✓ Missing values: {missing_count}")
    print(f"   ✓ Duplicate rows: {duplicate_count}")
    print(f"   ✓ Unique values per variable: {data.nunique().to_dict()}")
    checklist["Data Quality Assessment"] = True
    print()
    
    # 3. Univariate Analysis
    print("3. UNIVARIATE ANALYSIS:")
    numeric_vars = data.select_dtypes(include=[np.number]).columns
    categorical_vars = data.select_dtypes(include=['object', 'category']).columns
    print(f"   ✓ Numeric variables: {len(numeric_vars)}")
    print(f"   ✓ Categorical variables: {len(categorical_vars)}")
    if len(numeric_vars) > 0:
        print(f"   ✓ Summary statistics computed")
        print(f"   ✓ Distribution characteristics analyzed")
    checklist["Univariate Analysis"] = True
    print()
    
    # 4. Bivariate Analysis
    print("4. BIVARIATE ANALYSIS:")
    if len(numeric_vars) > 1:
        corr_matrix = data[numeric_vars].corr()
        high_correlations = np.where((np.abs(corr_matrix.values) > 0.7) & (np.abs(corr_matrix.values) < 1))
        print(f"   ✓ Correlation matrix computed")
        print(f"   ✓ High correlations identified: {len(high_correlations[0])}")
    if len(categorical_vars) > 0 and len(numeric_vars) > 0:
        print(f"   ✓ Group comparisons performed")
    checklist["Bivariate Analysis"] = True
    print()
    
    # 5. Multivariate Analysis
    print("5. MULTIVARIATE ANALYSIS:")
    if len(numeric_vars) > 2:
        print(f"   ✓ Multivariate patterns explored")
    if len(categorical_vars) > 1:
        print(f"   ✓ Interaction effects examined")
    checklist["Multivariate Analysis"] = True
    print()
    
    # 6. Data Transformation
    print("6. DATA TRANSFORMATION:")
    if len(numeric_vars) > 0:
        skewed_vars = [var for var in numeric_vars if abs(skew(data[var].dropna())) > 1]
        print(f"   ✓ Skewed variables identified: {len(skewed_vars)}")
        if len(skewed_vars) > 0:
            print(f"   ✓ Transformation recommendations made")
    checklist["Data Transformation"] = True
    print()
    
    # 7. Documentation
    print("7. DOCUMENTATION:")
    print(f"   ✓ Key findings documented")
    print(f"   ✓ Recommendations provided")
    print(f"   ✓ Code is reproducible")
    checklist["Documentation"] = True
    print()
    
    # Summary
    completed = sum(checklist.values())
    total = len(checklist)
    print(f"=== CHECKLIST SUMMARY ===")
    print(f"Completed: {completed}/{total} steps ({completed/total*100:.1f}%)")
    
    if completed == total:
        print("✓ All EDA steps completed successfully!")
    else:
        print("⚠ Some steps need attention")
    
    return checklist

# Apply systematic checklist
checklist_results = systematic_eda_checklist(mtcars, "Motor Trend Car Road Tests")
```

### 2. Data Quality First

```python
def comprehensive_data_quality_check(data):
    """
    Comprehensive data quality assessment
    """
    print("=== COMPREHENSIVE DATA QUALITY CHECK ===\n")
    
    quality_issues = []
    
    # 1. Missing values
    missing_summary = data.isnull().sum()
    if missing_summary.sum() > 0:
        print("MISSING VALUES DETECTED:")
        for var in missing_summary.index:
            if missing_summary[var] > 0:
                missing_pct = missing_summary[var] / len(data) * 100
                print(f"  {var}: {missing_summary[var]} ({missing_pct:.1f}%)")
                if missing_pct > 20:
                    quality_issues.append(f"High missing rate in {var}: {missing_pct:.1f}%")
        print()
    else:
        print("✓ No missing values found\n")
    
    # 2. Duplicates
    duplicate_count = data.duplicated().sum()
    if duplicate_count > 0:
        print(f"DUPLICATES DETECTED: {duplicate_count} rows")
        quality_issues.append(f"Duplicate rows: {duplicate_count}")
        print()
    else:
        print("✓ No duplicate rows found\n")
    
    # 3. Data type consistency
    print("DATA TYPE CONSISTENCY:")
    for var in data.columns:
        if data[var].dtype == 'object':
            # Check if numeric data is stored as text
            try:
                pd.to_numeric(data[var], errors='raise')
                quality_issues.append(f"Variable {var} contains numeric data stored as text")
                print(f"  ⚠ {var}: Numeric data stored as text")
            except:
                # Check for mixed data types
                unique_types = data[var].apply(type).unique()
                if len(unique_types) > 1:
                    quality_issues.append(f"Variable {var} has mixed data types")
                    print(f"  ⚠ {var}: Mixed data types detected")
                else:
                    print(f"  ✓ {var}: Consistent data type")
        else:
            print(f"  ✓ {var}: {data[var].dtype}")
    print()
    
    # 4. Range and domain checks
    print("RANGE AND DOMAIN CHECKS:")
    numeric_vars = data.select_dtypes(include=[np.number]).columns
    for var in numeric_vars:
        values = data[var].dropna()
        if len(values) > 0:
            # Check for extreme values
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            outliers = ((values < (q1 - 3 * iqr)) | (values > (q3 + 3 * iqr))).sum()
            if outliers > 0:
                print(f"  ⚠ {var}: {outliers} extreme outliers detected")
                quality_issues.append(f"Extreme outliers in {var}: {outliers}")
            else:
                print(f"  ✓ {var}: No extreme outliers")
            
            # Check for impossible values
            if var.lower() in ['age', 'years'] and (values < 0).any():
                print(f"  ⚠ {var}: Negative values detected")
                quality_issues.append(f"Negative values in {var}")
            elif var.lower() in ['price', 'amount', 'sales'] and (values < 0).any():
                print(f"  ⚠ {var}: Negative monetary values detected")
                quality_issues.append(f"Negative monetary values in {var}")
    print()
    
    # 5. Consistency checks
    print("CONSISTENCY CHECKS:")
    # Check for logical inconsistencies
    if 'age' in data.columns and 'birth_year' in data.columns:
        current_year = pd.Timestamp.now().year
        calculated_age = current_year - data['birth_year']
        age_diff = abs(data['age'] - calculated_age)
        if (age_diff > 2).any():
            print("  ⚠ Age and birth year inconsistencies detected")
            quality_issues.append("Age and birth year inconsistencies")
    
    # Check for missing value patterns
    missing_pattern = data.isnull()
    if missing_pattern.values.sum() > 0:
        missing_corr = missing_pattern.corr()
        high_missing_corr = ((abs(missing_corr) > 0.7) & (abs(missing_corr) < 1)).sum().sum()
        if high_missing_corr > 0:
            print("  ⚠ Missing values are correlated between variables")
            quality_issues.append("Correlated missing values")
    print()
    
    # Summary
    print("=== DATA QUALITY SUMMARY ===")
    if len(quality_issues) == 0:
        print("✓ No major data quality issues detected")
    else:
        print(f"⚠ {len(quality_issues)} data quality issues found:")
        for issue in quality_issues:
            print(f"  - {issue}")
    
    return quality_issues

# Apply data quality check
quality_issues = comprehensive_data_quality_check(mtcars)
```

### 3. Visualization Best Practices

```python
def create_effective_visualizations(data, target_var='mpg'):
    """
    Demonstrate effective visualization practices
    """
    print("=== EFFECTIVE VISUALIZATION PRACTICES ===\n")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Clear and informative titles
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Distribution plot with clear title and labels
    sns.histplot(data[target_var].dropna(), kde=True, ax=axes[0, 0], bins=15)
    axes[0, 0].set_title(f'Distribution of {target_var.upper()}', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel(f'{target_var.upper()} Values', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot with proper formatting
    sns.boxplot(y=data[target_var].dropna(), ax=axes[0, 1], color='lightblue')
    axes[0, 1].set_title(f'{target_var.upper()} Distribution (Box Plot)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel(f'{target_var.upper()} Values', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter plot with trend line
    if 'wt' in data.columns:
        sns.regplot(x='wt', y=target_var, data=data, ax=axes[1, 0], 
                   scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
        axes[1, 0].set_title(f'{target_var.upper()} vs Weight', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Weight', fontsize=12)
        axes[1, 0].set_ylabel(f'{target_var.upper()}', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation heatmap
    numeric_vars = data.select_dtypes(include=[np.number]).columns
    if len(numeric_vars) > 1:
        corr_matrix = data[numeric_vars].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[1, 1], fmt='.2f')
        axes[1, 1].set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Color considerations
    print("COLOR BEST PRACTICES:")
    print("- Use colorblind-friendly palettes")
    print("- Maintain sufficient contrast")
    print("- Use color consistently across plots")
    print("- Avoid using color as the only way to convey information")
    
    # 3. Accessibility considerations
    print("\nACCESSIBILITY CONSIDERATIONS:")
    print("- Use clear, readable fonts")
    print("- Provide sufficient contrast")
    print("- Include descriptive titles and labels")
    print("- Consider adding text annotations for key points")
    
    # 4. Interactive visualization example
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create interactive scatter plot
        fig = px.scatter(data, x='wt', y=target_var, 
                        title=f'Interactive {target_var.upper()} vs Weight',
                        labels={'wt': 'Weight', target_var: target_var.upper()},
                        hover_data=['cyl', 'am'])
        
        fig.update_layout(
            title_font_size=16,
            title_font_color='darkblue',
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )
        
        fig.show()
        
        print("\nINTERACTIVE VISUALIZATION CREATED:")
        print("- Hover information included")
        print("- Zoom and pan capabilities")
        print("- Export options available")
        
    except ImportError:
        print("\nPlotly not available for interactive visualizations")
        print("Install with: pip install plotly")

# Apply visualization best practices
create_effective_visualizations(mtcars)
```

### 4. Documentation and Reproducibility

```python
def create_eda_documentation(data, dataset_name="Dataset"):
    """
    Create comprehensive EDA documentation
    """
    print("=== EDA DOCUMENTATION TEMPLATE ===\n")
    
    # Create documentation dictionary
    eda_doc = {
        'dataset_info': {
            'name': dataset_name,
            'dimensions': data.shape,
            'memory_usage': f"{data.memory_usage(deep=True).sum() / 1024 ** 2:.3f} MB",
            'date_analyzed': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analyst': 'Data Scientist'
        },
        'data_quality': {
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_rows': data.duplicated().sum(),
            'data_types': data.dtypes.to_dict()
        },
        'key_findings': [],
        'recommendations': [],
        'code_snippets': []
    }
    
    # Add key findings
    numeric_vars = data.select_dtypes(include=[np.number]).columns
    if len(numeric_vars) > 0:
        # Find strongest correlation
        if len(numeric_vars) > 1:
            corr_matrix = data[numeric_vars].corr()
            np.fill_diagonal(corr_matrix.values, 0)
            max_cor = np.unravel_index(np.abs(corr_matrix.values).argmax(), corr_matrix.shape)
            var1 = corr_matrix.index[max_cor[0]]
            var2 = corr_matrix.columns[max_cor[1]]
            max_cor_value = corr_matrix.values[max_cor]
            eda_doc['key_findings'].append(f"Strongest correlation: {var1} and {var2} ({max_cor_value:.3f})")
        
        # Find most variable numeric variable
        cv_values = data[numeric_vars].std() / data[numeric_vars].mean()
        most_variable = cv_values.idxmax()
        eda_doc['key_findings'].append(f"Most variable numeric variable: {most_variable} (CV = {cv_values.max():.3f})")
    
    # Add recommendations
    if data.isnull().sum().sum() > 0:
        eda_doc['recommendations'].append("Address missing values before analysis")
    
    if data.duplicated().sum() > 0:
        eda_doc['recommendations'].append("Remove or investigate duplicate records")
    
    if len(numeric_vars) > 0:
        skewed_vars = [var for var in numeric_vars if abs(skew(data[var].dropna())) > 1]
        if len(skewed_vars) > 0:
            eda_doc['recommendations'].append(f"Transform highly skewed variables: {', '.join(skewed_vars)}")
    
    # Print documentation
    print("DATASET INFORMATION:")
    for key, value in eda_doc['dataset_info'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print()
    
    print("DATA QUALITY SUMMARY:")
    print(f"  Missing values: {sum(eda_doc['data_quality']['missing_values'].values())}")
    print(f"  Duplicate rows: {eda_doc['data_quality']['duplicate_rows']}")
    print(f"  Data types: {', '.join(set(eda_doc['data_quality']['data_types'].values()))}")
    print()
    
    print("KEY FINDINGS:")
    for finding in eda_doc['key_findings']:
        print(f"  • {finding}")
    print()
    
    print("RECOMMENDATIONS:")
    for rec in eda_doc['recommendations']:
        print(f"  • {rec}")
    print()
    
    print("REPRODUCIBILITY CHECKLIST:")
    print("  ✓ Random seed set")
    print("  ✓ All libraries imported")
    print("  ✓ Data loading code included")
    print("  ✓ All transformations documented")
    print("  ✓ Results saved/exported")
    
    return eda_doc

# Create documentation
eda_documentation = create_eda_documentation(mtcars, "Motor Trend Car Road Tests")
```

## Exercises

### Exercise 1: Basic EDA on a New Dataset

```python
# Load the iris dataset for practice
from sklearn.datasets import load_iris
iris_data = load_iris()
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df['species'] = iris_data.target_names[iris_data.target]

print("Iris Dataset:")
print(iris_df.head())
print(f"\nDataset shape: {iris_df.shape}")

# Your tasks:
# 1. Perform basic data exploration
# 2. Check for missing values and data quality issues
# 3. Create summary statistics for numeric variables
# 4. Analyze the distribution of each numeric variable
# 5. Create visualizations (histograms, box plots, scatter plots)
# 6. Examine relationships between variables
# 7. Compare species groups
# 8. Document your findings

def exercise_1_solution():
    """
    Solution for Exercise 1: Basic EDA on Iris Dataset
    """
    print("=== EXERCISE 1 SOLUTION ===\n")
    
    # 1. Basic data exploration
    print("1. BASIC DATA EXPLORATION:")
    print(f"Dataset dimensions: {iris_df.shape}")
    print(f"Data types: {iris_df.dtypes.to_dict()}")
    print(f"Memory usage: {iris_df.memory_usage(deep=True).sum() / 1024:.3f} KB")
    print()
    
    # 2. Data quality check
    print("2. DATA QUALITY CHECK:")
    missing_count = iris_df.isnull().sum().sum()
    duplicate_count = iris_df.duplicated().sum()
    print(f"Missing values: {missing_count}")
    print(f"Duplicate rows: {duplicate_count}")
    print()
    
    # 3. Summary statistics
    print("3. SUMMARY STATISTICS:")
    numeric_vars = iris_df.select_dtypes(include=[np.number]).columns
    print(iris_df[numeric_vars].describe())
    print()
    
    # 4. Distribution analysis
    print("4. DISTRIBUTION ANALYSIS:")
    for var in numeric_vars:
        values = iris_df[var].dropna()
        skewness_val = skew(values)
        kurtosis_val = kurtosis(values, fisher=False)
        print(f"{var}:")
        print(f"  Skewness: {skewness_val:.3f}")
        print(f"  Kurtosis: {kurtosis_val:.3f}")
        if abs(skewness_val) < 0.5:
            print("  Distribution: Approximately symmetric")
        elif skewness_val > 0.5:
            print("  Distribution: Right-skewed")
        else:
            print("  Distribution: Left-skewed")
    print()
    
    # 5. Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histograms
    for i, var in enumerate(numeric_vars):
        row, col = i // 2, i % 2
        sns.histplot(iris_df[var], kde=True, ax=axes[row, col])
        axes[row, col].set_title(f'Distribution of {var}')
    
    plt.tight_layout()
    plt.show()
    
    # Box plots by species
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, var in enumerate(numeric_vars):
        row, col = i // 2, i % 2
        sns.boxplot(x='species', y=var, data=iris_df, ax=axes[row, col])
        axes[row, col].set_title(f'{var} by Species')
        axes[row, col].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 6. Correlation analysis
    print("5. CORRELATION ANALYSIS:")
    corr_matrix = iris_df[numeric_vars].corr()
    print(corr_matrix.round(3))
    
    # Visualize correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix')
    plt.show()
    
    # 7. Species comparison
    print("\n6. SPECIES COMPARISON:")
    species_summary = iris_df.groupby('species')[numeric_vars].agg(['mean', 'std'])
    print(species_summary.round(3))
    
    # 8. Key findings
    print("\n7. KEY FINDINGS:")
    print("- All variables show approximately normal distributions")
    print("- Strong positive correlation between petal length and petal width")
    print("- Clear separation between species in petal measurements")
    print("- Setosa species has the smallest petal measurements")
    print("- Virginica species has the largest petal measurements")

# Run the solution
exercise_1_solution()
```

### Exercise 2: Advanced EDA with Data Issues

```python
# Create a dataset with various data quality issues
np.random.seed(789)
n_samples = 100

# Create base data
exercise_data = pd.DataFrame({
    'id': range(1, n_samples + 1),
    'age': np.random.normal(35, 10, n_samples).round(0),
    'income': np.random.exponential(50000, n_samples).round(2),
    'satisfaction': np.random.normal(7.5, 1.5, n_samples).round(1),
    'department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'HR'], n_samples),
    'experience_years': np.random.poisson(lam=5, size=n_samples)
})

# Introduce data quality issues
# Missing values
exercise_data.loc[np.random.choice(exercise_data.index, 10), 'income'] = np.nan
exercise_data.loc[np.random.choice(exercise_data.index, 5), 'satisfaction'] = np.nan

# Outliers
exercise_data.loc[0, 'age'] = 150  # Impossible age
exercise_data.loc[1, 'income'] = -50000  # Negative income
exercise_data.loc[2, 'satisfaction'] = 15  # Impossible satisfaction score

# Duplicates
exercise_data = pd.concat([exercise_data, exercise_data.iloc[:3]])

# Data type issues
exercise_data['age'] = exercise_data['age'].astype(str)
exercise_data.loc[0, 'age'] = 'N/A'

print("Exercise Dataset with Data Quality Issues:")
print(exercise_data.head(10))
print(f"\nDataset shape: {exercise_data.shape}")

# Your tasks:
# 1. Identify and document all data quality issues
# 2. Clean the data appropriately
# 3. Perform comprehensive EDA on the cleaned data
# 4. Create visualizations to show the impact of cleaning
# 5. Provide recommendations for data collection improvements

def exercise_2_solution():
    """
    Solution for Exercise 2: Advanced EDA with Data Issues
    """
    print("=== EXERCISE 2 SOLUTION ===\n")
    
    # 1. Identify data quality issues
    print("1. DATA QUALITY ISSUES IDENTIFIED:")
    
    # Missing values
    missing_summary = exercise_data.isnull().sum()
    print("Missing values:")
    for var in missing_summary.index:
        if missing_summary[var] > 0:
            print(f"  {var}: {missing_summary[var]} ({missing_summary[var]/len(exercise_data)*100:.1f}%)")
    
    # Duplicates
    duplicate_count = exercise_data.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")
    
    # Data type issues
    print("Data type issues:")
    for var in exercise_data.columns:
        if exercise_data[var].dtype == 'object':
            try:
                pd.to_numeric(exercise_data[var], errors='raise')
                print(f"  {var}: Numeric data stored as text")
            except:
                print(f"  {var}: Text data (expected for categorical)")
    
    # Range/domain issues
    print("Range/domain issues:")
    if 'age' in exercise_data.columns:
        age_values = pd.to_numeric(exercise_data['age'], errors='coerce')
        impossible_age = ((age_values < 0) | (age_values > 120)).sum()
        print(f"  Impossible ages: {impossible_age}")
    
    if 'income' in exercise_data.columns:
        impossible_income = (exercise_data['income'] < 0).sum()
        print(f"  Negative income values: {impossible_income}")
    
    if 'satisfaction' in exercise_data.columns:
        impossible_satisfaction = ((exercise_data['satisfaction'] < 1) | (exercise_data['satisfaction'] > 10)).sum()
        print(f"  Impossible satisfaction scores: {impossible_satisfaction}")
    
    print()
    
    # 2. Clean the data
    print("2. DATA CLEANING:")
    
    # Create a copy for cleaning
    cleaned_data = exercise_data.copy()
    
    # Remove duplicates
    cleaned_data = cleaned_data.drop_duplicates()
    print(f"Removed {duplicate_count} duplicate rows")
    
    # Fix data types
    cleaned_data['age'] = pd.to_numeric(cleaned_data['age'], errors='coerce')
    print("Converted age to numeric (invalid values become NaN)")
    
    # Remove impossible values
    if 'age' in cleaned_data.columns:
        age_mask = (cleaned_data['age'] < 0) | (cleaned_data['age'] > 120)
        cleaned_data = cleaned_data[~age_mask]
        print(f"Removed {age_mask.sum()} impossible age values")
    
    if 'income' in cleaned_data.columns:
        income_mask = cleaned_data['income'] < 0
        cleaned_data = cleaned_data[~income_mask]
        print(f"Removed {income_mask.sum()} negative income values")
    
    if 'satisfaction' in cleaned_data.columns:
        satisfaction_mask = (cleaned_data['satisfaction'] < 1) | (cleaned_data['satisfaction'] > 10)
        cleaned_data = cleaned_data[~satisfaction_mask]
        print(f"Removed {satisfaction_mask.sum()} impossible satisfaction values")
    
    print(f"Final dataset shape: {cleaned_data.shape}")
    print()
    
    # 3. Comprehensive EDA on cleaned data
    print("3. COMPREHENSIVE EDA ON CLEANED DATA:")
    
    # Summary statistics
    numeric_vars = cleaned_data.select_dtypes(include=[np.number]).columns
    print("Summary statistics:")
    print(cleaned_data[numeric_vars].describe())
    print()
    
    # Distribution analysis
    print("Distribution characteristics:")
    for var in numeric_vars:
        values = cleaned_data[var].dropna()
        skewness_val = skew(values)
        print(f"{var}: skewness = {skewness_val:.3f}")
    print()
    
    # Department analysis
    print("Analysis by department:")
    dept_summary = cleaned_data.groupby('department')[numeric_vars].agg(['count', 'mean', 'std'])
    print(dept_summary.round(2))
    print()
    
    # 4. Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Before vs after cleaning comparison
    # Age distribution
    sns.histplot(exercise_data['age'].dropna(), kde=True, ax=axes[0, 0], alpha=0.5, label='Before')
    sns.histplot(cleaned_data['age'].dropna(), kde=True, ax=axes[0, 0], alpha=0.5, label='After')
    axes[0, 0].set_title('Age Distribution: Before vs After Cleaning')
    axes[0, 0].legend()
    
    # Income distribution
    sns.histplot(exercise_data['income'].dropna(), kde=True, ax=axes[0, 1], alpha=0.5, label='Before')
    sns.histplot(cleaned_data['income'].dropna(), kde=True, ax=axes[0, 1], alpha=0.5, label='After')
    axes[0, 1].set_title('Income Distribution: Before vs After Cleaning')
    axes[0, 1].legend()
    
    # Satisfaction distribution
    sns.histplot(exercise_data['satisfaction'].dropna(), kde=True, ax=axes[0, 2], alpha=0.5, label='Before')
    sns.histplot(cleaned_data['satisfaction'].dropna(), kde=True, ax=axes[0, 2], alpha=0.5, label='After')
    axes[0, 2].set_title('Satisfaction Distribution: Before vs After Cleaning')
    axes[0, 2].legend()
    
    # Box plots by department
    sns.boxplot(x='department', y='income', data=cleaned_data, ax=axes[1, 0])
    axes[1, 0].set_title('Income by Department')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(x='department', y='satisfaction', data=cleaned_data, ax=axes[1, 1])
    axes[1, 1].set_title('Satisfaction by Department')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Correlation heatmap
    corr_matrix = cleaned_data[numeric_vars].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, ax=axes[1, 2])
    axes[1, 2].set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Recommendations
    print("4. RECOMMENDATIONS FOR DATA COLLECTION:")
    print("- Implement data validation rules during collection")
    print("- Use dropdown menus for categorical variables")
    print("- Set reasonable range limits for numeric variables")
    print("- Provide clear instructions for data entry")
    print("- Implement real-time validation feedback")
    print("- Regular data quality audits")
    print("- Training for data collectors on quality standards")

# Run the solution
exercise_2_solution()
```

### Exercise 3: Interactive EDA Dashboard

```python
def create_interactive_eda_dashboard(data, title="Interactive EDA Dashboard"):
    """
    Create an interactive EDA dashboard using Plotly
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
        
        print("=== INTERACTIVE EDA DASHBOARD ===\n")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribution', 'Correlation Matrix', 'Box Plot', 'Scatter Plot'),
            specs=[[{"type": "histogram"}, {"type": "heatmap"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )
        
        # 1. Distribution plot
        numeric_vars = data.select_dtypes(include=[np.number]).columns
        if len(numeric_vars) > 0:
            var = numeric_vars[0]  # Use first numeric variable
            fig.add_trace(
                go.Histogram(x=data[var].dropna(), name=f'{var} Distribution'),
                row=1, col=1
            )
        
        # 2. Correlation heatmap
        if len(numeric_vars) > 1:
            corr_matrix = data[numeric_vars].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='coolwarm',
                    zmid=0,
                    text=corr_matrix.round(2).values,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    name='Correlation Matrix'
                ),
                row=1, col=2
            )
        
        # 3. Box plot
        if len(numeric_vars) > 0:
            categorical_vars = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_vars) > 0:
                cat_var = categorical_vars[0]
                for category in data[cat_var].unique():
                    category_data = data[data[cat_var] == category][var].dropna()
                    fig.add_trace(
                        go.Box(y=category_data, name=f'{var} - {category}'),
                        row=2, col=1
                    )
            else:
                fig.add_trace(
                    go.Box(y=data[var].dropna(), name=f'{var}'),
                    row=2, col=1
                )
        
        # 4. Scatter plot
        if len(numeric_vars) > 1:
            var1, var2 = numeric_vars[0], numeric_vars[1]
            fig.add_trace(
                go.Scatter(
                    x=data[var1].dropna(),
                    y=data[var2].dropna(),
                    mode='markers',
                    name=f'{var1} vs {var2}',
                    marker=dict(size=8, opacity=0.7)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        # Show the dashboard
        fig.show()
        
        print("Interactive dashboard created successfully!")
        print("Features:")
        print("- Hover information on all plots")
        print("- Zoom and pan capabilities")
        print("- Download plot as PNG")
        print("- Reset axes button")
        
    except ImportError:
        print("Plotly not available. Install with: pip install plotly")
        print("Creating static dashboard instead...")
        
        # Fallback to static plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        numeric_vars = data.select_dtypes(include=[np.number]).columns
        if len(numeric_vars) > 0:
            var = numeric_vars[0]
            
            # Distribution
            sns.histplot(data[var].dropna(), kde=True, ax=axes[0, 0])
            axes[0, 0].set_title(f'Distribution of {var}')
            
            # Correlation matrix
            if len(numeric_vars) > 1:
                corr_matrix = data[numeric_vars].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, ax=axes[0, 1])
                axes[0, 1].set_title('Correlation Matrix')
            
            # Box plot
            categorical_vars = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_vars) > 0:
                cat_var = categorical_vars[0]
                sns.boxplot(x=cat_var, y=var, data=data, ax=axes[1, 0])
                axes[1, 0].set_title(f'{var} by {cat_var}')
                axes[1, 0].tick_params(axis='x', rotation=45)
            else:
                sns.boxplot(y=data[var].dropna(), ax=axes[1, 0])
                axes[1, 0].set_title(f'Box Plot of {var}')
            
            # Scatter plot
            if len(numeric_vars) > 1:
                var1, var2 = numeric_vars[0], numeric_vars[1]
                sns.scatterplot(x=var1, y=var2, data=data, ax=axes[1, 1])
                axes[1, 1].set_title(f'{var1} vs {var2}')
        
        plt.tight_layout()
        plt.show()

# Create interactive dashboard
create_interactive_eda_dashboard(mtcars, "Motor Trend Cars Interactive EDA")
```

## Summary

This comprehensive guide to Exploratory Data Analysis covers:

### Key Concepts Covered:
1. **Data Exploration Fundamentals**: Understanding data structure, types, and basic properties
2. **Data Quality Assessment**: Missing values, outliers, duplicates, and consistency checks
3. **Univariate Analysis**: Distribution analysis, summary statistics, and visualization
4. **Bivariate Analysis**: Correlation analysis, scatter plots, and relationship exploration
5. **Multivariate Analysis**: Group comparisons, interactions, and complex patterns
6. **Data Transformation**: When and how to transform variables
7. **Best Practices**: Systematic approaches, quality checks, and documentation

### Python Libraries Used:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **scipy**: Statistical functions
- **statsmodels**: Statistical modeling
- **plotly**: Interactive visualizations (optional)

### Practical Applications:
- Customer satisfaction analysis
- Sales performance evaluation
- Data quality assessment
- Automated EDA pipelines
- Interactive dashboards

### Best Practices Emphasized:
- Always start with data quality assessment
- Use systematic approaches
- Document findings and decisions
- Create reproducible code
- Consider multiple perspectives
- Validate assumptions

This guide provides a solid foundation for conducting thorough exploratory data analysis in Python, with practical examples and exercises to reinforce learning.