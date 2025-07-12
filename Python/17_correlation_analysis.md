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

**Code Reference**: See `simulate_correlation_data()` and `calculate_pearson_correlation()` functions in `17_correlation_analysis.py`.

The Pearson correlation analysis provides:
1. **Data Simulation**: Generate correlated data with specified correlation coefficient
2. **Correlation Calculation**: Compute Pearson correlation coefficient with significance testing
3. **Confidence Intervals**: Calculate 95% confidence intervals using Fisher's z-transformation
4. **Effect Size Interpretation**: Classify correlation strength (negligible, small, moderate, large, etc.)
5. **Comprehensive Results**: Returns correlation coefficient, p-value, confidence intervals, and effect size

This function provides a complete analysis of the linear relationship between two variables.

### Spearman and Kendall Correlation

**Code Reference**: See `calculate_spearman_kendall()` function in `17_correlation_analysis.py`.

The Spearman and Kendall correlation analysis provides:
1. **Spearman Correlation**: Rank-based correlation for monotonic relationships
2. **Kendall Correlation**: Ordinal correlation for small samples with many ties
3. **Significance Testing**: P-values for both correlation methods
4. **Effect Size Interpretation**: Strength classification for both correlations
5. **Comprehensive Results**: Returns correlation coefficients, p-values, and effect sizes for both methods

This function is particularly useful when data violate normality assumptions or when dealing with ordinal data.

### Correlation Matrix

**Code Reference**: See `simulate_multivariate_data()` and `create_correlation_matrix()` functions in `17_correlation_analysis.py`.

The correlation matrix analysis provides:
1. **Multivariate Data Simulation**: Generate multiple variables for correlation analysis
2. **Correlation Matrix**: Compute pairwise correlations between all variables
3. **Significance Testing**: Calculate p-values for each correlation pair
4. **Multiple Methods**: Support for Pearson, Spearman, and Kendall correlations
5. **Comprehensive Results**: Returns both correlation matrix and significance matrix

This function is essential for understanding relationships between multiple variables in a dataset.

## Visualization

### Scatter Plot with Correlation

**Code Reference**: See `plot_scatter_with_correlation()` function in `17_correlation_analysis.py`.

The scatter plot visualization provides:
1. **Scatter Plot**: Visual representation of the relationship between variables
2. **Regression Line**: Linear trend line with confidence intervals
3. **Correlation Information**: Display correlation coefficient and significance in title
4. **Customizable**: Adjustable plot size, colors, and styling
5. **Professional Output**: Publication-ready visualizations

This function creates informative scatter plots that clearly show the strength and direction of correlations.

### Correlation Matrix Heatmap

**Code Reference**: See `plot_correlation_matrix()` function in `17_correlation_analysis.py`.

The correlation matrix heatmap provides:
1. **Visual Matrix**: Color-coded representation of correlation strengths
2. **Numerical Values**: Display correlation coefficients in each cell
3. **Color Scheme**: Red-Yellow-Blue color map for intuitive interpretation
4. **Square Format**: Symmetric matrix layout for easy comparison
5. **Professional Styling**: Publication-ready heatmap visualizations

This function creates clear visual representations of correlation matrices, making it easy to identify strong and weak relationships.

## Assumption Checking

### Pearson Correlation Assumptions
- Linearity: Relationship between variables is linear
- Normality: Both variables are normally distributed
- Homoscedasticity: Constant variance of residuals
- No significant outliers

#### Checking Linearity and Outliers

**Code Reference**: See `plot_assumption_checks()` function in `17_correlation_analysis.py`.

The linearity and outlier checking provides:
1. **Scatter Plot**: Visual inspection of the relationship between variables
2. **Trend Line**: Linear regression line to assess linearity
3. **Outlier Detection**: Visual identification of potential outliers
4. **Comprehensive Assessment**: Part of the complete assumption checking suite
5. **Professional Visualization**: Clear, informative plots for assumption validation

This function helps assess whether the relationship between variables is linear and identifies any influential outliers.

#### Checking Normality

**Code Reference**: See `check_correlation_assumptions()` function in `17_correlation_analysis.py`.

The normality checking provides:
1. **Q-Q Plots**: Visual assessment of normality for both variables
2. **Shapiro-Wilk Test**: Formal statistical test for normality
3. **Comprehensive Results**: P-values and normality assessment for both variables
4. **Automated Assessment**: Automatic determination of whether normality assumptions are met
5. **Professional Output**: Clear, interpretable results for assumption validation

This function provides both visual and statistical evidence for normality, essential for determining whether Pearson correlation is appropriate.

#### Checking Homoscedasticity

**Code Reference**: See `check_correlation_assumptions()` function in `17_correlation_analysis.py`.

The homoscedasticity checking provides:
1. **Residual Analysis**: Plot of residuals against fitted values
2. **Variance Assessment**: Statistical test for constant variance
3. **Visual Inspection**: Clear visualization of variance patterns
4. **Quantitative Results**: Variance ratio and homoscedasticity assessment
5. **Automated Evaluation**: Automatic determination of homoscedasticity

This function helps assess whether the variance of residuals is constant across the range of predictor values, an important assumption for Pearson correlation.

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

**Code Reference**: See `correlation_confidence_interval()` function in `17_correlation_analysis.py`.

The confidence interval calculation provides:
1. **Fisher's z-Transformation**: Converts correlation to normal distribution
2. **Standard Error**: Calculates standard error of the transformed correlation
3. **Confidence Bounds**: Computes lower and upper confidence limits
4. **Back-Transformation**: Converts confidence limits back to correlation scale
5. **Flexible Alpha**: Adjustable significance level (default: 0.05)

This function provides precise confidence intervals for correlation coefficients, essential for understanding the uncertainty in correlation estimates.

### Fisher's z-Transformation

To compare correlations or compute confidence intervals:

```math
z = \frac{1}{2} \ln\left(\frac{1 + r}{1 - r}\right)
```

The standard error of $`z`$ is $`\frac{1}{\sqrt{n-3}}`$.

## Practical Examples

### Example 1: Height and Weight

**Code Reference**: See `height_weight_example()` function in `17_correlation_analysis.py`.

The height-weight example demonstrates:
1. **Realistic Data Simulation**: Generate height and weight data with known relationship
2. **Comprehensive Analysis**: Pearson, Spearman, and Kendall correlations
3. **Assumption Checking**: Full validation of correlation assumptions
4. **Visualization**: Scatter plots and assumption checking plots
5. **Complete Workflow**: End-to-end correlation analysis example

This example shows how to perform a complete correlation analysis on realistic data, including assumption checking and visualization.

### Example 2: Nonlinear Relationship

**Code Reference**: See `nonlinear_relationship_example()` function in `17_correlation_analysis.py`.

The nonlinear relationship example demonstrates:
1. **Nonlinear Data Simulation**: Generate quadratic relationship between variables
2. **Method Comparison**: Compare Pearson vs Spearman correlation results
3. **Visualization**: Scatter plot showing the nonlinear pattern
4. **Interpretation**: Understanding when different correlation methods are appropriate
5. **Educational Value**: Shows why Spearman correlation can be better for nonlinear relationships

This example illustrates the importance of choosing the right correlation method and understanding the limitations of Pearson correlation for nonlinear relationships.

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

## Using the Python Code

### Getting Started

To use the correlation analysis functions, import the module and run the main demonstration:

```python
# Run the complete demonstration
python 17_correlation_analysis.py
```

### Key Functions Overview

**Data Simulation:**
- `simulate_correlation_data()`: Generate correlated data with specified correlation
- `simulate_multivariate_data()`: Create multivariate datasets for matrix analysis

**Correlation Analysis:**
- `calculate_pearson_correlation()`: Complete Pearson correlation with confidence intervals
- `calculate_spearman_kendall()`: Spearman and Kendall correlations
- `create_correlation_matrix()`: Correlation matrices with significance testing

**Assumption Checking:**
- `check_correlation_assumptions()`: Comprehensive assumption validation
- `plot_assumption_checks()`: Visual assumption checking

**Visualization:**
- `plot_scatter_with_correlation()`: Scatter plots with correlation information
- `plot_correlation_matrix()`: Correlation matrix heatmaps

**Practical Examples:**
- `height_weight_example()`: Realistic height-weight correlation analysis
- `nonlinear_relationship_example()`: Demonstrating method differences
- `multivariate_correlation_example()`: Multi-variable correlation analysis

### Workflow Example

```python
# 1. Generate data
x, y = simulate_correlation_data(correlation=0.7)

# 2. Calculate correlations
pearson_result = calculate_pearson_correlation(x, y)
rank_results = calculate_spearman_kendall(x, y)

# 3. Check assumptions
assumptions = check_correlation_assumptions(x, y)

# 4. Create visualizations
plot_scatter_with_correlation(x, y, pearson_result)
plot_assumption_checks(x, y)

# 5. Interpret results
print(f"Correlation: {pearson_result['correlation']:.3f}")
print(f"Effect size: {pearson_result['effect_size']}")
print(f"Recommendation: {assumptions['overall_assessment']['recommendation']}")
```

### Function Reference

Each function includes comprehensive docstrings with:
- **Parameters**: Input variables and their types
- **Returns**: Output format and content
- **Examples**: Usage examples in the main execution block
- **Theory**: Connection to statistical concepts

### Integration with Theory

The Python functions implement the mathematical concepts discussed in this lesson:
- **Covariance and Correlation**: Mathematical foundations in `calculate_pearson_correlation()`
- **Fisher's z-Transformation**: Confidence intervals in `correlation_confidence_interval()`
- **Assumption Checking**: Statistical tests in `check_correlation_assumptions()`
- **Effect Size Interpretation**: Strength classification in `interpret_correlation_strength()`

---

**Key Takeaways:**
- Correlation quantifies the strength and direction of association
- Pearson for linear, normal data; Spearman/Kendall for ranks or non-normal data
- Always check assumptions and visualize
- Correlation ≠ causation
- Report effect size and confidence intervals
- Use the Python functions for comprehensive, reproducible analysis 