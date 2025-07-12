# Simple Linear Regression

## Overview

Simple linear regression models the relationship between a single predictor (independent variable) and a response (dependent variable) by fitting a straight line. It is foundational for prediction, understanding associations, and causal inference in statistics.

### Key Concepts
- **Regression**: Predicts the value of one variable based on another.
- **Line of Best Fit**: The straight line that minimizes the sum of squared residuals.
- **Slope and Intercept**: Quantify the direction and strength of the relationship.
- **Assumptions**: Linearity, independence, homoscedasticity, normality of residuals.

## Mathematical Foundations

### Model Equation

The simple linear regression model is:

```math
Y_i = \beta_0 + \beta_1 X_i + \epsilon_i
```

Where:
- $`Y_i`$: Response variable for observation $`i`$
- $`X_i`$: Predictor variable for observation $`i`$
- $`\beta_0`$: Intercept (expected value of $`Y`$ when $`X = 0`$)
- $`\beta_1`$: Slope (change in $`Y`$ for a one-unit increase in $`X`$)
- $`\epsilon_i`$: Error term (residual)

### Least Squares Estimation

The best-fitting line minimizes the sum of squared residuals:

```math
\text{RSS} = \sum_{i=1}^n (Y_i - \hat{Y}_i)^2
```

The estimated coefficients are:

```math
\hat{\beta}_1 = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^n (X_i - \bar{X})^2}
```

```math
\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{X}
```

### Interpretation
- $`\hat{\beta}_1`$: For each one-unit increase in $`X`$, $`Y`$ changes by $`\hat{\beta}_1`$ units (on average).
- $`\hat{\beta}_0`$: Expected value of $`Y`$ when $`X = 0`$ (may not always be meaningful).

### Goodness of Fit: $R^2$

$R^2$ measures the proportion of variance in $Y$ explained by $X$:

```math
R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = \frac{\text{ESS}}{\text{TSS}}
```
Where:
- $`\text{TSS} = \sum_{i=1}^n (Y_i - \bar{Y})^2`$ (total sum of squares)
- $`\text{ESS} = \sum_{i=1}^n (\hat{Y}_i - \bar{Y})^2`$ (explained sum of squares)
- $`\text{RSS} = \sum_{i=1}^n (Y_i - \hat{Y}_i)^2`$ (residual sum of squares)

$R^2$ ranges from 0 (no fit) to 1 (perfect fit).

## Fitting Simple Linear Regression in Python

### Example: Predicting MPG from Weight

**Code Reference**: See `mpg_weight_example()` function in `18_simple_linear_regression.py`.

The MPG vs Weight example demonstrates:
1. **Data Loading**: Load mtcars dataset with fallback simulation
2. **Model Fitting**: Complete simple linear regression analysis
3. **Results Interpretation**: Coefficient interpretation and model fit assessment
4. **Assumption Checking**: Comprehensive validation of regression assumptions
5. **Influence Analysis**: Outlier and influential point detection
6. **Visualization**: Regression plots and diagnostic visualizations

This example shows how to perform a complete simple linear regression analysis on real data, including all necessary diagnostics and interpretations.

### Interpreting Output

**Code Reference**: See `interpret_regression_results()` function in `18_simple_linear_regression.py`.

The interpretation function provides:
1. **Coefficient Interpretation**: Clear explanations of intercept and slope meanings
2. **Confidence Intervals**: Lower and upper bounds for coefficient estimates
3. **Model Fit Assessment**: R² interpretation and significance testing
4. **Statistical Significance**: F-statistic and p-value interpretation
5. **Practical Meaning**: Real-world interpretation of regression coefficients

This function automatically generates interpretable results for any simple linear regression model.

## Confidence and Prediction Intervals

### Confidence Interval for the Mean Response

**Code Reference**: See `calculate_confidence_intervals()` function in `18_simple_linear_regression.py`.

The confidence interval calculation provides:
1. **Mean Response Prediction**: Predict expected value for new observations
2. **Confidence Intervals**: Uncertainty bounds for the mean response
3. **Prediction Intervals**: Uncertainty bounds for individual observations
4. **Flexible Alpha**: Adjustable significance level (default: 0.05)
5. **Multiple Predictions**: Handle multiple new predictor values simultaneously

This function provides both confidence intervals (for the mean response) and prediction intervals (for individual observations), essential for understanding prediction uncertainty.

## Assumption Checking and Diagnostics

### 1. Linearity

- Relationship between $`X`$ and $`Y`$ should be linear.
- **Check**: Scatter plot, residuals vs fitted plot.

### 2. Independence

- Observations should be independent.
- **Check**: Study design, Durbin-Watson test for time series.

### 3. Homoscedasticity (Constant Variance)

- Residuals should have constant variance.
- **Check**: Plot residuals vs fitted values.

### 4. Normality of Residuals

- Residuals should be approximately normally distributed.
- **Check**: Q-Q plot, Shapiro-Wilk test.

**Code Reference**: See `create_diagnostic_plots()` and `check_regression_assumptions()` functions in `18_simple_linear_regression.py`.

The diagnostic plots and assumption checking provide:
1. **Residuals vs Fitted**: Check for linearity and homoscedasticity
2. **Normal Q-Q Plot**: Assess normality of residuals
3. **Scale-Location Plot**: Check for heteroscedasticity
4. **Residuals vs Leverage**: Identify influential points
5. **Statistical Tests**: Shapiro-Wilk test for normality, Breusch-Pagan for homoscedasticity
6. **Comprehensive Assessment**: Overall evaluation of all assumptions

These functions provide both visual and statistical evidence for assumption validation, essential for determining whether regression results are reliable.

### Outlier and Influence Diagnostics

**Code Reference**: See `analyze_influence()` and `plot_influence_diagnostics()` functions in `18_simple_linear_regression.py`.

The influence analysis provides:
1. **Leverage Analysis**: Identify high-leverage points using hat matrix diagonals
2. **Cook's Distance**: Measure influence of each observation on the model
3. **DFFITS**: Standardized influence measure for each observation
4. **Studentized Residuals**: Identify outliers in the response variable
5. **Visual Diagnostics**: Stem plots for Cook's distance and leverage
6. **Threshold Assessment**: Automatic identification of influential points

These functions help identify observations that may unduly influence the regression results, essential for robust model interpretation.

## Effect Size and Model Fit

### $R^2$ and Adjusted $R^2$
- $R^2$: Proportion of variance explained by the model.
- Adjusted $R^2$: Adjusts for number of predictors (useful for multiple regression).

### Standard Error of Estimate

```math
SE = \sqrt{\frac{\text{RSS}}{n-2}}
```

### F-statistic

Tests whether the model explains a significant amount of variance:

```math
F = \frac{\text{ESS}/1}{\text{RSS}/(n-2)}
```

## Practical Examples

### Example 1: Predicting House Prices

**Code Reference**: See `house_price_example()` function in `18_simple_linear_regression.py`.

The house price example demonstrates:
1. **Realistic Data Simulation**: Generate house size and price data with known relationship
2. **Model Fitting**: Complete simple linear regression analysis
3. **Results Interpretation**: Coefficient interpretation and model fit assessment
4. **Assumption Checking**: Comprehensive validation of regression assumptions
5. **Visualization**: Regression plots and diagnostic visualizations
6. **Practical Application**: Real-world scenario with meaningful variables

This example shows how to apply simple linear regression to a realistic business scenario, including all necessary diagnostics and interpretations.

### Example 2: Predicting Exam Scores from Study Hours

**Code Reference**: See `exam_score_example()` function in `18_simple_linear_regression.py`.

The exam score example demonstrates:
1. **Educational Data Simulation**: Generate study hours and exam score data
2. **Model Fitting**: Complete simple linear regression analysis
3. **Results Interpretation**: Coefficient interpretation and model fit assessment
4. **Assumption Checking**: Comprehensive validation of regression assumptions
5. **Visualization**: Regression plots and diagnostic visualizations
6. **Educational Application**: Academic scenario with clear causal interpretation

This example shows how to apply simple linear regression to educational research, demonstrating the relationship between study time and academic performance.

## Best Practices
- Always visualize data and check assumptions before interpreting results.
- Report coefficients, $R^2$, confidence intervals, and diagnostics.
- Avoid extrapolation beyond the range of observed data.
- Be cautious of outliers and influential points.
- Remember: correlation does not imply causation; regression does not prove causality without proper design.

## Reporting Guidelines

- Report the model equation, coefficients, $R^2$, confidence intervals, and p-values.
- Example: "A simple linear regression found that weight significantly predicted MPG, $`b = -5.34, 95\%\ CI [-7.0, -3.7], p < .001, R^2 = 0.75`$."

## Exercises

### Exercise 1: Fit and Interpret a Simple Linear Regression
Fit a model predicting MPG from horsepower in the mtcars dataset. Interpret the coefficients, $R^2$, and check assumptions.

### Exercise 2: Confidence and Prediction Intervals
Calculate and interpret confidence and prediction intervals for a new observation in your model.

### Exercise 3: Diagnostics and Outliers
Identify and interpret outliers and influential points in your regression model.

### Exercise 4: Real-World Application
Find a real dataset (e.g., from R's datasets package). Fit a simple linear regression, check assumptions, and report results.

## Using the Python Code

### Getting Started

To use the simple linear regression functions, import the module and run the main demonstration:

```python
# Run the complete demonstration
python 18_simple_linear_regression.py
```

### Key Functions Overview

**Data Preparation:**
- `load_mtcars_data()`: Load mtcars dataset with fallback simulation
- `simulate_regression_data()`: Generate data for regression analysis

**Model Fitting and Analysis:**
- `fit_simple_linear_regression()`: Complete regression analysis with diagnostics
- `interpret_regression_results()`: Automatic interpretation of results
- `calculate_confidence_intervals()`: Confidence and prediction intervals

**Assumption Checking:**
- `check_regression_assumptions()`: Comprehensive assumption validation
- `create_diagnostic_plots()`: Visual assumption checking
- `analyze_influence()`: Outlier and influence diagnostics
- `plot_influence_diagnostics()`: Visual influence analysis

**Visualization:**
- `plot_regression_results()`: Scatter plots with regression lines

**Practical Examples:**
- `mpg_weight_example()`: MPG vs Weight analysis using mtcars data
- `house_price_example()`: House price prediction example
- `exam_score_example()`: Educational regression example

### Workflow Example

```python
# 1. Generate or load data
x, y = simulate_regression_data(slope=2.0, intercept=10.0)

# 2. Fit regression model
results = fit_simple_linear_regression(x, y, 'x', 'y')

# 3. Interpret results
interpretation = interpret_regression_results(results)
print(f"Slope: {interpretation['slope']['value']:.3f}")
print(f"R²: {interpretation['model_fit']['r_squared']:.3f}")

# 4. Check assumptions
assumptions = check_regression_assumptions(results)
print(f"All assumptions met: {assumptions['overall_assessment']['all_assumptions_met']}")

# 5. Analyze influence
influence = analyze_influence(results)
print(f"Influential points: {influence['summary']['total_high_influence']}")

# 6. Create visualizations
plot_regression_results(results, "Sample Regression")
create_diagnostic_plots(results)
```

### Function Reference

Each function includes comprehensive docstrings with:
- **Parameters**: Input variables and their types
- **Returns**: Output format and content
- **Examples**: Usage examples in the main execution block
- **Theory**: Connection to statistical concepts

### Integration with Theory

The Python functions implement the mathematical concepts discussed in this lesson:
- **Least Squares Estimation**: Mathematical foundations in `fit_simple_linear_regression()`
- **Model Diagnostics**: Assumption checking in `check_regression_assumptions()`
- **Influence Analysis**: Outlier detection in `analyze_influence()`
- **Confidence Intervals**: Prediction uncertainty in `calculate_confidence_intervals()`

---

**Key Takeaways:**
- Simple linear regression models the relationship between two variables
- The slope quantifies the effect of the predictor
- $R^2$ measures goodness of fit
- Always check assumptions and diagnostics
- Report coefficients, intervals, and model fit
- Regression does not prove causation without proper design
- Use the Python functions for comprehensive, reproducible analysis 