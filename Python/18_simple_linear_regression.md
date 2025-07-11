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

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data (using seaborn's version of mtcars)
mtcars = sm.datasets.get_rdataset('mtcars').data

# Fit simple linear regression
model = smf.ols('mpg ~ wt', data=mtcars).fit()
print(model.summary())

# Extract coefficients
print("Coefficients:", model.params)

# Fitted values and residuals
fitted_vals = model.fittedvalues
residuals = model.resid

# Plot data and regression line
plt.figure(figsize=(8, 6))
plt.scatter(mtcars['wt'], mtcars['mpg'], color='blue', alpha=0.7, label='Data')
plt.plot(mtcars['wt'], fitted_vals, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression: MPG vs Weight')
plt.xlabel('Weight (1000 lbs)')
plt.ylabel('Miles per Gallon')
plt.legend()
plt.tight_layout()
plt.show()
```

### Interpreting Output
- **Intercept**: Estimated MPG when weight is zero (not always meaningful).
- **Slope**: Change in MPG for each 1000 lb increase in weight.
- **$R^2$**: Proportion of variance in MPG explained by weight.
- **p-value**: Tests if the slope is significantly different from zero.

## Confidence and Prediction Intervals

### Confidence Interval for the Mean Response

```python
# Predict mean MPG for a new weight value
new_data = pd.DataFrame({'wt': [3]})
pred_conf = model.get_prediction(new_data).summary_frame(alpha=0.05)
print("Confidence interval for mean response:")
print(pred_conf[['mean', 'mean_ci_lower', 'mean_ci_upper']])
```

### Prediction Interval for a New Observation

```python
# Predict individual MPG for a new car
print("Prediction interval for new observation:")
print(pred_conf[['mean', 'obs_ci_lower', 'obs_ci_upper']])
```

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

```python
# Diagnostic plots
import scipy.stats as stats
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
axes[0, 0].scatter(fitted_vals, residuals)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')

# Q-Q plot
sm.qqplot(residuals, line='s', ax=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

# Scale-Location plot
axes[1, 0].scatter(fitted_vals, np.sqrt(np.abs(residuals)))
axes[1, 0].set_title('Scale-Location')
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('Sqrt(|Residuals|)')

# Residuals vs Leverage
influence = model.get_influence()
leverage = influence.hat_matrix_diag
axes[1, 1].scatter(leverage, residuals)
axes[1, 1].set_title('Residuals vs Leverage')
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Residuals')

plt.tight_layout()
plt.show()

# Shapiro-Wilk test for normality
print("Shapiro-Wilk test for residuals:", stats.shapiro(residuals))
```

### Outlier and Influence Diagnostics

```python
# Leverage and influence
influence_measures = model.get_influence()
summary_frame = influence_measures.summary_frame()
print(summary_frame[['hat_diag', 'cooks_d', 'dffits', 'student_resid']])

# Cook's distance
cooksd = influence_measures.cooks_distance[0]
plt.figure(figsize=(8, 4))
plt.stem(np.arange(len(cooksd)), cooksd, use_line_collection=True)
plt.axhline(4/len(cooksd), color='red', linestyle='--')
plt.title("Cook's Distance")
plt.xlabel('Observation')
plt.ylabel("Cook's D")
plt.tight_layout()
plt.show()
```

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

```python
# Simulate data
np.random.seed(42)
house_size = np.random.normal(1500, 300, 100)
house_price = 50000 + 120 * house_size + np.random.normal(0, 20000, 100)

# Fit model
house_df = pd.DataFrame({'house_size': house_size, 'house_price': house_price})
house_model = smf.ols('house_price ~ house_size', data=house_df).fit()
print(house_model.summary())

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(house_size, house_price, color='darkgreen', alpha=0.7)
plt.plot(house_size, house_model.fittedvalues, color='red', linewidth=2)
plt.title('House Price vs Size')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.tight_layout()
plt.show()
```

### Example 2: Predicting Exam Scores from Study Hours

```python
# Simulate data
np.random.seed(123)
study_hours = np.random.normal(10, 2, 50)
exam_score = 40 + 5 * study_hours + np.random.normal(0, 5, 50)

# Fit model
exam_df = pd.DataFrame({'study_hours': study_hours, 'exam_score': exam_score})
exam_model = smf.ols('exam_score ~ study_hours', data=exam_df).fit()
print(exam_model.summary())

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(study_hours, exam_score, color='purple', alpha=0.7)
plt.plot(study_hours, exam_model.fittedvalues, color='red', linewidth=2)
plt.title('Exam Score vs Study Hours')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.tight_layout()
plt.show()
```

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

---

**Key Takeaways:**
- Simple linear regression models the relationship between two variables
- The slope quantifies the effect of the predictor
- $R^2$ measures goodness of fit
- Always check assumptions and diagnostics
- Report coefficients, intervals, and model fit
- Regression does not prove causation without proper design 