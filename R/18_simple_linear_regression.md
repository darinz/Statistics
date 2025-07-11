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

## Fitting Simple Linear Regression in R

### Example: Predicting MPG from Weight

```r
# Load data
data(mtcars)

# Fit simple linear regression
model <- lm(mpg ~ wt, data = mtcars)
summary(model)

# Extract coefficients
coef(model)

# Fitted values and residuals
fitted_vals <- fitted(model)
residuals <- resid(model)

# Plot data and regression line
plot(mtcars$wt, mtcars$mpg, pch = 16, col = "blue",
     main = "Simple Linear Regression: MPG vs Weight",
     xlab = "Weight (1000 lbs)", ylab = "Miles per Gallon")
abline(model, col = "red", lwd = 2)
```

### Interpreting Output
- **Intercept**: Estimated MPG when weight is zero (not always meaningful).
- **Slope**: Change in MPG for each 1000 lb increase in weight.
- **$R^2$**: Proportion of variance in MPG explained by weight.
- **p-value**: Tests if the slope is significantly different from zero.

## Confidence and Prediction Intervals

### Confidence Interval for the Mean Response

```r
# Predict mean MPG for a new weight value
new_data <- data.frame(wt = 3)
predict(model, newdata = new_data, interval = "confidence")
```

### Prediction Interval for a New Observation

```r
# Predict individual MPG for a new car
predict(model, newdata = new_data, interval = "prediction")
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

```r
# Diagnostic plots
par(mfrow = c(2, 2))
plot(model)
par(mfrow = c(1, 1))

# Shapiro-Wilk test for normality
shapiro.test(resid(model))
```

### Outlier and Influence Diagnostics

```r
# Leverage and influence
influence_measures <- influence.measures(model)
print(influence_measures)

# Cook's distance
cooksd <- cooks.distance(model)
plot(cooksd, type = "h", main = "Cook's Distance", ylab = "Cook's D")
abline(h = 4/length(cooksd), col = "red", lty = 2)
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

```r
# Simulate data
set.seed(42)
house_size <- rnorm(100, mean = 1500, sd = 300)
house_price <- 50000 + 120 * house_size + rnorm(100, mean = 0, sd = 20000)

# Fit model
house_model <- lm(house_price ~ house_size)
summary(house_model)

# Plot
plot(house_size, house_price, pch = 16, col = "darkgreen",
     main = "House Price vs Size",
     xlab = "Size (sq ft)", ylab = "Price ($)")
abline(house_model, col = "red", lwd = 2)
```

### Example 2: Predicting Exam Scores from Study Hours

```r
# Simulate data
set.seed(123)
study_hours <- rnorm(50, mean = 10, sd = 2)
exam_score <- 40 + 5 * study_hours + rnorm(50, mean = 0, sd = 5)

# Fit model
exam_model <- lm(exam_score ~ study_hours)
summary(exam_model)

# Plot
plot(study_hours, exam_score, pch = 16, col = "purple",
     main = "Exam Score vs Study Hours",
     xlab = "Study Hours", ylab = "Exam Score")
abline(exam_model, col = "red", lwd = 2)
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