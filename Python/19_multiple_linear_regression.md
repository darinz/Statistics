# Multiple Linear Regression

## Overview

Multiple linear regression extends simple linear regression to model the relationship between a dependent variable and two or more independent variables. It is a powerful tool for prediction, explanation, and understanding the joint effects of predictors.

### Key Concepts
- **Multiple Regression**: Predicts a response using several predictors.
- **Coefficients**: Quantify the effect of each predictor, holding others constant.
- **Assumptions**: Linearity, independence, homoscedasticity, normality, and no multicollinearity.
- **Model Selection**: Choosing the best set of predictors.
- **Regularization**: Techniques to prevent overfitting.

## Mathematical Foundations

### Model Equation

The multiple linear regression model is:

```math
Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \cdots + \beta_p X_{ip} + \epsilon_i
```

Where:
- $`Y_i`$: Response variable for observation $`i`$
- $`X_{ij}`$: Value of predictor $`j`$ for observation $`i`$
- $`\beta_0`$: Intercept
- $`\beta_j`$: Slope for predictor $`j`$ (effect of $`X_j`$ holding others constant)
- $`\epsilon_i`$: Error term

### Matrix Notation

The model can be written as:

```math
\mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}
```

Where:
- $`\mathbf{Y}`$: $n \times 1$ vector of responses
- $`\mathbf{X}`$: $n \times (p+1)$ matrix of predictors (including intercept)
- $`\boldsymbol{\beta}`$: $(p+1) \times 1$ vector of coefficients
- $`\boldsymbol{\epsilon}`$: $n \times 1$ vector of errors

### Least Squares Estimation

The estimated coefficients minimize the sum of squared residuals:

```math
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}
```

### Interpretation
- $`\hat{\beta}_j`$: Expected change in $`Y`$ for a one-unit increase in $`X_j`$, holding all other predictors constant.
- $`\hat{\beta}_0`$: Expected value of $`Y`$ when all $`X_j = 0`$ (may not always be meaningful).

### Goodness of Fit: $R^2$ and Adjusted $R^2$

$R^2$ measures the proportion of variance in $Y$ explained by all predictors:

```math
R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}
```

Adjusted $R^2$ penalizes for the number of predictors:

```math
R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
```
Where $`n`$ is the number of observations and $`p`$ is the number of predictors.

## Fitting Multiple Linear Regression in R

### Example: Predicting MPG from Multiple Predictors

```r
# Load data
data(mtcars)

# Fit multiple linear regression
multiple_model <- lm(mpg ~ wt + hp + disp, data = mtcars)
summary(multiple_model)

# Extract coefficients
coef(multiple_model)

# Fitted values and residuals
fitted_vals <- fitted(multiple_model)
residuals <- resid(multiple_model)
```

### Interpreting Output
- **Intercept**: Estimated MPG when all predictors are zero (may not be meaningful).
- **Coefficients**: Change in MPG for a one-unit increase in each predictor, holding others constant.
- **$R^2$ and Adjusted $R^2$**: Proportion of variance explained.
- **p-values**: Test if each coefficient is significantly different from zero.

## Confidence Intervals and Standardized Coefficients

```r
# Confidence intervals for coefficients
confint(multiple_model, level = 0.95)

# Standardized coefficients
library(lm.beta)
lm.beta(multiple_model)
```

## Model Building and Selection

### Stepwise Regression

- **Forward selection**: Start with no predictors, add one at a time.
- **Backward elimination**: Start with all predictors, remove one at a time.
- **Stepwise**: Combination of both.

```r
library(MASS)
# Forward selection
forward_model <- stepAIC(lm(mpg ~ 1, data = mtcars), direction = "forward", scope = ~ wt + hp + disp + drat + qsec)
# Backward elimination
backward_model <- stepAIC(multiple_model, direction = "backward")
# Stepwise
stepwise_model <- stepAIC(multiple_model, direction = "both")
```

### Best Subset Selection

```r
library(leaps)
best_subsets <- regsubsets(mpg ~ wt + hp + disp + drat + qsec, data = mtcars, nvmax = 5)
summary(best_subsets)
```

## Model Diagnostics

### Residual Analysis

- **Linearity**: Residuals vs fitted plot should show no pattern.
- **Normality**: Q-Q plot of residuals should be approximately linear.
- **Homoscedasticity**: Residuals should have constant variance.
- **Independence**: Residuals should not be autocorrelated.

```r
par(mfrow = c(2, 2))
plot(multiple_model)
par(mfrow = c(1, 1))
```

### Multicollinearity Detection

- **Variance Inflation Factor (VIF)**: $`\text{VIF}_j = \frac{1}{1 - R_j^2}`$ (where $`R_j^2`$ is the $R^2$ from regressing $X_j$ on all other predictors).
- **Tolerance**: $`1 / \text{VIF}`$
- **Condition number**: Large values indicate multicollinearity.

```r
library(car)
vif(multiple_model)
1 / vif(multiple_model)
```

### Outlier and Influence Diagnostics

- **Cook's distance**: Identifies influential points.
- **Leverage**: Points with unusual predictor values.
- **Studentized residuals**: Detects outliers in $Y$.

```r
cooksd <- cooks.distance(multiple_model)
plot(cooksd, type = "h", main = "Cook's Distance", ylab = "Cook's D")
abline(h = 4/length(cooksd), col = "red", lty = 2)

leverage <- hatvalues(multiple_model)
plot(leverage, type = "h", main = "Leverage", ylab = "Leverage")
abline(h = 2 * (length(coef(multiple_model)) + 1) / nrow(mtcars), col = "red", lty = 2)
```

## Model Comparison and Validation

### Information Criteria
- **AIC**: Penalizes model complexity, lower is better.
- **BIC**: Stronger penalty for complexity, lower is better.

```r
AIC(multiple_model)
BIC(multiple_model)
```

### Cross-Validation

- **LOOCV**: Leave-one-out cross-validation.
- **k-fold CV**: Split data into $k$ parts, train on $k-1$, test on 1.

```r
library(boot)
cv.glm(mtcars, multiple_model)$delta[1]  # LOOCV MSE
cv.glm(mtcars, multiple_model, K = 5)$delta[1]  # 5-fold CV MSE
```

## Interaction and Polynomial Terms

### Including Interactions

- **Interaction**: Effect of one predictor depends on another.

```r
interaction_model <- lm(mpg ~ wt * hp + disp, data = mtcars)
summary(interaction_model)
```

### Polynomial Terms

- **Polynomial**: Captures nonlinear relationships.

```r
poly_model <- lm(mpg ~ wt + I(wt^2) + hp + disp, data = mtcars)
summary(poly_model)
```

## Regularization Methods

### Ridge Regression

```r
library(glmnet)
x <- as.matrix(mtcars[, c("wt", "hp", "disp")])
y <- mtcars$mpg
ridge_model <- glmnet(x, y, alpha = 0)
cv_ridge <- cv.glmnet(x, y, alpha = 0)
best_lambda <- cv_ridge$lambda.min
predict(ridge_model, s = best_lambda, type = "coefficients")
```

### Lasso Regression

```r
lasso_model <- glmnet(x, y, alpha = 1)
cv_lasso <- cv.glmnet(x, y, alpha = 1)
best_lambda_lasso <- cv_lasso$lambda.min
predict(lasso_model, s = best_lambda_lasso, type = "coefficients")
```

### Elastic Net

```r
elastic_net <- glmnet(x, y, alpha = 0.5)
cv_elastic <- cv.glmnet(x, y, alpha = 0.5)
best_lambda_elastic <- cv_elastic$lambda.min
predict(elastic_net, s = best_lambda_elastic, type = "coefficients")
```

## Practical Examples

### Example 1: Real Estate Analysis

```r
set.seed(123)
n_properties <- 100
square_feet <- rnorm(n_properties, mean = 2000, sd = 500)
bedrooms <- sample(1:5, n_properties, replace = TRUE)
bathrooms <- sample(1:4, n_properties, replace = TRUE)
age <- rnorm(n_properties, mean = 15, sd = 8)
location_score <- rnorm(n_properties, mean = 7, sd = 1)
price <- 200000 + 100 * square_feet + 15000 * bedrooms + 25000 * bathrooms - 2000 * age + 15000 * location_score + rnorm(n_properties, mean = 0, sd = 15000)
real_estate_data <- data.frame(price, square_feet, bedrooms, bathrooms, age, location_score)
real_estate_model <- lm(price ~ square_feet + bedrooms + bathrooms + age + location_score, data = real_estate_data)
summary(real_estate_model)
```

### Example 2: Marketing Analysis

```r
set.seed(123)
n_campaigns <- 50
ad_spend <- rnorm(n_campaigns, mean = 10000, sd = 3000)
social_media_posts <- rpois(n_campaigns, lambda = 20)
email_sends <- rpois(n_campaigns, lambda = 1000)
season <- sample(c("Spring", "Summer", "Fall", "Winter"), n_campaigns, replace = TRUE)
sales <- 50000 + 2.5 * ad_spend + 500 * social_media_posts + 10 * email_sends + rnorm(n_campaigns, mean = 0, sd = 5000)
marketing_data <- data.frame(sales, ad_spend, social_media_posts, email_sends, season)
marketing_model <- lm(sales ~ ad_spend + social_media_posts + email_sends + season, data = marketing_data)
summary(marketing_model)
```

## Best Practices
- Always check assumptions and perform diagnostics before interpreting results.
- Use $R^2$, adjusted $R^2$, AIC, BIC, and cross-validation for model selection.
- Check for multicollinearity using VIF and correlation matrices.
- Use regularization for high-dimensional or collinear data.
- Report coefficients, intervals, diagnostics, and model fit.
- Interpret coefficients in the context of the model and data.

## Reporting Guidelines

- Report the model equation, coefficients, $R^2$, adjusted $R^2$, confidence intervals, p-values, and diagnostics.
- Example: "A multiple regression found that weight, horsepower, and displacement significantly predicted MPG, $`R^2 = 0.83, F(3, 28) = 45.2, p < .001`$. Weight had the largest negative effect ($`b = -3.5, 95\%\ CI [-4.7, -2.3], p < .001`$)."

## Exercises

### Exercise 1: Model Building and Selection
Build a multiple regression model to predict MPG using all available variables in the mtcars dataset. Use stepwise selection and compare models using $R^2$, AIC, and BIC.

### Exercise 2: Diagnostics and Multicollinearity
Perform comprehensive diagnostics on your model, including residual analysis, VIF, and outlier detection. Interpret the results.

### Exercise 3: Interaction and Polynomial Terms
Add interaction and polynomial terms to your model. Interpret their effects and compare model fit.

### Exercise 4: Regularization
Apply ridge and lasso regression to your data. Compare the coefficients and model fit to the standard regression.

### Exercise 5: Real-World Application
Find a real dataset (e.g., from R's datasets package). Build, validate, and report a multiple regression model.

---

**Key Takeaways:**
- Multiple regression models the effect of several predictors on a response
- Always check assumptions and diagnostics
- Use model selection and regularization to avoid overfitting
- Interpret coefficients in context
- Report all relevant statistics and diagnostics 