# Model Diagnostics and Validation

## Overview

Model diagnostics are essential for assessing the validity, reliability, and predictive power of regression models. Diagnostics help identify violations of assumptions, influential observations, and model fit issues, ensuring that statistical inferences and predictions are trustworthy.

### Key Concepts
- **Residuals**: Differences between observed and predicted values.
- **Leverage**: Measures how far an observation's predictor values are from the mean.
- **Influence**: Quantifies the impact of an observation on model estimates.
- **Assumptions**: Linearity, independence, homoscedasticity, normality, and no multicollinearity.
- **Model Fit**: How well the model explains the data.

## Mathematical Foundations

### Residuals

For observation $`i`$:

```math
r_i = y_i - \hat{y}_i
```

- $`y_i`$: Observed value
- $`\hat{y}_i`$: Predicted value

### Standardized and Studentized Residuals

- **Standardized residual**:

```math
r_i^* = \frac{r_i}{\hat{\sigma} \sqrt{1 - h_{ii}}}
```

- **Studentized residual** (externally studentized):

```math
t_i = \frac{r_i}{\hat{\sigma}_{(i)} \sqrt{1 - h_{ii}}}
```

Where $`h_{ii}`$ is the leverage for observation $`i`$.

### Leverage

Leverage measures the influence of an observation's predictor values:

```math
h_{ii} = \mathbf{x}_i^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_i
```

- $`\mathbf{x}_i`$: Row vector of predictors for observation $`i`$
- High leverage: $`h_{ii} > 2(p+1)/n`$ (rule of thumb)

### Cook's Distance

Cook's distance quantifies the influence of an observation on all fitted values:

```math
D_i = \frac{\sum_{j=1}^n (\hat{y}_j - \hat{y}_{j(i)})^2}{(p+1) \hat{\sigma}^2}
```

- $`\hat{y}_{j(i)}`$: Fitted value for $`j`$ when $`i`$ is omitted
- Large $`D_i`$ (e.g., $`> 4/n`$) indicates influential points

### Variance Inflation Factor (VIF)

VIF detects multicollinearity:

```math
\text{VIF}_j = \frac{1}{1 - R_j^2}
```

Where $`R_j^2`$ is the $R^2$ from regressing $`X_j`$ on all other predictors.

### Model Fit Metrics
- **$R^2$**: Proportion of variance explained (linear regression)
- **AIC/BIC**: Penalized likelihood criteria for model selection
- **Deviance**: Generalized measure of model fit (GLMs)

## Diagnostics for Linear Regression

### Residual Analysis

```r
# Fit model
data(mtcars)
model <- lm(mpg ~ wt + hp + disp, data = mtcars)

# Residual plots
par(mfrow = c(2, 2))
plot(model)
par(mfrow = c(1, 1))

# Standardized and studentized residuals
std_resid <- rstandard(model)
stud_resid <- rstudent(model)

# Q-Q plot for normality
qqnorm(stud_resid)
qqline(stud_resid, col = "red")

# Shapiro-Wilk test for normality
shapiro.test(stud_resid)
```

### Leverage and Influence

```r
# Leverage
leverage <- hatvalues(model)
plot(leverage, type = "h", main = "Leverage", ylab = "Leverage")
abline(h = 2 * (length(coef(model)) + 1) / nrow(mtcars), col = "red", lty = 2)

# Cook's distance
cooksd <- cooks.distance(model)
plot(cooksd, type = "h", main = "Cook's Distance", ylab = "Cook's D")
abline(h = 4/length(cooksd), col = "red", lty = 2)

# Influence plot
library(car)
influencePlot(model)
```

### Multicollinearity

```r
library(car)
vif(model)
1 / vif(model)  # Tolerance
```

### Heteroscedasticity

```r
# Breusch-Pagan test
library(lmtest)
bptest(model)

# Plot residuals vs fitted
plot(fitted(model), resid(model),
     main = "Residuals vs Fitted Values",
     xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red")
```

### Independence

```r
# Durbin-Watson test for autocorrelation
library(lmtest)
dwtest(model)
```

## Diagnostics for Logistic Regression

### Residuals and Influence

```r
# Fit logistic regression
mtcars$manual <- ifelse(mtcars$am == 1, 1, 0)
logit_model <- glm(manual ~ mpg + wt + hp, data = mtcars, family = binomial)

# Pearson and deviance residuals
pearson_resid <- residuals(logit_model, type = "pearson")
dev_resid <- residuals(logit_model, type = "deviance")

# Plot residuals
plot(fitted(logit_model), pearson_resid, main = "Pearson Residuals vs Fitted")
abline(h = 0, col = "red")

plot(fitted(logit_model), dev_resid, main = "Deviance Residuals vs Fitted")
abline(h = 0, col = "red")

# Leverage and Cook's distance
leverage <- hatvalues(logit_model)
cooksd <- cooks.distance(logit_model)
plot(leverage, dev_resid, main = "Deviance Residuals vs Leverage")
plot(cooksd, type = "h", main = "Cook's Distance (Logistic)")
abline(h = 4/length(cooksd), col = "red", lty = 2)
```

### Model Fit and Classification

```r
# Pseudo R-squared
library(pscl)
pR2(logit_model)

# Hosmer-Lemeshow test
library(ResourceSelection)
hoslem.test(logit_model$y, fitted(logit_model))

# ROC curve and AUC
library(pROC)
pred_probs <- predict(logit_model, type = "response")
roc_obj <- roc(mtcars$manual, pred_probs)
auc(roc_obj)
plot(roc_obj, main = "ROC Curve")
abline(a = 0, b = 1, col = "red", lty = 2)
```

## Generalized Linear Model (GLM) Diagnostics

- Many diagnostics for linear and logistic regression extend to other GLMs (e.g., Poisson, multinomial).
- Use deviance, residuals, influence, and fit statistics as appropriate for the model family.

## Visualization and Interpretation

### Diagnostic Plot Matrix

```r
# For any lm or glm model
par(mfrow = c(2, 2))
plot(model)
par(mfrow = c(1, 1))
```

### Influence Plot

```r
library(car)
influencePlot(model)
```

### Residuals vs Leverage Plot

```r
plot(hatvalues(model), rstudent(model),
     main = "Studentized Residuals vs Leverage",
     xlab = "Leverage", ylab = "Studentized Residuals")
abline(h = c(-2, 2), col = "red", lty = 2)
```

## Best Practices
- Always check all relevant assumptions before interpreting results.
- Use multiple diagnostics to identify problems (no single plot/test is sufficient).
- Investigate and report influential points and outliers.
- Address violations: transform variables, use robust methods, or reconsider model.
- Report all diagnostics and model fit statistics in publications.

## Reporting Guidelines

- Report which diagnostics were performed and their results.
- Example: "Model diagnostics indicated no major violations of assumptions. Residuals were approximately normal (Shapiro-Wilk $`p = 0.21`$), no high-leverage points (all $`h_{ii} < 0.2`$), and no influential observations (all Cook's $`D < 0.15`$)."

## Exercises

### Exercise 1: Linear Regression Diagnostics
Fit a multiple regression model to the mtcars dataset. Perform and interpret all key diagnostics (residuals, leverage, influence, multicollinearity, heteroscedasticity, independence).

### Exercise 2: Logistic Regression Diagnostics
Fit a logistic regression model to predict transmission type in mtcars. Perform and interpret residual, influence, and fit diagnostics.

### Exercise 3: Outlier and Influence Analysis
Simulate a dataset with outliers and high-leverage points. Fit a regression model and use diagnostics to identify and interpret these points.

### Exercise 4: Model Fit and Validation
Compare two regression models using AIC, BIC, and cross-validation. Assess which model is better and why.

### Exercise 5: Real-World Application
Find a real dataset. Fit a regression model, perform comprehensive diagnostics, and report your findings.

---

**Key Takeaways:**
- Diagnostics are essential for trustworthy regression analysis
- Always check residuals, leverage, influence, and assumptions
- Use multiple diagnostics and visualizations
- Address and report any violations or influential points
- Good diagnostics lead to better models and more reliable conclusions 