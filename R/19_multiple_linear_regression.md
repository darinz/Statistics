# Multiple Linear Regression

## Overview

Multiple linear regression extends simple linear regression to model the relationship between a dependent variable and multiple independent variables. It's one of the most widely used statistical techniques for prediction and understanding variable relationships.

## Basic Multiple Linear Regression

### Model Fitting

```r
# Load sample data
data(mtcars)

# Fit multiple linear regression model
# Predict MPG using weight, horsepower, and displacement
multiple_model <- lm(mpg ~ wt + hp + disp, data = mtcars)

# View model summary
summary(multiple_model)

# Extract key statistics
coefficients <- coef(multiple_model)
r_squared <- summary(multiple_model)$r.squared
adj_r_squared <- summary(multiple_model)$adj.r.squared
f_statistic <- summary(multiple_model)$fstatistic[1]
p_value <- pf(f_statistic, summary(multiple_model)$fstatistic[2], 
              summary(multiple_model)$fstatistic[3], lower.tail = FALSE)

cat("R-squared:", r_squared, "\n")
cat("Adjusted R-squared:", adj_r_squared, "\n")
cat("F-statistic:", f_statistic, "\n")
cat("p-value:", p_value, "\n")
```

### Model Interpretation

```r
# Extract coefficient table
coef_table <- summary(multiple_model)$coefficients
print(coef_table)

# Calculate confidence intervals for coefficients
conf_intervals <- confint(multiple_model, level = 0.95)
print(conf_intervals)

# Standardized coefficients
library(lm.beta)
standardized_model <- lm.beta(multiple_model)
print(standardized_model)
```

## Model Building Strategies

### Stepwise Regression

```r
# Forward selection
library(MASS)
forward_model <- stepAIC(lm(mpg ~ 1, data = mtcars), 
                        direction = "forward",
                        scope = ~ wt + hp + disp + drat + qsec)

print(forward_model)

# Backward elimination
backward_model <- stepAIC(multiple_model, direction = "backward")
print(backward_model)

# Stepwise (both directions)
stepwise_model <- stepAIC(multiple_model, direction = "both")
print(stepwise_model)
```

### Best Subset Selection

```r
library(leaps)

# Best subset selection
best_subsets <- regsubsets(mpg ~ wt + hp + disp + drat + qsec, 
                          data = mtcars, nvmax = 5)

# Summary of best models
summary_best <- summary(best_subsets)
print(summary_best)

# Plot results
par(mfrow = c(2, 2))
plot(best_subsets, scale = "r2")
plot(best_subsets, scale = "adjr2")
plot(best_subsets, scale = "Cp")
plot(best_subsets, scale = "bic")
par(mfrow = c(1, 1))
```

## Model Diagnostics

### Residual Analysis

```r
# Basic residual plots
par(mfrow = c(2, 2))
plot(multiple_model)
par(mfrow = c(1, 1))

# Additional diagnostic plots
library(car)

# Residual vs fitted plot
plot(multiple_model, which = 1, main = "Residuals vs Fitted")

# Normal Q-Q plot
plot(multiple_model, which = 2, main = "Normal Q-Q Plot")

# Scale-location plot
plot(multiple_model, which = 3, main = "Scale-Location Plot")

# Residuals vs leverage
plot(multiple_model, which = 5, main = "Residuals vs Leverage")
```

### Multicollinearity Detection

```r
# Variance Inflation Factor (VIF)
vif_values <- vif(multiple_model)
print(vif_values)

# Tolerance
tolerance <- 1 / vif_values
print(tolerance)

# Correlation matrix of predictors
predictor_cor <- cor(mtcars[, c("wt", "hp", "disp")])
print(predictor_cor)

# Condition number
library(perturb)
condition_number <- colldiag(multiple_model)
print(condition_number)
```

### Outlier Detection

```r
# Cook's distance
cooks_distance <- cooks.distance(multiple_model)
influential_points <- which(cooks_distance > 4/length(cooks_distance))

cat("Influential points (Cook's distance > 4/n):", influential_points, "\n")

# Leverage
leverage <- hatvalues(multiple_model)
high_leverage <- which(leverage > 2 * (length(coef(multiple_model)) + 1) / nrow(mtcars))

cat("High leverage points:", high_leverage, "\n")

# Studentized residuals
studentized_residuals <- rstudent(multiple_model)
outliers <- which(abs(studentized_residuals) > 2)

cat("Outliers (|studentized residuals| > 2):", outliers, "\n")
```

## Model Comparison

### Information Criteria

```r
# AIC and BIC
aic_value <- AIC(multiple_model)
bic_value <- BIC(multiple_model)

cat("AIC:", aic_value, "\n")
cat("BIC:", bic_value, "\n")

# Compare models with different predictors
model1 <- lm(mpg ~ wt, data = mtcars)
model2 <- lm(mpg ~ wt + hp, data = mtcars)
model3 <- lm(mpg ~ wt + hp + disp, data = mtcars)

# Create comparison table
model_comparison <- data.frame(
  Model = c("MPG ~ Weight", "MPG ~ Weight + HP", "MPG ~ Weight + HP + Disp"),
  R_squared = c(summary(model1)$r.squared, 
                summary(model2)$r.squared, 
                summary(model3)$r.squared),
  Adj_R_squared = c(summary(model1)$adj.r.squared, 
                    summary(model2)$adj.r.squared, 
                    summary(model3)$adj.r.squared),
  AIC = c(AIC(model1), AIC(model2), AIC(model3)),
  BIC = c(BIC(model1), BIC(model2), BIC(model3))
)

print(model_comparison)
```

### Cross-Validation

```r
# Leave-one-out cross-validation
library(boot)

cv_results <- cv.glm(mtcars, multiple_model)
cat("LOOCV MSE:", cv_results$delta[1], "\n")

# k-fold cross-validation
set.seed(123)
k_fold_cv <- cv.glm(mtcars, multiple_model, K = 5)
cat("5-fold CV MSE:", k_fold_cv$delta[1], "\n")

# Manual cross-validation
n <- nrow(mtcars)
k <- 5
folds <- sample(rep(1:k, length.out = n))
cv_errors <- numeric(k)

for (i in 1:k) {
  test_indices <- which(folds == i)
  train_data <- mtcars[-test_indices, ]
  test_data <- mtcars[test_indices, ]
  
  # Fit model on training data
  train_model <- lm(mpg ~ wt + hp + disp, data = train_data)
  
  # Predict on test data
  predictions <- predict(train_model, newdata = test_data)
  
  # Calculate MSE
  cv_errors[i] <- mean((test_data$mpg - predictions)^2)
}

cat("Manual 5-fold CV MSE:", mean(cv_errors), "\n")
cat("CV MSE standard error:", sd(cv_errors) / sqrt(k), "\n")
```

## Interaction Effects

### Including Interactions

```r
# Model with interaction terms
interaction_model <- lm(mpg ~ wt * hp + disp, data = mtcars)
summary(interaction_model)

# Compare models with and without interactions
anova(multiple_model, interaction_model)

# Visualize interactions
library(ggplot2)

# Create interaction plot
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(hp > median(hp)))) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "MPG vs Weight by HP Level",
       x = "Weight", y = "MPG",
       color = "HP Level") +
  theme_minimal()
```

### Polynomial Terms

```r
# Model with polynomial terms
poly_model <- lm(mpg ~ wt + I(wt^2) + hp + disp, data = mtcars)
summary(poly_model)

# Compare linear vs polynomial model
anova(multiple_model, poly_model)

# Visualize polynomial relationship
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ x + I(x^2), se = TRUE) +
  labs(title = "MPG vs Weight (Polynomial)",
       x = "Weight", y = "MPG") +
  theme_minimal()
```

## Model Validation

### Prediction Intervals

```r
# Create new data for prediction
new_data <- data.frame(
  wt = c(2.5, 3.0, 3.5),
  hp = c(100, 150, 200),
  disp = c(150, 200, 250)
)

# Point predictions
predictions <- predict(multiple_model, newdata = new_data, interval = "prediction")
print(predictions)

# Confidence intervals for predictions
confidence_intervals <- predict(multiple_model, newdata = new_data, interval = "confidence")
print(confidence_intervals)
```

### Model Assumptions

```r
# Test for normality of residuals
shapiro_test <- shapiro.test(residuals(multiple_model))
print(shapiro_test)

# Test for homoscedasticity
library(lmtest)
breusch_pagan_test <- bptest(multiple_model)
print(breusch_pagan_test)

# Test for independence of residuals
durbin_watson_test <- dwtest(multiple_model)
print(durbin_watson_test)

# Linearity test
linearity_test <- resettest(multiple_model)
print(linearity_test)
```

## Advanced Topics

### Ridge Regression

```r
library(glmnet)

# Prepare data for ridge regression
x <- as.matrix(mtcars[, c("wt", "hp", "disp")])
y <- mtcars$mpg

# Fit ridge regression
ridge_model <- glmnet(x, y, alpha = 0)

# Cross-validation for lambda selection
cv_ridge <- cv.glmnet(x, y, alpha = 0)
best_lambda <- cv_ridge$lambda.min

cat("Best lambda:", best_lambda, "\n")

# Coefficients with best lambda
ridge_coef <- predict(ridge_model, s = best_lambda, type = "coefficients")
print(ridge_coef)
```

### Lasso Regression

```r
# Fit lasso regression
lasso_model <- glmnet(x, y, alpha = 1)

# Cross-validation for lambda selection
cv_lasso <- cv.glmnet(x, y, alpha = 1)
best_lambda_lasso <- cv_lasso$lambda.min

cat("Best lambda (Lasso):", best_lambda_lasso, "\n")

# Coefficients with best lambda
lasso_coef <- predict(lasso_model, s = best_lambda_lasso, type = "coefficients")
print(lasso_coef)
```

### Elastic Net

```r
# Fit elastic net (alpha = 0.5)
elastic_net <- glmnet(x, y, alpha = 0.5)

# Cross-validation
cv_elastic <- cv.glmnet(x, y, alpha = 0.5)
best_lambda_elastic <- cv_elastic$lambda.min

cat("Best lambda (Elastic Net):", best_lambda_elastic, "\n")

# Coefficients
elastic_coef <- predict(elastic_net, s = best_lambda_elastic, type = "coefficients")
print(elastic_coef)
```

## Practical Examples

### Example 1: Real Estate Analysis

```r
# Simulate real estate data
set.seed(123)
n_properties <- 100

# Generate data
square_feet <- rnorm(n_properties, mean = 2000, sd = 500)
bedrooms <- sample(1:5, n_properties, replace = TRUE)
bathrooms <- sample(1:4, n_properties, replace = TRUE)
age <- rnorm(n_properties, mean = 15, sd = 8)
location_score <- rnorm(n_properties, mean = 7, sd = 1)

# Price depends on all factors
price <- 200000 + 100 * square_feet + 15000 * bedrooms + 
         25000 * bathrooms - 2000 * age + 15000 * location_score + 
         rnorm(n_properties, mean = 0, sd = 15000)

real_estate_data <- data.frame(
  price = price,
  square_feet = square_feet,
  bedrooms = bedrooms,
  bathrooms = bathrooms,
  age = age,
  location_score = location_score
)

# Fit multiple regression model
real_estate_model <- lm(price ~ square_feet + bedrooms + bathrooms + age + location_score, 
                        data = real_estate_data)

summary(real_estate_model)

# Check multicollinearity
vif_real_estate <- vif(real_estate_model)
print(vif_real_estate)
```

### Example 2: Marketing Analysis

```r
# Simulate marketing data
set.seed(123)
n_campaigns <- 50

# Generate data
ad_spend <- rnorm(n_campaigns, mean = 10000, sd = 3000)
social_media_posts <- rpois(n_campaigns, lambda = 20)
email_sends <- rpois(n_campaigns, lambda = 1000)
season <- sample(c("Spring", "Summer", "Fall", "Winter"), n_campaigns, replace = TRUE)

# Sales depends on marketing efforts
sales <- 50000 + 2.5 * ad_spend + 500 * social_media_posts + 
         10 * email_sends + rnorm(n_campaigns, mean = 0, sd = 5000)

marketing_data <- data.frame(
  sales = sales,
  ad_spend = ad_spend,
  social_media_posts = social_media_posts,
  email_sends = email_sends,
  season = season
)

# Fit model with categorical variable
marketing_model <- lm(sales ~ ad_spend + social_media_posts + email_sends + season, 
                     data = marketing_data)

summary(marketing_model)

# Calculate ROI for each channel
roi_ad_spend <- coef(marketing_model)["ad_spend"] * 1
roi_social <- coef(marketing_model)["social_media_posts"] * 1
roi_email <- coef(marketing_model)["email_sends"] * 1

cat("ROI per dollar spent on ads:", roi_ad_spend, "\n")
cat("ROI per social media post:", roi_social, "\n")
cat("ROI per email sent:", roi_email, "\n")
```

## Best Practices

### Model Selection Guidelines

```r
# Function to evaluate model quality
evaluate_model <- function(model, data) {
  # Basic statistics
  r_squared <- summary(model)$r.squared
  adj_r_squared <- summary(model)$adj.r.squared
  aic_value <- AIC(model)
  bic_value <- BIC(model)
  
  # Residual analysis
  residuals <- residuals(model)
  shapiro_p <- shapiro.test(residuals)$p.value
  
  # Multicollinearity
  vif_values <- vif(model)
  max_vif <- max(vif_values)
  
  # Outliers
  cooks_dist <- cooks.distance(model)
  influential_count <- sum(cooks_dist > 4/length(cooks_dist))
  
  return(list(
    R_squared = r_squared,
    Adj_R_squared = adj_r_squared,
    AIC = aic_value,
    BIC = bic_value,
    Shapiro_p = shapiro_p,
    Max_VIF = max_vif,
    Influential_points = influential_count
  ))
}

# Apply to our model
model_evaluation <- evaluate_model(multiple_model, mtcars)
print(model_evaluation)
```

### Reporting Guidelines

```r
# Function to generate regression report
generate_regression_report <- function(model, model_name = "Multiple Regression") {
  summary_stats <- summary(model)
  
  cat("=== REGRESSION ANALYSIS REPORT ===\n\n")
  cat("Model:", model_name, "\n")
  cat("Dependent variable:", all.vars(formula(model))[1], "\n")
  cat("Independent variables:", paste(all.vars(formula(model))[-1], collapse = ", "), "\n\n")
  
  # Model fit
  cat("MODEL FIT:\n")
  cat("R-squared:", round(summary_stats$r.squared, 3), "\n")
  cat("Adjusted R-squared:", round(summary_stats$adj.r.squared, 3), "\n")
  cat("F-statistic:", round(summary_stats$fstatistic[1], 3), "\n")
  cat("p-value:", ifelse(pf(summary_stats$fstatistic[1], 
                            summary_stats$fstatistic[2], 
                            summary_stats$fstatistic[3], 
                            lower.tail = FALSE) < 0.001, "< .001", 
                         round(pf(summary_stats$fstatistic[1], 
                                 summary_stats$fstatistic[2], 
                                 summary_stats$fstatistic[3], 
                                 lower.tail = FALSE), 3)), "\n\n")
  
  # Coefficients
  cat("COEFFICIENTS:\n")
  coef_table <- summary_stats$coefficients
  for (i in 1:nrow(coef_table)) {
    cat(rownames(coef_table)[i], ":\n")
    cat("  Estimate:", round(coef_table[i, 1], 3), "\n")
    cat("  Std. Error:", round(coef_table[i, 2], 3), "\n")
    cat("  t-value:", round(coef_table[i, 3], 3), "\n")
    cat("  p-value:", ifelse(coef_table[i, 4] < 0.001, "< .001", 
                             round(coef_table[i, 4], 3)), "\n\n")
  }
}

# Apply to our model
generate_regression_report(multiple_model, "MPG Prediction Model")
```

## Exercises

### Exercise 1: Model Building
Build a multiple regression model to predict MPG using all available variables in the mtcars dataset. Use stepwise selection to find the best model.

### Exercise 2: Diagnostics
Perform comprehensive diagnostics on your multiple regression model, including residual analysis, multicollinearity checks, and outlier detection.

### Exercise 3: Model Comparison
Compare different model selection criteria (AIC, BIC, adjusted R-squared) for predicting MPG.

### Exercise 4: Interaction Effects
Investigate interaction effects between variables in your regression model and interpret the results.

### Exercise 5: Prediction
Use your final model to make predictions for new data and calculate prediction intervals.

## Next Steps

In the next chapter, we'll learn about logistic regression for binary outcome variables.

---

**Key Takeaways:**
- Multiple regression extends simple regression to multiple predictors
- Always check assumptions and perform diagnostics
- Use model selection criteria to choose the best model
- Consider interaction effects and polynomial terms
- Validate models using cross-validation
- Be aware of multicollinearity and its effects
- Regularization methods can help with overfitting
- Always interpret coefficients in context of the model 