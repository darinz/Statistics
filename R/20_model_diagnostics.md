# Logistic Regression

## Overview

Logistic regression is used to model binary or categorical outcomes. It's one of the most important statistical techniques for classification problems and is widely used in medical research, marketing, and social sciences.

## Binary Logistic Regression

### Basic Model Fitting

```r
# Load sample data
data(mtcars)

# Convert transmission to binary outcome
mtcars$manual <- ifelse(mtcars$am == 1, 1, 0)

# Fit logistic regression model
logistic_model <- glm(manual ~ mpg + wt + hp, data = mtcars, family = binomial)

# View model summary
summary(logistic_model)

# Extract key statistics
coefficients <- coef(logistic_model)
aic_value <- AIC(logistic_model)
deviance <- deviance(logistic_model)
null_deviance <- logistic_model$null.deviance

cat("AIC:", aic_value, "\n")
cat("Deviance:", deviance, "\n")
cat("Null Deviance:", null_deviance, "\n")
```

### Model Interpretation

```r
# Extract coefficient table
coef_table <- summary(logistic_model)$coefficients
print(coef_table)

# Calculate odds ratios
odds_ratios <- exp(coef(logistic_model))
print(odds_ratios)

# Calculate confidence intervals for odds ratios
conf_intervals <- confint(logistic_model)
odds_ratios_ci <- exp(conf_intervals)
print(odds_ratios_ci)

# Standardized coefficients
library(arm)
standardized_coef <- standardize(logistic_model)
print(standardized_coef)
```

### Model Fit Assessment

```r
# Hosmer-Lemeshow test
library(ResourceSelection)
hl_test <- hoslem.test(logistic_model$y, fitted(logistic_model))
print(hl_test)

# Pseudo R-squared measures
library(pscl)
pseudo_r2 <- pR2(logistic_model)
print(pseudo_r2)

# McFadden's R-squared
mcfadden_r2 <- 1 - (logistic_model$deviance / logistic_model$null.deviance)
cat("McFadden's R-squared:", mcfadden_r2, "\n")

# Cox & Snell R-squared
cox_snell_r2 <- 1 - exp((logistic_model$deviance - logistic_model$null.deviance) / nrow(mtcars))
cat("Cox & Snell R-squared:", cox_snell_r2, "\n")
```

## Model Diagnostics

### Residual Analysis

```r
# Pearson residuals
pearson_residuals <- residuals(logistic_model, type = "pearson")
deviance_residuals <- residuals(logistic_model, type = "deviance")

# Plot residuals
par(mfrow = c(2, 2))
plot(fitted(logistic_model), pearson_residuals, 
     main = "Pearson Residuals vs Fitted",
     xlab = "Fitted Values", ylab = "Pearson Residuals")
abline(h = 0, col = "red")

plot(fitted(logistic_model), deviance_residuals,
     main = "Deviance Residuals vs Fitted",
     xlab = "Fitted Values", ylab = "Deviance Residuals")
abline(h = 0, col = "red")

# Q-Q plot for deviance residuals
qqnorm(deviance_residuals, main = "Q-Q Plot of Deviance Residuals")
qqline(deviance_residuals, col = "red")

# Leverage plot
leverage <- hatvalues(logistic_model)
plot(leverage, deviance_residuals,
     main = "Deviance Residuals vs Leverage",
     xlab = "Leverage", ylab = "Deviance Residuals")
par(mfrow = c(1, 1))
```

### Influence Diagnostics

```r
# Cook's distance
cooks_dist <- cooks.distance(logistic_model)
influential_points <- which(cooks_dist > 4/length(cooks_dist))

cat("Influential points (Cook's distance > 4/n):", influential_points, "\n")

# DFBETAS
dfbetas_values <- dfbetas(logistic_model)
print(head(dfbetas_values))

# DFFITS
dffits_values <- dffits(logistic_model)
high_dffits <- which(abs(dffits_values) > 2 * sqrt(length(coef(logistic_model)) / nrow(mtcars)))
cat("High DFFITS points:", high_dffits, "\n")
```

## Prediction and Classification

### Probability Predictions

```r
# Predict probabilities
predicted_probs <- predict(logistic_model, type = "response")
print(head(predicted_probs))

# Create classification table
predicted_class <- ifelse(predicted_probs > 0.5, 1, 0)
actual_class <- mtcars$manual

classification_table <- table(Actual = actual_class, Predicted = predicted_class)
print(classification_table)

# Calculate accuracy
accuracy <- sum(diag(classification_table)) / sum(classification_table)
cat("Accuracy:", accuracy, "\n")

# Calculate sensitivity and specificity
sensitivity <- classification_table[2, 2] / sum(classification_table[2, ])
specificity <- classification_table[1, 1] / sum(classification_table[1, ])

cat("Sensitivity:", sensitivity, "\n")
cat("Specificity:", specificity, "\n")
```

### ROC Curve Analysis

```r
library(pROC)

# Create ROC curve
roc_obj <- roc(actual_class, predicted_probs)
auc_value <- auc(roc_obj)

cat("AUC:", auc_value, "\n")

# Plot ROC curve
plot(roc_obj, main = "ROC Curve",
     xlab = "1 - Specificity", ylab = "Sensitivity")
abline(a = 0, b = 1, col = "red", lty = 2)

# Find optimal threshold
optimal_threshold <- coords(roc_obj, "best")
cat("Optimal threshold:", optimal_threshold$threshold, "\n")
cat("Optimal sensitivity:", optimal_threshold$sensitivity, "\n")
cat("Optimal specificity:", optimal_threshold$specificity, "\n")
```

## Model Selection

### Stepwise Selection

```r
library(MASS)

# Forward selection
forward_model <- stepAIC(glm(manual ~ 1, data = mtcars, family = binomial),
                        direction = "forward",
                        scope = ~ mpg + wt + hp + disp + drat + qsec)

print(forward_model)

# Backward elimination
backward_model <- stepAIC(logistic_model, direction = "backward")
print(backward_model)

# Stepwise selection
stepwise_model <- stepAIC(logistic_model, direction = "both")
print(stepwise_model)
```

### Model Comparison

```r
# Compare models
model1 <- glm(manual ~ mpg, data = mtcars, family = binomial)
model2 <- glm(manual ~ mpg + wt, data = mtcars, family = binomial)
model3 <- glm(manual ~ mpg + wt + hp, data = mtcars, family = binomial)

# Likelihood ratio test
lr_test_1_2 <- anova(model1, model2, test = "Chisq")
lr_test_2_3 <- anova(model2, model3, test = "Chisq")

print(lr_test_1_2)
print(lr_test_2_3)

# AIC comparison
aic_comparison <- data.frame(
  Model = c("MPG only", "MPG + Weight", "MPG + Weight + HP"),
  AIC = c(AIC(model1), AIC(model2), AIC(model3)),
  Deviance = c(deviance(model1), deviance(model2), deviance(model3))
)

print(aic_comparison)
```

## Multinomial Logistic Regression

### Basic Multinomial Model

```r
library(nnet)

# Create multinomial outcome (cylinder types)
mtcars$cyl_factor <- factor(mtcars$cyl)

# Fit multinomial logistic regression
multinomial_model <- multinom(cyl_factor ~ mpg + wt + hp, data = mtcars)

# View model summary
summary(multinomial_model)

# Calculate predicted probabilities
predicted_probs_mult <- predict(multinomial_model, type = "probs")
print(head(predicted_probs_mult))

# Predict classes
predicted_classes <- predict(multinomial_model, type = "class")
print(table(Actual = mtcars$cyl_factor, Predicted = predicted_classes))
```

### Ordinal Logistic Regression

```r
library(MASS)

# Create ordinal outcome (cylinder categories)
mtcars$cyl_ordinal <- ordered(mtcars$cyl)

# Fit ordinal logistic regression
ordinal_model <- polr(cyl_ordinal ~ mpg + wt + hp, data = mtcars)

# View model summary
summary(ordinal_model)

# Calculate odds ratios
ordinal_odds_ratios <- exp(coef(ordinal_model))
print(ordinal_odds_ratios)
```

## Advanced Topics

### Penalized Logistic Regression

```r
library(glmnet)

# Prepare data
x <- as.matrix(mtcars[, c("mpg", "wt", "hp", "disp", "drat", "qsec")])
y <- mtcars$manual

# Ridge logistic regression
ridge_logistic <- glmnet(x, y, alpha = 0, family = "binomial")

# Cross-validation for lambda selection
cv_ridge_logistic <- cv.glmnet(x, y, alpha = 0, family = "binomial")
best_lambda_ridge <- cv_ridge_logistic$lambda.min

cat("Best lambda (Ridge):", best_lambda_ridge, "\n")

# Lasso logistic regression
lasso_logistic <- glmnet(x, y, alpha = 1, family = "binomial")
cv_lasso_logistic <- cv.glmnet(x, y, alpha = 1, family = "binomial")
best_lambda_lasso <- cv_lasso_logistic$lambda.min

cat("Best lambda (Lasso):", best_lambda_lasso, "\n")
```

### Mixed Effects Logistic Regression

```r
library(lme4)

# Simulate grouped data
set.seed(123)
n_groups <- 10
n_per_group <- 20
total_n <- n_groups * n_per_group

# Generate data with random effects
group_id <- rep(1:n_groups, each = n_per_group)
x1 <- rnorm(total_n)
x2 <- rnorm(total_n)
random_effects <- rnorm(n_groups, sd = 0.5)
group_effects <- rep(random_effects, each = n_per_group)

# Generate binary outcome
linear_predictor <- 0.5 + 0.8 * x1 - 0.6 * x2 + group_effects
prob <- 1 / (1 + exp(-linear_predictor))
y <- rbinom(total_n, 1, prob)

mixed_data <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  group_id = factor(group_id)
)

# Fit mixed effects logistic regression
mixed_logistic <- glmer(y ~ x1 + x2 + (1 | group_id), 
                       data = mixed_data, family = binomial)

summary(mixed_logistic)
```

## Practical Examples

### Example 1: Medical Diagnosis

```r
# Simulate medical diagnosis data
set.seed(123)
n_patients <- 200

# Generate patient characteristics
age <- rnorm(n_patients, mean = 55, sd = 15)
bmi <- rnorm(n_patients, mean = 25, sd = 5)
blood_pressure <- rnorm(n_patients, mean = 120, sd = 20)
cholesterol <- rnorm(n_patients, mean = 200, sd = 40)

# Generate disease probability
linear_predictor <- -2 + 0.05 * age + 0.1 * bmi + 0.02 * blood_pressure + 0.01 * cholesterol
disease_prob <- 1 / (1 + exp(-linear_predictor))
disease <- rbinom(n_patients, 1, disease_prob)

medical_data <- data.frame(
  disease = disease,
  age = age,
  bmi = bmi,
  blood_pressure = blood_pressure,
  cholesterol = cholesterol
)

# Fit logistic regression
medical_model <- glm(disease ~ age + bmi + blood_pressure + cholesterol, 
                    data = medical_data, family = binomial)

summary(medical_model)

# Calculate odds ratios
medical_odds_ratios <- exp(coef(medical_model))
print(medical_odds_ratios)

# ROC analysis
medical_probs <- predict(medical_model, type = "response")
medical_roc <- roc(medical_data$disease, medical_probs)
medical_auc <- auc(medical_roc)

cat("Medical diagnosis AUC:", medical_auc, "\n")
```

### Example 2: Marketing Response

```r
# Simulate marketing campaign data
set.seed(123)
n_customers <- 500

# Generate customer characteristics
income <- rnorm(n_customers, mean = 50000, sd = 15000)
age <- rnorm(n_customers, mean = 45, sd = 15)
previous_purchases <- rpois(n_customers, lambda = 3)
email_frequency <- rpois(n_customers, lambda = 2)

# Generate response probability
linear_predictor <- -1.5 + 0.00002 * income + 0.02 * age + 0.3 * previous_purchases - 0.2 * email_frequency
response_prob <- 1 / (1 + exp(-linear_predictor))
response <- rbinom(n_customers, 1, response_prob)

marketing_data <- data.frame(
  response = response,
  income = income,
  age = age,
  previous_purchases = previous_purchases,
  email_frequency = email_frequency
)

# Fit logistic regression
marketing_model <- glm(response ~ income + age + previous_purchases + email_frequency, 
                      data = marketing_data, family = binomial)

summary(marketing_model)

# Calculate customer lifetime value
marketing_probs <- predict(marketing_model, type = "response")
customer_value <- marketing_probs * 100  # Assume $100 value per response

cat("Average customer value:", mean(customer_value), "\n")
cat("Total campaign value:", sum(customer_value), "\n")
```

## Best Practices

### Model Validation

```r
# Function to validate logistic regression model
validate_logistic_model <- function(model, data, outcome_var) {
  # Split data into training and test sets
  set.seed(123)
  train_indices <- sample(1:nrow(data), size = 0.7 * nrow(data))
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  # Fit model on training data
  train_model <- glm(formula(model), data = train_data, family = binomial)
  
  # Predict on test data
  test_probs <- predict(train_model, newdata = test_data, type = "response")
  test_preds <- ifelse(test_probs > 0.5, 1, 0)
  
  # Calculate performance metrics
  test_actual <- test_data[[outcome_var]]
  test_table <- table(Actual = test_actual, Predicted = test_preds)
  
  accuracy <- sum(diag(test_table)) / sum(test_table)
  sensitivity <- test_table[2, 2] / sum(test_table[2, ])
  specificity <- test_table[1, 1] / sum(test_table[1, ])
  
  # ROC analysis
  test_roc <- roc(test_actual, test_probs)
  test_auc <- auc(test_roc)
  
  return(list(
    Accuracy = accuracy,
    Sensitivity = sensitivity,
    Specificity = specificity,
    AUC = test_auc,
    Confusion_Matrix = test_table
  ))
}

# Apply to our model
validation_results <- validate_logistic_model(logistic_model, mtcars, "manual")
print(validation_results)
```

### Reporting Guidelines

```r
# Function to generate logistic regression report
generate_logistic_report <- function(model, model_name = "Logistic Regression") {
  summary_stats <- summary(model)
  
  cat("=== LOGISTIC REGRESSION REPORT ===\n\n")
  cat("Model:", model_name, "\n")
  cat("Dependent variable:", all.vars(formula(model))[1], "\n")
  cat("Independent variables:", paste(all.vars(formula(model))[-1], collapse = ", "), "\n\n")
  
  # Model fit
  cat("MODEL FIT:\n")
  cat("AIC:", round(AIC(model), 3), "\n")
  cat("Deviance:", round(deviance(model), 3), "\n")
  cat("Null Deviance:", round(model$null.deviance, 3), "\n")
  
  # Pseudo R-squared
  mcfadden_r2 <- 1 - (model$deviance / model$null.deviance)
  cat("McFadden's R-squared:", round(mcfadden_r2, 3), "\n\n")
  
  # Coefficients and odds ratios
  cat("COEFFICIENTS AND ODDS RATIOS:\n")
  coef_table <- summary_stats$coefficients
  odds_ratios <- exp(coef(model))
  
  for (i in 1:nrow(coef_table)) {
    cat(rownames(coef_table)[i], ":\n")
    cat("  Coefficient:", round(coef_table[i, 1], 3), "\n")
    cat("  Std. Error:", round(coef_table[i, 2], 3), "\n")
    cat("  z-value:", round(coef_table[i, 3], 3), "\n")
    cat("  p-value:", ifelse(coef_table[i, 4] < 0.001, "< .001", 
                             round(coef_table[i, 4], 3)), "\n")
    cat("  Odds Ratio:", round(odds_ratios[i], 3), "\n\n")
  }
}

# Apply to our model
generate_logistic_report(logistic_model, "Transmission Type Model")
```

## Exercises

### Exercise 1: Basic Logistic Regression
Fit a logistic regression model to predict manual transmission using MPG and weight. Interpret the coefficients and odds ratios.

### Exercise 2: Model Diagnostics
Perform comprehensive diagnostics on your logistic regression model, including residual analysis and influence diagnostics.

### Exercise 3: Classification Performance
Evaluate the classification performance of your model using ROC analysis and calculate sensitivity, specificity, and accuracy.

### Exercise 4: Model Selection
Compare different logistic regression models using stepwise selection and information criteria.

### Exercise 5: Multinomial Regression
Fit a multinomial logistic regression model to predict cylinder type using MPG, weight, and horsepower.

## Next Steps

In the next chapter, we'll learn about model diagnostics and validation techniques.

---

**Key Takeaways:**
- Logistic regression models binary and categorical outcomes
- Odds ratios provide interpretable effect measures
- Always assess model fit and perform diagnostics
- ROC curves help evaluate classification performance
- Cross-validation is important for model validation
- Consider regularization for high-dimensional data
- Mixed effects models handle clustered data
- Always interpret results in context of the problem 