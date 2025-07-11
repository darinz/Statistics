# Model Diagnostics and Validation

## Overview

Model diagnostics are essential for ensuring the validity and reliability of statistical models. This chapter covers comprehensive diagnostic techniques for identifying problems and validating model assumptions.

## Linear Regression Diagnostics

### Basic Diagnostic Plots

```r
# Load sample data
data(mtcars)

# Fit linear regression model
model <- lm(mpg ~ wt + hp + disp, data = mtcars)

# Standard diagnostic plots
par(mfrow = c(2, 2))
plot(model)
par(mfrow = c(1, 1))

# Extract diagnostic statistics
residuals_model <- residuals(model)
fitted_values <- fitted(model)
leverage_values <- hatvalues(model)
cooks_distance <- cooks.distance(model)
```

### Residual Analysis

```r
# Function to analyze residuals
analyze_residuals <- function(model) {
  residuals <- residuals(model)
  fitted <- fitted(model)
  
  # Basic statistics
  cat("=== RESIDUAL ANALYSIS ===\n")
  cat("Mean:", mean(residuals), "\n")
  cat("Standard deviation:", sd(residuals), "\n")
  cat("Skewness:", skewness(residuals), "\n")
  cat("Kurtosis:", kurtosis(residuals), "\n")
  
  # Normality test
  shapiro_test <- shapiro.test(residuals)
  cat("Shapiro-Wilk test p-value:", shapiro_test$p.value, "\n")
  
  # Independence test (Durbin-Watson)
  library(lmtest)
  dw_test <- dwtest(model)
  cat("Durbin-Watson test p-value:", dw_test$p.value, "\n")
  
  # Homoscedasticity test
  bp_test <- bptest(model)
  cat("Breusch-Pagan test p-value:", bp_test$p.value, "\n")
  
  return(list(
    residuals = residuals,
    fitted = fitted,
    shapiro_p = shapiro_test$p.value,
    dw_p = dw_test$p.value,
    bp_p = bp_test$p.value
  ))
}

# Apply to our model
residual_analysis <- analyze_residuals(model)
```

### Influence Diagnostics

```r
# Function to analyze influence measures
analyze_influence <- function(model) {
  # Cook's distance
  cooks_dist <- cooks.distance(model)
  influential_cooks <- which(cooks_dist > 4/length(cooks_dist))
  
  # Leverage
  leverage <- hatvalues(model)
  high_leverage <- which(leverage > 2 * (length(coef(model)) + 1) / nrow(model$model))
  
  # DFFITS
  dffits_values <- dffits(model)
  high_dffits <- which(abs(dffits_values) > 2 * sqrt(length(coef(model)) / nrow(model$model)))
  
  # DFBETAS
  dfbetas_values <- dfbetas(model)
  high_dfbetas <- which(apply(abs(dfbetas_values), 1, max) > 2/sqrt(nrow(model$model)))
  
  cat("=== INFLUENCE DIAGNOSTICS ===\n")
  cat("Influential points (Cook's distance):", influential_cooks, "\n")
  cat("High leverage points:", high_leverage, "\n")
  cat("High DFFITS points:", high_dffits, "\n")
  cat("High DFBETAS points:", high_dfbetas, "\n")
  
  return(list(
    cooks_distance = cooks_dist,
    leverage = leverage,
    dffits = dffits_values,
    dfbetas = dfbetas_values,
    influential_points = unique(c(influential_cooks, high_leverage, high_dffits, high_dfbetas))
  ))
}

# Apply to our model
influence_analysis <- analyze_influence(model)
```

### Multicollinearity Detection

```r
# Function to detect multicollinearity
detect_multicollinearity <- function(model) {
  library(car)
  
  # Variance Inflation Factor
  vif_values <- vif(model)
  
  # Tolerance
  tolerance <- 1 / vif_values
  
  # Condition number
  x_matrix <- model.matrix(model)
  eigen_values <- eigen(t(x_matrix) %*% x_matrix)$values
  condition_number <- sqrt(max(eigen_values) / min(eigen_values))
  
  cat("=== MULTICOLLINEARITY DIAGNOSTICS ===\n")
  cat("VIF values:\n")
  print(vif_values)
  cat("Tolerance values:\n")
  print(tolerance)
  cat("Condition number:", condition_number, "\n")
  
  # Interpretation
  high_vif <- which(vif_values > 10)
  if (length(high_vif) > 0) {
    cat("High VIF variables:", names(vif_values)[high_vif], "\n")
  }
  
  if (condition_number > 30) {
    cat("Condition number suggests multicollinearity\n")
  }
  
  return(list(
    vif = vif_values,
    tolerance = tolerance,
    condition_number = condition_number,
    high_vif_vars = names(vif_values)[high_vif]
  ))
}

# Apply to our model
multicollinearity_analysis <- detect_multicollinearity(model)
```

## Logistic Regression Diagnostics

### Logistic Model Diagnostics

```r
# Fit logistic regression model
mtcars$manual <- ifelse(mtcars$am == 1, 1, 0)
logistic_model <- glm(manual ~ mpg + wt + hp, data = mtcars, family = binomial)

# Function to analyze logistic residuals
analyze_logistic_residuals <- function(model) {
  # Pearson residuals
  pearson_residuals <- residuals(model, type = "pearson")
  deviance_residuals <- residuals(model, type = "deviance")
  
  # Hosmer-Lemeshow test
  library(ResourceSelection)
  hl_test <- hoslem.test(model$y, fitted(model))
  
  # Pseudo R-squared
  mcfadden_r2 <- 1 - (model$deviance / model$null.deviance)
  
  cat("=== LOGISTIC REGRESSION DIAGNOSTICS ===\n")
  cat("Hosmer-Lemeshow test p-value:", hl_test$p.value, "\n")
  cat("McFadden's R-squared:", mcfadden_r2, "\n")
  cat("Deviance:", model$deviance, "\n")
  cat("Null Deviance:", model$null.deviance, "\n")
  
  return(list(
    pearson_residuals = pearson_residuals,
    deviance_residuals = deviance_residuals,
    hosmer_lemeshow_p = hl_test$p.value,
    mcfadden_r2 = mcfadden_r2
  ))
}

# Apply to logistic model
logistic_diagnostics <- analyze_logistic_residuals(logistic_model)
```

### Classification Performance

```r
# Function to evaluate classification performance
evaluate_classification <- function(model, data, outcome_var, threshold = 0.5) {
  # Predict probabilities
  predicted_probs <- predict(model, type = "response")
  predicted_class <- ifelse(predicted_probs > threshold, 1, 0)
  actual_class <- data[[outcome_var]]
  
  # Confusion matrix
  confusion_matrix <- table(Actual = actual_class, Predicted = predicted_class)
  
  # Performance metrics
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  sensitivity <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
  specificity <- confusion_matrix[1, 1] / sum(confusion_matrix[1, ])
  precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
  f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
  
  # ROC analysis
  library(pROC)
  roc_obj <- roc(actual_class, predicted_probs)
  auc_value <- auc(roc_obj)
  
  cat("=== CLASSIFICATION PERFORMANCE ===\n")
  cat("Confusion Matrix:\n")
  print(confusion_matrix)
  cat("Accuracy:", accuracy, "\n")
  cat("Sensitivity:", sensitivity, "\n")
  cat("Specificity:", specificity, "\n")
  cat("Precision:", precision, "\n")
  cat("F1 Score:", f1_score, "\n")
  cat("AUC:", auc_value, "\n")
  
  return(list(
    confusion_matrix = confusion_matrix,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    f1_score = f1_score,
    auc = auc_value,
    roc_obj = roc_obj
  ))
}

# Apply to logistic model
classification_performance <- evaluate_classification(logistic_model, mtcars, "manual")
```

## Model Validation Techniques

### Cross-Validation

```r
# Function to perform cross-validation
perform_cross_validation <- function(model, data, k = 5, model_type = "linear") {
  set.seed(123)
  n <- nrow(data)
  folds <- sample(rep(1:k, length.out = n))
  cv_errors <- numeric(k)
  cv_predictions <- numeric(n)
  
  for (i in 1:k) {
    test_indices <- which(folds == i)
    train_data <- data[-test_indices, ]
    test_data <- data[test_indices, ]
    
    # Fit model on training data
    if (model_type == "linear") {
      train_model <- lm(formula(model), data = train_data)
      predictions <- predict(train_model, newdata = test_data)
      cv_errors[i] <- mean((test_data[[all.vars(formula(model))[1]]] - predictions)^2)
    } else if (model_type == "logistic") {
      train_model <- glm(formula(model), data = train_data, family = binomial)
      predictions <- predict(train_model, newdata = test_data, type = "response")
      cv_errors[i] <- mean((test_data[[all.vars(formula(model))[1]]] - predictions)^2)
    }
    
    cv_predictions[test_indices] <- predictions
  }
  
  cat("=== CROSS-VALIDATION RESULTS ===\n")
  cat("Mean CV MSE:", mean(cv_errors), "\n")
  cat("CV MSE Standard Error:", sd(cv_errors) / sqrt(k), "\n")
  cat("CV MSE by fold:\n")
  print(cv_errors)
  
  return(list(
    cv_errors = cv_errors,
    mean_cv_error = mean(cv_errors),
    cv_se = sd(cv_errors) / sqrt(k),
    cv_predictions = cv_predictions
  ))
}

# Apply to linear model
cv_linear <- perform_cross_validation(model, mtcars, k = 5, "linear")

# Apply to logistic model
cv_logistic <- perform_cross_validation(logistic_model, mtcars, k = 5, "logistic")
```

### Bootstrap Validation

```r
# Function to perform bootstrap validation
bootstrap_validation <- function(model, data, n_bootstrap = 1000) {
  library(boot)
  
  # Bootstrap function for MSE
  boot_mse <- function(data, indices) {
    d <- data[indices, ]
    fit <- lm(formula(model), data = d)
    predictions <- predict(fit, newdata = data)
    return(mean((data[[all.vars(formula(model))[1]]] - predictions)^2))
  }
  
  # Perform bootstrap
  boot_results <- boot(mtcars, boot_mse, R = n_bootstrap)
  
  cat("=== BOOTSTRAP VALIDATION ===\n")
  cat("Bootstrap MSE:", boot_results$t0, "\n")
  cat("Bootstrap MSE SE:", sd(boot_results$t), "\n")
  cat("Bootstrap MSE 95% CI:", quantile(boot_results$t, c(0.025, 0.975)), "\n")
  
  return(list(
    bootstrap_mse = boot_results$t0,
    bootstrap_se = sd(boot_results$t),
    bootstrap_ci = quantile(boot_results$t, c(0.025, 0.975)),
    bootstrap_samples = boot_results$t
  ))
}

# Apply to our model
bootstrap_results <- bootstrap_validation(model, mtcars)
```

## Advanced Diagnostics

### Partial Residual Plots

```r
# Function to create partial residual plots
partial_residual_plots <- function(model) {
  library(car)
  
  # Create partial residual plots
  par(mfrow = c(2, 2))
  crPlots(model)
  par(mfrow = c(1, 1))
  
  # Component + residual plots
  par(mfrow = c(2, 2))
  crPlots(model, layout = c(2, 2))
  par(mfrow = c(1, 1))
}

# Apply to our model
partial_residual_plots(model)
```

### Added Variable Plots

```r
# Function to create added variable plots
added_variable_plots <- function(model) {
  library(car)
  
  # Create added variable plots
  par(mfrow = c(2, 2))
  avPlots(model)
  par(mfrow = c(1, 1))
}

# Apply to our model
added_variable_plots(model)
```

### Influence Plot

```r
# Function to create comprehensive influence plot
influence_plot <- function(model) {
  library(car)
  
  # Create influence plot
  influencePlot(model, id.method = "identify", 
                main = "Influence Plot",
                sub = "Circle size is proportional to Cook's distance")
}

# Apply to our model
influence_plot(model)
```

## Model Comparison

### Information Criteria

```r
# Function to compare models using information criteria
compare_models <- function(model_list, model_names) {
  # Calculate information criteria
  aic_values <- sapply(model_list, AIC)
  bic_values <- sapply(model_list, BIC)
  
  # For linear models, also calculate R-squared
  r_squared_values <- sapply(model_list, function(m) {
    if (inherits(m, "lm")) {
      return(summary(m)$r.squared)
    } else {
      return(NA)
    }
  })
  
  # Create comparison table
  comparison_table <- data.frame(
    Model = model_names,
    AIC = aic_values,
    BIC = bic_values,
    R_squared = r_squared_values
  )
  
  cat("=== MODEL COMPARISON ===\n")
  print(comparison_table)
  
  # Find best model by each criterion
  best_aic <- model_names[which.min(aic_values)]
  best_bic <- model_names[which.min(bic_values)]
  best_r2 <- model_names[which.max(r_squared_values)]
  
  cat("Best model by AIC:", best_aic, "\n")
  cat("Best model by BIC:", best_bic, "\n")
  cat("Best model by R-squared:", best_r2, "\n")
  
  return(comparison_table)
}

# Create different models for comparison
model1 <- lm(mpg ~ wt, data = mtcars)
model2 <- lm(mpg ~ wt + hp, data = mtcars)
model3 <- lm(mpg ~ wt + hp + disp, data = mtcars)

model_list <- list(model1, model2, model3)
model_names <- c("MPG ~ Weight", "MPG ~ Weight + HP", "MPG ~ Weight + HP + Disp")

comparison_results <- compare_models(model_list, model_names)
```

### Likelihood Ratio Tests

```r
# Function to perform likelihood ratio tests
likelihood_ratio_tests <- function(model_list, model_names) {
  cat("=== LIKELIHOOD RATIO TESTS ===\n")
  
  for (i in 2:length(model_list)) {
    lr_test <- anova(model_list[[i-1]], model_list[[i]], test = "Chisq")
    cat("Comparing", model_names[i-1], "vs", model_names[i], ":\n")
    cat("  Chi-square:", lr_test$`Pr(>Chi)`[2], "\n")
    cat("  p-value:", lr_test$`Pr(>Chi)`[2], "\n\n")
  }
}

# Apply to our models
likelihood_ratio_tests(model_list, model_names)
```

## Practical Examples

### Example 1: Comprehensive Model Diagnostics

```r
# Function to perform comprehensive diagnostics
comprehensive_diagnostics <- function(model, data, model_name = "Model") {
  cat("=== COMPREHENSIVE DIAGNOSTICS FOR", model_name, "===\n\n")
  
  # 1. Residual analysis
  residual_analysis <- analyze_residuals(model)
  
  # 2. Influence analysis
  influence_analysis <- analyze_influence(model)
  
  # 3. Multicollinearity (for linear models)
  if (inherits(model, "lm")) {
    multicollinearity_analysis <- detect_multicollinearity(model)
  }
  
  # 4. Cross-validation
  cv_results <- perform_cross_validation(model, data, k = 5, 
                                       model_type = ifelse(inherits(model, "lm"), "linear", "logistic"))
  
  # 5. Model fit summary
  if (inherits(model, "lm")) {
    cat("R-squared:", summary(model)$r.squared, "\n")
    cat("Adjusted R-squared:", summary(model)$adj.r.squared, "\n")
  } else if (inherits(model, "glm")) {
    cat("McFadden's R-squared:", 1 - (model$deviance / model$null.deviance), "\n")
  }
  
  # 6. Recommendations
  cat("\n=== RECOMMENDATIONS ===\n")
  
  if (residual_analysis$shapiro_p < 0.05) {
    cat("- Residuals are not normally distributed\n")
  }
  
  if (residual_analysis$bp_p < 0.05) {
    cat("- Heteroscedasticity detected\n")
  }
  
  if (length(influence_analysis$influential_points) > 0) {
    cat("- Influential points detected\n")
  }
  
  if (inherits(model, "lm") && length(multicollinearity_analysis$high_vif_vars) > 0) {
    cat("- Multicollinearity detected\n")
  }
}

# Apply to our linear model
comprehensive_diagnostics(model, mtcars, "Linear Regression Model")

# Apply to our logistic model
comprehensive_diagnostics(logistic_model, mtcars, "Logistic Regression Model")
```

### Example 2: Model Validation Workflow

```r
# Function to create complete validation workflow
validation_workflow <- function(model, data, outcome_var, model_type = "linear") {
  cat("=== MODEL VALIDATION WORKFLOW ===\n\n")
  
  # 1. Split data
  set.seed(123)
  train_indices <- sample(1:nrow(data), size = 0.7 * nrow(data))
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  # 2. Fit model on training data
  if (model_type == "linear") {
    train_model <- lm(formula(model), data = train_data)
  } else {
    train_model <- glm(formula(model), data = train_data, family = binomial)
  }
  
  # 3. Predict on test data
  if (model_type == "linear") {
    test_predictions <- predict(train_model, newdata = test_data)
    test_mse <- mean((test_data[[outcome_var]] - test_predictions)^2)
    cat("Test MSE:", test_mse, "\n")
  } else {
    test_predictions <- predict(train_model, newdata = test_data, type = "response")
    test_auc <- auc(roc(test_data[[outcome_var]], test_predictions))
    cat("Test AUC:", test_auc, "\n")
  }
  
  # 4. Cross-validation
  cv_results <- perform_cross_validation(model, data, k = 5, model_type)
  
  # 5. Bootstrap validation
  bootstrap_results <- bootstrap_validation(model, data)
  
  # 6. Compare training vs test performance
  if (model_type == "linear") {
    train_mse <- mean(residuals(train_model)^2)
    cat("Training MSE:", train_mse, "\n")
    cat("Test MSE:", test_mse, "\n")
    cat("Overfitting ratio:", train_mse / test_mse, "\n")
  }
  
  return(list(
    train_model = train_model,
    test_predictions = test_predictions,
    cv_results = cv_results,
    bootstrap_results = bootstrap_results
  ))
}

# Apply to linear model
linear_validation <- validation_workflow(model, mtcars, "mpg", "linear")

# Apply to logistic model
logistic_validation <- validation_workflow(logistic_model, mtcars, "manual", "logistic")
```

## Best Practices

### Diagnostic Checklist

```r
# Function to create diagnostic checklist
diagnostic_checklist <- function(model, data) {
  cat("=== DIAGNOSTIC CHECKLIST ===\n\n")
  
  # 1. Residual analysis
  residuals <- residuals(model)
  cat("1. Residual Analysis:\n")
  cat("   - Normality: Shapiro-Wilk p =", shapiro.test(residuals)$p.value, "\n")
  cat("   - Independence: Durbin-Watson p =", dwtest(model)$p.value, "\n")
  cat("   - Homoscedasticity: Breusch-Pagan p =", bptest(model)$p.value, "\n\n")
  
  # 2. Influence analysis
  cooks_dist <- cooks.distance(model)
  influential_count <- sum(cooks_dist > 4/length(cooks_dist))
  cat("2. Influence Analysis:\n")
  cat("   - Influential points:", influential_count, "\n")
  cat("   - High leverage points:", sum(hatvalues(model) > 2 * (length(coef(model)) + 1) / nrow(data)), "\n\n")
  
  # 3. Multicollinearity (for linear models)
  if (inherits(model, "lm")) {
    vif_values <- vif(model)
    high_vif_count <- sum(vif_values > 10)
    cat("3. Multicollinearity:\n")
    cat("   - High VIF variables:", high_vif_count, "\n")
    cat("   - Max VIF:", max(vif_values), "\n\n")
  }
  
  # 4. Model fit
  if (inherits(model, "lm")) {
    cat("4. Model Fit:\n")
    cat("   - R-squared:", summary(model)$r.squared, "\n")
    cat("   - Adjusted R-squared:", summary(model)$adj.r.squared, "\n")
  } else if (inherits(model, "glm")) {
    cat("4. Model Fit:\n")
    cat("   - McFadden's R-squared:", 1 - (model$deviance / model$null.deviance), "\n")
  }
  
  # 5. Recommendations
  cat("5. Recommendations:\n")
  if (shapiro.test(residuals)$p.value < 0.05) {
    cat("   - Consider robust regression or transformations\n")
  }
  if (bptest(model)$p.value < 0.05) {
    cat("   - Consider weighted least squares\n")
  }
  if (influential_count > 0) {
    cat("   - Investigate influential points\n")
  }
  if (inherits(model, "lm") && max(vif_values) > 10) {
    cat("   - Consider removing highly correlated predictors\n")
  }
}

# Apply to our model
diagnostic_checklist(model, mtcars)
```

## Exercises

### Exercise 1: Comprehensive Diagnostics
Perform comprehensive diagnostics on a multiple regression model of your choice, including residual analysis, influence diagnostics, and multicollinearity checks.

### Exercise 2: Cross-Validation
Implement k-fold cross-validation for both linear and logistic regression models and compare the results.

### Exercise 3: Model Comparison
Compare different model specifications using information criteria and likelihood ratio tests.

### Exercise 4: Bootstrap Validation
Use bootstrap methods to validate your model and assess the stability of parameter estimates.

### Exercise 5: Diagnostic Reporting
Create a comprehensive diagnostic report for a regression model, including all relevant tests and recommendations.

## Next Steps

In the next chapter, we'll learn about nonparametric statistics and robust methods.

---

**Key Takeaways:**
- Always perform comprehensive diagnostics before interpreting results
- Check assumptions systematically using appropriate tests
- Use multiple validation techniques to assess model performance
- Consider the impact of influential points and outliers
- Be aware of multicollinearity in multiple regression
- Cross-validation provides unbiased performance estimates
- Bootstrap methods assess parameter stability
- Diagnostic results should guide model improvements 