# Variable Selection and Model Building

## Overview

Variable selection and model building are crucial steps in statistical modeling that help identify the most important predictors while avoiding overfitting. This chapter covers various methods for selecting variables, building optimal models, and validating model performance.

## Model Building Process

### Understanding the Modeling Workflow

```r
# Load required packages
library(tidyverse)
library(caret)
library(glmnet)
library(MASS)
library(leaps)
library(olsrr)

# Set seed for reproducibility
set.seed(123)

# Generate sample data for demonstration
n_samples <- 200
n_predictors <- 10

# Create correlated predictors
correlation_matrix <- matrix(0.3, nrow = n_predictors, ncol = n_predictors)
diag(correlation_matrix) <- 1

# Generate predictor variables
X <- MASS::mvrnorm(n_samples, mu = rep(0, n_predictors), Sigma = correlation_matrix)
colnames(X) <- paste0("X", 1:n_predictors)

# Create response variable with some predictors being important
beta_true <- c(2, 1.5, 0, 0, 0.8, 0, 0, 0, 0, 0)  # Only X1, X2, X5 are important
y <- X %*% beta_true + rnorm(n_samples, 0, 1)

# Create data frame
model_data <- data.frame(y = y, X)
print("Sample of the data:")
print(head(model_data))
```

### Data Splitting for Model Building

```r
# Split data into training and testing sets
train_index <- createDataPartition(model_data$y, p = 0.7, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")

# Function to calculate model performance metrics
calculate_performance <- function(actual, predicted) {
  mse <- mean((actual - predicted)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(actual - predicted))
  r_squared <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  
  return(list(
    MSE = mse,
    RMSE = rmse,
    MAE = mae,
    R_squared = r_squared
  ))
}
```

## Stepwise Selection Methods

### Forward Selection

```r
# Forward selection using leaps package
forward_model <- regsubsets(y ~ ., data = train_data, nvmax = n_predictors, 
                           method = "forward")

# Summary of forward selection
forward_summary <- summary(forward_model)
print("Forward Selection Results:")
print(forward_summary)

# Extract best models for different numbers of variables
forward_best <- which.min(forward_summary$bic)
cat("Best model by BIC has", forward_best, "variables\n")

# Get coefficients for best model
best_forward_coef <- coef(forward_model, forward_best)
print("Best Forward Selection Coefficients:")
print(round(best_forward_coef, 3))

# Plot selection criteria
par(mfrow = c(2, 2))
plot(forward_summary$rss, xlab = "Number of Variables", ylab = "RSS", 
     main = "RSS vs Number of Variables")
plot(forward_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted R²", 
     main = "Adjusted R² vs Number of Variables")
plot(forward_summary$bic, xlab = "Number of Variables", ylab = "BIC", 
     main = "BIC vs Number of Variables")
plot(forward_summary$cp, xlab = "Number of Variables", ylab = "Cp", 
     main = "Cp vs Number of Variables")
par(mfrow = c(1, 1))
```

### Backward Elimination

```r
# Backward elimination
backward_model <- regsubsets(y ~ ., data = train_data, nvmax = n_predictors, 
                            method = "backward")

# Summary of backward elimination
backward_summary <- summary(backward_model)
print("Backward Elimination Results:")
print(backward_summary)

# Extract best models
backward_best <- which.min(backward_summary$bic)
cat("Best model by BIC has", backward_best, "variables\n")

# Get coefficients for best model
best_backward_coef <- coef(backward_model, backward_best)
print("Best Backward Elimination Coefficients:")
print(round(best_backward_coef, 3))

# Compare forward vs backward
cat("Comparison of Forward vs Backward Selection:\n")
cat("Forward selection best model size:", forward_best, "\n")
cat("Backward elimination best model size:", backward_best, "\n")
```

### Stepwise Selection

```r
# Stepwise selection (both directions)
stepwise_model <- regsubsets(y ~ ., data = train_data, nvmax = n_predictors, 
                            method = "seqrep")

# Summary of stepwise selection
stepwise_summary <- summary(stepwise_model)
print("Stepwise Selection Results:")
print(stepwise_summary)

# Extract best models
stepwise_best <- which.min(stepwise_summary$bic)
cat("Best model by BIC has", stepwise_best, "variables\n")

# Get coefficients for best model
best_stepwise_coef <- coef(stepwise_model, stepwise_best)
print("Best Stepwise Selection Coefficients:")
print(round(best_stepwise_coef, 3))

# Compare all stepwise methods
comparison_df <- data.frame(
  Method = c("Forward", "Backward", "Stepwise"),
  Best_Size = c(forward_best, backward_best, stepwise_best),
  BIC = c(forward_summary$bic[forward_best], 
           backward_summary$bic[backward_best], 
           stepwise_summary$bic[stepwise_best]),
  Adj_R2 = c(forward_summary$adjr2[forward_best], 
              backward_summary$adjr2[backward_best], 
              stepwise_summary$adjr2[stepwise_best])
)

print("Comparison of Stepwise Methods:")
print(round(comparison_df, 3))
```

## Regularization Methods

### Ridge Regression

```r
# Prepare data for glmnet (matrix format)
X_train <- as.matrix(train_data[, -1])
y_train <- train_data$y
X_test <- as.matrix(test_data[, -1])
y_test <- test_data$y

# Ridge regression
ridge_model <- glmnet(X_train, y_train, alpha = 0, lambda = NULL)

# Cross-validation for optimal lambda
ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0)
print("Ridge Regression CV Results:")
print(ridge_cv)

# Optimal lambda
optimal_lambda_ridge <- ridge_cv$lambda.min
cat("Optimal lambda (ridge):", round(optimal_lambda_ridge, 4), "\n")

# Ridge coefficients
ridge_coef <- coef(ridge_model, s = optimal_lambda_ridge)
print("Ridge Regression Coefficients:")
print(round(as.matrix(ridge_coef), 3))

# Predictions
ridge_pred <- predict(ridge_model, newx = X_test, s = optimal_lambda_ridge)
ridge_performance <- calculate_performance(y_test, ridge_pred)
cat("Ridge Regression Performance:\n")
print(round(ridge_performance, 3))
```

### Lasso Regression

```r
# Lasso regression
lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = NULL)

# Cross-validation for optimal lambda
lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1)
print("Lasso Regression CV Results:")
print(lasso_cv)

# Optimal lambda
optimal_lambda_lasso <- lasso_cv$lambda.min
cat("Optimal lambda (lasso):", round(optimal_lambda_lasso, 4), "\n")

# Lasso coefficients
lasso_coef <- coef(lasso_model, s = optimal_lambda_lasso)
print("Lasso Regression Coefficients:")
print(round(as.matrix(lasso_coef), 3))

# Number of non-zero coefficients
n_nonzero_lasso <- sum(lasso_coef != 0) - 1  # Exclude intercept
cat("Number of non-zero coefficients (Lasso):", n_nonzero_lasso, "\n")

# Predictions
lasso_pred <- predict(lasso_model, newx = X_test, s = optimal_lambda_lasso)
lasso_performance <- calculate_performance(y_test, lasso_pred)
cat("Lasso Regression Performance:\n")
print(round(lasso_performance, 3))
```

### Elastic Net

```r
# Elastic net (alpha = 0.5)
elastic_model <- glmnet(X_train, y_train, alpha = 0.5, lambda = NULL)

# Cross-validation for optimal lambda
elastic_cv <- cv.glmnet(X_train, y_train, alpha = 0.5)
print("Elastic Net CV Results:")
print(elastic_cv)

# Optimal lambda
optimal_lambda_elastic <- elastic_cv$lambda.min
cat("Optimal lambda (elastic net):", round(optimal_lambda_elastic, 4), "\n")

# Elastic net coefficients
elastic_coef <- coef(elastic_model, s = optimal_lambda_elastic)
print("Elastic Net Coefficients:")
print(round(as.matrix(elastic_coef), 3))

# Number of non-zero coefficients
n_nonzero_elastic <- sum(elastic_coef != 0) - 1  # Exclude intercept
cat("Number of non-zero coefficients (Elastic Net):", n_nonzero_elastic, "\n")

# Predictions
elastic_pred <- predict(elastic_model, newx = X_test, s = optimal_lambda_elastic)
elastic_performance <- calculate_performance(y_test, elastic_pred)
cat("Elastic Net Performance:\n")
print(round(elastic_performance, 3))
```

## Model Comparison and Selection

### Information Criteria

```r
# Function to calculate information criteria
calculate_info_criteria <- function(model, data, k = 2) {
  # AIC: k = 2, BIC: k = log(n)
  n <- nrow(data)
  p <- length(coef(model)) - 1  # Number of predictors (exclude intercept)
  
  # Calculate RSS
  y_pred <- predict(model, data)
  y_actual <- data$y
  rss <- sum((y_actual - y_pred)^2)
  
  # Calculate criteria
  aic <- n * log(rss/n) + k * p
  bic <- n * log(rss/n) + log(n) * p
  
  return(list(AIC = aic, BIC = bic, RSS = rss))
}

# Fit full model for comparison
full_model <- lm(y ~ ., data = train_data)
full_criteria <- calculate_info_criteria(full_model, train_data)

cat("Full Model Information Criteria:\n")
print(round(full_criteria, 3))
```

### Cross-Validation for Model Selection

```r
# Function to perform k-fold CV for different models
cv_model_selection <- function(data, k = 5) {
  n <- nrow(data)
  fold_size <- floor(n / k)
  
  # Initialize results storage
  cv_results <- list()
  
  # Different model sizes to test
  model_sizes <- 1:min(10, ncol(data) - 1)
  
  for (size in model_sizes) {
    mse_folds <- numeric(k)
    
    for (fold in 1:k) {
      # Create fold indices
      start_idx <- (fold - 1) * fold_size + 1
      end_idx <- ifelse(fold == k, n, fold * fold_size)
      test_indices <- start_idx:end_idx
      
      # Split data
      train_fold <- data[-test_indices, ]
      test_fold <- data[test_indices, ]
      
      # Fit model with stepwise selection
      stepwise_fold <- regsubsets(y ~ ., data = train_fold, nvmax = size, 
                                 method = "seqrep")
      
      # Get best model of this size
      best_model_idx <- which.min(summary(stepwise_fold)$bic)
      
      # Get coefficients
      coef_fold <- coef(stepwise_fold, best_model_idx)
      
      # Make predictions
      X_test_fold <- as.matrix(test_fold[, -1])
      pred_fold <- X_test_fold %*% coef_fold[-1] + coef_fold[1]
      
      # Calculate MSE
      mse_folds[fold] <- mean((test_fold$y - pred_fold)^2)
    }
    
    cv_results[[paste0("size_", size)]] <- mean(mse_folds)
  }
  
  return(cv_results)
}

# Perform cross-validation
cv_results <- cv_model_selection(train_data, k = 5)
print("Cross-Validation Results:")
print(round(unlist(cv_results), 4))

# Find best model size
best_cv_size <- which.min(unlist(cv_results))
cat("Best model size by CV:", best_cv_size, "\n")
```

## Advanced Variable Selection Techniques

### Principal Component Regression (PCR)

```r
# Principal Component Analysis
pca_result <- prcomp(X_train, center = TRUE, scale = TRUE)

# Explained variance
explained_var <- pca_result$sdev^2 / sum(pca_result$sdev^2)
cumulative_var <- cumsum(explained_var)

cat("PCA Results:\n")
cat("Explained variance by first 3 components:", round(sum(explained_var[1:3]), 3), "\n")

# PCR with different numbers of components
pcr_performance <- list()

for (n_comp in 1:5) {
  # Use first n_comp principal components
  pc_scores <- pca_result$x[, 1:n_comp, drop = FALSE]
  
  # Fit regression on PC scores
  pcr_model <- lm(y_train ~ pc_scores)
  
  # Transform test data
  pc_scores_test <- predict(pca_result, X_test)[, 1:n_comp, drop = FALSE]
  
  # Make predictions
  pcr_pred <- predict(pcr_model, newdata = data.frame(pc_scores_test))
  
  # Calculate performance
  pcr_performance[[paste0("PC", n_comp)]] <- calculate_performance(y_test, pcr_pred)
}

print("PCR Performance Comparison:")
for (i in 1:length(pcr_performance)) {
  cat(names(pcr_performance)[i], "RMSE:", round(pcr_performance[[i]]$RMSE, 3), "\n")
}
```

### Partial Least Squares (PLS)

```r
# PLS regression using pls package
library(pls)

# Fit PLS model
pls_model <- plsr(y ~ ., data = train_data, ncomp = 5, validation = "CV")

# Summary
print("PLS Model Summary:")
print(summary(pls_model))

# Cross-validation results
pls_cv <- crossval(pls_model, segments = 5)
print("PLS Cross-Validation Results:")
print(pls_cv)

# Optimal number of components
optimal_comp <- which.min(pls_cv$val$PRESS)
cat("Optimal number of components:", optimal_comp, "\n")

# Predictions with optimal model
pls_pred <- predict(pls_model, newdata = test_data, ncomp = optimal_comp)
pls_performance <- calculate_performance(y_test, pls_pred)

cat("PLS Performance:\n")
print(round(pls_performance, 3))
```

## Model Validation and Diagnostics

### Residual Analysis

```r
# Function for comprehensive residual analysis
residual_analysis <- function(model, data, title = "Model") {
  # Get residuals
  residuals <- residuals(model)
  fitted_values <- fitted(model)
  
  # Create diagnostic plots
  par(mfrow = c(2, 2))
  
  # Residuals vs Fitted
  plot(fitted_values, residuals, main = paste(title, "- Residuals vs Fitted"),
       xlab = "Fitted Values", ylab = "Residuals")
  abline(h = 0, col = "red", lty = 2)
  
  # Q-Q plot
  qqnorm(residuals, main = paste(title, "- Q-Q Plot"))
  qqline(residuals, col = "red")
  
  # Histogram of residuals
  hist(residuals, main = paste(title, "- Residuals Distribution"),
       xlab = "Residuals", freq = FALSE)
  curve(dnorm(x, mean = mean(residuals), sd = sd(residuals)), 
        add = TRUE, col = "red")
  
  # Residuals vs Index
  plot(residuals, main = paste(title, "- Residuals vs Index"),
       xlab = "Index", ylab = "Residuals")
  abline(h = 0, col = "red", lty = 2)
  
  par(mfrow = c(1, 1))
  
  # Statistical tests
  shapiro_test <- shapiro.test(residuals)
  bp_test <- ncvTest(model)
  
  cat("Residual Analysis for", title, ":\n")
  cat("Shapiro-Wilk test p-value:", round(shapiro_test$p.value, 4), "\n")
  cat("Breusch-Pagan test p-value:", round(bp_test$p.value, 4), "\n")
  
  return(list(
    residuals = residuals,
    fitted = fitted_values,
    shapiro_p = shapiro_test$p.value,
    bp_p = bp_test$p.value
  ))
}

# Apply to full model
full_residuals <- residual_analysis(full_model, train_data, "Full Model")
```

### Multicollinearity Detection

```r
# Variance Inflation Factor (VIF)
library(car)

# Calculate VIF for full model
vif_values <- vif(full_model)
print("Variance Inflation Factors:")
print(round(vif_values, 3))

# Identify problematic variables
high_vif <- which(vif_values > 5)
if (length(high_vif) > 0) {
  cat("Variables with high VIF (> 5):\n")
  print(names(high_vif))
} else {
  cat("No variables with high VIF detected.\n")
}

# Condition number
X_matrix <- as.matrix(train_data[, -1])
eigen_values <- eigen(t(X_matrix) %*% X_matrix)$values
condition_number <- sqrt(max(eigen_values) / min(eigen_values))
cat("Condition number:", round(condition_number, 2), "\n")

if (condition_number > 30) {
  cat("Warning: High multicollinearity detected.\n")
} else {
  cat("Multicollinearity appears acceptable.\n")
}
```

## Best Practices and Guidelines

### Model Selection Strategy

```r
# Comprehensive model selection function
comprehensive_model_selection <- function(data, test_data = NULL) {
  cat("=== COMPREHENSIVE MODEL SELECTION ===\n\n")
  
  # 1. Stepwise selection
  cat("1. STEPWISE SELECTION:\n")
  stepwise_model <- regsubsets(y ~ ., data = data, nvmax = ncol(data) - 1, 
                              method = "seqrep")
  stepwise_summary <- summary(stepwise_model)
  best_stepwise <- which.min(stepwise_summary$bic)
  cat("   Best stepwise model size:", best_stepwise, "\n")
  cat("   BIC:", round(stepwise_summary$bic[best_stepwise], 2), "\n\n")
  
  # 2. Ridge regression
  cat("2. RIDGE REGRESSION:\n")
  X <- as.matrix(data[, -1])
  y <- data$y
  ridge_cv <- cv.glmnet(X, y, alpha = 0)
  cat("   Optimal lambda:", round(ridge_cv$lambda.min, 4), "\n")
  cat("   CV MSE:", round(min(ridge_cv$cvm), 4), "\n\n")
  
  # 3. Lasso regression
  cat("3. LASSO REGRESSION:\n")
  lasso_cv <- cv.glmnet(X, y, alpha = 1)
  cat("   Optimal lambda:", round(lasso_cv$lambda.min, 4), "\n")
  cat("   CV MSE:", round(min(lasso_cv$cvm), 4), "\n")
  
  # Count non-zero coefficients
  lasso_coef <- coef(lasso_cv, s = lasso_cv$lambda.min)
  n_nonzero <- sum(lasso_coef != 0) - 1
  cat("   Non-zero coefficients:", n_nonzero, "\n\n")
  
  # 4. Elastic net
  cat("4. ELASTIC NET:\n")
  elastic_cv <- cv.glmnet(X, y, alpha = 0.5)
  cat("   Optimal lambda:", round(elastic_cv$lambda.min, 4), "\n")
  cat("   CV MSE:", round(min(elastic_cv$cvm), 4), "\n")
  
  elastic_coef <- coef(elastic_cv, s = elastic_cv$lambda.min)
  n_nonzero_elastic <- sum(elastic_coef != 0) - 1
  cat("   Non-zero coefficients:", n_nonzero_elastic, "\n\n")
  
  # 5. Model comparison
  cat("5. MODEL COMPARISON:\n")
  comparison <- data.frame(
    Method = c("Stepwise", "Ridge", "Lasso", "Elastic Net"),
    CV_MSE = c(NA, min(ridge_cv$cvm), min(lasso_cv$cvm), min(elastic_cv$cvm)),
    Variables = c(best_stepwise, "All", n_nonzero, n_nonzero_elastic)
  )
  print(comparison)
  
  return(list(
    stepwise = stepwise_model,
    ridge = ridge_cv,
    lasso = lasso_cv,
    elastic = elastic_cv,
    comparison = comparison
  ))
}

# Apply comprehensive selection
selection_results <- comprehensive_model_selection(train_data, test_data)
```

### Reporting Guidelines

```r
# Function to generate comprehensive model building report
generate_model_report <- function(selection_results, test_data) {
  cat("=== MODEL BUILDING REPORT ===\n\n")
  
  # Data summary
  cat("DATA SUMMARY:\n")
  cat("Training set size:", nrow(train_data), "\n")
  cat("Testing set size:", nrow(test_data), "\n")
  cat("Number of predictors:", ncol(train_data) - 1, "\n\n")
  
  # Variable selection results
  cat("VARIABLE SELECTION RESULTS:\n")
  
  # Stepwise selection
  stepwise_summary <- summary(selection_results$stepwise)
  best_stepwise <- which.min(stepwise_summary$bic)
  cat("Stepwise Selection:\n")
  cat("  Best model size:", best_stepwise, "variables\n")
  cat("  BIC:", round(stepwise_summary$bic[best_stepwise], 2), "\n")
  cat("  Adjusted R²:", round(stepwise_summary$adjr2[best_stepwise], 3), "\n\n")
  
  # Regularization methods
  cat("Regularization Methods:\n")
  cat("  Ridge CV MSE:", round(min(selection_results$ridge$cvm), 4), "\n")
  cat("  Lasso CV MSE:", round(min(selection_results$lasso$cvm), 4), "\n")
  cat("  Elastic Net CV MSE:", round(min(selection_results$elastic$cvm), 4), "\n\n")
  
  # Model comparison
  cat("MODEL COMPARISON:\n")
  print(selection_results$comparison)
  
  # Recommendations
  cat("\nRECOMMENDATIONS:\n")
  best_method <- which.min(selection_results$comparison$CV_MSE[-1]) + 1
  cat("Best performing method:", selection_results$comparison$Method[best_method], "\n")
  
  if (best_method == 1) {
    cat("Stepwise selection provides good interpretability\n")
  } else if (best_method == 2) {
    cat("Ridge regression handles multicollinearity well\n")
  } else if (best_method == 3) {
    cat("Lasso provides sparse solutions\n")
  } else {
    cat("Elastic net balances ridge and lasso\n")
  }
}

# Generate report
generate_model_report(selection_results, test_data)
```

## Practical Examples

### Example 1: Real Estate Pricing

```r
# Simulate real estate data
set.seed(123)
n_properties <- 500

# Generate predictor variables
real_estate_data <- data.frame(
  square_feet = rnorm(n_properties, 2000, 500),
  bedrooms = rpois(n_properties, 3),
  bathrooms = rpois(n_properties, 2),
  age = rpois(n_properties, 15),
  distance_to_city = rnorm(n_properties, 10, 3),
  crime_rate = rnorm(n_properties, 50, 10),
  school_rating = rnorm(n_properties, 7, 1),
  lot_size = rnorm(n_properties, 8000, 2000),
  garage_spaces = rpois(n_properties, 2),
  pool = rbinom(n_properties, 1, 0.2)
)

# Create price with some variables being more important
price <- 200000 + 
  100 * real_estate_data$square_feet + 
  15000 * real_estate_data$bedrooms + 
  25000 * real_estate_data$bathrooms - 
  2000 * real_estate_data$age - 
  5000 * real_estate_data$distance_to_city - 
  1000 * real_estate_data$crime_rate + 
  15000 * real_estate_data$school_rating + 
  0.5 * real_estate_data$lot_size + 
  5000 * real_estate_data$garage_spaces + 
  25000 * real_estate_data$pool + 
  rnorm(n_properties, 0, 10000)

real_estate_data$price <- price

# Split data
re_train_index <- createDataPartition(real_estate_data$price, p = 0.7, list = FALSE)
re_train <- real_estate_data[re_train_index, ]
re_test <- real_estate_data[-re_train_index, ]

# Apply variable selection
re_selection <- comprehensive_model_selection(re_train, re_test)

# Compare with true model
cat("Real Estate Model Analysis:\n")
cat("True important variables: square_feet, bedrooms, bathrooms, age, distance_to_city\n")
cat("True unimportant variables: crime_rate, school_rating, lot_size, garage_spaces, pool\n")
```

### Example 2: Marketing Campaign Effectiveness

```r
# Simulate marketing data
set.seed(123)
n_campaigns <- 300

marketing_data <- data.frame(
  budget = rnorm(n_campaigns, 50000, 15000),
  duration = rpois(n_campaigns, 30),
  social_media = rnorm(n_campaigns, 0.3, 0.1),
  tv_ads = rnorm(n_campaigns, 0.4, 0.15),
  print_ads = rnorm(n_campaigns, 0.2, 0.08),
  influencer_marketing = rnorm(n_campaigns, 0.1, 0.05),
  target_audience_size = rnorm(n_campaigns, 1000000, 200000),
  competitor_activity = rnorm(n_campaigns, 0.5, 0.2),
  seasonality = rnorm(n_campaigns, 0, 1),
  market_sentiment = rnorm(n_campaigns, 0.6, 0.2)
)

# Create conversion rate
conversion_rate <- 0.02 + 
  0.000001 * marketing_data$budget + 
  0.0005 * marketing_data$duration + 
  0.1 * marketing_data$social_media + 
  0.05 * marketing_data$tv_ads + 
  0.02 * marketing_data$print_ads + 
  0.15 * marketing_data$influencer_marketing + 
  0.0000001 * marketing_data$target_audience_size - 
  0.05 * marketing_data$competitor_activity + 
  0.01 * marketing_data$seasonality + 
  0.03 * marketing_data$market_sentiment + 
  rnorm(n_campaigns, 0, 0.005)

marketing_data$conversion_rate <- conversion_rate

# Split data
mkt_train_index <- createDataPartition(marketing_data$conversion_rate, p = 0.7, list = FALSE)
mkt_train <- marketing_data[mkt_train_index, ]
mkt_test <- marketing_data[-mkt_train_index, ]

# Apply variable selection
mkt_selection <- comprehensive_model_selection(mkt_train, mkt_test)

cat("Marketing Campaign Analysis:\n")
cat("Important variables: budget, duration, social_media, influencer_marketing\n")
cat("Less important: print_ads, competitor_activity, seasonality\n")
```

## Exercises

### Exercise 1: Stepwise Selection
Apply forward, backward, and stepwise selection to a dataset and compare the results.

### Exercise 2: Regularization Methods
Compare ridge, lasso, and elastic net regression on a high-dimensional dataset.

### Exercise 3: Cross-Validation
Implement k-fold cross-validation to select the optimal number of variables.

### Exercise 4: Model Diagnostics
Perform comprehensive residual analysis and multicollinearity detection.

### Exercise 5: Real-World Application
Apply variable selection methods to a real dataset and interpret the results.

## Next Steps

In the next chapter, we'll learn about survival analysis for time-to-event data.

---

**Key Takeaways:**
- Variable selection helps identify the most important predictors
- Stepwise methods are interpretable but can be unstable
- Regularization methods handle multicollinearity and prevent overfitting
- Cross-validation is essential for model selection
- Always validate models on independent test sets
- Consider both statistical and practical significance
- Document the model building process thoroughly
- Choose methods based on data characteristics and goals 