# Logistic Regression

## Overview

Logistic regression is a fundamental statistical method for modeling the relationship between a categorical dependent variable and one or more independent variables. It is most commonly used for binary classification (two possible outcomes), but can be extended to handle multiple categories (multinomial) or ordered categories (ordinal).

### When to Use Logistic Regression
- When your outcome variable is categorical (e.g., Yes/No, Success/Failure, Disease/No Disease)
- When you want to estimate the probability of an event occurring as a function of predictor variables
- When you need interpretable effect sizes (odds ratios)

### Key Concepts
- **Odds**: The ratio of the probability of an event occurring to the probability of it not occurring.
  - $`\text{Odds} = \frac{p}{1-p}`$
- **Logit**: The natural logarithm of the odds.
  - $`\text{logit}(p) = \log\left(\frac{p}{1-p}\right)`$
- **Odds Ratio (OR)**: The ratio of the odds for one group to the odds for another group. In logistic regression, the exponentiated coefficient for a predictor is the odds ratio for a one-unit increase in that predictor.
- **Link Function**: Logistic regression uses the logit link, connecting the linear predictor to the probability:

```math
\text{logit}(p) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_k X_k
```

- **Probability**: The predicted probability is obtained by inverting the logit:

```math
p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_k X_k)}}
```

---

## Binary Logistic Regression

### Mathematical Foundation

Suppose $`Y`$ is a binary outcome ($`Y \in \{0, 1\}`$) and $`X_1, X_2, ..., X_k`$ are predictors. The model is:

```math
\log\left(\frac{P(Y=1)}{P(Y=0)}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_k X_k
```

- $`P(Y=1)`$ is the probability of the event (e.g., "Approved").
- $`\beta_j`$ is the change in the log-odds for a one-unit increase in $`X_j`$.
- $`e^{\beta_j}`$ is the odds ratio for $`X_j`$.

### Example: Simulated Credit Approval Data

```r
# Load required packages
library(tidyverse)
library(caret)
library(pROC)
library(car)
library(ResourceSelection)

# Set seed for reproducibility
set.seed(123)

# Generate sample data for binary logistic regression
n_samples <- 500

# Create predictor variables
logistic_data <- data.frame(
  age = rnorm(n_samples, 45, 10),
  income = rnorm(n_samples, 60000, 15000),
  education_years = rnorm(n_samples, 14, 3),
  credit_score = rnorm(n_samples, 700, 100),
  debt_ratio = rnorm(n_samples, 0.3, 0.1)
)

# Create binary outcome based on predictors
# Higher age, income, education, credit score, and lower debt ratio increase probability
log_odds <- -2 + 0.05 * logistic_data$age + 
            0.00001 * logistic_data$income + 
            0.2 * logistic_data$education_years + 
            0.01 * logistic_data$credit_score - 
            3 * logistic_data$debt_ratio

# Convert to probability
prob <- 1 / (1 + exp(-log_odds))

# Generate binary outcome
logistic_data$approved <- rbinom(n_samples, 1, prob)

# Convert to factor
logistic_data$approved <- factor(logistic_data$approved, levels = c(0, 1), 
                                labels = c("Rejected", "Approved"))

print("Sample of the data:")
print(head(logistic_data))
print(table(logistic_data$approved))
```

#### Code Explanation
- **Data Generation**: Simulates applicant features and approval status based on a linear combination of predictors.
- **Log Odds Calculation**: The log-odds are a linear function of predictors, reflecting their influence on approval.
- **Probability Conversion**: The log-odds are transformed to probabilities using the logistic function.
- **Outcome Generation**: The binary outcome is sampled from a Bernoulli distribution with the calculated probability.

---

### Fitting and Interpreting the Model

```r
# Fit logistic regression model
logistic_model <- glm(approved ~ age + income + education_years + credit_score + debt_ratio, 
                     data = logistic_data, family = binomial(link = "logit"))

# Model summary
print("Logistic Regression Model Summary:")
print(summary(logistic_model))

# Extract coefficients
coefficients <- coef(logistic_model)
print("Model Coefficients:")
print(round(coefficients, 4))

# Odds ratios
odds_ratios <- exp(coefficients)
print("Odds Ratios:")
print(round(odds_ratios, 4))

# Confidence intervals for odds ratios
conf_int <- confint(logistic_model)
odds_ratios_ci <- exp(conf_int)
print("Odds Ratios with 95% Confidence Intervals:")
print(round(odds_ratios_ci, 4))
```

#### Code Explanation
- **Model Fitting**: `glm(..., family = binomial(link = "logit"))` fits a logistic regression model.
- **Coefficients**: The output shows the estimated $`\beta`$ values for each predictor.
- **Odds Ratios**: Exponentiating the coefficients gives the odds ratios, which are easier to interpret.
- **Confidence Intervals**: The 95% CI for each odds ratio helps assess statistical significance and effect size.

#### Interpreting Odds Ratios
- If $`\text{OR} > 1`$: A one-unit increase in the predictor increases the odds of the event.
- If $`\text{OR} < 1`$: A one-unit increase in the predictor decreases the odds of the event.

---

### Model Interpretation

```r
# Function to interpret logistic regression coefficients
interpret_logistic_coefficients <- function(model) {
  coefficients <- coef(model)
  odds_ratios <- exp(coefficients)
  
  cat("=== LOGISTIC REGRESSION INTERPRETATION ===\n\n")
  
  for (i in 2:length(coefficients)) {  # Skip intercept
    var_name <- names(coefficients)[i]
    coef_value <- coefficients[i]
    odds_ratio <- odds_ratios[i]
    
    cat("Variable:", var_name, "\n")
    cat("Coefficient:", round(coef_value, 4), "\n")
    cat("Odds Ratio:", round(odds_ratio, 4), "\n")
    
    if (odds_ratio > 1) {
      cat("Interpretation: A one-unit increase in", var_name, 
          "increases the odds of the event by", round((odds_ratio - 1) * 100, 1), "%\n")
    } else {
      cat("Interpretation: A one-unit increase in", var_name, 
          "decreases the odds of the event by", round((1 - odds_ratio) * 100, 1), "%\n")
    }
    cat("\n")
  }
}

# Apply interpretation
interpret_logistic_coefficients(logistic_model)
```

#### Code Explanation
- This function prints a human-readable interpretation for each predictor, including the direction and magnitude of the effect.

---

### Model Assumptions

- **Linearity of the logit**: The logit of the outcome is a linear function of the predictors.
- **Independence**: Observations are independent.
- **No multicollinearity**: Predictors are not highly correlated.
- **Large sample size**: Ensures stable estimates.

#### Checking Assumptions
- Use VIF (Variance Inflation Factor) to check multicollinearity.
- Use diagnostic plots and tests (see below) to check fit and influential points.

---

### Model Diagnostics

```r
# Residual analysis for logistic regression
logistic_diagnostics <- function(model, data) {
  cat("=== LOGISTIC REGRESSION DIAGNOSTICS ===\n\n")
  
  # Deviance residuals
  deviance_residuals <- residuals(model, type = "deviance")
  
  # Pearson residuals
  pearson_residuals <- residuals(model, type = "pearson")
  
  # Fitted probabilities
  fitted_probs <- fitted(model)
  
  # Create diagnostic plots
  par(mfrow = c(2, 2))
  
  # Deviance residuals vs fitted values
  plot(fitted_probs, deviance_residuals, main = "Deviance Residuals vs Fitted",
       xlab = "Fitted Probabilities", ylab = "Deviance Residuals")
  abline(h = 0, col = "red", lty = 2)
  
  # Pearson residuals vs fitted values
  plot(fitted_probs, pearson_residuals, main = "Pearson Residuals vs Fitted",
       xlab = "Fitted Probabilities", ylab = "Pearson Residuals")
  abline(h = 0, col = "red", lty = 2)
  
  # Q-Q plot of deviance residuals
  qqnorm(deviance_residuals, main = "Q-Q Plot of Deviance Residuals")
  qqline(deviance_residuals, col = "red")
  
  # Leverage plot
  leverage <- hatvalues(model)
  plot(leverage, deviance_residuals, main = "Residuals vs Leverage",
       xlab = "Leverage", ylab = "Deviance Residuals")
  abline(h = 0, col = "red", lty = 2)
  
  par(mfrow = c(1, 1))
  
  # Hosmer-Lemeshow test
  hl_test <- hoslem.test(model$y, fitted_probs, g = 10)
  cat("Hosmer-Lemeshow Test:\n")
  cat("Chi-square statistic:", round(hl_test$statistic, 3), "\n")
  cat("p-value:", round(hl_test$p.value, 4), "\n")
  
  if (hl_test$p.value > 0.05) {
    cat("Model fits well (p > 0.05)\n")
  } else {
    cat("Model may not fit well (p ≤ 0.05)\n")
  }
  
  # Influential observations
  cook_dist <- cooks.distance(model)
  influential <- which(cook_dist > 4/length(cook_dist))
  
  cat("\nInfluential Observations (Cook's distance > 4/n):\n")
  if (length(influential) > 0) {
    print(influential)
  } else {
    cat("No influential observations detected.\n")
  }
  
  return(list(
    deviance_residuals = deviance_residuals,
    pearson_residuals = pearson_residuals,
    fitted_probs = fitted_probs,
    leverage = leverage,
    cook_dist = cook_dist,
    hl_test = hl_test
  ))
}

# Apply diagnostics
diagnostics_result <- logistic_diagnostics(logistic_model, logistic_data)
```

#### Code Explanation
- **Residuals**: Deviance and Pearson residuals help assess model fit.
- **Leverage and Influence**: Identifies points that have a large effect on the model.
- **Hosmer-Lemeshow Test**: Assesses goodness-of-fit; $`p > 0.05`$ suggests adequate fit.
- **Cook's Distance**: Identifies influential observations.

---

## Model Performance Evaluation

### Classification Metrics

```r
# Function to calculate classification metrics
calculate_classification_metrics <- function(actual, predicted, threshold = 0.5) {
  # Convert probabilities to predictions
  predicted_class <- ifelse(predicted >= threshold, 1, 0)
  
  # Create confusion matrix
  cm <- table(Actual = actual, Predicted = predicted_class)
  
  # Calculate metrics
  tp <- cm[2, 2]  # True positives
  tn <- cm[1, 1]  # True negatives
  fp <- cm[1, 2]  # False positives
  fn <- cm[2, 1]  # False negatives
  
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  sensitivity <- tp / (tp + fn)  # Recall
  specificity <- tn / (tn + fp)
  precision <- tp / (tp + fp)
  f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
  
  return(list(
    confusion_matrix = cm,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    f1_score = f1_score
  ))
}

# Split data for evaluation
set.seed(123)
train_index <- createDataPartition(logistic_data$approved, p = 0.7, list = FALSE)
train_data <- logistic_data[train_index, ]
test_data <- logistic_data[-train_index, ]

# Fit model on training data
train_model <- glm(approved ~ age + income + education_years + credit_score + debt_ratio, 
                   data = train_data, family = binomial(link = "logit"))

# Predictions on test data
test_predictions <- predict(train_model, newdata = test_data, type = "response")
test_actual <- as.numeric(test_data$approved) - 1  # Convert to 0/1

# Calculate metrics
metrics <- calculate_classification_metrics(test_actual, test_predictions)

cat("Classification Metrics:\n")
cat("Accuracy:", round(metrics$accuracy, 3), "\n")
cat("Sensitivity (Recall):", round(metrics$sensitivity, 3), "\n")
cat("Specificity:", round(metrics$specificity, 3), "\n")
cat("Precision:", round(metrics$precision, 3), "\n")
cat("F1 Score:", round(metrics$f1_score, 3), "\n")

print("Confusion Matrix:")
print(metrics$confusion_matrix)
```

#### Code Explanation
- **Confusion Matrix**: Shows counts of true/false positives/negatives.
- **Accuracy**: Proportion of correct predictions.
- **Sensitivity (Recall)**: Proportion of actual positives correctly identified.
- **Specificity**: Proportion of actual negatives correctly identified.
- **Precision**: Proportion of positive predictions that are correct.
- **F1 Score**: Harmonic mean of precision and recall.

---

### ROC Curve and AUC

```r
# ROC curve analysis
roc_obj <- roc(test_actual, test_predictions)
auc_value <- auc(roc_obj)

# Plot ROC curve
plot(roc_obj, main = "ROC Curve", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")
text(0.8, 0.2, paste("AUC =", round(auc_value, 3)), cex = 1.2)

cat("ROC Analysis:\n")
cat("AUC:", round(auc_value, 3), "\n")

# Find optimal threshold
optimal_threshold <- coords(roc_obj, "best")
cat("Optimal threshold:", round(optimal_threshold$threshold, 3), "\n")
cat("Optimal sensitivity:", round(optimal_threshold$sensitivity, 3), "\n")
cat("Optimal specificity:", round(optimal_threshold$specificity, 3), "\n")

# Calculate metrics with optimal threshold
optimal_metrics <- calculate_classification_metrics(test_actual, test_predictions, 
                                                 optimal_threshold$threshold)

cat("\nMetrics with Optimal Threshold:\n")
cat("Accuracy:", round(optimal_metrics$accuracy, 3), "\n")
cat("Sensitivity:", round(optimal_metrics$sensitivity, 3), "\n")
cat("Specificity:", round(optimal_metrics$specificity, 3), "\n")
```

#### Code Explanation
- **ROC Curve**: Plots sensitivity vs. 1-specificity for all thresholds.
- **AUC (Area Under Curve)**: Measures overall model discrimination (1 = perfect, 0.5 = random).
- **Optimal Threshold**: The threshold that maximizes sensitivity and specificity.

---

### Cross-Validation

```r
# K-fold cross-validation
cv_logistic <- function(data, k = 5) {
  n <- nrow(data)
  fold_size <- floor(n / k)
  
  cv_metrics <- list()
  
  for (fold in 1:k) {
    # Create fold indices
    start_idx <- (fold - 1) * fold_size + 1
    end_idx <- ifelse(fold == k, n, fold * fold_size)
    test_indices <- start_idx:end_idx
    
    # Split data
    train_fold <- data[-test_indices, ]
    test_fold <- data[test_indices, ]
    
    # Fit model
    fold_model <- glm(approved ~ age + income + education_years + credit_score + debt_ratio, 
                     data = train_fold, family = binomial(link = "logit"))
    
    # Predictions
    fold_pred <- predict(fold_model, newdata = test_fold, type = "response")
    fold_actual <- as.numeric(test_fold$approved) - 1
    
    # Calculate metrics
    fold_metrics <- calculate_classification_metrics(fold_actual, fold_pred)
    
    cv_metrics[[paste0("fold_", fold)]] <- fold_metrics
  }
  
  # Average metrics across folds
  avg_accuracy <- mean(sapply(cv_metrics, function(x) x$accuracy))
  avg_sensitivity <- mean(sapply(cv_metrics, function(x) x$sensitivity))
  avg_specificity <- mean(sapply(cv_metrics, function(x) x$specificity))
  avg_f1 <- mean(sapply(cv_metrics, function(x) x$f1_score))
  
  return(list(
    fold_metrics = cv_metrics,
    avg_accuracy = avg_accuracy,
    avg_sensitivity = avg_sensitivity,
    avg_specificity = avg_specificity,
    avg_f1 = avg_f1
  ))
}

# Apply cross-validation
cv_results <- cv_logistic(logistic_data, k = 5)

cat("Cross-Validation Results:\n")
cat("Average Accuracy:", round(cv_results$avg_accuracy, 3), "\n")
cat("Average Sensitivity:", round(cv_results$avg_sensitivity, 3), "\n")
cat("Average Specificity:", round(cv_results$avg_specificity, 3), "\n")
cat("Average F1 Score:", round(cv_results$avg_f1, 3), "\n")
```

#### Code Explanation
- **K-fold Cross-Validation**: Splits data into $`k`$ parts, trains on $`k-1`$, tests on 1, repeats, and averages metrics.
- **Purpose**: Provides a more robust estimate of model performance.

---

## Multinomial Logistic Regression

### Mathematical Foundation

For $`J`$ outcome categories ($`Y \in \{1, 2, ..., J\}`$), the model is:

```math
\log\left(\frac{P(Y=j)}{P(Y=1)}\right) = \beta_{0j} + \beta_{1j} X_1 + \cdots + \beta_{kj} X_k, \quad j = 2, ..., J
```

- $`P(Y=j)`$ is the probability of category $`j`$.
- $`P(Y=1)`$ is the probability of the reference category.
- Each category (except the reference) has its own set of coefficients.

### Example: Simulated Risk Category Data

```r
# Load nnet package for multinomial logistic regression
library(nnet)

# Generate multinomial data
set.seed(123)
n_multinomial <- 400

multinomial_data <- data.frame(
  age = rnorm(n_multinomial, 45, 10),
  income = rnorm(n_multinomial, 60000, 15000),
  education_years = rnorm(n_multinomial, 14, 3),
  credit_score = rnorm(n_multinomial, 700, 100),
  debt_ratio = rnorm(n_multinomial, 0.3, 0.1)
)

# Create multinomial outcome (3 categories)
# Category 1: Low risk, Category 2: Medium risk, Category 3: High risk
log_odds_medium <- -1 + 0.02 * multinomial_data$age + 
                   0.000005 * multinomial_data$income + 
                   0.1 * multinomial_data$education_years + 
                   0.005 * multinomial_data$credit_score - 
                   1.5 * multinomial_data$debt_ratio

log_odds_high <- -2 + 0.03 * multinomial_data$age + 
                 0.000008 * multinomial_data$income + 
                 0.15 * multinomial_data$education_years + 
                 0.008 * multinomial_data$credit_score - 
                 2 * multinomial_data$debt_ratio

# Calculate probabilities
prob_medium <- exp(log_odds_medium) / (1 + exp(log_odds_medium) + exp(log_odds_high))
prob_high <- exp(log_odds_high) / (1 + exp(log_odds_medium) + exp(log_odds_high))
prob_low <- 1 - prob_medium - prob_high

# Generate multinomial outcome
multinomial_data$risk_category <- apply(
  cbind(prob_low, prob_medium, prob_high), 1, 
  function(p) sample(1:3, 1, prob = p)
)

multinomial_data$risk_category <- factor(multinomial_data$risk_category,
                                       levels = 1:3,
                                       labels = c("Low Risk", "Medium Risk", "High Risk"))

print("Multinomial Data Summary:")
print(table(multinomial_data$risk_category))
```

#### Code Explanation
- **Data Generation**: Simulates predictors and assigns a risk category based on multinomial probabilities.

---

### Fitting and Interpreting the Multinomial Model

```r
# Load nnet package for multinomial logistic regression
library(nnet)

# Fit multinomial logistic regression
multinomial_model <- multinom(risk_category ~ age + income + education_years + 
                             credit_score + debt_ratio, data = multinomial_data)

# Model summary
print("Multinomial Logistic Regression Summary:")
print(summary(multinomial_model))

# Extract coefficients
multinomial_coef <- coef(multinomial_model)
print("Multinomial Coefficients:")
print(round(multinomial_coef, 4))

# Calculate odds ratios
multinomial_odds <- exp(multinomial_coef)
print("Multinomial Odds Ratios:")
print(round(multinomial_odds, 4))

# Z-statistics and p-values
z_stats <- summary(multinomial_model)$coefficients / summary(multinomial_model)$standard.errors
p_values <- 2 * (1 - pnorm(abs(z_stats)))

print("Z-statistics:")
print(round(z_stats, 3))
print("P-values:")
print(round(p_values, 4))
```

#### Code Explanation
- **Model Fitting**: `multinom()` fits a multinomial logistic regression.
- **Coefficients**: Each non-reference category has its own coefficients.
- **Odds Ratios**: Exponentiated coefficients for each category.
- **Z-statistics and p-values**: Assess statistical significance of predictors.

#### Interpreting Multinomial Odds Ratios
- For each predictor, the odds ratio compares the odds of being in a given category vs. the reference for a one-unit increase in the predictor.

---

## Advanced Topics

### Ordinal Logistic Regression

#### Mathematical Foundation

For ordered categories, the proportional odds model is:

```math
\log\left(\frac{P(Y \leq j)}{P(Y > j)}\right) = \theta_j - (\beta_1 X_1 + \cdots + \beta_k X_k), \quad j = 1, ..., J-1
```

- $`\theta_j`$ are threshold (cutpoint) parameters.
- $`\beta`$ coefficients are assumed constant across thresholds (proportional odds assumption).

```r
# Load MASS package for ordinal logistic regression
library(MASS)

# Create ordinal outcome (convert risk category to numeric)
ordinal_data <- multinomial_data
ordinal_data$risk_ordinal <- as.numeric(ordinal_data$risk_category)

# Fit ordinal logistic regression (proportional odds model)
ordinal_model <- polr(risk_category ~ age + income + education_years + 
                     credit_score + debt_ratio, data = ordinal_data, Hess = TRUE)

# Model summary
print("Ordinal Logistic Regression Summary:")
print(summary(ordinal_model))

# Extract coefficients
ordinal_coef <- coef(ordinal_model)
print("Ordinal Coefficients:")
print(round(ordinal_coef, 4))

# Calculate odds ratios
ordinal_odds <- exp(ordinal_coef)
print("Ordinal Odds Ratios:")
print(round(ordinal_odds, 4))

# Thresholds (cutpoints)
thresholds <- ordinal_model$zeta
print("Thresholds:")
print(round(thresholds, 4))
```

#### Code Explanation
- **Model Fitting**: `polr()` fits an ordinal logistic regression.
- **Thresholds**: Cutpoints between categories.
- **Odds Ratios**: Interpreted as the odds of being in a higher vs. lower category.

---

### Model Comparison

```r
# Compare binary, multinomial, and ordinal models
model_comparison <- function(binary_data, multinomial_data) {
  cat("=== MODEL COMPARISON ===\n\n")
  
  # Binary model (combine medium and high risk)
  binary_data_combined <- binary_data
  binary_data_combined$approved <- ifelse(binary_data_combined$approved == "Approved", 1, 0)
  
  # Fit binary model
  binary_model <- glm(approved ~ age + income + education_years + credit_score + debt_ratio, 
                     data = binary_data_combined, family = binomial(link = "logit"))
  
  # AIC comparison
  binary_aic <- AIC(binary_model)
  multinomial_aic <- AIC(multinomial_model)
  ordinal_aic <- AIC(ordinal_model)
  
  cat("AIC Comparison:\n")
  cat("Binary model:", round(binary_aic, 2), "\n")
  cat("Multinomial model:", round(multinomial_aic, 2), "\n")
  cat("Ordinal model:", round(ordinal_aic, 2), "\n\n")
  
  # BIC comparison
  binary_bic <- BIC(binary_model)
  multinomial_bic <- BIC(multinomial_model)
  ordinal_bic <- BIC(ordinal_model)
  
  cat("BIC Comparison:\n")
  cat("Binary model:", round(binary_bic, 2), "\n")
  cat("Multinomial model:", round(multinomial_bic, 2), "\n")
  cat("Ordinal model:", round(ordinal_bic, 2), "\n\n")
  
  # Likelihood ratio test for ordinal vs multinomial
  lr_stat <- -2 * (logLik(ordinal_model) - logLik(multinomial_model))
  lr_p_value <- pchisq(lr_stat, df = 1, lower.tail = FALSE)
  
  cat("Likelihood Ratio Test (Ordinal vs Multinomial):\n")
  cat("Chi-square statistic:", round(lr_stat, 3), "\n")
  cat("p-value:", round(lr_p_value, 4), "\n")
  
  if (lr_p_value > 0.05) {
    cat("Ordinal model is preferred (proportional odds assumption holds)\n")
  } else {
    cat("Multinomial model is preferred (proportional odds assumption violated)\n")
  }
  
  return(list(
    binary_aic = binary_aic,
    multinomial_aic = multinomial_aic,
    ordinal_aic = ordinal_aic,
    binary_bic = binary_bic,
    multinomial_bic = multinomial_bic,
    ordinal_bic = ordinal_bic,
    lr_stat = lr_stat,
    lr_p_value = lr_p_value
  ))
}

# Apply model comparison
comparison_results <- model_comparison(logistic_data, multinomial_data)
```

#### Code Explanation
- **AIC/BIC**: Lower values indicate better fit, penalizing for complexity.
- **Likelihood Ratio Test**: Compares nested models (e.g., ordinal vs. multinomial).
- **Interpretation**: If the proportional odds assumption holds, ordinal is preferred; otherwise, multinomial.

---

## Practical Examples

### Example 1: Credit Card Approval

```r
# Simulate credit card application data
set.seed(123)
n_applications <- 1000

credit_data <- data.frame(
  age = rnorm(n_applications, 35, 8),
  income = rnorm(n_applications, 75000, 20000),
  credit_score = rnorm(n_applications, 720, 80),
  debt_to_income = rnorm(n_applications, 0.25, 0.1),
  employment_years = rnorm(n_applications, 5, 3),
  previous_defaults = rpois(n_applications, 0.3),
  home_ownership = factor(sample(c("Rent", "Own", "Mortgage"), n_applications, 
                               prob = c(0.4, 0.3, 0.3), replace = TRUE)
)

# Create approval probability
log_odds_credit <- -3 + 0.02 * credit_data$age + 
                   0.00001 * credit_data$income + 
                   0.015 * credit_data$credit_score - 
                   2 * credit_data$debt_to_income + 
                   0.1 * credit_data$employment_years - 
                   0.5 * credit_data$previous_defaults

# Add home ownership effect
log_odds_credit[credit_data$home_ownership == "Own"] <- 
  log_odds_credit[credit_data$home_ownership == "Own"] + 0.3
log_odds_credit[credit_data$home_ownership == "Mortgage"] <- 
  log_odds_credit[credit_data$home_ownership == "Mortgage"] + 0.1

prob_credit <- 1 / (1 + exp(-log_odds_credit))
credit_data$approved <- rbinom(n_applications, 1, prob_credit)
credit_data$approved <- factor(credit_data$approved, levels = c(0, 1), 
                              labels = c("Rejected", "Approved"))

# Fit credit approval model
credit_model <- glm(approved ~ age + income + credit_score + debt_to_income + 
                    employment_years + previous_defaults + home_ownership, 
                    data = credit_data, family = binomial(link = "logit"))

print("Credit Card Approval Model:")
print(summary(credit_model))

# Odds ratios
credit_odds <- exp(coef(credit_model))
print("Odds Ratios for Credit Approval:")
print(round(credit_odds, 4))
```

#### Code Explanation
- **Simulated Data**: Models credit approval based on applicant features.
- **Model Fitting**: Includes categorical predictors (e.g., home ownership).
- **Interpretation**: Odds ratios for each predictor, including categorical levels.

---

### Example 2: Medical Diagnosis

```r
# Simulate medical diagnosis data
set.seed(123)
n_patients <- 800

medical_data <- data.frame(
  age = rnorm(n_patients, 60, 15),
  bmi = rnorm(n_patients, 28, 5),
  blood_pressure = rnorm(n_patients, 140, 20),
  cholesterol = rnorm(n_patients, 200, 40),
  glucose = rnorm(n_patients, 100, 20),
  family_history = factor(sample(c("No", "Yes"), n_patients, 
                               prob = c(0.7, 0.3), replace = TRUE),
  smoking = factor(sample(c("Never", "Former", "Current"), n_patients, 
                         prob = c(0.4, 0.3, 0.3), replace = TRUE)
)

# Create disease probability (diabetes)
log_odds_diabetes <- -4 + 0.03 * medical_data$age + 
                     0.05 * medical_data$bmi + 
                     0.01 * medical_data$blood_pressure + 
                     0.005 * medical_data$cholesterol + 
                     0.02 * medical_data$glucose

# Add categorical effects
log_odds_diabetes[medical_data$family_history == "Yes"] <- 
  log_odds_diabetes[medical_data$family_history == "Yes"] + 0.8
log_odds_diabetes[medical_data$smoking == "Current"] <- 
  log_odds_diabetes[medical_data$smoking == "Current"] + 0.3
log_odds_diabetes[medical_data$smoking == "Former"] <- 
  log_odds_diabetes[medical_data$smoking == "Former"] + 0.1

prob_diabetes <- 1 / (1 + exp(-log_odds_diabetes))
medical_data$diabetes <- rbinom(n_patients, 1, prob_diabetes)
medical_data$diabetes <- factor(medical_data$diabetes, levels = c(0, 1), 
                               labels = c("No Diabetes", "Diabetes"))

# Fit medical diagnosis model
medical_model <- glm(diabetes ~ age + bmi + blood_pressure + cholesterol + 
                     glucose + family_history + smoking, 
                     data = medical_data, family = binomial(link = "logit"))

print("Medical Diagnosis Model:")
print(summary(medical_model))

# Risk prediction for new patient
new_patient <- data.frame(
  age = 65,
  bmi = 32,
  blood_pressure = 150,
  cholesterol = 220,
  glucose = 120,
  family_history = "Yes",
  smoking = "Former"
)

predicted_risk <- predict(medical_model, newdata = new_patient, type = "response")
cat("Predicted diabetes risk for new patient:", round(predicted_risk, 3), "\n")
```

#### Code Explanation
- **Simulated Data**: Models disease risk based on clinical and lifestyle features.
- **Risk Prediction**: Predicts probability for a new patient.

---

## Best Practices

### Model Selection Guidelines

- Use binary logistic regression for two outcome categories.
- Use multinomial for more than two unordered categories.
- Use ordinal for ordered categories (if proportional odds assumption holds).
- Check sample size and class balance.
- Always check model assumptions and diagnostics.

```r
# Function to help choose appropriate logistic regression model
choose_logistic_model <- function(data, outcome_var) {
  cat("=== LOGISTIC REGRESSION MODEL SELECTION ===\n\n")
  
  # Check outcome variable
  outcome_levels <- levels(data[[outcome_var]])
  n_levels <- length(outcome_levels)
  
  cat("Outcome variable:", outcome_var, "\n")
  cat("Number of levels:", n_levels, "\n")
  cat("Levels:", paste(outcome_levels, collapse = ", "), "\n\n")
  
  if (n_levels == 2) {
    cat("RECOMMENDATION: Use binary logistic regression\n")
    cat("- Suitable for binary outcomes\n")
    cat("- Easy to interpret odds ratios\n")
    cat("- Good for classification problems\n")
  } else if (n_levels == 3) {
    cat("RECOMMENDATION: Consider both multinomial and ordinal\n")
    cat("- Multinomial: No ordering assumption\n")
    cat("- Ordinal: Assumes proportional odds\n")
    cat("- Test proportional odds assumption\n")
  } else {
    cat("RECOMMENDATION: Use multinomial logistic regression\n")
    cat("- Suitable for multiple unordered categories\n")
    cat("- More complex interpretation\n")
  }
  
  # Check sample size
  n_samples <- nrow(data)
  cat("\nSample size:", n_samples, "\n")
  
  if (n_samples < 100) {
    cat("WARNING: Small sample size may affect model stability\n")
  } else if (n_samples < 500) {
    cat("Sample size is adequate for most applications\n")
  } else {
    cat("Sample size is good for complex models\n")
  }
  
  # Check class balance
  outcome_counts <- table(data[[outcome_var]])
  min_count <- min(outcome_counts)
  max_count <- max(outcome_counts)
  imbalance_ratio <- max_count / min_count
  
  cat("\nClass balance:\n")
  print(outcome_counts)
  
  if (imbalance_ratio > 3) {
    cat("WARNING: Class imbalance detected\n")
    cat("Consider: resampling, different thresholds, or class weights\n")
  }
  
  return(list(
    n_levels = n_levels,
    n_samples = n_samples,
    imbalance_ratio = imbalance_ratio
  ))
}

# Apply model selection
model_selection <- choose_logistic_model(logistic_data, "approved")
```

---

### Reporting Guidelines

- Report coefficients, odds ratios, and confidence intervals.
- Include model fit statistics (AIC, BIC, log-likelihood).
- Report diagnostics (e.g., Hosmer-Lemeshow test, ROC/AUC).
- Discuss limitations, assumptions, and recommendations.

```r
# Function to generate comprehensive logistic regression report
generate_logistic_report <- function(model, data, test_data = NULL) {
  cat("=== LOGISTIC REGRESSION REPORT ===\n\n")
  
  # Model summary
  cat("MODEL SUMMARY:\n")
  print(summary(model))
  
  # Odds ratios with confidence intervals
  cat("\nODDS RATIOS WITH 95% CONFIDENCE INTERVALS:\n")
  odds_ci <- exp(confint(model))
  odds_table <- data.frame(
    Variable = rownames(odds_ci),
    Odds_Ratio = exp(coef(model)),
    CI_Lower = odds_ci[, 1],
    CI_Upper = odds_ci[, 2]
  )
  print(round(odds_table, 3))
  
  # Model fit statistics
  cat("\nMODEL FIT STATISTICS:\n")
  cat("AIC:", round(AIC(model), 2), "\n")
  cat("BIC:", round(BIC(model), 2), "\n")
  cat("Log-likelihood:", round(logLik(model), 2), "\n")
  
  # Hosmer-Lemeshow test
  hl_test <- hoslem.test(model$y, fitted(model), g = 10)
  cat("Hosmer-Lemeshow test p-value:", round(hl_test$p.value, 4), "\n")
  
  if (hl_test$p.value > 0.05) {
    cat("Model fits well\n")
  } else {
    cat("Model fit may be inadequate\n")
  }
  
  # Performance metrics (if test data provided)
  if (!is.null(test_data)) {
    cat("\nPERFORMANCE METRICS:\n")
    test_pred <- predict(model, newdata = test_data, type = "response")
    test_actual <- as.numeric(test_data$approved) - 1
    
    # ROC analysis
    roc_obj <- roc(test_actual, test_pred)
    auc_value <- auc(roc_obj)
    
    cat("AUC:", round(auc_value, 3), "\n")
    
    # Classification metrics
    metrics <- calculate_classification_metrics(test_actual, test_pred)
    cat("Accuracy:", round(metrics$accuracy, 3), "\n")
    cat("Sensitivity:", round(metrics$sensitivity, 3), "\n")
    cat("Specificity:", round(metrics$specificity, 3), "\n")
    cat("F1 Score:", round(metrics$f1_score, 3), "\n")
  }
  
  # Recommendations
  cat("\nRECOMMENDATIONS:\n")
  cat("- Check for influential observations\n")
  cat("- Validate model assumptions\n")
  cat("- Consider model diagnostics\n")
  cat("- Test model on new data\n")
}

# Generate report
generate_logistic_report(logistic_model, logistic_data, test_data)
```

---

## Exercises

### Exercise 1: Binary Logistic Regression
**Objective:** Fit a binary logistic regression model to a dataset. Interpret the coefficients and odds ratios.
- *Hint:* Use `glm()` with `family = binomial`.
- *Learning Outcome:* Understand model fitting and interpretation.

### Exercise 2: Model Diagnostics
**Objective:** Perform comprehensive diagnostics for a logistic regression model, including residual analysis and influence diagnostics.
- *Hint:* Use residual plots, Cook's distance, and the Hosmer-Lemeshow test.
- *Learning Outcome:* Assess model fit and identify influential points.

### Exercise 3: Model Performance
**Objective:** Evaluate model performance using ROC curves, classification metrics, and cross-validation.
- *Hint:* Use `pROC` for ROC/AUC and implement k-fold cross-validation.
- *Learning Outcome:* Understand model evaluation and validation.

### Exercise 4: Multinomial Logistic Regression
**Objective:** Fit a multinomial logistic regression model and compare it with binary and ordinal models.
- *Hint:* Use `nnet::multinom()` and `MASS::polr()`.
- *Learning Outcome:* Understand when and how to use multinomial and ordinal models.

### Exercise 5: Real-World Application
**Objective:** Apply logistic regression to a real-world classification problem and generate a comprehensive report.
- *Hint:* Use a dataset from healthcare, finance, or social sciences.
- *Learning Outcome:* Integrate model fitting, diagnostics, and reporting.

---

## Key Takeaways
- Logistic regression is ideal for binary and categorical outcomes
- Odds ratios provide intuitive interpretation of effects
- Model diagnostics are crucial for logistic regression
- ROC curves and AUC are important for model evaluation
- Cross-validation helps assess model performance
- Multinomial and ordinal models extend binary logistic regression
- Always check assumptions and validate models
- Consider class imbalance and sample size requirements

---

**Next Steps:**
In the next chapter, we'll learn about survival analysis for time-to-event data. 