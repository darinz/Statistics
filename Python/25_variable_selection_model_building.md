# Variable Selection and Model Building

## Introduction

Variable selection and model building are fundamental steps in statistical modeling. They help identify the most important predictors, improve model interpretability, reduce overfitting, and enhance predictive performance. The goal is to build a model that is both accurate and parsimonious.

**Key Goals:**
- Identify relevant predictors
- Avoid overfitting and multicollinearity
- Improve interpretability and generalizability
- Select models based on statistical and practical criteria

**When to Use Variable Selection:**
- Many potential predictors
- High-dimensional data
- Need for interpretable or sparse models
- Multicollinearity is a concern

## Mathematical Foundations

### Linear Regression Model

A multiple linear regression model with $`p`$ predictors:

```math
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \varepsilon
```

where $`Y`$ is the response, $`X_j`$ are predictors, $`\beta_j`$ are coefficients, and $`\varepsilon`$ is the error term.

### Model Selection Criteria

- **Akaike Information Criterion (AIC):**
```math
\text{AIC} = 2k - 2\log(L)
```
  where $`k`$ is the number of parameters, $`L`$ is the likelihood.
- **Bayesian Information Criterion (BIC):**
```math
\text{BIC} = k \log(n) - 2\log(L)
```
  where $`n`$ is the sample size.
- **Cross-Validation (CV):**
  - Partition data, fit model on training, evaluate on validation set.

Lower AIC/BIC or lower CV error indicates a better model.

---

# Practical Implementation

All Python code for this chapter has been moved to the companion file: `25_variable_selection_model_building.py`.

- For each theoretical section, refer to the corresponding function in the Python file.
- The Python file contains modular, well-documented functions and a main demonstration block.
- See the end of this document for a summary mapping theory sections to code functions.

---

## Stepwise Selection Methods

### Conceptual Overview

Stepwise selection methods iteratively add or remove predictors based on statistical criteria (e.g., AIC, BIC, adjusted $`R^2`$). The main approaches are:
- **Forward selection:** Start with no variables, add one at a time.
- **Backward elimination:** Start with all variables, remove one at a time.
- **Stepwise (both):** Combine forward and backward steps.

### Mathematical Rationale

At each step, the method chooses the variable that most improves the model according to a criterion (e.g., largest drop in AIC or BIC).

**Adjusted $`R^2`$:**
```math
R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
```
where $`n`$ is sample size, $`p`$ is number of predictors.

**Mallows' $`C_p`$:**
```math
C_p = \frac{\text{RSS}_p}{\hat{\sigma}^2} - (n - 2p)
```
where $`\text{RSS}_p`$ is residual sum of squares for model with $`p`$ predictors.

**Limitations:**
- Can be unstable (small data changes may change selected variables)
- Ignores model uncertainty
- May not find the true best model

### Python Implementation

**Reference:** Use `create_sample_data()`, `perform_stepwise_selection()`, and `evaluate_model()` functions in the Python file.

---

## Regularization Methods

### Conceptual Overview

Regularization methods add a penalty to the regression loss function to shrink coefficients and perform variable selection.

- **Ridge regression:** $`L_2`$ penalty, shrinks coefficients but does not set them to zero.
- **Lasso regression:** $`L_1`$ penalty, can set some coefficients exactly to zero (sparse solution).
- **Elastic net:** Combination of $`L_1`$ and $`L_2`$ penalties.

### Mathematical Foundation

- **Ridge regression:**
  ```math
  \min_{\beta} \left\{ \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\}
  ```
- **Lasso regression:**
  ```math
  \min_{\beta} \left\{ \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 + \lambda \sum_{j=1}^p |\beta_j| \right\}
  ```
- **Elastic net:**
  ```math
  \min_{\beta} \left\{ \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 + \lambda_1 \sum_{j=1}^p |\beta_j| + \lambda_2 \sum_{j=1}^p \beta_j^2 \right\}
  ```

**Tuning parameter $`\lambda`$ controls the strength of the penalty.**

**Advantages:**
- Handles multicollinearity
- Prevents overfitting
- Lasso/elastic net can select variables automatically

**Limitations:**
- Ridge does not produce sparse models
- Lasso can be unstable with highly correlated predictors

### Python Implementation

**Reference:** Use `perform_regularization_analysis()`, `plot_coefficient_paths()`, and `perform_cv_regularization()` functions in the Python file.

---

## Principal Component Regression (PCR) and Partial Least Squares (PLS)

### Conceptual Overview

- **PCR:** Regresses the response on principal components of the predictors.
- **PLS:** Finds components that maximize covariance between predictors and response.

### Mathematical Foundation

- **PCR:**
  1. Compute principal components $`Z_1, Z_2, \ldots, Z_m`$ of $`X`$.
  2. Fit regression: $`Y = \alpha_0 + \alpha_1 Z_1 + \cdots + \alpha_m Z_m + \varepsilon`$
- **PLS:**
  - Finds linear combinations of $`X`$ and $`Y`$ with maximal covariance.

**Advantages:**
- Useful when predictors are highly correlated
- Reduces dimensionality

**Limitations:**
- Components may be hard to interpret
- PCR does not use $`Y`$ to select components

### Python Implementation

**Reference:** Use `perform_pcr_pls_analysis()` and `plot_pcr_pls_comparison()` functions in the Python file.

---

## Model Diagnostics

### Residual Analysis

- **Check for:**
  - Nonlinearity
  - Heteroscedasticity
  - Non-normality
  - Outliers/influential points

- **Key plots:**
  - Residuals vs fitted
  - Q-Q plot
  - Histogram of residuals

### Multicollinearity Detection

- **Variance Inflation Factor (VIF):**
```math
VIF_j = \frac{1}{1 - R_j^2}
```
  where $`R_j^2`$ is the $`R^2`$ from regressing $`X_j`$ on all other predictors.
- **Condition number:** Large values ($`> 30`$) indicate multicollinearity.

### Python Implementation

**Reference:** Use `perform_model_diagnostics()` and `create_model_summary()` functions in the Python file.

---

## Best Practices

- Always split data into training and test sets
- Use cross-validation for model selection
- Standardize predictors for regularization methods
- Check assumptions and diagnostics
- Report model selection criteria and performance metrics
- Interpret both statistical and practical significance
- Document the model building process

---

## Code Reference Guide

| Theory Section                        | Python Function/Section                |
|---------------------------------------|----------------------------------------|
| Data Creation                          | `create_sample_data()`                 |
| Stepwise Selection Methods            | `perform_stepwise_selection()`, `evaluate_model()` |
| Regularization Methods                | `perform_regularization_analysis()`, `plot_coefficient_paths()`, `perform_cv_regularization()` |
| PCR and PLS Analysis                  | `perform_pcr_pls_analysis()`, `plot_pcr_pls_comparison()` |
| Model Diagnostics                     | `perform_model_diagnostics()`, `create_model_summary()` |
| Complete Workflow                     | See main block in `.py` file           |

For exercises, use the functions above as building blocks for your analysis.

---

## Exercises

### Exercise 1: Stepwise Selection
- **Objective:** Apply forward, backward, and stepwise selection to a dataset and compare the results.
- **Hint:** Use `perform_stepwise_selection()` and `evaluate_model()` functions.

### Exercise 2: Regularization Methods
- **Objective:** Compare ridge, lasso, and elastic net regression on a high-dimensional dataset.
- **Hint:** Use `perform_regularization_analysis()` and `perform_cv_regularization()` functions.

### Exercise 3: Cross-Validation
- **Objective:** Implement k-fold cross-validation to select the optimal number of variables.
- **Hint:** Use the cross-validation functionality built into the stepwise and regularization functions.

### Exercise 4: Model Diagnostics
- **Objective:** Perform comprehensive residual analysis and multicollinearity detection.
- **Hint:** Use `perform_model_diagnostics()` function.

### Exercise 5: Real-World Application
- **Objective:** Apply variable selection methods to a real dataset and interpret the results.
- **Hint:** Use all methods and compare their selected variables and performance using `create_model_summary()`.

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