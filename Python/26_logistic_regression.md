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

## Practical Implementation

All code examples and implementations for this chapter are available in the Python file: `26_logistic_regression.py`

### Code Reference Guide

| Theory Section | Python Function | Description |
|----------------|-----------------|-------------|
| Binary Logistic Regression - Data Generation | `generate_binary_logistic_data()` | Simulates credit approval data |
| Binary Logistic Regression - Model Fitting | `fit_binary_logistic_model()` | Fits binary logistic regression |
| Model Interpretation | `interpret_logistic_coefficients()` | Interprets coefficients and odds ratios |
| Model Diagnostics | `logistic_diagnostics()` | Performs comprehensive diagnostics |
| Classification Metrics | `calculate_classification_metrics()` | Calculates performance metrics |
| Model Performance | `evaluate_model_performance()` | Evaluates model using train-test split |
| ROC Curve Analysis | `plot_roc_curve()` | Plots ROC curve and finds optimal threshold |
| Cross-Validation | `cv_logistic()` | Performs k-fold cross-validation |
| Multinomial Logistic Regression | `generate_multinomial_data()`, `fit_multinomial_model()` | Multinomial logistic regression |
| Ordinal Logistic Regression | `fit_ordinal_model()` | Ordinal logistic regression |
| Model Comparison | `model_comparison()` | Compares different logistic regression models |
| Credit Approval Example | `generate_credit_approval_data()`, `fit_credit_approval_model()` | Credit card approval application |
| Medical Diagnosis Example | `generate_medical_data()`, `fit_medical_model()`, `predict_medical_risk()` | Medical diagnosis application |
| Model Selection | `choose_logistic_model()` | Helps choose appropriate model type |
| Comprehensive Reporting | `generate_logistic_report()` | Generates detailed model report |
| Exercise Data | `generate_exercise_data()` | Generates data for exercises |

### Running the Code

To run the complete demonstration:

```python
# Import the module
from logistic_regression import main

# Run the full demonstration
main()
```

To run specific sections:

```python
# Generate and fit binary logistic regression
from logistic_regression import generate_binary_logistic_data, fit_binary_logistic_model
data = generate_binary_logistic_data()
model, X, y = fit_binary_logistic_model(data)
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

**Implementation**: See `generate_binary_logistic_data()` in the Python file.

The function simulates applicant features and approval status based on a linear combination of predictors:
- **Data Generation**: Simulates applicant features and approval status based on a linear combination of predictors.
- **Log Odds Calculation**: The log-odds are a linear function of predictors, reflecting their influence on approval.
- **Probability Conversion**: The log-odds are transformed to probabilities using the logistic function.
- **Outcome Generation**: The binary outcome is sampled from a Bernoulli distribution with the calculated probability.

---

### Fitting and Interpreting the Model

**Implementation**: See `fit_binary_logistic_model()` and `interpret_logistic_coefficients()` in the Python file.

The model fitting process includes:
- **Model Fitting**: Uses `statsmodels.Logit()` to fit a logistic regression model.
- **Coefficients**: The output shows the estimated $`\beta`$ values for each predictor.
- **Odds Ratios**: Exponentiating the coefficients gives the odds ratios, which are easier to interpret.
- **Confidence Intervals**: The 95% CI for each odds ratio helps assess statistical significance and effect size.

#### Interpreting Odds Ratios
- If $`\text{OR} > 1`$: A one-unit increase in the predictor increases the odds of the event.
- If $`\text{OR} < 1`$: A one-unit increase in the predictor decreases the odds of the event.

---

### Model Interpretation

**Implementation**: See `interpret_logistic_coefficients()` in the Python file.

This function prints a human-readable interpretation for each predictor, including the direction and magnitude of the effect.

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

**Implementation**: See `logistic_diagnostics()` in the Python file.

The diagnostics include:
- **Residuals**: Deviance and Pearson residuals help assess model fit.
- **Leverage and Influence**: Identifies points that have a large effect on the model.
- **Hosmer-Lemeshow Test**: Assesses goodness-of-fit; $`p > 0.05`$ suggests adequate fit.
- **Cook's Distance**: Identifies influential observations.

---

## Model Performance Evaluation

### Classification Metrics

**Implementation**: See `calculate_classification_metrics()` in the Python file.

The metrics include:
- **Confusion Matrix**: Shows counts of true/false positives/negatives.
- **Accuracy**: Proportion of correct predictions.
- **Sensitivity (Recall)**: Proportion of actual positives correctly identified.
- **Specificity**: Proportion of actual negatives correctly identified.
- **Precision**: Proportion of positive predictions that are correct.
- **F1 Score**: Harmonic mean of precision and recall.

---

### ROC Curve and AUC

**Implementation**: See `plot_roc_curve()` in the Python file.

The ROC analysis includes:
- **ROC Curve**: Plots sensitivity vs. 1-specificity for all thresholds.
- **AUC (Area Under Curve)**: Measures overall model discrimination (1 = perfect, 0.5 = random).
- **Optimal Threshold**: The threshold that maximizes sensitivity and specificity.

---

### Cross-Validation

**Implementation**: See `cv_logistic()` in the Python file.

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

**Implementation**: See `generate_multinomial_data()` and `fit_multinomial_model()` in the Python file.

- **Data Generation**: Simulates predictors and assigns a risk category based on multinomial probabilities.

---

### Fitting and Interpreting the Multinomial Model

**Implementation**: See `fit_multinomial_model()` in the Python file.

- **Model Fitting**: Uses `statsmodels.MNLogit()` to fit a multinomial logistic regression.
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

**Implementation**: See `fit_ordinal_model()` in the Python file.

- **Model Fitting**: Uses `sklearn.LogisticRegression()` with multinomial option.
- **Thresholds**: Cutpoints between categories.
- **Odds Ratios**: Interpreted as the odds of being in a higher vs. lower category.

---

### Model Comparison

**Implementation**: See `model_comparison()` in the Python file.

- **AIC/BIC**: Lower values indicate better fit, penalizing for complexity.
- **Likelihood Ratio Test**: Compares nested models (e.g., ordinal vs. multinomial).
- **Interpretation**: If the proportional odds assumption holds, ordinal is preferred; otherwise, multinomial.

---

## Practical Examples

### Example 1: Credit Card Approval

**Implementation**: See `generate_credit_approval_data()` and `fit_credit_approval_model()` in the Python file.

- **Simulated Data**: Models credit approval based on applicant features.
- **Model Fitting**: Includes categorical predictors (e.g., home ownership).
- **Interpretation**: Odds ratios for each predictor, including categorical levels.

---

### Example 2: Medical Diagnosis

**Implementation**: See `generate_medical_data()`, `fit_medical_model()`, and `predict_medical_risk()` in the Python file.

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

**Implementation**: See `choose_logistic_model()` in the Python file.

---

### Reporting Guidelines

- Report coefficients, odds ratios, and confidence intervals.
- Include model fit statistics (AIC, BIC, log-likelihood).
- Report diagnostics (e.g., Hosmer-Lemeshow test, ROC/AUC).
- Discuss limitations, assumptions, and recommendations.

**Implementation**: See `generate_logistic_report()` in the Python file.

---

## Exercises

### Exercise 1: Binary Logistic Regression
**Objective:** Fit a binary logistic regression model to a dataset. Interpret the coefficients and odds ratios.
- *Hint:* Use the functions `generate_exercise_data()` and `fit_binary_logistic_model()` from the Python file.
- *Learning Outcome:* Understand model fitting and interpretation.

### Exercise 2: Model Diagnostics
**Objective:** Perform comprehensive diagnostics for a logistic regression model, including residual analysis and influence diagnostics.
- *Hint:* Use `logistic_diagnostics()` function from the Python file.
- *Learning Outcome:* Assess model fit and identify influential points.

### Exercise 3: Model Performance
**Objective:** Evaluate model performance using ROC curves, classification metrics, and cross-validation.
- *Hint:* Use `evaluate_model_performance()`, `plot_roc_curve()`, and `cv_logistic()` functions from the Python file.
- *Learning Outcome:* Understand model evaluation and validation.

### Exercise 4: Multinomial Logistic Regression
**Objective:** Fit a multinomial logistic regression model and compare it with binary and ordinal models.
- *Hint:* Use `generate_exercise_data()`, `fit_multinomial_model()`, and `model_comparison()` functions from the Python file.
- *Learning Outcome:* Understand when and how to use multinomial and ordinal models.

### Exercise 5: Real-World Application
**Objective:** Apply logistic regression to a real-world classification problem and generate a comprehensive report.
- *Hint:* Use the practical example functions and `generate_logistic_report()` from the Python file.
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
