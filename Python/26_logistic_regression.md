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

```python
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(123)

# Generate sample data for binary logistic regression
n_samples = 500

# Create predictor variables
logistic_data = pd.DataFrame({
    'age': np.random.normal(45, 10, n_samples),
    'income': np.random.normal(60000, 15000, n_samples),
    'education_years': np.random.normal(14, 3, n_samples),
    'credit_score': np.random.normal(700, 100, n_samples),
    'debt_ratio': np.random.normal(0.3, 0.1, n_samples)
})

# Create binary outcome based on predictors
# Higher age, income, education, credit score, and lower debt ratio increase probability
log_odds = (-2 + 0.05 * logistic_data['age'] + 
            0.00001 * logistic_data['income'] + 
            0.2 * logistic_data['education_years'] + 
            0.01 * logistic_data['credit_score'] - 
            3 * logistic_data['debt_ratio'])

# Convert to probability
prob = 1 / (1 + np.exp(-log_odds))

# Generate binary outcome
logistic_data['approved'] = np.random.binomial(1, prob)
logistic_data['approved'] = logistic_data['approved'].map({0: 'Rejected', 1: 'Approved'})

print("Sample of the data:")
print(logistic_data.head())
print(logistic_data['approved'].value_counts())
```

#### Code Explanation
- **Data Generation**: Simulates applicant features and approval status based on a linear combination of predictors.
- **Log Odds Calculation**: The log-odds are a linear function of predictors, reflecting their influence on approval.
- **Probability Conversion**: The log-odds are transformed to probabilities using the logistic function.
- **Outcome Generation**: The binary outcome is sampled from a Bernoulli distribution with the calculated probability.

---

### Fitting and Interpreting the Model

```python
# Prepare data for modeling
logistic_data['approved_bin'] = (logistic_data['approved'] == 'Approved').astype(int)
X = logistic_data[['age', 'income', 'education_years', 'credit_score', 'debt_ratio']]
X = sm.add_constant(X)
y = logistic_data['approved_bin']

# Fit logistic regression model
logistic_model = sm.Logit(y, X).fit()

# Model summary
print("Logistic Regression Model Summary:")
print(logistic_model.summary())

# Extract coefficients
coefficients = logistic_model.params
print("Model Coefficients:")
print(coefficients.round(4))

# Odds ratios
odds_ratios = np.exp(coefficients)
print("Odds Ratios:")
print(odds_ratios.round(4))

# Confidence intervals for odds ratios
conf_int = logistic_model.conf_int()
odds_ratios_ci = np.exp(conf_int)
print("Odds Ratios with 95% Confidence Intervals:")
print(odds_ratios_ci.round(4))
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

```python
# Function to interpret logistic regression coefficients
def interpret_logistic_coefficients(model):
    coefficients = model.params
    odds_ratios = np.exp(coefficients)
    
    print("=== LOGISTIC REGRESSION INTERPRETATION ===\n")
    
    for i, var_name in enumerate(coefficients.index[1:], 1):  # Skip intercept
        coef_value = coefficients[i]
        odds_ratio = odds_ratios[i]
        
        print(f"Variable: {var_name}")
        print(f"Coefficient: {coef_value:.4f}")
        print(f"Odds Ratio: {odds_ratio:.4f}")
        
        if odds_ratio > 1:
            print(f"Interpretation: A one-unit increase in {var_name} "
                  f"increases the odds of the event by {(odds_ratio - 1) * 100:.1f}%")
        else:
            print(f"Interpretation: A one-unit increase in {var_name} "
                  f"decreases the odds of the event by {(1 - odds_ratio) * 100:.1f}%")
        print()

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

```python
# Residual analysis for logistic regression
def logistic_diagnostics(model, data):
    print("=== LOGISTIC REGRESSION DIAGNOSTICS ===\n")
    
    # Get fitted probabilities
    fitted_probs = model.predict(X)
    
    # Calculate residuals (approximation for logistic regression)
    y_actual = y.values
    residuals = y_actual - fitted_probs
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs fitted values
    axes[0, 0].scatter(fitted_probs, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Probabilities')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True)
    
    # Q-Q plot of residuals
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot of Residuals')
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Histogram of Residuals')
    axes[1, 0].grid(True)
    
    # Leverage plot (simplified)
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    axes[1, 1].scatter(leverage, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Leverage')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Hosmer-Lemeshow test (approximation)
    def hosmer_lemeshow_test(y_true, y_pred, g=10):
        # Group observations by predicted probabilities
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        df['group'] = pd.qcut(df['y_pred'], g, labels=False)
        
        # Calculate observed and expected frequencies
        observed = df.groupby('group')['y_true'].sum()
        expected = df.groupby('group')['y_pred'].sum()
        
        # Chi-square statistic
        chi_square = ((observed - expected) ** 2 / expected).sum()
        p_value = 1 - stats.chi2.cdf(chi_square, g-2)
        
        return chi_square, p_value
    
    hl_stat, hl_p_value = hosmer_lemeshow_test(y_actual, fitted_probs)
    print(f"Hosmer-Lemeshow Test:")
    print(f"Chi-square statistic: {hl_stat:.3f}")
    print(f"p-value: {hl_p_value:.4f}")
    
    if hl_p_value > 0.05:
        print("Model fits well (p > 0.05)")
    else:
        print("Model may not fit well (p ≤ 0.05)")
    
    # Influential observations
    influence = model.get_influence()
    cook_dist = influence.cooks_distance[0]
    influential = np.where(cook_dist > 4/len(cook_dist))[0]
    
    print(f"\nInfluential Observations (Cook's distance > 4/n):")
    if len(influential) > 0:
        print(influential)
    else:
        print("No influential observations detected.")
    
    return {
        'residuals': residuals,
        'fitted_probs': fitted_probs,
        'leverage': leverage,
        'cook_dist': cook_dist,
        'hl_stat': hl_stat,
        'hl_p_value': hl_p_value
    }

# Apply diagnostics
diagnostics_result = logistic_diagnostics(logistic_model, logistic_data)
```

#### Code Explanation
- **Residuals**: Deviance and Pearson residuals help assess model fit.
- **Leverage and Influence**: Identifies points that have a large effect on the model.
- **Hosmer-Lemeshow Test**: Assesses goodness-of-fit; $`p > 0.05`$ suggests adequate fit.
- **Cook's Distance**: Identifies influential observations.

---

## Model Performance Evaluation

### Classification Metrics

```python
# Function to calculate classification metrics
def calculate_classification_metrics(actual, predicted, threshold=0.5):
    # Convert probabilities to predictions
    predicted_class = (predicted >= threshold).astype(int)
    
    # Create confusion matrix
    cm = confusion_matrix(actual, predicted_class)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # Recall
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score
    }

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123, stratify=y
)

# Fit model on training data
train_model = sm.Logit(y_train, X_train).fit()

# Predictions on test data
test_predictions = train_model.predict(X_test)
test_actual = y_test.values

# Calculate metrics
metrics = calculate_classification_metrics(test_actual, test_predictions)

print("Classification Metrics:")
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Sensitivity (Recall): {metrics['sensitivity']:.3f}")
print(f"Specificity: {metrics['specificity']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")

print("Confusion Matrix:")
print(metrics['confusion_matrix'])
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

```python
# ROC curve analysis
fpr, tpr, thresholds = roc_curve(test_actual, test_predictions)
auc_value = roc_auc_score(test_actual, test_predictions)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_value:.3f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("ROC Analysis:")
print(f"AUC: {auc_value:.3f}")

# Find optimal threshold (Youden's J statistic)
youden_j = tpr - fpr
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]
optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Optimal sensitivity: {optimal_sensitivity:.3f}")
print(f"Optimal specificity: {optimal_specificity:.3f}")

# Calculate metrics with optimal threshold
optimal_metrics = calculate_classification_metrics(test_actual, test_predictions, optimal_threshold)

print("\nMetrics with Optimal Threshold:")
print(f"Accuracy: {optimal_metrics['accuracy']:.3f}")
print(f"Sensitivity: {optimal_metrics['sensitivity']:.3f}")
print(f"Specificity: {optimal_metrics['specificity']:.3f}")
```

#### Code Explanation
- **ROC Curve**: Plots sensitivity vs. 1-specificity for all thresholds.
- **AUC (Area Under Curve)**: Measures overall model discrimination (1 = perfect, 0.5 = random).
- **Optimal Threshold**: The threshold that maximizes sensitivity and specificity.

---

### Cross-Validation

```python
# K-fold cross-validation
from sklearn.model_selection import KFold

def cv_logistic(data, k=5):
    n = len(data)
    kf = KFold(n_splits=k, shuffle=True, random_state=123)
    
    cv_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
        # Split data
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]
        
        # Fit model
        fold_model = sm.Logit(y_train_fold, X_train_fold).fit(disp=0)
        
        # Predictions
        fold_pred = fold_model.predict(X_test_fold)
        fold_actual = y_test_fold.values
        
        # Calculate metrics
        fold_metrics = calculate_classification_metrics(fold_actual, fold_pred)
        cv_metrics.append(fold_metrics)
    
    # Average metrics across folds
    avg_accuracy = np.mean([m['accuracy'] for m in cv_metrics])
    avg_sensitivity = np.mean([m['sensitivity'] for m in cv_metrics])
    avg_specificity = np.mean([m['specificity'] for m in cv_metrics])
    avg_f1 = np.mean([m['f1_score'] for m in cv_metrics])
    
    return {
        'fold_metrics': cv_metrics,
        'avg_accuracy': avg_accuracy,
        'avg_sensitivity': avg_sensitivity,
        'avg_specificity': avg_specificity,
        'avg_f1': avg_f1
    }

# Apply cross-validation
cv_results = cv_logistic(logistic_data, k=5)

print("Cross-Validation Results:")
print(f"Average Accuracy: {cv_results['avg_accuracy']:.3f}")
print(f"Average Sensitivity: {cv_results['avg_sensitivity']:.3f}")
print(f"Average Specificity: {cv_results['avg_specificity']:.3f}")
print(f"Average F1 Score: {cv_results['avg_f1']:.3f}")
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

```python
# Import multinomial logistic regression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

# Generate multinomial data
np.random.seed(123)
n_multinomial = 400

multinomial_data = pd.DataFrame({
    'age': np.random.normal(45, 10, n_multinomial),
    'income': np.random.normal(60000, 15000, n_multinomial),
    'education_years': np.random.normal(14, 3, n_multinomial),
    'credit_score': np.random.normal(700, 100, n_multinomial),
    'debt_ratio': np.random.normal(0.3, 0.1, n_multinomial)
})

# Create multinomial outcome (3 categories)
# Category 0: Low risk, Category 1: Medium risk, Category 2: High risk
log_odds_medium = (-1 + 0.02 * multinomial_data['age'] + 
                   0.000005 * multinomial_data['income'] + 
                   0.1 * multinomial_data['education_years'] + 
                   0.005 * multinomial_data['credit_score'] - 
                   1.5 * multinomial_data['debt_ratio'])

log_odds_high = (-2 + 0.03 * multinomial_data['age'] + 
                 0.000008 * multinomial_data['income'] + 
                 0.15 * multinomial_data['education_years'] + 
                 0.008 * multinomial_data['credit_score'] - 
                 2 * multinomial_data['debt_ratio'])

# Calculate probabilities
prob_medium = np.exp(log_odds_medium) / (1 + np.exp(log_odds_medium) + np.exp(log_odds_high))
prob_high = np.exp(log_odds_high) / (1 + np.exp(log_odds_medium) + np.exp(log_odds_high))
prob_low = 1 - prob_medium - prob_high

# Generate multinomial outcome
probs = np.column_stack([prob_low, prob_medium, prob_high])
multinomial_data['risk_category'] = np.array([np.random.choice(3, p=p) for p in probs])

# Create labels
risk_labels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
multinomial_data['risk_category_label'] = multinomial_data['risk_category'].map(risk_labels)

print("Multinomial Data Summary:")
print(multinomial_data['risk_category_label'].value_counts())
```

#### Code Explanation
- **Data Generation**: Simulates predictors and assigns a risk category based on multinomial probabilities.

---

### Fitting and Interpreting the Multinomial Model

```python
# Prepare data for multinomial regression
X_multinomial = multinomial_data[['age', 'income', 'education_years', 'credit_score', 'debt_ratio']]
X_multinomial = sm.add_constant(X_multinomial)
y_multinomial = multinomial_data['risk_category']

# Fit multinomial logistic regression using statsmodels
multinomial_model = sm.MNLogit(y_multinomial, X_multinomial).fit()

# Model summary
print("Multinomial Logistic Regression Summary:")
print(multinomial_model.summary())

# Extract coefficients
multinomial_coef = multinomial_model.params
print("Multinomial Coefficients:")
print(multinomial_coef.round(4))

# Calculate odds ratios
multinomial_odds = np.exp(multinomial_coef)
print("Multinomial Odds Ratios:")
print(multinomial_odds.round(4))

# Z-statistics and p-values
z_stats = multinomial_model.tvalues
p_values = multinomial_model.pvalues

print("Z-statistics:")
print(z_stats.round(3))
print("P-values:")
print(p_values.round(4))
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

```python
# For ordinal logistic regression, we'll use a custom implementation
# since statsmodels doesn't have a direct ordinal logistic regression function

# Create ordinal outcome (convert risk category to numeric)
ordinal_data = multinomial_data.copy()
ordinal_data['risk_ordinal'] = ordinal_data['risk_category']

# Fit ordinal logistic regression using sklearn (simplified approach)
from sklearn.linear_model import LogisticRegression

# Prepare data
X_ordinal = ordinal_data[['age', 'income', 'education_years', 'credit_score', 'debt_ratio']]
y_ordinal = ordinal_data['risk_ordinal']

# Fit ordinal logistic regression (using sklearn's multinomial option)
ordinal_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=123)
ordinal_model.fit(X_ordinal, y_ordinal)

# Model summary (simplified)
print("Ordinal Logistic Regression Summary:")
print(f"Intercept: {ordinal_model.intercept_}")
print("Coefficients:")
for i, feature in enumerate(X_ordinal.columns):
    print(f"{feature}: {ordinal_model.coef_[0][i]:.4f}")

# Calculate odds ratios
ordinal_odds = np.exp(ordinal_model.coef_[0])
print("Ordinal Odds Ratios:")
for i, feature in enumerate(X_ordinal.columns):
    print(f"{feature}: {ordinal_odds[i]:.4f}")

# Thresholds (intercepts for each class)
print("Thresholds (Intercepts):")
for i, intercept in enumerate(ordinal_model.intercept_):
    print(f"Class {i}: {intercept:.4f}")
```

#### Code Explanation
- **Model Fitting**: `polr()` fits an ordinal logistic regression.
- **Thresholds**: Cutpoints between categories.
- **Odds Ratios**: Interpreted as the odds of being in a higher vs. lower category.

---

### Model Comparison

```python
# Compare binary, multinomial, and ordinal models
def model_comparison(binary_data, multinomial_data):
    print("=== MODEL COMPARISON ===\n")
    
    # Binary model (combine medium and high risk)
    binary_data_combined = binary_data.copy()
    binary_data_combined['approved_bin'] = (binary_data_combined['approved'] == 'Approved').astype(int)
    
    # Fit binary model
    X_binary = binary_data_combined[['age', 'income', 'education_years', 'credit_score', 'debt_ratio']]
    X_binary = sm.add_constant(X_binary)
    y_binary = binary_data_combined['approved_bin']
    binary_model = sm.Logit(y_binary, X_binary).fit(disp=0)
    
    # AIC comparison
    binary_aic = binary_model.aic
    multinomial_aic = multinomial_model.aic
    
    # For ordinal model, we'll use a simplified AIC calculation
    ordinal_aic = len(X_ordinal.columns) * 2 + 2 * (-ordinal_model.score(X_ordinal, y_ordinal))
    
    print("AIC Comparison:")
    print(f"Binary model: {binary_aic:.2f}")
    print(f"Multinomial model: {multinomial_aic:.2f}")
    print(f"Ordinal model: {ordinal_aic:.2f}\n")
    
    # BIC comparison
    binary_bic = binary_model.bic
    multinomial_bic = multinomial_model.bic
    
    # For ordinal model, we'll use a simplified BIC calculation
    n = len(y_ordinal)
    ordinal_bic = len(X_ordinal.columns) * np.log(n) + 2 * (-ordinal_model.score(X_ordinal, y_ordinal))
    
    print("BIC Comparison:")
    print(f"Binary model: {binary_bic:.2f}")
    print(f"Multinomial model: {multinomial_bic:.2f}")
    print(f"Ordinal model: {ordinal_bic:.2f}\n")
    
    # Likelihood ratio test for ordinal vs multinomial (simplified)
    # Since we're using different implementations, we'll compare model performance
    print("Model Performance Comparison:")
    print("Note: Direct likelihood ratio test not available due to different implementations")
    print("Consider comparing classification accuracy and other metrics instead")
    
    return {
        'binary_aic': binary_aic,
        'multinomial_aic': multinomial_aic,
        'ordinal_aic': ordinal_aic,
        'binary_bic': binary_bic,
        'multinomial_bic': multinomial_bic,
        'ordinal_bic': ordinal_bic
    }

# Apply model comparison
comparison_results = model_comparison(logistic_data, multinomial_data)
```

#### Code Explanation
- **AIC/BIC**: Lower values indicate better fit, penalizing for complexity.
- **Likelihood Ratio Test**: Compares nested models (e.g., ordinal vs. multinomial).
- **Interpretation**: If the proportional odds assumption holds, ordinal is preferred; otherwise, multinomial.

---

## Practical Examples

### Example 1: Credit Card Approval

```python
# Simulate credit card application data
np.random.seed(123)
n_applications = 1000

credit_data = pd.DataFrame({
    'age': np.random.normal(35, 8, n_applications),
    'income': np.random.normal(75000, 20000, n_applications),
    'credit_score': np.random.normal(720, 80, n_applications),
    'debt_to_income': np.random.normal(0.25, 0.1, n_applications),
    'employment_years': np.random.normal(5, 3, n_applications),
    'previous_defaults': np.random.poisson(0.3, n_applications),
    'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], 
                                      n_applications, p=[0.4, 0.3, 0.3])
})

# Create approval probability
log_odds_credit = (-3 + 0.02 * credit_data['age'] + 
                   0.00001 * credit_data['income'] + 
                   0.015 * credit_data['credit_score'] - 
                   2 * credit_data['debt_to_income'] + 
                   0.1 * credit_data['employment_years'] - 
                   0.5 * credit_data['previous_defaults'])

# Add home ownership effect
log_odds_credit[credit_data['home_ownership'] == 'Own'] += 0.3
log_odds_credit[credit_data['home_ownership'] == 'Mortgage'] += 0.1

prob_credit = 1 / (1 + np.exp(-log_odds_credit))
credit_data['approved'] = np.random.binomial(1, prob_credit)
credit_data['approved'] = credit_data['approved'].map({0: 'Rejected', 1: 'Approved'})

# Prepare data for modeling
credit_data['approved_bin'] = (credit_data['approved'] == 'Approved').astype(int)

# Create dummy variables for categorical predictors
credit_data_encoded = pd.get_dummies(credit_data, columns=['home_ownership'], drop_first=True)

# Fit credit approval model
X_credit = credit_data_encoded[['age', 'income', 'credit_score', 'debt_to_income', 
                               'employment_years', 'previous_defaults', 
                               'home_ownership_Own', 'home_ownership_Rent']]
X_credit = sm.add_constant(X_credit)
y_credit = credit_data_encoded['approved_bin']

credit_model = sm.Logit(y_credit, X_credit).fit()

print("Credit Card Approval Model:")
print(credit_model.summary())

# Odds ratios
credit_odds = np.exp(credit_model.params)
print("Odds Ratios for Credit Approval:")
print(credit_odds.round(4))
```

#### Code Explanation
- **Simulated Data**: Models credit approval based on applicant features.
- **Model Fitting**: Includes categorical predictors (e.g., home ownership).
- **Interpretation**: Odds ratios for each predictor, including categorical levels.

---

### Example 2: Medical Diagnosis

```python
# Simulate medical diagnosis data
np.random.seed(123)
n_patients = 800

medical_data = pd.DataFrame({
    'age': np.random.normal(60, 15, n_patients),
    'bmi': np.random.normal(28, 5, n_patients),
    'blood_pressure': np.random.normal(140, 20, n_patients),
    'cholesterol': np.random.normal(200, 40, n_patients),
    'glucose': np.random.normal(100, 20, n_patients),
    'family_history': np.random.choice(['No', 'Yes'], n_patients, p=[0.7, 0.3]),
    'smoking': np.random.choice(['Never', 'Former', 'Current'], n_patients, p=[0.4, 0.3, 0.3])
})

# Create disease probability (diabetes)
log_odds_diabetes = (-4 + 0.03 * medical_data['age'] + 
                     0.05 * medical_data['bmi'] + 
                     0.01 * medical_data['blood_pressure'] + 
                     0.005 * medical_data['cholesterol'] + 
                     0.02 * medical_data['glucose'])

# Add categorical effects
log_odds_diabetes[medical_data['family_history'] == 'Yes'] += 0.8
log_odds_diabetes[medical_data['smoking'] == 'Current'] += 0.3
log_odds_diabetes[medical_data['smoking'] == 'Former'] += 0.1

prob_diabetes = 1 / (1 + np.exp(-log_odds_diabetes))
medical_data['diabetes'] = np.random.binomial(1, prob_diabetes)
medical_data['diabetes'] = medical_data['diabetes'].map({0: 'No Diabetes', 1: 'Diabetes'})

# Prepare data for modeling
medical_data['diabetes_bin'] = (medical_data['diabetes'] == 'Diabetes').astype(int)

# Create dummy variables for categorical predictors
medical_data_encoded = pd.get_dummies(medical_data, columns=['family_history', 'smoking'], drop_first=True)

# Fit medical diagnosis model
X_medical = medical_data_encoded[['age', 'bmi', 'blood_pressure', 'cholesterol', 'glucose',
                                 'family_history_Yes', 'smoking_Former', 'smoking_Current']]
X_medical = sm.add_constant(X_medical)
y_medical = medical_data_encoded['diabetes_bin']

medical_model = sm.Logit(y_medical, X_medical).fit()

print("Medical Diagnosis Model:")
print(medical_model.summary())

# Risk prediction for new patient
new_patient = pd.DataFrame({
    'age': [65],
    'bmi': [32],
    'blood_pressure': [150],
    'cholesterol': [220],
    'glucose': [120],
    'family_history_Yes': [1],
    'smoking_Former': [1],
    'smoking_Current': [0]
})
new_patient = sm.add_constant(new_patient)

predicted_risk = medical_model.predict(new_patient)[0]
print(f"Predicted diabetes risk for new patient: {predicted_risk:.3f}")
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

```python
# Function to help choose appropriate logistic regression model
def choose_logistic_model(data, outcome_var):
    print("=== LOGISTIC REGRESSION MODEL SELECTION ===\n")
    
    # Check outcome variable
    outcome_levels = data[outcome_var].unique()
    n_levels = len(outcome_levels)
    
    print(f"Outcome variable: {outcome_var}")
    print(f"Number of levels: {n_levels}")
    print(f"Levels: {', '.join(outcome_levels)}\n")
    
    if n_levels == 2:
        print("RECOMMENDATION: Use binary logistic regression")
        print("- Suitable for binary outcomes")
        print("- Easy to interpret odds ratios")
        print("- Good for classification problems")
    elif n_levels == 3:
        print("RECOMMENDATION: Consider both multinomial and ordinal")
        print("- Multinomial: No ordering assumption")
        print("- Ordinal: Assumes proportional odds")
        print("- Test proportional odds assumption")
    else:
        print("RECOMMENDATION: Use multinomial logistic regression")
        print("- Suitable for multiple unordered categories")
        print("- More complex interpretation")
    
    # Check sample size
    n_samples = len(data)
    print(f"\nSample size: {n_samples}")
    
    if n_samples < 100:
        print("WARNING: Small sample size may affect model stability")
    elif n_samples < 500:
        print("Sample size is adequate for most applications")
    else:
        print("Sample size is good for complex models")
    
    # Check class balance
    outcome_counts = data[outcome_var].value_counts()
    min_count = outcome_counts.min()
    max_count = outcome_counts.max()
    imbalance_ratio = max_count / min_count
    
    print("\nClass balance:")
    print(outcome_counts)
    
    if imbalance_ratio > 3:
        print("WARNING: Class imbalance detected")
        print("Consider: resampling, different thresholds, or class weights")
    
    return {
        'n_levels': n_levels,
        'n_samples': n_samples,
        'imbalance_ratio': imbalance_ratio
    }

# Apply model selection
model_selection = choose_logistic_model(logistic_data, "approved")
```

---

### Reporting Guidelines

- Report coefficients, odds ratios, and confidence intervals.
- Include model fit statistics (AIC, BIC, log-likelihood).
- Report diagnostics (e.g., Hosmer-Lemeshow test, ROC/AUC).
- Discuss limitations, assumptions, and recommendations.

```python
# Function to generate comprehensive logistic regression report
def generate_logistic_report(model, data, test_data=None):
    print("=== LOGISTIC REGRESSION REPORT ===\n")
    
    # Model summary
    print("MODEL SUMMARY:")
    print(model.summary())
    
    # Odds ratios with confidence intervals
    print("\nODDS RATIOS WITH 95% CONFIDENCE INTERVALS:")
    conf_int = model.conf_int()
    odds_ci = np.exp(conf_int)
    odds_table = pd.DataFrame({
        'Variable': model.params.index,
        'Odds_Ratio': np.exp(model.params),
        'CI_Lower': odds_ci.iloc[:, 0],
        'CI_Upper': odds_ci.iloc[:, 1]
    })
    print(odds_table.round(3))
    
    # Model fit statistics
    print("\nMODEL FIT STATISTICS:")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")
    print(f"Log-likelihood: {model.llf:.2f}")
    
    # Hosmer-Lemeshow test (approximation)
    fitted_probs = model.predict()
    y_actual = model.model.endog
    
    def hosmer_lemeshow_test(y_true, y_pred, g=10):
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        df['group'] = pd.qcut(df['y_pred'], g, labels=False)
        observed = df.groupby('group')['y_true'].sum()
        expected = df.groupby('group')['y_pred'].sum()
        chi_square = ((observed - expected) ** 2 / expected).sum()
        p_value = 1 - stats.chi2.cdf(chi_square, g-2)
        return chi_square, p_value
    
    hl_stat, hl_p_value = hosmer_lemeshow_test(y_actual, fitted_probs)
    print(f"Hosmer-Lemeshow test p-value: {hl_p_value:.4f}")
    
    if hl_p_value > 0.05:
        print("Model fits well")
    else:
        print("Model fit may be inadequate")
    
    # Performance metrics (if test data provided)
    if test_data is not None:
        print("\nPERFORMANCE METRICS:")
        test_pred = model.predict(test_data)
        test_actual = test_data.iloc[:, -1]  # Assuming last column is target
        
        # ROC analysis
        fpr, tpr, _ = roc_curve(test_actual, test_pred)
        auc_value = roc_auc_score(test_actual, test_pred)
        
        print(f"AUC: {auc_value:.3f}")
        
        # Classification metrics
        metrics = calculate_classification_metrics(test_actual, test_pred)
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"Specificity: {metrics['specificity']:.3f}")
        print(f"F1 Score: {metrics['f1_score']:.3f}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("- Check for influential observations")
    print("- Validate model assumptions")
    print("- Consider model diagnostics")
    print("- Test model on new data")

# Generate report
generate_logistic_report(logistic_model, logistic_data, test_data)
```

---

## Exercises

### Exercise 1: Binary Logistic Regression
**Objective:** Fit a binary logistic regression model to a dataset. Interpret the coefficients and odds ratios.
- *Hint:* Use `statsmodels.Logit()` or `sklearn.LogisticRegression()`.
- *Learning Outcome:* Understand model fitting and interpretation.

```python
# Exercise 1: Binary Logistic Regression
# Simulate data for customer churn prediction
np.random.seed(123)
n_customers = 600

churn_data = pd.DataFrame({
    'tenure': np.random.normal(24, 12, n_customers),
    'monthly_charges': np.random.normal(65, 20, n_customers),
    'total_charges': np.random.normal(1500, 800, n_customers),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                     n_customers, p=[0.5, 0.3, 0.2]),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
                                      n_customers, p=[0.3, 0.2, 0.25, 0.25])
})

# Create churn probability
log_odds_churn = (-2 + 0.05 * churn_data['tenure'] - 
                  0.02 * churn_data['monthly_charges'] + 
                  0.001 * churn_data['total_charges'])

# Add categorical effects
log_odds_churn[churn_data['contract_type'] == 'Month-to-month'] += 0.8
log_odds_churn[churn_data['contract_type'] == 'One year'] += 0.2
log_odds_churn[churn_data['payment_method'] == 'Electronic check'] += 0.5

prob_churn = 1 / (1 + np.exp(-log_odds_churn))
churn_data['churn'] = np.random.binomial(1, prob_churn)
churn_data['churn'] = churn_data['churn'].map({0: 'No Churn', 1: 'Churn'})
```

### Exercise 2: Model Diagnostics
**Objective:** Perform comprehensive diagnostics for a logistic regression model, including residual analysis and influence diagnostics.
- *Hint:* Use residual plots, Cook's distance, and the Hosmer-Lemeshow test.
- *Learning Outcome:* Assess model fit and identify influential points.

### Exercise 3: Model Performance
**Objective:** Evaluate model performance using ROC curves, classification metrics, and cross-validation.
- *Hint:* Use `sklearn.metrics` for ROC/AUC and implement k-fold cross-validation.
- *Learning Outcome:* Understand model evaluation and validation.

### Exercise 4: Multinomial Logistic Regression
**Objective:** Fit a multinomial logistic regression model and compare it with binary and ordinal models.
- *Hint:* Use `statsmodels.MNLogit()` and `sklearn.LogisticRegression()` with `multi_class='multinomial'`.
- *Learning Outcome:* Understand when and how to use multinomial and ordinal models.

```python
# Exercise 4: Multinomial Logistic Regression
# Simulate data for customer satisfaction levels
np.random.seed(123)
n_satisfaction = 500

satisfaction_data = pd.DataFrame({
    'age': np.random.normal(45, 12, n_satisfaction),
    'income': np.random.normal(60000, 20000, n_satisfaction),
    'service_quality': np.random.normal(7, 2, n_satisfaction),
    'wait_time': np.random.normal(10, 5, n_satisfaction),
    'previous_experience': np.random.choice(['Poor', 'Fair', 'Good'], n_satisfaction, p=[0.2, 0.3, 0.5])
})

# Create satisfaction probability (3 levels: Low, Medium, High)
log_odds_medium = (-1 + 0.02 * satisfaction_data['age'] + 
                   0.00001 * satisfaction_data['income'] + 
                   0.3 * satisfaction_data['service_quality'] - 
                   0.05 * satisfaction_data['wait_time'])

log_odds_high = (-2 + 0.03 * satisfaction_data['age'] + 
                 0.000015 * satisfaction_data['income'] + 
                 0.5 * satisfaction_data['service_quality'] - 
                 0.08 * satisfaction_data['wait_time'])

# Add categorical effects
log_odds_medium[satisfaction_data['previous_experience'] == 'Good'] += 0.5
log_odds_high[satisfaction_data['previous_experience'] == 'Good'] += 1.0

# Calculate probabilities
prob_medium = np.exp(log_odds_medium) / (1 + np.exp(log_odds_medium) + np.exp(log_odds_high))
prob_high = np.exp(log_odds_high) / (1 + np.exp(log_odds_medium) + np.exp(log_odds_high))
prob_low = 1 - prob_medium - prob_high

# Generate outcome
probs = np.column_stack([prob_low, prob_medium, prob_high])
satisfaction_data['satisfaction'] = np.array([np.random.choice(3, p=p) for p in probs])
satisfaction_data['satisfaction'] = satisfaction_data['satisfaction'].map({0: 'Low', 1: 'Medium', 2: 'High'})
```

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