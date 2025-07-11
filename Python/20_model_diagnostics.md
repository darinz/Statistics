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

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data (using sklearn's boston dataset as mtcars equivalent)
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target

# Fit model
X = data[['RM', 'LSTAT', 'PTRATIO']]  # Using room size, poverty level, and pupil-teacher ratio
y = data['MEDV']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Residual plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
axes[0, 0].scatter(model.fittedvalues, model.resid)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot for normality
stats.probplot(model.resid, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

# Scale-Location plot
axes[1, 0].scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)))
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('sqrt(|Residuals|)')
axes[1, 0].set_title('Scale-Location')

# Residuals vs Leverage
leverage = model.get_influence().hat_matrix_diag
axes[1, 1].scatter(leverage, model.resid)
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residuals vs Leverage')

plt.tight_layout()
plt.show()

# Standardized and studentized residuals
std_resid = model.get_influence().resid_studentized_internal
stud_resid = model.get_influence().resid_studentized_external

# Q-Q plot for normality
plt.figure(figsize=(8, 6))
stats.probplot(stud_resid, dist="norm", plot=plt)
plt.title('Q-Q Plot of Studentized Residuals')
plt.show()

# Shapiro-Wilk test for normality
shapiro_stat, shapiro_p = stats.shapiro(stud_resid)
print(f"Shapiro-Wilk test: statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")
```

### Leverage and Influence

```python
# Leverage
leverage = model.get_influence().hat_matrix_diag
plt.figure(figsize=(10, 6))
plt.bar(range(len(leverage)), leverage)
plt.axhline(y=2 * (len(model.params) + 1) / len(data), color='red', linestyle='--', 
           label=f'Threshold: {2 * (len(model.params) + 1) / len(data):.3f}')
plt.xlabel('Observation')
plt.ylabel('Leverage')
plt.title('Leverage')
plt.legend()
plt.show()

# Cook's distance
cooksd = model.get_influence().cooks_distance[0]
plt.figure(figsize=(10, 6))
plt.bar(range(len(cooksd)), cooksd)
plt.axhline(y=4/len(cooksd), color='red', linestyle='--', 
           label=f'Threshold: {4/len(cooksd):.3f}')
plt.xlabel('Observation')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance")
plt.legend()
plt.show()

# Influence plot
fig, ax = plt.subplots(figsize=(10, 8))
influence_plot = model.get_influence()
ax.scatter(leverage, stud_resid, s=100*cooksd, alpha=0.6)
ax.set_xlabel('Leverage')
ax.set_ylabel('Studentized Residuals')
ax.set_title('Influence Plot')
ax.axhline(y=0, color='red', linestyle='--')
ax.axhline(y=2, color='red', linestyle='--')
ax.axhline(y=-2, color='red', linestyle='--')
plt.show()
```

### Multicollinearity

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("Variance Inflation Factors:")
print(vif_data)

# Tolerance (1/VIF)
vif_data["Tolerance"] = 1 / vif_data["VIF"]
print("\nTolerance:")
print(vif_data[["Variable", "Tolerance"]])
```

### Heteroscedasticity

```python
# Breusch-Pagan test
bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(model.resid, X)
print(f"Breusch-Pagan test: statistic = {bp_stat:.4f}, p-value = {bp_p:.4f}")

# Plot residuals vs fitted
plt.figure(figsize=(10, 6))
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()
```

### Independence

```python
# Durbin-Watson test for autocorrelation
dw_stat = durbin_watson(model.resid)
print(f"Durbin-Watson statistic: {dw_stat:.4f}")
print("Interpretation:")
print("- DW ≈ 2: No autocorrelation")
print("- DW < 2: Positive autocorrelation")
print("- DW > 2: Negative autocorrelation")
```

## Diagnostics for Logistic Regression

### Residuals and Influence

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report
import statsmodels.api as sm

# Create binary outcome (high vs low median value)
data['high_value'] = (data['MEDV'] > data['MEDV'].median()).astype(int)

# Fit logistic regression with statsmodels
X_logit = data[['RM', 'LSTAT', 'PTRATIO']]
X_logit = sm.add_constant(X_logit)
y_logit = data['high_value']
logit_model = sm.Logit(y_logit, X_logit).fit()

# Pearson and deviance residuals
pearson_resid = logit_model.resid_pearson
dev_resid = logit_model.resid_dev

# Plot residuals
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(logit_model.fittedvalues, pearson_resid)
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Pearson Residuals')
axes[0].set_title('Pearson Residuals vs Fitted')

axes[1].scatter(logit_model.fittedvalues, dev_resid)
axes[1].axhline(y=0, color='red', linestyle='--')
axes[1].set_xlabel('Fitted Values')
axes[1].set_ylabel('Deviance Residuals')
axes[1].set_title('Deviance Residuals vs Fitted')

plt.tight_layout()
plt.show()

# Leverage and Cook's distance
leverage_logit = logit_model.get_influence().hat_matrix_diag
cooksd_logit = logit_model.get_influence().cooks_distance[0]

plt.figure(figsize=(10, 6))
plt.scatter(leverage_logit, dev_resid)
plt.xlabel('Leverage')
plt.ylabel('Deviance Residuals')
plt.title('Deviance Residuals vs Leverage')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(range(len(cooksd_logit)), cooksd_logit)
plt.axhline(y=4/len(cooksd_logit), color='red', linestyle='--', 
           label=f'Threshold: {4/len(cooksd_logit):.3f}')
plt.xlabel('Observation')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance (Logistic)")
plt.legend()
plt.show()
```

### Model Fit and Classification

```python
# Pseudo R-squared
print("Model Summary:")
print(logit_model.summary())

# Hosmer-Lemeshow test (approximation using chi-square test on grouped residuals)
def hosmer_lemeshow_test(y_true, y_pred, n_groups=10):
    # Sort by predicted probabilities
    sorted_indices = np.argsort(y_pred)
    y_true_sorted = y_true.iloc[sorted_indices]
    y_pred_sorted = y_pred.iloc[sorted_indices]
    
    # Group into deciles
    group_size = len(y_true) // n_groups
    groups = []
    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size if i < n_groups - 1 else len(y_true)
        groups.append({
            'observed': y_true_sorted.iloc[start_idx:end_idx],
            'predicted': y_pred_sorted.iloc[start_idx:end_idx]
        })
    
    # Calculate chi-square statistic
    chi_square = 0
    for group in groups:
        observed_1 = group['observed'].sum()
        observed_0 = len(group['observed']) - observed_1
        expected_1 = group['predicted'].sum()
        expected_0 = len(group['predicted']) - expected_1
        
        if expected_1 > 0 and expected_0 > 0:
            chi_square += ((observed_1 - expected_1)**2 / expected_1 + 
                          (observed_0 - expected_0)**2 / expected_0)
    
    p_value = 1 - stats.chi2.cdf(chi_square, n_groups - 2)
    return chi_square, p_value

hl_stat, hl_p = hosmer_lemeshow_test(y_logit, pd.Series(logit_model.fittedvalues))
print(f"Hosmer-Lemeshow test: chi-square = {hl_stat:.4f}, p-value = {hl_p:.4f}")

# ROC curve and AUC
y_pred_proba = logit_model.fittedvalues
fpr, tpr, _ = roc_curve(y_logit, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print(f"AUC: {roc_auc:.4f}")
```

## Generalized Linear Model (GLM) Diagnostics

- Many diagnostics for linear and logistic regression extend to other GLMs (e.g., Poisson, multinomial).
- Use deviance, residuals, influence, and fit statistics as appropriate for the model family.

## Visualization and Interpretation

### Diagnostic Plot Matrix

```python
# For any statsmodels model
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
axes[0, 0].scatter(model.fittedvalues, model.resid)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot
stats.probplot(model.resid, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

# Scale-Location
axes[1, 0].scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)))
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('sqrt(|Residuals|)')
axes[1, 0].set_title('Scale-Location')

# Residuals vs Leverage
axes[1, 1].scatter(leverage, model.resid)
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residuals vs Leverage')

plt.tight_layout()
plt.show()
```

### Influence Plot

```python
# Influence plot
fig, ax = plt.subplots(figsize=(10, 8))
influence_plot = model.get_influence()
ax.scatter(leverage, stud_resid, s=100*cooksd, alpha=0.6)
ax.set_xlabel('Leverage')
ax.set_ylabel('Studentized Residuals')
ax.set_title('Influence Plot')
ax.axhline(y=0, color='red', linestyle='--')
ax.axhline(y=2, color='red', linestyle='--')
ax.axhline(y=-2, color='red', linestyle='--')
plt.show()
```

### Residuals vs Leverage Plot

```python
plt.figure(figsize=(10, 6))
plt.scatter(leverage, stud_resid)
plt.axhline(y=2, color='red', linestyle='--', label='y = 2')
plt.axhline(y=-2, color='red', linestyle='--', label='y = -2')
plt.xlabel('Leverage')
plt.ylabel('Studentized Residuals')
plt.title('Studentized Residuals vs Leverage')
plt.legend()
plt.show()
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
Fit a multiple regression model to the boston housing dataset. Perform and interpret all key diagnostics (residuals, leverage, influence, multicollinearity, heteroscedasticity, independence).

### Exercise 2: Logistic Regression Diagnostics
Fit a logistic regression model to predict high-value homes in the boston dataset. Perform and interpret residual, influence, and fit diagnostics.

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