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

### Python Implementation

```python
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 200
n_features = 10

# Create correlated features
np.random.seed(42)
X = np.random.randn(n_samples, n_features)
# Add some correlation between features
X[:, 2] = X[:, 0] * 0.8 + X[:, 1] * 0.2 + np.random.randn(n_samples) * 0.1
X[:, 3] = X[:, 1] * 0.7 + np.random.randn(n_samples) * 0.3

# Create target variable (only some features are important)
beta_true = np.array([2.0, -1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
y = X @ beta_true + np.random.randn(n_samples) * 0.5

# Create DataFrame
feature_names = [f'X{i+1}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['y'] = y

print("Dataset shape:", df.shape)
print("True coefficients:", beta_true)
```

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

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Forward Selection
forward_selector = SequentialFeatureSelector(
    LinearRegression(), 
    n_features_to_select=5, 
    direction='forward',
    scoring='neg_mean_squared_error',
    cv=5
)
forward_selector.fit(X_train, y_train)
forward_selected = forward_selector.get_support()
forward_features = [feature_names[i] for i in range(len(feature_names)) if forward_selected[i]]

# Backward Elimination
backward_selector = SequentialFeatureSelector(
    LinearRegression(), 
    n_features_to_select=5, 
    direction='backward',
    scoring='neg_mean_squared_error',
    cv=5
)
backward_selector.fit(X_train, y_train)
backward_selected = backward_selector.get_support()
backward_features = [feature_names[i] for i in range(len(feature_names)) if backward_selected[i]]

# Recursive Feature Elimination (RFE)
rfe = RFE(estimator=LinearRegression(), n_features_to_select=5)
rfe.fit(X_train, y_train)
rfe_selected = rfe.get_support()
rfe_features = [feature_names[i] for i in range(len(feature_names)) if rfe_selected[i]]

print("Forward Selection Features:", forward_features)
print("Backward Elimination Features:", backward_features)
print("RFE Features:", rfe_features)

# Compare models
def evaluate_model(X_train, X_test, y_train, y_test, feature_names, method_name):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate AIC and BIC
    n = len(y_train)
    p = X_train.shape[1]
    rss = np.sum((y_train - model.predict(X_train))**2)
    sigma2_hat = rss / n
    aic = n * np.log(sigma2_hat) + 2 * p
    bic = n * np.log(sigma2_hat) + p * np.log(n)
    
    print(f"\n{method_name}:")
    print(f"Features: {feature_names}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"AIC: {aic:.4f}")
    print(f"BIC: {bic:.4f}")
    
    return model, mse, r2, aic, bic

# Evaluate all models
full_model, full_mse, full_r2, full_aic, full_bic = evaluate_model(
    X_train, X_test, y_train, y_test, feature_names, "Full Model"
)

forward_model, forward_mse, forward_r2, forward_aic, forward_aic = evaluate_model(
    X_train[:, forward_selected], X_test[:, forward_selected], 
    y_train, y_test, forward_features, "Forward Selection"
)

backward_model, backward_mse, backward_r2, backward_aic, backward_bic = evaluate_model(
    X_train[:, backward_selected], X_test[:, backward_selected], 
    y_train, y_test, backward_features, "Backward Elimination"
)

rfe_model, rfe_mse, rfe_r2, rfe_aic, rfe_bic = evaluate_model(
    X_train[:, rfe_selected], X_test[:, rfe_selected], 
    y_train, y_test, rfe_features, "RFE"
)
```

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

```python
# Standardize features for regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

# Elastic Net
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train_scaled, y_train)
elastic_pred = elastic_net.predict(X_test_scaled)
elastic_mse = mean_squared_error(y_test, elastic_pred)
elastic_r2 = r2_score(y_test, elastic_pred)

# Compare coefficients
coef_comparison = pd.DataFrame({
    'True': beta_true,
    'OLS': full_model.coef_,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_,
    'Elastic Net': elastic_net.coef_
}, index=feature_names)

print("\nCoefficient Comparison:")
print(coef_comparison)

# Plot coefficient paths for different alpha values
alphas = np.logspace(-3, 1, 50)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    ridge_temp = Ridge(alpha=alpha)
    ridge_temp.fit(X_train_scaled, y_train)
    ridge_coefs.append(ridge_temp.coef_)
    
    lasso_temp = Lasso(alpha=alpha)
    lasso_temp.fit(X_train_scaled, y_train)
    lasso_coefs.append(lasso_temp.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

# Plot coefficient paths
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
for i in range(len(feature_names)):
    plt.plot(alphas, ridge_coefs[:, i], label=feature_names[i])
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Ridge Regression Coefficient Paths')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for i in range(len(feature_names)):
    plt.plot(alphas, lasso_coefs[:, i], label=feature_names[i])
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Lasso Regression Coefficient Paths')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Cross-validation for optimal alpha
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print(f"Optimal Ridge alpha: {ridge_cv.alpha_:.4f}")

lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=1000)
lasso_cv.fit(X_train_scaled, y_train)
print(f"Optimal Lasso alpha: {lasso_cv.alpha_:.4f}")

elastic_cv = ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5, max_iter=1000)
elastic_cv.fit(X_train_scaled, y_train)
print(f"Optimal Elastic Net alpha: {elastic_cv.alpha_:.4f}, l1_ratio: {elastic_cv.l1_ratio_:.1f}")
```

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

```python
# Principal Component Regression (PCR)
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Fit PCR with different numbers of components
n_components_range = range(1, min(X_train.shape[1] + 1, 11))
pcr_scores = []

for n_comp in n_components_range:
    pcr_model = LinearRegression()
    pcr_model.fit(X_train_pca[:, :n_comp], y_train)
    pcr_pred = pcr_model.predict(X_test_pca[:, :n_comp])
    pcr_mse = mean_squared_error(y_test, pcr_pred)
    pcr_scores.append(pcr_mse)

# Partial Least Squares (PLS)
pls_scores = []
for n_comp in n_components_range:
    pls_model = PLSRegression(n_components=n_comp)
    pls_model.fit(X_train_scaled, y_train)
    pls_pred = pls_model.predict(X_test_scaled)
    pls_mse = mean_squared_error(y_test, pls_pred)
    pls_scores.append(pls_mse)

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(n_components_range, pcr_scores, 'bo-', label='PCR')
plt.xlabel('Number of Components')
plt.ylabel('Mean Squared Error')
plt.title('Principal Component Regression')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(n_components_range, pls_scores, 'ro-', label='PLS')
plt.xlabel('Number of Components')
plt.ylabel('Mean Squared Error')
plt.title('Partial Least Squares')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Find optimal number of components
optimal_pcr_components = n_components_range[np.argmin(pcr_scores)]
optimal_pls_components = n_components_range[np.argmin(pls_scores)]

print(f"Optimal PCR components: {optimal_pcr_components}")
print(f"Optimal PLS components: {optimal_pls_components}")

# Fit final models
pcr_final = LinearRegression()
pcr_final.fit(X_train_pca[:, :optimal_pcr_components], y_train)
pcr_final_pred = pcr_final.predict(X_test_pca[:, :optimal_pcr_components])
pcr_final_mse = mean_squared_error(y_test, pcr_final_pred)

pls_final = PLSRegression(n_components=optimal_pls_components)
pls_final.fit(X_train_scaled, y_train)
pls_final_pred = pls_final.predict(X_test_scaled)
pls_final_mse = mean_squared_error(y_test, pls_final_pred)

print(f"PCR MSE: {pcr_final_mse:.4f}")
print(f"PLS MSE: {pls_final_mse:.4f}")
```

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

```python
# Residual analysis for the best model (let's use Lasso)
best_model = lasso_cv
y_pred_best = best_model.predict(X_test_scaled)
residuals = y_test - y_pred_best

# Diagnostic plots
plt.figure(figsize=(15, 10))

# Residuals vs fitted
plt.subplot(2, 3, 1)
plt.scatter(y_pred_best, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.grid(True)

# Q-Q plot
plt.subplot(2, 3, 2)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')

# Histogram of residuals
plt.subplot(2, 3, 3)
plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.grid(True)

# Residuals vs index
plt.subplot(2, 3, 4)
plt.plot(residuals, 'o-', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Observation Index')
plt.ylabel('Residuals')
plt.title('Residuals vs Index')
plt.grid(True)

# Scale-location plot
plt.subplot(2, 3, 5)
plt.scatter(y_pred_best, np.sqrt(np.abs(residuals)), alpha=0.6)
plt.xlabel('Fitted Values')
plt.ylabel('√|Residuals|')
plt.title('Scale-Location Plot')
plt.grid(True)

# Leverage plot (simplified)
plt.subplot(2, 3, 6)
leverage = np.diag(X_train_scaled @ np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T)
plt.scatter(leverage, residuals[:len(leverage)], alpha=0.6)
plt.xlabel('Leverage')
plt.ylabel('Residuals')
plt.title('Leverage Plot')
plt.grid(True)

plt.tight_layout()
plt.show()

# Multicollinearity detection
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return vif_data

vif_results = calculate_vif(X_train)
print("\nVariance Inflation Factors:")
print(vif_results)

# Condition number
condition_number = np.linalg.cond(X_train)
print(f"\nCondition Number: {condition_number:.2f}")

# Correlation matrix
correlation_matrix = pd.DataFrame(X_train, columns=feature_names).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Summary of model performance
models_summary = pd.DataFrame({
    'Model': ['OLS', 'Forward Selection', 'Backward Elimination', 'RFE', 
              'Ridge', 'Lasso', 'Elastic Net', 'PCR', 'PLS'],
    'MSE': [full_mse, forward_mse, backward_mse, rfe_mse, 
            ridge_mse, lasso_mse, elastic_mse, pcr_final_mse, pls_final_mse],
    'R²': [full_r2, forward_r2, backward_r2, rfe_r2, 
           ridge_r2, lasso_r2, elastic_r2, r2_score(y_test, pcr_final_pred), 
           r2_score(y_test, pls_final_pred)]
})

print("\nModel Performance Summary:")
print(models_summary.sort_values('MSE'))
```

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

## Exercises

### Exercise 1: Stepwise Selection
- **Objective:** Apply forward, backward, and stepwise selection to a dataset and compare the results.
- **Hint:** Use `SequentialFeatureSelector` and compare AIC/BIC/adj $`R^2`$.

### Exercise 2: Regularization Methods
- **Objective:** Compare ridge, lasso, and elastic net regression on a high-dimensional dataset.
- **Hint:** Use `RidgeCV`, `LassoCV`, `ElasticNetCV` and cross-validation to select $`\lambda`$.

### Exercise 3: Cross-Validation
- **Objective:** Implement k-fold cross-validation to select the optimal number of variables.
- **Hint:** Use `cross_val_score` and `KFold` for custom CV.

### Exercise 4: Model Diagnostics
- **Objective:** Perform comprehensive residual analysis and multicollinearity detection.
- **Hint:** Use diagnostic plots and calculate VIF/condition number with `variance_inflation_factor`.

### Exercise 5: Real-World Application
- **Objective:** Apply variable selection methods to a real dataset and interpret the results.
- **Hint:** Use all methods and compare their selected variables and performance.

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