# Multiple Linear Regression

## Overview

Multiple linear regression extends simple linear regression to model the relationship between a dependent variable and two or more independent variables. It is a powerful tool for prediction, explanation, and understanding the joint effects of predictors.

### Key Concepts
- **Multiple Regression**: Predicts a response using several predictors.
- **Coefficients**: Quantify the effect of each predictor, holding others constant.
- **Assumptions**: Linearity, independence, homoscedasticity, normality, and no multicollinearity.
- **Model Selection**: Choosing the best set of predictors.
- **Regularization**: Techniques to prevent overfitting.

## Mathematical Foundations

### Model Equation

The multiple linear regression model is:

```math
Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \cdots + \beta_p X_{ip} + \epsilon_i
```

Where:
- $`Y_i`$: Response variable for observation $`i`$
- $`X_{ij}`$: Value of predictor $`j`$ for observation $`i`$
- $`\beta_0`$: Intercept
- $`\beta_j`$: Slope for predictor $`j`$ (effect of $`X_j`$ holding others constant)
- $`\epsilon_i`$: Error term

### Matrix Notation

The model can be written as:

```math
\mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}
```

Where:
- $`\mathbf{Y}`$: $n \times 1$ vector of responses
- $`\mathbf{X}`$: $n \times (p+1)$ matrix of predictors (including intercept)
- $`\boldsymbol{\beta}`$: $(p+1) \times 1$ vector of coefficients
- $`\boldsymbol{\epsilon}`$: $n \times 1$ vector of errors

### Least Squares Estimation

The estimated coefficients minimize the sum of squared residuals:

```math
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}
```

### Interpretation
- $`\hat{\beta}_j`$: Expected change in $`Y`$ for a one-unit increase in $`X_j`$, holding all other predictors constant.
- $`\hat{\beta}_0`$: Expected value of $`Y`$ when all $`X_j = 0`$ (may not always be meaningful).

### Goodness of Fit: $R^2$ and Adjusted $R^2$

$R^2$ measures the proportion of variance in $Y$ explained by all predictors:

```math
R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}
```

Adjusted $R^2$ penalizes for the number of predictors:

```math
R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
```
Where $`n`$ is the number of observations and $`p`$ is the number of predictors.

## Fitting Multiple Linear Regression in Python

### Example: Predicting MPG from Multiple Predictors

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load data
mtcars = sm.datasets.get_rdataset('mtcars').data

# Fit multiple linear regression
multiple_model = smf.ols('mpg ~ wt + hp + disp', data=mtcars).fit()
print(multiple_model.summary())

# Extract coefficients
print("Coefficients:", multiple_model.params)

# Fitted values and residuals
fitted_vals = multiple_model.fittedvalues
residuals = multiple_model.resid
```

### Interpreting Output
- **Intercept**: Estimated MPG when all predictors are zero (may not be meaningful).
- **Coefficients**: Change in MPG for a one-unit increase in each predictor, holding others constant.
- **$R^2$ and Adjusted $R^2$**: Proportion of variance explained.
- **p-values**: Test if each coefficient is significantly different from zero.

## Confidence Intervals and Standardized Coefficients

```python
# Confidence intervals for coefficients
print("Confidence intervals:")
print(multiple_model.conf_int(alpha=0.05))

# Standardized coefficients
def standardize_coefficients(model, data):
    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[['wt', 'hp', 'disp']])
    y_scaled = (data['mpg'] - data['mpg'].mean()) / data['mpg'].std()
    
    # Fit model with standardized data
    X_scaled_df = pd.DataFrame(X_scaled, columns=['wt', 'hp', 'disp'])
    X_scaled_df['const'] = 1
    model_scaled = sm.OLS(y_scaled, X_scaled_df).fit()
    
    return model_scaled.params

std_coeffs = standardize_coefficients(multiple_model, mtcars)
print("Standardized coefficients:")
print(std_coeffs)
```

## Model Building and Selection

### Stepwise Regression

- **Forward selection**: Start with no predictors, add one at a time.
- **Backward elimination**: Start with all predictors, remove one at a time.
- **Stepwise**: Combination of both.

```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

# Forward selection
X = mtcars[['wt', 'hp', 'disp', 'drat', 'qsec']]
y = mtcars['mpg']

forward_selector = SequentialFeatureSelector(
    LinearRegression(), n_features_to_select=3, direction='forward'
)
forward_selector.fit(X, y)
print("Forward selection features:", X.columns[forward_selector.get_support()].tolist())

# Backward elimination
backward_selector = SequentialFeatureSelector(
    LinearRegression(), n_features_to_select=2, direction='backward'
)
backward_selector.fit(X, y)
print("Backward elimination features:", X.columns[backward_selector.get_support()].tolist())
```

### Best Subset Selection

```python
from itertools import combinations
from sklearn.metrics import r2_score

def best_subset_selection(X, y, max_features):
    best_score = -np.inf
    best_features = None
    
    for k in range(1, max_features + 1):
        for features in combinations(X.columns, k):
            X_subset = X[list(features)]
            model = LinearRegression().fit(X_subset, y)
            score = r2_score(y, model.predict(X_subset))
            
            if score > best_score:
                best_score = score
                best_features = features
    
    return best_features, best_score

best_features, best_score = best_subset_selection(X, y, 5)
print(f"Best subset: {best_features}")
print(f"Best R²: {best_score:.3f}")
```

## Model Diagnostics

### Residual Analysis

- **Linearity**: Residuals vs fitted plot should show no pattern.
- **Normality**: Q-Q plot of residuals should be approximately linear.
- **Homoscedasticity**: Residuals should have constant variance.
- **Independence**: Residuals should not be autocorrelated.

```python
# Diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
axes[0, 0].scatter(fitted_vals, residuals)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')

# Q-Q plot
sm.qqplot(residuals, line='s', ax=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

# Scale-Location plot
axes[1, 0].scatter(fitted_vals, np.sqrt(np.abs(residuals)))
axes[1, 0].set_title('Scale-Location')
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('Sqrt(|Residuals|)')

# Residuals vs Leverage
influence = multiple_model.get_influence()
leverage = influence.hat_matrix_diag
axes[1, 1].scatter(leverage, residuals)
axes[1, 1].set_title('Residuals vs Leverage')
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Residuals')

plt.tight_layout()
plt.show()
```

### Multicollinearity Detection

- **Variance Inflation Factor (VIF)**: $`\text{VIF}_j = \frac{1}{1 - R_j^2}`$ (where $`R_j^2`$ is the $R^2$ from regressing $X_j$ on all other predictors).
- **Tolerance**: $`1 / \text{VIF}`$
- **Condition number**: Large values indicate multicollinearity.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

X_vif = mtcars[['wt', 'hp', 'disp']]
vif_results = calculate_vif(X_vif)
print("Variance Inflation Factors:")
print(vif_results)

# Tolerance
vif_results['Tolerance'] = 1 / vif_results['VIF']
print("\nTolerance:")
print(vif_results[['Variable', 'Tolerance']])
```

### Outlier and Influence Diagnostics

- **Cook's distance**: Identifies influential points.
- **Leverage**: Points with unusual predictor values.
- **Studentized residuals**: Detects outliers in $Y$.

```python
# Cook's distance
cooksd = influence.cooks_distance[0]
plt.figure(figsize=(8, 4))
plt.stem(np.arange(len(cooksd)), cooksd, use_line_collection=True)
plt.axhline(4/len(cooksd), color='red', linestyle='--')
plt.title("Cook's Distance")
plt.xlabel('Observation')
plt.ylabel("Cook's D")
plt.tight_layout()
plt.show()

# Leverage
leverage = influence.hat_matrix_diag
plt.figure(figsize=(8, 4))
plt.stem(np.arange(len(leverage)), leverage, use_line_collection=True)
leverage_threshold = 2 * (len(multiple_model.params) + 1) / len(mtcars)
plt.axhline(leverage_threshold, color='red', linestyle='--')
plt.title("Leverage")
plt.xlabel('Observation')
plt.ylabel('Leverage')
plt.tight_layout()
plt.show()
```

## Model Comparison and Validation

### Information Criteria
- **AIC**: Penalizes model complexity, lower is better.
- **BIC**: Stronger penalty for complexity, lower is better.

```python
# AIC and BIC
print(f"AIC: {multiple_model.aic:.2f}")
print(f"BIC: {multiple_model.bic:.2f}")
```

### Cross-Validation

- **LOOCV**: Leave-one-out cross-validation.
- **k-fold CV**: Split data into $k$ parts, train on $k-1$, test on 1.

```python
from sklearn.model_selection import cross_val_score, LeaveOneOut

# LOOCV
loo = LeaveOneOut()
cv_scores_loo = cross_val_score(LinearRegression(), X, y, cv=loo, scoring='neg_mean_squared_error')
mse_loo = -cv_scores_loo.mean()
print(f"LOOCV MSE: {mse_loo:.2f}")

# 5-fold CV
cv_scores_5fold = cross_val_score(LinearRegression(), X, y, cv=5, scoring='neg_mean_squared_error')
mse_5fold = -cv_scores_5fold.mean()
print(f"5-fold CV MSE: {mse_5fold:.2f}")
```

## Interaction and Polynomial Terms

### Including Interactions

- **Interaction**: Effect of one predictor depends on another.

```python
# Interaction model
interaction_model = smf.ols('mpg ~ wt * hp + disp', data=mtcars).fit()
print(interaction_model.summary())
```

### Polynomial Terms

- **Polynomial**: Captures nonlinear relationships.

```python
# Polynomial model
mtcars['wt_squared'] = mtcars['wt'] ** 2
poly_model = smf.ols('mpg ~ wt + wt_squared + hp + disp', data=mtcars).fit()
print(poly_model.summary())
```

## Regularization Methods

### Ridge Regression

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

# Ridge regression
ridge = Ridge()
param_grid = {'alpha': np.logspace(-3, 3, 100)}
ridge_cv = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X, y)

best_ridge = Ridge(alpha=ridge_cv.best_params_['alpha'])
best_ridge.fit(X, y)
print("Ridge coefficients:", best_ridge.coef_)
print("Best alpha:", ridge_cv.best_params_['alpha'])
```

### Lasso Regression

```python
# Lasso regression
lasso = Lasso()
lasso_cv = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_cv.fit(X, y)

best_lasso = Lasso(alpha=lasso_cv.best_params_['alpha'])
best_lasso.fit(X, y)
print("Lasso coefficients:", best_lasso.coef_)
print("Best alpha:", lasso_cv.best_params_['alpha'])
```

### Elastic Net

```python
# Elastic Net
elastic = ElasticNet()
param_grid_elastic = {
    'alpha': np.logspace(-3, 3, 50),
    'l1_ratio': np.linspace(0, 1, 20)
}
elastic_cv = GridSearchCV(elastic, param_grid_elastic, cv=5, scoring='neg_mean_squared_error')
elastic_cv.fit(X, y)

best_elastic = ElasticNet(
    alpha=elastic_cv.best_params_['alpha'],
    l1_ratio=elastic_cv.best_params_['l1_ratio']
)
best_elastic.fit(X, y)
print("Elastic Net coefficients:", best_elastic.coef_)
print("Best parameters:", elastic_cv.best_params_)
```

## Practical Examples

### Example 1: Real Estate Analysis

```python
# Simulate real estate data
np.random.seed(123)
n_properties = 100
square_feet = np.random.normal(2000, 500, n_properties)
bedrooms = np.random.randint(1, 6, n_properties)
bathrooms = np.random.randint(1, 5, n_properties)
age = np.random.normal(15, 8, n_properties)
location_score = np.random.normal(7, 1, n_properties)
price = (200000 + 100 * square_feet + 15000 * bedrooms + 
         25000 * bathrooms - 2000 * age + 15000 * location_score + 
         np.random.normal(0, 15000, n_properties))

real_estate_data = pd.DataFrame({
    'price': price,
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'location_score': location_score
})

real_estate_model = smf.ols(
    'price ~ square_feet + bedrooms + bathrooms + age + location_score', 
    data=real_estate_data
).fit()
print(real_estate_model.summary())
```

### Example 2: Marketing Analysis

```python
# Simulate marketing data
np.random.seed(123)
n_campaigns = 50
ad_spend = np.random.normal(10000, 3000, n_campaigns)
social_media_posts = np.random.poisson(20, n_campaigns)
email_sends = np.random.poisson(1000, n_campaigns)
season = np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_campaigns)
sales = (50000 + 2.5 * ad_spend + 500 * social_media_posts + 
         10 * email_sends + np.random.normal(0, 5000, n_campaigns))

marketing_data = pd.DataFrame({
    'sales': sales,
    'ad_spend': ad_spend,
    'social_media_posts': social_media_posts,
    'email_sends': email_sends,
    'season': season
})

marketing_model = smf.ols(
    'sales ~ ad_spend + social_media_posts + email_sends + season', 
    data=marketing_data
).fit()
print(marketing_model.summary())
```

## Best Practices
- Always check assumptions and perform diagnostics before interpreting results.
- Use $R^2$, adjusted $R^2$, AIC, BIC, and cross-validation for model selection.
- Check for multicollinearity using VIF and correlation matrices.
- Use regularization for high-dimensional or collinear data.
- Report coefficients, intervals, diagnostics, and model fit.
- Interpret coefficients in the context of the model and data.

## Reporting Guidelines

- Report the model equation, coefficients, $R^2$, adjusted $R^2$, confidence intervals, p-values, and diagnostics.
- Example: "A multiple regression found that weight, horsepower, and displacement significantly predicted MPG, $`R^2 = 0.83, F(3, 28) = 45.2, p < .001`$. Weight had the largest negative effect ($`b = -3.5, 95\%\ CI [-4.7, -2.3], p < .001`$)."

## Exercises

### Exercise 1: Model Building and Selection
Build a multiple regression model to predict MPG using all available variables in the mtcars dataset. Use stepwise selection and compare models using $R^2$, AIC, and BIC.

### Exercise 2: Diagnostics and Multicollinearity
Perform comprehensive diagnostics on your model, including residual analysis, VIF, and outlier detection. Interpret the results.

### Exercise 3: Interaction and Polynomial Terms
Add interaction and polynomial terms to your model. Interpret their effects and compare model fit.

### Exercise 4: Regularization
Apply ridge and lasso regression to your data. Compare the coefficients and model fit to the standard regression.

### Exercise 5: Real-World Application
Find a real dataset (e.g., from R's datasets package). Build, validate, and report a multiple regression model.

---

**Key Takeaways:**
- Multiple regression models the effect of several predictors on a response
- Always check assumptions and diagnostics
- Use model selection and regularization to avoid overfitting
- Interpret coefficients in context
- Report all relevant statistics and diagnostics 