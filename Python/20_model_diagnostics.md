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

**Code Reference**: See `perform_residual_analysis()` and `create_residual_plots()` functions in `20_model_diagnostics.py`.

The residual analysis provides:
1. **Comprehensive Residual Calculation**: Ordinary, standardized, and studentized residuals
2. **Normality Testing**: Shapiro-Wilk test and Q-Q plots for residual normality
3. **Residual Statistics**: Mean, standard deviation, skewness, and kurtosis
4. **Visual Diagnostics**: Four-panel residual plots for comprehensive assessment
5. **Automatic Assessment**: Automatic determination of normality assumptions
6. **Professional Output**: Publication-ready diagnostic visualizations

This function provides complete residual analysis essential for validating linear regression assumptions.

### Leverage and Influence

**Code Reference**: See `analyze_leverage_and_influence()` and `plot_leverage_and_influence()` functions in `20_model_diagnostics.py`.

The leverage and influence analysis provides:
1. **Leverage Calculation**: Hat matrix diagonals to identify high-leverage points
2. **Cook's Distance**: Measure influence of each observation on all fitted values
3. **DFFITS**: Standardized influence measure for each observation
4. **Threshold Assessment**: Automatic identification of influential points
5. **Visual Diagnostics**: Leverage plots, Cook's distance plots, and influence plots
6. **Comprehensive Summary**: Summary statistics for all influence measures

These functions help identify observations that may unduly influence the regression results.

### Multicollinearity

**Code Reference**: See `check_multicollinearity()` function in `20_model_diagnostics.py`.

The multicollinearity detection provides:
1. **VIF Calculation**: Variance Inflation Factor for each predictor
2. **Tolerance Analysis**: Reciprocal of VIF for additional insight
3. **Correlation Matrix**: Pairwise correlations between predictors
4. **High Correlation Detection**: Identify problematic predictor pairs
5. **Severity Assessment**: Classify multicollinearity severity
6. **Recommendations**: Provide guidance for addressing multicollinearity

This function provides comprehensive multicollinearity diagnostics essential for multiple regression model validation.

### Heteroscedasticity

**Code Reference**: See `test_heteroscedasticity()` function in `20_model_diagnostics.py`.

The heteroscedasticity testing provides:
1. **Breusch-Pagan Test**: Formal statistical test for heteroscedasticity
2. **Variance Ratio Analysis**: Simple visual assessment of variance patterns
3. **Statistical Assessment**: P-values and significance testing
4. **Visual Diagnostics**: Residual plots for heteroscedasticity detection
5. **Comprehensive Evaluation**: Multiple approaches to detect variance issues
6. **Recommendations**: Guidance for addressing heteroscedasticity

This function provides both statistical and visual evidence for heteroscedasticity detection.

### Independence

**Code Reference**: See `test_independence()` function in `20_model_diagnostics.py`.

The independence testing provides:
1. **Durbin-Watson Test**: Formal test for autocorrelation in residuals
2. **Autocorrelation Detection**: Identify positive or negative autocorrelation
3. **Statistical Assessment**: Durbin-Watson statistic interpretation
4. **Clear Interpretation**: Automatic classification of autocorrelation type
5. **Recommendations**: Guidance for addressing independence violations
6. **Comprehensive Analysis**: Complete independence assessment

This function provides essential diagnostics for assessing the independence assumption in regression models.

## Diagnostics for Logistic Regression

### Residuals and Influence

**Code Reference**: See `perform_logistic_regression_diagnostics()` and `create_logistic_diagnostic_plots()` functions in `20_model_diagnostics.py`.

The logistic regression diagnostics provide:
1. **Pearson Residuals**: Standardized residuals for logistic regression
2. **Deviance Residuals**: Deviance-based residuals for model fit assessment
3. **Leverage Analysis**: Hat matrix diagonals for logistic models
4. **Cook's Distance**: Influence measures for logistic regression
5. **Visual Diagnostics**: Comprehensive plots for logistic model assessment
6. **Model Fit Statistics**: Pseudo R-squared and other fit measures

These functions provide essential diagnostics for validating logistic regression models.

### Model Fit and Classification

**Code Reference**: See `hosmer_lemeshow_test()` function and the model fit assessment within `perform_logistic_regression_diagnostics()` in `20_model_diagnostics.py`.

The model fit and classification assessment provides:
1. **Pseudo R-squared**: Multiple pseudo R-squared measures for logistic regression
2. **Hosmer-Lemeshow Test**: Goodness-of-fit test for logistic regression
3. **ROC Curve Analysis**: Receiver Operating Characteristic curve
4. **AUC Calculation**: Area Under the Curve for classification performance
5. **Model Summary**: Comprehensive statistical summary
6. **Classification Performance**: Overall model performance assessment

These functions provide essential validation metrics for logistic regression model performance.

## Generalized Linear Model (GLM) Diagnostics

- Many diagnostics for linear and logistic regression extend to other GLMs (e.g., Poisson, multinomial).
- Use deviance, residuals, influence, and fit statistics as appropriate for the model family.

## Visualization and Interpretation

### Diagnostic Plot Matrix

**Code Reference**: See `create_residual_plots()` and `create_comprehensive_diagnostic_plots()` functions in `20_model_diagnostics.py`.

The diagnostic plot matrix provides:
1. **Four-Panel Layout**: Comprehensive diagnostic visualization
2. **Residuals vs Fitted**: Linearity and homoscedasticity assessment
3. **Normal Q-Q Plot**: Normality assumption validation
4. **Scale-Location Plot**: Heteroscedasticity detection
5. **Residuals vs Leverage**: Influence and leverage assessment
6. **Professional Formatting**: Publication-ready visualizations

These functions create comprehensive diagnostic plots essential for model validation.

### Influence Plot

**Code Reference**: See `plot_leverage_and_influence()` function in `20_model_diagnostics.py`.

The influence plot provides:
1. **Bubble Plot**: Point size represents Cook's distance
2. **Leverage vs Residuals**: X-axis shows leverage, Y-axis shows studentized residuals
3. **Influence Assessment**: Visual identification of influential points
4. **Threshold Lines**: Reference lines for leverage and residual thresholds
5. **Comprehensive View**: Single plot showing all influence measures
6. **Professional Formatting**: Publication-ready influence visualization

This function creates the essential influence plot for identifying problematic observations.

### Residuals vs Leverage Plot

**Code Reference**: See `plot_leverage_and_influence()` function in `20_model_diagnostics.py`.

The residuals vs leverage plot provides:
1. **Scatter Plot**: Studentized residuals vs leverage values
2. **Threshold Lines**: Reference lines at ±2 for studentized residuals
3. **Outlier Detection**: Visual identification of high-leverage and high-residual points
4. **Influence Assessment**: Combined view of leverage and residual patterns
5. **Clear Visualization**: Easy interpretation of problematic observations
6. **Professional Formatting**: Publication-ready diagnostic plot

This function creates essential plots for identifying influential observations in regression models.

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

## Python Implementation Reference

### Core Diagnostic Functions

**Linear Regression Diagnostics:**
- `perform_residual_analysis(X, y, feature_names=None)`: Comprehensive residual analysis
- `analyze_leverage_and_influence(results)`: Leverage and influence diagnostics
- `check_multicollinearity(X)`: VIF and multicollinearity analysis
- `test_heteroscedasticity(results)`: Heteroscedasticity testing
- `test_independence(results)`: Independence and autocorrelation testing

**Logistic Regression Diagnostics:**
- `perform_logistic_regression_diagnostics(X, y, feature_names=None)`: Complete logistic diagnostics
- `hosmer_lemeshow_test(y_true, y_pred, n_groups=10)`: Goodness-of-fit test

**Visualization Functions:**
- `create_residual_plots(results, figsize=(12, 10))`: Four-panel residual plots
- `plot_leverage_and_influence(results, influence_results)`: Leverage and influence plots
- `create_logistic_diagnostic_plots(results)`: Logistic regression plots
- `create_comprehensive_diagnostic_plots(diagnostic_results)`: All diagnostic plots

**Comprehensive Analysis:**
- `perform_comprehensive_diagnostics(X, y, model_type='linear', feature_names=None)`: Complete diagnostics
- `boston_housing_diagnostics_example()`: Real-world linear regression example
- `logistic_regression_example()`: Real-world logistic regression example
- `outlier_influence_example()`: Outlier and influence analysis example

### Data Preparation Functions
- `load_boston_data()`: Load Boston housing dataset with fallback simulation
- `simulate_data_with_outliers(seed=42, n_samples=100, outlier_fraction=0.05)`: Generate test data with outliers

### Usage Example

```python
# Load the module
from model_diagnostics import *

# Perform comprehensive diagnostics
data = load_boston_data()
X = data[['RM', 'LSTAT', 'PTRATIO']]
y = data['MEDV']

# Complete diagnostic analysis
diagnostics = perform_comprehensive_diagnostics(X, y, model_type='linear')

# Create all diagnostic plots
create_comprehensive_diagnostic_plots(diagnostics)

# Access specific results
print(f"Residuals normal: {diagnostics['residual_analysis']['residual_stats']['normal_residuals']}")
print(f"Heteroscedasticity: {diagnostics['heteroscedasticity']['overall_assessment']['heteroscedastic']}")
print(f"High leverage points: {diagnostics['leverage_and_influence']['summary']['total_high_leverage']}")
```

### Function Cross-References

**Theory → Code Mapping:**
- **Residual Analysis** → `perform_residual_analysis()`, `create_residual_plots()`
- **Leverage & Influence** → `analyze_leverage_and_influence()`, `plot_leverage_and_influence()`
- **Multicollinearity** → `check_multicollinearity()`
- **Heteroscedasticity** → `test_heteroscedasticity()`
- **Independence** → `test_independence()`
- **Logistic Diagnostics** → `perform_logistic_regression_diagnostics()`, `create_logistic_diagnostic_plots()`
- **Model Fit** → `hosmer_lemeshow_test()`, ROC analysis in logistic diagnostics

**Complete Workflow:**
1. **Data Preparation** → `load_boston_data()` or `simulate_data_with_outliers()`
2. **Comprehensive Analysis** → `perform_comprehensive_diagnostics()`
3. **Visualization** → `create_comprehensive_diagnostic_plots()`
4. **Interpretation** → Access results from diagnostic dictionaries

---

**Key Takeaways:**
- Diagnostics are essential for trustworthy regression analysis
- Always check residuals, leverage, influence, and assumptions
- Use multiple diagnostics and visualizations
- Address and report any violations or influential points
- Good diagnostics lead to better models and more reliable conclusions
- The Python implementation provides comprehensive, professional diagnostic tools 