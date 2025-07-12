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

**Code Reference**: See `mpg_multiple_predictors_example()` function in `19_multiple_linear_regression.py`.

The MPG multiple predictors example demonstrates:
1. **Data Loading**: Load mtcars dataset with fallback simulation
2. **Model Fitting**: Complete multiple linear regression analysis
3. **Results Interpretation**: Coefficient interpretation and model fit assessment
4. **Multicollinearity Checking**: VIF analysis and correlation assessment
5. **Diagnostics**: Comprehensive model diagnostics and influence analysis
6. **Cross-Validation**: Model validation using k-fold cross-validation

This example shows how to perform a complete multiple linear regression analysis on real data, including all necessary diagnostics and validations.

### Interpreting Output

**Code Reference**: See `interpret_multiple_regression_results()` function in `19_multiple_linear_regression.py`.

The interpretation function provides:
1. **Coefficient Interpretation**: Clear explanations of intercept and slope meanings
2. **Confidence Intervals**: Lower and upper bounds for coefficient estimates
3. **Model Fit Assessment**: R² and adjusted R² interpretation
4. **Statistical Significance**: F-statistic and p-value interpretation
5. **Practical Meaning**: Real-world interpretation of regression coefficients
6. **Predictor Effects**: Individual effect of each predictor holding others constant

This function automatically generates interpretable results for any multiple linear regression model.

## Confidence Intervals and Standardized Coefficients

**Code Reference**: See `calculate_standardized_coefficients()` function in `19_multiple_linear_regression.py`.

The standardized coefficients analysis provides:
1. **Coefficient Standardization**: Convert coefficients to standardized scale
2. **Predictor Comparison**: Compare relative importance of predictors
3. **Scale-Free Interpretation**: Coefficients independent of variable scales
4. **Effect Size Assessment**: Quantify relative predictor importance
5. **Automatic Calculation**: Handle any number of predictors automatically

This function provides standardized coefficients that allow direct comparison of predictor effects regardless of their original scales.

## Model Building and Selection

### Stepwise Regression

**Code Reference**: See `perform_stepwise_selection()` function in `19_multiple_linear_regression.py`.

The stepwise regression analysis provides:
1. **Forward Selection**: Start with no predictors, add one at a time
2. **Backward Elimination**: Start with all predictors, remove one at a time
3. **Automatic Selection**: Determine optimal number of features
4. **Model Comparison**: Compare different selection strategies
5. **Feature Ranking**: Identify most important predictors
6. **Model Validation**: Validate selected models

This function implements both forward and backward stepwise selection methods for optimal predictor subset identification.

### Best Subset Selection

**Code Reference**: See `perform_best_subset_selection()` function in `19_multiple_linear_regression.py`.

The best subset selection analysis provides:
1. **Exhaustive Search**: Evaluate all possible predictor combinations
2. **Optimal Subset**: Find the best predictor combination for given criteria
3. **Model Comparison**: Compare all possible models systematically
4. **Feature Ranking**: Identify optimal feature subsets of different sizes
5. **Performance Metrics**: R² and other fit measures for each subset
6. **Computational Efficiency**: Handle large numbers of predictors efficiently

This function performs exhaustive search to find the optimal subset of predictors, providing comprehensive model comparison.

## Model Diagnostics

### Residual Analysis

- **Linearity**: Residuals vs fitted plot should show no pattern.
- **Normality**: Q-Q plot of residuals should be approximately linear.
- **Homoscedasticity**: Residuals should have constant variance.
- **Independence**: Residuals should not be autocorrelated.

**Code Reference**: See `create_multiple_regression_diagnostics()` function in `19_multiple_linear_regression.py`.

The diagnostic plots provide:
1. **Residuals vs Fitted**: Check for linearity and homoscedasticity
2. **Normal Q-Q Plot**: Assess normality of residuals
3. **Scale-Location Plot**: Check for heteroscedasticity
4. **Residuals vs Leverage**: Identify influential points
5. **Comprehensive Assessment**: Visual evaluation of all assumptions
6. **Professional Output**: Publication-ready diagnostic visualizations

These plots provide essential visual diagnostics for multiple regression assumption validation.

### Multicollinearity Detection

- **Variance Inflation Factor (VIF)**: $`\text{VIF}_j = \frac{1}{1 - R_j^2}`$ (where $`R_j^2`$ is the $R^2$ from regressing $X_j$ on all other predictors).
- **Tolerance**: $`1 / \text{VIF}`$
- **Condition number**: Large values indicate multicollinearity.

**Code Reference**: See `check_multicollinearity()` function in `19_multiple_linear_regression.py`.

The multicollinearity detection provides:
1. **VIF Calculation**: Variance Inflation Factor for each predictor
2. **Tolerance Analysis**: Reciprocal of VIF for additional insight
3. **Correlation Matrix**: Pairwise correlations between predictors
4. **High Correlation Detection**: Identify problematic predictor pairs
5. **Severity Assessment**: Classify multicollinearity severity
6. **Recommendations**: Provide guidance for addressing multicollinearity

This function provides comprehensive multicollinearity diagnostics essential for multiple regression model validation.

### Outlier and Influence Diagnostics

- **Cook's distance**: Identifies influential points.
- **Leverage**: Points with unusual predictor values.
- **Studentized residuals**: Detects outliers in $Y$.

**Code Reference**: See `analyze_multiple_regression_influence()` function in `19_multiple_linear_regression.py`.

The influence analysis provides:
1. **Cook's Distance**: Measure influence of each observation on the model
2. **Leverage Analysis**: Identify high-leverage points using hat matrix diagonals
3. **DFFITS**: Standardized influence measure for each observation
4. **Studentized Residuals**: Identify outliers in the response variable
5. **Threshold Assessment**: Automatic identification of influential points
6. **Comprehensive Summary**: Summary statistics for all influence measures

These functions help identify observations that may unduly influence the multiple regression results.

## Model Comparison and Validation

### Information Criteria
- **AIC**: Penalizes model complexity, lower is better.
- **BIC**: Stronger penalty for complexity, lower is better.

**Code Reference**: See `fit_multiple_linear_regression()` function in `19_multiple_linear_regression.py`.

The information criteria analysis provides:
1. **AIC Calculation**: Akaike Information Criterion for model comparison
2. **BIC Calculation**: Bayesian Information Criterion for model comparison
3. **Model Selection**: Lower values indicate better models
4. **Complexity Penalty**: Balance between fit and model complexity
5. **Automatic Computation**: Included in comprehensive model results

These criteria help select the optimal model by balancing goodness of fit with model complexity.

### Cross-Validation

- **LOOCV**: Leave-one-out cross-validation.
- **k-fold CV**: Split data into $k$ parts, train on $k-1$, test on 1.

**Code Reference**: See `perform_cross_validation()` function in `19_multiple_linear_regression.py`.

The cross-validation analysis provides:
1. **Leave-One-Out CV**: Comprehensive validation using all data points
2. **K-Fold CV**: Efficient validation with configurable fold number
3. **Multiple Metrics**: MSE, RMSE, and R² scores
4. **Performance Assessment**: Mean and standard deviation of scores
5. **Model Validation**: Unbiased estimate of model performance
6. **Flexible Configuration**: Support for different CV strategies

This function provides robust model validation essential for assessing predictive performance.

## Interaction and Polynomial Terms

### Including Interactions

- **Interaction**: Effect of one predictor depends on another.

**Code Reference**: See `add_interaction_terms()` function in `19_multiple_linear_regression.py`.

The interaction terms analysis provides:
1. **Interaction Creation**: Automatically generate interaction terms
2. **Flexible Naming**: Customizable interaction term names
3. **Multiple Interactions**: Handle multiple variable interactions
4. **Data Preparation**: Prepare data for interaction analysis
5. **Model Integration**: Seamless integration with regression models

This function facilitates the inclusion of interaction effects in multiple regression models.

### Polynomial Terms

- **Polynomial**: Captures nonlinear relationships.

**Code Reference**: See `add_polynomial_terms()` function in `19_multiple_linear_regression.py`.

The polynomial terms analysis provides:
1. **Polynomial Creation**: Generate polynomial terms up to specified degree
2. **Flexible Degrees**: Support for quadratic, cubic, and higher-order terms
3. **Custom Naming**: Configurable naming conventions for polynomial terms
4. **Data Preparation**: Automatic data preparation for polynomial regression
5. **Model Integration**: Seamless integration with regression models

This function facilitates the inclusion of nonlinear relationships in multiple regression models.

## Regularization Methods

### Ridge Regression

**Code Reference**: See `apply_ridge_regression()` function in `19_multiple_linear_regression.py`.

The ridge regression analysis provides:
1. **L2 Regularization**: Ridge regression with L2 penalty
2. **Cross-Validation**: Automatic hyperparameter tuning
3. **Optimal Alpha**: Find best regularization strength
4. **Coefficient Shrinkage**: Shrink coefficients toward zero
5. **Multicollinearity Handling**: Address multicollinearity issues
6. **Performance Assessment**: R² and other fit measures

This function implements ridge regression with automatic hyperparameter optimization.

### Lasso Regression

**Code Reference**: See `apply_lasso_regression()` function in `19_multiple_linear_regression.py`.

The lasso regression analysis provides:
1. **L1 Regularization**: Lasso regression with L1 penalty
2. **Feature Selection**: Automatic feature selection via coefficient sparsity
3. **Cross-Validation**: Automatic hyperparameter tuning
4. **Optimal Alpha**: Find best regularization strength
5. **Sparse Solutions**: Produce sparse coefficient vectors
6. **Variable Selection**: Identify important predictors

This function implements lasso regression with automatic feature selection and hyperparameter optimization.

### Elastic Net

**Code Reference**: See `apply_elastic_net()` function in `19_multiple_linear_regression.py`.

The elastic net analysis provides:
1. **Combined Regularization**: L1 and L2 penalties combined
2. **Flexible Penalty**: Adjustable balance between L1 and L2
3. **Cross-Validation**: Automatic hyperparameter tuning
4. **Optimal Parameters**: Find best alpha and l1_ratio
5. **Feature Selection**: Combines benefits of ridge and lasso
6. **Performance Assessment**: Comprehensive model evaluation

This function implements elastic net regression with automatic hyperparameter optimization for both regularization parameters.

## Practical Examples

### Example 1: Real Estate Analysis

**Code Reference**: See `real_estate_example()` function in `19_multiple_linear_regression.py`.

The real estate example demonstrates:
1. **Realistic Data Simulation**: Generate property data with known relationships
2. **Multiple Predictors**: Square footage, bedrooms, bathrooms, age, location
3. **Model Selection**: Forward stepwise and best subset selection
4. **Regularization Methods**: Ridge, Lasso, and Elastic Net comparison
5. **Comprehensive Analysis**: Full multiple regression workflow
6. **Business Application**: Real-world real estate valuation scenario

This example shows how to apply multiple regression to a realistic business scenario with model selection and regularization.

### Example 2: Marketing Analysis

**Code Reference**: See `marketing_example()` function in `19_multiple_linear_regression.py`.

The marketing example demonstrates:
1. **Marketing Data Simulation**: Generate campaign data with known effects
2. **Categorical Variables**: Handle seasonal effects with dummy variables
3. **Interaction Terms**: Add interaction effects between variables
4. **Multiple Predictors**: Ad spend, social media, email, season
5. **Business Metrics**: Sales prediction and ROI analysis
6. **Marketing Application**: Real-world marketing campaign analysis

This example shows how to apply multiple regression to marketing analytics, including categorical variables and interaction effects.

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

## Using the Python Code

### Getting Started

To use the multiple linear regression functions, import the module and run the main demonstration:

```python
# Run the complete demonstration
python 19_multiple_linear_regression.py
```

### Key Functions Overview

**Data Preparation:**
- `load_mtcars_data()`: Load mtcars dataset with fallback simulation
- `simulate_multiple_regression_data()`: Generate data for multiple regression

**Model Fitting and Analysis:**
- `fit_multiple_linear_regression()`: Complete multiple regression analysis
- `interpret_multiple_regression_results()`: Automatic result interpretation
- `calculate_standardized_coefficients()`: Standardized coefficient analysis

**Model Selection:**
- `perform_stepwise_selection()`: Forward and backward stepwise selection
- `perform_best_subset_selection()`: Exhaustive subset selection

**Diagnostics and Validation:**
- `check_multicollinearity()`: VIF and multicollinearity diagnostics
- `create_multiple_regression_diagnostics()`: Visual assumption checking
- `analyze_multiple_regression_influence()`: Outlier and influence diagnostics
- `perform_cross_validation()`: Cross-validation for model validation

**Advanced Features:**
- `add_interaction_terms()`: Create interaction effects
- `add_polynomial_terms()`: Add polynomial terms
- `apply_ridge_regression()`: Ridge regression with CV
- `apply_lasso_regression()`: Lasso regression with CV
- `apply_elastic_net()`: Elastic net regression with CV

**Practical Examples:**
- `mpg_multiple_predictors_example()`: MPG prediction with multiple variables
- `real_estate_example()`: Real estate price prediction
- `marketing_example()`: Marketing campaign analysis

### Workflow Example

```python
# 1. Generate or load data
X, y = simulate_multiple_regression_data(n_predictors=3)
X_df = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
y_series = pd.Series(y, name='y')

# 2. Fit multiple regression model
results = fit_multiple_linear_regression(X_df, y_series)

# 3. Interpret results
interpretation = interpret_multiple_regression_results(results)
print(f"R²: {interpretation['model_fit']['r_squared']:.3f}")

# 4. Check multicollinearity
multicollinearity = check_multicollinearity(X_df)
print(f"Severe multicollinearity: {multicollinearity['assessment']['severe_multicollinearity']}")

# 5. Perform model selection
stepwise_results = perform_stepwise_selection(X_df, y_series, direction='forward')
print(f"Selected features: {stepwise_results['selected_features']}")

# 6. Cross-validation
cv_results = perform_cross_validation(X_df, y_series, cv_method='5fold')
print(f"CV R²: {cv_results['r2_mean']:.3f}")

# 7. Apply regularization
ridge_results = apply_ridge_regression(X_df, y_series)
lasso_results = apply_lasso_regression(X_df, y_series)
print(f"Ridge R²: {ridge_results['r_squared']:.3f}")
print(f"Lasso selected {lasso_results['n_non_zero']} variables")
```

### Function Reference

Each function includes comprehensive docstrings with:
- **Parameters**: Input variables and their types
- **Returns**: Output format and content
- **Examples**: Usage examples in the main execution block
- **Theory**: Connection to statistical concepts

### Integration with Theory

The Python functions implement the mathematical concepts discussed in this lesson:
- **Matrix Notation**: Mathematical foundations in `fit_multiple_linear_regression()`
- **Model Selection**: Stepwise and subset selection methods
- **Multicollinearity**: VIF and correlation analysis
- **Regularization**: Ridge, Lasso, and Elastic Net implementations
- **Cross-Validation**: Model validation techniques

---

**Key Takeaways:**
- Multiple regression models the effect of several predictors on a response
- Always check assumptions and diagnostics
- Use model selection and regularization to avoid overfitting
- Interpret coefficients in context
- Report all relevant statistics and diagnostics
- Use the Python functions for comprehensive, reproducible analysis 