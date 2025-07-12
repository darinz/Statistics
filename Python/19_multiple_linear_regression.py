"""
Multiple Linear Regression in Python

This module provides comprehensive functions for performing multiple linear regression,
including model fitting, diagnostics, assumption checking, model selection, and regularization.

Key Functions:
- fit_multiple_linear_regression: Fit and analyze multiple linear regression models
- calculate_standardized_coefficients: Standardized coefficient analysis
- perform_model_selection: Stepwise and best subset selection
- check_multicollinearity: VIF and multicollinearity diagnostics
- create_diagnostic_plots: Regression diagnostic visualizations
- apply_regularization: Ridge, Lasso, and Elastic Net regression
- practical_examples: Real-world regression examples

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score, LeaveOneOut, GridSearchCV
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations

# --- Data Loading and Preparation ---

def load_mtcars_data():
    """
    Load the mtcars dataset for multiple regression analysis.
    
    Returns:
    --------
    DataFrame : mtcars dataset
    """
    try:
        mtcars = sm.datasets.get_rdataset('mtcars').data
        return mtcars
    except:
        # Fallback: create simulated mtcars-like data
        np.random.seed(42)
        n = 32
        wt = np.random.normal(3.2, 0.9, n)
        hp = np.random.normal(146.7, 68.6, n)
        disp = np.random.normal(230.7, 123.9, n)
        drat = np.random.normal(3.6, 0.5, n)
        qsec = np.random.normal(17.8, 1.8, n)
        
        # Generate correlated MPG
        mpg = 37.3 - 5.34 * wt - 0.03 * hp - 0.01 * disp + np.random.normal(0, 2.5, n)
        
        mtcars = pd.DataFrame({
            'mpg': mpg,
            'wt': wt,
            'hp': hp,
            'disp': disp,
            'drat': drat,
            'qsec': qsec
        })
        return mtcars

def simulate_multiple_regression_data(seed=42, n_samples=100, n_predictors=3, noise_std=1.0):
    """
    Simulate data for multiple linear regression demonstration.
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    n_samples : int
        Number of samples to generate
    n_predictors : int
        Number of predictor variables
    noise_std : float
        Standard deviation of noise
        
    Returns:
    --------
    tuple : (X, y) arrays for regression analysis
    """
    np.random.seed(seed)
    
    # Generate predictors
    X = np.random.normal(0, 2, (n_samples, n_predictors))
    
    # Generate response with known coefficients
    coefficients = np.array([10.0, 2.0, -1.5, 0.8])  # intercept + slopes
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    y = X_with_intercept @ coefficients + np.random.normal(0, noise_std, n_samples)
    
    return X, y

# --- Multiple Linear Regression Analysis ---

def fit_multiple_linear_regression(X, y, feature_names=None):
    """
    Fit multiple linear regression model and provide comprehensive analysis.
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Predictor variables
    y : array-like
        Response variable
    feature_names : list, optional
        Names for the predictor variables
        
    Returns:
    --------
    dict : Dictionary containing model results and diagnostics
    """
    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = [f'X{i+1}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name='y')
    
    # Create formula for statsmodels
    formula = f"{y.name} ~ " + " + ".join(X.columns)
    
    # Combine data
    data = pd.concat([y, X], axis=1)
    
    # Fit model
    model = smf.ols(formula, data=data).fit()
    
    # Extract key results
    coefficients = model.params
    conf_int = model.conf_int()
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    f_statistic = model.fvalue
    f_pvalue = model.f_pvalue
    aic = model.aic
    bic = model.bic
    
    # Calculate fitted values and residuals
    fitted_vals = model.fittedvalues
    residuals = model.resid
    
    return {
        'model': model,
        'coefficients': coefficients,
        'confidence_intervals': conf_int,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'f_statistic': f_statistic,
        'f_pvalue': f_pvalue,
        'aic': aic,
        'bic': bic,
        'fitted_values': fitted_vals,
        'residuals': residuals,
        'data': data,
        'formula': formula,
        'X': X,
        'y': y
    }

def interpret_multiple_regression_results(results):
    """
    Provide interpretation of multiple regression results.
    
    Parameters:
    -----------
    results : dict
        Results from fit_multiple_linear_regression
        
    Returns:
    --------
    dict : Dictionary with interpretations
    """
    coef = results['coefficients']
    conf_int = results['confidence_intervals']
    
    interpretation = {
        'intercept': {
            'value': coef.iloc[0],
            'interpretation': f"When all predictors are zero, the predicted value of {results['y'].name} is {coef.iloc[0]:.3f}",
            'ci_lower': conf_int.iloc[0, 0],
            'ci_upper': conf_int.iloc[0, 1]
        },
        'predictors': {},
        'model_fit': {
            'r_squared': results['r_squared'],
            'adj_r_squared': results['adj_r_squared'],
            'interpretation': f"The model explains {results['r_squared']*100:.1f}% of the variance in {results['y'].name}",
            'f_statistic': results['f_statistic'],
            'f_pvalue': results['f_pvalue'],
            'significant': results['f_pvalue'] < 0.05,
            'aic': results['aic'],
            'bic': results['bic']
        }
    }
    
    # Interpret each predictor
    for i, predictor in enumerate(results['X'].columns):
        interpretation['predictors'][predictor] = {
            'value': coef.iloc[i+1],
            'interpretation': f"For each one-unit increase in {predictor}, {results['y'].name} changes by {coef.iloc[i+1]:.3f} units on average, holding all other predictors constant",
            'ci_lower': conf_int.iloc[i+1, 0],
            'ci_upper': conf_int.iloc[i+1, 1]
        }
    
    return interpretation

# --- Standardized Coefficients ---

def calculate_standardized_coefficients(results):
    """
    Calculate standardized coefficients for multiple regression.
    
    Parameters:
    -----------
    results : dict
        Results from fit_multiple_linear_regression
        
    Returns:
    --------
    dict : Dictionary containing standardized coefficients
    """
    X = results['X']
    y = results['y']
    
    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = (y - y.mean()) / y.std()
    
    # Fit model with standardized data
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled_df['const'] = 1
    
    model_scaled = sm.OLS(y_scaled, X_scaled_df).fit()
    
    return {
        'standardized_coefficients': model_scaled.params,
        'X_scaled': X_scaled_df,
        'y_scaled': y_scaled,
        'model_scaled': model_scaled
    }

# --- Model Selection ---

def perform_stepwise_selection(X, y, direction='forward', n_features_to_select=None):
    """
    Perform stepwise feature selection.
    
    Parameters:
    -----------
    X : DataFrame
        Predictor variables
    y : Series
        Response variable
    direction : str
        'forward' or 'backward'
    n_features_to_select : int, optional
        Number of features to select
        
    Returns:
    --------
    dict : Dictionary containing selection results
    """
    if n_features_to_select is None:
        n_features_to_select = X.shape[1] // 2
    
    # Forward selection
    if direction == 'forward':
        selector = SequentialFeatureSelector(
            LinearRegression(), n_features_to_select=n_features_to_select, direction='forward'
        )
    else:  # backward
        selector = SequentialFeatureSelector(
            LinearRegression(), n_features_to_select=n_features_to_select, direction='backward'
        )
    
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Fit model with selected features
    X_selected = X[selected_features]
    selected_results = fit_multiple_linear_regression(X_selected, y)
    
    return {
        'selected_features': selected_features,
        'selection_direction': direction,
        'n_selected': len(selected_features),
        'model_results': selected_results
    }

def perform_best_subset_selection(X, y, max_features=None):
    """
    Perform best subset selection.
    
    Parameters:
    -----------
    X : DataFrame
        Predictor variables
    y : Series
        Response variable
    max_features : int, optional
        Maximum number of features to consider
        
    Returns:
    --------
    dict : Dictionary containing selection results
    """
    if max_features is None:
        max_features = min(X.shape[1], 5)
    
    best_score = -np.inf
    best_features = None
    best_model = None
    all_results = []
    
    for k in range(1, max_features + 1):
        for features in combinations(X.columns, k):
            X_subset = X[list(features)]
            model = LinearRegression().fit(X_subset, y)
            score = r2_score(y, model.predict(X_subset))
            
            all_results.append({
                'features': features,
                'n_features': k,
                'r_squared': score
            })
            
            if score > best_score:
                best_score = score
                best_features = features
                best_model = model
    
    # Fit full model with best features
    X_best = X[list(best_features)]
    best_results = fit_multiple_linear_regression(X_best, y)
    
    return {
        'best_features': list(best_features),
        'best_r_squared': best_score,
        'n_best_features': len(best_features),
        'all_results': all_results,
        'model_results': best_results
    }

# --- Multicollinearity Diagnostics ---

def check_multicollinearity(X):
    """
    Check for multicollinearity using VIF and correlation analysis.
    
    Parameters:
    -----------
    X : DataFrame
        Predictor variables
        
    Returns:
    --------
    dict : Dictionary containing multicollinearity diagnostics
    """
    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data["Tolerance"] = 1 / vif_data["VIF"]
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Identify high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # Threshold for high correlation
                high_corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    # Assess multicollinearity
    high_vif = vif_data[vif_data['VIF'] > 10]
    moderate_vif = vif_data[(vif_data['VIF'] > 5) & (vif_data['VIF'] <= 10)]
    
    assessment = {
        'severe_multicollinearity': len(high_vif) > 0,
        'moderate_multicollinearity': len(moderate_vif) > 0,
        'recommendation': 'No multicollinearity issues' if len(high_vif) == 0 else 
                         'Consider removing variables with high VIF' if len(high_vif) > 0 else
                         'Monitor variables with moderate VIF'
    }
    
    return {
        'vif_results': vif_data,
        'correlation_matrix': corr_matrix,
        'high_correlation_pairs': high_corr_pairs,
        'high_vif_variables': high_vif,
        'moderate_vif_variables': moderate_vif,
        'assessment': assessment
    }

# --- Model Diagnostics ---

def create_multiple_regression_diagnostics(results, figsize=(12, 10)):
    """
    Create comprehensive diagnostic plots for multiple regression.
    
    Parameters:
    -----------
    results : dict
        Results from fit_multiple_linear_regression
    figsize : tuple
        Figure size for the plots
        
    Returns:
    --------
    matplotlib.figure.Figure : Figure object with diagnostic plots
    """
    model = results['model']
    residuals = results['residuals']
    fitted_vals = results['fitted_values']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted_vals, residuals, alpha=0.7)
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    
    # 2. Q-Q plot
    sm.qqplot(residuals, line='s', ax=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    
    # 3. Scale-Location plot
    axes[1, 0].scatter(fitted_vals, np.sqrt(np.abs(residuals)), alpha=0.7)
    axes[1, 0].set_title('Scale-Location Plot')
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('√|Residuals|')
    
    # 4. Residuals vs Leverage
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    axes[1, 1].scatter(leverage, residuals, alpha=0.7)
    axes[1, 1].set_title('Residuals vs Leverage')
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('Residuals')
    
    plt.tight_layout()
    return fig

def analyze_multiple_regression_influence(results):
    """
    Analyze outliers and influential points in multiple regression.
    
    Parameters:
    -----------
    results : dict
        Results from fit_multiple_linear_regression
        
    Returns:
    --------
    dict : Dictionary containing influence analysis results
    """
    model = results['model']
    influence_measures = model.get_influence()
    
    # Get influence measures
    hat_diag = influence_measures.hat_matrix_diag
    cooks_d = influence_measures.cooks_distance[0]
    dffits = influence_measures.dffits[0]
    student_resid = influence_measures.resid_studentized
    
    # Identify influential points
    n = len(results['y'])
    p = len(results['model'].params)
    leverage_threshold = 2 * (p + 1) / n
    cooks_threshold = 4 / n
    
    high_leverage = hat_diag > leverage_threshold
    high_cooks = cooks_d > cooks_threshold
    high_dffits = np.abs(dffits) > 2 * np.sqrt(p/n)
    
    influence_results = {
        'leverage': {
            'values': hat_diag,
            'threshold': leverage_threshold,
            'high_leverage_points': np.where(high_leverage)[0],
            'max_leverage': np.max(hat_diag)
        },
        'cooks_distance': {
            'values': cooks_d,
            'threshold': cooks_threshold,
            'high_influence_points': np.where(high_cooks)[0],
            'max_cooks': np.max(cooks_d)
        },
        'dffits': {
            'values': dffits,
            'threshold': 2 * np.sqrt(p/n),
            'high_dffits_points': np.where(high_dffits)[0],
            'max_dffits': np.max(np.abs(dffits))
        },
        'studentized_residuals': {
            'values': student_resid,
            'outliers': np.where(np.abs(student_resid) > 2)[0]
        },
        'summary': {
            'total_high_leverage': np.sum(high_leverage),
            'total_high_influence': np.sum(high_cooks),
            'total_outliers': np.sum(np.abs(student_resid) > 2)
        }
    }
    
    return influence_results

# --- Cross-Validation ---

def perform_cross_validation(X, y, cv_method='5fold', n_splits=5):
    """
    Perform cross-validation for model evaluation.
    
    Parameters:
    -----------
    X : DataFrame
        Predictor variables
    y : Series
        Response variable
    cv_method : str
        '5fold', '10fold', or 'loo' (leave-one-out)
    n_splits : int
        Number of folds for k-fold CV
        
    Returns:
    --------
    dict : Dictionary containing cross-validation results
    """
    if cv_method == 'loo':
        cv = LeaveOneOut()
        cv_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring='neg_mean_squared_error')
        mse = -cv_scores.mean()
        rmse = np.sqrt(mse)
        r2_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring='r2')
        r2_mean = r2_scores.mean()
    else:
        cv_scores = cross_val_score(LinearRegression(), X, y, cv=n_splits, scoring='neg_mean_squared_error')
        mse = -cv_scores.mean()
        rmse = np.sqrt(mse)
        r2_scores = cross_val_score(LinearRegression(), X, y, cv=n_splits, scoring='r2')
        r2_mean = r2_scores.mean()
    
    return {
        'cv_method': cv_method,
        'n_splits': n_splits if cv_method != 'loo' else len(y),
        'mse': mse,
        'rmse': rmse,
        'r2_mean': r2_mean,
        'r2_std': r2_scores.std(),
        'mse_scores': -cv_scores,
        'r2_scores': r2_scores
    }

# --- Regularization Methods ---

def apply_ridge_regression(X, y, cv_folds=5):
    """
    Apply ridge regression with cross-validation.
    
    Parameters:
    -----------
    X : DataFrame
        Predictor variables
    y : Series
        Response variable
    cv_folds : int
        Number of CV folds
        
    Returns:
    --------
    dict : Dictionary containing ridge regression results
    """
    # Grid search for optimal alpha
    ridge = Ridge()
    param_grid = {'alpha': np.logspace(-3, 3, 100)}
    ridge_cv = GridSearchCV(ridge, param_grid, cv=cv_folds, scoring='neg_mean_squared_error')
    ridge_cv.fit(X, y)
    
    # Fit best model
    best_ridge = Ridge(alpha=ridge_cv.best_params_['alpha'])
    best_ridge.fit(X, y)
    
    # Calculate R²
    r2 = r2_score(y, best_ridge.predict(X))
    
    return {
        'best_alpha': ridge_cv.best_params_['alpha'],
        'coefficients': best_ridge.coef_,
        'intercept': best_ridge.intercept_,
        'r_squared': r2,
        'cv_results': ridge_cv.cv_results_,
        'model': best_ridge
    }

def apply_lasso_regression(X, y, cv_folds=5):
    """
    Apply lasso regression with cross-validation.
    
    Parameters:
    -----------
    X : DataFrame
        Predictor variables
    y : Series
        Response variable
    cv_folds : int
        Number of CV folds
        
    Returns:
    --------
    dict : Dictionary containing lasso regression results
    """
    # Grid search for optimal alpha
    lasso = Lasso()
    param_grid = {'alpha': np.logspace(-3, 3, 100)}
    lasso_cv = GridSearchCV(lasso, param_grid, cv=cv_folds, scoring='neg_mean_squared_error')
    lasso_cv.fit(X, y)
    
    # Fit best model
    best_lasso = Lasso(alpha=lasso_cv.best_params_['alpha'])
    best_lasso.fit(X, y)
    
    # Calculate R²
    r2 = r2_score(y, best_lasso.predict(X))
    
    # Identify non-zero coefficients
    non_zero_coeffs = X.columns[best_lasso.coef_ != 0].tolist()
    
    return {
        'best_alpha': lasso_cv.best_params_['alpha'],
        'coefficients': best_lasso.coef_,
        'intercept': best_lasso.intercept_,
        'r_squared': r2,
        'non_zero_coefficients': non_zero_coeffs,
        'n_non_zero': len(non_zero_coeffs),
        'cv_results': lasso_cv.cv_results_,
        'model': best_lasso
    }

def apply_elastic_net(X, y, cv_folds=5):
    """
    Apply elastic net regression with cross-validation.
    
    Parameters:
    -----------
    X : DataFrame
        Predictor variables
    y : Series
        Response variable
    cv_folds : int
        Number of CV folds
        
    Returns:
    --------
    dict : Dictionary containing elastic net results
    """
    # Grid search for optimal parameters
    elastic = ElasticNet()
    param_grid = {
        'alpha': np.logspace(-3, 3, 50),
        'l1_ratio': np.linspace(0, 1, 20)
    }
    elastic_cv = GridSearchCV(elastic, param_grid, cv=cv_folds, scoring='neg_mean_squared_error')
    elastic_cv.fit(X, y)
    
    # Fit best model
    best_elastic = ElasticNet(
        alpha=elastic_cv.best_params_['alpha'],
        l1_ratio=elastic_cv.best_params_['l1_ratio']
    )
    best_elastic.fit(X, y)
    
    # Calculate R²
    r2 = r2_score(y, best_elastic.predict(X))
    
    return {
        'best_alpha': elastic_cv.best_params_['alpha'],
        'best_l1_ratio': elastic_cv.best_params_['l1_ratio'],
        'coefficients': best_elastic.coef_,
        'intercept': best_elastic.intercept_,
        'r_squared': r2,
        'cv_results': elastic_cv.cv_results_,
        'model': best_elastic
    }

# --- Interaction and Polynomial Terms ---

def add_interaction_terms(data, var1, var2, interaction_name=None):
    """
    Add interaction term to the dataset.
    
    Parameters:
    -----------
    data : DataFrame
        Original dataset
    var1, var2 : str
        Variable names to interact
    interaction_name : str, optional
        Name for the interaction term
        
    Returns:
    --------
    DataFrame : Dataset with interaction term
    """
    if interaction_name is None:
        interaction_name = f"{var1}_x_{var2}"
    
    data[interaction_name] = data[var1] * data[var2]
    return data

def add_polynomial_terms(data, variable, degree=2, prefix=None):
    """
    Add polynomial terms to the dataset.
    
    Parameters:
    -----------
    data : DataFrame
        Original dataset
    variable : str
        Variable name
    degree : int
        Highest polynomial degree
    prefix : str, optional
        Prefix for polynomial terms
        
    Returns:
    --------
    DataFrame : Dataset with polynomial terms
    """
    if prefix is None:
        prefix = variable
    
    for d in range(2, degree + 1):
        data[f"{prefix}^{d}"] = data[variable] ** d
    
    return data

# --- Practical Examples ---

def mpg_multiple_predictors_example():
    """
    Example: Predicting MPG from multiple predictors using mtcars dataset.
    
    Returns:
    --------
    dict : Complete analysis results
    """
    # Load data
    mtcars = load_mtcars_data()
    
    # Select predictors
    predictors = ['wt', 'hp', 'disp']
    X = mtcars[predictors]
    y = mtcars['mpg']
    
    # Fit multiple regression
    results = fit_multiple_linear_regression(X, y)
    
    # Interpret results
    interpretation = interpret_multiple_regression_results(results)
    
    # Check multicollinearity
    multicollinearity = check_multicollinearity(X)
    
    # Perform diagnostics
    diagnostics = create_multiple_regression_diagnostics(results)
    influence = analyze_multiple_regression_influence(results)
    
    # Cross-validation
    cv_results = perform_cross_validation(X, y)
    
    return {
        'results': results,
        'interpretation': interpretation,
        'multicollinearity': multicollinearity,
        'influence': influence,
        'cv_results': cv_results,
        'data': mtcars
    }

def real_estate_example():
    """
    Example: Real estate price prediction with multiple predictors.
    
    Returns:
    --------
    dict : Complete analysis results
    """
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
    
    # Fit model
    predictors = ['square_feet', 'bedrooms', 'bathrooms', 'age', 'location_score']
    X = real_estate_data[predictors]
    y = real_estate_data['price']
    
    results = fit_multiple_linear_regression(X, y)
    interpretation = interpret_multiple_regression_results(results)
    
    # Model selection
    stepwise_forward = perform_stepwise_selection(X, y, direction='forward', n_features_to_select=3)
    best_subset = perform_best_subset_selection(X, y, max_features=5)
    
    # Regularization
    ridge_results = apply_ridge_regression(X, y)
    lasso_results = apply_lasso_regression(X, y)
    elastic_results = apply_elastic_net(X, y)
    
    return {
        'results': results,
        'interpretation': interpretation,
        'stepwise_forward': stepwise_forward,
        'best_subset': best_subset,
        'ridge': ridge_results,
        'lasso': lasso_results,
        'elastic_net': elastic_results,
        'data': real_estate_data
    }

def marketing_example():
    """
    Example: Marketing sales prediction with multiple predictors.
    
    Returns:
    --------
    dict : Complete analysis results
    """
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
    
    # Create dummy variables for season
    marketing_data = pd.get_dummies(marketing_data, columns=['season'], drop_first=True)
    
    # Fit model
    predictors = ['ad_spend', 'social_media_posts', 'email_sends', 
                  'season_Summer', 'season_Fall', 'season_Winter']
    X = marketing_data[predictors]
    y = marketing_data['sales']
    
    results = fit_multiple_linear_regression(X, y)
    interpretation = interpret_multiple_regression_results(results)
    
    # Add interaction terms
    marketing_data_interaction = add_interaction_terms(marketing_data, 'ad_spend', 'social_media_posts')
    X_interaction = marketing_data_interaction[predictors + ['ad_spend_x_social_media_posts']]
    
    interaction_results = fit_multiple_linear_regression(X_interaction, y)
    
    return {
        'results': results,
        'interpretation': interpretation,
        'interaction_results': interaction_results,
        'data': marketing_data
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    """
    Main execution block demonstrating multiple linear regression workflow.
    This shows how to use all the functions from the lesson.
    """
    print("=== MULTIPLE LINEAR REGRESSION DEMONSTRATION ===\n")
    
    # 1. Generate sample data
    print("1. Generating sample data...")
    X, y = simulate_multiple_regression_data(seed=42, n_samples=100, n_predictors=3)
    X_df = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
    y_series = pd.Series(y, name='y')
    print(f"   Created dataset with {len(y)} observations and {X.shape[1]} predictors\n")
    
    # 2. Fit multiple regression model
    print("2. Fitting multiple linear regression...")
    results = fit_multiple_linear_regression(X_df, y_series)
    print(f"   Model fitted successfully")
    print(f"   R² = {results['r_squared']:.3f}")
    print(f"   Adjusted R² = {results['adj_r_squared']:.3f}\n")
    
    # 3. Interpret results
    print("3. Interpreting results...")
    interpretation = interpret_multiple_regression_results(results)
    print(f"   F-statistic: {interpretation['model_fit']['f_statistic']:.2f} (p = {interpretation['model_fit']['f_pvalue']:.4f})")
    print(f"   Model explains {interpretation['model_fit']['r_squared']*100:.1f}% of variance\n")
    
    # 4. Check multicollinearity
    print("4. Checking multicollinearity...")
    multicollinearity = check_multicollinearity(X_df)
    print(f"   Severe multicollinearity: {multicollinearity['assessment']['severe_multicollinearity']}")
    print(f"   Recommendation: {multicollinearity['assessment']['recommendation']}\n")
    
    # 5. Perform model selection
    print("5. Performing model selection...")
    stepwise_results = perform_stepwise_selection(X_df, y_series, direction='forward', n_features_to_select=2)
    print(f"   Forward selection selected: {stepwise_results['selected_features']}")
    print(f"   Best subset selection: {perform_best_subset_selection(X_df, y_series, max_features=3)['best_features']}\n")
    
    # 6. Cross-validation
    print("6. Performing cross-validation...")
    cv_results = perform_cross_validation(X_df, y_series, cv_method='5fold')
    print(f"   5-fold CV R²: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
    print(f"   CV RMSE: {cv_results['rmse']:.3f}\n")
    
    # 7. Apply regularization
    print("7. Applying regularization...")
    ridge_results = apply_ridge_regression(X_df, y_series)
    lasso_results = apply_lasso_regression(X_df, y_series)
    print(f"   Ridge R²: {ridge_results['r_squared']:.3f} (α = {ridge_results['best_alpha']:.3f})")
    print(f"   Lasso R²: {lasso_results['r_squared']:.3f} (α = {lasso_results['best_alpha']:.3f})")
    print(f"   Lasso selected {lasso_results['n_non_zero']} variables\n")
    
    # 8. Create diagnostics
    print("8. Creating diagnostic plots...")
    create_multiple_regression_diagnostics(results)
    print("   Diagnostic plots created\n")
    
    # 9. Run practical examples
    print("9. Running practical examples...")
    mpg_analysis = mpg_multiple_predictors_example()
    real_estate_analysis = real_estate_example()
    marketing_analysis = marketing_example()
    print("   All practical examples completed\n")
    
    print("=== DEMONSTRATION COMPLETE ===")
    print("\nAll functions from the multiple linear regression lesson have been demonstrated.")
    print("Refer to the markdown file for detailed explanations and theory.") 