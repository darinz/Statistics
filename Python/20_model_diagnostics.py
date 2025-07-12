"""
Model Diagnostics and Validation in Python

This module provides comprehensive functions for performing model diagnostics,
including residual analysis, influence diagnostics, assumption checking, and validation.

Key Functions:
- perform_residual_analysis: Comprehensive residual analysis for linear regression
- analyze_leverage_and_influence: Leverage and influence diagnostics
- check_multicollinearity: VIF and multicollinearity analysis
- test_heteroscedasticity: Heteroscedasticity testing
- test_independence: Independence and autocorrelation testing
- logistic_regression_diagnostics: Diagnostics for logistic regression
- create_diagnostic_plots: Visual diagnostic plots
- practical_examples: Real-world diagnostic examples

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, classification_report
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- Data Loading and Preparation ---

def load_boston_data():
    """
    Load the Boston housing dataset for diagnostic analysis.
    
    Returns:
    --------
    DataFrame : Boston housing dataset
    """
    try:
        boston = load_boston()
        data = pd.DataFrame(boston.data, columns=boston.feature_names)
        data['MEDV'] = boston.target
        return data
    except:
        # Fallback: create simulated boston-like data
        np.random.seed(42)
        n = 506
        rm = np.random.normal(6.3, 0.7, n)
        lstat = np.random.normal(12.7, 7.1, n)
        ptratio = np.random.normal(18.5, 2.1, n)
        
        # Generate correlated MEDV
        medv = 30.0 + 5.0 * rm - 0.5 * lstat - 0.5 * ptratio + np.random.normal(0, 3, n)
        
        data = pd.DataFrame({
            'RM': rm,
            'LSTAT': lstat,
            'PTRATIO': ptratio,
            'MEDV': medv
        })
        return data

def simulate_data_with_outliers(seed=42, n_samples=100, outlier_fraction=0.05):
    """
    Simulate data with outliers for diagnostic demonstration.
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    n_samples : int
        Number of samples to generate
    outlier_fraction : float
        Fraction of outliers to add
        
    Returns:
    --------
    tuple : (X, y) arrays with outliers
    """
    np.random.seed(seed)
    
    # Generate normal data
    X = np.random.normal(0, 2, (n_samples, 2))
    y = 10 + 2 * X[:, 0] - 1.5 * X[:, 1] + np.random.normal(0, 1, n_samples)
    
    # Add outliers
    n_outliers = int(n_samples * outlier_fraction)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    
    # High-leverage outliers (extreme X values)
    X[outlier_indices[::2], 0] = np.random.normal(8, 1, len(outlier_indices[::2]))
    
    # High-influence outliers (extreme Y values)
    y[outlier_indices[1::2]] = np.random.normal(50, 5, len(outlier_indices[1::2]))
    
    return X, y

# --- Linear Regression Diagnostics ---

def perform_residual_analysis(X, y, feature_names=None):
    """
    Perform comprehensive residual analysis for linear regression.
    
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
    dict : Dictionary containing residual analysis results
    """
    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = [f'X{i+1}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name='y')
    
    # Add constant for statsmodels
    X_with_const = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y, X_with_const).fit()
    
    # Calculate various residuals
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    # Standardized and studentized residuals
    influence = model.get_influence()
    std_resid = influence.resid_studentized_internal
    stud_resid = influence.resid_studentized_external
    
    # Normality tests
    shapiro_stat, shapiro_p = stats.shapiro(stud_resid)
    
    # Residual statistics
    residual_stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals),
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'normal_residuals': shapiro_p > 0.05
    }
    
    return {
        'model': model,
        'residuals': residuals,
        'fitted_values': fitted_values,
        'std_residuals': std_resid,
        'stud_residuals': stud_resid,
        'residual_stats': residual_stats,
        'X': X,
        'y': y
    }

def create_residual_plots(results, figsize=(12, 10)):
    """
    Create comprehensive residual diagnostic plots.
    
    Parameters:
    -----------
    results : dict
        Results from perform_residual_analysis
    figsize : tuple
        Figure size for the plots
        
    Returns:
    --------
    matplotlib.figure.Figure : Figure object with residual plots
    """
    model = results['model']
    residuals = results['residuals']
    fitted_values = results['fitted_values']
    stud_resid = results['stud_residuals']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted_values, residuals, alpha=0.7)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # 2. Q-Q plot for normality
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q')
    
    # 3. Scale-Location plot
    axes[1, 0].scatter(fitted_values, np.sqrt(np.abs(residuals)), alpha=0.7)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('sqrt(|Residuals|)')
    axes[1, 0].set_title('Scale-Location')
    
    # 4. Studentized residuals vs fitted
    axes[1, 1].scatter(fitted_values, stud_resid, alpha=0.7)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].axhline(y=2, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].axhline(y=-2, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('Studentized Residuals')
    axes[1, 1].set_title('Studentized Residuals vs Fitted')
    
    plt.tight_layout()
    return fig

def analyze_leverage_and_influence(results):
    """
    Analyze leverage and influence for linear regression.
    
    Parameters:
    -----------
    results : dict
        Results from perform_residual_analysis
        
    Returns:
    --------
    dict : Dictionary containing leverage and influence analysis
    """
    model = results['model']
    influence = model.get_influence()
    
    # Leverage
    leverage = influence.hat_matrix_diag
    n = len(results['y'])
    p = len(model.params)
    leverage_threshold = 2 * (p + 1) / n
    
    # Cook's distance
    cooksd = influence.cooks_distance[0]
    cooksd_threshold = 4 / n
    
    # DFFITS
    dffits = influence.dffits[0]
    dffits_threshold = 2 * np.sqrt(p / n)
    
    # Identify influential points
    high_leverage = leverage > leverage_threshold
    high_cooks = cooksd > cooksd_threshold
    high_dffits = np.abs(dffits) > dffits_threshold
    
    influence_results = {
        'leverage': {
            'values': leverage,
            'threshold': leverage_threshold,
            'high_leverage_points': np.where(high_leverage)[0],
            'max_leverage': np.max(leverage)
        },
        'cooks_distance': {
            'values': cooksd,
            'threshold': cooksd_threshold,
            'high_influence_points': np.where(high_cooks)[0],
            'max_cooks': np.max(cooksd)
        },
        'dffits': {
            'values': dffits,
            'threshold': dffits_threshold,
            'high_dffits_points': np.where(high_dffits)[0],
            'max_dffits': np.max(np.abs(dffits))
        },
        'summary': {
            'total_high_leverage': np.sum(high_leverage),
            'total_high_influence': np.sum(high_cooks),
            'total_high_dffits': np.sum(high_dffits)
        }
    }
    
    return influence_results

def plot_leverage_and_influence(results, influence_results):
    """
    Create plots for leverage and influence analysis.
    
    Parameters:
    -----------
    results : dict
        Results from perform_residual_analysis
    influence_results : dict
        Results from analyze_leverage_and_influence
    """
    leverage = influence_results['leverage']['values']
    cooksd = influence_results['cooks_distance']['values']
    stud_resid = results['stud_residuals']
    
    # Leverage plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(leverage)), leverage, alpha=0.7)
    plt.axhline(y=influence_results['leverage']['threshold'], color='red', linestyle='--', 
               label=f'Threshold: {influence_results["leverage"]["threshold"]:.3f}')
    plt.xlabel('Observation')
    plt.ylabel('Leverage')
    plt.title('Leverage')
    plt.legend()
    plt.show()
    
    # Cook's distance plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(cooksd)), cooksd, alpha=0.7)
    plt.axhline(y=influence_results['cooks_distance']['threshold'], color='red', linestyle='--', 
               label=f'Threshold: {influence_results["cooks_distance"]["threshold"]:.3f}')
    plt.xlabel('Observation')
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance")
    plt.legend()
    plt.show()
    
    # Influence plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(leverage, stud_resid, s=100*cooksd, alpha=0.6)
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Studentized Residuals')
    ax.set_title('Influence Plot')
    ax.axhline(y=0, color='red', linestyle='--')
    ax.axhline(y=2, color='red', linestyle='--')
    ax.axhline(y=-2, color='red', linestyle='--')
    plt.show()

def check_multicollinearity(X):
    """
    Check for multicollinearity using VIF analysis.
    
    Parameters:
    -----------
    X : DataFrame
        Predictor variables
        
    Returns:
    --------
    dict : Dictionary containing multicollinearity diagnostics
    """
    # Add constant for VIF calculation
    X_with_const = sm.add_constant(X)
    
    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    vif_data["Tolerance"] = 1 / vif_data["VIF"]
    
    # Identify problematic variables
    high_vif = vif_data[vif_data['VIF'] > 10]
    moderate_vif = vif_data[(vif_data['VIF'] > 5) & (vif_data['VIF'] <= 10)]
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Identify high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
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

def test_heteroscedasticity(results):
    """
    Test for heteroscedasticity using Breusch-Pagan test.
    
    Parameters:
    -----------
    results : dict
        Results from perform_residual_analysis
        
    Returns:
    --------
    dict : Dictionary containing heteroscedasticity test results
    """
    model = results['model']
    residuals = results['residuals']
    X_with_const = sm.add_constant(results['X'])
    
    # Breusch-Pagan test
    bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(residuals, X_with_const)
    
    # Visual assessment
    fitted_values = results['fitted_values']
    
    # Calculate variance ratio (simple test)
    n = len(residuals)
    mid_point = n // 2
    var_low = np.var(residuals[:mid_point])
    var_high = np.var(residuals[mid_point:])
    var_ratio = var_high / var_low if var_low > 0 else np.inf
    
    heteroscedasticity_results = {
        'breusch_pagan': {
            'statistic': bp_stat,
            'p_value': bp_p,
            'f_statistic': bp_f,
            'f_p_value': bp_f_p,
            'heteroscedastic': bp_p < 0.05
        },
        'variance_ratio': {
            'ratio': var_ratio,
            'var_low': var_low,
            'var_high': var_high,
            'heteroscedastic': var_ratio > 2.0 or var_ratio < 0.5
        },
        'overall_assessment': {
            'heteroscedastic': bp_p < 0.05 or (var_ratio > 2.0 or var_ratio < 0.5),
            'recommendation': 'Consider robust standard errors or transformations' if bp_p < 0.05 else 'No evidence of heteroscedasticity'
        }
    }
    
    return heteroscedasticity_results

def test_independence(results):
    """
    Test for independence using Durbin-Watson test.
    
    Parameters:
    -----------
    results : dict
        Results from perform_residual_analysis
        
    Returns:
    --------
    dict : Dictionary containing independence test results
    """
    residuals = results['residuals']
    
    # Durbin-Watson test
    dw_stat = durbin_watson(residuals)
    
    # Interpretation
    if dw_stat < 1.5:
        autocorr_type = "Positive autocorrelation"
        autocorr_present = True
    elif dw_stat > 2.5:
        autocorr_type = "Negative autocorrelation"
        autocorr_present = True
    else:
        autocorr_type = "No autocorrelation"
        autocorr_present = False
    
    independence_results = {
        'durbin_watson': {
            'statistic': dw_stat,
            'autocorrelation_type': autocorr_type,
            'autocorrelation_present': autocorr_present
        },
        'interpretation': {
            'dw_approx_2': 'No autocorrelation',
            'dw_less_2': 'Positive autocorrelation',
            'dw_greater_2': 'Negative autocorrelation'
        },
        'recommendation': 'Consider time series methods' if autocorr_present else 'No independence issues detected'
    }
    
    return independence_results

# --- Logistic Regression Diagnostics ---

def perform_logistic_regression_diagnostics(X, y, feature_names=None):
    """
    Perform diagnostics for logistic regression.
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Predictor variables
    y : array-like
        Binary response variable
    feature_names : list, optional
        Names for the predictor variables
        
    Returns:
    --------
    dict : Dictionary containing logistic regression diagnostics
    """
    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = [f'X{i+1}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name='y')
    
    # Add constant for statsmodels
    X_with_const = sm.add_constant(X)
    
    # Fit logistic regression
    logit_model = sm.Logit(y, X_with_const).fit()
    
    # Calculate residuals
    pearson_resid = logit_model.resid_pearson
    dev_resid = logit_model.resid_dev
    
    # Leverage and influence
    influence = logit_model.get_influence()
    leverage = influence.hat_matrix_diag
    cooksd = influence.cooks_distance[0]
    
    # Model fit statistics
    pseudo_r_squared = 1 - (logit_model.llf / logit_model.llnull)
    
    # Hosmer-Lemeshow test
    hl_stat, hl_p = hosmer_lemeshow_test(y, pd.Series(logit_model.fittedvalues))
    
    # ROC curve and AUC
    y_pred_proba = logit_model.fittedvalues
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        'model': logit_model,
        'pearson_residuals': pearson_resid,
        'deviance_residuals': dev_resid,
        'leverage': leverage,
        'cooks_distance': cooksd,
        'pseudo_r_squared': pseudo_r_squared,
        'hosmer_lemeshow': {
            'statistic': hl_stat,
            'p_value': hl_p,
            'good_fit': hl_p > 0.05
        },
        'roc_curve': {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        },
        'X': X,
        'y': y
    }

def hosmer_lemeshow_test(y_true, y_pred, n_groups=10):
    """
    Perform Hosmer-Lemeshow goodness-of-fit test.
    
    Parameters:
    -----------
    y_true : Series
        True binary outcomes
    y_pred : Series
        Predicted probabilities
    n_groups : int
        Number of groups for the test
        
    Returns:
    --------
    tuple : (chi_square_statistic, p_value)
    """
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

def create_logistic_diagnostic_plots(results):
    """
    Create diagnostic plots for logistic regression.
    
    Parameters:
    -----------
    results : dict
        Results from perform_logistic_regression_diagnostics
    """
    model = results['model']
    pearson_resid = results['pearson_residuals']
    dev_resid = results['deviance_residuals']
    leverage = results['leverage']
    cooksd = results['cooks_distance']
    roc_data = results['roc_curve']
    
    # Residual plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(model.fittedvalues, pearson_resid, alpha=0.7)
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('Fitted Values')
    axes[0].set_ylabel('Pearson Residuals')
    axes[0].set_title('Pearson Residuals vs Fitted')
    
    axes[1].scatter(model.fittedvalues, dev_resid, alpha=0.7)
    axes[1].axhline(y=0, color='red', linestyle='--')
    axes[1].set_xlabel('Fitted Values')
    axes[1].set_ylabel('Deviance Residuals')
    axes[1].set_title('Deviance Residuals vs Fitted')
    
    plt.tight_layout()
    plt.show()
    
    # Leverage and influence plots
    plt.figure(figsize=(10, 6))
    plt.scatter(leverage, dev_resid, alpha=0.7)
    plt.xlabel('Leverage')
    plt.ylabel('Deviance Residuals')
    plt.title('Deviance Residuals vs Leverage')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(cooksd)), cooksd, alpha=0.7)
    plt.axhline(y=4/len(cooksd), color='red', linestyle='--', 
               label=f'Threshold: {4/len(cooksd):.3f}')
    plt.xlabel('Observation')
    plt.ylabel("Cook's Distance")
    plt.title("Cook's Distance (Logistic)")
    plt.legend()
    plt.show()
    
    # ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(roc_data['fpr'], roc_data['tpr'], color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_data["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# --- Comprehensive Diagnostic Functions ---

def perform_comprehensive_diagnostics(X, y, model_type='linear', feature_names=None):
    """
    Perform comprehensive diagnostics for regression models.
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Predictor variables
    y : array-like
        Response variable
    model_type : str
        'linear' or 'logistic'
    feature_names : list, optional
        Names for the predictor variables
        
    Returns:
    --------
    dict : Dictionary containing all diagnostic results
    """
    if model_type == 'linear':
        # Linear regression diagnostics
        residual_results = perform_residual_analysis(X, y, feature_names)
        influence_results = analyze_leverage_and_influence(residual_results)
        multicollinearity_results = check_multicollinearity(X)
        heteroscedasticity_results = test_heteroscedasticity(residual_results)
        independence_results = test_independence(residual_results)
        
        return {
            'residual_analysis': residual_results,
            'leverage_and_influence': influence_results,
            'multicollinearity': multicollinearity_results,
            'heteroscedasticity': heteroscedasticity_results,
            'independence': independence_results,
            'model_type': 'linear'
        }
    
    elif model_type == 'logistic':
        # Logistic regression diagnostics
        logit_results = perform_logistic_regression_diagnostics(X, y, feature_names)
        
        return {
            'logistic_diagnostics': logit_results,
            'model_type': 'logistic'
        }
    
    else:
        raise ValueError("model_type must be 'linear' or 'logistic'")

def create_comprehensive_diagnostic_plots(diagnostic_results):
    """
    Create comprehensive diagnostic plots.
    
    Parameters:
    -----------
    diagnostic_results : dict
        Results from perform_comprehensive_diagnostics
    """
    if diagnostic_results['model_type'] == 'linear':
        # Linear regression plots
        create_residual_plots(diagnostic_results['residual_analysis'])
        plot_leverage_and_influence(
            diagnostic_results['residual_analysis'], 
            diagnostic_results['leverage_and_influence']
        )
        
        # Additional plots
        residuals = diagnostic_results['residual_analysis']['residuals']
        fitted_values = diagnostic_results['residual_analysis']['fitted_values']
        
        # Residuals vs fitted for heteroscedasticity
        plt.figure(figsize=(10, 6))
        plt.scatter(fitted_values, residuals, alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted Values (Heteroscedasticity Check)')
        plt.show()
        
    elif diagnostic_results['model_type'] == 'logistic':
        # Logistic regression plots
        create_logistic_diagnostic_plots(diagnostic_results['logistic_diagnostics'])

# --- Practical Examples ---

def boston_housing_diagnostics_example():
    """
    Example: Comprehensive diagnostics for Boston housing dataset.
    
    Returns:
    --------
    dict : Complete diagnostic results
    """
    # Load data
    data = load_boston_data()
    
    # Select predictors
    X = data[['RM', 'LSTAT', 'PTRATIO']]
    y = data['MEDV']
    
    # Perform comprehensive diagnostics
    diagnostics = perform_comprehensive_diagnostics(X, y, model_type='linear')
    
    # Create plots
    create_comprehensive_diagnostic_plots(diagnostics)
    
    return {
        'diagnostics': diagnostics,
        'data': data
    }

def logistic_regression_example():
    """
    Example: Logistic regression diagnostics for binary outcome.
    
    Returns:
    --------
    dict : Complete diagnostic results
    """
    # Load data
    data = load_boston_data()
    
    # Create binary outcome (high vs low median value)
    data['high_value'] = (data['MEDV'] > data['MEDV'].median()).astype(int)
    
    # Select predictors
    X = data[['RM', 'LSTAT', 'PTRATIO']]
    y = data['high_value']
    
    # Perform logistic regression diagnostics
    diagnostics = perform_comprehensive_diagnostics(X, y, model_type='logistic')
    
    # Create plots
    create_comprehensive_diagnostic_plots(diagnostics)
    
    return {
        'diagnostics': diagnostics,
        'data': data
    }

def outlier_influence_example():
    """
    Example: Diagnostics with simulated outliers and influential points.
    
    Returns:
    --------
    dict : Complete diagnostic results
    """
    # Generate data with outliers
    X, y = simulate_data_with_outliers(n_samples=100, outlier_fraction=0.1)
    X_df = pd.DataFrame(X, columns=['X1', 'X2'])
    y_series = pd.Series(y, name='y')
    
    # Perform comprehensive diagnostics
    diagnostics = perform_comprehensive_diagnostics(X_df, y_series, model_type='linear')
    
    # Create plots
    create_comprehensive_diagnostic_plots(diagnostics)
    
    return {
        'diagnostics': diagnostics,
        'data': {'X': X_df, 'y': y_series}
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    """
    Main execution block demonstrating model diagnostics workflow.
    This shows how to use all the functions from the lesson.
    """
    print("=== MODEL DIAGNOSTICS DEMONSTRATION ===\n")
    
    # 1. Boston housing diagnostics
    print("1. Boston Housing Dataset Diagnostics...")
    boston_results = boston_housing_diagnostics_example()
    print("   Linear regression diagnostics completed")
    print(f"   Residuals normal: {boston_results['diagnostics']['residual_analysis']['residual_stats']['normal_residuals']}")
    print(f"   Heteroscedasticity: {boston_results['diagnostics']['heteroscedasticity']['overall_assessment']['heteroscedastic']}")
    print(f"   Independence issues: {boston_results['diagnostics']['independence']['durbin_watson']['autocorrelation_present']}\n")
    
    # 2. Logistic regression diagnostics
    print("2. Logistic Regression Diagnostics...")
    logit_results = logistic_regression_example()
    print("   Logistic regression diagnostics completed")
    print(f"   Hosmer-Lemeshow p-value: {logit_results['diagnostics']['logistic_diagnostics']['hosmer_lemeshow']['p_value']:.4f}")
    print(f"   AUC: {logit_results['diagnostics']['logistic_diagnostics']['roc_curve']['auc']:.3f}\n")
    
    # 3. Outlier and influence analysis
    print("3. Outlier and Influence Analysis...")
    outlier_results = outlier_influence_example()
    print("   Outlier analysis completed")
    print(f"   High leverage points: {outlier_results['diagnostics']['leverage_and_influence']['summary']['total_high_leverage']}")
    print(f"   High influence points: {outlier_results['diagnostics']['leverage_and_influence']['summary']['total_high_influence']}\n")
    
    # 4. Individual diagnostic tests
    print("4. Individual Diagnostic Tests...")
    data = load_boston_data()
    X = data[['RM', 'LSTAT', 'PTRATIO']]
    y = data['MEDV']
    
    # Residual analysis
    residual_results = perform_residual_analysis(X, y)
    print("   Residual analysis completed")
    
    # Multicollinearity
    multicollinearity = check_multicollinearity(X)
    print(f"   Severe multicollinearity: {multicollinearity['assessment']['severe_multicollinearity']}")
    
    # Heteroscedasticity
    heteroscedasticity = test_heteroscedasticity(residual_results)
    print(f"   Heteroscedasticity: {heteroscedasticity['overall_assessment']['heteroscedastic']}")
    
    # Independence
    independence = test_independence(residual_results)
    print(f"   Independence issues: {independence['durbin_watson']['autocorrelation_present']}\n")
    
    print("=== DEMONSTRATION COMPLETE ===")
    print("\nAll functions from the model diagnostics lesson have been demonstrated.")
    print("Refer to the markdown file for detailed explanations and theory.") 