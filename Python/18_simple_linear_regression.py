"""
Simple Linear Regression in Python

This module provides comprehensive functions for performing simple linear regression,
including model fitting, diagnostics, assumption checking, and visualization.

Key Functions:
- fit_simple_linear_regression: Fit and analyze simple linear regression models
- calculate_confidence_intervals: Confidence and prediction intervals
- check_regression_assumptions: Comprehensive assumption checking
- create_diagnostic_plots: Regression diagnostic visualizations
- analyze_influence: Outlier and influence diagnostics
- practical_examples: Real-world regression examples

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# --- Data Loading and Preparation ---

def load_mtcars_data():
    """
    Load the mtcars dataset for regression analysis.
    
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
        mpg = 37.3 - 5.34 * wt + np.random.normal(0, 2.5, n)
        hp = np.random.normal(146.7, 68.6, n)
        
        mtcars = pd.DataFrame({
            'mpg': mpg,
            'wt': wt,
            'hp': hp
        })
        return mtcars

def simulate_regression_data(seed=42, n_samples=100, slope=2.0, intercept=10.0, noise_std=1.0):
    """
    Simulate data for simple linear regression demonstration.
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    n_samples : int
        Number of samples to generate
    slope : float
        True slope parameter
    intercept : float
        True intercept parameter
    noise_std : float
        Standard deviation of noise
        
    Returns:
    --------
    tuple : (x, y) arrays for regression analysis
    """
    np.random.seed(seed)
    x = np.random.normal(0, 2, n_samples)
    y = intercept + slope * x + np.random.normal(0, noise_std, n_samples)
    return x, y

# --- Simple Linear Regression Analysis ---

def fit_simple_linear_regression(x, y, x_name='x', y_name='y'):
    """
    Fit simple linear regression model and provide comprehensive analysis.
    
    Parameters:
    -----------
    x, y : array-like
        Predictor and response variables
    x_name, y_name : str
        Names for the variables (for formula)
        
    Returns:
    --------
    dict : Dictionary containing model results and diagnostics
    """
    # Create DataFrame
    data = pd.DataFrame({x_name: x, y_name: y})
    
    # Fit model
    formula = f'{y_name} ~ {x_name}'
    model = smf.ols(formula, data=data).fit()
    
    # Extract key results
    coefficients = model.params
    conf_int = model.conf_int()
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    f_statistic = model.fvalue
    f_pvalue = model.f_pvalue
    
    # Calculate fitted values and residuals
    fitted_vals = model.fittedvalues
    residuals = model.resid
    
    # Calculate standard error of estimate
    n = len(x)
    rss = np.sum(residuals**2)
    se_estimate = np.sqrt(rss / (n - 2))
    
    return {
        'model': model,
        'coefficients': coefficients,
        'confidence_intervals': conf_int,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'f_statistic': f_statistic,
        'f_pvalue': f_pvalue,
        'fitted_values': fitted_vals,
        'residuals': residuals,
        'se_estimate': se_estimate,
        'n': n,
        'data': data,
        'formula': formula
    }

def interpret_regression_results(results):
    """
    Provide interpretation of regression results.
    
    Parameters:
    -----------
    results : dict
        Results from fit_simple_linear_regression
        
    Returns:
    --------
    dict : Dictionary with interpretations
    """
    coef = results['coefficients']
    conf_int = results['confidence_intervals']
    
    interpretation = {
        'intercept': {
            'value': coef.iloc[0],
            'interpretation': f"When {list(results['data'].columns)[0]} = 0, the predicted value of {list(results['data'].columns)[1]} is {coef.iloc[0]:.3f}",
            'ci_lower': conf_int.iloc[0, 0],
            'ci_upper': conf_int.iloc[0, 1]
        },
        'slope': {
            'value': coef.iloc[1],
            'interpretation': f"For each one-unit increase in {list(results['data'].columns)[0]}, {list(results['data'].columns)[1]} changes by {coef.iloc[1]:.3f} units on average",
            'ci_lower': conf_int.iloc[1, 0],
            'ci_upper': conf_int.iloc[1, 1]
        },
        'model_fit': {
            'r_squared': results['r_squared'],
            'interpretation': f"The model explains {results['r_squared']*100:.1f}% of the variance in {list(results['data'].columns)[1]}",
            'f_statistic': results['f_statistic'],
            'f_pvalue': results['f_pvalue'],
            'significant': results['f_pvalue'] < 0.05
        }
    }
    
    return interpretation

# --- Confidence and Prediction Intervals ---

def calculate_confidence_intervals(model, new_x_values, alpha=0.05):
    """
    Calculate confidence and prediction intervals for new values.
    
    Parameters:
    -----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted regression model
    new_x_values : array-like
        New predictor values for prediction
    alpha : float
        Significance level (default: 0.05)
        
    Returns:
    --------
    dict : Dictionary containing confidence and prediction intervals
    """
    # Create DataFrame for new data
    x_name = model.model.exog_names[1]  # Skip intercept
    new_data = pd.DataFrame({x_name: new_x_values})
    
    # Get predictions with intervals
    predictions = model.get_prediction(new_data).summary_frame(alpha=alpha)
    
    return {
        'new_x': new_x_values,
        'predictions': predictions,
        'confidence_intervals': predictions[['mean', 'mean_ci_lower', 'mean_ci_upper']],
        'prediction_intervals': predictions[['mean', 'obs_ci_lower', 'obs_ci_upper']],
        'alpha': alpha
    }

# --- Assumption Checking and Diagnostics ---

def check_regression_assumptions(results, alpha=0.05):
    """
    Check all assumptions for simple linear regression.
    
    Parameters:
    -----------
    results : dict
        Results from fit_simple_linear_regression
    alpha : float
        Significance level for normality tests
        
    Returns:
    --------
    dict : Dictionary containing assumption check results
    """
    model = results['model']
    residuals = results['residuals']
    fitted_vals = results['fitted_values']
    x = results['data'].iloc[:, 0]  # Predictor variable
    
    assumption_results = {}
    
    # 1. Linearity
    # Calculate R-squared for linear relationship
    linear_r_squared = results['r_squared']
    assumption_results['linearity'] = {
        'r_squared': linear_r_squared,
        'linear_relationship': linear_r_squared > 0.1,  # Arbitrary threshold
        'interpretation': f"R² = {linear_r_squared:.3f} indicates {'strong' if linear_r_squared > 0.5 else 'moderate' if linear_r_squared > 0.1 else 'weak'} linear relationship"
    }
    
    # 2. Independence
    # Durbin-Watson test for autocorrelation
    dw_stat = sm.stats.durbin_watson(residuals)
    assumption_results['independence'] = {
        'durbin_watson': dw_stat,
        'independent': 1.5 < dw_stat < 2.5,  # Range for independence
        'interpretation': f"Durbin-Watson = {dw_stat:.3f} indicates {'independent' if 1.5 < dw_stat < 2.5 else 'potential autocorrelation'}"
    }
    
    # 3. Homoscedasticity
    # Breusch-Pagan test for heteroscedasticity
    try:
        bp_stat, bp_pvalue = sm.stats.diagnostic.het_breuschpagan(residuals, model.model.exog)
        homoscedastic = bp_pvalue > alpha
    except:
        # Fallback: visual assessment using variance ratio
        var_ratio = np.var(residuals[x < np.median(x)]) / np.var(residuals[x >= np.median(x)])
        homoscedastic = 0.5 < var_ratio < 2.0
        bp_stat, bp_pvalue = None, None
    
    assumption_results['homoscedasticity'] = {
        'breusch_pagan_stat': bp_stat,
        'breusch_pagan_pvalue': bp_pvalue,
        'homoscedastic': homoscedastic,
        'interpretation': f"Residuals are {'homoscedastic' if homoscedastic else 'heteroscedastic'}"
    }
    
    # 4. Normality of residuals
    shapiro_stat, shapiro_pvalue = stats.shapiro(residuals)
    assumption_results['normality'] = {
        'shapiro_statistic': shapiro_stat,
        'shapiro_pvalue': shapiro_pvalue,
        'normal_residuals': shapiro_pvalue > alpha,
        'interpretation': f"Residuals are {'normally distributed' if shapiro_pvalue > alpha else 'not normally distributed'} (Shapiro-Wilk p = {shapiro_pvalue:.4f})"
    }
    
    # Overall assessment
    all_assumptions_met = (
        assumption_results['linearity']['linear_relationship'] and
        assumption_results['independence']['independent'] and
        assumption_results['homoscedasticity']['homoscedastic'] and
        assumption_results['normality']['normal_residuals']
    )
    
    assumption_results['overall_assessment'] = {
        'all_assumptions_met': all_assumptions_met,
        'recommendation': 'Model assumptions are met' if all_assumptions_met else 'Consider transformations or robust methods'
    }
    
    return assumption_results

def create_diagnostic_plots(results, figsize=(12, 10)):
    """
    Create comprehensive diagnostic plots for regression analysis.
    
    Parameters:
    -----------
    results : dict
        Results from fit_simple_linear_regression
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

# --- Outlier and Influence Diagnostics ---

def analyze_influence(results):
    """
    Analyze outliers and influential points in the regression model.
    
    Parameters:
    -----------
    results : dict
        Results from fit_simple_linear_regression
        
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
    n = results['n']
    leverage_threshold = 2 * (2 + 1) / n  # 2*(p+1)/n where p=1 for simple regression
    cooks_threshold = 4 / n
    
    high_leverage = hat_diag > leverage_threshold
    high_cooks = cooks_d > cooks_threshold
    high_dffits = np.abs(dffits) > 2 * np.sqrt(2/n)
    
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
            'threshold': 2 * np.sqrt(2/n),
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

def plot_influence_diagnostics(results, influence_results):
    """
    Create plots for influence diagnostics.
    
    Parameters:
    -----------
    results : dict
        Results from fit_simple_linear_regression
    influence_results : dict
        Results from analyze_influence
    """
    # Cook's distance plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    cooksd = influence_results['cooks_distance']['values']
    threshold = influence_results['cooks_distance']['threshold']
    plt.stem(np.arange(len(cooksd)), cooksd, use_line_collection=True)
    plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
    plt.title("Cook's Distance")
    plt.xlabel('Observation')
    plt.ylabel("Cook's D")
    plt.legend()
    
    # Leverage plot
    plt.subplot(1, 2, 2)
    leverage = influence_results['leverage']['values']
    threshold = influence_results['leverage']['threshold']
    plt.stem(np.arange(len(leverage)), leverage, use_line_collection=True)
    plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
    plt.title('Leverage')
    plt.xlabel('Observation')
    plt.ylabel('Leverage')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# --- Visualization Functions ---

def plot_regression_results(results, title="Simple Linear Regression"):
    """
    Create scatter plot with regression line and confidence intervals.
    
    Parameters:
    -----------
    results : dict
        Results from fit_simple_linear_regression
    title : str
        Plot title
    """
    data = results['data']
    x_col = data.columns[0]
    y_col = data.columns[1]
    
    plt.figure(figsize=(8, 6))
    
    # Scatter plot
    plt.scatter(data[x_col], data[y_col], color='blue', alpha=0.7, label='Data')
    
    # Regression line
    plt.plot(data[x_col], results['fitted_values'], color='red', linewidth=2, label='Regression Line')
    
    # Add R² to title
    title += f" (R² = {results['r_squared']:.3f})"
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Practical Examples ---

def mpg_weight_example():
    """
    Example: Predicting MPG from Weight using mtcars dataset.
    
    Returns:
    --------
    dict : Complete analysis results
    """
    # Load data
    mtcars = load_mtcars_data()
    
    # Fit model
    results = fit_simple_linear_regression(mtcars['wt'], mtcars['mpg'], 'wt', 'mpg')
    
    # Interpret results
    interpretation = interpret_regression_results(results)
    
    # Check assumptions
    assumptions = check_regression_assumptions(results)
    
    # Analyze influence
    influence = analyze_influence(results)
    
    # Create visualizations
    plot_regression_results(results, "MPG vs Weight")
    create_diagnostic_plots(results)
    plot_influence_diagnostics(results, influence)
    
    return {
        'results': results,
        'interpretation': interpretation,
        'assumptions': assumptions,
        'influence': influence,
        'data': mtcars
    }

def house_price_example():
    """
    Example: Predicting House Prices from Size.
    
    Returns:
    --------
    dict : Complete analysis results
    """
    # Simulate data
    np.random.seed(42)
    house_size = np.random.normal(1500, 300, 100)
    house_price = 50000 + 120 * house_size + np.random.normal(0, 20000, 100)
    
    # Fit model
    results = fit_simple_linear_regression(house_size, house_price, 'house_size', 'house_price')
    
    # Interpret results
    interpretation = interpret_regression_results(results)
    
    # Check assumptions
    assumptions = check_regression_assumptions(results)
    
    # Create visualizations
    plot_regression_results(results, "House Price vs Size")
    create_diagnostic_plots(results)
    
    return {
        'results': results,
        'interpretation': interpretation,
        'assumptions': assumptions,
        'data': {'house_size': house_size, 'house_price': house_price}
    }

def exam_score_example():
    """
    Example: Predicting Exam Scores from Study Hours.
    
    Returns:
    --------
    dict : Complete analysis results
    """
    # Simulate data
    np.random.seed(123)
    study_hours = np.random.normal(10, 2, 50)
    exam_score = 40 + 5 * study_hours + np.random.normal(0, 5, 50)
    
    # Fit model
    results = fit_simple_linear_regression(study_hours, exam_score, 'study_hours', 'exam_score')
    
    # Interpret results
    interpretation = interpret_regression_results(results)
    
    # Check assumptions
    assumptions = check_regression_assumptions(results)
    
    # Create visualizations
    plot_regression_results(results, "Exam Score vs Study Hours")
    create_diagnostic_plots(results)
    
    return {
        'results': results,
        'interpretation': interpretation,
        'assumptions': assumptions,
        'data': {'study_hours': study_hours, 'exam_score': exam_score}
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    """
    Main execution block demonstrating simple linear regression workflow.
    This shows how to use all the functions from the lesson.
    """
    print("=== SIMPLE LINEAR REGRESSION DEMONSTRATION ===\n")
    
    # 1. Generate sample data
    print("1. Generating sample data...")
    x, y = simulate_regression_data(seed=42, n_samples=100, slope=2.0, intercept=10.0)
    print(f"   Created dataset with {len(x)} observations")
    print(f"   True slope: 2.0, True intercept: 10.0\n")
    
    # 2. Fit regression model
    print("2. Fitting simple linear regression...")
    results = fit_simple_linear_regression(x, y, 'x', 'y')
    print(f"   Model fitted successfully")
    print(f"   R² = {results['r_squared']:.3f}")
    print(f"   Adjusted R² = {results['adj_r_squared']:.3f}\n")
    
    # 3. Interpret results
    print("3. Interpreting results...")
    interpretation = interpret_regression_results(results)
    print(f"   Intercept: {interpretation['intercept']['value']:.3f}")
    print(f"   Slope: {interpretation['slope']['value']:.3f}")
    print(f"   Model explains {interpretation['model_fit']['r_squared']*100:.1f}% of variance")
    print(f"   F-statistic: {interpretation['model_fit']['f_statistic']:.2f} (p = {interpretation['model_fit']['f_pvalue']:.4f})\n")
    
    # 4. Check assumptions
    print("4. Checking assumptions...")
    assumptions = check_regression_assumptions(results)
    print(f"   Linearity: {assumptions['linearity']['linear_relationship']}")
    print(f"   Independence: {assumptions['independence']['independent']}")
    print(f"   Homoscedasticity: {assumptions['homoscedasticity']['homoscedastic']}")
    print(f"   Normality: {assumptions['normality']['normal_residuals']}")
    print(f"   Overall: {assumptions['overall_assessment']['recommendation']}\n")
    
    # 5. Analyze influence
    print("5. Analyzing influence...")
    influence = analyze_influence(results)
    print(f"   High leverage points: {influence['summary']['total_high_leverage']}")
    print(f"   High influence points: {influence['summary']['total_high_influence']}")
    print(f"   Outliers: {influence['summary']['total_outliers']}\n")
    
    # 6. Calculate confidence intervals
    print("6. Calculating confidence intervals...")
    new_x = np.array([-2, 0, 2, 4])
    intervals = calculate_confidence_intervals(results['model'], new_x)
    print("   Confidence intervals calculated for new x values")
    print("   Prediction intervals calculated for new observations\n")
    
    # 7. Create visualizations
    print("7. Creating visualizations...")
    plot_regression_results(results, "Sample Regression Analysis")
    create_diagnostic_plots(results)
    plot_influence_diagnostics(results, influence)
    print("   All visualizations created\n")
    
    # 8. Run practical examples
    print("8. Running practical examples...")
    mpg_analysis = mpg_weight_example()
    house_analysis = house_price_example()
    exam_analysis = exam_score_example()
    print("   All practical examples completed\n")
    
    print("=== DEMONSTRATION COMPLETE ===")
    print("\nAll functions from the simple linear regression lesson have been demonstrated.")
    print("Refer to the markdown file for detailed explanations and theory.") 