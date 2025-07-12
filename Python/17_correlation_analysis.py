"""
Correlation Analysis in Python

This module provides comprehensive functions for performing correlation analysis,
including Pearson, Spearman, and Kendall correlations, assumption checking, and visualization.

Key Functions:
- calculate_pearson_correlation: Pearson correlation with significance testing
- calculate_spearman_kendall: Spearman and Kendall correlations
- create_correlation_matrix: Correlation matrix with significance testing
- check_correlation_assumptions: Comprehensive assumption checking
- visualize_correlations: Various correlation visualizations
- calculate_confidence_intervals: Confidence intervals for correlations
- practical_examples: Real-world correlation examples

"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# --- Data Simulation for Correlation Analysis ---

def simulate_correlation_data(seed=42, n_samples=100, correlation=0.7, noise_scale=0.5):
    """
    Simulate correlated data for demonstration.
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    n_samples : int
        Number of samples to generate
    correlation : float
        Target correlation coefficient
    noise_scale : float
        Scale of noise to add
        
    Returns:
    --------
    tuple : (x, y) arrays of correlated data
    """
    np.random.seed(seed)
    x = np.random.normal(size=n_samples)
    y = correlation * x + np.random.normal(scale=noise_scale, size=n_samples)
    return x, y

def simulate_multivariate_data(seed=123, n_samples=100, n_variables=3):
    """
    Simulate multivariate data for correlation matrix analysis.
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    n_samples : int
        Number of samples to generate
    n_variables : int
        Number of variables to generate
        
    Returns:
    --------
    DataFrame : Simulated multivariate data
    """
    np.random.seed(seed)
    data = pd.DataFrame({
        'A': np.random.normal(size=n_samples),
        'B': np.random.normal(size=n_samples),
        'C': np.random.normal(size=n_samples)
    })
    return data

# --- Pearson Correlation Analysis ---

def calculate_pearson_correlation(x, y):
    """
    Calculate Pearson correlation coefficient with significance testing.
    
    Parameters:
    -----------
    x, y : array-like
        Variables to correlate
        
    Returns:
    --------
    dict : Dictionary containing correlation coefficient, p-value, and test results
    """
    # Calculate Pearson correlation
    cor_pearson, p_value = stats.pearsonr(x, y)
    
    # Calculate sample size
    n = len(x)
    
    # Calculate confidence interval
    ci_low, ci_up = correlation_confidence_interval(cor_pearson, n)
    
    # Determine effect size
    effect_size = interpret_correlation_strength(abs(cor_pearson))
    
    return {
        'correlation': cor_pearson,
        'p_value': p_value,
        'n': n,
        'ci_lower': ci_low,
        'ci_upper': ci_up,
        'effect_size': effect_size,
        'significant': p_value < 0.05
    }

def correlation_confidence_interval(r, n, alpha=0.05):
    """
    Calculate confidence interval for correlation coefficient using Fisher's z-transformation.
    
    Parameters:
    -----------
    r : float
        Correlation coefficient
    n : int
        Sample size
    alpha : float
        Significance level (default: 0.05)
        
    Returns:
    --------
    tuple : (lower_bound, upper_bound) confidence interval
    """
    from scipy.stats import norm
    
    # Fisher's z-transformation
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha/2)
    
    # Calculate confidence interval
    ci_lower = np.tanh(z - z_crit * se)
    ci_upper = np.tanh(z + z_crit * se)
    
    return ci_lower, ci_upper

def interpret_correlation_strength(abs_r):
    """
    Interpret the strength of correlation based on absolute value.
    
    Parameters:
    -----------
    abs_r : float
        Absolute value of correlation coefficient
        
    Returns:
    --------
    str : Interpretation of correlation strength
    """
    if abs_r < 0.10:
        return "Negligible"
    elif abs_r < 0.30:
        return "Small"
    elif abs_r < 0.50:
        return "Moderate"
    elif abs_r < 0.70:
        return "Large"
    elif abs_r < 0.90:
        return "Very large"
    else:
        return "Nearly perfect"

# --- Spearman and Kendall Correlation ---

def calculate_spearman_kendall(x, y):
    """
    Calculate Spearman and Kendall correlations with significance testing.
    
    Parameters:
    -----------
    x, y : array-like
        Variables to correlate
        
    Returns:
    --------
    dict : Dictionary containing Spearman and Kendall correlation results
    """
    # Spearman correlation
    cor_spearman, p_spearman = stats.spearmanr(x, y)
    
    # Kendall correlation
    cor_kendall, p_kendall = stats.kendalltau(x, y)
    
    return {
        'spearman': {
            'correlation': cor_spearman,
            'p_value': p_spearman,
            'effect_size': interpret_correlation_strength(abs(cor_spearman)),
            'significant': p_spearman < 0.05
        },
        'kendall': {
            'correlation': cor_kendall,
            'p_value': p_kendall,
            'effect_size': interpret_correlation_strength(abs(cor_kendall)),
            'significant': p_kendall < 0.05
        }
    }

# --- Correlation Matrix Analysis ---

def create_correlation_matrix(data, method='pearson'):
    """
    Create correlation matrix with significance testing.
    
    Parameters:
    -----------
    data : DataFrame
        Data to analyze
    method : str
        Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
    --------
    dict : Dictionary containing correlation matrix and significance matrix
    """
    # Calculate correlation matrix
    if method == 'pearson':
        cor_matrix = data.corr(method='pearson')
    elif method == 'spearman':
        cor_matrix = data.corr(method='spearman')
    elif method == 'kendall':
        cor_matrix = data.corr(method='kendall')
    else:
        raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
    
    # Calculate significance matrix
    p_matrix = pd.DataFrame(np.ones((data.shape[1], data.shape[1])), 
                           columns=data.columns, index=data.columns)
    
    for i in data.columns:
        for j in data.columns:
            if i != j:
                if method == 'pearson':
                    _, p = stats.pearsonr(data[i], data[j])
                elif method == 'spearman':
                    _, p = stats.spearmanr(data[i], data[j])
                elif method == 'kendall':
                    _, p = stats.kendalltau(data[i], data[j])
                p_matrix.loc[i, j] = p
            else:
                p_matrix.loc[i, j] = np.nan
    
    return {
        'correlation_matrix': cor_matrix,
        'p_value_matrix': p_matrix,
        'method': method
    }

# --- Assumption Checking ---

def check_correlation_assumptions(x, y, alpha=0.05):
    """
    Check assumptions for Pearson correlation.
    
    Parameters:
    -----------
    x, y : array-like
        Variables to check
    alpha : float
        Significance level for normality tests
        
    Returns:
    --------
    dict : Dictionary containing assumption check results
    """
    results = {}
    
    # 1. Check normality using Shapiro-Wilk test
    shapiro_x = stats.shapiro(x)
    shapiro_y = stats.shapiro(y)
    
    results['normality'] = {
        'x_normal': shapiro_x.pvalue >= alpha,
        'y_normal': shapiro_y.pvalue >= alpha,
        'x_p_value': shapiro_x.pvalue,
        'y_p_value': shapiro_y.pvalue,
        'both_normal': shapiro_x.pvalue >= alpha and shapiro_y.pvalue >= alpha
    }
    
    # 2. Check for outliers using IQR method
    def detect_outliers(data):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers
    
    outliers_x = detect_outliers(x)
    outliers_y = detect_outliers(y)
    
    results['outliers'] = {
        'x_outliers': np.sum(outliers_x),
        'y_outliers': np.sum(outliers_y),
        'total_outliers': np.sum(outliers_x | outliers_y),
        'outlier_percentage': np.mean(outliers_x | outliers_y) * 100
    }
    
    # 3. Check linearity using R-squared from linear regression
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    r_squared = model.score(x.reshape(-1, 1), y)
    
    results['linearity'] = {
        'r_squared': r_squared,
        'linear_relationship': r_squared > 0.1  # Arbitrary threshold
    }
    
    # 4. Check homoscedasticity using residuals
    fitted = model.predict(x.reshape(-1, 1))
    residuals = y - fitted
    
    # Simple homoscedasticity check (variance ratio)
    var_ratio = np.var(residuals[x < np.median(x)]) / np.var(residuals[x >= np.median(x)])
    homoscedastic = 0.5 < var_ratio < 2.0  # Arbitrary threshold
    
    results['homoscedasticity'] = {
        'variance_ratio': var_ratio,
        'homoscedastic': homoscedastic
    }
    
    # Overall assessment
    results['overall_assessment'] = {
        'pearson_appropriate': (results['normality']['both_normal'] and 
                              results['outliers']['outlier_percentage'] < 5 and
                              results['linearity']['linear_relationship'] and
                              results['homoscedasticity']['homoscedastic']),
        'recommendation': 'Use Pearson correlation' if results['overall_assessment']['pearson_appropriate'] 
                         else 'Consider Spearman or Kendall correlation'
    }
    
    return results

# --- Visualization Functions ---

def plot_scatter_with_correlation(x, y, correlation_result=None, title="Scatter Plot with Correlation"):
    """
    Create scatter plot with regression line and correlation information.
    
    Parameters:
    -----------
    x, y : array-like
        Variables to plot
    correlation_result : dict, optional
        Result from correlation analysis
    title : str
        Plot title
    """
    plt.figure(figsize=(8, 6))
    
    # Create scatter plot with regression line
    plot_data = pd.DataFrame({'x': x, 'y': y})
    sns.regplot(x='x', y='y', data=plot_data, ci=95, line_kws={'color': 'blue'})
    
    # Add correlation information to title
    if correlation_result:
        title += f"\nPearson r = {correlation_result['correlation']:.3f}"
        if correlation_result['significant']:
            title += " (p < 0.05)"
        else:
            title += " (p ≥ 0.05)"
    
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(cor_matrix, title="Correlation Matrix Heatmap"):
    """
    Create correlation matrix heatmap.
    
    Parameters:
    -----------
    cor_matrix : DataFrame
        Correlation matrix
    title : str
        Plot title
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cor_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', square=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_assumption_checks(x, y):
    """
    Create comprehensive assumption checking plots.
    
    Parameters:
    -----------
    x, y : array-like
        Variables to check
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Scatter plot for linearity and outliers
    axes[0, 0].scatter(x, y, alpha=0.7)
    slope, intercept = np.polyfit(x, y, 1)
    axes[0, 0].plot(x, slope * x + intercept, color='red')
    axes[0, 0].set_title("Scatter Plot for Linearity and Outliers")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    
    # 2. Q-Q plots for normality
    stats.probplot(x, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot for x")
    
    stats.probplot(y, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot for y")
    
    # 3. Residuals vs fitted for homoscedasticity
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    fitted = model.predict(x.reshape(-1, 1))
    residuals = y - fitted
    axes[1, 1].scatter(fitted, residuals)
    axes[1, 1].axhline(0, color='red')
    axes[1, 1].set_title("Residuals vs Fitted Values")
    axes[1, 1].set_xlabel("Fitted Values")
    axes[1, 1].set_ylabel("Residuals")
    
    plt.tight_layout()
    plt.show()

# --- Practical Examples ---

def height_weight_example():
    """
    Example: Height and weight correlation analysis.
    
    Returns:
    --------
    dict : Analysis results
    """
    # Simulate height and weight data
    np.random.seed(1)
    height = np.random.normal(170, 10, 100)
    weight = 0.5 * height + np.random.normal(0, 8, 100)
    
    # Calculate Pearson correlation
    pearson_result = calculate_pearson_correlation(height, weight)
    
    # Calculate Spearman and Kendall
    rank_results = calculate_spearman_kendall(height, weight)
    
    # Check assumptions
    assumptions = check_correlation_assumptions(height, weight)
    
    # Create visualizations
    plot_scatter_with_correlation(height, weight, pearson_result, 
                                 "Height vs Weight Correlation")
    plot_assumption_checks(height, weight)
    
    return {
        'pearson': pearson_result,
        'spearman_kendall': rank_results,
        'assumptions': assumptions,
        'data': {'height': height, 'weight': weight}
    }

def nonlinear_relationship_example():
    """
    Example: Nonlinear relationship showing difference between Pearson and Spearman.
    
    Returns:
    --------
    dict : Analysis results
    """
    # Simulate nonlinear data
    np.random.seed(2)
    x_nl = np.random.normal(size=100)
    y_nl = x_nl ** 2 + np.random.normal(size=100)
    
    # Calculate different correlations
    pearson_result = calculate_pearson_correlation(x_nl, y_nl)
    rank_results = calculate_spearman_kendall(x_nl, y_nl)
    
    # Create visualization
    plot_scatter_with_correlation(x_nl, y_nl, pearson_result, 
                                 "Nonlinear Relationship Example")
    
    return {
        'pearson': pearson_result,
        'spearman_kendall': rank_results,
        'data': {'x': x_nl, 'y': y_nl}
    }

def multivariate_correlation_example():
    """
    Example: Multivariate correlation matrix analysis.
    
    Returns:
    --------
    dict : Analysis results
    """
    # Generate multivariate data
    data = simulate_multivariate_data(n_samples=100, n_variables=3)
    
    # Calculate correlation matrices for different methods
    pearson_matrix = create_correlation_matrix(data, method='pearson')
    spearman_matrix = create_correlation_matrix(data, method='spearman')
    
    # Create visualizations
    plot_correlation_matrix(pearson_matrix['correlation_matrix'], 
                           "Pearson Correlation Matrix")
    plot_correlation_matrix(spearman_matrix['correlation_matrix'], 
                           "Spearman Correlation Matrix")
    
    return {
        'pearson_matrix': pearson_matrix,
        'spearman_matrix': spearman_matrix,
        'data': data
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    """
    Main execution block demonstrating correlation analysis workflow.
    This shows how to use all the functions from the lesson.
    """
    print("=== CORRELATION ANALYSIS DEMONSTRATION ===\n")
    
    # 1. Generate sample data
    print("1. Generating sample data...")
    x, y = simulate_correlation_data(seed=42, n_samples=100, correlation=0.7)
    print(f"   Created dataset with {len(x)} observations")
    print(f"   Target correlation: 0.7\n")
    
    # 2. Calculate Pearson correlation
    print("2. Pearson correlation analysis...")
    pearson_result = calculate_pearson_correlation(x, y)
    print(f"   Correlation: {pearson_result['correlation']:.3f}")
    print(f"   p-value: {pearson_result['p_value']:.4f}")
    print(f"   95% CI: [{pearson_result['ci_lower']:.3f}, {pearson_result['ci_upper']:.3f}]")
    print(f"   Effect size: {pearson_result['effect_size']}")
    print(f"   Significant: {pearson_result['significant']}\n")
    
    # 3. Calculate Spearman and Kendall correlations
    print("3. Spearman and Kendall correlations...")
    rank_results = calculate_spearman_kendall(x, y)
    print(f"   Spearman: {rank_results['spearman']['correlation']:.3f} (p={rank_results['spearman']['p_value']:.4f})")
    print(f"   Kendall: {rank_results['kendall']['correlation']:.3f} (p={rank_results['kendall']['p_value']:.4f})\n")
    
    # 4. Check assumptions
    print("4. Checking assumptions...")
    assumptions = check_correlation_assumptions(x, y)
    print(f"   Normality (both variables): {assumptions['normality']['both_normal']}")
    print(f"   Outliers: {assumptions['outliers']['total_outliers']} detected")
    print(f"   Linearity (R²): {assumptions['linearity']['r_squared']:.3f}")
    print(f"   Homoscedasticity: {assumptions['homoscedasticity']['homoscedastic']}")
    print(f"   Recommendation: {assumptions['overall_assessment']['recommendation']}\n")
    
    # 5. Create visualizations
    print("5. Creating visualizations...")
    plot_scatter_with_correlation(x, y, pearson_result)
    plot_assumption_checks(x, y)
    print("   Visualizations created\n")
    
    # 6. Multivariate correlation example
    print("6. Multivariate correlation analysis...")
    multivariate_data = simulate_multivariate_data()
    cor_matrix_result = create_correlation_matrix(multivariate_data, method='pearson')
    print("   Correlation matrix calculated")
    plot_correlation_matrix(cor_matrix_result['correlation_matrix'])
    print("   Correlation matrix heatmap created\n")
    
    # 7. Practical examples
    print("7. Running practical examples...")
    height_weight = height_weight_example()
    nonlinear = nonlinear_relationship_example()
    multivariate = multivariate_correlation_example()
    print("   All practical examples completed\n")
    
    print("=== DEMONSTRATION COMPLETE ===")
    print("\nAll functions from the correlation analysis lesson have been demonstrated.")
    print("Refer to the markdown file for detailed explanations and theory.") 