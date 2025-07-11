"""
Two-Way ANOVA Analysis in Python

This module provides comprehensive tools for performing two-way Analysis of Variance (ANOVA)
including manual calculations, assumption checking, effect size analysis, simple effects,
post hoc tests, and practical examples.

Author: Statistics Course
Date: 2024
"""

# --- Imports ---
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f, shapiro, anderson, ks_1samp, levene, bartlett, fligner, norm, probplot
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Preparation ---
def load_sample_data():
    """Load and prepare sample data for two-way ANOVA analysis."""
    # Load sample data (using seaborn's mpg dataset as a proxy for mtcars)
    mtcars = sns.load_dataset('mpg').dropna(subset=['mpg', 'cylinders', 'origin'])
    
    # Create factors for analysis
    mtcars['cyl_factor'] = pd.Categorical(mtcars['cylinders'], categories=[4, 6, 8], ordered=True)
    mtcars['cyl_factor'] = mtcars['cyl_factor'].map({4: '4-cylinder', 6: '6-cylinder', 8: '8-cylinder'})
    mtcars['am_factor'] = pd.Categorical(mtcars['origin'], categories=['usa', 'europe', 'japan'])
    
    return mtcars

# --- Manual Two-Way ANOVA Calculation ---
def manual_two_way_anova(data, factor1, factor2, response):
    """
    Perform manual two-way ANOVA calculation for educational purposes.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    factor1 : str
        Name of first factor variable
    factor2 : str
        Name of second factor variable
    response : str
        Name of response variable
    
    Returns:
    --------
    dict : Dictionary containing all ANOVA results
    """
    # Drop missing
    data = data.dropna(subset=[factor1, factor2, response])
    overall_mean = data[response].mean()
    factor1_levels = data[factor1].unique()
    factor2_levels = data[factor2].unique()
    n_total = len(data)
    n_factor1_levels = len(factor1_levels)
    n_factor2_levels = len(factor2_levels)

    # Cell means
    cell_means = data.pivot_table(index=factor1, columns=factor2, values=response, aggfunc='mean')
    factor1_means = data.groupby(factor1)[response].mean()
    factor2_means = data.groupby(factor2)[response].mean()

    # SS Total
    ss_total = ((data[response] - overall_mean) ** 2).sum()
    # SS Factor 1
    ss_factor1 = sum(data[factor1].value_counts()[lvl] * (factor1_means[lvl] - overall_mean) ** 2 for lvl in factor1_levels)
    # SS Factor 2
    ss_factor2 = sum(data[factor2].value_counts()[lvl] * (factor2_means[lvl] - overall_mean) ** 2 for lvl in factor2_levels)
    # SS Interaction
    ss_interaction = 0
    for i in factor1_levels:
        for j in factor2_levels:
            n_cell = len(data[(data[factor1] == i) & (data[factor2] == j)])
            if n_cell == 0:
                continue
            cell_mean = cell_means.loc[i, j]
            interaction_effect = cell_mean - factor1_means[i] - factor2_means[j] + overall_mean
            ss_interaction += n_cell * interaction_effect ** 2
    # SS Error
    ss_error = ss_total - ss_factor1 - ss_factor2 - ss_interaction

    # Degrees of freedom
    df_factor1 = n_factor1_levels - 1
    df_factor2 = n_factor2_levels - 1
    df_interaction = (n_factor1_levels - 1) * (n_factor2_levels - 1)
    df_error = n_total - n_factor1_levels * n_factor2_levels
    df_total = n_total - 1

    # Mean Squares
    ms_factor1 = ss_factor1 / df_factor1
    ms_factor2 = ss_factor2 / df_factor2
    ms_interaction = ss_interaction / df_interaction
    ms_error = ss_error / df_error

    # F-statistics
    f_factor1 = ms_factor1 / ms_error
    f_factor2 = ms_factor2 / ms_error
    f_interaction = ms_interaction / ms_error

    # p-values
    p_factor1 = 1 - f.cdf(f_factor1, df_factor1, df_error)
    p_factor2 = 1 - f.cdf(f_factor2, df_factor2, df_error)
    p_interaction = 1 - f.cdf(f_interaction, df_interaction, df_error)

    # Effect sizes (partial eta-squared)
    partial_eta2_factor1 = ss_factor1 / (ss_factor1 + ss_error)
    partial_eta2_factor2 = ss_factor2 / (ss_factor2 + ss_error)
    partial_eta2_interaction = ss_interaction / (ss_interaction + ss_error)

    return {
        'ss_factor1': ss_factor1,
        'ss_factor2': ss_factor2,
        'ss_interaction': ss_interaction,
        'ss_error': ss_error,
        'ss_total': ss_total,
        'df_factor1': df_factor1,
        'df_factor2': df_factor2,
        'df_interaction': df_interaction,
        'df_error': df_error,
        'df_total': df_total,
        'ms_factor1': ms_factor1,
        'ms_factor2': ms_factor2,
        'ms_interaction': ms_interaction,
        'ms_error': ms_error,
        'f_factor1': f_factor1,
        'f_factor2': f_factor2,
        'f_interaction': f_interaction,
        'p_factor1': p_factor1,
        'p_factor2': p_factor2,
        'p_interaction': p_interaction,
        'partial_eta2_factor1': partial_eta2_factor1,
        'partial_eta2_factor2': partial_eta2_factor2,
        'partial_eta2_interaction': partial_eta2_interaction
    } 

# --- Built-in Two-Way ANOVA ---
def builtin_two_way_anova(data, factor1, factor2, response):
    """
    Perform two-way ANOVA using statsmodels.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    factor1 : str
        Name of first factor variable
    factor2 : str
        Name of second factor variable
    response : str
        Name of response variable
    
    Returns:
    --------
    tuple : (model, anova_table, key_statistics)
    """
    # Perform two-way ANOVA using statsmodels
    model = ols(f'{response} ~ C({factor1}) * C({factor2})', data=data).fit()
    two_way_anova = anova_lm(model, typ=2)
    
    # Extract key statistics
    f_factor1 = two_way_anova.loc[f'C({factor1})', 'F']
    f_factor2 = two_way_anova.loc[f'C({factor2})', 'F']
    f_interaction = two_way_anova.loc[f'C({factor1}):C({factor2})', 'F']
    p_factor1 = two_way_anova.loc[f'C({factor1})', 'PR(>F)']
    p_factor2 = two_way_anova.loc[f'C({factor2})', 'PR(>F)']
    p_interaction = two_way_anova.loc[f'C({factor1}):C({factor2})', 'PR(>F)']
    
    key_stats = {
        'f_factor1': f_factor1,
        'f_factor2': f_factor2,
        'f_interaction': f_interaction,
        'p_factor1': p_factor1,
        'p_factor2': p_factor2,
        'p_interaction': p_interaction
    }
    
    return model, two_way_anova, key_stats

# --- Descriptive Statistics ---
def calculate_descriptive_stats(data, factor1, factor2, response):
    """
    Calculate comprehensive descriptive statistics for two-way ANOVA.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    factor1 : str
        Name of first factor variable
    factor2 : str
        Name of second factor variable
    response : str
        Name of response variable
    
    Returns:
    --------
    dict : Dictionary containing cell means, marginal means, and grand mean
    """
    # Cell means - these represent the mean for each combination of factor levels
    cell_means = data.groupby([factor1, factor2]).agg({
        response: ['count', 'mean', 'std']
    }).round(3)
    cell_means.columns = ['n', 'mean', 'sd']
    cell_means = cell_means.reset_index()

    # Calculate standard error and confidence intervals
    cell_means['se'] = cell_means['sd'] / np.sqrt(cell_means['n'])
    cell_means['ci_lower'] = cell_means['mean'] - stats.t.ppf(0.975, cell_means['n']-1) * cell_means['se']
    cell_means['ci_upper'] = cell_means['mean'] + stats.t.ppf(0.975, cell_means['n']-1) * cell_means['se']

    # Marginal means for Factor 1 - averaged across factor2 levels
    marginal_factor1 = data.groupby(factor1).agg({
        response: ['count', 'mean', 'std']
    }).round(3)
    marginal_factor1.columns = ['n', 'mean', 'sd']
    marginal_factor1 = marginal_factor1.reset_index()

    marginal_factor1['se'] = marginal_factor1['sd'] / np.sqrt(marginal_factor1['n'])
    marginal_factor1['ci_lower'] = marginal_factor1['mean'] - stats.t.ppf(0.975, marginal_factor1['n']-1) * marginal_factor1['se']
    marginal_factor1['ci_upper'] = marginal_factor1['mean'] + stats.t.ppf(0.975, marginal_factor1['n']-1) * marginal_factor1['se']

    # Marginal means for Factor 2 - averaged across factor1 levels
    marginal_factor2 = data.groupby(factor2).agg({
        response: ['count', 'mean', 'std']
    }).round(3)
    marginal_factor2.columns = ['n', 'mean', 'sd']
    marginal_factor2 = marginal_factor2.reset_index()

    marginal_factor2['se'] = marginal_factor2['sd'] / np.sqrt(marginal_factor2['n'])
    marginal_factor2['ci_lower'] = marginal_factor2['mean'] - stats.t.ppf(0.975, marginal_factor2['n']-1) * marginal_factor2['se']
    marginal_factor2['ci_upper'] = marginal_factor2['mean'] + stats.t.ppf(0.975, marginal_factor2['n']-1) * marginal_factor2['se']

    # Grand mean
    grand_mean = data[response].mean()
    
    return {
        'cell_means': cell_means,
        'marginal_factor1': marginal_factor1,
        'marginal_factor2': marginal_factor2,
        'grand_mean': grand_mean
    }

def calculate_interaction_effects(cell_means, marginal_factor1, marginal_factor2, grand_mean, factor1_name, factor2_name):
    """
    Calculate interaction effects matrix.
    
    Parameters:
    -----------
    cell_means : pandas.DataFrame
        Cell means data
    marginal_factor1 : pandas.DataFrame
        Marginal means for factor1
    marginal_factor2 : pandas.DataFrame
        Marginal means for factor2
    grand_mean : float
        Grand mean
    factor1_name : str
        Name of factor1 column
    factor2_name : str
        Name of factor2 column
    
    Returns:
    --------
    numpy.ndarray : Interaction effects matrix
    """
    # Create a matrix of interaction effects
    interaction_matrix = np.zeros((len(marginal_factor1), len(marginal_factor2)))
    
    for i in range(len(marginal_factor1)):
        for j in range(len(marginal_factor2)):
            # Find the cell mean for this combination
            cell_data = cell_means[
                (cell_means[factor1_name] == marginal_factor1.iloc[i][factor1_name]) & 
                (cell_means[factor2_name] == marginal_factor2.iloc[j][factor2_name])
            ]
            
            if len(cell_data) > 0:
                cell_mean = cell_data.iloc[0]['mean']
                factor1_mean = marginal_factor1.iloc[i]['mean']
                factor2_mean = marginal_factor2.iloc[j]['mean']
                
                interaction_effect = cell_mean - factor1_mean - factor2_mean + grand_mean
                interaction_matrix[i, j] = interaction_effect
    
    return interaction_matrix

# --- Visualization Functions ---
def create_interaction_plot(data, factor1, factor2, response, title="Interaction Plot"):
    """Create interaction plot for two-way ANOVA."""
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=data, x=factor1, y=response, hue=factor2, 
                  capsize=0.1, markers=['o', 's', '^'], markersize=8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(factor1)
    plt.ylabel(response)
    plt.legend(title=factor2)
    plt.grid(True, alpha=0.3)
    plt.show()

def create_box_plot(data, factor1, factor2, response, title="Box Plot with Individual Points"):
    """Create box plot with individual points for two-way ANOVA."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=factor1, y=response, hue=factor2, alpha=0.7)
    sns.stripplot(data=data, x=factor1, y=response, hue=factor2, 
                  dodge=True, alpha=0.6, size=3, legend=False)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(factor1)
    plt.ylabel(response)
    plt.legend(title=factor2)
    plt.grid(True, alpha=0.3)
    plt.show()

def create_heatmap(cell_means, factor1, factor2, title="Cell Means Heatmap"):
    """Create heatmap of cell means."""
    plt.figure(figsize=(8, 6))
    pivot_means = cell_means.pivot(index=factor2, columns=factor1, values='mean')
    sns.heatmap(pivot_means, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Mean'}, square=True)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(factor1)
    plt.ylabel(factor2)
    plt.show()

def create_marginal_effects_plot(marginal_factor1, marginal_factor2, factor1, factor2, response):
    """Create marginal effects plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Marginal effects for factor1
    ax1.errorbar(marginal_factor1[factor1], marginal_factor1['mean'], 
                 yerr=marginal_factor1['se'], marker='o', capsize=5, capthick=2)
    ax1.set_title(f'Marginal Effects - {factor1}', fontweight='bold')
    ax1.set_xlabel(factor1)
    ax1.set_ylabel(f'Mean {response}')
    ax1.grid(True, alpha=0.3)

    # Marginal effects for factor2
    ax2.errorbar(marginal_factor2[factor2], marginal_factor2['mean'], 
                 yerr=marginal_factor2['se'], marker='s', capsize=5, capthick=2)
    ax2.set_title(f'Marginal Effects - {factor2}', fontweight='bold')
    ax2.set_xlabel(factor2)
    ax2.set_ylabel(f'Mean {response}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def create_residuals_plot(model, title="Residuals vs Fitted Values"):
    """Create residuals plot for model diagnostics."""
    model_residuals = model.resid
    fitted_values = model.fittedvalues

    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_values, model_residuals, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show() 

# --- Effect Size Analysis ---
def calculate_two_way_effect_sizes(anova_result, data, factor1, factor2, response):
    """
    Calculate comprehensive effect sizes for two-way ANOVA.
    
    Parameters:
    -----------
    anova_result : dict
        Results from manual ANOVA calculation
    data : pandas.DataFrame
        Input data
    factor1 : str
        Name of first factor variable
    factor2 : str
        Name of second factor variable
    response : str
        Name of response variable
    
    Returns:
    --------
    dict : Dictionary containing all effect size measures
    """
    # Extract Sum of Squares
    ss_factor1 = anova_result['ss_factor1']
    ss_factor2 = anova_result['ss_factor2']
    ss_interaction = anova_result['ss_interaction']
    ss_error = anova_result['ss_error']
    ss_total = anova_result['ss_total']
    
    # Degrees of freedom
    df_factor1 = anova_result['df_factor1']
    df_factor2 = anova_result['df_factor2']
    df_interaction = anova_result['df_interaction']
    df_error = anova_result['df_error']
    
    # Mean Squares
    ms_error = anova_result['ms_error']
    
    # Partial eta-squared
    partial_eta2_factor1 = ss_factor1 / (ss_factor1 + ss_error)
    partial_eta2_factor2 = ss_factor2 / (ss_factor2 + ss_error)
    partial_eta2_interaction = ss_interaction / (ss_interaction + ss_error)
    
    # Eta-squared (total)
    eta2_factor1 = ss_factor1 / ss_total
    eta2_factor2 = ss_factor2 / ss_total
    eta2_interaction = ss_interaction / ss_total
    
    # Omega-squared (unbiased)
    omega2_factor1 = (ss_factor1 - df_factor1 * ms_error) / (ss_total + ms_error)
    omega2_factor2 = (ss_factor2 - df_factor2 * ms_error) / (ss_total + ms_error)
    omega2_interaction = (ss_interaction - df_interaction * ms_error) / (ss_total + ms_error)
    
    # Cohen's f (for power analysis)
    f_factor1 = np.sqrt(partial_eta2_factor1 / (1 - partial_eta2_factor1))
    f_factor2 = np.sqrt(partial_eta2_factor2 / (1 - partial_eta2_factor2))
    f_interaction = np.sqrt(partial_eta2_interaction / (1 - partial_eta2_interaction))
    
    # Bootstrap confidence intervals for partial eta-squared
    def bootstrap_effect_size(data, indices, factor1, factor2, response):
        d = data.iloc[indices]
        model = ols(f'{response} ~ C({factor1}) * C({factor2})', data=d).fit()
        anova_table = anova_lm(model, typ=2)
        
        ss_factor1 = anova_table.loc[f'C({factor1})', 'sum_sq']
        ss_factor2 = anova_table.loc[f'C({factor2})', 'sum_sq']
        ss_interaction = anova_table.loc[f'C({factor1}):C({factor2})', 'sum_sq']
        ss_error = anova_table.loc['Residual', 'sum_sq']
        
        partial_eta2_factor1 = ss_factor1 / (ss_factor1 + ss_error)
        partial_eta2_factor2 = ss_factor2 / (ss_factor2 + ss_error)
        partial_eta2_interaction = ss_interaction / (ss_interaction + ss_error)
        
        return [partial_eta2_factor1, partial_eta2_factor2, partial_eta2_interaction]
    
    # Bootstrap for confidence intervals
    from scipy.stats import bootstrap
    bootstrap_result = bootstrap((data,), bootstrap_effect_size, n_resamples=1000,
                               args=(factor1, factor2, response))
    
    ci_factor1 = (np.percentile(bootstrap_result.bootstrap_distribution[:, 0], 2.5),
                  np.percentile(bootstrap_result.bootstrap_distribution[:, 0], 97.5))
    ci_factor2 = (np.percentile(bootstrap_result.bootstrap_distribution[:, 1], 2.5),
                  np.percentile(bootstrap_result.bootstrap_distribution[:, 1], 97.5))
    ci_interaction = (np.percentile(bootstrap_result.bootstrap_distribution[:, 2], 2.5),
                      np.percentile(bootstrap_result.bootstrap_distribution[:, 2], 97.5))
    
    return {
        'partial_eta2_factor1': partial_eta2_factor1,
        'partial_eta2_factor2': partial_eta2_factor2,
        'partial_eta2_interaction': partial_eta2_interaction,
        'eta2_factor1': eta2_factor1,
        'eta2_factor2': eta2_factor2,
        'eta2_interaction': eta2_interaction,
        'omega2_factor1': omega2_factor1,
        'omega2_factor2': omega2_factor2,
        'omega2_interaction': omega2_interaction,
        'f_factor1': f_factor1,
        'f_factor2': f_factor2,
        'f_interaction': f_interaction,
        'ci_factor1': ci_factor1,
        'ci_factor2': ci_factor2,
        'ci_interaction': ci_interaction
    }

def interpret_effect_size(eta_sq, measure="partial_eta2"):
    """
    Interpret effect size values.
    
    Parameters:
    -----------
    eta_sq : float
        Effect size value
    measure : str
        Type of effect size measure
    
    Returns:
    --------
    str : Interpretation of effect size
    """
    if measure == "partial_eta2" or measure == "eta2":
        if eta_sq < 0.01:
            return "Negligible effect (< 1% of variance explained)"
        elif eta_sq < 0.06:
            return "Small effect (1-6% of variance explained)"
        elif eta_sq < 0.14:
            return "Medium effect (6-14% of variance explained)"
        else:
            return "Large effect (> 14% of variance explained)"
    elif measure == "f":
        if eta_sq < 0.10:
            return "Small effect"
        elif eta_sq < 0.25:
            return "Medium effect"
        elif eta_sq < 0.40:
            return "Large effect"
        else:
            return "Very large effect"

def create_effect_size_plot(effect_sizes, title="Effect Size Comparison"):
    """Create effect size comparison plot."""
    effect_size_data = pd.DataFrame({
        'Effect': ['Factor 1', 'Factor 2', 'Interaction'] * 3,
        'Measure': ['Partial η²'] * 3 + ['η²'] * 3 + ['ω²'] * 3,
        'Value': [effect_sizes['partial_eta2_factor1'], effect_sizes['partial_eta2_factor2'], effect_sizes['partial_eta2_interaction'],
                  effect_sizes['eta2_factor1'], effect_sizes['eta2_factor2'], effect_sizes['eta2_interaction'],
                  effect_sizes['omega2_factor1'], effect_sizes['omega2_factor2'], effect_sizes['omega2_interaction']]
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=effect_size_data, x='Effect', y='Value', hue='Measure', alpha=0.8)
    plt.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Negligible')
    plt.axhline(y=0.06, color='red', linestyle='--', alpha=0.7, label='Small')
    plt.axhline(y=0.14, color='red', linestyle='--', alpha=0.7, label='Medium')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Effect')
    plt.ylabel('Effect Size')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Measure')
    plt.tight_layout()
    plt.show() 

# --- Simple Effects Analysis ---
def simple_effects_analysis(data, factor1, factor2, response, alpha=0.05):
    """
    Perform comprehensive simple effects analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    factor1 : str
        Name of first factor variable
    factor2 : str
        Name of second factor variable
    response : str
        Name of response variable
    alpha : float
        Significance level
    
    Returns:
    --------
    dict : Dictionary containing simple effects results
    """
    results = {}
    
    # Simple effects of Factor 1 at each level of Factor 2
    factor2_levels = data[factor2].unique()
    
    for level in factor2_levels:
        subset_data = data[data[factor2] == level]
        
        if len(subset_data[factor1].unique()) > 1:
            # Perform one-way ANOVA
            simple_model = ols(f'{response} ~ C({factor1})', data=subset_data).fit()
            simple_anova = anova_lm(simple_model, typ=2)
            
            f_stat = simple_anova.loc[f'C({factor1})', 'F']
            p_value = simple_anova.loc[f'C({factor1})', 'PR(>F)']
            df_factor = simple_anova.loc[f'C({factor1})', 'df']
            df_error = simple_anova.loc['Residual', 'df']
            
            # Calculate effect size
            ss_factor = simple_anova.loc[f'C({factor1})', 'sum_sq']
            ss_error = simple_anova.loc['Residual', 'sum_sq']
            partial_eta2 = ss_factor / (ss_factor + ss_error)
            
            # Calculate descriptive statistics
            desc_stats = subset_data.groupby(factor1).agg({
                response: ['count', 'mean', 'std']
            }).round(3)
            desc_stats.columns = ['n', 'mean', 'sd']
            desc_stats = desc_stats.reset_index()
            desc_stats['se'] = desc_stats['sd'] / np.sqrt(desc_stats['n'])
            
            # Store results
            results[f"{factor1}_at_{factor2}_{level}"] = {
                'f_stat': f_stat,
                'p_value': p_value,
                'partial_eta2': partial_eta2,
                'df_factor': df_factor,
                'df_error': df_error,
                'desc_stats': desc_stats,
                'significant': p_value < alpha
            }
    
    # Simple effects of Factor 2 at each level of Factor 1
    factor1_levels = data[factor1].unique()
    
    for level in factor1_levels:
        subset_data = data[data[factor1] == level]
        
        if len(subset_data[factor2].unique()) > 1:
            # Perform one-way ANOVA
            simple_model = ols(f'{response} ~ C({factor2})', data=subset_data).fit()
            simple_anova = anova_lm(simple_model, typ=2)
            
            f_stat = simple_anova.loc[f'C({factor2})', 'F']
            p_value = simple_anova.loc[f'C({factor2})', 'PR(>F)']
            df_factor = simple_anova.loc[f'C({factor2})', 'df']
            df_error = simple_anova.loc['Residual', 'df']
            
            # Calculate effect size
            ss_factor = simple_anova.loc[f'C({factor2})', 'sum_sq']
            ss_error = simple_anova.loc['Residual', 'sum_sq']
            partial_eta2 = ss_factor / (ss_factor + ss_error)
            
            # Calculate descriptive statistics
            desc_stats = subset_data.groupby(factor2).agg({
                response: ['count', 'mean', 'std']
            }).round(3)
            desc_stats.columns = ['n', 'mean', 'sd']
            desc_stats = desc_stats.reset_index()
            desc_stats['se'] = desc_stats['sd'] / np.sqrt(desc_stats['n'])
            
            # Store results
            results[f"{factor2}_at_{factor1}_{level}"] = {
                'f_stat': f_stat,
                'p_value': p_value,
                'partial_eta2': partial_eta2,
                'df_factor': df_factor,
                'df_error': df_error,
                'desc_stats': desc_stats,
                'significant': p_value < alpha
            }
    
    return results

def plot_simple_effects(data, factor1, factor2, response):
    """Create visualization for simple effects."""
    factor2_levels = data[factor2].unique()
    
    fig, axes = plt.subplots(1, len(factor2_levels), figsize=(5*len(factor2_levels), 5))
    if len(factor2_levels) == 1:
        axes = [axes]
    
    for i, level in enumerate(factor2_levels):
        subset_data = data[data[factor2] == level]
        
        # Box plot with individual points
        sns.boxplot(data=subset_data, x=factor1, y=response, ax=axes[i], alpha=0.7, color='lightblue')
        sns.stripplot(data=subset_data, x=factor1, y=response, ax=axes[i], alpha=0.6, size=3, color='black')
        
        # Add mean points
        means = subset_data.groupby(factor1)[response].mean()
        for j, (group, mean_val) in enumerate(means.items()):
            axes[i].plot(j, mean_val, 'ro', markersize=8, markeredgecolor='red', markeredgewidth=2)
        
        axes[i].set_title(f'Simple Effect of {factor1} at {factor2} = {level}', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel(factor1)
        axes[i].set_ylabel(response)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# --- Assumption Checking ---
def check_normality_two_way(data, factor1, factor2, response, alpha=0.05):
    """
    Check normality assumption for two-way ANOVA.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    factor1 : str
        Name of first factor variable
    factor2 : str
        Name of second factor variable
    response : str
        Name of response variable
    alpha : float
        Significance level
    
    Returns:
    --------
    dict : Dictionary containing normality test results
    """
    # Fit the model
    model = ols(f'{response} ~ C({factor1}) * C({factor2})', data=data).fit()
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    # Multiple normality tests
    tests = {}
    
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = shapiro(residuals)
    tests['shapiro'] = {'statistic': shapiro_stat, 'pvalue': shapiro_p}
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = ks_1samp(residuals, norm.cdf, args=(np.mean(residuals), np.std(residuals)))
    tests['ks'] = {'statistic': ks_stat, 'pvalue': ks_p}
    
    # Anderson-Darling test
    ad_result = anderson(residuals)
    tests['ad'] = {'statistic': ad_result.statistic, 'pvalue': ad_result.significance_level[2]/100}
    
    # Summary of normality tests
    p_values = [shapiro_p, ks_p, ad_result.significance_level[2]/100]
    test_names = ["Shapiro-Wilk", "Kolmogorov-Smirnov", "Anderson-Darling"]
    
    normal_tests = sum(p >= alpha for p in p_values)
    total_tests = len(p_values)
    
    # Create diagnostic plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Q-Q plot
    probplot(residuals, dist="norm", plot=ax1)
    ax1.set_title("Q-Q Plot of Residuals", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram with normal curve
    ax2.hist(residuals, bins=15, density=True, alpha=0.7, color='steelblue')
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x, norm.pdf(x, np.mean(residuals), np.std(residuals)), 'r-', linewidth=2)
    ax2.set_title("Histogram of Residuals with Normal Curve", fontweight='bold')
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Density")
    ax2.grid(True, alpha=0.3)
    
    # Residuals vs fitted values
    ax3.scatter(fitted_values, residuals, alpha=0.7)
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_title("Residuals vs Fitted Values", fontweight='bold')
    ax3.set_xlabel("Fitted Values")
    ax3.set_ylabel("Residuals")
    ax3.grid(True, alpha=0.3)
    
    # Remove the fourth subplot
    ax4.remove()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'tests': tests,
        'p_values': p_values,
        'test_names': test_names,
        'normal_tests': normal_tests,
        'total_tests': total_tests
    }

def check_homogeneity_two_way(data, factor1, factor2, response, alpha=0.05):
    """
    Check homogeneity of variance assumption for two-way ANOVA.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    factor1 : str
        Name of first factor variable
    factor2 : str
        Name of second factor variable
    response : str
        Name of response variable
    alpha : float
        Significance level
    
    Returns:
    --------
    dict : Dictionary containing homogeneity test results
    """
    # Create interaction factor for testing
    data['interaction_factor'] = data[factor1].astype(str) + ':' + data[factor2].astype(str)
    
    # Multiple homogeneity tests
    tests = {}
    
    # Prepare groups for testing
    groups = [group[response].values for name, group in data.groupby('interaction_factor')]
    
    # Levene's test (most robust)
    levene_stat, levene_p = levene(*groups)
    tests['levene'] = {'statistic': levene_stat, 'pvalue': levene_p}
    
    # Bartlett's test (sensitive to non-normality)
    bartlett_stat, bartlett_p = bartlett(*groups)
    tests['bartlett'] = {'statistic': bartlett_stat, 'pvalue': bartlett_p}
    
    # Fligner-Killeen test (robust)
    fligner_stat, fligner_p = fligner(*groups)
    tests['fligner'] = {'statistic': fligner_stat, 'pvalue': fligner_p}
    
    # Summary of homogeneity tests
    p_values = [levene_p, bartlett_p, fligner_p]
    test_names = ["Levene's", "Bartlett's", "Fligner-Killeen"]
    
    equal_var_tests = sum(p >= alpha for p in p_values)
    total_tests = len(p_values)
    
    # Calculate and display group variances
    group_vars = data.groupby([factor1, factor2]).agg({
        response: ['count', 'var', 'std']
    }).round(3)
    group_vars.columns = ['n', 'variance', 'sd']
    group_vars = group_vars.reset_index()
    
    # Calculate variance ratio (largest/smallest)
    max_var = group_vars['variance'].max()
    min_var = group_vars['variance'].min()
    var_ratio = max_var / min_var
    
    # Create diagnostic plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot to visualize variance differences
    sns.boxplot(data=data, x='interaction_factor', y=response, hue=factor1, ax=ax1, alpha=0.7)
    ax1.set_title("Box Plot by Group", fontweight='bold')
    ax1.set_xlabel("Group")
    ax1.set_ylabel(response)
    ax1.tick_params(axis='x', rotation=45)
    
    # Variance vs mean plot
    group_means = data.groupby('interaction_factor')[response].mean()
    group_vars_dict = dict(zip(group_vars[factor1] + ':' + group_vars[factor2], group_vars['variance']))
    
    means = [group_means[group] for group in group_means.index]
    variances = [group_vars_dict[group] for group in group_means.index]
    
    ax2.scatter(means, variances, s=50, alpha=0.7)
    ax2.set_title("Variance vs Mean Plot", fontweight='bold')
    ax2.set_xlabel("Group Mean")
    ax2.set_ylabel("Group Variance")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'tests': tests,
        'p_values': p_values,
        'test_names': test_names,
        'equal_var_tests': equal_var_tests,
        'total_tests': total_tests,
        'group_vars': group_vars,
        'var_ratio': var_ratio
    } 

# --- Post Hoc Tests ---
def perform_posthoc_tests(data, factor1, factor2, response, key_stats):
    """
    Perform post hoc tests for significant effects.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    factor1 : str
        Name of first factor variable
    factor2 : str
        Name of second factor variable
    response : str
        Name of response variable
    key_stats : dict
        Key statistics from ANOVA
    
    Returns:
    --------
    dict : Dictionary containing post hoc test results
    """
    results = {}
    
    # Post hoc for Factor 1 if significant
    if key_stats['p_factor1'] < 0.05:
        cyl_posthoc = pairwise_tukeyhsd(data[response], data[factor1])
        results['factor1_posthoc'] = cyl_posthoc
    
    # Post hoc for Factor 2 if significant
    if key_stats['p_factor2'] < 0.05:
        # Since factor2 has only 2 levels, no post hoc needed
        results['factor2_posthoc'] = "Only 2 levels - no post hoc tests needed"
    
    # Post hoc for Interaction Effects if significant
    if key_stats['p_interaction'] < 0.05:
        # Create interaction factor for cell comparisons
        data['interaction_factor'] = data[factor1] + ':' + data[factor2]
        
        # Pairwise comparisons for all cells
        cell_posthoc = pairwise_tukeyhsd(data[response], data['interaction_factor'])
        results['interaction_posthoc'] = cell_posthoc
    
    return results

def simple_effects_posthoc(simple_effects_results, data, factor1, factor2, response):
    """
    Perform post hoc tests for significant simple effects.
    
    Parameters:
    -----------
    simple_effects_results : dict
        Results from simple effects analysis
    data : pandas.DataFrame
        Input data
    factor1 : str
        Name of first factor variable
    factor2 : str
        Name of second factor variable
    response : str
        Name of response variable
    
    Returns:
    --------
    dict : Dictionary containing post hoc test results
    """
    results = {}
    
    for effect_name, result in simple_effects_results.items():
        if result['significant']:
            # Extract factor information from effect name
            if factor1 + "_at_" in effect_name:
                # Simple effect of factor1 at a level of factor2
                level = effect_name.replace(f"{factor1}_at_{factor2}_", "")
                subset_data = data[data[factor2] == level]
                
                if len(subset_data[factor1].unique()) > 2:
                    # Perform Tukey's HSD
                    tukey_result = pairwise_tukeyhsd(subset_data[response], subset_data[factor1])
                    results[effect_name] = tukey_result
    
    return results

# --- Practical Examples ---
def create_educational_example():
    """Create educational intervention example data."""
    np.random.seed(123)
    n_per_cell = 15

    # Generate data for 2x3 factorial design
    # Factor 1: Teaching Method (A, B)
    # Factor 2: Class Size (Small, Medium, Large)

    teaching_method = ["Method A"] * (n_per_cell * 3) + ["Method B"] * (n_per_cell * 3)
    class_size = ["Small"] * n_per_cell + ["Medium"] * n_per_cell + ["Large"] * n_per_cell
    class_size = class_size * 2

    # Generate scores with interaction effects
    scores = []
    for i in range(len(teaching_method)):
        if teaching_method[i] == "Method A":
            if class_size[i] == "Small":
                scores.append(np.random.normal(85, 8))
            elif class_size[i] == "Medium":
                scores.append(np.random.normal(80, 10))
            else:
                scores.append(np.random.normal(75, 12))
        else:
            if class_size[i] == "Small":
                scores.append(np.random.normal(82, 9))
            elif class_size[i] == "Medium":
                scores.append(np.random.normal(85, 8))
            else:
                scores.append(np.random.normal(88, 7))

    # Create data frame
    education_data = pd.DataFrame({
        'score': scores,
        'method': teaching_method,
        'class_size': class_size
    })
    
    return education_data

def create_clinical_example():
    """Create clinical trial example data."""
    np.random.seed(123)
    n_per_cell = 20

    # Generate data for 2x2 factorial design
    # Factor 1: Treatment (Drug A, Drug B)
    # Factor 2: Dosage (Low, High)

    treatment = ["Drug A"] * (n_per_cell * 2) + ["Drug B"] * (n_per_cell * 2)
    dosage = ["Low"] * n_per_cell + ["High"] * n_per_cell
    dosage = dosage * 2

    # Generate outcomes with interaction
    outcomes = []
    for i in range(len(treatment)):
        if treatment[i] == "Drug A":
            if dosage[i] == "Low":
                outcomes.append(np.random.normal(60, 12))
            else:
                outcomes.append(np.random.normal(75, 10))
        else:
            if dosage[i] == "Low":
                outcomes.append(np.random.normal(65, 11))
            else:
                outcomes.append(np.random.normal(85, 9))

    # Create data frame
    clinical_data = pd.DataFrame({
        'outcome': outcomes,
        'treatment': treatment,
        'dosage': dosage
    })
    
    return clinical_data

def create_manufacturing_example():
    """Create manufacturing quality example data."""
    np.random.seed(123)
    n_per_cell = 12

    # Generate data for 3x2 factorial design
    # Factor 1: Machine Type (A, B, C)
    # Factor 2: Shift (Day, Night)

    machine = ["Machine A"] * (n_per_cell * 2) + ["Machine B"] * (n_per_cell * 2) + ["Machine C"] * (n_per_cell * 2)
    shift = ["Day"] * n_per_cell + ["Night"] * n_per_cell
    shift = shift * 3

    # Generate quality scores
    quality_scores = []
    for i in range(len(machine)):
        if machine[i] == "Machine A":
            if shift[i] == "Day":
                quality_scores.append(np.random.normal(95, 3))
            else:
                quality_scores.append(np.random.normal(92, 4))
        elif machine[i] == "Machine B":
            if shift[i] == "Day":
                quality_scores.append(np.random.normal(88, 5))
            else:
                quality_scores.append(np.random.normal(85, 6))
        else:
            if shift[i] == "Day":
                quality_scores.append(np.random.normal(90, 4))
            else:
                quality_scores.append(np.random.normal(87, 5))

    # Create data frame
    quality_data = pd.DataFrame({
        'quality': quality_scores,
        'machine': machine,
        'shift': shift
    })
    
    return quality_data

# --- Main Execution Block ---
if __name__ == "__main__":
    print("=== TWO-WAY ANOVA ANALYSIS DEMONSTRATION ===\n")
    
    # Load sample data
    print("1. Loading sample data...")
    mtcars = load_sample_data()
    print(f"   Loaded {len(mtcars)} observations\n")
    
    # Manual two-way ANOVA
    print("2. Performing manual two-way ANOVA...")
    anova_result = manual_two_way_anova(mtcars, 'cyl_factor', 'am_factor', 'mpg')
    print(f"   Factor 1 (Cylinders) F-statistic: {anova_result['f_factor1']:.3f}")
    print(f"   Factor 1 p-value: {anova_result['p_factor1']:.4f}")
    print(f"   Factor 2 (Transmission) F-statistic: {anova_result['f_factor2']:.3f}")
    print(f"   Factor 2 p-value: {anova_result['p_factor2']:.4f}")
    print(f"   Interaction F-statistic: {anova_result['f_interaction']:.3f}")
    print(f"   Interaction p-value: {anova_result['p_interaction']:.4f}\n")
    
    # Built-in two-way ANOVA
    print("3. Performing built-in two-way ANOVA...")
    model, two_way_anova, key_stats = builtin_two_way_anova(mtcars, "cyl_factor", "am_factor", "mpg")
    print("   ANOVA Table:")
    print(two_way_anova)
    print()
    
    # Descriptive statistics
    print("4. Calculating descriptive statistics...")
    desc_stats = calculate_descriptive_stats(mtcars, "cyl_factor", "am_factor", "mpg")
    print("   Cell means:")
    print(desc_stats['cell_means'])
    print(f"   Grand mean: {desc_stats['grand_mean']:.2f}\n")
    
    # Effect size analysis
    print("5. Calculating effect sizes...")
    effect_sizes = calculate_two_way_effect_sizes(anova_result, mtcars, "cyl_factor", "am_factor", "mpg")
    print(f"   Partial η² - Factor 1: {effect_sizes['partial_eta2_factor1']:.4f}")
    print(f"   Partial η² - Factor 2: {effect_sizes['partial_eta2_factor2']:.4f}")
    print(f"   Partial η² - Interaction: {effect_sizes['partial_eta2_interaction']:.4f}\n")
    
    # Assumption checking
    print("6. Checking assumptions...")
    normality_results = check_normality_two_way(mtcars, "cyl_factor", "am_factor", "mpg")
    homogeneity_results = check_homogeneity_two_way(mtcars, "cyl_factor", "am_factor", "mpg")
    print(f"   Normality tests passed: {normality_results['normal_tests']}/{normality_results['total_tests']}")
    print(f"   Homogeneity tests passed: {homogeneity_results['equal_var_tests']}/{homogeneity_results['total_tests']}\n")
    
    # Simple effects analysis (if interaction is significant)
    if key_stats['p_interaction'] < 0.05:
        print("7. Performing simple effects analysis...")
        simple_effects_results = simple_effects_analysis(mtcars, "cyl_factor", "am_factor", "mpg")
        print(f"   Found {sum(r['significant'] for r in simple_effects_results.values())} significant simple effects\n")
    
    # Post hoc tests
    print("8. Performing post hoc tests...")
    posthoc_results = perform_posthoc_tests(mtcars, "cyl_factor", "am_factor", "mpg", key_stats)
    print(f"   Completed {len(posthoc_results)} post hoc test sets\n")
    
    # Create visualizations
    print("9. Creating visualizations...")
    create_interaction_plot(mtcars, "cyl_factor", "am_factor", "mpg", "MPG by Cylinders and Transmission")
    create_box_plot(mtcars, "cyl_factor", "am_factor", "mpg", "MPG Distribution by Cylinders and Transmission")
    create_heatmap(desc_stats['cell_means'], "cyl_factor", "am_factor", "Cell Means Heatmap")
    
    # Practical examples
    print("10. Running practical examples...")
    
    # Educational example
    education_data = create_educational_example()
    education_model, education_anova, education_stats = builtin_two_way_anova(education_data, "method", "class_size", "score")
    print("   Educational example completed")
    
    # Clinical example
    clinical_data = create_clinical_example()
    clinical_model, clinical_anova, clinical_stats = builtin_two_way_anova(clinical_data, "treatment", "dosage", "outcome")
    print("   Clinical example completed")
    
    # Manufacturing example
    quality_data = create_manufacturing_example()
    quality_model, quality_anova, quality_stats = builtin_two_way_anova(quality_data, "machine", "shift", "quality")
    print("   Manufacturing example completed\n")
    
    print("=== ANALYSIS COMPLETE ===")
    print("All functions are now available for use in your own analyses.")
    print("See the documentation for each function for detailed usage instructions.") 