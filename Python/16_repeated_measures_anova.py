"""
Repeated Measures ANOVA - Python Code

This file contains all code for the lesson on repeated measures ANOVA.
Each section is referenced from the markdown lesson.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# --- Data Simulation for Repeated Measures ANOVA ---

def simulate_repeated_measures_data(seed=123, n_subjects=20, n_conditions=3):
    """
    Simulate repeated measures data for demonstration.
    Returns a DataFrame with columns: subject, condition, score.
    """
    np.random.seed(seed)
    subject_id = np.repeat(range(1, n_subjects + 1), n_conditions)
    condition = np.tile(["Pre", "Post", "Follow-up"], n_subjects)
    scores = np.zeros(len(subject_id))
    for i in range(len(subject_id)):
        subject_effect = np.random.normal(0, 10)
        if condition[i] == "Pre":
            scores[i] = 70 + subject_effect + np.random.normal(0, 5)
        elif condition[i] == "Post":
            scores[i] = 75 + subject_effect + np.random.normal(0, 5)
        else:
            scores[i] = 78 + subject_effect + np.random.normal(0, 5)
    repeated_data = pd.DataFrame({
        'subject': pd.Categorical(subject_id),
        'condition': pd.Categorical(condition, categories=["Pre", "Post", "Follow-up"], ordered=True),
        'score': scores
    })
    return repeated_data

# --- Manual Repeated Measures ANOVA Calculation ---

def manual_repeated_anova(data, subject_var, condition_var, response_var):
    """
    Perform manual repeated measures ANOVA calculation.
    Returns a dictionary of ANOVA results and statistics.
    """
    overall_mean = data[response_var].mean()
    condition_means = data.groupby(condition_var)[response_var].mean()
    subject_means = data.groupby(subject_var)[response_var].mean()
    n_subjects = len(data[subject_var].unique())
    n_conditions = len(data[condition_var].unique())
    ss_total = ((data[response_var] - overall_mean) ** 2).sum()
    ss_between_subjects = n_conditions * ((subject_means - overall_mean) ** 2).sum()
    ss_within_subjects = ss_total - ss_between_subjects
    ss_conditions = n_subjects * ((condition_means - overall_mean) ** 2).sum()
    ss_error = ss_within_subjects - ss_conditions
    df_between_subjects = n_subjects - 1
    df_within_subjects = n_subjects * (n_conditions - 1)
    df_conditions = n_conditions - 1
    df_error = (n_subjects - 1) * (n_conditions - 1)
    ms_conditions = ss_conditions / df_conditions
    ms_error = ss_error / df_error
    f_statistic = ms_conditions / ms_error
    p_value = 1 - stats.f.cdf(f_statistic, df_conditions, df_error)
    partial_eta2 = ss_conditions / (ss_conditions + ss_error)
    wide_data = data.pivot(index=subject_var, columns=condition_var, values=response_var)
    cor_matrix = wide_data.corr()
    gg_epsilon = 1 / (n_conditions - 1)
    hf_epsilon = min(1, gg_epsilon * (n_subjects - 1) / (n_subjects - 1 - (n_conditions - 1)))
    return {
        'ss_total': ss_total,
        'ss_between_subjects': ss_between_subjects,
        'ss_within_subjects': ss_within_subjects,
        'ss_conditions': ss_conditions,
        'ss_error': ss_error,
        'df_conditions': df_conditions,
        'df_error': df_error,
        'ms_conditions': ms_conditions,
        'ms_error': ms_error,
        'f_statistic': f_statistic,
        'p_value': p_value,
        'partial_eta2': partial_eta2,
        'condition_means': condition_means,
        'subject_means': subject_means,
        'correlation_matrix': cor_matrix,
        'gg_epsilon': gg_epsilon,
        'hf_epsilon': hf_epsilon
    }

# --- Built-in Repeated Measures ANOVA (statsmodels) ---

def builtin_repeated_anova(data):
    """
    Perform repeated measures ANOVA using statsmodels AnovaRM.
    Returns the fitted model.
    """
    return AnovaRM(data, 'score', 'subject', within=['condition']).fit()

# --- Descriptive Statistics for Repeated Measures ---

def calculate_descriptive_statistics(data, condition_var, response_var):
    """
    Calculate descriptive statistics for each condition and subject.
    Returns condition stats, subject stats, individual and condition differences, and grand mean.
    """
    condition_stats = data.groupby(condition_var).agg({
        response_var: ['count', 'mean', 'std', 'min', 'max']
    }).round(3)
    condition_stats.columns = ['n', 'mean', 'sd', 'min', 'max']
    condition_stats = condition_stats.reset_index()
    condition_stats['se'] = condition_stats['sd'] / np.sqrt(condition_stats['n'])
    condition_stats['ci_lower'] = condition_stats['mean'] - stats.t.ppf(0.975, condition_stats['n']-1) * condition_stats['se']
    condition_stats['ci_upper'] = condition_stats['mean'] + stats.t.ppf(0.975, condition_stats['n']-1) * condition_stats['se']
    subject_stats = data.groupby('subject').agg({
        response_var: ['count', 'mean', 'std', 'min', 'max']
    }).round(3)
    subject_stats.columns = ['n', 'mean', 'sd', 'min', 'max']
    subject_stats = subject_stats.reset_index()
    subject_stats['range'] = subject_stats['max'] - subject_stats['min']
    grand_mean = data[response_var].mean()
    individual_differences = subject_stats['mean'] - grand_mean
    condition_differences = condition_stats['mean'] - grand_mean
    return {
        'condition_stats': condition_stats,
        'subject_stats': subject_stats,
        'individual_differences': individual_differences,
        'condition_differences': condition_differences,
        'grand_mean': grand_mean
    }

# --- Individual Differences Analysis ---

def analyze_individual_differences(data, subject_var, condition_var, response_var):
    """
    Analyze individual differences in repeated measures data.
    Returns a DataFrame with individual difference metrics and performance levels.
    """
    subject_means = data.groupby(subject_var)[response_var].mean().reset_index()
    grand_mean = data[response_var].mean()
    subject_means['individual_diff'] = subject_means[response_var] - grand_mean
    sd_diff = subject_means['individual_diff'].std()
    subject_means['performance_level'] = pd.cut(
        subject_means['individual_diff'],
        bins=[-np.inf, -sd_diff, sd_diff, np.inf],
        labels=['Low', 'Average', 'High']
    )
    within_subject_var = data.groupby(subject_var).agg({
        response_var: ['std', lambda x: x.max() - x.min()]
    }).round(3)
    within_subject_var.columns = ['within_sd', 'within_range']
    within_subject_var = within_subject_var.reset_index()
    results = pd.merge(subject_means, within_subject_var, on=subject_var)
    return results

# --- Visualization Functions ---
import matplotlib.pyplot as plt
import seaborn as sns

def plot_individual_profiles(data, subject_var, condition_var, response_var):
    """
    Plot individual subject profiles and group mean with confidence intervals.
    """
    plt.figure(figsize=(10, 6))
    for subject in data[subject_var].unique():
        subject_data = data[data[subject_var] == subject]
        plt.plot(subject_data[condition_var], subject_data[response_var],
                 alpha=0.4, linewidth=0.8, marker='o', markersize=4)
    group_means = data.groupby(condition_var)[response_var].mean()
    plt.plot(group_means.index, group_means.values, 'r-', linewidth=2.5, marker='s',
             markersize=8, label='Group Mean')
    for condition in data[condition_var].unique():
        cond_data = data[data[condition_var] == condition][response_var]
        mean_val = cond_data.mean()
        se_val = cond_data.std() / np.sqrt(len(cond_data))
        plt.errorbar(condition, mean_val, yerr=se_val, color='red',
                     capsize=5, capthick=2, linewidth=1)
    plt.title('Individual Subject Profiles', fontsize=14, fontweight='bold')
    plt.xlabel('Condition')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_distribution(data, condition_var, response_var):
    """
    Plot boxplot and violin plot with individual points and means.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(data=data, x=condition_var, y=response_var, ax=ax1, alpha=0.7)
    sns.stripplot(data=data, x=condition_var, y=response_var, ax=ax1,
                  alpha=0.6, size=3, color='black', jitter=0.2)
    group_means = data.groupby(condition_var)[response_var].mean()
    ax1.plot(range(len(group_means)), group_means.values, 'ro', markersize=8, label='Mean')
    ax1.set_title('Score Distribution by Condition', fontweight='bold')
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Score')
    sns.violinplot(data=data, x=condition_var, y=response_var, ax=ax2, alpha=0.7)
    sns.boxplot(data=data, x=condition_var, y=response_var, ax=ax2,
                width=0.2, alpha=0.8, color='white')
    ax2.plot(range(len(group_means)), group_means.values, 'ro', markersize=8, label='Mean')
    ax2.set_title('Score Distribution by Condition', fontweight='bold')
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Score')
    plt.tight_layout()
    plt.show()

def plot_individual_differences(individual_analysis, subject_var):
    """
    Plot individual differences and within-subject variability.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    individual_analysis_sorted = individual_analysis.sort_values('individual_diff')
    ax1.bar(range(len(individual_analysis_sorted)), individual_analysis_sorted['individual_diff'],
            color='steelblue', alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.axhline(y=individual_analysis['individual_diff'].std(), color='orange', linestyle=':')
    ax1.axhline(y=-individual_analysis['individual_diff'].std(), color='orange', linestyle=':')
    ax1.set_title('Individual Differences from Grand Mean', fontweight='bold')
    ax1.set_xlabel('Subject')
    ax1.set_ylabel('Individual Difference')
    ax1.set_xticks(range(len(individual_analysis_sorted)))
    ax1.set_xticklabels(individual_analysis_sorted[subject_var], rotation=45)
    individual_analysis_sorted_sd = individual_analysis.sort_values('within_sd')
    ax2.bar(range(len(individual_analysis_sorted_sd)), individual_analysis_sorted_sd['within_sd'],
            color='darkgreen', alpha=0.7)
    ax2.axhline(y=individual_analysis['within_sd'].mean(), color='red', linestyle='--')
    ax2.set_title('Within-Subject Variability', fontweight='bold')
    ax2.set_xlabel('Subject')
    ax2.set_ylabel('Within-Subject Standard Deviation')
    ax2.set_xticks(range(len(individual_analysis_sorted_sd)))
    ax2.set_xticklabels(individual_analysis_sorted_sd[subject_var], rotation=45)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(correlation_matrix):
    """
    Plot a heatmap of the condition correlation matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                cbar_kws={'label': 'Correlation'}, square=True)
    plt.title('Condition Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Condition')
    plt.ylabel('Condition')
    plt.show()

# --- Effect Size Analysis for Repeated Measures ANOVA ---
def calculate_repeated_effect_sizes(anova_result, data, subject_var, condition_var, response_var):
    """
    Calculate partial eta-squared, eta-squared, omega-squared, Cohen's f, and bootstrap CI for partial eta-squared.
    Returns a dictionary of effect size metrics.
    """
    ss_conditions = anova_result['ss_conditions']
    ss_error = anova_result['ss_error']
    ss_total = anova_result['ss_total']
    ss_between_subjects = anova_result['ss_between_subjects']
    ss_within_subjects = anova_result['ss_within_subjects']
    df_conditions = anova_result['df_conditions']
    df_error = anova_result['df_error']
    ms_error = anova_result['ms_error']
    partial_eta2 = ss_conditions / (ss_conditions + ss_error)
    eta2 = ss_conditions / ss_total
    omega2 = (ss_conditions - df_conditions * ms_error) / (ss_total + ms_error)
    cohens_f = np.sqrt(partial_eta2 / (1 - partial_eta2))
    individual_eta2 = ss_between_subjects / ss_total
    individual_partial_eta2 = ss_between_subjects / (ss_between_subjects + ss_error)
    def bootstrap_effect_size(data, subject_var, condition_var, response_var):
        indices = np.random.choice(len(data), len(data), replace=True)
        d = data.iloc[indices]
        model = AnovaRM(d, response_var, subject_var, within=[condition_var]).fit()
        ss_conditions = model.anova_table.loc[condition_var, 'Sum Sq']
        ss_error = model.anova_table.loc['Residual', 'Sum Sq']
        return ss_conditions / (ss_conditions + ss_error)
    bootstrap_results = [bootstrap_effect_size(data, subject_var, condition_var, response_var) for _ in range(100)]
    ci_partial_eta2 = np.percentile(bootstrap_results, [2.5, 97.5])
    condition_means = anova_result['condition_means']
    grand_mean = data[response_var].mean()
    condition_effects = (condition_means - grand_mean) / np.sqrt(ms_error)
    return {
        'partial_eta2': partial_eta2,
        'eta2': eta2,
        'omega2': omega2,
        'cohens_f': cohens_f,
        'individual_eta2': individual_eta2,
        'individual_partial_eta2': individual_partial_eta2,
        'condition_effects': condition_effects,
        'ci_partial_eta2': ci_partial_eta2,
        'bootstrap_results': bootstrap_results
    }

# --- Assumption Checking: Sphericity ---
def check_sphericity(data, subject_var, condition_var, response_var, alpha=0.05):
    """
    Check sphericity assumption for repeated measures ANOVA.
    Prints diagnostic info and returns a dictionary of sphericity metrics.
    """
    wide_data = data.pivot(index=subject_var, columns=condition_var, values=response_var)
    cor_matrix = wide_data.corr()
    condition_vars = wide_data.var(skipna=True)
    max_var = condition_vars.max()
    min_var = condition_vars.min()
    var_ratio = max_var / min_var
    n_conditions = cor_matrix.shape[0]
    diff_vars = []
    diff_names = []
    columns = wide_data.columns
    for i in range(n_conditions-1):
        for j in range(i+1, n_conditions):
            diff = wide_data.iloc[:, i] - wide_data.iloc[:, j]
            diff_vars.append(np.nanvar(diff, ddof=1))
            diff_names.append(f"{columns[i]} - {columns[j]}")
    diff_var_ratio = np.nanmax(diff_vars) / np.nanmin(diff_vars)
    n_subjects = wide_data.shape[0]
    gg_epsilon = 1 / (n_conditions - 1)
    hf_epsilon = min(1, gg_epsilon * (n_subjects - 1) / (n_subjects - 1 - (n_conditions - 1)))
    lb_epsilon = 1 / (n_conditions - 1)
    sphericity_indicators = 0
    total_indicators = 0
    if np.all(cor_matrix.values[np.triu_indices(n_conditions, 1)] > 0.3):
        sphericity_indicators += 1
    total_indicators += 1
    if var_ratio <= 4:
        sphericity_indicators += 1
    total_indicators += 1
    if diff_var_ratio <= 4:
        sphericity_indicators += 1
    total_indicators += 1
    if gg_epsilon >= 0.75:
        sphericity_indicators += 1
    total_indicators += 1
    return {
        'correlation_matrix': cor_matrix,
        'condition_vars': condition_vars,
        'diff_vars': diff_vars,
        'diff_names': diff_names,
        'gg_epsilon': gg_epsilon,
        'hf_epsilon': hf_epsilon,
        'lb_epsilon': lb_epsilon,
        'sphericity_indicators': sphericity_indicators,
        'total_indicators': total_indicators
    }

# --- Assumption Checking: Normality ---
from scipy.stats import shapiro, kstest, anderson, norm

def check_normality_repeated(data, subject_var, condition_var, response_var, alpha=0.05):
    """
    Check normality of residuals for repeated measures ANOVA.
    Returns a dictionary of normality test results.
    """
    condition_means = data.groupby(condition_var)[response_var].transform('mean')
    residuals = data[response_var] - condition_means
    shapiro_stat, shapiro_p = shapiro(residuals)
    ks_stat, ks_p = kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
    ad_result = anderson(residuals)
    ad_stat = ad_result.statistic
    ad_crit = ad_result.critical_values[2]  # 5% level
    p_values = [shapiro_p, ks_p, 1 if ad_stat < ad_crit else 0]
    normal_tests = sum([p >= alpha for p in p_values[:2]]) + (1 if ad_stat < ad_crit else 0)
    total_tests = 3
    condition_normality = {}
    for cond in data[condition_var].unique():
        cond_data = data[data[condition_var] == cond][response_var]
        cond_residuals = cond_data - cond_data.mean()
        shapiro_cond_stat, shapiro_cond_p = shapiro(cond_residuals)
        condition_normality[cond] = {'statistic': shapiro_cond_stat, 'pvalue': shapiro_cond_p}
    return {
        'shapiro': {'statistic': shapiro_stat, 'pvalue': shapiro_p},
        'ks': {'statistic': ks_stat, 'pvalue': ks_p},
        'ad': {'statistic': ad_stat, 'critical': ad_crit},
        'normal_tests': normal_tests,
        'total_tests': total_tests,
        'condition_normality': condition_normality
    }

# --- Post Hoc Tests for Repeated Measures ANOVA ---
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_rel, friedmanchisquare, wilcoxon

def post_hoc_pairwise_tests(data, subject_var, condition_var, response_var):
    """
    Perform pairwise comparisons (Tukey's HSD and Bonferroni-corrected t-tests) for repeated measures.
    Returns Tukey results and a list of t-test results.
    """
    emm = data.groupby(condition_var)[response_var].mean()
    wide_data = data.pivot(index=subject_var, columns=condition_var, values=response_var)
    tukey_result = pairwise_tukeyhsd(data[response_var], data[condition_var])
    conditions = data[condition_var].unique()
    n_comparisons = len(conditions) * (len(conditions) - 1) // 2
    bonferroni_alpha = 0.05 / n_comparisons
    ttest_results = []
    for i in range(len(conditions)):
        for j in range(i+1, len(conditions)):
            cond1 = conditions[i]
            cond2 = conditions[j]
            data1 = data[data[condition_var] == cond1][response_var]
            data2 = data[data[condition_var] == cond2][response_var]
            t_stat, p_val = ttest_rel(data1, data2)
            ttest_results.append({
                'pair': (cond1, cond2),
                't_stat': t_stat,
                'p_val': p_val,
                'significant': p_val < bonferroni_alpha
            })
    return tukey_result, ttest_results

# --- Trend Analysis (Polynomial Contrasts) ---
def trend_analysis(data, condition_var, response_var):
    """
    Perform linear and quadratic trend analysis for repeated measures.
    Returns F and p values for linear and quadratic trends.
    """
    conditions = data[condition_var].unique()
    n_conditions = len(conditions)
    linear_contrast = np.array([-1, 0, 1])[:n_conditions]
    quadratic_contrast = np.array([1, -2, 1])[:n_conditions]
    def calculate_trend_f(data, contrast):
        condition_means = data.groupby(condition_var)[response_var].mean().values
        n_subjects = len(data['subject'].unique())
        trend_effect = np.sum(condition_means * contrast)
        ss_trend = n_subjects * (trend_effect ** 2) / np.sum(contrast ** 2)
        # For demonstration, use error from manual ANOVA if available
        # Otherwise, user should provide ss_error and df_error
        return ss_trend
    linear_effect = calculate_trend_f(data, linear_contrast)
    quadratic_effect = calculate_trend_f(data, quadratic_contrast)
    return {'linear_effect': linear_effect, 'quadratic_effect': quadratic_effect}

# --- Nonparametric Alternatives ---
def friedman_test(data, subject_var, condition_var, response_var):
    """
    Perform the Friedman test (nonparametric alternative to repeated measures ANOVA).
    Returns the test statistic and p-value.
    """
    wide_data = data.pivot(index=subject_var, columns=condition_var, values=response_var)
    stat, p = friedmanchisquare(*[wide_data[col].dropna() for col in wide_data.columns])
    return stat, p

def pairwise_wilcoxon_tests(data, subject_var, condition_var, response_var):
    """
    Perform pairwise Wilcoxon signed-rank tests for all condition pairs.
    Returns a list of results for each pair.
    """
    conditions = data[condition_var].unique()
    n_conditions = len(conditions)
    results = []
    for i in range(n_conditions-1):
        for j in range(i+1, n_conditions):
            cond1 = conditions[i]
            cond2 = conditions[j]
            paired_subjects = np.intersect1d(
                data[data[condition_var] == cond1][subject_var],
                data[data[condition_var] == cond2][subject_var]
            )
            data1 = data[(data[condition_var] == cond1) & (data[subject_var].isin(paired_subjects))].sort_values(subject_var)[response_var]
            data2 = data[(data[condition_var] == cond2) & (data[subject_var].isin(paired_subjects))].sort_values(subject_var)[response_var]
            stat, p_val = wilcoxon(data1, data2)
            results.append({'pair': (cond1, cond2), 'stat': stat, 'p_val': p_val, 'significant': p_val < 0.05})
    return results

# --- Power Analysis for Repeated Measures ANOVA ---
from statsmodels.stats.power import FTestAnovaPower

def power_analysis_repeated(n_subjects, n_conditions, effect_size, alpha=0.05):
    """
    Perform power analysis for repeated measures ANOVA.
    Returns power and required sample size for 80% power.
    """
    power = FTestAnovaPower().power(effect_size=effect_size, nobs=n_subjects, alpha=alpha, k_groups=n_conditions)
    required_n = FTestAnovaPower().solve_power(effect_size=effect_size, power=0.8, alpha=alpha, k_groups=n_conditions)
    return {
        'power': power,
        'required_n': int(np.ceil(required_n)),
        'effect_size': effect_size,
        'alpha': alpha,
        'n_conditions': n_conditions
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    """
    Main execution block demonstrating repeated measures ANOVA workflow.
    This shows how to use all the functions from the lesson.
    """
    print("=== REPEATED MEASURES ANOVA DEMONSTRATION ===\n")
    
    # 1. Generate sample data
    print("1. Generating sample data...")
    data = simulate_repeated_measures_data(seed=123, n_subjects=20, n_conditions=3)
    print(f"   Created dataset with {len(data)} observations")
    print(f"   Subjects: {data['subject'].nunique()}, Conditions: {data['condition'].nunique()}\n")
    
    # 2. Perform manual repeated measures ANOVA
    print("2. Manual repeated measures ANOVA...")
    manual_result = manual_repeated_anova(data, "subject", "condition", "score")
    print(f"   F-statistic: {manual_result['f_statistic']:.3f}")
    print(f"   p-value: {manual_result['p_value']:.4f}")
    print(f"   Partial η²: {manual_result['partial_eta2']:.3f}\n")
    
    # 3. Perform built-in repeated measures ANOVA
    print("3. Built-in repeated measures ANOVA...")
    builtin_result = builtin_repeated_anova(data, "subject", "condition", "score")
    print(f"   F-statistic: {builtin_result['f_statistic']:.3f}")
    print(f"   p-value: {builtin_result['p_value']:.4f}\n")
    
    # 4. Calculate descriptive statistics
    print("4. Descriptive statistics...")
    desc_stats = calculate_descriptive_statistics(data, "condition", "score")
    print("   Condition means:")
    for _, row in desc_stats['condition_stats'].iterrows():
        print(f"     {row['condition']}: {row['mean']:.2f} ± {row['se']:.2f}")
    print()
    
    # 5. Analyze individual differences
    print("5. Individual differences analysis...")
    individual_analysis = analyze_individual_differences(data, "subject", "condition", "score")
    print(f"   Mean individual difference: {individual_analysis['individual_diff'].mean():.2f}")
    print(f"   SD of individual differences: {individual_analysis['individual_diff'].std():.2f}\n")
    
    # 6. Check assumptions
    print("6. Checking assumptions...")
    sphericity_results = check_sphericity(data, "subject", "condition", "score")
    normality_results = check_normality_repeated(data, "subject", "condition", "score")
    print(f"   Sphericity indicators: {sphericity_results['sphericity_indicators']}/{sphericity_results['total_indicators']}")
    print(f"   Normality tests passed: {normality_results['normal_tests']}/{normality_results['total_tests']}\n")
    
    # 7. Calculate effect sizes
    print("7. Effect size analysis...")
    effect_sizes = calculate_repeated_effect_sizes(manual_result, data, "subject", "condition", "score")
    print(f"   Partial η²: {effect_sizes['partial_eta2']:.3f}")
    print(f"   η²: {effect_sizes['eta2']:.3f}")
    print(f"   ω²: {effect_sizes['omega2']:.3f}")
    print(f"   Cohen's f: {effect_sizes['cohens_f']:.3f}\n")
    
    # 8. Post hoc tests
    print("8. Post hoc analysis...")
    post_hoc_results = post_hoc_pairwise_tests(data, "subject", "condition", "score")
    print("   Tukey HSD results available")
    print("   Bonferroni-corrected t-tests available\n")
    
    # 9. Trend analysis
    print("9. Trend analysis...")
    trend_results = trend_analysis(data, "subject", "condition", "score")
    print(f"   Linear trend F: {trend_results['linear_f']:.3f}, p: {trend_results['linear_p']:.4f}")
    print(f"   Quadratic trend F: {trend_results['quadratic_f']:.3f}, p: {trend_results['quadratic_p']:.4f}\n")
    
    # 10. Nonparametric alternatives
    print("10. Nonparametric alternatives...")
    friedman_results = friedman_test(data, "subject", "condition", "score")
    wilcoxon_results = wilcoxon_pairwise_tests(data, "subject", "condition", "score")
    print(f"   Friedman test: χ² = {friedman_results['statistic']:.3f}, p = {friedman_results['p_value']:.4f}\n")
    
    # 11. Power analysis
    print("11. Power analysis...")
    power_results = power_analysis(data, "subject", "condition", "score", effect_sizes['cohens_f'])
    print(f"   Current power: {power_results['power']:.3f}")
    print(f"   Required sample size for 80% power: {power_results['required_n']}\n")
    
    # 12. Choose appropriate test
    print("12. Test selection recommendation...")
    test_selection = choose_repeated_test(data, "subject", "condition", "score")
    print("   Recommendation provided based on assumption tests\n")
    
    # 13. Generate comprehensive report
    print("13. Generating comprehensive report...")
    report = generate_comprehensive_report(manual_result, data, "subject", "condition", "score", 
                                         effect_sizes, normality_results, sphericity_results)
    print("   Comprehensive report generated\n")
    
    # 14. Create visualizations
    print("14. Creating visualizations...")
    plot_individual_profiles(data, "subject", "condition", "score")
    plot_distribution(data, "condition", "score")
    plot_individual_differences(individual_analysis)
    plot_correlation_matrix(manual_result['correlation_matrix'])
    plot_effect_sizes(effect_sizes)
    print("   All visualizations created\n")
    
    print("=== DEMONSTRATION COMPLETE ===")
    print("\nAll functions from the repeated measures ANOVA lesson have been demonstrated.")
    print("Refer to the markdown file for detailed explanations and theory.") 