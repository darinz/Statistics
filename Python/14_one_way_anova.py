"""
One-Way ANOVA Analysis in Python

This module provides comprehensive tools for performing one-way Analysis of Variance (ANOVA)
including manual calculations, assumption checking, effect size analysis, post hoc tests,
power analysis, and practical examples.

"""

# --- Imports ---
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f, shapiro, anderson, ks_2samp, levene, bartlett, fligner, kruskal, mannwhitneyu, rankdata, skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statsmodels.stats.multitest import multipletests

# --- Data Loading ---
def load_sample_data():
    """Load and prepare the mtcars dataset for ANOVA examples."""
    mtcars = sns.load_dataset('mpg')
    mtcars['cyl_factor'] = pd.Categorical(mtcars['cylinders'], categories=[4, 6, 8], ordered=True)
    mtcars['cyl_factor'] = mtcars['cyl_factor'].map({4: '4-cylinder', 6: '6-cylinder', 8: '8-cylinder'})
    return mtcars

# --- Manual ANOVA Calculation ---
def manual_anova(group_data):
    """Perform manual one-way ANOVA calculation. Returns a dict of results."""
    k = len(group_data)
    n_per_group = [len(group) for group in group_data]
    total_n = sum(n_per_group)
    group_means = [np.mean(group) for group in group_data]
    overall_mean = np.mean(np.concatenate(group_data))
    ss_between = sum(n_per_group[i] * (group_means[i] - overall_mean)**2 for i in range(k))
    ss_within = sum(sum((group_data[i] - group_means[i])**2) for i in range(k))
    ss_total = ss_between + ss_within
    df_between = k - 1
    df_within = total_n - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_statistic = ms_between / ms_within
    p_value = 1 - f.cdf(f_statistic, df_between, df_within)
    eta_squared = ss_between / ss_total
    omega_squared = (ss_between - (df_between * ms_within)) / (ss_total + ms_within)
    return {
        'ss_between': ss_between,
        'ss_within': ss_within,
        'ss_total': ss_total,
        'df_between': df_between,
        'df_within': df_within,
        'ms_between': ms_between,
        'ms_within': ms_within,
        'f_statistic': f_statistic,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'omega_squared': omega_squared,
        'group_means': group_means,
        'overall_mean': overall_mean
    }

# --- Built-in ANOVA (scipy, statsmodels) ---
def builtin_anova(group_data):
    """Perform ANOVA using scipy's f_oneway."""
    f_stat, p_val = stats.f_oneway(*group_data)
    return f_stat, p_val

def statsmodels_anova(data, group_var, response_var):
    """Perform ANOVA using statsmodels and return the ANOVA table and model."""
    model = ols(f'{response_var} ~ {group_var}', data=data).fit()
    anova_table = anova_lm(model, typ=2)
    return anova_table, model

# --- Descriptive Statistics ---
def descriptive_stats(data, group_var, response_var):
    """Return descriptive statistics for each group and overall."""
    group_stats = data.groupby(group_var)[response_var].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
    group_stats['se'] = group_stats['std'] / np.sqrt(group_stats['count'])
    group_stats['cv'] = group_stats['std'] / group_stats['mean']
    group_stats['skewness'] = data.groupby(group_var)[response_var].apply(skew)
    group_stats['kurtosis'] = data.groupby(group_var)[response_var].apply(kurtosis)
    overall = data[response_var].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
    return group_stats, overall

def pooled_variance(group_data):
    n_per_group = [len(group) for group in group_data]
    var_per_group = [np.var(group, ddof=1) for group in group_data]
    numerator = sum((n_per_group[i] - 1) * var_per_group[i] for i in range(len(group_data)))
    denominator = sum(n_per_group[i] - 1 for i in range(len(group_data)))
    return numerator / denominator

# --- Visualization ---
def anova_visualizations(data, group_var, response_var, fitted_values, residuals):
    """Create key plots for ANOVA diagnostics and group comparison."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("Set2")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive ANOVA Visualization', fontsize=16, fontweight='bold')
    # Boxplot
    ax1 = axes[0, 0]
    sns.boxplot(data=data, x=group_var, y=response_var, ax=ax1)
    means = data.groupby(group_var)[response_var].mean()
    ax1.plot(range(len(means)), means, 'o', color='red', markersize=8, label='Mean')
    ax1.set_title('Boxplot with Means')
    ax1.legend()
    # Violin
    ax2 = axes[0, 1]
    sns.violinplot(data=data, x=group_var, y=response_var, ax=ax2, inner='box')
    ax2.set_title('Violin Plot')
    # Histogram
    ax3 = axes[0, 2]
    for group in data[group_var].unique():
        group_data = data[data[group_var] == group][response_var]
        ax3.hist(group_data, alpha=0.7, label=group, bins=8, density=True)
        x_range = np.linspace(group_data.min(), group_data.max(), 100)
        density = stats.gaussian_kde(group_data)(x_range)
        ax3.plot(x_range, density, linewidth=2)
    ax3.set_title('Histogram by Group')
    ax3.legend()
    # Q-Q plot
    ax4 = axes[1, 0]
    group = data[group_var].unique()[0]
    group_data = data[data[group_var] == group][response_var]
    stats.probplot(group_data, dist="norm", plot=ax4)
    ax4.set_title(f'Q-Q Plot for {group}')
    # Residuals vs Fitted
    ax5 = axes[1, 1]
    ax5.scatter(fitted_values, residuals, alpha=0.7, c=data[group_var].astype('category').cat.codes)
    ax5.axhline(y=0, color='red', linestyle='--')
    ax5.set_title('Residuals vs Fitted')
    # Means with CI
    ax6 = axes[1, 2]
    group_stats = data.groupby(group_var)[response_var].agg(['mean', 'std', 'count'])
    group_stats['se'] = group_stats['std'] / np.sqrt(group_stats['count'])
    x_pos = np.arange(len(group_stats))
    ax6.bar(x_pos, group_stats['mean'], alpha=0.7, yerr=1.96 * group_stats['se'], capsize=5)
    ax6.set_title('Means with 95% CI')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(group_stats.index)
    plt.tight_layout()
    plt.show()

# --- Effect Size ---
def anova_effect_sizes(anova_result):
    """Calculate eta-squared, omega-squared, Cohen's f, and confidence intervals."""
    ss_between = anova_result['ss_between']
    ss_within = anova_result['ss_within']
    ss_total = anova_result['ss_total']
    df_between = anova_result['df_between']
    ms_within = anova_result['ms_within']
    eta_squared = ss_between / ss_total
    omega_squared = (ss_between - (df_between * ms_within)) / (ss_total + ms_within)
    cohens_f = np.sqrt(eta_squared / (1 - eta_squared))
    return {'eta_squared': eta_squared, 'omega_squared': omega_squared, 'cohens_f': cohens_f}

# --- Post Hoc Tests ---
def posthoc_tests(data, group_var, response_var, alpha=0.05):
    """Run Tukey's HSD, Bonferroni, Holm, and FDR post hoc tests."""
    mc = MultiComparison(data[response_var], data[group_var])
    tukey = mc.tukeyhsd(alpha=alpha)
    groups = data[group_var].unique()
    pvals = []
    comps = []
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1 = data[data[group_var] == groups[i]][response_var]
            g2 = data[data[group_var] == groups[j]][response_var]
            _, p = stats.ttest_ind(g1, g2)
            pvals.append(p)
            comps.append(f"{groups[i]} vs {groups[j]}")
    bonf = multipletests(pvals, method='bonferroni')[1]
    holm = multipletests(pvals, method='holm')[1]
    fdr = multipletests(pvals, method='fdr_bh')[1]
    return {'tukey': tukey, 'bonferroni': bonf, 'holm': holm, 'fdr': fdr, 'comparisons': comps}

# --- Assumption Checking ---
def check_assumptions(data, group_var, response_var, alpha=0.05):
    """Check ANOVA assumptions: normality, homogeneity, independence, outliers."""
    groups = data[group_var].unique()
    results = {}
    # Normality
    results['normality'] = {g: shapiro(data[data[group_var] == g][response_var]) for g in groups}
    # Homogeneity
    group_data = [data[data[group_var] == g][response_var] for g in groups]
    results['levene'] = levene(*group_data)
    results['bartlett'] = bartlett(*group_data)
    results['fligner'] = fligner(*group_data)
    # Outliers (IQR method)
    outliers = {}
    for g in groups:
        vals = data[data[group_var] == g][response_var]
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        outliers[g] = vals[(vals < lower) | (vals > upper)]
    results['outliers'] = outliers
    return results

# --- Nonparametric Alternatives ---
def kruskal_wallis(data, group_var, response_var):
    """Run Kruskal-Wallis test and pairwise Mann-Whitney U tests."""
    groups = data[group_var].unique()
    group_data = [data[data[group_var] == g][response_var] for g in groups]
    kw = kruskal(*group_data)
    # Pairwise Mann-Whitney U
    pvals = []
    comps = []
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1 = group_data[i]
            g2 = group_data[j]
            _, p = mannwhitneyu(g1, g2, alternative='two-sided')
            pvals.append(p)
            comps.append(f"{groups[i]} vs {groups[j]}")
    bonf = multipletests(pvals, method='bonferroni')[1]
    return {'kruskal': kw, 'pairwise_bonferroni': bonf, 'comparisons': comps}

# --- Power Analysis ---
def power_analysis_anova(k, n_per_group, f_effect_size, alpha=0.05):
    """Calculate power for one-way ANOVA."""
    df1 = k - 1
    df2 = k * (n_per_group - 1)
    lambda_param = f_effect_size**2 * k * n_per_group
    f_critical = f.ppf(1 - alpha, df1, df2)
    power = 1 - f.cdf(f_critical, df1, df2, lambda_param)
    return power

# --- Practical Examples ---
def example_educational():
    """Simulate and analyze educational intervention data."""
    np.random.seed(123)
    n = 25
    method_a = np.random.normal(72, 8, n)
    method_b = np.random.normal(78, 9, n)
    method_c = np.random.normal(75, 7, n)
    scores = np.concatenate([method_a, method_b, method_c])
    methods = pd.Categorical(['Traditional']*n + ['Interactive']*n + ['Technology']*n)
    df = pd.DataFrame({'score': scores, 'method': methods})
    return df

def example_clinical():
    """Simulate and analyze clinical trial data."""
    np.random.seed(456)
    n = 30
    placebo = np.random.normal(145, 12, n)
    drug_a = np.random.normal(135, 11, n)
    drug_b = np.random.normal(125, 10, n)
    bp = np.concatenate([placebo, drug_a, drug_b])
    treatments = pd.Categorical(['Placebo']*n + ['Drug A']*n + ['Drug B']*n)
    df = pd.DataFrame({'blood_pressure': bp, 'treatment': treatments})
    return df

def example_quality():
    """Simulate and analyze quality control data."""
    np.random.seed(789)
    n = 20
    a = np.random.normal(100, 2, n)
    b = np.random.normal(98, 3.5, n)
    c = np.random.normal(102, 4.5, n)
    output = np.concatenate([a, b, c])
    machines = pd.Categorical(['Machine A']*n + ['Machine B']*n + ['Machine C']*n)
    df = pd.DataFrame({'output': output, 'machine': machines})
    return df

# --- Main block for demonstration ---
if __name__ == "__main__":
    # Example: One-way ANOVA on mtcars
    mtcars = load_sample_data()
    mpg_4cyl = mtcars[mtcars['cylinders'] == 4]['mpg'].dropna().values
    mpg_6cyl = mtcars[mtcars['cylinders'] == 6]['mpg'].dropna().values
    mpg_8cyl = mtcars[mtcars['cylinders'] == 8]['mpg'].dropna().values
    print("\n--- Manual ANOVA ---")
    anova_res = manual_anova([mpg_4cyl, mpg_6cyl, mpg_8cyl])
    print("\n--- Built-in ANOVA ---")
    f_stat, p_val = builtin_anova([mpg_4cyl, mpg_6cyl, mpg_8cyl])
    print(f"F = {f_stat:.3f}, p = {p_val:.4f}")
    print("\n--- Statsmodels ANOVA ---")
    anova_table, model = statsmodels_anova(mtcars, 'cyl_factor', 'mpg')
    print(anova_table)
    # Visualizations
    anova_visualizations(mtcars, 'cyl_factor', 'mpg', model.fittedvalues, model.resid)
    # Descriptive stats
    group_stats, overall = descriptive_stats(mtcars, 'cyl_factor', 'mpg')
    print("\nGroup stats:\n", group_stats)
    print("\nOverall stats:\n", overall)
    # Effect size
    eff = anova_effect_sizes(anova_res)
    print("\nEffect sizes:", eff)
    # Post hoc
    posthoc = posthoc_tests(mtcars, 'cyl_factor', 'mpg')
    print("\nTukey's HSD:\n", posthoc['tukey'])
    # Assumptions
    assumptions = check_assumptions(mtcars, 'cyl_factor', 'mpg')
    print("\nAssumption checks:\n", assumptions)
    # Nonparametric
    np_results = kruskal_wallis(mtcars, 'cyl_factor', 'mpg')
    print("\nKruskal-Wallis:\n", np_results['kruskal'])
    # Power
    k = 3
    n_per_group = min([len(mpg_4cyl), len(mpg_6cyl), len(mpg_8cyl)])
    power = power_analysis_anova(k, n_per_group, eff['cohens_f'])
    print(f"\nPower: {power:.3f}")
    # Practical example
    print("\n--- Educational Example ---")
    edu = example_educational()
    print(edu.groupby('method')['score'].describe()) 