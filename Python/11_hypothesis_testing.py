import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower
import matplotlib.pyplot as plt
from itertools import combinations


def coin_fairness_test(n_flips=100, true_prob=0.6, hypothesized_prob=0.5, alpha=0.05):
    """
    Test if a coin is fair using binomial test.
    Demonstrates basic hypothesis testing concepts.
    """
    np.random.seed(123)
    coin_flips = np.random.binomial(1, true_prob, n_flips)
    head_count = np.sum(coin_flips)
    observed_proportion = head_count / n_flips
    
    # Calculate test statistic
    expected_heads = n_flips * hypothesized_prob
    standard_error = np.sqrt(n_flips * hypothesized_prob * (1 - hypothesized_prob))
    z_statistic = (head_count - expected_heads) / standard_error
    
    # Calculate p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    
    return {
        'head_count': head_count,
        'observed_proportion': observed_proportion,
        'z_statistic': z_statistic,
        'p_value': p_value,
        'decision': 'Reject H0' if p_value < alpha else 'Fail to reject H0'
    }


def calculate_power(effect_size, n, alpha=0.05):
    """
    Calculate power for different effect sizes and sample sizes.
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    ncp = effect_size * np.sqrt(n/2)
    power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    return power


def power_analysis_demo():
    """
    Demonstrate power analysis for different scenarios.
    """
    effect_sizes = [0.2, 0.5, 0.8]
    sample_sizes = [20, 50, 100]
    alpha_levels = [0.01, 0.05, 0.10]
    
    results = {}
    for d in effect_sizes:
        for n in sample_sizes:
            for alpha in alpha_levels:
                power = calculate_power(d, n, alpha)
                results[(d, n, alpha)] = power
    
    return results


def one_sample_t_test(data, hypothesized_mean, alpha=0.05):
    """
    Perform one-sample t-test.
    """
    t_stat, p_value = stats.ttest_1samp(data, hypothesized_mean)
    
    # Calculate effect size
    effect_size = (data.mean() - hypothesized_mean) / data.std(ddof=1)
    
    # Calculate confidence interval
    desc_stats = DescrStatsW(data)
    ci = desc_stats.tconfint_mean(alpha=alpha)
    
    # Check normality
    shapiro_stat, shapiro_p = stats.shapiro(data)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'confidence_interval': ci,
        'normality_test': (shapiro_stat, shapiro_p),
        'decision': 'Reject H0' if p_value < alpha else 'Fail to reject H0'
    }


def one_sample_proportion_test(successes, n, hypothesized_prop=0.5, alpha=0.05):
    """
    Perform one-sample proportion test using exact binomial test.
    """
    # Exact binomial test
    binom_result = stats.binomtest(successes, n, p=hypothesized_prop)
    
    # Normal approximation
    sample_prop = successes / n
    z_stat = (sample_prop - hypothesized_prop) / np.sqrt(hypothesized_prop * (1 - hypothesized_prop) / n)
    p_value_normal = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Check large sample conditions
    np_check = n * hypothesized_prop
    nq_check = n * (1 - hypothesized_prop)
    large_sample_ok = np_check >= 10 and nq_check >= 10
    
    return {
        'exact_p_value': binom_result.pvalue,
        'normal_p_value': p_value_normal,
        'z_statistic': z_stat,
        'large_sample_conditions': large_sample_ok,
        'confidence_interval': binom_result.proportions_ci(confidence_level=1-alpha),
        'decision': 'Reject H0' if binom_result.pvalue < alpha else 'Fail to reject H0'
    }


def one_sample_variance_test(data, hypothesized_variance, alpha=0.05):
    """
    Perform one-sample variance test using chi-square distribution.
    """
    sample_variance = data.var()
    n = len(data)
    df = n - 1
    
    # Calculate test statistic
    test_statistic = (df * sample_variance) / hypothesized_variance
    
    # Calculate p-value (two-tailed)
    p_value_lower = stats.chi2.cdf(test_statistic, df)
    p_value_upper = 1 - stats.chi2.cdf(test_statistic, df)
    p_value = 2 * min(p_value_lower, p_value_upper)
    
    # Calculate confidence interval
    chi_lower = stats.chi2.ppf(alpha/2, df)
    chi_upper = stats.chi2.ppf(1-alpha/2, df)
    ci_lower = (df * sample_variance) / chi_upper
    ci_upper = (df * sample_variance) / chi_lower
    
    return {
        'chi_square_statistic': test_statistic,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper),
        'variance_ratio': sample_variance / hypothesized_variance,
        'decision': 'Reject H0' if p_value < alpha else 'Fail to reject H0'
    }


def independent_t_test(group1, group2, equal_var=True, alpha=0.05):
    """
    Perform independent t-test with assumption checking.
    """
    # Check assumptions
    shapiro1_stat, shapiro1_p = stats.shapiro(group1)
    shapiro2_stat, shapiro2_p = stats.shapiro(group2)
    levene_stat, levene_p = stats.levene(group1, group2)
    
    # Perform tests
    pooled_t_stat, pooled_p = stats.ttest_ind(group1, group2, equal_var=True)
    welch_t_stat, welch_p = stats.ttest_ind(group1, group2, equal_var=False)
    
    # Calculate effect size
    def cohens_d_two_sample(x1, x2):
        pooled_std = np.sqrt(((len(x1) - 1) * x1.var() + (len(x2) - 1) * x2.var()) / (len(x1) + len(x2) - 2))
        return (x1.mean() - x2.mean()) / pooled_std
    
    effect_size = cohens_d_two_sample(group1, group2)
    
    # Confidence interval
    cm = CompareMeans.from_data(group1, group2)
    ci = cm.tconfint_diff(alpha=alpha, usevar='unequal')
    
    return {
        'normality_tests': [(shapiro1_stat, shapiro1_p), (shapiro2_stat, shapiro2_p)],
        'levene_test': (levene_stat, levene_p),
        'pooled_t_test': (pooled_t_stat, pooled_p),
        'welch_t_test': (welch_t_stat, welch_p),
        'effect_size': effect_size,
        'confidence_interval': ci,
        'recommended_test': 'Welch' if levene_p < alpha else 'Pooled'
    }


def paired_t_test(before, after, alpha=0.05):
    """
    Perform paired t-test.
    """
    differences = after - before
    t_stat, p_value = stats.ttest_rel(after, before)
    
    # Calculate effect size
    def cohens_d_paired(differences):
        return differences.mean() / differences.std(ddof=1)
    
    effect_size = cohens_d_paired(differences)
    
    # Confidence interval
    desc_stats = DescrStatsW(differences)
    ci = desc_stats.tconfint_mean(alpha=alpha)
    
    # Check normality of differences
    shapiro_stat, shapiro_p = stats.shapiro(differences)
    
    # Correlation
    correlation = np.corrcoef(before, after)[0, 1]
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'confidence_interval': ci,
        'normality_test': (shapiro_stat, shapiro_p),
        'correlation': correlation,
        'decision': 'Reject H0' if p_value < alpha else 'Fail to reject H0'
    }


def two_sample_proportion_test(group1_success, group1_total, group2_success, group2_total, alpha=0.05):
    """
    Perform two-sample proportion test.
    """
    p1 = group1_success / group1_total
    p2 = group2_success / group2_total
    
    # Check large sample conditions
    large_sample_ok = ((group1_total * p1 >= 10) and (group1_total * (1-p1) >= 10) and
                       (group2_total * p2 >= 10) and (group2_total * (1-p2) >= 10))
    
    # Chi-square test
    contingency_table = np.array([[group1_success, group1_total - group1_success],
                                  [group2_success, group2_total - group2_success]])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Confidence interval
    ci = confint_proportions_2indep(group1_success, group1_total, 
                                   group2_success, group2_total, 
                                   method='wald')
    
    # Effect size
    prop_diff = p2 - p1
    risk_ratio = p2 / p1
    odds1 = p1 / (1 - p1)
    odds2 = p2 / (1 - p2)
    odds_ratio = odds2 / odds1
    
    return {
        'chi_square_statistic': chi2_stat,
        'p_value': p_value,
        'large_sample_conditions': large_sample_ok,
        'confidence_interval': ci,
        'proportion_difference': prop_diff,
        'risk_ratio': risk_ratio,
        'odds_ratio': odds_ratio,
        'decision': 'Reject H0' if p_value < alpha else 'Fail to reject H0'
    }


def wilcoxon_rank_sum_test(group1, group2, alpha=0.05):
    """
    Perform Wilcoxon rank-sum test (Mann-Whitney U).
    """
    wilcox_stat, wilcox_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Compare with parametric test
    t_test_result = stats.ttest_ind(group1, group2)
    
    # Effect size (rank-biserial correlation)
    def rank_biserial_correlation(x1, x2):
        combined = np.concatenate([x1, x2])
        ranks = stats.rankdata(combined)
        n1 = len(x1)
        n2 = len(x2)
        ranks1 = ranks[:n1]
        U = np.sum(ranks1) - n1 * (n1 + 1) / 2
        r_rb = 1 - (2 * U) / (n1 * n2)
        return r_rb
    
    rank_biserial = rank_biserial_correlation(group1, group2)
    
    return {
        'u_statistic': wilcox_stat,
        'p_value': wilcox_p,
        't_test_p_value': t_test_result.pvalue,
        'rank_biserial_correlation': rank_biserial,
        'decision': 'Reject H0' if wilcox_p < alpha else 'Fail to reject H0'
    }


def wilcoxon_signed_rank_test(before, after, alpha=0.05):
    """
    Perform Wilcoxon signed-rank test.
    """
    wilcox_stat, wilcox_p = stats.wilcoxon(after, before)
    
    # Compare with parametric test
    paired_t_test = stats.ttest_rel(after, before)
    
    # Effect size
    def rank_biserial_paired(differences):
        abs_diffs = np.abs(differences)
        ranks = stats.rankdata(abs_diffs)
        signed_ranks = np.where(differences > 0, ranks, -ranks)
        n = len(differences)
        r_rb = np.sum(signed_ranks) / (n * (n + 1) / 2)
        return r_rb
    
    differences = after - before
    rank_biserial = rank_biserial_paired(differences)
    
    return {
        'w_statistic': wilcox_stat,
        'p_value': wilcox_p,
        'paired_t_p_value': paired_t_test.pvalue,
        'rank_biserial_correlation': rank_biserial,
        'decision': 'Reject H0' if wilcox_p < alpha else 'Fail to reject H0'
    }


def kruskal_wallis_test(groups, group_names=None, alpha=0.05):
    """
    Perform Kruskal-Wallis test.
    """
    kruskal_stat, kruskal_p = stats.kruskal(*groups)
    
    # Effect size (epsilon-squared)
    def epsilon_squared(h_stat, n, k):
        return (h_stat - k + 1) / (n - k)
    
    n_total = sum(len(g) for g in groups)
    k_groups = len(groups)
    epsilon_sq = epsilon_squared(kruskal_stat, n_total, k_groups)
    
    # Compare with parametric ANOVA
    f_stat, f_p = f_oneway(*groups)
    
    # Post-hoc analysis if significant
    post_hoc_results = {}
    if kruskal_p < alpha:
        n_comparisons = len(list(combinations(range(len(groups)), 2)))
        alpha_corrected = alpha / n_comparisons
        
        for i, j in combinations(range(len(groups)), 2):
            u_stat, u_p = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
            name1 = group_names[i] if group_names else f'Group {i}'
            name2 = group_names[j] if group_names else f'Group {j}'
            post_hoc_results[f'{name1} vs {name2}'] = {
                'u_statistic': u_stat,
                'p_value': u_p,
                'significant': u_p < alpha_corrected
            }
    
    return {
        'h_statistic': kruskal_stat,
        'p_value': kruskal_p,
        'epsilon_squared': epsilon_sq,
        'anova_f_statistic': f_stat,
        'anova_p_value': f_p,
        'post_hoc_results': post_hoc_results,
        'decision': 'Reject H0' if kruskal_p < alpha else 'Fail to reject H0'
    }


def one_way_anova(groups, group_names=None, alpha=0.05):
    """
    Perform one-way ANOVA with post-hoc tests.
    """
    f_stat, f_p = f_oneway(*groups)
    
    # Post-hoc tests using Tukey's HSD
    all_data = np.concatenate(groups)
    if group_names:
        group_labels = np.repeat(group_names, [len(g) for g in groups])
    else:
        group_labels = np.repeat([f'Group {i}' for i in range(len(groups))], [len(g) for g in groups])
    
    tukey = pairwise_tukeyhsd(all_data, group_labels)
    
    return {
        'f_statistic': f_stat,
        'p_value': f_p,
        'tukey_results': tukey,
        'decision': 'Reject H0' if f_p < alpha else 'Fail to reject H0'
    }


def chi_square_test(data, row_col, col_col, alpha=0.05):
    """
    Perform chi-square test of independence.
    """
    contingency_table = pd.crosstab(data[row_col], data[col_col])
    chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
    
    return {
        'chi_square_statistic': chi2_stat,
        'p_value': chi2_p,
        'degrees_of_freedom': dof,
        'contingency_table': contingency_table,
        'expected_frequencies': expected,
        'decision': 'Reject H0' if chi2_p < alpha else 'Fail to reject H0'
    }


def cohens_d_two_sample(x1, x2):
    """
    Calculate Cohen's d for two independent samples.
    """
    pooled_std = np.sqrt(((len(x1) - 1) * x1.var() + (len(x2) - 1) * x2.var()) / (len(x1) + len(x2) - 2))
    return (x1.mean() - x2.mean()) / pooled_std


def power_analysis_t_test(effect_size=0.5, alpha=0.05, power=0.8):
    """
    Perform power analysis for t-test.
    """
    power_analysis = TTestPower()
    sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power)
    
    # Power for different scenarios
    effect_sizes = [0.2, 0.5, 0.8]
    sample_sizes = [20, 50, 100]
    
    power_matrix = {}
    for d in effect_sizes:
        for n in sample_sizes:
            power_val = power_analysis.power(effect_size=d, nobs=n, alpha=alpha)
            power_matrix[(d, n)] = power_val
    
    return {
        'required_sample_size': sample_size,
        'power_matrix': power_matrix
    }


def multiple_testing_correction(p_values, alpha=0.05, method='bonferroni'):
    """
    Apply multiple testing correction.
    """
    rejected, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method=method)
    
    return {
        'original_p_values': p_values,
        'corrected_p_values': corrected_p_values,
        'significant': rejected,
        'method': method
    }


def check_t_test_assumptions(x, y=None):
    """
    Check assumptions for t-test.
    """
    results = {}
    
    if y is None:
        # One-sample t-test
        shapiro_stat, shapiro_p = stats.shapiro(x)
        results['normality_test'] = (shapiro_stat, shapiro_p)
        results['sample_size'] = len(x)
    else:
        # Two-sample t-test
        shapiro_x_stat, shapiro_x_p = stats.shapiro(x)
        shapiro_y_stat, shapiro_y_p = stats.shapiro(y)
        results['normality_tests'] = [(shapiro_x_stat, shapiro_x_p), (shapiro_y_stat, shapiro_y_p)]
        
        variance_ratio = x.var() / y.var()
        results['variance_ratio'] = variance_ratio
        results['levene_test'] = stats.levene(x, y)
    
    return results


def format_test_results(test_statistic, p_value, test_name):
    """
    Format test results for reporting.
    """
    if p_value < 0.001:
        significance = "p < 0.001"
    elif p_value < 0.01:
        significance = "p < 0.01"
    elif p_value < 0.05:
        significance = "p < 0.05"
    else:
        significance = "p >= 0.05"
    
    return {
        'test_name': test_name,
        'test_statistic': test_statistic,
        'p_value': p_value,
        'significance': significance
    }


def practical_examples():
    """
    Demonstrate practical examples of hypothesis testing.
    """
    np.random.seed(123)
    
    # Example 1: Drug Efficacy Study
    placebo = np.random.normal(50, 10, 30)
    treatment = np.random.normal(55, 10, 30)
    drug_results = stats.ttest_ind(treatment, placebo, alternative='greater')
    
    # Example 2: Customer Satisfaction Survey
    store_a = np.random.normal(7.5, 1.2, 50)
    store_b = np.random.normal(7.8, 1.1, 50)
    satisfaction_results = stats.ttest_ind(store_a, store_b)
    
    # Example 3: A/B Testing
    version_a_conversions = np.random.binomial(1, 0.05, 1000)
    version_b_conversions = np.random.binomial(1, 0.06, 1000)
    contingency_table = np.array([[np.sum(version_a_conversions), len(version_a_conversions) - np.sum(version_a_conversions)],
                                  [np.sum(version_b_conversions), len(version_b_conversions) - np.sum(version_b_conversions)]])
    ab_results = chi2_contingency(contingency_table)
    
    return {
        'drug_efficacy': {
            't_statistic': drug_results.statistic,
            'p_value': drug_results.pvalue,
            'effect_size': cohens_d_two_sample(treatment, placebo)
        },
        'customer_satisfaction': {
            't_statistic': satisfaction_results.statistic,
            'p_value': satisfaction_results.pvalue
        },
        'ab_testing': {
            'chi_square_statistic': ab_results[0],
            'p_value': ab_results[1],
            'degrees_of_freedom': ab_results[2]
        }
    }


def main():
    """
    Main function to demonstrate all hypothesis testing functions.
    """
    # Load data
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    sepal_length = data['sepal length (cm)']
    
    print("=== Hypothesis Testing Demonstrations ===\n")
    
    # Basic concepts
    print("1. Coin Fairness Test:")
    coin_results = coin_fairness_test()
    print(f"   Z-statistic: {coin_results['z_statistic']:.3f}")
    print(f"   P-value: {coin_results['p_value']:.4f}")
    print(f"   Decision: {coin_results['decision']}\n")
    
    # One-sample tests
    print("2. One-Sample t-Test:")
    t_results = one_sample_t_test(sepal_length, 5.5)
    print(f"   t-statistic: {t_results['t_statistic']:.3f}")
    print(f"   P-value: {t_results['p_value']:.4f}")
    print(f"   Effect size: {t_results['effect_size']:.3f}\n")
    
    # Two-sample tests
    setosa = data[data['target'] == 0]['sepal length (cm)']
    versicolor = data[data['target'] == 1]['sepal length (cm)']
    
    print("3. Independent t-Test:")
    indep_results = independent_t_test(setosa, versicolor)
    print(f"   Welch t-statistic: {indep_results['welch_t_test'][0]:.3f}")
    print(f"   P-value: {indep_results['welch_t_test'][1]:.4f}")
    print(f"   Effect size: {indep_results['effect_size']:.3f}\n")
    
    # Paired t-test
    np.random.seed(123)
    before = np.random.normal(100, 15, 20)
    after = before + np.random.normal(5, 10, 20)
    
    print("4. Paired t-Test:")
    paired_results = paired_t_test(before, after)
    print(f"   t-statistic: {paired_results['t_statistic']:.3f}")
    print(f"   P-value: {paired_results['p_value']:.4f}")
    print(f"   Effect size: {paired_results['effect_size']:.3f}\n")
    
    # Nonparametric tests
    print("5. Wilcoxon Rank-Sum Test:")
    wilcox_results = wilcoxon_rank_sum_test(setosa, versicolor)
    print(f"   U-statistic: {wilcox_results['u_statistic']:.3f}")
    print(f"   P-value: {wilcox_results['p_value']:.4f}\n")
    
    # Multiple testing correction
    print("6. Multiple Testing Correction:")
    p_values = [0.01, 0.03, 0.05, 0.08, 0.12]
    correction_results = multiple_testing_correction(p_values)
    print(f"   Original p-values: {p_values}")
    print(f"   Bonferroni corrected: {correction_results['corrected_p_values']}\n")
    
    # Power analysis
    print("7. Power Analysis:")
    power_results = power_analysis_t_test()
    print(f"   Required sample size for 80% power: {power_results['required_sample_size']:.0f}\n")
    
    # Practical examples
    print("8. Practical Examples:")
    examples = practical_examples()
    print(f"   Drug efficacy p-value: {examples['drug_efficacy']['p_value']:.4f}")
    print(f"   Customer satisfaction p-value: {examples['customer_satisfaction']['p_value']:.4f}")
    print(f"   A/B testing p-value: {examples['ab_testing']['p_value']:.4f}\n")
    
    print("=== End of Demonstrations ===")


if __name__ == "__main__":
    main() 