"""
Nonparametric Tests Toolkit

This module provides functions and demonstrations for common nonparametric statistical tests, including Wilcoxon, Mann-Whitney U, Kruskal-Wallis, Friedman, Sign, McNemar, Chi-Square, Fisher's Exact, and effect size calculations.

Each function is documented and can be referenced from the corresponding theory in the markdown file.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# --- 1. Wilcoxon Signed-Rank Test ---
def wilcoxon_signed_rank(before, after, plot=True):
    """
    Perform Wilcoxon signed-rank test for paired samples and plot results.
    """
    statistic, p_value = stats.wilcoxon(before, after)
    print(f"Wilcoxon signed-rank test:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    if plot:
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(before, after)
        plt.plot([min(before), max(before)], [min(before), max(before)], 'r--', alpha=0.7)
        plt.xlabel('Before')
        plt.ylabel('After')
        plt.title('Paired Data')
        plt.subplot(1, 2, 2)
        differences = before - after
        plt.hist(differences, bins=5, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.xlabel('Difference (Before - After)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Differences')
        plt.tight_layout()
        plt.show()
    return statistic, p_value

# --- 2. Mann-Whitney U Test ---
def mann_whitney_u(group1, group2, plot=True):
    """
    Perform Mann-Whitney U test for two independent samples and plot results.
    """
    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    print(f"Mann-Whitney U test:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    if plot:
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.boxplot([group1, group2], labels=['Group 1', 'Group 2'])
        plt.ylabel('Values')
        plt.title('Boxplot Comparison')
        plt.subplot(1, 2, 2)
        all_data = np.concatenate([group1, group2])
        all_labels = ['Group 1'] * len(group1) + ['Group 2'] * len(group2)
        sns.violinplot(x=all_labels, y=all_data)
        plt.ylabel('Values')
        plt.title('Violin Plot Comparison')
        plt.tight_layout()
        plt.show()
    return statistic, p_value

# --- 3. Kruskal-Wallis Test ---
def kruskal_wallis(*groups, plot=True):
    """
    Perform Kruskal-Wallis test for more than two independent groups and plot results.
    """
    statistic, p_value = stats.kruskal(*groups)
    print(f"Kruskal-Wallis test:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.boxplot(groups, labels=[f'Group {i+1}' for i in range(len(groups))])
        plt.ylabel('Values')
        plt.title('Boxplot Comparison')
        plt.subplot(1, 2, 2)
        values = np.concatenate(groups)
        group_labels = sum([[f'Group {i+1}'] * len(g) for i, g in enumerate(groups)], [])
        sns.violinplot(x=group_labels, y=values)
        plt.ylabel('Values')
        plt.title('Violin Plot Comparison')
        plt.tight_layout()
        plt.show()
    return statistic, p_value

# --- 4. Friedman Test ---
def friedman_test(*treatments, plot=True):
    """
    Perform Friedman test for repeated measures and plot results.
    Each argument is a treatment (array-like), all of the same length (subjects).
    """
    statistic, p_value = stats.friedmanchisquare(*treatments)
    print(f"Friedman test:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    if plot:
        values = np.column_stack(treatments)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.boxplot([values[:, i] for i in range(values.shape[1])], 
                   labels=[f'Treatment {i+1}' for i in range(values.shape[1])])
        plt.ylabel('Values')
        plt.title('Boxplot by Treatment')
        plt.subplot(1, 2, 2)
        for i in range(values.shape[0]):
            plt.plot(range(1, values.shape[1]+1), values[i], 'o-', alpha=0.7, label=f'Subject {i+1}')
        plt.xlabel('Treatment')
        plt.ylabel('Values')
        plt.title('Individual Profiles')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    return statistic, p_value 

# --- 5. Sign Test ---
def sign_test(before, after, plot=True):
    """
    Perform the sign test for paired data and plot results.
    """
    diff = before - after
    n_pos = np.sum(diff > 0)
    n_total = np.sum(diff != 0)
    # Binomial test
    result = stats.binomtest(n_pos, n_total, p=0.5)
    p_value = result.pvalue
    print(f"Sign test:")
    print(f"Positive differences: {n_pos}")
    print(f"Total non-zero differences: {n_total}")
    print(f"P-value: {p_value:.4f}")
    if plot:
        plt.figure(figsize=(8, 6))
        signs = ['Positive' if d > 0 else 'Negative' for d in diff if d != 0]
        sign_counts = pd.Series(signs).value_counts()
        plt.bar(sign_counts.index, sign_counts.values)
        plt.ylabel('Count')
        plt.title('Sign Test Results')
        plt.show()
    return n_pos, n_total, p_value

# --- 6. McNemar's Test ---
def mcnemar_test(tab, plot=True):
    """
    Perform McNemar's test for paired binary data (2x2 table) and plot results.
    """
    result = stats.mcnemar(tab, exact=False)
    statistic = result.statistic
    p_value = result.pvalue
    print(f"McNemar's test:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    if plot:
        plt.figure(figsize=(8, 6))
        sns.heatmap(tab, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['After: No', 'After: Yes'],
                   yticklabels=['Before: No', 'Before: Yes'])
        plt.title('McNemar Test Contingency Table')
        plt.show()
    return statistic, p_value

# --- 7. Chi-Square Test of Independence ---
def chi_square_independence(tab, plot=True):
    """
    Perform chi-square test of independence for a contingency table and plot results.
    """
    statistic, p_value, dof, expected = stats.chi2_contingency(tab)
    print(f"Chi-square test of independence:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of freedom: {dof}")
    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(tab, annot=True, fmt='d', cmap='Blues')
        plt.title('Observed Frequencies')
        plt.subplot(1, 2, 2)
        sns.heatmap(expected, annot=True, fmt='.1f', cmap='Greens')
        plt.title('Expected Frequencies')
        plt.tight_layout()
        plt.show()
    return statistic, p_value, dof, expected

# --- 8. Fisher's Exact Test ---
def fishers_exact(tab, plot=True):
    """
    Perform Fisher's exact test for a 2x2 table and plot results.
    """
    odds_ratio, p_value = stats.fisher_exact(tab)
    print(f"Fisher's exact test:")
    print(f"Odds ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")
    if plot:
        plt.figure(figsize=(6, 6))
        sns.heatmap(tab, annot=True, fmt='d', cmap='Reds')
        plt.title("Fisher's Exact Test Table")
        plt.show()
    return odds_ratio, p_value

# --- Effect Size Calculations ---
def wilcox_effsize(x, y):
    """
    Calculate rank-biserial correlation effect size for Mann-Whitney U test.
    """
    statistic, _ = stats.mannwhitneyu(x, y, alternative='two-sided')
    n1, n2 = len(x), len(y)
    r = 1 - (2 * statistic) / (n1 * n2)
    print(f"Rank-biserial correlation: {r:.4f}")
    return r

def cramers_v(contingency_table):
    """
    Calculate Cramér's V for chi-square test.
    """
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = np.sum(contingency_table)
    k = min(contingency_table.shape)
    v = np.sqrt(chi2 / (n * (k - 1)))
    print(f"Cramér's V: {v:.4f}")
    return v

# --- Practical Example: Kruskal-Wallis and Post Hoc ---
def kruskal_posthoc_example():
    """
    Simulate data for 3 groups, perform Kruskal-Wallis test and post hoc pairwise Mann-Whitney U tests with Bonferroni correction. Visualize results.
    """
    np.random.seed(42)
    g1 = np.random.normal(10, 2, 20)
    g2 = np.random.normal(12, 2, 20)
    g3 = np.random.normal(15, 2, 20)
    values = np.concatenate([g1, g2, g3])
    group = np.repeat([1, 2, 3], 20)
    # Kruskal-Wallis test
    statistic, p_value = stats.kruskal(g1, g2, g3)
    print(f"Kruskal-Wallis test:")
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    # Post hoc pairwise Wilcoxon tests with Bonferroni correction
    groups = [g1, g2, g3]
    group_names = ['Group 1', 'Group 2', 'Group 3']
    print("\nPost hoc pairwise comparisons (Bonferroni corrected):")
    for i, (g1_name, g2_name) in enumerate(combinations(group_names, 2)):
        idx1, idx2 = i, i + 1
        stat, p = stats.mannwhitneyu(groups[idx1], groups[idx2], alternative='two-sided')
        p_corrected = min(p * 3, 1.0)
        print(f"{g1_name} vs {g2_name}: p = {p_corrected:.4f}")
    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.boxplot([g1, g2, g3], labels=['Group 1', 'Group 2', 'Group 3'])
    plt.ylabel('Values')
    plt.title('Group Comparison')
    plt.subplot(1, 2, 2)
    group_labels = ['Group 1'] * len(g1) + ['Group 2'] * len(g2) + ['Group 3'] * len(g3)
    sns.violinplot(x=group_labels, y=values)
    plt.ylabel('Values')
    plt.title('Distribution Comparison')
    plt.tight_layout()
    plt.show()

# --- Main Demonstration Block ---
if __name__ == "__main__":
    # Example data for paired tests
    before = np.array([120, 115, 130, 140, 125])
    after = np.array([118, 117, 128, 135, 130])
    wilcoxon_signed_rank(before, after)
    sign_test(before, after)
    # Example data for independent tests
    group1 = np.array([12, 15, 14, 10, 13])
    group2 = np.array([18, 20, 17, 16, 19])
    mann_whitney_u(group1, group2)
    wilcox_effsize(group1, group2)
    # Kruskal-Wallis and Friedman
    np.random.seed(42)
    g1 = np.random.normal(10, 2, 5)
    g2 = np.random.normal(12, 2, 5)
    g3 = np.random.normal(15, 2, 5)
    kruskal_wallis(g1, g2, g3)
    friedman_test(g1, g2, g3)
    # McNemar's test
    tab_mcnemar = np.array([[30, 10], [5, 55]])
    mcnemar_test(tab_mcnemar)
    # Chi-square and Fisher's exact
    tab_chi = np.array([[20, 15], [30, 35]])
    chi_square_independence(tab_chi)
    cramers_v(tab_chi)
    tab_fisher = np.array([[2, 3], [8, 7]])
    fishers_exact(tab_fisher)
    # Practical example
    kruskal_posthoc_example() 