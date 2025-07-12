"""
Chi-Square Tests Toolkit

This module provides functions and demonstrations for chi-square tests: goodness of fit, independence, homogeneity, effect sizes, assumption checking, Fisher's exact, likelihood ratio, post hoc, power analysis, reporting, and practical examples.

Each function is documented and can be referenced from the corresponding theory in the markdown file.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Chi-Square Goodness of Fit Test ---
def chi_square_goodness_of_fit(observed, expected, categories=None, plot=True):
    """
    Perform chi-square goodness of fit test and plot observed vs expected frequencies.
    """
    chi_square_test = stats.chisquare(observed, f_exp=expected)
    print("Chi-Square Goodness of Fit Test Results:")
    print(f"Chi-square statistic: {chi_square_test.statistic:.3f}")
    print(f"p-value: {chi_square_test.pvalue:.4f}")
    print(f"Expected frequencies: {expected}")
    print(f"Observed frequencies: {observed}")
    # Manual calculation
    residuals = (observed - expected) / np.sqrt(expected)
    print(f"Standardized residuals: {np.round(residuals, 3)}")
    if plot:
        if categories is None:
            categories = [f'Category {i+1}' for i in range(len(observed))]
        x = np.arange(len(categories))
        width = 0.35
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.bar(x - width/2, observed, width, label='Observed', alpha=0.8)
        plt.bar(x + width/2, expected, width, label='Expected', alpha=0.8)
        plt.xlabel('Categories')
        plt.ylabel('Frequency')
        plt.title('Observed vs Expected Frequencies')
        plt.xticks(x, categories)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.bar(categories, residuals)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Categories')
        plt.ylabel('Standardized Residuals')
        plt.title('Standardized Residuals')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    return chi_square_test

# --- 2. Chi-Square Test of Independence ---
def chi_square_independence(contingency_table, row_names=None, col_names=None, plot=True):
    """
    Perform chi-square test of independence for a contingency table and plot results.
    """
    result = stats.chi2_contingency(contingency_table)
    print("Chi-Square Test of Independence Results:")
    print(f"Chi-square statistic: {result.statistic:.3f}")
    print(f"Degrees of freedom: {result.dof}")
    print(f"p-value: {result.pvalue:.4f}")
    expected = result.expected
    residuals = (contingency_table - expected) / np.sqrt(expected)
    if row_names is None:
        row_names = [f'Row {i+1}' for i in range(contingency_table.shape[0])]
    if col_names is None:
        col_names = [f'Col {j+1}' for j in range(contingency_table.shape[1])]
    df_table = pd.DataFrame(contingency_table, index=row_names, columns=col_names)
    expected_df = pd.DataFrame(expected, index=row_names, columns=col_names)
    residuals_df = pd.DataFrame(residuals, index=row_names, columns=col_names)
    print("\nExpected Frequencies:")
    print(expected_df.round(2))
    print("\nStandardized Residuals:")
    print(residuals_df.round(3))
    significant_cells = np.where(np.abs(residuals) > 2)
    if len(significant_cells[0]) > 0:
        print("\nSignificant cells (|residual| > 2):")
        for i, j in zip(significant_cells[0], significant_cells[1]):
            print(f"  {row_names[i]} - {col_names[j]}: {residuals[i, j]:.3f}")
    if plot:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        sns.heatmap(df_table, annot=True, fmt='d', cmap='Blues')
        plt.title('Observed Frequencies')
        plt.subplot(1, 3, 2)
        sns.heatmap(expected_df, annot=True, fmt='.1f', cmap='Greens')
        plt.title('Expected Frequencies')
        plt.subplot(1, 3, 3)
        sns.heatmap(residuals_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
        plt.title('Standardized Residuals')
        plt.tight_layout()
        plt.show()
    return result

# --- 3. Chi-Square Test of Homogeneity ---
def chi_square_homogeneity(homogeneity_data, row_names=None, col_names=None, plot=True):
    """
    Perform chi-square test of homogeneity and plot observed frequencies and stacked bar chart.
    """
    result = stats.chi2_contingency(homogeneity_data)
    print("Chi-Square Test of Homogeneity Results:")
    print(f"Chi-square statistic: {result.statistic:.3f}")
    print(f"Degrees of freedom: {result.dof}")
    print(f"p-value: {result.pvalue:.4f}")
    if row_names is None:
        row_names = [f'Row {i+1}' for i in range(homogeneity_data.shape[0])]
    if col_names is None:
        col_names = [f'Col {j+1}' for j in range(homogeneity_data.shape[1])]
    df_homogeneity = pd.DataFrame(homogeneity_data, index=row_names, columns=col_names)
    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(df_homogeneity, annot=True, fmt='d', cmap='Blues')
        plt.title('Observed Frequencies')
        plt.subplot(1, 2, 2)
        df_homogeneity.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Stacked Bar Chart by Group')
        plt.xlabel('Group')
        plt.ylabel('Count')
        plt.legend(title='Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    return result 

# --- Effect Size Measures ---
def calculate_chi_square_effect_sizes(chi_square_result, n):
    """
    Calculate Cramér's V, Phi coefficient, and contingency coefficient for a chi-square result.
    """
    df = chi_square_result.dof
    min_dim = min(chi_square_result.expected.shape)
    cramers_v = np.sqrt(chi_square_result.statistic / (n * (min_dim - 1)))
    if chi_square_result.expected.shape == (2, 2):
        phi = np.sqrt(chi_square_result.statistic / n)
    else:
        phi = np.nan
    contingency_coef = np.sqrt(chi_square_result.statistic / (chi_square_result.statistic + n))
    return {
        'cramers_v': cramers_v,
        'phi': phi,
        'contingency_coef': contingency_coef
    }

def interpret_cramers_v(v):
    if v < 0.1:
        return "Negligible effect"
    elif v < 0.3:
        return "Small effect"
    elif v < 0.5:
        return "Medium effect"
    else:
        return "Large effect"

# --- Assumption Checking ---
def check_chi_square_assumptions(contingency_table):
    """
    Check chi-square test assumptions: expected frequencies, independence, etc.
    """
    chi_square_result = stats.chi2_contingency(contingency_table)
    expected_freq = chi_square_result.expected
    min_expected = np.min(expected_freq)
    low_expected_cells = np.where(expected_freq < 5)
    print("Expected frequencies:")
    print(pd.DataFrame(expected_freq,
                      index=[f'Row {i+1}' for i in range(expected_freq.shape[0])],
                      columns=[f'Col {j+1}' for j in range(expected_freq.shape[1])]).round(2))
    print(f"Minimum expected frequency: {min_expected:.2f}")
    if len(low_expected_cells[0]) > 0:
        print("WARNING: Cells with expected frequency < 5:")
        for i, j in zip(low_expected_cells[0], low_expected_cells[1]):
            expected_val = expected_freq[i, j]
            print(f"  Row {i+1} - Col {j+1}: {expected_val:.2f}")
        print("Consider using Fisher's exact test or combining categories")
    else:
        print("All expected frequencies ≥ 5. Chi-square test is appropriate.")
    print("\nIndependence assumption: Data should be from independent observations.")
    return {
        'expected_frequencies': expected_freq,
        'min_expected': min_expected,
        'low_expected_cells': low_expected_cells
    }

# --- Fisher's Exact Test ---
def fishers_exact(contingency_table, plot=True):
    """
    Perform Fisher's exact test for a 2x2 table and plot results.
    """
    fisher_test = stats.fisher_exact(contingency_table)
    print("Fisher's Exact Test Results:")
    print(f"p-value: {fisher_test.pvalue:.4f}")
    print(f"Odds ratio: {fisher_test.statistic:.3f}")
    if plot and contingency_table.shape == (2, 2):
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.DataFrame(contingency_table,
                                index=['Row 1', 'Row 2'],
                                columns=['Col 1', 'Col 2']),
                   annot=True, fmt='d', cmap='Reds')
        plt.title("Fisher's Exact Test Table")
        plt.show()
    return fisher_test

# --- Likelihood Ratio Test ---
def likelihood_ratio_test(observed):
    """
    Perform likelihood ratio (G-test) for a contingency table.
    """
    row_totals = np.sum(observed, axis=1)
    col_totals = np.sum(observed, axis=0)
    total = np.sum(observed)
    expected = np.outer(row_totals, col_totals) / total
    lr_statistic = 2 * np.sum(observed * np.log(observed / expected))
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    p_value = 1 - stats.chi2.cdf(lr_statistic, df)
    print("Likelihood Ratio Test Results:")
    print(f"G-statistic: {lr_statistic:.3f}")
    print(f"Degrees of freedom: {df}")
    print(f"p-value: {p_value:.4f}")
    return {'statistic': lr_statistic, 'df': df, 'p_value': p_value}

# --- Post Hoc Analysis: Pairwise Chi-Square ---
def pairwise_chi_square(contingency_table, alpha=0.05):
    """
    Perform pairwise chi-square tests with Bonferroni correction for rows.
    """
    n_rows = contingency_table.shape[0]
    n_cols = contingency_table.shape[1]
    n_comparisons = int(n_rows * (n_rows - 1) / 2)
    alpha_corrected = alpha / n_comparisons
    results = []
    for i in range(n_rows - 1):
        for j in range(i + 1, n_rows):
            subtable = contingency_table[[i, j], :]
            test_result = stats.chi2_contingency(subtable)
            results.append({
                'comparison': f"Row {i+1} vs Row {j+1}",
                'chi_square': test_result.statistic,
                'p_value': test_result.pvalue,
                'significant': test_result.pvalue < alpha_corrected
            })
    return results

# --- Power Analysis ---
def power_analysis_chi_square(n, w, df, alpha=0.05):
    """
    Power analysis for chi-square test (approximate).
    """
    critical_value = stats.chi2.ppf(1 - alpha, df)
    power = 1 - stats.chi2.cdf(critical_value, df, ncp=n * w**2)
    target_power = 0.8
    required_n = int(critical_value / (w**2 * (1 - target_power)))
    print("Power Analysis Results:")
    print(f"Current power: {power:.3f}")
    print(f"Required sample size for 80% power: {required_n}")
    return {
        'power': power,
        'required_n': required_n,
        'effect_size': w,
        'alpha': alpha
    }

# --- Reporting ---
def generate_chi_square_report(chi_square_result, contingency_table, test_type="independence"):
    """
    Generate a comprehensive chi-square test report.
    """
    print("=== CHI-SQUARE TEST REPORT ===\n")
    print("CONTINGENCY TABLE:")
    print(pd.DataFrame(contingency_table))
    print()
    print("TEST RESULTS:")
    print(f"Test type: {test_type}")
    print(f"Chi-square statistic: {chi_square_result.statistic:.3f}")
    print(f"Degrees of freedom: {chi_square_result.dof}")
    print(f"p-value: {chi_square_result.pvalue:.4f}")
    effect_sizes = calculate_chi_square_effect_sizes(chi_square_result, np.sum(contingency_table))
    print(f"Cramer's V: {effect_sizes['cramers_v']:.3f}")
    print(f"Effect size interpretation: {interpret_cramers_v(effect_sizes['cramers_v'])}\n")
    print("EXPECTED FREQUENCIES:")
    print(pd.DataFrame(chi_square_result.expected).round(2))
    print()
    observed = contingency_table
    expected = chi_square_result.expected
    residuals = (observed - expected) / np.sqrt(expected)
    print("STANDARDIZED RESIDUALS:")
    print(pd.DataFrame(residuals).round(3))
    print()
    alpha = 0.05
    if chi_square_result.pvalue < alpha:
        print("CONCLUSION:")
        print(f"Reject the null hypothesis (p < {alpha})")
        if test_type == "independence":
            print("There is a significant relationship between the variables")
        elif test_type == "homogeneity":
            print("The proportions are significantly different across groups")
        else:
            print("The observed frequencies differ significantly from expected")
    else:
        print("CONCLUSION:")
        print(f"Fail to reject the null hypothesis (p >= {alpha})")
        print("There is insufficient evidence of a relationship")

# --- Practical Examples ---
def survey_analysis_example():
    """
    Simulate survey data and perform chi-square analysis.
    """
    np.random.seed(123)
    n_responses = 200
    age_group = np.random.choice(['18-25', '26-35', '36-45', '46+'], n_responses, p=[0.3, 0.35, 0.25, 0.1])
    satisfaction = np.random.choice(['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied'], n_responses, p=[0.4, 0.3, 0.2, 0.1])
    survey_data = pd.DataFrame({'Age Group': age_group, 'Satisfaction': satisfaction})
    survey_table = pd.crosstab(survey_data['Age Group'], survey_data['Satisfaction'])
    print("Survey Results:")
    print(survey_table)
    survey_test = stats.chi2_contingency(survey_table.values)
    survey_effect = calculate_chi_square_effect_sizes(survey_test, n_responses)
    print(f"Cramer's V: {survey_effect['cramers_v']:.3f}")
    print(f"Interpretation: {interpret_cramers_v(survey_effect['cramers_v'])}")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(survey_table, annot=True, fmt='d', cmap='Blues')
    plt.title('Survey Results')
    plt.subplot(1, 2, 2)
    survey_table.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Satisfaction by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.legend(title='Satisfaction')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def clinical_trial_example():
    """
    Simulate clinical trial data and perform chi-square and Fisher's exact tests.
    """
    np.random.seed(123)
    clinical_data = np.array([[45, 15], [30, 25]])
    row_names = ['Treatment', 'Control']
    col_names = ['Improved', 'No Improvement']
    df_clinical = pd.DataFrame(clinical_data, index=row_names, columns=col_names)
    print("Clinical Trial Results:")
    print(df_clinical)
    clinical_test = stats.chi2_contingency(clinical_data)
    clinical_fisher = stats.fisher_exact(clinical_data)
    clinical_effect = calculate_chi_square_effect_sizes(clinical_test, np.sum(clinical_data))
    print(f"Phi coefficient: {clinical_effect['phi']:.3f}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(df_clinical, annot=True, fmt='d', cmap='Blues')
    plt.title('Clinical Trial Results')
    plt.subplot(1, 2, 2)
    df_clinical.plot(kind='bar', ax=plt.gca())
    plt.title('Improvement by Treatment Group')
    plt.xlabel('Group')
    plt.ylabel('Count')
    plt.legend(title='Outcome')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def quality_control_example():
    """
    Simulate quality control data and perform chi-square analysis.
    """
    np.random.seed(123)
    quality_data = np.array([
        [85, 10, 5],
        [70, 20, 10],
        [60, 25, 15]
    ])
    row_names = ['Machine A', 'Machine B', 'Machine C']
    col_names = ['Excellent', 'Good', 'Poor']
    df_quality = pd.DataFrame(quality_data, index=row_names, columns=col_names)
    print("Quality Control Results:")
    print(df_quality)
    quality_test = stats.chi2_contingency(quality_data)
    quality_assumptions = check_chi_square_assumptions(quality_data)
    quality_effect = calculate_chi_square_effect_sizes(quality_test, np.sum(quality_data))
    print(f"Cramer's V: {quality_effect['cramers_v']:.3f}")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(df_quality, annot=True, fmt='d', cmap='Blues')
    plt.title('Quality Control Results')
    plt.subplot(1, 2, 2)
    df_quality.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Quality by Machine')
    plt.xlabel('Machine')
    plt.ylabel('Count')
    plt.legend(title='Quality')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Main Demonstration Block ---
if __name__ == "__main__":
    # Goodness of fit example
    observed = np.array([45, 52, 38, 65])
    expected = np.array([50, 50, 50, 50])
    chi_square_goodness_of_fit(observed, expected)
    # Independence example
    contingency_table = np.array([
        [45, 35, 25],
        [55, 40, 30],
        [30, 25, 15]
    ])
    row_names = ['Low', 'Medium', 'High']
    col_names = ['Group A', 'Group B', 'Group C']
    chi_square_independence(contingency_table, row_names, col_names)
    # Homogeneity example
    homogeneity_data = np.array([
        [20, 30, 25],
        [15, 25, 20],
        [10, 15, 10]
    ])
    row_names = ['Treatment A', 'Treatment B', 'Treatment C']
    col_names = ['Success', 'Partial', 'Failure']
    chi_square_homogeneity(homogeneity_data, row_names, col_names)
    # Effect size and interpretation
    result = stats.chi2_contingency(contingency_table)
    effect_sizes = calculate_chi_square_effect_sizes(result, np.sum(contingency_table))
    print(f"Cramer's V: {effect_sizes['cramers_v']:.3f}")
    print(f"Interpretation: {interpret_cramers_v(effect_sizes['cramers_v'])}")
    # Assumption checking
    check_chi_square_assumptions(contingency_table)
    # Fisher's exact test
    fishers_exact(np.array([[20, 15], [30, 35]]))
    # Likelihood ratio test
    likelihood_ratio_test(contingency_table)
    # Post hoc analysis
    pairwise_results = pairwise_chi_square(contingency_table)
    for result in pairwise_results:
        print(f"{result['comparison']}: Chi-square = {result['chi_square']:.3f}, p = {result['p_value']:.4f}, Significant: {result['significant']}")
    # Power analysis
    power_analysis_chi_square(n=100, w=0.3, df=1)
    # Reporting
    generate_chi_square_report(result, contingency_table, "independence")
    # Practical examples
    survey_analysis_example()
    clinical_trial_example()
    quality_control_example() 