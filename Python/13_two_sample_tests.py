import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris
from statsmodels.stats.power import TTestPower
import matplotlib.pyplot as plt
import seaborn as sns


def independent_samples_t_test():
    """
    Demonstrates an independent samples t-test (Welch's and pooled),
    including manual calculations and confidence intervals.
    """
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    species_0 = iris_df[iris_df['target'] == 0]['sepal length (cm)']
    species_1 = iris_df[iris_df['target'] == 1]['sepal length (cm)']
    t_stat, p_val = stats.ttest_ind(species_0, species_1, equal_var=False)
    n1, n2 = len(species_0), len(species_1)
    mean1, mean2 = species_0.mean(), species_1.mean()
    var1, var2 = species_0.var(), species_1.var()
    se_welch = np.sqrt(var1/n1 + var2/n2)
    manual_t = (mean1 - mean2) / se_welch
    df_welch = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    t_critical = stats.t.ppf(0.975, df_welch)
    ci_lower = (mean1 - mean2) - t_critical * se_welch
    ci_upper = (mean1 - mean2) + t_critical * se_welch
    print("Test Results:")
    print(f"Mean difference (Species 0 - Species 1): {mean1 - mean2:.3f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_val:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print("\nManual Calculation Verification:")
    print(f"Group 1 (Species 0): n = {n1}, mean = {mean1:.3f}, var = {var1:.3f}")
    print(f"Group 2 (Species 1): n = {n2}, mean = {mean2:.3f}, var = {var2:.3f}")
    print(f"Standard Error (Welch): {se_welch:.3f}")
    print(f"Manual t-statistic: {manual_t:.3f}")
    print(f"Degrees of freedom: {df_welch:.1f}")
    return t_stat, p_val, ci_lower, ci_upper

def paired_samples_t_test():
    """
    Demonstrates paired samples t-test for dependent groups.
    Includes manual calculations, confidence intervals, and effect size.
    """
    # Generate paired data (before/after treatment)
    np.random.seed(42)
    before = np.random.normal(50, 10, 30)
    after = before + np.random.normal(5, 3, 30)  # Treatment effect
    
    # Calculate differences
    differences = after - before
    
    # Manual calculations
    n = len(differences)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    t_stat = mean_diff / se_diff
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Confidence interval
    t_critical = stats.t.ppf(0.975, df)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    # Effect size (Cohen's d for paired samples)
    cohens_d = mean_diff / std_diff
    
    print("=== Paired Samples t-Test ===")
    print(f"Sample size: {n}")
    print(f"Mean difference: {mean_diff:.3f}")
    print(f"Standard deviation of differences: {std_diff:.3f}")
    print(f"Standard error: {se_diff:.3f}")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"Degrees of freedom: {df}")
    print(f"p-value: {p_value:.6f}")
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"Cohen's d: {cohens_d:.3f}")
    
    # Using scipy
    t_stat_scipy, p_value_scipy = stats.ttest_rel(after, before)
    print(f"\nSciPy results:")
    print(f"t-statistic: {t_stat_scipy:.3f}")
    print(f"p-value: {p_value_scipy:.6f}")
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'ci': (ci_lower, ci_upper)
    }


def mann_whitney_u_test():
    """
    Demonstrates Mann-Whitney U test for independent samples.
    Nonparametric alternative to independent t-test.
    """
    # Generate two independent samples
    np.random.seed(42)
    group1 = np.random.normal(50, 10, 25)
    group2 = np.random.normal(55, 12, 30)
    
    # Manual calculation
    n1, n2 = len(group1), len(group2)
    
    # Combine and rank all data
    combined = np.concatenate([group1, group2])
    ranks = stats.rankdata(combined)
    
    # Sum ranks for each group
    ranks1 = ranks[:n1]
    ranks2 = ranks[n1:]
    sum_ranks1 = np.sum(ranks1)
    sum_ranks2 = np.sum(ranks2)
    
    # Calculate U statistics
    U1 = sum_ranks1 - (n1 * (n1 + 1)) / 2
    U2 = sum_ranks2 - (n2 * (n2 + 1)) / 2
    
    # Use the smaller U value
    U = min(U1, U2)
    
    # Expected value and standard deviation
    mu_U = (n1 * n2) / 2
    sigma_U = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
    
    # Z-score and p-value
    z_score = (U - mu_U) / sigma_U
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Effect size (r = Z/√N)
    N = n1 + n2
    r_effect_size = abs(z_score) / np.sqrt(N)
    
    print("=== Mann-Whitney U Test ===")
    print(f"Group 1 size: {n1}, Group 2 size: {n2}")
    print(f"U1: {U1:.1f}, U2: {U2:.1f}")
    print(f"U statistic: {U:.1f}")
    print(f"Expected U: {mu_U:.1f}")
    print(f"Standard deviation: {sigma_U:.3f}")
    print(f"Z-score: {z_score:.3f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Effect size (r): {r_effect_size:.3f}")
    
    # Using scipy
    u_stat, p_value_scipy = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    print(f"\nSciPy results:")
    print(f"U statistic: {u_stat:.1f}")
    print(f"p-value: {p_value_scipy:.6f}")
    
    return {
        'u_stat': U,
        'p_value': p_value,
        'effect_size': r_effect_size
    }


def wilcoxon_signed_rank_test():
    """
    Demonstrates Wilcoxon signed-rank test for paired samples.
    Nonparametric alternative to paired t-test.
    """
    # Generate paired data
    np.random.seed(42)
    before = np.random.normal(50, 10, 25)
    after = before + np.random.normal(3, 4, 25)
    
    # Calculate differences
    differences = after - before
    
    # Manual calculation
    # Remove zero differences
    non_zero_diffs = differences[differences != 0]
    n = len(non_zero_diffs)
    
    # Calculate absolute differences and ranks
    abs_diffs = np.abs(non_zero_diffs)
    ranks = stats.rankdata(abs_diffs)
    
    # Assign signs back
    signs = np.sign(non_zero_diffs)
    signed_ranks = signs * ranks
    
    # Calculate W+ and W-
    w_plus = np.sum(signed_ranks[signed_ranks > 0])
    w_minus = abs(np.sum(signed_ranks[signed_ranks < 0]))
    
    # Use the smaller value
    W = min(w_plus, w_minus)
    
    # Expected value and standard deviation
    mu_W = (n * (n + 1)) / 4
    sigma_W = np.sqrt((n * (n + 1) * (2 * n + 1)) / 24)
    
    # Z-score and p-value
    z_score = (W - mu_W) / sigma_W
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Effect size
    r_effect_size = abs(z_score) / np.sqrt(n)
    
    print("=== Wilcoxon Signed-Rank Test ===")
    print(f"Sample size (non-zero differences): {n}")
    print(f"W+: {w_plus:.1f}, W-: {w_minus:.1f}")
    print(f"W statistic: {W:.1f}")
    print(f"Expected W: {mu_W:.1f}")
    print(f"Standard deviation: {sigma_W:.3f}")
    print(f"Z-score: {z_score:.3f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Effect size (r): {r_effect_size:.3f}")
    
    # Using scipy
    w_stat, p_value_scipy = stats.wilcoxon(before, after)
    print(f"\nSciPy results:")
    print(f"W statistic: {w_stat:.1f}")
    print(f"p-value: {p_value_scipy:.6f}")
    
    return {
        'w_stat': W,
        'p_value': p_value,
        'effect_size': r_effect_size
    }


def effect_size_calculations():
    """
    Demonstrates various effect size measures for two-sample tests.
    """
    # Generate sample data
    np.random.seed(42)
    group1 = np.random.normal(50, 10, 30)
    group2 = np.random.normal(55, 12, 35)
    
    # Cohen's d (independent samples)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                         (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    cohens_d = (np.mean(group2) - np.mean(group1)) / pooled_std
    
    # Hedges' g (bias-corrected Cohen's d)
    df = n1 + n2 - 2
    correction_factor = 1 - (3 / (4 * df - 1))
    hedges_g = cohens_d * correction_factor
    
    # Glass's delta (using control group SD)
    glass_delta = (np.mean(group2) - np.mean(group1)) / np.std(group1, ddof=1)
    
    # Common Language Effect Size (CLES)
    # Probability that a randomly selected score from group 2 is greater than from group 1
    from scipy.stats import norm
    cles = norm.cdf(cohens_d / np.sqrt(2))
    
    # Point-biserial correlation
    # For binary grouping variable
    all_scores = np.concatenate([group1, group2])
    group_labels = np.concatenate([np.zeros(n1), np.ones(n2)])
    point_biserial = stats.pointbiserialr(group_labels, all_scores)[0]
    
    print("=== Effect Size Measures ===")
    print(f"Cohen's d: {cohens_d:.3f}")
    print(f"Hedges' g: {hedges_g:.3f}")
    print(f"Glass's delta: {glass_delta:.3f}")
    print(f"Common Language Effect Size: {cles:.3f}")
    print(f"Point-biserial correlation: {point_biserial:.3f}")
    
    # Effect size interpretation
    print(f"\nEffect Size Interpretation:")
    if abs(cohens_d) < 0.2:
        print("Small effect size")
    elif abs(cohens_d) < 0.5:
        print("Medium effect size")
    else:
        print("Large effect size")
    
    return {
        'cohens_d': cohens_d,
        'hedges_g': hedges_g,
        'glass_delta': glass_delta,
        'cles': cles,
        'point_biserial': point_biserial
    }


def power_analysis():
    """
    Demonstrates power analysis for two-sample t-tests.
    """
    from statsmodels.stats.power import TTestPower
    
    # Initialize power analysis
    power_analysis = TTestPower()
    
    # Parameters
    alpha = 0.05
    effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large
    sample_sizes = np.arange(10, 101, 10)
    
    print("=== Power Analysis for Two-Sample t-Test ===")
    print(f"Alpha level: {alpha}")
    
    # Power curves for different effect sizes
    for effect_size in effect_sizes:
        print(f"\nEffect size (Cohen's d): {effect_size}")
        powers = power_analysis.power(effect_size, sample_sizes, alpha)
        
        # Find sample size needed for 80% power
        target_power = 0.8
        required_n = power_analysis.solve_power(effect_size, power=target_power, alpha=alpha)
        
        print(f"Sample size per group needed for 80% power: {required_n:.0f}")
        print(f"Power with n=30 per group: {powers[2]:.3f}")
        print(f"Power with n=50 per group: {powers[4]:.3f}")
    
    # Effect size needed for given power and sample size
    n_per_group = 30
    target_power = 0.8
    required_effect = power_analysis.solve_power(power=target_power, nobs1=n_per_group, alpha=alpha)
    print(f"\nEffect size needed for 80% power with n={n_per_group} per group: {required_effect:.3f}")
    
    return {
        'power_analysis': power_analysis,
        'required_sample_sizes': {es: power_analysis.solve_power(es, power=0.8, alpha=alpha) 
                                for es in effect_sizes}
    }


def assumption_checking():
    """
    Demonstrates checking assumptions for two-sample tests.
    """
    # Generate sample data
    np.random.seed(42)
    group1 = np.random.normal(50, 10, 30)
    group2 = np.random.normal(55, 12, 35)
    
    print("=== Assumption Checking ===")
    
    # Normality tests
    print("\n1. Normality Tests:")
    for i, (group, name) in enumerate([(group1, "Group 1"), (group2, "Group 2")]):
        shapiro_stat, shapiro_p = stats.shapiro(group)
        print(f"{name}: Shapiro-Wilk p-value = {shapiro_p:.4f}")
    
    # Homogeneity of variance (Levene's test)
    levene_stat, levene_p = stats.levene(group1, group2)
    print(f"\n2. Homogeneity of Variance (Levene's test):")
    print(f"Statistic: {levene_stat:.3f}, p-value: {levene_p:.4f}")
    
    # Independence (assumed based on study design)
    print(f"\n3. Independence: Assumed based on study design")
    
    # Sample size adequacy
    print(f"\n4. Sample Size Adequacy:")
    print(f"Group 1: n = {len(group1)}")
    print(f"Group 2: n = {len(group2)}")
    print(f"Total: N = {len(group1) + len(group2)}")
    
    # Recommendations
    print(f"\n5. Recommendations:")
    if levene_p < 0.05:
        print("- Use Welch's t-test (unequal variances)")
    else:
        print("- Pooled t-test is appropriate")
    
    if any([stats.shapiro(group)[1] < 0.05 for group in [group1, group2]]):
        print("- Consider nonparametric alternatives")
    
    return {
        'normality_tests': [(stats.shapiro(group1), stats.shapiro(group2))],
        'levene_test': (levene_stat, levene_p)
    }


def robust_methods():
    """
    Demonstrates robust alternatives to traditional two-sample tests.
    """
    # Generate data with outliers
    np.random.seed(42)
    group1 = np.random.normal(50, 10, 30)
    group2 = np.random.normal(55, 12, 35)
    
    # Add outliers
    group1[0] = 100  # Outlier in group 1
    group2[0] = 20   # Outlier in group 2
    
    print("=== Robust Methods ===")
    
    # Trimmed means t-test
    trim_proportion = 0.1
    trimmed_group1 = stats.trim_mean(group1, trim_proportion)
    trimmed_group2 = stats.trim_mean(group2, trim_proportion)
    
    print(f"\n1. Trimmed Means ({trim_proportion*100}% trimmed):")
    print(f"Group 1: {trimmed_group1:.2f} (original: {np.mean(group1):.2f})")
    print(f"Group 2: {trimmed_group2:.2f} (original: {np.mean(group2):.2f})")
    
    # Winsorized t-test
    from scipy.stats.mstats import winsorize
    winsorized_group1 = winsorize(group1, limits=[0.1, 0.1])
    winsorized_group2 = winsorize(group2, limits=[0.1, 0.1])
    
    winsorized_t, winsorized_p = stats.ttest_ind(winsorized_group1, winsorized_group2)
    print(f"\n2. Winsorized t-test:")
    print(f"t-statistic: {winsorized_t:.3f}, p-value: {winsorized_p:.4f}")
    
    # Bootstrap confidence interval
    def bootstrap_diff(data1, data2, n_bootstrap=1000):
        diffs = []
        for _ in range(n_bootstrap):
            boot1 = np.random.choice(data1, size=len(data1), replace=True)
            boot2 = np.random.choice(data2, size=len(data2), replace=True)
            diffs.append(np.mean(boot2) - np.mean(boot1))
        return np.percentile(diffs, [2.5, 97.5])
    
    bootstrap_ci = bootstrap_diff(group1, group2)
    print(f"\n3. Bootstrap 95% CI for mean difference: [{bootstrap_ci[0]:.3f}, {bootstrap_ci[1]:.3f}]")
    
    # Permutation test
    def permutation_test(data1, data2, n_permutations=1000):
        observed_diff = np.mean(data2) - np.mean(data1)
        combined = np.concatenate([data1, data2])
        n1 = len(data1)
        
        perm_diffs = []
        for _ in range(n_permutations):
            perm_indices = np.random.permutation(len(combined))
            perm_group1 = combined[perm_indices[:n1]]
            perm_group2 = combined[perm_indices[n1:]]
            perm_diffs.append(np.mean(perm_group2) - np.mean(perm_group1))
        
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        return observed_diff, p_value
    
    perm_diff, perm_p = permutation_test(group1, group2)
    print(f"\n4. Permutation test:")
    print(f"Observed difference: {perm_diff:.3f}")
    print(f"p-value: {perm_p:.4f}")
    
    return {
        'trimmed_means': (trimmed_group1, trimmed_group2),
        'winsorized_test': (winsorized_t, winsorized_p),
        'bootstrap_ci': bootstrap_ci,
        'permutation_test': (perm_diff, perm_p)
    }


def practical_examples():
    """
    Demonstrates practical applications of two-sample tests.
    """
    print("=== Practical Examples ===")
    
    # Example 1: Treatment effectiveness
    print("\n1. Treatment Effectiveness Study:")
    print("Comparing test scores between treatment and control groups")
    
    # Generate treatment data
    np.random.seed(42)
    control_scores = np.random.normal(70, 15, 40)
    treatment_scores = control_scores + np.random.normal(8, 5, 40)  # Treatment effect
    
    t_stat, p_val = stats.ttest_ind(treatment_scores, control_scores)
    cohens_d = (np.mean(treatment_scores) - np.mean(control_scores)) / np.sqrt(
        ((len(treatment_scores) - 1) * np.var(treatment_scores, ddof=1) + 
         (len(control_scores) - 1) * np.var(control_scores, ddof=1)) / 
        (len(treatment_scores) + len(control_scores) - 2))
    
    print(f"Control mean: {np.mean(control_scores):.1f}")
    print(f"Treatment mean: {np.mean(treatment_scores):.1f}")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")
    
    # Example 2: Gender differences
    print("\n2. Gender Differences Study:")
    print("Comparing salary between male and female employees")
    
    male_salary = np.random.normal(60000, 15000, 50)
    female_salary = np.random.normal(55000, 12000, 45)
    
    t_stat, p_val = stats.ttest_ind(male_salary, female_salary)
    print(f"Male mean salary: ${np.mean(male_salary):.0f}")
    print(f"Female mean salary: ${np.mean(female_salary):.0f}")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
    
    # Example 3: Before/After intervention
    print("\n3. Before/After Intervention Study:")
    print("Comparing blood pressure before and after medication")
    
    before_bp = np.random.normal(140, 20, 30)
    after_bp = before_bp - np.random.normal(15, 8, 30)  # Medication effect
    
    t_stat, p_val = stats.ttest_rel(after_bp, before_bp)
    print(f"Before mean BP: {np.mean(before_bp):.1f}")
    print(f"After mean BP: {np.mean(after_bp):.1f}")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
    
    return {
        'treatment_study': (t_stat, p_val, cohens_d),
        'gender_study': (t_stat, p_val),
        'intervention_study': (t_stat, p_val)
    }


if __name__ == "__main__":
    print("Two-Sample Tests Demonstration\n")
    print("=" * 50)
    
    # Run all demonstrations
    print("\n1. Independent Samples t-Test:")
    independent_results = independent_samples_t_test()
    
    print("\n2. Paired Samples t-Test:")
    paired_results = paired_samples_t_test()
    
    print("\n3. Mann-Whitney U Test:")
    mw_results = mann_whitney_u_test()
    
    print("\n4. Wilcoxon Signed-Rank Test:")
    wilcoxon_results = wilcoxon_signed_rank_test()
    
    print("\n5. Effect Size Calculations:")
    effect_results = effect_size_calculations()
    
    print("\n6. Power Analysis:")
    power_results = power_analysis()
    
    print("\n7. Assumption Checking:")
    assumption_results = assumption_checking()
    
    print("\n8. Robust Methods:")
    robust_results = robust_methods()
    
    print("\n9. Practical Examples:")
    practical_results = practical_examples()
    
    print("\n" + "=" * 50)
    print("All demonstrations completed successfully!") 