import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, anderson, kstest, skew, kurtosis, trim_mean
from sklearn.datasets import load_iris
from sklearn.robust import HuberRegressor
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.power import TTestPower
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap


def basic_one_sample_t_test():
    """
    Demonstrates basic one-sample t-test with manual calculations.
    
    This function shows how to perform a one-sample t-test comparing
    a sample mean to a hypothesized population mean, including manual
    calculations for understanding the underlying mathematics.
    """
    # Load iris dataset as example
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    
    # Test if the mean sepal length is significantly different from 5.5
    sepal_length = data['sepal length (cm)']
    t_statistic, p_value = stats.ttest_1samp(sepal_length, 5.5)
    
    # Calculate confidence interval
    desc_stats = DescrStatsW(sepal_length)
    confidence_interval = desc_stats.tconfint_mean(alpha=0.05)
    sample_mean = sepal_length.mean()
    hypothesized_mean = 5.5
    
    print("Test Results:")
    print(f"Sample mean: {sample_mean:.3f}")
    print(f"Hypothesized mean: {hypothesized_mean}")
    print(f"t-statistic: {t_statistic:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"95% Confidence Interval: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
    
    # Manual calculation for understanding
    n = len(sepal_length)
    sample_sd = sepal_length.std(ddof=1)
    standard_error = sample_sd / np.sqrt(n)
    manual_t = (sample_mean - hypothesized_mean) / standard_error
    
    print(f"\nManual Calculation Verification:")
    print(f"Sample SD: {sample_sd:.3f}")
    print(f"Standard Error: {standard_error:.3f}")
    print(f"Manual t-statistic: {manual_t:.3f}")
    print(f"Degrees of freedom: {n - 1}")
    
    return {
        't_statistic': t_statistic,
        'p_value': p_value,
        'sample_mean': sample_mean,
        'confidence_interval': confidence_interval,
        'manual_t': manual_t,
        'n': n
    }


def t_test_alternatives():
    """
    Demonstrates one-sample t-test with different alternative hypotheses.
    
    Shows how to perform two-tailed and one-tailed tests, and explains
    the relationship between different p-values and critical values.
    """
    # Load data
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    sepal_length = data['sepal length (cm)']
    
    # Two-tailed test (default)
    two_tailed_stat, two_tailed_p = stats.ttest_1samp(sepal_length, 5.5)
    print("Two-tailed test:")
    print(f"t-statistic: {two_tailed_stat:.3f}, p-value: {two_tailed_p:.4f}")
    
    # One-tailed test (greater than)
    greater_stat, greater_p = stats.ttest_1samp(sepal_length, 5.5, alternative='greater')
    print("\nOne-tailed test (greater than):")
    print(f"t-statistic: {greater_stat:.3f}, p-value: {greater_p:.4f}")
    
    # One-tailed test (less than)
    less_stat, less_p = stats.ttest_1samp(sepal_length, 5.5, alternative='less')
    print("\nOne-tailed test (less than):")
    print(f"t-statistic: {less_stat:.3f}, p-value: {less_p:.4f}")
    
    # Compare p-values
    print(f"\nP-values comparison:")
    print(f"Two-tailed: {two_tailed_p:.4f}")
    print(f"Greater than: {greater_p:.4f}")
    print(f"Less than: {less_p:.4f}")
    
    # Relationship between p-values
    print(f"\nP-value relationships:")
    print(f"Two-tailed p-value ≈ 2 × min(one-tailed p-values)")
    print(f"Greater p-value + Less p-value = 1")
    
    # Critical values comparison
    alpha = 0.05
    df = len(sepal_length) - 1
    critical_two_tailed = stats.t.ppf(1 - alpha/2, df)
    critical_one_tailed = stats.t.ppf(1 - alpha, df)
    
    print(f"\nCritical values (α = 0.05):")
    print(f"Two-tailed critical t: {critical_two_tailed:.3f}")
    print(f"One-tailed critical t: {critical_one_tailed:.3f}")
    
    return {
        'two_tailed': (two_tailed_stat, two_tailed_p),
        'greater': (greater_stat, greater_p),
        'less': (less_stat, less_p),
        'critical_two_tailed': critical_two_tailed,
        'critical_one_tailed': critical_one_tailed
    }


def calculate_cohens_d(sample_data, hypothesized_mean):
    """
    Calculate Cohen's d effect size for one-sample t-test.
    
    Parameters:
    -----------
    sample_data : array-like
        Sample data
    hypothesized_mean : float
        Hypothesized population mean
        
    Returns:
    --------
    dict : Dictionary containing effect size measures and confidence intervals
    """
    sample_mean = sample_data.mean()
    sample_sd = sample_data.std(ddof=1)
    n = len(sample_data)
    
    # Cohen's d
    cohens_d = (sample_mean - hypothesized_mean) / sample_sd
    
    # Hedges' g (unbiased estimator)
    hedges_g = cohens_d * (1 - 3 / (4 * (n - 1) - 1))
    
    # Standard error of effect size
    se_d = np.sqrt(1/n + cohens_d**2/(2*n))
    
    # Confidence interval for effect size
    df = n - 1
    t_critical = stats.t.ppf(0.975, df)
    ci_lower = cohens_d - t_critical * se_d
    ci_upper = cohens_d + t_critical * se_d
    
    return {
        'cohens_d': cohens_d,
        'hedges_g': hedges_g,
        'se_d': se_d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'sample_mean': sample_mean,
        'sample_sd': sample_sd,
        'n': n
    }


def interpret_effect_size(d):
    """
    Interpret Cohen's d effect size.
    
    Parameters:
    -----------
    d : float
        Cohen's d effect size
        
    Returns:
    --------
    str : Interpretation of effect size
    """
    if abs(d) < 0.2:
        return "Small effect"
    elif abs(d) < 0.5:
        return "Medium effect"
    elif abs(d) < 0.8:
        return "Large effect"
    else:
        return "Very large effect"


def effect_size_analysis():
    """
    Demonstrates effect size calculation and interpretation for one-sample t-test.
    
    Shows how to calculate Cohen's d, Hedges' g, and confidence intervals
    for effect sizes, along with power analysis based on effect size.
    """
    # Load data
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    sepal_length = data['sepal length (cm)']
    
    # Apply to sepal length data
    sepal_effect_size = calculate_cohens_d(sepal_length, 5.5)
    
    print("Effect Size Analysis:")
    print(f"Cohen's d: {sepal_effect_size['cohens_d']:.3f}")
    print(f"Hedges' g: {sepal_effect_size['hedges_g']:.3f}")
    print(f"Standard Error of d: {sepal_effect_size['se_d']:.3f}")
    print(f"95% CI for d: [{sepal_effect_size['ci_lower']:.3f}, {sepal_effect_size['ci_upper']:.3f}]")
    
    print(f"Effect size interpretation: {interpret_effect_size(sepal_effect_size['cohens_d'])}")
    
    # Power analysis based on effect size
    power_analysis = TTestPower()
    power = power_analysis.power(effect_size=sepal_effect_size['cohens_d'], 
                               nobs=sepal_effect_size['n'], alpha=0.05)
    print(f"Power for current effect size: {power:.3f}")
    
    return sepal_effect_size


def one_sample_z_test(sample_data, hypothesized_mean, population_sd, alpha=0.05):
    """
    Perform one-sample z-test when population standard deviation is known.
    
    Parameters:
    -----------
    sample_data : array-like
        Sample data
    hypothesized_mean : float
        Hypothesized population mean
    population_sd : float
        Known population standard deviation
    alpha : float
        Significance level
        
    Returns:
    --------
    dict : Dictionary containing test results
    """
    sample_mean = sample_data.mean()
    n = len(sample_data)
    
    # Calculate z-statistic
    z_statistic = (sample_mean - hypothesized_mean) / (population_sd / np.sqrt(n))
    
    # Calculate p-value
    p_value_two_tailed = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    p_value_greater = 1 - stats.norm.cdf(z_statistic)
    p_value_less = stats.norm.cdf(z_statistic)
    
    # Calculate confidence interval
    margin_of_error = stats.norm.ppf(1 - alpha/2) * (population_sd / np.sqrt(n))
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    # Calculate effect size (Cohen's d using population SD)
    cohens_d = (sample_mean - hypothesized_mean) / population_sd
    
    return {
        'z_statistic': z_statistic,
        'p_value_two_tailed': p_value_two_tailed,
        'p_value_greater': p_value_greater,
        'p_value_less': p_value_less,
        'sample_mean': sample_mean,
        'hypothesized_mean': hypothesized_mean,
        'confidence_interval': [ci_lower, ci_upper],
        'margin_of_error': margin_of_error,
        'cohens_d': cohens_d,
        'n': n
    }


def z_test_demonstration():
    """
    Demonstrates one-sample z-test and comparison with t-test.
    
    Shows how to perform z-test when population standard deviation is known,
    and compares results with t-test to illustrate the differences.
    """
    # Load data
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    sepal_length = data['sepal length (cm)']
    
    # Example: Test if sepal length mean is different from 5.5 (assuming known population SD = 0.5)
    sepal_z_test = one_sample_z_test(sepal_length, 5.5, 0.5)
    
    print("One-Sample Z-Test Results:")
    print(f"Sample mean: {sepal_z_test['sample_mean']:.3f}")
    print(f"Hypothesized mean: {sepal_z_test['hypothesized_mean']}")
    print(f"Population SD: 0.5")
    print(f"z-statistic: {sepal_z_test['z_statistic']:.3f}")
    print(f"Two-tailed p-value: {sepal_z_test['p_value_two_tailed']:.4f}")
    print(f"95% Confidence Interval: [{sepal_z_test['confidence_interval'][0]:.3f}, {sepal_z_test['confidence_interval'][1]:.3f}]")
    print(f"Effect size (Cohen's d): {sepal_z_test['cohens_d']:.3f}")
    
    # Compare with t-test results
    sepal_t_stat, sepal_t_p = stats.ttest_1samp(sepal_length, 5.5)
    print(f"\nComparison with t-test:")
    print(f"t-statistic: {sepal_t_stat:.3f}")
    print(f"t-test p-value: {sepal_t_p:.4f}")
    print(f"z-test p-value: {sepal_z_test['p_value_two_tailed']:.4f}")
    
    return sepal_z_test


def wilcoxon_signed_rank_test():
    """
    Demonstrates Wilcoxon signed-rank test as nonparametric alternative to t-test.
    
    Shows how to perform the Wilcoxon test, manual calculations for understanding,
    and comparison with parametric t-test results.
    """
    # Load data
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    sepal_length = data['sepal length (cm)']
    
    # Wilcoxon signed-rank test (nonparametric alternative to t-test)
    wilcox_stat, wilcox_p = stats.wilcoxon(sepal_length - 5.5)
    print(f"Wilcoxon signed-rank test:")
    print(f"W-statistic: {wilcox_stat:.3f}")
    print(f"p-value: {wilcox_p:.4f}")
    
    # Manual calculation for understanding
    differences = sepal_length - 5.5
    abs_differences = np.abs(differences)
    ranks = stats.rankdata(abs_differences)
    signed_ranks = np.sign(differences) * ranks
    W_statistic = np.sum(signed_ranks)
    
    print(f"\nManual Wilcoxon Calculation:")
    print(f"W statistic: {W_statistic:.3f}")
    print(f"SciPy W statistic: {wilcox_stat:.3f}")
    
    # Compare with t-test results
    sepal_t_stat, sepal_t_p = stats.ttest_1samp(sepal_length, 5.5)
    print(f"\nComparison of t-test and Wilcoxon test:")
    print(f"t-test p-value: {sepal_t_p:.4f}")
    print(f"Wilcoxon p-value: {wilcox_p:.4f}")
    
    # Effect size for Wilcoxon test
    wilcox_effect_size = abs(stats.norm.ppf(wilcox_p / 2)) / np.sqrt(len(sepal_length))
    print(f"Wilcoxon effect size (r): {wilcox_effect_size:.3f}")
    
    # Interpret Wilcoxon effect size
    def interpret_wilcox_effect(r):
        if abs(r) < 0.1:
            return "Small effect"
        elif abs(r) < 0.3:
            return "Medium effect"
        elif abs(r) < 0.5:
            return "Large effect"
        else:
            return "Very large effect"
    
    print(f"Wilcoxon effect interpretation: {interpret_wilcox_effect(wilcox_effect_size)}")
    
    return {
        'wilcox_stat': wilcox_stat,
        'wilcox_p': wilcox_p,
        'manual_W': W_statistic,
        'effect_size': wilcox_effect_size
    }


def sign_test(sample_data, hypothesized_median):
    """
    Perform sign test as simplest nonparametric alternative.
    
    Parameters:
    -----------
    sample_data : array-like
        Sample data
    hypothesized_median : float
        Hypothesized median value
        
    Returns:
    --------
    dict : Dictionary containing test results
    """
    differences = sample_data - hypothesized_median
    positive_signs = np.sum(differences > 0)
    negative_signs = np.sum(differences < 0)
    n = positive_signs + negative_signs
    
    # Binomial test
    p_value_two_tailed = 2 * stats.binom.cdf(min(positive_signs, negative_signs), n, 0.5)
    p_value_greater = 1 - stats.binom.cdf(positive_signs - 1, n, 0.5)
    p_value_less = stats.binom.cdf(positive_signs, n, 0.5)
    
    # Normal approximation for large samples
    if n > 20:
        z_statistic = (positive_signs - n/2) / np.sqrt(n/4)
        p_value_normal = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    else:
        z_statistic = np.nan
        p_value_normal = np.nan
    
    return {
        'positive_signs': positive_signs,
        'negative_signs': negative_signs,
        'n': n,
        'p_value_two_tailed': p_value_two_tailed,
        'p_value_greater': p_value_greater,
        'p_value_less': p_value_less,
        'z_statistic': z_statistic,
        'p_value_normal': p_value_normal
    }


def sign_test_demonstration():
    """
    Demonstrates sign test and comparison with other tests.
    
    Shows how to perform the sign test, interpret results, and compare
    with parametric and other nonparametric alternatives.
    """
    # Load data
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    sepal_length = data['sepal length (cm)']
    
    # Apply sign test to sepal length data
    sepal_sign_test = sign_test(sepal_length, 5.5)
    
    print("Sign Test Results:")
    print(f"Positive signs: {sepal_sign_test['positive_signs']}")
    print(f"Negative signs: {sepal_sign_test['negative_signs']}")
    print(f"Total observations (excluding ties): {sepal_sign_test['n']}")
    print(f"Two-tailed p-value (exact): {sepal_sign_test['p_value_two_tailed']:.4f}")
    
    if not np.isnan(sepal_sign_test['z_statistic']):
        print(f"z-statistic (normal approx): {sepal_sign_test['z_statistic']:.3f}")
        print(f"Two-tailed p-value (normal): {sepal_sign_test['p_value_normal']:.4f}")
    
    # Compare with other tests
    sepal_t_stat, sepal_t_p = stats.ttest_1samp(sepal_length, 5.5)
    wilcox_stat, wilcox_p = stats.wilcoxon(sepal_length - 5.5)
    
    print(f"\nComparison with other tests:")
    print(f"t-test p-value: {sepal_t_p:.4f}")
    print(f"Wilcoxon p-value: {wilcox_p:.4f}")
    print(f"Sign test p-value: {sepal_sign_test['p_value_two_tailed']:.4f}")
    
    return sepal_sign_test


def power_analysis(sample_size, effect_size, alpha=0.05):
    """
    Comprehensive power analysis for one-sample t-test.
    
    Parameters:
    -----------
    sample_size : int
        Current sample size
    effect_size : float
        Effect size (Cohen's d)
    alpha : float
        Significance level
        
    Returns:
    --------
    dict : Dictionary containing power analysis results
    """
    # Calculate power for current sample size
    power_result = TTestPower()
    power = power_result.power(effect_size=effect_size, nobs=sample_size, alpha=alpha)
    
    # Calculate required sample size for 80% power
    sample_size_80 = power_result.solve_power(effect_size=effect_size, alpha=alpha, power=0.8)
    
    # Calculate required sample size for 90% power
    sample_size_90 = power_result.solve_power(effect_size=effect_size, alpha=alpha, power=0.9)
    
    # Calculate power for different effect sizes
    small_effect_power = power_result.power(effect_size=0.2, nobs=sample_size, alpha=alpha)
    medium_effect_power = power_result.power(effect_size=0.5, nobs=sample_size, alpha=alpha)
    large_effect_power = power_result.power(effect_size=0.8, nobs=sample_size, alpha=alpha)
    
    return {
        'power': power,
        'required_n_80': sample_size_80,
        'required_n_90': sample_size_90,
        'effect_size': effect_size,
        'alpha': alpha,
        'small_effect_power': small_effect_power,
        'medium_effect_power': medium_effect_power,
        'large_effect_power': large_effect_power
    }


def power_analysis_demonstration():
    """
    Demonstrates power analysis for one-sample t-test.
    
    Shows how to calculate power for current effect size, required sample sizes
    for different power levels, and power curves for different effect sizes.
    """
    # Load data and calculate effect size
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    sepal_length = data['sepal length (cm)']
    sepal_effect_size = calculate_cohens_d(sepal_length, 5.5)
    
    # Apply to sepal length data
    sepal_power = power_analysis(len(sepal_length), sepal_effect_size['cohens_d'])
    
    print("Power Analysis Results:")
    print(f"Current effect size: {sepal_power['effect_size']:.3f}")
    print(f"Current power: {sepal_power['power']:.3f}")
    print(f"Required sample size for 80% power: {int(np.ceil(sepal_power['required_n_80']))}")
    print(f"Required sample size for 90% power: {int(np.ceil(sepal_power['required_n_90']))}")
    
    print(f"\nPower for different effect sizes:")
    print(f"Small effect (d = 0.2): {sepal_power['small_effect_power']:.3f}")
    print(f"Medium effect (d = 0.5): {sepal_power['medium_effect_power']:.3f}")
    print(f"Large effect (d = 0.8): {sepal_power['large_effect_power']:.3f}")
    
    # Power curve analysis
    effect_sizes = np.arange(0.1, 1.1, 0.1)
    power_result = TTestPower()
    power_curve = [power_result.power(effect_size=d, nobs=len(sepal_length), alpha=0.05) 
                   for d in effect_sizes]
    
    print(f"\nPower curve (effect size vs power):")
    for i, d in enumerate(effect_sizes):
        print(f"d = {d:.1f}: Power = {power_curve[i]:.3f}")
    
    return sepal_power


def check_normality(data):
    """
    Comprehensive normality checking for a dataset.
    Includes Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov tests, skewness, kurtosis, and plots.
    """
    shapiro_stat, shapiro_p = shapiro(data)
    anderson_result = anderson(data)
    ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))
    data_skewness = skew(data)
    data_kurtosis = kurtosis(data)
    # Plots (optional, can be enabled for interactive use)
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # stats.probplot(data, dist="norm", plot=axes[0])
    # axes[0].set_title(f"Q-Q Plot for Normality Check\nShapiro-Wilk p = {shapiro_p:.4f}")
    # axes[1].hist(data, bins=15, density=True, alpha=0.7, color='steelblue')
    # x = np.linspace(data.min(), data.max(), 100)
    # axes[1].plot(x, stats.norm.pdf(x, data.mean(), data.std()), 'r-', linewidth=2)
    # axes[1].set_title(f"Histogram with Normal Curve\nSkewness = {data_skewness:.3f}, Kurtosis = {data_kurtosis:.3f}")
    # axes[2].boxplot(data, patch_artist=True, boxprops=dict(facecolor='steelblue', alpha=0.7))
    # axes[2].set_title("Box Plot for Outlier Detection")
    # plt.tight_layout()
    # plt.show()
    print("=== NORMALITY ASSESSMENT ===")
    print(f"Sample size: {len(data)}")
    print(f"Mean: {data.mean():.3f}")
    print(f"SD: {data.std():.3f}")
    print(f"Skewness: {data_skewness:.3f}")
    print(f"Kurtosis: {data_kurtosis:.3f}\n")
    print("Normality Tests:")
    print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
    print(f"Anderson-Darling statistic: {anderson_result.statistic:.4f}")
    print(f"Kolmogorov-Smirnov p-value: {ks_p:.4f}\n")
    return {
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'anderson_stat': anderson_result.statistic,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'skewness': data_skewness,
        'kurtosis': data_kurtosis
    }

def detect_outliers(data):
    """
    Detect outliers in a dataset using IQR, Z-score, and modified Z-score methods.
    Returns indices of outliers for each method.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound_iqr = q1 - 1.5 * iqr
    upper_bound_iqr = q3 + 1.5 * iqr
    outliers_iqr = (data < lower_bound_iqr) | (data > upper_bound_iqr)
    z_scores = np.abs((data - data.mean()) / data.std())
    outliers_z = z_scores > 3
    median_val = np.median(data)
    mad_val = stats.median_abs_deviation(data)
    modified_z_scores = np.abs((data - median_val) / mad_val)
    outliers_modified_z = modified_z_scores > 3.5
    return {
        'outliers_iqr': np.where(outliers_iqr)[0],
        'outliers_z': np.where(outliers_z)[0],
        'outliers_modified_z': np.where(outliers_modified_z)[0],
        'bounds_iqr': [lower_bound_iqr, upper_bound_iqr],
        'z_scores': z_scores,
        'modified_z_scores': modified_z_scores,
        'q1': q1,
        'q3': q3,
        'iqr': iqr
    }

def bootstrap_mean_ci(data, n_resamples=10000, alpha=0.05):
    """
    Compute bootstrap confidence intervals for the mean.
    Returns percentile and normal-based CIs.
    """
    bootstrap_means = []
    for _ in range(n_resamples):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    bootstrap_means = np.array(bootstrap_means)
    percentile_ci = np.percentile(bootstrap_means, [100*alpha/2, 100*(1-alpha/2)])
    normal_ci = [np.mean(bootstrap_means) - 1.96 * np.std(bootstrap_means),
                 np.mean(bootstrap_means) + 1.96 * np.std(bootstrap_means)]
    print(f"Bootstrap percentile {100*(1-alpha):.1f}% CI: [{percentile_ci[0]:.3f}, {percentile_ci[1]:.3f}]")
    print(f"Bootstrap normal {100*(1-alpha):.1f}% CI: [{normal_ci[0]:.3f}, {normal_ci[1]:.3f}]")
    return {'percentile_ci': percentile_ci, 'normal_ci': normal_ci, 'bootstrap_means': bootstrap_means}

def robust_t_test(data, mu, trim=0.1):
    """
    Perform robust t-test using trimmed mean and winsorized mean.
    Returns robust statistics and p-value.
    """
    trimmed_data = data.dropna()
    n = len(trimmed_data)
    k = int(n * trim)
    if k > 0:
        sorted_data = np.sort(trimmed_data)
        trimmed_values = sorted_data[k:(n - k)]
    else:
        trimmed_values = trimmed_data
    trimmed_mean_val = np.mean(trimmed_values)
    trimmed_var = np.var(trimmed_values, ddof=1)
    trimmed_se = np.sqrt(trimmed_var / (n - 2 * k))
    t_statistic = (trimmed_mean_val - mu) / trimmed_se
    df = n - 2 * k - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
    # Winsorized statistics
    winsorized_data = data.copy()
    if k > 0:
        winsorized_data[winsorized_data < sorted_data[k]] = sorted_data[k]
        winsorized_data[winsorized_data > sorted_data[n - k - 1]] = sorted_data[n - k - 1]
    winsorized_mean = winsorized_data.mean()
    winsorized_var = winsorized_data.var(ddof=1)
    # M-estimator (Huber's method)
    huber_reg = HuberRegressor(fit_intercept=True)
    huber_reg.fit(np.ones((len(data), 1)), data)
    huber_estimate = huber_reg.intercept_
    print(f"Trimmed mean: {trimmed_mean_val:.3f}, Winsorized mean: {winsorized_mean:.3f}, Huber M-estimate: {huber_estimate:.3f}")
    print(f"Robust t-statistic: {t_statistic:.3f}, p-value: {p_value:.4f}")
    return {
        'trimmed_mean': trimmed_mean_val,
        'winsorized_mean': winsorized_mean,
        'huber_estimate': huber_estimate,
        't_statistic': t_statistic,
        'p_value': p_value,
        'df': df
    }

def generate_test_report(test_result, data, hypothesized_value, test_type="t-test"):
    """
    Generate a comprehensive report for a one-sample test.
    Includes descriptive statistics, test results, effect size, and recommendations.
    """
    print("=== COMPREHENSIVE ONE-SAMPLE TEST REPORT ===\n")
    print("RESEARCH CONTEXT:")
    print(f"Test type: {test_type}")
    print(f"Sample size: {len(data)}")
    print(f"Hypothesized value: {hypothesized_value}")
    print(f"Alpha level: 0.05\n")
    print("DESCRIPTIVE STATISTICS:")
    print(f"Sample mean: {data.mean():.3f}")
    print(f"Sample SD: {data.std():.3f}")
    print(f"Sample median: {data.median():.3f}")
    print(f"Range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"IQR: {data.quantile(0.75) - data.quantile(0.25):.3f}\n")
    shapiro_stat, shapiro_p = shapiro(data)
    print("ASSUMPTION CHECKS:")
    print(f"Normality (Shapiro-Wilk): W = {shapiro_stat:.3f}, p = {shapiro_p:.4f}")
    outliers = detect_outliers(data)
    print(f"Outliers (IQR method): {len(outliers['outliers_iqr'])}")
    print(f"Outliers (Z-score method): {len(outliers['outliers_z'])}\n")
    if test_type == "t-test":
        print("T-TEST RESULTS:")
        print(f"t-statistic: {test_result[0]:.3f}")
        print(f"p-value: {test_result[1]:.4f}")
        ci_lower, ci_upper = stats.t.interval(0.95, len(data)-1, loc=data.mean(), scale=stats.sem(data))
        print(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"Standard Error: {stats.sem(data):.3f}\n")
    elif test_type == "wilcoxon":
        print("WILCOXON SIGNED-RANK TEST RESULTS:")
        print(f"W-statistic: {test_result[0]:.3f}")
        print(f"p-value: {test_result[1]:.4f}\n")
    effect_size = calculate_cohens_d(data, hypothesized_value)
    print("EFFECT SIZE:")
    print(f"Cohen's d: {effect_size['cohens_d']:.3f}")
    print(f"Interpretation: {interpret_effect_size(effect_size['cohens_d'])}")
    print(f"95% CI for effect size: [{effect_size['ci_lower']:.3f}, {effect_size['ci_upper']:.3f}]\n")
    power_analysis = TTestPower()
    power = power_analysis.power(effect_size=effect_size['cohens_d'], nobs=len(data), alpha=0.05)
    print("POWER ANALYSIS:")
    print(f"Observed power: {power:.3f}")
    if power < 0.8:
        recommended_n = power_analysis.solve_power(effect_size=effect_size['cohens_d'], alpha=0.05, power=0.8)
        print(f"Recommended sample size for 80% power: {int(np.ceil(recommended_n))}")
    print()
    alpha = 0.05
    print("STATISTICAL DECISION:")
    if test_result[1] < alpha:
        print(f"✓ Reject the null hypothesis (p < {alpha})")
        print(f"✓ There is significant evidence that the population mean differs from {hypothesized_value}")
    else:
        print(f"✗ Fail to reject the null hypothesis (p >= {alpha})")
        print(f"✗ There is insufficient evidence that the population mean differs from {hypothesized_value}")
    print()
    mean_diff = data.mean() - hypothesized_value
    print("PRACTICAL INTERPRETATION:")
    if abs(effect_size['cohens_d']) < 0.2:
        print("The effect is small and may not be practically meaningful.")
    elif abs(effect_size['cohens_d']) < 0.5:
        print("The effect is moderate and may be practically meaningful.")
    else:
        print("The effect is large and likely practically meaningful.")
    print(f"\nAPA STYLE REPORTING:")
    if test_type == "t-test":
        print(f"A one-sample t-test was conducted to compare the sample mean (M = {data.mean():.2f}, SD = {data.std():.2f}) to the hypothesized value of {hypothesized_value}. ", end="")
        if test_result[1] < alpha:
            print(f"The test was significant, t({len(data)-1}) = {test_result[0]:.2f}, p = {test_result[1]:.3f}, d = {effect_size['cohens_d']:.2f}. ", end="")
            print(f"The 95% confidence interval for the mean difference was [{ci_lower:.2f}, {ci_upper:.2f}].")
        else:
            print(f"The test was not significant, t({len(data)-1}) = {test_result[0]:.2f}, p = {test_result[1]:.3f}, d = {effect_size['cohens_d']:.2f}.")

# In __main__, you can demonstrate these as needed, e.g.:
#
# iris = load_iris()
# data = pd.DataFrame(iris.data, columns=iris.feature_names)
# sepal_length = data['sepal length (cm)']
# check_normality(sepal_length)
# detect_outliers(sepal_length)
# bootstrap_mean_ci(sepal_length)
# robust_t_test(sepal_length, 5.5)
# sepal_t_result = stats.ttest_1samp(sepal_length, 5.5)
# generate_test_report(sepal_t_result, sepal_length, 5.5, "t-test")
#
# These are commented out to avoid excessive output by default.


if __name__ == "__main__":
    print("=== ONE-SAMPLE TESTS DEMONSTRATION ===\n")
    
    # Basic t-test
    print("1. Basic One-Sample t-Test")
    print("=" * 40)
    basic_t_result = basic_one_sample_t_test()
    print()
    
    # t-test alternatives
    print("2. t-Test with Different Alternatives")
    print("=" * 40)
    alternatives_result = t_test_alternatives()
    print()
    
    # Effect size analysis
    print("3. Effect Size Analysis")
    print("=" * 40)
    effect_result = effect_size_analysis()
    print()
    
    # Z-test demonstration
    print("4. One-Sample Z-Test")
    print("=" * 40)
    z_test_result = z_test_demonstration()
    print()
    
    # Wilcoxon test
    print("5. Wilcoxon Signed-Rank Test")
    print("=" * 40)
    wilcox_result = wilcoxon_signed_rank_test()
    print()
    
    # Sign test
    print("6. Sign Test")
    print("=" * 40)
    sign_result = sign_test_demonstration()
    print()
    
    # Power analysis
    print("7. Power Analysis")
    print("=" * 40)
    power_result = power_analysis_demonstration()
    print()
    
    # Normality check
    print("8. Normality Check")
    print("=" * 40)
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    sepal_length = data['sepal length (cm)']
    check_normality(sepal_length)
    print()

    # Outlier detection
    print("9. Outlier Detection")
    print("=" * 40)
    detect_outliers(sepal_length)
    print()

    # Bootstrap CI
    print("10. Bootstrap Confidence Intervals")
    print("=" * 40)
    bootstrap_mean_ci(sepal_length)
    print()

    # Robust t-test
    print("11. Robust t-Test")
    print("=" * 40)
    robust_t_test(sepal_length, 5.5)
    print()

    # Generate report (example)
    print("12. Comprehensive Test Report (Example)")
    print("=" * 40)
    sepal_t_result = stats.ttest_1samp(sepal_length, 5.5)
    generate_test_report(sepal_t_result, sepal_length, 5.5, "t-test")
    print()
    
    print("=== DEMONSTRATION COMPLETE ===") 