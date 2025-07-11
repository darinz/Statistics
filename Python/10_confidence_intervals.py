import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
import matplotlib.pyplot as plt


def ci_mean_known_sd(data, population_sd=0.8, confidence_level=0.95):
    """
    Calculate confidence interval for the mean with known population standard deviation (z-distribution).
    """
    sample_mean = data.mean()
    n = len(data)
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    standard_error = population_sd / np.sqrt(n)
    margin_of_error = z_score * standard_error
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    return ci_lower, ci_upper, sample_mean, z_score, standard_error, margin_of_error


def ci_mean_unknown_sd(data, confidence_level=0.95):
    """
    Calculate confidence interval for the mean with unknown population standard deviation (t-distribution).
    """
    sample_mean = data.mean()
    sample_sd = data.std()
    n = len(data)
    alpha = 1 - confidence_level
    df = n - 1
    t_score = stats.t.ppf(1 - alpha/2, df=df)
    standard_error = sample_sd / np.sqrt(n)
    margin_of_error = t_score * standard_error
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    return ci_lower, ci_upper, sample_mean, t_score, standard_error, margin_of_error


def ci_mean_statsmodels(data, confidence_level=0.95):
    """
    Calculate confidence interval for the mean using statsmodels DescrStatsW.
    """
    desc_stats = DescrStatsW(data)
    ci = desc_stats.tconfint_mean(alpha=1-confidence_level)
    return ci, desc_stats


def ci_proportion(successes, n, confidence_level=0.95):
    """
    Calculate confidence interval for a proportion using normal approximation.
    """
    sample_proportion = successes / n
    standard_error = np.sqrt(sample_proportion * (1 - sample_proportion) / n)
    z_score = stats.norm.ppf(1 - (1-confidence_level)/2)
    margin_of_error = z_score * standard_error
    ci_lower = sample_proportion - margin_of_error
    ci_upper = sample_proportion + margin_of_error
    return ci_lower, ci_upper, sample_proportion, z_score, standard_error, margin_of_error


def ci_proportion_exact(successes, n, confidence_level=0.95):
    """
    Calculate exact binomial confidence interval for a proportion using scipy.stats.binomtest.
    """
    binom_result = stats.binomtest(successes, n)
    ci = binom_result.proportions_ci(confidence_level=confidence_level)
    return ci


def ci_difference_means(group1, group2, equal_var=True, confidence_level=0.95):
    """
    Calculate confidence interval for the difference between two means (independent samples).
    """
    cm = CompareMeans.from_data(group1, group2)
    usevar = 'pooled' if equal_var else 'unequal'
    ci = cm.tconfint_diff(alpha=1-confidence_level, usevar=usevar)
    return ci


def ci_paired_samples(before, after, confidence_level=0.95):
    """
    Calculate confidence interval for the mean difference in paired samples.
    """
    differences = after - before
    desc_stats = DescrStatsW(differences)
    ci = desc_stats.tconfint_mean(alpha=1-confidence_level)
    return ci


def ci_variance(data, confidence_level=0.95):
    """
    Calculate confidence interval for the variance using chi-square distribution.
    """
    sample_variance = data.var()
    n = len(data)
    df = n - 1
    alpha = 1 - confidence_level
    chi_lower = stats.chi2.ppf(alpha/2, df=df)
    chi_upper = stats.chi2.ppf(1 - alpha/2, df=df)
    ci_lower = (df * sample_variance) / chi_upper
    ci_upper = (df * sample_variance) / chi_lower
    return ci_lower, ci_upper, sample_variance


def bootstrap_ci_mean(data, n_bootstrap=1000, confidence_level=0.95):
    """
    Bootstrap confidence interval for the mean.
    """
    np.random.seed(123)
    bootstrap_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(bootstrap_means, (1-confidence_level)/2*100)
    upper = np.percentile(bootstrap_means, (1+(confidence_level))/2*100)
    return lower, upper, np.mean(bootstrap_means), np.std(bootstrap_means)


def bootstrap_ci_median(data, n_bootstrap=1000, confidence_level=0.95):
    """
    Bootstrap confidence interval for the median.
    """
    np.random.seed(123)
    bootstrap_medians = [np.median(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(bootstrap_medians, (1-confidence_level)/2*100)
    upper = np.percentile(bootstrap_medians, (1+(confidence_level))/2*100)
    return lower, upper, np.mean(bootstrap_medians), np.std(bootstrap_medians)


def ci_width_vs_sample_size(data, sample_sizes, confidence_level=0.95):
    """
    Show effect of sample size on CI width for the mean.
    """
    widths = []
    for size in sample_sizes:
        sample = np.random.choice(data, size=size, replace=True)
        desc_stats = DescrStatsW(sample)
        ci = desc_stats.tconfint_mean(alpha=1-confidence_level)
        widths.append(ci[1] - ci[0])
    return widths


def ci_width_vs_confidence_level(data, confidence_levels):
    """
    Show effect of confidence level on CI width for the mean.
    """
    widths = []
    for level in confidence_levels:
        desc_stats = DescrStatsW(data)
        ci = desc_stats.tconfint_mean(alpha=1-level)
        widths.append(ci[1] - ci[0])
    return widths


def bonferroni_cis(data, group_col, value_col, alpha_family=0.05):
    """
    Calculate Bonferroni-corrected and uncorrected CIs for each group.
    """
    groups = data[group_col].unique()
    n_comparisons = len(groups)
    alpha_individual = alpha_family / n_comparisons
    results = {}
    for group in groups:
        group_data = data[data[group_col] == group][value_col]
        desc_stats = DescrStatsW(group_data)
        ci_bonf = desc_stats.tconfint_mean(alpha=alpha_individual)
        ci_uncorr = desc_stats.tconfint_mean(alpha=0.05)
        results[group] = {'bonferroni': ci_bonf, 'uncorrected': ci_uncorr}
    return results


def interpret_ci(ci, parameter_name="parameter", confidence_level=0.95):
    """
    Print interpretation of a confidence interval.
    """
    print(f"We are {confidence_level*100:.1f}% confident that the true {parameter_name} lies between {ci[0]:.3f} and {ci[1]:.3f}.")


def demonstrate_mistakes():
    """
    Print common mistakes in CI interpretation.
    """
    print("1. ❌ Saying '95% probability that the parameter is in the interval'")
    print("2. ❌ Comparing confidence intervals for significance")
    print("3. ❌ Using confidence intervals for individual predictions")
    print("4. ❌ Ignoring multiple comparisons")
    print("5. ❌ Focusing only on whether 0 is in the interval")
    print("6. ❌ Using the same confidence level for all analyses")


def main():
    # Load sample data
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    sepal_length = data['sepal length (cm)']

    # CI for mean (known SD)
    ci1 = ci_mean_known_sd(sepal_length)
    print(f"CI for mean (known SD): {ci1[0]:.3f} to {ci1[1]:.3f}")

    # CI for mean (unknown SD)
    ci2 = ci_mean_unknown_sd(sepal_length)
    print(f"CI for mean (unknown SD): {ci2[0]:.3f} to {ci2[1]:.3f}")

    # CI for mean (statsmodels)
    ci3, _ = ci_mean_statsmodels(sepal_length)
    print(f"CI for mean (statsmodels): {ci3[0]:.3f} to {ci3[1]:.3f}")

    # CI for proportion
    data['long_sepal'] = (sepal_length > 5.5).astype(int)
    successes = data['long_sepal'].sum()
    n = len(data['long_sepal'])
    ci4 = ci_proportion(successes, n)
    print(f"CI for proportion: {ci4[0]:.3f} to {ci4[1]:.3f}")

    # Exact CI for proportion
    ci5 = ci_proportion_exact(successes, n)
    print(f"Exact CI for proportion: {ci5[0]:.3f} to {ci5[1]:.3f}")

    # CI for difference of means (independent)
    setosa = data[data['target'] == 0]['sepal length (cm)']
    versicolor = data[data['target'] == 1]['sepal length (cm)']
    ci6 = ci_difference_means(setosa, versicolor)
    print(f"CI for difference of means: {ci6[0]:.3f} to {ci6[1]:.3f}")

    # CI for paired samples
    before = np.random.normal(50, 10, 20)
    after = before + np.random.normal(5, 3, 20)
    ci7 = ci_paired_samples(before, after)
    print(f"CI for paired samples: {ci7[0]:.3f} to {ci7[1]:.3f}")

    # CI for variance
    ci8 = ci_variance(sepal_length)
    print(f"CI for variance: {ci8[0]:.3f} to {ci8[1]:.3f}")

    # Bootstrap CI for mean
    ci9 = bootstrap_ci_mean(sepal_length)
    print(f"Bootstrap CI for mean: {ci9[0]:.3f} to {ci9[1]:.3f}")

    # Bootstrap CI for median
    ci10 = bootstrap_ci_median(sepal_length)
    print(f"Bootstrap CI for median: {ci10[0]:.3f} to {ci10[1]:.3f}")

    # CI width vs sample size
    sample_sizes = [5, 10, 20, 30, 50, 100]
    widths = ci_width_vs_sample_size(sepal_length, sample_sizes)
    print(f"CI widths for sample sizes: {list(zip(sample_sizes, widths))}")

    # CI width vs confidence level
    confidence_levels = [0.80, 0.85, 0.90, 0.95, 0.99]
    widths2 = ci_width_vs_confidence_level(sepal_length, confidence_levels)
    print(f"CI widths for confidence levels: {list(zip(confidence_levels, widths2))}")

    # Bonferroni CIs
    bonferroni = bonferroni_cis(data, 'target', 'sepal length (cm)')
    print(f"Bonferroni CIs: {bonferroni}")

    # Interpretation
    ci, _ = ci_mean_statsmodels(sepal_length)
    interpret_ci(ci, "mean sepal length", 0.95)

    # Demonstrate mistakes
    demonstrate_mistakes()

if __name__ == "__main__":
    main() 