# One-Sample Tests

## Overview

One-sample tests are fundamental statistical procedures used to determine whether a sample mean differs significantly from a hypothesized population mean. These tests form the cornerstone of statistical inference and are widely used across research disciplines including psychology, medicine, education, and business analytics.

### When to Use One-Sample Tests

One-sample tests are appropriate when you want to:
- Compare a sample mean to a known or hypothesized population value
- Test whether a treatment has a significant effect compared to a baseline
- Validate whether a process meets quality control standards
- Assess whether educational interventions improve performance above a threshold

### Key Concepts

**Null Hypothesis (H₀):** The population mean equals the hypothesized value
**Alternative Hypothesis (H₁):** The population mean differs from the hypothesized value

The test determines whether observed differences are statistically significant or due to random sampling variation.

### Mathematical Foundation

The one-sample t-test is based on the t-distribution, which accounts for the uncertainty in estimating the population standard deviation from sample data.

For a sample of size $n$ with mean $\bar{x}$ and standard deviation $s$, the t-statistic is:

```math
t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}
```

where $\mu_0$ is the hypothesized population mean.

The degrees of freedom are $df = n - 1$, and the p-value is calculated from the t-distribution with these degrees of freedom.

## One-Sample t-Test

The one-sample t-test is the most commonly used parametric test for comparing a sample mean to a hypothesized population mean. It's robust and works well even with moderate violations of normality assumptions, especially for larger sample sizes.

### Mathematical Foundation

The t-test is based on the t-distribution, which was developed by William Gosset (publishing under the pseudonym "Student") in 1908. The t-distribution accounts for the uncertainty in estimating the population standard deviation from sample data.

**Test Statistic:**
```math
t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}
```

**Degrees of Freedom:**
```math
df = n - 1
```

**Confidence Interval:**
```math
CI = \bar{x} \pm t_{\alpha/2, df} \cdot \frac{s}{\sqrt{n}}
```

where:
- $\bar{x}$ = sample mean
- $\mu_0$ = hypothesized population mean
- $s$ = sample standard deviation
- $n$ = sample size
- $t_{\alpha/2, df}$ = critical t-value for confidence level $\alpha$

### Basic One-Sample t-Test

**See:** `basic_one_sample_t_test()` in `12_one_sample_tests.py`

*This function demonstrates how to perform a one-sample t-test comparing a sample mean to a hypothesized population mean, including manual calculations for understanding the underlying mathematics. Run the function to see the test results and manual verification.*

---

### One-Sample t-Test with Different Alternatives

**See:** `t_test_alternatives()` in `12_one_sample_tests.py`

*This function demonstrates how to perform two-tailed and one-tailed one-sample t-tests, and explains the relationship between different p-values and critical values. Use it to understand hypothesis directionality in t-tests.*

---

### Effect Size for One-Sample t-Test

**See:** `effect_size_analysis()` and `calculate_cohens_d()` in `12_one_sample_tests.py`

*These functions show how to calculate Cohen's d, Hedges' g, and confidence intervals for effect sizes, along with power analysis based on effect size. Use them to interpret the magnitude and practical significance of your results.*

---

## One-Sample z-Test

**See:** `z_test_demonstration()` and `one_sample_z_test()` in `12_one_sample_tests.py`

*These functions demonstrate how to perform a one-sample z-test when the population standard deviation is known, and compare results with the t-test. Use them to understand the difference between z- and t-tests.*

---

## Nonparametric One-Sample Tests

### Wilcoxon Signed-Rank Test

**See:** `wilcoxon_signed_rank_test()` in `12_one_sample_tests.py`

*This function demonstrates the Wilcoxon signed-rank test as a nonparametric alternative to the t-test, including manual calculations and comparison with the t-test. Use it for data that may not be normally distributed.*

---

### Sign Test

**See:** `sign_test_demonstration()` and `sign_test()` in `12_one_sample_tests.py`

*These functions demonstrate the sign test, the simplest nonparametric test, and compare it with the t-test and Wilcoxon test. Use them for robust inference when only the direction of differences matters.*

---

## Power Analysis

**See:** `power_analysis_demonstration()` and `power_analysis()` in `12_one_sample_tests.py`

*These functions show how to calculate power for a one-sample t-test, required sample sizes for different power levels, and power curves for different effect sizes. Use them for study design and to ensure adequate statistical power.*

---

## Assumption Checking

### Normality Test

**See:** `check_normality()` in `12_one_sample_tests.py`

*This function provides a comprehensive normality assessment using Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov tests, skewness, kurtosis, and (optionally) plots. Use it to check if your data meet the assumptions for parametric tests.*

---

### Outlier Detection

**See:** `detect_outliers()` in `12_one_sample_tests.py`

*This function detects outliers using IQR, Z-score, and modified Z-score methods. Use it to identify and handle outliers before performing statistical tests.*

---

## Advanced Topics

### Bootstrap Confidence Intervals

**See:** `bootstrap_mean_ci()` in `12_one_sample_tests.py`

*This function computes bootstrap confidence intervals for the mean using both percentile and normal-based methods. Use it for robust inference when normality is questionable.*

---

### Robust One-Sample Tests

**See:** `robust_t_test()` in `12_one_sample_tests.py`

*This function performs a robust t-test using trimmed means and winsorized means, and provides robust statistics and p-values. Use it when your data contain outliers or are not normally distributed.*

---

## Best Practices & Reporting

**See:** `generate_test_report()` in `12_one_sample_tests.py`

*This function generates a comprehensive report for a one-sample test, including descriptive statistics, test results, effect size, and recommendations. Use it to communicate your results clearly and in APA style.*

---

## Practical Examples

All practical examples (quality control, educational assessment, medical research, etc.) are implemented as demonstrations in the functions above. See the `__main__` block in `12_one_sample_tests.py` for example usage and outputs for each scenario.

---

## Exercises

For each exercise, use the relevant functions in `12_one_sample_tests.py` to perform the analysis. The code for each exercise can be implemented by calling and combining these functions as described in the exercise instructions.

---

## Next Steps

In the next chapter, we'll learn about two-sample tests for comparing means between groups.

---

**Key Takeaways:**
- One-sample tests compare a sample mean to a hypothesized population mean
- t-test is appropriate for normally distributed data or large samples
- Nonparametric alternatives exist for non-normal data
- Always check assumptions before interpreting results
- Effect sizes provide important information about practical significance
- Power analysis helps determine appropriate sample sizes
- Bootstrap methods provide robust confidence intervals