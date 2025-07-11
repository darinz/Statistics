# Confidence Intervals

## Overview

Confidence intervals (CIs) are one of the most important tools in statistical inference, providing a range of plausible values for population parameters based on sample data. Unlike point estimates that give a single value, confidence intervals acknowledge the inherent uncertainty in sampling and provide a range that likely contains the true population parameter.

### What is a Confidence Interval?

A confidence interval is an interval estimate that, with a specified level of confidence, contains the true population parameter. The confidence level (typically 90%, 95%, or 99%) represents the long-run frequency with which the method produces intervals that contain the true parameter.

### Key Concepts

1. **Point Estimate**: A single value estimate of a population parameter (e.g., sample mean)
2. **Interval Estimate**: A range of values that likely contains the population parameter
3. **Confidence Level**: The probability that the interval contains the true parameter (long-run interpretation)
4. **Margin of Error**: Half the width of the confidence interval
5. **Standard Error**: The standard deviation of the sampling distribution

### Mathematical Foundation

The general form of a confidence interval is:

```math
\text{Point Estimate} \pm \text{Critical Value} \times \text{Standard Error}
```

For a population mean with known standard deviation:

```math
\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
```

Where:
- $\bar{x}$ is the sample mean
- $z_{\alpha/2}$ is the critical value from the standard normal distribution
- $\sigma$ is the population standard deviation
- $n$ is the sample size
- $\alpha = 1 - \text{confidence level}$

### Interpretation

The correct interpretation of a 95% confidence interval is:
"We are 95% confident that the true population parameter lies within this interval."

**Important**: This does NOT mean there's a 95% probability that the parameter is in the interval. The parameter is fixed; the interval varies across samples.

### Properties of Confidence Intervals

1. **Width**: Wider intervals indicate more uncertainty
2. **Sample Size**: Larger samples produce narrower intervals
3. **Confidence Level**: Higher confidence levels produce wider intervals
4. **Population Variability**: More variable populations produce wider intervals

## Confidence Interval for the Mean

The mean is one of the most commonly estimated parameters. The method for constructing confidence intervals depends on whether we know the population standard deviation.

### Normal Distribution (Known Population Standard Deviation)

When the population standard deviation $\sigma$ is known, we use the standard normal distribution (z-distribution) to construct confidence intervals.

#### Mathematical Foundation

The confidence interval formula is:

```math
\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
```

Where:
- $\bar{x}$ is the sample mean
- $z_{\alpha/2}$ is the critical value from the standard normal distribution
- $\sigma$ is the population standard deviation
- $n$ is the sample size
- $\alpha = 1 - \text{confidence level}$

The standard error of the mean is:

```math
SE(\bar{x}) = \frac{\sigma}{\sqrt{n}}
```

#### Python Example

See `ci_mean_known_sd` in `10_confidence_intervals.py` for a function that calculates the confidence interval for the mean when the population standard deviation is known. This function demonstrates the use of the z-distribution for confidence intervals.

---

### t-Distribution (Unknown Population Standard Deviation)

In practice, we rarely know the population standard deviation. When $\sigma$ is unknown, we use the sample standard deviation $s$ and the t-distribution instead of the normal distribution.

#### Mathematical Foundation

The confidence interval formula becomes:

```math
\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}
```

Where:
- $\bar{x}$ is the sample mean
- $t_{\alpha/2, n-1}$ is the critical value from the t-distribution with $n-1$ degrees of freedom
- $s$ is the sample standard deviation
- $n$ is the sample size

The standard error is estimated as:

```math
SE(\bar{x}) = \frac{s}{\sqrt{n}}
```

#### Python Example

See `ci_mean_unknown_sd` in `10_confidence_intervals.py` for a function that calculates the confidence interval for the mean when the population standard deviation is unknown, using the t-distribution.

---

### Using Built-in Functions

Python's `statsmodels` library provides convenient functions for calculating confidence intervals. The `ci_mean_statsmodels` function in `10_confidence_intervals.py` demonstrates how to use `DescrStatsW` to compute confidence intervals for the mean.

---

## Confidence Interval for the Proportion

Proportions represent the fraction of a population that has a particular characteristic. Confidence intervals for proportions are essential in surveys, quality control, and many other applications.

### Mathematical Foundation

For a population proportion $p$, the sample proportion $\hat{p}$ is:

```math
\hat{p} = \frac{x}{n}
```

Where $x$ is the number of successes and $n$ is the sample size.

The standard error of the proportion is:

```math
SE(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
```

### Large Sample Approximation

When $n\hat{p} \geq 10$ and $n(1-\hat{p}) \geq 10$, we can use the normal approximation:

```math
\hat{p} \pm z_{\alpha/2} \cdot \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
```

#### Python Example

See `ci_proportion` in `10_confidence_intervals.py` for a function that calculates the confidence interval for a proportion using the normal approximation.

---

### Exact Binomial Confidence Interval

When the large sample conditions are not met, or when we want the most accurate confidence interval, we use the exact binomial method.

#### Python Example

See `ci_proportion_exact` in `10_confidence_intervals.py` for a function that calculates the exact binomial confidence interval for a proportion using the Clopper-Pearson method.

---

## Confidence Interval for the Difference Between Two Means

Comparing two population means is a common statistical task. We can construct confidence intervals for the difference between two means using either independent or paired samples.

### Independent Samples

See `ci_difference_means` in `10_confidence_intervals.py` for a function that calculates the confidence interval for the difference between two means for independent samples. This function supports both pooled and Welch's (unequal variance) methods.

---

### Paired Samples

See `ci_paired_samples` in `10_confidence_intervals.py` for a function that calculates the confidence interval for the mean difference in paired samples.

---

## Confidence Interval for the Variance

See `ci_variance` in `10_confidence_intervals.py` for a function that calculates the confidence interval for the variance using the chi-square distribution.

---

## Bootstrap Confidence Intervals

Bootstrap methods provide a non-parametric approach to constructing confidence intervals. They are particularly useful when the sampling distribution is unknown or when the assumptions of parametric methods are violated.

- For the mean, see `bootstrap_ci_mean` in `10_confidence_intervals.py`.
- For the median, see `bootstrap_ci_median` in `10_confidence_intervals.py`.

Each function demonstrates how to use bootstrap resampling to estimate confidence intervals for the mean and median, respectively.

---

## Effect of Sample Size and Confidence Level

- To see the effect of sample size on CI width, use `ci_width_vs_sample_size` in `10_confidence_intervals.py`.
- To see the effect of confidence level on CI width, use `ci_width_vs_confidence_level` in `10_confidence_intervals.py`.

---

## Multiple Confidence Intervals

To calculate Bonferroni-corrected and uncorrected confidence intervals for multiple groups, see `bonferroni_cis` in `10_confidence_intervals.py`.

---

## Practical Examples

- **Quality Control**: Use the mean CI functions to assess if a process is on target.
- **Survey Results**: Use the proportion CI functions to estimate population proportions from survey data.
- **Treatment Effect**: Use the difference of means CI functions to estimate the effect of a treatment.

---

## Best Practices

- For correct interpretation of confidence intervals, see the `interpret_ci` function in `10_confidence_intervals.py`.
- For common mistakes to avoid, see the `demonstrate_mistakes` function in `10_confidence_intervals.py`.

---

## Exercises

1. Calculate 90%, 95%, and 99% confidence intervals for the mean of a subset of data using the provided functions.
2. Calculate confidence intervals for a proportion using both normal approximation and exact binomial methods.
3. Calculate a confidence interval for the difference in means between two groups.
4. Use bootstrap sampling to calculate confidence intervals for the median and standard deviation of a dataset.
5. Determine the sample size needed to achieve a desired margin of error using the CI width functions.

Refer to the corresponding functions in `10_confidence_intervals.py` for implementation guidance. 