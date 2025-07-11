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

#### Code Implementation

```r
# Load sample data
data(mtcars)

# Calculate confidence interval for mean MPG
sample_mean <- mean(mtcars$mpg)
sample_sd <- sd(mtcars$mpg)
n <- length(mtcars$mpg)

# For demonstration, assume we know population standard deviation
population_sd <- 6  # Hypothetical known population SD
confidence_level <- 0.95
alpha <- 1 - confidence_level

# Calculate z-score (critical value)
z_score <- qnorm(1 - alpha/2)

# Calculate standard error
standard_error <- population_sd / sqrt(n)

# Calculate margin of error
margin_of_error <- z_score * standard_error

# Calculate confidence interval
ci_lower <- sample_mean - margin_of_error
ci_upper <- sample_mean + margin_of_error

# Display results
cat("=== Confidence Interval for Mean (Known Population SD) ===\n")
cat("Sample mean:", round(sample_mean, 3), "\n")
cat("Population SD:", population_sd, "\n")
cat("Sample size:", n, "\n")
cat("Confidence level:", confidence_level * 100, "%\n")
cat("Z-score:", round(z_score, 3), "\n")
cat("Standard error:", round(standard_error, 3), "\n")
cat("Margin of error:", round(margin_of_error, 3), "\n")
cat("95% CI:", round(ci_lower, 3), "to", round(ci_upper, 3), "\n")

# Verify the calculation
cat("\nVerification:\n")
cat("CI width:", round(ci_upper - ci_lower, 3), "\n")
cat("Expected width:", round(2 * margin_of_error, 3), "\n")
```

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

#### Properties of the t-Distribution

1. **Degrees of Freedom**: The t-distribution has $n-1$ degrees of freedom
2. **Shape**: Similar to normal distribution but with heavier tails
3. **Convergence**: As $n \to \infty$, the t-distribution approaches the normal distribution
4. **Critical Values**: t-critical values are always larger than z-critical values for the same confidence level

#### Code Implementation

```r
# Calculate confidence interval using t-distribution
confidence_level <- 0.95
alpha <- 1 - confidence_level
df <- n - 1  # degrees of freedom

# Calculate t-score (critical value)
t_score <- qt(1 - alpha/2, df = df)

# Calculate standard error using sample SD
standard_error <- sample_sd / sqrt(n)

# Calculate margin of error
margin_of_error <- t_score * standard_error

# Calculate confidence interval
ci_lower <- sample_mean - margin_of_error
ci_upper <- sample_mean + margin_of_error

# Display results
cat("=== Confidence Interval for Mean (Unknown Population SD) ===\n")
cat("Sample mean:", round(sample_mean, 3), "\n")
cat("Sample SD:", round(sample_sd, 3), "\n")
cat("Sample size:", n, "\n")
cat("Degrees of freedom:", df, "\n")
cat("Confidence level:", confidence_level * 100, "%\n")
cat("t-score:", round(t_score, 3), "\n")
cat("Standard error:", round(standard_error, 3), "\n")
cat("Margin of error:", round(margin_of_error, 3), "\n")
cat("95% CI:", round(ci_lower, 3), "to", round(ci_upper, 3), "\n")

# Compare with z-distribution (for demonstration)
z_score <- qnorm(1 - alpha/2)
z_margin_of_error <- z_score * standard_error
cat("\nComparison with z-distribution:\n")
cat("Z-score:", round(z_score, 3), "\n")
cat("Z-based margin of error:", round(z_margin_of_error, 3), "\n")
cat("Difference in margin of error:", round(margin_of_error - z_margin_of_error, 3), "\n")
```

### Using Built-in Functions

R provides convenient built-in functions for calculating confidence intervals. The `t.test()` function is particularly useful for confidence intervals of the mean.

#### Understanding the t.test() Function

The `t.test()` function performs a t-test and automatically calculates confidence intervals. It handles:
- Unknown population standard deviation (uses t-distribution)
- Degrees of freedom calculation
- Standard error estimation
- Critical value selection

#### Code Implementation

```r
# Using t.test() for confidence interval
t_test_result <- t.test(mtcars$mpg, conf.level = 0.95)
print(t_test_result)

# Extract confidence interval
ci_from_test <- t_test_result$conf.int
cat("95% CI from t.test():", round(ci_from_test[1], 3), "to", round(ci_from_test[2], 3), "\n")

# Extract other useful information
cat("\nDetailed Information:\n")
cat("Sample mean:", round(t_test_result$estimate, 3), "\n")
cat("Standard error:", round(t_test_result$stderr, 3), "\n")
cat("Degrees of freedom:", t_test_result$parameter, "\n")
cat("Confidence level:", (1 - attr(t_test_result$conf.int, "conf.level")) * 100, "%\n")

# Verify our manual calculation
cat("\nVerification with manual calculation:\n")
manual_ci_lower <- sample_mean - margin_of_error
manual_ci_upper <- sample_mean + margin_of_error
cat("Manual CI:", round(manual_ci_lower, 3), "to", round(manual_ci_upper, 3), "\n")
cat("t.test() CI:", round(ci_from_test[1], 3), "to", round(ci_from_test[2], 3), "\n")
cat("Difference:", round(abs(manual_ci_lower - ci_from_test[1]), 6), "\n")
```

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

#### Code Implementation

```r
# Calculate confidence interval for proportion
# Example: proportion of manual transmissions
manual_count <- sum(mtcars$am)
total_count <- length(mtcars$am)
sample_proportion <- manual_count / total_count

# Check large sample conditions
np <- total_count * sample_proportion
nq <- total_count * (1 - sample_proportion)

cat("=== Confidence Interval for Proportion ===\n")
cat("Manual transmissions:", manual_count, "\n")
cat("Total cars:", total_count, "\n")
cat("Sample proportion:", round(sample_proportion, 3), "\n")

# Check large sample approximation conditions
cat("\nLarge Sample Approximation Check:\n")
cat("n*p =", np, "(should be ≥ 10):", ifelse(np >= 10, "✓", "✗"), "\n")
cat("n*(1-p) =", nq, "(should be ≥ 10):", ifelse(nq >= 10, "✓", "✗"), "\n")

# Calculate standard error
standard_error <- sqrt(sample_proportion * (1 - sample_proportion) / total_count)

# Calculate margin of error
z_score <- qnorm(0.975)  # 95% confidence level
margin_of_error <- z_score * standard_error

# Calculate confidence interval
ci_lower <- sample_proportion - margin_of_error
ci_upper <- sample_proportion + margin_of_error

cat("\nNormal Approximation Results:\n")
cat("Standard error:", round(standard_error, 4), "\n")
cat("Z-score:", round(z_score, 3), "\n")
cat("Margin of error:", round(margin_of_error, 4), "\n")
cat("95% CI:", round(ci_lower, 4), "to", round(ci_upper, 4), "\n")

# Calculate confidence interval width
ci_width <- ci_upper - ci_lower
cat("CI width:", round(ci_width, 4), "\n")
```

### Exact Binomial Confidence Interval

When the large sample conditions are not met, or when we want the most accurate confidence interval, we use the exact binomial method. This method is based on the binomial distribution and doesn't rely on normal approximation.

#### Mathematical Foundation

The exact binomial confidence interval uses the Clopper-Pearson method, which is based on the relationship between the binomial distribution and the beta distribution. The confidence interval $(p_L, p_U)$ satisfies:

```math
P(X \geq x | p = p_L) = \frac{\alpha}{2}
```

```math
P(X \leq x | p = p_U) = \frac{\alpha}{2}
```

Where $X$ follows a binomial distribution with parameters $n$ and $p$.

#### Code Implementation

```r
# Using binom.test() for exact confidence interval
binom_result <- binom.test(manual_count, total_count, conf.level = 0.95)
print(binom_result)

# Extract confidence interval
ci_from_binom <- binom_result$conf.int
cat("Exact 95% CI:", round(ci_from_binom[1], 4), "to", round(ci_from_binom[2], 4), "\n")

# Compare normal approximation vs exact method
cat("\n=== Comparison: Normal vs Exact ===\n")
cat("Normal approximation CI:", round(ci_lower, 4), "to", round(ci_upper, 4), "\n")
cat("Exact binomial CI:", round(ci_from_binom[1], 4), "to", round(ci_from_binom[2], 4), "\n")

# Calculate differences
normal_width <- ci_upper - ci_lower
exact_width <- ci_from_binom[2] - ci_from_binom[1]
cat("Normal CI width:", round(normal_width, 4), "\n")
cat("Exact CI width:", round(exact_width, 4), "\n")
cat("Width difference:", round(exact_width - normal_width, 4), "\n")

# Check if the intervals overlap significantly
overlap_lower <- max(ci_lower, ci_from_binom[1])
overlap_upper <- min(ci_upper, ci_from_binom[2])
overlap_width <- max(0, overlap_upper - overlap_lower)
cat("Overlap width:", round(overlap_width, 4), "\n")

# When to use each method
cat("\n=== When to Use Each Method ===\n")
if (np >= 10 && nq >= 10) {
  cat("✓ Large sample conditions met - both methods are reasonable\n")
  cat("✓ Normal approximation is simpler and often sufficient\n")
  cat("✓ Exact method is more accurate but computationally intensive\n")
} else {
  cat("✗ Large sample conditions not met\n")
  cat("✓ Use exact binomial method\n")
  cat("✗ Normal approximation may be unreliable\n")
}
```

## Confidence Interval for the Difference Between Two Means

Comparing two population means is a common statistical task. We can construct confidence intervals for the difference between two means using either independent or paired samples.

### Independent Samples

When the two samples are independent (e.g., different groups of subjects), we use the two-sample t-test approach.

#### Mathematical Foundation

For independent samples, the confidence interval for $\mu_1 - \mu_2$ is:

```math
(\bar{x}_1 - \bar{x}_2) \pm t_{\alpha/2, df} \cdot SE(\bar{x}_1 - \bar{x}_2)
```

Where the standard error depends on whether we assume equal variances:

**Equal Variances (Pooled):**
```math
SE(\bar{x}_1 - \bar{x}_2) = s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}
```

Where the pooled standard deviation is:
```math
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
```

**Unequal Variances (Welch's):**
```math
SE(\bar{x}_1 - \bar{x}_2) = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
```

The degrees of freedom for unequal variances (Welch's approximation):
```math
df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
```

#### Code Implementation

```r
# Compare MPG between automatic and manual transmissions
auto_mpg <- mtcars$mpg[mtcars$am == 0]
manual_mpg <- mtcars$mpg[mtcars$am == 1]

# Calculate sample statistics
n1 <- length(auto_mpg)
n2 <- length(manual_mpg)
mean1 <- mean(auto_mpg)
mean2 <- mean(manual_mpg)
sd1 <- sd(auto_mpg)
sd2 <- sd(manual_mpg)

cat("=== Confidence Interval for Difference Between Two Means ===\n")
cat("Group 1 (Auto): n =", n1, ", mean =", round(mean1, 3), ", SD =", round(sd1, 3), "\n")
cat("Group 2 (Manual): n =", n2, ", mean =", round(mean2, 3), ", SD =", round(sd2, 3), "\n")

# Test for equal variances (F-test)
var_test <- var.test(auto_mpg, manual_mpg)
cat("\nVariance Test (F-test):\n")
cat("F-statistic:", round(var_test$statistic, 3), "\n")
cat("p-value:", round(var_test$p.value, 4), "\n")
cat("Equal variances assumption:", ifelse(var_test$p.value > 0.05, "✓", "✗"), "\n")

# Calculate pooled standard deviation (equal variances)
pooled_sd <- sqrt(((n1 - 1) * sd1^2 + (n2 - 1) * sd2^2) / (n1 + n2 - 2))

# Calculate standard error
standard_error <- pooled_sd * sqrt(1/n1 + 1/n2)

# Calculate degrees of freedom
df <- n1 + n2 - 2

# Calculate t-score
t_score <- qt(0.975, df = df)

# Calculate margin of error
margin_of_error <- t_score * standard_error

# Calculate confidence interval
difference <- mean2 - mean1
ci_lower <- difference - margin_of_error
ci_upper <- difference + margin_of_error

cat("\nPooled Variance Results:\n")
cat("Pooled SD:", round(pooled_sd, 3), "\n")
cat("Standard error:", round(standard_error, 3), "\n")
cat("Degrees of freedom:", df, "\n")
cat("t-score:", round(t_score, 3), "\n")
cat("Margin of error:", round(margin_of_error, 3), "\n")
cat("Difference (Manual - Auto):", round(difference, 3), "\n")
cat("95% CI for difference:", round(ci_lower, 3), "to", round(ci_upper, 3), "\n")

# Using built-in function (pooled variance)
t_test_diff <- t.test(manual_mpg, auto_mpg, conf.level = 0.95, var.equal = TRUE)
print(t_test_diff)

# Using built-in function (Welch's - unequal variances)
t_test_welch <- t.test(manual_mpg, auto_mpg, conf.level = 0.95, var.equal = FALSE)
cat("\nWelch's t-test (unequal variances):\n")
cat("95% CI for difference:", round(t_test_welch$conf.int[1], 3), "to", round(t_test_welch$conf.int[2], 3), "\n")
cat("Degrees of freedom:", round(t_test_welch$parameter, 1), "\n")
```

### Paired Samples

When the two samples are paired (e.g., same subjects measured before and after treatment), we analyze the differences between pairs rather than the individual measurements.

#### Mathematical Foundation

For paired samples, we work with the differences $d_i = x_{i1} - x_{i2}$. The confidence interval for the mean difference $\mu_d$ is:

```math
\bar{d} \pm t_{\alpha/2, n-1} \cdot \frac{s_d}{\sqrt{n}}
```

Where:
- $\bar{d}$ is the mean of the differences
- $s_d$ is the standard deviation of the differences
- $n$ is the number of pairs
- $t_{\alpha/2, n-1}$ is the critical value from the t-distribution with $n-1$ degrees of freedom

The standard error of the mean difference is:

```math
SE(\bar{d}) = \frac{s_d}{\sqrt{n}}
```

#### Advantages of Paired Analysis

1. **Reduced Variability**: Paired analysis removes between-subject variability
2. **Increased Power**: More sensitive to detect treatment effects
3. **Control for Confounding**: Each subject serves as their own control

#### Code Implementation

```r
# Simulate paired data (before and after treatment)
set.seed(123)
before <- rnorm(20, mean = 50, sd = 10)
after <- before + rnorm(20, mean = 5, sd = 3)  # Treatment effect

# Calculate differences
differences <- after - before

# Calculate confidence interval for mean difference
n_diff <- length(differences)
mean_diff <- mean(differences)
sd_diff <- sd(differences)

cat("=== Paired Samples Confidence Interval ===\n")
cat("Sample size (pairs):", n_diff, "\n")
cat("Mean difference:", round(mean_diff, 3), "\n")
cat("SD of differences:", round(sd_diff, 3), "\n")

# Calculate standard error
standard_error <- sd_diff / sqrt(n_diff)

# Calculate t-score
df <- n_diff - 1
t_score <- qt(0.975, df = df)

# Calculate margin of error
margin_of_error <- t_score * standard_error

# Calculate confidence interval
ci_lower <- mean_diff - margin_of_error
ci_upper <- mean_diff + margin_of_error

cat("\nPaired Analysis Results:\n")
cat("Standard error:", round(standard_error, 3), "\n")
cat("Degrees of freedom:", df, "\n")
cat("t-score:", round(t_score, 3), "\n")
cat("Margin of error:", round(margin_of_error, 3), "\n")
cat("95% CI for mean difference:", round(ci_lower, 3), "to", round(ci_upper, 3), "\n")

# Using built-in function
paired_test <- t.test(after, before, paired = TRUE, conf.level = 0.95)
print(paired_test)

# Compare with independent samples analysis (incorrect for paired data)
independent_test <- t.test(after, before, paired = FALSE, conf.level = 0.95)
cat("\n=== Comparison: Paired vs Independent Analysis ===\n")
cat("Paired CI width:", round(ci_upper - ci_lower, 3), "\n")
cat("Independent CI width:", round(independent_test$conf.int[2] - independent_test$conf.int[1], 3), "\n")
cat("Width ratio (Independent/Paired):", round((independent_test$conf.int[2] - independent_test$conf.int[1]) / (ci_upper - ci_lower), 2), "\n")
```

## Confidence Interval for the Variance

Confidence intervals for variance are useful when we need to estimate the variability of a population or compare the variability between groups.

### Mathematical Foundation

For a normal population, the sampling distribution of the sample variance follows a chi-square distribution:

```math
\frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}
```

Where:
- $s^2$ is the sample variance
- $\sigma^2$ is the population variance
- $n$ is the sample size
- $\chi^2_{n-1}$ is the chi-square distribution with $n-1$ degrees of freedom

The confidence interval for the population variance is:

```math
\left(\frac{(n-1)s^2}{\chi^2_{\alpha/2, n-1}}, \frac{(n-1)s^2}{\chi^2_{1-\alpha/2, n-1}}\right)
```

Where $\chi^2_{\alpha/2, n-1}$ and $\chi^2_{1-\alpha/2, n-1}$ are the critical values from the chi-square distribution.

### Chi-Square Based Confidence Interval

#### Code Implementation

```r
# Calculate confidence interval for variance
sample_variance <- var(mtcars$mpg)
n <- length(mtcars$mpg)
df <- n - 1

cat("=== Confidence Interval for Variance ===\n")
cat("Sample variance:", round(sample_variance, 3), "\n")
cat("Sample size:", n, "\n")
cat("Degrees of freedom:", df, "\n")

# Calculate chi-square critical values
chi_lower <- qchisq(0.025, df = df)
chi_upper <- qchisq(0.975, df = df)

cat("\nChi-square Critical Values:\n")
cat("Lower critical value (2.5%):", round(chi_lower, 3), "\n")
cat("Upper critical value (97.5%):", round(chi_upper, 3), "\n")

# Calculate confidence interval
ci_lower_var <- (df * sample_variance) / chi_upper
ci_upper_var <- (df * sample_variance) / chi_lower

cat("\nVariance Confidence Interval:\n")
cat("95% CI for variance:", round(ci_lower_var, 3), "to", round(ci_upper_var, 3), "\n")

# Confidence interval for standard deviation
ci_lower_sd <- sqrt(ci_lower_var)
ci_upper_sd <- sqrt(ci_upper_var)
cat("95% CI for standard deviation:", round(ci_lower_sd, 3), "to", round(ci_upper_sd, 3), "\n")

# Verify the calculation
cat("\nVerification:\n")
cat("Sample SD:", round(sd(mtcars$mpg), 3), "\n")
cat("CI width (variance):", round(ci_upper_var - ci_lower_var, 3), "\n")
cat("CI width (SD):", round(ci_upper_sd - ci_lower_sd, 3), "\n")

# Check if the intervals are reasonable
cat("\nReasonableness Check:\n")
if (ci_lower_var > 0) {
  cat("✓ Lower bound is positive (reasonable)\n")
} else {
  cat("✗ Lower bound is negative (unreasonable)\n")
}

if (ci_upper_var > sample_variance) {
  cat("✓ Upper bound is greater than sample variance\n")
} else {
  cat("✗ Upper bound is less than sample variance\n")
}
```

## Bootstrap Confidence Intervals

Bootstrap methods provide a non-parametric approach to constructing confidence intervals. They are particularly useful when the sampling distribution is unknown or when the assumptions of parametric methods are violated.

### Mathematical Foundation

The bootstrap method works by:
1. Resampling with replacement from the original sample
2. Computing the statistic of interest for each bootstrap sample
3. Using the distribution of bootstrap statistics to construct confidence intervals

For a sample $x_1, x_2, \ldots, x_n$, we create bootstrap samples $x_1^*, x_2^*, \ldots, x_n^*$ by sampling with replacement from the original data.

The bootstrap estimate of the standard error is:

```math
SE_{boot}(\hat{\theta}) = \sqrt{\frac{1}{B-1} \sum_{b=1}^B (\hat{\theta}_b^* - \bar{\theta}^*)^2}
```

Where:
- $\hat{\theta}_b^*$ is the estimate from the $b$-th bootstrap sample
- $\bar{\theta}^*$ is the mean of all bootstrap estimates
- $B$ is the number of bootstrap samples

### Types of Bootstrap Confidence Intervals

1. **Normal Bootstrap**: Assumes bootstrap distribution is normal
2. **Percentile Bootstrap**: Uses percentiles of bootstrap distribution
3. **Basic Bootstrap**: Adjusts for bias in the bootstrap distribution
4. **BCa Bootstrap**: Bias-corrected and accelerated method

### Bootstrap for Mean

#### Code Implementation

```r
library(boot)

# Bootstrap function for mean
boot_mean <- function(data, indices) {
  return(mean(data[indices]))
}

# Perform bootstrap
set.seed(123)
boot_results <- boot(mtcars$mpg, boot_mean, R = 1000)

cat("=== Bootstrap Confidence Intervals ===\n")
cat("Original sample mean:", round(mean(mtcars$mpg), 3), "\n")
cat("Bootstrap replications:", length(boot_results$t), "\n")
cat("Bootstrap mean:", round(mean(boot_results$t), 3), "\n")
cat("Bootstrap SE:", round(sd(boot_results$t), 3), "\n")

# Different types of bootstrap confidence intervals
boot_normal <- boot.ci(boot_results, type = "norm")
boot_percentile <- boot.ci(boot_results, type = "perc")
boot_basic <- boot.ci(boot_results, type = "basic")

cat("\nBootstrap Confidence Intervals:\n")
cat("Normal bootstrap CI:", round(boot_normal$normal[2], 3), "to", round(boot_normal$normal[3], 3), "\n")
cat("Percentile bootstrap CI:", round(boot_percentile$percent[4], 3), "to", round(boot_percentile$percent[5], 3), "\n")
cat("Basic bootstrap CI:", round(boot_basic$basic[4], 3), "to", round(boot_basic$basic[5], 3), "\n")

# Compare with parametric CI
parametric_ci <- t.test(mtcars$mpg, conf.level = 0.95)$conf.int
cat("Parametric t-test CI:", round(parametric_ci[1], 3), "to", round(parametric_ci[2], 3), "\n")

# Calculate bootstrap bias
bootstrap_bias <- mean(boot_results$t) - mean(mtcars$mpg)
cat("\nBootstrap Analysis:\n")
cat("Bootstrap bias:", round(bootstrap_bias, 4), "\n")
cat("Bootstrap variance:", round(var(boot_results$t), 4), "\n")

# Plot bootstrap distribution
hist(boot_results$t, main = "Bootstrap Distribution of Mean",
     xlab = "Bootstrap Mean", col = "lightblue", freq = FALSE, 
     breaks = 30, border = "white")

# Add confidence interval lines
abline(v = boot_percentile$percent[4:5], col = "red", lty = 2, lwd = 2)
abline(v = mean(mtcars$mpg), col = "green", lwd = 2)
abline(v = mean(boot_results$t), col = "blue", lty = 3, lwd = 2)

# Add legend
legend("topright", legend = c("Original Mean", "Bootstrap Mean", "95% CI"),
       col = c("green", "blue", "red"), lty = c(1, 3, 2), lwd = 2)
```

### Bootstrap for Median

Bootstrap methods are particularly valuable for statistics like the median, which don't have simple parametric sampling distributions.

#### Mathematical Foundation

The median is a robust measure of central tendency that is less sensitive to outliers than the mean. The bootstrap provides a way to estimate the sampling distribution of the median without making distributional assumptions.

For a sample $x_1, x_2, \ldots, x_n$, the sample median is:
```math
\text{median} = \begin{cases}
x_{(n+1)/2} & \text{if } n \text{ is odd} \\
\frac{x_{n/2} + x_{n/2+1}}{2} & \text{if } n \text{ is even}
\end{cases}
```

Where $x_{(i)}$ denotes the $i$-th order statistic.

#### Code Implementation

```r
# Bootstrap function for median
boot_median <- function(data, indices) {
  return(median(data[indices]))
}

# Perform bootstrap
boot_median_results <- boot(mtcars$mpg, boot_median, R = 1000)
boot_median_ci <- boot.ci(boot_median_results, type = "perc")

cat("=== Bootstrap Confidence Interval for Median ===\n")
cat("Original sample median:", round(median(mtcars$mpg), 3), "\n")
cat("Bootstrap median:", round(mean(boot_median_results$t), 3), "\n")
cat("Bootstrap SE:", round(sd(boot_median_results$t), 3), "\n")
cat("Bootstrap CI for median:", round(boot_median_ci$percent[4], 3), "to", round(boot_median_ci$percent[5], 3), "\n")

# Compare with mean
cat("\nComparison with Mean:\n")
cat("Sample mean:", round(mean(mtcars$mpg), 3), "\n")
cat("Sample median:", round(median(mtcars$mpg), 3), "\n")
cat("Difference (mean - median):", round(mean(mtcars$mpg) - median(mtcars$mpg), 3), "\n")

# Check for skewness
if (mean(mtcars$mpg) > median(mtcars$mpg)) {
  cat("Distribution appears right-skewed (mean > median)\n")
} else if (mean(mtcars$mpg) < median(mtcars$mpg)) {
  cat("Distribution appears left-skewed (mean < median)\n")
} else {
  cat("Distribution appears symmetric (mean ≈ median)\n")
}

# Plot bootstrap distribution
hist(boot_median_results$t, main = "Bootstrap Distribution of Median",
     xlab = "Bootstrap Median", col = "lightgreen", freq = FALSE,
     breaks = 30, border = "white")
abline(v = boot_median_ci$percent[4:5], col = "red", lty = 2, lwd = 2)
abline(v = median(mtcars$mpg), col = "green", lwd = 2)
abline(v = mean(boot_median_results$t), col = "blue", lty = 3, lwd = 2)

# Add legend
legend("topright", legend = c("Original Median", "Bootstrap Median", "95% CI"),
       col = c("green", "blue", "red"), lty = c(1, 3, 2), lwd = 2)

# Compare bootstrap distributions of mean and median
par(mfrow = c(1, 2))
hist(boot_results$t, main = "Bootstrap Mean", xlab = "Bootstrap Mean", 
     col = "lightblue", freq = FALSE, breaks = 30)
hist(boot_median_results$t, main = "Bootstrap Median", xlab = "Bootstrap Median", 
     col = "lightgreen", freq = FALSE, breaks = 30)
par(mfrow = c(1, 1))
```

## Effect of Sample Size and Confidence Level

The width of confidence intervals is influenced by both sample size and confidence level. Understanding these relationships is crucial for study design and interpretation.

### Mathematical Foundation

The width of a confidence interval is determined by:

```math
\text{Width} = 2 \times \text{Critical Value} \times \text{Standard Error}
```

For the mean with unknown population standard deviation:
```math
\text{Width} = 2 \times t_{\alpha/2, n-1} \times \frac{s}{\sqrt{n}}
```

#### Sample Size Effect

As sample size increases:
- Standard error decreases: $SE \propto \frac{1}{\sqrt{n}}$
- Critical value decreases (t-distribution approaches normal)
- Overall effect: Width decreases approximately as $\frac{1}{\sqrt{n}}$

#### Confidence Level Effect

As confidence level increases:
- Critical value increases
- Width increases
- Trade-off: Higher confidence vs. wider intervals

### Sample Size Effect

#### Code Implementation

```r
# Function to calculate confidence interval width
ci_width <- function(sample_size, confidence_level = 0.95) {
  # Simulate sample
  set.seed(123)
  sample_data <- sample(mtcars$mpg, size = sample_size, replace = TRUE)
  
  # Calculate confidence interval
  t_test_result <- t.test(sample_data, conf.level = confidence_level)
  ci <- t_test_result$conf.int
  
  # Return width
  return(ci[2] - ci[1])
}

# Test different sample sizes
sample_sizes <- c(5, 10, 20, 30, 50, 100)
ci_widths <- sapply(sample_sizes, ci_width)

cat("=== Effect of Sample Size on CI Width ===\n")
for (i in 1:length(sample_sizes)) {
  cat("Sample size:", sample_sizes[i], "→ CI width:", round(ci_widths[i], 3), "\n")
}

# Calculate theoretical relationship
theoretical_widths <- ci_widths[1] * sqrt(sample_sizes[1] / sample_sizes)
cat("\nTheoretical widths (based on 1/√n relationship):\n")
for (i in 1:length(sample_sizes)) {
  cat("Sample size:", sample_sizes[i], "→ Theoretical width:", round(theoretical_widths[i], 3), "\n")
}

# Plot results
plot(sample_sizes, ci_widths, type = "b", 
     main = "Effect of Sample Size on CI Width",
     xlab = "Sample Size", ylab = "CI Width",
     col = "blue", lwd = 2, pch = 19)

# Add theoretical curve
lines(sample_sizes, theoretical_widths, col = "red", lty = 2, lwd = 2)

# Add legend
legend("topright", legend = c("Observed", "Theoretical (1/√n)"),
       col = c("blue", "red"), lty = c(1, 2), lwd = 2, pch = c(19, NA))

# Calculate efficiency gains
cat("\nEfficiency Analysis:\n")
for (i in 2:length(sample_sizes)) {
  efficiency_gain <- (ci_widths[1] / ci_widths[i])^2
  cat("Sample size", sample_sizes[i], "is", round(efficiency_gain, 1), "times more efficient than size", sample_sizes[1], "\n")
}
```

### Confidence Level Effect

#### Code Implementation

```r
# Function to calculate confidence interval for different levels
ci_by_level <- function(confidence_levels) {
  widths <- numeric(length(confidence_levels))
  critical_values <- numeric(length(confidence_levels))
  
  for (i in 1:length(confidence_levels)) {
    t_test_result <- t.test(mtcars$mpg, conf.level = confidence_levels[i])
    ci <- t_test_result$conf.int
    widths[i] <- ci[2] - ci[1]
    
    # Calculate critical value
    alpha <- 1 - confidence_levels[i]
    critical_values[i] <- qt(1 - alpha/2, df = length(mtcars$mpg) - 1)
  }
  
  return(list(widths = widths, critical_values = critical_values))
}

# Test different confidence levels
confidence_levels <- c(0.80, 0.85, 0.90, 0.95, 0.99)
ci_results <- ci_by_level(confidence_levels)
ci_widths_by_level <- ci_results$widths
critical_values <- ci_results$critical_values

cat("=== Effect of Confidence Level on CI Width ===\n")
for (i in 1:length(confidence_levels)) {
  cat("Confidence level:", confidence_levels[i] * 100, "% → CI width:", round(ci_widths_by_level[i], 3), 
      "→ Critical value:", round(critical_values[i], 3), "\n")
}

# Calculate relative increases
cat("\nRelative Increases:\n")
for (i in 2:length(confidence_levels)) {
  relative_increase <- (ci_widths_by_level[i] / ci_widths_by_level[1]) - 1
  cat("From", confidence_levels[1] * 100, "% to", confidence_levels[i] * 100, "%: +", 
      round(relative_increase * 100, 1), "% wider\n")
}

# Plot results
plot(confidence_levels, ci_widths_by_level, type = "b",
     main = "Effect of Confidence Level on CI Width",
     xlab = "Confidence Level", ylab = "CI Width",
     col = "red", lwd = 2, pch = 19)

# Add critical values on secondary axis
par(new = TRUE)
plot(confidence_levels, critical_values, type = "b", 
     axes = FALSE, xlab = "", ylab = "", col = "blue", lty = 2, lwd = 2)
axis(4, col = "blue", col.axis = "blue")
mtext("Critical Value", side = 4, line = 3, col = "blue")

# Add legend
legend("topleft", legend = c("CI Width", "Critical Value"),
       col = c("red", "blue"), lty = c(1, 2), lwd = 2, pch = c(19, NA))

# Reset plot parameters
par(new = FALSE)

# Trade-off analysis
cat("\nTrade-off Analysis:\n")
cat("Higher confidence levels provide more certainty but wider intervals.\n")
cat("This creates a trade-off between precision and confidence.\n\n")

# Practical recommendations
cat("Practical Recommendations:\n")
cat("• 90% CI: Good balance for most applications\n")
cat("• 95% CI: Standard choice for most research\n")
cat("• 99% CI: Use when high confidence is critical\n")
cat("• 80% CI: Use when precision is more important than confidence\n")
```

## Multiple Confidence Intervals

When constructing multiple confidence intervals, we need to account for the increased probability of making at least one Type I error (false positive). This is known as the multiple comparisons problem.

### Mathematical Foundation

When making $k$ independent confidence intervals, each with confidence level $1 - \alpha$, the probability that all intervals contain their true parameters is:

```math
P(\text{All intervals correct}) = (1 - \alpha)^k
```

The family-wise error rate (FWER) is:
```math
\text{FWER} = 1 - (1 - \alpha)^k
```

For small $\alpha$, this is approximately:
```math
\text{FWER} \approx k\alpha
```

### Bonferroni Correction

The Bonferroni correction adjusts the individual significance level to maintain the desired family-wise error rate:

```math
\alpha_{individual} = \frac{\alpha_{family}}{k}
```

### Simultaneous Confidence Intervals

#### Code Implementation

```r
# Bonferroni correction for multiple comparisons
# Example: confidence intervals for MPG by cylinders
cylinders <- unique(mtcars$cyl)
n_comparisons <- length(cylinders)
alpha_family <- 0.05
alpha_individual <- alpha_family / n_comparisons

cat("=== Multiple Confidence Intervals ===\n")
cat("Number of comparisons:", n_comparisons, "\n")
cat("Family-wise alpha:", alpha_family, "\n")
cat("Individual alpha:", round(alpha_individual, 4), "\n")
cat("Individual confidence level:", round((1 - alpha_individual) * 100, 2), "%\n")

# Calculate confidence intervals for each cylinder type
ci_results <- list()
sample_sizes <- list()
sample_means <- list()

for (cyl in cylinders) {
  cyl_data <- mtcars$mpg[mtcars$cyl == cyl]
  t_test_result <- t.test(cyl_data, conf.level = 1 - alpha_individual)
  ci_results[[as.character(cyl)]] <- t_test_result$conf.int
  sample_sizes[[as.character(cyl)]] <- length(cyl_data)
  sample_means[[as.character(cyl)]] <- mean(cyl_data)
}

# Display results
cat("\nBonferroni-corrected confidence intervals (α =", alpha_family, "):\n")
for (cyl in cylinders) {
  ci <- ci_results[[as.character(cyl)]]
  n <- sample_sizes[[as.character(cyl)]]
  mean_val <- sample_means[[as.character(cyl)]]
  cat("Cylinders", cyl, ":", "n =", n, ", mean =", round(mean_val, 2), 
      ", CI =", round(ci[1], 2), "to", round(ci[2], 2), "\n")
}

# Compare with uncorrected intervals
cat("\nUncorrected confidence intervals (95%):\n")
for (cyl in cylinders) {
  cyl_data <- mtcars$mpg[mtcars$cyl == cyl]
  t_test_result <- t.test(cyl_data, conf.level = 0.95)
  ci <- t_test_result$conf.int
  cat("Cylinders", cyl, ":", round(ci[1], 2), "to", round(ci[2], 2), "\n")
}

# Calculate width differences
cat("\nWidth Comparison:\n")
for (cyl in cylinders) {
  cyl_data <- mtcars$mpg[mtcars$cyl == cyl]
  
  # Bonferroni-corrected
  t_test_bonf <- t.test(cyl_data, conf.level = 1 - alpha_individual)
  width_bonf <- t_test_bonf$conf.int[2] - t_test_bonf$conf.int[1]
  
  # Uncorrected
  t_test_uncorr <- t.test(cyl_data, conf.level = 0.95)
  width_uncorr <- t_test_uncorr$conf.int[2] - t_test_uncorr$conf.int[1]
  
  width_ratio <- width_bonf / width_uncorr
  cat("Cylinders", cyl, ": width ratio =", round(width_ratio, 2), "\n")
}

# Family-wise error rate calculation
cat("\nFamily-wise Error Rate Analysis:\n")
cat("Uncorrected FWER:", round(1 - (1 - 0.05)^n_comparisons, 4), "\n")
cat("Bonferroni-corrected FWER:", round(1 - (1 - alpha_individual)^n_comparisons, 4), "\n")
cat("Bonferroni upper bound:", round(n_comparisons * alpha_individual, 4), "\n")
```

## Practical Examples

### Example 1: Quality Control

Quality control is a common application of confidence intervals in manufacturing and production processes.

#### Mathematical Foundation

In quality control, we often want to estimate the mean of a production process and determine if it meets specifications. The confidence interval helps us understand the uncertainty in our estimate.

For a production process with target value $\mu_0$, we construct a confidence interval for the population mean $\mu$. If $\mu_0$ falls within the interval, we have evidence that the process is on target.

#### Code Implementation

```r
# Quality control example
set.seed(123)
production_batch <- rnorm(100, mean = 100, sd = 5)

# Calculate confidence interval for mean weight
t_test_qc <- t.test(production_batch, conf.level = 0.95)
ci_qc <- t_test_qc$conf.int

cat("=== Quality Control Example ===\n")
cat("Sample size:", length(production_batch), "\n")
cat("Production batch mean:", round(mean(production_batch), 2), "\n")
cat("Production batch SD:", round(sd(production_batch), 2), "\n")
cat("95% CI for mean weight:", round(ci_qc[1], 2), "to", round(ci_qc[2], 2), "\n")

# Check if target weight (100) is in confidence interval
target_weight <- 100
if (target_weight >= ci_qc[1] && target_weight <= ci_qc[2]) {
  cat("✓ Target weight is within the confidence interval.\n")
  cat("  → Process appears to be on target.\n")
} else {
  cat("✗ Target weight is outside the confidence interval.\n")
  cat("  → Process may need adjustment.\n")
}

# Calculate process capability
process_capability <- abs(mean(production_batch) - target_weight) / sd(production_batch)
cat("\nProcess Capability Analysis:\n")
cat("Process capability index:", round(process_capability, 3), "\n")

if (process_capability < 0.5) {
  cat("✓ Process is well-controlled\n")
} else if (process_capability < 1.0) {
  cat("⚠ Process needs monitoring\n")
} else {
  cat("✗ Process needs immediate attention\n")
}

# Tolerance analysis
tolerance <- 2 * sd(production_batch)  # ±2 SD tolerance
cat("Process tolerance (±2 SD):", round(mean(production_batch) - tolerance, 2), 
    "to", round(mean(production_batch) + tolerance, 2), "\n")
```

### Example 2: Survey Results

Survey research is another important application of confidence intervals, particularly for proportions.

#### Mathematical Foundation

In survey research, we often want to estimate the proportion of a population that has a particular characteristic. The confidence interval for a proportion helps us understand the precision of our estimate.

For a population proportion $p$, the sample proportion $\hat{p}$ follows approximately a normal distribution when the sample size is large enough:

```math
\hat{p} \sim N\left(p, \sqrt{\frac{p(1-p)}{n}}\right)
```

#### Code Implementation

```r
# Survey example
set.seed(123)
survey_responses <- sample(c(0, 1), size = 100, replace = TRUE, prob = c(0.7, 0.3))
# 1 = satisfied, 0 = not satisfied

# Calculate confidence interval for satisfaction proportion
satisfied_count <- sum(survey_responses)
total_responses <- length(survey_responses)
satisfaction_proportion <- satisfied_count / total_responses

cat("=== Survey Results Example ===\n")
cat("Total responses:", total_responses, "\n")
cat("Satisfied responses:", satisfied_count, "\n")
cat("Satisfaction proportion:", round(satisfaction_proportion, 3), "\n")

# Check large sample conditions
np <- total_responses * satisfaction_proportion
nq <- total_responses * (1 - satisfaction_proportion)
cat("Large sample check - n*p =", np, ", n*(1-p) =", nq, "\n")

# Calculate confidence interval
binom_result <- binom.test(satisfied_count, total_responses, conf.level = 0.95)
ci_survey <- binom_result$conf.int

cat("95% CI for satisfaction proportion:", round(ci_survey[1], 4), "to", round(ci_survey[2], 4), "\n")

# Interpret the results
cat("\nInterpretation:\n")
cat("We are 95% confident that the true satisfaction proportion\n")
cat("in the population lies between", round(ci_survey[1] * 100, 1), "% and", 
    round(ci_survey[2] * 100, 1), "%.\n")

# Margin of error
margin_of_error <- (ci_survey[2] - ci_survey[1]) / 2
cat("Margin of error: ±", round(margin_of_error * 100, 1), "%\n")

# Sample size planning for future surveys
cat("\nSample Size Planning:\n")
cat("For a margin of error of ±5% at 95% confidence:\n")
required_n <- ceiling(1.96^2 * 0.5 * 0.5 / 0.05^2)
cat("Required sample size:", required_n, "\n")

# Compare with different confidence levels
confidence_levels <- c(0.90, 0.95, 0.99)
for (level in confidence_levels) {
  binom_result_level <- binom.test(satisfied_count, total_responses, conf.level = level)
  ci_level <- binom_result_level$conf.int
  cat("Confidence level", level * 100, "%: CI =", round(ci_level[1], 4), "to", round(ci_level[2], 4), "\n")
}
```

### Example 3: Treatment Effect

Clinical trials and experimental studies often use confidence intervals to estimate treatment effects.

#### Mathematical Foundation

In experimental studies, we want to estimate the difference between treatment and control groups. The confidence interval for the treatment effect helps us understand both the magnitude and precision of the effect.

For independent samples, the treatment effect $\delta = \mu_1 - \mu_2$ is estimated by $\bar{x}_1 - \bar{x}_2$ with standard error:

```math
SE(\bar{x}_1 - \bar{x}_2) = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
```

#### Code Implementation

```r
# Treatment effect example
set.seed(123)
control_group <- rnorm(30, mean = 50, sd = 10)
treatment_group <- rnorm(30, mean = 55, sd = 10)

cat("=== Treatment Effect Example ===\n")
cat("Control group: n =", length(control_group), ", mean =", round(mean(control_group), 2), 
    ", SD =", round(sd(control_group), 2), "\n")
cat("Treatment group: n =", length(treatment_group), ", mean =", round(mean(treatment_group), 2), 
    ", SD =", round(sd(treatment_group), 2), "\n")

# Calculate confidence interval for treatment effect
t_test_treatment <- t.test(treatment_group, control_group, conf.level = 0.95)
ci_treatment <- t_test_treatment$conf.int

treatment_effect <- mean(treatment_group) - mean(control_group)
cat("\nTreatment Effect Analysis:\n")
cat("Point estimate of treatment effect:", round(treatment_effect, 2), "\n")
cat("95% CI for treatment effect:", round(ci_treatment[1], 2), "to", round(ci_treatment[2], 2), "\n")

# Check if treatment effect is significant (CI doesn't include 0)
if (ci_treatment[1] > 0 || ci_treatment[2] < 0) {
  cat("✓ Treatment effect is significant (CI doesn't include 0).\n")
} else {
  cat("✗ Treatment effect is not significant (CI includes 0).\n")
}

# Effect size calculation (Cohen's d)
pooled_sd <- sqrt(((length(control_group) - 1) * var(control_group) + 
                   (length(treatment_group) - 1) * var(treatment_group)) / 
                  (length(control_group) + length(treatment_group) - 2))
cohens_d <- treatment_effect / pooled_sd

cat("\nEffect Size Analysis:\n")
cat("Cohen's d:", round(cohens_d, 3), "\n")

if (abs(cohens_d) < 0.2) {
  cat("Effect size: Small\n")
} else if (abs(cohens_d) < 0.5) {
  cat("Effect size: Medium\n")
} else if (abs(cohens_d) < 0.8) {
  cat("Effect size: Large\n")
} else {
  cat("Effect size: Very large\n")
}

# Power analysis
cat("\nPower Analysis:\n")
cat("Sample size per group:", length(control_group), "\n")
cat("Effect size (Cohen's d):", round(cohens_d, 3), "\n")

# Approximate power calculation
power_approx <- pnorm(abs(cohens_d) * sqrt(length(control_group) / 2) - 1.96)
cat("Approximate power:", round(power_approx, 3), "\n")

if (power_approx < 0.8) {
  cat("⚠ Study may be underpowered\n")
} else {
  cat("✓ Study appears adequately powered\n")
}

# Clinical significance
cat("\nClinical Significance:\n")
cat("Treatment effect:", round(treatment_effect, 2), "units\n")
cat("95% CI width:", round(ci_treatment[2] - ci_treatment[1], 2), "units\n")
cat("Precision (1/width):", round(1 / (ci_treatment[2] - ci_treatment[1]), 3), "\n")
```

## Best Practices

### Interpretation Guidelines

Proper interpretation of confidence intervals is crucial for statistical inference. Here are guidelines for correct interpretation.

#### Mathematical Foundation

The correct interpretation of a $(1-\alpha) \times 100\%$ confidence interval is:
"We are $(1-\alpha) \times 100\%$ confident that the true parameter lies within this interval."

This means that if we were to repeat the sampling process many times, approximately $(1-\alpha) \times 100\%$ of the intervals would contain the true parameter.

#### Code Implementation

```r
# Function to interpret confidence intervals
interpret_ci <- function(ci, parameter_name = "parameter", confidence_level = 0.95) {
  cat("=== CONFIDENCE INTERVAL INTERPRETATION ===\n")
  cat("Confidence Level:", confidence_level * 100, "%\n")
  cat("Parameter:", parameter_name, "\n")
  cat("Interval:", round(ci[1], 3), "to", round(ci[2], 3), "\n\n")
  
  cat("CORRECT INTERPRETATION:\n")
  cat("We are", confidence_level * 100, "% confident that the true", parameter_name, "\n")
  cat("lies between", round(ci[1], 3), "and", round(ci[2], 3), ".\n\n")
  
  cat("This means that if we were to repeat this study many times,\n")
  cat("approximately", confidence_level * 100, "% of the confidence intervals\n")
  cat("would contain the true", parameter_name, ".\n\n")
  
  cat("IMPORTANT NOTES:\n")
  cat("✓ The confidence interval is about the method, not the parameter\n")
  cat("✓ The parameter is fixed; the interval varies across samples\n")
  cat("✓ A wider interval indicates more uncertainty\n")
  cat("✓ A narrower interval indicates more precision\n")
  cat("✓ The confidence level refers to the long-run frequency\n")
  
  # Calculate interval width
  width <- ci[2] - ci[1]
  cat("\nINTERVAL ANALYSIS:\n")
  cat("Interval width:", round(width, 3), "\n")
  cat("Margin of error: ±", round(width/2, 3), "\n")
  
  if (width < 1) {
    cat("Precision: High (narrow interval)\n")
  } else if (width < 5) {
    cat("Precision: Moderate\n")
  } else {
    cat("Precision: Low (wide interval)\n")
  }
}

# Example interpretation
example_ci <- t.test(mtcars$mpg, conf.level = 0.95)$conf.int
interpret_ci(example_ci, "mean MPG", 0.95)

# Compare different confidence levels
cat("\n=== COMPARISON OF CONFIDENCE LEVELS ===\n")
confidence_levels <- c(0.90, 0.95, 0.99)
for (level in confidence_levels) {
  ci_level <- t.test(mtcars$mpg, conf.level = level)$conf.int
  cat("Confidence level", level * 100, "%: CI width =", 
      round(ci_level[2] - ci_level[1], 3), "\n")
}
```

### Common Mistakes to Avoid

Understanding common mistakes helps prevent misinterpretation and misuse of confidence intervals.

#### Mathematical Foundation

The key principle is that confidence intervals are about the method, not the parameter. The parameter is fixed, but the interval varies across samples.

#### Code Implementation

```r
# Function to demonstrate common mistakes
demonstrate_mistakes <- function() {
  cat("=== COMMON MISTAKES IN CONFIDENCE INTERVALS ===\n\n")
  
  cat("1. ❌ Saying '95% probability that the parameter is in the interval'\n")
  cat("   ✓ Correct: 'We are 95% confident that the interval contains the parameter'\n")
  cat("   - The parameter is fixed, not random\n")
  cat("   - The interval is random, not the parameter\n\n")
  
  cat("2. ❌ Comparing confidence intervals for significance\n")
  cat("   ✓ Correct: Use hypothesis tests for significance\n")
  cat("   - Overlapping CIs don't necessarily mean no significant difference\n")
  cat("   - Non-overlapping CIs don't necessarily mean significant difference\n\n")
  
  cat("3. ❌ Using confidence intervals for individual predictions\n")
  cat("   ✓ Correct: Use prediction intervals for individual predictions\n")
  cat("   - CIs are for population parameters, not individual values\n")
  cat("   - Prediction intervals are wider than confidence intervals\n\n")
  
  cat("4. ❌ Ignoring multiple comparisons\n")
  cat("   ✓ Correct: Use corrections like Bonferroni for multiple comparisons\n")
  cat("   - Multiple CIs increase family-wise error rate\n")
  cat("   - Each additional comparison increases the chance of false positives\n\n")
  
  cat("5. ❌ Focusing only on whether 0 is in the interval\n")
  cat("   ✓ Correct: Consider the practical significance of the interval\n")
  cat("   - Statistical significance ≠ practical significance\n")
  cat("   - Consider the context and magnitude of effects\n\n")
  
  cat("6. ❌ Using the same confidence level for all analyses\n")
  cat("   ✓ Correct: Choose confidence level based on the context\n")
  cat("   - Higher confidence = wider intervals = less precision\n")
  cat("   - Lower confidence = narrower intervals = less certainty\n\n")
}

demonstrate_mistakes()

# Demonstrate the overlap fallacy
cat("\n=== DEMONSTRATION: CI OVERLAP FALLACY ===\n")

# Create two groups with overlapping CIs but significant difference
set.seed(123)
group1 <- rnorm(20, mean = 10, sd = 2)
group2 <- rnorm(20, mean = 12, sd = 2)

ci1 <- t.test(group1, conf.level = 0.95)$conf.int
ci2 <- t.test(group2, conf.level = 0.95)$conf.int
t_test_result <- t.test(group1, group2, conf.level = 0.95)

cat("Group 1 CI:", round(ci1[1], 2), "to", round(ci1[2], 2), "\n")
cat("Group 2 CI:", round(ci2[1], 2), "to", round(ci2[2], 2), "\n")
cat("CIs overlap:", (ci1[2] > ci2[1] && ci2[2] > ci1[1]), "\n")
cat("t-test p-value:", round(t_test_result$p.value, 4), "\n")
cat("Significant difference:", t_test_result$p.value < 0.05, "\n")

if (t_test_result$p.value < 0.05 && (ci1[2] > ci2[1] && ci2[2] > ci1[1])) {
  cat("⚠ This demonstrates the overlap fallacy!\n")
  cat("   CIs overlap but groups are significantly different.\n")
}
```

## Exercises

### Exercise 1: Basic Confidence Intervals
Calculate 90%, 95%, and 99% confidence intervals for the mean MPG of cars with 6 cylinders.

**Hints:**
- Use `subset()` or logical indexing to get cars with 6 cylinders
- Use `t.test()` with different `conf.level` values
- Compare the widths of the intervals

### Exercise 2: Proportion Confidence Intervals
Calculate confidence intervals for the proportion of cars with manual transmission, using both normal approximation and exact binomial methods.

**Hints:**
- Check the large sample conditions for normal approximation
- Use `binom.test()` for exact method
- Compare the results and explain any differences

### Exercise 3: Difference Between Means
Calculate a confidence interval for the difference in MPG between cars with 4 and 8 cylinders.

**Hints:**
- Test for equal variances using `var.test()`
- Use both pooled and Welch's t-test
- Interpret the results in context

### Exercise 4: Bootstrap Confidence Intervals
Use bootstrap sampling to calculate confidence intervals for the median and standard deviation of the iris sepal length data.

**Hints:**
- Load the `iris` dataset: `data(iris)`
- Use the `boot` package
- Compare bootstrap CIs with parametric CIs

### Exercise 5: Sample Size Planning
Determine the sample size needed to estimate the mean MPG with a margin of error of ±1 MPG at 95% confidence.

**Hints:**
- Use the formula: $n = \left(\frac{z_{\alpha/2} \cdot s}{E}\right)^2$
- Estimate $s$ from the current data
- Consider different confidence levels

### Exercise 6: Multiple Comparisons
Calculate confidence intervals for the mean MPG by cylinder type (4, 6, 8) using Bonferroni correction.

**Hints:**
- Calculate the number of comparisons
- Adjust the individual confidence level
- Compare with uncorrected intervals

### Exercise 7: Effect Size and Power
For the treatment effect example in the chapter:
- Calculate Cohen's d effect size
- Determine if the study is adequately powered
- Suggest sample size for 80% power

### Exercise 8: Confidence Interval Width Analysis
Investigate how confidence interval width changes with:
- Sample size (use different subsamples)
- Confidence level (80%, 90%, 95%, 99%)
- Population variability (compare different variables)

### Exercise 9: Practical Applications
Choose a real dataset and:
- Calculate confidence intervals for relevant parameters
- Interpret the results in context
- Discuss practical implications

### Exercise 10: Advanced Bootstrap
Implement different bootstrap confidence interval methods:
- Normal bootstrap
- Percentile bootstrap
- Basic bootstrap
- BCa bootstrap (if available)

Compare the results and discuss when each method is appropriate.

---

**Solutions and Additional Resources:**
- Use `?t.test` for help with t-test functions
- Use `?binom.test` for help with binomial tests
- Use `?boot` for help with bootstrap functions
- Consider using `ggplot2` for visualization
- Practice with different datasets to build intuition

## Next Steps

In the next chapter, we'll learn about hypothesis testing, which complements confidence intervals for statistical inference.

---

**Key Takeaways:**
- Confidence intervals provide ranges of plausible parameter values
- Use t-distribution when population standard deviation is unknown
- Bootstrap methods are useful for non-parametric inference
- Sample size affects the width of confidence intervals
- Higher confidence levels result in wider intervals
- Always interpret confidence intervals correctly
- Consider multiple comparisons when making many inferences 