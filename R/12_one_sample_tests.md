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

```r
# Load sample data
data(mtcars)

# Test if the mean MPG is significantly different from 20
mpg_test <- t.test(mtcars$mpg, mu = 20)
print(mpg_test)

# Extract key statistics
t_statistic <- mpg_test$statistic
p_value <- mpg_test$p.value
confidence_interval <- mpg_test$conf.int
sample_mean <- mpg_test$estimate
hypothesized_mean <- mpg_test$null.value

cat("Test Results:\n")
cat("Sample mean:", round(sample_mean, 3), "\n")
cat("Hypothesized mean:", hypothesized_mean, "\n")
cat("t-statistic:", round(t_statistic, 3), "\n")
cat("p-value:", round(p_value, 4), "\n")
cat("95% Confidence Interval:", round(confidence_interval, 3), "\n")

# Manual calculation for understanding
n <- length(mtcars$mpg)
sample_sd <- sd(mtcars$mpg)
standard_error <- sample_sd / sqrt(n)
manual_t <- (sample_mean - hypothesized_mean) / standard_error

cat("\nManual Calculation Verification:\n")
cat("Sample SD:", round(sample_sd, 3), "\n")
cat("Standard Error:", round(standard_error, 3), "\n")
cat("Manual t-statistic:", round(manual_t, 3), "\n")
cat("Degrees of freedom:", n - 1, "\n")
```

### One-Sample t-Test with Different Alternatives

The choice of alternative hypothesis depends on your research question and prior knowledge about the direction of the effect.

**Types of Alternative Hypotheses:**

1. **Two-tailed (two.sided):** Tests for any difference from the hypothesized value
   - H₀: $\mu = \mu_0$
   - H₁: $\mu \neq \mu_0$

2. **One-tailed greater:** Tests if the population mean is greater than hypothesized value
   - H₀: $\mu = \mu_0$
   - H₁: $\mu > \mu_0$

3. **One-tailed less:** Tests if the population mean is less than hypothesized value
   - H₀: $\mu = \mu_0$
   - H₁: $\mu < \mu_0$

**Mathematical Differences:**
- Two-tailed p-value: $P(|T| > |t_{obs}|)$
- One-tailed greater: $P(T > t_{obs})$
- One-tailed less: $P(T < t_{obs})$

```r
# Two-tailed test (default)
two_tailed_test <- t.test(mtcars$mpg, mu = 20, alternative = "two.sided")
print(two_tailed_test)

# One-tailed test (greater than)
greater_test <- t.test(mtcars$mpg, mu = 20, alternative = "greater")
print(greater_test)

# One-tailed test (less than)
less_test <- t.test(mtcars$mpg, mu = 20, alternative = "less")
print(less_test)

# Compare p-values
cat("P-values comparison:\n")
cat("Two-tailed:", round(two_tailed_test$p.value, 4), "\n")
cat("Greater than:", round(greater_test$p.value, 4), "\n")
cat("Less than:", round(less_test$p.value, 4), "\n")

# Relationship between p-values
cat("\nP-value relationships:\n")
cat("Two-tailed p-value ≈ 2 × min(one-tailed p-values)\n")
cat("Greater p-value + Less p-value = 1\n")

# Critical values comparison
alpha <- 0.05
df <- length(mtcars$mpg) - 1
critical_two_tailed <- qt(1 - alpha/2, df)
critical_one_tailed <- qt(1 - alpha, df)

cat("\nCritical values (α = 0.05):\n")
cat("Two-tailed critical t:", round(critical_two_tailed, 3), "\n")
cat("One-tailed critical t:", round(critical_one_tailed, 3), "\n")
```

### Effect Size for One-Sample t-Test

Effect size measures the magnitude of the difference between the sample mean and hypothesized population mean, independent of sample size. This is crucial for understanding practical significance beyond statistical significance.

**Mathematical Foundation:**

**Cohen's d (Standardized Mean Difference):**
```math
d = \frac{\bar{x} - \mu_0}{s}
```

**Hedges' g (Unbiased Estimator):**
```math
g = d \cdot \left(1 - \frac{3}{4(n-1) - 1}\right)
```

**Confidence Interval for Effect Size:**
```math
CI_d = d \pm t_{\alpha/2, df} \cdot SE_d
```

where $SE_d = \sqrt{\frac{1}{n} + \frac{d^2}{2n}}$

**Interpretation Guidelines:**
- Small effect: $|d| < 0.2$
- Medium effect: $0.2 \leq |d| < 0.5$
- Large effect: $0.5 \leq |d| < 0.8$
- Very large effect: $|d| \geq 0.8$

```r
# Calculate Cohen's d effect size
calculate_cohens_d <- function(sample_data, hypothesized_mean) {
  sample_mean <- mean(sample_data, na.rm = TRUE)
  sample_sd <- sd(sample_data, na.rm = TRUE)
  n <- length(sample_data)
  
  # Cohen's d
  cohens_d <- (sample_mean - hypothesized_mean) / sample_sd
  
  # Hedges' g (unbiased estimator)
  hedges_g <- cohens_d * (1 - 3 / (4 * (n - 1) - 1))
  
  # Standard error of effect size
  se_d <- sqrt(1/n + cohens_d^2/(2*n))
  
  # Confidence interval for effect size
  df <- n - 1
  t_critical <- qt(0.975, df)
  ci_lower <- cohens_d - t_critical * se_d
  ci_upper <- cohens_d + t_critical * se_d
  
  return(list(
    cohens_d = cohens_d,
    hedges_g = hedges_g,
    se_d = se_d,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    sample_mean = sample_mean,
    sample_sd = sample_sd,
    n = n
  ))
}

# Apply to MPG data
mpg_effect_size <- calculate_cohens_d(mtcars$mpg, 20)

cat("Effect Size Analysis:\n")
cat("Cohen's d:", round(mpg_effect_size$cohens_d, 3), "\n")
cat("Hedges' g:", round(mpg_effect_size$hedges_g, 3), "\n")
cat("Standard Error of d:", round(mpg_effect_size$se_d, 3), "\n")
cat("95% CI for d:", round(c(mpg_effect_size$ci_lower, mpg_effect_size$ci_upper), 3), "\n")

# Interpret effect size
interpret_effect_size <- function(d) {
  if (abs(d) < 0.2) {
    return("Small effect")
  } else if (abs(d) < 0.5) {
    return("Medium effect")
  } else if (abs(d) < 0.8) {
    return("Large effect")
  } else {
    return("Very large effect")
  }
}

cat("Effect size interpretation:", interpret_effect_size(mpg_effect_size$cohens_d), "\n")

# Power analysis based on effect size
library(pwr)
power_analysis <- pwr.t.test(d = mpg_effect_size$cohens_d, n = mpg_effect_size$n, 
                            sig.level = 0.05, type = "one.sample")
cat("Power for current effect size:", round(power_analysis$power, 3), "\n")
```

## One-Sample z-Test

The z-test is used when the population standard deviation is known, which is rare in practice but theoretically important. It's more powerful than the t-test when the population standard deviation is known with certainty.

### Mathematical Foundation

**Test Statistic:**
```math
z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}
```

**Confidence Interval:**
```math
CI = \bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
```

**P-value Calculations:**
- Two-tailed: $P(|Z| > |z_{obs}|) = 2(1 - \Phi(|z_{obs}|))$
- One-tailed greater: $P(Z > z_{obs}) = 1 - \Phi(z_{obs})$
- One-tailed less: $P(Z < z_{obs}) = \Phi(z_{obs})$

where $\Phi(z)$ is the standard normal cumulative distribution function.

**When to Use z-test vs t-test:**
- Use z-test when population standard deviation ($\sigma$) is known
- Use t-test when population standard deviation is unknown (estimated from sample)
- For large samples ($n > 30$), t-test approximates z-test due to Central Limit Theorem

### Basic One-Sample z-Test

```r
# Function to perform one-sample z-test
one_sample_z_test <- function(sample_data, hypothesized_mean, population_sd, alpha = 0.05) {
  sample_mean <- mean(sample_data, na.rm = TRUE)
  n <- length(sample_data)
  
  # Calculate z-statistic
  z_statistic <- (sample_mean - hypothesized_mean) / (population_sd / sqrt(n))
  
  # Calculate p-value
  p_value_two_tailed <- 2 * (1 - pnorm(abs(z_statistic)))
  p_value_greater <- 1 - pnorm(z_statistic)
  p_value_less <- pnorm(z_statistic)
  
  # Calculate confidence interval
  margin_of_error <- qnorm(1 - alpha/2) * (population_sd / sqrt(n))
  ci_lower <- sample_mean - margin_of_error
  ci_upper <- sample_mean + margin_of_error
  
  # Calculate effect size (Cohen's d using population SD)
  cohens_d <- (sample_mean - hypothesized_mean) / population_sd
  
  return(list(
    z_statistic = z_statistic,
    p_value_two_tailed = p_value_two_tailed,
    p_value_greater = p_value_greater,
    p_value_less = p_value_less,
    sample_mean = sample_mean,
    hypothesized_mean = hypothesized_mean,
    confidence_interval = c(ci_lower, ci_upper),
    margin_of_error = margin_of_error,
    cohens_d = cohens_d,
    n = n
  ))
}

# Example: Test if MPG mean is different from 20 (assuming known population SD = 6)
mpg_z_test <- one_sample_z_test(mtcars$mpg, 20, 6)

cat("One-Sample Z-Test Results:\n")
cat("Sample mean:", round(mpg_z_test$sample_mean, 3), "\n")
cat("Hypothesized mean:", mpg_z_test$hypothesized_mean, "\n")
cat("Population SD:", 6, "\n")
cat("z-statistic:", round(mpg_z_test$z_statistic, 3), "\n")
cat("Two-tailed p-value:", round(mpg_z_test$p_value_two_tailed, 4), "\n")
cat("95% Confidence Interval:", round(mpg_z_test$confidence_interval, 3), "\n")
cat("Effect size (Cohen's d):", round(mpg_z_test$cohens_d, 3), "\n")

# Compare with t-test results
mpg_t_test <- t.test(mtcars$mpg, mu = 20)
cat("\nComparison with t-test:\n")
cat("t-statistic:", round(mpg_t_test$statistic, 3), "\n")
cat("t-test p-value:", round(mpg_t_test$p.value, 4), "\n")
cat("z-test p-value:", round(mpg_z_test$p_value_two_tailed, 4), "\n")
```

## Nonparametric One-Sample Tests

Nonparametric tests make fewer assumptions about the underlying population distribution and are robust to violations of normality. They're particularly useful for small samples or when data is skewed or contains outliers.

### Wilcoxon Signed-Rank Test

The Wilcoxon signed-rank test is the most commonly used nonparametric alternative to the one-sample t-test. It tests whether the median differs from a hypothesized value by ranking the absolute differences.

**Mathematical Foundation:**

1. Calculate differences: $d_i = x_i - \mu_0$
2. Rank absolute differences: $R_i = rank(|d_i|)$
3. Assign signs: $s_i = sign(d_i)$
4. Calculate test statistic: $W = \sum_{i=1}^{n} s_i \cdot R_i$

**For large samples ($n > 20$), W is approximately normal:**
```math
Z = \frac{W}{\sqrt{\frac{n(n+1)(2n+1)}{6}}}
```

**Effect Size (r):**
```math
r = \frac{Z}{\sqrt{n}}
```

```r
# Wilcoxon signed-rank test (nonparametric alternative to t-test)
wilcox_test <- wilcox.test(mtcars$mpg, mu = 20, alternative = "two.sided")
print(wilcox_test)

# Manual calculation for understanding
differences <- mtcars$mpg - 20
abs_differences <- abs(differences)
ranks <- rank(abs_differences)
signed_ranks <- sign(differences) * ranks
W_statistic <- sum(signed_ranks)

cat("\nManual Wilcoxon Calculation:\n")
cat("W statistic:", W_statistic, "\n")
cat("R function W statistic:", wilcox_test$statistic, "\n")

# Compare with t-test results
cat("\nComparison of t-test and Wilcoxon test:\n")
cat("t-test p-value:", round(t.test(mtcars$mpg, mu = 20)$p.value, 4), "\n")
cat("Wilcoxon p-value:", round(wilcox_test$p.value, 4), "\n")

# Effect size for Wilcoxon test
wilcox_effect_size <- abs(qnorm(wilcox_test$p.value / 2)) / sqrt(length(mtcars$mpg))
cat("Wilcoxon effect size (r):", round(wilcox_effect_size, 3), "\n")

# Interpret Wilcoxon effect size
interpret_wilcox_effect <- function(r) {
  if (abs(r) < 0.1) {
    return("Small effect")
  } else if (abs(r) < 0.3) {
    return("Medium effect")
  } else if (abs(r) < 0.5) {
    return("Large effect")
  } else {
    return("Very large effect")
  }
}

cat("Wilcoxon effect interpretation:", interpret_wilcox_effect(wilcox_effect_size), "\n")
```

### Sign Test

The sign test is the simplest nonparametric test that only considers whether observations are above or below the hypothesized median. It's less powerful than the Wilcoxon test but makes the fewest assumptions.

**Mathematical Foundation:**

The sign test is based on the binomial distribution with $p = 0.5$ under the null hypothesis.

**Test Statistic:**
- $S^+$ = number of positive differences
- $S^-$ = number of negative differences
- $n = S^+ + S^-$ (excluding ties)

**Under H₀:** $S^+ \sim Binomial(n, 0.5)$

**P-value Calculations:**
- Two-tailed: $P = 2 \cdot P(S^+ \leq \min(S^+, S^-))$
- One-tailed greater: $P = P(S^+ \geq S^+_{obs})$
- One-tailed less: $P = P(S^+ \leq S^+_{obs})$

**For large samples ($n > 20$), normal approximation:**
```math
Z = \frac{S^+ - n/2}{\sqrt{n/4}}
```

```r
# Function to perform sign test
sign_test <- function(sample_data, hypothesized_median) {
  differences <- sample_data - hypothesized_median
  positive_signs <- sum(differences > 0, na.rm = TRUE)
  negative_signs <- sum(differences < 0, na.rm = TRUE)
  n <- positive_signs + negative_signs
  
  # Binomial test
  p_value_two_tailed <- 2 * pbinom(min(positive_signs, negative_signs), n, 0.5)
  p_value_greater <- 1 - pbinom(positive_signs - 1, n, 0.5)
  p_value_less <- pbinom(positive_signs, n, 0.5)
  
  # Normal approximation for large samples
  if (n > 20) {
    z_statistic <- (positive_signs - n/2) / sqrt(n/4)
    p_value_normal <- 2 * (1 - pnorm(abs(z_statistic)))
  } else {
    z_statistic <- NA
    p_value_normal <- NA
  }
  
  return(list(
    positive_signs = positive_signs,
    negative_signs = negative_signs,
    n = n,
    p_value_two_tailed = p_value_two_tailed,
    p_value_greater = p_value_greater,
    p_value_less = p_value_less,
    z_statistic = z_statistic,
    p_value_normal = p_value_normal
  ))
}

# Apply sign test to MPG data
mpg_sign_test <- sign_test(mtcars$mpg, 20)

cat("Sign Test Results:\n")
cat("Positive signs:", mpg_sign_test$positive_signs, "\n")
cat("Negative signs:", mpg_sign_test$negative_signs, "\n")
cat("Total observations (excluding ties):", mpg_sign_test$n, "\n")
cat("Two-tailed p-value (exact):", round(mpg_sign_test$p_value_two_tailed, 4), "\n")

if (!is.na(mpg_sign_test$z_statistic)) {
  cat("z-statistic (normal approx):", round(mpg_sign_test$z_statistic, 3), "\n")
  cat("Two-tailed p-value (normal):", round(mpg_sign_test$p_value_normal, 4), "\n")
}

# Compare with other tests
cat("\nComparison with other tests:\n")
cat("t-test p-value:", round(t.test(mtcars$mpg, mu = 20)$p.value, 4), "\n")
cat("Wilcoxon p-value:", round(wilcox.test(mtcars$mpg, mu = 20)$p.value, 4), "\n")
cat("Sign test p-value:", round(mpg_sign_test$p_value_two_tailed, 4), "\n")
```

## Power Analysis

Power analysis helps determine the probability of detecting a true effect and the sample size needed for adequate statistical power. This is crucial for study design and interpreting results.

### Mathematical Foundation

**Power Definition:**
```math
Power = P(\text{Reject } H_0 | H_1 \text{ is true}) = 1 - \beta
```

**For one-sample t-test:**
```math
Power = P\left(|T| > t_{\alpha/2, df} \mid \mu = \mu_0 + \delta\right)
```

where $\delta$ is the true effect size and $t_{\alpha/2, df}$ is the critical t-value.

**Effect Size and Power Relationship:**
- Larger effect sizes require smaller sample sizes for the same power
- Smaller effect sizes require larger sample sizes for adequate power
- Power increases with sample size for fixed effect sizes

### Power Analysis for One-Sample t-Test

```r
library(pwr)

# Comprehensive power analysis function
power_analysis <- function(sample_size, effect_size, alpha = 0.05) {
  # Calculate power for current sample size
  power_result <- pwr.t.test(n = sample_size, d = effect_size, sig.level = alpha, type = "one.sample")
  
  # Calculate required sample size for 80% power
  sample_size_80 <- pwr.t.test(d = effect_size, sig.level = alpha, power = 0.8, type = "one.sample")
  
  # Calculate required sample size for 90% power
  sample_size_90 <- pwr.t.test(d = effect_size, sig.level = alpha, power = 0.9, type = "one.sample")
  
  # Calculate power for different effect sizes
  small_effect_power <- pwr.t.test(n = sample_size, d = 0.2, sig.level = alpha, type = "one.sample")$power
  medium_effect_power <- pwr.t.test(n = sample_size, d = 0.5, sig.level = alpha, type = "one.sample")$power
  large_effect_power <- pwr.t.test(n = sample_size, d = 0.8, sig.level = alpha, type = "one.sample")$power
  
  return(list(
    power = power_result$power,
    required_n_80 = sample_size_80$n,
    required_n_90 = sample_size_90$n,
    effect_size = effect_size,
    alpha = alpha,
    small_effect_power = small_effect_power,
    medium_effect_power = medium_effect_power,
    large_effect_power = large_effect_power
  ))
}

# Apply to MPG data
mpg_power <- power_analysis(length(mtcars$mpg), mpg_effect_size$cohens_d)

cat("Power Analysis Results:\n")
cat("Current effect size:", round(mpg_power$effect_size, 3), "\n")
cat("Current power:", round(mpg_power$power, 3), "\n")
cat("Required sample size for 80% power:", ceiling(mpg_power$required_n_80), "\n")
cat("Required sample size for 90% power:", ceiling(mpg_power$required_n_90), "\n")

cat("\nPower for different effect sizes:\n")
cat("Small effect (d = 0.2):", round(mpg_power$small_effect_power, 3), "\n")
cat("Medium effect (d = 0.5):", round(mpg_power$medium_effect_power, 3), "\n")
cat("Large effect (d = 0.8):", round(mpg_power$large_effect_power, 3), "\n")

# Power curve analysis
effect_sizes <- seq(0.1, 1.0, by = 0.1)
power_curve <- sapply(effect_sizes, function(d) {
  pwr.t.test(n = length(mtcars$mpg), d = d, sig.level = 0.05, type = "one.sample")$power
})

cat("\nPower curve (effect size vs power):\n")
for (i in 1:length(effect_sizes)) {
  cat("d =", effect_sizes[i], ": Power =", round(power_curve[i], 3), "\n")
}
```

## Assumption Checking

Proper assumption checking is crucial for valid statistical inference. Violations of assumptions can lead to incorrect conclusions and inflated Type I error rates.

### Key Assumptions for One-Sample Tests

1. **Independence:** Observations are independent of each other
2. **Normality:** Data follows a normal distribution (for parametric tests)
3. **Random Sampling:** Data represents a random sample from the population
4. **No Outliers:** Extreme values don't unduly influence results

### Normality Test

The normality assumption is critical for parametric tests like the t-test. Several methods can assess normality:

**Mathematical Tests:**
- **Shapiro-Wilk:** Most powerful test for normality
- **Anderson-Darling:** Good for detecting departures from normality
- **Kolmogorov-Smirnov:** Tests against a specified distribution

**Graphical Methods:**
- **Q-Q plots:** Compare sample quantiles to theoretical normal quantiles
- **Histograms:** Visual assessment of distribution shape
- **Box plots:** Detect skewness and outliers

```r
# Comprehensive normality checking function
check_normality <- function(data) {
  # Shapiro-Wilk test
  shapiro_test <- shapiro.test(data)
  
  # Anderson-Darling test
  library(nortest)
  ad_test <- ad.test(data)
  
  # Kolmogorov-Smirnov test
  ks_test <- ks.test(data, "pnorm", mean = mean(data), sd = sd(data))
  
  # Descriptive statistics for normality
  skewness <- moments::skewness(data)
  kurtosis <- moments::kurtosis(data)
  
  # Q-Q plot
  qq_plot <- ggplot(data.frame(x = data), aes(sample = x)) +
    stat_qq() +
    stat_qq_line() +
    labs(title = "Q-Q Plot for Normality Check",
         subtitle = paste("Shapiro-Wilk p =", round(shapiro_test$p.value, 4))) +
    theme_minimal()
  
  # Histogram with normal curve
  hist_plot <- ggplot(data.frame(x = data), aes(x = x)) +
    geom_histogram(aes(y = ..density..), bins = 15, fill = "steelblue", alpha = 0.7) +
    stat_function(fun = dnorm, args = list(mean = mean(data), sd = sd(data)), 
                 color = "red", size = 1) +
    labs(title = "Histogram with Normal Curve",
         subtitle = paste("Skewness =", round(skewness, 3), 
                         "Kurtosis =", round(kurtosis, 3))) +
    theme_minimal()
  
  # Box plot
  box_plot <- ggplot(data.frame(x = data), aes(y = x)) +
    geom_boxplot(fill = "steelblue", alpha = 0.7) +
    labs(title = "Box Plot for Outlier Detection") +
    theme_minimal()
  
  cat("=== NORMALITY ASSESSMENT ===\n")
  cat("Sample size:", length(data), "\n")
  cat("Mean:", round(mean(data), 3), "\n")
  cat("SD:", round(sd(data), 3), "\n")
  cat("Skewness:", round(skewness, 3), "\n")
  cat("Kurtosis:", round(kurtosis, 3), "\n\n")
  
  cat("Normality Tests:\n")
  cat("Shapiro-Wilk p-value:", round(shapiro_test$p.value, 4), "\n")
  cat("Anderson-Darling p-value:", round(ad_test$p.value, 4), "\n")
  cat("Kolmogorov-Smirnov p-value:", round(ks_test$p.value, 4), "\n\n")
  
  # Interpretation
  cat("Interpretation:\n")
  if (shapiro_test$p.value >= 0.05) {
    cat("✓ Data appears to be normally distributed\n")
  } else {
    cat("✗ Data is not normally distributed\n")
  }
  
  if (abs(skewness) < 1) {
    cat("✓ Skewness is acceptable\n")
  } else {
    cat("✗ Data is significantly skewed\n")
  }
  
  if (abs(kurtosis - 3) < 2) {
    cat("✓ Kurtosis is acceptable\n")
  } else {
    cat("✗ Data has unusual kurtosis\n")
  }
  
  return(list(
    shapiro_test = shapiro_test,
    ad_test = ad_test,
    ks_test = ks_test,
    skewness = skewness,
    kurtosis = kurtosis,
    qq_plot = qq_plot,
    hist_plot = hist_plot,
    box_plot = box_plot
  ))
}

# Check normality of MPG data
mpg_normality <- check_normality(mtcars$mpg)
```

### Outlier Detection

Outliers can significantly influence test results and should be identified and handled appropriately. Different methods have different sensitivities to outliers.

**Mathematical Methods:**

1. **IQR Method (Tukey's Fences):**
   ```math
   \text{Lower bound} = Q_1 - 1.5 \times IQR
   \text{Upper bound} = Q_3 + 1.5 \times IQR
   ```

2. **Z-score Method:**
   ```math
   z_i = \frac{x_i - \bar{x}}{s}
   ```
   Values with $|z_i| > 3$ are considered outliers.

3. **Modified Z-score Method (Robust):**
   ```math
   M_i = \frac{0.6745(x_i - \text{median})}{\text{MAD}}
   ```
   Values with $|M_i| > 3.5$ are considered outliers.

4. **Mahalanobis Distance (Multivariate):**
   ```math
   D^2 = (x - \mu)^T \Sigma^{-1} (x - \mu)
   ```

```r
# Comprehensive outlier detection function
detect_outliers <- function(data) {
  # IQR method
  q1 <- quantile(data, 0.25, na.rm = TRUE)
  q3 <- quantile(data, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound_iqr <- q1 - 1.5 * iqr
  upper_bound_iqr <- q3 + 1.5 * iqr
  
  outliers_iqr <- data < lower_bound_iqr | data > upper_bound_iqr
  
  # Z-score method
  z_scores <- abs((data - mean(data, na.rm = TRUE)) / sd(data, na.rm = TRUE))
  outliers_z <- z_scores > 3
  
  # Modified Z-score method (robust)
  median_val <- median(data, na.rm = TRUE)
  mad_val <- mad(data, na.rm = TRUE)
  modified_z_scores <- abs((data - median_val) / mad_val)
  outliers_modified_z <- modified_z_scores > 3.5
  
  # Extreme IQR method (more conservative)
  lower_bound_extreme <- q1 - 3 * iqr
  upper_bound_extreme <- q3 + 3 * iqr
  outliers_extreme_iqr <- data < lower_bound_extreme | data > upper_bound_extreme
  
  # Percentile method
  outliers_percentile <- data < quantile(data, 0.01, na.rm = TRUE) | 
                        data > quantile(data, 0.99, na.rm = TRUE)
  
  return(list(
    outliers_iqr = which(outliers_iqr),
    outliers_z = which(outliers_z),
    outliers_modified_z = which(outliers_modified_z),
    outliers_extreme_iqr = which(outliers_extreme_iqr),
    outliers_percentile = which(outliers_percentile),
    bounds_iqr = c(lower_bound_iqr, upper_bound_iqr),
    bounds_extreme_iqr = c(lower_bound_extreme, upper_bound_extreme),
    z_scores = z_scores,
    modified_z_scores = modified_z_scores,
    q1 = q1,
    q3 = q3,
    iqr = iqr
  ))
}

# Detect outliers in MPG data
mpg_outliers <- detect_outliers(mtcars$mpg)

cat("=== OUTLIER DETECTION ===\n")
cat("Sample size:", length(mtcars$mpg), "\n")
cat("Range:", round(range(mtcars$mpg), 2), "\n")
cat("Q1:", round(mpg_outliers$q1, 2), "\n")
cat("Q3:", round(mpg_outliers$q3, 2), "\n")
cat("IQR:", round(mpg_outliers$iqr, 2), "\n\n")

cat("Outlier Detection Results:\n")
cat("IQR method outliers:", length(mpg_outliers$outliers_iqr), "\n")
cat("Z-score method outliers:", length(mpg_outliers$outliers_z), "\n")
cat("Modified Z-score outliers:", length(mpg_outliers$outliers_modified_z), "\n")
cat("Extreme IQR outliers:", length(mpg_outliers$outliers_extreme_iqr), "\n")
cat("Percentile outliers:", length(mpg_outliers$outliers_percentile), "\n\n")

# Show outlier values
if (length(mpg_outliers$outliers_iqr) > 0) {
  cat("IQR outliers:", round(mtcars$mpg[mpg_outliers$outliers_iqr], 2), "\n")
}
if (length(mpg_outliers$outliers_z) > 0) {
  cat("Z-score outliers:", round(mtcars$mpg[mpg_outliers$outliers_z], 2), "\n")
}

# Impact analysis
cat("\nImpact Analysis:\n")
original_mean <- mean(mtcars$mpg)
original_sd <- sd(mtcars$mpg)

# Remove IQR outliers
clean_data_iqr <- mtcars$mpg[-mpg_outliers$outliers_iqr]
if (length(clean_data_iqr) < length(mtcars$mpg)) {
  cat("Mean without IQR outliers:", round(mean(clean_data_iqr), 3), 
      "(original:", round(original_mean, 3), ")\n")
  cat("SD without IQR outliers:", round(sd(clean_data_iqr), 3), 
      "(original:", round(original_sd, 3), ")\n")
}
```

## Practical Examples

Real-world applications of one-sample tests demonstrate their importance across various fields. These examples show how to apply the concepts in practice.

### Example 1: Quality Control

Quality control is a common application where one-sample tests are used to ensure products meet specifications.

**Scenario:** A manufacturing plant produces widgets with a target weight of 100 grams. A sample of 50 widgets is tested to ensure the production process is working correctly.

```r
# Simulate quality control data
set.seed(123)
n_products <- 50
product_weights <- rnorm(n_products, mean = 100, sd = 5)

# Comprehensive quality control analysis
cat("=== QUALITY CONTROL ANALYSIS ===\n")
cat("Target weight: 100 grams\n")
cat("Sample size:", n_products, "\n")
cat("Sample mean:", round(mean(product_weights), 2), "grams\n")
cat("Sample SD:", round(sd(product_weights), 2), "grams\n")

# Test if mean weight is 100 grams
weight_test <- t.test(product_weights, mu = 100)
print(weight_test)

# Calculate effect size
weight_effect <- calculate_cohens_d(product_weights, 100)

cat("\nQuality Control Results:\n")
cat("Effect size:", round(weight_effect$cohens_d, 3), "\n")
cat("Interpretation:", interpret_effect_size(weight_effect$cohens_d), "\n")

# Tolerance analysis
tolerance_limits <- c(95, 105)  # ±5 grams tolerance
within_tolerance <- sum(product_weights >= tolerance_limits[1] & 
                       product_weights <= tolerance_limits[2])
tolerance_percentage <- within_tolerance / n_products * 100

cat("\nTolerance Analysis:\n")
cat("Within ±5g tolerance:", within_tolerance, "/", n_products, 
    "(", round(tolerance_percentage, 1), "%)\n")

# Process capability
process_capability <- (tolerance_limits[2] - tolerance_limits[1]) / (6 * sd(product_weights))
cat("Process capability (Cpk):", round(process_capability, 3), "\n")

# Decision making
alpha <- 0.05
if (weight_test$p.value < alpha) {
  cat("\nDECISION: Process adjustment needed (p <", alpha, ")\n")
} else {
  cat("\nDECISION: Process is in control (p >=", alpha, ")\n")
}
```

### Example 2: Educational Assessment

Educational assessment uses one-sample tests to evaluate whether students meet learning objectives or pass thresholds.

**Scenario:** A teacher wants to determine if her class of 30 students has achieved a passing score (70%) on a standardized test, with the goal of demonstrating that the class performs above the minimum threshold.

```r
# Simulate test scores
set.seed(123)
n_students <- 30
test_scores <- rnorm(n_students, mean = 75, sd = 10)

# Comprehensive educational assessment
cat("=== EDUCATIONAL ASSESSMENT ===\n")
cat("Passing threshold: 70%\n")
cat("Sample size:", n_students, "students\n")
cat("Sample mean:", round(mean(test_scores), 1), "%\n")
cat("Sample SD:", round(sd(test_scores), 1), "%\n")
cat("Range:", round(range(test_scores), 1), "%\n")

# Descriptive statistics
passing_students <- sum(test_scores >= 70)
passing_rate <- passing_students / n_students * 100

cat("\nDescriptive Statistics:\n")
cat("Students passing (≥70%):", passing_students, "/", n_students, 
    "(", round(passing_rate, 1), "%)\n")
cat("Students below threshold:", n_students - passing_students, "\n")

# Test if mean score is above 70 (passing threshold)
passing_test <- t.test(test_scores, mu = 70, alternative = "greater")
print(passing_test)

# Calculate confidence interval
ci_result <- t.test(test_scores, mu = 70, conf.level = 0.95)
cat("95% Confidence Interval:", round(ci_result$conf.int, 2), "%\n")

# Effect size
score_effect <- calculate_cohens_d(test_scores, 70)
cat("Effect size:", round(score_effect$cohens_d, 3), "\n")
cat("Effect interpretation:", interpret_effect_size(score_effect$cohens_d), "\n")

# Power analysis
power_analysis <- pwr.t.test(n = n_students, d = score_effect$cohens_d, 
                            sig.level = 0.05, type = "one.sample", alternative = "greater")
cat("Power for detecting this effect:", round(power_analysis$power, 3), "\n")

# Educational interpretation
alpha <- 0.05
if (passing_test$p.value < alpha) {
  cat("\nEDUCATIONAL CONCLUSION:\n")
  cat("✓ The class significantly exceeds the passing threshold (p <", alpha, ")\n")
  cat("✓ The intervention/teaching method appears effective\n")
  cat("✓ Consider advancing to more challenging material\n")
} else {
  cat("\nEDUCATIONAL CONCLUSION:\n")
  cat("✗ The class does not significantly exceed the passing threshold (p >=", alpha, ")\n")
  cat("✗ Additional instruction may be needed\n")
  cat("✗ Consider reviewing foundational concepts\n")
}

# Performance categories
excellent <- sum(test_scores >= 90)
good <- sum(test_scores >= 80 & test_scores < 90)
satisfactory <- sum(test_scores >= 70 & test_scores < 80)
needs_improvement <- sum(test_scores < 70)

cat("\nPerformance Distribution:\n")
cat("Excellent (≥90%):", excellent, "students\n")
cat("Good (80-89%):", good, "students\n")
cat("Satisfactory (70-79%):", satisfactory, "students\n")
cat("Needs Improvement (<70%):", needs_improvement, "students\n")
```

### Example 3: Medical Research

Medical research frequently uses one-sample tests to compare patient outcomes against established norms or baseline values.

**Scenario:** A researcher wants to determine if patients with a specific condition have elevated systolic blood pressure compared to the normal range of 120 mmHg.

```r
# Simulate blood pressure data
set.seed(123)
n_patients <- 25
systolic_bp <- rnorm(n_patients, mean = 130, sd = 15)

# Comprehensive medical analysis
cat("=== MEDICAL RESEARCH ANALYSIS ===\n")
cat("Normal systolic BP: 120 mmHg\n")
cat("Sample size:", n_patients, "patients\n")
cat("Sample mean:", round(mean(systolic_bp), 1), "mmHg\n")
cat("Sample SD:", round(sd(systolic_bp), 1), "mmHg\n")
cat("Range:", round(range(systolic_bp), 1), "mmHg\n")

# Clinical categories
normal_bp <- sum(systolic_bp < 120)
elevated_bp <- sum(systolic_bp >= 120 & systolic_bp < 130)
stage1_hypertension <- sum(systolic_bp >= 130 & systolic_bp < 140)
stage2_hypertension <- sum(systolic_bp >= 140)

cat("\nClinical Classification:\n")
cat("Normal (<120):", normal_bp, "patients\n")
cat("Elevated (120-129):", elevated_bp, "patients\n")
cat("Stage 1 Hypertension (130-139):", stage1_hypertension, "patients\n")
cat("Stage 2 Hypertension (≥140):", stage2_hypertension, "patients\n")

# Test if mean systolic BP is different from 120 (normal)
bp_test <- t.test(systolic_bp, mu = 120)
print(bp_test)

# Nonparametric alternative
bp_wilcox <- wilcox.test(systolic_bp, mu = 120)
print(bp_wilcox)

# Effect size
bp_effect <- calculate_cohens_d(systolic_bp, 120)
cat("Effect size:", round(bp_effect$cohens_d, 3), "\n")
cat("Effect interpretation:", interpret_effect_size(bp_effect$cohens_d), "\n")

# Compare parametric and nonparametric results
cat("\nTest Comparison:\n")
cat("t-test p-value:", round(bp_test$p.value, 4), "\n")
cat("Wilcoxon p-value:", round(bp_wilcox$p.value, 4), "\n")

# Clinical interpretation
alpha <- 0.05
if (bp_test$p.value < alpha) {
  cat("\nCLINICAL CONCLUSION:\n")
  cat("✓ Patients have significantly elevated BP compared to normal (p <", alpha, ")\n")
  cat("✓ Clinical intervention may be warranted\n")
  cat("✓ Consider lifestyle modifications or medication\n")
} else {
  cat("\nCLINICAL CONCLUSION:\n")
  cat("✗ No significant elevation in BP compared to normal (p >=", alpha, ")\n")
  cat("✗ Continue monitoring as standard\n")
}

# Risk assessment
high_risk_patients <- sum(systolic_bp >= 140)
risk_percentage <- high_risk_patients / n_patients * 100

cat("\nRisk Assessment:\n")
cat("High-risk patients (≥140 mmHg):", high_risk_patients, "/", n_patients, 
    "(", round(risk_percentage, 1), "%)\n")

# Confidence interval for clinical decision making
bp_ci <- t.test(systolic_bp, mu = 120, conf.level = 0.95)
cat("95% CI for mean BP:", round(bp_ci$conf.int, 1), "mmHg\n")

# Power analysis for future studies
bp_power <- pwr.t.test(n = n_patients, d = bp_effect$cohens_d, 
                       sig.level = 0.05, type = "one.sample")
cat("Power for detecting this effect:", round(bp_power$power, 3), "\n")

# Sample size recommendation for future studies
if (bp_power$power < 0.8) {
  recommended_n <- pwr.t.test(d = bp_effect$cohens_d, sig.level = 0.05, 
                             power = 0.8, type = "one.sample")$n
  cat("Recommended sample size for 80% power:", ceiling(recommended_n), "\n")
}
```

## Advanced Topics

Advanced methods provide robust alternatives and enhanced inference capabilities for one-sample tests.

### Bootstrap Confidence Intervals

Bootstrap methods provide nonparametric confidence intervals that don't rely on distributional assumptions. They're particularly useful when data doesn't meet normality assumptions.

**Mathematical Foundation:**

The bootstrap estimates the sampling distribution by resampling with replacement from the original data.

**Bootstrap Algorithm:**
1. Draw $B$ bootstrap samples of size $n$ with replacement
2. Calculate statistic of interest for each bootstrap sample
3. Use empirical distribution of bootstrap statistics for inference

**Bootstrap Confidence Interval:**
```math
CI_{1-\alpha} = [\hat{\theta}_{\alpha/2}, \hat{\theta}_{1-\alpha/2}]
```

where $\hat{\theta}_p$ is the $p$-th percentile of bootstrap statistics.

```r
library(boot)

# Bootstrap function for mean
boot_mean <- function(data, indices) {
  d <- data[indices]
  return(mean(d))
}

# Comprehensive bootstrap analysis
boot_results <- boot(mtcars$mpg, boot_mean, R = 10000)
boot_ci_perc <- boot.ci(boot_results, type = "perc")
boot_ci_bca <- boot.ci(boot_results, type = "bca")
boot_ci_norm <- boot.ci(boot_results, type = "norm")

cat("=== BOOTSTRAP ANALYSIS ===\n")
cat("Number of bootstrap samples: 10,000\n")
cat("Original sample mean:", round(boot_results$t0, 3), "\n")
cat("Bootstrap mean:", round(mean(boot_results$t), 3), "\n")
cat("Bootstrap SD:", round(sd(boot_results$t), 3), "\n")

cat("\nBootstrap Confidence Intervals:\n")
cat("Percentile 95% CI:", round(boot_ci_perc$percent[4:5], 3), "\n")
cat("BCa 95% CI:", round(boot_ci_bca$bca[4:5], 3), "\n")
cat("Normal 95% CI:", round(boot_ci_norm$normal[2:3], 3), "\n")

# Compare with parametric methods
t_ci <- t.test(mtcars$mpg)$conf.int
cat("\nComparison with Parametric Methods:\n")
cat("t-test 95% CI:", round(t_ci, 3), "\n")
cat("Bootstrap percentile 95% CI:", round(boot_ci_perc$percent[4:5], 3), "\n")

# Bootstrap bias and standard error
bootstrap_bias <- mean(boot_results$t) - boot_results$t0
bootstrap_se <- sd(boot_results$t)

cat("\nBootstrap Diagnostics:\n")
cat("Bias:", round(bootstrap_bias, 4), "\n")
cat("Standard Error:", round(bootstrap_se, 4), "\n")
cat("Bias-corrected mean:", round(boot_results$t0 - bootstrap_bias, 3), "\n")
```

### Robust One-Sample Tests

Robust methods provide resistance to outliers and violations of distributional assumptions while maintaining good statistical properties.

**Mathematical Foundation:**

**Trimmed Mean:**
```math
\bar{x}_\alpha = \frac{1}{n-2k} \sum_{i=k+1}^{n-k} x_{(i)}
```

where $k = \lfloor n\alpha \rfloor$ and $x_{(i)}$ are the ordered observations.

**Winsorized Variance:**
```math
s_w^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x}_w)^2
```

where $\bar{x}_w$ is the Winsorized mean.

**Robust t-statistic:**
```math
t_{robust} = \frac{\bar{x}_\alpha - \mu_0}{s_w/\sqrt{n}}
```

```r
# Comprehensive robust testing function
robust_t_test <- function(data, mu, trim = 0.1) {
  trimmed_data <- data[!is.na(data)]
  n <- length(trimmed_data)
  k <- floor(n * trim)
  
  if (k > 0) {
    sorted_data <- sort(trimmed_data)
    trimmed_values <- sorted_data[(k + 1):(n - k)]
  } else {
    trimmed_values <- trimmed_data
  }
  
  trimmed_mean <- mean(trimmed_values)
  trimmed_var <- var(trimmed_values)
  trimmed_se <- sqrt(trimmed_var / (n - 2 * k))
  
  t_statistic <- (trimmed_mean - mu) / trimmed_se
  df <- n - 2 * k - 1
  p_value <- 2 * (1 - pt(abs(t_statistic), df))
  
  # Winsorized statistics
  winsorized_data <- data
  if (k > 0) {
    winsorized_data[winsorized_data < sorted_data[k + 1]] <- sorted_data[k + 1]
    winsorized_data[winsorized_data > sorted_data[n - k]] <- sorted_data[n - k]
  }
  winsorized_mean <- mean(winsorized_data)
  winsorized_var <- var(winsorized_data)
  
  # M-estimator (Huber's method)
  library(MASS)
  huber_result <- huber(data, mu = mu)
  
  return(list(
    trimmed_mean = trimmed_mean,
    trimmed_var = trimmed_var,
    trimmed_se = trimmed_se,
    t_statistic = t_statistic,
    p_value = p_value,
    df = df,
    winsorized_mean = winsorized_mean,
    winsorized_var = winsorized_var,
    huber_estimate = huber_result$mu,
    huber_scale = huber_result$s,
    trim_proportion = trim,
    n_trimmed = 2 * k
  ))
}

# Apply robust t-test to MPG data
mpg_robust <- robust_t_test(mtcars$mpg, 20, trim = 0.1)

cat("=== ROBUST ONE-SAMPLE TEST ===\n")
cat("Trim proportion:", mpg_robust$trim_proportion, "\n")
cat("Number of observations trimmed:", mpg_robust$n_trimmed, "\n\n")

cat("Robust Statistics:\n")
cat("Trimmed mean:", round(mpg_robust$trimmed_mean, 3), "\n")
cat("Winsorized mean:", round(mpg_robust$winsorized_mean, 3), "\n")
cat("Huber M-estimate:", round(mpg_robust$huber_estimate, 3), "\n")
cat("Original mean:", round(mean(mtcars$mpg), 3), "\n\n")

cat("Test Results:\n")
cat("Robust t-statistic:", round(mpg_robust$t_statistic, 3), "\n")
cat("Degrees of freedom:", mpg_robust$df, "\n")
cat("p-value:", round(mpg_robust$p_value, 4), "\n")

# Compare with standard t-test
standard_t <- t.test(mtcars$mpg, mu = 20)
cat("\nComparison with Standard t-test:\n")
cat("Standard t-statistic:", round(standard_t$statistic, 3), "\n")
cat("Standard p-value:", round(standard_t$p.value, 4), "\n")
cat("Robust t-statistic:", round(mpg_robust$t_statistic, 3), "\n")
cat("Robust p-value:", round(mpg_robust$p_value, 4), "\n")

# Effect size for robust test
robust_effect_size <- (mpg_robust$trimmed_mean - 20) / sqrt(mpg_robust$trimmed_var)
cat("\nRobust Effect Size:\n")
cat("Cohen's d (robust):", round(robust_effect_size, 3), "\n")
cat("Interpretation:", interpret_effect_size(robust_effect_size), "\n")
```

## Best Practices

Following best practices ensures valid statistical inference and meaningful results. These guidelines help researchers make appropriate choices and avoid common pitfalls.

### Test Selection Guidelines

The choice of test depends on data characteristics, sample size, and research objectives. This decision tree helps select the most appropriate method.

**Decision Framework:**

1. **Sample Size Consideration:**
   - $n < 15$: Nonparametric tests recommended
   - $15 \leq n < 30$: Check normality carefully
   - $n \geq 30$: Central Limit Theorem applies

2. **Distribution Assessment:**
   - Normal data: Parametric tests preferred
   - Non-normal data: Nonparametric or robust methods

3. **Outlier Sensitivity:**
   - Outliers present: Robust methods recommended
   - Clean data: Standard methods appropriate

```r
# Comprehensive test selection function
choose_one_sample_test <- function(data, hypothesized_value) {
  cat("=== ONE-SAMPLE TEST SELECTION GUIDE ===\n")
  
  # Basic information
  n <- length(data)
  cat("Sample size:", n, "\n")
  cat("Hypothesized value:", hypothesized_value, "\n")
  cat("Sample mean:", round(mean(data), 3), "\n")
  cat("Sample SD:", round(sd(data), 3), "\n\n")
  
  # Normality assessment
  shapiro_test <- shapiro.test(data)
  cat("=== NORMALITY ASSESSMENT ===\n")
  cat("Shapiro-Wilk p-value:", round(shapiro_test$p.value, 4), "\n")
  
  # Skewness and kurtosis
  skewness <- moments::skewness(data)
  kurtosis <- moments::kurtosis(data)
  cat("Skewness:", round(skewness, 3), "\n")
  cat("Kurtosis:", round(kurtosis, 3), "\n")
  
  # Outlier assessment
  outliers <- detect_outliers(data)
  n_outliers_iqr <- length(outliers$outliers_iqr)
  n_outliers_z <- length(outliers$outliers_z)
  
  cat("\n=== OUTLIER ASSESSMENT ===\n")
  cat("IQR method outliers:", n_outliers_iqr, "\n")
  cat("Z-score method outliers:", n_outliers_z, "\n")
  
  # Effect size
  effect_size <- calculate_cohens_d(data, hypothesized_value)
  cat("\n=== EFFECT SIZE ===\n")
  cat("Cohen's d:", round(effect_size$cohens_d, 3), "\n")
  cat("Interpretation:", interpret_effect_size(effect_size$cohens_d), "\n")
  
  # Recommendations
  cat("\n=== RECOMMENDATIONS ===\n")
  
  # Sample size recommendations
  if (n < 15) {
    cat("⚠️  Small sample size (n < 15):\n")
    cat("   - Nonparametric tests strongly recommended\n")
    cat("   - Consider collecting more data if possible\n")
    cat("   - Use Wilcoxon signed-rank test\n")
  } else if (n < 30) {
    cat("⚠️  Moderate sample size (15 ≤ n < 30):\n")
    cat("   - Check normality carefully\n")
    if (shapiro_test$p.value >= 0.05) {
      cat("   - Data appears normal: t-test appropriate\n")
    } else {
      cat("   - Data not normal: Use Wilcoxon test\n")
    }
  } else {
    cat("✓ Large sample size (n ≥ 30):\n")
    cat("   - Central Limit Theorem applies\n")
    cat("   - t-test is robust to normality violations\n")
  }
  
  # Normality recommendations
  if (shapiro_test$p.value < 0.05) {
    cat("✗ Normality assumption violated:\n")
    cat("   - Consider nonparametric alternatives\n")
    cat("   - Robust methods may be appropriate\n")
  } else {
    cat("✓ Normality assumption met:\n")
    cat("   - Parametric tests are appropriate\n")
  }
  
  # Outlier recommendations
  if (n_outliers_iqr > 0) {
    cat("⚠️  Outliers detected:\n")
    cat("   - Consider robust methods\n")
    cat("   - Trimmed t-test recommended\n")
    cat("   - Investigate outlier causes\n")
  } else {
    cat("✓ No significant outliers detected\n")
  }
  
  # Final recommendation
  cat("\n=== FINAL RECOMMENDATION ===\n")
  if (n >= 30) {
    cat("Primary: One-sample t-test\n")
    cat("Alternative: Bootstrap confidence intervals\n")
  } else if (shapiro_test$p.value >= 0.05 && n_outliers_iqr == 0) {
    cat("Primary: One-sample t-test\n")
    cat("Alternative: Wilcoxon signed-rank test\n")
  } else {
    cat("Primary: Wilcoxon signed-rank test\n")
    cat("Alternative: Robust t-test (trimmed mean)\n")
  }
  
  # Power consideration
  power_analysis <- pwr.t.test(n = n, d = effect_size$cohens_d, 
                              sig.level = 0.05, type = "one.sample")
  power <- power_analysis$power
  
  cat("\n=== POWER ANALYSIS ===\n")
  cat("Current power:", round(power, 3), "\n")
  if (power < 0.8) {
    recommended_n <- pwr.t.test(d = effect_size$cohens_d, sig.level = 0.05, 
                               power = 0.8, type = "one.sample")$n
    cat("Recommended sample size for 80% power:", ceiling(recommended_n), "\n")
  }
}

# Apply to MPG data
choose_one_sample_test(mtcars$mpg, 20)
```

### Reporting Guidelines

Proper reporting of statistical results is essential for transparency and reproducibility. These guidelines ensure comprehensive and clear communication of findings.

**Essential Elements for Reporting:**

1. **Descriptive Statistics:** Mean, standard deviation, sample size
2. **Test Statistics:** t-value, degrees of freedom, p-value
3. **Effect Size:** Cohen's d with interpretation
4. **Confidence Intervals:** 95% CI for the mean difference
5. **Assumption Checks:** Normality tests, outlier assessment
6. **Practical Significance:** Interpretation of results

**APA Style Reporting:**
- t-test: $t(df) = t\text{-value}, p = p\text{-value}$
- Effect size: $d = \text{value}$
- Confidence interval: 95% CI [lower, upper]

```r
# Comprehensive reporting function
generate_test_report <- function(test_result, data, hypothesized_value, test_type = "t-test") {
  cat("=== COMPREHENSIVE ONE-SAMPLE TEST REPORT ===\n\n")
  
  # Basic information
  cat("RESEARCH CONTEXT:\n")
  cat("Test type:", test_type, "\n")
  cat("Sample size:", length(data), "\n")
  cat("Hypothesized value:", hypothesized_value, "\n")
  cat("Alpha level: 0.05\n\n")
  
  # Descriptive statistics
  cat("DESCRIPTIVE STATISTICS:\n")
  cat("Sample mean:", round(mean(data), 3), "\n")
  cat("Sample SD:", round(sd(data), 3), "\n")
  cat("Sample median:", round(median(data), 3), "\n")
  cat("Range:", round(range(data), 3), "\n")
  cat("IQR:", round(IQR(data), 3), "\n\n")
  
  # Assumption checks
  cat("ASSUMPTION CHECKS:\n")
  shapiro_test <- shapiro.test(data)
  cat("Normality (Shapiro-Wilk): W =", round(shapiro_test$statistic, 3), 
      ", p =", round(shapiro_test$p.value, 4), "\n")
  
  outliers <- detect_outliers(data)
  cat("Outliers (IQR method):", length(outliers$outliers_iqr), "\n")
  cat("Outliers (Z-score method):", length(outliers$outliers_z), "\n\n")
  
  # Test results
  if (test_type == "t-test") {
    cat("T-TEST RESULTS:\n")
    cat("t-statistic:", round(test_result$statistic, 3), "\n")
    cat("Degrees of freedom:", test_result$parameter, "\n")
    cat("p-value:", round(test_result$p.value, 4), "\n")
    cat("95% Confidence Interval:", round(test_result$conf.int, 3), "\n")
    cat("Standard Error:", round(sd(data)/sqrt(length(data)), 3), "\n\n")
  } else if (test_type == "wilcoxon") {
    cat("WILCOXON SIGNED-RANK TEST RESULTS:\n")
    cat("V-statistic:", test_result$statistic, "\n")
    cat("p-value:", round(test_result$p.value, 4), "\n\n")
  }
  
  # Effect size
  effect_size <- calculate_cohens_d(data, hypothesized_value)
  cat("EFFECT SIZE:\n")
  cat("Cohen's d:", round(effect_size$cohens_d, 3), "\n")
  cat("Interpretation:", interpret_effect_size(effect_size$cohens_d), "\n")
  cat("95% CI for effect size:", round(c(effect_size$ci_lower, effect_size$ci_upper), 3), "\n\n")
  
  # Power analysis
  power_analysis <- pwr.t.test(n = length(data), d = effect_size$cohens_d, 
                              sig.level = 0.05, type = "one.sample")
  cat("POWER ANALYSIS:\n")
  cat("Observed power:", round(power_analysis$power, 3), "\n")
  if (power_analysis$power < 0.8) {
    recommended_n <- pwr.t.test(d = effect_size$cohens_d, sig.level = 0.05, 
                               power = 0.8, type = "one.sample")$n
    cat("Recommended sample size for 80% power:", ceiling(recommended_n), "\n")
  }
  cat("\n")
  
  # Statistical decision
  alpha <- 0.05
  cat("STATISTICAL DECISION:\n")
  if (test_result$p.value < alpha) {
    cat("✓ Reject the null hypothesis (p <", alpha, ")\n")
    cat("✓ There is significant evidence that the population mean differs from", hypothesized_value, "\n")
  } else {
    cat("✗ Fail to reject the null hypothesis (p >=", alpha, ")\n")
    cat("✗ There is insufficient evidence that the population mean differs from", hypothesized_value, "\n")
  }
  cat("\n")
  
  # Practical interpretation
  cat("PRACTICAL INTERPRETATION:\n")
  mean_diff <- mean(data) - hypothesized_value
  if (abs(effect_size$cohens_d) < 0.2) {
    cat("The effect is small and may not be practically meaningful.\n")
  } else if (abs(effect_size$cohens_d) < 0.5) {
    cat("The effect is moderate and may be practically meaningful.\n")
  } else {
    cat("The effect is large and likely practically meaningful.\n")
  }
  
  # APA style reporting
  cat("\nAPA STYLE REPORTING:\n")
  if (test_type == "t-test") {
    cat("A one-sample t-test was conducted to compare the sample mean (M =", 
        round(mean(data), 2), ", SD =", round(sd(data), 2), 
        ") to the hypothesized value of", hypothesized_value, ". ")
    
    if (test_result$p.value < alpha) {
      cat("The test was significant, t(", test_result$parameter, ") =", 
          round(test_result$statistic, 2), ", p =", round(test_result$p.value, 3), 
          ", d =", round(effect_size$cohens_d, 2), ". ")
      cat("The 95% confidence interval for the mean difference was [", 
          round(test_result$conf.int[1], 2), ",", round(test_result$conf.int[2], 2), "].\n")
    } else {
      cat("The test was not significant, t(", test_result$parameter, ") =", 
          round(test_result$statistic, 2), ", p =", round(test_result$p.value, 3), 
          ", d =", round(effect_size$cohens_d, 2), ".\n")
    }
  }
}

# Generate comprehensive report for MPG t-test
mpg_t_test <- t.test(mtcars$mpg, mu = 20)
generate_test_report(mpg_t_test, mtcars$mpg, 20, "t-test")
```

## Exercises

These exercises provide hands-on practice with one-sample tests, covering various scenarios and applications. Work through them systematically to build your understanding.

### Exercise 1: Basic One-Sample t-Test

**Scenario:** A car manufacturer claims their vehicles have an average weight of 3.0 thousand pounds. Test this claim using the mtcars dataset.

**Tasks:**
1. Perform a one-sample t-test to compare the mean weight to 3.0
2. Calculate and interpret the effect size
3. Report the results in APA style
4. Provide a practical interpretation

```r
# Your code here
# Hint: Use t.test() function with mu = 3.0
# Hint: Use the calculate_cohens_d() function for effect size
```

### Exercise 2: Effect Size Analysis

**Scenario:** Compare effect sizes across different variables in the mtcars dataset to understand which variables show the strongest deviations from hypothesized values.

**Tasks:**
1. Test MPG against hypothesized mean of 20
2. Test horsepower against hypothesized mean of 100
3. Test displacement against hypothesized mean of 200
4. Compare effect sizes and interpret practical significance
5. Create a summary table of results

```r
# Your code here
# Hint: Test multiple variables and store results in a data frame
# Hint: Use sapply() or a loop to test multiple variables efficiently
```

### Exercise 3: Nonparametric Alternatives

**Scenario:** Investigate whether nonparametric tests provide different conclusions than parametric tests for skewed data.

**Tasks:**
1. Generate skewed data using rgamma() or rchisq()
2. Perform both t-test and Wilcoxon signed-rank test
3. Compare p-values and conclusions
4. Assess normality of the generated data
5. Explain when each test is more appropriate

```r
# Your code here
# Hint: skewed_data <- rgamma(30, shape = 2, scale = 2)
# Hint: Use shapiro.test() to check normality
# Hint: Compare t.test() and wilcox.test() results
```

### Exercise 4: Power Analysis

**Scenario:** Design a study to detect a medium effect size (d = 0.5) with adequate power.

**Tasks:**
1. Calculate required sample size for 80% power
2. Calculate required sample size for 90% power
3. Create a power curve for different effect sizes
4. Simulate data and verify power calculations
5. Discuss implications for study design

```r
# Your code here
# Hint: Use pwr.t.test() with different power levels
# Hint: Use seq() to create a range of effect sizes
# Hint: Use replicate() to simulate multiple studies
```

### Exercise 5: Assumption Checking

**Scenario:** Perform comprehensive assumption checking for a dataset and recommend appropriate statistical methods.

**Tasks:**
1. Load a dataset (e.g., iris$Sepal.Length)
2. Check normality using multiple methods
3. Detect outliers using different criteria
4. Assess sample size adequacy
5. Recommend appropriate test based on findings
6. Perform the recommended test and interpret results

```r
# Your code here
# Hint: Use the check_normality() and detect_outliers() functions
# Hint: Use the choose_one_sample_test() function for recommendations
```

### Exercise 6: Bootstrap Confidence Intervals

**Scenario:** Compare bootstrap confidence intervals with parametric confidence intervals for non-normal data.

**Tasks:**
1. Generate non-normal data (e.g., exponential distribution)
2. Calculate parametric confidence interval using t.test()
3. Calculate bootstrap confidence interval using boot()
4. Compare the two methods
5. Discuss advantages and limitations of each approach

```r
# Your code here
# Hint: non_normal_data <- rexp(50, rate = 0.5)
# Hint: Use boot() function with appropriate statistic function
# Hint: Use boot.ci() to get different types of bootstrap CIs
```

### Exercise 7: Robust Methods

**Scenario:** Compare robust methods with standard methods when outliers are present.

**Tasks:**
1. Create a dataset with outliers
2. Perform standard t-test
3. Perform robust t-test (trimmed mean)
4. Perform Wilcoxon signed-rank test
5. Compare results and discuss implications
6. Recommend the best approach for this data

```r
# Your code here
# Hint: Add outliers to existing data: data_with_outliers <- c(data, 100, -50)
# Hint: Use the robust_t_test() function
# Hint: Compare effect sizes across methods
```

### Exercise 8: Real-World Application

**Scenario:** Analyze a real-world dataset and provide comprehensive statistical reporting.

**Tasks:**
1. Choose a variable from a dataset (e.g., mtcars$mpg, iris$Sepal.Length)
2. Formulate a research question
3. Perform comprehensive analysis including:
   - Descriptive statistics
   - Assumption checking
   - Appropriate statistical test
   - Effect size calculation
   - Power analysis
   - APA style reporting
4. Provide practical interpretation and recommendations

```r
# Your code here
# Hint: Use the generate_test_report() function
# Hint: Consider the context and implications of your findings
# Hint: Provide actionable recommendations based on results
```

### Exercise Solutions and Hints

**Exercise 1 Solution:**
```r
# Test weight against 3.0
weight_test <- t.test(mtcars$wt, mu = 3.0)
print(weight_test)

# Effect size
weight_effect <- calculate_cohens_d(mtcars$wt, 3.0)
cat("Effect size:", round(weight_effect$cohens_d, 3), "\n")
```

**Exercise 2 Solution:**
```r
# Test multiple variables
variables <- c("mpg", "hp", "disp")
hypothesized_values <- c(20, 100, 200)

results <- data.frame(
  Variable = variables,
  Sample_Mean = sapply(mtcars[variables], mean),
  Hypothesized = hypothesized_values,
  t_statistic = NA,
  p_value = NA,
  effect_size = NA
)

for (i in 1:length(variables)) {
  test_result <- t.test(mtcars[[variables[i]]], mu = hypothesized_values[i])
  effect_size <- calculate_cohens_d(mtcars[[variables[i]]], hypothesized_values[i])
  
  results$t_statistic[i] <- test_result$statistic
  results$p_value[i] <- test_result$p.value
  results$effect_size[i] <- effect_size$cohens_d
}

print(results)
```

**Exercise 3 Solution:**
```r
# Generate skewed data
set.seed(123)
skewed_data <- rgamma(30, shape = 2, scale = 2)

# Compare tests
t_result <- t.test(skewed_data, mu = 4)
wilcox_result <- wilcox.test(skewed_data, mu = 4)

cat("t-test p-value:", round(t_result$p.value, 4), "\n")
cat("Wilcoxon p-value:", round(wilcox_result$p.value, 4), "\n")
```

**Exercise 4 Solution:**
```r
# Power analysis
library(pwr)
sample_size_80 <- pwr.t.test(d = 0.5, sig.level = 0.05, power = 0.8, type = "one.sample")
sample_size_90 <- pwr.t.test(d = 0.5, sig.level = 0.05, power = 0.9, type = "one.sample")

cat("Required n for 80% power:", ceiling(sample_size_80$n), "\n")
cat("Required n for 90% power:", ceiling(sample_size_90$n), "\n")
```

**Exercise 5 Solution:**
```r
# Comprehensive assumption checking
data <- iris$Sepal.Length
choose_one_sample_test(data, 5.5)
```

These exercises provide comprehensive practice with one-sample tests and help develop critical thinking about statistical methodology.

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