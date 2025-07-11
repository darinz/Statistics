# One-Sample Tests

## Overview

One-sample tests are used to determine whether a sample mean differs significantly from a hypothesized population mean. These tests are fundamental to statistical inference and are widely used in research and data analysis.

## One-Sample t-Test

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
```

### One-Sample t-Test with Different Alternatives

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
```

### Effect Size for One-Sample t-Test

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
  
  return(list(
    cohens_d = cohens_d,
    hedges_g = hedges_g,
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
```

## One-Sample z-Test

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
  
  return(list(
    z_statistic = z_statistic,
    p_value_two_tailed = p_value_two_tailed,
    p_value_greater = p_value_greater,
    p_value_less = p_value_less,
    sample_mean = sample_mean,
    hypothesized_mean = hypothesized_mean,
    confidence_interval = c(ci_lower, ci_upper),
    margin_of_error = margin_of_error
  ))
}

# Example: Test if MPG mean is different from 20 (assuming known population SD = 6)
mpg_z_test <- one_sample_z_test(mtcars$mpg, 20, 6)

cat("One-Sample Z-Test Results:\n")
cat("Sample mean:", round(mpg_z_test$sample_mean, 3), "\n")
cat("Hypothesized mean:", mpg_z_test$hypothesized_mean, "\n")
cat("z-statistic:", round(mpg_z_test$z_statistic, 3), "\n")
cat("Two-tailed p-value:", round(mpg_z_test$p_value_two_tailed, 4), "\n")
cat("95% Confidence Interval:", round(mpg_z_test$confidence_interval, 3), "\n")
```

## Nonparametric One-Sample Tests

### Wilcoxon Signed-Rank Test

```r
# Wilcoxon signed-rank test (nonparametric alternative to t-test)
wilcox_test <- wilcox.test(mtcars$mpg, mu = 20, alternative = "two.sided")
print(wilcox_test)

# Compare with t-test results
cat("Comparison of t-test and Wilcoxon test:\n")
cat("t-test p-value:", round(t.test(mtcars$mpg, mu = 20)$p.value, 4), "\n")
cat("Wilcoxon p-value:", round(wilcox_test$p.value, 4), "\n")

# Effect size for Wilcoxon test
wilcox_effect_size <- abs(qnorm(wilcox_test$p.value / 2)) / sqrt(length(mtcars$mpg))
cat("Wilcoxon effect size (r):", round(wilcox_effect_size, 3), "\n")
```

### Sign Test

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
  
  return(list(
    positive_signs = positive_signs,
    negative_signs = negative_signs,
    n = n,
    p_value_two_tailed = p_value_two_tailed,
    p_value_greater = p_value_greater,
    p_value_less = p_value_less
  ))
}

# Apply sign test to MPG data
mpg_sign_test <- sign_test(mtcars$mpg, 20)

cat("Sign Test Results:\n")
cat("Positive signs:", mpg_sign_test$positive_signs, "\n")
cat("Negative signs:", mpg_sign_test$negative_signs, "\n")
cat("Two-tailed p-value:", round(mpg_sign_test$p_value_two_tailed, 4), "\n")
```

## Power Analysis

### Power Analysis for One-Sample t-Test

```r
library(pwr)

# Power analysis for one-sample t-test
power_analysis <- function(sample_size, effect_size, alpha = 0.05) {
  # Calculate power
  power_result <- pwr.t.test(n = sample_size, d = effect_size, sig.level = alpha, type = "one.sample")
  
  # Calculate required sample size for 80% power
  sample_size_80 <- pwr.t.test(d = effect_size, sig.level = alpha, power = 0.8, type = "one.sample")
  
  return(list(
    power = power_result$power,
    required_n_80 = sample_size_80$n,
    effect_size = effect_size,
    alpha = alpha
  ))
}

# Apply to MPG data
mpg_power <- power_analysis(length(mtcars$mpg), mpg_effect_size$cohens_d)

cat("Power Analysis Results:\n")
cat("Current power:", round(mpg_power$power, 3), "\n")
cat("Required sample size for 80% power:", ceiling(mpg_power$required_n_80), "\n")
```

## Assumption Checking

### Normality Test

```r
# Function to check normality assumption
check_normality <- function(data) {
  # Shapiro-Wilk test
  shapiro_test <- shapiro.test(data)
  
  # Q-Q plot
  qq_plot <- ggplot(data.frame(x = data), aes(sample = x)) +
    stat_qq() +
    stat_qq_line() +
    labs(title = "Q-Q Plot for Normality Check") +
    theme_minimal()
  
  # Histogram with normal curve
  hist_plot <- ggplot(data.frame(x = data), aes(x = x)) +
    geom_histogram(aes(y = ..density..), bins = 15, fill = "steelblue", alpha = 0.7) +
    stat_function(fun = dnorm, args = list(mean = mean(data), sd = sd(data)), 
                 color = "red", size = 1) +
    labs(title = "Histogram with Normal Curve") +
    theme_minimal()
  
  cat("Normality Test Results:\n")
  cat("Shapiro-Wilk p-value:", round(shapiro_test$p.value, 4), "\n")
  cat("Conclusion:", ifelse(shapiro_test$p.value < 0.05, 
                           "Data is not normally distributed", 
                           "Data appears to be normally distributed"), "\n")
  
  return(list(
    shapiro_test = shapiro_test,
    qq_plot = qq_plot,
    hist_plot = hist_plot
  ))
}

# Check normality of MPG data
mpg_normality <- check_normality(mtcars$mpg)
```

### Outlier Detection

```r
# Function to detect outliers
detect_outliers <- function(data) {
  # IQR method
  q1 <- quantile(data, 0.25, na.rm = TRUE)
  q3 <- quantile(data, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  
  outliers_iqr <- data < lower_bound | data > upper_bound
  
  # Z-score method
  z_scores <- abs((data - mean(data, na.rm = TRUE)) / sd(data, na.rm = TRUE))
  outliers_z <- z_scores > 3
  
  # Modified Z-score method
  median_val <- median(data, na.rm = TRUE)
  mad_val <- mad(data, na.rm = TRUE)
  modified_z_scores <- abs((data - median_val) / mad_val)
  outliers_modified_z <- modified_z_scores > 3.5
  
  return(list(
    outliers_iqr = which(outliers_iqr),
    outliers_z = which(outliers_z),
    outliers_modified_z = which(outliers_modified_z),
    bounds_iqr = c(lower_bound, upper_bound)
  ))
}

# Detect outliers in MPG data
mpg_outliers <- detect_outliers(mtcars$mpg)

cat("Outlier Detection Results:\n")
cat("IQR method outliers:", length(mpg_outliers$outliers_iqr), "\n")
cat("Z-score method outliers:", length(mpg_outliers$outliers_z), "\n")
cat("Modified Z-score outliers:", length(mpg_outliers$outliers_modified_z), "\n")
```

## Practical Examples

### Example 1: Quality Control

```r
# Simulate quality control data
set.seed(123)
n_products <- 50
product_weights <- rnorm(n_products, mean = 100, sd = 5)

# Test if mean weight is 100 grams
weight_test <- t.test(product_weights, mu = 100)
print(weight_test)

# Calculate effect size
weight_effect <- calculate_cohens_d(product_weights, 100)

cat("Quality Control Results:\n")
cat("Sample mean:", round(mean(product_weights), 2), "grams\n")
cat("Target mean: 100 grams\n")
cat("Effect size:", round(weight_effect$cohens_d, 3), "\n")
cat("Interpretation:", interpret_effect_size(weight_effect$cohens_d), "\n")
```

### Example 2: Educational Assessment

```r
# Simulate test scores
set.seed(123)
n_students <- 30
test_scores <- rnorm(n_students, mean = 75, sd = 10)

# Test if mean score is above 70 (passing threshold)
passing_test <- t.test(test_scores, mu = 70, alternative = "greater")
print(passing_test)

# Calculate confidence interval
ci_result <- t.test(test_scores, mu = 70, conf.level = 0.95)
cat("95% Confidence Interval:", round(ci_result$conf.int, 2), "\n")

# Effect size
score_effect <- calculate_cohens_d(test_scores, 70)
cat("Effect size:", round(score_effect$cohens_d, 3), "\n")
```

### Example 3: Medical Research

```r
# Simulate blood pressure data
set.seed(123)
n_patients <- 25
systolic_bp <- rnorm(n_patients, mean = 130, sd = 15)

# Test if mean systolic BP is different from 120 (normal)
bp_test <- t.test(systolic_bp, mu = 120)
print(bp_test)

# Nonparametric alternative
bp_wilcox <- wilcox.test(systolic_bp, mu = 120)
print(bp_wilcox)

# Compare parametric and nonparametric results
cat("Comparison:\n")
cat("t-test p-value:", round(bp_test$p.value, 4), "\n")
cat("Wilcoxon p-value:", round(bp_wilcox$p.value, 4), "\n")
```

## Advanced Topics

### Bootstrap Confidence Intervals

```r
library(boot)

# Bootstrap function for mean
boot_mean <- function(data, indices) {
  d <- data[indices]
  return(mean(d))
}

# Bootstrap confidence interval for MPG mean
boot_results <- boot(mtcars$mpg, boot_mean, R = 1000)
boot_ci <- boot.ci(boot_results, type = "perc")

cat("Bootstrap Results:\n")
cat("Bootstrap mean:", round(boot_results$t0, 3), "\n")
cat("Bootstrap 95% CI:", round(boot_ci$percent[4:5], 3), "\n")

# Compare with t-test CI
t_ci <- t.test(mtcars$mpg)$conf.int
cat("t-test 95% CI:", round(t_ci, 3), "\n")
```

### Robust One-Sample Tests

```r
# Trimmed mean t-test
trimmed_t_test <- function(data, mu, trim = 0.1) {
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
  
  return(list(
    trimmed_mean = trimmed_mean,
    t_statistic = t_statistic,
    p_value = p_value,
    df = df
  ))
}

# Apply trimmed t-test to MPG data
mpg_trimmed <- trimmed_t_test(mtcars$mpg, 20, trim = 0.1)
cat("Trimmed t-test results:\n")
cat("Trimmed mean:", round(mpg_trimmed$trimmed_mean, 3), "\n")
cat("t-statistic:", round(mpg_trimmed$t_statistic, 3), "\n")
cat("p-value:", round(mpg_trimmed$p_value, 4), "\n")
```

## Best Practices

### Test Selection Guidelines

```r
# Function to help choose appropriate one-sample test
choose_one_sample_test <- function(data, hypothesized_value) {
  cat("=== ONE-SAMPLE TEST SELECTION ===\n")
  
  # Check sample size
  n <- length(data)
  cat("Sample size:", n, "\n")
  
  # Check normality
  shapiro_test <- shapiro.test(data)
  cat("Normality test p-value:", round(shapiro_test$p.value, 4), "\n")
  
  # Check for outliers
  outliers <- detect_outliers(data)
  n_outliers <- length(outliers$outliers_iqr)
  cat("Number of outliers (IQR method):", n_outliers, "\n")
  
  # Recommendations
  cat("\nRECOMMENDATIONS:\n")
  
  if (n >= 30) {
    cat("- Large sample size: Central Limit Theorem applies\n")
    cat("- t-test is appropriate regardless of normality\n")
  } else {
    if (shapiro_test$p.value >= 0.05) {
      cat("- Data appears normally distributed\n")
      cat("- t-test is appropriate\n")
    } else {
      cat("- Data is not normally distributed\n")
      cat("- Consider Wilcoxon signed-rank test\n")
    }
  }
  
  if (n_outliers > 0) {
    cat("- Outliers detected: Consider robust methods\n")
    cat("- Trimmed t-test or nonparametric test recommended\n")
  }
  
  # Effect size calculation
  effect_size <- calculate_cohens_d(data, hypothesized_value)
  cat("- Effect size (Cohen's d):", round(effect_size$cohens_d, 3), "\n")
  cat("- Interpretation:", interpret_effect_size(effect_size$cohens_d), "\n")
}

# Apply to MPG data
choose_one_sample_test(mtcars$mpg, 20)
```

### Reporting Guidelines

```r
# Function to generate comprehensive test report
generate_test_report <- function(test_result, data, hypothesized_value, test_type = "t-test") {
  cat("=== ONE-SAMPLE TEST REPORT ===\n\n")
  
  cat("TEST INFORMATION:\n")
  cat("Test type:", test_type, "\n")
  cat("Sample size:", length(data), "\n")
  cat("Hypothesized value:", hypothesized_value, "\n\n")
  
  cat("DESCRIPTIVE STATISTICS:\n")
  cat("Sample mean:", round(mean(data), 3), "\n")
  cat("Sample SD:", round(sd(data), 3), "\n")
  cat("Sample median:", round(median(data), 3), "\n\n")
  
  if (test_type == "t-test") {
    cat("T-TEST RESULTS:\n")
    cat("t-statistic:", round(test_result$statistic, 3), "\n")
    cat("Degrees of freedom:", test_result$parameter, "\n")
    cat("p-value:", round(test_result$p.value, 4), "\n")
    cat("95% Confidence Interval:", round(test_result$conf.int, 3), "\n\n")
  } else if (test_type == "wilcoxon") {
    cat("WILCOXON SIGNED-RANK TEST RESULTS:\n")
    cat("V-statistic:", test_result$statistic, "\n")
    cat("p-value:", round(test_result$p.value, 4), "\n\n")
  }
  
  # Effect size
  effect_size <- calculate_cohens_d(data, hypothesized_value)
  cat("EFFECT SIZE:\n")
  cat("Cohen's d:", round(effect_size$cohens_d, 3), "\n")
  cat("Interpretation:", interpret_effect_size(effect_size$cohens_d), "\n\n")
  
  # Conclusion
  alpha <- 0.05
  if (test_result$p.value < alpha) {
    cat("CONCLUSION:\n")
    cat("Reject the null hypothesis (p <", alpha, ")\n")
    cat("There is significant evidence that the population mean differs from", hypothesized_value, "\n")
  } else {
    cat("CONCLUSION:\n")
    cat("Fail to reject the null hypothesis (p >=", alpha, ")\n")
    cat("There is insufficient evidence that the population mean differs from", hypothesized_value, "\n")
  }
}

# Generate report for MPG t-test
mpg_t_test <- t.test(mtcars$mpg, mu = 20)
generate_test_report(mpg_t_test, mtcars$mpg, 20, "t-test")
```

## Exercises

### Exercise 1: Basic One-Sample t-Test
Test whether the mean weight of cars in the mtcars dataset is significantly different from 3.0 (thousand pounds).

### Exercise 2: Effect Size Analysis
Calculate and interpret effect sizes for one-sample tests on different variables in the mtcars dataset.

### Exercise 3: Nonparametric Alternatives
Compare t-test results with Wilcoxon signed-rank test results for skewed data.

### Exercise 4: Power Analysis
Conduct power analysis to determine required sample sizes for detecting different effect sizes.

### Exercise 5: Assumption Checking
Perform comprehensive assumption checking for one-sample tests and recommend appropriate alternatives.

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