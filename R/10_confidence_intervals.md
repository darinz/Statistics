# Confidence Intervals

## Overview

Confidence intervals provide a range of plausible values for population parameters based on sample data. They are fundamental to statistical inference and help us quantify the uncertainty in our estimates.

## Confidence Interval for the Mean

### Normal Distribution (Known Population Standard Deviation)

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

# Calculate z-score
z_score <- qnorm(1 - alpha/2)

# Calculate margin of error
margin_of_error <- z_score * (population_sd / sqrt(n))

# Calculate confidence interval
ci_lower <- sample_mean - margin_of_error
ci_upper <- sample_mean + margin_of_error

cat("Sample mean:", sample_mean, "\n")
cat("Margin of error:", margin_of_error, "\n")
cat("95% CI:", ci_lower, "to", ci_upper, "\n")
```

### t-Distribution (Unknown Population Standard Deviation)

```r
# Calculate confidence interval using t-distribution
confidence_level <- 0.95
alpha <- 1 - confidence_level
df <- n - 1  # degrees of freedom

# Calculate t-score
t_score <- qt(1 - alpha/2, df = df)

# Calculate margin of error
margin_of_error <- t_score * (sample_sd / sqrt(n))

# Calculate confidence interval
ci_lower <- sample_mean - margin_of_error
ci_upper <- sample_mean + margin_of_error

cat("Sample mean:", sample_mean, "\n")
cat("Sample SD:", sample_sd, "\n")
cat("Degrees of freedom:", df, "\n")
cat("t-score:", t_score, "\n")
cat("Margin of error:", margin_of_error, "\n")
cat("95% CI:", ci_lower, "to", ci_upper, "\n")
```

### Using Built-in Functions

```r
# Using t.test() for confidence interval
t_test_result <- t.test(mtcars$mpg, conf.level = 0.95)
print(t_test_result)

# Extract confidence interval
ci_from_test <- t_test_result$conf.int
cat("95% CI from t.test():", ci_from_test[1], "to", ci_from_test[2], "\n")
```

## Confidence Interval for the Proportion

### Large Sample Approximation

```r
# Calculate confidence interval for proportion
# Example: proportion of manual transmissions
manual_count <- sum(mtcars$am)
total_count <- length(mtcars$am)
sample_proportion <- manual_count / total_count

# Calculate standard error
standard_error <- sqrt(sample_proportion * (1 - sample_proportion) / total_count)

# Calculate margin of error
z_score <- qnorm(0.975)  # 95% confidence level
margin_of_error <- z_score * standard_error

# Calculate confidence interval
ci_lower <- sample_proportion - margin_of_error
ci_upper <- sample_proportion + margin_of_error

cat("Sample proportion:", sample_proportion, "\n")
cat("Standard error:", standard_error, "\n")
cat("Margin of error:", margin_of_error, "\n")
cat("95% CI:", ci_lower, "to", ci_upper, "\n")
```

### Exact Binomial Confidence Interval

```r
# Using binom.test() for exact confidence interval
binom_result <- binom.test(manual_count, total_count, conf.level = 0.95)
print(binom_result)

# Extract confidence interval
ci_from_binom <- binom_result$conf.int
cat("Exact 95% CI:", ci_from_binom[1], "to", ci_from_binom[2], "\n")
```

## Confidence Interval for the Difference Between Two Means

### Independent Samples

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

# Calculate pooled standard deviation
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

cat("Auto mean:", mean1, "\n")
cat("Manual mean:", mean2, "\n")
cat("Difference:", difference, "\n")
cat("Pooled SD:", pooled_sd, "\n")
cat("Standard error:", standard_error, "\n")
cat("95% CI for difference:", ci_lower, "to", ci_upper, "\n")

# Using built-in function
t_test_diff <- t.test(manual_mpg, auto_mpg, conf.level = 0.95)
print(t_test_diff)
```

### Paired Samples

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

cat("Mean difference:", mean_diff, "\n")
cat("SD of differences:", sd_diff, "\n")
cat("Standard error:", standard_error, "\n")
cat("95% CI for mean difference:", ci_lower, "to", ci_upper, "\n")

# Using built-in function
paired_test <- t.test(after, before, paired = TRUE, conf.level = 0.95)
print(paired_test)
```

## Confidence Interval for the Variance

### Chi-Square Based Confidence Interval

```r
# Calculate confidence interval for variance
sample_variance <- var(mtcars$mpg)
n <- length(mtcars$mpg)
df <- n - 1

# Calculate chi-square critical values
chi_lower <- qchisq(0.025, df = df)
chi_upper <- qchisq(0.975, df = df)

# Calculate confidence interval
ci_lower_var <- (df * sample_variance) / chi_upper
ci_upper_var <- (df * sample_variance) / chi_lower

cat("Sample variance:", sample_variance, "\n")
cat("Degrees of freedom:", df, "\n")
cat("95% CI for variance:", ci_lower_var, "to", ci_upper_var, "\n")

# Confidence interval for standard deviation
ci_lower_sd <- sqrt(ci_lower_var)
ci_upper_sd <- sqrt(ci_upper_var)
cat("95% CI for standard deviation:", ci_lower_sd, "to", ci_upper_sd, "\n")
```

## Bootstrap Confidence Intervals

### Bootstrap for Mean

```r
library(boot)

# Bootstrap function for mean
boot_mean <- function(data, indices) {
  return(mean(data[indices]))
}

# Perform bootstrap
set.seed(123)
boot_results <- boot(mtcars$mpg, boot_mean, R = 1000)

# Different types of bootstrap confidence intervals
boot_normal <- boot.ci(boot_results, type = "norm")
boot_percentile <- boot.ci(boot_results, type = "perc")
boot_basic <- boot.ci(boot_results, type = "basic")

cat("Normal bootstrap CI:", boot_normal$normal[2:3], "\n")
cat("Percentile bootstrap CI:", boot_percentile$percent[4:5], "\n")
cat("Basic bootstrap CI:", boot_basic$basic[4:5], "\n")

# Plot bootstrap distribution
hist(boot_results$t, main = "Bootstrap Distribution of Mean",
     xlab = "Bootstrap Mean", col = "lightblue", freq = FALSE)

# Add confidence interval lines
abline(v = boot_percentile$percent[4:5], col = "red", lty = 2, lwd = 2)
abline(v = mean(mtcars$mpg), col = "green", lwd = 2)
```

### Bootstrap for Median

```r
# Bootstrap function for median
boot_median <- function(data, indices) {
  return(median(data[indices]))
}

# Perform bootstrap
boot_median_results <- boot(mtcars$mpg, boot_median, R = 1000)
boot_median_ci <- boot.ci(boot_median_results, type = "perc")

cat("Bootstrap CI for median:", boot_median_ci$percent[4:5], "\n")

# Plot bootstrap distribution
hist(boot_median_results$t, main = "Bootstrap Distribution of Median",
     xlab = "Bootstrap Median", col = "lightgreen", freq = FALSE)
abline(v = boot_median_ci$percent[4:5], col = "red", lty = 2, lwd = 2)
abline(v = median(mtcars$mpg), col = "green", lwd = 2)
```

## Effect of Sample Size and Confidence Level

### Sample Size Effect

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
sample_sizes <- c(5, 10, 20, 30, 50)
ci_widths <- sapply(sample_sizes, ci_width)

# Plot results
plot(sample_sizes, ci_widths, type = "b", 
     main = "Effect of Sample Size on CI Width",
     xlab = "Sample Size", ylab = "CI Width",
     col = "blue", lwd = 2)
```

### Confidence Level Effect

```r
# Function to calculate confidence interval for different levels
ci_by_level <- function(confidence_levels) {
  widths <- numeric(length(confidence_levels))
  for (i in 1:length(confidence_levels)) {
    t_test_result <- t.test(mtcars$mpg, conf.level = confidence_levels[i])
    ci <- t_test_result$conf.int
    widths[i] <- ci[2] - ci[1]
  }
  return(widths)
}

# Test different confidence levels
confidence_levels <- c(0.80, 0.85, 0.90, 0.95, 0.99)
ci_widths_by_level <- ci_by_level(confidence_levels)

# Plot results
plot(confidence_levels, ci_widths_by_level, type = "b",
     main = "Effect of Confidence Level on CI Width",
     xlab = "Confidence Level", ylab = "CI Width",
     col = "red", lwd = 2)
```

## Multiple Confidence Intervals

### Simultaneous Confidence Intervals

```r
# Bonferroni correction for multiple comparisons
# Example: confidence intervals for MPG by cylinders
cylinders <- unique(mtcars$cyl)
n_comparisons <- length(cylinders)
alpha_family <- 0.05
alpha_individual <- alpha_family / n_comparisons

# Calculate confidence intervals for each cylinder type
ci_results <- list()
for (cyl in cylinders) {
  cyl_data <- mtcars$mpg[mtcars$cyl == cyl]
  t_test_result <- t.test(cyl_data, conf.level = 1 - alpha_individual)
  ci_results[[as.character(cyl)]] <- t_test_result$conf.int
}

# Display results
cat("Bonferroni-corrected confidence intervals (α =", alpha_family, "):\n")
for (cyl in cylinders) {
  ci <- ci_results[[as.character(cyl)]]
  cat("Cylinders", cyl, ":", ci[1], "to", ci[2], "\n")
}
```

## Practical Examples

### Example 1: Quality Control

```r
# Quality control example
set.seed(123)
production_batch <- rnorm(100, mean = 100, sd = 5)

# Calculate confidence interval for mean weight
t_test_qc <- t.test(production_batch, conf.level = 0.95)
ci_qc <- t_test_qc$conf.int

cat("Production batch mean:", mean(production_batch), "\n")
cat("95% CI for mean weight:", ci_qc[1], "to", ci_qc[2], "\n")

# Check if target weight (100) is in confidence interval
target_weight <- 100
if (target_weight >= ci_qc[1] && target_weight <= ci_qc[2]) {
  cat("Target weight is within the confidence interval.\n")
} else {
  cat("Target weight is outside the confidence interval.\n")
}
```

### Example 2: Survey Results

```r
# Survey example
set.seed(123)
survey_responses <- sample(c(0, 1), size = 100, replace = TRUE, prob = c(0.7, 0.3))
# 1 = satisfied, 0 = not satisfied

# Calculate confidence interval for satisfaction proportion
satisfied_count <- sum(survey_responses)
total_responses <- length(survey_responses)
satisfaction_proportion <- satisfied_count / total_responses

# Calculate confidence interval
binom_result <- binom.test(satisfied_count, total_responses, conf.level = 0.95)
ci_survey <- binom_result$conf.int

cat("Satisfaction proportion:", satisfaction_proportion, "\n")
cat("95% CI for satisfaction proportion:", ci_survey[1], "to", ci_survey[2], "\n")
```

### Example 3: Treatment Effect

```r
# Treatment effect example
set.seed(123)
control_group <- rnorm(30, mean = 50, sd = 10)
treatment_group <- rnorm(30, mean = 55, sd = 10)

# Calculate confidence interval for treatment effect
t_test_treatment <- t.test(treatment_group, control_group, conf.level = 0.95)
ci_treatment <- t_test_treatment$conf.int

cat("Control group mean:", mean(control_group), "\n")
cat("Treatment group mean:", mean(treatment_group), "\n")
cat("Treatment effect:", mean(treatment_group) - mean(control_group), "\n")
cat("95% CI for treatment effect:", ci_treatment[1], "to", ci_treatment[2], "\n")

# Check if treatment effect is significant (CI doesn't include 0)
if (ci_treatment[1] > 0 || ci_treatment[2] < 0) {
  cat("Treatment effect is significant (CI doesn't include 0).\n")
} else {
  cat("Treatment effect is not significant (CI includes 0).\n")
}
```

## Best Practices

### Interpretation Guidelines

```r
# Function to interpret confidence intervals
interpret_ci <- function(ci, parameter_name = "parameter", confidence_level = 0.95) {
  cat("INTERPRETATION OF", confidence_level * 100, "% CONFIDENCE INTERVAL:\n")
  cat("We are", confidence_level * 100, "% confident that the true", parameter_name, "\n")
  cat("lies between", ci[1], "and", ci[2], ".\n\n")
  
  cat("This means that if we were to repeat this study many times,\n")
  cat("approximately", confidence_level * 100, "% of the confidence intervals\n")
  cat("would contain the true", parameter_name, ".\n\n")
  
  cat("IMPORTANT NOTES:\n")
  cat("- The confidence interval is about the method, not the parameter\n")
  cat("- The parameter is fixed; the interval varies across samples\n")
  cat("- A wider interval indicates more uncertainty\n")
  cat("- A narrower interval indicates more precision\n")
}

# Example interpretation
example_ci <- t.test(mtcars$mpg, conf.level = 0.95)$conf.int
interpret_ci(example_ci, "mean MPG", 0.95)
```

### Common Mistakes to Avoid

```r
# Function to demonstrate common mistakes
demonstrate_mistakes <- function() {
  cat("COMMON MISTAKES IN CONFIDENCE INTERVALS:\n\n")
  
  cat("1. Saying '95% probability that the parameter is in the interval'\n")
  cat("   - The parameter is fixed, not random\n")
  cat("   - The interval is random, not the parameter\n\n")
  
  cat("2. Comparing confidence intervals for significance\n")
  cat("   - Overlapping CIs don't necessarily mean no significant difference\n")
  cat("   - Non-overlapping CIs don't necessarily mean significant difference\n\n")
  
  cat("3. Using confidence intervals for individual predictions\n")
  cat("   - CIs are for population parameters, not individual values\n")
  cat("   - Use prediction intervals for individual predictions\n\n")
  
  cat("4. Ignoring multiple comparisons\n")
  cat("   - Multiple CIs increase family-wise error rate\n")
  cat("   - Use corrections like Bonferroni for multiple comparisons\n\n")
}

demonstrate_mistakes()
```

## Exercises

### Exercise 1: Basic Confidence Intervals
Calculate 90%, 95%, and 99% confidence intervals for the mean MPG of cars with 6 cylinders.

### Exercise 2: Proportion Confidence Intervals
Calculate confidence intervals for the proportion of cars with manual transmission, using both normal approximation and exact binomial methods.

### Exercise 3: Difference Between Means
Calculate a confidence interval for the difference in MPG between cars with 4 and 8 cylinders.

### Exercise 4: Bootstrap Confidence Intervals
Use bootstrap sampling to calculate confidence intervals for the median and standard deviation of the iris sepal length data.

### Exercise 5: Sample Size Planning
Determine the sample size needed to estimate the mean MPG with a margin of error of ±1 MPG at 95% confidence.

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