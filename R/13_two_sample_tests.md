# Two-Sample Tests

## Overview

Two-sample tests are used to compare means or distributions between two groups. These tests are fundamental to experimental design and observational studies, allowing researchers to determine if differences between groups are statistically significant.

## Independent Samples t-Test

### Basic Independent Samples t-Test

```r
# Load sample data
data(mtcars)

# Compare MPG between automatic and manual transmission
automatic_mpg <- mtcars$mpg[mtcars$am == 0]
manual_mpg <- mtcars$mpg[mtcars$am == 1]

# Perform independent samples t-test
t_test_result <- t.test(automatic_mpg, manual_mpg, var.equal = FALSE)
print(t_test_result)

# Extract key statistics
t_statistic <- t_test_result$statistic
p_value <- t_test_result$p.value
confidence_interval <- t_test_result$conf.int
mean_diff <- t_test_result$estimate[1] - t_test_result$estimate[2]

cat("Test Results:\n")
cat("Mean difference (Manual - Automatic):", round(mean_diff, 3), "\n")
cat("t-statistic:", round(t_statistic, 3), "\n")
cat("p-value:", round(p_value, 4), "\n")
cat("95% Confidence Interval:", round(confidence_interval, 3), "\n")
```

### Equal vs Unequal Variances

```r
# Test for equal variances
var_test <- var.test(automatic_mpg, manual_mpg)
print(var_test)

# Perform t-test with equal variances (if appropriate)
if (var_test$p.value > 0.05) {
  t_test_equal_var <- t.test(automatic_mpg, manual_mpg, var.equal = TRUE)
  cat("Using equal variances t-test:\n")
  print(t_test_equal_var)
} else {
  cat("Using Welch's t-test (unequal variances):\n")
  print(t_test_result)
}

# Compare results
cat("Comparison of t-tests:\n")
cat("Equal variances p-value:", round(t.test(automatic_mpg, manual_mpg, var.equal = TRUE)$p.value, 4), "\n")
cat("Unequal variances p-value:", round(t_test_result$p.value, 4), "\n")
```

### Effect Size for Independent Samples

```r
# Calculate Cohen's d for independent samples
calculate_cohens_d_independent <- function(group1, group2) {
  n1 <- length(group1)
  n2 <- length(group2)
  
  # Pooled standard deviation
  pooled_sd <- sqrt(((n1 - 1) * var(group1) + (n2 - 1) * var(group2)) / (n1 + n2 - 2))
  
  # Cohen's d
  cohens_d <- (mean(group1) - mean(group2)) / pooled_sd
  
  # Hedges' g (unbiased estimator)
  hedges_g <- cohens_d * (1 - 3 / (4 * (n1 + n2) - 9))
  
  return(list(
    cohens_d = cohens_d,
    hedges_g = hedges_g,
    pooled_sd = pooled_sd,
    n1 = n1,
    n2 = n2
  ))
}

# Apply to transmission comparison
transmission_effect <- calculate_cohens_d_independent(automatic_mpg, manual_mpg)

cat("Effect Size Analysis:\n")
cat("Cohen's d:", round(transmission_effect$cohens_d, 3), "\n")
cat("Hedges' g:", round(transmission_effect$hedges_g, 3), "\n")

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

cat("Effect size interpretation:", interpret_effect_size(transmission_effect$cohens_d), "\n")
```

## Paired Samples t-Test

### Basic Paired Samples t-Test

```r
# Simulate paired data (before and after treatment)
set.seed(123)
n_subjects <- 20
before_scores <- rnorm(n_subjects, mean = 75, sd = 10)
after_scores <- before_scores + rnorm(n_subjects, mean = 5, sd = 8)

# Perform paired samples t-test
paired_test <- t.test(before_scores, after_scores, paired = TRUE)
print(paired_test)

# Calculate paired effect size
paired_effect <- calculate_cohens_d_independent(before_scores, after_scores)

cat("Paired Samples Results:\n")
cat("Mean difference (After - Before):", round(mean(after_scores - before_scores), 3), "\n")
cat("t-statistic:", round(paired_test$statistic, 3), "\n")
cat("p-value:", round(paired_test$p.value, 4), "\n")
cat("Effect size (Cohen's d):", round(paired_effect$cohens_d, 3), "\n")
```

### Paired Data Analysis

```r
# Create paired data frame
paired_data <- data.frame(
  subject = 1:n_subjects,
  before = before_scores,
  after = after_scores,
  difference = after_scores - before_scores
)

# Summary statistics for paired data
cat("Paired Data Summary:\n")
cat("Mean difference:", round(mean(paired_data$difference), 3), "\n")
cat("SD of differences:", round(sd(paired_data$difference), 3), "\n")
cat("Correlation between before and after:", round(cor(paired_data$before, paired_data$after), 3), "\n")

# Visualize paired data
library(ggplot2)
library(gridExtra)

# Before vs After plot
p1 <- ggplot(paired_data, aes(x = before, y = after)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Before vs After Scores") +
  theme_minimal()

# Difference plot
p2 <- ggplot(paired_data, aes(x = difference)) +
  geom_histogram(bins = 10, fill = "steelblue", alpha = 0.7) +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Distribution of Differences") +
  theme_minimal()

# Combine plots
grid.arrange(p1, p2, ncol = 2)
```

## Nonparametric Two-Sample Tests

### Mann-Whitney U Test (Wilcoxon Rank-Sum)

```r
# Mann-Whitney U test for independent samples
wilcox_test <- wilcox.test(automatic_mpg, manual_mpg)
print(wilcox_test)

# Compare with t-test results
cat("Comparison of parametric and nonparametric tests:\n")
cat("t-test p-value:", round(t_test_result$p.value, 4), "\n")
cat("Wilcoxon p-value:", round(wilcox_test$p.value, 4), "\n")

# Effect size for Wilcoxon test
wilcox_effect_size <- abs(qnorm(wilcox_test$p.value / 2)) / sqrt(length(automatic_mpg) + length(manual_mpg))
cat("Wilcoxon effect size (r):", round(wilcox_effect_size, 3), "\n")
```

### Wilcoxon Signed-Rank Test for Paired Data

```r
# Wilcoxon signed-rank test for paired samples
paired_wilcox <- wilcox.test(before_scores, after_scores, paired = TRUE)
print(paired_wilcox)

# Compare with paired t-test
cat("Comparison of paired tests:\n")
cat("Paired t-test p-value:", round(paired_test$p.value, 4), "\n")
cat("Paired Wilcoxon p-value:", round(paired_wilcox$p.value, 4), "\n")
```

## Assumption Checking

### Normality Tests

```r
# Function to check normality for both groups
check_normality_groups <- function(group1, group2, group_names = c("Group 1", "Group 2")) {
  cat("=== NORMALITY TESTS ===\n")
  
  # Shapiro-Wilk tests
  shapiro1 <- shapiro.test(group1)
  shapiro2 <- shapiro.test(group2)
  
  cat(group_names[1], "Shapiro-Wilk p-value:", round(shapiro1$p.value, 4), "\n")
  cat(group_names[2], "Shapiro-Wilk p-value:", round(shapiro2$p.value, 4), "\n")
  
  # Q-Q plots
  par(mfrow = c(1, 2))
  qqnorm(group1, main = paste("Q-Q Plot:", group_names[1]))
  qqline(group1, col = "red")
  
  qqnorm(group2, main = paste("Q-Q Plot:", group_names[2]))
  qqline(group2, col = "red")
  par(mfrow = c(1, 1))
  
  # Recommendations
  cat("\nRECOMMENDATIONS:\n")
  if (shapiro1$p.value < 0.05 || shapiro2$p.value < 0.05) {
    cat("- At least one group is not normally distributed\n")
    cat("- Consider nonparametric alternatives\n")
  } else {
    cat("- Both groups appear normally distributed\n")
    cat("- Parametric tests are appropriate\n")
  }
  
  return(list(
    shapiro1 = shapiro1,
    shapiro2 = shapiro2
  ))
}

# Check normality for transmission groups
normality_results <- check_normality_groups(automatic_mpg, manual_mpg, 
                                          c("Automatic", "Manual"))
```

### Homogeneity of Variance

```r
# Function to test homogeneity of variance
check_homogeneity <- function(group1, group2) {
  cat("=== HOMOGENEITY OF VARIANCE TESTS ===\n")
  
  # F-test for equality of variances
  f_test <- var.test(group1, group2)
  cat("F-test p-value:", round(f_test$p.value, 4), "\n")
  
  # Levene's test (more robust)
  library(car)
  combined_data <- c(group1, group2)
  group_labels <- factor(c(rep("Group1", length(group1)), 
                        rep("Group2", length(group2))))
  levene_test <- leveneTest(combined_data, group_labels)
  cat("Levene's test p-value:", round(levene_test$`Pr(>F)`[1], 4), "\n")
  
  # Recommendations
  cat("\nRECOMMENDATIONS:\n")
  if (f_test$p.value < 0.05) {
    cat("- Variances are significantly different\n")
    cat("- Use Welch's t-test (unequal variances)\n")
  } else {
    cat("- Variances appear equal\n")
    cat("- Standard t-test is appropriate\n")
  }
  
  return(list(
    f_test = f_test,
    levene_test = levene_test
  ))
}

# Check homogeneity for transmission groups
homogeneity_results <- check_homogeneity(automatic_mpg, manual_mpg)
```

## Power Analysis

### Power Analysis for Two-Sample t-Test

```r
library(pwr)

# Power analysis for two-sample t-test
power_analysis_two_sample <- function(n1, n2, effect_size, alpha = 0.05) {
  # Calculate power
  power_result <- pwr.t2n.test(n1 = n1, n2 = n2, d = effect_size, sig.level = alpha)
  
  # Calculate required sample sizes for 80% power
  sample_size_result <- pwr.t.test(d = effect_size, sig.level = alpha, power = 0.8)
  
  return(list(
    power = power_result$power,
    required_n_per_group = ceiling(sample_size_result$n),
    effect_size = effect_size,
    alpha = alpha
  ))
}

# Apply to transmission comparison
transmission_power <- power_analysis_two_sample(
  length(automatic_mpg), 
  length(manual_mpg), 
  transmission_effect$cohens_d
)

cat("Power Analysis Results:\n")
cat("Current power:", round(transmission_power$power, 3), "\n")
cat("Required sample size per group for 80% power:", transmission_power$required_n_per_group, "\n")
```

## Practical Examples

### Example 1: Clinical Trial

```r
# Simulate clinical trial data
set.seed(123)
n_treatment <- 25
n_control <- 25

# Generate treatment and control group data
treatment_scores <- rnorm(n_treatment, mean = 85, sd = 12)
control_scores <- rnorm(n_control, mean = 75, sd = 10)

# Perform t-test
clinical_t_test <- t.test(treatment_scores, control_scores)
print(clinical_t_test)

# Calculate effect size
clinical_effect <- calculate_cohens_d_independent(treatment_scores, control_scores)

cat("Clinical Trial Results:\n")
cat("Treatment mean:", round(mean(treatment_scores), 2), "\n")
cat("Control mean:", round(mean(control_scores), 2), "\n")
cat("Mean difference:", round(mean(treatment_scores) - mean(control_scores), 2), "\n")
cat("Effect size:", round(clinical_effect$cohens_d, 3), "\n")
cat("Interpretation:", interpret_effect_size(clinical_effect$cohens_d), "\n")
```

### Example 2: Educational Research

```r
# Simulate educational intervention data
set.seed(123)
n_experimental <- 30
n_control <- 30

# Generate pre-test and post-test data
experimental_pre <- rnorm(n_experimental, mean = 70, sd = 15)
experimental_post <- experimental_pre + rnorm(n_experimental, mean = 8, sd = 6)

control_pre <- rnorm(n_control, mean = 72, sd = 14)
control_post <- control_pre + rnorm(n_control, mean = 2, sd = 5)

# Analyze gain scores
experimental_gains <- experimental_post - experimental_pre
control_gains <- control_post - control_pre

# Independent samples t-test on gains
gains_test <- t.test(experimental_gains, control_gains)
print(gains_test)

# Effect size for gains
gains_effect <- calculate_cohens_d_independent(experimental_gains, control_gains)

cat("Educational Intervention Results:\n")
cat("Experimental gain mean:", round(mean(experimental_gains), 2), "\n")
cat("Control gain mean:", round(mean(control_gains), 2), "\n")
cat("Gain difference:", round(mean(experimental_gains) - mean(control_gains), 2), "\n")
cat("Effect size:", round(gains_effect$cohens_d, 3), "\n")
```

### Example 3: Quality Control

```r
# Simulate quality control data
set.seed(123)
n_machine1 <- 20
n_machine2 <- 20

# Generate production data
machine1_output <- rnorm(n_machine1, mean = 100, sd = 5)
machine2_output <- rnorm(n_machine2, mean = 98, sd = 6)

# Perform quality control test
quality_test <- t.test(machine1_output, machine2_output)
print(quality_test)

# Nonparametric alternative
quality_wilcox <- wilcox.test(machine1_output, machine2_output)
print(quality_wilcox)

# Compare results
cat("Quality Control Comparison:\n")
cat("t-test p-value:", round(quality_test$p.value, 4), "\n")
cat("Wilcoxon p-value:", round(quality_wilcox$p.value, 4), "\n")
```

## Advanced Topics

### Bootstrap Confidence Intervals

```r
library(boot)

# Bootstrap function for mean difference
boot_mean_diff <- function(data, indices) {
  d <- data[indices, ]
  group1 <- d$value[d$group == "Group1"]
  group2 <- d$value[d$group == "Group2"]
  return(mean(group1) - mean(group2))
}

# Create data frame for bootstrap
boot_data <- data.frame(
  value = c(automatic_mpg, manual_mpg),
  group = c(rep("Group1", length(automatic_mpg)), 
           rep("Group2", length(manual_mpg)))
)

# Bootstrap confidence interval
boot_results <- boot(boot_data, boot_mean_diff, R = 1000)
boot_ci <- boot.ci(boot_results, type = "perc")

cat("Bootstrap Results:\n")
cat("Bootstrap mean difference:", round(boot_results$t0, 3), "\n")
cat("Bootstrap 95% CI:", round(boot_ci$percent[4:5], 3), "\n")

# Compare with t-test CI
t_ci <- t.test(automatic_mpg, manual_mpg)$conf.int
cat("t-test 95% CI:", round(t_ci, 3), "\n")
```

### Robust Two-Sample Tests

```r
# Yuen's t-test for trimmed means
yuen_test <- function(group1, group2, trim = 0.1) {
  # Trim the data
  n1 <- length(group1)
  n2 <- length(group2)
  k1 <- floor(n1 * trim)
  k2 <- floor(n2 * trim)
  
  # Sort and trim
  sorted1 <- sort(group1)
  sorted2 <- sort(group2)
  
  trimmed1 <- sorted1[(k1 + 1):(n1 - k1)]
  trimmed2 <- sorted2[(k2 + 1):(n2 - k2)]
  
  # Calculate trimmed statistics
  mean1 <- mean(trimmed1)
  mean2 <- mean(trimmed2)
  var1 <- var(trimmed1)
  var2 <- var(trimmed2)
  
  # Calculate test statistic
  se <- sqrt(var1 / (n1 - 2 * k1) + var2 / (n2 - 2 * k2))
  t_stat <- (mean1 - mean2) / se
  
  # Degrees of freedom
  df <- (var1 / (n1 - 2 * k1) + var2 / (n2 - 2 * k2))^2 /
        ((var1 / (n1 - 2 * k1))^2 / (n1 - 2 * k1 - 1) + 
         (var2 / (n2 - 2 * k2))^2 / (n2 - 2 * k2 - 1))
  
  p_value <- 2 * (1 - pt(abs(t_stat), df))
  
  return(list(
    t_statistic = t_stat,
    p_value = p_value,
    df = df,
    trimmed_means = c(mean1, mean2)
  ))
}

# Apply Yuen's test
yuen_result <- yuen_test(automatic_mpg, manual_mpg)
cat("Yuen's t-test results:\n")
cat("t-statistic:", round(yuen_result$t_statistic, 3), "\n")
cat("p-value:", round(yuen_result$p_value, 4), "\n")
```

## Best Practices

### Test Selection Guidelines

```r
# Function to help choose appropriate two-sample test
choose_two_sample_test <- function(group1, group2, paired = FALSE) {
  cat("=== TWO-SAMPLE TEST SELECTION ===\n")
  
  if (paired) {
    cat("PAIRED SAMPLES ANALYSIS:\n")
    
    # Check normality of differences
    differences <- group1 - group2
    shapiro_diff <- shapiro.test(differences)
    cat("Normality of differences p-value:", round(shapiro_diff$p.value, 4), "\n")
    
    if (shapiro_diff$p.value >= 0.05) {
      cat("RECOMMENDATION: Paired t-test\n")
    } else {
      cat("RECOMMENDATION: Wilcoxon signed-rank test\n")
    }
    
  } else {
    cat("INDEPENDENT SAMPLES ANALYSIS:\n")
    
    # Check normality
    shapiro1 <- shapiro.test(group1)
    shapiro2 <- shapiro.test(group2)
    cat("Group 1 normality p-value:", round(shapiro1$p.value, 4), "\n")
    cat("Group 2 normality p-value:", round(shapiro2$p.value, 4), "\n")
    
    # Check homogeneity of variance
    var_test <- var.test(group1, group2)
    cat("Homogeneity of variance p-value:", round(var_test$p.value, 4), "\n")
    
    # Recommendations
    if (shapiro1$p.value >= 0.05 && shapiro2$p.value >= 0.05) {
      if (var_test$p.value >= 0.05) {
        cat("RECOMMENDATION: Standard t-test (equal variances)\n")
      } else {
        cat("RECOMMENDATION: Welch's t-test (unequal variances)\n")
      }
    } else {
      cat("RECOMMENDATION: Mann-Whitney U test\n")
    }
  }
  
  # Effect size calculation
  if (paired) {
    effect_size <- abs(mean(differences)) / sd(differences)
  } else {
    effect_size <- calculate_cohens_d_independent(group1, group2)$cohens_d
  }
  
  cat("Effect size (Cohen's d):", round(effect_size, 3), "\n")
  cat("Interpretation:", interpret_effect_size(effect_size), "\n")
}

# Apply to transmission comparison
choose_two_sample_test(automatic_mpg, manual_mpg, paired = FALSE)
```

### Reporting Guidelines

```r
# Function to generate comprehensive two-sample test report
generate_two_sample_report <- function(test_result, group1, group2, test_type = "t-test", paired = FALSE) {
  cat("=== TWO-SAMPLE TEST REPORT ===\n\n")
  
  cat("TEST INFORMATION:\n")
  cat("Test type:", test_type, "\n")
  cat("Design:", ifelse(paired, "Paired samples", "Independent samples"), "\n")
  cat("Group 1 n:", length(group1), "\n")
  cat("Group 2 n:", length(group2), "\n\n")
  
  cat("DESCRIPTIVE STATISTICS:\n")
  cat("Group 1 mean:", round(mean(group1), 3), "\n")
  cat("Group 1 SD:", round(sd(group1), 3), "\n")
  cat("Group 2 mean:", round(mean(group2), 3), "\n")
  cat("Group 2 SD:", round(sd(group2), 3), "\n")
  cat("Mean difference:", round(mean(group1) - mean(group2), 3), "\n\n")
  
  if (test_type == "t-test") {
    cat("T-TEST RESULTS:\n")
    cat("t-statistic:", round(test_result$statistic, 3), "\n")
    cat("Degrees of freedom:", round(test_result$parameter, 1), "\n")
    cat("p-value:", round(test_result$p.value, 4), "\n")
    cat("95% Confidence Interval:", round(test_result$conf.int, 3), "\n\n")
  } else if (test_type == "wilcoxon") {
    cat("WILCOXON TEST RESULTS:\n")
    cat("W-statistic:", test_result$statistic, "\n")
    cat("p-value:", round(test_result$p.value, 4), "\n\n")
  }
  
  # Effect size
  if (paired) {
    differences <- group1 - group2
    effect_size <- abs(mean(differences)) / sd(differences)
  } else {
    effect_size <- calculate_cohens_d_independent(group1, group2)$cohens_d
  }
  
  cat("EFFECT SIZE:\n")
  cat("Cohen's d:", round(effect_size, 3), "\n")
  cat("Interpretation:", interpret_effect_size(effect_size), "\n\n")
  
  # Conclusion
  alpha <- 0.05
  if (test_result$p.value < alpha) {
    cat("CONCLUSION:\n")
    cat("Reject the null hypothesis (p <", alpha, ")\n")
    cat("There is significant evidence of a difference between groups\n")
  } else {
    cat("CONCLUSION:\n")
    cat("Fail to reject the null hypothesis (p >=", alpha, ")\n")
    cat("There is insufficient evidence of a difference between groups\n")
  }
}

# Generate report for transmission comparison
transmission_t_test <- t.test(automatic_mpg, manual_mpg)
generate_two_sample_report(transmission_t_test, automatic_mpg, manual_mpg, "t-test", paired = FALSE)
```

## Exercises

### Exercise 1: Independent Samples t-Test
Compare the horsepower of cars with different cylinder counts using independent samples t-tests.

### Exercise 2: Paired Samples Analysis
Create a paired dataset and perform both parametric and nonparametric paired tests.

### Exercise 3: Effect Size Analysis
Calculate and interpret effect sizes for various two-sample comparisons in the mtcars dataset.

### Exercise 4: Assumption Checking
Perform comprehensive assumption checking for two-sample tests and recommend appropriate alternatives.

### Exercise 5: Power Analysis
Conduct power analysis to determine required sample sizes for detecting different effect sizes in two-sample designs.

## Next Steps

In the next chapter, we'll learn about one-way ANOVA for comparing means across multiple groups.

---

**Key Takeaways:**
- Two-sample tests compare means or distributions between two groups
- Independent samples tests are for unrelated groups
- Paired samples tests are for related observations
- Always check assumptions before choosing a test
- Effect sizes provide important information about practical significance
- Nonparametric alternatives exist for non-normal data
- Power analysis helps determine appropriate sample sizes 