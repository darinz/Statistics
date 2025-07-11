# Two-Sample Tests

## Overview

Two-sample tests are fundamental statistical procedures used to compare means or distributions between two groups. These tests form the backbone of experimental design and observational studies, allowing researchers to determine if observed differences between groups are statistically significant or due to random variation.

### When to Use Two-Sample Tests

Two-sample tests are appropriate when you want to:
- Compare treatment and control groups in experiments
- Assess differences between two populations
- Evaluate the effectiveness of interventions
- Test hypotheses about group differences
- Analyze before-and-after measurements

### Types of Two-Sample Tests

**Independent Samples:** Groups are unrelated (e.g., treatment vs control)
**Paired Samples:** Groups are related (e.g., before vs after, matched pairs)

### Key Concepts

**Null Hypothesis (H₀):** The population means are equal ($\mu_1 = \mu_2$)
**Alternative Hypothesis (H₁):** The population means are different ($\mu_1 \neq \mu_2$)

The test determines whether observed differences are statistically significant or due to random sampling variation.

### Mathematical Foundation

Two-sample tests are based on the sampling distribution of the difference between sample means:

```math
\bar{X}_1 - \bar{X}_2 \sim N\left(\mu_1 - \mu_2, \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}\right)
```

For large samples, the Central Limit Theorem ensures this distribution is approximately normal regardless of the underlying population distributions.

## Independent Samples t-Test

The independent samples t-test is the most commonly used parametric test for comparing means between two unrelated groups. It's robust and works well even with moderate violations of normality assumptions, especially for larger sample sizes.

### Mathematical Foundation

The independent samples t-test is based on the t-distribution and accounts for the uncertainty in estimating population standard deviations from sample data.

**Test Statistic:**
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
```

**Degrees of Freedom (Welch's approximation):**
```math
df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
```

**Pooled Standard Deviation (for equal variances):**
```math
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
```

**Confidence Interval:**
```math
CI = (\bar{x}_1 - \bar{x}_2) \pm t_{\alpha/2, df} \cdot \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
```

where:
- $\bar{x}_1, \bar{x}_2$ = sample means
- $s_1^2, s_2^2$ = sample variances
- $n_1, n_2$ = sample sizes
- $t_{\alpha/2, df}$ = critical t-value for confidence level $\alpha$

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

# Manual calculation for understanding
n1 <- length(automatic_mpg)
n2 <- length(manual_mpg)
mean1 <- mean(automatic_mpg)
mean2 <- mean(manual_mpg)
var1 <- var(automatic_mpg)
var2 <- var(manual_mpg)

# Welch's t-test (unequal variances)
se_welch <- sqrt(var1/n1 + var2/n2)
manual_t_welch <- (mean1 - mean2) / se_welch

# Degrees of freedom (Welch's approximation)
df_welch <- (var1/n1 + var2/n2)^2 / ((var1/n1)^2/(n1-1) + (var2/n2)^2/(n2-1))

cat("\nManual Calculation Verification:\n")
cat("Group 1 (Automatic): n =", n1, ", mean =", round(mean1, 3), 
    ", var =", round(var1, 3), "\n")
cat("Group 2 (Manual): n =", n2, ", mean =", round(mean2, 3), 
    ", var =", round(var2, 3), "\n")
cat("Standard Error (Welch):", round(se_welch, 3), "\n")
cat("Manual t-statistic:", round(manual_t_welch, 3), "\n")
cat("Degrees of freedom:", round(df_welch, 1), "\n")
```

### Equal vs Unequal Variances

The choice between equal and unequal variance t-tests depends on whether the population variances are assumed to be equal. This assumption significantly affects the test statistic and degrees of freedom.

**Mathematical Differences:**

**Equal Variances (Pooled t-test):**
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
```

where $s_p$ is the pooled standard deviation.

**Unequal Variances (Welch's t-test):**
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
```

**Degrees of Freedom:**
- Equal variances: $df = n_1 + n_2 - 2$
- Unequal variances: $df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}$

**When to Use Each:**
- **Equal variances:** When population variances are known to be equal or similar
- **Unequal variances:** When variances may differ (more conservative approach)

```r
# Test for equal variances
var_test <- var.test(automatic_mpg, manual_mpg)
print(var_test)

# Manual F-test calculation
f_statistic <- var(automatic_mpg) / var(manual_mpg)
df1 <- length(automatic_mpg) - 1
df2 <- length(manual_mpg) - 1
f_p_value <- 2 * (1 - pf(f_statistic, df1, df2))

cat("\nManual F-test calculation:\n")
cat("F-statistic:", round(f_statistic, 3), "\n")
cat("Degrees of freedom:", df1, ",", df2, "\n")
cat("p-value:", round(f_p_value, 4), "\n")

# Perform t-test with equal variances (if appropriate)
if (var_test$p.value > 0.05) {
  t_test_equal_var <- t.test(automatic_mpg, manual_mpg, var.equal = TRUE)
  cat("\nUsing equal variances t-test:\n")
  print(t_test_equal_var)
  
  # Manual pooled t-test calculation
  pooled_sd <- sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
  pooled_se <- pooled_sd * sqrt(1/n1 + 1/n2)
  manual_t_pooled <- (mean1 - mean2) / pooled_se
  df_pooled <- n1 + n2 - 2
  
  cat("\nManual pooled t-test calculation:\n")
  cat("Pooled SD:", round(pooled_sd, 3), "\n")
  cat("Pooled SE:", round(pooled_se, 3), "\n")
  cat("t-statistic:", round(manual_t_pooled, 3), "\n")
  cat("Degrees of freedom:", df_pooled, "\n")
  
} else {
  cat("\nUsing Welch's t-test (unequal variances):\n")
  print(t_test_result)
}

# Compare results
cat("\nComparison of t-tests:\n")
cat("Equal variances p-value:", round(t.test(automatic_mpg, manual_mpg, var.equal = TRUE)$p.value, 4), "\n")
cat("Unequal variances p-value:", round(t_test_result$p.value, 4), "\n")

# Effect on confidence intervals
ci_equal <- t.test(automatic_mpg, manual_mpg, var.equal = TRUE)$conf.int
ci_unequal <- t_test_result$conf.int

cat("\nConfidence Intervals:\n")
cat("Equal variances 95% CI:", round(ci_equal, 3), "\n")
cat("Unequal variances 95% CI:", round(ci_unequal, 3), "\n")

# Width comparison
width_equal <- ci_equal[2] - ci_equal[1]
width_unequal <- ci_unequal[2] - ci_unequal[1]

cat("CI width (equal variances):", round(width_equal, 3), "\n")
cat("CI width (unequal variances):", round(width_unequal, 3), "\n")
```

### Effect Size for Independent Samples

Effect size measures the magnitude of the difference between groups, independent of sample size. This is crucial for understanding practical significance beyond statistical significance.

**Mathematical Foundation:**

**Cohen's d (Standardized Mean Difference):**
```math
d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}
```

where $s_p$ is the pooled standard deviation.

**Pooled Standard Deviation:**
```math
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
```

**Hedges' g (Unbiased Estimator):**
```math
g = d \cdot \left(1 - \frac{3}{4(n_1 + n_2) - 9}\right)
```

**Confidence Interval for Effect Size:**
```math
CI_d = d \pm t_{\alpha/2, df} \cdot SE_d
```

where $SE_d = \sqrt{\frac{n_1 + n_2}{n_1 n_2} + \frac{d^2}{2(n_1 + n_2)}}$

**Interpretation Guidelines:**
- Small effect: $|d| < 0.2$
- Medium effect: $0.2 \leq |d| < 0.5$
- Large effect: $0.5 \leq |d| < 0.8$
- Very large effect: $|d| \geq 0.8$

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
  
  # Standard error of effect size
  se_d <- sqrt((n1 + n2)/(n1 * n2) + cohens_d^2/(2*(n1 + n2)))
  
  # Confidence interval for effect size
  df <- n1 + n2 - 2
  t_critical <- qt(0.975, df)
  ci_lower <- cohens_d - t_critical * se_d
  ci_upper <- cohens_d + t_critical * se_d
  
  return(list(
    cohens_d = cohens_d,
    hedges_g = hedges_g,
    pooled_sd = pooled_sd,
    se_d = se_d,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    n1 = n1,
    n2 = n2
  ))
}

# Apply to transmission comparison
transmission_effect <- calculate_cohens_d_independent(automatic_mpg, manual_mpg)

cat("Effect Size Analysis:\n")
cat("Cohen's d:", round(transmission_effect$cohens_d, 3), "\n")
cat("Hedges' g:", round(transmission_effect$hedges_g, 3), "\n")
cat("Pooled SD:", round(transmission_effect$pooled_sd, 3), "\n")
cat("Standard Error of d:", round(transmission_effect$se_d, 3), "\n")
cat("95% CI for d:", round(c(transmission_effect$ci_lower, transmission_effect$ci_upper), 3), "\n")

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

# Power analysis based on effect size
library(pwr)
power_analysis <- pwr.t2n.test(n1 = transmission_effect$n1, n2 = transmission_effect$n2, 
                               d = transmission_effect$cohens_d, sig.level = 0.05)
cat("Power for current effect size:", round(power_analysis$power, 3), "\n")

# Required sample size for 80% power
if (power_analysis$power < 0.8) {
  required_n <- pwr.t.test(d = transmission_effect$cohens_d, sig.level = 0.05, 
                           power = 0.8, type = "two.sample")$n
  cat("Required sample size per group for 80% power:", ceiling(required_n), "\n")
}
```

## Paired Samples t-Test

The paired samples t-test is used when observations in the two groups are related or matched. This design increases statistical power by controlling for individual differences and reducing error variance.

### Mathematical Foundation

The paired t-test analyzes the differences between paired observations rather than the raw scores.

**Test Statistic:**
```math
t = \frac{\bar{d}}{s_d/\sqrt{n}}
```

where:
- $\bar{d} = \frac{1}{n}\sum_{i=1}^n d_i$ (mean of differences)
- $d_i = x_{1i} - x_{2i}$ (difference for pair $i$)
- $s_d = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (d_i - \bar{d})^2}$ (standard deviation of differences)

**Degrees of Freedom:**
```math
df = n - 1
```

**Confidence Interval:**
```math
CI = \bar{d} \pm t_{\alpha/2, n-1} \cdot \frac{s_d}{\sqrt{n}}
```

**Effect Size for Paired Data:**
```math
d = \frac{\bar{d}}{s_d}
```

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

# Manual calculation for understanding
differences <- after_scores - before_scores
mean_diff <- mean(differences)
sd_diff <- sd(differences)
se_diff <- sd_diff / sqrt(n_subjects)
manual_t_paired <- mean_diff / se_diff
df_paired <- n_subjects - 1

cat("\nManual Paired t-test Calculation:\n")
cat("Mean difference:", round(mean_diff, 3), "\n")
cat("SD of differences:", round(sd_diff, 3), "\n")
cat("Standard error:", round(se_diff, 3), "\n")
cat("t-statistic:", round(manual_t_paired, 3), "\n")
cat("Degrees of freedom:", df_paired, "\n")

# Calculate paired effect size
paired_effect <- mean_diff / sd_diff

cat("\nPaired Samples Results:\n")
cat("Mean difference (After - Before):", round(mean_diff, 3), "\n")
cat("t-statistic:", round(paired_test$statistic, 3), "\n")
cat("p-value:", round(paired_test$p.value, 4), "\n")
cat("Effect size (Cohen's d):", round(paired_effect, 3), "\n")

# Compare with independent samples approach (incorrect for paired data)
independent_effect <- calculate_cohens_d_independent(before_scores, after_scores)
cat("Independent samples effect size (incorrect):", round(independent_effect$cohens_d, 3), "\n")
cat("Paired samples effect size (correct):", round(paired_effect, 3), "\n")

# Power comparison
paired_power <- pwr.t.test(n = n_subjects, d = paired_effect, sig.level = 0.05, 
                           type = "paired")$power
independent_power <- pwr.t2n.test(n1 = n_subjects, n2 = n_subjects, 
                                  d = independent_effect$cohens_d, sig.level = 0.05)$power

cat("\nPower Comparison:\n")
cat("Paired design power:", round(paired_power, 3), "\n")
cat("Independent design power:", round(independent_power, 3), "\n")
cat("Power advantage:", round(paired_power - independent_power, 3), "\n")
```

### Paired Data Analysis

Paired data analysis provides insights into the relationship between measurements and the distribution of differences, which is crucial for understanding the effectiveness of interventions.

**Key Concepts:**

**Correlation in Paired Data:**
```math
r = \frac{\sum_{i=1}^n (x_{1i} - \bar{x}_1)(x_{2i} - \bar{x}_2)}{\sqrt{\sum_{i=1}^n (x_{1i} - \bar{x}_1)^2 \sum_{i=1}^n (x_{2i} - \bar{x}_2)^2}}
```

**Variance Reduction:**
```math
\text{Var}(\bar{d}) = \frac{\sigma_d^2}{n} = \frac{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}{n}
```

where $\rho$ is the correlation between paired observations.

**Power Advantage:**
The paired design increases power by reducing error variance through the correlation between measurements.

```r
# Create paired data frame
paired_data <- data.frame(
  subject = 1:n_subjects,
  before = before_scores,
  after = after_scores,
  difference = after_scores - before_scores
)

# Comprehensive paired data analysis
cat("=== PAIRED DATA ANALYSIS ===\n")
cat("Sample size:", n_subjects, "\n")
cat("Mean difference:", round(mean(paired_data$difference), 3), "\n")
cat("SD of differences:", round(sd(paired_data$difference), 3), "\n")
cat("SE of mean difference:", round(sd(paired_data$difference)/sqrt(n_subjects), 3), "\n")

# Correlation analysis
correlation <- cor(paired_data$before, paired_data$after)
cat("Correlation between before and after:", round(correlation, 3), "\n")

# Variance analysis
var_before <- var(paired_data$before)
var_after <- var(paired_data$after)
var_diff <- var(paired_data$difference)

cat("\nVariance Analysis:\n")
cat("Variance (before):", round(var_before, 3), "\n")
cat("Variance (after):", round(var_after, 3), "\n")
cat("Variance (differences):", round(var_diff, 3), "\n")
cat("Theoretical variance (if independent):", round(var_before + var_after, 3), "\n")
cat("Variance reduction:", round(var_before + var_after - var_diff, 3), "\n")

# Effect of correlation on power
theoretical_var_independent <- var_before + var_after
actual_var_paired <- var_diff
power_ratio <- theoretical_var_independent / actual_var_paired

cat("\nPower Analysis:\n")
cat("Variance ratio (independent/paired):", round(power_ratio, 3), "\n")
cat("Effective sample size increase:", round(power_ratio, 1), "times\n")

# Individual change analysis
positive_changes <- sum(paired_data$difference > 0)
negative_changes <- sum(paired_data$difference < 0)
no_change <- sum(paired_data$difference == 0)

cat("\nIndividual Change Analysis:\n")
cat("Subjects with improvement:", positive_changes, "(", round(positive_changes/n_subjects*100, 1), "%)\n")
cat("Subjects with decline:", negative_changes, "(", round(negative_changes/n_subjects*100, 1), "%)\n")
cat("Subjects with no change:", no_change, "(", round(no_change/n_subjects*100, 1), "%)\n")

# Normality of differences
shapiro_diff <- shapiro.test(paired_data$difference)
cat("\nNormality of Differences:\n")
cat("Shapiro-Wilk p-value:", round(shapiro_diff$p.value, 4), "\n")
if (shapiro_diff$p.value < 0.05) {
  cat("Differences are not normally distributed - consider nonparametric test\n")
} else {
  cat("Differences appear normally distributed - parametric test appropriate\n")
}

# Visualize paired data
library(ggplot2)
library(gridExtra)

# Before vs After plot
p1 <- ggplot(paired_data, aes(x = before, y = after)) +
  geom_point(alpha = 0.7) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Before vs After Scores",
       subtitle = paste("r =", round(correlation, 3))) +
  theme_minimal()

# Difference plot
p2 <- ggplot(paired_data, aes(x = difference)) +
  geom_histogram(bins = 10, fill = "steelblue", alpha = 0.7) +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
  geom_vline(xintercept = mean(paired_data$difference), color = "blue", size = 1) +
  labs(title = "Distribution of Differences",
       subtitle = paste("Mean =", round(mean(paired_data$difference), 2))) +
  theme_minimal()

# Combine plots
grid.arrange(p1, p2, ncol = 2)
```

## Nonparametric Two-Sample Tests

Nonparametric tests make fewer assumptions about the underlying population distributions and are robust to violations of normality. They're particularly useful for small samples or when data is skewed or contains outliers.

### Mann-Whitney U Test (Wilcoxon Rank-Sum)

The Mann-Whitney U test is the most commonly used nonparametric alternative to the independent samples t-test. It tests whether the distributions of two groups differ by analyzing the ranks of the combined data.

**Mathematical Foundation:**

1. **Combine and rank all observations:**
   ```math
   R_i = \text{rank}(x_i) \text{ in combined sample}
   ```

2. **Calculate rank sums:**
   ```math
   W_1 = \sum_{i=1}^{n_1} R_i \text{ (Group 1)}
   ```
   ```math
   W_2 = \sum_{i=1}^{n_2} R_i \text{ (Group 2)}
   ```

3. **Calculate U statistics:**
   ```math
   U_1 = W_1 - \frac{n_1(n_1 + 1)}{2}
   ```
   ```math
   U_2 = W_2 - \frac{n_2(n_2 + 1)}{2}
   ```

4. **Test statistic:**
   ```math
   U = \min(U_1, U_2)
   ```

**For large samples ($n_1, n_2 > 20$), U is approximately normal:**
```math
Z = \frac{U - \mu_U}{\sigma_U}
```

where:
```math
\mu_U = \frac{n_1 n_2}{2}
```
```math
\sigma_U = \sqrt{\frac{n_1 n_2(n_1 + n_2 + 1)}{12}}
```

**Effect Size (r):**
```math
r = \frac{Z}{\sqrt{n_1 + n_2}}
```

```r
# Mann-Whitney U test for independent samples
wilcox_test <- wilcox.test(automatic_mpg, manual_mpg)
print(wilcox_test)

# Manual calculation for understanding
combined_data <- c(automatic_mpg, manual_mpg)
ranks <- rank(combined_data)
n1 <- length(automatic_mpg)
n2 <- length(manual_mpg)

# Rank sums
ranks_group1 <- ranks[1:n1]
ranks_group2 <- ranks[(n1+1):(n1+n2)]
W1 <- sum(ranks_group1)
W2 <- sum(ranks_group2)

# U statistics
U1 <- W1 - (n1 * (n1 + 1)) / 2
U2 <- W2 - (n2 * (n2 + 1)) / 2
U_statistic <- min(U1, U2)

cat("\nManual Mann-Whitney Calculation:\n")
cat("Rank sum (Group 1):", W1, "\n")
cat("Rank sum (Group 2):", W2, "\n")
cat("U1:", U1, "\n")
cat("U2:", U2, "\n")
cat("U statistic:", U_statistic, "\n")
cat("R function U statistic:", wilcox_test$statistic, "\n")

# Compare with t-test results
cat("\nComparison of parametric and nonparametric tests:\n")
cat("t-test p-value:", round(t_test_result$p.value, 4), "\n")
cat("Wilcoxon p-value:", round(wilcox_test$p.value, 4), "\n")

# Effect size for Wilcoxon test
wilcox_effect_size <- abs(qnorm(wilcox_test$p.value / 2)) / sqrt(n1 + n2)
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

# Power comparison
wilcox_power <- pwr.t2n.test(n1 = n1, n2 = n2, d = wilcox_effect_size, sig.level = 0.05)$power
t_power <- pwr.t2n.test(n1 = n1, n2 = n2, d = transmission_effect$cohens_d, sig.level = 0.05)$power

cat("\nPower Comparison:\n")
cat("t-test power:", round(t_power, 3), "\n")
cat("Wilcoxon power:", round(wilcox_power, 3), "\n")
```

### Wilcoxon Signed-Rank Test for Paired Data

The Wilcoxon signed-rank test is the nonparametric alternative to the paired t-test. It analyzes the ranks of the absolute differences between paired observations.

**Mathematical Foundation:**

1. **Calculate differences:**
   ```math
   d_i = x_{1i} - x_{2i}
   ```

2. **Rank absolute differences:**
   ```math
   R_i = \text{rank}(|d_i|)
   ```

3. **Assign signs:**
   ```math
   s_i = \text{sign}(d_i)
   ```

4. **Calculate test statistic:**
   ```math
   W = \sum_{i=1}^{n} s_i \cdot R_i
   ```

**For large samples ($n > 20$), W is approximately normal:**
```math
Z = \frac{W}{\sqrt{\frac{n(n+1)(2n+1)}{6}}}
```

**Effect Size (r):**
```math
r = \frac{Z}{\sqrt{n}}
```

```r
# Wilcoxon signed-rank test for paired samples
paired_wilcox <- wilcox.test(before_scores, after_scores, paired = TRUE)
print(paired_wilcox)

# Manual calculation for understanding
differences <- before_scores - after_scores
abs_differences <- abs(differences)
ranks <- rank(abs_differences)
signed_ranks <- sign(differences) * ranks
W_statistic <- sum(signed_ranks)

cat("\nManual Wilcoxon Signed-Rank Calculation:\n")
cat("W statistic:", W_statistic, "\n")
cat("R function W statistic:", paired_wilcox$statistic, "\n")

# Compare with paired t-test
cat("\nComparison of paired tests:\n")
cat("Paired t-test p-value:", round(paired_test$p.value, 4), "\n")
cat("Paired Wilcoxon p-value:", round(paired_wilcox$p.value, 4), "\n")

# Effect size for paired Wilcoxon
paired_wilcox_effect <- abs(qnorm(paired_wilcox$p.value / 2)) / sqrt(n_subjects)
cat("Paired Wilcoxon effect size (r):", round(paired_wilcox_effect, 3), "\n")
cat("Effect interpretation:", interpret_wilcox_effect(paired_wilcox_effect), "\n")

# Power comparison for paired tests
paired_wilcox_power <- pwr.t.test(n = n_subjects, d = paired_wilcox_effect, 
                                  sig.level = 0.05, type = "paired")$power

cat("\nPaired Tests Power Comparison:\n")
cat("Paired t-test power:", round(paired_power, 3), "\n")
cat("Paired Wilcoxon power:", round(paired_wilcox_power, 3), "\n")

# Assumption checking for paired data
cat("\nAssumption Checking for Paired Data:\n")
cat("Normality of differences (Shapiro-Wilk):", round(shapiro_diff$p.value, 4), "\n")
if (shapiro_diff$p.value < 0.05) {
  cat("RECOMMENDATION: Use Wilcoxon signed-rank test\n")
} else {
  cat("RECOMMENDATION: Paired t-test is appropriate\n")
}
```

## Assumption Checking

Proper assumption checking is crucial for valid statistical inference in two-sample tests. Violations of assumptions can lead to incorrect conclusions and inflated Type I error rates.

### Key Assumptions for Two-Sample Tests

1. **Independence:** Observations within and between groups are independent
2. **Normality:** Data follows normal distributions (for parametric tests)
3. **Homogeneity of Variance:** Population variances are equal (for pooled t-test)
4. **Random Sampling:** Data represents random samples from populations
5. **No Outliers:** Extreme values don't unduly influence results

### Normality Tests

The normality assumption is critical for parametric tests. Several methods can assess normality for each group.

**Mathematical Tests:**
- **Shapiro-Wilk:** Most powerful test for normality
- **Anderson-Darling:** Good for detecting departures from normality
- **Kolmogorov-Smirnov:** Tests against a specified distribution

**Graphical Methods:**
- **Q-Q plots:** Compare sample quantiles to theoretical normal quantiles
- **Histograms:** Visual assessment of distribution shape
- **Box plots:** Detect skewness and outliers

```r
# Comprehensive normality checking function for two groups
check_normality_groups <- function(group1, group2, group_names = c("Group 1", "Group 2")) {
  cat("=== COMPREHENSIVE NORMALITY ASSESSMENT ===\n")
  
  # Basic information
  n1 <- length(group1)
  n2 <- length(group2)
  cat("Sample sizes:", n1, "and", n2, "\n\n")
  
  # Shapiro-Wilk tests
  shapiro1 <- shapiro.test(group1)
  shapiro2 <- shapiro.test(group2)
  
  cat("Shapiro-Wilk Tests:\n")
  cat(group_names[1], "p-value:", round(shapiro1$p.value, 4), "\n")
  cat(group_names[2], "p-value:", round(shapiro2$p.value, 4), "\n\n")
  
  # Anderson-Darling tests
  library(nortest)
  ad1 <- ad.test(group1)
  ad2 <- ad.test(group2)
  
  cat("Anderson-Darling Tests:\n")
  cat(group_names[1], "p-value:", round(ad1$p.value, 4), "\n")
  cat(group_names[2], "p-value:", round(ad2$p.value, 4), "\n\n")
  
  # Descriptive statistics
  cat("Descriptive Statistics:\n")
  cat(group_names[1], "mean:", round(mean(group1), 3), "SD:", round(sd(group1), 3), "\n")
  cat(group_names[2], "mean:", round(mean(group2), 3), "SD:", round(sd(group2), 3), "\n")
  
  # Skewness and kurtosis
  skewness1 <- moments::skewness(group1)
  kurtosis1 <- moments::kurtosis(group1)
  skewness2 <- moments::skewness(group2)
  kurtosis2 <- moments::kurtosis(group2)
  
  cat("\nSkewness and Kurtosis:\n")
  cat(group_names[1], "skewness:", round(skewness1, 3), "kurtosis:", round(kurtosis1, 3), "\n")
  cat(group_names[2], "skewness:", round(skewness2, 3), "kurtosis:", round(kurtosis2, 3), "\n\n")
  
  # Q-Q plots
  par(mfrow = c(1, 2))
  qqnorm(group1, main = paste("Q-Q Plot:", group_names[1]))
  qqline(group1, col = "red")
  
  qqnorm(group2, main = paste("Q-Q Plot:", group_names[2]))
  qqline(group2, col = "red")
  par(mfrow = c(1, 1))
  
  # Comprehensive recommendations
  cat("=== RECOMMENDATIONS ===\n")
  
  # Sample size considerations
  if (n1 >= 30 && n2 >= 30) {
    cat("✓ Large sample sizes: Central Limit Theorem applies\n")
    cat("✓ Parametric tests are robust to moderate normality violations\n")
  } else if (n1 >= 15 && n2 >= 15) {
    cat("⚠️  Moderate sample sizes: Check normality carefully\n")
  } else {
    cat("⚠️  Small sample sizes: Nonparametric tests recommended\n")
  }
  
  # Normality assessment
  if (shapiro1$p.value >= 0.05 && shapiro2$p.value >= 0.05) {
    cat("✓ Both groups appear normally distributed\n")
    cat("✓ Parametric tests are appropriate\n")
  } else {
    cat("✗ At least one group is not normally distributed\n")
    if (n1 < 30 || n2 < 30) {
      cat("✗ Consider nonparametric alternatives\n")
    } else {
      cat("⚠️  Parametric tests may still be appropriate due to large sample sizes\n")
    }
  }
  
  # Skewness assessment
  if (abs(skewness1) < 1 && abs(skewness2) < 1) {
    cat("✓ Both groups have acceptable skewness\n")
  } else {
    cat("✗ At least one group is significantly skewed\n")
  }
  
  # Kurtosis assessment
  if (abs(kurtosis1 - 3) < 2 && abs(kurtosis2 - 3) < 2) {
    cat("✓ Both groups have acceptable kurtosis\n")
  } else {
    cat("✗ At least one group has unusual kurtosis\n")
  }
  
  return(list(
    shapiro1 = shapiro1,
    shapiro2 = shapiro2,
    ad1 = ad1,
    ad2 = ad2,
    skewness1 = skewness1,
    skewness2 = skewness2,
    kurtosis1 = kurtosis1,
    kurtosis2 = kurtosis2
  ))
}

# Check normality for transmission groups
normality_results <- check_normality_groups(automatic_mpg, manual_mpg, 
                                          c("Automatic", "Manual"))
```

### Homogeneity of Variance

The homogeneity of variance assumption is crucial for choosing between pooled and Welch's t-tests. Violations can lead to incorrect Type I error rates and reduced power.

**Mathematical Foundation:**

**F-test for Equality of Variances:**
```math
F = \frac{s_1^2}{s_2^2} \sim F_{n_1-1, n_2-1}
```

**Levene's Test (More Robust):**
```math
W = \frac{(N-k)}{(k-1)} \cdot \frac{\sum_{i=1}^k n_i(\bar{Z}_i - \bar{Z})^2}{\sum_{i=1}^k \sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_i)^2}
```

where $Z_{ij} = |X_{ij} - \bar{X}_i|$ and $\bar{Z}_i$ is the mean of $Z_{ij}$ for group $i$.

**Variance Ratio Guidelines:**
- Ratio < 2: Variances considered equal
- Ratio 2-4: Moderate difference, use Welch's test
- Ratio > 4: Large difference, use Welch's test

```r
# Comprehensive homogeneity of variance testing function
check_homogeneity <- function(group1, group2, group_names = c("Group 1", "Group 2")) {
  cat("=== HOMOGENEITY OF VARIANCE ASSESSMENT ===\n")
  
  # Basic statistics
  n1 <- length(group1)
  n2 <- length(group2)
  var1 <- var(group1)
  var2 <- var(group2)
  ratio <- max(var1, var2) / min(var1, var2)
  
  cat("Sample sizes:", n1, "and", n2, "\n")
  cat("Variances:", round(var1, 3), "and", round(var2, 3), "\n")
  cat("Variance ratio:", round(ratio, 3), "\n\n")
  
  # F-test for equality of variances
  f_test <- var.test(group1, group2)
  cat("F-test for Equality of Variances:\n")
  cat("F-statistic:", round(f_test$statistic, 3), "\n")
  cat("p-value:", round(f_test$p.value, 4), "\n")
  cat("95% CI for ratio:", round(f_test$conf.int, 3), "\n\n")
  
  # Levene's test (more robust)
  library(car)
  combined_data <- c(group1, group2)
  group_labels <- factor(c(rep(group_names[1], length(group1)), 
                        rep(group_names[2], length(group2))))
  levene_test <- leveneTest(combined_data, group_labels)
  cat("Levene's Test (Robust):\n")
  cat("F-statistic:", round(levene_test$`F value`[1], 3), "\n")
  cat("p-value:", round(levene_test$`Pr(>F)`[1], 4), "\n\n")
  
  # Brown-Forsythe test (median-based)
  bf_test <- leveneTest(combined_data, group_labels, center = "median")
  cat("Brown-Forsythe Test (Median-based):\n")
  cat("F-statistic:", round(bf_test$`F value`[1], 3), "\n")
  cat("p-value:", round(bf_test$`Pr(>F)`[1], 4), "\n\n")
  
  # Comprehensive recommendations
  cat("=== RECOMMENDATIONS ===\n")
  
  # Variance ratio assessment
  if (ratio < 2) {
    cat("✓ Variance ratio < 2: Variances appear equal\n")
  } else if (ratio < 4) {
    cat("⚠️  Variance ratio 2-4: Moderate difference in variances\n")
  } else {
    cat("✗ Variance ratio > 4: Large difference in variances\n")
  }
  
  # F-test assessment
  if (f_test$p.value >= 0.05) {
    cat("✓ F-test: Variances appear equal (p >=", 0.05, ")\n")
  } else {
    cat("✗ F-test: Variances are significantly different (p <", 0.05, ")\n")
  }
  
  # Levene's test assessment
  if (levene_test$`Pr(>F)`[1] >= 0.05) {
    cat("✓ Levene's test: Variances appear equal (p >=", 0.05, ")\n")
  } else {
    cat("✗ Levene's test: Variances are significantly different (p <", 0.05, ")\n")
  }
  
  # Final recommendation
  cat("\nFINAL RECOMMENDATION:\n")
  if (f_test$p.value >= 0.05 && levene_test$`Pr(>F)`[1] >= 0.05 && ratio < 2) {
    cat("✓ Use pooled t-test (equal variances)\n")
    cat("✓ Standard t-test with var.equal = TRUE\n")
  } else {
    cat("✗ Use Welch's t-test (unequal variances)\n")
    cat("✗ Standard t-test with var.equal = FALSE\n")
  }
  
  # Impact on results
  t_pooled <- t.test(group1, group2, var.equal = TRUE)
  t_welch <- t.test(group1, group2, var.equal = FALSE)
  
  cat("\nImpact on Results:\n")
  cat("Pooled t-test p-value:", round(t_pooled$p.value, 4), "\n")
  cat("Welch's t-test p-value:", round(t_welch$p.value, 4), "\n")
  cat("Pooled t-test df:", round(t_pooled$parameter, 1), "\n")
  cat("Welch's t-test df:", round(t_welch$parameter, 1), "\n")
  
  return(list(
    f_test = f_test,
    levene_test = levene_test,
    bf_test = bf_test,
    variance_ratio = ratio,
    t_pooled = t_pooled,
    t_welch = t_welch
  ))
}

# Check homogeneity for transmission groups
homogeneity_results <- check_homogeneity(automatic_mpg, manual_mpg, 
                                        c("Automatic", "Manual"))
```

## Power Analysis

Power analysis helps determine the probability of detecting a true effect and the sample size needed for adequate statistical power in two-sample designs. This is crucial for study design and interpreting results.

### Mathematical Foundation

**Power Definition:**
```math
Power = P(\text{Reject } H_0 | H_1 \text{ is true}) = 1 - \beta
```

**For two-sample t-test:**
```math
Power = P\left(|T| > t_{\alpha/2, df} \mid \mu_1 - \mu_2 = \delta\right)
```

where $\delta$ is the true effect size and $t_{\alpha/2, df}$ is the critical t-value.

**Effect Size and Power Relationship:**
- Larger effect sizes require smaller sample sizes for the same power
- Smaller effect sizes require larger sample sizes for adequate power
- Power increases with sample size for fixed effect sizes
- Unequal sample sizes reduce power compared to equal sample sizes

### Power Analysis for Two-Sample t-Test

```r
library(pwr)

# Comprehensive power analysis function for two-sample designs
power_analysis_two_sample <- function(n1, n2, effect_size, alpha = 0.05) {
  # Calculate power for current sample sizes
  power_result <- pwr.t2n.test(n1 = n1, n2 = n2, d = effect_size, sig.level = alpha)
  
  # Calculate required sample sizes for different power levels
  sample_size_80 <- pwr.t.test(d = effect_size, sig.level = alpha, power = 0.8, type = "two.sample")
  sample_size_90 <- pwr.t.test(d = effect_size, sig.level = alpha, power = 0.9, type = "two.sample")
  
  # Calculate power for different effect sizes
  small_effect_power <- pwr.t2n.test(n1 = n1, n2 = n2, d = 0.2, sig.level = alpha)$power
  medium_effect_power <- pwr.t2n.test(n1 = n1, n2 = n2, d = 0.5, sig.level = alpha)$power
  large_effect_power <- pwr.t2n.test(n1 = n1, n2 = n2, d = 0.8, sig.level = alpha)$power
  
  # Calculate power for equal sample sizes
  n_equal <- (n1 + n2) / 2
  power_equal <- pwr.t.test(n = n_equal, d = effect_size, sig.level = alpha, type = "two.sample")$power
  
  return(list(
    power = power_result$power,
    required_n_80 = ceiling(sample_size_80$n),
    required_n_90 = ceiling(sample_size_90$n),
    effect_size = effect_size,
    alpha = alpha,
    small_effect_power = small_effect_power,
    medium_effect_power = medium_effect_power,
    large_effect_power = large_effect_power,
    power_equal = power_equal,
    n1 = n1,
    n2 = n2
  ))
}

# Apply to transmission comparison
transmission_power <- power_analysis_two_sample(
  length(automatic_mpg), 
  length(manual_mpg), 
  transmission_effect$cohens_d
)

cat("=== POWER ANALYSIS FOR TWO-SAMPLE DESIGN ===\n")
cat("Current sample sizes:", transmission_power$n1, "and", transmission_power$n2, "\n")
cat("Effect size (Cohen's d):", round(transmission_power$effect_size, 3), "\n")
cat("Current power:", round(transmission_power$power, 3), "\n")

cat("\nRequired Sample Sizes:\n")
cat("For 80% power:", transmission_power$required_n_80, "per group\n")
cat("For 90% power:", transmission_power$required_n_90, "per group\n")

cat("\nPower for Different Effect Sizes:\n")
cat("Small effect (d = 0.2):", round(transmission_power$small_effect_power, 3), "\n")
cat("Medium effect (d = 0.5):", round(transmission_power$medium_effect_power, 3), "\n")
cat("Large effect (d = 0.8):", round(transmission_power$large_effect_power, 3), "\n")

cat("\nSample Size Efficiency:\n")
cat("Current power (unequal n):", round(transmission_power$power, 3), "\n")
cat("Power with equal n:", round(transmission_power$power_equal, 3), "\n")
cat("Power loss due to unequal n:", round(transmission_power$power_equal - transmission_power$power, 3), "\n")

# Power curve analysis
effect_sizes <- seq(0.1, 1.0, by = 0.1)
power_curve <- sapply(effect_sizes, function(d) {
  pwr.t2n.test(n1 = transmission_power$n1, n2 = transmission_power$n2, 
               d = d, sig.level = 0.05)$power
})

cat("\nPower Curve (Effect Size vs Power):\n")
for (i in 1:length(effect_sizes)) {
  cat("d =", effect_sizes[i], ": Power =", round(power_curve[i], 3), "\n")
}

# Sample size recommendations
cat("\nSample Size Recommendations:\n")
if (transmission_power$power < 0.8) {
  cat("⚠️  Current power is below 80%\n")
  cat("📈 Consider increasing sample size to", transmission_power$required_n_80, "per group\n")
} else {
  cat("✓ Current power is adequate (≥80%)\n")
}

if (abs(transmission_power$n1 - transmission_power$n2) > 0) {
  cat("⚠️  Unequal sample sizes detected\n")
  cat("📈 Consider equal sample sizes for maximum power\n")
} else {
  cat("✓ Equal sample sizes (optimal for power)\n")
}
```

## Practical Examples

Real-world applications of two-sample tests demonstrate their importance across various fields. These examples show how to apply the concepts in practice with comprehensive analysis.

### Example 1: Clinical Trial

Clinical trials are a common application where two-sample tests are used to evaluate treatment effectiveness compared to control conditions.

**Scenario:** A pharmaceutical company conducts a clinical trial to test a new medication for reducing blood pressure. Patients are randomly assigned to either the treatment group (new medication) or control group (placebo).

```r
# Simulate clinical trial data
set.seed(123)
n_treatment <- 25
n_control <- 25

# Generate treatment and control group data
treatment_scores <- rnorm(n_treatment, mean = 85, sd = 12)
control_scores <- rnorm(n_control, mean = 75, sd = 10)

# Comprehensive clinical trial analysis
cat("=== CLINICAL TRIAL ANALYSIS ===\n")
cat("Treatment group n:", n_treatment, "\n")
cat("Control group n:", n_control, "\n")
cat("Treatment mean:", round(mean(treatment_scores), 2), "\n")
cat("Control mean:", round(mean(control_scores), 2), "\n")
cat("Treatment SD:", round(sd(treatment_scores), 2), "\n")
cat("Control SD:", round(sd(control_scores), 2), "\n")

# Perform comprehensive t-test analysis
clinical_t_test <- t.test(treatment_scores, control_scores)
print(clinical_t_test)

# Calculate effect size
clinical_effect <- calculate_cohens_d_independent(treatment_scores, control_scores)

cat("\nClinical Trial Results:\n")
cat("Mean difference (Treatment - Control):", round(mean(treatment_scores) - mean(control_scores), 2), "\n")
cat("Effect size (Cohen's d):", round(clinical_effect$cohens_d, 3), "\n")
cat("Effect interpretation:", interpret_effect_size(clinical_effect$cohens_d), "\n")

# Power analysis
clinical_power <- power_analysis_two_sample(n_treatment, n_control, clinical_effect$cohens_d)
cat("Power for detecting this effect:", round(clinical_power$power, 3), "\n")

# Clinical significance assessment
mean_diff <- mean(treatment_scores) - mean(control_scores)
ci_width <- clinical_t_test$conf.int[2] - clinical_t_test$conf.int[1]

cat("\nClinical Significance Assessment:\n")
cat("95% CI for mean difference:", round(clinical_t_test$conf.int, 2), "\n")
cat("CI width:", round(ci_width, 2), "\n")

# Clinical interpretation
alpha <- 0.05
if (clinical_t_test$p.value < alpha) {
  cat("\nCLINICAL CONCLUSION:\n")
  cat("✓ Treatment shows significant improvement (p <", alpha, ")\n")
  cat("✓ Mean improvement:", round(mean_diff, 1), "points\n")
  if (clinical_effect$cohens_d >= 0.5) {
    cat("✓ Effect size is clinically meaningful\n")
  } else {
    cat("⚠️  Effect size may not be clinically meaningful\n")
  }
} else {
  cat("\nCLINICAL CONCLUSION:\n")
  cat("✗ No significant treatment effect (p >=", alpha, ")\n")
  cat("✗ Consider larger sample size or different treatment\n")
}

# Number needed to treat (NNT) calculation
# Assuming higher scores are better
improvement_rate_treatment <- sum(treatment_scores > mean(control_scores)) / n_treatment
improvement_rate_control <- sum(control_scores > mean(control_scores)) / n_control
nnt <- 1 / (improvement_rate_treatment - improvement_rate_control)

cat("\nNumber Needed to Treat (NNT):\n")
cat("NNT:", round(nnt, 1), "patients\n")
cat("Interpretation: Treat", round(nnt, 0), "patients to see 1 additional benefit\n")

# Safety analysis (outliers)
treatment_outliers <- sum(abs(scale(treatment_scores)) > 3)
control_outliers <- sum(abs(scale(control_scores)) > 3)

cat("\nSafety Analysis:\n")
cat("Treatment outliers:", treatment_outliers, "\n")
cat("Control outliers:", control_outliers, "\n")
```

### Example 2: Educational Research

Educational research frequently uses two-sample tests to evaluate the effectiveness of teaching interventions, comparing experimental and control groups.

**Scenario:** A researcher wants to evaluate the effectiveness of a new teaching method on student performance. Students are randomly assigned to either the experimental group (new method) or control group (traditional method).

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

# Comprehensive educational analysis
cat("=== EDUCATIONAL INTERVENTION ANALYSIS ===\n")
cat("Experimental group n:", n_experimental, "\n")
cat("Control group n:", n_control, "\n")

# Pre-test analysis
cat("\nPre-test Analysis:\n")
cat("Experimental pre-test mean:", round(mean(experimental_pre), 2), "\n")
cat("Control pre-test mean:", round(mean(control_pre), 2), "\n")
pre_test_diff <- mean(experimental_pre) - mean(control_pre)
cat("Pre-test difference:", round(pre_test_diff, 2), "\n")

# Check for baseline differences
pre_test_t <- t.test(experimental_pre, control_pre)
cat("Pre-test t-test p-value:", round(pre_test_t$p.value, 4), "\n")
if (pre_test_t$p.value < 0.05) {
  cat("⚠️  Significant baseline differences detected\n")
} else {
  cat("✓ No significant baseline differences\n")
}

# Analyze gain scores
experimental_gains <- experimental_post - experimental_pre
control_gains <- control_post - control_pre

cat("\nGain Score Analysis:\n")
cat("Experimental gain mean:", round(mean(experimental_gains), 2), "\n")
cat("Control gain mean:", round(mean(control_gains), 2), "\n")
cat("Experimental gain SD:", round(sd(experimental_gains), 2), "\n")
cat("Control gain SD:", round(sd(control_gains), 2), "\n")

# Independent samples t-test on gains
gains_test <- t.test(experimental_gains, control_gains)
print(gains_test)

# Effect size for gains
gains_effect <- calculate_cohens_d_independent(experimental_gains, control_gains)

cat("\nEducational Intervention Results:\n")
cat("Gain difference (Experimental - Control):", round(mean(experimental_gains) - mean(control_gains), 2), "\n")
cat("Effect size (Cohen's d):", round(gains_effect$cohens_d, 3), "\n")
cat("Effect interpretation:", interpret_effect_size(gains_effect$cohens_d), "\n")

# Power analysis
edu_power <- power_analysis_two_sample(n_experimental, n_control, gains_effect$cohens_d)
cat("Power for detecting this effect:", round(edu_power$power, 3), "\n")

# Educational significance assessment
gain_diff <- mean(experimental_gains) - mean(control_gains)
gain_ci <- gains_test$conf.int

cat("\nEducational Significance Assessment:\n")
cat("95% CI for gain difference:", round(gain_ci, 2), "\n")

# Practical significance (minimum important difference)
# Assume 5 points is educationally meaningful
mid <- 5
if (abs(gain_diff) >= mid) {
  cat("✓ Effect exceeds minimum important difference (", mid, "points)\n")
} else {
  cat("⚠️  Effect below minimum important difference (", mid, "points)\n")
}

# Educational interpretation
alpha <- 0.05
if (gains_test$p.value < alpha) {
  cat("\nEDUCATIONAL CONCLUSION:\n")
  cat("✓ New teaching method shows significant improvement (p <", alpha, ")\n")
  cat("✓ Average improvement:", round(gain_diff, 1), "points\n")
  
  # Effect size interpretation for education
  if (gains_effect$cohens_d >= 0.5) {
    cat("✓ Large practical effect on learning\n")
  } else if (gains_effect$cohens_d >= 0.3) {
    cat("✓ Moderate practical effect on learning\n")
  } else {
    cat("⚠️  Small practical effect on learning\n")
  }
} else {
  cat("\nEDUCATIONAL CONCLUSION:\n")
  cat("✗ No significant improvement with new method (p >=", alpha, ")\n")
  cat("✗ Consider revising the intervention or increasing sample size\n")
}

# Individual student analysis
experimental_improvers <- sum(experimental_gains > 0)
control_improvers <- sum(control_gains > 0)

cat("\nIndividual Student Analysis:\n")
cat("Experimental students improving:", experimental_improvers, "/", n_experimental, 
    "(", round(experimental_improvers/n_experimental*100, 1), "%)\n")
cat("Control students improving:", control_improvers, "/", n_control, 
    "(", round(control_improvers/n_control*100, 1), "%)\n")

# Cost-effectiveness analysis
# Assume new method costs $100 more per student
cost_per_student <- 100
total_cost <- n_experimental * cost_per_student
cost_per_point <- total_cost / (gain_diff * n_experimental)

cat("\nCost-Effectiveness Analysis:\n")
cat("Additional cost per student: $", cost_per_student, "\n")
cat("Total additional cost: $", total_cost, "\n")
cat("Cost per point of improvement: $", round(cost_per_point, 2), "\n")
```

### Example 3: Quality Control

Quality control applications use two-sample tests to compare production processes, machine performance, and product specifications.

**Scenario:** A manufacturing plant wants to compare the output quality of two production machines to determine if they produce significantly different results.

```r
# Simulate quality control data
set.seed(123)
n_machine1 <- 20
n_machine2 <- 20

# Generate production data
machine1_output <- rnorm(n_machine1, mean = 100, sd = 5)
machine2_output <- rnorm(n_machine2, mean = 98, sd = 6)

# Comprehensive quality control analysis
cat("=== QUALITY CONTROL ANALYSIS ===\n")
cat("Machine 1 n:", n_machine1, "\n")
cat("Machine 2 n:", n_machine2, "\n")
cat("Machine 1 mean:", round(mean(machine1_output), 2), "\n")
cat("Machine 2 mean:", round(mean(machine2_output), 2), "\n")
cat("Machine 1 SD:", round(sd(machine1_output), 2), "\n")
cat("Machine 2 SD:", round(sd(machine2_output), 2), "\n")

# Specification limits (assume 95-105 is acceptable)
spec_lower <- 95
spec_upper <- 105

cat("\nSpecification Analysis:\n")
cat("Specification limits:", spec_lower, "-", spec_upper, "\n")

# Calculate process capability
machine1_capable <- sum(machine1_output >= spec_lower & machine1_output <= spec_upper)
machine2_capable <- sum(machine2_output >= spec_lower & machine2_output <= spec_upper)

cat("Machine 1 within specs:", machine1_capable, "/", n_machine1, 
    "(", round(machine1_capable/n_machine1*100, 1), "%)\n")
cat("Machine 2 within specs:", machine2_capable, "/", n_machine2, 
    "(", round(machine2_capable/n_machine2*100, 1), "%)\n")

# Perform quality control test
quality_test <- t.test(machine1_output, machine2_output)
print(quality_test)

# Nonparametric alternative
quality_wilcox <- wilcox.test(machine1_output, machine2_output)
print(quality_wilcox)

# Effect size
quality_effect <- calculate_cohens_d_independent(machine1_output, machine2_output)

cat("\nQuality Control Results:\n")
cat("Mean difference (Machine 1 - Machine 2):", round(mean(machine1_output) - mean(machine2_output), 2), "\n")
cat("Effect size (Cohen's d):", round(quality_effect$cohens_d, 3), "\n")
cat("Effect interpretation:", interpret_effect_size(quality_effect$cohens_d), "\n")

# Compare parametric and nonparametric results
cat("\nTest Comparison:\n")
cat("t-test p-value:", round(quality_test$p.value, 4), "\n")
cat("Wilcoxon p-value:", round(quality_wilcox$p.value, 4), "\n")

# Process capability indices
machine1_cpk <- min((mean(machine1_output) - spec_lower) / (3 * sd(machine1_output)),
                    (spec_upper - mean(machine1_output)) / (3 * sd(machine1_output)))
machine2_cpk <- min((mean(machine2_output) - spec_lower) / (3 * sd(machine2_output)),
                    (spec_upper - mean(machine2_output)) / (3 * sd(machine2_output)))

cat("\nProcess Capability Analysis:\n")
cat("Machine 1 Cpk:", round(machine1_cpk, 3), "\n")
cat("Machine 2 Cpk:", round(machine2_cpk, 3), "\n")

# Cpk interpretation
interpret_cpk <- function(cpk) {
  if (cpk >= 1.33) {
    return("Excellent capability")
  } else if (cpk >= 1.0) {
    return("Adequate capability")
  } else if (cpk >= 0.67) {
    return("Marginal capability")
  } else {
    return("Poor capability")
  }
}

cat("Machine 1 capability:", interpret_cpk(machine1_cpk), "\n")
cat("Machine 2 capability:", interpret_cpk(machine2_cpk), "\n")

# Quality control interpretation
alpha <- 0.05
mean_diff <- mean(machine1_output) - mean(machine2_output)

if (quality_test$p.value < alpha) {
  cat("\nQUALITY CONTROL CONCLUSION:\n")
  cat("✓ Machines produce significantly different outputs (p <", alpha, ")\n")
  cat("✓ Mean difference:", round(mean_diff, 2), "units\n")
  
  # Practical significance
  if (abs(mean_diff) > 2) {
    cat("⚠️  Difference exceeds practical tolerance (2 units)\n")
    cat("⚠️  Consider machine calibration or maintenance\n")
  } else {
    cat("✓ Difference within practical tolerance\n")
  }
  
  # Which machine is better
  if (mean_diff > 0) {
    cat("✓ Machine 1 produces higher quality output\n")
  } else {
    cat("✓ Machine 2 produces higher quality output\n")
  }
} else {
  cat("\nQUALITY CONTROL CONCLUSION:\n")
  cat("✗ No significant difference between machines (p >=", alpha, ")\n")
  cat("✓ Both machines perform similarly\n")
}

# Economic analysis
# Assume cost of poor quality is $10 per unit outside specs
cost_per_defect <- 10
machine1_defects <- n_machine1 - machine1_capable
machine2_defects <- n_machine2 - machine2_capable

machine1_cost <- machine1_defects * cost_per_defect
machine2_cost <- machine2_defects * cost_per_defect

cat("\nEconomic Analysis:\n")
cat("Machine 1 defect cost: $", machine1_cost, "\n")
cat("Machine 2 defect cost: $", machine2_cost, "\n")
cat("Cost difference: $", abs(machine1_cost - machine2_cost), "\n")

# Recommendations
cat("\nRECOMMENDATIONS:\n")
if (quality_test$p.value < alpha) {
  if (machine1_cpk > machine2_cpk) {
    cat("✓ Prefer Machine 1 for production\n")
  } else {
    cat("✓ Prefer Machine 2 for production\n")
  }
} else {
  cat("✓ Both machines are equivalent\n")
  cat("✓ Choose based on other factors (speed, cost, availability)\n")
}
```

## Advanced Topics

Advanced techniques extend the basic two-sample tests to handle complex scenarios, provide robust alternatives, and offer deeper insights into the data.

### Bootstrap Confidence Intervals

Bootstrap methods provide nonparametric confidence intervals that don't rely on distributional assumptions. They're particularly useful when data doesn't meet normality assumptions or when sample sizes are small.

**Mathematical Foundation:**

The bootstrap estimates the sampling distribution by resampling with replacement from the observed data:

```math
\hat{F}_n(x) = \frac{1}{n} \sum_{i=1}^n I(X_i \leq x)
```

**Bootstrap Confidence Interval:**
```math
CI_{1-\alpha} = [\hat{\theta}_{\alpha/2}^*, \hat{\theta}_{1-\alpha/2}^*]
```

where $\hat{\theta}_\alpha^*$ is the $\alpha$-th percentile of bootstrap estimates.

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

# Comprehensive bootstrap analysis
cat("=== BOOTSTRAP ANALYSIS ===\n")
cat("Original sample sizes:", length(automatic_mpg), "and", length(manual_mpg), "\n")
cat("Bootstrap replications: 1000\n")

# Bootstrap confidence interval
boot_results <- boot(boot_data, boot_mean_diff, R = 1000)
boot_ci_perc <- boot.ci(boot_results, type = "perc")
boot_ci_bca <- boot.ci(boot_results, type = "bca")

cat("\nBootstrap Results:\n")
cat("Original mean difference:", round(boot_results$t0, 3), "\n")
cat("Bootstrap mean:", round(mean(boot_results$t), 3), "\n")
cat("Bootstrap SE:", round(sd(boot_results$t), 3), "\n")

cat("\nBootstrap Confidence Intervals:\n")
cat("Percentile 95% CI:", round(boot_ci_perc$percent[4:5], 3), "\n")
cat("BCa 95% CI:", round(boot_ci_bca$bca[4:5], 3), "\n")

# Compare with parametric CI
t_ci <- t.test(automatic_mpg, manual_mpg)$conf.int
cat("t-test 95% CI:", round(t_ci, 3), "\n")

# Bootstrap bias and acceleration
bias <- mean(boot_results$t) - boot_results$t0
cat("\nBootstrap Diagnostics:\n")
cat("Bias:", round(bias, 4), "\n")
cat("Bias-corrected estimate:", round(boot_results$t0 - bias, 3), "\n")

# Bootstrap distribution analysis
cat("\nBootstrap Distribution:\n")
cat("2.5th percentile:", round(quantile(boot_results$t, 0.025), 3), "\n")
cat("50th percentile (median):", round(quantile(boot_results$t, 0.5), 3), "\n")
cat("97.5th percentile:", round(quantile(boot_results$t, 0.975), 3), "\n")

# Compare CI widths
t_ci_width <- t_ci[2] - t_ci[1]
boot_ci_width <- boot_ci_perc$percent[5] - boot_ci_perc$percent[4]

cat("\nConfidence Interval Comparison:\n")
cat("t-test CI width:", round(t_ci_width, 3), "\n")
cat("Bootstrap CI width:", round(boot_ci_width, 3), "\n")
cat("Width ratio (Bootstrap/t-test):", round(boot_ci_width/t_ci_width, 3), "\n")

# Bootstrap histogram
hist(boot_results$t, main = "Bootstrap Distribution of Mean Difference",
     xlab = "Mean Difference", col = "lightblue", border = "white")
abline(v = boot_results$t0, col = "red", lwd = 2)
abline(v = boot_ci_perc$percent[4:5], col = "blue", lty = 2)
```

### Robust Two-Sample Tests

Robust methods provide alternatives to traditional t-tests that are less sensitive to outliers and violations of normality assumptions. These methods use trimmed means, winsorized data, or other robust estimators.

**Mathematical Foundation:**

**Yuen's t-test (Trimmed Means):**
```math
t_Y = \frac{\bar{X}_{t1} - \bar{X}_{t2}}{\sqrt{\frac{s_{w1}^2}{h_1} + \frac{s_{w2}^2}{h_2}}}
```

where:
- $\bar{X}_{ti}$ = trimmed mean for group $i$
- $s_{wi}^2$ = winsorized variance for group $i$
- $h_i = n_i - 2g_i$ = effective sample size after trimming

**Winsorized Variance:**
```math
s_w^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X}_t)^2
```

**M-estimators:**
```math
\psi(x) = \begin{cases}
x & \text{if } |x| \leq c \\
c \cdot \text{sign}(x) & \text{if } |x| > c
\end{cases}
```

```r
# Comprehensive robust testing function
robust_two_sample_test <- function(group1, group2, group_names = c("Group 1", "Group 2")) {
  cat("=== ROBUST TWO-SAMPLE ANALYSIS ===\n")
  
  # Basic statistics
  n1 <- length(group1)
  n2 <- length(group2)
  
  cat("Sample sizes:", n1, "and", n2, "\n")
  cat("Original means:", round(mean(group1), 3), "and", round(mean(group2), 3), "\n")
  cat("Original SDs:", round(sd(group1), 3), "and", round(sd(group2), 3), "\n")
  
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
      trimmed_means = c(mean1, mean2),
      trimmed_sds = c(sqrt(var1), sqrt(var2)),
      trim_proportion = trim
    ))
  }
  
  # Apply Yuen's test with different trim levels
  yuen_10 <- yuen_test(group1, group2, trim = 0.1)
  yuen_20 <- yuen_test(group1, group2, trim = 0.2)
  
  cat("\nYuen's t-test Results:\n")
  cat("10% trimmed means:", round(yuen_10$trimmed_means, 3), "\n")
  cat("20% trimmed means:", round(yuen_20$trimmed_means, 3), "\n")
  
  cat("\n10% Trim Results:\n")
  cat("t-statistic:", round(yuen_10$t_statistic, 3), "\n")
  cat("p-value:", round(yuen_10$p_value, 4), "\n")
  cat("df:", round(yuen_10$df, 1), "\n")
  
  cat("\n20% Trim Results:\n")
  cat("t-statistic:", round(yuen_20$t_statistic, 3), "\n")
  cat("p-value:", round(yuen_20$p_value, 4), "\n")
  cat("df:", round(yuen_20$df, 1), "\n")
  
  # Compare with standard t-test
  standard_t <- t.test(group1, group2)
  
  cat("\nComparison with Standard t-test:\n")
  cat("Standard t-test p-value:", round(standard_t$p.value, 4), "\n")
  cat("Yuen 10% p-value:", round(yuen_10$p_value, 4), "\n")
  cat("Yuen 20% p-value:", round(yuen_20$p_value, 4), "\n")
  
  # Outlier detection
  outliers1 <- sum(abs(scale(group1)) > 2)
  outliers2 <- sum(abs(scale(group2)) > 2)
  
  cat("\nOutlier Analysis:\n")
  cat(group_names[1], "outliers (|z| > 2):", outliers1, "\n")
  cat(group_names[2], "outliers (|z| > 2):", outliers2, "\n")
  
  # Robust effect size
  robust_effect_10 <- (yuen_10$trimmed_means[1] - yuen_10$trimmed_means[2]) / 
                      sqrt((yuen_10$trimmed_sds[1]^2 + yuen_10$trimmed_sds[2]^2) / 2)
  
  cat("\nRobust Effect Size (10% trim):", round(robust_effect_10, 3), "\n")
  cat("Effect interpretation:", interpret_effect_size(robust_effect_10), "\n")
  
  # Recommendations
  cat("\nRobust Analysis Recommendations:\n")
  if (outliers1 > 0 || outliers2 > 0) {
    cat("⚠️  Outliers detected - robust methods recommended\n")
    if (abs(yuen_10$p_value - standard_t$p.value) > 0.01) {
      cat("⚠️  Results differ from standard t-test\n")
      cat("✓ Use robust methods for inference\n")
    } else {
      cat("✓ Results consistent with standard t-test\n")
    }
  } else {
    cat("✓ No outliers detected\n")
    cat("✓ Standard t-test is appropriate\n")
  }
  
  return(list(
    yuen_10 = yuen_10,
    yuen_20 = yuen_20,
    standard_t = standard_t,
    outliers1 = outliers1,
    outliers2 = outliers2,
    robust_effect = robust_effect_10
  ))
}

# Apply robust analysis
robust_results <- robust_two_sample_test(automatic_mpg, manual_mpg, 
                                        c("Automatic", "Manual"))
```

## Best Practices

Following best practices ensures valid statistical inference and meaningful interpretation of two-sample test results. These guidelines help avoid common pitfalls and improve the quality of statistical analysis.

### Test Selection Guidelines

Proper test selection is crucial for valid statistical inference. The choice depends on data characteristics, sample sizes, and research design.

**Decision Tree for Two-Sample Tests:**

1. **Independent vs Paired:** Are observations related?
2. **Sample Size:** Large ($n \geq 30$) vs Small ($n < 30$)
3. **Normality:** Check distribution assumptions
4. **Variance Equality:** For independent samples
5. **Outliers:** Assess influence of extreme values

```r
# Comprehensive test selection function
choose_two_sample_test <- function(group1, group2, paired = FALSE, alpha = 0.05) {
  cat("=== COMPREHENSIVE TWO-SAMPLE TEST SELECTION ===\n")
  
  n1 <- length(group1)
  n2 <- length(group2)
  
  cat("Sample sizes:", n1, "and", n2, "\n")
  cat("Design:", ifelse(paired, "Paired samples", "Independent samples"), "\n\n")
  
  if (paired) {
    cat("PAIRED SAMPLES ANALYSIS:\n")
    
    # Check normality of differences
    differences <- group1 - group2
    shapiro_diff <- shapiro.test(differences)
    cat("Normality of differences p-value:", round(shapiro_diff$p.value, 4), "\n")
    
    # Sample size considerations
    if (n1 >= 30) {
      cat("✓ Large sample size: Central Limit Theorem applies\n")
      cat("✓ Parametric tests are robust\n")
    } else {
      cat("⚠️  Small sample size: Check normality carefully\n")
    }
    
    # Test recommendation
    if (shapiro_diff$p.value >= alpha || n1 >= 30) {
      cat("RECOMMENDATION: Paired t-test\n")
      cat("REASON: Normal differences or large sample size\n")
    } else {
      cat("RECOMMENDATION: Wilcoxon signed-rank test\n")
      cat("REASON: Non-normal differences in small sample\n")
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
    
    # Sample size considerations
    if (n1 >= 30 && n2 >= 30) {
      cat("✓ Large sample sizes: Central Limit Theorem applies\n")
      cat("✓ Parametric tests are robust to moderate violations\n")
    } else if (n1 >= 15 && n2 >= 15) {
      cat("⚠️  Moderate sample sizes: Check assumptions carefully\n")
    } else {
      cat("⚠️  Small sample sizes: Nonparametric tests recommended\n")
    }
    
    # Test recommendation logic
    cat("\nTEST SELECTION LOGIC:\n")
    
    # Normality assessment
    if (shapiro1$p.value >= alpha && shapiro2$p.value >= alpha) {
      cat("✓ Both groups appear normally distributed\n")
      normality_ok <- TRUE
    } else if (n1 >= 30 && n2 >= 30) {
      cat("✓ Large sample sizes compensate for non-normality\n")
      normality_ok <- TRUE
    } else {
      cat("✗ Non-normal distributions in small samples\n")
      normality_ok <- FALSE
    }
    
    # Variance assessment
    if (var_test$p.value >= alpha) {
      cat("✓ Variances appear equal\n")
      variance_ok <- TRUE
    } else {
      cat("✗ Variances are significantly different\n")
      variance_ok <- FALSE
    }
    
    # Final recommendation
    cat("\nFINAL RECOMMENDATION:\n")
    if (normality_ok) {
      if (variance_ok) {
        cat("✓ Standard t-test (equal variances)\n")
        cat("REASON: Normal distributions and equal variances\n")
      } else {
        cat("✓ Welch's t-test (unequal variances)\n")
        cat("REASON: Normal distributions but unequal variances\n")
      }
    } else {
      cat("✓ Mann-Whitney U test\n")
      cat("REASON: Non-normal distributions or small samples\n")
    }
  }
  
  # Effect size calculation
  if (paired) {
    effect_size <- abs(mean(differences)) / sd(differences)
  } else {
    effect_size <- calculate_cohens_d_independent(group1, group2)$cohens_d
  }
  
  cat("\nEFFECT SIZE ANALYSIS:\n")
  cat("Cohen's d:", round(effect_size, 3), "\n")
  cat("Interpretation:", interpret_effect_size(effect_size), "\n")
  
  # Power considerations
  if (paired) {
    power_est <- pwr.t.test(n = n1, d = effect_size, sig.level = alpha, type = "paired")$power
  } else {
    power_est <- pwr.t2n.test(n1 = n1, n2 = n2, d = effect_size, sig.level = alpha)$power
  }
  
  cat("Estimated power:", round(power_est, 3), "\n")
  
  if (power_est < 0.8) {
    cat("⚠️  Power below 80% - consider larger sample size\n")
  } else {
    cat("✓ Adequate power for detecting effects\n")
  }
  
  return(list(
    paired = paired,
    n1 = n1,
    n2 = n2,
    effect_size = effect_size,
    power = power_est,
    normality_ok = ifelse(paired, shapiro_diff$p.value >= alpha, 
                         shapiro1$p.value >= alpha && shapiro2$p.value >= alpha),
    variance_ok = ifelse(paired, NA, var_test$p.value >= alpha)
  ))
}

# Apply comprehensive test selection
test_selection <- choose_two_sample_test(automatic_mpg, manual_mpg, paired = FALSE)
```

### Reporting Guidelines

Proper reporting of two-sample test results is essential for transparency, reproducibility, and scientific communication. Following standardized reporting guidelines ensures that results are clear, complete, and interpretable.

**Essential Elements for Reporting:**

1. **Study Design:** Independent vs paired samples
2. **Descriptive Statistics:** Means, standard deviations, sample sizes
3. **Test Results:** Test statistic, degrees of freedom, p-value
4. **Effect Size:** Cohen's d and interpretation
5. **Confidence Intervals:** Precision of estimates
6. **Assumption Checks:** Normality, homogeneity of variance
7. **Practical Significance:** Clinical or practical relevance

```r
# Comprehensive reporting function for two-sample tests
generate_two_sample_report <- function(test_result, group1, group2, test_type = "t-test", 
                                      paired = FALSE, alpha = 0.05) {
  cat("=== COMPREHENSIVE TWO-SAMPLE TEST REPORT ===\n\n")
  
  # Basic information
  n1 <- length(group1)
  n2 <- length(group2)
  mean1 <- mean(group1)
  mean2 <- mean(group2)
  sd1 <- sd(group1)
  sd2 <- sd(group2)
  
  cat("STUDY DESIGN:\n")
  cat("Test type:", test_type, "\n")
  cat("Design:", ifelse(paired, "Paired samples", "Independent samples"), "\n")
  cat("Group 1 sample size:", n1, "\n")
  cat("Group 2 sample size:", n2, "\n")
  cat("Total sample size:", n1 + n2, "\n")
  cat("Significance level (α):", alpha, "\n\n")
  
  cat("DESCRIPTIVE STATISTICS:\n")
  cat("Group 1: M =", round(mean1, 2), ", SD =", round(sd1, 2), "\n")
  cat("Group 2: M =", round(mean2, 2), ", SD =", round(sd2, 2), "\n")
  
  if (paired) {
    differences <- group1 - group2
    mean_diff <- mean(differences)
    sd_diff <- sd(differences)
    cat("Differences: M =", round(mean_diff, 2), ", SD =", round(sd_diff, 2), "\n")
  } else {
    mean_diff <- mean1 - mean2
    cat("Mean difference (Group 1 - Group 2):", round(mean_diff, 2), "\n")
  }
  cat("\n")
  
  # Test results
  if (test_type == "t-test") {
    cat("T-TEST RESULTS:\n")
    cat("t(", round(test_result$parameter, 1), ") =", round(test_result$statistic, 3), "\n")
    cat("p-value:", round(test_result$p.value, 4), "\n")
    cat("95% CI:", round(test_result$conf.int, 3), "\n")
    
    # Effect size
    if (paired) {
      effect_size <- abs(mean_diff) / sd_diff
    } else {
      effect_size <- calculate_cohens_d_independent(group1, group2)$cohens_d
    }
    
    cat("Cohen's d =", round(effect_size, 3), "\n")
    cat("Effect size interpretation:", interpret_effect_size(effect_size), "\n\n")
    
  } else if (test_type == "wilcoxon") {
    cat("WILCOXON TEST RESULTS:\n")
    cat("W =", test_result$statistic, "\n")
    cat("p-value:", round(test_result$p.value, 4), "\n")
    
    # Effect size for nonparametric test
    wilcox_effect <- abs(qnorm(test_result$p.value / 2)) / sqrt(n1 + n2)
    cat("Effect size (r) =", round(wilcox_effect, 3), "\n")
    cat("Effect size interpretation:", interpret_wilcox_effect(wilcox_effect), "\n\n")
  }
  
  # Assumption checks
  cat("ASSUMPTION CHECKS:\n")
  
  if (paired) {
    differences <- group1 - group2
    shapiro_diff <- shapiro.test(differences)
    cat("Normality of differences: Shapiro-Wilk p =", round(shapiro_diff$p.value, 4), "\n")
    if (shapiro_diff$p.value < alpha) {
      cat("⚠️  Differences are not normally distributed\n")
    } else {
      cat("✓ Differences appear normally distributed\n")
    }
  } else {
    shapiro1 <- shapiro.test(group1)
    shapiro2 <- shapiro.test(group2)
    cat("Group 1 normality: Shapiro-Wilk p =", round(shapiro1$p.value, 4), "\n")
    cat("Group 2 normality: Shapiro-Wilk p =", round(shapiro2$p.value, 4), "\n")
    
    var_test <- var.test(group1, group2)
    cat("Homogeneity of variance: F-test p =", round(var_test$p.value, 4), "\n")
    
    if (shapiro1$p.value < alpha || shapiro2$p.value < alpha) {
      cat("⚠️  At least one group is not normally distributed\n")
    } else {
      cat("✓ Both groups appear normally distributed\n")
    }
    
    if (var_test$p.value < alpha) {
      cat("⚠️  Variances are significantly different\n")
    } else {
      cat("✓ Variances appear equal\n")
    }
  }
  cat("\n")
  
  # Power analysis
  if (test_type == "t-test") {
    if (paired) {
      power_est <- pwr.t.test(n = n1, d = effect_size, sig.level = alpha, type = "paired")$power
    } else {
      power_est <- pwr.t2n.test(n1 = n1, n2 = n2, d = effect_size, sig.level = alpha)$power
    }
    cat("POWER ANALYSIS:\n")
    cat("Estimated power =", round(power_est, 3), "\n")
    if (power_est < 0.8) {
      cat("⚠️  Power below 80% - results should be interpreted cautiously\n")
    } else {
      cat("✓ Adequate power for detecting effects\n")
    }
    cat("\n")
  }
  
  # Statistical conclusion
  cat("STATISTICAL CONCLUSION:\n")
  if (test_result$p.value < alpha) {
    cat("✓ Reject the null hypothesis (p <", alpha, ")\n")
    cat("✓ There is significant evidence of a difference between groups\n")
  } else {
    cat("✗ Fail to reject the null hypothesis (p >=", alpha, ")\n")
    cat("✗ There is insufficient evidence of a difference between groups\n")
  }
  cat("\n")
  
  # Practical significance
  cat("PRACTICAL SIGNIFICANCE:\n")
  if (test_type == "t-test") {
    if (abs(effect_size) >= 0.8) {
      cat("✓ Large practical effect\n")
    } else if (abs(effect_size) >= 0.5) {
      cat("✓ Medium practical effect\n")
    } else if (abs(effect_size) >= 0.2) {
      cat("✓ Small practical effect\n")
    } else {
      cat("⚠️  Very small practical effect\n")
    }
  }
  
  # APA style reporting
  cat("\nAPA STYLE REPORTING:\n")
  if (test_type == "t-test") {
    if (test_result$p.value < 0.001) {
      p_report <- "p < .001"
    } else {
      p_report <- paste("p =", round(test_result$p.value, 3))
    }
    
    cat("A", ifelse(paired, "paired", "independent"), "samples t-test was conducted to compare", 
        ifelse(paired, "the two conditions", "the two groups"), ".\n")
    cat("There was", ifelse(test_result$p.value < alpha, "", "no"), "significant difference between groups, ",
        "t(", round(test_result$parameter, 1), ") =", round(test_result$statistic, 3), ",", p_report, 
        ", d =", round(effect_size, 3), ".\n")
  }
  
  return(list(
    test_type = test_type,
    paired = paired,
    n1 = n1,
    n2 = n2,
    mean1 = mean1,
    mean2 = mean2,
    sd1 = sd1,
    sd2 = sd2,
    test_statistic = test_result$statistic,
    p_value = test_result$p.value,
    effect_size = ifelse(test_type == "t-test", effect_size, wilcox_effect),
    significant = test_result$p.value < alpha
  ))
}

# Generate comprehensive report for transmission comparison
transmission_t_test <- t.test(automatic_mpg, manual_mpg)
transmission_report <- generate_two_sample_report(transmission_t_test, automatic_mpg, manual_mpg, 
                                                 "t-test", paired = FALSE)
```

## Exercises

These exercises provide hands-on practice with two-sample tests, helping you develop proficiency in statistical analysis and interpretation.

### Exercise 1: Independent Samples t-Test

**Objective:** Compare the horsepower of cars with different cylinder counts using independent samples t-tests.

**Data:** Use the `mtcars` dataset.

**Tasks:**
1. Create two groups: cars with 4 cylinders vs cars with 8 cylinders
2. Perform descriptive statistics for both groups
3. Check assumptions (normality, homogeneity of variance)
4. Conduct independent samples t-test
5. Calculate and interpret effect size
6. Generate a comprehensive report

**Hints:**
- Use `mtcars$cyl == 4` and `mtcars$cyl == 8` to create groups
- Remember to handle unequal sample sizes
- Consider both pooled and Welch's t-tests

**Expected Learning Outcomes:**
- Understanding when to use pooled vs Welch's t-test
- Interpreting effect sizes in context
- Recognizing the importance of assumption checking

### Exercise 2: Paired Samples Analysis

**Objective:** Create a paired dataset and perform both parametric and nonparametric paired tests.

**Scenario:** Simulate before-and-after data for a weight loss program.

**Tasks:**
1. Generate paired data (before and after weights)
2. Calculate and analyze differences
3. Check normality of differences
4. Perform paired t-test and Wilcoxon signed-rank test
5. Compare results and interpret differences
6. Calculate paired effect size

**Hints:**
- Use `rnorm()` to generate realistic weight data
- Ensure positive correlation between before and after scores
- Consider the power advantage of paired designs

**Expected Learning Outcomes:**
- Understanding the power advantage of paired designs
- Recognizing when paired vs independent tests are appropriate
- Interpreting correlation in paired data

### Exercise 3: Effect Size Analysis

**Objective:** Calculate and interpret effect sizes for various two-sample comparisons.

**Data:** Use multiple variables from the `mtcars` dataset.

**Tasks:**
1. Compare multiple variables between transmission types
2. Calculate Cohen's d for each comparison
3. Create a table of effect sizes and interpretations
4. Identify which comparisons show the largest effects
5. Discuss practical significance vs statistical significance

**Variables to Compare:**
- MPG (miles per gallon)
- HP (horsepower)
- WT (weight)
- QSEC (quarter mile time)

**Hints:**
- Use a loop or apply function for efficiency
- Consider creating a function for effect size calculation
- Think about which effects are most practically meaningful

**Expected Learning Outcomes:**
- Understanding effect size interpretation
- Recognizing the difference between statistical and practical significance
- Developing intuition for meaningful effect sizes

### Exercise 4: Assumption Checking

**Objective:** Perform comprehensive assumption checking for two-sample tests.

**Data:** Create datasets with various characteristics (normal, skewed, with outliers).

**Tasks:**
1. Generate three datasets:
   - Normal distributions with equal variances
   - Normal distributions with unequal variances
   - Non-normal distributions with outliers
2. For each dataset, perform comprehensive assumption checking
3. Recommend appropriate tests based on findings
4. Compare results across different test approaches
5. Discuss the robustness of different methods

**Hints:**
- Use `rnorm()`, `rgamma()`, and `c()` with outliers
- Test normality with multiple methods
- Consider sample size effects on assumption violations

**Expected Learning Outcomes:**
- Understanding when assumption violations matter
- Recognizing the robustness of different tests
- Developing judgment for test selection

### Exercise 5: Power Analysis

**Objective:** Conduct power analysis for two-sample designs.

**Scenario:** Design a study to compare two teaching methods.

**Tasks:**
1. Determine required sample sizes for different effect sizes (0.2, 0.5, 0.8)
2. Calculate power for different sample sizes (20, 50, 100 per group)
3. Create power curves
4. Consider cost-benefit analysis of sample sizes
5. Make recommendations for study design

**Hints:**
- Use the `pwr` package
- Consider practical constraints (time, cost, availability)
- Think about minimum important differences

**Expected Learning Outcomes:**
- Understanding the relationship between effect size, sample size, and power
- Developing intuition for study design
- Making informed decisions about sample size

### Exercise 6: Real-World Application

**Objective:** Apply two-sample tests to a real-world scenario.

**Scenario:** Analyze customer satisfaction data for two different service providers.

**Tasks:**
1. Create realistic customer satisfaction data
2. Perform comprehensive analysis including:
   - Descriptive statistics
   - Assumption checking
   - Appropriate statistical tests
   - Effect size calculation
   - Practical interpretation
3. Write a professional report
4. Make business recommendations

**Hints:**
- Consider the business context
- Think about practical significance
- Include confidence intervals
- Consider multiple outcome measures

**Expected Learning Outcomes:**
- Applying statistical concepts to real problems
- Communicating results to non-statisticians
- Making data-driven recommendations

### Exercise 7: Advanced Topics

**Objective:** Explore advanced two-sample testing methods.

**Tasks:**
1. Implement bootstrap confidence intervals
2. Perform Yuen's t-test for trimmed means
3. Compare results with standard methods
4. Discuss when advanced methods are beneficial
5. Analyze the trade-offs between methods

**Hints:**
- Use the `boot` package for bootstrap analysis
- Consider different trim levels
- Compare computational efficiency

**Expected Learning Outcomes:**
- Understanding when advanced methods are needed
- Recognizing the limitations of standard methods
- Developing expertise in robust statistics

### Solutions and Additional Resources

**For each exercise:**
- Start with small datasets to verify your approach
- Use the functions developed in this chapter
- Check your results with built-in R functions
- Consider multiple approaches to the same problem

**Common Mistakes to Avoid:**
- Using independent tests for paired data
- Ignoring assumption violations in small samples
- Focusing only on p-values without effect sizes
- Not considering practical significance

**Next Steps:**
- Practice with your own datasets
- Explore related topics (ANOVA, regression)
- Learn about multiple comparison corrections
- Study experimental design principles

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