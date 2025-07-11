# Hypothesis Testing

## Overview

Hypothesis testing is a fundamental concept in statistics that allows us to make decisions about populations based on sample data. It provides a framework for testing claims about parameters or relationships in data through a systematic process of statistical inference.

### What is Hypothesis Testing?

Hypothesis testing is a statistical method used to determine whether there is enough evidence in a sample to support a particular claim about a population parameter. It involves making an initial assumption (the null hypothesis) and then determining whether the observed data provides sufficient evidence to reject this assumption.

### The Scientific Method and Hypothesis Testing

Hypothesis testing follows the scientific method:
1. **Observation**: Collect data from a sample
2. **Hypothesis**: Formulate null and alternative hypotheses
3. **Prediction**: Determine what we expect to observe if the null hypothesis is true
4. **Testing**: Calculate a test statistic and compare it to a critical value
5. **Conclusion**: Make a decision based on the evidence

### Key Components of Hypothesis Testing

1. **Null Hypothesis ($H_0$)**: The default assumption about the population parameter
2. **Alternative Hypothesis ($H_1$ or $H_a$)**: The claim we want to support
3. **Test Statistic**: A calculated value that measures the strength of evidence
4. **Significance Level ($\alpha$)**: The probability of rejecting $H_0$ when it's true
5. **P-value**: The probability of observing the test statistic or more extreme values
6. **Decision Rule**: Criteria for rejecting or failing to reject $H_0$

### Mathematical Foundation

The general framework for hypothesis testing involves:

```math
\text{Test Statistic} = \frac{\text{Sample Statistic} - \text{Parameter under } H_0}{\text{Standard Error}}
```

The decision rule is:
- Reject $H_0$ if $|T| > t_{\alpha/2, df}$ (two-tailed test)
- Reject $H_0$ if $T > t_{\alpha, df}$ (one-tailed test, upper)
- Reject $H_0$ if $T < -t_{\alpha, df}$ (one-tailed test, lower)

Where $T$ is the test statistic and $t_{\alpha, df}$ is the critical value from the appropriate distribution.

## Basic Concepts

### Null and Alternative Hypotheses

The foundation of hypothesis testing lies in formulating two competing hypotheses about a population parameter.

#### Mathematical Foundation

**Null Hypothesis ($H_0$)**: The default assumption that there is no effect, no difference, or no relationship. It represents the status quo or the claim we want to test against.

**Alternative Hypothesis ($H_1$ or $H_a$)**: The claim we want to support. It represents the research hypothesis or the effect we're trying to detect.

For a population parameter $\theta$, the hypotheses can be:

**Two-tailed test:**
```math
H_0: \theta = \theta_0 \quad \text{vs.} \quad H_1: \theta \neq \theta_0
```

**One-tailed test (upper):**
```math
H_0: \theta \leq \theta_0 \quad \text{vs.} \quad H_1: \theta > \theta_0
```

**One-tailed test (lower):**
```math
H_0: \theta \geq \theta_0 \quad \text{vs.} \quad H_1: \theta < \theta_0
```

#### Code Implementation

```r
# Example: Testing if a coin is fair
# H0: p = 0.5 (coin is fair)
# H1: p ≠ 0.5 (coin is not fair)

# Simulate coin flips
set.seed(123)
coin_flips <- rbinom(100, 1, 0.6)  # Biased coin
head_count <- sum(coin_flips)
total_flips <- length(coin_flips)
observed_proportion <- head_count / total_flips

cat("=== Coin Fairness Test ===\n")
cat("Null hypothesis (H0): p = 0.5 (coin is fair)\n")
cat("Alternative hypothesis (H1): p ≠ 0.5 (coin is not fair)\n")
cat("Test type: Two-tailed\n\n")

cat("Data Summary:\n")
cat("Total flips:", total_flips, "\n")
cat("Heads observed:", head_count, "\n")
cat("Observed proportion of heads:", round(observed_proportion, 3), "\n")
cat("Expected proportion under H0:", 0.5, "\n")
cat("Difference from expected:", round(observed_proportion - 0.5, 3), "\n")

# Calculate test statistic
expected_heads <- total_flips * 0.5
standard_error <- sqrt(total_flips * 0.5 * 0.5)
z_statistic <- (head_count - expected_heads) / standard_error

cat("\nTest Statistic Calculation:\n")
cat("Expected heads under H0:", expected_heads, "\n")
cat("Standard error:", round(standard_error, 3), "\n")
cat("Z-statistic:", round(z_statistic, 3), "\n")

# Calculate p-value
p_value <- 2 * (1 - pnorm(abs(z_statistic)))
cat("P-value:", round(p_value, 4), "\n")

# Decision
alpha <- 0.05
if (p_value < alpha) {
  cat("Decision: Reject H0 (coin is not fair)\n")
} else {
  cat("Decision: Fail to reject H0 (insufficient evidence)\n")
}
```

### Type I and Type II Errors

Understanding the types of errors that can occur in hypothesis testing is crucial for interpreting results and making informed decisions.

#### Mathematical Foundation

**Type I Error ($\alpha$)**: The probability of rejecting $H_0$ when it's actually true.
```math
\alpha = P(\text{Reject } H_0 | H_0 \text{ is true})
```

**Type II Error ($\beta$)**: The probability of failing to reject $H_0$ when it's actually false.
```math
\beta = P(\text{Fail to reject } H_0 | H_1 \text{ is true})
```

**Power ($1 - \beta$)**: The probability of correctly rejecting $H_0$ when it's false.
```math
\text{Power} = 1 - \beta = P(\text{Reject } H_0 | H_1 \text{ is true})
```

The relationship between these concepts:
```math
\alpha + \text{Power} \leq 1
```

#### Decision Matrix

| Decision | $H_0$ True | $H_0$ False |
|----------|-------------|-------------|
| Reject $H_0$ | Type I Error ($\alpha$) | Correct Decision (Power) |
| Fail to reject $H_0$ | Correct Decision ($1-\alpha$) | Type II Error ($\beta$) |

#### Code Implementation

```r
# Understanding error types
cat("=== Type I and Type II Errors ===\n")
cat("Type I Error (α): Rejecting H0 when it's true\n")
cat("Type II Error (β): Failing to reject H0 when it's false\n")
cat("Power (1-β): Probability of correctly rejecting H0\n\n")

# Example: Setting significance level
alpha <- 0.05
cat("Significance level (α):", alpha, "\n")
cat("This means we're willing to make a Type I error 5% of the time\n\n")

# Demonstrate the trade-off between Type I and Type II errors
cat("=== Error Trade-off Demonstration ===\n")

# Function to calculate power for different effect sizes
calculate_power <- function(effect_size, n, alpha = 0.05) {
  # For a two-sample t-test
  power <- pwr.t.test(d = effect_size, n = n, sig.level = alpha, type = "two.sample")
  return(power$power)
}

# Test different scenarios
effect_sizes <- c(0.2, 0.5, 0.8)  # Small, medium, large
sample_sizes <- c(20, 50, 100)
alpha_levels <- c(0.01, 0.05, 0.10)

cat("Power Analysis for Different Scenarios:\n")
cat("Effect Size | Sample Size | α = 0.01 | α = 0.05 | α = 0.10\n")
cat("-----------|-------------|-----------|-----------|-----------\n")

for (d in effect_sizes) {
  for (n in sample_sizes) {
    cat(sprintf("%.1f        | %d          | %.3f     | %.3f     | %.3f\n", 
                d, n, 
                calculate_power(d, n, 0.01),
                calculate_power(d, n, 0.05),
                calculate_power(d, n, 0.10)))
  }
}

cat("\nKey Insights:\n")
cat("• Lower α (Type I error) → Lower power (higher Type II error)\n")
cat("• Larger effect size → Higher power\n")
cat("• Larger sample size → Higher power\n")
cat("• There's always a trade-off between Type I and Type II errors\n")
```

## One-Sample Tests

One-sample tests are used to compare a sample statistic to a hypothesized population parameter.

### One-Sample t-Test

The one-sample t-test is used to determine whether the mean of a single sample differs from a hypothesized population mean.

#### Mathematical Foundation

For a sample of size $n$ with mean $\bar{x}$ and standard deviation $s$, the test statistic is:

```math
t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}
```

Where $\mu_0$ is the hypothesized population mean under $H_0$.

The test statistic follows a t-distribution with $n-1$ degrees of freedom.

**Hypotheses:**
```math
H_0: \mu = \mu_0 \quad \text{vs.} \quad H_1: \mu \neq \mu_0
```

#### Assumptions

1. **Independence**: Observations are independent
2. **Normality**: The population is normally distributed (or sample size is large)
3. **Random Sampling**: Data comes from a random sample

#### Code Implementation

```r
# Load data
data(mtcars)

# Test if mean MPG is different from 20
mpg_test <- t.test(mtcars$mpg, mu = 20)

cat("=== One-Sample t-Test ===\n")
cat("Hypothesis: H0: μ = 20 vs. H1: μ ≠ 20\n")
cat("Test type: Two-tailed\n\n")

# View results
print(mpg_test)

# Extract components
cat("\nDetailed Results:\n")
cat("Sample mean:", round(mpg_test$estimate, 3), "\n")
cat("Hypothesized mean:", 20, "\n")
cat("Test statistic (t):", round(mpg_test$statistic, 3), "\n")
cat("Degrees of freedom:", mpg_test$parameter, "\n")
cat("P-value:", round(mpg_test$p.value, 4), "\n")
cat("95% Confidence interval:", round(mpg_test$conf.int[1], 3), "to", round(mpg_test$conf.int[2], 3), "\n")

# Decision
alpha <- 0.05
if (mpg_test$p.value < alpha) {
  cat("Decision: Reject H0 (mean MPG differs from 20)\n")
} else {
  cat("Decision: Fail to reject H0 (insufficient evidence)\n")
}

# Effect size calculation
library(effectsize)
cohens_d <- cohens_d(mtcars$mpg, mu = 20)
cat("Effect size (Cohen's d):", round(cohens_d$Cohens_d, 3), "\n")

# Interpret effect size
if (abs(cohens_d$Cohens_d) < 0.2) {
  cat("Effect size: Small\n")
} else if (abs(cohens_d$Cohens_d) < 0.5) {
  cat("Effect size: Medium\n")
} else if (abs(cohens_d$Cohens_d) < 0.8) {
  cat("Effect size: Large\n")
} else {
  cat("Effect size: Very large\n")
}

# Check assumptions
cat("\n=== Assumption Checking ===\n")

# Normality test
shapiro_result <- shapiro.test(mtcars$mpg)
cat("Shapiro-Wilk normality test:\n")
cat("W =", round(shapiro_result$statistic, 3), "\n")
cat("P-value =", round(shapiro_result$p.value, 4), "\n")

if (shapiro_result$p.value < 0.05) {
  cat("Warning: Data may not be normally distributed\n")
} else {
  cat("✓ Data appears to be normally distributed\n")
}

# Sample size check
cat("Sample size:", length(mtcars$mpg), "\n")
if (length(mtcars$mpg) >= 30) {
  cat("✓ Sample size is adequate for t-test (Central Limit Theorem)\n")
} else {
  cat("⚠ Small sample size - normality assumption is important\n")
}
```

### One-Sample Proportion Test

The one-sample proportion test is used to determine whether a sample proportion differs from a hypothesized population proportion.

#### Mathematical Foundation

For a sample of size $n$ with $x$ successes, the sample proportion is $\hat{p} = x/n$.

**Large Sample Approximation (Normal):**
When $np \geq 10$ and $n(1-p) \geq 10$, the test statistic is:

```math
z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}
```

Where $p_0$ is the hypothesized population proportion under $H_0$.

**Exact Test (Binomial):**
For small samples, use the exact binomial test:

```math
P(X \geq x) = \sum_{k=x}^n \binom{n}{k} p_0^k (1-p_0)^{n-k}
```

**Hypotheses:**
```math
H_0: p = p_0 \quad \text{vs.} \quad H_1: p \neq p_0
```

#### Code Implementation

```r
# Test proportion of automatic transmissions
auto_count <- sum(mtcars$am == 0)
total_cars <- nrow(mtcars)
auto_proportion <- auto_count / total_cars

cat("=== One-Sample Proportion Test ===\n")
cat("Hypothesis: H0: p = 0.5 vs. H1: p ≠ 0.5\n")
cat("Test type: Two-tailed\n\n")

cat("Data Summary:\n")
cat("Total cars:", total_cars, "\n")
cat("Automatic transmissions:", auto_count, "\n")
cat("Sample proportion:", round(auto_proportion, 3), "\n")
cat("Hypothesized proportion:", 0.5, "\n")

# Check large sample conditions
np <- total_cars * 0.5
nq <- total_cars * 0.5
cat("\nLarge Sample Conditions:\n")
cat("n*p =", np, "(should be ≥ 10):", ifelse(np >= 10, "✓", "✗"), "\n")
cat("n*(1-p) =", nq, "(should be ≥ 10):", ifelse(nq >= 10, "✓", "✗"), "\n")

# Test if proportion is different from 0.5
prop_test <- prop.test(auto_count, total_cars, p = 0.5)

cat("\nNormal Approximation Results:\n")
print(prop_test)

# Extract components
cat("\nDetailed Results:\n")
cat("Test statistic (z):", round(prop_test$statistic, 3), "\n")
cat("P-value:", round(prop_test$p.value, 4), "\n")
cat("95% Confidence interval:", round(prop_test$conf.int[1], 4), "to", round(prop_test$conf.int[2], 4), "\n")

# Decision
alpha <- 0.05
if (prop_test$p.value < alpha) {
  cat("Decision: Reject H0 (proportion differs from 0.5)\n")
} else {
  cat("Decision: Fail to reject H0 (insufficient evidence)\n")
}

# Exact binomial test for comparison
binom_test <- binom.test(auto_count, total_cars, p = 0.5)
cat("\nExact Binomial Test Results:\n")
cat("P-value:", round(binom_test$p.value, 4), "\n")
cat("95% Confidence interval:", round(binom_test$conf.int[1], 4), "to", round(binom_test$conf.int[2], 4), "\n")

# Compare methods
cat("\nMethod Comparison:\n")
cat("Normal approximation p-value:", round(prop_test$p.value, 4), "\n")
cat("Exact binomial p-value:", round(binom_test$p.value, 4), "\n")
cat("Difference:", round(abs(prop_test$p.value - binom_test$p.value), 4), "\n")

if (abs(prop_test$p.value - binom_test$p.value) < 0.01) {
  cat("✓ Methods give similar results (large sample conditions met)\n")
} else {
  cat("⚠ Methods differ - use exact test for small samples\n")
}

# Effect size (difference from hypothesized proportion)
effect_size <- abs(auto_proportion - 0.5)
cat("\nEffect Size Analysis:\n")
cat("Absolute difference from 0.5:", round(effect_size, 3), "\n")

if (effect_size < 0.1) {
  cat("Effect size: Small\n")
} else if (effect_size < 0.2) {
  cat("Effect size: Medium\n")
} else {
  cat("Effect size: Large\n")
}
```

### One-Sample Variance Test

The one-sample variance test (chi-square test for variance) is used to determine whether a sample variance differs from a hypothesized population variance.

#### Mathematical Foundation

For a sample of size $n$ with variance $s^2$, the test statistic is:

```math
\chi^2 = \frac{(n-1)s^2}{\sigma_0^2}
```

Where $\sigma_0^2$ is the hypothesized population variance under $H_0$.

The test statistic follows a chi-square distribution with $n-1$ degrees of freedom.

**Hypotheses:**
```math
H_0: \sigma^2 = \sigma_0^2 \quad \text{vs.} \quad H_1: \sigma^2 \neq \sigma_0^2
```

**Assumptions:**
1. **Normality**: The population is normally distributed
2. **Independence**: Observations are independent
3. **Random Sampling**: Data comes from a random sample

#### Code Implementation

```r
# Test if variance of MPG is different from 25
mpg_variance <- var(mtcars$mpg)
n <- length(mtcars$mpg)
hypothesized_variance <- 25

cat("=== One-Sample Variance Test ===\n")
cat("Hypothesis: H0: σ² = 25 vs. H1: σ² ≠ 25\n")
cat("Test type: Two-tailed\n\n")

cat("Data Summary:\n")
cat("Sample size:", n, "\n")
cat("Sample variance:", round(mpg_variance, 3), "\n")
cat("Hypothesized variance:", hypothesized_variance, "\n")
cat("Sample standard deviation:", round(sd(mtcars$mpg), 3), "\n")
cat("Hypothesized standard deviation:", sqrt(hypothesized_variance), "\n")

# Calculate test statistic
test_statistic <- (n - 1) * mpg_variance / hypothesized_variance
df <- n - 1

cat("\nTest Statistic Calculation:\n")
cat("Degrees of freedom:", df, "\n")
cat("Chi-square statistic:", round(test_statistic, 3), "\n")

# Calculate p-value (two-tailed)
p_value_lower <- pchisq(test_statistic, df)
p_value_upper <- 1 - pchisq(test_statistic, df)
p_value <- 2 * min(p_value_lower, p_value_upper)

cat("P-value (two-tailed):", round(p_value, 4), "\n")

# Decision
alpha <- 0.05
if (p_value < alpha) {
  cat("Decision: Reject H0 (variance differs from 25)\n")
} else {
  cat("Decision: Fail to reject H0 (insufficient evidence)\n")
}

# Calculate confidence interval for variance
chi_lower <- qchisq(0.025, df)
chi_upper <- qchisq(0.975, df)
ci_lower_var <- (df * mpg_variance) / chi_upper
ci_upper_var <- (df * mpg_variance) / chi_lower

cat("\nConfidence Interval for Variance:\n")
cat("95% CI for variance:", round(ci_lower_var, 3), "to", round(ci_upper_var, 3), "\n")

# Check if hypothesized variance is in confidence interval
if (hypothesized_variance >= ci_lower_var && hypothesized_variance <= ci_upper_var) {
  cat("✓ Hypothesized variance is within the confidence interval\n")
} else {
  cat("✗ Hypothesized variance is outside the confidence interval\n")
}

# Effect size (ratio of variances)
variance_ratio <- mpg_variance / hypothesized_variance
cat("\nEffect Size Analysis:\n")
cat("Variance ratio (sample/hypothesized):", round(variance_ratio, 3), "\n")

if (abs(variance_ratio - 1) < 0.2) {
  cat("Effect size: Small\n")
} else if (abs(variance_ratio - 1) < 0.5) {
  cat("Effect size: Medium\n")
} else {
  cat("Effect size: Large\n")
}

# Check normality assumption
cat("\n=== Assumption Checking ===\n")
shapiro_result <- shapiro.test(mtcars$mpg)
cat("Shapiro-Wilk normality test:\n")
cat("W =", round(shapiro_result$statistic, 3), "\n")
cat("P-value =", round(shapiro_result$p.value, 4), "\n")

if (shapiro_result$p.value < 0.05) {
  cat("⚠ Warning: Data may not be normally distributed\n")
  cat("   Chi-square test for variance is sensitive to non-normality\n")
  cat("   Consider using nonparametric alternatives\n")
} else {
  cat("✓ Data appears to be normally distributed\n")
}

# Alternative: Levene's test for homogeneity of variance
cat("\nAlternative: Levene's Test\n")
# This would require multiple groups, but we can demonstrate the concept
cat("Note: Levene's test is typically used for comparing variances across groups\n")
```

## Two-Sample Tests

Two-sample tests are used to compare parameters between two independent or paired groups.

### Independent t-Test

The independent t-test (two-sample t-test) is used to compare the means of two independent groups.

#### Mathematical Foundation

For two independent samples with means $\bar{x}_1$ and $\bar{x}_2$, the test statistic is:

**Equal Variances (Pooled):**
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
```

Where the pooled standard deviation is:
```math
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
```

**Unequal Variances (Welch's):**
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
```

With degrees of freedom (Welch's approximation):
```math
df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
```

**Hypotheses:**
```math
H_0: \mu_1 = \mu_2 \quad \text{vs.} \quad H_1: \mu_1 \neq \mu_2
```

#### Assumptions

1. **Independence**: Observations are independent within and between groups
2. **Normality**: Both populations are normally distributed (or sample sizes are large)
3. **Equal Variances**: Population variances are equal (for pooled t-test)
4. **Random Sampling**: Data comes from random samples

#### Code Implementation

```r
# Compare MPG between automatic and manual transmissions
auto_mpg <- mtcars$mpg[mtcars$am == 0]
manual_mpg <- mtcars$mpg[mtcars$am == 1]

cat("=== Independent t-Test ===\n")
cat("Hypothesis: H0: μ_auto = μ_manual vs. H1: μ_auto ≠ μ_manual\n")
cat("Test type: Two-tailed\n\n")

cat("Data Summary:\n")
cat("Automatic transmissions: n =", length(auto_mpg), ", mean =", round(mean(auto_mpg), 3), 
    ", SD =", round(sd(auto_mpg), 3), "\n")
cat("Manual transmissions: n =", length(manual_mpg), ", mean =", round(mean(manual_mpg), 3), 
    ", SD =", round(sd(manual_mpg), 3), "\n")

# Check assumptions
cat("\n=== Assumption Checking ===\n")

# Normality tests
shapiro_auto <- shapiro.test(auto_mpg)
shapiro_manual <- shapiro.test(manual_mpg)

cat("Normality Tests:\n")
cat("Automatic - W =", round(shapiro_auto$statistic, 3), ", p =", round(shapiro_auto$p.value, 4), "\n")
cat("Manual - W =", round(shapiro_manual$statistic, 3), ", p =", round(shapiro_manual$p.value, 4), "\n")

# Variance ratio test
variance_ratio <- var(auto_mpg) / var(manual_mpg)
cat("Variance ratio:", round(variance_ratio, 3), "\n")

if (variance_ratio > 2 || variance_ratio < 0.5) {
  cat("⚠ Variance ratio suggests unequal variances - use Welch's t-test\n")
} else {
  cat("✓ Variances appear approximately equal\n")
}

# F-test for equality of variances
var_test <- var.test(auto_mpg, manual_mpg)
cat("F-test for equality of variances: F =", round(var_test$statistic, 3), 
    ", p =", round(var_test$p.value, 4), "\n")

if (var_test$p.value < 0.05) {
  cat("✓ Variances are significantly different - use Welch's t-test\n")
} else {
  cat("✓ Variances are not significantly different - pooled t-test is appropriate\n")
}

# Perform t-tests
cat("\n=== Test Results ===\n")

# Pooled t-test (equal variances)
pooled_test <- t.test(auto_mpg, manual_mpg, var.equal = TRUE)
cat("Pooled t-test (equal variances):\n")
cat("t =", round(pooled_test$statistic, 3), ", df =", pooled_test$parameter, 
    ", p =", round(pooled_test$p.value, 4), "\n")

# Welch's t-test (unequal variances)
welch_test <- t.test(auto_mpg, manual_mpg, var.equal = FALSE)
cat("Welch's t-test (unequal variances):\n")
cat("t =", round(welch_test$statistic, 3), ", df =", round(welch_test$parameter, 1), 
    ", p =", round(welch_test$p.value, 4), "\n")

# Decision
alpha <- 0.05
if (welch_test$p.value < alpha) {
  cat("Decision: Reject H0 (means differ significantly)\n")
} else {
  cat("Decision: Fail to reject H0 (insufficient evidence)\n")
}

# Effect size
library(effectsize)
cohens_d <- cohens_d(auto_mpg, manual_mpg)
cat("\nEffect Size (Cohen's d):", round(cohens_d$Cohens_d, 3), "\n")

if (abs(cohens_d$Cohens_d) < 0.2) {
  cat("Effect size: Small\n")
} else if (abs(cohens_d$Cohens_d) < 0.5) {
  cat("Effect size: Medium\n")
} else if (abs(cohens_d$Cohens_d) < 0.8) {
  cat("Effect size: Large\n")
} else {
  cat("Effect size: Very large\n")
}

# Confidence interval
cat("95% CI for difference:", round(welch_test$conf.int[1], 3), "to", round(welch_test$conf.int[2], 3), "\n")

# Practical significance
mean_diff <- mean(manual_mpg) - mean(auto_mpg)
cat("Mean difference (manual - auto):", round(mean_diff, 3), "\n")

if (abs(mean_diff) > 2) {
  cat("✓ Practical significance: Large difference in MPG\n")
} else if (abs(mean_diff) > 1) {
  cat("⚠ Practical significance: Moderate difference in MPG\n")
} else {
  cat("✗ Practical significance: Small difference in MPG\n")
}
```

### Paired t-Test

The paired t-test is used to compare the means of two related groups, such as before/after measurements on the same subjects.

#### Mathematical Foundation

For paired observations $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$, we work with the differences $d_i = x_i - y_i$.

The test statistic is:
```math
t = \frac{\bar{d}}{s_d/\sqrt{n}}
```

Where:
- $\bar{d}$ is the mean of the differences
- $s_d$ is the standard deviation of the differences
- $n$ is the number of pairs

The test statistic follows a t-distribution with $n-1$ degrees of freedom.

**Hypotheses:**
```math
H_0: \mu_d = 0 \quad \text{vs.} \quad H_1: \mu_d \neq 0
```

#### Advantages of Paired Tests

1. **Reduced Variability**: Removes between-subject variability
2. **Increased Power**: More sensitive to detect treatment effects
3. **Control for Confounding**: Each subject serves as their own control

#### Code Implementation

```r
# Simulate paired data (before/after treatment)
set.seed(123)
before <- rnorm(20, mean = 100, sd = 15)
after <- before + rnorm(20, mean = 5, sd = 10)

cat("=== Paired t-Test ===\n")
cat("Hypothesis: H0: μ_d = 0 vs. H1: μ_d ≠ 0\n")
cat("Test type: Two-tailed\n\n")

cat("Data Summary:\n")
cat("Sample size (pairs):", length(before), "\n")
cat("Before treatment - mean =", round(mean(before), 2), ", SD =", round(sd(before), 2), "\n")
cat("After treatment - mean =", round(mean(after), 2), ", SD =", round(sd(after), 2), "\n")

# Calculate differences
differences <- after - before
cat("Differences - mean =", round(mean(differences), 2), ", SD =", round(sd(differences), 2), "\n")

# Perform paired t-test
paired_test <- t.test(after, before, paired = TRUE)

cat("\n=== Test Results ===\n")
cat("t-statistic:", round(paired_test$statistic, 3), "\n")
cat("Degrees of freedom:", paired_test$parameter, "\n")
cat("P-value:", round(paired_test$p.value, 4), "\n")
cat("95% CI for mean difference:", round(paired_test$conf.int[1], 2), "to", round(paired_test$conf.int[2], 2), "\n")

# Decision
alpha <- 0.05
if (paired_test$p.value < alpha) {
  cat("Decision: Reject H0 (treatment has significant effect)\n")
} else {
  cat("Decision: Fail to reject H0 (insufficient evidence)\n")
}

# Effect size
library(effectsize)
cohens_d_paired <- cohens_d(after, before, paired = TRUE)
cat("\nEffect Size (Cohen's d for paired samples):", round(cohens_d_paired$Cohens_d, 3), "\n")

if (abs(cohens_d_paired$Cohens_d) < 0.2) {
  cat("Effect size: Small\n")
} else if (abs(cohens_d_paired$Cohens_d) < 0.5) {
  cat("Effect size: Medium\n")
} else if (abs(cohens_d_paired$Cohens_d) < 0.8) {
  cat("Effect size: Large\n")
} else {
  cat("Effect size: Very large\n")
}

# Compare with independent t-test (incorrect for paired data)
independent_test <- t.test(after, before, paired = FALSE)
cat("\n=== Comparison with Independent t-Test ===\n")
cat("Independent t-test p-value:", round(independent_test$p.value, 4), "\n")
cat("Paired t-test p-value:", round(paired_test$p.value, 4), "\n")

if (paired_test$p.value < independent_test$p.value) {
  cat("✓ Paired test is more powerful (as expected)\n")
} else {
  cat("⚠ Paired test is not more powerful (unusual)\n")
}

# Check normality of differences
cat("\n=== Assumption Checking ===\n")
shapiro_diff <- shapiro.test(differences)
cat("Shapiro-Wilk test for differences:\n")
cat("W =", round(shapiro_diff$statistic, 3), ", p =", round(shapiro_diff$p.value, 4), "\n")

if (shapiro_diff$p.value < 0.05) {
  cat("⚠ Differences may not be normally distributed\n")
  cat("   Consider using Wilcoxon signed-rank test\n")
} else {
  cat("✓ Differences appear to be normally distributed\n")
}

# Correlation between before and after
correlation <- cor(before, after)
cat("Correlation between before and after:", round(correlation, 3), "\n")

if (correlation > 0.5) {
  cat("✓ Strong correlation - paired test is appropriate\n")
} else if (correlation > 0.3) {
  cat("⚠ Moderate correlation - paired test may still be beneficial\n")
} else {
  cat("✗ Weak correlation - independent test might be more appropriate\n")
}

# Practical significance
mean_diff <- mean(differences)
cat("\nPractical Significance:\n")
cat("Mean treatment effect:", round(mean_diff, 2), "units\n")

if (abs(mean_diff) > 10) {
  cat("✓ Large practical effect\n")
} else if (abs(mean_diff) > 5) {
  cat("⚠ Moderate practical effect\n")
} else {
  cat("✗ Small practical effect\n")
}
```

### Two-Sample Proportion Test

The two-sample proportion test is used to compare proportions between two independent groups.

#### Mathematical Foundation

For two independent samples with proportions $\hat{p}_1$ and $\hat{p}_2$, the test statistic is:

**Large Sample Approximation:**
```math
z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}
```

Where the pooled proportion is:
```math
\hat{p} = \frac{x_1 + x_2}{n_1 + n_2}
```

**Hypotheses:**
```math
H_0: p_1 = p_2 \quad \text{vs.} \quad H_1: p_1 \neq p_2
```

#### Assumptions

1. **Independence**: Observations are independent within and between groups
2. **Large Sample**: $n_1p_1 \geq 10$, $n_1(1-p_1) \geq 10$, $n_2p_2 \geq 10$, $n_2(1-p_2) \geq 10$
3. **Random Sampling**: Data comes from random samples

#### Code Implementation

```r
# Compare proportions between two groups
group1_success <- 15
group1_total <- 50
group2_success <- 25
group2_total <- 60

cat("=== Two-Sample Proportion Test ===\n")
cat("Hypothesis: H0: p1 = p2 vs. H1: p1 ≠ p2\n")
cat("Test type: Two-tailed\n\n")

cat("Data Summary:\n")
cat("Group 1: Successes =", group1_success, ", Total =", group1_total, 
    ", Proportion =", round(group1_success/group1_total, 3), "\n")
cat("Group 2: Successes =", group2_success, ", Total =", group2_total, 
    ", Proportion =", round(group2_success/group2_total, 3), "\n")

# Check large sample conditions
p1 <- group1_success / group1_total
p2 <- group2_success / group2_total

cat("\nLarge Sample Conditions:\n")
cat("Group 1 - n1*p1 =", group1_total * p1, ", n1*(1-p1) =", group1_total * (1-p1), "\n")
cat("Group 2 - n2*p2 =", group2_total * p2, ", n2*(1-p2) =", group2_total * (1-p2), "\n")

large_sample_ok <- (group1_total * p1 >= 10) && (group1_total * (1-p1) >= 10) &&
                   (group2_total * p2 >= 10) && (group2_total * (1-p2) >= 10)

if (large_sample_ok) {
  cat("✓ Large sample conditions are met\n")
} else {
  cat("⚠ Large sample conditions may not be met\n")
}

# Perform proportion test
prop_test_2sample <- prop.test(c(group1_success, group2_success),
                               c(group1_total, group2_total))

cat("\n=== Test Results ===\n")
cat("Chi-square statistic:", round(prop_test_2sample$statistic, 3), "\n")
cat("Degrees of freedom:", prop_test_2sample$parameter, "\n")
cat("P-value:", round(prop_test_2sample$p.value, 4), "\n")
cat("95% CI for difference:", round(prop_test_2sample$conf.int[1], 4), "to", round(prop_test_2sample$conf.int[2], 4), "\n")

# Decision
alpha <- 0.05
if (prop_test_2sample$p.value < alpha) {
  cat("Decision: Reject H0 (proportions differ significantly)\n")
} else {
  cat("Decision: Fail to reject H0 (insufficient evidence)\n")
}

# Effect size (difference in proportions)
prop_diff <- p2 - p1
cat("\nEffect Size Analysis:\n")
cat("Difference in proportions (p2 - p1):", round(prop_diff, 3), "\n")

if (abs(prop_diff) < 0.1) {
  cat("Effect size: Small\n")
} else if (abs(prop_diff) < 0.2) {
  cat("Effect size: Medium\n")
} else {
  cat("Effect size: Large\n")
}

# Risk ratio and odds ratio
risk_ratio <- p2 / p1
odds1 <- p1 / (1 - p1)
odds2 <- p2 / (1 - p2)
odds_ratio <- odds2 / odds1

cat("\nAdditional Measures:\n")
cat("Risk ratio (p2/p1):", round(risk_ratio, 3), "\n")
cat("Odds ratio:", round(odds_ratio, 3), "\n")

# Practical significance
cat("\nPractical Significance:\n")
if (abs(prop_diff) > 0.2) {
  cat("✓ Large practical difference\n")
} else if (abs(prop_diff) > 0.1) {
  cat("⚠ Moderate practical difference\n")
} else {
  cat("✗ Small practical difference\n")
}

# Fisher's exact test for small samples
if (!large_sample_ok) {
  cat("\n=== Fisher's Exact Test (for small samples) ===\n")
  
  # Create contingency table
  contingency_table <- matrix(c(group1_success, group1_total - group1_success,
                               group2_success, group2_total - group2_success), 
                             nrow = 2, byrow = TRUE)
  
  fisher_test <- fisher.test(contingency_table)
  cat("Fisher's exact test p-value:", round(fisher_test$p.value, 4), "\n")
  
  if (fisher_test$p.value < alpha) {
    cat("Decision: Reject H0 (proportions differ significantly)\n")
  } else {
    cat("Decision: Fail to reject H0 (insufficient evidence)\n")
  }
}
```

## Nonparametric Tests

Nonparametric tests make fewer assumptions about the underlying population distribution and are often used when parametric assumptions are violated.

### Wilcoxon Rank-Sum Test (Mann-Whitney U)

The Wilcoxon rank-sum test is a nonparametric alternative to the independent t-test that tests whether two independent samples come from populations with the same distribution.

#### Mathematical Foundation

The test statistic is based on the sum of ranks. For two samples of sizes $n_1$ and $n_2$:

```math
W = \sum_{i=1}^{n_1} R_i
```

Where $R_i$ are the ranks of the first sample when all observations are ranked together.

The test statistic follows approximately a normal distribution:
```math
Z = \frac{W - \mu_W}{\sigma_W}
```

Where:
```math
\mu_W = \frac{n_1(n_1 + n_2 + 1)}{2}
```

```math
\sigma_W = \sqrt{\frac{n_1 n_2(n_1 + n_2 + 1)}{12}}
```

**Hypotheses:**
```math
H_0: \text{The two populations have the same distribution}
```

```math
H_1: \text{The two populations have different distributions}
```

#### Assumptions

1. **Independence**: Observations are independent within and between groups
2. **Random Sampling**: Data comes from random samples
3. **Ordinal Data**: Data can be ranked (no normality assumption)

#### Code Implementation

```r
# Nonparametric alternative to independent t-test
wilcox_test <- wilcox.test(auto_mpg, manual_mpg)

cat("=== Wilcoxon Rank-Sum Test ===\n")
cat("Hypothesis: H0: Same distribution vs. H1: Different distributions\n")
cat("Test type: Two-tailed\n\n")

cat("Data Summary:\n")
cat("Automatic transmissions: n =", length(auto_mpg), ", median =", round(median(auto_mpg), 3), "\n")
cat("Manual transmissions: n =", length(manual_mpg), ", median =", round(median(manual_mpg), 3), "\n")

cat("\n=== Test Results ===\n")
cat("W-statistic:", wilcox_test$statistic, "\n")
cat("P-value:", round(wilcox_test$p.value, 4), "\n")

# Decision
alpha <- 0.05
if (wilcox_test$p.value < alpha) {
  cat("Decision: Reject H0 (distributions differ significantly)\n")
} else {
  cat("Decision: Fail to reject H0 (insufficient evidence)\n")
}

# Compare with parametric t-test
cat("\n=== Comparison with Parametric Test ===\n")
t_test_result <- t.test(auto_mpg, manual_mpg)
cat("t-test p-value:", round(t_test_result$p.value, 4), "\n")
cat("Wilcoxon p-value:", round(wilcox_test$p.value, 4), "\n")

if (abs(t_test_result$p.value - wilcox_test$p.value) < 0.01) {
  cat("✓ Both tests give similar results\n")
} else {
  cat("⚠ Tests give different results - check assumptions\n")
}

# Effect size (rank-biserial correlation)
library(effectsize)
rank_biserial <- rank_biserial_correlation(auto_mpg, manual_mpg)
cat("\nEffect Size (Rank-biserial correlation):", round(rank_biserial$r_rank_biserial, 3), "\n")

if (abs(rank_biserial$r_rank_biserial) < 0.1) {
  cat("Effect size: Small\n")
} else if (abs(rank_biserial$r_rank_biserial) < 0.3) {
  cat("Effect size: Medium\n")
} else if (abs(rank_biserial$r_rank_biserial) < 0.5) {
  cat("Effect size: Large\n")
} else {
  cat("Effect size: Very large\n")
}

# Check when to use nonparametric test
cat("\n=== When to Use Nonparametric Tests ===\n")

# Normality check
shapiro_auto <- shapiro.test(auto_mpg)
shapiro_manual <- shapiro.test(manual_mpg)

cat("Normality tests:\n")
cat("Automatic - p =", round(shapiro_auto$p.value, 4), "\n")
cat("Manual - p =", round(shapiro_manual$p.value, 4), "\n")

if (shapiro_auto$p.value < 0.05 || shapiro_manual$p.value < 0.05) {
  cat("✓ Nonparametric test is appropriate (data not normal)\n")
} else {
  cat("✓ Both parametric and nonparametric tests are valid\n")
}

# Sample size consideration
total_n <- length(auto_mpg) + length(manual_mpg)
if (total_n < 30) {
  cat("✓ Nonparametric test is recommended for small samples\n")
} else {
  cat("✓ Parametric test is generally robust for large samples\n")
}
```

### Wilcoxon Signed-Rank Test

The Wilcoxon signed-rank test is a nonparametric alternative to the paired t-test that tests whether the median of the differences is zero.

#### Mathematical Foundation

For paired differences $d_i = x_i - y_i$, the test statistic is:

```math
W = \sum_{i=1}^{n} \text{sgn}(d_i) R_i
```

Where:
- $\text{sgn}(d_i)$ is the sign of the difference
- $R_i$ is the rank of $|d_i|$ when all absolute differences are ranked

The test statistic follows approximately a normal distribution:
```math
Z = \frac{W}{\sqrt{\frac{n(n+1)(2n+1)}{6}}}
```

**Hypotheses:**
```math
H_0: \text{The median of differences is zero}
```

```math
H_1: \text{The median of differences is not zero}
```

#### Code Implementation

```r
# Nonparametric alternative to paired t-test
wilcox_paired <- wilcox.test(after, before, paired = TRUE)

cat("=== Wilcoxon Signed-Rank Test ===\n")
cat("Hypothesis: H0: Median of differences = 0 vs. H1: Median of differences ≠ 0\n")
cat("Test type: Two-tailed\n\n")

cat("Data Summary:\n")
cat("Sample size (pairs):", length(before), "\n")
cat("Before treatment - median =", round(median(before), 2), "\n")
cat("After treatment - median =", round(median(after), 2), "\n")

# Calculate differences
differences <- after - before
cat("Differences - median =", round(median(differences), 2), "\n")

cat("\n=== Test Results ===\n")
cat("V-statistic:", wilcox_paired$statistic, "\n")
cat("P-value:", round(wilcox_paired$p.value, 4), "\n")

# Decision
alpha <- 0.05
if (wilcox_paired$p.value < alpha) {
  cat("Decision: Reject H0 (median difference is significant)\n")
} else {
  cat("Decision: Fail to reject H0 (insufficient evidence)\n")
}

# Compare with parametric paired t-test
cat("\n=== Comparison with Parametric Test ===\n")
paired_t_test <- t.test(after, before, paired = TRUE)
cat("Paired t-test p-value:", round(paired_t_test$p.value, 4), "\n")
cat("Wilcoxon signed-rank p-value:", round(wilcox_paired$p.value, 4), "\n")

if (abs(paired_t_test$p.value - wilcox_paired$p.value) < 0.01) {
  cat("✓ Both tests give similar results\n")
} else {
  cat("⚠ Tests give different results - check assumptions\n")
}

# Effect size (rank-biserial correlation for paired data)
library(effectsize)
rank_biserial_paired <- rank_biserial_correlation(after, before, paired = TRUE)
cat("\nEffect Size (Rank-biserial correlation):", round(rank_biserial_paired$r_rank_biserial, 3), "\n")

if (abs(rank_biserial_paired$r_rank_biserial) < 0.1) {
  cat("Effect size: Small\n")
} else if (abs(rank_biserial_paired$r_rank_biserial) < 0.3) {
  cat("Effect size: Medium\n")
} else if (abs(rank_biserial_paired$r_rank_biserial) < 0.5) {
  cat("Effect size: Large\n")
} else {
  cat("Effect size: Very large\n")
}

# Check normality of differences
cat("\n=== Assumption Checking ===\n")
shapiro_diff <- shapiro.test(differences)
cat("Shapiro-Wilk test for differences:\n")
cat("W =", round(shapiro_diff$statistic, 3), ", p =", round(shapiro_diff$p.value, 4), "\n")

if (shapiro_diff$p.value < 0.05) {
  cat("✓ Nonparametric test is appropriate (differences not normal)\n")
} else {
  cat("✓ Both parametric and nonparametric tests are valid\n")
}

# Practical significance
median_diff <- median(differences)
cat("\nPractical Significance:\n")
cat("Median treatment effect:", round(median_diff, 2), "units\n")

if (abs(median_diff) > 10) {
  cat("✓ Large practical effect\n")
} else if (abs(median_diff) > 5) {
  cat("⚠ Moderate practical effect\n")
} else {
  cat("✗ Small practical effect\n")
}

# Number of positive vs negative differences
positive_diff <- sum(differences > 0)
negative_diff <- sum(differences < 0)
zero_diff <- sum(differences == 0)

cat("\nDifference Analysis:\n")
cat("Positive differences:", positive_diff, "\n")
cat("Negative differences:", negative_diff, "\n")
cat("Zero differences:", zero_diff, "\n")

if (positive_diff > negative_diff) {
  cat("✓ More positive differences (treatment appears effective)\n")
} else if (negative_diff > positive_diff) {
  cat("✗ More negative differences (treatment appears harmful)\n")
} else {
  cat("⚠ Equal positive and negative differences\n")
}
```

### Kruskal-Wallis Test

The Kruskal-Wallis test is a nonparametric alternative to one-way ANOVA that tests whether multiple independent groups have the same distribution.

#### Mathematical Foundation

For $k$ groups with sample sizes $n_1, n_2, \ldots, n_k$, the test statistic is:

```math
H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)
```

Where:
- $N = n_1 + n_2 + \ldots + n_k$ is the total sample size
- $R_i$ is the sum of ranks for group $i$

The test statistic follows approximately a chi-square distribution with $k-1$ degrees of freedom.

**Hypotheses:**
```math
H_0: \text{All groups have the same distribution}
```

```math
H_1: \text{At least one group has a different distribution}
```

#### Code Implementation

```r
# Nonparametric alternative to one-way ANOVA
kruskal_test <- kruskal.test(mpg ~ factor(cyl), data = mtcars)

cat("=== Kruskal-Wallis Test ===\n")
cat("Hypothesis: H0: All groups have same distribution vs. H1: At least one group differs\n")
cat("Test type: Two-tailed\n\n")

# Data summary by group
cat("Data Summary by Cylinders:\n")
cyl_groups <- split(mtcars$mpg, mtcars$cyl)
for (cyl in names(cyl_groups)) {
  group_data <- cyl_groups[[cyl]]
  cat("Cylinders", cyl, ": n =", length(group_data), 
      ", median =", round(median(group_data), 2),
      ", mean =", round(mean(group_data), 2), "\n")
}

cat("\n=== Test Results ===\n")
cat("H-statistic:", round(kruskal_test$statistic, 3), "\n")
cat("Degrees of freedom:", kruskal_test$parameter, "\n")
cat("P-value:", round(kruskal_test$p.value, 4), "\n")

# Decision
alpha <- 0.05
if (kruskal_test$p.value < alpha) {
  cat("Decision: Reject H0 (at least one group differs)\n")
} else {
  cat("Decision: Fail to reject H0 (insufficient evidence)\n")
}

# Effect size (epsilon-squared)
library(effectsize)
epsilon_squared <- epsilon_squared(kruskal_test)
cat("\nEffect Size (Epsilon-squared):", round(epsilon_squared$Epsilon2, 3), "\n")

if (epsilon_squared$Epsilon2 < 0.06) {
  cat("Effect size: Small\n")
} else if (epsilon_squared$Epsilon2 < 0.14) {
  cat("Effect size: Medium\n")
} else {
  cat("Effect size: Large\n")
}

# Compare with parametric ANOVA
cat("\n=== Comparison with Parametric ANOVA ===\n")
anova_result <- aov(mpg ~ factor(cyl), data = mtcars)
anova_summary <- summary(anova_result)
cat("F-statistic:", round(anova_summary[[1]]$"F value"[1], 3), "\n")
cat("ANOVA p-value:", round(anova_summary[[1]]$"Pr(>F)"[1], 4), "\n")
cat("Kruskal-Wallis p-value:", round(kruskal_test$p.value, 4), "\n")

if (abs(anova_summary[[1]]$"Pr(>F)"[1] - kruskal_test$p.value) < 0.01) {
  cat("✓ Both tests give similar results\n")
} else {
  cat("⚠ Tests give different results - check assumptions\n")
}

# Post-hoc analysis (if significant)
if (kruskal_test$p.value < alpha) {
  cat("\n=== Post-hoc Analysis ===\n")
  cat("Since the overall test is significant, we need post-hoc tests\n")
  cat("to determine which specific groups differ.\n")
  
  # Pairwise Wilcoxon tests with Bonferroni correction
  library(pairwiseComparisons)
  pairwise_wilcox <- pairwise_wilcox_test(mpg ~ cyl, data = mtcars, p.adjust.method = "bonferroni")
  cat("Pairwise Wilcoxon tests with Bonferroni correction:\n")
  print(pairwise_wilcox)
}

# Check assumptions
cat("\n=== Assumption Checking ===\n")

# Normality check for each group
cat("Normality tests by group:\n")
for (cyl in names(cyl_groups)) {
  group_data <- cyl_groups[[cyl]]
  shapiro_result <- shapiro.test(group_data)
  cat("Cylinders", cyl, "- W =", round(shapiro_result$statistic, 3), 
      ", p =", round(shapiro_result$p.value, 4), "\n")
}

# Homogeneity of variance
cat("\nHomogeneity of variance test:\n")
bartlett_result <- bartlett.test(mpg ~ factor(cyl), data = mtcars)
cat("Bartlett's test - p =", round(bartlett_result$p.value, 4), "\n")

if (bartlett_result$p.value < 0.05) {
  cat("✓ Nonparametric test is appropriate (variances not equal)\n")
} else {
  cat("✓ Both parametric and nonparametric tests are valid\n")
}

# Sample size consideration
total_n <- nrow(mtcars)
if (total_n < 30) {
  cat("✓ Nonparametric test is recommended for small samples\n")
} else {
  cat("✓ Parametric test is generally robust for large samples\n")
}

# Practical significance
cat("\nPractical Significance:\n")
overall_median <- median(mtcars$mpg)
cat("Overall median MPG:", round(overall_median, 2), "\n")

for (cyl in names(cyl_groups)) {
  group_data <- cyl_groups[[cyl]]
  group_median <- median(group_data)
  cat("Cylinders", cyl, "median:", round(group_median, 2), 
      "(difference from overall:", round(group_median - overall_median, 2), ")\n")
}
```

## Multiple Comparison Tests

### One-Way ANOVA

```r
# Test if mean MPG differs by number of cylinders
anova_result <- aov(mpg ~ factor(cyl), data = mtcars)

print(anova_result)
print(summary(anova_result))

# Post-hoc tests
library(multcomp)
posthoc <- glht(anova_result, linfct = mcp(cyl = "Tukey"))
print(summary(posthoc))
```

### Chi-Square Test

```r
# Test independence between transmission and cylinders
contingency_table <- table(mtcars$am, mtcars$cyl)
chi_square_test <- chisq.test(contingency_table)

print(chi_square_test)
print(contingency_table)
```

## Effect Size

### Cohen's d

```r
# Calculate Cohen's d for t-test
library(effectsize)

cohens_d <- cohens_d(auto_mpg, manual_mpg)
print(cohens_d)

# Interpret effect size
cat("Effect size interpretation:\n")
cat("d = 0.2: Small effect\n")
cat("d = 0.5: Medium effect\n")
cat("d = 0.8: Large effect\n")
```

### Eta-squared

```r
# Calculate eta-squared for ANOVA
library(effectsize)

eta_squared <- eta_squared(anova_result)
print(eta_squared)
```

## Power Analysis

```r
# Install and load pwr package
install.packages("pwr")
library(pwr)

# Power analysis for t-test
power_analysis <- pwr.t.test(d = 0.5, sig.level = 0.05, power = 0.8)
print(power_analysis)

# Sample size needed
cat("Sample size needed per group:", ceiling(power_analysis$n), "\n")
```

## Multiple Testing Correction

### Bonferroni Correction

```r
# Multiple p-values
p_values <- c(0.01, 0.03, 0.05, 0.08, 0.12)

# Bonferroni correction
alpha <- 0.05
bonferroni_corrected <- p.adjust(p_values, method = "bonferroni")

cat("Original p-values:", p_values, "\n")
cat("Bonferroni corrected:", bonferroni_corrected, "\n")
cat("Significant at α = 0.05:", bonferroni_corrected < alpha, "\n")
```

### False Discovery Rate

```r
# FDR correction
fdr_corrected <- p.adjust(p_values, method = "fdr")

cat("FDR corrected:", fdr_corrected, "\n")
cat("Significant at α = 0.05:", fdr_corrected < alpha, "\n")
```

## Practical Examples

### Example 1: Drug Efficacy Study

```r
# Simulate drug efficacy data
set.seed(123)
placebo <- rnorm(30, mean = 50, sd = 10)
treatment <- rnorm(30, mean = 55, sd = 10)

# Test if treatment is effective
efficacy_test <- t.test(treatment, placebo, alternative = "greater")

print(efficacy_test)

# Calculate effect size
effect_size <- cohens_d(treatment, placebo)
print(effect_size)
```

### Example 2: Customer Satisfaction Survey

```r
# Simulate satisfaction scores
set.seed(123)
store_a <- rnorm(50, mean = 7.5, sd = 1.2)
store_b <- rnorm(50, mean = 7.8, sd = 1.1)

# Test if satisfaction differs
satisfaction_test <- t.test(store_a, store_b)

print(satisfaction_test)

# Check normality
shapiro.test(store_a)
shapiro.test(store_b)
```

### Example 3: A/B Testing

```r
# Simulate A/B test data
set.seed(123)
version_a_conversions <- rbinom(1000, 1, 0.05)
version_b_conversions <- rbinom(1000, 1, 0.06)

# Test conversion rates
ab_test <- prop.test(c(sum(version_a_conversions), sum(version_b_conversions)),
                     c(length(version_a_conversions), length(version_b_conversions)))

print(ab_test)
```

## Best Practices

### Assumption Checking

```r
# Function to check t-test assumptions
check_t_test_assumptions <- function(x, y = NULL) {
  cat("Normality test (Shapiro-Wilk):\n")
  if (is.null(y)) {
    print(shapiro.test(x))
  } else {
    print(shapiro.test(x))
    print(shapiro.test(y))
  }
  
  if (!is.null(y)) {
    cat("\nVariance ratio:", var(x) / var(y), "\n")
    cat("If ratio > 2, consider Welch's t-test\n")
  }
}

# Apply to our data
check_t_test_assumptions(auto_mpg, manual_mpg)
```

### Reporting Results

```r
# Function to format test results
format_test_results <- function(test_result, test_name) {
  cat(test_name, "\n")
  cat("Test statistic:", round(test_result$statistic, 3), "\n")
  cat("P-value:", round(test_result$p.value, 4), "\n")
  
  if (test_result$p.value < 0.001) {
    cat("Significance: p < 0.001\n")
  } else if (test_result$p.value < 0.01) {
    cat("Significance: p < 0.01\n")
  } else if (test_result$p.value < 0.05) {
    cat("Significance: p < 0.05\n")
  } else {
    cat("Significance: p >= 0.05\n")
  }
}

# Apply to t-test
format_test_results(t_test_result, "Independent t-test")
```

## Common Mistakes to Avoid

```r
# 1. Multiple testing without correction
cat("Mistake: Running many tests without correction\n")
cat("Solution: Use Bonferroni, FDR, or other corrections\n\n")

# 2. Ignoring effect size
cat("Mistake: Only reporting p-values\n")
cat("Solution: Always report effect sizes\n\n")

# 3. Data dredging
cat("Mistake: Testing many hypotheses without pre-specification\n")
cat("Solution: Pre-specify hypotheses and analysis plan\n\n")

# 4. Ignoring assumptions
cat("Mistake: Not checking test assumptions\n")
cat("Solution: Always verify assumptions before testing\n")
```

## Exercises

### Exercise 1: Basic Hypothesis Testing
Test if the mean MPG in the mtcars dataset is different from 20 using a t-test.

### Exercise 2: Two-Sample Comparison
Compare the MPG between automatic and manual transmissions using both parametric and nonparametric tests.

### Exercise 3: Multiple Testing
Perform multiple t-tests and apply correction methods to control for multiple comparisons.

### Exercise 4: Power Analysis
Calculate the required sample size for detecting a medium effect size with 80% power.

### Exercise 5: Real-world Application
Design and conduct a hypothesis test for a real-world scenario of your choice.

## Next Steps

In the next chapter, we'll learn about confidence intervals and their relationship with hypothesis testing.

---

**Key Takeaways:**
- Always state null and alternative hypotheses clearly
- Choose appropriate tests based on data type and assumptions
- Report both p-values and effect sizes
- Check assumptions before conducting tests
- Use correction methods for multiple testing
- Consider power analysis for study design
- Interpret results in context of practical significance
- Avoid common pitfalls like data dredging 