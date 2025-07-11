# Hypothesis Testing

## Overview

Hypothesis testing is a fundamental concept in statistics that allows us to make decisions about populations based on sample data. It provides a framework for testing claims about parameters or relationships in data.

## Basic Concepts

### Null and Alternative Hypotheses

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

cat("Observed proportion of heads:", observed_proportion, "\n")
cat("Expected proportion under H0:", 0.5, "\n")
```

### Type I and Type II Errors

```r
# Understanding error types
cat("Type I Error (α): Rejecting H0 when it's true\n")
cat("Type II Error (β): Failing to reject H0 when it's false\n")
cat("Power (1-β): Probability of correctly rejecting H0\n")

# Example: Setting significance level
alpha <- 0.05
cat("Significance level (α):", alpha, "\n")
cat("This means we're willing to make a Type I error 5% of the time\n")
```

## One-Sample Tests

### One-Sample t-Test

```r
# Load data
data(mtcars)

# Test if mean MPG is different from 20
mpg_test <- t.test(mtcars$mpg, mu = 20)

# View results
print(mpg_test)

# Extract components
cat("Test statistic:", mpg_test$statistic, "\n")
cat("P-value:", mpg_test$p.value, "\n")
cat("Confidence interval:", mpg_test$conf.int, "\n")
cat("Sample mean:", mpg_test$estimate, "\n")
```

### One-Sample Proportion Test

```r
# Test proportion of automatic transmissions
auto_count <- sum(mtcars$am == 0)
total_cars <- nrow(mtcars)
auto_proportion <- auto_count / total_cars

# Test if proportion is different from 0.5
prop_test <- prop.test(auto_count, total_cars, p = 0.5)

print(prop_test)
```

### One-Sample Variance Test

```r
# Test if variance of MPG is different from 25
mpg_variance <- var(mtcars$mpg)
n <- length(mtcars$mpg)
test_statistic <- (n - 1) * mpg_variance / 25
p_value <- 2 * (1 - pchisq(test_statistic, n - 1))

cat("Test statistic:", test_statistic, "\n")
cat("P-value:", p_value, "\n")
cat("Sample variance:", mpg_variance, "\n")
```

## Two-Sample Tests

### Independent t-Test

```r
# Compare MPG between automatic and manual transmissions
auto_mpg <- mtcars$mpg[mtcars$am == 0]
manual_mpg <- mtcars$mpg[mtcars$am == 1]

# Perform t-test
t_test_result <- t.test(auto_mpg, manual_mpg)

print(t_test_result)

# Check assumptions
cat("Variance ratio:", var(auto_mpg) / var(manual_mpg), "\n")
cat("If ratio > 2, consider Welch's t-test\n")
```

### Paired t-Test

```r
# Simulate paired data (before/after treatment)
set.seed(123)
before <- rnorm(20, mean = 100, sd = 15)
after <- before + rnorm(20, mean = 5, sd = 10)

# Perform paired t-test
paired_test <- t.test(after, before, paired = TRUE)

print(paired_test)
```

### Two-Sample Proportion Test

```r
# Compare proportions between two groups
group1_success <- 15
group1_total <- 50
group2_success <- 25
group2_total <- 60

prop_test_2sample <- prop.test(c(group1_success, group2_success),
                               c(group1_total, group2_total))

print(prop_test_2sample)
```

## Nonparametric Tests

### Wilcoxon Rank-Sum Test (Mann-Whitney U)

```r
# Nonparametric alternative to independent t-test
wilcox_test <- wilcox.test(auto_mpg, manual_mpg)

print(wilcox_test)
```

### Wilcoxon Signed-Rank Test

```r
# Nonparametric alternative to paired t-test
wilcox_paired <- wilcox.test(after, before, paired = TRUE)

print(wilcox_paired)
```

### Kruskal-Wallis Test

```r
# Nonparametric alternative to one-way ANOVA
kruskal_test <- kruskal.test(mpg ~ factor(cyl), data = mtcars)

print(kruskal_test)
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