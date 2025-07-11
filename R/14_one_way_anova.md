# One-Way ANOVA

## Overview

One-way Analysis of Variance (ANOVA) is a powerful statistical technique used to compare means across three or more groups simultaneously. It extends the two-sample t-test to multiple groups and is fundamental to experimental design and observational studies.

### When to Use One-Way ANOVA

One-way ANOVA is appropriate when you want to:
- Compare means across three or more independent groups
- Test the null hypothesis that all group means are equal
- Determine if there are significant differences between any groups
- Analyze experimental data with one categorical independent variable
- Compare multiple treatments, conditions, or categories

### Key Concepts

**Null Hypothesis (H₀):** All population means are equal ($\mu_1 = \mu_2 = \mu_3 = ... = \mu_k$)
**Alternative Hypothesis (H₁):** At least one population mean differs from the others

The test determines whether observed differences between group means are statistically significant or due to random sampling variation.

### Mathematical Foundation

One-way ANOVA is based on partitioning the total variance in the data into two components:

**Total Sum of Squares (SST):**
```math
SST = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_{..})^2
```

**Between-Groups Sum of Squares (SSB):**
```math
SSB = \sum_{i=1}^{k} n_i(\bar{X}_{i.} - \bar{X}_{..})^2
```

**Within-Groups Sum of Squares (SSW):**
```math
SSW = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_{i.})^2
```

where:
- $X_{ij}$ = observation $j$ in group $i$
- $\bar{X}_{i.}$ = mean of group $i$
- $\bar{X}_{..}$ = overall mean
- $n_i$ = sample size of group $i$
- $k$ = number of groups

**F-Statistic:**
```math
F = \frac{MSB}{MSW} = \frac{SSB/(k-1)}{SSW/(N-k)}
```

where $N = \sum_{i=1}^{k} n_i$ is the total sample size.

The F-statistic follows an F-distribution with $(k-1, N-k)$ degrees of freedom under the null hypothesis.

## Basic One-Way ANOVA

The basic one-way ANOVA procedure involves calculating the F-statistic by comparing the variance between groups to the variance within groups. Understanding the manual calculation helps clarify the underlying principles.

### Mathematical Foundation

**Variance Partitioning:**
The total variance in the data is partitioned into two components:
1. **Between-groups variance:** Measures differences between group means
2. **Within-groups variance:** Measures variability within each group

**Expected Values Under Null Hypothesis:**
- $E(MSB) = \sigma^2 + \frac{\sum_{i=1}^{k} n_i(\mu_i - \bar{\mu})^2}{k-1}$
- $E(MSW) = \sigma^2$

where $\bar{\mu} = \frac{\sum_{i=1}^{k} n_i\mu_i}{N}$ is the weighted grand mean.

Under the null hypothesis ($\mu_1 = \mu_2 = ... = \mu_k$), both expected values equal $\sigma^2$, so $E(F) = 1$.

**F-Distribution Properties:**
- $F \sim F(k-1, N-k)$ under $H_0$
- $F \geq 0$ (always positive)
- Reject $H_0$ if $F > F_{\alpha, k-1, N-k}$

### Manual Calculation

```r
# Load sample data
data(mtcars)

# Create groups based on number of cylinders
mtcars$cyl_factor <- factor(mtcars$cyl, levels = c(4, 6, 8), 
                           labels = c("4-cylinder", "6-cylinder", "8-cylinder"))

# Extract MPG data for each cylinder group
mpg_4cyl <- mtcars$mpg[mtcars$cyl == 4]
mpg_6cyl <- mtcars$mpg[mtcars$cyl == 6]
mpg_8cyl <- mtcars$mpg[mtcars$cyl == 8]

# Comprehensive manual ANOVA calculation
manual_anova <- function(group_data) {
  # Basic information
  k <- length(group_data)
  n_per_group <- sapply(group_data, length)
  total_n <- sum(n_per_group)
  
  cat("=== MANUAL ANOVA CALCULATION ===\n")
  cat("Number of groups (k):", k, "\n")
  cat("Sample sizes per group:", n_per_group, "\n")
  cat("Total sample size (N):", total_n, "\n\n")
  
  # Calculate group means and overall mean
  group_means <- sapply(group_data, mean)
  overall_mean <- mean(unlist(group_data))
  
  cat("Group means:", round(group_means, 3), "\n")
  cat("Overall mean:", round(overall_mean, 3), "\n\n")
  
  # Calculate Sum of Squares
  # Between-groups SS
  ss_between <- sum(n_per_group * (group_means - overall_mean)^2)
  
  # Within-groups SS
  ss_within <- sum(sapply(1:length(group_data), function(i) {
    sum((group_data[[i]] - group_means[i])^2)
  }))
  
  # Total SS
  ss_total <- ss_between + ss_within
  
  cat("Sum of Squares:\n")
  cat("Between-groups SS:", round(ss_between, 3), "\n")
  cat("Within-groups SS:", round(ss_within, 3), "\n")
  cat("Total SS:", round(ss_total, 3), "\n")
  cat("Verification (SSB + SSW = SST):", round(ss_between + ss_within, 3), "=", round(ss_total, 3), "\n\n")
  
  # Degrees of freedom
  df_between <- k - 1
  df_within <- total_n - k
  df_total <- total_n - 1
  
  cat("Degrees of Freedom:\n")
  cat("Between-groups df:", df_between, "\n")
  cat("Within-groups df:", df_within, "\n")
  cat("Total df:", df_total, "\n")
  cat("Verification (dfB + dfW = dfT):", df_between + df_within, "=", df_total, "\n\n")
  
  # Mean Squares
  ms_between <- ss_between / df_between
  ms_within <- ss_within / df_within
  
  cat("Mean Squares:\n")
  cat("Between-groups MS:", round(ms_between, 3), "\n")
  cat("Within-groups MS:", round(ms_within, 3), "\n\n")
  
  # F-statistic
  f_statistic <- ms_between / ms_within
  
  # p-value
  p_value <- 1 - pf(f_statistic, df_between, df_within)
  
  # Critical F-value
  f_critical <- qf(0.95, df_between, df_within)
  
  cat("F-Test Results:\n")
  cat("F-statistic:", round(f_statistic, 3), "\n")
  cat("Critical F-value (α = 0.05):", round(f_critical, 3), "\n")
  cat("p-value:", round(p_value, 4), "\n")
  cat("Significant:", p_value < 0.05, "\n\n")
  
  # Effect size (eta-squared)
  eta_squared <- ss_between / ss_total
  
  # Partial eta-squared (same as eta-squared for one-way ANOVA)
  partial_eta_squared <- eta_squared
  
  # Omega-squared (unbiased estimator)
  omega_squared <- (ss_between - (df_between * ms_within)) / (ss_total + ms_within)
  
  cat("Effect Sizes:\n")
  cat("Eta-squared:", round(eta_squared, 3), "\n")
  cat("Partial eta-squared:", round(partial_eta_squared, 3), "\n")
  cat("Omega-squared:", round(omega_squared, 3), "\n\n")
  
  # ANOVA table
  cat("ANOVA Table:\n")
  cat("Source\t\tSS\t\t\tDF\tMS\t\t\tF\t\tp-value\n")
  cat("Between\t", round(ss_between, 3), "\t", df_between, "\t", round(ms_between, 3), "\t", round(f_statistic, 3), "\t", round(p_value, 4), "\n")
  cat("Within\t\t", round(ss_within, 3), "\t", df_within, "\t", round(ms_within, 3), "\n")
  cat("Total\t\t", round(ss_total, 3), "\t", df_total, "\n\n")
  
  return(list(
    ss_between = ss_between,
    ss_within = ss_within,
    ss_total = ss_total,
    df_between = df_between,
    df_within = df_within,
    df_total = df_total,
    ms_between = ms_between,
    ms_within = ms_within,
    f_statistic = f_statistic,
    p_value = p_value,
    eta_squared = eta_squared,
    partial_eta_squared = partial_eta_squared,
    omega_squared = omega_squared,
    group_means = group_means,
    overall_mean = overall_mean,
    f_critical = f_critical
  ))
}

# Apply manual calculation
anova_result <- manual_anova(list(mpg_4cyl, mpg_6cyl, mpg_8cyl))

# Verify with R's built-in function
cat("=== VERIFICATION WITH R'S BUILT-IN FUNCTION ===\n")
```

### Using R's Built-in ANOVA

R's built-in ANOVA functions provide efficient and comprehensive analysis. The `aov()` function uses the same mathematical principles as manual calculation but with optimized algorithms.

**Model Specification:**
The ANOVA model can be written as:
```math
Y_{ij} = \mu + \alpha_i + \epsilon_{ij}
```

where:
- $Y_{ij}$ = observation $j$ in group $i$
- $\mu$ = overall mean
- $\alpha_i$ = effect of group $i$ (deviation from overall mean)
- $\epsilon_{ij}$ = random error term

**Assumptions:**
1. **Independence:** Observations are independent
2. **Normality:** Error terms are normally distributed
3. **Homoscedasticity:** Error terms have constant variance
4. **Linearity:** Effects are additive

```r
# Perform ANOVA using R's aov function
anova_model <- aov(mpg ~ cyl_factor, data = mtcars)
print(anova_model)

# Get comprehensive ANOVA summary
anova_summary <- summary(anova_model)
print(anova_summary)

# Extract key statistics
f_statistic <- anova_summary[[1]]$`F value`[1]
p_value <- anova_summary[[1]]$`Pr(>F)`[1]
df_between <- anova_summary[[1]]$Df[1]
df_within <- anova_summary[[1]]$Df[2]
ss_between <- anova_summary[[1]]$`Sum Sq`[1]
ss_within <- anova_summary[[1]]$`Sum Sq`[2]
ms_between <- anova_summary[[1]]$`Mean Sq`[1]
ms_within <- anova_summary[[1]]$`Mean Sq`[2]

cat("=== R BUILT-IN ANOVA RESULTS ===\n")
cat("F-statistic:", round(f_statistic, 3), "\n")
cat("p-value:", round(p_value, 4), "\n")
cat("Degrees of freedom:", df_between, ",", df_within, "\n")
cat("Sum of Squares (Between):", round(ss_between, 3), "\n")
cat("Sum of Squares (Within):", round(ss_within, 3), "\n")
cat("Mean Square (Between):", round(ms_between, 3), "\n")
cat("Mean Square (Within):", round(ms_within, 3), "\n\n")

# Verify manual calculation matches R results
cat("=== VERIFICATION ===\n")
cat("Manual F-statistic:", round(anova_result$f_statistic, 3), "\n")
cat("R F-statistic:", round(f_statistic, 3), "\n")
cat("Match:", abs(anova_result$f_statistic - f_statistic) < 0.001, "\n\n")

cat("Manual p-value:", round(anova_result$p_value, 4), "\n")
cat("R p-value:", round(p_value, 4), "\n")
cat("Match:", abs(anova_result$p_value - p_value) < 0.0001, "\n\n")

# Model diagnostics
cat("=== MODEL DIAGNOSTICS ===\n")
cat("Model formula:", deparse(formula(anova_model)), "\n")
cat("Number of observations:", length(residuals(anova_model)), "\n")
cat("Number of groups:", length(unique(mtcars$cyl_factor)), "\n")

# Residual analysis
residuals_model <- residuals(anova_model)
fitted_values <- fitted(anova_model)

cat("\nResidual Analysis:\n")
cat("Mean of residuals:", round(mean(residuals_model), 6), "(should be ~0)\n")
cat("SD of residuals:", round(sd(residuals_model), 3), "\n")
cat("Min residual:", round(min(residuals_model), 3), "\n")
cat("Max residual:", round(max(residuals_model), 3), "\n")

# Model fit statistics
cat("\nModel Fit:\n")
cat("R-squared:", round(ss_between / (ss_between + ss_within), 3), "\n")
cat("Adjusted R-squared:", round(1 - (ss_within / df_within) / ((ss_between + ss_within) / (df_between + df_within)), 3), "\n")

# Confidence intervals for group means
cat("\n=== CONFIDENCE INTERVALS FOR GROUP MEANS ===\n")
group_means_ci <- tapply(mtcars$mpg, mtcars$cyl_factor, function(x) {
  n <- length(x)
  mean_val <- mean(x)
  se <- sd(x) / sqrt(n)
  t_critical <- qt(0.975, n - 1)
  ci_lower <- mean_val - t_critical * se
  ci_upper <- mean_val + t_critical * se
  return(c(mean = mean_val, lower = ci_lower, upper = ci_upper))
})

for (i in 1:length(group_means_ci)) {
  group_name <- names(group_means_ci)[i]
  ci <- group_means_ci[[i]]
  cat(group_name, ": ", round(ci["mean"], 2), " [", round(ci["lower"], 2), ", ", round(ci["upper"], 2), "]\n", sep = "")
}
```

## Descriptive Statistics

Descriptive statistics provide essential information about the data distribution and help assess ANOVA assumptions. They also aid in interpreting the practical significance of results.

### Mathematical Foundation

**Group Statistics:**
For each group $i$:
- **Mean:** $\bar{X}_i = \frac{1}{n_i}\sum_{j=1}^{n_i} X_{ij}$
- **Variance:** $s_i^2 = \frac{1}{n_i-1}\sum_{j=1}^{n_i} (X_{ij} - \bar{X}_i)^2$
- **Standard Error:** $SE_i = \frac{s_i}{\sqrt{n_i}}$

**Overall Statistics:**
- **Grand Mean:** $\bar{X}_{..} = \frac{1}{N}\sum_{i=1}^{k}\sum_{j=1}^{n_i} X_{ij}$
- **Pooled Variance:** $s_p^2 = \frac{\sum_{i=1}^{k} (n_i-1)s_i^2}{\sum_{i=1}^{k} (n_i-1)}$

**Effect Size Indicators:**
- **Coefficient of Variation:** $CV_i = \frac{s_i}{\bar{X}_i}$
- **Standardized Mean Difference:** $d_i = \frac{\bar{X}_i - \bar{X}_{..}}{s_p}$

### Group Comparisons

```r
# Comprehensive descriptive statistics for each group
library(dplyr)

group_stats <- mtcars %>%
  group_by(cyl_factor) %>%
  summarise(
    n = n(),
    mean = mean(mpg, na.rm = TRUE),
    sd = sd(mpg, na.rm = TRUE),
    median = median(mpg, na.rm = TRUE),
    min = min(mpg, na.rm = TRUE),
    max = max(mpg, na.rm = TRUE),
    q25 = quantile(mpg, 0.25, na.rm = TRUE),
    q75 = quantile(mpg, 0.75, na.rm = TRUE),
    iqr = q75 - q25,
    se = sd / sqrt(n),
    cv = sd / mean,
    skewness = moments::skewness(mpg),
    kurtosis = moments::kurtosis(mpg)
  )

cat("=== COMPREHENSIVE GROUP STATISTICS ===\n")
print(group_stats)

# Overall statistics
overall_stats <- mtcars %>%
  summarise(
    n = n(),
    mean = mean(mpg, na.rm = TRUE),
    sd = sd(mpg, na.rm = TRUE),
    median = median(mpg, na.rm = TRUE),
    min = min(mpg, na.rm = TRUE),
    max = max(mpg, na.rm = TRUE),
    q25 = quantile(mpg, 0.25, na.rm = TRUE),
    q75 = quantile(mpg, 0.75, na.rm = TRUE),
    iqr = q75 - q25,
    cv = sd / mean,
    skewness = moments::skewness(mpg),
    kurtosis = moments::kurtosis(mpg)
  )

cat("\n=== OVERALL STATISTICS ===\n")
print(overall_stats)

# Pooled variance calculation
pooled_variance <- function(group_data) {
  n_per_group <- sapply(group_data, length)
  var_per_group <- sapply(group_data, var)
  
  numerator <- sum((n_per_group - 1) * var_per_group)
  denominator <- sum(n_per_group - 1)
  
  return(numerator / denominator)
}

pooled_var <- pooled_variance(list(mpg_4cyl, mpg_6cyl, mpg_8cyl))
pooled_sd <- sqrt(pooled_var)

cat("\n=== POOLED STATISTICS ===\n")
cat("Pooled variance:", round(pooled_var, 3), "\n")
cat("Pooled standard deviation:", round(pooled_sd, 3), "\n")

# Effect size calculations for each group
group_means <- group_stats$mean
overall_mean <- overall_stats$mean

effect_sizes <- (group_means - overall_mean) / pooled_sd
names(effect_sizes) <- group_stats$cyl_factor

cat("\n=== EFFECT SIZE INDICATORS ===\n")
cat("Standardized mean differences (Cohen's d relative to overall mean):\n")
for (i in 1:length(effect_sizes)) {
  cat(names(effect_sizes)[i], ": d =", round(effect_sizes[i], 3), "\n")
}

# Variance ratio analysis
variances <- group_stats$sd^2
max_var <- max(variances)
min_var <- min(variances)
var_ratio <- max_var / min_var

cat("\n=== VARIANCE ANALYSIS ===\n")
cat("Group variances:", round(variances, 3), "\n")
cat("Variance ratio (max/min):", round(var_ratio, 3), "\n")
if (var_ratio > 4) {
  cat("⚠️  Large variance ratio - consider assumption violations\n")
} else if (var_ratio > 2) {
  cat("⚠️  Moderate variance ratio - check homogeneity assumption\n")
} else {
  cat("✓ Variance ratio acceptable\n")
}

# Sample size analysis
n_per_group <- group_stats$n
balanced_design <- length(unique(n_per_group)) == 1

cat("\n=== SAMPLE SIZE ANALYSIS ===\n")
cat("Sample sizes per group:", n_per_group, "\n")
cat("Design balanced:", balanced_design, "\n")
if (!balanced_design) {
  cat("⚠️  Unbalanced design - consider Type III SS for interactions\n")
} else {
  cat("✓ Balanced design\n")
}

# Power analysis based on sample sizes
min_n <- min(n_per_group)
cat("Minimum sample size per group:", min_n, "\n")
if (min_n < 10) {
  cat("⚠️  Small sample sizes - consider nonparametric alternatives\n")
} else if (min_n < 30) {
  cat("⚠️  Moderate sample sizes - check normality carefully\n")
} else {
  cat("✓ Adequate sample sizes for parametric tests\n")
}
```

### Visualization

Visualization is crucial for understanding data distributions, identifying patterns, and assessing ANOVA assumptions. Different plots provide complementary information about the data structure.

**Key Visualization Goals:**
1. **Distribution Comparison:** Assess normality and homogeneity
2. **Central Tendency:** Compare group means and medians
3. **Variability:** Examine spread and outliers
4. **Effect Size:** Visualize practical significance

```r
library(ggplot2)
library(gridExtra)

# Enhanced box plot with statistics
p1 <- ggplot(mtcars, aes(x = cyl_factor, y = mpg, fill = cyl_factor)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 16, outlier.size = 2) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white") +
  labs(title = "MPG by Number of Cylinders", 
       subtitle = "Boxes show IQR, lines show medians, diamonds show means",
       x = "Cylinders", y = "MPG") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")

# Violin plot with box plot overlay
p2 <- ggplot(mtcars, aes(x = cyl_factor, y = mpg, fill = cyl_factor)) +
  geom_violin(alpha = 0.7, scale = "width") +
  geom_boxplot(width = 0.2, alpha = 0.8, outlier.shape = NA) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white") +
  labs(title = "MPG Distribution by Cylinders", 
       subtitle = "Violin shows density, box shows quartiles",
       x = "Cylinders", y = "MPG") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")

# Histogram by group with density curves
p3 <- ggplot(mtcars, aes(x = mpg, fill = cyl_factor)) +
  geom_histogram(bins = 8, alpha = 0.7, position = "identity") +
  geom_density(aes(y = ..density.. * 8 * 2), alpha = 0.5) +
  facet_wrap(~cyl_factor, scales = "free_y") +
  labs(title = "MPG Distribution by Cylinders", 
       subtitle = "Histograms with density curves",
       x = "MPG", y = "Count") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")

# Q-Q plots for normality assessment
p4 <- ggplot(mtcars, aes(sample = mpg, color = cyl_factor)) +
  stat_qq() +
  stat_qq_line() +
  facet_wrap(~cyl_factor) +
  labs(title = "Q-Q Plots for Normality Assessment", 
       subtitle = "Points should follow the line for normal distributions",
       x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal() +
  scale_color_brewer(palette = "Set2")

# Residuals vs fitted plot
p5 <- ggplot(data.frame(
  fitted = fitted(anova_model),
  residuals = residuals(anova_model),
  group = mtcars$cyl_factor
), aes(x = fitted, y = residuals, color = group)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +
  labs(title = "Residuals vs Fitted Values", 
       subtitle = "Should show no pattern for valid ANOVA",
       x = "Fitted Values", y = "Residuals") +
  theme_minimal() +
  scale_color_brewer(palette = "Set2")

# Mean comparison plot with confidence intervals
p6 <- ggplot(group_stats, aes(x = cyl_factor, y = mean, fill = cyl_factor)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  geom_errorbar(aes(ymin = mean - 1.96 * se, ymax = mean + 1.96 * se), 
                width = 0.2, size = 1) +
  labs(title = "Group Means with 95% Confidence Intervals", 
       subtitle = "Bars show means, error bars show 95% CI",
       x = "Cylinders", y = "Mean MPG") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")

# Combine plots in a comprehensive layout
cat("=== COMPREHENSIVE ANOVA VISUALIZATION ===\n")
cat("Generating 6 different plots for complete data analysis...\n")

# Display plots in a 2x3 grid
grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 2, nrow = 3)

# Additional diagnostic plots
cat("\n=== ADDITIONAL DIAGNOSTIC PLOTS ===\n")

# Scale-location plot for homoscedasticity
p7 <- ggplot(data.frame(
  fitted = fitted(anova_model),
  sqrt_abs_resid = sqrt(abs(residuals(anova_model))),
  group = mtcars$cyl_factor
), aes(x = fitted, y = sqrt_abs_resid, color = group)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +
  labs(title = "Scale-Location Plot", 
       subtitle = "Should show constant spread for homoscedasticity",
       x = "Fitted Values", y = "√|Residuals|") +
  theme_minimal() +
  scale_color_brewer(palette = "Set2")

# Leverage plot
p8 <- ggplot(data.frame(
  leverage = hatvalues(anova_model),
  residuals = residuals(anova_model),
  group = mtcars$cyl_factor
), aes(x = leverage, y = residuals, color = group)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Leverage Plot", 
       subtitle = "High leverage points may be influential",
       x = "Leverage", y = "Residuals") +
  theme_minimal() +
  scale_color_brewer(palette = "Set2")

# Display diagnostic plots
grid.arrange(p7, p8, ncol = 2)

# Summary of visual findings
cat("\n=== VISUAL ANALYSIS SUMMARY ===\n")
cat("1. Box plots: Show group differences and outliers\n")
cat("2. Violin plots: Show distribution shapes and density\n")
cat("3. Histograms: Show data distribution and normality\n")
cat("4. Q-Q plots: Assess normality assumption\n")
cat("5. Residual plots: Check model assumptions\n")
cat("6. Mean plots: Show effect sizes with uncertainty\n")
cat("7. Scale-location: Check homoscedasticity\n")
cat("8. Leverage: Identify influential observations\n")
```

## Effect Size

Effect size measures the magnitude of the relationship between the independent variable and the dependent variable, independent of sample size. This is crucial for understanding practical significance beyond statistical significance.

### Mathematical Foundation

**Eta-Squared (η²):**
```math
\eta^2 = \frac{SS_{between}}{SS_{total}} = \frac{SS_{between}}{SS_{between} + SS_{within}}
```

**Partial Eta-Squared (η²p):**
```math
\eta_p^2 = \frac{SS_{between}}{SS_{between} + SS_{within}}
```

For one-way ANOVA, partial eta-squared equals eta-squared.

**Omega-Squared (ω²):**
```math
\omega^2 = \frac{SS_{between} - (k-1)MS_{within}}{SS_{total} + MS_{within}}
```

**Cohen's f:**
```math
f = \sqrt{\frac{\eta^2}{1 - \eta^2}}
```

**Interpretation Guidelines:**
- **Eta-squared:** Small (0.01), Medium (0.06), Large (0.14)
- **Omega-squared:** Small (0.01), Medium (0.06), Large (0.14)
- **Cohen's f:** Small (0.10), Medium (0.25), Large (0.40)

### Comprehensive Effect Size Analysis

```r
# Comprehensive effect size calculation function
calculate_anova_effect_sizes <- function(anova_result) {
  # Extract components
  ss_between <- anova_result$ss_between
  ss_within <- anova_result$ss_within
  ss_total <- anova_result$ss_total
  df_between <- anova_result$df_between
  ms_within <- anova_result$ms_within
  
  # Eta-squared
  eta_squared <- ss_between / ss_total
  
  # Partial eta-squared (same as eta-squared for one-way ANOVA)
  partial_eta_squared <- eta_squared
  
  # Omega-squared (unbiased estimator)
  omega_squared <- (ss_between - (df_between * ms_within)) / (ss_total + ms_within)
  
  # Cohen's f
  cohens_f <- sqrt(eta_squared / (1 - eta_squared))
  
  # Epsilon-squared (for nonparametric ANOVA)
  epsilon_squared <- (ss_between - (df_between * ms_within)) / ss_total
  
  # Confidence interval for eta-squared
  f_stat <- anova_result$f_statistic
  df1 <- df_between
  df2 <- anova_result$df_within
  
  # Noncentrality parameter
  lambda <- f_stat * df1
  
  # Confidence interval using noncentral F distribution
  ci_lower <- 1 - 1 / (1 + lambda / qf(0.975, df1, df2))
  ci_upper <- 1 - 1 / (1 + lambda / qf(0.025, df1, df2))
  
  return(list(
    eta_squared = eta_squared,
    partial_eta_squared = partial_eta_squared,
    omega_squared = omega_squared,
    cohens_f = cohens_f,
    epsilon_squared = epsilon_squared,
    ci_lower = ci_lower,
    ci_upper = ci_upper
  ))
}

# Apply to our ANOVA results
effect_sizes <- calculate_anova_effect_sizes(anova_result)

cat("=== COMPREHENSIVE EFFECT SIZE ANALYSIS ===\n")
cat("Eta-squared (η²):", round(effect_sizes$eta_squared, 3), "\n")
cat("Partial eta-squared (η²p):", round(effect_sizes$partial_eta_squared, 3), "\n")
cat("Omega-squared (ω²):", round(effect_sizes$omega_squared, 3), "\n")
cat("Cohen's f:", round(effect_sizes$cohens_f, 3), "\n")
cat("Epsilon-squared (ε²):", round(effect_sizes$epsilon_squared, 3), "\n")
cat("95% CI for η²:", round(c(effect_sizes$ci_lower, effect_sizes$ci_upper), 3), "\n\n")

# Enhanced effect size interpretation
interpret_effect_size_comprehensive <- function(eta_sq, omega_sq, cohens_f) {
  cat("=== EFFECT SIZE INTERPRETATION ===\n")
  
  # Eta-squared interpretation
  cat("Eta-squared (η²):", round(eta_sq, 3), "\n")
  if (eta_sq < 0.01) {
    cat("  Interpretation: Negligible effect\n")
  } else if (eta_sq < 0.06) {
    cat("  Interpretation: Small effect\n")
  } else if (eta_sq < 0.14) {
    cat("  Interpretation: Medium effect\n")
  } else {
    cat("  Interpretation: Large effect\n")
  }
  
  # Omega-squared interpretation
  cat("\nOmega-squared (ω²):", round(omega_sq, 3), "\n")
  if (omega_sq < 0.01) {
    cat("  Interpretation: Negligible effect\n")
  } else if (omega_sq < 0.06) {
    cat("  Interpretation: Small effect\n")
  } else if (omega_sq < 0.14) {
    cat("  Interpretation: Medium effect\n")
  } else {
    cat("  Interpretation: Large effect\n")
  }
  
  # Cohen's f interpretation
  cat("\nCohen's f:", round(cohens_f, 3), "\n")
  if (cohens_f < 0.10) {
    cat("  Interpretation: Small effect\n")
  } else if (cohens_f < 0.25) {
    cat("  Interpretation: Medium effect\n")
  } else if (cohens_f < 0.40) {
    cat("  Interpretation: Large effect\n")
  } else {
    cat("  Interpretation: Very large effect\n")
  }
  
  # Practical significance assessment
  cat("\n=== PRACTICAL SIGNIFICANCE ASSESSMENT ===\n")
  if (eta_sq >= 0.14) {
    cat("✓ Large practical effect - results are practically meaningful\n")
  } else if (eta_sq >= 0.06) {
    cat("⚠️  Medium practical effect - consider context for interpretation\n")
  } else if (eta_sq >= 0.01) {
    cat("⚠️  Small practical effect - may not be practically meaningful\n")
  } else {
    cat("✗ Negligible practical effect - results may not be practically useful\n")
  }
}

# Apply comprehensive interpretation
interpret_effect_size_comprehensive(effect_sizes$eta_squared, 
                                   effect_sizes$omega_squared, 
                                   effect_sizes$cohens_f)

# Effect size comparison with other measures
cat("\n=== EFFECT SIZE COMPARISON ===\n")
cat("Comparison of different effect size measures:\n")
cat("Eta-squared (biased):", round(effect_sizes$eta_squared, 3), "\n")
cat("Omega-squared (unbiased):", round(effect_sizes$omega_squared, 3), "\n")
cat("Difference (bias):", round(effect_sizes$eta_squared - effect_sizes$omega_squared, 3), "\n")

if (abs(effect_sizes$eta_squared - effect_sizes$omega_squared) < 0.01) {
  cat("✓ Bias is minimal - eta-squared is acceptable\n")
} else {
  cat("⚠️  Notable bias - prefer omega-squared for small samples\n")
}

# Power analysis based on effect size
library(pwr)
f_effect_size <- effect_sizes$cohens_f
current_power <- pwr.anova.test(k = 3, n = min(table(mtcars$cyl_factor)), 
                                f = f_effect_size, sig.level = 0.05)$power

cat("\n=== POWER ANALYSIS ===\n")
cat("Effect size f:", round(f_effect_size, 3), "\n")
cat("Current power:", round(current_power, 3), "\n")

if (current_power < 0.8) {
  required_n <- pwr.anova.test(k = 3, f = f_effect_size, sig.level = 0.05, power = 0.8)$n
  cat("Required sample size per group for 80% power:", ceiling(required_n), "\n")
} else {
  cat("✓ Adequate power for detecting this effect size\n")
}
```

## Post Hoc Tests

When ANOVA reveals significant differences between groups, post hoc tests identify which specific group pairs differ significantly. These tests control for multiple comparisons to maintain the family-wise error rate.

### Mathematical Foundation

**Multiple Comparison Problem:**
With $k$ groups, there are $\binom{k}{2} = \frac{k(k-1)}{2}$ pairwise comparisons. If each test uses $\alpha = 0.05$, the family-wise error rate becomes:
```math
\alpha_{FW} = 1 - (1 - \alpha)^m
```
where $m$ is the number of comparisons.

**Tukey's HSD (Honestly Significant Difference):**
```math
HSD = q_{\alpha,k,N-k} \sqrt{\frac{MS_{within}}{n}}
```
where $q_{\alpha,k,N-k}$ is the critical value from the studentized range distribution.

**Bonferroni Correction:**
```math
\alpha_{adjusted} = \frac{\alpha}{m}
```

**Scheffe's Test:**
```math
S = \sqrt{(k-1)F_{\alpha,k-1,N-k}}
```

### Comprehensive Post Hoc Analysis

```r
# Comprehensive post hoc analysis function
comprehensive_posthoc <- function(anova_model, data, group_var, response_var, alpha = 0.05) {
  cat("=== COMPREHENSIVE POST HOC ANALYSIS ===\n")
  
  # Get group information
  groups <- unique(data[[group_var]])
  k <- length(groups)
  m <- k * (k - 1) / 2  # number of pairwise comparisons
  
  cat("Number of groups (k):", k, "\n")
  cat("Number of pairwise comparisons (m):", m, "\n")
  cat("Family-wise error rate without correction:", round(1 - (1 - alpha)^m, 4), "\n\n")
  
  # 1. Tukey's HSD Test
  cat("=== TUKEY'S HSD TEST ===\n")
  tukey_result <- TukeyHSD(anova_model, conf.level = 1 - alpha)
  print(tukey_result)
  
  # Extract significant differences
  tukey_matrix <- tukey_result[[1]]
  significant_tukey <- tukey_matrix[tukey_matrix[, "p adj"] < alpha, ]
  
  if (nrow(significant_tukey) > 0) {
    cat("\nSignificant pairwise differences (Tukey's HSD, p <", alpha, "):\n")
    for (i in 1:nrow(significant_tukey)) {
      row_name <- rownames(significant_tukey)[i]
      diff <- significant_tukey[i, "diff"]
      lwr <- significant_tukey[i, "lwr"]
      upr <- significant_tukey[i, "upr"]
      p_adj <- significant_tukey[i, "p adj"]
      cat(row_name, ": diff =", round(diff, 3), 
          "95% CI [", round(lwr, 3), ",", round(upr, 3), "], p =", round(p_adj, 4), "\n")
    }
  } else {
    cat("No significant pairwise differences found with Tukey's HSD.\n")
  }
  
  # 2. Bonferroni Correction
  cat("\n=== BONFERRONI CORRECTION ===\n")
  pairwise_tests <- pairwise.t.test(data[[response_var]], data[[group_var]], 
                                   p.adjust.method = "bonferroni")
  print(pairwise_tests)
  
  # Extract significant Bonferroni results
  bonferroni_matrix <- pairwise_tests$p.value
  significant_bonferroni <- which(bonferroni_matrix < alpha, arr.ind = TRUE)
  
  if (nrow(significant_bonferroni) > 0) {
    cat("\nSignificant differences (Bonferroni-corrected p <", alpha, "):\n")
    for (i in 1:nrow(significant_bonferroni)) {
      row_idx <- significant_bonferroni[i, 1]
      col_idx <- significant_bonferroni[i, 2]
      group1 <- rownames(bonferroni_matrix)[row_idx]
      group2 <- colnames(bonferroni_matrix)[col_idx]
      p_value <- bonferroni_matrix[row_idx, col_idx]
      cat(group1, "vs", group2, ": p =", round(p_value, 4), "\n")
    }
  } else {
    cat("No significant differences found with Bonferroni correction.\n")
  }
  
  # 3. Scheffe's Test
  cat("\n=== SCHEFFE'S TEST ===\n")
  scheffe_result <- scheffe_test(anova_result, list(mpg_4cyl, mpg_6cyl, mpg_8cyl), alpha)
  
  cat("Scheffe's critical value:", round(scheffe_result$scheffe_critical, 3), "\n\n")
  
  for (comp in scheffe_result$comparisons) {
    group_names <- c("4-cylinder", "6-cylinder", "8-cylinder")
    cat(group_names[comp$group1], "vs", group_names[comp$group2], ":\n")
    cat("  Mean difference:", round(comp$mean_diff, 3), "\n")
    cat("  Test statistic:", round(comp$test_stat, 3), "\n")
    cat("  p-value:", round(comp$p_value, 4), "\n")
    cat("  Significant:", comp$significant, "\n\n")
  }
  
  # 4. Holm's Method
  cat("=== HOLM'S METHOD ===\n")
  holm_tests <- pairwise.t.test(data[[response_var]], data[[group_var]], 
                               p.adjust.method = "holm")
  print(holm_tests)
  
  # 5. False Discovery Rate (FDR)
  cat("=== FALSE DISCOVERY RATE (FDR) ===\n")
  fdr_tests <- pairwise.t.test(data[[response_var]], data[[group_var]], 
                              p.adjust.method = "fdr")
  print(fdr_tests)
  
  # Comparison of methods
  cat("\n=== METHOD COMPARISON ===\n")
  cat("Method comparison summary:\n")
  cat("- Tukey's HSD: Controls FWER, most powerful for balanced designs\n")
  cat("- Bonferroni: Most conservative, controls FWER\n")
  cat("- Holm's method: Less conservative than Bonferroni, controls FWER\n")
  cat("- FDR: Controls false discovery rate, less conservative\n")
  cat("- Scheffe's: Most conservative, allows any contrast\n\n")
  
  return(list(
    tukey = tukey_result,
    bonferroni = pairwise_tests,
    scheffe = scheffe_result,
    holm = holm_tests,
    fdr = fdr_tests
  ))
}

# Apply comprehensive post hoc analysis
posthoc_results <- comprehensive_posthoc(anova_model, mtcars, "cyl_factor", "mpg")

# Visualization of post hoc results
cat("\n=== POST HOC VISUALIZATION ===\n")

# Tukey's HSD plot
plot(posthoc_results$tukey, main = "Tukey's HSD Test Results")

# Create a summary table of all methods
create_posthoc_summary <- function(posthoc_results, alpha = 0.05) {
  cat("=== POST HOC SUMMARY TABLE ===\n")
  
  # Extract significant pairs from each method
  tukey_sig <- posthoc_results$tukey[[1]][posthoc_results$tukey[[1]][, "p adj"] < alpha, ]
  bonferroni_sig <- which(posthoc_results$bonferroni$p.value < alpha, arr.ind = TRUE)
  holm_sig <- which(posthoc_results$holm$p.value < alpha, arr.ind = TRUE)
  fdr_sig <- which(posthoc_results$fdr$p.value < alpha, arr.ind = TRUE)
  
  cat("Method\t\tSignificant Pairs\n")
  cat("Tukey's HSD\t", nrow(tukey_sig), "\n")
  cat("Bonferroni\t", nrow(bonferroni_sig), "\n")
  cat("Holm's\t\t", nrow(holm_sig), "\n")
  cat("FDR\t\t", nrow(fdr_sig), "\n")
  
  # Agreement analysis
  cat("\n=== AGREEMENT ANALYSIS ===\n")
  cat("Methods that found the most significant differences:\n")
  method_counts <- c(
    "Tukey" = nrow(tukey_sig),
    "Bonferroni" = nrow(bonferroni_sig),
    "Holm" = nrow(holm_sig),
    "FDR" = nrow(fdr_sig)
  )
  
  for (method in names(sort(method_counts, decreasing = TRUE))) {
    cat(method, ":", method_counts[method], "significant pairs\n")
  }
}

create_posthoc_summary(posthoc_results)
```

### Bonferroni Correction

```r
# Perform pairwise t-tests with Bonferroni correction
pairwise_tests <- pairwise.t.test(mtcars$mpg, mtcars$cyl_factor, p.adjust.method = "bonferroni")
print(pairwise_tests)

# Extract significant pairs
bonferroni_matrix <- pairwise_tests$p.value
significant_bonferroni <- which(bonferroni_matrix < 0.05, arr.ind = TRUE)

if (nrow(significant_bonferroni) > 0) {
  cat("Significant differences (Bonferroni-corrected p < 0.05):\n")
  for (i in 1:nrow(significant_bonferroni)) {
    row_idx <- significant_bonferroni[i, 1]
    col_idx <- significant_bonferroni[i, 2]
    group1 <- rownames(bonferroni_matrix)[row_idx]
    group2 <- colnames(bonferroni_matrix)[col_idx]
    p_value <- bonferroni_matrix[row_idx, col_idx]
    cat(group1, "vs", group2, ": p =", round(p_value, 4), "\n")
  }
} else {
  cat("No significant differences found with Bonferroni correction.\n")
}
```

### Scheffe's Test

```r
# Function to perform Scheffe's test
scheffe_test <- function(anova_result, group_data, alpha = 0.05) {
  k <- length(group_data)
  n_per_group <- sapply(group_data, length)
  total_n <- sum(n_per_group)
  
  # Calculate critical value
  f_critical <- qf(1 - alpha, k - 1, total_n - k)
  scheffe_critical <- sqrt((k - 1) * f_critical)
  
  # Calculate group means
  group_means <- sapply(group_data, mean)
  
  # Perform all pairwise comparisons
  comparisons <- list()
  pair_count <- 1
  
  for (i in 1:(k-1)) {
    for (j in (i+1):k) {
      mean_diff <- group_means[i] - group_means[j]
      
      # Standard error for the difference
      se_diff <- sqrt(anova_result$ms_within * (1/n_per_group[i] + 1/n_per_group[j]))
      
      # Test statistic
      test_stat <- abs(mean_diff) / se_diff
      
      # p-value
      p_value <- 1 - pf(test_stat^2 / (k - 1), k - 1, total_n - k)
      
      comparisons[[pair_count]] <- list(
        group1 = i,
        group2 = j,
        mean_diff = mean_diff,
        test_stat = test_stat,
        p_value = p_value,
        significant = test_stat > scheffe_critical
      )
      
      pair_count <- pair_count + 1
    }
  }
  
  return(list(
    comparisons = comparisons,
    scheffe_critical = scheffe_critical,
    alpha = alpha
  ))
}

# Apply Scheffe's test
scheffe_result <- scheffe_test(anova_result, list(mpg_4cyl, mpg_6cyl, mpg_8cyl))

cat("Scheffe's Test Results:\n")
cat("Critical value:", round(scheffe_result$scheffe_critical, 3), "\n\n")

for (comp in scheffe_result$comparisons) {
  group_names <- c("4-cylinder", "6-cylinder", "8-cylinder")
  cat(group_names[comp$group1], "vs", group_names[comp$group2], ":\n")
  cat("  Mean difference:", round(comp$mean_diff, 3), "\n")
  cat("  Test statistic:", round(comp$test_stat, 3), "\n")
  cat("  p-value:", round(comp$p_value, 4), "\n")
  cat("  Significant:", comp$significant, "\n\n")
}
```

## Assumption Checking

ANOVA relies on several key assumptions. Violations can affect the validity of results and may require alternative approaches. Comprehensive assumption checking is essential for robust statistical inference.

### Mathematical Foundation

**ANOVA Assumptions:**

1. **Independence:** Observations are independent within and between groups
2. **Normality:** Error terms follow a normal distribution
3. **Homoscedasticity:** Error terms have constant variance across groups
4. **Linearity:** Effects are additive

**Test Statistics:**
- **Shapiro-Wilk:** $W = \frac{(\sum_{i=1}^{n} a_i x_{(i)})^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$
- **Levene's Test:** $W = \frac{(N-k)\sum_{i=1}^{k} n_i(\bar{Z}_i - \bar{Z})^2}{(k-1)\sum_{i=1}^{k}\sum_{j=1}^{n_i}(Z_{ij} - \bar{Z}_i)^2}$
- **Bartlett's Test:** $\chi^2 = \frac{(N-k)\ln(s_p^2) - \sum_{i=1}^{k}(n_i-1)\ln(s_i^2)}{1 + \frac{1}{3(k-1)}(\sum_{i=1}^{k}\frac{1}{n_i-1} - \frac{1}{N-k})}$

### Comprehensive Assumption Checking

```r
# Comprehensive assumption checking function
comprehensive_assumption_check <- function(data, group_var, response_var, alpha = 0.05) {
  cat("=== COMPREHENSIVE ANOVA ASSUMPTION CHECKING ===\n")
  
  # Get basic information
  groups <- unique(data[[group_var]])
  k <- length(groups)
  n_per_group <- sapply(groups, function(g) sum(data[[group_var]] == g))
  total_n <- sum(n_per_group)
  
  cat("Number of groups:", k, "\n")
  cat("Sample sizes per group:", n_per_group, "\n")
  cat("Total sample size:", total_n, "\n\n")
  
  # 1. Independence Assessment
  cat("=== 1. INDEPENDENCE ASSESSMENT ===\n")
  
  # Check for balanced design
  balanced_design <- length(unique(n_per_group)) == 1
  cat("Balanced design:", balanced_design, "\n")
  
  # Check for random sampling (simulation-based)
  set.seed(123)
  independence_test <- function(data, group_var, response_var) {
    # Create a simple test for independence
    model <- aov(as.formula(paste(response_var, "~", group_var)), data = data)
    residuals <- residuals(model)
    
    # Test for autocorrelation in residuals
    acf_result <- acf(residuals, plot = FALSE, lag.max = 1)
    lag1_corr <- acf_result$acf[2]
    
    return(list(
      lag1_correlation = lag1_corr,
      independent = abs(lag1_corr) < 0.3
    ))
  }
  
  independence_result <- independence_test(data, group_var, response_var)
  cat("Lag-1 autocorrelation:", round(independence_result$lag1_correlation, 3), "\n")
  cat("Independence assumption met:", independence_result$independent, "\n\n")
  
  # 2. Normality Assessment
  cat("=== 2. NORMALITY ASSESSMENT ===\n")
  
  # Shapiro-Wilk test for each group
  normality_results <- list()
  all_normal <- TRUE
  
  for (group in groups) {
    group_data <- data[[response_var]][data[[group_var]] == group]
    shapiro_result <- shapiro.test(group_data)
    normality_results[[as.character(group)]] <- shapiro_result
    
    cat("Group", group, "Shapiro-Wilk:\n")
    cat("  W =", round(shapiro_result$statistic, 4), "\n")
    cat("  p-value =", round(shapiro_result$p.value, 4), "\n")
    cat("  Normal:", shapiro_result$p.value >= alpha, "\n\n")
    
    if (shapiro_result$p.value < alpha) {
      all_normal <- FALSE
    }
  }
  
  # Overall normality test on residuals
  model <- aov(as.formula(paste(response_var, "~", group_var)), data = data)
  residuals <- residuals(model)
  overall_shapiro <- shapiro.test(residuals)
  
  cat("Overall residuals Shapiro-Wilk:\n")
  cat("  W =", round(overall_shapiro$statistic, 4), "\n")
  cat("  p-value =", round(overall_shapiro$p.value, 4), "\n")
  cat("  Normal:", overall_shapiro$p.value >= alpha, "\n\n")
  
  # Additional normality tests
  library(nortest)
  
  # Anderson-Darling test
  ad_test <- ad.test(residuals)
  cat("Anderson-Darling test:\n")
  cat("  A =", round(ad_test$statistic, 4), "\n")
  cat("  p-value =", round(ad_test$p.value, 4), "\n")
  cat("  Normal:", ad_test$p.value >= alpha, "\n\n")
  
  # Kolmogorov-Smirnov test
  ks_test <- ks.test(residuals, "pnorm", mean = mean(residuals), sd = sd(residuals))
  cat("Kolmogorov-Smirnov test:\n")
  cat("  D =", round(ks_test$statistic, 4), "\n")
  cat("  p-value =", round(ks_test$p.value, 4), "\n")
  cat("  Normal:", ks_test$p.value >= alpha, "\n\n")
  
  # 3. Homoscedasticity Assessment
  cat("=== 3. HOMOSCEDASTICITY ASSESSMENT ===\n")
  
  # Levene's test
  library(car)
  levene_result <- leveneTest(as.formula(paste(response_var, "~", group_var)), data = data)
  cat("Levene's test:\n")
  cat("  F =", round(levene_result$`F value`[1], 4), "\n")
  cat("  p-value =", round(levene_result$`Pr(>F)`[1], 4), "\n")
  cat("  Equal variances:", levene_result$`Pr(>F)`[1] >= alpha, "\n\n")
  
  # Bartlett's test
  bartlett_result <- bartlett.test(as.formula(paste(response_var, "~", group_var)), data = data)
  cat("Bartlett's test:\n")
  cat("  Chi-squared =", round(bartlett_result$statistic, 4), "\n")
  cat("  p-value =", round(bartlett_result$p.value, 4), "\n")
  cat("  Equal variances:", bartlett_result$p.value >= alpha, "\n\n")
  
  # Fligner-Killeen test (more robust)
  fligner_result <- fligner.test(as.formula(paste(response_var, "~", group_var)), data = data)
  cat("Fligner-Killeen test:\n")
  cat("  Chi-squared =", round(fligner_result$statistic, 4), "\n")
  cat("  p-value =", round(fligner_result$p.value, 4), "\n")
  cat("  Equal variances:", fligner_result$p.value >= alpha, "\n\n")
  
  # Variance ratio analysis
  group_variances <- sapply(groups, function(g) var(data[[response_var]][data[[group_var]] == g]))
  max_var <- max(group_variances)
  min_var <- min(group_variances)
  var_ratio <- max_var / min_var
  
  cat("Variance analysis:\n")
  cat("  Group variances:", round(group_variances, 3), "\n")
  cat("  Variance ratio (max/min):", round(var_ratio, 3), "\n")
  cat("  Acceptable ratio (< 4):", var_ratio < 4, "\n\n")
  
  # 4. Linearity Assessment
  cat("=== 4. LINEARITY ASSESSMENT ===\n")
  
  # Check for interaction effects (not applicable in one-way ANOVA)
  cat("Linearity assumption is inherent in one-way ANOVA design.\n")
  cat("No interaction effects to test.\n\n")
  
  # 5. Outlier Detection
  cat("=== 5. OUTLIER DETECTION ===\n")
  
  outliers_by_group <- list()
  total_outliers <- 0
  
  for (group in groups) {
    group_data <- data[[response_var]][data[[group_var]] == group]
    
    # IQR method
    q1 <- quantile(group_data, 0.25)
    q3 <- quantile(group_data, 0.75)
    iqr <- q3 - q1
    lower_bound <- q1 - 1.5 * iqr
    upper_bound <- q3 + 1.5 * iqr
    
    outliers <- group_data < lower_bound | group_data > upper_bound
    outlier_indices <- which(outliers)
    outliers_by_group[[as.character(group)]] <- outlier_indices
    
    cat("Group", group, "outliers (IQR method):\n")
    cat("  Number of outliers:", length(outlier_indices), "\n")
    if (length(outlier_indices) > 0) {
      cat("  Outlier values:", round(group_data[outlier_indices], 3), "\n")
    }
    cat("\n")
    
    total_outliers <- total_outliers + length(outlier_indices)
  }
  
  # 6. Comprehensive Assessment
  cat("=== 6. COMPREHENSIVE ASSESSMENT ===\n")
  
  # Count assumption violations
  violations <- 0
  violation_details <- c()
  
  if (!independence_result$independent) {
    violations <- violations + 1
    violation_details <- c(violation_details, "Independence")
  }
  
  if (!all_normal || overall_shapiro$p.value < alpha) {
    violations <- violations + 1
    violation_details <- c(violation_details, "Normality")
  }
  
  if (levene_result$`Pr(>F)`[1] < alpha || var_ratio >= 4) {
    violations <- violations + 1
    violation_details <- c(violation_details, "Homoscedasticity")
  }
  
  if (total_outliers > total_n * 0.1) {
    violations <- violations + 1
    violation_details <- c(violation_details, "Outliers")
  }
  
  cat("Total assumption violations:", violations, "\n")
  if (violations > 0) {
    cat("Violated assumptions:", paste(violation_details, collapse = ", "), "\n")
  }
  
  # Recommendations
  cat("\n=== RECOMMENDATIONS ===\n")
  
  if (violations == 0) {
    cat("✓ All assumptions met - standard ANOVA is appropriate\n")
  } else {
    cat("⚠️  Some assumptions violated - consider alternatives:\n")
    
    if ("Normality" %in% violation_details) {
      cat("- Use Kruskal-Wallis test for non-normal data\n")
      cat("- Consider data transformation\n")
    }
    
    if ("Homoscedasticity" %in% violation_details) {
      cat("- Use Welch's ANOVA for unequal variances\n")
      cat("- Consider robust ANOVA methods\n")
    }
    
    if ("Outliers" %in% violation_details) {
      cat("- Investigate outliers for data entry errors\n")
      cat("- Consider robust statistical methods\n")
      cat("- Report results with and without outliers\n")
    }
    
    if ("Independence" %in% violation_details) {
      cat("- Check study design for independence violations\n")
      cat("- Consider mixed-effects models if appropriate\n")
    }
  }
  
  return(list(
    independence = independence_result,
    normality = list(group_tests = normality_results, overall = overall_shapiro, ad = ad_test, ks = ks_test),
    homoscedasticity = list(levene = levene_result, bartlett = bartlett_result, fligner = fligner_result, var_ratio = var_ratio),
    outliers = outliers_by_group,
    violations = violations,
    violation_details = violation_details
  ))
}

# Apply comprehensive assumption checking
assumption_results <- comprehensive_assumption_check(mtcars, "cyl_factor", "mpg")
```

### Homogeneity of Variance

```r
# Function to test homogeneity of variance
check_homogeneity_anova <- function(data, group_var, response_var) {
  cat("=== HOMOGENEITY OF VARIANCE TESTS ===\n")
  
  # Levene's test
  library(car)
  levene_result <- leveneTest(as.formula(paste(response_var, "~", group_var)), data = data)
  cat("Levene's test p-value:", round(levene_result$`Pr(>F)`[1], 4), "\n")
  
  # Bartlett's test
  bartlett_result <- bartlett.test(as.formula(paste(response_var, "~", group_var)), data = data)
  cat("Bartlett's test p-value:", round(bartlett_result$p.value, 4), "\n")
  
  # Fligner-Killeen test (more robust)
  fligner_result <- fligner.test(as.formula(paste(response_var, "~", group_var)), data = data)
  cat("Fligner-Killeen test p-value:", round(fligner_result$p.value, 4), "\n")
  
  # Recommendations
  cat("\nRECOMMENDATIONS:\n")
  if (levene_result$`Pr(>F)`[1] >= 0.05) {
    cat("- Variances appear equal across groups\n")
    cat("- Standard ANOVA is appropriate\n")
  } else {
    cat("- Variances are significantly different\n")
    cat("- Consider Welch's ANOVA or nonparametric alternatives\n")
  }
  
  return(list(
    levene = levene_result,
    bartlett = bartlett_result,
    fligner = fligner_result
  ))
}

# Check homogeneity for cylinder groups
homogeneity_results <- check_homogeneity_anova(mtcars, "cyl_factor", "mpg")
```

### Independence and Random Sampling

```r
# Function to check independence assumption
check_independence_anova <- function(data, group_var, response_var) {
  cat("=== INDEPENDENCE ASSESSMENT ===\n")
  
  # Check for balanced design
  group_counts <- table(data[[group_var]])
  cat("Group sample sizes:\n")
  print(group_counts)
  
  # Check for equal sample sizes
  n_groups <- length(group_counts)
  equal_sizes <- length(unique(group_counts)) == 1
  
  if (equal_sizes) {
    cat("Design is balanced (equal sample sizes)\n")
  } else {
    cat("Design is unbalanced (unequal sample sizes)\n")
  }
  
  # Check for outliers
  outliers_by_group <- list()
  for (group in unique(data[[group_var]])) {
    group_data <- data[[response_var]][data[[group_var]] == group]
    
    # IQR method
    q1 <- quantile(group_data, 0.25)
    q3 <- quantile(group_data, 0.75)
    iqr <- q3 - q1
    lower_bound <- q1 - 1.5 * iqr
    upper_bound <- q3 + 1.5 * iqr
    
    outliers <- group_data < lower_bound | group_data > upper_bound
    outliers_by_group[[as.character(group)]] <- which(outliers)
  }
  
  cat("Outliers by group (IQR method):\n")
  for (group in names(outliers_by_group)) {
    n_outliers <- length(outliers_by_group[[group]])
    cat("Group", group, ":", n_outliers, "outliers\n")
  }
  
  return(list(
    group_counts = group_counts,
    balanced = equal_sizes,
    outliers = outliers_by_group
  ))
}

# Check independence for cylinder groups
independence_results <- check_independence_anova(mtcars, "cyl_factor", "mpg")
```

## Nonparametric Alternatives

When ANOVA assumptions are violated, nonparametric alternatives provide robust statistical inference without requiring normality or equal variances. These methods are based on ranks rather than actual values.

### Mathematical Foundation

**Kruskal-Wallis Test:**
The test statistic is:
```math
H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)
```

where:
- $R_i$ = sum of ranks for group $i$
- $n_i$ = sample size of group $i$
- $N$ = total sample size
- $k$ = number of groups

Under the null hypothesis, $H$ follows a chi-squared distribution with $k-1$ degrees of freedom.

**Effect Size (Epsilon-squared):**
```math
\varepsilon^2 = \frac{H - (k-1)}{N - k}
```

**Mann-Whitney U Test (for pairwise comparisons):**
```math
U = n_1n_2 + \frac{n_1(n_1+1)}{2} - R_1
```

### Comprehensive Nonparametric Analysis

```r
# Comprehensive nonparametric analysis function
comprehensive_nonparametric <- function(data, group_var, response_var, alpha = 0.05) {
  cat("=== COMPREHENSIVE NONPARAMETRIC ANALYSIS ===\n")
  
  # Get basic information
  groups <- unique(data[[group_var]])
  k <- length(groups)
  n_per_group <- sapply(groups, function(g) sum(data[[group_var]] == g))
  total_n <- sum(n_per_group)
  
  cat("Number of groups:", k, "\n")
  cat("Sample sizes per group:", n_per_group, "\n")
  cat("Total sample size:", total_n, "\n\n")
  
  # 1. Kruskal-Wallis Test
  cat("=== 1. KRUSKAL-WALLIS TEST ===\n")
  kruskal_result <- kruskal.test(as.formula(paste(response_var, "~", group_var)), data = data)
  print(kruskal_result)
  
  # Calculate effect size
  h_statistic <- kruskal_result$statistic
  epsilon_squared <- (h_statistic - (k - 1)) / (total_n - k)
  
  cat("\nEffect size analysis:\n")
  cat("H-statistic:", round(h_statistic, 3), "\n")
  cat("Epsilon-squared (ε²):", round(epsilon_squared, 3), "\n")
  cat("p-value:", round(kruskal_result$p.value, 4), "\n")
  cat("Significant:", kruskal_result$p.value < alpha, "\n\n")
  
  # 2. Rank-based descriptive statistics
  cat("=== 2. RANK-BASED DESCRIPTIVE STATISTICS ===\n")
  
  # Calculate ranks for the entire dataset
  data$ranks <- rank(data[[response_var]])
  
  rank_stats <- data %>%
    group_by(!!sym(group_var)) %>%
    summarise(
      n = n(),
      mean_rank = mean(ranks, na.rm = TRUE),
      median_rank = median(ranks, na.rm = TRUE),
      min_rank = min(ranks, na.rm = TRUE),
      max_rank = max(ranks, na.rm = TRUE),
      rank_sum = sum(ranks, na.rm = TRUE)
    )
  
  print(rank_stats)
  
  # 3. Pairwise Mann-Whitney U tests
  cat("\n=== 3. PAIRWISE MANN-WHITNEY U TESTS ===\n")
  
  # Perform all pairwise comparisons
  pairwise_results <- list()
  pair_count <- 1
  
  for (i in 1:(k-1)) {
    for (j in (i+1):k) {
      group1 <- groups[i]
      group2 <- groups[j]
      
      data1 <- data[[response_var]][data[[group_var]] == group1]
      data2 <- data[[response_var]][data[[group_var]] == group2]
      
      # Mann-Whitney U test
      mw_result <- wilcox.test(data1, data2, alternative = "two.sided")
      
      # Calculate effect size (r)
      z_stat <- qnorm(mw_result$p.value / 2)
      n1 <- length(data1)
      n2 <- length(data2)
      r_effect_size <- abs(z_stat) / sqrt(n1 + n2)
      
      pairwise_results[[pair_count]] <- list(
        group1 = group1,
        group2 = group2,
        u_statistic = mw_result$statistic,
        p_value = mw_result$p.value,
        effect_size_r = r_effect_size,
        significant = mw_result$p.value < alpha
      )
      
      cat("Comparison:", group1, "vs", group2, "\n")
      cat("  U-statistic:", round(mw_result$statistic, 3), "\n")
      cat("  p-value:", round(mw_result$p.value, 4), "\n")
      cat("  Effect size (r):", round(r_effect_size, 3), "\n")
      cat("  Significant:", mw_result$p.value < alpha, "\n\n")
      
      pair_count <- pair_count + 1
    }
  }
  
  # 4. Dunn's test (with multiple corrections)
  cat("=== 4. DUNN'S TEST ===\n")
  library(dunn.test)
  
  # Perform Dunn's test with different correction methods
  dunn_bonferroni <- dunn.test(data[[response_var]], data[[group_var]], 
                              method = "bonferroni")
  dunn_holm <- dunn.test(data[[response_var]], data[[group_var]], 
                        method = "holm")
  dunn_fdr <- dunn.test(data[[response_var]], data[[group_var]], 
                       method = "fdr")
  
  cat("Dunn's test with Bonferroni correction:\n")
  print(dunn_bonferroni)
  
  cat("\nDunn's test with Holm correction:\n")
  print(dunn_holm)
  
  cat("\nDunn's test with FDR correction:\n")
  print(dunn_fdr)
  
  # 5. Comparison with parametric ANOVA
  cat("\n=== 5. COMPARISON WITH PARAMETRIC ANOVA ===\n")
  
  # Perform parametric ANOVA for comparison
  anova_model <- aov(as.formula(paste(response_var, "~", group_var)), data = data)
  anova_summary <- summary(anova_model)
  f_stat <- anova_summary[[1]]$`F value`[1]
  anova_p_value <- anova_summary[[1]]$`Pr(>F)`[1]
  
  cat("Parametric ANOVA results:\n")
  cat("  F-statistic:", round(f_stat, 3), "\n")
  cat("  p-value:", round(anova_p_value, 4), "\n")
  cat("  Significant:", anova_p_value < alpha, "\n\n")
  
  cat("Nonparametric Kruskal-Wallis results:\n")
  cat("  H-statistic:", round(h_statistic, 3), "\n")
  cat("  p-value:", round(kruskal_result$p.value, 4), "\n")
  cat("  Significant:", kruskal_result$p.value < alpha, "\n\n")
  
  # Agreement analysis
  cat("=== 6. AGREEMENT ANALYSIS ===\n")
  anova_sig <- anova_p_value < alpha
  kruskal_sig <- kruskal_result$p.value < alpha
  
  cat("Agreement between parametric and nonparametric tests:\n")
  cat("  ANOVA significant:", anova_sig, "\n")
  cat("  Kruskal-Wallis significant:", kruskal_sig, "\n")
  cat("  Agreement:", anova_sig == kruskal_sig, "\n")
  
  if (anova_sig == kruskal_sig) {
    cat("  ✓ Both tests agree on significance\n")
  } else {
    cat("  ⚠️  Tests disagree - check assumptions carefully\n")
  }
  
  # 6. Effect size comparison
  cat("\n=== 7. EFFECT SIZE COMPARISON ===\n")
  
  # Calculate eta-squared for ANOVA
  ss_between <- anova_summary[[1]]$`Sum Sq`[1]
  ss_total <- sum(anova_summary[[1]]$`Sum Sq`)
  eta_squared <- ss_between / ss_total
  
  cat("Effect size comparison:\n")
  cat("  ANOVA eta-squared (η²):", round(eta_squared, 3), "\n")
  cat("  Kruskal-Wallis epsilon-squared (ε²):", round(epsilon_squared, 3), "\n")
  cat("  Difference:", round(abs(eta_squared - epsilon_squared), 3), "\n")
  
  # 7. Recommendations
  cat("\n=== 8. RECOMMENDATIONS ===\n")
  
  if (kruskal_result$p.value < alpha) {
    cat("✓ Kruskal-Wallis test is significant\n")
    cat("  - There are significant differences between groups\n")
    cat("  - Use Dunn's test for pairwise comparisons\n")
  } else {
    cat("✗ Kruskal-Wallis test is not significant\n")
    cat("  - No evidence of group differences\n")
  }
  
  if (abs(eta_squared - epsilon_squared) < 0.05) {
    cat("✓ Effect sizes are similar\n")
    cat("  - Both parametric and nonparametric approaches are valid\n")
  } else {
    cat("⚠️  Effect sizes differ substantially\n")
    cat("  - Consider which approach better fits your data\n")
  }
  
  return(list(
    kruskal_wallis = kruskal_result,
    epsilon_squared = epsilon_squared,
    rank_stats = rank_stats,
    pairwise_tests = pairwise_results,
    dunn_bonferroni = dunn_bonferroni,
    dunn_holm = dunn_holm,
    dunn_fdr = dunn_fdr,
    anova_comparison = list(f_stat = f_stat, p_value = anova_p_value, eta_squared = eta_squared),
    agreement = (anova_sig == kruskal_sig)
  ))
}

# Apply comprehensive nonparametric analysis
nonparametric_results <- comprehensive_nonparametric(mtcars, "cyl_factor", "mpg")

# Additional robust methods
cat("\n=== ADDITIONAL ROBUST METHODS ===\n")

# Bootstrap confidence intervals for group differences
library(boot)

bootstrap_group_diff <- function(data, indices, group1, group2) {
  d <- data[indices, ]
  mean1 <- mean(d[[response_var]][d[[group_var]] == group1])
  mean2 <- mean(d[[response_var]][d[[group_var]] == group2])
  return(mean1 - mean2)
}

# Example bootstrap for 4-cylinder vs 8-cylinder
boot_result <- boot(mtcars, bootstrap_group_diff, R = 1000, 
                   group1 = "4-cylinder", group2 = "8-cylinder")
boot_ci <- boot.ci(boot_result, type = "perc")

cat("Bootstrap 95% CI for 4-cylinder vs 8-cylinder difference:\n")
cat("  Lower:", round(boot_ci$percent[4], 3), "\n")
cat("  Upper:", round(boot_ci$percent[5], 3), "\n")
```

### Post Hoc Tests for Nonparametric ANOVA

```r
# Dunn's test for Kruskal-Wallis
library(dunn.test)

dunn_result <- dunn.test(mtcars$mpg, mtcars$cyl_factor, method = "bonferroni")
print(dunn_result)

# Extract significant pairs
significant_dunn <- dunn_result$P.adjusted < 0.05
if (any(significant_dunn)) {
  cat("Significant differences (Dunn's test with Bonferroni correction):\n")
  significant_pairs <- dunn_result$comparisons[significant_dunn]
  significant_pvalues <- dunn_result$P.adjusted[significant_dunn]
  
  for (i in 1:length(significant_pairs)) {
    cat(significant_pairs[i], ": p =", round(significant_pvalues[i], 4), "\n")
  }
} else {
  cat("No significant differences found with Dunn's test.\n")
}
```

## Power Analysis

Power analysis helps determine the probability of detecting a true effect and guides sample size planning. It's essential for study design and interpreting results.

### Mathematical Foundation

**Power Definition:**
Power = $P(\text{Reject } H_0 | H_1 \text{ is true}) = 1 - \beta$

**Effect Size Measures:**
- **Cohen's f:** $f = \sqrt{\frac{\eta^2}{1 - \eta^2}}$
- **Eta-squared:** $\eta^2 = \frac{SS_{between}}{SS_{total}}$

**Noncentrality Parameter:**
```math
\lambda = n \sum_{i=1}^{k} \frac{(\mu_i - \bar{\mu})^2}{\sigma^2}
```

**Power Calculation:**
Power depends on the noncentral F-distribution:
```math
\text{Power} = P(F > F_{\alpha, k-1, N-k} | \lambda)
```

### Comprehensive Power Analysis

```r
# Comprehensive power analysis function
comprehensive_power_analysis <- function(data, group_var, response_var, alpha = 0.05) {
  cat("=== COMPREHENSIVE POWER ANALYSIS ===\n")
  
  # Get basic information
  groups <- unique(data[[group_var]])
  k <- length(groups)
  n_per_group <- sapply(groups, function(g) sum(data[[group_var]] == g))
  total_n <- sum(n_per_group)
  
  cat("Study design:\n")
  cat("  Number of groups (k):", k, "\n")
  cat("  Sample sizes per group:", n_per_group, "\n")
  cat("  Total sample size (N):", total_n, "\n")
  cat("  Significance level (α):", alpha, "\n\n")
  
  # 1. Calculate observed effect size
  cat("=== 1. OBSERVED EFFECT SIZE ===\n")
  
  # Perform ANOVA to get effect size
  anova_model <- aov(as.formula(paste(response_var, "~", group_var)), data = data)
  anova_summary <- summary(anova_model)
  
  ss_between <- anova_summary[[1]]$`Sum Sq`[1]
  ss_total <- sum(anova_summary[[1]]$`Sum Sq`)
  eta_squared <- ss_between / ss_total
  
  # Calculate Cohen's f
  f_effect_size <- sqrt(eta_squared / (1 - eta_squared))
  
  cat("Observed effect sizes:\n")
  cat("  Eta-squared (η²):", round(eta_squared, 3), "\n")
  cat("  Cohen's f:", round(f_effect_size, 3), "\n")
  
  # Interpret effect size
  if (f_effect_size < 0.10) {
    effect_interpretation <- "Small"
  } else if (f_effect_size < 0.25) {
    effect_interpretation <- "Medium"
  } else if (f_effect_size < 0.40) {
    effect_interpretation <- "Large"
  } else {
    effect_interpretation <- "Very large"
  }
  
  cat("  Interpretation:", effect_interpretation, "effect\n\n")
  
  # 2. Current power analysis
  cat("=== 2. CURRENT POWER ANALYSIS ===\n")
  
  library(pwr)
  
  # Calculate current power
  current_power <- pwr.anova.test(k = k, n = min(n_per_group), f = f_effect_size, 
                                 sig.level = alpha)$power
  
  cat("Current power analysis:\n")
  cat("  Effect size f:", round(f_effect_size, 3), "\n")
  cat("  Sample size per group:", min(n_per_group), "\n")
  cat("  Current power:", round(current_power, 3), "\n")
  cat("  Type II error rate (β):", round(1 - current_power, 3), "\n\n")
  
  # 3. Sample size planning
  cat("=== 3. SAMPLE SIZE PLANNING ===\n")
  
  # Calculate required sample sizes for different power levels
  power_levels <- c(0.8, 0.85, 0.9, 0.95)
  sample_size_results <- list()
  
  cat("Required sample sizes per group:\n")
  for (power_level in power_levels) {
    required_n <- pwr.anova.test(k = k, f = f_effect_size, sig.level = alpha, 
                                power = power_level)$n
    sample_size_results[[as.character(power_level)]] <- ceiling(required_n)
    
    cat("  Power", power_level, ":", ceiling(required_n), "per group\n")
  }
  
  # 4. Effect size sensitivity analysis
  cat("\n=== 4. EFFECT SIZE SENSITIVITY ANALYSIS ===\n")
  
  # Test different effect sizes
  effect_sizes_to_test <- c(0.1, 0.25, 0.4, 0.6)
  current_n <- min(n_per_group)
  
  cat("Power for different effect sizes (n =", current_n, "per group):\n")
  for (f_test in effect_sizes_to_test) {
    power_test <- pwr.anova.test(k = k, n = current_n, f = f_test, 
                                sig.level = alpha)$power
    cat("  f =", f_test, ":", round(power_test, 3), "\n")
  }
  
  # 5. Power curve analysis
  cat("\n=== 5. POWER CURVE ANALYSIS ===\n")
  
  # Generate power curve data
  sample_sizes <- seq(5, 50, by = 5)
  power_curve <- sapply(sample_sizes, function(n) {
    pwr.anova.test(k = k, n = n, f = f_effect_size, sig.level = alpha)$power
  })
  
  cat("Power curve (effect size f =", round(f_effect_size, 3), "):\n")
  for (i in 1:length(sample_sizes)) {
    cat("  n =", sample_sizes[i], "per group: power =", round(power_curve[i], 3), "\n")
  }
  
  # 6. Multiple comparison correction impact
  cat("\n=== 6. MULTIPLE COMPARISON IMPACT ===\n")
  
  # Number of pairwise comparisons
  m <- k * (k - 1) / 2
  
  # Bonferroni correction
  alpha_bonferroni <- alpha / m
  power_bonferroni <- pwr.anova.test(k = k, n = min(n_per_group), f = f_effect_size, 
                                    sig.level = alpha_bonferroni)$power
  
  cat("Multiple comparison impact:\n")
  cat("  Number of pairwise comparisons:", m, "\n")
  cat("  Bonferroni α:", round(alpha_bonferroni, 4), "\n")
  cat("  Power with Bonferroni correction:", round(power_bonferroni, 3), "\n")
  cat("  Power loss:", round(current_power - power_bonferroni, 3), "\n\n")
  
  # 7. Recommendations
  cat("=== 7. RECOMMENDATIONS ===\n")
  
  if (current_power >= 0.8) {
    cat("✓ Current power is adequate (≥ 0.8)\n")
    cat("  - Study has sufficient power to detect the observed effect\n")
  } else if (current_power >= 0.6) {
    cat("⚠️  Current power is moderate (0.6-0.8)\n")
    cat("  - Consider increasing sample size for better power\n")
  } else {
    cat("✗ Current power is low (< 0.6)\n")
    cat("  - Study may be underpowered\n")
    cat("  - Consider larger sample size or different design\n")
  }
  
  # Sample size recommendations
  if (current_power < 0.8) {
    required_n_80 <- sample_size_results[["0.8"]]
    cat("\nSample size recommendations:\n")
    cat("  For 80% power:", required_n_80, "per group\n")
    cat("  Total sample size needed:", required_n_80 * k, "\n")
    cat("  Additional participants needed:", (required_n_80 * k) - total_n, "\n")
  }
  
  # Effect size recommendations
  cat("\nEffect size considerations:\n")
  if (f_effect_size < 0.1) {
    cat("  - Very small effect size detected\n")
    cat("  - Consider if this effect is practically meaningful\n")
    cat("  - Large sample sizes may be needed for adequate power\n")
  } else if (f_effect_size > 0.4) {
    cat("  - Large effect size detected\n")
    cat("  - Current sample size likely provides adequate power\n")
    cat("  - Effect is likely to be practically significant\n")
  }
  
  return(list(
    observed_effect = list(eta_squared = eta_squared, f = f_effect_size),
    current_power = current_power,
    sample_size_planning = sample_size_results,
    power_curve = data.frame(n = sample_sizes, power = power_curve),
    multiple_comparison_impact = list(
      comparisons = m,
      bonferroni_alpha = alpha_bonferroni,
      bonferroni_power = power_bonferroni
    )
  ))
}

# Apply comprehensive power analysis
power_results <- comprehensive_power_analysis(mtcars, "cyl_factor", "mpg")

# Additional power analysis tools
cat("\n=== ADDITIONAL POWER ANALYSIS TOOLS ===\n")

# Monte Carlo power simulation
monte_carlo_power <- function(k, n_per_group, effect_size, alpha = 0.05, n_sim = 1000) {
  set.seed(123)
  
  # Simulate data and test power
  significant_tests <- 0
  
  for (i in 1:n_sim) {
    # Generate data with specified effect size
    group_means <- c(0, effect_size, effect_size * 2)  # Example means
    data_sim <- data.frame(
      y = c(rnorm(n_per_group, group_means[1], 1),
            rnorm(n_per_group, group_means[2], 1),
            rnorm(n_per_group, group_means[3], 1)),
      group = factor(rep(1:k, each = n_per_group))
    )
    
    # Perform ANOVA
    anova_result <- aov(y ~ group, data = data_sim)
    p_value <- summary(anova_result)[[1]]$`Pr(>F)`[1]
    
    if (p_value < alpha) {
      significant_tests <- significant_tests + 1
    }
  }
  
  return(significant_tests / n_sim)
}

# Example Monte Carlo power calculation
mc_power <- monte_carlo_power(k = 3, n_per_group = 10, effect_size = 0.5)
cat("Monte Carlo power simulation (n = 1000):\n")
cat("  Simulated power:", round(mc_power, 3), "\n")
cat("  Theoretical power:", round(pwr.anova.test(k = 3, n = 10, f = 0.5)$power, 3), "\n")
```

## Practical Examples

Real-world applications demonstrate how one-way ANOVA is used across different fields. These examples show the complete analysis workflow from data preparation to interpretation.

### Example 1: Educational Research

**Research Question:** Do different teaching methods affect student performance on standardized tests?

**Study Design:** Randomized controlled trial with three teaching methods (Traditional, Interactive, Technology-based)

```r
# Simulate comprehensive educational intervention data
set.seed(123)
n_per_group <- 25

# Generate realistic data for three teaching methods
# Method A: Traditional lecture-based (baseline)
# Method B: Interactive group learning (expected improvement)
# Method C: Technology-enhanced learning (moderate improvement)

method_a_scores <- rnorm(n_per_group, mean = 72, sd = 8)
method_b_scores <- rnorm(n_per_group, mean = 78, sd = 9)
method_c_scores <- rnorm(n_per_group, mean = 75, sd = 7)

# Create comprehensive data frame
education_data <- data.frame(
  score = c(method_a_scores, method_b_scores, method_c_scores),
  method = factor(rep(c("Traditional", "Interactive", "Technology"), each = n_per_group)),
  student_id = 1:(n_per_group * 3),
  study_time = c(rnorm(n_per_group, 5, 1), rnorm(n_per_group, 6, 1), rnorm(n_per_group, 5.5, 1))
)

# Add some realistic variation
education_data$score <- education_data$score + rnorm(nrow(education_data), 0, 2)

cat("=== EDUCATIONAL RESEARCH EXAMPLE ===\n")
cat("Research Question: Do different teaching methods affect student performance?\n")
cat("Sample size per group:", n_per_group, "\n")
cat("Total participants:", nrow(education_data), "\n\n")

# 1. Descriptive Statistics
cat("=== 1. DESCRIPTIVE STATISTICS ===\n")
desc_stats <- education_data %>%
  group_by(method) %>%
  summarise(
    n = n(),
    mean_score = mean(score, na.rm = TRUE),
    sd_score = sd(score, na.rm = TRUE),
    median_score = median(score, na.rm = TRUE),
    min_score = min(score, na.rm = TRUE),
    max_score = max(score, na.rm = TRUE),
    se_score = sd_score / sqrt(n)
  )

print(desc_stats)

# 2. Assumption Checking
cat("\n=== 2. ASSUMPTION CHECKING ===\n")
education_assumptions <- comprehensive_assumption_check(education_data, "method", "score")

# 3. One-Way ANOVA
cat("\n=== 3. ONE-WAY ANOVA ===\n")
education_anova <- aov(score ~ method, data = education_data)
print(summary(education_anova))

# Extract key statistics
education_ss <- summary(education_anova)[[1]]
f_stat <- education_ss$`F value`[1]
p_value <- education_ss$`Pr(>F)`[1]
eta_squared <- education_ss$`Sum Sq`[1] / sum(education_ss$`Sum Sq`)

# 4. Effect Size Analysis
cat("\n=== 4. EFFECT SIZE ANALYSIS ===\n")
f_effect_size <- sqrt(eta_squared / (1 - eta_squared))

cat("Effect size measures:\n")
cat("  Eta-squared (η²):", round(eta_squared, 3), "\n")
cat("  Cohen's f:", round(f_effect_size, 3), "\n")

# Interpret effect size
if (eta_squared < 0.06) {
  effect_interpretation <- "Small"
} else if (eta_squared < 0.14) {
  effect_interpretation <- "Medium"
} else {
  effect_interpretation <- "Large"
}
cat("  Interpretation:", effect_interpretation, "effect\n")

# 5. Post Hoc Analysis
cat("\n=== 5. POST HOC ANALYSIS ===\n")
education_tukey <- TukeyHSD(education_anova)
print(education_tukey)

# Extract significant differences
significant_pairs <- education_tukey$method[education_tukey$method[, "p adj"] < 0.05, ]
if (nrow(significant_pairs) > 0) {
  cat("\nSignificant pairwise differences:\n")
  for (i in 1:nrow(significant_pairs)) {
    row_name <- rownames(significant_pairs)[i]
    diff <- significant_pairs[i, "diff"]
    p_adj <- significant_pairs[i, "p adj"]
    cat("  ", row_name, ": diff =", round(diff, 2), ", p =", round(p_adj, 4), "\n")
  }
} else {
  cat("No significant pairwise differences found.\n")
}

# 6. Power Analysis
cat("\n=== 6. POWER ANALYSIS ===\n")
education_power <- comprehensive_power_analysis(education_data, "method", "score")

# 7. Visualization
cat("\n=== 7. VISUALIZATION ===\n")
library(ggplot2)

# Create comprehensive visualization
p1 <- ggplot(education_data, aes(x = method, y = score, fill = method)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white") +
  labs(title = "Student Performance by Teaching Method", 
       subtitle = "Boxes show IQR, lines show medians, diamonds show means",
       x = "Teaching Method", y = "Test Score") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")

p2 <- ggplot(education_data, aes(x = method, y = score, fill = method)) +
  geom_violin(alpha = 0.7) +
  geom_boxplot(width = 0.2, alpha = 0.8) +
  labs(title = "Distribution of Scores by Teaching Method", 
       x = "Teaching Method", y = "Test Score") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")

# Display plots
print(p1)
print(p2)

# 8. Comprehensive Results Summary
cat("\n=== 8. COMPREHENSIVE RESULTS SUMMARY ===\n")
cat("Educational Research Results:\n")
cat("  F-statistic:", round(f_stat, 3), "\n")
cat("  p-value:", round(p_value, 4), "\n")
cat("  Effect size (η²):", round(eta_squared, 3), "\n")
cat("  Effect size interpretation:", effect_interpretation, "\n")
cat("  Power:", round(education_power$current_power, 3), "\n")

# 9. Practical Interpretation
cat("\n=== 9. PRACTICAL INTERPRETATION ===\n")
if (p_value < 0.05) {
  cat("✓ Statistically significant differences found between teaching methods\n")
  cat("  - Teaching method significantly affects student performance\n")
  
  if (eta_squared >= 0.14) {
    cat("  - Large practical effect: teaching method has substantial impact\n")
  } else if (eta_squared >= 0.06) {
    cat("  - Medium practical effect: teaching method has moderate impact\n")
  } else {
    cat("  - Small practical effect: teaching method has limited impact\n")
  }
  
  # Identify best method
  best_method <- desc_stats$method[which.max(desc_stats$mean_score)]
  cat("  - Best performing method:", best_method, "\n")
  
} else {
  cat("✗ No statistically significant differences found\n")
  cat("  - Teaching method does not significantly affect performance\n")
  cat("  - Consider other factors that may influence student success\n")
}

# 10. Recommendations
cat("\n=== 10. RECOMMENDATIONS ===\n")
if (p_value < 0.05 && eta_squared >= 0.06) {
  cat("✓ Evidence supports the effectiveness of different teaching methods\n")
  cat("  - Consider implementing the best-performing method\n")
  cat("  - Conduct follow-up studies to confirm findings\n")
  cat("  - Consider cost-benefit analysis of different methods\n")
} else {
  cat("⚠️  Limited evidence for teaching method effectiveness\n")
  cat("  - Consider larger sample sizes for future studies\n")
  cat("  - Investigate other factors affecting student performance\n")
  cat("  - Consider qualitative research to understand student experiences\n")
}
```

### Example 2: Clinical Trial

**Research Question:** Do different drug treatments reduce blood pressure in hypertensive patients?

**Study Design:** Randomized controlled trial with three treatment groups (Placebo, Drug A, Drug B)

```r
# Simulate comprehensive clinical trial data
set.seed(456)
n_per_treatment <- 30

# Generate realistic clinical data
# Placebo: minimal effect (baseline)
# Drug A: moderate blood pressure reduction
# Drug B: strong blood pressure reduction

placebo_bp <- rnorm(n_per_treatment, mean = 145, sd = 12)
drug_a_bp <- rnorm(n_per_treatment, mean = 135, sd = 11)
drug_b_bp <- rnorm(n_per_treatment, mean = 125, sd = 10)

# Create comprehensive data frame
clinical_data <- data.frame(
  blood_pressure = c(placebo_bp, drug_a_bp, drug_b_bp),
  treatment = factor(rep(c("Placebo", "Drug A", "Drug B"), each = n_per_treatment)),
  patient_id = 1:(n_per_treatment * 3),
  age = c(rnorm(n_per_treatment, 55, 8), rnorm(n_per_treatment, 57, 9), rnorm(n_per_treatment, 54, 7)),
  baseline_bp = c(rnorm(n_per_treatment, 150, 15), rnorm(n_per_treatment, 148, 14), rnorm(n_per_treatment, 152, 16))
)

# Calculate change from baseline
clinical_data$bp_change <- clinical_data$baseline_bp - clinical_data$blood_pressure

cat("=== CLINICAL TRIAL EXAMPLE ===\n")
cat("Research Question: Do different drug treatments reduce blood pressure?\n")
cat("Sample size per group:", n_per_treatment, "\n")
cat("Total participants:", nrow(clinical_data), "\n\n")

# 1. Descriptive Statistics
cat("=== 1. DESCRIPTIVE STATISTICS ===\n")
clinical_desc <- clinical_data %>%
  group_by(treatment) %>%
  summarise(
    n = n(),
    mean_bp = mean(blood_pressure, na.rm = TRUE),
    sd_bp = sd(blood_pressure, na.rm = TRUE),
    mean_change = mean(bp_change, na.rm = TRUE),
    sd_change = sd(bp_change, na.rm = TRUE),
    se_bp = sd_bp / sqrt(n)
  )

print(clinical_desc)

# 2. Assumption Checking
cat("\n=== 2. ASSUMPTION CHECKING ===\n")
clinical_assumptions <- comprehensive_assumption_check(clinical_data, "treatment", "blood_pressure")

# 3. One-Way ANOVA
cat("\n=== 3. ONE-WAY ANOVA ===\n")
clinical_anova <- aov(blood_pressure ~ treatment, data = clinical_data)
print(summary(clinical_anova))

# Extract key statistics
clinical_ss <- summary(clinical_anova)[[1]]
clinical_f_stat <- clinical_ss$`F value`[1]
clinical_p_value <- clinical_ss$`Pr(>F)`[1]
clinical_eta_squared <- clinical_ss$`Sum Sq`[1] / sum(clinical_ss$`Sum Sq`)

# 4. Effect Size and Power
cat("\n=== 4. EFFECT SIZE AND POWER ===\n")
clinical_f_effect <- sqrt(clinical_eta_squared / (1 - clinical_eta_squared))
clinical_power <- comprehensive_power_analysis(clinical_data, "treatment", "blood_pressure")

# 5. Post Hoc Analysis
cat("\n=== 5. POST HOC ANALYSIS ===\n")
clinical_tukey <- TukeyHSD(clinical_anova)
print(clinical_tukey)

# 6. Nonparametric Alternative
cat("\n=== 6. NONPARAMETRIC ALTERNATIVE ===\n")
clinical_kruskal <- kruskal.test(blood_pressure ~ treatment, data = clinical_data)
print(clinical_kruskal)

# 7. Clinical Significance
cat("\n=== 7. CLINICAL SIGNIFICANCE ===\n")
# Calculate clinically meaningful differences (e.g., 5 mmHg reduction)
clinical_significance <- clinical_desc %>%
  mutate(
    clinically_meaningful = mean_change >= 5,
    effect_category = case_when(
      mean_change >= 10 ~ "Large clinical effect",
      mean_change >= 5 ~ "Moderate clinical effect",
      mean_change >= 2 ~ "Small clinical effect",
      TRUE ~ "No clinical effect"
    )
  )

print(clinical_significance)

# 8. Safety Analysis
cat("\n=== 8. SAFETY ANALYSIS ===\n")
# Check for adverse effects (hypotension)
clinical_data$hypotension <- clinical_data$blood_pressure < 90
safety_summary <- clinical_data %>%
  group_by(treatment) %>%
  summarise(
    n_hypotension = sum(hypotension),
    pct_hypotension = mean(hypotension) * 100
  )

cat("Safety analysis (hypotension < 90 mmHg):\n")
print(safety_summary)

# 9. Comprehensive Results
cat("\n=== 9. COMPREHENSIVE RESULTS ===\n")
cat("Clinical Trial Results:\n")
cat("  F-statistic:", round(clinical_f_stat, 3), "\n")
cat("  p-value:", round(clinical_p_value, 4), "\n")
cat("  Effect size (η²):", round(clinical_eta_squared, 3), "\n")
cat("  Power:", round(clinical_power$current_power, 3), "\n")

# 10. Clinical Recommendations
cat("\n=== 10. CLINICAL RECOMMENDATIONS ===\n")
if (clinical_p_value < 0.05) {
  best_treatment <- clinical_desc$treatment[which.min(clinical_desc$mean_bp)]
  cat("✓ Statistically significant treatment effects found\n")
  cat("  - Best treatment:", best_treatment, "\n")
  cat("  - Consider safety profile when choosing treatment\n")
  cat("  - Monitor for adverse effects\n")
} else {
  cat("✗ No statistically significant treatment effects\n")
  cat("  - Consider larger sample size or different endpoints\n")
  cat("  - Investigate patient compliance and adherence\n")
}
```

### Example 3: Quality Control

**Research Question:** Do different manufacturing machines produce products with consistent quality?

**Study Design:** Quality control study comparing three production machines

```r
# Simulate comprehensive quality control data
set.seed(789)
n_per_machine <- 20

# Generate realistic quality data
# Machine A: High precision, consistent output
# Machine B: Moderate precision, some variation
# Machine C: Lower precision, more variation

machine_a_output <- rnorm(n_per_machine, mean = 100, sd = 2)
machine_b_output <- rnorm(n_per_machine, mean = 98, sd = 3.5)
machine_c_output <- rnorm(n_per_machine, mean = 102, sd = 4.5)

# Create comprehensive data frame
quality_data <- data.frame(
  output = c(machine_a_output, machine_b_output, machine_c_output),
  machine = factor(rep(c("Machine A", "Machine B", "Machine C"), each = n_per_machine)),
  batch_id = 1:(n_per_machine * 3),
  temperature = c(rnorm(n_per_machine, 25, 2), rnorm(n_per_machine, 26, 3), rnorm(n_per_machine, 24, 2.5)),
  humidity = c(rnorm(n_per_machine, 50, 5), rnorm(n_per_machine, 52, 6), rnorm(n_per_machine, 48, 4))
)

# Add quality control specifications
quality_data$within_spec <- abs(quality_data$output - 100) <= 5
quality_data$defect_rate <- ifelse(quality_data$within_spec, 0, 1)

cat("=== QUALITY CONTROL EXAMPLE ===\n")
cat("Research Question: Do different machines produce consistent quality?\n")
cat("Sample size per machine:", n_per_machine, "\n")
cat("Total measurements:", nrow(quality_data), "\n\n")

# 1. Descriptive Statistics
cat("=== 1. DESCRIPTIVE STATISTICS ===\n")
quality_desc <- quality_data %>%
  group_by(machine) %>%
  summarise(
    n = n(),
    mean_output = mean(output, na.rm = TRUE),
    sd_output = sd(output, na.rm = TRUE),
    cv = sd_output / mean_output * 100,  # Coefficient of variation
    defect_rate = mean(defect_rate, na.rm = TRUE) * 100,
    within_spec_rate = mean(within_spec, na.rm = TRUE) * 100
  )

print(quality_desc)

# 2. Assumption Checking
cat("\n=== 2. ASSUMPTION CHECKING ===\n")
quality_assumptions <- comprehensive_assumption_check(quality_data, "machine", "output")

# 3. One-Way ANOVA
cat("\n=== 3. ONE-WAY ANOVA ===\n")
quality_anova <- aov(output ~ machine, data = quality_data)
print(summary(quality_anova))

# Extract key statistics
quality_ss <- summary(quality_anova)[[1]]
quality_f_stat <- quality_ss$`F value`[1]
quality_p_value <- quality_ss$`Pr(>F)`[1]
quality_eta_squared <- quality_ss$`Sum Sq`[1] / sum(quality_ss$`Sum Sq`)

# 4. Post Hoc Analysis
cat("\n=== 4. POST HOC ANALYSIS ===\n")
quality_tukey <- TukeyHSD(quality_anova)
print(quality_tukey)

# 5. Quality Control Analysis
cat("\n=== 5. QUALITY CONTROL ANALYSIS ===\n")
# Process capability analysis
quality_data$deviation_from_target <- quality_data$output - 100

capability_analysis <- quality_data %>%
  group_by(machine) %>%
  summarise(
    mean_deviation = mean(deviation_from_target),
    sd_deviation = sd(deviation_from_target),
    process_capability = 6 / (6 * sd_deviation),  # Cp index
    process_capability_centered = (6 - abs(mean_deviation)) / (6 * sd_deviation)  # Cpk index
  )

cat("Process capability analysis:\n")
print(capability_analysis)

# 6. Economic Impact Analysis
cat("\n=== 6. ECONOMIC IMPACT ANALYSIS ===\n")
# Calculate cost implications
quality_data$cost_per_unit <- ifelse(quality_data$within_spec, 10, 25)  # Defects cost more

economic_analysis <- quality_data %>%
  group_by(machine) %>%
  summarise(
    total_cost = sum(cost_per_unit),
    avg_cost_per_unit = mean(cost_per_unit),
    defect_cost = sum(cost_per_unit[!within_spec]),
    efficiency_score = (n() - sum(defect_rate)) / n() * 100
  )

cat("Economic analysis:\n")
print(economic_analysis)

# 7. Visualization
cat("\n=== 7. VISUALIZATION ===\n")
library(ggplot2)

# Create comprehensive quality control plots
p1 <- ggplot(quality_data, aes(x = machine, y = output, fill = machine)) +
  geom_boxplot(alpha = 0.7) +
  geom_hline(yintercept = 100, linetype = "dashed", color = "red") +
  geom_hline(yintercept = c(95, 105), linetype = "dotted", color = "orange") +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white") +
  labs(title = "Output Quality by Machine", 
       subtitle = "Red line = target, Orange lines = specification limits",
       x = "Machine", y = "Output Quality") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")

p2 <- ggplot(quality_data, aes(x = machine, y = deviation_from_target, fill = machine)) +
  geom_boxplot(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Deviation from Target by Machine", 
       x = "Machine", y = "Deviation from Target (100)") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_brewer(palette = "Set2")

# Display plots
print(p1)
print(p2)

# 8. Quality Control Recommendations
cat("\n=== 8. QUALITY CONTROL RECOMMENDATIONS ===\n")
best_machine <- quality_desc$machine[which.max(quality_desc$within_spec_rate)]
worst_machine <- quality_desc$machine[which.min(quality_desc$within_spec_rate)]

cat("Quality control recommendations:\n")
cat("  - Best performing machine:", best_machine, "\n")
cat("  - Machine needing improvement:", worst_machine, "\n")

if (quality_p_value < 0.05) {
  cat("  - Significant differences in machine performance detected\n")
  cat("  - Consider machine maintenance or replacement for underperforming units\n")
  cat("  - Implement quality control procedures for all machines\n")
} else {
  cat("  - No significant differences in machine performance\n")
  cat("  - All machines appear to be operating within acceptable parameters\n")
}

# 9. Process Improvement Suggestions
cat("\n=== 9. PROCESS IMPROVEMENT SUGGESTIONS ===\n")
cat("Process improvement recommendations:\n")
for (i in 1:nrow(quality_desc)) {
  machine <- quality_desc$machine[i]
  defect_rate <- quality_desc$defect_rate[i]
  
  if (defect_rate > 10) {
    cat("  - ", machine, ": High defect rate (", round(defect_rate, 1), "%) - needs immediate attention\n")
  } else if (defect_rate > 5) {
    cat("  - ", machine, ": Moderate defect rate (", round(defect_rate, 1), "%) - monitor closely\n")
  } else {
    cat("  - ", machine, ": Low defect rate (", round(defect_rate, 1), "%) - performing well\n")
  }
}
```

## Best Practices

Following best practices ensures robust statistical analysis and valid conclusions. These guidelines help researchers make appropriate methodological choices and interpret results correctly.

### Comprehensive Test Selection Guidelines

```r
# Comprehensive function to help choose appropriate ANOVA test
comprehensive_test_selection <- function(data, group_var, response_var, alpha = 0.05) {
  cat("=== COMPREHENSIVE ANOVA TEST SELECTION ===\n")
  
  # Get basic information
  groups <- unique(data[[group_var]])
  k <- length(groups)
  n_per_group <- sapply(groups, function(g) sum(data[[group_var]] == g))
  total_n <- sum(n_per_group)
  
  cat("Study characteristics:\n")
  cat("  Number of groups:", k, "\n")
  cat("  Sample sizes per group:", n_per_group, "\n")
  cat("  Total sample size:", total_n, "\n")
  cat("  Significance level:", alpha, "\n\n")
  
  # 1. Check all assumptions comprehensively
  cat("=== 1. ASSUMPTION ASSESSMENT ===\n")
  
  assumption_results <- comprehensive_assumption_check(data, group_var, response_var)
  
  # 2. Evaluate assumption violations
  cat("\n=== 2. ASSUMPTION VIOLATION EVALUATION ===\n")
  
  violations <- assumption_results$violations
  violation_details <- assumption_results$violation_details
  
  cat("Total assumption violations:", violations, "\n")
  if (violations > 0) {
    cat("Violated assumptions:", paste(violation_details, collapse = ", "), "\n")
  }
  
  # 3. Sample size considerations
  cat("\n=== 3. SAMPLE SIZE CONSIDERATIONS ===\n")
  
  min_n <- min(n_per_group)
  max_n <- max(n_per_group)
  balanced <- length(unique(n_per_group)) == 1
  
  cat("Sample size analysis:\n")
  cat("  Minimum sample size per group:", min_n, "\n")
  cat("  Maximum sample size per group:", max_n, "\n")
  cat("  Balanced design:", balanced, "\n")
  
  if (min_n < 10) {
    cat("  ⚠️  Small sample sizes - consider nonparametric alternatives\n")
  } else if (min_n < 30) {
    cat("  ⚠️  Moderate sample sizes - check normality carefully\n")
  } else {
    cat("  ✓ Adequate sample sizes for parametric tests\n")
  }
  
  # 4. Effect size considerations
  cat("\n=== 4. EFFECT SIZE CONSIDERATIONS ===\n")
  
  # Perform ANOVA to get effect size
  anova_model <- aov(as.formula(paste(response_var, "~", group_var)), data = data)
  anova_summary <- summary(anova_model)
  eta_squared <- anova_summary[[1]]$`Sum Sq`[1] / sum(anova_summary[[1]]$`Sum Sq`)
  
  cat("Observed effect size (η²):", round(eta_squared, 3), "\n")
  
  if (eta_squared < 0.01) {
    cat("  ⚠️  Very small effect size - consider practical significance\n")
  } else if (eta_squared < 0.06) {
    cat("  ⚠️  Small effect size - may not be practically meaningful\n")
  } else if (eta_squared < 0.14) {
    cat("  ✓ Medium effect size - likely practically meaningful\n")
  } else {
    cat("  ✓ Large effect size - clearly practically meaningful\n")
  }
  
  # 5. Decision tree for test selection
  cat("\n=== 5. TEST SELECTION DECISION TREE ===\n")
  
  # Check specific assumptions
  normality_ok <- all(sapply(assumption_results$normality$group_tests, function(x) x$p.value >= alpha))
  homogeneity_ok <- assumption_results$homoscedasticity$levene$`Pr(>F)`[1] >= alpha
  independence_ok <- assumption_results$independence$independent
  outliers_ok <- length(unlist(assumption_results$outliers)) <= total_n * 0.1
  
  cat("Assumption status:\n")
  cat("  Normality:", normality_ok, "\n")
  cat("  Homogeneity of variance:", homogeneity_ok, "\n")
  cat("  Independence:", independence_ok, "\n")
  cat("  Outliers acceptable:", outliers_ok, "\n")
  
  # Decision logic
  cat("\n=== 6. RECOMMENDED APPROACH ===\n")
  
  if (normality_ok && homogeneity_ok && independence_ok && outliers_ok) {
    cat("✓ RECOMMENDATION: Standard one-way ANOVA\n")
    cat("  - All assumptions are met\n")
    cat("  - Most powerful parametric approach\n")
    cat("  - Use Tukey's HSD for post hoc tests\n")
  } else if (normality_ok && !homogeneity_ok && independence_ok) {
    cat("✓ RECOMMENDATION: Welch's ANOVA\n")
    cat("  - Data is normal but variances are unequal\n")
    cat("  - More robust to variance heterogeneity\n")
    cat("  - Use Games-Howell for post hoc tests\n")
  } else if (!normality_ok && min_n >= 15) {
    cat("✓ RECOMMENDATION: Kruskal-Wallis test\n")
    cat("  - Data is not normally distributed\n")
    cat("  - Nonparametric alternative\n")
    cat("  - Use Dunn's test for post hoc comparisons\n")
  } else if (min_n < 15) {
    cat("✓ RECOMMENDATION: Kruskal-Wallis test\n")
    cat("  - Small sample sizes\n")
    cat("  - Nonparametric approach is more robust\n")
    cat("  - Use Dunn's test for post hoc comparisons\n")
  } else {
    cat("⚠️  RECOMMENDATION: Multiple approaches\n")
    cat("  - Complex assumption violations\n")
    cat("  - Consider both parametric and nonparametric approaches\n")
    cat("  - Report results from both methods\n")
  }
  
  # 7. Additional considerations
  cat("\n=== 7. ADDITIONAL CONSIDERATIONS ===\n")
  
  # Power analysis
  f_effect_size <- sqrt(eta_squared / (1 - eta_squared))
  current_power <- pwr.anova.test(k = k, n = min_n, f = f_effect_size, sig.level = alpha)$power
  
  cat("Power analysis:\n")
  cat("  Current power:", round(current_power, 3), "\n")
  
  if (current_power < 0.8) {
    cat("  ⚠️  Low power - consider larger sample size\n")
    required_n <- pwr.anova.test(k = k, f = f_effect_size, sig.level = alpha, power = 0.8)$n
    cat("  Required sample size per group for 80% power:", ceiling(required_n), "\n")
  } else {
    cat("  ✓ Adequate power for detecting effects\n")
  }
  
  # Multiple comparison considerations
  m <- k * (k - 1) / 2
  cat("\nMultiple comparison considerations:\n")
  cat("  Number of pairwise comparisons:", m, "\n")
  cat("  Family-wise error rate without correction:", round(1 - (1 - alpha)^m, 4), "\n")
  
  if (m > 6) {
    cat("  ⚠️  Many comparisons - use conservative correction methods\n")
  } else {
    cat("  ✓ Reasonable number of comparisons\n")
  }
  
  # 8. Reporting recommendations
  cat("\n=== 8. REPORTING RECOMMENDATIONS ===\n")
  
  cat("Essential elements to report:\n")
  cat("  - Descriptive statistics for each group\n")
  cat("  - Assumption checking results\n")
  cat("  - Test statistic and p-value\n")
  cat("  - Effect size measures\n")
  cat("  - Post hoc test results (if significant)\n")
  cat("  - Power analysis results\n")
  cat("  - Practical significance assessment\n")
  
  return(list(
    assumptions = assumption_results,
    effect_size = eta_squared,
    power = current_power,
    violations = violations,
    violation_details = violation_details,
    recommendation = ifelse(normality_ok && homogeneity_ok && independence_ok && outliers_ok,
                           "Standard ANOVA",
                           ifelse(normality_ok && !homogeneity_ok && independence_ok,
                                  "Welch's ANOVA",
                                  "Kruskal-Wallis"))
  ))
}

# Apply comprehensive test selection
test_selection_results <- comprehensive_test_selection(mtcars, "cyl_factor", "mpg")
```

### Comprehensive Reporting Guidelines

Proper reporting of ANOVA results is essential for transparency and reproducibility. This section provides a comprehensive template for reporting one-way ANOVA analyses.

```r
# Comprehensive function to generate detailed ANOVA report
generate_comprehensive_anova_report <- function(data, group_var, response_var, alpha = 0.05) {
  cat("=== COMPREHENSIVE ONE-WAY ANOVA REPORT ===\n")
  cat("Analysis Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
  
  # Study Information
  cat("=== STUDY INFORMATION ===\n")
  groups <- unique(data[[group_var]])
  k <- length(groups)
  n_per_group <- sapply(groups, function(g) sum(data[[group_var]] == g))
  total_n <- sum(n_per_group)
  
  cat("Research Design: One-way ANOVA\n")
  cat("Number of groups:", k, "\n")
  cat("Group names:", paste(groups, collapse = ", "), "\n")
  cat("Sample sizes per group:", paste(n_per_group, collapse = ", "), "\n")
  cat("Total sample size:", total_n, "\n")
  cat("Significance level (α):", alpha, "\n\n")
  
  # 1. Descriptive Statistics
  cat("=== 1. DESCRIPTIVE STATISTICS ===\n")
  
  desc_stats <- data %>%
    group_by(!!sym(group_var)) %>%
    summarise(
      n = n(),
      mean = mean(!!sym(response_var), na.rm = TRUE),
      sd = sd(!!sym(response_var), na.rm = TRUE),
      median = median(!!sym(response_var), na.rm = TRUE),
      min = min(!!sym(response_var), na.rm = TRUE),
      max = max(!!sym(response_var), na.rm = TRUE),
      se = sd / sqrt(n)
    )
  
  print(desc_stats)
  
  # Overall statistics
  overall_stats <- data %>%
    summarise(
      n = n(),
      mean = mean(!!sym(response_var), na.rm = TRUE),
      sd = sd(!!sym(response_var), na.rm = TRUE),
      median = median(!!sym(response_var), na.rm = TRUE)
    )
  
  cat("\nOverall statistics:\n")
  print(overall_stats)
  cat("\n")
  
  # 2. Assumption Checking
  cat("=== 2. ASSUMPTION CHECKING ===\n")
  
  assumption_results <- comprehensive_assumption_check(data, group_var, response_var)
  
  cat("Assumption violations:", assumption_results$violations, "\n")
  if (assumption_results$violations > 0) {
    cat("Violated assumptions:", paste(assumption_results$violation_details, collapse = ", "), "\n")
  } else {
    cat("All assumptions met ✓\n")
  }
  cat("\n")
  
  # 3. Primary Analysis
  cat("=== 3. PRIMARY ANALYSIS ===\n")
  
  # Perform ANOVA
  anova_model <- aov(as.formula(paste(response_var, "~", group_var)), data = data)
  anova_summary <- summary(anova_model)
  
  # Extract key statistics
  f_stat <- anova_summary[[1]]$`F value`[1]
  p_value <- anova_summary[[1]]$`Pr(>F)`[1]
  df_between <- anova_summary[[1]]$Df[1]
  df_within <- anova_summary[[1]]$Df[2]
  ss_between <- anova_summary[[1]]$`Sum Sq`[1]
  ss_within <- anova_summary[[1]]$`Sum Sq`[2]
  ms_between <- anova_summary[[1]]$`Mean Sq`[1]
  ms_within <- anova_summary[[1]]$`Mean Sq`[2]
  
  cat("ANOVA Results:\n")
  cat("  F-statistic:", round(f_stat, 3), "\n")
  cat("  Degrees of freedom:", df_between, ",", df_within, "\n")
  cat("  p-value:", round(p_value, 4), "\n")
  cat("  Critical F-value:", round(qf(1 - alpha, df_between, df_within), 3), "\n")
  cat("  Significant:", p_value < alpha, "\n\n")
  
  # 4. Effect Size Analysis
  cat("=== 4. EFFECT SIZE ANALYSIS ===\n")
  
  # Calculate effect sizes
  ss_total <- ss_between + ss_within
  eta_squared <- ss_between / ss_total
  omega_squared <- (ss_between - (df_between * ms_within)) / (ss_total + ms_within)
  cohens_f <- sqrt(eta_squared / (1 - eta_squared))
  
  cat("Effect size measures:\n")
  cat("  Eta-squared (η²):", round(eta_squared, 3), "\n")
  cat("  Omega-squared (ω²):", round(omega_squared, 3), "\n")
  cat("  Cohen's f:", round(cohens_f, 3), "\n")
  
  # Interpret effect size
  if (eta_squared < 0.01) {
    effect_interpretation <- "Negligible"
  } else if (eta_squared < 0.06) {
    effect_interpretation <- "Small"
  } else if (eta_squared < 0.14) {
    effect_interpretation <- "Medium"
  } else {
    effect_interpretation <- "Large"
  }
  
  cat("  Effect size interpretation:", effect_interpretation, "\n\n")
  
  # 5. Post Hoc Analysis
  cat("=== 5. POST HOC ANALYSIS ===\n")
  
  if (p_value < alpha) {
    cat("Post hoc tests performed (ANOVA was significant):\n")
    
    # Tukey's HSD
    tukey_result <- TukeyHSD(anova_model)
    significant_tukey <- tukey_result[[1]][tukey_result[[1]][, "p adj"] < alpha, ]
    
    if (nrow(significant_tukey) > 0) {
      cat("Significant pairwise differences (Tukey's HSD):\n")
      for (i in 1:nrow(significant_tukey)) {
        row_name <- rownames(significant_tukey)[i]
        diff <- significant_tukey[i, "diff"]
        lwr <- significant_tukey[i, "lwr"]
        upr <- significant_tukey[i, "upr"]
        p_adj <- significant_tukey[i, "p adj"]
        cat("  ", row_name, ": diff =", round(diff, 3), 
            "95% CI [", round(lwr, 3), ",", round(upr, 3), "], p =", round(p_adj, 4), "\n")
      }
    } else {
      cat("No significant pairwise differences found with Tukey's HSD.\n")
    }
  } else {
    cat("Post hoc tests not performed (ANOVA was not significant).\n")
  }
  cat("\n")
  
  # 6. Power Analysis
  cat("=== 6. POWER ANALYSIS ===\n")
  
  current_power <- pwr.anova.test(k = k, n = min(n_per_group), f = cohens_f, sig.level = alpha)$power
  
  cat("Power analysis:\n")
  cat("  Current power:", round(current_power, 3), "\n")
  cat("  Type II error rate (β):", round(1 - current_power, 3), "\n")
  
  if (current_power < 0.8) {
    required_n <- pwr.anova.test(k = k, f = cohens_f, sig.level = alpha, power = 0.8)$n
    cat("  Required sample size per group for 80% power:", ceiling(required_n), "\n")
  }
  cat("\n")
  
  # 7. Nonparametric Alternative
  cat("=== 7. NONPARAMETRIC ALTERNATIVE ===\n")
  
  kruskal_result <- kruskal.test(as.formula(paste(response_var, "~", group_var)), data = data)
  
  cat("Kruskal-Wallis test results:\n")
  cat("  H-statistic:", round(kruskal_result$statistic, 3), "\n")
  cat("  p-value:", round(kruskal_result$p.value, 4), "\n")
  cat("  Significant:", kruskal_result$p.value < alpha, "\n")
  
  # Agreement between parametric and nonparametric tests
  agreement <- (p_value < alpha) == (kruskal_result$p.value < alpha)
  cat("  Agreement with parametric ANOVA:", agreement, "\n\n")
  
  # 8. Confidence Intervals
  cat("=== 8. CONFIDENCE INTERVALS ===\n")
  
  # Confidence intervals for group means
  ci_data <- data %>%
    group_by(!!sym(group_var)) %>%
    summarise(
      n = n(),
      mean = mean(!!sym(response_var), na.rm = TRUE),
      sd = sd(!!sym(response_var), na.rm = TRUE),
      se = sd / sqrt(n),
      t_critical = qt(1 - alpha/2, n - 1),
      ci_lower = mean - t_critical * se,
      ci_upper = mean + t_critical * se
    )
  
  cat("95% Confidence intervals for group means:\n")
  for (i in 1:nrow(ci_data)) {
    group_name <- ci_data[[group_var]][i]
    mean_val <- ci_data$mean[i]
    ci_lower <- ci_data$ci_lower[i]
    ci_upper <- ci_data$ci_upper[i]
    cat("  ", group_name, ": ", round(mean_val, 2), " [", round(ci_lower, 2), ", ", round(ci_upper, 2), "]\n", sep = "")
  }
  cat("\n")
  
  # 9. Practical Significance
  cat("=== 9. PRACTICAL SIGNIFICANCE ===\n")
  
  cat("Practical significance assessment:\n")
  if (p_value < alpha) {
    cat("  ✓ Statistically significant differences detected\n")
    
    if (eta_squared >= 0.14) {
      cat("  ✓ Large practical effect - results are practically meaningful\n")
    } else if (eta_squared >= 0.06) {
      cat("  ⚠️  Medium practical effect - consider context for interpretation\n")
    } else {
      cat("  ⚠️  Small practical effect - may not be practically meaningful\n")
    }
    
    # Identify best performing group
    best_group <- desc_stats[[group_var]][which.max(desc_stats$mean)]
    cat("  - Best performing group:", best_group, "\n")
    
  } else {
    cat("  ✗ No statistically significant differences detected\n")
    cat("  - Consider power analysis and effect size\n")
    cat("  - May need larger sample size or different design\n")
  }
  cat("\n")
  
  # 10. Conclusions and Recommendations
  cat("=== 10. CONCLUSIONS AND RECOMMENDATIONS ===\n")
  
  cat("Primary conclusion:\n")
  if (p_value < alpha) {
    cat("  Reject the null hypothesis (p <", alpha, ")\n")
    cat("  There are significant differences between group means\n")
  } else {
    cat("  Fail to reject the null hypothesis (p >=", alpha, ")\n")
    cat("  There is insufficient evidence of differences between group means\n")
  }
  
  cat("\nRecommendations:\n")
  if (p_value < alpha && eta_squared >= 0.06) {
    cat("  - Results support the effectiveness of the intervention/factor\n")
    cat("  - Consider implementing the best-performing condition\n")
    cat("  - Conduct follow-up studies to confirm findings\n")
  } else if (p_value < alpha && eta_squared < 0.06) {
    cat("  - Statistically significant but small practical effect\n")
    cat("  - Consider whether the effect is meaningful in practice\n")
    cat("  - May need larger sample size for adequate power\n")
  } else {
    cat("  - No significant differences detected\n")
    cat("  - Consider increasing sample size for future studies\n")
    cat("  - Investigate other factors that may influence the outcome\n")
  }
  
  # 11. Limitations
  cat("\n=== 11. LIMITATIONS ===\n")
  
  cat("Study limitations:\n")
  if (current_power < 0.8) {
    cat("  - Low statistical power may have missed true effects\n")
  }
  if (assumption_results$violations > 0) {
    cat("  - Some ANOVA assumptions were violated\n")
  }
  if (min(n_per_group) < 30) {
    cat("  - Small sample sizes may affect robustness\n")
  }
  if (!length(unique(n_per_group)) == 1) {
    cat("  - Unbalanced design may affect power\n")
  }
  
  cat("\n=== END OF REPORT ===\n")
  
  return(list(
    descriptive_stats = desc_stats,
    anova_results = list(f_stat = f_stat, p_value = p_value, df = c(df_between, df_within)),
    effect_sizes = list(eta_squared = eta_squared, omega_squared = omega_squared, cohens_f = cohens_f),
    power = current_power,
    assumptions = assumption_results,
    kruskal_wallis = kruskal_result,
    significant = p_value < alpha
  ))
}

# Generate comprehensive report
comprehensive_report <- generate_comprehensive_anova_report(mtcars, "cyl_factor", "mpg")
```

## Exercises

These exercises provide hands-on practice with one-way ANOVA concepts and techniques. Each exercise builds upon the previous ones and includes comprehensive analysis workflows.

### Exercise 1: Basic One-Way ANOVA with Transmission Types

**Objective:** Perform a complete one-way ANOVA analysis comparing MPG across different transmission types.

**Dataset:** Use the mtcars dataset and create a transmission factor variable.

**Tasks:**
1. Create a transmission factor variable (automatic vs manual)
2. Perform descriptive statistics for each transmission type
3. Conduct one-way ANOVA
4. Calculate and interpret effect sizes
5. Perform post hoc tests if significant
6. Create appropriate visualizations

**Expected Learning Outcomes:**
- Understand the complete ANOVA workflow
- Interpret ANOVA results correctly
- Calculate and interpret effect sizes
- Perform post hoc analysis appropriately

```r
# Exercise 1 Solution Framework
# Your code here...

# 1. Data preparation
# 2. Descriptive statistics
# 3. Assumption checking
# 4. ANOVA analysis
# 5. Effect size calculation
# 6. Post hoc tests
# 7. Visualization
# 8. Interpretation
```

### Exercise 2: Comprehensive Assumption Checking

**Objective:** Conduct thorough assumption checking and recommend appropriate statistical approaches.

**Dataset:** Use the built-in `PlantGrowth` dataset or create simulated data with known violations.

**Tasks:**
1. Check normality for each group using multiple tests
2. Test homogeneity of variance using different methods
3. Assess independence and identify outliers
4. Evaluate the impact of assumption violations
5. Recommend appropriate alternative approaches
6. Compare parametric and nonparametric results

**Expected Learning Outcomes:**
- Master assumption checking procedures
- Understand when to use alternative methods
- Interpret assumption violation impacts
- Make informed methodological decisions

```r
# Exercise 2 Solution Framework
# Your code here...

# 1. Load and prepare data
# 2. Comprehensive normality testing
# 3. Homogeneity of variance testing
# 4. Independence assessment
# 5. Outlier detection
# 6. Alternative method comparison
# 7. Recommendations
```

### Exercise 3: Advanced Post Hoc Analysis

**Objective:** Compare multiple post hoc methods and understand their differences.

**Dataset:** Create simulated data with 4 groups and known differences.

**Tasks:**
1. Perform ANOVA and confirm significance
2. Apply Tukey's HSD test
3. Apply Bonferroni correction
4. Apply Holm's method
5. Apply FDR correction
6. Compare results across methods
7. Create visualization of post hoc results

**Expected Learning Outcomes:**
- Understand different multiple comparison methods
- Compare conservative vs. liberal approaches
- Interpret post hoc results correctly
- Choose appropriate methods for different scenarios

```r
# Exercise 3 Solution Framework
# Your code here...

# 1. Generate simulated data
# 2. Perform ANOVA
# 3. Multiple post hoc methods
# 4. Result comparison
# 5. Visualization
# 6. Interpretation
```

### Exercise 4: Effect Size Analysis and Interpretation

**Objective:** Calculate and interpret various effect size measures comprehensively.

**Dataset:** Use the `ToothGrowth` dataset or create data with different effect sizes.

**Tasks:**
1. Calculate eta-squared, omega-squared, and Cohen's f
2. Compute confidence intervals for effect sizes
3. Compare effect sizes across different datasets
4. Assess practical significance
5. Create effect size visualizations
6. Provide comprehensive interpretation

**Expected Learning Outcomes:**
- Master effect size calculations
- Understand effect size interpretation
- Distinguish statistical vs. practical significance
- Communicate effect sizes effectively

```r
# Exercise 4 Solution Framework
# Your code here...

# 1. Data preparation
# 2. Effect size calculations
# 3. Confidence intervals
# 4. Practical significance assessment
# 5. Visualization
# 6. Comprehensive interpretation
```

### Exercise 5: Power Analysis and Sample Size Planning

**Objective:** Conduct comprehensive power analysis for ANOVA designs.

**Dataset:** Use existing data or create scenarios with different effect sizes.

**Tasks:**
1. Calculate current power for existing data
2. Determine required sample sizes for different power levels
3. Create power curves for different effect sizes
4. Assess the impact of multiple comparisons on power
5. Plan sample sizes for future studies
6. Consider cost-benefit analysis

**Expected Learning Outcomes:**
- Master power analysis techniques
- Plan sample sizes effectively
- Understand power trade-offs
- Design efficient studies

```r
# Exercise 5 Solution Framework
# Your code here...

# 1. Current power analysis
# 2. Sample size planning
# 3. Power curves
# 4. Multiple comparison impact
# 5. Study design recommendations
```

### Exercise 6: Real-World Application

**Objective:** Apply one-way ANOVA to a real-world research question.

**Research Question:** Do different study environments (library, coffee shop, home) affect student concentration levels?

**Tasks:**
1. Design a study to address the research question
2. Simulate realistic data based on the design
3. Perform complete ANOVA analysis
4. Check all assumptions thoroughly
5. Calculate effect sizes and power
6. Provide practical recommendations
7. Write a comprehensive report

**Expected Learning Outcomes:**
- Apply ANOVA to real research questions
- Design appropriate studies
- Conduct complete statistical analysis
- Communicate results effectively

```r
# Exercise 6 Solution Framework
# Your code here...

# 1. Study design
# 2. Data simulation
# 3. Complete analysis
# 4. Assumption checking
# 5. Effect size and power
# 6. Practical recommendations
# 7. Report writing
```

### Exercise 7: Robust Methods and Alternatives

**Objective:** Explore robust alternatives to traditional ANOVA.

**Dataset:** Create data with various assumption violations.

**Tasks:**
1. Create data with normality violations
2. Create data with variance heterogeneity
3. Apply traditional ANOVA
4. Apply Welch's ANOVA
5. Apply Kruskal-Wallis test
6. Apply bootstrap methods
7. Compare results across methods

**Expected Learning Outcomes:**
- Understand robust alternatives
- Choose appropriate methods for violated assumptions
- Compare parametric and nonparametric approaches
- Apply modern statistical techniques

```r
# Exercise 7 Solution Framework
# Your code here...

# 1. Data generation with violations
# 2. Traditional ANOVA
# 3. Welch's ANOVA
# 4. Kruskal-Wallis
# 5. Bootstrap methods
# 6. Method comparison
# 7. Recommendations
```

### Exercise 8: Advanced Visualization and Reporting

**Objective:** Create comprehensive visualizations and reports for ANOVA results.

**Dataset:** Use any of the previous datasets or create new data.

**Tasks:**
1. Create publication-quality visualizations
2. Generate comprehensive statistical reports
3. Create interactive visualizations (if possible)
4. Design presentation materials
5. Write statistical interpretation
6. Provide practical recommendations

**Expected Learning Outcomes:**
- Create effective visualizations
- Write comprehensive reports
- Communicate statistical results clearly
- Provide actionable recommendations

```r
# Exercise 8 Solution Framework
# Your code here...

# 1. Advanced visualizations
# 2. Comprehensive reporting
# 3. Interactive elements
# 4. Presentation materials
# 5. Statistical interpretation
# 6. Practical recommendations
```

### Exercise Hints and Solutions

**General Tips:**
- Always start with descriptive statistics
- Check assumptions before interpreting results
- Consider both statistical and practical significance
- Use appropriate visualizations for your data
- Report results comprehensively and transparently

**Common Pitfalls to Avoid:**
- Ignoring assumption violations
- Focusing only on p-values
- Neglecting effect size interpretation
- Performing post hoc tests without significant ANOVA
- Not considering practical significance

**Advanced Considerations:**
- Multiple comparison corrections
- Power analysis for study design
- Robust alternatives for assumption violations
- Effect size confidence intervals
- Practical significance assessment

### Expected Learning Progression

**Beginner Level (Exercises 1-3):**
- Basic ANOVA workflow
- Assumption checking
- Post hoc analysis

**Intermediate Level (Exercises 4-6):**
- Effect size analysis
- Power analysis
- Real-world applications

**Advanced Level (Exercises 7-8):**
- Robust methods
- Advanced visualization
- Comprehensive reporting

Each exercise builds upon previous knowledge and introduces new concepts progressively.

## Next Steps

In the next chapter, we'll learn about two-way ANOVA for analyzing the effects of two independent variables.

---

**Key Takeaways:**
- One-way ANOVA compares means across three or more groups
- Always check assumptions before interpreting results
- Effect sizes provide important information about practical significance
- Post hoc tests are necessary when ANOVA is significant
- Nonparametric alternatives exist for non-normal data
- Power analysis helps determine appropriate sample sizes
- Proper reporting includes descriptive statistics, test results, and effect sizes 