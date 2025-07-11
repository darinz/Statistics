# One-Way ANOVA

## Overview

One-way Analysis of Variance (ANOVA) is used to compare means across three or more groups. It extends the t-test to multiple groups and is a fundamental statistical method for experimental design and observational studies.

## Basic One-Way ANOVA

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

# Manual ANOVA calculation
manual_anova <- function(group_data) {
  # Calculate group means and overall mean
  group_means <- sapply(group_data, mean)
  overall_mean <- mean(unlist(group_data))
  
  # Calculate sample sizes
  n_per_group <- sapply(group_data, length)
  total_n <- sum(n_per_group)
  
  # Calculate Sum of Squares
  # Between-groups SS
  ss_between <- sum(n_per_group * (group_means - overall_mean)^2)
  
  # Within-groups SS
  ss_within <- sum(sapply(1:length(group_data), function(i) {
    sum((group_data[[i]] - group_means[i])^2)
  }))
  
  # Total SS
  ss_total <- ss_between + ss_within
  
  # Degrees of freedom
  df_between <- length(group_data) - 1
  df_within <- total_n - length(group_data)
  df_total <- total_n - 1
  
  # Mean Squares
  ms_between <- ss_between / df_between
  ms_within <- ss_within / df_within
  
  # F-statistic
  f_statistic <- ms_between / ms_within
  
  # p-value
  p_value <- 1 - pf(f_statistic, df_between, df_within)
  
  # Effect size (eta-squared)
  eta_squared <- ss_between / ss_total
  
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
    group_means = group_means,
    overall_mean = overall_mean
  ))
}

# Apply manual calculation
anova_result <- manual_anova(list(mpg_4cyl, mpg_6cyl, mpg_8cyl))

cat("Manual ANOVA Results:\n")
cat("F-statistic:", round(anova_result$f_statistic, 3), "\n")
cat("p-value:", round(anova_result$p_value, 4), "\n")
cat("Eta-squared:", round(anova_result$eta_squared, 3), "\n")
cat("Group means:", round(anova_result$group_means, 2), "\n")
```

### Using R's Built-in ANOVA

```r
# Perform ANOVA using R's aov function
anova_model <- aov(mpg ~ cyl_factor, data = mtcars)
print(anova_model)

# Get ANOVA summary
anova_summary <- summary(anova_model)
print(anova_summary)

# Extract key statistics
f_statistic <- anova_summary[[1]]$`F value`[1]
p_value <- anova_summary[[1]]$`Pr(>F)`[1]
df_between <- anova_summary[[1]]$Df[1]
df_within <- anova_summary[[1]]$Df[2]

cat("ANOVA Results:\n")
cat("F-statistic:", round(f_statistic, 3), "\n")
cat("p-value:", round(p_value, 4), "\n")
cat("Degrees of freedom:", df_between, ",", df_within, "\n")
```

## Descriptive Statistics

### Group Comparisons

```r
# Calculate descriptive statistics for each group
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
    se = sd / sqrt(n)
  )

print(group_stats)

# Overall statistics
overall_stats <- mtcars %>%
  summarise(
    n = n(),
    mean = mean(mpg, na.rm = TRUE),
    sd = sd(mpg, na.rm = TRUE),
    median = median(mpg, na.rm = TRUE)
  )

cat("Overall Statistics:\n")
print(overall_stats)
```

### Visualization

```r
library(ggplot2)
library(gridExtra)

# Box plot
p1 <- ggplot(mtcars, aes(x = cyl_factor, y = mpg, fill = cyl_factor)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "MPG by Number of Cylinders", x = "Cylinders", y = "MPG") +
  theme_minimal() +
  theme(legend.position = "none")

# Violin plot
p2 <- ggplot(mtcars, aes(x = cyl_factor, y = mpg, fill = cyl_factor)) +
  geom_violin(alpha = 0.7) +
  geom_boxplot(width = 0.2, alpha = 0.8) +
  labs(title = "MPG Distribution by Cylinders", x = "Cylinders", y = "MPG") +
  theme_minimal() +
  theme(legend.position = "none")

# Histogram by group
p3 <- ggplot(mtcars, aes(x = mpg, fill = cyl_factor)) +
  geom_histogram(bins = 10, alpha = 0.7, position = "identity") +
  facet_wrap(~cyl_factor) +
  labs(title = "MPG Distribution by Cylinders", x = "MPG", y = "Count") +
  theme_minimal()

# Combine plots
grid.arrange(p1, p2, p3, ncol = 2)
```

## Effect Size

### Eta-Squared and Partial Eta-Squared

```r
# Calculate effect sizes
calculate_anova_effect_sizes <- function(anova_result) {
  # Eta-squared
  eta_squared <- anova_result$ss_between / anova_result$ss_total
  
  # Partial eta-squared (same as eta-squared for one-way ANOVA)
  partial_eta_squared <- eta_squared
  
  # Omega-squared (unbiased estimator)
  omega_squared <- (anova_result$ss_between - (anova_result$df_between * anova_result$ms_within)) /
                   (anova_result$ss_total + anova_result$ms_within)
  
  return(list(
    eta_squared = eta_squared,
    partial_eta_squared = partial_eta_squared,
    omega_squared = omega_squared
  ))
}

# Apply to our ANOVA results
effect_sizes <- calculate_anova_effect_sizes(anova_result)

cat("Effect Size Analysis:\n")
cat("Eta-squared:", round(effect_sizes$eta_squared, 3), "\n")
cat("Partial eta-squared:", round(effect_sizes$partial_eta_squared, 3), "\n")
cat("Omega-squared:", round(effect_sizes$omega_squared, 3), "\n")

# Interpret effect size
interpret_eta_squared <- function(eta_sq) {
  if (eta_sq < 0.01) {
    return("Negligible effect")
  } else if (eta_sq < 0.06) {
    return("Small effect")
  } else if (eta_sq < 0.14) {
    return("Medium effect")
  } else {
    return("Large effect")
  }
}

cat("Effect size interpretation:", interpret_eta_squared(effect_sizes$eta_squared), "\n")
```

## Post Hoc Tests

### Tukey's HSD Test

```r
# Perform Tukey's HSD test
tukey_result <- TukeyHSD(anova_model)
print(tukey_result)

# Extract significant differences
significant_pairs <- tukey_result$cyl_factor[tukey_result$cyl_factor[, "p adj"] < 0.05, ]
if (nrow(significant_pairs) > 0) {
  cat("Significant pairwise differences (p < 0.05):\n")
  print(significant_pairs)
} else {
  cat("No significant pairwise differences found.\n")
}

# Visualize Tukey results
plot(tukey_result)
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

### Normality Test

```r
# Function to check normality for all groups
check_normality_anova <- function(data, group_var, response_var) {
  cat("=== NORMALITY TESTS FOR ANOVA ===\n")
  
  # Get unique groups
  groups <- unique(data[[group_var]])
  
  # Perform Shapiro-Wilk test for each group
  normality_results <- list()
  for (group in groups) {
    group_data <- data[[response_var]][data[[group_var]] == group]
    shapiro_result <- shapiro.test(group_data)
    normality_results[[as.character(group)]] <- shapiro_result
    
    cat("Group", group, "Shapiro-Wilk p-value:", round(shapiro_result$p.value, 4), "\n")
  }
  
  # Overall normality test on residuals
  model <- aov(as.formula(paste(response_var, "~", group_var)), data = data)
  residuals <- residuals(model)
  overall_shapiro <- shapiro.test(residuals)
  
  cat("Overall residuals Shapiro-Wilk p-value:", round(overall_shapiro$p.value, 4), "\n")
  
  # Recommendations
  cat("\nRECOMMENDATIONS:\n")
  all_normal <- all(sapply(normality_results, function(x) x$p.value >= 0.05))
  if (all_normal && overall_shapiro$p.value >= 0.05) {
    cat("- All groups appear normally distributed\n")
    cat("- ANOVA assumptions are met\n")
  } else {
    cat("- Some groups may not be normally distributed\n")
    cat("- Consider nonparametric alternatives (Kruskal-Wallis)\n")
  }
  
  return(list(
    group_tests = normality_results,
    overall_test = overall_shapiro
  ))
}

# Check normality for cylinder groups
normality_results <- check_normality_anova(mtcars, "cyl_factor", "mpg")
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

### Kruskal-Wallis Test

```r
# Perform Kruskal-Wallis test
kruskal_result <- kruskal.test(mpg ~ cyl_factor, data = mtcars)
print(kruskal_result)

# Compare with ANOVA results
cat("Comparison of parametric and nonparametric tests:\n")
cat("ANOVA F-statistic:", round(f_statistic, 3), "\n")
cat("ANOVA p-value:", round(p_value, 4), "\n")
cat("Kruskal-Wallis chi-squared:", round(kruskal_result$statistic, 3), "\n")
cat("Kruskal-Wallis p-value:", round(kruskal_result$p.value, 4), "\n")

# Effect size for Kruskal-Wallis
kruskal_effect_size <- (kruskal_result$statistic - length(unique(mtcars$cyl_factor)) + 1) /
                       (nrow(mtcars) - length(unique(mtcars$cyl_factor)))
cat("Kruskal-Wallis effect size (epsilon-squared):", round(kruskal_effect_size, 3), "\n")
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

### Power Analysis for One-Way ANOVA

```r
library(pwr)

# Power analysis for one-way ANOVA
power_analysis_anova <- function(k, n_per_group, effect_size, alpha = 0.05) {
  # Calculate power
  power_result <- pwr.anova.test(k = k, n = n_per_group, f = effect_size, sig.level = alpha)
  
  # Calculate required sample size for 80% power
  sample_size_result <- pwr.anova.test(k = k, f = effect_size, sig.level = alpha, power = 0.8)
  
  return(list(
    power = power_result$power,
    required_n_per_group = ceiling(sample_size_result$n),
    effect_size = effect_size,
    alpha = alpha,
    k = k
  ))
}

# Calculate effect size f for ANOVA
calculate_f_effect_size <- function(eta_squared) {
  f <- sqrt(eta_squared / (1 - eta_squared))
  return(f)
}

# Apply power analysis
f_effect_size <- calculate_f_effect_size(effect_sizes$eta_squared)
anova_power <- power_analysis_anova(
  k = 3, 
  n_per_group = min(table(mtcars$cyl_factor)), 
  effect_size = f_effect_size
)

cat("Power Analysis Results:\n")
cat("Effect size f:", round(f_effect_size, 3), "\n")
cat("Current power:", round(anova_power$power, 3), "\n")
cat("Required sample size per group for 80% power:", anova_power$required_n_per_group, "\n")
```

## Practical Examples

### Example 1: Educational Research

```r
# Simulate educational intervention data
set.seed(123)
n_per_group <- 20

# Generate data for three teaching methods
method_a_scores <- rnorm(n_per_group, mean = 75, sd = 10)
method_b_scores <- rnorm(n_per_group, mean = 82, sd = 12)
method_c_scores <- rnorm(n_per_group, mean = 78, sd = 11)

# Create data frame
education_data <- data.frame(
  score = c(method_a_scores, method_b_scores, method_c_scores),
  method = factor(rep(c("Method A", "Method B", "Method C"), each = n_per_group))
)

# Perform ANOVA
education_anova <- aov(score ~ method, data = education_data)
print(summary(education_anova))

# Post hoc tests
education_tukey <- TukeyHSD(education_anova)
print(education_tukey)

# Effect size
education_ss <- summary(education_anova)[[1]]
education_eta_squared <- education_ss$`Sum Sq`[1] / sum(education_ss$`Sum Sq`)

cat("Educational Research Results:\n")
cat("F-statistic:", round(education_ss$`F value`[1], 3), "\n")
cat("p-value:", round(education_ss$`Pr(>F)`[1], 4), "\n")
cat("Eta-squared:", round(education_eta_squared, 3), "\n")
```

### Example 2: Clinical Trial

```r
# Simulate clinical trial data
set.seed(123)
n_per_treatment <- 25

# Generate data for three treatments
treatment_a_scores <- rnorm(n_per_treatment, mean = 65, sd = 15)
treatment_b_scores <- rnorm(n_per_treatment, mean = 72, sd = 14)
treatment_c_scores <- rnorm(n_per_treatment, mean = 68, sd = 16)

# Create data frame
clinical_data <- data.frame(
  outcome = c(treatment_a_scores, treatment_b_scores, treatment_c_scores),
  treatment = factor(rep(c("Treatment A", "Treatment B", "Treatment C"), each = n_per_treatment))
)

# Perform ANOVA
clinical_anova <- aov(outcome ~ treatment, data = clinical_data)
print(summary(clinical_anova))

# Check assumptions
clinical_normality <- check_normality_anova(clinical_data, "treatment", "outcome")
clinical_homogeneity <- check_homogeneity_anova(clinical_data, "treatment", "outcome")

# Nonparametric alternative
clinical_kruskal <- kruskal.test(outcome ~ treatment, data = clinical_data)
print(clinical_kruskal)
```

### Example 3: Quality Control

```r
# Simulate quality control data
set.seed(123)
n_per_machine <- 15

# Generate data for three machines
machine_a_output <- rnorm(n_per_machine, mean = 100, sd = 3)
machine_b_output <- rnorm(n_per_machine, mean = 98, sd = 4)
machine_c_output <- rnorm(n_per_machine, mean = 102, sd = 3.5)

# Create data frame
quality_data <- data.frame(
  output = c(machine_a_output, machine_b_output, machine_c_output),
  machine = factor(rep(c("Machine A", "Machine B", "Machine C"), each = n_per_machine))
)

# Perform ANOVA
quality_anova <- aov(output ~ machine, data = quality_data)
print(summary(quality_anova))

# Post hoc analysis
quality_tukey <- TukeyHSD(quality_anova)
print(quality_tukey)

# Visualize results
ggplot(quality_data, aes(x = machine, y = output, fill = machine)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Output Quality by Machine", x = "Machine", y = "Output") +
  theme_minimal() +
  theme(legend.position = "none")
```

## Best Practices

### Test Selection Guidelines

```r
# Function to help choose appropriate ANOVA test
choose_anova_test <- function(data, group_var, response_var) {
  cat("=== ANOVA TEST SELECTION ===\n")
  
  # Check normality
  normality_results <- check_normality_anova(data, group_var, response_var)
  
  # Check homogeneity
  homogeneity_results <- check_homogeneity_anova(data, group_var, response_var)
  
  # Check independence
  independence_results <- check_independence_anova(data, group_var, response_var)
  
  cat("\nFINAL RECOMMENDATION:\n")
  
  # Decision logic
  normal_data <- all(sapply(normality_results$group_tests, function(x) x$p.value >= 0.05))
  equal_variances <- homogeneity_results$levene$`Pr(>F)`[1] >= 0.05
  
  if (normal_data && equal_variances) {
    cat("Use standard one-way ANOVA\n")
    cat("All assumptions are met\n")
  } else if (normal_data && !equal_variances) {
    cat("Use Welch's ANOVA (unequal variances)\n")
    cat("Data is normal but variances are unequal\n")
  } else {
    cat("Use Kruskal-Wallis test\n")
    cat("Data is not normal or variances are unequal\n")
  }
  
  return(list(
    normality = normality_results,
    homogeneity = homogeneity_results,
    independence = independence_results
  ))
}

# Apply to cylinder data
test_selection <- choose_anova_test(mtcars, "cyl_factor", "mpg")
```

### Reporting Guidelines

```r
# Function to generate comprehensive ANOVA report
generate_anova_report <- function(anova_result, data, group_var, response_var) {
  cat("=== ONE-WAY ANOVA REPORT ===\n\n")
  
  # Descriptive statistics
  desc_stats <- data %>%
    group_by(!!sym(group_var)) %>%
    summarise(
      n = n(),
      mean = mean(!!sym(response_var), na.rm = TRUE),
      sd = sd(!!sym(response_var), na.rm = TRUE)
    )
  
  cat("DESCRIPTIVE STATISTICS:\n")
  print(desc_stats)
  cat("\n")
  
  # ANOVA results
  anova_summary <- summary(anova_result)
  f_stat <- anova_summary[[1]]$`F value`[1]
  p_value <- anova_summary[[1]]$`Pr(>F)`[1]
  df_between <- anova_summary[[1]]$Df[1]
  df_within <- anova_summary[[1]]$Df[2]
  
  cat("ANOVA RESULTS:\n")
  cat("F-statistic:", round(f_stat, 3), "\n")
  cat("Degrees of freedom:", df_between, ",", df_within, "\n")
  cat("p-value:", round(p_value, 4), "\n")
  
  # Effect size
  ss_between <- anova_summary[[1]]$`Sum Sq`[1]
  ss_total <- sum(anova_summary[[1]]$`Sum Sq`)
  eta_squared <- ss_between / ss_total
  
  cat("Effect size (eta-squared):", round(eta_squared, 3), "\n")
  cat("Interpretation:", interpret_eta_squared(eta_squared), "\n\n")
  
  # Post hoc results
  tukey_result <- TukeyHSD(anova_result)
  significant_pairs <- tukey_result[[1]][tukey_result[[1]][, "p adj"] < 0.05, ]
  
  if (nrow(significant_pairs) > 0) {
    cat("SIGNIFICANT PAIRWISE DIFFERENCES (Tukey's HSD):\n")
    print(significant_pairs)
  } else {
    cat("No significant pairwise differences found.\n")
  }
  cat("\n")
  
  # Conclusion
  alpha <- 0.05
  if (p_value < alpha) {
    cat("CONCLUSION:\n")
    cat("Reject the null hypothesis (p <", alpha, ")\n")
    cat("There are significant differences between group means\n")
  } else {
    cat("CONCLUSION:\n")
    cat("Fail to reject the null hypothesis (p >=", alpha, ")\n")
    cat("There is insufficient evidence of differences between group means\n")
  }
}

# Generate report for cylinder ANOVA
generate_anova_report(anova_model, mtcars, "cyl_factor", "mpg")
```

## Exercises

### Exercise 1: Basic One-Way ANOVA
Perform one-way ANOVA to compare MPG across different transmission types in the mtcars dataset.

### Exercise 2: Assumption Checking
Conduct comprehensive assumption checking for ANOVA and recommend appropriate alternatives.

### Exercise 3: Post Hoc Analysis
Perform multiple post hoc tests and compare their results for a given dataset.

### Exercise 4: Effect Size Analysis
Calculate and interpret different effect size measures for ANOVA results.

### Exercise 5: Power Analysis
Conduct power analysis to determine required sample sizes for detecting different effect sizes in ANOVA designs.

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