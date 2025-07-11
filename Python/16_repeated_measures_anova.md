# Repeated Measures ANOVA

## Overview

Repeated Measures Analysis of Variance (ANOVA) is a powerful statistical technique used to analyze data where the same subjects are measured multiple times under different conditions. This within-subjects design is more powerful than between-subjects designs because it controls for individual differences and reduces error variance, leading to increased statistical power and more precise effect estimates.

### Key Concepts

**Within-Subjects Design**: A research design where each participant is measured under all experimental conditions, allowing for direct comparison of conditions within the same individuals.

**Individual Differences**: The natural variation between participants that is controlled for in repeated measures designs, reducing error variance.

**Sphericity**: The assumption that the variances of the differences between all pairs of conditions are equal, which is crucial for the validity of repeated measures ANOVA.

**Compound Symmetry**: A stronger assumption than sphericity, requiring equal variances and equal covariances between all conditions.

### Mathematical Foundation

The repeated measures ANOVA model can be expressed as:

```math
Y_{ij} = \mu + \alpha_i + \pi_j + \epsilon_{ij}
```

Where:
- $`Y_{ij}`$ is the observed value for the jth subject in the ith condition
- $`\mu`$ is the overall population mean
- $`\alpha_i`$ is the effect of the ith condition (treatment effect)
- $`\pi_j`$ is the effect of the jth subject (individual differences)
- $`\epsilon_{ij}`$ is the random error term

### Sum of Squares Decomposition

The total variability in repeated measures data is partitioned into three components:

```math
SS_{Total} = SS_{Between Subjects} + SS_{Within Subjects}
```

Where:
- $`SS_{Total} = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{..})^2`$
- $`SS_{Between Subjects} = k\sum_{j=1}^{n}(\bar{Y}_{.j} - \bar{Y}_{..})^2`$ (individual differences)
- $`SS_{Within Subjects} = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{.j})^2`$ (within-subject variation)

The within-subjects sum of squares is further partitioned into:

```math
SS_{Within Subjects} = SS_{Conditions} + SS_{Error}
```

Where:
- $`SS_{Conditions} = n\sum_{i=1}^{k}(\bar{Y}_{i.} - \bar{Y}_{..})^2`$ (treatment effect)
- $`SS_{Error} = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{i.} - \bar{Y}_{.j} + \bar{Y}_{..})^2`$ (residual error)

### Degrees of Freedom

```math
df_{Total} = nk - 1
df_{Between Subjects} = n - 1
df_{Within Subjects} = n(k - 1)
df_{Conditions} = k - 1
df_{Error} = (n - 1)(k - 1)
```

Where $`n`$ is the number of subjects and $`k`$ is the number of conditions.

### F-Statistic

```math
F = \frac{MS_{Conditions}}{MS_{Error}} = \frac{SS_{Conditions}/df_{Conditions}}{SS_{Error}/df_{Error}}
```

### Effect Size Measures

**Partial Eta-Squared**:
```math
\eta_p^2 = \frac{SS_{Conditions}}{SS_{Conditions} + SS_{Error}}
```

**Eta-Squared**:
```math
\eta^2 = \frac{SS_{Conditions}}{SS_{Total}}
```

**Omega-Squared** (unbiased estimate):
```math
\omega^2 = \frac{SS_{Conditions} - df_{Conditions} \times MS_{Error}}{SS_{Total} + MS_{Error}}
```

### Sphericity Assumption

The sphericity assumption requires that the variance of the differences between any two conditions is constant:

```math
Var(Y_{i} - Y_{j}) = \sigma^2 \text{ for all } i \neq j
```

This can be tested using Mauchly's test of sphericity, which examines the correlation matrix of the conditions.

## Basic Repeated Measures ANOVA

### Manual Calculation

The manual calculation helps understand the underlying mathematics and provides insight into how the statistical software performs the analysis.

```r
# Simulate repeated measures data
set.seed(123)
n_subjects <- 20
n_conditions <- 3

# Generate data for 3 conditions (e.g., pre-test, post-test, follow-up)
subject_id <- rep(1:n_subjects, each = n_conditions)
condition <- rep(c("Pre", "Post", "Follow-up"), times = n_subjects)

# Generate scores with subject effects and condition effects
scores <- numeric(length(subject_id))
for (i in 1:length(subject_id)) {
  subject_effect <- rnorm(1, 0, 10)  # Individual differences
  if (condition[i] == "Pre") {
    scores[i] <- 70 + subject_effect + rnorm(1, 0, 5)
  } else if (condition[i] == "Post") {
    scores[i] <- 75 + subject_effect + rnorm(1, 0, 5)
  } else {
    scores[i] <- 78 + subject_effect + rnorm(1, 0, 5)
  }
}

# Create data frame
repeated_data <- data.frame(
  subject = factor(subject_id),
  condition = factor(condition, levels = c("Pre", "Post", "Follow-up")),
  score = scores
)

# Manual repeated measures ANOVA calculation
manual_repeated_anova <- function(data, subject_var, condition_var, response_var) {
  # Calculate overall mean
  overall_mean <- mean(data[[response_var]], na.rm = TRUE)
  
  # Calculate condition means
  condition_means <- tapply(data[[response_var]], data[[condition_var]], mean, na.rm = TRUE)
  
  # Calculate subject means
  subject_means <- tapply(data[[response_var]], data[[subject_var]], mean, na.rm = TRUE)
  
  # Calculate sample sizes
  n_subjects <- length(unique(data[[subject_var]]))
  n_conditions <- length(unique(data[[condition_var]]))
  n_total <- nrow(data)
  
  # Calculate Sum of Squares
  # SS Total
  ss_total <- sum((data[[response_var]] - overall_mean)^2, na.rm = TRUE)
  
  # SS Between Subjects (individual differences)
  ss_between_subjects <- n_conditions * sum((subject_means - overall_mean)^2)
  
  # SS Within Subjects
  ss_within_subjects <- ss_total - ss_between_subjects
  
  # SS Conditions (Treatment effect)
  ss_conditions <- n_subjects * sum((condition_means - overall_mean)^2)
  
  # SS Error (Residual)
  ss_error <- ss_within_subjects - ss_conditions
  
  # Degrees of freedom
  df_between_subjects <- n_subjects - 1
  df_within_subjects <- n_subjects * (n_conditions - 1)
  df_conditions <- n_conditions - 1
  df_error <- (n_subjects - 1) * (n_conditions - 1)
  
  # Mean Squares
  ms_conditions <- ss_conditions / df_conditions
  ms_error <- ss_error / df_error
  
  # F-statistic
  f_statistic <- ms_conditions / ms_error
  
  # p-value
  p_value <- 1 - pf(f_statistic, df_conditions, df_error)
  
  # Effect size (partial eta-squared)
  partial_eta2 <- ss_conditions / (ss_conditions + ss_error)
  
  # Calculate sphericity measures
  # Create wide format for correlation analysis
  wide_data <- data %>%
    tidyr::pivot_wider(
      names_from = condition_var,
      values_from = response_var,
      names_prefix = "condition_"
    )
  
  # Remove subject column for correlation matrix
  cor_matrix <- wide_data %>%
    select(-subject_var) %>%
    cor(use = "complete.obs")
  
  # Greenhouse-Geisser epsilon
  gg_epsilon <- 1 / (n_conditions - 1)
  
  # Huynh-Feldt epsilon (approximation)
  hf_epsilon <- min(1, gg_epsilon * (n_subjects - 1) / (n_subjects - 1 - (n_conditions - 1)))
  
  return(list(
    ss_total = ss_total,
    ss_between_subjects = ss_between_subjects,
    ss_within_subjects = ss_within_subjects,
    ss_conditions = ss_conditions,
    ss_error = ss_error,
    df_conditions = df_conditions,
    df_error = df_error,
    ms_conditions = ms_conditions,
    ms_error = ms_error,
    f_statistic = f_statistic,
    p_value = p_value,
    partial_eta2 = partial_eta2,
    condition_means = condition_means,
    subject_means = subject_means,
    correlation_matrix = cor_matrix,
    gg_epsilon = gg_epsilon,
    hf_epsilon = hf_epsilon
  ))
}

# Apply manual calculation
anova_result <- manual_repeated_anova(repeated_data, "subject", "condition", "score")

cat("Manual Repeated Measures ANOVA Results:\n")
cat("F-statistic:", round(anova_result$f_statistic, 3), "\n")
cat("p-value:", round(anova_result$p_value, 4), "\n")
cat("Partial η²:", round(anova_result$partial_eta2, 3), "\n")
cat("Condition means:", round(anova_result$condition_means, 2), "\n")
cat("Greenhouse-Geisser ε:", round(anova_result$gg_epsilon, 3), "\n")
cat("Huynh-Feldt ε:", round(anova_result$hf_epsilon, 3), "\n")
```

### Using R's Built-in Repeated Measures ANOVA

R's `aov()` function with the `Error()` term provides a convenient way to perform repeated measures ANOVA.

```r
# Perform repeated measures ANOVA using aov
repeated_model <- aov(score ~ condition + Error(subject/condition), data = repeated_data)
print(repeated_model)

# Get ANOVA summary
anova_summary <- summary(repeated_model)
print(anova_summary)

# Extract key statistics
f_statistic <- anova_summary[[2]][[1]]$`F value`[1]
p_value <- anova_summary[[2]][[1]]$`Pr(>F)`[1]
df_conditions <- anova_summary[[2]][[1]]$Df[1]
df_error <- anova_summary[[2]][[1]]$Df[2]

cat("Repeated Measures ANOVA Results:\n")
cat("F-statistic:", round(f_statistic, 3), "\n")
cat("p-value:", round(p_value, 4), "\n")
cat("Degrees of freedom:", df_conditions, ",", df_error, "\n")
```

## Descriptive Statistics

### Understanding Repeated Measures Descriptive Statistics

In repeated measures designs, we need to understand several types of statistics:

1. **Condition Means**: The average performance across all subjects for each condition
2. **Subject Means**: The average performance across all conditions for each subject
3. **Individual Differences**: The variation between subjects
4. **Within-Subject Variation**: The variation within each subject across conditions

### Mathematical Definitions

**Condition Mean**: $`\bar{Y}_{i.} = \frac{1}{n}\sum_{j=1}^{n}Y_{ij}`$

**Subject Mean**: $`\bar{Y}_{.j} = \frac{1}{k}\sum_{i=1}^{k}Y_{ij}`$

**Grand Mean**: $`\bar{Y}_{..} = \frac{1}{nk}\sum_{i=1}^{k}\sum_{j=1}^{n}Y_{ij}`$

**Individual Differences**: $`SS_{Between Subjects} = k\sum_{j=1}^{n}(\bar{Y}_{.j} - \bar{Y}_{..})^2`$

**Within-Subject Variation**: $`SS_{Within Subjects} = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{.j})^2`$

### Condition and Subject Statistics

```r
# Calculate comprehensive descriptive statistics
library(dplyr)

# Condition means with confidence intervals
condition_stats <- repeated_data %>%
  group_by(condition) %>%
  summarise(
    n = n(),
    mean = mean(score, na.rm = TRUE),
    sd = sd(score, na.rm = TRUE),
    se = sd / sqrt(n),
    ci_lower = mean - qt(0.975, n-1) * se,
    ci_upper = mean + qt(0.975, n-1) * se,
    min = min(score, na.rm = TRUE),
    max = max(score, na.rm = TRUE)
  )

cat("Condition Statistics:\n")
print(condition_stats)

# Subject means with individual differences analysis
subject_stats <- repeated_data %>%
  group_by(subject) %>%
  summarise(
    n = n(),
    mean = mean(score, na.rm = TRUE),
    sd = sd(score, na.rm = TRUE),
    min = min(score, na.rm = TRUE),
    max = max(score, na.rm = TRUE),
    range = max - min
  )

cat("\nSubject Statistics Summary:\n")
cat("Mean subject score:", round(mean(subject_stats$mean), 2), "\n")
cat("SD of subject scores:", round(sd(subject_stats$mean), 2), "\n")
cat("Range of subject scores:", round(range(subject_stats$mean), 2), "\n")
cat("Mean within-subject range:", round(mean(subject_stats$range), 2), "\n")
cat("SD of within-subject ranges:", round(sd(subject_stats$range), 2), "\n")

# Calculate individual differences
grand_mean <- mean(repeated_data$score, na.rm = TRUE)
individual_differences <- subject_stats$mean - grand_mean

cat("\nIndividual Differences Analysis:\n")
cat("Mean individual difference:", round(mean(individual_differences), 2), "\n")
cat("SD of individual differences:", round(sd(individual_differences), 2), "\n")
cat("Range of individual differences:", round(range(individual_differences), 2), "\n")

# Calculate condition differences
condition_differences <- condition_stats$mean - grand_mean
cat("\nCondition Differences from Grand Mean:\n")
for (i in 1:nrow(condition_stats)) {
  cat(condition_stats$condition[i], ":", round(condition_differences[i], 2), "\n")
}
```

### Understanding Individual Differences

```r
# Analyze individual differences in detail
analyze_individual_differences <- function(data, subject_var, condition_var, response_var) {
  # Calculate subject means
  subject_means <- data %>%
    group_by(!!sym(subject_var)) %>%
    summarise(mean = mean(!!sym(response_var), na.rm = TRUE))
  
  # Calculate grand mean
  grand_mean <- mean(data[[response_var]], na.rm = TRUE)
  
  # Individual differences
  subject_means$individual_diff <- subject_means$mean - grand_mean
  
  # Categorize subjects by performance level
  subject_means$performance_level <- cut(subject_means$individual_diff,
                                        breaks = c(-Inf, -sd(subject_means$individual_diff), 
                                                 sd(subject_means$individual_diff), Inf),
                                        labels = c("Low", "Average", "High"))
  
  # Calculate within-subject variability
  within_subject_var <- data %>%
    group_by(!!sym(subject_var)) %>%
    summarise(
      within_sd = sd(!!sym(response_var), na.rm = TRUE),
      within_range = max(!!sym(response_var), na.rm = TRUE) - min(!!sym(response_var), na.rm = TRUE)
    )
  
  # Combine results
  results <- merge(subject_means, within_subject_var, by = subject_var)
  
  cat("INDIVIDUAL DIFFERENCES ANALYSIS:\n")
  cat("=" * 35, "\n")
  cat("Grand mean:", round(grand_mean, 2), "\n")
  cat("SD of individual differences:", round(sd(results$individual_diff), 2), "\n")
  cat("Range of individual differences:", round(range(results$individual_diff), 2), "\n\n")
  
  cat("PERFORMANCE LEVEL DISTRIBUTION:\n")
  cat("=" * 32, "\n")
  print(table(results$performance_level))
  
  cat("\nWITHIN-SUBJECT VARIABILITY:\n")
  cat("=" * 26, "\n")
  cat("Mean within-subject SD:", round(mean(results$within_sd), 2), "\n")
  cat("SD of within-subject SDs:", round(sd(results$within_sd), 2), "\n")
  cat("Mean within-subject range:", round(mean(results$within_range), 2), "\n")
  
  return(results)
}

# Apply individual differences analysis
individual_analysis <- analyze_individual_differences(repeated_data, "subject", "condition", "score")
```

### Visualization

Visualization is crucial for understanding repeated measures data, as it helps identify patterns, individual differences, and potential violations of assumptions.

#### Individual Subject Profiles

The individual subject profiles plot is the most important visualization for repeated measures data as it shows how each subject changes across conditions.

```r
library(ggplot2)
library(gridExtra)

# Enhanced individual subject profiles
p1 <- ggplot(repeated_data, aes(x = condition, y = score, group = subject)) +
  geom_line(alpha = 0.4, size = 0.8) +
  geom_point(alpha = 0.6, size = 2) +
  stat_summary(aes(group = 1), fun = mean, geom = "line", color = "red", size = 2.5) +
  stat_summary(aes(group = 1), fun = mean, geom = "point", color = "red", size = 4, shape = 23, fill = "red") +
  stat_summary(aes(group = 1), fun.data = "mean_cl_normal", geom = "errorbar", width = 0.2, color = "red", size = 1) +
  labs(title = "Individual Subject Profiles",
       subtitle = "Each line represents one subject, red line shows group mean",
       x = "Condition", y = "Score") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"))

# Interpret individual profiles
cat("Individual Profiles Interpretation:\n")
cat("- Parallel lines indicate consistent individual differences\n")
cat("- Non-parallel lines indicate individual differences in treatment effects\n")
cat("- Steep slopes indicate strong treatment effects\n")
cat("- Flat lines indicate no treatment effect\n\n")
```

#### Enhanced Distribution Plots

```r
# Enhanced box plot with individual points
p2 <- ggplot(repeated_data, aes(x = condition, y = score, fill = condition)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.6, size = 2) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "red") +
  labs(title = "Score Distribution by Condition",
       subtitle = "Boxes show quartiles, points show individual scores",
       x = "Condition", y = "Score") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"),
        legend.position = "none")

# Enhanced violin plot
p3 <- ggplot(repeated_data, aes(x = condition, y = score, fill = condition)) +
  geom_violin(alpha = 0.7) +
  geom_boxplot(width = 0.2, alpha = 0.8, outlier.shape = NA) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "red") +
  labs(title = "Score Distribution by Condition",
       subtitle = "Violin shows density, box shows quartiles",
       x = "Condition", y = "Score") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"),
        legend.position = "none")
```

#### Individual Differences Visualization

```r
# Individual differences plot
individual_plot <- ggplot(individual_analysis, aes(x = reorder(subject, individual_diff), y = individual_diff)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_hline(yintercept = c(-sd(individual_analysis$individual_diff), sd(individual_analysis$individual_diff)), 
             linetype = "dotted", color = "orange") +
  labs(title = "Individual Differences from Grand Mean",
       subtitle = "Positive values = above average, negative = below average",
       x = "Subject", y = "Individual Difference") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"),
        axis.text.x = element_text(angle = 45, hjust = 1))

# Within-subject variability plot
within_var_plot <- ggplot(individual_analysis, aes(x = reorder(subject, within_sd), y = within_sd)) +
  geom_bar(stat = "identity", fill = "darkgreen", alpha = 0.7) +
  geom_hline(yintercept = mean(individual_analysis$within_sd), linetype = "dashed", color = "red") +
  labs(title = "Within-Subject Variability",
       subtitle = "Higher bars = more variable performance across conditions",
       x = "Subject", y = "Within-Subject Standard Deviation") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"),
        axis.text.x = element_text(angle = 45, hjust = 1))
```

#### Correlation Matrix Visualization

```r
# Correlation matrix heatmap
library(reshape2)

# Create correlation matrix data for plotting
cor_data <- melt(anova_result$correlation_matrix)
names(cor_data) <- c("Condition1", "Condition2", "Correlation")

# Correlation heatmap
cor_plot <- ggplot(cor_data, aes(x = Condition1, y = Condition2, fill = Correlation)) +
  geom_tile(color = "white", size = 1) +
  geom_text(aes(label = sprintf("%.2f", Correlation)), color = "white", size = 4, fontface = "bold") +
  scale_fill_gradient2(low = "#4575B4", mid = "#FFFFBF", high = "#D73027", 
                       midpoint = 0.5, limits = c(0, 1)) +
  labs(title = "Condition Correlation Matrix",
       subtitle = "Higher correlations support sphericity assumption",
       x = "Condition", y = "Condition") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"))

# Combine all plots
grid.arrange(p1, p2, p3, individual_plot, within_var_plot, cor_plot, ncol = 2)
```

#### Trend Analysis Visualization

```r
# Trend analysis plot
trend_plot <- ggplot(repeated_data, aes(x = as.numeric(condition), y = score, group = subject)) +
  geom_line(alpha = 0.3, size = 0.8) +
  geom_point(alpha = 0.5, size = 2) +
  stat_summary(aes(group = 1), fun = mean, geom = "line", color = "red", size = 2.5) +
  stat_summary(aes(group = 1), fun = mean, geom = "point", color = "red", size = 4) +
  stat_summary(aes(group = 1), fun.data = "mean_cl_normal", geom = "errorbar", width = 0.1, color = "red") +
  geom_smooth(aes(group = 1), method = "lm", se = TRUE, color = "blue", linetype = "dashed") +
  labs(title = "Trend Analysis",
       subtitle = "Blue dashed line shows linear trend",
       x = "Condition (Numeric)", y = "Score") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"))

print(trend_plot)
```

## Effect Size Analysis

Effect sizes are crucial for understanding the practical significance of repeated measures ANOVA results, as they provide information about the magnitude of effects independent of sample size.

### Understanding Effect Size Measures in Repeated Measures

#### Partial Eta-Squared ($`\eta_p^2`$)

Partial eta-squared is the most commonly used effect size measure in repeated measures ANOVA. It represents the proportion of variance in the dependent variable that is explained by the condition effect, controlling for individual differences.

```math
\eta_p^2 = \frac{SS_{Conditions}}{SS_{Conditions} + SS_{Error}}
```

**Advantages:**
- Controls for individual differences
- Ranges from 0 to 1
- Easy to interpret
- Most commonly reported in literature

**Disadvantages:**
- Can be biased upward in small samples
- Values can be larger than in between-subjects designs

#### Eta-Squared ($`\eta^2`$)

Eta-squared represents the proportion of total variance explained by the condition effect.

```math
\eta^2 = \frac{SS_{Conditions}}{SS_{Total}}
```

**Advantages:**
- Intuitive interpretation
- Values sum to 1 across all effects

**Disadvantages:**
- Does not control for individual differences
- Can be misleading in repeated measures designs

#### Omega-Squared ($`\omega^2`$)

Omega-squared is an unbiased estimate of the population effect size.

```math
\omega^2 = \frac{SS_{Conditions} - df_{Conditions} \times MS_{Error}}{SS_{Total} + MS_{Error}}
```

**Advantages:**
- Unbiased estimate
- More conservative than eta-squared

**Disadvantages:**
- Can be negative for small effects
- Less commonly reported

### Comprehensive Effect Size Calculation

```r
# Calculate comprehensive effect sizes for repeated measures ANOVA
calculate_repeated_effect_sizes <- function(anova_result, data, subject_var, condition_var, response_var) {
  # Extract Sum of Squares
  ss_conditions <- anova_result$ss_conditions
  ss_error <- anova_result$ss_error
  ss_total <- anova_result$ss_total
  ss_between_subjects <- anova_result$ss_between_subjects
  ss_within_subjects <- anova_result$ss_within_subjects
  
  # Degrees of freedom
  df_conditions <- anova_result$df_conditions
  df_error <- anova_result$df_error
  df_between_subjects <- anova_result$df_between_subjects
  df_within_subjects <- anova_result$df_within_subjects
  
  # Mean Squares
  ms_error <- anova_result$ms_error
  
  # Partial eta-squared
  partial_eta2 <- ss_conditions / (ss_conditions + ss_error)
  
  # Eta-squared (total)
  eta2 <- ss_conditions / ss_total
  
  # Omega-squared (unbiased)
  omega2 <- (ss_conditions - df_conditions * ms_error) / (ss_total + ms_error)
  
  # Cohen's f (for power analysis)
  cohens_f <- sqrt(partial_eta2 / (1 - partial_eta2))
  
  # Individual differences effect size
  individual_eta2 <- ss_between_subjects / ss_total
  individual_partial_eta2 <- ss_between_subjects / (ss_between_subjects + ss_error)
  
  # Bootstrap confidence intervals for partial eta-squared
  library(boot)
  
  bootstrap_effect_size <- function(data, indices, subject_var, condition_var, response_var) {
    d <- data[indices, ]
    model <- aov(as.formula(paste(response_var, "~", condition_var, "+ Error(", subject_var, "/", condition_var, ")")), data = d)
    summary_model <- summary(model)
    
    ss_conditions <- summary_model[[2]][[1]]$`Sum Sq`[1]
    ss_error <- summary_model[[2]][[1]]$`Sum Sq`[2]
    
    partial_eta2 <- ss_conditions / (ss_conditions + ss_error)
    return(partial_eta2)
  }
  
  # Bootstrap for confidence intervals
  boot_results <- boot(data, bootstrap_effect_size, R = 1000, 
                      subject_var = subject_var, condition_var = condition_var, response_var = response_var)
  
  ci_partial_eta2 <- boot.ci(boot_results, type = "perc")
  
  # Calculate condition-specific effect sizes
  condition_means <- anova_result$condition_means
  grand_mean <- mean(data[[response_var]], na.rm = TRUE)
  
  # Standardized mean differences (Cohen's d) for each condition
  condition_effects <- (condition_means - grand_mean) / sqrt(ms_error)
  
  return(list(
    partial_eta2 = partial_eta2,
    eta2 = eta2,
    omega2 = omega2,
    cohens_f = cohens_f,
    individual_eta2 = individual_eta2,
    individual_partial_eta2 = individual_partial_eta2,
    condition_effects = condition_effects,
    ci_partial_eta2 = ci_partial_eta2,
    boot_results = boot_results
  ))
}

# Apply comprehensive effect size calculation
effect_sizes <- calculate_repeated_effect_sizes(anova_result, repeated_data, "subject", "condition", "score")

# Display comprehensive effect size results
cat("=== COMPREHENSIVE EFFECT SIZE ANALYSIS ===\n\n")

cat("CONDITION EFFECT SIZES:\n")
cat("=" * 22, "\n")
cat("Partial η²:", round(effect_sizes$partial_eta2, 4), "\n")
cat("η²:", round(effect_sizes$eta2, 4), "\n")
cat("ω²:", round(effect_sizes$omega2, 4), "\n")
cat("Cohen's f:", round(effect_sizes$cohens_f, 4), "\n\n")

cat("INDIVIDUAL DIFFERENCES EFFECT SIZES:\n")
cat("=" * 33, "\n")
cat("Individual differences η²:", round(effect_sizes$individual_eta2, 4), "\n")
cat("Individual differences partial η²:", round(effect_sizes$individual_partial_eta2, 4), "\n\n")

cat("CONDITION-SPECIFIC EFFECTS (Cohen's d):\n")
cat("=" * 35, "\n")
for (i in 1:length(effect_sizes$condition_effects)) {
  cat(names(effect_sizes$condition_effects)[i], ":", round(effect_sizes$condition_effects[i], 3), "\n")
}
```

### Effect Size Interpretation

```r
# Enhanced effect size interpretation
interpret_effect_size <- function(eta_sq, measure = "partial_eta2") {
  if (measure == "partial_eta2" || measure == "eta2") {
    if (eta_sq < 0.01) {
      return("Negligible effect (< 1% of variance explained)")
    } else if (eta_sq < 0.06) {
      return("Small effect (1-6% of variance explained)")
    } else if (eta_sq < 0.14) {
      return("Medium effect (6-14% of variance explained)")
    } else {
      return("Large effect (> 14% of variance explained)")
    }
  } else if (measure == "f") {
    if (eta_sq < 0.10) {
      return("Small effect")
    } else if (eta_sq < 0.25) {
      return("Medium effect")
    } else if (eta_sq < 0.40) {
      return("Large effect")
    } else {
      return("Very large effect")
    }
  } else if (measure == "d") {
    if (abs(eta_sq) < 0.20) {
      return("Negligible effect")
    } else if (abs(eta_sq) < 0.50) {
      return("Small effect")
    } else if (abs(eta_sq) < 0.80) {
      return("Medium effect")
    } else {
      return("Large effect")
    }
  }
}

cat("EFFECT SIZE INTERPRETATION:\n")
cat("=" * 25, "\n")
cat("Condition effect:", interpret_effect_size(effect_sizes$partial_eta2), "\n")
cat("Individual differences:", interpret_effect_size(effect_sizes$individual_eta2), "\n\n")

# Practical significance assessment
cat("PRACTICAL SIGNIFICANCE ASSESSMENT:\n")
cat("=" * 32, "\n")
if (effect_sizes$partial_eta2 > 0.06) {
  cat("✓ Condition effect is practically significant\n")
} else {
  cat("✗ Condition effect may not be practically significant\n")
}

if (effect_sizes$individual_partial_eta2 > 0.06) {
  cat("✓ Individual differences are practically significant\n")
} else {
  cat("✗ Individual differences may not be practically significant\n")
}
```

### Effect Size Visualization

```r
# Create effect size comparison plot
effect_size_data <- data.frame(
  Effect = rep(c("Condition Effect", "Individual Differences"), 3),
  Measure = rep(c("Partial η²", "η²", "ω²"), each = 2),
  Value = c(effect_sizes$partial_eta2, effect_sizes$individual_partial_eta2,
            effect_sizes$eta2, effect_sizes$individual_eta2,
            effect_sizes$omega2, effect_sizes$individual_partial_eta2)  # Using individual partial eta2 as proxy
)

ggplot(effect_size_data, aes(x = Effect, y = Value, fill = Measure)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  geom_hline(yintercept = c(0.01, 0.06, 0.14), linetype = "dashed", color = "red", alpha = 0.7) +
  annotate("text", x = 0.5, y = 0.01, label = "Negligible", hjust = 0, color = "red") +
  annotate("text", x = 0.5, y = 0.06, label = "Small", hjust = 0, color = "red") +
  annotate("text", x = 0.5, y = 0.14, label = "Medium", hjust = 0, color = "red") +
  labs(title = "Effect Size Comparison",
       subtitle = "Condition effects vs individual differences",
       x = "Effect Type", y = "Effect Size", fill = "Measure") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"))

# Condition-specific effects plot
condition_effects_data <- data.frame(
  Condition = names(effect_sizes$condition_effects),
  Effect_Size = effect_sizes$condition_effects
)

ggplot(condition_effects_data, aes(x = Condition, y = Effect_Size, fill = Condition)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  geom_hline(yintercept = c(-0.2, 0, 0.2), linetype = c("dashed", "solid", "dashed"), 
             color = c("red", "black", "red"), alpha = 0.7) +
  annotate("text", x = 0.5, y = -0.2, label = "Small", hjust = 0, color = "red") +
  annotate("text", x = 0.5, y = 0.2, label = "Small", hjust = 0, color = "red") +
  labs(title = "Condition-Specific Effect Sizes (Cohen's d)",
       subtitle = "Relative to grand mean",
       x = "Condition", y = "Effect Size (Cohen's d)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"),
        legend.position = "none")
```

## Post Hoc Tests

### Pairwise Comparisons

```r
# Perform pairwise comparisons
library(emmeans)

# Get estimated marginal means
emm <- emmeans(repeated_model, ~ condition)
print(emm)

# Pairwise comparisons with different corrections
pairs_result <- pairs(emm, adjust = "bonferroni")
print(pairs_result)

# Tukey's HSD
tukey_result <- pairs(emm, adjust = "tukey")
print(tukey_result)

# Extract significant pairs
significant_pairs <- tukey_result[tukey_result$p.value < 0.05, ]
if (nrow(significant_pairs) > 0) {
  cat("Significant pairwise differences (Tukey's HSD, p < 0.05):\n")
  print(significant_pairs)
} else {
  cat("No significant pairwise differences found.\n")
}
```

### Trend Analysis

```r
# Polynomial contrasts for trend analysis
contrasts(repeated_data$condition) <- contr.poly(3)
trend_model <- aov(score ~ condition + Error(subject/condition), data = repeated_data)
trend_summary <- summary(trend_model)
print(trend_summary)

# Extract trend components
linear_f <- trend_summary[[2]][[1]]$`F value`[1]
quadratic_f <- trend_summary[[2]][[1]]$`F value`[2]
linear_p <- trend_summary[[2]][[1]]$`Pr(>F)`[1]
quadratic_p <- trend_summary[[2]][[1]]$`Pr(>F)`[2]

cat("Trend Analysis Results:\n")
cat("Linear trend F-statistic:", round(linear_f, 3), "\n")
cat("Linear trend p-value:", round(linear_p, 4), "\n")
cat("Quadratic trend F-statistic:", round(quadratic_f, 3), "\n")
cat("Quadratic trend p-value:", round(quadratic_p, 4), "\n")
```

## Assumption Checking

Repeated measures ANOVA relies on several key assumptions that must be verified before interpreting results. Violations of these assumptions can lead to incorrect conclusions and inflated Type I error rates.

### Key Assumptions

1. **Sphericity**: The variances of the differences between all pairs of conditions are equal
2. **Normality**: Residuals should be normally distributed
3. **Independence**: Observations should be independent (within-subjects correlation is expected)
4. **Linearity**: The relationship between conditions and the dependent variable should be linear

### Comprehensive Sphericity Testing

The sphericity assumption is the most critical assumption for repeated measures ANOVA. It requires that the variance of the differences between any two conditions is constant.

```r
# Enhanced function to check sphericity assumption
check_sphericity <- function(data, subject_var, condition_var, response_var, alpha = 0.05) {
  cat("=== COMPREHENSIVE SPHERICITY TESTING ===\n\n")
  
  # Create wide format for analysis
  wide_data <- data %>%
    tidyr::pivot_wider(
      names_from = condition_var,
      values_from = response_var,
      names_prefix = "condition_"
    )
  
  # Remove subject column for correlation matrix
  cor_matrix <- wide_data %>%
    select(-subject_var) %>%
    cor(use = "complete.obs")
  
  cat("CORRELATION MATRIX OF CONDITIONS:\n")
  cat("=" * 35, "\n")
  print(round(cor_matrix, 3))
  
  # Calculate condition variances
  condition_vars <- apply(wide_data %>% select(-subject_var), 2, var, na.rm = TRUE)
  cat("\nCONDITION VARIANCES:\n")
  cat("=" * 20, "\n")
  for (i in 1:length(condition_vars)) {
    cat(names(condition_vars)[i], ":", round(condition_vars[i], 2), "\n")
  }
  
  # Calculate variance ratios
  max_var <- max(condition_vars)
  min_var <- min(condition_vars)
  var_ratio <- max_var / min_var
  
  cat("\nVARIANCE RATIO (max/min):", round(var_ratio, 2), "\n")
  
  # Calculate difference variances
  n_conditions <- ncol(cor_matrix)
  diff_vars <- numeric(0)
  diff_names <- character(0)
  
  for (i in 1:(n_conditions-1)) {
    for (j in (i+1):n_conditions) {
      diff <- wide_data[, i+1] - wide_data[, j+1]  # +1 because first column is subject
      diff_vars <- c(diff_vars, var(diff, na.rm = TRUE))
      diff_names <- c(diff_names, paste(names(wide_data)[i+1], "-", names(wide_data)[j+1]))
    }
  }
  
  cat("\nDIFFERENCE VARIANCES:\n")
  cat("=" * 20, "\n")
  for (i in 1:length(diff_vars)) {
    cat(diff_names[i], ":", round(diff_vars[i], 2), "\n")
  }
  
  # Test for homogeneity of difference variances
  diff_var_ratio <- max(diff_vars) / min(diff_vars)
  cat("\nDIFFERENCE VARIANCE RATIO (max/min):", round(diff_var_ratio, 2), "\n")
  
  # Calculate sphericity measures
  n_subjects <- nrow(wide_data)
  
  # Greenhouse-Geisser epsilon
  gg_epsilon <- 1 / (n_conditions - 1)
  
  # Huynh-Feldt epsilon (approximation)
  hf_epsilon <- min(1, gg_epsilon * (n_subjects - 1) / (n_subjects - 1 - (n_conditions - 1)))
  
  # Lower-bound epsilon
  lb_epsilon <- 1 / (n_conditions - 1)
  
  cat("\nSPHERICITY CORRECTIONS:\n")
  cat("=" * 22, "\n")
  cat("Greenhouse-Geisser ε:", round(gg_epsilon, 3), "\n")
  cat("Huynh-Feldt ε:", round(hf_epsilon, 3), "\n")
  cat("Lower-bound ε:", round(lb_epsilon, 3), "\n")
  
  # Mauchly's test (if available)
  library(car)
  tryCatch({
    model <- aov(as.formula(paste(response_var, "~", condition_var, "+ Error(", subject_var, "/", condition_var, ")")), data = data)
    mauchly_result <- mauchly.test(model, id = data[[subject_var]])
    cat("\nMAUCHLY'S TEST OF SPHERICITY:\n")
    cat("=" * 30, "\n")
    cat("W-statistic:", round(mauchly_result$statistic, 4), "\n")
    cat("p-value:", round(mauchly_result$p.value, 4), "\n")
    cat("Decision:", ifelse(mauchly_result$p.value >= alpha, "✓ Sphericity met", "✗ Sphericity violated"), "\n")
  }, error = function(e) {
    cat("\nMauchly's test not available for this data structure\n")
  })
  
  # Sphericity assessment
  cat("\nSPHERICITY ASSESSMENT:\n")
  cat("=" * 21, "\n")
  
  sphericity_indicators <- 0
  total_indicators <- 0
  
  # Check correlation matrix
  if (all(cor_matrix[upper.tri(cor_matrix)] > 0.3)) {
    cat("✓ Correlations are reasonably high\n")
    sphericity_indicators <- sphericity_indicators + 1
  } else {
    cat("✗ Some correlations are low\n")
  }
  total_indicators <- total_indicators + 1
  
  # Check variance ratio
  if (var_ratio <= 4) {
    cat("✓ Condition variances are reasonably equal\n")
    sphericity_indicators <- sphericity_indicators + 1
  } else {
    cat("✗ Condition variances are unequal\n")
  }
  total_indicators <- total_indicators + 1
  
  # Check difference variance ratio
  if (diff_var_ratio <= 4) {
    cat("✓ Difference variances are reasonably equal\n")
    sphericity_indicators <- sphericity_indicators + 1
  } else {
    cat("✗ Difference variances are unequal\n")
  }
  total_indicators <- total_indicators + 1
  
  # Check epsilon values
  if (gg_epsilon >= 0.75) {
    cat("✓ Greenhouse-Geisser epsilon suggests sphericity\n")
    sphericity_indicators <- sphericity_indicators + 1
  } else {
    cat("✗ Greenhouse-Geisser epsilon suggests sphericity violation\n")
  }
  total_indicators <- total_indicators + 1
  
  cat("\nSphericity indicators met:", sphericity_indicators, "out of", total_indicators, "\n")
  
  # Recommendations
  cat("\nRECOMMENDATIONS:\n")
  cat("=" * 15, "\n")
  
  if (sphericity_indicators >= 3) {
    cat("✓ Sphericity assumption appears to be met\n")
    cat("✓ Standard repeated measures ANOVA is appropriate\n")
  } else if (sphericity_indicators >= 2) {
    cat("⚠ Sphericity assumption may be questionable\n")
    cat("⚠ Consider using Greenhouse-Geisser or Huynh-Feldt corrections\n")
  } else {
    cat("✗ Sphericity assumption appears to be violated\n")
    cat("✗ Use Greenhouse-Geisser or Huynh-Feldt corrections\n")
    cat("✗ Consider nonparametric alternatives\n")
  }
  
  # Create diagnostic plots
  library(ggplot2)
  library(gridExtra)
  
  # Correlation heatmap
  library(reshape2)
  cor_data <- melt(cor_matrix)
  names(cor_data) <- c("Condition1", "Condition2", "Correlation")
  
  cor_plot <- ggplot(cor_data, aes(x = Condition1, y = Condition2, fill = Correlation)) +
    geom_tile(color = "white", size = 1) +
    geom_text(aes(label = sprintf("%.2f", Correlation)), color = "white", size = 4, fontface = "bold") +
    scale_fill_gradient2(low = "#4575B4", mid = "#FFFFBF", high = "#D73027", 
                         midpoint = 0.5, limits = c(0, 1)) +
    labs(title = "Condition Correlation Matrix",
         subtitle = "Higher correlations support sphericity",
         x = "Condition", y = "Condition") +
    theme_minimal() +
    theme(plot.title = element_text(size = 12, face = "bold"),
          plot.subtitle = element_text(size = 10, color = "gray50"))
  
  # Difference variances plot
  diff_var_data <- data.frame(
    Comparison = diff_names,
    Variance = diff_vars
  )
  
  diff_var_plot <- ggplot(diff_var_data, aes(x = reorder(Comparison, Variance), y = Variance)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
    geom_hline(yintercept = mean(diff_vars), linetype = "dashed", color = "red") +
    labs(title = "Variances of Condition Differences",
         subtitle = "Equal variances support sphericity",
         x = "Condition Comparison", y = "Variance") +
    theme_minimal() +
    theme(plot.title = element_text(size = 12, face = "bold"),
          plot.subtitle = element_text(size = 10, color = "gray50"),
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Combine plots
  combined_plot <- grid.arrange(cor_plot, diff_var_plot, ncol = 2)
  
  return(list(
    correlation_matrix = cor_matrix,
    condition_vars = condition_vars,
    diff_vars = diff_vars,
    diff_names = diff_names,
    gg_epsilon = gg_epsilon,
    hf_epsilon = hf_epsilon,
    lb_epsilon = lb_epsilon,
    sphericity_indicators = sphericity_indicators,
    total_indicators = total_indicators,
    cor_plot = cor_plot,
    diff_var_plot = diff_var_plot,
    combined_plot = combined_plot
  ))
}

# Check sphericity with enhanced function
sphericity_results <- check_sphericity(repeated_data, "subject", "condition", "score")
```

### Comprehensive Normality Testing

```r
# Enhanced function to check normality for repeated measures
check_normality_repeated <- function(data, subject_var, condition_var, response_var, alpha = 0.05) {
  cat("=== COMPREHENSIVE NORMALITY TESTS FOR REPEATED MEASURES ===\n\n")
  
  # Fit the model
  model <- aov(as.formula(paste(response_var, "~", condition_var, "+ Error(", subject_var, "/", condition_var, ")")), data = data)
  
  # Extract residuals properly
  # For repeated measures, we need to extract residuals from the within-subjects error term
  model_residuals <- residuals(model)
  
  # If the above doesn't work, calculate residuals manually
  if (length(model_residuals) == 0) {
    # Calculate residuals as deviations from condition means
    condition_means <- tapply(data[[response_var]], data[[condition_var]], mean, na.rm = TRUE)
    residuals <- data[[response_var]] - condition_means[data[[condition_var]]]
  } else {
    residuals <- model_residuals
  }
  
  cat("NORMALITY TESTS ON RESIDUALS:\n")
  cat("=" * 35, "\n")
  
  # Multiple normality tests
  tests <- list()
  
  # Shapiro-Wilk test
  shapiro_result <- shapiro.test(residuals)
  tests$shapiro <- shapiro_result
  cat("Shapiro-Wilk test:\n")
  cat("  W =", round(shapiro_result$statistic, 4), "\n")
  cat("  p-value =", round(shapiro_result$p.value, 4), "\n")
  cat("  Decision:", ifelse(shapiro_result$p.value >= alpha, "✓ Normal", "✗ Non-normal"), "\n\n")
  
  # Kolmogorov-Smirnov test
  ks_result <- ks.test(residuals, "pnorm", mean = mean(residuals), sd = sd(residuals))
  tests$ks <- ks_result
  cat("Kolmogorov-Smirnov test:\n")
  cat("  D =", round(ks_result$statistic, 4), "\n")
  cat("  p-value =", round(ks_result$p.value, 4), "\n")
  cat("  Decision:", ifelse(ks_result$p.value >= alpha, "✓ Normal", "✗ Non-normal"), "\n\n")
  
  # Anderson-Darling test
  library(nortest)
  ad_result <- ad.test(residuals)
  tests$ad <- ad_result
  cat("Anderson-Darling test:\n")
  cat("  A =", round(ad_result$statistic, 4), "\n")
  cat("  p-value =", round(ad_result$p.value, 4), "\n")
  cat("  Decision:", ifelse(ad_result$p.value >= alpha, "✓ Normal", "✗ Non-normal"), "\n\n")
  
  # Lilliefors test
  lillie_result <- lillie.test(residuals)
  tests$lillie <- lillie_result
  cat("Lilliefors test:\n")
  cat("  D =", round(lillie_result$statistic, 4), "\n")
  cat("  p-value =", round(lillie_result$p.value, 4), "\n")
  cat("  Decision:", ifelse(lillie_result$p.value >= alpha, "✓ Normal", "✗ Non-normal"), "\n\n")
  
  # Summary of normality tests
  cat("NORMALITY TEST SUMMARY:\n")
  cat("=" * 25, "\n")
  
  p_values <- c(shapiro_result$p.value, ks_result$p.value, ad_result$p.value, lillie_result$p.value)
  test_names <- c("Shapiro-Wilk", "Kolmogorov-Smirnov", "Anderson-Darling", "Lilliefors")
  
  normal_tests <- sum(p_values >= alpha)
  total_tests <- length(p_values)
  
  cat("Tests supporting normality:", normal_tests, "out of", total_tests, "\n")
  
  if (normal_tests >= 3) {
    cat("✓ Normality assumption appears to be met\n")
  } else if (normal_tests >= 2) {
    cat("⚠ Normality assumption may be questionable\n")
  } else {
    cat("✗ Normality assumption appears to be violated\n")
  }
  
  # Check normality within each condition
  cat("\nNORMALITY WITHIN CONDITIONS:\n")
  cat("=" * 27, "\n")
  
  conditions <- unique(data[[condition_var]])
  condition_normality <- list()
  
  for (cond in conditions) {
    cond_data <- data[data[[condition_var]] == cond, response_var]
    cond_residuals <- cond_data - mean(cond_data, na.rm = TRUE)
    
    shapiro_cond <- shapiro.test(cond_residuals)
    condition_normality[[cond]] <- shapiro_cond
    
    cat(cond, "Shapiro-Wilk p-value:", round(shapiro_cond$p.value, 4))
    if (shapiro_cond$p.value >= alpha) {
      cat(" ✓ Normal\n")
    } else {
      cat(" ✗ Non-normal\n")
    }
  }
  
  # Create diagnostic plots
  library(ggplot2)
  library(gridExtra)
  
  # Q-Q plot with confidence bands
  qq_plot <- ggplot(data.frame(residuals = residuals), aes(sample = residuals)) +
    stat_qq() +
    stat_qq_line(color = "red", linetype = "dashed") +
    labs(title = "Q-Q Plot of Residuals",
         subtitle = "Points should fall along the red line",
         x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_minimal() +
    theme(plot.title = element_text(size = 12, face = "bold"),
          plot.subtitle = element_text(size = 10, color = "gray50"))
  
  # Histogram with normal curve
  hist_plot <- ggplot(data.frame(residuals = residuals), aes(x = residuals)) +
    geom_histogram(aes(y = ..density..), bins = 15, fill = "steelblue", alpha = 0.7) +
    stat_function(fun = dnorm, args = list(mean = mean(residuals), sd = sd(residuals)), 
                 color = "red", size = 1) +
    labs(title = "Histogram of Residuals with Normal Curve",
         subtitle = "Histogram should approximate the red normal curve",
         x = "Residuals", y = "Density") +
    theme_minimal() +
    theme(plot.title = element_text(size = 12, face = "bold"),
          plot.subtitle = element_text(size = 10, color = "gray50"))
  
  # Residuals vs fitted values
  fitted_values <- fitted(model)
  if (length(fitted_values) == 0) {
    # Calculate fitted values manually
    fitted_values <- condition_means[data[[condition_var]]]
  }
  
  fitted_plot <- ggplot(data.frame(residuals = residuals, fitted = fitted_values), 
                        aes(x = fitted, y = residuals)) +
    geom_point(alpha = 0.7) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", se = TRUE, color = "blue") +
    labs(title = "Residuals vs Fitted Values",
         subtitle = "Check for homoscedasticity and linearity",
         x = "Fitted Values", y = "Residuals") +
    theme_minimal() +
    theme(plot.title = element_text(size = 12, face = "bold"),
          plot.subtitle = element_text(size = 10, color = "gray50"))
  
  # Condition-specific normality plots
  cond_norm_plots <- list()
  for (i in 1:length(conditions)) {
    cond <- conditions[i]
    cond_data <- data[data[[condition_var]] == cond, response_var]
    cond_residuals <- cond_data - mean(cond_data, na.rm = TRUE)
    
    p <- ggplot(data.frame(residuals = cond_residuals), aes(sample = residuals)) +
      stat_qq() +
      stat_qq_line(color = "red", linetype = "dashed") +
      labs(title = paste("Q-Q Plot for", cond),
           x = "Theoretical Quantiles", y = "Sample Quantiles") +
      theme_minimal() +
      theme(plot.title = element_text(size = 10, face = "bold"))
    
    cond_norm_plots[[i]] <- p
  }
  
  # Combine plots
  combined_plot <- grid.arrange(qq_plot, hist_plot, fitted_plot, ncol = 2)
  
  # Recommendations based on results
  cat("\nRECOMMENDATIONS:\n")
  cat("=" * 15, "\n")
  
  if (normal_tests >= 3) {
    cat("✓ Repeated measures ANOVA is appropriate for these data\n")
    cat("✓ All normality assumptions appear to be met\n")
  } else if (normal_tests >= 2) {
    cat("⚠ Consider robust alternatives or data transformation\n")
    cat("⚠ Repeated measures ANOVA may still be appropriate with caution\n")
  } else {
    cat("✗ Consider nonparametric alternatives:\n")
    cat("  - Friedman test\n")
    cat("  - Wilcoxon signed-rank test for pairwise comparisons\n")
    cat("  - Permutation tests\n")
    cat("  - Data transformation (log, square root, etc.)\n")
  }
  
  return(list(
    tests = tests,
    p_values = p_values,
    test_names = test_names,
    normal_tests = normal_tests,
    total_tests = total_tests,
    condition_normality = condition_normality,
    qq_plot = qq_plot,
    hist_plot = hist_plot,
    fitted_plot = fitted_plot,
    cond_norm_plots = cond_norm_plots,
    combined_plot = combined_plot
  ))
}

# Check normality with enhanced function
normality_results <- check_normality_repeated(repeated_data, "subject", "condition", "score")
```

## Mixed ANOVA

### Between-Subjects and Within-Subjects Factors

```r
# Simulate mixed ANOVA data
set.seed(123)
n_per_group <- 15

# Generate data for 2x3 mixed design
# Between-subjects factor: Group (A, B)
# Within-subjects factor: Time (Pre, Post, Follow-up)

group <- rep(c("Group A", "Group B"), each = n_per_group * 3)
subject_id <- rep(1:(n_per_group * 2), each = 3)
time <- rep(c("Pre", "Post", "Follow-up"), times = n_per_group * 2)

# Generate scores with group and time effects
scores <- numeric(length(group))
for (i in 1:length(group)) {
  subject_effect <- rnorm(1, 0, 8)  # Individual differences
  
  if (group[i] == "Group A") {
    if (time[i] == "Pre") {
      scores[i] <- 70 + subject_effect + rnorm(1, 0, 4)
    } else if (time[i] == "Post") {
      scores[i] <- 75 + subject_effect + rnorm(1, 0, 4)
    } else {
      scores[i] <- 78 + subject_effect + rnorm(1, 0, 4)
    }
  } else {
    if (time[i] == "Pre") {
      scores[i] <- 72 + subject_effect + rnorm(1, 0, 4)
    } else if (time[i] == "Post") {
      scores[i] <- 80 + subject_effect + rnorm(1, 0, 4)
    } else {
      scores[i] <- 85 + subject_effect + rnorm(1, 0, 4)
    }
  }
}

# Create data frame
mixed_data <- data.frame(
  subject = factor(subject_id),
  group = factor(group),
  time = factor(time, levels = c("Pre", "Post", "Follow-up")),
  score = scores
)

# Perform mixed ANOVA
mixed_model <- aov(score ~ group * time + Error(subject/time), data = mixed_data)
print(summary(mixed_model))

# Extract results
mixed_summary <- summary(mixed_model)

# Between-subjects effect (Group)
group_f <- mixed_summary[[1]][[1]]$`F value`[1]
group_p <- mixed_summary[[1]][[1]]$`Pr(>F)`[1]

# Within-subjects effects (Time and Group:Time interaction)
time_f <- mixed_summary[[2]][[1]]$`F value`[1]
time_p <- mixed_summary[[2]][[1]]$`Pr(>F)`[1]
interaction_f <- mixed_summary[[2]][[1]]$`F value`[2]
interaction_p <- mixed_summary[[2]][[1]]$`Pr(>F)`[2]

cat("Mixed ANOVA Results:\n")
cat("Group (between-subjects) F-statistic:", round(group_f, 3), "\n")
cat("Group p-value:", round(group_p, 4), "\n")
cat("Time (within-subjects) F-statistic:", round(time_f, 3), "\n")
cat("Time p-value:", round(time_p, 4), "\n")
cat("Group:Time interaction F-statistic:", round(interaction_f, 3), "\n")
cat("Group:Time interaction p-value:", round(interaction_p, 4), "\n")
```

## Nonparametric Alternatives

### Friedman Test

```r
# Perform Friedman test (nonparametric alternative)
friedman_result <- friedman.test(score ~ condition | subject, data = repeated_data)
print(friedman_result)

# Compare with parametric results
cat("Comparison of parametric and nonparametric tests:\n")
cat("Repeated measures ANOVA F-statistic:", round(f_statistic, 3), "\n")
cat("Repeated measures ANOVA p-value:", round(p_value, 4), "\n")
cat("Friedman chi-squared:", round(friedman_result$statistic, 3), "\n")
cat("Friedman p-value:", round(friedman_result$p.value, 4), "\n")

# Post hoc for Friedman test
library(PMCMRplus)
friedman_posthoc <- frdAllPairsConoverTest(score ~ condition | subject, data = repeated_data)
print(friedman_posthoc)
```

### Wilcoxon Signed-Rank Test for Pairwise Comparisons

```r
# Pairwise Wilcoxon signed-rank tests
pairwise_wilcox <- function(data, subject_var, condition_var, response_var) {
  conditions <- unique(data[[condition_var]])
  n_conditions <- length(conditions)
  
  results <- list()
  pair_count <- 1
  
  for (i in 1:(n_conditions-1)) {
    for (j in (i+1):n_conditions) {
      cond1 <- conditions[i]
      cond2 <- conditions[j]
      
      # Extract paired data
      data1 <- data[data[[condition_var]] == cond1, response_var]
      data2 <- data[data[[condition_var]] == cond2, response_var]
      
      # Perform Wilcoxon signed-rank test
      wilcox_result <- wilcox.test(data1, data2, paired = TRUE)
      
      results[[pair_count]] <- list(
        comparison = paste(cond1, "vs", cond2),
        statistic = wilcox_result$statistic,
        p_value = wilcox_result$p.value
      )
      
      pair_count <- pair_count + 1
    }
  }
  
  return(results)
}

# Apply pairwise Wilcoxon tests
wilcox_results <- pairwise_wilcox(repeated_data, "subject", "condition", "score")

cat("Pairwise Wilcoxon Signed-Rank Tests:\n")
for (result in wilcox_results) {
  cat(result$comparison, ":\n")
  cat("  V-statistic:", result$statistic, "\n")
  cat("  p-value:", round(result$p_value, 4), "\n")
  if (result$p_value < 0.05) {
    cat("  Significant\n")
  } else {
    cat("  Non-significant\n")
  }
  cat("\n")
}
```

## Power Analysis

### Power Analysis for Repeated Measures ANOVA

```r
library(pwr)

# Power analysis for repeated measures ANOVA
power_analysis_repeated <- function(n_subjects, n_conditions, effect_size, alpha = 0.05) {
  # Calculate power using F-test
  power_result <- pwr.f.test(k = n_conditions, n = n_subjects, f = effect_size, sig.level = alpha)
  
  # Calculate required sample size for 80% power
  sample_size_result <- pwr.f.test(k = n_conditions, f = effect_size, sig.level = alpha, power = 0.8)
  
  return(list(
    power = power_result$power,
    required_n = ceiling(sample_size_result$n),
    effect_size = effect_size,
    alpha = alpha,
    n_conditions = n_conditions
  ))
}

# Apply power analysis
f_effect_size <- effect_sizes$cohens_f
repeated_power <- power_analysis_repeated(
  n_subjects = length(unique(repeated_data$subject)),
  n_conditions = length(unique(repeated_data$condition)),
  effect_size = f_effect_size
)

cat("Power Analysis Results:\n")
cat("Effect size f:", round(f_effect_size, 3), "\n")
cat("Current power:", round(repeated_power$power, 3), "\n")
cat("Required sample size for 80% power:", repeated_power$required_n, "\n")
```

## Practical Examples

### Example 1: Clinical Trial

```r
# Simulate clinical trial data
set.seed(123)
n_patients <- 25

# Generate data for 4 time points
patient_id <- rep(1:n_patients, each = 4)
time_point <- rep(c("Baseline", "Week 4", "Week 8", "Week 12"), times = n_patients)

# Generate symptom scores
scores <- numeric(length(patient_id))
for (i in 1:length(patient_id)) {
  patient_effect <- rnorm(1, 0, 15)  # Individual differences
  
  if (time_point[i] == "Baseline") {
    scores[i] <- 80 + patient_effect + rnorm(1, 0, 8)
  } else if (time_point[i] == "Week 4") {
    scores[i] <- 70 + patient_effect + rnorm(1, 0, 8)
  } else if (time_point[i] == "Week 8") {
    scores[i] <- 60 + patient_effect + rnorm(1, 0, 8)
  } else {
    scores[i] <- 55 + patient_effect + rnorm(1, 0, 8)
  }
}

# Create data frame
clinical_data <- data.frame(
  patient = factor(patient_id),
  time = factor(time_point, levels = c("Baseline", "Week 4", "Week 8", "Week 12")),
  symptom_score = scores
)

# Perform repeated measures ANOVA
clinical_model <- aov(symptom_score ~ time + Error(patient/time), data = clinical_data)
print(summary(clinical_model))

# Visualize results
ggplot(clinical_data, aes(x = time, y = symptom_score, group = patient)) +
  geom_line(alpha = 0.3) +
  geom_point(alpha = 0.5) +
  stat_summary(aes(group = 1), fun = mean, geom = "line", color = "red", size = 2) +
  stat_summary(aes(group = 1), fun = mean, geom = "point", color = "red", size = 3) +
  labs(title = "Symptom Scores Over Time", x = "Time Point", y = "Symptom Score") +
  theme_minimal()
```

### Example 2: Learning Study

```r
# Simulate learning study data
set.seed(123)
n_students <- 30

# Generate data for 3 learning sessions
student_id <- rep(1:n_students, each = 3)
session <- rep(c("Session 1", "Session 2", "Session 3"), times = n_students)

# Generate performance scores
scores <- numeric(length(student_id))
for (i in 1:length(student_id)) {
  student_effect <- rnorm(1, 0, 12)  # Individual differences
  
  if (session[i] == "Session 1") {
    scores[i] <- 65 + student_effect + rnorm(1, 0, 6)
  } else if (session[i] == "Session 2") {
    scores[i] <- 75 + student_effect + rnorm(1, 0, 6)
  } else {
    scores[i] <- 82 + student_effect + rnorm(1, 0, 6)
  }
}

# Create data frame
learning_data <- data.frame(
  student = factor(student_id),
  session = factor(session, levels = c("Session 1", "Session 2", "Session 3")),
  performance = scores
)

# Perform repeated measures ANOVA
learning_model <- aov(performance ~ session + Error(student/session), data = learning_data)
print(summary(learning_model))

# Post hoc analysis
learning_emm <- emmeans(learning_model, ~ session)
learning_pairs <- pairs(learning_emm, adjust = "tukey")
print(learning_pairs)
```

### Example 3: Exercise Study

```r
# Simulate exercise study data
set.seed(123)
n_participants <- 20

# Generate data for 2x3 mixed design
# Between-subjects: Exercise Type (Aerobic, Strength)
# Within-subjects: Time (Pre, Mid, Post)

exercise_type <- rep(c("Aerobic", "Strength"), each = n_participants * 3)
participant_id <- rep(1:(n_participants * 2), each = 3)
time_point <- rep(c("Pre", "Mid", "Post"), times = n_participants * 2)

# Generate fitness scores
scores <- numeric(length(exercise_type))
for (i in 1:length(exercise_type)) {
  participant_effect <- rnorm(1, 0, 10)  # Individual differences
  
  if (exercise_type[i] == "Aerobic") {
    if (time_point[i] == "Pre") {
      scores[i] <- 60 + participant_effect + rnorm(1, 0, 5)
    } else if (time_point[i] == "Mid") {
      scores[i] <- 70 + participant_effect + rnorm(1, 0, 5)
    } else {
      scores[i] <- 75 + participant_effect + rnorm(1, 0, 5)
    }
  } else {
    if (time_point[i] == "Pre") {
      scores[i] <- 55 + participant_effect + rnorm(1, 0, 5)
    } else if (time_point[i] == "Mid") {
      scores[i] <- 65 + participant_effect + rnorm(1, 0, 5)
    } else {
      scores[i] <- 80 + participant_effect + rnorm(1, 0, 5)
    }
  }
}

# Create data frame
exercise_data <- data.frame(
  participant = factor(participant_id),
  exercise_type = factor(exercise_type),
  time = factor(time_point, levels = c("Pre", "Mid", "Post")),
  fitness_score = scores
)

# Perform mixed ANOVA
exercise_model <- aov(fitness_score ~ exercise_type * time + Error(participant/time), data = exercise_data)
print(summary(exercise_model))

# Visualize interaction
ggplot(exercise_data, aes(x = time, y = fitness_score, color = exercise_type, group = exercise_type)) +
  stat_summary(fun = mean, geom = "line", size = 1) +
  stat_summary(fun = mean, geom = "point", size = 3) +
  labs(title = "Fitness Scores by Exercise Type and Time",
       x = "Time Point", y = "Fitness Score", color = "Exercise Type") +
  theme_minimal()
```

## Best Practices

### Test Selection Guidelines

```r
# Function to help choose appropriate repeated measures test
choose_repeated_test <- function(data, subject_var, condition_var, response_var) {
  cat("=== REPEATED MEASURES TEST SELECTION ===\n")
  
  # Check normality
  normality_results <- check_normality_repeated(data, subject_var, condition_var, response_var)
  
  # Check sphericity
  sphericity_results <- check_sphericity(data, subject_var, condition_var, response_var)
  
  # Check sample size
  n_subjects <- length(unique(data[[subject_var]]))
  n_conditions <- length(unique(data[[condition_var]]))
  cat("Sample size:", n_subjects, "subjects,", n_conditions, "conditions\n")
  
  cat("\nFINAL RECOMMENDATION:\n")
  
  # Decision logic
  normal_residuals <- normality_results$shapiro_test$p.value >= 0.05
  sphericity_ok <- sphericity_results$gg_epsilon >= 0.75
  
  if (normal_residuals && sphericity_ok) {
    cat("Use standard repeated measures ANOVA\n")
    cat("All assumptions are met\n")
  } else if (normal_residuals && !sphericity_ok) {
    cat("Use repeated measures ANOVA with Greenhouse-Geisser correction\n")
    cat("Residuals are normal but sphericity is violated\n")
  } else {
    cat("Use Friedman test (nonparametric alternative)\n")
    cat("Residuals are not normal\n")
  }
  
  return(list(
    normality = normality_results,
    sphericity = sphericity_results,
    n_subjects = n_subjects,
    n_conditions = n_conditions
  ))
}

# Apply to repeated measures data
test_selection <- choose_repeated_test(repeated_data, "subject", "condition", "score")
```

### Reporting Guidelines

```r
# Function to generate comprehensive repeated measures ANOVA report
generate_repeated_report <- function(anova_result, data, subject_var, condition_var, response_var) {
  cat("=== REPEATED MEASURES ANOVA REPORT ===\n\n")
  
  # Descriptive statistics
  desc_stats <- data %>%
    group_by(!!sym(condition_var)) %>%
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
  f_stat <- anova_summary[[2]][[1]]$`F value`[1]
  p_value <- anova_summary[[2]][[1]]$`Pr(>F)`[1]
  df_conditions <- anova_summary[[2]][[1]]$Df[1]
  df_error <- anova_summary[[2]][[1]]$Df[2]
  
  cat("REPEATED MEASURES ANOVA RESULTS:\n")
  cat("F-statistic:", round(f_stat, 3), "\n")
  cat("Degrees of freedom:", df_conditions, ",", df_error, "\n")
  cat("p-value:", round(p_value, 4), "\n")
  
  # Effect size
  ss_conditions <- anova_summary[[2]][[1]]$`Sum Sq`[1]
  ss_error <- anova_summary[[2]][[1]]$`Sum Sq`[2]
  partial_eta2 <- ss_conditions / (ss_conditions + ss_error)
  
  cat("Effect size (partial η²):", round(partial_eta2, 3), "\n")
  cat("Interpretation:", interpret_effect_size(partial_eta2), "\n\n")
  
  # Post hoc results
  emm <- emmeans(anova_result, ~ condition)
  pairs_result <- pairs(emm, adjust = "tukey")
  significant_pairs <- pairs_result[pairs_result$p.value < 0.05, ]
  
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
    cat("There are significant differences between conditions\n")
  } else {
    cat("CONCLUSION:\n")
    cat("Fail to reject the null hypothesis (p >=", alpha, ")\n")
    cat("There is insufficient evidence of differences between conditions\n")
  }
}

# Generate report for repeated measures ANOVA
generate_repeated_report(repeated_model, repeated_data, "subject", "condition", "score")
```

## Comprehensive Exercises

### Exercise 1: Basic Repeated Measures ANOVA Analysis

**Objective**: Perform a complete repeated measures ANOVA analysis from data preparation to interpretation.

**Scenario**: A researcher is studying the effectiveness of a new learning intervention. 15 students are tested on their problem-solving skills at three time points: before the intervention (baseline), immediately after (post-test), and 3 months later (follow-up).

**Tasks**:
1. Generate realistic data for this scenario
2. Perform descriptive statistics and visualization
3. Conduct repeated measures ANOVA with manual calculations
4. Check assumptions thoroughly
5. Calculate and interpret effect sizes
6. Perform post hoc analysis if needed
7. Write a comprehensive report

**Requirements**:
- Use 15 subjects and 3 time points
- Include individual differences and learning effects
- Generate data with realistic effect sizes
- Provide detailed interpretation of results

**Expected Learning Outcomes**:
- Understanding of repeated measures ANOVA workflow
- Ability to interpret individual differences
- Knowledge of assumption checking procedures
- Skills in effect size calculation and interpretation

**Solution Framework**:
```r
# Exercise 1 Solution Framework
set.seed(123)
n_subjects <- 15
n_timepoints <- 3

# Generate data with learning effects
# ... (students will implement this)

# Perform comprehensive analysis
# ... (students will implement this)

# Check assumptions
# ... (students will implement this)

# Calculate effect sizes
# ... (students will implement this)

# Write report
# ... (students will implement this)
```

### Exercise 2: Mixed ANOVA with Interaction Effects

**Objective**: Analyze a mixed ANOVA design with both between-subjects and within-subjects factors.

**Scenario**: A clinical trial comparing two treatments (A and B) for anxiety reduction. Patients are randomly assigned to treatments and measured at baseline, week 4, and week 8.

**Tasks**:
1. Create a 2×3 mixed design dataset
2. Analyze main effects and interaction effects
3. Perform simple effects analysis
4. Create interaction plots
5. Conduct post hoc tests for significant effects
6. Report results comprehensively

**Requirements**:
- 20 patients per treatment group
- 3 measurement time points
- Include treatment × time interaction
- Realistic anxiety scores (0-100 scale)

**Expected Learning Outcomes**:
- Understanding of mixed ANOVA designs
- Ability to interpret interaction effects
- Skills in simple effects analysis
- Knowledge of appropriate post hoc procedures

### Exercise 3: Advanced Assumption Checking and Remedies

**Objective**: Practice comprehensive assumption checking and implement appropriate remedies for violations.

**Scenario**: You have repeated measures data that may violate sphericity and normality assumptions.

**Tasks**:
1. Generate data with known assumption violations
2. Perform comprehensive assumption checking
3. Apply appropriate corrections (Greenhouse-Geisser, Huynh-Feldt)
4. Compare results with and without corrections
5. Implement nonparametric alternatives
6. Compare parametric and nonparametric results

**Requirements**:
- Create data with sphericity violations
- Create data with normality violations
- Apply multiple correction methods
- Use Friedman test and Wilcoxon tests
- Provide recommendations for each scenario

**Expected Learning Outcomes**:
- Understanding of assumption violations
- Knowledge of correction methods
- Skills in nonparametric alternatives
- Ability to make informed decisions about analysis methods

### Exercise 4: Effect Size Analysis and Power

**Objective**: Conduct comprehensive effect size analysis and power calculations.

**Scenario**: You need to design a study to detect different effect sizes with adequate power.

**Tasks**:
1. Calculate effect sizes for existing data
2. Perform power analysis for different scenarios
3. Determine required sample sizes
4. Analyze effect size confidence intervals
5. Compare different effect size measures
6. Make recommendations for study design

**Requirements**:
- Use multiple effect size measures
- Calculate bootstrap confidence intervals
- Perform power analysis for different effect sizes
- Consider practical significance
- Provide sample size recommendations

**Expected Learning Outcomes**:
- Understanding of effect size interpretation
- Knowledge of power analysis procedures
- Skills in sample size determination
- Ability to assess practical significance

### Exercise 5: Real-World Application

**Objective**: Apply repeated measures ANOVA to a real-world research scenario.

**Scenario**: Choose one of the following:
- **Educational Research**: Student performance across multiple exams
- **Clinical Research**: Patient outcomes over treatment period
- **Sports Science**: Athlete performance across training phases
- **Psychology**: Cognitive task performance under different conditions

**Tasks**:
1. Design a realistic research study
2. Generate appropriate data
3. Perform complete analysis workflow
4. Create publication-ready visualizations
5. Write a research report
6. Address potential limitations

**Requirements**:
- Realistic research design
- Appropriate sample sizes
- Meaningful dependent variables
- Comprehensive analysis
- Professional reporting

**Expected Learning Outcomes**:
- Application of repeated measures ANOVA to real problems
- Skills in research design
- Ability to communicate results effectively
- Understanding of research limitations

### Exercise 6: Advanced Topics

**Objective**: Explore advanced topics in repeated measures analysis.

**Tasks**:
1. **Unbalanced Designs**: Handle missing data and unbalanced designs
2. **Robust Methods**: Implement robust repeated measures ANOVA
3. **Multivariate Approaches**: Use MANOVA for multiple dependent variables
4. **Trend Analysis**: Perform polynomial contrasts and trend analysis
5. **Bootstrap Methods**: Use bootstrap for confidence intervals and hypothesis testing

**Requirements**:
- Implement multiple advanced techniques
- Compare results across methods
- Understand when to use each approach
- Provide practical recommendations

**Expected Learning Outcomes**:
- Knowledge of advanced repeated measures techniques
- Understanding of robust methods
- Skills in multivariate analysis
- Ability to choose appropriate methods for different situations

## Best Practices and Guidelines

### Test Selection Guidelines

```r
# Function to help choose appropriate repeated measures test
choose_repeated_test <- function(data, subject_var, condition_var, response_var, alpha = 0.05) {
  cat("=== REPEATED MEASURES TEST SELECTION GUIDE ===\n\n")
  
  # Check sample size
  n_subjects <- length(unique(data[[subject_var]]))
  n_conditions <- length(unique(data[[condition_var]]))
  
  cat("SAMPLE SIZE ASSESSMENT:\n")
  cat("=" * 24, "\n")
  cat("Number of subjects:", n_subjects, "\n")
  cat("Number of conditions:", n_conditions, "\n")
  
  if (n_subjects < 10) {
    cat("⚠ Small sample size - consider nonparametric alternatives\n")
  } else if (n_subjects < 20) {
    cat("⚠ Moderate sample size - use with caution\n")
  } else {
    cat("✓ Adequate sample size for parametric tests\n")
  }
  
  # Check normality
  normality_results <- check_normality_repeated(data, subject_var, condition_var, response_var, alpha)
  
  # Check sphericity
  sphericity_results <- check_sphericity(data, subject_var, condition_var, response_var, alpha)
  
  # Decision matrix
  cat("\nDECISION MATRIX:\n")
  cat("=" * 16, "\n")
  
  normal_data <- normality_results$normal_tests >= 3
  sphericity_ok <- sphericity_results$sphericity_indicators >= 3
  
  if (normal_data && sphericity_ok) {
    cat("✓ RECOMMENDATION: Standard repeated measures ANOVA\n")
    cat("  - All assumptions met\n")
    cat("  - Most powerful test available\n")
  } else if (normal_data && !sphericity_ok) {
    cat("✓ RECOMMENDATION: Repeated measures ANOVA with corrections\n")
    cat("  - Use Greenhouse-Geisser or Huynh-Feldt corrections\n")
    cat("  - Data are normal but sphericity violated\n")
  } else if (!normal_data && sphericity_ok) {
    cat("✓ RECOMMENDATION: Robust repeated measures ANOVA\n")
    cat("  - Consider Yuen's t-test or trimmed means\n")
    cat("  - Sphericity met but normality violated\n")
  } else {
    cat("✓ RECOMMENDATION: Nonparametric alternatives\n")
    cat("  - Use Friedman test\n")
    cat("  - Consider permutation tests\n")
    cat("  - Both normality and sphericity violated\n")
  }
  
  # Additional considerations
  cat("\nADDITIONAL CONSIDERATIONS:\n")
  cat("=" * 25, "\n")
  
  if (n_conditions == 2) {
    cat("• For 2 conditions, consider paired t-test\n")
    cat("• Simpler and equivalent to repeated measures ANOVA\n")
  }
  
  if (n_conditions > 5) {
    cat("• Many conditions - consider trend analysis\n")
    cat("• Polynomial contrasts may be more informative\n")
  }
  
  if (n_subjects > 50) {
    cat("• Large sample - parametric tests are robust\n")
    cat("• Minor assumption violations may be acceptable\n")
  }
  
  return(list(
    n_subjects = n_subjects,
    n_conditions = n_conditions,
    normality_results = normality_results,
    sphericity_results = sphericity_results,
    recommendation = ifelse(normal_data && sphericity_ok, "Standard ANOVA",
                           ifelse(normal_data && !sphericity_ok, "ANOVA with corrections",
                                  ifelse(!normal_data && sphericity_ok, "Robust ANOVA", "Nonparametric")))
  ))
}

# Apply test selection guide
test_selection <- choose_repeated_test(repeated_data, "subject", "condition", "score")
```

### Data Preparation Best Practices

```r
# Function to prepare data for repeated measures analysis
prepare_repeated_data <- function(data, subject_var, condition_var, response_var) {
  cat("=== DATA PREPARATION CHECKLIST ===\n\n")
  
  # Check for missing data
  missing_data <- sum(is.na(data[[response_var]]))
  total_obs <- nrow(data)
  missing_pct <- (missing_data / total_obs) * 100
  
  cat("MISSING DATA ASSESSMENT:\n")
  cat("=" * 24, "\n")
  cat("Missing observations:", missing_data, "out of", total_obs, "(", round(missing_pct, 1), "%)\n")
  
  if (missing_pct > 5) {
    cat("⚠ High missing data - consider imputation or exclusion\n")
  } else if (missing_pct > 0) {
    cat("⚠ Some missing data - check for patterns\n")
  } else {
    cat("✓ No missing data\n")
  }
  
  # Check data structure
  cat("\nDATA STRUCTURE CHECK:\n")
  cat("=" * 20, "\n")
  
  # Check for balanced design
  subject_counts <- table(data[[subject_var]])
  condition_counts <- table(data[[condition_var]])
  
  if (length(unique(subject_counts)) == 1) {
    cat("✓ Balanced design (all subjects have same number of observations)\n")
  } else {
    cat("⚠ Unbalanced design - some subjects have different numbers of observations\n")
  }
  
  # Check for outliers
  outliers <- boxplot.stats(data[[response_var]])$out
  cat("Number of outliers:", length(outliers), "\n")
  
  if (length(outliers) > 0) {
    cat("⚠ Outliers detected - consider robust methods or data transformation\n")
  } else {
    cat("✓ No outliers detected\n")
  }
  
  # Data quality recommendations
  cat("\nDATA QUALITY RECOMMENDATIONS:\n")
  cat("=" * 29, "\n")
  
  if (missing_pct > 10) {
    cat("• Consider multiple imputation\n")
    cat("• Check for missing data patterns\n")
    cat("• Consider excluding subjects with too much missing data\n")
  }
  
  if (length(unique(subject_counts)) > 1) {
    cat("• Consider mixed-effects models for unbalanced data\n")
    cat("• Check if missing data is missing at random\n")
  }
  
  if (length(outliers) > 0) {
    cat("• Consider robust statistical methods\n")
    cat("• Check if outliers are data entry errors\n")
    cat("• Consider data transformation\n")
  }
  
  # Create cleaned dataset
  cleaned_data <- data[!is.na(data[[response_var]]), ]
  
  return(list(
    original_data = data,
    cleaned_data = cleaned_data,
    missing_data = missing_data,
    missing_pct = missing_pct,
    outliers = outliers,
    subject_counts = subject_counts,
    condition_counts = condition_counts
  ))
}

# Apply data preparation
data_prep <- prepare_repeated_data(repeated_data, "subject", "condition", "score")
```

### Reporting Guidelines

```r
# Function to generate comprehensive repeated measures ANOVA report
generate_comprehensive_report <- function(anova_result, data, subject_var, condition_var, response_var, 
                                        effect_sizes, normality_results, sphericity_results) {
  cat("=== COMPREHENSIVE REPEATED MEASURES ANOVA REPORT ===\n\n")
  
  # Study description
  cat("STUDY DESCRIPTION:\n")
  cat("=" * 19, "\n")
  cat("Design: Repeated measures ANOVA\n")
  cat("Subjects:", length(unique(data[[subject_var]])), "\n")
  cat("Conditions:", length(unique(data[[condition_var]])), "\n")
  cat("Total observations:", nrow(data), "\n\n")
  
  # Descriptive statistics
  desc_stats <- data %>%
    group_by(!!sym(condition_var)) %>%
    summarise(
      n = n(),
      mean = mean(!!sym(response_var), na.rm = TRUE),
      sd = sd(!!sym(response_var), na.rm = TRUE),
      se = sd / sqrt(n),
      ci_lower = mean - qt(0.975, n-1) * se,
      ci_upper = mean + qt(0.975, n-1) * se
    )
  
  cat("DESCRIPTIVE STATISTICS:\n")
  cat("=" * 23, "\n")
  print(desc_stats)
  cat("\n")
  
  # Assumption checking summary
  cat("ASSUMPTION CHECKING:\n")
  cat("=" * 19, "\n")
  
  # Normality
  normal_tests <- normality_results$normal_tests
  total_tests <- normality_results$total_tests
  cat("Normality:", normal_tests, "out of", total_tests, "tests passed\n")
  
  # Sphericity
  sphericity_indicators <- sphericity_results$sphericity_indicators
  total_indicators <- sphericity_results$total_indicators
  cat("Sphericity:", sphericity_indicators, "out of", total_indicators, "indicators met\n")
  
  # ANOVA results
  anova_summary <- summary(anova_result)
  f_stat <- anova_summary[[2]][[1]]$`F value`[1]
  p_value <- anova_summary[[2]][[1]]$`Pr(>F)`[1]
  df_conditions <- anova_summary[[2]][[1]]$Df[1]
  df_error <- anova_summary[[2]][[1]]$Df[2]
  
  cat("\nREPEATED MEASURES ANOVA RESULTS:\n")
  cat("=" * 32, "\n")
  cat("F-statistic:", round(f_stat, 3), "\n")
  cat("Degrees of freedom:", df_conditions, ",", df_error, "\n")
  cat("p-value:", round(p_value, 4), "\n")
  
  # Effect sizes
  cat("\nEFFECT SIZES:\n")
  cat("=" * 12, "\n")
  cat("Partial η²:", round(effect_sizes$partial_eta2, 4), "\n")
  cat("η²:", round(effect_sizes$eta2, 4), "\n")
  cat("ω²:", round(effect_sizes$omega2, 4), "\n")
  cat("Cohen's f:", round(effect_sizes$cohens_f, 4), "\n")
  
  # Interpretation
  cat("\nINTERPRETATION:\n")
  cat("=" * 14, "\n")
  
  alpha <- 0.05
  if (p_value < alpha) {
    cat("✓ Significant main effect of condition (p <", alpha, ")\n")
    cat("✓ There are significant differences between conditions\n")
  } else {
    cat("✗ No significant main effect of condition (p >=", alpha, ")\n")
    cat("✗ There is insufficient evidence of differences between conditions\n")
  }
  
  # Effect size interpretation
  if (effect_sizes$partial_eta2 > 0.14) {
    cat("✓ Large effect size\n")
  } else if (effect_sizes$partial_eta2 > 0.06) {
    cat("✓ Medium effect size\n")
  } else if (effect_sizes$partial_eta2 > 0.01) {
    cat("✓ Small effect size\n")
  } else {
    cat("✓ Negligible effect size\n")
  }
  
  # Recommendations
  cat("\nRECOMMENDATIONS:\n")
  cat("=" * 15, "\n")
  
  if (normal_tests >= 3 && sphericity_indicators >= 3) {
    cat("✓ Results are reliable and valid\n")
    cat("✓ Standard repeated measures ANOVA is appropriate\n")
  } else if (normal_tests >= 2 && sphericity_indicators >= 2) {
    cat("⚠ Results should be interpreted with caution\n")
    cat("⚠ Consider robust alternatives for future studies\n")
  } else {
    cat("✗ Results may not be reliable\n")
    cat("✗ Consider nonparametric alternatives\n")
  }
  
  # Post hoc recommendations
  if (p_value < alpha) {
    cat("✓ Perform post hoc tests to identify specific differences\n")
    cat("✓ Consider trend analysis for ordered conditions\n")
  }
  
  return(list(
    descriptive_stats = desc_stats,
    anova_results = list(f_stat = f_stat, p_value = p_value, df = c(df_conditions, df_error)),
    effect_sizes = effect_sizes,
    interpretation = list(significant = p_value < alpha, effect_size = effect_sizes$partial_eta2)
  ))
}

# Generate comprehensive report
comprehensive_report <- generate_comprehensive_report(
  repeated_model, repeated_data, "subject", "condition", "score",
  effect_sizes, normality_results, sphericity_results
)
```

## Next Steps

In the next chapter, we'll learn about correlation analysis for examining relationships between variables.

---

**Key Takeaways:**
- Repeated measures ANOVA is more powerful than between-subjects designs
- Always check sphericity and normality assumptions
- Effect sizes provide important information about practical significance
- Post hoc tests are necessary when the main effect is significant
- Nonparametric alternatives exist for non-normal data
- Mixed ANOVA combines between-subjects and within-subjects factors
- Proper reporting includes descriptive statistics, test results, and effect sizes 