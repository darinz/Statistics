# Repeated Measures ANOVA

## Overview

Repeated Measures ANOVA is used to analyze data where the same subjects are measured multiple times under different conditions. This design is more powerful than between-subjects designs because it controls for individual differences and reduces error variance.

## Basic Repeated Measures ANOVA

### Manual Calculation

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
  
  # SS Between Subjects
  ss_between_subjects <- n_conditions * sum((subject_means - overall_mean)^2)
  
  # SS Within Subjects
  ss_within_subjects <- ss_total - ss_between_subjects
  
  # SS Conditions (Treatment)
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
    subject_means = subject_means
  ))
}

# Apply manual calculation
anova_result <- manual_repeated_anova(repeated_data, "subject", "condition", "score")

cat("Manual Repeated Measures ANOVA Results:\n")
cat("F-statistic:", round(anova_result$f_statistic, 3), "\n")
cat("p-value:", round(anova_result$p_value, 4), "\n")
cat("Partial η²:", round(anova_result$partial_eta2, 3), "\n")
cat("Condition means:", round(anova_result$condition_means, 2), "\n")
```

### Using R's Built-in Repeated Measures ANOVA

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

### Condition and Subject Statistics

```r
# Calculate descriptive statistics
library(dplyr)

# Condition means
condition_stats <- repeated_data %>%
  group_by(condition) %>%
  summarise(
    n = n(),
    mean = mean(score, na.rm = TRUE),
    sd = sd(score, na.rm = TRUE),
    se = sd / sqrt(n)
  )

print(condition_stats)

# Subject means
subject_stats <- repeated_data %>%
  group_by(subject) %>%
  summarise(
    n = n(),
    mean = mean(score, na.rm = TRUE),
    sd = sd(score, na.rm = TRUE)
  )

cat("Subject Statistics Summary:\n")
cat("Mean subject score:", round(mean(subject_stats$mean), 2), "\n")
cat("SD of subject scores:", round(sd(subject_stats$mean), 2), "\n")
cat("Range of subject scores:", round(range(subject_stats$mean), 2), "\n")
```

### Visualization

```r
library(ggplot2)
library(gridExtra)

# Individual subject profiles
p1 <- ggplot(repeated_data, aes(x = condition, y = score, group = subject)) +
  geom_line(alpha = 0.3) +
  geom_point(alpha = 0.5) +
  stat_summary(aes(group = 1), fun = mean, geom = "line", color = "red", size = 2) +
  stat_summary(aes(group = 1), fun = mean, geom = "point", color = "red", size = 3) +
  labs(title = "Individual Subject Profiles", x = "Condition", y = "Score") +
  theme_minimal()

# Box plot
p2 <- ggplot(repeated_data, aes(x = condition, y = score, fill = condition)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Score Distribution by Condition", x = "Condition", y = "Score") +
  theme_minimal() +
  theme(legend.position = "none")

# Violin plot
p3 <- ggplot(repeated_data, aes(x = condition, y = score, fill = condition)) +
  geom_violin(alpha = 0.7) +
  geom_boxplot(width = 0.2, alpha = 0.8) +
  labs(title = "Score Distribution by Condition", x = "Condition", y = "Score") +
  theme_minimal() +
  theme(legend.position = "none")

# Combine plots
grid.arrange(p1, p2, p3, ncol = 2)
```

## Effect Size Analysis

### Partial Eta-Squared and Other Effect Sizes

```r
# Calculate effect sizes for repeated measures ANOVA
calculate_repeated_effect_sizes <- function(anova_result) {
  # Extract Sum of Squares
  ss_conditions <- anova_result$ss_conditions
  ss_error <- anova_result$ss_error
  ss_total <- anova_result$ss_total
  
  # Partial eta-squared
  partial_eta2 <- ss_conditions / (ss_conditions + ss_error)
  
  # Eta-squared (total)
  eta2 <- ss_conditions / ss_total
  
  # Omega-squared (unbiased)
  df_conditions <- anova_result$df_conditions
  df_error <- anova_result$df_error
  ms_error <- anova_result$ms_error
  
  omega2 <- (ss_conditions - df_conditions * ms_error) / (ss_total + ms_error)
  
  # Cohen's f (for power analysis)
  cohens_f <- sqrt(partial_eta2 / (1 - partial_eta2))
  
  return(list(
    partial_eta2 = partial_eta2,
    eta2 = eta2,
    omega2 = omega2,
    cohens_f = cohens_f
  ))
}

# Apply to our results
effect_sizes <- calculate_repeated_effect_sizes(anova_result)

cat("Effect Size Analysis:\n")
cat("Partial η²:", round(effect_sizes$partial_eta2, 3), "\n")
cat("η²:", round(effect_sizes$eta2, 3), "\n")
cat("ω²:", round(effect_sizes$omega2, 3), "\n")
cat("Cohen's f:", round(effect_sizes$cohens_f, 3), "\n")

# Interpret effect size
interpret_effect_size <- function(eta_sq) {
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

cat("Effect size interpretation:", interpret_effect_size(effect_sizes$partial_eta2), "\n")
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

### Sphericity Test

```r
# Function to check sphericity assumption
check_sphericity <- function(data, subject_var, condition_var, response_var) {
  cat("=== SPHERICITY TEST ===\n")
  
  # Create wide format for Mauchly's test
  wide_data <- data %>%
    pivot_wider(
      names_from = condition_var,
      values_from = response_var,
      names_prefix = "condition_"
    )
  
  # Remove subject column for correlation matrix
  cor_matrix <- wide_data %>%
    select(-subject_var) %>%
    cor(use = "complete.obs")
  
  cat("Correlation matrix of conditions:\n")
  print(round(cor_matrix, 3))
  
  # Mauchly's test using car package
  library(car)
  model <- aov(as.formula(paste(response_var, "~", condition_var, "+ Error(", subject_var, "/", condition_var, ")")), data = data)
  
  # Note: Mauchly's test is not directly available in base R
  # We can check sphericity informally by examining the correlation matrix
  
  # Calculate Greenhouse-Geisser and Huynh-Feldt corrections
  n_conditions <- ncol(cor_matrix)
  n_subjects <- nrow(wide_data)
  
  # Greenhouse-Geisser epsilon
  gg_epsilon <- 1 / (n_conditions - 1)
  
  # Huynh-Feldt epsilon (approximation)
  hf_epsilon <- min(1, gg_epsilon * (n_subjects - 1) / (n_subjects - 1 - (n_conditions - 1)))
  
  cat("\nSphericity Corrections:\n")
  cat("Greenhouse-Geisser ε:", round(gg_epsilon, 3), "\n")
  cat("Huynh-Feldt ε:", round(hf_epsilon, 3), "\n")
  
  # Recommendations
  cat("\nRECOMMENDATIONS:\n")
  if (gg_epsilon < 0.75) {
    cat("- Sphericity assumption may be violated\n")
    cat("- Use Greenhouse-Geisser or Huynh-Feldt corrections\n")
  } else {
    cat("- Sphericity assumption appears to be met\n")
    cat("- Standard repeated measures ANOVA is appropriate\n")
  }
  
  return(list(
    correlation_matrix = cor_matrix,
    gg_epsilon = gg_epsilon,
    hf_epsilon = hf_epsilon
  ))
}

# Check sphericity
sphericity_results <- check_sphericity(repeated_data, "subject", "condition", "score")
```

### Normality Test

```r
# Function to check normality for repeated measures
check_normality_repeated <- function(data, subject_var, condition_var, response_var) {
  cat("=== NORMALITY TESTS FOR REPEATED MEASURES ===\n")
  
  # Check normality of residuals
  model <- aov(as.formula(paste(response_var, "~", condition_var, "+ Error(", subject_var, "/", condition_var, ")")), data = data)
  
  # Extract residuals (this is simplified - in practice, you'd need to extract from the model)
  residuals <- data[[response_var]] - ave(data[[response_var]], data[[condition_var]])
  
  # Shapiro-Wilk test on residuals
  shapiro_result <- shapiro.test(residuals)
  cat("Residuals Shapiro-Wilk p-value:", round(shapiro_result$p.value, 4), "\n")
  
  # Q-Q plot of residuals
  qq_plot <- ggplot(data.frame(residuals = residuals), aes(sample = residuals)) +
    stat_qq() +
    stat_qq_line() +
    labs(title = "Q-Q Plot of Residuals") +
    theme_minimal()
  
  # Histogram of residuals
  hist_plot <- ggplot(data.frame(residuals = residuals), aes(x = residuals)) +
    geom_histogram(bins = 15, fill = "steelblue", alpha = 0.7) +
    labs(title = "Histogram of Residuals") +
    theme_minimal()
  
  # Recommendations
  cat("\nRECOMMENDATIONS:\n")
  if (shapiro_result$p.value >= 0.05) {
    cat("- Residuals appear normally distributed\n")
    cat("- Repeated measures ANOVA assumptions are met\n")
  } else {
    cat("- Residuals may not be normally distributed\n")
    cat("- Consider nonparametric alternatives\n")
  }
  
  return(list(
    shapiro_test = shapiro_result,
    qq_plot = qq_plot,
    hist_plot = hist_plot
  ))
}

# Check normality
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

## Exercises

### Exercise 1: Basic Repeated Measures ANOVA
Perform repeated measures ANOVA to analyze the effects of time on a dependent variable in a within-subjects design.

### Exercise 2: Mixed ANOVA Analysis
Create a mixed ANOVA dataset and analyze both between-subjects and within-subjects effects.

### Exercise 3: Assumption Checking
Conduct comprehensive assumption checking for repeated measures ANOVA and recommend appropriate alternatives.

### Exercise 4: Post Hoc Analysis
Perform appropriate post hoc tests for significant repeated measures effects.

### Exercise 5: Power Analysis
Conduct power analysis to determine required sample sizes for detecting different effect sizes in repeated measures designs.

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