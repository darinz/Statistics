# Two-Way ANOVA

## Overview

Two-way Analysis of Variance (ANOVA) is used to analyze the effects of two independent variables (factors) on a dependent variable, including their interaction effects. This design is more powerful than one-way ANOVA as it allows researchers to examine both main effects and interaction effects simultaneously.

## Basic Two-Way ANOVA

### Manual Calculation

```r
# Load sample data
data(mtcars)

# Create factors for analysis
mtcars$cyl_factor <- factor(mtcars$cyl, levels = c(4, 6, 8), 
                           labels = c("4-cylinder", "6-cylinder", "8-cylinder"))
mtcars$am_factor <- factor(mtcars$am, levels = c(0, 1), 
                          labels = c("Automatic", "Manual"))

# Manual two-way ANOVA calculation
manual_two_way_anova <- function(data, factor1, factor2, response) {
  # Calculate overall mean
  overall_mean <- mean(data[[response]], na.rm = TRUE)
  
  # Calculate factor means
  factor1_means <- tapply(data[[response]], data[[factor1]], mean, na.rm = TRUE)
  factor2_means <- tapply(data[[response]], data[[factor2]], mean, na.rm = TRUE)
  
  # Calculate cell means
  cell_means <- tapply(data[[response]], list(data[[factor1]], data[[factor2]]), mean, na.rm = TRUE)
  
  # Calculate sample sizes
  n_total <- nrow(data)
  n_factor1_levels <- length(unique(data[[factor1]]))
  n_factor2_levels <- length(unique(data[[factor2]]))
  
  # Calculate Sum of Squares
  # SS Total
  ss_total <- sum((data[[response]] - overall_mean)^2, na.rm = TRUE)
  
  # SS Factor 1 (Main effect)
  ss_factor1 <- sum(table(data[[factor1]]) * (factor1_means - overall_mean)^2)
  
  # SS Factor 2 (Main effect)
  ss_factor2 <- sum(table(data[[factor2]]) * (factor2_means - overall_mean)^2)
  
  # SS Interaction
  ss_interaction <- 0
  for (i in 1:n_factor1_levels) {
    for (j in 1:n_factor2_levels) {
      cell_mean <- cell_means[i, j]
      factor1_mean <- factor1_means[i]
      factor2_mean <- factor2_means[j]
      n_cell <- sum(data[[factor1]] == levels(data[[factor1]])[i] & 
                  data[[factor2]] == levels(data[[factor2]])[j])
      
      interaction_effect <- cell_mean - factor1_mean - factor2_mean + overall_mean
      ss_interaction <- ss_interaction + n_cell * interaction_effect^2
    }
  }
  
  # SS Error (Within)
  ss_error <- ss_total - ss_factor1 - ss_factor2 - ss_interaction
  
  # Degrees of freedom
  df_factor1 <- n_factor1_levels - 1
  df_factor2 <- n_factor2_levels - 1
  df_interaction <- (n_factor1_levels - 1) * (n_factor2_levels - 1)
  df_error <- n_total - n_factor1_levels * n_factor2_levels
  df_total <- n_total - 1
  
  # Mean Squares
  ms_factor1 <- ss_factor1 / df_factor1
  ms_factor2 <- ss_factor2 / df_factor2
  ms_interaction <- ss_interaction / df_interaction
  ms_error <- ss_error / df_error
  
  # F-statistics
  f_factor1 <- ms_factor1 / ms_error
  f_factor2 <- ms_factor2 / ms_error
  f_interaction <- ms_interaction / ms_error
  
  # p-values
  p_factor1 <- 1 - pf(f_factor1, df_factor1, df_error)
  p_factor2 <- 1 - pf(f_factor2, df_factor2, df_error)
  p_interaction <- 1 - pf(f_interaction, df_interaction, df_error)
  
  # Effect sizes (partial eta-squared)
  partial_eta2_factor1 <- ss_factor1 / (ss_factor1 + ss_error)
  partial_eta2_factor2 <- ss_factor2 / (ss_factor2 + ss_error)
  partial_eta2_interaction <- ss_interaction / (ss_interaction + ss_error)
  
  return(list(
    ss_factor1 = ss_factor1,
    ss_factor2 = ss_factor2,
    ss_interaction = ss_interaction,
    ss_error = ss_error,
    ss_total = ss_total,
    df_factor1 = df_factor1,
    df_factor2 = df_factor2,
    df_interaction = df_interaction,
    df_error = df_error,
    df_total = df_total,
    ms_factor1 = ms_factor1,
    ms_factor2 = ms_factor2,
    ms_interaction = ms_interaction,
    ms_error = ms_error,
    f_factor1 = f_factor1,
    f_factor2 = f_factor2,
    f_interaction = f_interaction,
    p_factor1 = p_factor1,
    p_factor2 = p_factor2,
    p_interaction = p_interaction,
    partial_eta2_factor1 = partial_eta2_factor1,
    partial_eta2_factor2 = partial_eta2_factor2,
    partial_eta2_interaction = partial_eta2_interaction,
    cell_means = cell_means,
    factor1_means = factor1_means,
    factor2_means = factor2_means
  ))
}

# Apply manual calculation
anova_result <- manual_two_way_anova(mtcars, "cyl_factor", "am_factor", "mpg")

cat("Manual Two-Way ANOVA Results:\n")
cat("Factor 1 (Cylinders) F-statistic:", round(anova_result$f_factor1, 3), "\n")
cat("Factor 1 p-value:", round(anova_result$p_factor1, 4), "\n")
cat("Factor 2 (Transmission) F-statistic:", round(anova_result$f_factor2, 3), "\n")
cat("Factor 2 p-value:", round(anova_result$p_factor2, 4), "\n")
cat("Interaction F-statistic:", round(anova_result$f_interaction, 3), "\n")
cat("Interaction p-value:", round(anova_result$p_interaction, 4), "\n")
```

### Using R's Built-in Two-Way ANOVA

```r
# Perform two-way ANOVA using R's aov function
two_way_model <- aov(mpg ~ cyl_factor * am_factor, data = mtcars)
print(two_way_model)

# Get ANOVA summary
anova_summary <- summary(two_way_model)
print(anova_summary)

# Extract key statistics
f_cylinders <- anova_summary[[1]]$`F value`[1]
f_transmission <- anova_summary[[1]]$`F value`[2]
f_interaction <- anova_summary[[1]]$`F value`[3]
p_cylinders <- anova_summary[[1]]$`Pr(>F)`[1]
p_transmission <- anova_summary[[1]]$`Pr(>F)`[2]
p_interaction <- anova_summary[[1]]$`Pr(>F)`[3]

cat("Two-Way ANOVA Results:\n")
cat("Cylinders F-statistic:", round(f_cylinders, 3), "\n")
cat("Cylinders p-value:", round(p_cylinders, 4), "\n")
cat("Transmission F-statistic:", round(f_transmission, 3), "\n")
cat("Transmission p-value:", round(p_transmission, 4), "\n")
cat("Interaction F-statistic:", round(f_interaction, 3), "\n")
cat("Interaction p-value:", round(p_interaction, 4), "\n")
```

## Descriptive Statistics

### Cell Means and Marginal Means

```r
# Calculate descriptive statistics
library(dplyr)

# Cell means
cell_means <- mtcars %>%
  group_by(cyl_factor, am_factor) %>%
  summarise(
    n = n(),
    mean = mean(mpg, na.rm = TRUE),
    sd = sd(mpg, na.rm = TRUE),
    se = sd / sqrt(n)
  )

print(cell_means)

# Marginal means for Factor 1 (Cylinders)
marginal_cylinders <- mtcars %>%
  group_by(cyl_factor) %>%
  summarise(
    n = n(),
    mean = mean(mpg, na.rm = TRUE),
    sd = sd(mpg, na.rm = TRUE),
    se = sd / sqrt(n)
  )

cat("Marginal Means - Cylinders:\n")
print(marginal_cylinders)

# Marginal means for Factor 2 (Transmission)
marginal_transmission <- mtcars %>%
  group_by(am_factor) %>%
  summarise(
    n = n(),
    mean = mean(mpg, na.rm = TRUE),
    sd = sd(mpg, na.rm = TRUE),
    se = sd / sqrt(n)
  )

cat("Marginal Means - Transmission:\n")
print(marginal_transmission)
```

### Visualization

```r
library(ggplot2)
library(gridExtra)

# Interaction plot
p1 <- ggplot(mtcars, aes(x = cyl_factor, y = mpg, color = am_factor, group = am_factor)) +
  stat_summary(fun = mean, geom = "line", size = 1) +
  stat_summary(fun = mean, geom = "point", size = 3) +
  labs(title = "Interaction Plot: MPG by Cylinders and Transmission",
       x = "Number of Cylinders", y = "MPG", color = "Transmission") +
  theme_minimal()

# Box plot
p2 <- ggplot(mtcars, aes(x = cyl_factor, y = mpg, fill = am_factor)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "MPG by Cylinders and Transmission",
       x = "Number of Cylinders", y = "MPG", fill = "Transmission") +
  theme_minimal()

# Heatmap of cell means
cell_means_wide <- cell_means %>%
  select(cyl_factor, am_factor, mean) %>%
  pivot_wider(names_from = am_factor, values_from = mean)

p3 <- ggplot(cell_means, aes(x = cyl_factor, y = am_factor, fill = mean)) +
  geom_tile() +
  geom_text(aes(label = round(mean, 1)), color = "white", size = 4) +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Cell Means Heatmap", x = "Cylinders", y = "Transmission") +
  theme_minimal()

# Combine plots
grid.arrange(p1, p2, p3, ncol = 2)
```

## Effect Size Analysis

### Partial Eta-Squared

```r
# Calculate effect sizes
calculate_two_way_effect_sizes <- function(anova_result) {
  # Extract Sum of Squares
  ss_factor1 <- anova_result$ss_factor1
  ss_factor2 <- anova_result$ss_factor2
  ss_interaction <- anova_result$ss_interaction
  ss_error <- anova_result$ss_error
  ss_total <- anova_result$ss_total
  
  # Partial eta-squared
  partial_eta2_factor1 <- ss_factor1 / (ss_factor1 + ss_error)
  partial_eta2_factor2 <- ss_factor2 / (ss_factor2 + ss_error)
  partial_eta2_interaction <- ss_interaction / (ss_interaction + ss_error)
  
  # Eta-squared (total)
  eta2_factor1 <- ss_factor1 / ss_total
  eta2_factor2 <- ss_factor2 / ss_total
  eta2_interaction <- ss_interaction / ss_total
  
  # Omega-squared (unbiased)
  df_factor1 <- anova_result$df_factor1
  df_factor2 <- anova_result$df_factor2
  df_interaction <- anova_result$df_interaction
  ms_error <- anova_result$ms_error
  
  omega2_factor1 <- (ss_factor1 - df_factor1 * ms_error) / (ss_total + ms_error)
  omega2_factor2 <- (ss_factor2 - df_factor2 * ms_error) / (ss_total + ms_error)
  omega2_interaction <- (ss_interaction - df_interaction * ms_error) / (ss_total + ms_error)
  
  return(list(
    partial_eta2_factor1 = partial_eta2_factor1,
    partial_eta2_factor2 = partial_eta2_factor2,
    partial_eta2_interaction = partial_eta2_interaction,
    eta2_factor1 = eta2_factor1,
    eta2_factor2 = eta2_factor2,
    eta2_interaction = eta2_interaction,
    omega2_factor1 = omega2_factor1,
    omega2_factor2 = omega2_factor2,
    omega2_interaction = omega2_interaction
  ))
}

# Apply to our results
effect_sizes <- calculate_two_way_effect_sizes(anova_result)

cat("Effect Size Analysis:\n")
cat("Factor 1 (Cylinders) Partial η²:", round(effect_sizes$partial_eta2_factor1, 3), "\n")
cat("Factor 2 (Transmission) Partial η²:", round(effect_sizes$partial_eta2_factor2, 3), "\n")
cat("Interaction Partial η²:", round(effect_sizes$partial_eta2_interaction, 3), "\n")

# Interpret effect sizes
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

cat("Factor 1 interpretation:", interpret_effect_size(effect_sizes$partial_eta2_factor1), "\n")
cat("Factor 2 interpretation:", interpret_effect_size(effect_sizes$partial_eta2_factor2), "\n")
cat("Interaction interpretation:", interpret_effect_size(effect_sizes$partial_eta2_interaction), "\n")
```

## Simple Effects Analysis

### Simple Effects for Significant Interactions

```r
# Function to perform simple effects analysis
simple_effects_analysis <- function(data, factor1, factor2, response) {
  cat("=== SIMPLE EFFECTS ANALYSIS ===\n")
  
  # Simple effects of Factor 1 at each level of Factor 2
  cat("Simple effects of", factor1, "at each level of", factor2, ":\n")
  
  factor2_levels <- unique(data[[factor2]])
  
  for (level in factor2_levels) {
    subset_data <- data[data[[factor2]] == level, ]
    
    if (length(unique(subset_data[[factor1]])) > 1) {
      # Perform one-way ANOVA
      simple_model <- aov(as.formula(paste(response, "~", factor1)), data = subset_data)
      simple_summary <- summary(simple_model)
      
      f_stat <- simple_summary[[1]]$`F value`[1]
      p_value <- simple_summary[[1]]$`Pr(>F)`[1]
      
      cat("At", factor2, "=", level, ":\n")
      cat("  F-statistic:", round(f_stat, 3), "\n")
      cat("  p-value:", round(p_value, 4), "\n")
      
      if (p_value < 0.05) {
        cat("  Significant effect\n")
      } else {
        cat("  Non-significant effect\n")
      }
      cat("\n")
    }
  }
  
  # Simple effects of Factor 2 at each level of Factor 1
  cat("Simple effects of", factor2, "at each level of", factor1, ":\n")
  
  factor1_levels <- unique(data[[factor1]])
  
  for (level in factor1_levels) {
    subset_data <- data[data[[factor1]] == level, ]
    
    if (length(unique(subset_data[[factor2]])) > 1) {
      # Perform one-way ANOVA
      simple_model <- aov(as.formula(paste(response, "~", factor2)), data = subset_data)
      simple_summary <- summary(simple_model)
      
      f_stat <- simple_summary[[1]]$`F value`[1]
      p_value <- simple_summary[[1]]$`Pr(>F)`[1]
      
      cat("At", factor1, "=", level, ":\n")
      cat("  F-statistic:", round(f_stat, 3), "\n")
      cat("  p-value:", round(p_value, 4), "\n")
      
      if (p_value < 0.05) {
        cat("  Significant effect\n")
      } else {
        cat("  Non-significant effect\n")
      }
      cat("\n")
    }
  }
}

# Apply simple effects analysis
simple_effects_analysis(mtcars, "cyl_factor", "am_factor", "mpg")
```

## Post Hoc Tests

### Post Hoc for Main Effects

```r
# Post hoc tests for significant main effects
library(multcomp)

# Post hoc for Factor 1 (Cylinders) if significant
if (p_cylinders < 0.05) {
  cat("Post hoc tests for Cylinders (Factor 1):\n")
  
  # Tukey's HSD for cylinders
  cyl_posthoc <- glht(two_way_model, linfct = mcp(cyl_factor = "Tukey"))
  cyl_summary <- summary(cyl_posthoc)
  print(cyl_summary)
  
  # Extract significant pairs
  cyl_pvalues <- cyl_summary$test$pvalues
  cyl_comparisons <- names(cyl_pvalues)
  
  cat("Significant pairwise differences (p < 0.05):\n")
  for (i in 1:length(cyl_pvalues)) {
    if (cyl_pvalues[i] < 0.05) {
      cat(cyl_comparisons[i], ": p =", round(cyl_pvalues[i], 4), "\n")
    }
  }
  cat("\n")
}

# Post hoc for Factor 2 (Transmission) if significant
if (p_transmission < 0.05) {
  cat("Post hoc tests for Transmission (Factor 2):\n")
  
  # Since transmission has only 2 levels, no post hoc needed
  cat("Transmission has only 2 levels - no post hoc tests needed\n")
  cat("Automatic vs Manual transmission difference is already tested\n\n")
}
```

### Post Hoc for Interaction Effects

```r
# Post hoc tests for interaction effects
if (p_interaction < 0.05) {
  cat("Post hoc tests for Interaction Effects:\n")
  
  # Pairwise comparisons for all cells
  cell_comparisons <- glht(two_way_model, linfct = mcp(cyl_factor:am_factor = "Tukey"))
  cell_summary <- summary(cell_comparisons)
  print(cell_summary)
  
  # Extract significant cell comparisons
  cell_pvalues <- cell_summary$test$pvalues
  cell_comparisons <- names(cell_pvalues)
  
  cat("Significant cell differences (p < 0.05):\n")
  for (i in 1:length(cell_pvalues)) {
    if (cell_pvalues[i] < 0.05) {
      cat(cell_comparisons[i], ": p =", round(cell_pvalues[i], 4), "\n")
    }
  }
  cat("\n")
} else {
  cat("No post hoc tests needed for interaction (not significant)\n\n")
}
```

## Assumption Checking

### Normality Test

```r
# Function to check normality for two-way ANOVA
check_normality_two_way <- function(data, factor1, factor2, response) {
  cat("=== NORMALITY TESTS FOR TWO-WAY ANOVA ===\n")
  
  # Check normality of residuals
  model <- aov(as.formula(paste(response, "~", factor1, "*", factor2)), data = data)
  residuals <- residuals(model)
  
  # Shapiro-Wilk test on residuals
  shapiro_result <- shapiro.test(residuals)
  cat("Overall residuals Shapiro-Wilk p-value:", round(shapiro_result$p.value, 4), "\n")
  
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
    cat("- Two-way ANOVA assumptions are met\n")
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
normality_results <- check_normality_two_way(mtcars, "cyl_factor", "am_factor", "mpg")
```

### Homogeneity of Variance

```r
# Function to test homogeneity of variance for two-way ANOVA
check_homogeneity_two_way <- function(data, factor1, factor2, response) {
  cat("=== HOMOGENEITY OF VARIANCE TESTS ===\n")
  
  # Levene's test
  library(car)
  levene_result <- leveneTest(as.formula(paste(response, "~", factor1, "*", factor2)), data = data)
  cat("Levene's test p-value:", round(levene_result$`Pr(>F)`[1], 4), "\n")
  
  # Bartlett's test
  bartlett_result <- bartlett.test(as.formula(paste(response, "~", factor1, "*", factor2)), data = data)
  cat("Bartlett's test p-value:", round(bartlett_result$p.value, 4), "\n")
  
  # Fligner-Killeen test
  fligner_result <- fligner.test(as.formula(paste(response, "~", factor1, "*", factor2)), data = data)
  cat("Fligner-Killeen test p-value:", round(fligner_result$p.value, 4), "\n")
  
  # Recommendations
  cat("\nRECOMMENDATIONS:\n")
  if (levene_result$`Pr(>F)`[1] >= 0.05) {
    cat("- Variances appear equal across groups\n")
    cat("- Standard two-way ANOVA is appropriate\n")
  } else {
    cat("- Variances are significantly different\n")
    cat("- Consider robust alternatives or data transformation\n")
  }
  
  return(list(
    levene = levene_result,
    bartlett = bartlett_result,
    fligner = fligner_result
  ))
}

# Check homogeneity
homogeneity_results <- check_homogeneity_two_way(mtcars, "cyl_factor", "am_factor", "mpg")
```

## Practical Examples

### Example 1: Educational Research

```r
# Simulate educational intervention data
set.seed(123)
n_per_cell <- 15

# Generate data for 2x3 factorial design
# Factor 1: Teaching Method (A, B)
# Factor 2: Class Size (Small, Medium, Large)

teaching_method <- rep(c("Method A", "Method B"), each = n_per_cell * 3)
class_size <- rep(rep(c("Small", "Medium", "Large"), each = n_per_cell), 2)

# Generate scores with interaction effects
scores <- numeric(length(teaching_method))
for (i in 1:length(teaching_method)) {
  if (teaching_method[i] == "Method A") {
    if (class_size[i] == "Small") scores[i] <- rnorm(1, 85, 8)
    else if (class_size[i] == "Medium") scores[i] <- rnorm(1, 80, 10)
    else scores[i] <- rnorm(1, 75, 12)
  } else {
    if (class_size[i] == "Small") scores[i] <- rnorm(1, 82, 9)
    else if (class_size[i] == "Medium") scores[i] <- rnorm(1, 85, 8)
    else scores[i] <- rnorm(1, 88, 7)
  }
}

# Create data frame
education_data <- data.frame(
  score = scores,
  method = factor(teaching_method),
  class_size = factor(class_size, levels = c("Small", "Medium", "Large"))
)

# Perform two-way ANOVA
education_model <- aov(score ~ method * class_size, data = education_data)
print(summary(education_model))

# Visualize interaction
ggplot(education_data, aes(x = class_size, y = score, color = method, group = method)) +
  stat_summary(fun = mean, geom = "line", size = 1) +
  stat_summary(fun = mean, geom = "point", size = 3) +
  labs(title = "Interaction Plot: Score by Teaching Method and Class Size",
       x = "Class Size", y = "Score", color = "Teaching Method") +
  theme_minimal()
```

### Example 2: Clinical Trial

```r
# Simulate clinical trial data
set.seed(123)
n_per_cell <- 20

# Generate data for 2x2 factorial design
# Factor 1: Treatment (Drug A, Drug B)
# Factor 2: Dosage (Low, High)

treatment <- rep(c("Drug A", "Drug B"), each = n_per_cell * 2)
dosage <- rep(rep(c("Low", "High"), each = n_per_cell), 2)

# Generate outcomes with interaction
outcomes <- numeric(length(treatment))
for (i in 1:length(treatment)) {
  if (treatment[i] == "Drug A") {
    if (dosage[i] == "Low") outcomes[i] <- rnorm(1, 60, 12)
    else outcomes[i] <- rnorm(1, 75, 10)
  } else {
    if (dosage[i] == "Low") outcomes[i] <- rnorm(1, 65, 11)
    else outcomes[i] <- rnorm(1, 85, 9)
  }
}

# Create data frame
clinical_data <- data.frame(
  outcome = outcomes,
  treatment = factor(treatment),
  dosage = factor(dosage, levels = c("Low", "High"))
)

# Perform two-way ANOVA
clinical_model <- aov(outcome ~ treatment * dosage, data = clinical_data)
print(summary(clinical_model))

# Simple effects analysis
simple_effects_analysis(clinical_data, "treatment", "dosage", "outcome")
```

### Example 3: Manufacturing Quality

```r
# Simulate manufacturing quality data
set.seed(123)
n_per_cell <- 12

# Generate data for 3x2 factorial design
# Factor 1: Machine Type (A, B, C)
# Factor 2: Shift (Day, Night)

machine <- rep(c("Machine A", "Machine B", "Machine C"), each = n_per_cell * 2)
shift <- rep(rep(c("Day", "Night"), each = n_per_cell), 3)

# Generate quality scores
quality_scores <- numeric(length(machine))
for (i in 1:length(machine)) {
  if (machine[i] == "Machine A") {
    if (shift[i] == "Day") quality_scores[i] <- rnorm(1, 95, 3)
    else quality_scores[i] <- rnorm(1, 92, 4)
  } else if (machine[i] == "Machine B") {
    if (shift[i] == "Day") quality_scores[i] <- rnorm(1, 88, 5)
    else quality_scores[i] <- rnorm(1, 85, 6)
  } else {
    if (shift[i] == "Day") quality_scores[i] <- rnorm(1, 90, 4)
    else quality_scores[i] <- rnorm(1, 87, 5)
  }
}

# Create data frame
quality_data <- data.frame(
  quality = quality_scores,
  machine = factor(machine),
  shift = factor(shift, levels = c("Day", "Night"))
)

# Perform two-way ANOVA
quality_model <- aov(quality ~ machine * shift, data = quality_data)
print(summary(quality_model))

# Visualize results
ggplot(quality_data, aes(x = machine, y = quality, fill = shift)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Quality Scores by Machine and Shift",
       x = "Machine", y = "Quality Score", fill = "Shift") +
  theme_minimal()
```

## Best Practices

### Test Selection Guidelines

```r
# Function to help choose appropriate two-way ANOVA test
choose_two_way_test <- function(data, factor1, factor2, response) {
  cat("=== TWO-WAY ANOVA TEST SELECTION ===\n")
  
  # Check normality
  normality_results <- check_normality_two_way(data, factor1, factor2, response)
  
  # Check homogeneity
  homogeneity_results <- check_homogeneity_two_way(data, factor1, factor2, response)
  
  # Check sample sizes
  cell_counts <- table(data[[factor1]], data[[factor2]])
  cat("Cell sample sizes:\n")
  print(cell_counts)
  
  # Check for balanced design
  balanced <- length(unique(as.vector(cell_counts))) == 1
  cat("Balanced design:", balanced, "\n")
  
  cat("\nFINAL RECOMMENDATION:\n")
  
  # Decision logic
  normal_residuals <- normality_results$shapiro_test$p.value >= 0.05
  equal_variances <- homogeneity_results$levene$`Pr(>F)`[1] >= 0.05
  
  if (normal_residuals && equal_variances) {
    cat("Use standard two-way ANOVA\n")
    cat("All assumptions are met\n")
  } else if (normal_residuals && !equal_variances) {
    cat("Consider robust alternatives or data transformation\n")
    cat("Residuals are normal but variances are unequal\n")
  } else {
    cat("Consider nonparametric alternatives\n")
    cat("Residuals are not normal\n")
  }
  
  return(list(
    normality = normality_results,
    homogeneity = homogeneity_results,
    cell_counts = cell_counts,
    balanced = balanced
  ))
}

# Apply to cylinder and transmission data
test_selection <- choose_two_way_test(mtcars, "cyl_factor", "am_factor", "mpg")
```

### Reporting Guidelines

```r
# Function to generate comprehensive two-way ANOVA report
generate_two_way_report <- function(anova_result, data, factor1, factor2, response) {
  cat("=== TWO-WAY ANOVA REPORT ===\n\n")
  
  # Descriptive statistics
  desc_stats <- data %>%
    group_by(!!sym(factor1), !!sym(factor2)) %>%
    summarise(
      n = n(),
      mean = mean(!!sym(response), na.rm = TRUE),
      sd = sd(!!sym(response), na.rm = TRUE)
    )
  
  cat("DESCRIPTIVE STATISTICS:\n")
  print(desc_stats)
  cat("\n")
  
  # ANOVA results
  anova_summary <- summary(anova_result)
  f_factor1 <- anova_summary[[1]]$`F value`[1]
  f_factor2 <- anova_summary[[1]]$`F value`[2]
  f_interaction <- anova_summary[[1]]$`F value`[3]
  p_factor1 <- anova_summary[[1]]$`Pr(>F)`[1]
  p_factor2 <- anova_summary[[1]]$`Pr(>F)`[2]
  p_interaction <- anova_summary[[1]]$`Pr(>F)`[3]
  
  cat("ANOVA RESULTS:\n")
  cat("Factor 1 (", factor1, ") F-statistic:", round(f_factor1, 3), "\n")
  cat("Factor 1 p-value:", round(p_factor1, 4), "\n")
  cat("Factor 2 (", factor2, ") F-statistic:", round(f_factor2, 3), "\n")
  cat("Factor 2 p-value:", round(p_factor2, 4), "\n")
  cat("Interaction F-statistic:", round(f_interaction, 3), "\n")
  cat("Interaction p-value:", round(p_interaction, 4), "\n\n")
  
  # Effect sizes
  ss_factor1 <- anova_summary[[1]]$`Sum Sq`[1]
  ss_factor2 <- anova_summary[[1]]$`Sum Sq`[2]
  ss_interaction <- anova_summary[[1]]$`Sum Sq`[3]
  ss_error <- anova_summary[[1]]$`Sum Sq`[4]
  
  partial_eta2_factor1 <- ss_factor1 / (ss_factor1 + ss_error)
  partial_eta2_factor2 <- ss_factor2 / (ss_factor2 + ss_error)
  partial_eta2_interaction <- ss_interaction / (ss_interaction + ss_error)
  
  cat("EFFECT SIZES (Partial η²):\n")
  cat("Factor 1:", round(partial_eta2_factor1, 3), "\n")
  cat("Factor 2:", round(partial_eta2_factor2, 3), "\n")
  cat("Interaction:", round(partial_eta2_interaction, 3), "\n\n")
  
  # Interpretation
  alpha <- 0.05
  cat("INTERPRETATION:\n")
  
  if (p_factor1 < alpha) {
    cat("- Significant main effect of", factor1, "\n")
  } else {
    cat("- No significant main effect of", factor1, "\n")
  }
  
  if (p_factor2 < alpha) {
    cat("- Significant main effect of", factor2, "\n")
  } else {
    cat("- No significant main effect of", factor2, "\n")
  }
  
  if (p_interaction < alpha) {
    cat("- Significant interaction effect\n")
    cat("- Simple effects analysis recommended\n")
  } else {
    cat("- No significant interaction effect\n")
    cat("- Main effects can be interpreted independently\n")
  }
}

# Generate report for cylinder and transmission ANOVA
generate_two_way_report(two_way_model, mtcars, "cyl_factor", "am_factor", "mpg")
```

## Exercises

### Exercise 1: Basic Two-Way ANOVA
Perform two-way ANOVA to analyze the effects of cylinder count and transmission type on horsepower in the mtcars dataset.

### Exercise 2: Interaction Analysis
Create a dataset with a significant interaction effect and perform simple effects analysis.

### Exercise 3: Assumption Checking
Conduct comprehensive assumption checking for two-way ANOVA and recommend appropriate alternatives.

### Exercise 4: Effect Size Analysis
Calculate and interpret different effect size measures for two-way ANOVA results.

### Exercise 5: Post Hoc Analysis
Perform appropriate post hoc tests for significant main effects and interactions.

## Next Steps

In the next chapter, we'll learn about repeated measures ANOVA for analyzing within-subject designs.

---

**Key Takeaways:**
- Two-way ANOVA analyzes effects of two independent variables and their interaction
- Always check assumptions before interpreting results
- Effect sizes provide important information about practical significance
- Simple effects analysis is needed when interactions are significant
- Post hoc tests are necessary for significant main effects
- Proper reporting includes descriptive statistics, test results, and effect sizes
- Interaction effects can modify the interpretation of main effects 