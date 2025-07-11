# Two-Way ANOVA

## Overview

Two-way Analysis of Variance (ANOVA) is a powerful statistical technique used to analyze the effects of two independent variables (factors) on a dependent variable, including their interaction effects. This design is more sophisticated than one-way ANOVA as it allows researchers to examine both main effects and interaction effects simultaneously, providing a more complete understanding of the relationships between variables.

### Key Concepts

**Main Effects**: The individual effect of each factor on the dependent variable, ignoring the other factor.

**Interaction Effects**: The combined effect of both factors that cannot be explained by their individual main effects alone. An interaction occurs when the effect of one factor depends on the level of the other factor.

**Factorial Design**: A research design where all combinations of factor levels are included in the study.

### Mathematical Foundation

The two-way ANOVA model can be expressed as:

```math
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}
```

Where:
- $`Y_{ijk}`$ is the observed value for the kth observation in the ith level of factor A and jth level of factor B
- $`\mu`$ is the overall population mean
- $`\alpha_i`$ is the effect of the ith level of factor A (main effect of factor A)
- $`\beta_j`$ is the effect of the jth level of factor B (main effect of factor B)
- $`(\alpha\beta)_{ij}`$ is the interaction effect between the ith level of factor A and jth level of factor B
- $`\epsilon_{ijk}`$ is the random error term

### Sum of Squares Decomposition

The total variability in the data is partitioned into four components:

```math
SS_{Total} = SS_A + SS_B + SS_{AB} + SS_{Error}
```

Where:
- $`SS_{Total} = \sum_{i=1}^{a}\sum_{j=1}^{b}\sum_{k=1}^{n_{ij}}(Y_{ijk} - \bar{Y}_{...})^2`$
- $`SS_A = \sum_{i=1}^{a}bn_i(\bar{Y}_{i..} - \bar{Y}_{...})^2`$ (Factor A main effect)
- $`SS_B = \sum_{j=1}^{b}an_j(\bar{Y}_{.j.} - \bar{Y}_{...})^2`$ (Factor B main effect)
- $`SS_{AB} = \sum_{i=1}^{a}\sum_{j=1}^{b}n_{ij}(\bar{Y}_{ij.} - \bar{Y}_{i..} - \bar{Y}_{.j.} + \bar{Y}_{...})^2`$ (Interaction effect)
- $`SS_{Error} = \sum_{i=1}^{a}\sum_{j=1}^{b}\sum_{k=1}^{n_{ij}}(Y_{ijk} - \bar{Y}_{ij.})^2`$ (Error term)

### Degrees of Freedom

```math
df_A = a - 1
df_B = b - 1
df_{AB} = (a - 1)(b - 1)
df_{Error} = N - ab
df_{Total} = N - 1
```

Where $`a`$ and $`b`$ are the number of levels for factors A and B respectively, and $`N`$ is the total sample size.

### F-Statistics

```math
F_A = \frac{MS_A}{MS_{Error}} = \frac{SS_A/df_A}{SS_{Error}/df_{Error}}
```

```math
F_B = \frac{MS_B}{MS_{Error}} = \frac{SS_B/df_B}{SS_{Error}/df_{Error}}
```

```math
F_{AB} = \frac{MS_{AB}}{MS_{Error}} = \frac{SS_{AB}/df_{AB}}{SS_{Error}/df_{Error}}
```

### Effect Size Measures

**Partial Eta-Squared** (most commonly used):
```math
\eta_p^2 = \frac{SS_{Effect}}{SS_{Effect} + SS_{Error}}
```

**Eta-Squared**:
```math
\eta^2 = \frac{SS_{Effect}}{SS_{Total}}
```

**Omega-Squared** (unbiased estimate):
```math
\omega^2 = \frac{SS_{Effect} - df_{Effect} \times MS_{Error}}{SS_{Total} + MS_{Error}}
```

## Basic Two-Way ANOVA

### Manual Calculation

The manual calculation helps understand the underlying mathematics and provides insight into how the statistical software performs the analysis.

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

R's `aov()` function provides a convenient way to perform two-way ANOVA with automatic calculation of all statistics.

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

### Understanding Cell Means and Marginal Means

In two-way ANOVA, we need to understand three types of means:

1. **Cell Means**: The mean of the dependent variable for each combination of factor levels
2. **Marginal Means**: The mean of the dependent variable for each level of a factor, averaged across all levels of the other factor
3. **Grand Mean**: The overall mean of the dependent variable across all observations

### Mathematical Definitions

**Cell Mean**: $`\bar{Y}_{ij.} = \frac{1}{n_{ij}}\sum_{k=1}^{n_{ij}}Y_{ijk}`$

**Marginal Mean for Factor A**: $`\bar{Y}_{i..} = \frac{1}{bn_i}\sum_{j=1}^{b}\sum_{k=1}^{n_{ij}}Y_{ijk}`$

**Marginal Mean for Factor B**: $`\bar{Y}_{.j.} = \frac{1}{an_j}\sum_{i=1}^{a}\sum_{k=1}^{n_{ij}}Y_{ijk}`$

**Grand Mean**: $`\bar{Y}_{...} = \frac{1}{N}\sum_{i=1}^{a}\sum_{j=1}^{b}\sum_{k=1}^{n_{ij}}Y_{ijk}`$

### Cell Means and Marginal Means

```r
# Calculate descriptive statistics
library(dplyr)

# Cell means - these represent the mean for each combination of factor levels
cell_means <- mtcars %>%
  group_by(cyl_factor, am_factor) %>%
  summarise(
    n = n(),
    mean = mean(mpg, na.rm = TRUE),
    sd = sd(mpg, na.rm = TRUE),
    se = sd / sqrt(n),
    ci_lower = mean - qt(0.975, n-1) * se,
    ci_upper = mean + qt(0.975, n-1) * se
  )

cat("Cell Means (Mean MPG for each combination):\n")
print(cell_means)

# Marginal means for Factor 1 (Cylinders) - averaged across transmission types
marginal_cylinders <- mtcars %>%
  group_by(cyl_factor) %>%
  summarise(
    n = n(),
    mean = mean(mpg, na.rm = TRUE),
    sd = sd(mpg, na.rm = TRUE),
    se = sd / sqrt(n),
    ci_lower = mean - qt(0.975, n-1) * se,
    ci_upper = mean + qt(0.975, n-1) * se
  )

cat("Marginal Means - Cylinders (averaged across transmission types):\n")
print(marginal_cylinders)

# Marginal means for Factor 2 (Transmission) - averaged across cylinder types
marginal_transmission <- mtcars %>%
  group_by(am_factor) %>%
  summarise(
    n = n(),
    mean = mean(mpg, na.rm = TRUE),
    sd = sd(mpg, na.rm = TRUE),
    se = sd / sqrt(n),
    ci_lower = mean - qt(0.975, n-1) * se,
    ci_upper = mean + qt(0.975, n-1) * se
  )

cat("Marginal Means - Transmission (averaged across cylinder types):\n")
print(marginal_transmission)

# Grand mean
grand_mean <- mean(mtcars$mpg, na.rm = TRUE)
cat("Grand Mean (overall average MPG):", round(grand_mean, 2), "\n")
```

### Understanding Interaction Effects

Interaction effects can be calculated as the difference between the observed cell mean and what would be expected based on the main effects alone:

```math
(\alpha\beta)_{ij} = \bar{Y}_{ij.} - \bar{Y}_{i..} - \bar{Y}_{.j.} + \bar{Y}_{...}
```

```r
# Calculate interaction effects
interaction_effects <- function(cell_means, marginal_factor1, marginal_factor2, grand_mean) {
  # Create a matrix of interaction effects
  interaction_matrix <- matrix(NA, nrow = nrow(marginal_factor1), ncol = nrow(marginal_factor2))
  rownames(interaction_matrix) <- marginal_factor1$cyl_factor
  colnames(interaction_matrix) <- marginal_factor2$am_factor
  
  for (i in 1:nrow(marginal_factor1)) {
    for (j in 1:nrow(marginal_factor2)) {
      cell_mean <- cell_means$mean[cell_means$cyl_factor == marginal_factor1$cyl_factor[i] & 
                                   cell_means$am_factor == marginal_factor2$am_factor[j]]
      factor1_mean <- marginal_factor1$mean[i]
      factor2_mean <- marginal_factor2$mean[j]
      
      interaction_effect <- cell_mean - factor1_mean - factor2_mean + grand_mean
      interaction_matrix[i, j] <- interaction_effect
    }
  }
  
  return(interaction_matrix)
}

# Calculate and display interaction effects
interaction_matrix <- interaction_effects(cell_means, marginal_cylinders, marginal_transmission, grand_mean)
cat("Interaction Effects Matrix:\n")
print(round(interaction_matrix, 3))

# Interpret interaction effects
cat("\nInteraction Effect Interpretation:\n")
for (i in 1:nrow(interaction_matrix)) {
  for (j in 1:ncol(interaction_matrix)) {
    effect <- interaction_matrix[i, j]
    if (abs(effect) > 0.5) {
      cat("Strong interaction between", rownames(interaction_matrix)[i], "and", 
          colnames(interaction_matrix)[j], ":", round(effect, 2), "\n")
    }
  }
}
```

### Visualization

Visualization is crucial for understanding two-way ANOVA results, especially for detecting interaction effects and patterns in the data.

#### Interaction Plot

The interaction plot is the most important visualization for two-way ANOVA as it shows how the effect of one factor changes across the levels of the other factor.

```r
library(ggplot2)
library(gridExtra)

# Interaction plot with confidence intervals
p1 <- ggplot(mtcars, aes(x = cyl_factor, y = mpg, color = am_factor, group = am_factor)) +
  stat_summary(fun = mean, geom = "line", size = 1.5) +
  stat_summary(fun = mean, geom = "point", size = 4) +
  stat_summary(fun.data = "mean_cl_normal", geom = "errorbar", width = 0.2, size = 1) +
  labs(title = "Interaction Plot: MPG by Cylinders and Transmission",
       subtitle = "Lines show the relationship between factors",
       x = "Number of Cylinders", y = "MPG", color = "Transmission") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"))

# Interpret interaction plot
cat("Interaction Plot Interpretation:\n")
cat("- Parallel lines indicate no interaction\n")
cat("- Non-parallel lines indicate interaction\n")
cat("- Crossing lines indicate strong interaction\n\n")
```

#### Box Plot with Individual Points

Box plots show the distribution of data within each cell, while individual points show the actual data distribution.

```r
# Box plot with individual points
p2 <- ggplot(mtcars, aes(x = cyl_factor, y = mpg, fill = am_factor)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.2, alpha = 0.6, size = 2) +
  labs(title = "MPG Distribution by Cylinders and Transmission",
       subtitle = "Boxes show quartiles, points show individual cars",
       x = "Number of Cylinders", y = "MPG", fill = "Transmission") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"))
```

#### Heatmap of Cell Means

A heatmap provides a visual representation of cell means, making it easy to identify patterns and interactions.

```r
# Heatmap of cell means with enhanced styling
p3 <- ggplot(cell_means, aes(x = cyl_factor, y = am_factor, fill = mean)) +
  geom_tile(color = "white", size = 1) +
  geom_text(aes(label = sprintf("%.1f", mean)), color = "white", size = 5, fontface = "bold") +
  scale_fill_gradient2(low = "#4575B4", mid = "#FFFFBF", high = "#D73027", 
                       midpoint = mean(cell_means$mean)) +
  labs(title = "Cell Means Heatmap",
       subtitle = "Color intensity represents MPG values",
       x = "Cylinders", y = "Transmission", fill = "Mean MPG") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"))
```

#### Marginal Effects Plot

This plot shows the main effects of each factor independently.

```r
# Marginal effects plot
marginal_data <- rbind(
  data.frame(factor = "Cylinders", level = marginal_cylinders$cyl_factor, 
             mean = marginal_cylinders$mean, se = marginal_cylinders$se),
  data.frame(factor = "Transmission", level = marginal_transmission$am_factor, 
             mean = marginal_transmission$mean, se = marginal_transmission$se)
)

p4 <- ggplot(marginal_data, aes(x = level, y = mean, color = factor)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2, size = 1) +
  geom_line(aes(group = factor), size = 1) +
  facet_wrap(~factor, scales = "free_x") +
  labs(title = "Marginal Effects",
       subtitle = "Main effects of each factor independently",
       x = "Factor Level", y = "Mean MPG", color = "Factor") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"))

# Combine all plots
grid.arrange(p1, p2, p3, p4, ncol = 2)
```

#### Residuals Plot

Checking residuals is important for assumption validation.

```r
# Residuals plot
model_residuals <- residuals(two_way_model)
fitted_values <- fitted(two_way_model)

p5 <- ggplot(data.frame(residuals = model_residuals, fitted = fitted_values), 
             aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_smooth(method = "loess", se = TRUE, color = "blue") +
  labs(title = "Residuals vs Fitted Values",
       subtitle = "Check for homoscedasticity and linearity",
       x = "Fitted Values", y = "Residuals") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"))

print(p5)
```

## Effect Size Analysis

Effect sizes are crucial for understanding the practical significance of statistical results, as they provide information about the magnitude of effects independent of sample size.

### Understanding Effect Size Measures

#### Partial Eta-Squared ($`\eta_p^2`$)

Partial eta-squared is the most commonly used effect size measure in ANOVA. It represents the proportion of variance in the dependent variable that is explained by a specific effect, controlling for other effects in the model.

```math
\eta_p^2 = \frac{SS_{Effect}}{SS_{Effect} + SS_{Error}}
```

**Advantages:**
- Controls for other effects in the model
- Ranges from 0 to 1
- Easy to interpret

**Disadvantages:**
- Can be biased upward in small samples
- Values can sum to more than 1

#### Eta-Squared ($`\eta^2`$)

Eta-squared represents the proportion of total variance explained by an effect.

```math
\eta^2 = \frac{SS_{Effect}}{SS_{Total}}
```

**Advantages:**
- Values sum to 1 across all effects
- Intuitive interpretation

**Disadvantages:**
- Does not control for other effects
- Can be misleading in factorial designs

#### Omega-Squared ($`\omega^2`$)

Omega-squared is an unbiased estimate of the population effect size.

```math
\omega^2 = \frac{SS_{Effect} - df_{Effect} \times MS_{Error}}{SS_{Total} + MS_{Error}}
```

**Advantages:**
- Unbiased estimate
- More conservative than eta-squared

**Disadvantages:**
- Can be negative for small effects
- Less commonly reported

### Comprehensive Effect Size Calculation

```r
# Calculate effect sizes with confidence intervals
calculate_two_way_effect_sizes <- function(anova_result, data, factor1, factor2, response) {
  # Extract Sum of Squares
  ss_factor1 <- anova_result$ss_factor1
  ss_factor2 <- anova_result$ss_factor2
  ss_interaction <- anova_result$ss_interaction
  ss_error <- anova_result$ss_error
  ss_total <- anova_result$ss_total
  
  # Degrees of freedom
  df_factor1 <- anova_result$df_factor1
  df_factor2 <- anova_result$df_factor2
  df_interaction <- anova_result$df_interaction
  df_error <- anova_result$df_error
  
  # Mean Squares
  ms_error <- anova_result$ms_error
  
  # Partial eta-squared
  partial_eta2_factor1 <- ss_factor1 / (ss_factor1 + ss_error)
  partial_eta2_factor2 <- ss_factor2 / (ss_factor2 + ss_error)
  partial_eta2_interaction <- ss_interaction / (ss_interaction + ss_error)
  
  # Eta-squared (total)
  eta2_factor1 <- ss_factor1 / ss_total
  eta2_factor2 <- ss_factor2 / ss_total
  eta2_interaction <- ss_interaction / ss_total
  
  # Omega-squared (unbiased)
  omega2_factor1 <- (ss_factor1 - df_factor1 * ms_error) / (ss_total + ms_error)
  omega2_factor2 <- (ss_factor2 - df_factor2 * ms_error) / (ss_total + ms_error)
  omega2_interaction <- (ss_interaction - df_interaction * ms_error) / (ss_total + ms_error)
  
  # Cohen's f (for power analysis)
  f_factor1 <- sqrt(partial_eta2_factor1 / (1 - partial_eta2_factor1))
  f_factor2 <- sqrt(partial_eta2_factor2 / (1 - partial_eta2_factor2))
  f_interaction <- sqrt(partial_eta2_interaction / (1 - partial_eta2_interaction))
  
  # Bootstrap confidence intervals for partial eta-squared
  library(boot)
  
  bootstrap_effect_size <- function(data, indices, factor1, factor2, response) {
    d <- data[indices, ]
    model <- aov(as.formula(paste(response, "~", factor1, "*", factor2)), data = d)
    summary_model <- summary(model)
    
    ss_factor1 <- summary_model[[1]]$`Sum Sq`[1]
    ss_factor2 <- summary_model[[1]]$`Sum Sq`[2]
    ss_interaction <- summary_model[[1]]$`Sum Sq`[3]
    ss_error <- summary_model[[1]]$`Sum Sq`[4]
    
    partial_eta2_factor1 <- ss_factor1 / (ss_factor1 + ss_error)
    partial_eta2_factor2 <- ss_factor2 / (ss_factor2 + ss_error)
    partial_eta2_interaction <- ss_interaction / (ss_interaction + ss_error)
    
    return(c(partial_eta2_factor1, partial_eta2_factor2, partial_eta2_interaction))
  }
  
  # Bootstrap for confidence intervals
  boot_results <- boot(data, bootstrap_effect_size, R = 1000, 
                      factor1 = factor1, factor2 = factor2, response = response)
  
  ci_factor1 <- boot.ci(boot_results, type = "perc", index = 1)
  ci_factor2 <- boot.ci(boot_results, type = "perc", index = 2)
  ci_interaction <- boot.ci(boot_results, type = "perc", index = 3)
  
  return(list(
    partial_eta2_factor1 = partial_eta2_factor1,
    partial_eta2_factor2 = partial_eta2_factor2,
    partial_eta2_interaction = partial_eta2_interaction,
    eta2_factor1 = eta2_factor1,
    eta2_factor2 = eta2_factor2,
    eta2_interaction = eta2_interaction,
    omega2_factor1 = omega2_factor1,
    omega2_factor2 = omega2_factor2,
    omega2_interaction = omega2_interaction,
    f_factor1 = f_factor1,
    f_factor2 = f_factor2,
    f_interaction = f_interaction,
    ci_factor1 = ci_factor1,
    ci_factor2 = ci_factor2,
    ci_interaction = ci_interaction
  ))
}

# Apply to our results
effect_sizes <- calculate_two_way_effect_sizes(anova_result, mtcars, "cyl_factor", "am_factor", "mpg")

# Display comprehensive effect size results
cat("=== COMPREHENSIVE EFFECT SIZE ANALYSIS ===\n\n")

cat("PARTIAL ETA-SQUARED (η²p):\n")
cat("Factor 1 (Cylinders):", round(effect_sizes$partial_eta2_factor1, 4), "\n")
cat("Factor 2 (Transmission):", round(effect_sizes$partial_eta2_factor2, 4), "\n")
cat("Interaction:", round(effect_sizes$partial_eta2_interaction, 4), "\n\n")

cat("ETA-SQUARED (η²):\n")
cat("Factor 1 (Cylinders):", round(effect_sizes$eta2_factor1, 4), "\n")
cat("Factor 2 (Transmission):", round(effect_sizes$eta2_factor2, 4), "\n")
cat("Interaction:", round(effect_sizes$eta2_interaction, 4), "\n\n")

cat("OMEGA-SQUARED (ω²):\n")
cat("Factor 1 (Cylinders):", round(effect_sizes$omega2_factor1, 4), "\n")
cat("Factor 2 (Transmission):", round(effect_sizes$omega2_factor2, 4), "\n")
cat("Interaction:", round(effect_sizes$omega2_interaction, 4), "\n\n")

cat("COHEN'S f (for power analysis):\n")
cat("Factor 1 (Cylinders):", round(effect_sizes$f_factor1, 4), "\n")
cat("Factor 2 (Transmission):", round(effect_sizes$f_factor2, 4), "\n")
cat("Interaction:", round(effect_sizes$f_interaction, 4), "\n\n")
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
  }
}

cat("EFFECT SIZE INTERPRETATION:\n")
cat("Factor 1 (Cylinders):", interpret_effect_size(effect_sizes$partial_eta2_factor1), "\n")
cat("Factor 2 (Transmission):", interpret_effect_size(effect_sizes$partial_eta2_factor2), "\n")
cat("Interaction:", interpret_effect_size(effect_sizes$partial_eta2_interaction), "\n\n")

# Practical significance assessment
cat("PRACTICAL SIGNIFICANCE ASSESSMENT:\n")
if (effect_sizes$partial_eta2_factor1 > 0.06) {
  cat("- Cylinder count has a practically significant effect on MPG\n")
}
if (effect_sizes$partial_eta2_factor2 > 0.06) {
  cat("- Transmission type has a practically significant effect on MPG\n")
}
if (effect_sizes$partial_eta2_interaction > 0.06) {
  cat("- There is a practically significant interaction effect\n")
}
```

### Effect Size Visualization

```r
# Create effect size comparison plot
effect_size_data <- data.frame(
  Effect = rep(c("Factor 1 (Cylinders)", "Factor 2 (Transmission)", "Interaction"), 3),
  Measure = rep(c("Partial η²", "η²", "ω²"), each = 3),
  Value = c(effect_sizes$partial_eta2_factor1, effect_sizes$partial_eta2_factor2, effect_sizes$partial_eta2_interaction,
            effect_sizes$eta2_factor1, effect_sizes$eta2_factor2, effect_sizes$eta2_interaction,
            effect_sizes$omega2_factor1, effect_sizes$omega2_factor2, effect_sizes$omega2_interaction)
)

ggplot(effect_size_data, aes(x = Effect, y = Value, fill = Measure)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  geom_hline(yintercept = c(0.01, 0.06, 0.14), linetype = "dashed", color = "red", alpha = 0.7) +
  annotate("text", x = 0.5, y = 0.01, label = "Negligible", hjust = 0, color = "red") +
  annotate("text", x = 0.5, y = 0.06, label = "Small", hjust = 0, color = "red") +
  annotate("text", x = 0.5, y = 0.14, label = "Medium", hjust = 0, color = "red") +
  labs(title = "Effect Size Comparison",
       subtitle = "Different measures of effect size for each factor and interaction",
       x = "Effect", y = "Effect Size", fill = "Measure") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray50"),
        axis.text.x = element_text(angle = 45, hjust = 1))
```

## Simple Effects Analysis

When a significant interaction is found in two-way ANOVA, the main effects cannot be interpreted independently. Instead, we must examine **simple effects** - the effect of one factor at each level of the other factor.

### Understanding Simple Effects

Simple effects analysis answers questions like:
- "Does the effect of teaching method differ across class sizes?"
- "Is there a difference between automatic and manual transmission for 4-cylinder engines?"
- "Do the effects of drug treatment vary by dosage level?"

### Mathematical Foundation

The simple effect of Factor A at level j of Factor B is:

```math
SS_{A|B_j} = \sum_{i=1}^{a} n_{ij}(\bar{Y}_{ij.} - \bar{Y}_{.j.})^2
```

With degrees of freedom: $`df_{A|B_j} = a - 1`$

The F-statistic for the simple effect is:

```math
F_{A|B_j} = \frac{MS_{A|B_j}}{MS_{Error}} = \frac{SS_{A|B_j}/df_{A|B_j}}{SS_{Error}/df_{Error}}
```

### Comprehensive Simple Effects Analysis

```r
# Enhanced function to perform simple effects analysis with effect sizes
simple_effects_analysis <- function(data, factor1, factor2, response, alpha = 0.05) {
  cat("=== COMPREHENSIVE SIMPLE EFFECTS ANALYSIS ===\n\n")
  
  # Store results for summary
  results <- list()
  
  # Simple effects of Factor 1 at each level of Factor 2
  cat("SIMPLE EFFECTS OF", factor1, "AT EACH LEVEL OF", factor2, ":\n")
  cat("=" * 50, "\n")
  
  factor2_levels <- unique(data[[factor2]])
  
  for (level in factor2_levels) {
    subset_data <- data[data[[factor2]] == level, ]
    
    if (length(unique(subset_data[[factor1]])) > 1) {
      # Perform one-way ANOVA
      simple_model <- aov(as.formula(paste(response, "~", factor1)), data = subset_data)
      simple_summary <- summary(simple_model)
      
      f_stat <- simple_summary[[1]]$`F value`[1]
      p_value <- simple_summary[[1]]$`Pr(>F)`[1]
      df_factor <- simple_summary[[1]]$`Df`[1]
      df_error <- simple_summary[[1]]$`Df`[2]
      
      # Calculate effect size
      ss_factor <- simple_summary[[1]]$`Sum Sq`[1]
      ss_error <- simple_summary[[1]]$`Sum Sq`[2]
      partial_eta2 <- ss_factor / (ss_factor + ss_error)
      
      # Calculate descriptive statistics
      desc_stats <- subset_data %>%
        group_by(!!sym(factor1)) %>%
        summarise(
          n = n(),
          mean = mean(!!sym(response), na.rm = TRUE),
          sd = sd(!!sym(response), na.rm = TRUE),
          se = sd / sqrt(n)
        )
      
      cat("At", factor2, "=", level, ":\n")
      cat("  F-statistic:", round(f_stat, 3), "\n")
      cat("  p-value:", round(p_value, 4), "\n")
      cat("  Partial η²:", round(partial_eta2, 4), "\n")
      cat("  Degrees of freedom:", df_factor, ",", df_error, "\n")
      
      if (p_value < alpha) {
        cat("  ✓ Significant effect (p <", alpha, ")\n")
      } else {
        cat("  ✗ Non-significant effect (p ≥", alpha, ")\n")
      }
      
      cat("  Descriptive statistics:\n")
      for (i in 1:nrow(desc_stats)) {
        cat("    ", desc_stats[[factor1]][i], ": M =", round(desc_stats$mean[i], 2), 
            ", SD =", round(desc_stats$sd[i], 2), ", n =", desc_stats$n[i], "\n")
      }
      
      # Store results
      results[[paste0(factor1, "_at_", factor2, "_", level)]] <- list(
        f_stat = f_stat,
        p_value = p_value,
        partial_eta2 = partial_eta2,
        df_factor = df_factor,
        df_error = df_error,
        desc_stats = desc_stats,
        significant = p_value < alpha
      )
      
      cat("\n")
    }
  }
  
  # Simple effects of Factor 2 at each level of Factor 1
  cat("SIMPLE EFFECTS OF", factor2, "AT EACH LEVEL OF", factor1, ":\n")
  cat("=" * 50, "\n")
  
  factor1_levels <- unique(data[[factor1]])
  
  for (level in factor1_levels) {
    subset_data <- data[data[[factor1]] == level, ]
    
    if (length(unique(subset_data[[factor2]])) > 1) {
      # Perform one-way ANOVA
      simple_model <- aov(as.formula(paste(response, "~", factor2)), data = subset_data)
      simple_summary <- summary(simple_model)
      
      f_stat <- simple_summary[[1]]$`F value`[1]
      p_value <- simple_summary[[1]]$`Pr(>F)`[1]
      df_factor <- simple_summary[[1]]$`Df`[1]
      df_error <- simple_summary[[1]]$`Df`[2]
      
      # Calculate effect size
      ss_factor <- simple_summary[[1]]$`Sum Sq`[1]
      ss_error <- simple_summary[[1]]$`Sum Sq`[2]
      partial_eta2 <- ss_factor / (ss_factor + ss_error)
      
      # Calculate descriptive statistics
      desc_stats <- subset_data %>%
        group_by(!!sym(factor2)) %>%
        summarise(
          n = n(),
          mean = mean(!!sym(response), na.rm = TRUE),
          sd = sd(!!sym(response), na.rm = TRUE),
          se = sd / sqrt(n)
        )
      
      cat("At", factor1, "=", level, ":\n")
      cat("  F-statistic:", round(f_stat, 3), "\n")
      cat("  p-value:", round(p_value, 4), "\n")
      cat("  Partial η²:", round(partial_eta2, 4), "\n")
      cat("  Degrees of freedom:", df_factor, ",", df_error, "\n")
      
      if (p_value < alpha) {
        cat("  ✓ Significant effect (p <", alpha, ")\n")
      } else {
        cat("  ✗ Non-significant effect (p ≥", alpha, ")\n")
      }
      
      cat("  Descriptive statistics:\n")
      for (i in 1:nrow(desc_stats)) {
        cat("    ", desc_stats[[factor2]][i], ": M =", round(desc_stats$mean[i], 2), 
            ", SD =", round(desc_stats$sd[i], 2), ", n =", desc_stats$n[i], "\n")
      }
      
      # Store results
      results[[paste0(factor2, "_at_", factor1, "_", level)]] <- list(
        f_stat = f_stat,
        p_value = p_value,
        partial_eta2 = partial_eta2,
        df_factor = df_factor,
        df_error = df_error,
        desc_stats = desc_stats,
        significant = p_value < alpha
      )
      
      cat("\n")
    }
  }
  
  # Summary of significant effects
  cat("SUMMARY OF SIGNIFICANT SIMPLE EFFECTS:\n")
  cat("=" * 40, "\n")
  
  significant_effects <- 0
  for (effect_name in names(results)) {
    if (results[[effect_name]]$significant) {
      cat("✓", effect_name, "(p =", round(results[[effect_name]]$p_value, 4), ")\n")
      significant_effects <- significant_effects + 1
    }
  }
  
  if (significant_effects == 0) {
    cat("No significant simple effects found.\n")
  } else {
    cat("Total significant simple effects:", significant_effects, "\n")
  }
  
  return(results)
}

# Apply enhanced simple effects analysis
simple_effects_results <- simple_effects_analysis(mtcars, "cyl_factor", "am_factor", "mpg")
```

### Simple Effects Visualization

```r
# Create visualization for simple effects
library(ggplot2)

# Function to create simple effects plots
plot_simple_effects <- function(data, factor1, factor2, response) {
  # Create plots for simple effects of factor1 at each level of factor2
  factor2_levels <- unique(data[[factor2]])
  
  plots <- list()
  
  for (i in 1:length(factor2_levels)) {
    level <- factor2_levels[i]
    subset_data <- data[data[[factor2]] == level, ]
    
    p <- ggplot(subset_data, aes(x = !!sym(factor1), y = !!sym(response))) +
      geom_boxplot(alpha = 0.7, fill = "lightblue") +
      geom_jitter(width = 0.2, alpha = 0.6, size = 2) +
      stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "red") +
      labs(title = paste("Simple Effect of", factor1, "at", factor2, "=", level),
           subtitle = paste("One-way ANOVA within", factor2, "=", level),
           x = factor1, y = response) +
      theme_minimal() +
      theme(plot.title = element_text(size = 12, face = "bold"),
            plot.subtitle = element_text(size = 10, color = "gray50"))
    
    plots[[i]] <- p
  }
  
  # Combine plots
  do.call(grid.arrange, c(plots, ncol = 2))
}

# Create simple effects plots
plot_simple_effects(mtcars, "cyl_factor", "am_factor", "mpg")
```

### Post Hoc Tests for Simple Effects

```r
# Post hoc tests for significant simple effects
library(multcomp)

# Function to perform post hoc tests for simple effects
simple_effects_posthoc <- function(simple_effects_results, data, factor1, factor2, response) {
  cat("=== POST HOC TESTS FOR SIGNIFICANT SIMPLE EFFECTS ===\n\n")
  
  for (effect_name in names(simple_effects_results)) {
    if (simple_effects_results[[effect_name]]$significant) {
      cat("Post hoc tests for:", effect_name, "\n")
      cat("-" * 40, "\n")
      
      # Extract factor information from effect name
      if (grepl(paste0(factor1, "_at_"), effect_name)) {
        # Simple effect of factor1 at a level of factor2
        level <- gsub(paste0(factor1, "_at_", factor2, "_"), "", effect_name)
        subset_data <- data[data[[factor2]] == level, ]
        
        if (length(unique(subset_data[[factor1]])) > 2) {
          # Perform Tukey's HSD
          model <- aov(as.formula(paste(response, "~", factor1)), data = subset_data)
          tukey_result <- glht(model, linfct = mcp(!!sym(factor1) = "Tukey"))
          tukey_summary <- summary(tukey_result)
          
          cat("Tukey's HSD results:\n")
          print(tukey_summary)
          
          # Extract significant pairwise comparisons
          pvalues <- tukey_summary$test$pvalues
          comparisons <- names(pvalues)
          
          cat("Significant pairwise differences (p < 0.05):\n")
          significant_comparisons <- 0
          for (j in 1:length(pvalues)) {
            if (pvalues[j] < 0.05) {
              cat("  ", comparisons[j], ": p =", round(pvalues[j], 4), "\n")
              significant_comparisons <- significant_comparisons + 1
            }
          }
          
          if (significant_comparisons == 0) {
            cat("  No significant pairwise differences found.\n")
          }
        } else {
          cat("Only 2 levels - no post hoc tests needed.\n")
        }
      }
      
      cat("\n")
    }
  }
}

# Apply post hoc tests
simple_effects_posthoc(simple_effects_results, mtcars, "cyl_factor", "am_factor", "mpg")
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

Two-way ANOVA relies on several key assumptions that must be verified before interpreting results. Violations of these assumptions can lead to incorrect conclusions.

### Key Assumptions

1. **Normality**: Residuals should be normally distributed
2. **Homogeneity of Variance**: Variances should be equal across all groups
3. **Independence**: Observations should be independent
4. **Linearity**: The relationship between factors and the dependent variable should be linear

### Comprehensive Normality Testing

```r
# Enhanced function to check normality for two-way ANOVA
check_normality_two_way <- function(data, factor1, factor2, response, alpha = 0.05) {
  cat("=== COMPREHENSIVE NORMALITY TESTS FOR TWO-WAY ANOVA ===\n\n")
  
  # Fit the model
  model <- aov(as.formula(paste(response, "~", factor1, "*", factor2)), data = data)
  residuals <- residuals(model)
  fitted_values <- fitted(model)
  
  cat("NORMALITY TESTS ON RESIDUALS:\n")
  cat("=" * 40, "\n")
  
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
  
  # Combine plots
  combined_plot <- grid.arrange(qq_plot, hist_plot, fitted_plot, ncol = 2)
  
  # Recommendations based on results
  cat("\nRECOMMENDATIONS:\n")
  cat("=" * 15, "\n")
  
  if (normal_tests >= 3) {
    cat("✓ Two-way ANOVA is appropriate for these data\n")
    cat("✓ All normality assumptions appear to be met\n")
  } else if (normal_tests >= 2) {
    cat("⚠ Consider robust alternatives or data transformation\n")
    cat("⚠ Two-way ANOVA may still be appropriate with caution\n")
  } else {
    cat("✗ Consider nonparametric alternatives:\n")
    cat("  - Kruskal-Wallis test for main effects\n")
    cat("  - Permutation tests\n")
    cat("  - Data transformation (log, square root, etc.)\n")
  }
  
  return(list(
    tests = tests,
    p_values = p_values,
    test_names = test_names,
    normal_tests = normal_tests,
    total_tests = total_tests,
    qq_plot = qq_plot,
    hist_plot = hist_plot,
    fitted_plot = fitted_plot,
    combined_plot = combined_plot
  ))
}

# Check normality with enhanced function
normality_results <- check_normality_two_way(mtcars, "cyl_factor", "am_factor", "mpg")
```

### Comprehensive Homogeneity of Variance Testing

```r
# Enhanced function to test homogeneity of variance for two-way ANOVA
check_homogeneity_two_way <- function(data, factor1, factor2, response, alpha = 0.05) {
  cat("=== COMPREHENSIVE HOMOGENEITY OF VARIANCE TESTS ===\n\n")
  
  # Create interaction factor for testing
  data$interaction_factor <- interaction(data[[factor1]], data[[factor2]])
  
  cat("HOMOGENEITY OF VARIANCE TESTS:\n")
  cat("=" * 35, "\n")
  
  # Multiple homogeneity tests
  tests <- list()
  
  # Levene's test (most robust)
  library(car)
  levene_result <- leveneTest(as.formula(paste(response, "~", factor1, "*", factor2)), data = data)
  tests$levene <- levene_result
  cat("Levene's test (robust to non-normality):\n")
  cat("  F =", round(levene_result$`F value`[1], 4), "\n")
  cat("  p-value =", round(levene_result$`Pr(>F)`[1], 4), "\n")
  cat("  Decision:", ifelse(levene_result$`Pr(>F)`[1] >= alpha, "✓ Equal variances", "✗ Unequal variances"), "\n\n")
  
  # Bartlett's test (sensitive to non-normality)
  bartlett_result <- bartlett.test(as.formula(paste(response, "~", factor1, "*", factor2)), data = data)
  tests$bartlett <- bartlett_result
  cat("Bartlett's test (sensitive to non-normality):\n")
  cat("  K-squared =", round(bartlett_result$statistic, 4), "\n")
  cat("  p-value =", round(bartlett_result$p.value, 4), "\n")
  cat("  Decision:", ifelse(bartlett_result$p.value >= alpha, "✓ Equal variances", "✗ Unequal variances"), "\n\n")
  
  # Fligner-Killeen test (robust)
  fligner_result <- fligner.test(as.formula(paste(response, "~", factor1, "*", factor2)), data = data)
  tests$fligner <- fligner_result
  cat("Fligner-Killeen test (robust):\n")
  cat("  chi-squared =", round(fligner_result$statistic, 4), "\n")
  cat("  p-value =", round(fligner_result$p.value, 4), "\n")
  cat("  Decision:", ifelse(fligner_result$p.value >= alpha, "✓ Equal variances", "✗ Unequal variances"), "\n\n")
  
  # Brown-Forsythe test (robust)
  library(onewaytests)
  bf_result <- bf.test(as.formula(paste(response, "~", "interaction_factor")), data = data)
  tests$brown_forsythe <- bf_result
  cat("Brown-Forsythe test (robust):\n")
  cat("  F =", round(bf_result$statistic, 4), "\n")
  cat("  p-value =", round(bf_result$p.value, 4), "\n")
  cat("  Decision:", ifelse(bf_result$p.value >= alpha, "✓ Equal variances", "✗ Unequal variances"), "\n\n")
  
  # Summary of homogeneity tests
  cat("HOMOGENEITY TEST SUMMARY:\n")
  cat("=" * 28, "\n")
  
  p_values <- c(levene_result$`Pr(>F)`[1], bartlett_result$p.value, 
                fligner_result$p.value, bf_result$p.value)
  test_names <- c("Levene's", "Bartlett's", "Fligner-Killeen", "Brown-Forsythe")
  
  equal_var_tests <- sum(p_values >= alpha)
  total_tests <- length(p_values)
  
  cat("Tests supporting equal variances:", equal_var_tests, "out of", total_tests, "\n")
  
  if (equal_var_tests >= 3) {
    cat("✓ Homogeneity of variance assumption appears to be met\n")
  } else if (equal_var_tests >= 2) {
    cat("⚠ Homogeneity of variance assumption may be questionable\n")
  } else {
    cat("✗ Homogeneity of variance assumption appears to be violated\n")
  }
  
  # Calculate and display group variances
  cat("\nGROUP VARIANCES:\n")
  cat("=" * 15, "\n")
  
  group_vars <- data %>%
    group_by(!!sym(factor1), !!sym(factor2)) %>%
    summarise(
      n = n(),
      variance = var(!!sym(response), na.rm = TRUE),
      sd = sd(!!sym(response), na.rm = TRUE)
    )
  
  print(group_vars)
  
  # Calculate variance ratio (largest/smallest)
  max_var <- max(group_vars$variance, na.rm = TRUE)
  min_var <- min(group_vars$variance, na.rm = TRUE)
  var_ratio <- max_var / min_var
  
  cat("\nVariance ratio (largest/smallest):", round(var_ratio, 2), "\n")
  
  if (var_ratio <= 4) {
    cat("✓ Variance ratio is acceptable (≤ 4)\n")
  } else if (var_ratio <= 10) {
    cat("⚠ Variance ratio is questionable (4-10)\n")
  } else {
    cat("✗ Variance ratio is too large (> 10)\n")
  }
  
  # Create diagnostic plots
  library(ggplot2)
  library(gridExtra)
  
  # Box plot to visualize variance differences
  box_plot <- ggplot(data, aes(x = interaction(!!sym(factor1), !!sym(factor2)), 
                               y = !!sym(response), fill = !!sym(factor1))) +
    geom_boxplot(alpha = 0.7) +
    labs(title = "Box Plot by Group",
         subtitle = "Check for differences in spread (variance)",
         x = "Group", y = response, fill = factor1) +
    theme_minimal() +
    theme(plot.title = element_text(size = 12, face = "bold"),
          plot.subtitle = element_text(size = 10, color = "gray50"),
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Residuals vs fitted values (already created in normality check)
  # This plot also helps assess homogeneity
  
  # Variance vs mean plot
  var_mean_data <- group_vars %>%
    mutate(mean = tapply(data[[response]], 
                        interaction(data[[factor1]], data[[factor2]]), 
                        mean, na.rm = TRUE))
  
  var_mean_plot <- ggplot(var_mean_data, aes(x = mean, y = variance)) +
    geom_point(size = 3, alpha = 0.7) +
    geom_smooth(method = "lm", se = TRUE, color = "red") +
    labs(title = "Variance vs Mean Plot",
         subtitle = "Check for heteroscedasticity patterns",
         x = "Group Mean", y = "Group Variance") +
    theme_minimal() +
    theme(plot.title = element_text(size = 12, face = "bold"),
          plot.subtitle = element_text(size = 10, color = "gray50"))
  
  # Combine plots
  combined_plot <- grid.arrange(box_plot, var_mean_plot, ncol = 2)
  
  # Recommendations based on results
  cat("\nRECOMMENDATIONS:\n")
  cat("=" * 15, "\n")
  
  if (equal_var_tests >= 3 && var_ratio <= 4) {
    cat("✓ Standard two-way ANOVA is appropriate\n")
    cat("✓ All homogeneity assumptions appear to be met\n")
  } else if (equal_var_tests >= 2 && var_ratio <= 10) {
    cat("⚠ Consider robust alternatives:\n")
    cat("  - Welch's ANOVA (if available for two-way)\n")
    cat("  - Data transformation (log, square root, etc.)\n")
    cat("  - Two-way ANOVA may still be appropriate with caution\n")
  } else {
    cat("✗ Consider alternatives:\n")
    cat("  - Robust statistical methods\n")
    cat("  - Nonparametric alternatives\n")
    cat("  - Data transformation\n")
    cat("  - Bootstrapping methods\n")
  }
  
  return(list(
    tests = tests,
    p_values = p_values,
    test_names = test_names,
    equal_var_tests = equal_var_tests,
    total_tests = total_tests,
    group_vars = group_vars,
    var_ratio = var_ratio,
    box_plot = box_plot,
    var_mean_plot = var_mean_plot,
    combined_plot = combined_plot
  ))
}

# Check homogeneity with enhanced function
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

## Best Practices and Guidelines

Following best practices ensures reliable, reproducible, and interpretable two-way ANOVA results. This section provides comprehensive guidelines for conducting and reporting two-way ANOVA analyses.

### Test Selection Guidelines

```r
# Comprehensive function to help choose appropriate two-way ANOVA test
choose_two_way_test <- function(data, factor1, factor2, response, alpha = 0.05) {
  cat("=== COMPREHENSIVE TWO-WAY ANOVA TEST SELECTION ===\n\n")
  
  # Check normality
  cat("STEP 1: NORMALITY ASSESSMENT\n")
  cat("=" * 30, "\n")
  normality_results <- check_normality_two_way(data, factor1, factor2, response, alpha)
  
  # Check homogeneity
  cat("\nSTEP 2: HOMOGENEITY ASSESSMENT\n")
  cat("=" * 32, "\n")
  homogeneity_results <- check_homogeneity_two_way(data, factor1, factor2, response, alpha)
  
  # Check sample sizes
  cat("\nSTEP 3: SAMPLE SIZE ASSESSMENT\n")
  cat("=" * 28, "\n")
  cell_counts <- table(data[[factor1]], data[[factor2]])
  cat("Cell sample sizes:\n")
  print(cell_counts)
  
  # Check for balanced design
  balanced <- length(unique(as.vector(cell_counts))) == 1
  cat("Balanced design:", balanced, "\n")
  
  # Calculate total sample size
  total_n <- sum(cell_counts)
  cat("Total sample size:", total_n, "\n")
  
  # Check minimum cell size
  min_cell_size <- min(cell_counts)
  cat("Minimum cell size:", min_cell_size, "\n")
  
  # Sample size recommendations
  if (min_cell_size < 5) {
    cat("⚠ Warning: Very small cell sizes (< 5)\n")
  } else if (min_cell_size < 10) {
    cat("⚠ Warning: Small cell sizes (< 10)\n")
  } else if (min_cell_size >= 20) {
    cat("✓ Adequate cell sizes (≥ 20)\n")
  }
  
  # Check for missing data
  missing_data <- sum(is.na(data[[response]]))
  if (missing_data > 0) {
    cat("Missing data points:", missing_data, "\n")
  } else {
    cat("✓ No missing data\n")
  }
  
  cat("\nSTEP 4: FINAL RECOMMENDATION\n")
  cat("=" * 25, "\n")
  
  # Decision logic based on comprehensive assessment
  normal_tests_passed <- normality_results$normal_tests >= 3
  homogeneity_tests_passed <- homogeneity_results$equal_var_tests >= 3
  adequate_sample_size <- min_cell_size >= 10
  
  if (normal_tests_passed && homogeneity_tests_passed && adequate_sample_size) {
    cat("✓ RECOMMENDATION: Standard Two-Way ANOVA\n")
    cat("  - All assumptions appear to be met\n")
    cat("  - Sample sizes are adequate\n")
    cat("  - Results should be reliable\n")
  } else if (normal_tests_passed && !homogeneity_tests_passed && adequate_sample_size) {
    cat("⚠ RECOMMENDATION: Robust Alternatives\n")
    cat("  - Normality assumption met\n")
    cat("  - Homogeneity assumption violated\n")
    cat("  - Consider: Welch's ANOVA, data transformation, or robust methods\n")
  } else if (!normal_tests_passed && adequate_sample_size) {
    cat("⚠ RECOMMENDATION: Nonparametric or Robust Methods\n")
    cat("  - Normality assumption violated\n")
    cat("  - Consider: Kruskal-Wallis, permutation tests, or robust ANOVA\n")
  } else if (!adequate_sample_size) {
    cat("⚠ RECOMMENDATION: Caution Required\n")
    cat("  - Sample sizes are small\n")
    cat("  - Consider: Larger sample size, nonparametric methods, or bootstrapping\n")
  }
  
  # Additional considerations
  cat("\nADDITIONAL CONSIDERATIONS:\n")
  cat("=" * 25, "\n")
  
  if (!balanced) {
    cat("- Unbalanced design detected\n")
    cat("- Consider Type III Sum of Squares\n")
    cat("- Be cautious with interaction interpretation\n")
  }
  
  if (missing_data > 0) {
    cat("- Missing data present\n")
    cat("- Consider multiple imputation or complete case analysis\n")
  }
  
  # Power analysis recommendation
  if (total_n < 50) {
    cat("- Small total sample size\n")
    cat("- Consider power analysis for future studies\n")
  }
  
  return(list(
    normality = normality_results,
    homogeneity = homogeneity_results,
    cell_counts = cell_counts,
    balanced = balanced,
    total_n = total_n,
    min_cell_size = min_cell_size,
    missing_data = missing_data,
    recommendation = list(
      normal_tests_passed = normal_tests_passed,
      homogeneity_tests_passed = homogeneity_tests_passed,
      adequate_sample_size = adequate_sample_size
    )
  ))
}

# Apply comprehensive test selection
test_selection <- choose_two_way_test(mtcars, "cyl_factor", "am_factor", "mpg")
```

### Data Preparation Best Practices

```r
# Function to prepare data for two-way ANOVA
prepare_data_for_two_way_anova <- function(data, factor1, factor2, response) {
  cat("=== DATA PREPARATION FOR TWO-WAY ANOVA ===\n\n")
  
  # Check data structure
  cat("DATA STRUCTURE CHECK:\n")
  cat("=" * 20, "\n")
  cat("Number of observations:", nrow(data), "\n")
  cat("Number of variables:", ncol(data), "\n")
  cat("Factor 1 levels:", length(unique(data[[factor1]])), "\n")
  cat("Factor 2 levels:", length(unique(data[[factor2]])), "\n\n")
  
  # Check for missing values
  missing_summary <- data %>%
    summarise(
      factor1_missing = sum(is.na(!!sym(factor1))),
      factor2_missing = sum(is.na(!!sym(factor2))),
      response_missing = sum(is.na(!!sym(response)))
    )
  
  cat("MISSING DATA CHECK:\n")
  cat("=" * 18, "\n")
  cat("Factor 1 missing:", missing_summary$factor1_missing, "\n")
  cat("Factor 2 missing:", missing_summary$factor2_missing, "\n")
  cat("Response missing:", missing_summary$response_missing, "\n\n")
  
  # Create clean dataset
  clean_data <- data %>%
    filter(!is.na(!!sym(factor1)) & !is.na(!!sym(factor2)) & !is.na(!!sym(response)))
  
  cat("CLEAN DATA SUMMARY:\n")
  cat("=" * 18, "\n")
  cat("Observations removed:", nrow(data) - nrow(clean_data), "\n")
  cat("Remaining observations:", nrow(clean_data), "\n\n")
  
  # Ensure factors are properly coded
  clean_data[[factor1]] <- factor(clean_data[[factor1]])
  clean_data[[factor2]] <- factor(clean_data[[factor2]])
  
  # Check factor levels
  cat("FACTOR LEVELS:\n")
  cat("=" * 14, "\n")
  cat("Factor 1 levels:", levels(clean_data[[factor1]]), "\n")
  cat("Factor 2 levels:", levels(clean_data[[factor2]]), "\n\n")
  
  # Check for empty cells
  cell_counts <- table(clean_data[[factor1]], clean_data[[factor2]])
  empty_cells <- sum(cell_counts == 0)
  
  if (empty_cells > 0) {
    cat("⚠ WARNING: Empty cells detected\n")
    cat("Empty cells:", empty_cells, "\n")
    cat("This may cause issues with the analysis\n\n")
  } else {
    cat("✓ No empty cells detected\n\n")
  }
  
  # Check for outliers
  cat("OUTLIER DETECTION:\n")
  cat("=" * 18, "\n")
  
  # Boxplot method
  boxplot_stats <- boxplot.stats(clean_data[[response]])
  outliers <- boxplot_stats$out
  
  if (length(outliers) > 0) {
    cat("Potential outliers detected:", length(outliers), "\n")
    cat("Outlier values:", round(outliers, 2), "\n")
    cat("Consider investigating these values\n\n")
  } else {
    cat("✓ No outliers detected by boxplot method\n\n")
  }
  
  return(list(
    original_data = data,
    clean_data = clean_data,
    missing_summary = missing_summary,
    cell_counts = cell_counts,
    empty_cells = empty_cells,
    outliers = outliers
  ))
}

# Prepare data
data_prep <- prepare_data_for_two_way_anova(mtcars, "cyl_factor", "am_factor", "mpg")
```

### Reporting Guidelines

```r
# Function to generate comprehensive two-way ANOVA report
generate_two_way_report <- function(anova_result, data, factor1, factor2, response, 
                                   normality_results = NULL, homogeneity_results = NULL) {
  cat("=== COMPREHENSIVE TWO-WAY ANOVA REPORT ===\n\n")
  
  # 1. Descriptive Statistics
  cat("1. DESCRIPTIVE STATISTICS\n")
  cat("=" * 25, "\n")
  
  desc_stats <- data %>%
    group_by(!!sym(factor1), !!sym(factor2)) %>%
    summarise(
      n = n(),
      mean = mean(!!sym(response), na.rm = TRUE),
      sd = sd(!!sym(response), na.rm = TRUE),
      se = sd / sqrt(n),
      min = min(!!sym(response), na.rm = TRUE),
      max = max(!!sym(response), na.rm = TRUE)
    )
  
  print(desc_stats)
  
  # Marginal means
  marginal_factor1 <- data %>%
    group_by(!!sym(factor1)) %>%
    summarise(
      n = n(),
      mean = mean(!!sym(response), na.rm = TRUE),
      sd = sd(!!sym(response), na.rm = TRUE)
    )
  
  marginal_factor2 <- data %>%
    group_by(!!sym(factor2)) %>%
    summarise(
      n = n(),
      mean = mean(!!sym(response), na.rm = TRUE),
      sd = sd(!!sym(response), na.rm = TRUE)
    )
  
  cat("\nMarginal means for", factor1, ":\n")
  print(marginal_factor1)
  
  cat("\nMarginal means for", factor2, ":\n")
  print(marginal_factor2)
  
  # 2. ANOVA Results
  cat("\n2. TWO-WAY ANOVA RESULTS\n")
  cat("=" * 25, "\n")
  
  anova_summary <- summary(anova_result)
  print(anova_summary)
  
  # Extract key statistics
  f_factor1 <- anova_summary[[1]]$`F value`[1]
  f_factor2 <- anova_summary[[1]]$`F value`[2]
  f_interaction <- anova_summary[[1]]$`F value`[3]
  p_factor1 <- anova_summary[[1]]$`Pr(>F)`[1]
  p_factor2 <- anova_summary[[1]]$`Pr(>F)`[2]
  p_interaction <- anova_summary[[1]]$`Pr(>F)`[3]
  
  # 3. Effect Sizes
  cat("\n3. EFFECT SIZES\n")
  cat("=" * 15, "\n")
  
  ss_factor1 <- anova_summary[[1]]$`Sum Sq`[1]
  ss_factor2 <- anova_summary[[1]]$`Sum Sq`[2]
  ss_interaction <- anova_summary[[1]]$`Sum Sq`[3]
  ss_error <- anova_summary[[1]]$`Sum Sq`[4]
  ss_total <- sum(anova_summary[[1]]$`Sum Sq`)
  
  partial_eta2_factor1 <- ss_factor1 / (ss_factor1 + ss_error)
  partial_eta2_factor2 <- ss_factor2 / (ss_factor2 + ss_error)
  partial_eta2_interaction <- ss_interaction / (ss_interaction + ss_error)
  
  cat("Partial η² values:\n")
  cat("  ", factor1, ":", round(partial_eta2_factor1, 4), "\n")
  cat("  ", factor2, ":", round(partial_eta2_factor2, 4), "\n")
  cat("  Interaction:", round(partial_eta2_interaction, 4), "\n")
  
  # 4. Assumption Checking Summary
  if (!is.null(normality_results) && !is.null(homogeneity_results)) {
    cat("\n4. ASSUMPTION CHECKING SUMMARY\n")
    cat("=" * 30, "\n")
    
    cat("Normality tests passed:", normality_results$normal_tests, "/", normality_results$total_tests, "\n")
    cat("Homogeneity tests passed:", homogeneity_results$equal_var_tests, "/", homogeneity_results$total_tests, "\n")
    
    if (normality_results$normal_tests >= 3 && homogeneity_results$equal_var_tests >= 3) {
      cat("✓ All assumptions appear to be met\n")
    } else {
      cat("⚠ Some assumptions may be violated\n")
    }
  }
  
  # 5. Interpretation
  cat("\n5. INTERPRETATION\n")
  cat("=" * 15, "\n")
  
  alpha <- 0.05
  
  # Main effects
  if (p_factor1 < alpha) {
    cat("✓ Significant main effect of", factor1, "(p =", round(p_factor1, 4), ")\n")
    if (partial_eta2_factor1 >= 0.14) {
      cat("  Large effect size (η²p =", round(partial_eta2_factor1, 3), ")\n")
    } else if (partial_eta2_factor1 >= 0.06) {
      cat("  Medium effect size (η²p =", round(partial_eta2_factor1, 3), ")\n")
    } else {
      cat("  Small effect size (η²p =", round(partial_eta2_factor1, 3), ")\n")
    }
  } else {
    cat("✗ No significant main effect of", factor1, "(p =", round(p_factor1, 4), ")\n")
  }
  
  if (p_factor2 < alpha) {
    cat("✓ Significant main effect of", factor2, "(p =", round(p_factor2, 4), ")\n")
    if (partial_eta2_factor2 >= 0.14) {
      cat("  Large effect size (η²p =", round(partial_eta2_factor2, 3), ")\n")
    } else if (partial_eta2_factor2 >= 0.06) {
      cat("  Medium effect size (η²p =", round(partial_eta2_factor2, 3), ")\n")
    } else {
      cat("  Small effect size (η²p =", round(partial_eta2_factor2, 3), ")\n")
    }
  } else {
    cat("✗ No significant main effect of", factor2, "(p =", round(p_factor2, 4), ")\n")
  }
  
  # Interaction effect
  if (p_interaction < alpha) {
    cat("✓ Significant interaction effect (p =", round(p_interaction, 4), ")\n")
    if (partial_eta2_interaction >= 0.14) {
      cat("  Large interaction effect (η²p =", round(partial_eta2_interaction, 3), ")\n")
    } else if (partial_eta2_interaction >= 0.06) {
      cat("  Medium interaction effect (η²p =", round(partial_eta2_interaction, 3), ")\n")
    } else {
      cat("  Small interaction effect (η²p =", round(partial_eta2_interaction, 3), ")\n")
    }
    cat("  → Simple effects analysis recommended\n")
  } else {
    cat("✗ No significant interaction effect (p =", round(p_interaction, 4), ")\n")
    cat("  → Main effects can be interpreted independently\n")
  }
  
  # 6. Recommendations
  cat("\n6. RECOMMENDATIONS\n")
  cat("=" * 17, "\n")
  
  if (p_interaction < alpha) {
    cat("- Conduct simple effects analysis\n")
    cat("- Perform post hoc tests for significant simple effects\n")
    cat("- Focus interpretation on interaction effects\n")
  } else {
    if (p_factor1 < alpha || p_factor2 < alpha) {
      cat("- Perform post hoc tests for significant main effects\n")
      cat("- Consider planned comparisons if theoretically justified\n")
    }
  }
  
  cat("- Report effect sizes alongside p-values\n")
  cat("- Consider practical significance of results\n")
  cat("- Validate findings with additional analyses if needed\n")
  
  return(list(
    descriptive_stats = desc_stats,
    marginal_factor1 = marginal_factor1,
    marginal_factor2 = marginal_factor2,
    anova_results = anova_summary,
    effect_sizes = list(
      partial_eta2_factor1 = partial_eta2_factor1,
      partial_eta2_factor2 = partial_eta2_factor2,
      partial_eta2_interaction = partial_eta2_interaction
    ),
    interpretation = list(
      factor1_significant = p_factor1 < alpha,
      factor2_significant = p_factor2 < alpha,
      interaction_significant = p_interaction < alpha
    )
  ))
}

# Generate comprehensive report
comprehensive_report <- generate_two_way_report(two_way_model, mtcars, "cyl_factor", "am_factor", "mpg",
                                               normality_results, homogeneity_results)
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

## Comprehensive Exercises

The following exercises are designed to help you master two-way ANOVA concepts, from basic applications to advanced analyses.

### Exercise 1: Basic Two-Way ANOVA Analysis

**Objective**: Perform a complete two-way ANOVA analysis on the mtcars dataset.

**Task**: Analyze the effects of cylinder count (4, 6, 8) and transmission type (automatic, manual) on horsepower.

**Requirements**:
1. Create appropriate factor variables
2. Perform manual calculations for all components (SS, df, MS, F, p-values)
3. Use R's built-in `aov()` function
4. Calculate descriptive statistics for all cells
5. Create interaction plots and box plots
6. Calculate effect sizes (partial η², η², ω²)
7. Interpret results comprehensively

**Expected Learning Outcomes**:
- Understanding of two-way ANOVA mathematical foundations
- Ability to perform both manual and automated calculations
- Skills in creating informative visualizations
- Competence in effect size interpretation

**Hints**:
- Use `factor()` to create categorical variables
- Remember to check assumptions before interpretation
- Consider the practical significance of results

### Exercise 2: Interaction Effects Analysis

**Objective**: Create and analyze a dataset with significant interaction effects.

**Task**: Generate a simulated dataset with a 2×3 factorial design where:
- Factor A has 2 levels
- Factor B has 3 levels
- There is a significant interaction effect
- Sample size is 20 per cell

**Requirements**:
1. Generate data with known interaction effects
2. Perform two-way ANOVA
3. Conduct simple effects analysis
4. Create interaction plots
5. Perform post hoc tests for significant simple effects
6. Calculate and interpret effect sizes for all effects

**Expected Learning Outcomes**:
- Understanding of interaction effects
- Ability to perform simple effects analysis
- Skills in interpreting complex factorial designs
- Competence in post hoc testing for interactions

**Hints**:
- Use different means for different cells to create interactions
- Consider using `rnorm()` with different parameters for each cell
- Remember that interactions can be ordinal or disordinal

### Exercise 3: Comprehensive Assumption Checking

**Objective**: Develop expertise in assumption checking and alternative methods.

**Task**: Using the mtcars dataset, perform comprehensive assumption checking and recommend appropriate analyses.

**Requirements**:
1. Test normality using multiple methods (Shapiro-Wilk, KS, Anderson-Darling, Lilliefors)
2. Test homogeneity of variance using multiple methods (Levene's, Bartlett's, Fligner-Killeen, Brown-Forsythe)
3. Create diagnostic plots (Q-Q plots, residual plots, variance plots)
4. If assumptions are violated, perform appropriate alternatives:
   - Data transformations (log, square root, etc.)
   - Nonparametric alternatives
   - Robust methods
5. Compare results between parametric and nonparametric approaches

**Expected Learning Outcomes**:
- Comprehensive understanding of ANOVA assumptions
- Ability to choose appropriate alternative methods
- Skills in diagnostic plotting and interpretation
- Competence in data transformation

**Hints**:
- Use `shapiro.test()`, `ks.test()`, `ad.test()`, `lillie.test()`
- For homogeneity: `leveneTest()`, `bartlett.test()`, `fligner.test()`
- Consider `kruskal.test()` for nonparametric alternatives

### Exercise 4: Advanced Effect Size Analysis

**Objective**: Master effect size calculations and interpretation.

**Task**: Perform comprehensive effect size analysis for a two-way ANOVA.

**Requirements**:
1. Calculate all effect size measures (partial η², η², ω², Cohen's f)
2. Compute bootstrap confidence intervals for effect sizes
3. Create effect size comparison plots
4. Assess practical significance
5. Perform power analysis
6. Compare effect sizes across different datasets

**Expected Learning Outcomes**:
- Deep understanding of effect size measures
- Ability to interpret practical significance
- Skills in power analysis
- Competence in bootstrap methods

**Hints**:
- Use the `boot` package for bootstrap confidence intervals
- Consider using `pwr` package for power analysis
- Remember that effect sizes are sample size independent

### Exercise 5: Real-World Application

**Objective**: Apply two-way ANOVA to a real-world research scenario.

**Task**: Design and analyze a research study using two-way ANOVA.

**Scenario Options**:
- **Educational Research**: Teaching method × Class size on student performance
- **Clinical Trial**: Drug treatment × Dosage level on patient outcomes
- **Manufacturing**: Machine type × Shift on product quality
- **Marketing**: Advertisement type × Target audience on purchase behavior

**Requirements**:
1. Design the study with appropriate sample sizes
2. Generate realistic data based on the scenario
3. Perform complete two-way ANOVA analysis
4. Create publication-ready visualizations
5. Write a comprehensive results section
6. Discuss practical implications

**Expected Learning Outcomes**:
- Ability to design factorial research studies
- Skills in data generation and analysis
- Competence in results presentation
- Understanding of practical applications

**Hints**:
- Consider effect sizes when determining sample sizes
- Use realistic means and standard deviations
- Include interaction effects that make sense for the scenario

### Exercise 6: Advanced Topics

**Objective**: Explore advanced two-way ANOVA topics.

**Task**: Investigate one or more advanced topics:

**Options**:
1. **Unbalanced Designs**: Analyze data with unequal cell sizes
2. **Mixed Models**: Use `lme4` package for mixed-effects ANOVA
3. **Robust ANOVA**: Implement robust alternatives using `WRS2` package
4. **Bootstrap ANOVA**: Perform bootstrap-based ANOVA
5. **Bayesian ANOVA**: Use `BayesFactor` package for Bayesian ANOVA

**Requirements**:
1. Implement the chosen advanced method
2. Compare results with standard two-way ANOVA
3. Discuss advantages and limitations
4. Create appropriate visualizations
5. Provide recommendations for when to use each method

**Expected Learning Outcomes**:
- Understanding of advanced ANOVA methods
- Ability to choose appropriate methods for different situations
- Skills in implementing specialized packages
- Competence in method comparison

**Hints**:
- For unbalanced designs, consider Type I vs Type III SS
- For mixed models, understand random vs fixed effects
- For robust methods, understand when they're most useful

### Exercise Solutions Framework

For each exercise, follow this systematic approach:

1. **Data Preparation**:
   - Check data structure and quality
   - Create appropriate factor variables
   - Handle missing values appropriately

2. **Exploratory Analysis**:
   - Calculate descriptive statistics
   - Create initial visualizations
   - Identify potential issues

3. **Assumption Checking**:
   - Test normality and homogeneity
   - Create diagnostic plots
   - Decide on appropriate analysis method

4. **Statistical Analysis**:
   - Perform the chosen analysis
   - Calculate effect sizes
   - Conduct post hoc tests if needed

5. **Results Interpretation**:
   - Interpret statistical significance
   - Assess practical significance
   - Consider limitations and assumptions

6. **Reporting**:
   - Create publication-ready tables and figures
   - Write clear interpretations
   - Provide recommendations

**Learning Progression**:
- Start with Exercise 1 to build foundational skills
- Progress through exercises 2-4 to develop advanced competencies
- Complete Exercise 5 to apply skills to real-world scenarios
- Attempt Exercise 6 to explore cutting-edge methods

**Assessment Criteria**:
- Correct implementation of statistical methods
- Appropriate interpretation of results
- Quality of visualizations and reporting
- Understanding of underlying concepts
- Ability to make practical recommendations

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