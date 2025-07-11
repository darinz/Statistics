# Analysis of Variance (ANOVA)

## Overview

Analysis of Variance (ANOVA) is a statistical method used to compare means across multiple groups. It extends the t-test to handle more than two groups and can test for interactions between factors.

## One-Way ANOVA

### Basic One-Way ANOVA

```r
# Load sample data
data(mtcars)

# One-way ANOVA: Compare MPG across different cylinder types
one_way_anova <- aov(mpg ~ factor(cyl), data = mtcars)
summary(one_way_anova)

# Extract results
f_statistic <- summary(one_way_anova)[[1]]$"F value"[1]
p_value <- summary(one_way_anova)[[1]]$"Pr(>F)"[1]
df_between <- summary(one_way_anova)[[1]]$"Df"[1]
df_within <- summary(one_way_anova)[[1]]$"Df"[2]

cat("F-statistic:", f_statistic, "\n")
cat("p-value:", p_value, "\n")
cat("Degrees of freedom (between):", df_between, "\n")
cat("Degrees of freedom (within):", df_within, "\n")
```

### Descriptive Statistics by Group

```r
# Calculate descriptive statistics by cylinder type
library(dplyr)

group_stats <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    n = n(),
    mean_mpg = mean(mpg),
    sd_mpg = sd(mpg),
    se_mpg = sd_mpg / sqrt(n)
  )

print(group_stats)

# Create box plot
boxplot(mpg ~ cyl, data = mtcars, 
        main = "MPG by Number of Cylinders",
        xlab = "Number of Cylinders", 
        ylab = "Miles per Gallon",
        col = c("lightblue", "lightgreen", "lightcoral"))
```

### Assumptions Check

```r
# Check ANOVA assumptions
par(mfrow = c(2, 2))

# 1. Normality within each group
cyl_4 <- mtcars$mpg[mtcars$cyl == 4]
cyl_6 <- mtcars$mpg[mtcars$cyl == 6]
cyl_8 <- mtcars$mpg[mtcars$cyl == 8]

qqnorm(cyl_4, main = "Q-Q Plot: 4 Cylinders")
qqline(cyl_4)

qqnorm(cyl_6, main = "Q-Q Plot: 6 Cylinders")
qqline(cyl_6)

qqnorm(cyl_8, main = "Q-Q Plot: 8 Cylinders")
qqline(cyl_8)

# 2. Homogeneity of variances
plot(one_way_anova, which = 1, main = "Residuals vs Fitted")

par(mfrow = c(1, 1))

# Levene's test for homogeneity of variances
library(car)
levene_test <- leveneTest(mpg ~ factor(cyl), data = mtcars)
print(levene_test)

# Shapiro-Wilk test for normality of residuals
shapiro_test <- shapiro.test(residuals(one_way_anova))
print(shapiro_test)
```

### Post Hoc Tests

```r
# Tukey's HSD test
tukey_result <- TukeyHSD(one_way_anova)
print(tukey_result)

# Plot Tukey results
plot(tukey_result)

# Pairwise t-tests with Bonferroni correction
pairwise_tests <- pairwise.t.test(mtcars$mpg, mtcars$cyl, p.adjust.method = "bonferroni")
print(pairwise_tests)

# Effect size (eta-squared)
ss_between <- summary(one_way_anova)[[1]]$"Sum Sq"[1]
ss_total <- sum(summary(one_way_anova)[[1]]$"Sum Sq")
eta_squared <- ss_between / ss_total

cat("Effect size (eta-squared):", eta_squared, "\n")
```

## Two-Way ANOVA

### Basic Two-Way ANOVA

```r
# Two-way ANOVA: MPG by cylinders and transmission type
two_way_anova <- aov(mpg ~ factor(cyl) * factor(am), data = mtcars)
summary(two_way_anova)

# Extract results
anova_table <- summary(two_way_anova)[[1]]
print(anova_table)

# Calculate effect sizes
ss_cyl <- anova_table$"Sum Sq"[1]
ss_am <- anova_table$"Sum Sq"[2]
ss_interaction <- anova_table$"Sum Sq"[3]
ss_total <- sum(anova_table$"Sum Sq")

eta_squared_cyl <- ss_cyl / ss_total
eta_squared_am <- ss_am / ss_total
eta_squared_interaction <- ss_interaction / ss_total

cat("Effect size for cylinders:", eta_squared_cyl, "\n")
cat("Effect size for transmission:", eta_squared_am, "\n")
cat("Effect size for interaction:", eta_squared_interaction, "\n")
```

### Interaction Plot

```r
# Create interaction plot
interaction.plot(mtcars$cyl, mtcars$am, mtcars$mpg,
                main = "Interaction Plot: MPG by Cylinders and Transmission",
                xlab = "Number of Cylinders",
                ylab = "Mean MPG",
                trace.label = "Transmission",
                col = c("blue", "red"),
                lwd = 2)

# Alternative using ggplot2
library(ggplot2)
ggplot(mtcars, aes(x = factor(cyl), y = mpg, color = factor(am))) +
  stat_summary(fun = mean, geom = "line", aes(group = factor(am)), size = 1) +
  stat_summary(fun = mean, geom = "point", size = 3) +
  labs(title = "Interaction Plot: MPG by Cylinders and Transmission",
       x = "Number of Cylinders",
       y = "Mean MPG",
       color = "Transmission") +
  theme_minimal()
```

### Descriptive Statistics for Two-Way ANOVA

```r
# Calculate descriptive statistics for each combination
two_way_stats <- mtcars %>%
  group_by(cyl, am) %>%
  summarise(
    n = n(),
    mean_mpg = mean(mpg),
    sd_mpg = sd(mpg),
    se_mpg = sd_mpg / sqrt(n)
  ) %>%
  ungroup()

print(two_way_stats)

# Create heatmap of means
library(reshape2)
means_matrix <- dcast(two_way_stats, cyl ~ am, value.var = "mean_mpg")
print(means_matrix)
```

## Repeated Measures ANOVA

### Simulating Repeated Measures Data

```r
# Simulate repeated measures data
set.seed(123)
n_subjects <- 20
n_timepoints <- 3

# Create data frame
subject_id <- rep(1:n_subjects, each = n_timepoints)
time_point <- rep(c("Baseline", "Week 4", "Week 8"), times = n_subjects)

# Simulate scores with treatment effect
baseline <- rnorm(n_subjects, mean = 50, sd = 10)
week4 <- baseline + rnorm(n_subjects, mean = 5, sd = 3)
week8 <- baseline + rnorm(n_subjects, mean = 10, sd = 3)

scores <- c(baseline, week4, week8)

repeated_data <- data.frame(
  subject_id = subject_id,
  time_point = time_point,
  score = scores
)

# Convert to wide format for analysis
repeated_wide <- reshape(repeated_data, 
                        idvar = "subject_id", 
                        timevar = "time_point", 
                        direction = "wide")

print(head(repeated_wide))
```

### Repeated Measures ANOVA Analysis

```r
# Perform repeated measures ANOVA
library(ez)

repeated_anova <- ezANOVA(data = repeated_data,
                          dv = score,
                          wid = subject_id,
                          within = time_point,
                          detailed = TRUE)

print(repeated_anova)

# Alternative using aov() with Error term
repeated_aov <- aov(score ~ time_point + Error(subject_id/time_point), 
                    data = repeated_data)
summary(repeated_aov)
```

### Post Hoc Tests for Repeated Measures

```r
# Pairwise comparisons with Bonferroni correction
pairwise_repeated <- pairwise.t.test(repeated_data$score, 
                                    repeated_data$time_point,
                                    p.adjust.method = "bonferroni",
                                    paired = TRUE)
print(pairwise_repeated)

# Calculate effect size
ss_time <- repeated_anova$ANOVA$SSn[1]
ss_error <- repeated_anova$ANOVA$SSd[1]
partial_eta_squared <- ss_time / (ss_time + ss_error)

cat("Partial eta-squared:", partial_eta_squared, "\n")
```

## Mixed Design ANOVA

### Simulating Mixed Design Data

```r
# Simulate mixed design data (between-subject factor: treatment group)
set.seed(123)
n_per_group <- 15
n_timepoints <- 3

# Create data frame
subject_id <- rep(1:(2 * n_per_group), each = n_timepoints)
group <- rep(c("Control", "Treatment"), each = n_per_group * n_timepoints)
time_point <- rep(c("Baseline", "Week 4", "Week 8"), times = 2 * n_per_group)

# Simulate scores
control_baseline <- rnorm(n_per_group, mean = 50, sd = 10)
control_week4 <- control_baseline + rnorm(n_per_group, mean = 2, sd = 3)
control_week8 <- control_baseline + rnorm(n_per_group, mean = 3, sd = 3)

treatment_baseline <- rnorm(n_per_group, mean = 50, sd = 10)
treatment_week4 <- treatment_baseline + rnorm(n_per_group, mean = 8, sd = 3)
treatment_week8 <- treatment_baseline + rnorm(n_per_group, mean = 15, sd = 3)

scores <- c(control_baseline, control_week4, control_week8,
            treatment_baseline, treatment_week4, treatment_week8)

mixed_data <- data.frame(
  subject_id = subject_id,
  group = group,
  time_point = time_point,
  score = scores
)

print(head(mixed_data))
```

### Mixed Design ANOVA Analysis

```r
# Perform mixed design ANOVA
mixed_anova <- ezANOVA(data = mixed_data,
                       dv = score,
                       wid = subject_id,
                       between = group,
                       within = time_point,
                       detailed = TRUE)

print(mixed_anova)

# Alternative using aov()
mixed_aov <- aov(score ~ group * time_point + Error(subject_id/time_point), 
                 data = mixed_data)
summary(mixed_aov)
```

### Interaction Analysis

```r
# Create interaction plot
ggplot(mixed_data, aes(x = time_point, y = score, color = group, group = group)) +
  stat_summary(fun = mean, geom = "line", size = 1) +
  stat_summary(fun = mean, geom = "point", size = 3) +
  labs(title = "Mixed Design ANOVA: Score by Group and Time",
       x = "Time Point",
       y = "Mean Score",
       color = "Group") +
  theme_minimal()

# Simple effects analysis
library(emmeans)

# Test simple effects of time within each group
simple_effects <- emmeans(mixed_aov, ~ time_point | group)
pairs(simple_effects, adjust = "bonferroni")

# Test simple effects of group at each time point
simple_effects_group <- emmeans(mixed_aov, ~ group | time_point)
pairs(simple_effects_group, adjust = "bonferroni")
```

## Nonparametric Alternatives

### Kruskal-Wallis Test (One-Way)

```r
# Kruskal-Wallis test for nonparametric one-way ANOVA
kruskal_result <- kruskal.test(mpg ~ factor(cyl), data = mtcars)
print(kruskal_result)

# Post hoc tests for Kruskal-Wallis
library(dunn.test)
dunn_result <- dunn.test(mtcars$mpg, mtcars$cyl, method = "bonferroni")
print(dunn_result)
```

### Friedman Test (Repeated Measures)

```r
# Friedman test for nonparametric repeated measures
friedman_result <- friedman.test(score ~ time_point | subject_id, data = repeated_data)
print(friedman_result)

# Post hoc tests for Friedman
library(PMCMRplus)
friedman_posthoc <- friedmanTest(score ~ time_point | subject_id, data = repeated_data)
print(friedman_posthoc)
```

## Effect Size and Power Analysis

### Effect Size Calculations

```r
# Function to calculate effect sizes for ANOVA
calculate_effect_sizes <- function(anova_result) {
  anova_table <- summary(anova_result)[[1]]
  
  # Calculate eta-squared
  ss_effects <- anova_table$"Sum Sq"[-nrow(anova_table)]
  ss_total <- sum(anova_table$"Sum Sq")
  eta_squared <- ss_effects / ss_total
  
  # Calculate partial eta-squared
  ss_error <- anova_table$"Sum Sq"[nrow(anova_table)]
  partial_eta_squared <- ss_effects / (ss_effects + ss_error)
  
  # Calculate Cohen's f
  cohens_f <- sqrt(partial_eta_squared / (1 - partial_eta_squared))
  
  return(data.frame(
    Effect = rownames(anova_table)[-nrow(anova_table)],
    Eta_Squared = eta_squared,
    Partial_Eta_Squared = partial_eta_squared,
    Cohens_f = cohens_f
  ))
}

# Apply to one-way ANOVA
effect_sizes <- calculate_effect_sizes(one_way_anova)
print(effect_sizes)
```

### Power Analysis

```r
# Power analysis for one-way ANOVA
library(pwr)

# Calculate power for given effect size and sample size
effect_size <- 0.25  # Medium effect
n_per_group <- 10
n_groups <- 3

power_result <- pwr.anova.test(k = n_groups, 
                               n = n_per_group, 
                               f = effect_size, 
                               sig.level = 0.05)
print(power_result)

# Calculate required sample size for desired power
required_sample <- pwr.anova.test(k = n_groups, 
                                  f = effect_size, 
                                  sig.level = 0.05, 
                                  power = 0.80)
print(required_sample)
```

## Practical Examples

### Example 1: Educational Research

```r
# Simulate educational data
set.seed(123)
n_students <- 60
teaching_methods <- c("Traditional", "Interactive", "Online")

# Simulate test scores
traditional <- rnorm(20, mean = 75, sd = 10)
interactive <- rnorm(20, mean = 82, sd = 10)
online <- rnorm(20, mean = 78, sd = 10)

scores <- c(traditional, interactive, online)
method <- rep(teaching_methods, each = 20)

education_data <- data.frame(method = method, score = scores)

# Perform ANOVA
edu_anova <- aov(score ~ method, data = education_data)
summary(edu_anova)

# Post hoc tests
edu_tukey <- TukeyHSD(edu_anova)
print(edu_tukey)

# Effect size
edu_effect <- calculate_effect_sizes(edu_anova)
print(edu_effect)
```

### Example 2: Clinical Trial

```r
# Simulate clinical trial data
set.seed(123)
n_patients <- 30
time_points <- c("Baseline", "Week 2", "Week 4", "Week 6")

# Simulate pain scores
baseline <- rnorm(n_patients, mean = 7, sd = 1.5)
week2 <- baseline - rnorm(n_patients, mean = 1, sd = 0.5)
week4 <- baseline - rnorm(n_patients, mean = 2, sd = 0.5)
week6 <- baseline - rnorm(n_patients, mean = 2.5, sd = 0.5)

pain_scores <- c(baseline, week2, week4, week6)
patient_id <- rep(1:n_patients, times = 4)
time <- rep(time_points, each = n_patients)

clinical_data <- data.frame(
  patient_id = patient_id,
  time = time,
  pain_score = pain_scores
)

# Perform repeated measures ANOVA
clinical_anova <- ezANOVA(data = clinical_data,
                          dv = pain_score,
                          wid = patient_id,
                          within = time,
                          detailed = TRUE)

print(clinical_anova)

# Post hoc tests
clinical_pairwise <- pairwise.t.test(clinical_data$pain_score,
                                    clinical_data$time,
                                    p.adjust.method = "bonferroni",
                                    paired = TRUE)
print(clinical_pairwise)
```

## Best Practices

### Assumption Checking

```r
# Comprehensive assumption checking function
check_anova_assumptions <- function(anova_result, data, group_var, dv_var) {
  cat("=== ANOVA ASSUMPTION CHECKS ===\n\n")
  
  # 1. Normality of residuals
  residuals <- residuals(anova_result)
  shapiro_test <- shapiro.test(residuals)
  cat("1. Normality of Residuals (Shapiro-Wilk):\n")
  cat("   W =", shapiro_test$statistic, ", p =", shapiro_test$p.value, "\n")
  cat("   Interpretation:", ifelse(shapiro_test$p.value > 0.05, "Normal", "Not normal"), "\n\n")
  
  # 2. Homogeneity of variances
  levene_test <- leveneTest(as.formula(paste(dv_var, "~", group_var)), data = data)
  cat("2. Homogeneity of Variances (Levene's Test):\n")
  cat("   F =", levene_test$`F value`[1], ", p =", levene_test$`Pr(>F)`[1], "\n")
  cat("   Interpretation:", ifelse(levene_test$`Pr(>F)`[1] > 0.05, "Equal variances", "Unequal variances"), "\n\n")
  
  # 3. Independence (assumed if design is correct)
  cat("3. Independence:\n")
  cat("   Checked through study design\n\n")
  
  # 4. Visual diagnostics
  par(mfrow = c(2, 2))
  plot(anova_result)
  par(mfrow = c(1, 1))
}

# Apply to one-way ANOVA
check_anova_assumptions(one_way_anova, mtcars, "cyl", "mpg")
```

### Reporting Guidelines

```r
# Function to generate APA-style report
generate_anova_report <- function(anova_result, effect_name = "effect") {
  anova_table <- summary(anova_result)[[1]]
  
  cat("=== APA-STYLE ANOVA REPORT ===\n\n")
  
  # Main effect
  f_stat <- anova_table$"F value"[1]
  p_value <- anova_table$"Pr(>F)"[1]
  df_between <- anova_table$"Df"[1]
  df_within <- anova_table$"Df"[2]
  
  cat("A one-way analysis of variance was conducted to compare", effect_name, "\n")
  cat("across groups. There was a", ifelse(p_value < 0.001, "highly significant",
                                           ifelse(p_value < 0.01, "significant",
                                                  ifelse(p_value < 0.05, "marginally significant", "non-significant"))), "\n")
  cat("effect, F(", df_between, ",", df_within, ") =", round(f_stat, 3), ", p =", 
      ifelse(p_value < 0.001, "< .001", round(p_value, 3)), ".\n\n")
  
  # Effect size
  ss_between <- anova_table$"Sum Sq"[1]
  ss_total <- sum(anova_table$"Sum Sq")
  eta_squared <- ss_between / ss_total
  
  cat("The effect size (η²) was", round(eta_squared, 3), ", indicating a", 
      ifelse(eta_squared < 0.01, "small",
             ifelse(eta_squared < 0.06, "medium", "large")), "effect.\n\n")
}

# Apply to one-way ANOVA
generate_anova_report(one_way_anova, "MPG across cylinder types")
```

## Exercises

### Exercise 1: One-Way ANOVA
Perform a one-way ANOVA to compare MPG across different transmission types (automatic vs manual) and interpret the results.

### Exercise 2: Two-Way ANOVA
Conduct a two-way ANOVA to examine the effects of cylinder type and transmission type on MPG, including the interaction effect.

### Exercise 3: Repeated Measures ANOVA
Design a repeated measures study and analyze the data using appropriate ANOVA techniques.

### Exercise 4: Nonparametric Alternatives
Compare the results of parametric ANOVA with nonparametric alternatives for the same dataset.

### Exercise 5: Effect Size and Power
Calculate effect sizes for your ANOVA results and perform power analysis to determine required sample sizes.

## Next Steps

In the next chapter, we'll learn about correlation analysis, which examines relationships between variables.

---

**Key Takeaways:**
- ANOVA extends t-tests to multiple groups
- Check assumptions before interpreting results
- Use appropriate post hoc tests for multiple comparisons
- Consider effect sizes alongside p-values
- Nonparametric alternatives exist for violated assumptions
- Repeated measures ANOVA handles within-subject designs
- Mixed designs combine between- and within-subject factors 