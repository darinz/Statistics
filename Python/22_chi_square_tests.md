# Chi-Square Tests

## Introduction

Chi-square tests are fundamental statistical methods for analyzing categorical data. They are nonparametric tests that examine relationships between categorical variables, test for independence, and assess goodness of fit. Chi-square tests are widely used in research across disciplines including psychology, medicine, social sciences, and business.

**Key Applications:**
- Testing independence between categorical variables
- Comparing observed frequencies to expected frequencies
- Assessing homogeneity across groups
- Analyzing survey responses and contingency tables
- Quality control and process monitoring

**When to Use Chi-Square Tests:**
- Data are categorical (nominal or ordinal)
- Observations are independent
- Expected frequencies meet minimum requirements
- Testing for relationships or differences in proportions

## Mathematical Foundations

### Chi-Square Distribution

The chi-square distribution is a continuous probability distribution with $`k`$ degrees of freedom. It is the distribution of the sum of squares of $`k`$ independent standard normal random variables.

**Probability Density Function:**
```math
f(x; k) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{k/2-1} e^{-x/2}, \quad x > 0
```

**Properties:**
- Always positive and right-skewed
- Mean = $`k`$ (degrees of freedom)
- Variance = $`2k`$
- As $`k`$ increases, approaches normal distribution

### Chi-Square Test Statistic

The general form of the chi-square test statistic is:
```math
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
```

where:
- $`O_i`$ = observed frequency for category $`i`$
- $`E_i`$ = expected frequency for category $`i`$
- $`k`$ = number of categories

Under the null hypothesis, this statistic follows a chi-square distribution with appropriate degrees of freedom.

## Types of Chi-Square Tests

### 1. Chi-Square Goodness of Fit Test

**Purpose:** Test whether observed frequencies match expected frequencies from a theoretical distribution.

**Null Hypothesis:** $`H_0: O_i = E_i`$ for all categories
**Alternative Hypothesis:** $`H_1: O_i \neq E_i`$ for at least one category

**Test Statistic:**
```math
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
```

**Degrees of Freedom:** $`df = k - 1 - p`$ where $`p`$ is the number of parameters estimated from the data.

**R Implementation:**
```r
# Basic goodness of fit test
observed <- c(45, 52, 38, 65)
expected <- c(50, 50, 50, 50)  # Equal expected frequencies

# Perform chi-square goodness of fit test
chi_square_test <- chisq.test(observed, p = rep(1/length(observed), length(observed)))

# Extract and display results
cat("Chi-Square Goodness of Fit Test Results:\n")
cat("Chi-square statistic:", round(chi_square_test$statistic, 3), "\n")
cat("Degrees of freedom:", chi_square_test$parameter, "\n")
cat("p-value:", round(chi_square_test$p.value, 4), "\n")
cat("Expected frequencies:", expected, "\n")
cat("Observed frequencies:", observed, "\n")

# Manual calculation for understanding
manual_chi_square <- function(observed, expected) {
  chi_square <- sum((observed - expected)^2 / expected)
  df <- length(observed) - 1
  p_value <- 1 - pchisq(chi_square, df)
  residuals <- (observed - expected) / sqrt(expected)
  
  return(list(
    chi_square = chi_square,
    df = df,
    p_value = p_value,
    residuals = residuals
  ))
}

manual_result <- manual_chi_square(observed, expected)
cat("\nManual Calculation:\n")
cat("Chi-square statistic:", round(manual_result$chi_square, 3), "\n")
cat("Standardized residuals:", round(manual_result$residuals, 3), "\n")
```

### 2. Chi-Square Test of Independence

**Purpose:** Test whether two categorical variables are independent (not associated).

**Null Hypothesis:** $`H_0: \text{Variables are independent}`$
**Alternative Hypothesis:** $`H_1: \text{Variables are dependent}`$

**Test Statistic:**
```math
\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
```

where $`E_{ij} = \frac{R_i \times C_j}{N}`$ is the expected frequency for cell $`(i,j)`$.

**Degrees of Freedom:** $`df = (r-1) \times (c-1)`$ where $`r`$ and $`c`$ are the number of rows and columns.

**R Implementation:**
```r
# Create contingency table
contingency_table <- matrix(c(45, 35, 25, 55, 40, 30, 30, 25, 15), nrow = 3, ncol = 3)
rownames(contingency_table) <- c("Low", "Medium", "High")
colnames(contingency_table) <- c("Group A", "Group B", "Group C")

print("Contingency Table:")
print(contingency_table)

# Perform chi-square test of independence
independence_test <- chisq.test(contingency_table)

# Display results
cat("Chi-Square Test of Independence Results:\n")
cat("Chi-square statistic:", round(independence_test$statistic, 3), "\n")
cat("Degrees of freedom:", independence_test$parameter, "\n")
cat("p-value:", round(independence_test$p.value, 4), "\n")

# Expected frequencies and residuals
cat("\nExpected Frequencies:\n")
print(round(independence_test$expected, 2))

cat("\nStandardized Residuals:\n")
print(round(independence_test$residuals, 3))

# Identify significant cells
significant_cells <- which(abs(independence_test$residuals) > 2, arr.ind = TRUE)
if (nrow(significant_cells) > 0) {
  cat("\nSignificant cells (|residual| > 2):\n")
  for (i in 1:nrow(significant_cells)) {
    row_idx <- significant_cells[i, 1]
    col_idx <- significant_cells[i, 2]
    row_name <- rownames(contingency_table)[row_idx]
    col_name <- colnames(contingency_table)[col_idx]
    residual <- independence_test$residuals[row_idx, col_idx]
    cat(row_name, "-", col_name, ":", round(residual, 3), "\n")
  }
}
```

### 3. Chi-Square Test of Homogeneity

**Purpose:** Test whether the distribution of a categorical variable is the same across different groups.

**Null Hypothesis:** $`H_0: \text{Proportions are equal across groups}`$
**Alternative Hypothesis:** $`H_1: \text{Proportions differ across groups}`$

**Test Statistic:** Same as independence test, but interpretation differs.

**R Implementation:**
```r
# Create homogeneity test data
homogeneity_data <- matrix(c(20, 30, 25, 15, 25, 20, 10, 15, 10), nrow = 3, ncol = 3)
rownames(homogeneity_data) <- c("Treatment A", "Treatment B", "Treatment C")
colnames(homogeneity_data) <- c("Success", "Partial", "Failure")

print("Homogeneity Test Data:")
print(homogeneity_data)

# Perform chi-square test of homogeneity
homogeneity_test <- chisq.test(homogeneity_data)

cat("Chi-Square Test of Homogeneity Results:\n")
cat("Chi-square statistic:", round(homogeneity_test$statistic, 3), "\n")
cat("Degrees of freedom:", homogeneity_test$parameter, "\n")
cat("p-value:", round(homogeneity_test$p.value, 4), "\n")
```

## Effect Size Measures

### Cramér's V

**Formula:**
```math
V = \sqrt{\frac{\chi^2}{N(k-1)}}
```

where $`k = \min(r, c)`$ is the smaller of the number of rows or columns.

**Interpretation:**
- $`V < 0.1`$: Negligible effect
- $`0.1 \leq V < 0.3`$: Small effect
- $`0.3 \leq V < 0.5`$: Medium effect
- $`V \geq 0.5`$: Large effect

### Phi Coefficient (2x2 tables)

**Formula:**
```math
\phi = \sqrt{\frac{\chi^2}{N}}
```

### Contingency Coefficient

**Formula:**
```math
C = \sqrt{\frac{\chi^2}{\chi^2 + N}}
```

**R Implementation:**
```r
# Function to calculate effect sizes
calculate_chi_square_effect_sizes <- function(chi_square_result, n) {
  # Cramer's V
  df <- chi_square_result$parameter
  min_dim <- min(nrow(chi_square_result$observed), ncol(chi_square_result$observed))
  cramers_v <- sqrt(chi_square_result$statistic / (n * (min_dim - 1)))
  
  # Phi coefficient (for 2x2 tables)
  if (nrow(chi_square_result$observed) == 2 && ncol(chi_square_result$observed) == 2) {
    phi <- sqrt(chi_square_result$statistic / n)
  } else {
    phi <- NA
  }
  
  # Contingency coefficient
  contingency_coef <- sqrt(chi_square_result$statistic / (chi_square_result$statistic + n))
  
  return(list(
    cramers_v = cramers_v,
    phi = phi,
    contingency_coef = contingency_coef
  ))
}

# Apply to independence test
effect_sizes <- calculate_chi_square_effect_sizes(independence_test, sum(contingency_table))

cat("Effect Size Analysis:\n")
cat("Cramer's V:", round(effect_sizes$cramers_v, 3), "\n")
if (!is.na(effect_sizes$phi)) {
  cat("Phi coefficient:", round(effect_sizes$phi, 3), "\n")
}
cat("Contingency coefficient:", round(effect_sizes$contingency_coef, 3), "\n")

# Interpretation function
interpret_cramers_v <- function(v) {
  if (v < 0.1) {
    return("Negligible effect")
  } else if (v < 0.3) {
    return("Small effect")
  } else if (v < 0.5) {
    return("Medium effect")
  } else {
    return("Large effect")
  }
}

cat("Effect size interpretation:", interpret_cramers_v(effect_sizes$cramers_v), "\n")
```

## Assumptions and Violations

### Key Assumptions

1. **Independence:** Observations must be independent
2. **Expected Frequencies:** All expected frequencies should be ≥ 5
3. **Random Sampling:** Data should come from random sampling
4. **Mutually Exclusive Categories:** Each observation belongs to exactly one category

### Checking Assumptions

```r
# Function to check chi-square assumptions
check_chi_square_assumptions <- function(contingency_table) {
  cat("=== CHI-SQUARE ASSUMPTIONS CHECK ===\n")
  
  # Calculate expected frequencies
  chi_square_result <- chisq.test(contingency_table)
  expected_freq <- chi_square_result$expected
  
  cat("Expected frequencies:\n")
  print(round(expected_freq, 2))
  
  # Check minimum expected frequency
  min_expected <- min(expected_freq)
  cat("Minimum expected frequency:", round(min_expected, 2), "\n")
  
  # Check for cells with expected frequency < 5
  low_expected_cells <- which(expected_freq < 5, arr.ind = TRUE)
  
  if (nrow(low_expected_cells) > 0) {
    cat("WARNING: Cells with expected frequency < 5:\n")
    for (i in 1:nrow(low_expected_cells)) {
      row_idx <- low_expected_cells[i, 1]
      col_idx <- low_expected_cells[i, 2]
      row_name <- rownames(contingency_table)[row_idx]
      col_name <- colnames(contingency_table)[col_idx]
      expected_val <- expected_freq[row_idx, col_idx]
      cat("  ", row_name, "-", col_name, ":", round(expected_val, 2), "\n")
    }
    cat("Consider using Fisher's exact test or combining categories\n")
  } else {
    cat("All expected frequencies ≥ 5. Chi-square test is appropriate.\n")
  }
  
  # Check for independence
  cat("\nIndependence assumption: Data should be from independent observations.\n")
  
  return(list(
    expected_frequencies = expected_freq,
    min_expected = min_expected,
    low_expected_cells = low_expected_cells
  ))
}

# Check assumptions
assumption_results <- check_chi_square_assumptions(contingency_table)
```

## Alternative Tests

### Fisher's Exact Test

**When to Use:**
- Small expected frequencies (< 5)
- 2x2 contingency tables
- Exact p-values needed

**R Implementation:**
```r
# Fisher's exact test for small expected frequencies
fisher_test <- fisher.test(contingency_table)

cat("Fisher's Exact Test Results:\n")
cat("p-value:", round(fisher_test$p.value, 4), "\n")
cat("Odds ratio:", round(fisher_test$estimate, 3), "\n")
cat("95% Confidence interval:", round(fisher_test$conf.int, 3), "\n")

# Compare with chi-square results
cat("\nComparison:\n")
cat("Chi-square p-value:", round(independence_test$p.value, 4), "\n")
cat("Fisher's exact p-value:", round(fisher_test$p.value, 4), "\n")
```

### Likelihood Ratio Test

**Formula:**
```math
G = 2 \sum_{i=1}^{r} \sum_{j=1}^{c} O_{ij} \ln\left(\frac{O_{ij}}{E_{ij}}\right)
```

**R Implementation:**
```r
# Likelihood ratio test
likelihood_ratio_test <- function(observed) {
  # Calculate expected frequencies under independence
  row_totals <- rowSums(observed)
  col_totals <- colSums(observed)
  total <- sum(observed)
  
  expected <- outer(row_totals, col_totals) / total
  
  # Calculate likelihood ratio statistic
  lr_statistic <- 2 * sum(observed * log(observed / expected))
  
  # Degrees of freedom
  df <- (nrow(observed) - 1) * (ncol(observed) - 1)
  
  # p-value
  p_value <- 1 - pchisq(lr_statistic, df)
  
  return(list(
    statistic = lr_statistic,
    df = df,
    p_value = p_value
  ))
}

# Apply likelihood ratio test
lr_result <- likelihood_ratio_test(contingency_table)

cat("Likelihood Ratio Test Results:\n")
cat("G-statistic:", round(lr_result$statistic, 3), "\n")
cat("Degrees of freedom:", lr_result$df, "\n")
cat("p-value:", round(lr_result$p_value, 4), "\n")
```

## Post Hoc Analysis

### Pairwise Chi-Square Tests

```r
# Function to perform pairwise chi-square tests
pairwise_chi_square <- function(contingency_table, alpha = 0.05) {
  n_rows <- nrow(contingency_table)
  n_cols <- ncol(contingency_table)
  
  # Calculate number of pairwise comparisons
  n_comparisons <- choose(n_rows, 2)
  
  # Bonferroni correction
  alpha_corrected <- alpha / n_comparisons
  
  results <- list()
  pair_count <- 1
  
  for (i in 1:(n_rows-1)) {
    for (j in (i+1):n_rows) {
      # Extract 2x2 subtable
      subtable <- contingency_table[c(i, j), ]
      
      # Perform chi-square test
      test_result <- chisq.test(subtable)
      
      results[[pair_count]] <- list(
        comparison = paste(rownames(contingency_table)[i], "vs", rownames(contingency_table)[j]),
        chi_square = test_result$statistic,
        p_value = test_result$p.value,
        significant = test_result$p.value < alpha_corrected
      )
      
      pair_count <- pair_count + 1
    }
  }
  
  return(results)
}

# Apply pairwise tests
pairwise_results <- pairwise_chi_square(contingency_table)

cat("Pairwise Chi-Square Tests (Bonferroni-corrected α = 0.017):\n")
for (result in pairwise_results) {
  cat(result$comparison, ":\n")
  cat("  Chi-square:", round(result$chi_square, 3), "\n")
  cat("  p-value:", round(result$p_value, 4), "\n")
  cat("  Significant:", result$significant, "\n\n")
}
```

## Power Analysis

### Power Analysis for Chi-Square

```r
library(pwr)

# Power analysis for chi-square test
power_analysis_chi_square <- function(n, w, df, alpha = 0.05) {
  # Calculate power
  power_result <- pwr.chisq.test(w = w, N = n, df = df, sig.level = alpha)
  
  # Calculate required sample size for 80% power
  sample_size_result <- pwr.chisq.test(w = w, df = df, sig.level = alpha, power = 0.8)
  
  return(list(
    power = power_result$power,
    required_n = ceiling(sample_size_result$N),
    effect_size = w,
    alpha = alpha
  ))
}

# Apply power analysis
# For 2x2 table, df = 1, w = 0.3 (medium effect)
power_result <- power_analysis_chi_square(n = 100, w = 0.3, df = 1)

cat("Power Analysis Results:\n")
cat("Current power:", round(power_result$power, 3), "\n")
cat("Required sample size for 80% power:", power_result$required_n, "\n")
```

## Practical Examples

### Example 1: Survey Analysis

```r
# Simulate survey data
set.seed(123)
n_responses <- 200

# Generate survey responses
age_group <- sample(c("18-25", "26-35", "36-45", "46+"), n_responses, replace = TRUE, 
                   prob = c(0.3, 0.35, 0.25, 0.1))
satisfaction <- sample(c("Very Satisfied", "Satisfied", "Neutral", "Dissatisfied"), 
                      n_responses, replace = TRUE, prob = c(0.4, 0.3, 0.2, 0.1))

# Create contingency table
survey_table <- table(age_group, satisfaction)
print("Survey Results:")
print(survey_table)

# Perform chi-square test
survey_test <- chisq.test(survey_table)

# Effect size
survey_effect <- calculate_chi_square_effect_sizes(survey_test, n_responses)
cat("Cramer's V:", round(survey_effect$cramers_v, 3), "\n")
cat("Interpretation:", interpret_cramers_v(survey_effect$cramers_v), "\n")
```

### Example 2: Clinical Trial

```r
# Simulate clinical trial data
set.seed(123)

# Create 2x2 contingency table
clinical_data <- matrix(c(45, 15, 30, 25), nrow = 2, ncol = 2)
rownames(clinical_data) <- c("Treatment", "Control")
colnames(clinical_data) <- c("Improved", "No Improvement")

print("Clinical Trial Results:")
print(clinical_data)

# Chi-square test
clinical_test <- chisq.test(clinical_data)

# Fisher's exact test
clinical_fisher <- fisher.test(clinical_data)

# Effect size
clinical_effect <- calculate_chi_square_effect_sizes(clinical_test, sum(clinical_data))
cat("Phi coefficient:", round(clinical_effect$phi, 3), "\n")
```

### Example 3: Quality Control

```r
# Simulate quality control data
set.seed(123)

# Create 3x3 contingency table
quality_data <- matrix(c(85, 10, 5, 70, 20, 10, 60, 25, 15), nrow = 3, ncol = 3)
rownames(quality_data) <- c("Machine A", "Machine B", "Machine C")
colnames(quality_data) <- c("Excellent", "Good", "Poor")

print("Quality Control Results:")
print(quality_data)

# Chi-square test of homogeneity
quality_test <- chisq.test(quality_data)

# Check assumptions
quality_assumptions <- check_chi_square_assumptions(quality_data)

# Effect size
quality_effect <- calculate_chi_square_effect_sizes(quality_test, sum(quality_data))
cat("Cramer's V:", round(quality_effect$cramers_v, 3), "\n")
```

## Best Practices

### Test Selection Guidelines

```r
# Function to help choose appropriate chi-square test
choose_chi_square_test <- function(contingency_table) {
  cat("=== CHI-SQUARE TEST SELECTION ===\n")
  
  # Check expected frequencies
  chi_square_result <- chisq.test(contingency_table)
  expected_freq <- chi_square_result$expected
  min_expected <- min(expected_freq)
  
  cat("Minimum expected frequency:", round(min_expected, 2), "\n")
  
  # Check table dimensions
  n_rows <- nrow(contingency_table)
  n_cols <- ncol(contingency_table)
  cat("Table dimensions:", n_rows, "x", n_cols, "\n")
  
  cat("\nRECOMMENDATIONS:\n")
  
  if (min_expected >= 5) {
    cat("- Use chi-square test of independence\n")
    cat("- All expected frequencies ≥ 5\n")
  } else if (min_expected >= 1 && n_rows == 2 && n_cols == 2) {
    cat("- Use Fisher's exact test\n")
    cat("- Small expected frequencies in 2x2 table\n")
  } else {
    cat("- Use Fisher's exact test or combine categories\n")
    cat("- Very small expected frequencies\n")
  }
  
  # Effect size calculation
  effect_sizes <- calculate_chi_square_effect_sizes(chi_square_result, sum(contingency_table))
  cat("- Effect size (Cramer's V):", round(effect_sizes$cramers_v, 3), "\n")
  cat("- Interpretation:", interpret_cramers_v(effect_sizes$cramers_v), "\n")
  
  return(list(
    min_expected = min_expected,
    table_dimensions = c(n_rows, n_cols),
    effect_sizes = effect_sizes
  ))
}

# Apply to contingency table
test_selection <- choose_chi_square_test(contingency_table)
```

### Reporting Guidelines

```r
# Function to generate comprehensive chi-square report
generate_chi_square_report <- function(chi_square_result, contingency_table, test_type = "independence") {
  cat("=== CHI-SQUARE TEST REPORT ===\n\n")
  
  cat("CONTINGENCY TABLE:\n")
  print(contingency_table)
  cat("\n")
  
  cat("TEST RESULTS:\n")
  cat("Test type:", test_type, "\n")
  cat("Chi-square statistic:", round(chi_square_result$statistic, 3), "\n")
  cat("Degrees of freedom:", chi_square_result$parameter, "\n")
  cat("p-value:", round(chi_square_result$p.value, 4), "\n")
  
  # Effect size
  effect_sizes <- calculate_chi_square_effect_sizes(chi_square_result, sum(contingency_table))
  cat("Cramer's V:", round(effect_sizes$cramers_v, 3), "\n")
  cat("Effect size interpretation:", interpret_cramers_v(effect_sizes$cramers_v), "\n\n")
  
  # Expected frequencies
  cat("EXPECTED FREQUENCIES:\n")
  print(round(chi_square_result$expected, 2))
  cat("\n")
  
  # Standardized residuals
  cat("STANDARDIZED RESIDUALS:\n")
  print(round(chi_square_result$residuals, 3))
  cat("\n")
  
  # Conclusion
  alpha <- 0.05
  if (chi_square_result$p.value < alpha) {
    cat("CONCLUSION:\n")
    cat("Reject the null hypothesis (p <", alpha, ")\n")
    if (test_type == "independence") {
      cat("There is a significant relationship between the variables\n")
    } else if (test_type == "homogeneity") {
      cat("The proportions are significantly different across groups\n")
    } else {
      cat("The observed frequencies differ significantly from expected\n")
    }
  } else {
    cat("CONCLUSION:\n")
    cat("Fail to reject the null hypothesis (p >=", alpha, ")\n")
    cat("There is insufficient evidence of a relationship\n")
  }
}

# Generate report
generate_chi_square_report(independence_test, contingency_table, "independence")
```

## Exercises

### Exercise 1: Goodness of Fit Test
- **Objective:** Test whether observed frequencies match expected frequencies in a categorical variable.
- **Data:** Create a dataset with observed frequencies and test against equal expected frequencies.
- **Hint:** Use `chisq.test()` with the `p` parameter for expected proportions.

### Exercise 2: Independence Test
- **Objective:** Analyze the relationship between two categorical variables using chi-square test of independence.
- **Data:** Create a contingency table and test for independence.
- **Hint:** Use `chisq.test()` on a matrix or table object.

### Exercise 3: Homogeneity Test
- **Objective:** Test whether proportions are the same across different groups.
- **Data:** Create data with multiple groups and test for homogeneity.
- **Hint:** Use `chisq.test()` and interpret as homogeneity test.

### Exercise 4: Assumption Checking
- **Objective:** Check chi-square assumptions and recommend appropriate alternatives when violated.
- **Data:** Create data with small expected frequencies.
- **Hint:** Use `check_chi_square_assumptions()` and `fisher.test()`.

### Exercise 5: Effect Size Analysis
- **Objective:** Calculate and interpret different effect size measures for chi-square tests.
- **Data:** Use any contingency table from previous exercises.
- **Hint:** Use `calculate_chi_square_effect_sizes()` function.

### Exercise 6: Post Hoc Analysis
- **Objective:** Perform pairwise chi-square tests with multiple comparisons correction.
- **Data:** Use a 3x3 or larger contingency table.
- **Hint:** Use `pairwise_chi_square()` function.

### Exercise 7: Power Analysis
- **Objective:** Conduct power analysis for a chi-square test.
- **Data:** Determine required sample size for desired power.
- **Hint:** Use `pwr.chisq.test()` from the `pwr` package.

### Exercise 8: Comprehensive Analysis
- **Objective:** Perform a complete chi-square analysis including assumption checking, test selection, and reporting.
- **Data:** Create a realistic dataset (e.g., survey responses, clinical trial results).
- **Hint:** Use all the functions developed in this chapter.

## Next Steps

In the next chapter, we'll learn about time series analysis for analyzing temporal data.

---

**Key Takeaways:**
- Chi-square tests are essential for categorical data analysis
- Three main types: goodness of fit, independence, and homogeneity
- Always check expected frequency requirements before using chi-square tests
- Effect sizes (Cramer's V, Phi) provide important information about practical significance
- Fisher's exact test is preferred for small expected frequencies
- Post hoc analyses help identify specific differences when overall test is significant
- Power analysis helps determine appropriate sample sizes
- Proper reporting includes contingency tables, test results, effect sizes, and conclusions
- Assumption violations require alternative approaches or data modifications 