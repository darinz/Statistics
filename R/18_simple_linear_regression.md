# Correlation Analysis

## Overview

Correlation analysis examines the strength and direction of relationships between variables. It's fundamental to understanding associations in data and is often a precursor to regression analysis.

## Pearson Correlation

### Basic Pearson Correlation

```r
# Load sample data
data(mtcars)

# Calculate Pearson correlation between MPG and weight
pearson_cor <- cor(mtcars$mpg, mtcars$wt, method = "pearson")
cat("Pearson correlation (MPG vs Weight):", pearson_cor, "\n")

# Calculate correlation matrix for multiple variables
correlation_matrix <- cor(mtcars[, c("mpg", "wt", "hp", "disp")], method = "pearson")
print(correlation_matrix)

# Test significance of correlation
cor_test <- cor.test(mtcars$mpg, mtcars$wt, method = "pearson")
print(cor_test)
```

### Correlation with Confidence Intervals

```r
# Calculate correlation with confidence interval
library(psych)

# Using psych package for correlation with CI
cor_with_ci <- corr.test(mtcars[, c("mpg", "wt", "hp", "disp")], 
                         use = "pairwise", 
                         method = "pearson")

print(cor_with_ci$r)  # Correlation matrix
print(cor_with_ci$p)  # P-values
print(cor_with_ci$ci) # Confidence intervals
```

### Visualizing Correlations

```r
# Scatter plot with correlation line
plot(mtcars$wt, mtcars$mpg, 
     main = "MPG vs Weight",
     xlab = "Weight (1000 lbs)", 
     ylab = "Miles per Gallon",
     pch = 16, col = "blue")

# Add correlation line
abline(lm(mpg ~ wt, data = mtcars), col = "red", lwd = 2)

# Add correlation coefficient text
text(4, 30, paste("r =", round(pearson_cor, 3)), col = "red", cex = 1.2)

# Correlation matrix heatmap
library(corrplot)
corrplot(correlation_matrix, method = "color", type = "upper", 
         addCoef.col = "black", tl.col = "black", tl.srt = 45)
```

## Spearman Correlation

### Basic Spearman Correlation

```r
# Calculate Spearman correlation
spearman_cor <- cor(mtcars$mpg, mtcars$wt, method = "spearman")
cat("Spearman correlation (MPG vs Weight):", spearman_cor, "\n")

# Test significance
spearman_test <- cor.test(mtcars$mpg, mtcars$wt, method = "spearman")
print(spearman_test)

# Compare Pearson vs Spearman
cat("Pearson correlation:", pearson_cor, "\n")
cat("Spearman correlation:", spearman_cor, "\n")
```

### Rank Correlation Analysis

```r
# Create ranked data
mpg_ranks <- rank(mtcars$mpg)
wt_ranks <- rank(mtcars$wt)

# Manual calculation of Spearman correlation
n <- length(mtcars$mpg)
sum_d_squared <- sum((mpg_ranks - wt_ranks)^2)
spearman_manual <- 1 - (6 * sum_d_squared) / (n * (n^2 - 1))

cat("Manual Spearman calculation:", spearman_manual, "\n")
cat("Built-in Spearman:", spearman_cor, "\n")
```

## Kendall's Tau

### Basic Kendall's Tau

```r
# Calculate Kendall's tau
kendall_tau <- cor(mtcars$mpg, mtcars$wt, method = "kendall")
cat("Kendall's tau (MPG vs Weight):", kendall_tau, "\n")

# Test significance
kendall_test <- cor.test(mtcars$mpg, mtcars$wt, method = "kendall")
print(kendall_test)

# Compare all three correlation measures
cat("Pearson:", pearson_cor, "\n")
cat("Spearman:", spearman_cor, "\n")
cat("Kendall's tau:", kendall_tau, "\n")
```

## Partial Correlation

### Controlling for Third Variables

```r
# Calculate partial correlation (MPG vs Weight, controlling for HP)
library(ppcor)

partial_cor <- pcor.test(mtcars$mpg, mtcars$wt, mtcars$hp, method = "pearson")
print(partial_cor)

# Compare zero-order vs partial correlation
cat("Zero-order correlation (MPG vs Weight):", pearson_cor, "\n")
cat("Partial correlation (MPG vs Weight | HP):", partial_cor$estimate, "\n")

# Multiple partial correlations
partial_matrix <- pcor(mtcars[, c("mpg", "wt", "hp", "disp")])
print(partial_matrix$estimate)
```

## Correlation Matrix Analysis

### Comprehensive Correlation Analysis

```r
# Function to analyze correlation matrix
analyze_correlations <- function(data, method = "pearson") {
  # Calculate correlation matrix
  cor_matrix <- cor(data, method = method, use = "pairwise.complete.obs")
  
  # Calculate p-values
  p_matrix <- matrix(NA, nrow = ncol(data), ncol = ncol(data))
  for (i in 1:ncol(data)) {
    for (j in 1:ncol(data)) {
      if (i != j) {
        test_result <- cor.test(data[, i], data[, j], method = method)
        p_matrix[i, j] <- test_result$p.value
      }
    }
  }
  
  # Create significance matrix
  sig_matrix <- p_matrix < 0.05
  
  return(list(
    correlations = cor_matrix,
    p_values = p_matrix,
    significant = sig_matrix
  ))
}

# Apply to mtcars data
selected_vars <- mtcars[, c("mpg", "wt", "hp", "disp", "drat", "qsec")]
cor_analysis <- analyze_correlations(selected_vars)

print("Correlation Matrix:")
print(round(cor_analysis$correlations, 3))

print("P-values:")
print(round(cor_analysis$p_values, 3))

print("Significant correlations (p < 0.05):")
print(cor_analysis$significant)
```

### Correlation Network Analysis

```r
# Create correlation network
library(qgraph)

# Prepare correlation matrix for network
cor_network <- cor_analysis$correlations
diag(cor_network) <- 0  # Remove self-correlations

# Create network plot
qgraph(cor_network, 
       layout = "spring",
       labels = colnames(selected_vars),
       title = "Correlation Network",
       edge.color = ifelse(cor_network > 0, "green", "red"),
       edge.width = abs(cor_network) * 3)
```

## Correlation vs Causation

### Understanding Correlation Limits

```r
# Function to demonstrate correlation vs causation
demonstrate_correlation_limits <- function() {
  cat("=== CORRELATION vs CAUSATION ===\n\n")
  
  cat("1. Correlation does not imply causation\n")
  cat("   - Two variables can be correlated without one causing the other\n")
  cat("   - Third variables may explain the relationship\n\n")
  
  cat("2. Examples of spurious correlations:\n")
  cat("   - Ice cream sales and crime rates (both increase in summer)\n")
  cat("   - Number of firefighters and damage (more fires = more damage)\n\n")
  
  cat("3. Types of relationships:\n")
  cat("   - Direct causation: A → B\n")
  cat("   - Reverse causation: B → A\n")
  cat("   - Common cause: C → A, C → B\n")
  cat("   - Coincidence: No real relationship\n\n")
  
  cat("4. Establishing causation requires:\n")
  cat("   - Temporal precedence\n")
  cat("   - Covariation\n")
  cat("   - Elimination of alternative explanations\n")
  cat("   - Theoretical plausibility\n")
}

demonstrate_correlation_limits()
```

## Robust Correlation Methods

### Winsorized Correlation

```r
# Function to calculate winsorized correlation
winsorized_correlation <- function(x, y, k = 2) {
  # Winsorize the data
  x_winsorized <- winsorize(x, k)
  y_winsorized <- winsorize(y, k)
  
  # Calculate correlation
  return(cor(x_winsorized, y_winsorized))
}

# Winsorize function
winsorize <- function(x, k) {
  n <- length(x)
  sorted_x <- sort(x)
  
  # Replace k smallest and k largest values
  x_winsorized <- x
  x_winsorized[x <= sorted_x[k]] <- sorted_x[k]
  x_winsorized[x >= sorted_x[n - k + 1]] <- sorted_x[n - k + 1]
  
  return(x_winsorized)
}

# Compare robust correlations
pearson_robust <- winsorized_correlation(mtcars$mpg, mtcars$wt)
cat("Winsorized correlation:", pearson_robust, "\n")
cat("Original Pearson:", pearson_cor, "\n")
```

### Biweight Midcorrelation

```r
# Function to calculate biweight midcorrelation
biweight_midcorrelation <- function(x, y) {
  # Calculate median absolute deviation
  mad_x <- mad(x)
  mad_y <- mad(y)
  
  # Calculate u-values
  u_x <- (x - median(x)) / (9 * mad_x)
  u_y <- (y - median(y)) / (9 * mad_y)
  
  # Calculate weights
  w_x <- (1 - u_x^2)^2 * (abs(u_x) < 1)
  w_y <- (1 - u_y^2)^2 * (abs(u_y) < 1)
  
  # Calculate biweight midcorrelation
  numerator <- sum(w_x * w_y * (x - median(x)) * (y - median(y)))
  denominator <- sqrt(sum(w_x * (x - median(x))^2) * sum(w_y * (y - median(y))^2))
  
  return(numerator / denominator)
}

# Compare with other methods
biweight_cor <- biweight_midcorrelation(mtcars$mpg, mtcars$wt)
cat("Biweight midcorrelation:", biweight_cor, "\n")
cat("Pearson correlation:", pearson_cor, "\n")
cat("Spearman correlation:", spearman_cor, "\n")
```

## Correlation in Different Contexts

### Time Series Correlation

```r
# Simulate time series data
set.seed(123)
n <- 100
time <- 1:n
series1 <- cumsum(rnorm(n, mean = 0, sd = 1))
series2 <- cumsum(rnorm(n, mean = 0, sd = 1)) + 0.5 * series1

# Calculate correlation
ts_cor <- cor(series1, series2)
cat("Time series correlation:", ts_cor, "\n")

# Plot time series
plot(time, series1, type = "l", col = "blue", 
     main = "Time Series Correlation",
     xlab = "Time", ylab = "Value")
lines(time, series2, col = "red")
legend("topleft", legend = c("Series 1", "Series 2"), 
       col = c("blue", "red"), lty = 1)
```

### Categorical Variable Correlation

```r
# Correlation with categorical variables
# Convert transmission to numeric for correlation
mtcars$am_numeric <- as.numeric(mtcars$am)

# Point-biserial correlation (binary variable)
point_biserial <- cor(mtcars$mpg, mtcars$am_numeric)
cat("Point-biserial correlation (MPG vs Transmission):", point_biserial, "\n")

# Biserial correlation (assuming underlying normal distribution)
library(polycor)
biserial_cor <- polyserial(mtcars$mpg, mtcars$am_numeric)
cat("Biserial correlation:", biserial_cor, "\n")
```

## Correlation Diagnostics

### Outlier Detection in Correlation

```r
# Function to detect influential points in correlation
correlation_outliers <- function(x, y, threshold = 2) {
  # Calculate standardized residuals
  model <- lm(y ~ x)
  residuals <- rstandard(model)
  
  # Identify outliers
  outliers <- abs(residuals) > threshold
  
  # Calculate correlation with and without outliers
  cor_all <- cor(x, y)
  cor_clean <- cor(x[!outliers], y[!outliers])
  
  return(list(
    outliers = which(outliers),
    correlation_all = cor_all,
    correlation_clean = cor_clean,
    change = cor_all - cor_clean
  ))
}

# Apply to MPG vs Weight
outlier_analysis <- correlation_outliers(mtcars$wt, mtcars$mpg)
print(outlier_analysis)

# Plot with outliers highlighted
plot(mtcars$wt, mtcars$mpg, 
     main = "MPG vs Weight with Outliers",
     xlab = "Weight", ylab = "MPG")
points(mtcars$wt[outlier_analysis$outliers], 
       mtcars$mpg[outlier_analysis$outliers], 
       col = "red", pch = 16, cex = 1.5)
```

### Heteroscedasticity in Correlation

```r
# Check for heteroscedasticity
plot(mtcars$wt, mtcars$mpg, 
     main = "MPG vs Weight",
     xlab = "Weight", ylab = "MPG")

# Add regression line
abline(lm(mpg ~ wt, data = mtcars), col = "red")

# Residual plot
model <- lm(mpg ~ wt, data = mtcars)
plot(fitted(model), residuals(model), 
     main = "Residual Plot",
     xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)
```

## Practical Examples

### Example 1: Financial Data Analysis

```r
# Simulate financial data
set.seed(123)
n_days <- 252
returns_stock1 <- rnorm(n_days, mean = 0.001, sd = 0.02)
returns_stock2 <- 0.7 * returns_stock1 + rnorm(n_days, mean = 0, sd = 0.015)

# Calculate rolling correlation
window_size <- 30
rolling_cor <- numeric(n_days - window_size + 1)

for (i in 1:(n_days - window_size + 1)) {
  rolling_cor[i] <- cor(returns_stock1[i:(i + window_size - 1)], 
                        returns_stock2[i:(i + window_size - 1)])
}

# Plot rolling correlation
plot(rolling_cor, type = "l", 
     main = "Rolling 30-Day Correlation",
     xlab = "Time", ylab = "Correlation")
abline(h = 0.7, col = "red", lty = 2)
```

### Example 2: Survey Data Analysis

```r
# Simulate survey data
set.seed(123)
n_respondents <- 200

# Simulate correlated variables
satisfaction <- rnorm(n_respondents, mean = 7, sd = 1.5)
loyalty <- 0.6 * satisfaction + rnorm(n_respondents, mean = 0, sd = 1)
recommendation <- 0.5 * satisfaction + 0.3 * loyalty + rnorm(n_respondents, mean = 0, sd = 0.8)

survey_data <- data.frame(
  satisfaction = satisfaction,
  loyalty = loyalty,
  recommendation = recommendation
)

# Calculate correlation matrix
survey_cor <- cor(survey_data)
print(round(survey_cor, 3))

# Test significance
survey_tests <- corr.test(survey_data)
print(survey_tests$p)
```

### Example 3: Scientific Research

```r
# Simulate scientific data
set.seed(123)
n_subjects <- 50

# Simulate correlated physiological measures
heart_rate <- rnorm(n_subjects, mean = 75, sd = 10)
blood_pressure <- 0.8 * heart_rate + rnorm(n_subjects, mean = 120, sd = 15)
stress_level <- -0.6 * heart_rate + rnorm(n_subjects, mean = 5, sd = 2)

scientific_data <- data.frame(
  heart_rate = heart_rate,
  blood_pressure = blood_pressure,
  stress_level = stress_level
)

# Calculate partial correlations
partial_cor_scientific <- pcor(scientific_data)
print(round(partial_cor_scientific$estimate, 3))

# Compare zero-order vs partial correlations
zero_order <- cor(scientific_data)
print("Zero-order correlations:")
print(round(zero_order, 3))

print("Partial correlations:")
print(round(partial_cor_scientific$estimate, 3))
```

## Best Practices

### Correlation Interpretation Guidelines

```r
# Function to interpret correlation strength
interpret_correlation <- function(r, method = "Pearson") {
  abs_r <- abs(r)
  
  strength <- ifelse(abs_r >= 0.9, "very strong",
                     ifelse(abs_r >= 0.7, "strong",
                            ifelse(abs_r >= 0.5, "moderate",
                                   ifelse(abs_r >= 0.3, "weak",
                                          "very weak"))))
  
  direction <- ifelse(r > 0, "positive", "negative")
  
  cat("=== CORRELATION INTERPRETATION ===\n")
  cat("Method:", method, "\n")
  cat("Correlation coefficient:", round(r, 3), "\n")
  cat("Strength:", strength, "\n")
  cat("Direction:", direction, "\n")
  cat("Interpretation: There is a", strength, direction, "relationship.\n")
  
  # Guidelines for interpretation
  cat("\nGuidelines for interpretation:\n")
  cat("- |r| ≥ 0.9: Very strong relationship\n")
  cat("- |r| ≥ 0.7: Strong relationship\n")
  cat("- |r| ≥ 0.5: Moderate relationship\n")
  cat("- |r| ≥ 0.3: Weak relationship\n")
  cat("- |r| < 0.3: Very weak relationship\n")
}

# Apply to MPG vs Weight
interpret_correlation(pearson_cor, "Pearson")
```

### Reporting Guidelines

```r
# Function to generate correlation report
generate_correlation_report <- function(x, y, method = "pearson") {
  # Calculate correlation and test
  cor_result <- cor.test(x, y, method = method)
  
  cat("=== CORRELATION REPORT ===\n\n")
  cat("Variables:", deparse(substitute(x)), "and", deparse(substitute(y)), "\n")
  cat("Method:", method, "\n")
  cat("Correlation coefficient:", round(cor_result$estimate, 3), "\n")
  cat("95% Confidence Interval:", round(cor_result$conf.int, 3), "\n")
  cat("t-statistic:", round(cor_result$statistic, 3), "\n")
  cat("Degrees of freedom:", cor_result$parameter, "\n")
  cat("p-value:", ifelse(cor_result$p.value < 0.001, "< .001", 
                         round(cor_result$p.value, 3)), "\n")
  
  # Interpretation
  if (cor_result$p.value < 0.05) {
    cat("Conclusion: Significant correlation (p < .05)\n")
  } else {
    cat("Conclusion: Non-significant correlation (p ≥ .05)\n")
  }
}

# Apply to MPG vs Weight
generate_correlation_report(mtcars$mpg, mtcars$wt, "pearson")
```

## Exercises

### Exercise 1: Basic Correlation Analysis
Calculate and interpret Pearson, Spearman, and Kendall correlations between MPG and weight in the mtcars dataset.

### Exercise 2: Correlation Matrix
Create a correlation matrix for all numeric variables in the mtcars dataset and identify the strongest relationships.

### Exercise 3: Partial Correlation
Calculate partial correlations between MPG, weight, and horsepower, controlling for each variable in turn.

### Exercise 4: Robust Correlation
Compare different correlation methods (Pearson, Spearman, winsorized) on a dataset with outliers.

### Exercise 5: Time Series Correlation
Generate two time series with known correlation and calculate rolling correlations to demonstrate how relationships change over time.

## Next Steps

In the next chapter, we'll learn about simple linear regression, which builds upon correlation analysis to make predictions.

---

**Key Takeaways:**
- Pearson correlation measures linear relationships
- Spearman correlation measures monotonic relationships
- Kendall's tau is robust to outliers
- Partial correlation controls for third variables
- Correlation does not imply causation
- Always check assumptions and outliers
- Consider effect size alongside significance
- Use appropriate correlation measures for your data type 