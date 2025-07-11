# Measures of Variability

## Overview

Measures of variability describe how spread out or dispersed the data is around the central tendency. They complement measures of central tendency to give us a complete picture of our data distribution. Understanding variability is crucial for statistical inference, quality control, and decision-making.

### Why Variability Matters

- **Data Quality**: High variability may indicate data quality issues or heterogeneity
- **Statistical Inference**: Variability affects confidence intervals and hypothesis tests
- **Process Control**: Low variability often indicates stable, predictable processes
- **Comparison**: Allows comparison of consistency across different groups
- **Risk Assessment**: Variability is directly related to uncertainty and risk
- **Model Selection**: Influences choice of statistical models and methods

### Types of Variability Measures

1. **Range**: Simple measure of spread (max - min)
2. **Interquartile Range (IQR)**: Robust measure of middle 50% spread
3. **Variance**: Average squared deviation from mean
4. **Standard Deviation**: Square root of variance, same units as data
5. **Coefficient of Variation**: Relative measure of variability
6. **Mean Absolute Deviation (MAD)**: Average absolute deviation
7. **Percentiles**: Position-based measures of spread

## Range

The range is the difference between the maximum and minimum values in a dataset. It's the simplest measure of variability but is highly sensitive to outliers.

### Mathematical Foundation

**Range Formula**:
```math
\text{Range} = x_{\max} - x_{\min}
```

**Properties of Range**:
- Easy to calculate and understand
- Sensitive to outliers
- Depends only on two values
- Not affected by the distribution shape
- Always non-negative

### Basic Range Calculation

```r
# Create sample data
data <- c(2, 4, 6, 8, 10, 12, 14, 16, 18, 20)

# Calculate range
data_range <- range(data)
cat("Range:", data_range[2] - data_range[1], "\n")
cat("Min:", data_range[1], "\n")
cat("Max:", data_range[2], "\n")

# Using built-in function
data_range_alt <- max(data) - min(data)
cat("Range (alternative):", data_range_alt, "\n")

# Understanding the effect of outliers
data_with_outlier <- c(data, 100)
cat("Original range:", max(data) - min(data), "\n")
cat("Range with outlier:", max(data_with_outlier) - min(data_with_outlier), "\n")

# Range as percentage of mean
range_percent <- ((max(data) - min(data)) / mean(data)) * 100
cat("Range as % of mean:", range_percent, "%\n")
```

### Range by Group

```r
# Load data
data(mtcars)

# Calculate range by cylinders
library(dplyr)
range_by_cyl <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    range_mpg = max(mpg) - min(mpg),
    min_mpg = min(mpg),
    max_mpg = max(mpg),
    mean_mpg = mean(mpg),
    range_percent = ((max(mpg) - min(mpg)) / mean(mpg)) * 100
  )

print(range_by_cyl)

# Visualize ranges by group
ggplot(range_by_cyl, aes(x = factor(cyl), y = range_mpg)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  labs(title = "MPG Range by Number of Cylinders",
       x = "Number of Cylinders",
       y = "Range (Max - Min)") +
  theme_minimal()
```

## Interquartile Range (IQR)

The IQR is the difference between the 75th percentile (Q3) and the 25th percentile (Q1). It's a robust measure that is not affected by outliers.

### Mathematical Foundation

**IQR Formula**:
```math
\text{IQR} = Q_3 - Q_1
```

where $Q_1$ is the 25th percentile and $Q_3$ is the 75th percentile.

**Properties of IQR**:
- Robust to outliers
- Represents the middle 50% of data
- Used in box plots and outlier detection
- Not affected by extreme values
- Good for skewed distributions

### Basic IQR Calculation

```r
# Calculate IQR
data_iqr <- IQR(data)
cat("IQR:", data_iqr, "\n")

# Manual calculation
q1 <- quantile(data, 0.25)
q3 <- quantile(data, 0.75)
manual_iqr <- q3 - q1
cat("Manual IQR:", manual_iqr, "\n")

# Verify with built-in function
cat("Built-in IQR:", IQR(data), "\n")

# Understanding quartiles
cat("Q1 (25th percentile):", q1, "\n")
cat("Q2 (50th percentile/median):", quantile(data, 0.5), "\n")
cat("Q3 (75th percentile):", q3, "\n")

# IQR as percentage of median
iqr_percent <- (data_iqr / median(data)) * 100
cat("IQR as % of median:", iqr_percent, "%\n")
```

### IQR by Group

```r
# IQR by cylinders
iqr_by_cyl <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    iqr_mpg = IQR(mpg),
    q1_mpg = quantile(mpg, 0.25),
    q3_mpg = quantile(mpg, 0.75),
    median_mpg = median(mpg),
    iqr_percent = (IQR(mpg) / median(mpg)) * 100
  )

print(iqr_by_cyl)

# Compare IQR vs Range
comparison_by_cyl <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    range_mpg = max(mpg) - min(mpg),
    iqr_mpg = IQR(mpg),
    ratio = (max(mpg) - min(mpg)) / IQR(mpg)
  )

print(comparison_by_cyl)
```

## Variance

Variance measures the average squared deviation from the mean. It's the foundation for many statistical methods.

### Mathematical Foundation

**Population Variance**:
```math
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2
```

**Sample Variance**:
```math
s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2
```

**Properties of Variance**:
- Always non-negative
- Sensitive to outliers
- Units are squared
- Used in statistical inference
- Minimized by the mean

### Population vs Sample Variance

```r
# Population variance (n denominator)
population_variance <- function(x) {
  n <- length(x)
  mean_x <- mean(x)
  sum((x - mean_x)^2) / n
}

# Sample variance (n-1 denominator)
sample_variance <- function(x) {
  n <- length(x)
  mean_x <- mean(x)
  sum((x - mean_x)^2) / (n - 1)
}

# Compare with built-in function
cat("Population variance:", population_variance(data), "\n")
cat("Sample variance:", sample_variance(data), "\n")
cat("Built-in var():", var(data), "\n")

# Understanding the difference
n <- length(data)
cat("Sample size (n):", n, "\n")
cat("Degrees of freedom (n-1):", n-1, "\n")
cat("Ratio n/(n-1):", n/(n-1), "\n")
cat("Population variance * (n/(n-1)) = Sample variance:", 
    population_variance(data) * (n/(n-1)), "\n")
```

### Variance by Group

```r
# Variance by cylinders
variance_by_cyl <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    variance_mpg = var(mpg),
    mean_mpg = mean(mpg),
    sd_mpg = sd(mpg),
    cv_mpg = (sd(mpg) / mean(mpg)) * 100
  )

print(variance_by_cyl)

# Visualize variance by group
ggplot(variance_by_cyl, aes(x = factor(cyl), y = variance_mpg)) +
  geom_bar(stat = "identity", fill = "lightgreen", alpha = 0.7) +
  labs(title = "MPG Variance by Number of Cylinders",
       x = "Number of Cylinders",
       y = "Variance") +
  theme_minimal()
```

## Standard Deviation

Standard deviation is the square root of variance and is in the same units as the original data. It's the most commonly used measure of variability.

### Mathematical Foundation

**Population Standard Deviation**:
```math
\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2}
```

**Sample Standard Deviation**:
```math
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2}
```

**Properties of Standard Deviation**:
- Same units as original data
- Sensitive to outliers
- Used in normal distribution properties
- Basis for confidence intervals
- Used in z-scores and standardization

### Basic Standard Deviation

```r
# Calculate standard deviation
data_sd <- sd(data)
cat("Standard deviation:", data_sd, "\n")

# Manual calculation
manual_sd <- sqrt(var(data))
cat("Manual SD:", manual_sd, "\n")

# Population standard deviation
population_sd <- function(x) {
  sqrt(population_variance(x))
}

cat("Population SD:", population_sd(data), "\n")

# Understanding the empirical rule
mean_data <- mean(data)
sd_data <- sd(data)
cat("Mean:", mean_data, "\n")
cat("Standard deviation:", sd_data, "\n")
cat("68% of data within:", mean_data - sd_data, "to", mean_data + sd_data, "\n")
cat("95% of data within:", mean_data - 2*sd_data, "to", mean_data + 2*sd_data, "\n")
cat("99.7% of data within:", mean_data - 3*sd_data, "to", mean_data + 3*sd_data, "\n")
```

### Standard Deviation by Group

```r
# SD by cylinders
sd_by_cyl <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    sd_mpg = sd(mpg),
    mean_mpg = mean(mpg),
    cv_mpg = sd_mpg / mean_mpg,  # Coefficient of variation
    se_mpg = sd_mpg / sqrt(n())  # Standard error
  )

print(sd_by_cyl)

# Visualize mean ± SD by group
ggplot(sd_by_cyl, aes(x = factor(cyl), y = mean_mpg)) +
  geom_point(size = 3, color = "red") +
  geom_errorbar(aes(ymin = mean_mpg - sd_mpg, ymax = mean_mpg + sd_mpg), 
                width = 0.2, color = "blue") +
  labs(title = "Mean ± SD by Cylinders",
       x = "Number of Cylinders",
       y = "MPG") +
  theme_minimal()
```

## Coefficient of Variation

The coefficient of variation (CV) is the ratio of standard deviation to mean, expressed as a percentage. It allows comparison of variability across different scales.

### Mathematical Foundation

**Coefficient of Variation Formula**:
```math
CV = \frac{s}{\bar{x}} \times 100\%
```

**Properties of CV**:
- Dimensionless measure
- Allows comparison across different scales
- Sensitive to small means
- Used in quality control
- Good for positive data only

```r
# Calculate CV
cv <- function(x) {
  (sd(x) / mean(x)) * 100
}

# Apply to different variables
cat("CV for MPG:", cv(mtcars$mpg), "%\n")
cat("CV for Weight:", cv(mtcars$wt), "%\n")
cat("CV for Horsepower:", cv(mtcars$hp), "%\n")

# Compare variability across different scales
cv_comparison <- data.frame(
  Variable = c("MPG", "Weight", "Horsepower"),
  Mean = c(mean(mtcars$mpg), mean(mtcars$wt), mean(mtcars$hp)),
  SD = c(sd(mtcars$mpg), sd(mtcars$wt), sd(mtcars$hp)),
  CV = c(cv(mtcars$mpg), cv(mtcars$wt), cv(mtcars$hp))
)

print(cv_comparison)

# Visualize CV comparison
ggplot(cv_comparison, aes(x = Variable, y = CV)) +
  geom_bar(stat = "identity", fill = "orange", alpha = 0.7) +
  labs(title = "Coefficient of Variation Comparison",
       x = "Variable",
       y = "CV (%)") +
  theme_minimal()

# Understanding CV interpretation
cat("CV Interpretation:\n")
cat("CV < 15%: Low variability\n")
cat("CV 15-35%: Moderate variability\n")
cat("CV > 35%: High variability\n")
```

## Mean Absolute Deviation (MAD)

MAD is the average of the absolute deviations from the mean. It's more robust than standard deviation but less commonly used.

### Mathematical Foundation

**MAD Formula**:
```math
\text{MAD} = \frac{1}{n}\sum_{i=1}^{n} |x_i - \bar{x}|
```

**Properties of MAD**:
- Same units as original data
- More robust than standard deviation
- Less sensitive to outliers
- Not used in normal distribution properties
- Computationally simple

```r
# Calculate MAD
mad_calculation <- function(x) {
  mean(abs(x - mean(x)))
}

# Compare with built-in function
cat("Manual MAD:", mad_calculation(data), "\n")
cat("Built-in MAD:", mad(data), "\n")

# Note: R's mad() function uses median by default
cat("MAD around median:", mad(data), "\n")
cat("MAD around mean:", mad(data, center = mean(data)), "\n")

# Compare MAD vs SD
cat("Standard deviation:", sd(data), "\n")
cat("MAD around mean:", mad(data, center = mean(data)), "\n")
cat("Ratio SD/MAD:", sd(data) / mad(data, center = mean(data)), "\n")

# For normal distribution, SD ≈ 1.253 × MAD
cat("Expected ratio for normal distribution: 1.253\n")
```

## Percentiles and Quantiles

Percentiles divide the data into 100 equal parts, while quantiles divide into any number of parts.

### Mathematical Foundation

**Percentile Definition**:
The $p$th percentile is the value below which $p\%$ of the data falls.

**Common Percentiles**:
- 25th percentile (Q1): First quartile
- 50th percentile: Median
- 75th percentile (Q3): Third quartile
- 90th percentile: 90% of data below this value

```r
# Calculate percentiles
percentiles <- quantile(mtcars$mpg, probs = c(0.1, 0.25, 0.5, 0.75, 0.9))
print(percentiles)

# Calculate deciles (10th, 20th, ..., 90th percentiles)
deciles <- quantile(mtcars$mpg, probs = seq(0.1, 0.9, by = 0.1))
print(deciles)

# Five-number summary
five_number <- fivenum(mtcars$mpg)
cat("Five-number summary:\n")
cat("Min:", five_number[1], "\n")
cat("Q1:", five_number[2], "\n")
cat("Median:", five_number[3], "\n")
cat("Q3:", five_number[4], "\n")
cat("Max:", five_number[5], "\n")

# Percentile ranks
percentile_rank <- function(x, value) {
  sum(x <= value) / length(x) * 100
}

cat("Percentile rank of 20 MPG:", percentile_rank(mtcars$mpg, 20), "%\n")
cat("Percentile rank of 25 MPG:", percentile_rank(mtcars$mpg, 25), "%\n")
```

## Robust Measures of Variability

### Median Absolute Deviation (MAD)

MAD around the median is a robust measure of variability.

```r
# MAD around median
mad_median <- mad(mtcars$mpg)
cat("MAD around median:", mad_median, "\n")

# MAD around mean
mad_mean <- mad(mtcars$mpg, center = mean(mtcars$mpg))
cat("MAD around mean:", mad_mean, "\n")

# Compare MAD with IQR
cat("IQR:", IQR(mtcars$mpg), "\n")
cat("MAD around median:", mad_median, "\n")
cat("Ratio IQR/MAD:", IQR(mtcars$mpg) / mad_median, "\n")

# For normal distribution, IQR ≈ 1.349 × MAD
cat("Expected ratio for normal distribution: 1.349\n")
```

### Quartile Coefficient of Dispersion

The quartile coefficient of dispersion is a robust relative measure of variability.

```r
# Quartile coefficient of dispersion
qcd <- function(x) {
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  (q3 - q1) / (q3 + q1)
}

cat("Quartile coefficient of dispersion:", qcd(mtcars$mpg), "\n")

# Compare with CV
cat("Coefficient of variation:", cv(mtcars$mpg), "%\n")
cat("Quartile coefficient of dispersion:", qcd(mtcars$mpg), "\n")
```

## Comparing Variability Measures

```r
# Comprehensive variability summary
variability_summary <- function(x, variable_name = "Variable") {
  cat("=== VARIABILITY SUMMARY FOR", variable_name, "===\n")
  cat("Data length:", length(x), "\n")
  cat("Missing values:", sum(is.na(x)), "\n\n")
  
  cat("Basic Measures:\n")
  cat("Range:", max(x, na.rm = TRUE) - min(x, na.rm = TRUE), "\n")
  cat("IQR:", IQR(x, na.rm = TRUE), "\n")
  cat("Variance:", var(x, na.rm = TRUE), "\n")
  cat("Standard Deviation:", sd(x, na.rm = TRUE), "\n")
  cat("Coefficient of Variation:", (sd(x, na.rm = TRUE) / mean(x, na.rm = TRUE)) * 100, "%\n")
  
  cat("\nRobust Measures:\n")
  cat("MAD (around median):", mad(x, na.rm = TRUE), "\n")
  cat("MAD (around mean):", mad(x, center = mean(x, na.rm = TRUE), na.rm = TRUE), "\n")
  cat("Quartile coefficient of dispersion:", qcd(x), "\n")
  
  cat("\nPercentiles:\n")
  percentiles <- quantile(x, probs = c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE)
  cat("5th percentile:", percentiles[1], "\n")
  cat("25th percentile:", percentiles[2], "\n")
  cat("50th percentile (median):", percentiles[3], "\n")
  cat("75th percentile:", percentiles[4], "\n")
  cat("95th percentile:", percentiles[5], "\n")
}

# Apply to different variables
variability_summary(mtcars$mpg, "MPG")
cat("\n")
variability_summary(mtcars$wt, "Weight")
```

## Outlier Detection Using Variability Measures

```r
# Function to detect outliers using IQR method
detect_outliers_iqr <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  
  outliers <- x < lower_bound | x > upper_bound
  return(list(
    outliers = x[outliers],
    outlier_indices = which(outliers),
    bounds = c(lower_bound, upper_bound),
    n_outliers = sum(outliers, na.rm = TRUE)
  ))
}

# Function to detect outliers using z-score method
detect_outliers_zscore <- function(x, threshold = 3) {
  z_scores <- abs((x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE))
  outliers <- z_scores > threshold
  
  return(list(
    outliers = x[outliers],
    outlier_indices = which(outliers),
    z_scores = z_scores,
    n_outliers = sum(outliers, na.rm = TRUE)
  ))
}

# Detect outliers in MPG
mpg_outliers_iqr <- detect_outliers_iqr(mtcars$mpg)
mpg_outliers_zscore <- detect_outliers_zscore(mtcars$mpg)

cat("=== OUTLIER DETECTION FOR MPG ===\n")
cat("IQR method outliers:", mpg_outliers_iqr$outliers, "\n")
cat("IQR method count:", mpg_outliers_iqr$n_outliers, "\n")
cat("Z-score method outliers:", mpg_outliers_zscore$outliers, "\n")
cat("Z-score method count:", mpg_outliers_zscore$n_outliers, "\n")
```

## Practical Examples

### Example 1: Comparing Variability Across Groups

```r
# Compare MPG variability across transmission types
auto_mpg <- mtcars$mpg[mtcars$am == 0]
manual_mpg <- mtcars$mpg[mtcars$am == 1]

cat("=== MPG VARIABILITY BY TRANSMISSION ===\n")
cat("Automatic transmission:\n")
variability_summary(auto_mpg, "Auto MPG")

cat("\nManual transmission:\n")
variability_summary(manual_mpg, "Manual MPG")

# Test for equal variances
var_test <- var.test(auto_mpg, manual_mpg)
print(var_test)

# Visualize variability comparison
transmission_data <- data.frame(
  transmission = rep(c("Automatic", "Manual"), c(length(auto_mpg), length(manual_mpg))),
  mpg = c(auto_mpg, manual_mpg)
)

ggplot(transmission_data, aes(x = transmission, y = mpg)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7) +
  labs(title = "MPG Variability by Transmission Type",
       x = "Transmission",
       y = "MPG") +
  theme_minimal()
```

### Example 2: Time Series Variability

```r
# Simulate time series data
set.seed(123)
time_series <- cumsum(rnorm(100, mean = 0, sd = 1))

# Calculate rolling standard deviation
rolling_sd <- function(x, window = 10) {
  n <- length(x)
  result <- numeric(n - window + 1)
  
  for (i in 1:(n - window + 1)) {
    result[i] <- sd(x[i:(i + window - 1)])
  }
  
  return(result)
}

# Apply rolling SD
rolling_volatility <- rolling_sd(time_series, window = 10)

# Plot
plot(rolling_volatility, type = "l", 
     main = "Rolling Standard Deviation",
     xlab = "Time", ylab = "Standard Deviation")

# Calculate overall variability measures
cat("=== TIME SERIES VARIABILITY ===\n")
cat("Overall SD:", sd(time_series), "\n")
cat("Mean rolling SD:", mean(rolling_volatility), "\n")
cat("SD of rolling SD:", sd(rolling_volatility), "\n")
```

### Example 3: Financial Data Analysis

```r
# Simulate stock returns
set.seed(123)
returns <- rnorm(252, mean = 0.001, sd = 0.02)  # Daily returns

# Calculate volatility measures
cat("=== STOCK RETURN VOLATILITY ===\n")
cat("Daily volatility (SD):", sd(returns), "\n")
cat("Annualized volatility:", sd(returns) * sqrt(252), "\n")
cat("Downside deviation:", sd(returns[returns < 0]), "\n")

# Value at Risk (VaR)
var_95 <- quantile(returns, 0.05)
cat("95% VaR:", var_95, "\n")

# Expected Shortfall (Conditional VaR)
es_95 <- mean(returns[returns <= var_95])
cat("95% Expected Shortfall:", es_95, "\n")

# Rolling volatility
rolling_vol <- rolling_sd(returns, window = 20)
cat("Mean 20-day rolling volatility:", mean(rolling_vol), "\n")
cat("Volatility of volatility:", sd(rolling_vol), "\n")
```

## Visualization of Variability

```r
# Box plots to show variability
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "MPG Variability by Cylinders",
       x = "Number of Cylinders",
       y = "Miles per Gallon") +
  theme_minimal()

# Violin plots for density and variability
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  geom_violin(fill = "lightgreen", alpha = 0.7) +
  geom_boxplot(width = 0.2, fill = "white") +
  labs(title = "MPG Distribution by Cylinders",
       x = "Number of Cylinders",
       y = "Miles per Gallon") +
  theme_minimal()

# Histogram with variability measures
ggplot(mtcars, aes(x = mpg)) +
  geom_histogram(bins = 15, fill = "lightblue", color = "black") +
  geom_vline(xintercept = mean(mtcars$mpg), color = "red", linetype = "dashed") +
  geom_vline(xintercept = mean(mtcars$mpg) + c(-1, 1) * sd(mtcars$mpg), 
             color = "blue", linetype = "dotted") +
  labs(title = "MPG Distribution with Mean and ±1 SD",
       x = "Miles per Gallon",
       y = "Frequency") +
  theme_minimal()

# Scatter plot with error bars
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(title = "MPG vs Weight with Regression Line",
       x = "Weight",
       y = "MPG") +
  theme_minimal()
```

## Best Practices

### When to Use Each Measure

```r
# Guidelines for choosing variability measures
cat("GUIDELINES FOR CHOOSING VARIABILITY MEASURES:\n")
cat("1. Standard Deviation: Use for symmetric, normal distributions\n")
cat("2. IQR: Use for skewed distributions or when outliers are present\n")
cat("3. Coefficient of Variation: Use to compare variability across different scales\n")
cat("4. MAD: Use when you want robustness to outliers\n")
cat("5. Range: Use for quick assessment, but sensitive to outliers\n")
cat("6. Percentiles: Use for non-parametric analysis\n")

# Decision tree function
choose_variability_measure <- function(x) {
  cat("=== VARIABILITY MEASURE DECISION TREE ===\n")
  
  # Check for missing values
  if (sum(is.na(x)) > 0) {
    cat("Data contains missing values - use na.rm = TRUE\n")
  }
  
  # Check for outliers
  outliers_iqr <- detect_outliers_iqr(x)
  outliers_zscore <- detect_outliers_zscore(x)
  
  if (outliers_iqr$n_outliers > 0 || outliers_zscore$n_outliers > 0) {
    cat("Outliers detected\n")
    cat("Recommendation: Use IQR or MAD (robust measures)\n")
    return("IQR")
  }
  
  # Check for skewness
  mean_val <- mean(x, na.rm = TRUE)
  median_val <- median(x, na.rm = TRUE)
  skewness_indicator <- abs(mean_val - median_val) / mean_val
  
  if (skewness_indicator > 0.1) {
    cat("Skewed distribution detected\n")
    cat("Recommendation: Use IQR or MAD\n")
    return("IQR")
  } else {
    cat("Symmetric distribution\n")
    cat("Recommendation: Use Standard Deviation\n")
    return("SD")
  }
}

# Test the decision function
choose_variability_measure(mtcars$mpg)
```

### Handling Missing Data

```r
# Function to handle missing data in variability calculations
robust_variability <- function(x) {
  # Remove missing values
  x_clean <- x[!is.na(x)]
  
  if (length(x_clean) == 0) {
    return(NA)
  }
  
  # Calculate measures
  result <- list(
    range = max(x_clean) - min(x_clean),
    iqr = IQR(x_clean),
    variance = var(x_clean),
    sd = sd(x_clean),
    cv = (sd(x_clean) / mean(x_clean)) * 100,
    mad = mad(x_clean),
    qcd = qcd(x_clean),
    n = length(x_clean),
    missing = sum(is.na(x)),
    missing_percent = sum(is.na(x)) / length(x) * 100
  )
  
  return(result)
}

# Test with missing data
data_with_na <- c(1, 2, NA, 4, 5, NA, 7, 8)
result <- robust_variability(data_with_na)
print(result)
```

## Exercises

### Exercise 1: Basic Variability Calculations
Calculate range, IQR, variance, and standard deviation for the following dataset: [12, 15, 18, 22, 25, 28, 30, 35, 40, 45]

```r
# Your solution here
exercise_data <- c(12, 15, 18, 22, 25, 28, 30, 35, 40, 45)

cat("Dataset:", exercise_data, "\n")
cat("Range:", max(exercise_data) - min(exercise_data), "\n")
cat("IQR:", IQR(exercise_data), "\n")
cat("Variance:", var(exercise_data), "\n")
cat("Standard Deviation:", sd(exercise_data), "\n")
cat("Coefficient of Variation:", (sd(exercise_data) / mean(exercise_data)) * 100, "%\n")
```

### Exercise 2: Comparing Variability
Compare the variability of different variables in the mtcars dataset using appropriate measures.

```r
# Your solution here
# Compare variability across different variables
variables <- c("mpg", "wt", "hp", "disp")
comparison_df <- data.frame(
  Variable = variables,
  Mean = sapply(mtcars[variables], mean),
  SD = sapply(mtcars[variables], sd),
  CV = sapply(mtcars[variables], function(x) (sd(x) / mean(x)) * 100),
  IQR = sapply(mtcars[variables], IQR),
  Range = sapply(mtcars[variables], function(x) max(x) - min(x))
)

print(comparison_df)

# Visualize CV comparison
ggplot(comparison_df, aes(x = Variable, y = CV)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  labs(title = "Coefficient of Variation Comparison",
       x = "Variable",
       y = "CV (%)") +
  theme_minimal()
```

### Exercise 3: Outlier Detection
Use the IQR method to detect outliers in the iris dataset for each species.

```r
# Your solution here
data(iris)

# Detect outliers by species
species_list <- unique(iris$Species)
for (species in species_list) {
  species_data <- iris$Sepal.Length[iris$Species == species]
  outliers <- detect_outliers_iqr(species_data)
  
  cat("===", species, "===\n")
  cat("Outliers:", outliers$outliers, "\n")
  cat("Number of outliers:", outliers$n_outliers, "\n")
  cat("Total observations:", length(species_data), "\n")
  cat("Outlier percentage:", (outliers$n_outliers / length(species_data)) * 100, "%\n\n")
}
```

### Exercise 4: Robust Measures
Compare standard deviation and MAD for datasets with and without outliers.

```r
# Your solution here
# Create datasets
clean_data <- rnorm(100, mean = 10, sd = 2)
outlier_data <- c(clean_data, 50, 60, 70)

# Compare measures
cat("=== CLEAN DATA ===\n")
cat("SD:", sd(clean_data), "\n")
cat("MAD:", mad(clean_data), "\n")
cat("Ratio SD/MAD:", sd(clean_data) / mad(clean_data), "\n")

cat("\n=== DATA WITH OUTLIERS ===\n")
cat("SD:", sd(outlier_data), "\n")
cat("MAD:", mad(outlier_data), "\n")
cat("Ratio SD/MAD:", sd(outlier_data) / mad(outlier_data), "\n")

cat("\nChange in SD:", sd(outlier_data) - sd(clean_data), "\n")
cat("Change in MAD:", mad(outlier_data) - mad(clean_data), "\n")
```

### Exercise 5: Real-world Application
Analyze the variability of a real dataset of your choice and interpret the results.

```r
# Your solution here
# Use built-in airquality dataset
data(airquality)

# Comprehensive variability analysis
cat("=== AIR QUALITY VARIABILITY ANALYSIS ===\n")

# Analyze Ozone
variability_summary(airquality$Ozone, "Ozone")

# Compare variability by month
airquality %>%
  group_by(Month) %>%
  summarise(
    n = n(),
    mean_ozone = mean(Ozone, na.rm = TRUE),
    sd_ozone = sd(Ozone, na.rm = TRUE),
    cv_ozone = (sd(Ozone, na.rm = TRUE) / mean(Ozone, na.rm = TRUE)) * 100,
    iqr_ozone = IQR(Ozone, na.rm = TRUE)
  )

# Visualize variability by month
ggplot(airquality, aes(x = factor(Month), y = Ozone)) +
  geom_boxplot(fill = "lightgreen", alpha = 0.7) +
  labs(title = "Ozone Variability by Month",
       x = "Month",
       y = "Ozone") +
  theme_minimal()
```

## Next Steps

In the next chapter, we'll learn about data visualization techniques to complement our understanding of central tendency and variability. We'll cover:

- **Histograms and Density Plots**: Visualizing distributions
- **Box Plots and Violin Plots**: Showing variability and shape
- **Scatter Plots**: Exploring relationships between variables
- **Q-Q Plots**: Assessing normality
- **Time Series Plots**: Visualizing temporal patterns

---

**Key Takeaways:**
- Standard deviation is most common but sensitive to outliers
- IQR is robust and good for skewed distributions
- Coefficient of variation allows comparison across different scales
- MAD is robust to outliers
- Always consider the data distribution when choosing measures
- Use multiple measures for comprehensive understanding
- Visualize variability alongside central tendency
- Outliers can significantly affect variance and standard deviation
- Robust measures are preferred for non-normal distributions
- Variability measures are essential for statistical inference 