# Measures of Variability

## Overview

Measures of variability describe how spread out or dispersed the data is around the central tendency. They complement measures of central tendency to give us a complete picture of our data distribution.

## Range

The range is the difference between the maximum and minimum values in a dataset.

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
    max_mpg = max(mpg)
  )

print(range_by_cyl)
```

## Interquartile Range (IQR)

The IQR is the difference between the 75th percentile (Q3) and the 25th percentile (Q1).

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
```

### IQR by Group

```r
# IQR by cylinders
iqr_by_cyl <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    iqr_mpg = IQR(mpg),
    q1_mpg = quantile(mpg, 0.25),
    q3_mpg = quantile(mpg, 0.75)
  )

print(iqr_by_cyl)
```

## Variance

Variance measures the average squared deviation from the mean.

### Population Variance

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
```

### Variance by Group

```r
# Variance by cylinders
variance_by_cyl <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    variance_mpg = var(mpg),
    mean_mpg = mean(mpg)
  )

print(variance_by_cyl)
```

## Standard Deviation

Standard deviation is the square root of variance and is in the same units as the original data.

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
```

### Standard Deviation by Group

```r
# SD by cylinders
sd_by_cyl <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    sd_mpg = sd(mpg),
    mean_mpg = mean(mpg),
    cv_mpg = sd_mpg / mean_mpg  # Coefficient of variation
  )

print(sd_by_cyl)
```

## Coefficient of Variation

The coefficient of variation (CV) is the ratio of standard deviation to mean, expressed as a percentage.

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
```

## Mean Absolute Deviation (MAD)

MAD is the average of the absolute deviations from the mean.

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
```

## Percentiles and Quantiles

Percentiles divide the data into 100 equal parts, while quantiles divide into any number of parts.

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
```

## Robust Measures of Variability

### Median Absolute Deviation (MAD)

```r
# MAD around median
mad_median <- mad(mtcars$mpg)
cat("MAD around median:", mad_median, "\n")

# MAD around mean
mad_mean <- mad(mtcars$mpg, center = mean(mtcars$mpg))
cat("MAD around mean:", mad_mean, "\n")
```

### Quartile Coefficient of Dispersion

```r
# Quartile coefficient of dispersion
qcd <- function(x) {
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  (q3 - q1) / (q3 + q1)
}

cat("Quartile coefficient of dispersion:", qcd(mtcars$mpg), "\n")
```

## Comparing Variability Measures

```r
# Comprehensive variability summary
variability_summary <- function(x, variable_name = "Variable") {
  cat("=== VARIABILITY SUMMARY FOR", variable_name, "===\n")
  cat("Range:", max(x) - min(x), "\n")
  cat("IQR:", IQR(x), "\n")
  cat("Variance:", var(x), "\n")
  cat("Standard Deviation:", sd(x), "\n")
  cat("Coefficient of Variation:", (sd(x) / mean(x)) * 100, "%\n")
  cat("MAD (around median):", mad(x), "\n")
  cat("MAD (around mean):", mad(x, center = mean(x)), "\n")
  cat("Quartile coefficient of dispersion:", qcd(x), "\n")
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
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  
  outliers <- x < lower_bound | x > upper_bound
  return(list(
    outliers = x[outliers],
    outlier_indices = which(outliers),
    bounds = c(lower_bound, upper_bound)
  ))
}

# Detect outliers in MPG
mpg_outliers <- detect_outliers_iqr(mtcars$mpg)
cat("MPG outliers:", mpg_outliers$outliers, "\n")
cat("Number of outliers:", length(mpg_outliers$outliers), "\n")
cat("Outlier bounds:", mpg_outliers$bounds, "\n")
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
    n = length(x_clean),
    missing = sum(is.na(x))
  )
  
  return(result)
}

# Test with missing data
data_with_na <- c(1, 2, NA, 4, 5, NA, 7, 8)
robust_variability(data_with_na)
```

## Exercises

### Exercise 1: Basic Variability Calculations
Calculate range, IQR, variance, and standard deviation for the following dataset: [12, 15, 18, 22, 25, 28, 30, 35, 40, 45]

### Exercise 2: Comparing Variability
Compare the variability of different variables in the mtcars dataset using appropriate measures.

### Exercise 3: Outlier Detection
Use the IQR method to detect outliers in the iris dataset for each species.

### Exercise 4: Robust Measures
Compare standard deviation and MAD for datasets with and without outliers.

### Exercise 5: Real-world Application
Analyze the variability of a real dataset of your choice and interpret the results.

## Next Steps

In the next chapter, we'll learn about data visualization techniques to complement our understanding of central tendency and variability.

---

**Key Takeaways:**
- Standard deviation is most common but sensitive to outliers
- IQR is robust and good for skewed distributions
- Coefficient of variation allows comparison across different scales
- MAD is robust to outliers
- Always consider the data distribution when choosing measures
- Use multiple measures for comprehensive understanding
- Visualize variability alongside central tendency 