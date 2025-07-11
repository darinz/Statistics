# Measures of Central Tendency

## Overview

Measures of central tendency describe the center or typical value of a dataset. They help us understand what the "average" or "typical" value is in our data.

## Mean (Arithmetic Mean)

The mean is the sum of all values divided by the number of values.

### Basic Mean Calculation

```r
# Create sample data
data <- c(2, 4, 6, 8, 10)

# Calculate mean
mean_value <- mean(data)
mean_value

# Mean with missing values
data_with_na <- c(2, 4, NA, 8, 10)
mean(data_with_na, na.rm = TRUE)
```

### Mean by Group

```r
# Load data
data(mtcars)

# Calculate mean by group
library(dplyr)

# Mean MPG by number of cylinders
mpg_by_cyl <- mtcars %>%
  group_by(cyl) %>%
  summarise(mean_mpg = mean(mpg),
            count = n())

mpg_by_cyl
```

### Weighted Mean

```r
# Create data with weights
values <- c(85, 92, 78, 96, 88)
weights <- c(0.2, 0.3, 0.1, 0.25, 0.15)

# Calculate weighted mean
weighted_mean <- weighted.mean(values, weights)
weighted_mean

# Verify calculation
sum(values * weights) / sum(weights)
```

## Median

The median is the middle value when data is ordered from smallest to largest.

### Basic Median Calculation

```r
# Calculate median
data <- c(1, 3, 5, 7, 9)
median_value <- median(data)
median_value

# Median with even number of observations
data_even <- c(1, 3, 5, 7, 9, 11)
median(data_even)

# Median with missing values
data_with_na <- c(1, 3, NA, 7, 9)
median(data_with_na, na.rm = TRUE)
```

### Median by Group

```r
# Median MPG by cylinders
median_by_cyl <- mtcars %>%
  group_by(cyl) %>%
  summarise(median_mpg = median(mpg),
            count = n())

median_by_cyl
```

## Mode

The mode is the most frequently occurring value in a dataset.

### Finding the Mode

```r
# Function to find mode
find_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Test the function
data <- c(1, 2, 2, 3, 4, 4, 4, 5)
mode_value <- find_mode(data)
mode_value

# Using table to see frequency
table(data)
```

### Mode for Categorical Data

```r
# Create categorical data
colors <- c("red", "blue", "red", "green", "blue", "red", "yellow")
mode_color <- find_mode(colors)
mode_color

# Frequency table
color_freq <- table(colors)
color_freq
```

## Geometric Mean

The geometric mean is useful for data that represents rates of change or ratios.

```r
# Function to calculate geometric mean
geometric_mean <- function(x, na.rm = TRUE) {
  if (na.rm) x <- x[!is.na(x)]
  exp(mean(log(x)))
}

# Example: Growth rates
growth_rates <- c(1.05, 1.08, 1.12, 1.06)
geometric_mean(growth_rates)

# Verify with built-in function
exp(mean(log(growth_rates)))
```

## Harmonic Mean

The harmonic mean is useful for rates and ratios.

```r
# Function to calculate harmonic mean
harmonic_mean <- function(x, na.rm = TRUE) {
  if (na.rm) x <- x[!is.na(x)]
  length(x) / sum(1/x)
}

# Example: Average speed
speeds <- c(60, 40, 80)  # km/h
harmonic_mean(speeds)

# Verify calculation
3 / (1/60 + 1/40 + 1/80)
```

## Trimmed Mean

The trimmed mean removes a percentage of extreme values before calculating the mean.

```r
# Calculate trimmed mean
data <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 100)

# 10% trimmed mean (removes 10% from each end)
trimmed_mean <- mean(data, trim = 0.1)
trimmed_mean

# Compare with regular mean
regular_mean <- mean(data)
regular_mean
```

## Comparing Measures of Central Tendency

```r
# Create data with outliers
data_with_outliers <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 100)

# Calculate different measures
mean_val <- mean(data_with_outliers)
median_val <- median(data_with_outliers)
mode_val <- find_mode(data_with_outliers)

# Compare results
cat("Mean:", mean_val, "\n")
cat("Median:", median_val, "\n")
cat("Mode:", mode_val, "\n")

# Create summary function
central_tendency_summary <- function(x) {
  cat("Measures of Central Tendency:\n")
  cat("Mean:", mean(x, na.rm = TRUE), "\n")
  cat("Median:", median(x, na.rm = TRUE), "\n")
  cat("Mode:", find_mode(x), "\n")
  cat("Geometric Mean:", geometric_mean(x), "\n")
  cat("Harmonic Mean:", harmonic_mean(x), "\n")
  cat("10% Trimmed Mean:", mean(x, trim = 0.1, na.rm = TRUE), "\n")
}

# Apply to mtcars MPG
central_tendency_summary(mtcars$mpg)
```

## Robust Measures

### Winsorized Mean

```r
# Function to calculate winsorized mean
winsorized_mean <- function(x, k = 1) {
  sorted_x <- sort(x)
  n <- length(x)
  sorted_x[1:k] <- sorted_x[k + 1]
  sorted_x[(n - k + 1):n] <- sorted_x[n - k]
  mean(sorted_x)
}

# Example
data <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 100)
winsorized_mean(data, k = 1)
```

## Practical Examples

### Example 1: Student Test Scores

```r
# Create student test scores
test_scores <- c(65, 72, 78, 85, 88, 90, 92, 95, 98, 100)

# Calculate various measures
cat("Test Score Analysis:\n")
central_tendency_summary(test_scores)

# Visualize the distribution
hist(test_scores, main = "Distribution of Test Scores", 
     xlab = "Score", col = "lightblue")
abline(v = mean(test_scores), col = "red", lwd = 2)
abline(v = median(test_scores), col = "blue", lwd = 2)
legend("topright", legend = c("Mean", "Median"), 
       col = c("red", "blue"), lwd = 2)
```

### Example 2: Income Data

```r
# Simulate income data (right-skewed)
set.seed(123)
income <- rlnorm(1000, meanlog = 10, sdlog = 0.5)

# Calculate measures
cat("Income Analysis:\n")
central_tendency_summary(income)

# Compare mean and median for skewed data
cat("Mean vs Median difference:", mean(income) - median(income), "\n")
```

### Example 3: Temperature Data

```r
# Create temperature data
temperatures <- c(15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42)

# Calculate measures
cat("Temperature Analysis:\n")
central_tendency_summary(temperatures)

# Seasonal analysis
winter_temps <- temperatures[1:3]
summer_temps <- temperatures[10:12]

cat("\nWinter temperatures:\n")
central_tendency_summary(winter_temps)

cat("\nSummer temperatures:\n")
central_tendency_summary(summer_temps)
```

## Using Built-in Functions

```r
# Load required packages
library(dplyr)
library(psych)

# Comprehensive summary
describe(mtcars$mpg)

# Summary by group
mtcars %>%
  group_by(cyl) %>%
  summarise(
    n = n(),
    mean_mpg = mean(mpg),
    median_mpg = median(mpg),
    sd_mpg = sd(mpg),
    min_mpg = min(mpg),
    max_mpg = max(mpg)
  )
```

## Best Practices

### When to Use Each Measure

```r
# Guidelines for choosing measures
cat("Guidelines for Measures of Central Tendency:\n")
cat("1. Mean: Use for symmetric, continuous data\n")
cat("2. Median: Use for skewed data or data with outliers\n")
cat("3. Mode: Use for categorical data or discrete data\n")
cat("4. Geometric Mean: Use for rates, ratios, growth rates\n")
cat("5. Harmonic Mean: Use for rates, speeds, ratios\n")
cat("6. Trimmed Mean: Use when outliers are present\n")
```

### Handling Missing Data

```r
# Function to handle missing data
robust_central_tendency <- function(x) {
  # Remove missing values
  x_clean <- x[!is.na(x)]
  
  if (length(x_clean) == 0) {
    return(NA)
  }
  
  # Calculate measures
  result <- list(
    mean = mean(x_clean),
    median = median(x_clean),
    mode = find_mode(x_clean),
    n = length(x_clean),
    missing = sum(is.na(x))
  )
  
  return(result)
}

# Test with missing data
data_with_missing <- c(1, 2, NA, 4, 5, NA, 7, 8)
robust_central_tendency(data_with_missing)
```

## Exercises

### Exercise 1: Basic Calculations
Create a dataset with 10 values and calculate the mean, median, and mode. Compare the results.

### Exercise 2: Outlier Impact
Create a dataset, then add an outlier and observe how it affects the mean vs. median.

### Exercise 3: Group Analysis
Using the `iris` dataset, calculate the mean and median of sepal length by species.

### Exercise 4: Robust Measures
Create a dataset with outliers and compare the regular mean, trimmed mean, and winsorized mean.

### Exercise 5: Real-world Application
Find a real dataset (or use built-in data) and perform a comprehensive central tendency analysis.

## Next Steps

In the next chapter, we'll learn about measures of variability, which complement central tendency measures to give us a complete picture of our data.

---

**Key Takeaways:**
- Mean is sensitive to outliers, median is robust
- Mode is best for categorical data
- Geometric mean for rates and ratios
- Harmonic mean for rates and speeds
- Always consider the data type and distribution
- Use multiple measures for comprehensive analysis
- Handle missing data appropriately 