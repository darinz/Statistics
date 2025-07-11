# Measures of Central Tendency

## Overview

Measures of central tendency describe the center or typical value of a dataset. They help us understand what the "average" or "typical" value is in our data. These measures are fundamental to statistical analysis and provide the foundation for understanding data distributions.

### Why Central Tendency Matters

- **Data Summarization**: Provides a single value that represents the entire dataset
- **Comparison**: Allows comparison between different groups or datasets
- **Decision Making**: Helps in making informed decisions based on typical values
- **Statistical Inference**: Forms the basis for more advanced statistical methods
- **Communication**: Simplifies complex data for stakeholders

### Types of Central Tendency Measures

1. **Mean**: Arithmetic average, most common measure
2. **Median**: Middle value, robust to outliers
3. **Mode**: Most frequent value, useful for categorical data
4. **Geometric Mean**: For rates and ratios
5. **Harmonic Mean**: For rates and speeds
6. **Trimmed Mean**: Robust version of mean
7. **Winsorized Mean**: Another robust alternative

## Mean (Arithmetic Mean)

The arithmetic mean is the sum of all values divided by the number of values. It's the most commonly used measure of central tendency.

### Mathematical Foundation

**Population Mean**:
```math
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i
```

**Sample Mean**:
```math
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
```

**Properties of the Mean**:
- The sum of deviations from the mean equals zero: $\sum_{i=1}^{n} (x_i - \bar{x}) = 0$
- The mean minimizes the sum of squared deviations: $\sum_{i=1}^{n} (x_i - \bar{x})^2$ is minimized
- The mean is affected by every value in the dataset
- The mean is sensitive to outliers

### Basic Mean Calculation

```r
# Create sample data
data <- c(2, 4, 6, 8, 10)

# Calculate mean
mean_value <- mean(data)
cat("Mean:", mean_value, "\n")

# Verify calculation manually
manual_mean <- sum(data) / length(data)
cat("Manual calculation:", manual_mean, "\n")

# Mean with missing values
data_with_na <- c(2, 4, NA, 8, 10)
mean_with_na <- mean(data_with_na, na.rm = TRUE)
cat("Mean with NA removed:", mean_with_na, "\n")

# Understanding the effect of outliers
data_with_outlier <- c(2, 4, 6, 8, 10, 100)
cat("Original mean:", mean(data), "\n")
cat("Mean with outlier:", mean(data_with_outlier), "\n")
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
  summarise(
    mean_mpg = mean(mpg),
    count = n(),
    sd_mpg = sd(mpg),
    se_mpg = sd(mpg) / sqrt(n())  # Standard error
  )

print(mpg_by_cyl)

# Visualize means by group
library(ggplot2)
ggplot(mpg_by_cyl, aes(x = factor(cyl), y = mean_mpg)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  geom_errorbar(aes(ymin = mean_mpg - se_mpg, ymax = mean_mpg + se_mpg), 
                width = 0.2) +
  labs(title = "Mean MPG by Number of Cylinders",
       x = "Number of Cylinders",
       y = "Mean MPG") +
  theme_minimal()
```

### Weighted Mean

The weighted mean is useful when different observations have different importance or reliability.

**Mathematical Formula**:
```math
\bar{x}_w = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}
```

```r
# Create data with weights
values <- c(85, 92, 78, 96, 88)
weights <- c(0.2, 0.3, 0.1, 0.25, 0.15)

# Calculate weighted mean
weighted_mean <- weighted.mean(values, weights)
cat("Weighted mean:", weighted_mean, "\n")

# Verify calculation manually
manual_weighted_mean <- sum(values * weights) / sum(weights)
cat("Manual calculation:", manual_weighted_mean, "\n")

# Example: Course grades with different weights
course_grades <- data.frame(
  assignment = c("Homework", "Midterm", "Final", "Participation"),
  grade = c(85, 78, 92, 95),
  weight = c(0.3, 0.25, 0.35, 0.1)
)

course_weighted_mean <- weighted.mean(course_grades$grade, course_grades$weight)
cat("Course weighted mean:", course_weighted_mean, "\n")
```

## Median

The median is the middle value when data is ordered from smallest to largest. It's robust to outliers and represents the 50th percentile.

### Mathematical Foundation

**For odd number of observations**:
```math
\text{Median} = x_{\left(\frac{n+1}{2}\right)}
```

**For even number of observations**:
```math
\text{Median} = \frac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)}}{2}
```

**Properties of the Median**:
- Resistant to outliers
- Minimizes the sum of absolute deviations: $\sum_{i=1}^{n} |x_i - \text{median}|$ is minimized
- Represents the 50th percentile
- Not affected by extreme values

### Basic Median Calculation

```r
# Calculate median
data <- c(1, 3, 5, 7, 9)
median_value <- median(data)
cat("Median:", median_value, "\n")

# Median with even number of observations
data_even <- c(1, 3, 5, 7, 9, 11)
median_even <- median(data_even)
cat("Median (even n):", median_even, "\n")

# Verify calculation for odd n
sorted_data <- sort(data)
n <- length(data)
manual_median_odd <- sorted_data[(n + 1) / 2]
cat("Manual median (odd):", manual_median_odd, "\n")

# Verify calculation for even n
sorted_even <- sort(data_even)
n_even <- length(data_even)
manual_median_even <- (sorted_even[n_even/2] + sorted_even[n_even/2 + 1]) / 2
cat("Manual median (even):", manual_median_even, "\n")

# Median with missing values
data_with_na <- c(1, 3, NA, 7, 9)
median_with_na <- median(data_with_na, na.rm = TRUE)
cat("Median with NA removed:", median_with_na, "\n")
```

### Median by Group

```r
# Median MPG by cylinders
median_by_cyl <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    median_mpg = median(mpg),
    count = n(),
    q25 = quantile(mpg, 0.25),
    q75 = quantile(mpg, 0.75)
  )

print(median_by_cyl)

# Compare mean vs median by group
comparison_by_cyl <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    mean_mpg = mean(mpg),
    median_mpg = median(mpg),
    difference = mean_mpg - median_mpg,
    count = n()
  )

print(comparison_by_cyl)

# Visualize mean vs median
ggplot(comparison_by_cyl, aes(x = factor(cyl))) +
  geom_point(aes(y = mean_mpg, color = "Mean"), size = 3) +
  geom_point(aes(y = median_mpg, color = "Median"), size = 3) +
  labs(title = "Mean vs Median MPG by Cylinders",
       x = "Number of Cylinders",
       y = "MPG",
       color = "Measure") +
  theme_minimal()
```

## Mode

The mode is the most frequently occurring value in a dataset. It's the only measure of central tendency that can be used with categorical data.

### Mathematical Foundation

**Mode Definition**:
```math
\text{Mode} = \arg\max_{x} f(x)
```

where $f(x)$ is the frequency of value $x$.

**Properties of the Mode**:
- Can be used with any data type (numeric, categorical, ordinal)
- May not be unique (bimodal, multimodal)
- May not exist (uniform distribution)
- Not affected by extreme values
- Represents the most common value

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
cat("Mode:", mode_value, "\n")

# Using table to see frequency
freq_table <- table(data)
print(freq_table)

# Find all modes (for multimodal data)
find_all_modes <- function(x) {
  freq_table <- table(x)
  max_freq <- max(freq_table)
  modes <- names(freq_table[freq_table == max_freq])
  return(as.numeric(modes))
}

# Test with multimodal data
multimodal_data <- c(1, 2, 2, 3, 4, 4, 5)
all_modes <- find_all_modes(multimodal_data)
cat("All modes:", all_modes, "\n")
```

### Mode for Categorical Data

```r
# Create categorical data
colors <- c("red", "blue", "red", "green", "blue", "red", "yellow")
mode_color <- find_mode(colors)
cat("Mode color:", mode_color, "\n")

# Frequency table
color_freq <- table(colors)
print(color_freq)

# Visualize frequency distribution
color_df <- data.frame(
  color = names(color_freq),
  frequency = as.numeric(color_freq)
)

ggplot(color_df, aes(x = color, y = frequency)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  labs(title = "Frequency Distribution of Colors",
       x = "Color",
       y = "Frequency") +
  theme_minimal()
```

## Geometric Mean

The geometric mean is useful for data that represents rates of change, ratios, or multiplicative relationships.

### Mathematical Foundation

**Geometric Mean Formula**:
```math
GM = \sqrt[n]{x_1 \times x_2 \times \cdots \times x_n} = \left(\prod_{i=1}^{n} x_i\right)^{\frac{1}{n}}
```

**Logarithmic Form**:
```math
GM = \exp\left(\frac{1}{n}\sum_{i=1}^{n} \ln(x_i)\right)
```

**Properties of Geometric Mean**:
- Always less than or equal to arithmetic mean
- Useful for rates of change and ratios
- Sensitive to zeros and negative values
- Multiplicative equivalent of arithmetic mean

```r
# Function to calculate geometric mean
geometric_mean <- function(x, na.rm = TRUE) {
  if (na.rm) x <- x[!is.na(x)]
  if (any(x <= 0)) {
    warning("Geometric mean requires positive values")
    return(NA)
  }
  exp(mean(log(x)))
}

# Example: Growth rates
growth_rates <- c(1.05, 1.08, 1.12, 1.06)
gm_growth <- geometric_mean(growth_rates)
cat("Geometric mean of growth rates:", gm_growth, "\n")

# Verify with built-in function
manual_gm <- exp(mean(log(growth_rates)))
cat("Manual calculation:", manual_gm, "\n")

# Compare with arithmetic mean
am_growth <- mean(growth_rates)
cat("Arithmetic mean:", am_growth, "\n")
cat("Geometric mean:", gm_growth, "\n")

# Example: Investment returns
returns <- c(0.05, 0.08, -0.02, 0.12, 0.06)
# Convert to growth factors
growth_factors <- 1 + returns
gm_return <- geometric_mean(growth_factors) - 1
cat("Geometric mean return:", gm_return, "\n")
```

## Harmonic Mean

The harmonic mean is useful for rates, speeds, and situations involving reciprocals.

### Mathematical Foundation

**Harmonic Mean Formula**:
```math
HM = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}
```

**Properties of Harmonic Mean**:
- Always less than or equal to geometric mean
- Useful for rates and speeds
- Sensitive to small values
- Reciprocal of arithmetic mean of reciprocals

```r
# Function to calculate harmonic mean
harmonic_mean <- function(x, na.rm = TRUE) {
  if (na.rm) x <- x[!is.na(x)]
  if (any(x <= 0)) {
    warning("Harmonic mean requires positive values")
    return(NA)
  }
  length(x) / sum(1/x)
}

# Example: Average speed
speeds <- c(60, 40, 80)  # km/h
hm_speed <- harmonic_mean(speeds)
cat("Harmonic mean speed:", hm_speed, "km/h\n")

# Verify calculation
manual_hm <- 3 / (1/60 + 1/40 + 1/80)
cat("Manual calculation:", manual_hm, "km/h\n")

# Compare with arithmetic mean
am_speed <- mean(speeds)
cat("Arithmetic mean speed:", am_speed, "km/h\n")
cat("Harmonic mean speed:", hm_speed, "km/h\n")

# Example: Parallel resistors
resistors <- c(10, 20, 30)  # ohms
hm_resistance <- harmonic_mean(resistors)
cat("Equivalent parallel resistance:", hm_resistance, "ohms\n")
```

## Trimmed Mean

The trimmed mean removes a percentage of extreme values before calculating the mean, making it more robust to outliers.

### Mathematical Foundation

**Trimmed Mean Formula**:
```math
\bar{x}_{\alpha} = \frac{1}{n(1-2\alpha)}\sum_{i=\alpha n + 1}^{(1-\alpha)n} x_{(i)}
```

where $\alpha$ is the trimming proportion and $x_{(i)}$ are the ordered values.

**Properties of Trimmed Mean**:
- More robust than arithmetic mean
- Less sensitive to outliers
- Maintains some efficiency for normal data
- Common trimming levels: 5%, 10%, 20%

```r
# Calculate trimmed mean
data <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 100)

# 10% trimmed mean (removes 10% from each end)
trimmed_mean <- mean(data, trim = 0.1)
cat("10% trimmed mean:", trimmed_mean, "\n")

# Compare with regular mean
regular_mean <- mean(data)
cat("Regular mean:", regular_mean, "\n")

# Manual calculation of 10% trimmed mean
sorted_data <- sort(data)
n <- length(data)
trim_n <- round(n * 0.1)
trimmed_data <- sorted_data[(trim_n + 1):(n - trim_n)]
manual_trimmed_mean <- mean(trimmed_data)
cat("Manual trimmed mean:", manual_trimmed_mean, "\n")

# Compare different trimming levels
trim_levels <- c(0, 0.05, 0.1, 0.2)
trimmed_means <- sapply(trim_levels, function(trim) mean(data, trim = trim))
names(trimmed_means) <- paste0(trim_levels * 100, "%")
print(trimmed_means)
```

## Winsorized Mean

The winsorized mean replaces extreme values with less extreme values rather than removing them.

### Mathematical Foundation

**Winsorized Mean Formula**:
```math
\bar{x}_w = \frac{1}{n}\left(k \cdot x_{(k+1)} + \sum_{i=k+1}^{n-k} x_{(i)} + k \cdot x_{(n-k)}\right)
```

where $k$ is the number of values to be winsorized from each end.

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

# Compare different robust measures
cat("Original data:", data, "\n")
cat("Arithmetic mean:", mean(data), "\n")
cat("Median:", median(data), "\n")
cat("10% trimmed mean:", mean(data, trim = 0.1), "\n")
cat("Winsorized mean (k=1):", winsorized_mean(data, k = 1), "\n")
```

## Comparing Measures of Central Tendency

```r
# Create data with outliers
data_with_outliers <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 100)

# Calculate different measures
mean_val <- mean(data_with_outliers)
median_val <- median(data_with_outliers)
mode_val <- find_mode(data_with_outliers)
geom_mean <- geometric_mean(data_with_outliers)
harm_mean <- harmonic_mean(data_with_outliers)
trimmed_mean <- mean(data_with_outliers, trim = 0.1)
winsorized_mean_val <- winsorized_mean(data_with_outliers, k = 1)

# Compare results
cat("Measures of Central Tendency:\n")
cat("Mean:", mean_val, "\n")
cat("Median:", median_val, "\n")
cat("Mode:", mode_val, "\n")
cat("Geometric Mean:", geom_mean, "\n")
cat("Harmonic Mean:", harm_mean, "\n")
cat("10% Trimmed Mean:", trimmed_mean, "\n")
cat("Winsorized Mean:", winsorized_mean_val, "\n")

# Create comprehensive summary function
central_tendency_summary <- function(x) {
  cat("=== CENTRAL TENDENCY SUMMARY ===\n")
  cat("Data length:", length(x), "\n")
  cat("Missing values:", sum(is.na(x)), "\n\n")
  
  cat("Measures of Central Tendency:\n")
  cat("Mean:", mean(x, na.rm = TRUE), "\n")
  cat("Median:", median(x, na.rm = TRUE), "\n")
  cat("Mode:", find_mode(x), "\n")
  cat("Geometric Mean:", geometric_mean(x), "\n")
  cat("Harmonic Mean:", harmonic_mean(x), "\n")
  cat("10% Trimmed Mean:", mean(x, trim = 0.1, na.rm = TRUE), "\n")
  cat("Winsorized Mean:", winsorized_mean(x, k = 1), "\n\n")
  
  # Compare mean and median
  mean_val <- mean(x, na.rm = TRUE)
  median_val <- median(x, na.rm = TRUE)
  cat("Mean - Median:", mean_val - median_val, "\n")
  
  if (abs(mean_val - median_val) > 0.1 * mean_val) {
    cat("Note: Large difference suggests skewed distribution\n")
  }
}

# Apply to mtcars MPG
central_tendency_summary(mtcars$mpg)
```

## Mathematical Relationships

### Inequality Relationships

For positive data, the following relationship holds:
```math
\text{Harmonic Mean} \leq \text{Geometric Mean} \leq \text{Arithmetic Mean}
```

This is known as the **Arithmetic Mean-Geometric Mean-Harmonic Mean Inequality**.

```r
# Demonstrate the inequality
positive_data <- c(2, 4, 8, 16)

hm <- harmonic_mean(positive_data)
gm <- geometric_mean(positive_data)
am <- mean(positive_data)

cat("Harmonic Mean:", hm, "\n")
cat("Geometric Mean:", gm, "\n")
cat("Arithmetic Mean:", am, "\n")

cat("Inequality holds:", hm <= gm && gm <= am, "\n")
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
     xlab = "Score", col = "lightblue", border = "white")
abline(v = mean(test_scores), col = "red", lwd = 2, lty = 2)
abline(v = median(test_scores), col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c("Mean", "Median"), 
       col = c("red", "blue"), lwd = 2, lty = 2)

# Analyze by grade level
grade_analysis <- function(scores) {
  cat("Grade Level Analysis:\n")
  cat("A (90+):", mean(scores[scores >= 90]), "\n")
  cat("B (80-89):", mean(scores[scores >= 80 & scores < 90]), "\n")
  cat("C (70-79):", mean(scores[scores >= 70 & scores < 80]), "\n")
  cat("D (60-69):", mean(scores[scores >= 60 & scores < 70]), "\n")
}

grade_analysis(test_scores)
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
cat("Skewness indicator:", ifelse(mean(income) > median(income), "Right-skewed", "Left-skewed"), "\n")

# Visualize with log scale
hist(income, main = "Distribution of Income (Log Scale)", 
     xlab = "Income", col = "lightgreen", border = "white", breaks = 50)
abline(v = mean(income), col = "red", lwd = 2, lty = 2)
abline(v = median(income), col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c("Mean", "Median"), 
       col = c("red", "blue"), lwd = 2, lty = 2)

# Income percentiles
percentiles <- quantile(income, c(0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
cat("Income Percentiles:\n")
print(percentiles)
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

# Temperature range analysis
temp_range <- max(temperatures) - min(temperatures)
cat("Temperature range:", temp_range, "degrees\n")
cat("Mean temperature:", mean(temperatures), "degrees\n")
cat("Temperature variability:", sd(temperatures), "degrees\n")
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
    mode_mpg = find_mode(mpg),
    sd_mpg = sd(mpg),
    min_mpg = min(mpg),
    max_mpg = max(mpg),
    q25 = quantile(mpg, 0.25),
    q75 = quantile(mpg, 0.75)
  )

# Robust summary statistics
library(robustbase)
mc(mtcars$mpg)  # Median absolute deviation
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
cat("7. Winsorized Mean: Use when outliers are present but you want to keep all data\n")

# Decision tree function
choose_central_tendency <- function(x) {
  cat("=== CENTRAL TENDENCY DECISION TREE ===\n")
  
  # Check for missing values
  if (sum(is.na(x)) > 0) {
    cat("Data contains missing values - use na.rm = TRUE\n")
  }
  
  # Check data type
  if (is.factor(x) || is.character(x)) {
    cat("Recommendation: Use MODE (categorical data)\n")
    return("Mode")
  }
  
  # Check for outliers
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  outliers <- sum(x < (q1 - 1.5 * iqr) | x > (q3 + 1.5 * iqr), na.rm = TRUE)
  
  if (outliers > 0) {
    cat("Outliers detected:", outliers, "\n")
    cat("Recommendation: Use MEDIAN or TRIMMED MEAN\n")
    return("Median")
  }
  
  # Check for skewness
  mean_val <- mean(x, na.rm = TRUE)
  median_val <- median(x, na.rm = TRUE)
  skewness_indicator <- abs(mean_val - median_val) / mean_val
  
  if (skewness_indicator > 0.1) {
    cat("Skewed distribution detected\n")
    cat("Recommendation: Use MEDIAN\n")
    return("Median")
  } else {
    cat("Symmetric distribution\n")
    cat("Recommendation: Use MEAN\n")
    return("Mean")
  }
}

# Test the decision function
choose_central_tendency(mtcars$mpg)
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
    geometric_mean = geometric_mean(x_clean),
    harmonic_mean = harmonic_mean(x_clean),
    trimmed_mean = mean(x_clean, trim = 0.1),
    winsorized_mean = winsorized_mean(x_clean, k = 1),
    n = length(x_clean),
    missing = sum(is.na(x)),
    missing_percent = sum(is.na(x)) / length(x) * 100
  )
  
  return(result)
}

# Test with missing data
data_with_missing <- c(1, 2, NA, 4, 5, NA, 7, 8)
result <- robust_central_tendency(data_with_missing)
print(result)
```

## Exercises

### Exercise 1: Basic Calculations
Create a dataset with 10 values and calculate the mean, median, and mode. Compare the results.

```r
# Your solution here
# Create dataset
my_data <- c(3, 7, 2, 9, 5, 7, 8, 4, 6, 7)

# Calculate measures
cat("Dataset:", my_data, "\n")
cat("Mean:", mean(my_data), "\n")
cat("Median:", median(my_data), "\n")
cat("Mode:", find_mode(my_data), "\n")

# Compare results
cat("Mean - Median:", mean(my_data) - median(my_data), "\n")
```

### Exercise 2: Outlier Impact
Create a dataset, then add an outlier and observe how it affects the mean vs. median.

```r
# Your solution here
# Original dataset
original_data <- c(10, 12, 15, 18, 20, 22, 25, 28, 30, 32)

# Add outlier
data_with_outlier <- c(original_data, 100)

# Compare measures
cat("Original data measures:\n")
cat("Mean:", mean(original_data), "\n")
cat("Median:", median(original_data), "\n")

cat("Data with outlier measures:\n")
cat("Mean:", mean(data_with_outlier), "\n")
cat("Median:", median(data_with_outlier), "\n")

cat("Change in mean:", mean(data_with_outlier) - mean(original_data), "\n")
cat("Change in median:", median(data_with_outlier) - median(original_data), "\n")
```

### Exercise 3: Group Analysis
Using the `iris` dataset, calculate the mean and median of sepal length by species.

```r
# Your solution here
data(iris)

# Group analysis
iris_summary <- iris %>%
  group_by(Species) %>%
  summarise(
    n = n(),
    mean_sepal_length = mean(Sepal.Length),
    median_sepal_length = median(Sepal.Length),
    sd_sepal_length = sd(Sepal.Length)
  )

print(iris_summary)

# Visualize
ggplot(iris, aes(x = Species, y = Sepal.Length)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "red") +
  labs(title = "Sepal Length by Species",
       subtitle = "Box shows median, red point shows mean") +
  theme_minimal()
```

### Exercise 4: Robust Measures
Create a dataset with outliers and compare the regular mean, trimmed mean, and winsorized mean.

```r
# Your solution here
# Create dataset with outliers
robust_data <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100)

# Compare robust measures
cat("Dataset:", robust_data, "\n")
cat("Regular mean:", mean(robust_data), "\n")
cat("Median:", median(robust_data), "\n")
cat("10% trimmed mean:", mean(robust_data, trim = 0.1), "\n")
cat("20% trimmed mean:", mean(robust_data, trim = 0.2), "\n")
cat("Winsorized mean (k=1):", winsorized_mean(robust_data, k = 1), "\n")
cat("Winsorized mean (k=2):", winsorized_mean(robust_data, k = 2), "\n")
```

### Exercise 5: Real-world Application
Find a real dataset (or use built-in data) and perform a comprehensive central tendency analysis.

```r
# Your solution here
# Use built-in airquality dataset
data(airquality)

# Comprehensive analysis of Ozone
cat("Ozone Analysis:\n")
central_tendency_summary(airquality$Ozone)

# Analysis by month
airquality %>%
  group_by(Month) %>%
  summarise(
    n = n(),
    mean_ozone = mean(Ozone, na.rm = TRUE),
    median_ozone = median(Ozone, na.rm = TRUE),
    sd_ozone = sd(Ozone, na.rm = TRUE)
  )

# Visualize
ggplot(airquality, aes(x = factor(Month), y = Ozone)) +
  geom_boxplot(fill = "lightgreen", alpha = 0.7) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "red") +
  labs(title = "Ozone Levels by Month",
       x = "Month",
       y = "Ozone") +
  theme_minimal()
```

## Next Steps

In the next chapter, we'll learn about measures of variability, which complement central tendency measures to give us a complete picture of our data. We'll cover:

- **Range and Interquartile Range**: Measures of spread
- **Variance and Standard Deviation**: Most common measures of variability
- **Coefficient of Variation**: Relative measure of variability
- **Skewness and Kurtosis**: Measures of distribution shape
- **Robust Measures of Variability**: Median absolute deviation

---

**Key Takeaways:**
- Mean is sensitive to outliers, median is robust
- Mode is best for categorical data
- Geometric mean for rates and ratios
- Harmonic mean for rates and speeds
- Always consider the data type and distribution
- Use multiple measures for comprehensive analysis
- Handle missing data appropriately
- Consider the context when choosing measures
- Outliers can significantly affect the mean
- Different measures serve different purposes 