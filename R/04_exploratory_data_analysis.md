# Exploratory Data Analysis

## Overview

Exploratory Data Analysis (EDA) is the first step in any statistical analysis. It involves examining and understanding your data through visualization and summary statistics before applying formal statistical methods. EDA is both an art and a science - it requires creativity in visualization and rigor in statistical thinking.

### The Philosophy of EDA

EDA follows these key principles:
- **Data-Driven Discovery**: Let the data guide your analysis, not preconceived notions
- **Iterative Process**: EDA is not linear; you may need to revisit earlier steps
- **Multiple Perspectives**: Use different techniques to understand the same data
- **Documentation**: Record your findings and decisions for reproducibility

### Why EDA Matters

- **Data Quality**: Identify issues before they affect your analysis
- **Pattern Recognition**: Discover relationships and trends in the data
- **Assumption Validation**: Check if your statistical assumptions are met
- **Hypothesis Generation**: EDA often leads to new research questions
- **Communication**: Visualizations help communicate findings to stakeholders

### The EDA Framework

1. **Data Overview**: Understand structure, size, and basic properties
2. **Data Quality Assessment**: Check for missing values, outliers, inconsistencies
3. **Univariate Analysis**: Examine individual variables in isolation
4. **Bivariate Analysis**: Explore relationships between pairs of variables
5. **Multivariate Analysis**: Understand complex interactions
6. **Data Transformation**: Apply transformations if needed
7. **Summary and Recommendations**: Document findings and next steps

## Data Exploration Fundamentals

### Initial Data Inspection

```r
# Load sample data
data(mtcars)

# Basic data structure
str(mtcars)        # Structure: data types and first few values
dim(mtcars)        # Dimensions: rows and columns
names(mtcars)      # Variable names

# Data types and summary
sapply(mtcars, class)  # Class of each variable
sapply(mtcars, typeof) # Internal type of each variable

# First few observations
head(mtcars, 10)   # First 10 rows
tail(mtcars, 5)    # Last 5 rows

# Memory usage
object.size(mtcars) # Size in bytes
format(object.size(mtcars), units = "MB") # Size in MB
```

### Understanding Data Types

Different data types require different analytical approaches:

- **Numeric (Continuous)**: Can take any value within a range
- **Numeric (Discrete)**: Can only take specific values (e.g., counts)
- **Categorical (Nominal)**: Categories with no natural order
- **Categorical (Ordinal)**: Categories with natural order
- **Binary**: Two possible values (0/1, TRUE/FALSE)
- **Date/Time**: Temporal data requiring special handling

### Summary Statistics

```r
# Comprehensive summary
summary(mtcars)

# Detailed summary for numeric variables
library(dplyr)
mtcars %>%
  select_if(is.numeric) %>%
  summary()

# Summary by groups
mtcars %>%
  group_by(am) %>%
  summarise(
    n = n(),                                    # Count
    mean_mpg = mean(mpg, na.rm = TRUE),        # Arithmetic mean
    sd_mpg = sd(mpg, na.rm = TRUE),            # Standard deviation
    median_mpg = median(mpg, na.rm = TRUE),    # Median
    min_mpg = min(mpg, na.rm = TRUE),          # Minimum
    max_mpg = max(mpg, na.rm = TRUE),          # Maximum
    q25 = quantile(mpg, 0.25, na.rm = TRUE),   # 25th percentile
    q75 = quantile(mpg, 0.75, na.rm = TRUE),   # 75th percentile
    iqr = q75 - q25                            # Interquartile range
  )

# Coefficient of variation (relative variability)
cv_mpg <- sd(mtcars$mpg, na.rm = TRUE) / mean(mtcars$mpg, na.rm = TRUE)
cat("Coefficient of variation for MPG:", round(cv_mpg, 3), "\n")
```

### Mathematical Foundation: Descriptive Statistics

**Arithmetic Mean**: The sum of all values divided by the number of values
```math
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
```

**Median**: The middle value when data is ordered
- For odd n: middle value
- For even n: average of two middle values

**Standard Deviation**: Measure of variability around the mean
```math
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2}
```

**Coefficient of Variation**: Relative measure of variability
```math
CV = \frac{s}{\bar{x}} \times 100\%
```

**Interquartile Range (IQR)**: Range containing the middle 50% of data
```math
IQR = Q_3 - Q_1
```

## Data Quality Assessment

### Understanding Data Quality Issues

Data quality problems can significantly impact analysis:

- **Missing Values**: Can bias results and reduce sample size
- **Outliers**: May represent errors or genuine extreme values
- **Inconsistencies**: Logical contradictions in the data
- **Duplicates**: Can inflate sample size and bias results
- **Data Type Issues**: Numbers stored as text, incorrect formats

### Missing Values Analysis

```r
# Check for missing values
missing_summary <- sapply(mtcars, function(x) sum(is.na(x)))
print(missing_summary)

# Calculate missing percentage
missing_percent <- (missing_summary / nrow(mtcars)) * 100

# Create missing data summary
missing_data <- data.frame(
  variable = names(missing_summary),
  missing_count = missing_summary,
  missing_percent = missing_percent,
  data_type = sapply(mtcars, class)
)

# Display missing data summary
print(missing_data[missing_data$missing_count > 0, ])

# Visualize missing data
library(ggplot2)
ggplot(missing_data, aes(x = reorder(variable, missing_percent), y = missing_percent)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Missing Values by Variable",
       x = "Variable",
       y = "Percentage Missing") +
  theme_minimal()

# Check for patterns in missing data
# Are missing values random or systematic?
missing_pattern <- is.na(mtcars)
missing_correlations <- cor(missing_pattern)
print(round(missing_correlations, 3))
```

### Duplicate Detection

```r
# Check for duplicate rows
duplicates <- sum(duplicated(mtcars))
cat("Number of duplicate rows:", duplicates, "\n")

# Check for duplicate values in specific columns
duplicate_mpg <- sum(duplicated(mtcars$mpg))
cat("Duplicate MPG values:", duplicate_mpg, "\n")

# Find duplicate rows
if (duplicates > 0) {
  duplicate_indices <- which(duplicated(mtcars))
  cat("Duplicate row indices:", duplicate_indices, "\n")
  
  # Show duplicate rows
  print(mtcars[duplicate_indices, ])
}

# Check for near-duplicates (similar but not identical)
# This is useful for detecting data entry errors
near_duplicates <- function(data, threshold = 0.95) {
  # Calculate similarity matrix (simplified)
  # In practice, you might use more sophisticated methods
  n_rows <- nrow(data)
  similarity_matrix <- matrix(0, n_rows, n_rows)
  
  for (i in 1:n_rows) {
    for (j in 1:n_rows) {
      if (i != j) {
        # Calculate similarity (simplified)
        similarity <- sum(data[i, ] == data[j, ]) / ncol(data)
        similarity_matrix[i, j] = similarity
      }
    }
  }
  
  # Find pairs with high similarity
  high_similarity <- which(similarity_matrix > threshold, arr.ind = TRUE)
  return(high_similarity)
}
```

### Outlier Detection

Outliers are data points that deviate significantly from the rest of the data. They can be:
- **True outliers**: Genuine extreme values
- **Errors**: Data entry mistakes
- **Influential points**: Extreme values that affect analysis

```r
# Function to detect outliers using multiple methods
detect_outliers <- function(x, method = "iqr") {
  if (method == "iqr") {
    # IQR method (most common)
    q1 <- quantile(x, 0.25, na.rm = TRUE)
    q3 <- quantile(x, 0.75, na.rm = TRUE)
    iqr <- q3 - q1
    lower_bound <- q1 - 1.5 * iqr
    upper_bound <- q3 + 1.5 * iqr
    
    outliers <- x < lower_bound | x > upper_bound
    
  } else if (method == "zscore") {
    # Z-score method
    z_scores <- abs((x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE))
    outliers <- z_scores > 3  # Values more than 3 standard deviations away
    
  } else if (method == "modified_zscore") {
    # Modified Z-score (more robust)
    median_val <- median(x, na.rm = TRUE)
    mad_val <- median(abs(x - median_val), na.rm = TRUE)
    modified_z_scores <- abs(0.6745 * (x - median_val) / mad_val)
    outliers <- modified_z_scores > 3.5
  }
  
  return(list(
    outliers = outliers,
    outlier_indices = which(outliers),
    outlier_values = x[outliers],
    method = method
  ))
}

# Apply to numeric variables
numeric_vars <- sapply(mtcars, is.numeric)
outlier_summary <- lapply(mtcars[, numeric_vars], detect_outliers)

# Display outlier summary
for (var in names(outlier_summary)) {
  n_outliers <- sum(outlier_summary[[var]]$outliers)
  if (n_outliers > 0) {
    cat(var, ":", n_outliers, "outliers detected\n")
    cat("  Outlier values:", outlier_summary[[var]]$outlier_values, "\n")
  }
}

# Compare different outlier detection methods
compare_outlier_methods <- function(x, var_name) {
  iqr_result <- detect_outliers(x, "iqr")
  zscore_result <- detect_outliers(x, "zscore")
  modified_z_result <- detect_outliers(x, "modified_zscore")
  
  cat("=== OUTLIER DETECTION COMPARISON FOR", var_name, "===\n")
  cat("IQR method:", sum(iqr_result$outliers), "outliers\n")
  cat("Z-score method:", sum(zscore_result$outliers), "outliers\n")
  cat("Modified Z-score method:", sum(modified_z_result$outliers), "outliers\n")
  
  return(list(
    iqr = iqr_result,
    zscore = zscore_result,
    modified_zscore = modified_z_result
  ))
}

# Compare methods for MPG
mpg_outlier_comparison <- compare_outlier_methods(mtcars$mpg, "MPG")
```

### Mathematical Foundation: Outlier Detection

**IQR Method**: Based on quartiles
```math
\text{Lower bound} = Q_1 - 1.5 \times IQR
\text{Upper bound} = Q_3 + 1.5 \times IQR
```

**Z-Score Method**: Based on standard deviations
```math
Z = \frac{x - \mu}{\sigma}
```
Values with |Z| > 3 are considered outliers.

**Modified Z-Score**: More robust to extreme values
```math
M = \frac{0.6745(x - \text{median})}{\text{MAD}}
```
where MAD is the Median Absolute Deviation.

## Univariate Analysis

### Distribution Analysis

Understanding the distribution of a variable is crucial for choosing appropriate statistical methods.

```r
# Function to analyze distribution of a variable
analyze_distribution <- function(x, var_name) {
  cat("=== DISTRIBUTION ANALYSIS FOR", var_name, "===\n")
  
  # Basic statistics
  mean_val <- mean(x, na.rm = TRUE)
  median_val <- median(x, na.rm = TRUE)
  sd_val <- sd(x, na.rm = TRUE)
  
  cat("Mean:", round(mean_val, 3), "\n")
  cat("Median:", round(median_val, 3), "\n")
  cat("Standard Deviation:", round(sd_val, 3), "\n")
  
  # Skewness and kurtosis
  library(moments)
  skewness_val <- skewness(x, na.rm = TRUE)
  kurtosis_val <- kurtosis(x, na.rm = TRUE)
  
  cat("Skewness:", round(skewness_val, 3), "\n")
  cat("Kurtosis:", round(kurtosis_val, 3), "\n")
  
  # Interpret skewness
  if (abs(skewness_val) < 0.5) {
    cat("Distribution is approximately symmetric\n")
  } else if (skewness_val > 0.5) {
    cat("Distribution is right-skewed (positive skew)\n")
  } else {
    cat("Distribution is left-skewed (negative skew)\n")
  }
  
  # Interpret kurtosis
  if (kurtosis_val < 3) {
    cat("Distribution has lighter tails than normal (platykurtic)\n")
  } else if (kurtosis_val > 3) {
    cat("Distribution has heavier tails than normal (leptokurtic)\n")
  } else {
    cat("Distribution has normal-like tails (mesokurtic)\n")
  }
  
  # Quantiles
  cat("Quantiles:\n")
  quantiles <- quantile(x, probs = c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE)
  print(round(quantiles, 3))
  
  # Normality test
  shapiro_test <- shapiro.test(x)
  cat("Shapiro-Wilk normality test p-value:", round(shapiro_test$p.value, 4), "\n")
  
  if (shapiro_test$p.value < 0.05) {
    cat("Data is NOT normally distributed (p < 0.05)\n")
  } else {
    cat("Data appears to be normally distributed (p >= 0.05)\n")
  }
  
  return(list(
    mean = mean_val,
    median = median_val,
    sd = sd_val,
    skewness = skewness_val,
    kurtosis = kurtosis_val,
    shapiro_p = shapiro_test$p.value
  ))
}

# Apply to MPG
mpg_analysis <- analyze_distribution(mtcars$mpg, "MPG")
```

### Mathematical Foundation: Distribution Characteristics

**Skewness**: Measures asymmetry of the distribution
```math
\text{Skewness} = \frac{E[(X - \mu)^3]}{\sigma^3}
```

**Kurtosis**: Measures the "tailedness" of the distribution
```math
\text{Kurtosis} = \frac{E[(X - \mu)^4]}{\sigma^4}
```

**Shapiro-Wilk Test**: Tests for normality
```math
W = \frac{(\sum_{i=1}^{n} a_i x_{(i)})^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
```

### Visualization for Univariate Data

```r
# Create comprehensive univariate plots
create_univariate_plots <- function(data, variable) {
  # Histogram with density curve
  p1 <- ggplot(data, aes_string(x = variable)) +
    geom_histogram(aes(y = ..density..), bins = 15, fill = "steelblue", alpha = 0.7) +
    geom_density(color = "red", size = 1) +
    geom_vline(aes(xintercept = mean(!!sym(variable), na.rm = TRUE)), 
               color = "green", linetype = "dashed", size = 1) +
    geom_vline(aes(xintercept = median(!!sym(variable), na.rm = TRUE)), 
               color = "orange", linetype = "dashed", size = 1) +
    labs(title = paste("Distribution of", variable),
         subtitle = "Green = Mean, Orange = Median") +
    theme_minimal()
  
  # Box plot
  p2 <- ggplot(data, aes_string(y = variable)) +
    geom_boxplot(fill = "lightgreen", alpha = 0.7) +
    geom_jitter(width = 0.2, alpha = 0.5) +
    labs(title = paste("Boxplot of", variable)) +
    theme_minimal()
  
  # Q-Q plot for normality assessment
  p3 <- ggplot(data, aes_string(sample = variable)) +
    stat_qq() +
    stat_qq_line() +
    labs(title = paste("Q-Q Plot of", variable),
         subtitle = "Points should follow the line for normality") +
    theme_minimal()
  
  # Cumulative distribution function
  p4 <- ggplot(data, aes_string(x = variable)) +
    stat_ecdf() +
    labs(title = paste("Cumulative Distribution of", variable)) +
    theme_minimal()
  
  # Combine plots
  library(gridExtra)
  grid.arrange(p1, p2, p3, p4, ncol = 2)
}

# Apply to MPG
create_univariate_plots(mtcars, "mpg")
```

### Understanding Distribution Shapes

**Normal Distribution**: Bell-shaped, symmetric
- Mean = Median = Mode
- 68% of data within ±1 standard deviation
- 95% of data within ±2 standard deviations

**Skewed Distributions**:
- **Right-skewed**: Long tail to the right, mean > median
- **Left-skewed**: Long tail to the left, mean < median

**Bimodal/Multimodal**: Multiple peaks, may indicate mixed populations

**Uniform Distribution**: All values equally likely

**Exponential Distribution**: Rapidly decreasing probability 

## Bivariate Analysis

### Correlation Analysis

Correlation measures the strength and direction of linear relationships between variables.

```r
# Correlation matrix for numeric variables
numeric_data <- mtcars[, sapply(mtcars, is.numeric)]
correlation_matrix <- cor(numeric_data, use = "complete.obs")

# Display correlation matrix
print(round(correlation_matrix, 3))

# Visualize correlation matrix
library(corrplot)
corrplot(correlation_matrix, method = "color", type = "upper", 
         order = "hclust", tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.7)

# Detailed correlation analysis
correlation_analysis <- function(data, var1, var2) {
  cat("=== CORRELATION ANALYSIS ===\n")
  cat("Variables:", var1, "and", var2, "\n")
  
  # Pearson correlation (parametric)
  pearson_cor <- cor.test(data[[var1]], data[[var2]], method = "pearson")
  cat("Pearson correlation:", round(pearson_cor$estimate, 3), "\n")
  cat("95% Confidence Interval:", round(pearson_cor$conf.int, 3), "\n")
  cat("p-value:", round(pearson_cor$p.value, 4), "\n")
  
  # Spearman correlation (nonparametric)
  spearman_cor <- cor.test(data[[var1]], data[[var2]], method = "spearman")
  cat("Spearman correlation:", round(spearman_cor$estimate, 3), "\n")
  cat("p-value:", round(spearman_cor$p.value, 4), "\n")
  
  # Kendall's tau (nonparametric, for ordinal data)
  kendall_cor <- cor.test(data[[var1]], data[[var2]], method = "kendall")
  cat("Kendall's tau:", round(kendall_cor$estimate, 3), "\n")
  cat("p-value:", round(kendall_cor$p.value, 4), "\n")
  
  # Interpret correlation strength
  pearson_abs <- abs(pearson_cor$estimate)
  if (pearson_abs < 0.1) {
    cat("Correlation strength: Negligible\n")
  } else if (pearson_abs < 0.3) {
    cat("Correlation strength: Weak\n")
  } else if (pearson_abs < 0.5) {
    cat("Correlation strength: Moderate\n")
  } else if (pearson_abs < 0.7) {
    cat("Correlation strength: Strong\n")
  } else {
    cat("Correlation strength: Very strong\n")
  }
  
  return(list(
    pearson = pearson_cor,
    spearman = spearman_cor,
    kendall = kendall_cor
  ))
}

# Analyze correlation between MPG and Weight
mpg_wt_correlation <- correlation_analysis(mtcars, "mpg", "wt")
```

### Mathematical Foundation: Correlation

**Pearson Correlation Coefficient**:
```math
r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}
```

**Spearman Rank Correlation**:
```math
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
```
where $d_i$ is the difference between ranks.

**Kendall's Tau**:
```math
\tau = \frac{n_c - n_d}{\frac{n(n-1)}{2}}
```
where $n_c$ is the number of concordant pairs and $n_d$ is the number of discordant pairs.

### Scatter Plot Analysis

```r
# Create comprehensive scatter plots
create_scatter_analysis <- function(data, x_var, y_var) {
  # Basic scatter plot with regression line
  p1 <- ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_point(alpha = 0.7, size = 2) +
    geom_smooth(method = "lm", se = TRUE, color = "red", fill = "red", alpha = 0.2) +
    labs(title = paste(y_var, "vs", x_var),
         subtitle = "Red line = Linear regression with 95% confidence interval") +
    theme_minimal()
  
  # Scatter plot with LOESS curve
  p2 <- ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_point(alpha = 0.7, size = 2) +
    geom_smooth(method = "loess", se = TRUE, color = "blue", fill = "blue", alpha = 0.2) +
    labs(title = paste(y_var, "vs", x_var, "(LOESS)"),
         subtitle = "Blue line = LOESS curve with 95% confidence interval") +
    theme_minimal()
  
  # Residual plot
  model <- lm(as.formula(paste(y_var, "~", x_var)), data = data)
  residuals_data <- data.frame(
    fitted = fitted(model),
    residuals = residuals(model),
    x = data[[x_var]]
  )
  
  p3 <- ggplot(residuals_data, aes(x = fitted, y = residuals)) +
    geom_point(alpha = 0.7) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    geom_smooth(method = "loess", se = TRUE, color = "blue", alpha = 0.2) +
    labs(title = "Residual Plot",
         subtitle = "Check for patterns in residuals") +
    theme_minimal()
  
  # Residuals vs predictor
  p4 <- ggplot(residuals_data, aes(x = x, y = residuals)) +
    geom_point(alpha = 0.7) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    geom_smooth(method = "loess", se = TRUE, color = "blue", alpha = 0.2) +
    labs(title = "Residuals vs Predictor",
         subtitle = "Check for heteroscedasticity") +
    theme_minimal()
  
  # Combine plots
  library(gridExtra)
  grid.arrange(p1, p2, p3, p4, ncol = 2)
  
  # Print model summary
  cat("=== LINEAR REGRESSION SUMMARY ===\n")
  print(summary(model))
  
  return(model)
}

# Apply to MPG vs Weight
mpg_wt_model <- create_scatter_analysis(mtcars, "wt", "mpg")
```

### Understanding Scatter Plots

**Patterns to Look For**:
- **Linear**: Points follow a straight line
- **Nonlinear**: Curved patterns (quadratic, exponential, etc.)
- **No relationship**: Random scatter
- **Clusters**: Groups of points
- **Outliers**: Points far from the main pattern

**Regression Assumptions**:
1. **Linearity**: Relationship is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed

## Multivariate Analysis

### Group Comparisons

Comparing groups helps understand how categorical variables affect continuous outcomes.

```r
# Compare groups using box plots
create_group_comparison <- function(data, group_var, response_var) {
  # Box plot by group
  p1 <- ggplot(data, aes_string(x = group_var, y = response_var, fill = group_var)) +
    geom_boxplot(alpha = 0.7) +
    geom_jitter(width = 0.2, alpha = 0.5) +
    labs(title = paste(response_var, "by", group_var)) +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Violin plot (shows full distribution)
  p2 <- ggplot(data, aes_string(x = group_var, y = response_var, fill = group_var)) +
    geom_violin(alpha = 0.7) +
    geom_boxplot(width = 0.2, alpha = 0.8) +
    labs(title = paste(response_var, "by", group_var, "(Violin Plot)")) +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Summary statistics by group
  group_summary <- data %>%
    group_by(!!sym(group_var)) %>%
    summarise(
      n = n(),
      mean = mean(!!sym(response_var), na.rm = TRUE),
      sd = sd(!!sym(response_var), na.rm = TRUE),
      median = median(!!sym(response_var), na.rm = TRUE),
      min = min(!!sym(response_var), na.rm = TRUE),
      max = max(!!sym(response_var), na.rm = TRUE),
      q25 = quantile(!!sym(response_var), 0.25, na.rm = TRUE),
      q75 = quantile(!!sym(response_var), 0.75, na.rm = TRUE)
    )
  
  print(group_summary)
  
  # Statistical test for group differences
  if (is.factor(data[[group_var]]) || length(unique(data[[group_var]])) <= 10) {
    # ANOVA for multiple groups
    if (length(unique(data[[group_var]])) > 2) {
      anova_result <- aov(as.formula(paste(response_var, "~", group_var)), data = data)
      cat("\n=== ANOVA RESULTS ===\n")
      print(summary(anova_result))
    } else {
      # t-test for two groups
      groups <- unique(data[[group_var]])
      group1_data <- data[data[[group_var]] == groups[1], response_var]
      group2_data <- data[data[[group_var]] == groups[2], response_var]
      
      t_test_result <- t.test(group1_data, group2_data)
      cat("\n=== T-TEST RESULTS ===\n")
      print(t_test_result)
    }
  }
  
  # Combine plots
  grid.arrange(p1, p2, ncol = 2)
}

# Compare MPG by transmission type
mtcars$am_factor <- factor(mtcars$am, labels = c("Automatic", "Manual"))
create_group_comparison(mtcars, "am_factor", "mpg")
```

### Mathematical Foundation: Group Comparisons

**T-Test Statistic**:
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
```

**ANOVA F-Statistic**:
```math
F = \frac{\text{Between-group variance}}{\text{Within-group variance}} = \frac{MSB}{MSW}
```

**Effect Size (Cohen's d)**:
```math
d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}
```
where $s_{pooled} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$

### Interaction Analysis

Interactions occur when the effect of one variable depends on the level of another variable.

```r
# Analyze interactions between variables
analyze_interactions <- function(data, var1, var2, response_var) {
  # Create interaction plot
  p1 <- ggplot(data, aes_string(x = var1, y = response_var, color = var2)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", se = TRUE) +
    labs(title = paste("Interaction:", response_var, "~", var1, "*", var2)) +
    theme_minimal()
  
  # Faceted plot
  p2 <- ggplot(data, aes_string(x = var1, y = response_var)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", se = TRUE) +
    facet_wrap(as.formula(paste("~", var2))) +
    labs(title = paste(response_var, "by", var1, "faceted by", var2)) +
    theme_minimal()
  
  # Statistical test for interaction
  model_formula <- as.formula(paste(response_var, "~", var1, "*", var2))
  interaction_model <- lm(model_formula, data = data)
  
  cat("=== INTERACTION ANALYSIS ===\n")
  print(summary(interaction_model))
  
  # Test for interaction significance
  no_interaction_formula <- as.formula(paste(response_var, "~", var1, "+", var2))
  no_interaction_model <- lm(no_interaction_formula, data = data)
  
  anova_result <- anova(no_interaction_model, interaction_model)
  cat("\n=== INTERACTION SIGNIFICANCE TEST ===\n")
  print(anova_result)
  
  # Combine plots
  grid.arrange(p1, p2, ncol = 1)
  
  return(interaction_model)
}

# Analyze interaction between weight and transmission on MPG
analyze_interactions(mtcars, "wt", "am_factor", "mpg")
```

### Understanding Interactions

**Types of Interactions**:
- **Additive**: Effects of variables simply add together
- **Multiplicative**: Effects multiply together
- **Antagonistic**: One variable reduces the effect of another
- **Synergistic**: Variables work together to enhance effects

**Visualizing Interactions**:
- **Parallel lines**: No interaction
- **Converging lines**: Interaction
- **Crossing lines**: Strong interaction

## Data Transformation

### Variable Transformations

Transformations can help meet statistical assumptions and improve model performance.

```r
# Function to apply common transformations
apply_transformations <- function(data, variable) {
  # Original variable
  original <- data[[variable]]
  
  # Log transformation (for right-skewed data)
  log_transformed <- log(original)
  
  # Square root transformation (for moderate right skew)
  sqrt_transformed <- sqrt(original)
  
  # Reciprocal transformation (for extreme right skew)
  reciprocal_transformed <- 1 / original
  
  # Box-Cox transformation (optimal transformation)
  library(MASS)
  bc_result <- boxcox(original ~ 1)
  lambda <- bc_result$x[which.max(bc_result$y)]
  
  if (abs(lambda) < 0.001) {
    bc_transformed <- log(original)
  } else {
    bc_transformed <- (original^lambda - 1) / lambda
  }
  
  # Yeo-Johnson transformation (handles negative values)
  library(bestNormalize)
  yj_result <- yeojohnson(original)
  yj_transformed <- yj_result$x.t
  
  # Create comparison plots
  transformed_data <- data.frame(
    Original = original,
    Log = log_transformed,
    Sqrt = sqrt_transformed,
    Reciprocal = reciprocal_transformed,
    BoxCox = bc_transformed,
    YeoJohnson = yj_transformed
  )
  
  # Plot distributions
  library(reshape2)
  melted_data <- melt(transformed_data)
  
  ggplot(melted_data, aes(x = value)) +
    geom_histogram(bins = 15, fill = "steelblue", alpha = 0.7) +
    facet_wrap(~variable, scales = "free") +
    labs(title = paste("Distribution of", variable, "under different transformations")) +
    theme_minimal()
  
  # Test normality for each transformation
  cat("=== NORMALITY TESTS ===\n")
  for (col in names(transformed_data)) {
    shapiro_result <- shapiro.test(transformed_data[[col]])
    cat(col, "Shapiro-Wilk p-value:", round(shapiro_result$p.value, 4), "\n")
  }
  
  # Return transformation results
  return(list(
    original = original,
    log = log_transformed,
    sqrt = sqrt_transformed,
    reciprocal = reciprocal_transformed,
    boxcox = bc_transformed,
    yeojohnson = yj_transformed,
    lambda = lambda
  ))
}

# Apply transformations to MPG
mpg_transformations <- apply_transformations(mtcars, "mpg")
```

### Mathematical Foundation: Transformations

**Log Transformation**:
```math
y' = \log(y)
```
Useful for right-skewed data, stabilizes variance.

**Square Root Transformation**:
```math
y' = \sqrt{y}
```
Less aggressive than log, good for count data.

**Box-Cox Transformation**:
```math
y' = \begin{cases}
\frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(y) & \text{if } \lambda = 0
\end{cases}
```

**Yeo-Johnson Transformation**:
```math
y' = \begin{cases}
\frac{(y+1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, y \geq 0 \\
\log(y+1) & \text{if } \lambda = 0, y \geq 0 \\
-\frac{(-y+1)^{2-\lambda} - 1}{2-\lambda} & \text{if } \lambda \neq 2, y < 0 \\
-\log(-y+1) & \text{if } \lambda = 2, y < 0
\end{cases}
```

### When to Transform

**Transform when**:
- Data is highly skewed
- Variance is not constant (heteroscedasticity)
- Residuals are not normal
- Relationships are nonlinear

**Common transformations**:
- **Right skew**: Log, square root, reciprocal
- **Left skew**: Square, cube, exponential
- **Count data**: Square root, log
- **Proportions**: Logit, arcsine 

## Data Quality Checks

### Consistency Checks

Data consistency ensures that your data makes logical sense and doesn't contain contradictions.

```r
# Function to check data consistency
check_data_consistency <- function(data) {
  cat("=== DATA CONSISTENCY CHECKS ===\n")
  
  # Check for logical inconsistencies
  if ("mpg" %in% names(data) && "wt" %in% names(data)) {
    # Check if heavier cars have lower MPG (general trend)
    correlation <- cor(data$mpg, data$wt, use = "complete.obs")
    cat("MPG-Weight correlation:", round(correlation, 3), "\n")
    
    if (correlation > 0) {
      cat("WARNING: Positive correlation between weight and MPG (unexpected)\n")
    }
  }
  
  # Check for extreme values
  numeric_vars <- sapply(data, is.numeric)
  for (var in names(data)[numeric_vars]) {
    values <- data[[var]]
    
    # Check for extreme values using percentiles
    extreme_low <- sum(values < quantile(values, 0.01, na.rm = TRUE), na.rm = TRUE)
    extreme_high <- sum(values > quantile(values, 0.99, na.rm = TRUE), na.rm = TRUE)
    
    if (extreme_low > 0 || extreme_high > 0) {
      cat(var, ":", extreme_low, "extreme low values,", extreme_high, "extreme high values\n")
    }
    
    # Check for impossible values
    if (var == "mpg" && any(values < 0, na.rm = TRUE)) {
      cat("WARNING:", var, "contains negative values (impossible for MPG)\n")
    }
    
    if (var == "wt" && any(values <= 0, na.rm = TRUE)) {
      cat("WARNING:", var, "contains zero or negative values (impossible for weight)\n")
    }
  }
  
  # Check for data type consistency
  for (var in names(data)) {
    if (is.factor(data[[var]])) {
      n_levels <- nlevels(data[[var]])
      if (n_levels > 20) {
        cat("WARNING:", var, "has", n_levels, "levels (consider if this is correct)\n")
      }
      
      # Check for levels with very few observations
      level_counts <- table(data[[var]])
      rare_levels <- level_counts[level_counts < 3]
      if (length(rare_levels) > 0) {
        cat("WARNING:", var, "has rare levels:", names(rare_levels), "\n")
      }
    }
  }
  
  # Check for duplicate observations
  duplicates <- sum(duplicated(data))
  if (duplicates > 0) {
    cat("WARNING:", duplicates, "duplicate rows found\n")
  }
  
  # Check for missing value patterns
  missing_pattern <- is.na(data)
  if (sum(missing_pattern) > 0) {
    missing_correlations <- cor(missing_pattern)
    high_missing_correlations <- which(abs(missing_correlations) > 0.7 & 
                                     abs(missing_correlations) < 1, arr.ind = TRUE)
    if (nrow(high_missing_correlations) > 0) {
      cat("WARNING: Missing values are correlated between variables\n")
    }
  }
}

# Apply consistency checks
check_data_consistency(mtcars)
```

### Data Validation Rules

```r
# Function to create and apply data validation rules
create_validation_rules <- function(data) {
  cat("=== DATA VALIDATION RULES ===\n")
  
  # Define validation rules
  rules <- list(
    mpg = list(
      min = 0,
      max = 50,
      description = "MPG should be between 0 and 50"
    ),
    wt = list(
      min = 0.5,
      max = 10,
      description = "Weight should be between 0.5 and 10 tons"
    ),
    cyl = list(
      valid_values = c(4, 6, 8),
      description = "Cylinders should be 4, 6, or 8"
    )
  )
  
  # Apply validation rules
  violations <- list()
  
  for (var in names(rules)) {
    if (var %in% names(data)) {
      rule <- rules[[var]]
      values <- data[[var]]
      
      if ("min" %in% names(rule) && "max" %in% names(rule)) {
        # Range validation
        violations[[var]] <- values < rule$min | values > rule$max
        n_violations <- sum(violations[[var]], na.rm = TRUE)
        if (n_violations > 0) {
          cat(var, ":", n_violations, "values outside range [", rule$min, ",", rule$max, "]\n")
        }
      } else if ("valid_values" %in% names(rule)) {
        # Value validation
        violations[[var]] <- !(values %in% rule$valid_values)
        n_violations <- sum(violations[[var]], na.rm = TRUE)
        if (n_violations > 0) {
          cat(var, ":", n_violations, "invalid values\n")
        }
      }
    }
  }
  
  return(violations)
}

# Apply validation rules
validation_violations <- create_validation_rules(mtcars)
```

## EDA Report Generation

### Comprehensive EDA Report

```r
# Function to generate comprehensive EDA report
generate_eda_report <- function(data, dataset_name = "Dataset") {
  cat("=== EXPLORATORY DATA ANALYSIS REPORT ===\n\n")
  cat("Dataset:", dataset_name, "\n")
  cat("Dimensions:", nrow(data), "rows ×", ncol(data), "columns\n")
  cat("Memory usage:", format(object.size(data), units = "MB"), "\n\n")
  
  # Data types summary
  cat("DATA TYPES:\n")
  type_summary <- sapply(data, class)
  for (var in names(type_summary)) {
    cat("  ", var, ":", type_summary[var], "\n")
  }
  cat("\n")
  
  # Missing values summary
  missing_summary <- sapply(data, function(x) sum(is.na(x)))
  if (sum(missing_summary) > 0) {
    cat("MISSING VALUES:\n")
    for (var in names(missing_summary)) {
      if (missing_summary[var] > 0) {
        cat("  ", var, ":", missing_summary[var], "(", 
            round(missing_summary[var]/nrow(data)*100, 1), "%)\n")
      }
    }
    cat("\n")
  } else {
    cat("No missing values found.\n\n")
  }
  
  # Summary statistics for numeric variables
  numeric_vars <- sapply(data, is.numeric)
  if (sum(numeric_vars) > 0) {
    cat("NUMERIC VARIABLES SUMMARY:\n")
    numeric_summary <- summary(data[, numeric_vars])
    print(numeric_summary)
    cat("\n")
    
    # Distribution characteristics
    cat("DISTRIBUTION CHARACTERISTICS:\n")
    for (var in names(data)[numeric_vars]) {
      values <- data[[var]]
      skewness_val <- moments::skewness(values, na.rm = TRUE)
      kurtosis_val <- moments::kurtosis(values, na.rm = TRUE)
      
      cat("  ", var, ":\n")
      cat("    Skewness:", round(skewness_val, 3))
      if (abs(skewness_val) < 0.5) {
        cat(" (symmetric)\n")
      } else if (skewness_val > 0.5) {
        cat(" (right-skewed)\n")
      } else {
        cat(" (left-skewed)\n")
      }
      
      cat("    Kurtosis:", round(kurtosis_val, 3))
      if (kurtosis_val < 3) {
        cat(" (light tails)\n")
      } else if (kurtosis_val > 3) {
        cat(" (heavy tails)\n")
      } else {
        cat(" (normal-like)\n")
      }
    }
    cat("\n")
  }
  
  # Key findings
  cat("KEY FINDINGS:\n")
  
  # Find strongest correlations
  if (sum(numeric_vars) > 1) {
    cor_matrix <- cor(data[, numeric_vars], use = "complete.obs")
    # Remove diagonal
    diag(cor_matrix) <- 0
    max_cor <- which(abs(cor_matrix) == max(abs(cor_matrix)), arr.ind = TRUE)
    var1 <- rownames(cor_matrix)[max_cor[1, 1]]
    var2 <- colnames(cor_matrix)[max_cor[1, 2]]
    max_cor_value <- cor_matrix[max_cor[1, 1], max_cor[1, 2]]
    
    cat("  Strongest correlation:", var1, "and", var2, "(", round(max_cor_value, 3), ")\n")
    
    # Find high correlations
    high_correlations <- which(abs(cor_matrix) > 0.7 & abs(cor_matrix) < 1, arr.ind = TRUE)
    if (nrow(high_correlations) > 0) {
      cat("  High correlations (|r| > 0.7):\n")
      for (i in 1:nrow(high_correlations)) {
        var1 <- rownames(cor_matrix)[high_correlations[i, 1]]
        var2 <- colnames(cor_matrix)[high_correlations[i, 2]]
        cor_value <- cor_matrix[high_correlations[i, 1], high_correlations[i, 2]]
        cat("    ", var1, "-", var2, ":", round(cor_value, 3), "\n")
      }
    }
  }
  
  # Find variables with most variation
  if (sum(numeric_vars) > 0) {
    cv_values <- sapply(data[, numeric_vars], function(x) {
      sd(x, na.rm = TRUE) / mean(x, na.rm = TRUE)
    })
    most_variable <- names(cv_values)[which.max(cv_values)]
    cat("  Most variable numeric variable:", most_variable, 
        "(CV =", round(max(cv_values), 3), ")\n")
  }
  
  # Outlier summary
  if (sum(numeric_vars) > 0) {
    outlier_counts <- sapply(data[, numeric_vars], function(x) {
      q1 <- quantile(x, 0.25, na.rm = TRUE)
      q3 <- quantile(x, 0.75, na.rm = TRUE)
      iqr <- q3 - q1
      sum(x < (q1 - 1.5 * iqr) | x > (q3 + 1.5 * iqr), na.rm = TRUE)
    })
    
    variables_with_outliers <- names(outlier_counts)[outlier_counts > 0]
    if (length(variables_with_outliers) > 0) {
      cat("  Variables with outliers:\n")
      for (var in variables_with_outliers) {
        cat("    ", var, ":", outlier_counts[var], "outliers\n")
      }
    }
  }
  
  cat("\nRECOMMENDATIONS:\n")
  cat("1. Consider transformations for highly skewed variables\n")
  cat("2. Investigate outliers and their impact on analysis\n")
  cat("3. Check for multicollinearity in regression models\n")
  cat("4. Consider interaction effects in modeling\n")
  cat("5. Validate data quality before proceeding with analysis\n")
}

# Generate EDA report
generate_eda_report(mtcars, "Motor Trend Car Road Tests")
```

### Automated EDA Pipeline

```r
# Function to create automated EDA pipeline
automated_eda_pipeline <- function(data, dataset_name = "Dataset") {
  cat("=== AUTOMATED EDA PIPELINE ===\n\n")
  
  # Step 1: Data overview
  cat("STEP 1: DATA OVERVIEW\n")
  cat("Dimensions:", nrow(data), "×", ncol(data), "\n")
  cat("Memory usage:", format(object.size(data), units = "MB"), "\n")
  cat("Data types:", paste(unique(sapply(data, class)), collapse = ", "), "\n\n")
  
  # Step 2: Data quality assessment
  cat("STEP 2: DATA QUALITY ASSESSMENT\n")
  missing_count <- sum(sapply(data, function(x) sum(is.na(x))))
  cat("Total missing values:", missing_count, "\n")
  
  duplicate_count <- sum(duplicated(data))
  cat("Duplicate rows:", duplicate_count, "\n")
  
  # Step 3: Univariate analysis
  cat("STEP 3: UNIVARIATE ANALYSIS\n")
  numeric_vars <- sapply(data, is.numeric)
  cat("Numeric variables:", sum(numeric_vars), "\n")
  cat("Categorical variables:", sum(!numeric_vars), "\n")
  
  # Analyze distributions for numeric variables
  if (sum(numeric_vars) > 0) {
    cat("Distribution analysis:\n")
    for (var in names(data)[numeric_vars]) {
      values <- data[[var]]
      skewness_val <- moments::skewness(values, na.rm = TRUE)
      cat("  ", var, ": skewness =", round(skewness_val, 3))
      if (abs(skewness_val) > 1) {
        cat(" (highly skewed - consider transformation)\n")
      } else {
        cat(" (moderate skew)\n")
      }
    }
  }
  cat("\n")
  
  # Step 4: Bivariate analysis
  cat("STEP 4: BIVARIATE ANALYSIS\n")
  if (sum(numeric_vars) > 1) {
    cor_matrix <- cor(data[, numeric_vars], use = "complete.obs")
    high_correlations <- which(abs(cor_matrix) > 0.7 & abs(cor_matrix) < 1, arr.ind = TRUE)
    if (nrow(high_correlations) > 0) {
      cat("High correlations detected:\n")
      for (i in 1:nrow(high_correlations)) {
        var1 <- rownames(cor_matrix)[high_correlations[i, 1]]
        var2 <- colnames(cor_matrix)[high_correlations[i, 2]]
        cor_value <- cor_matrix[high_correlations[i, 1], high_correlations[i, 2]]
        cat("  ", var1, "-", var2, ":", round(cor_value, 3), "\n")
      }
    } else {
      cat("No high correlations detected\n")
    }
  }
  cat("\n")
  
  # Step 5: Recommendations
  cat("STEP 5: RECOMMENDATIONS\n")
  if (missing_count > 0) {
    cat("- Address missing values before analysis\n")
  }
  if (duplicate_count > 0) {
    cat("- Remove or investigate duplicate records\n")
  }
  if (sum(numeric_vars) > 1 && nrow(high_correlations) > 0) {
    cat("- Consider multicollinearity in regression models\n")
  }
  
  # Check for skewed variables
  if (sum(numeric_vars) > 0) {
    skewed_vars <- sapply(data[, numeric_vars], function(x) {
      abs(moments::skewness(x, na.rm = TRUE)) > 1
    })
    if (sum(skewed_vars) > 0) {
      cat("- Transform highly skewed variables\n")
    }
  }
  
  cat("- Check for outliers and their impact\n")
  cat("- Validate statistical assumptions\n")
  cat("- Consider interaction effects in modeling\n")
}

# Apply pipeline to mtcars
automated_eda_pipeline(mtcars, "Motor Trend Car Road Tests")
```

## Practical Examples

### Example 1: Real Estate Data Analysis

```r
# Simulate comprehensive real estate data
set.seed(123)
n_properties <- 200

real_estate_data <- data.frame(
  price = rnorm(n_properties, mean = 300000, sd = 75000),
  square_feet = rnorm(n_properties, mean = 2000, sd = 500),
  bedrooms = sample(1:5, n_properties, replace = TRUE),
  bathrooms = sample(1:4, n_properties, replace = TRUE),
  age = rnorm(n_properties, mean = 15, sd = 8),
  location_score = rnorm(n_properties, mean = 7, sd = 1),
  property_type = sample(c("Single Family", "Condo", "Townhouse"), n_properties, replace = TRUE),
  garage_spaces = sample(0:3, n_properties, replace = TRUE)
)

# Add some realistic relationships
real_estate_data$price <- real_estate_data$price + 
  50000 * real_estate_data$square_feet / 1000 +
  25000 * real_estate_data$bedrooms +
  -5000 * real_estate_data$age +
  15000 * real_estate_data$location_score

# Perform comprehensive EDA
generate_eda_report(real_estate_data, "Real Estate Dataset")

# Create visualizations
create_univariate_plots(real_estate_data, "price")
create_scatter_analysis(real_estate_data, "square_feet", "price")

# Group comparisons
real_estate_data$property_type <- factor(real_estate_data$property_type)
create_group_comparison(real_estate_data, "property_type", "price")

# Correlation analysis
numeric_cols <- sapply(real_estate_data, is.numeric)
correlation_matrix <- cor(real_estate_data[, numeric_cols], use = "complete.obs")
print(round(correlation_matrix, 3))
```

### Example 2: Customer Satisfaction Analysis

```r
# Simulate comprehensive customer satisfaction data
set.seed(123)
n_customers <- 500

customer_data <- data.frame(
  satisfaction = sample(1:10, n_customers, replace = TRUE, 
                       prob = c(0.05, 0.08, 0.12, 0.15, 0.20, 0.18, 0.12, 0.08, 0.02, 0.01)),
  age_group = sample(c("18-25", "26-35", "36-45", "46+"), n_customers, replace = TRUE),
  income_level = sample(c("Low", "Medium", "High"), n_customers, replace = TRUE),
  purchase_amount = rnorm(n_customers, mean = 150, sd = 50),
  customer_service_rating = sample(1:5, n_customers, replace = TRUE),
  product_quality_rating = sample(1:5, n_customers, replace = TRUE),
  loyalty_program = sample(c("Yes", "No"), n_customers, replace = TRUE)
)

# Add realistic relationships
customer_data$satisfaction <- customer_data$satisfaction + 
  0.5 * customer_data$customer_service_rating +
  0.3 * customer_data$product_quality_rating +
  ifelse(customer_data$loyalty_program == "Yes", 1, 0)

# Ensure satisfaction stays within bounds
customer_data$satisfaction <- pmin(pmax(customer_data$satisfaction, 1), 10)

# Perform comprehensive EDA
generate_eda_report(customer_data, "Customer Satisfaction Dataset")

# Group comparisons
customer_data$age_group <- factor(customer_data$age_group, 
                                 levels = c("18-25", "26-35", "36-45", "46+"))
create_group_comparison(customer_data, "age_group", "satisfaction")

# Correlation analysis
numeric_cols <- sapply(customer_data, is.numeric)
correlation_matrix <- cor(customer_data[, numeric_cols], use = "complete.obs")
print(round(correlation_matrix, 3))

# Interaction analysis
customer_data$loyalty_factor <- factor(customer_data$loyalty_program)
analyze_interactions(customer_data, "purchase_amount", "loyalty_factor", "satisfaction")
```

### Example 3: Healthcare Data Analysis

```r
# Simulate healthcare data
set.seed(123)
n_patients <- 300

healthcare_data <- data.frame(
  age = rnorm(n_patients, mean = 45, sd = 15),
  bmi = rnorm(n_patients, mean = 25, sd = 5),
  blood_pressure_systolic = rnorm(n_patients, mean = 120, sd = 15),
  blood_pressure_diastolic = rnorm(n_patients, mean = 80, sd = 10),
  cholesterol = rnorm(n_patients, mean = 200, sd = 40),
  glucose = rnorm(n_patients, mean = 100, sd = 20),
  gender = sample(c("Male", "Female"), n_patients, replace = TRUE),
  smoking_status = sample(c("Never", "Former", "Current"), n_patients, replace = TRUE),
  diabetes = sample(c("No", "Yes"), n_patients, replace = TRUE, prob = c(0.85, 0.15))
)

# Add realistic relationships
healthcare_data$bmi <- healthcare_data$bmi + 0.1 * healthcare_data$age
healthcare_data$blood_pressure_systolic <- healthcare_data$blood_pressure_systolic + 
  0.5 * healthcare_data$age + 2 * healthcare_data$bmi
healthcare_data$glucose <- healthcare_data$glucose + 
  ifelse(healthcare_data$diabetes == "Yes", 50, 0)

# Perform comprehensive EDA
generate_eda_report(healthcare_data, "Healthcare Dataset")

# Create visualizations for key variables
create_univariate_plots(healthcare_data, "bmi")
create_scatter_analysis(healthcare_data, "age", "bmi")

# Group comparisons
healthcare_data$gender <- factor(healthcare_data$gender)
create_group_comparison(healthcare_data, "gender", "bmi")

# Correlation analysis
numeric_cols <- sapply(healthcare_data, is.numeric)
correlation_matrix <- cor(healthcare_data[, numeric_cols], use = "complete.obs")
print(round(correlation_matrix, 3))
```

## Best Practices

### EDA Workflow

```r
# Function to create systematic EDA workflow
eda_workflow <- function(data, dataset_name = "Dataset") {
  cat("=== SYSTEMATIC EDA WORKFLOW ===\n\n")
  
  # Step 1: Data overview
  cat("STEP 1: DATA OVERVIEW\n")
  cat("Dimensions:", nrow(data), "×", ncol(data), "\n")
  cat("Memory usage:", format(object.size(data), units = "MB"), "\n")
  cat("Data types:", paste(unique(sapply(data, class)), collapse = ", "), "\n\n")
  
  # Step 2: Data quality assessment
  cat("STEP 2: DATA QUALITY ASSESSMENT\n")
  missing_count <- sum(sapply(data, function(x) sum(is.na(x))))
  cat("Total missing values:", missing_count, "\n")
  
  duplicate_count <- sum(duplicated(data))
  cat("Duplicate rows:", duplicate_count, "\n")
  
  # Check for outliers
  numeric_vars <- sapply(data, is.numeric)
  if (sum(numeric_vars) > 0) {
    outlier_counts <- sapply(data[, numeric_vars], function(x) {
      q1 <- quantile(x, 0.25, na.rm = TRUE)
      q3 <- quantile(x, 0.75, na.rm = TRUE)
      iqr <- q3 - q1
      sum(x < (q1 - 1.5 * iqr) | x > (q3 + 1.5 * iqr), na.rm = TRUE)
    })
    total_outliers <- sum(outlier_counts)
    cat("Total outliers:", total_outliers, "\n")
  }
  cat("\n")
  
  # Step 3: Univariate analysis
  cat("STEP 3: UNIVARIATE ANALYSIS\n")
  cat("Numeric variables:", sum(numeric_vars), "\n")
  cat("Categorical variables:", sum(!numeric_vars), "\n")
  
  # Analyze distributions for numeric variables
  if (sum(numeric_vars) > 0) {
    cat("Distribution analysis:\n")
    for (var in names(data)[numeric_vars]) {
      values <- data[[var]]
      skewness_val <- moments::skewness(values, na.rm = TRUE)
      cat("  ", var, ": skewness =", round(skewness_val, 3))
      if (abs(skewness_val) > 1) {
        cat(" (highly skewed - consider transformation)\n")
      } else {
        cat(" (moderate skew)\n")
      }
    }
  }
  cat("\n")
  
  # Step 4: Bivariate analysis
  cat("STEP 4: BIVARIATE ANALYSIS\n")
  if (sum(numeric_vars) > 1) {
    cor_matrix <- cor(data[, numeric_vars], use = "complete.obs")
    high_correlations <- which(abs(cor_matrix) > 0.7 & abs(cor_matrix) < 1, arr.ind = TRUE)
    if (nrow(high_correlations) > 0) {
      cat("High correlations detected:\n")
      for (i in 1:nrow(high_correlations)) {
        var1 <- rownames(cor_matrix)[high_correlations[i, 1]]
        var2 <- colnames(cor_matrix)[high_correlations[i, 2]]
        cor_value <- cor_matrix[high_correlations[i, 1], high_correlations[i, 2]]
        cat("  ", var1, "-", var2, ":", round(cor_value, 3), "\n")
      }
    } else {
      cat("No high correlations detected\n")
    }
  }
  cat("\n")
  
  # Step 5: Recommendations
  cat("STEP 5: RECOMMENDATIONS\n")
  if (missing_count > 0) {
    cat("- Address missing values before analysis\n")
  }
  if (duplicate_count > 0) {
    cat("- Remove or investigate duplicate records\n")
  }
  if (sum(numeric_vars) > 1 && nrow(high_correlations) > 0) {
    cat("- Consider multicollinearity in regression models\n")
  }
  
  # Check for skewed variables
  if (sum(numeric_vars) > 0) {
    skewed_vars <- sapply(data[, numeric_vars], function(x) {
      abs(moments::skewness(x, na.rm = TRUE)) > 1
    })
    if (sum(skewed_vars) > 0) {
      cat("- Transform highly skewed variables\n")
    }
  }
  
  cat("- Check for outliers and their impact\n")
  cat("- Validate statistical assumptions\n")
  cat("- Consider interaction effects in modeling\n")
  cat("- Document all findings and decisions\n")
}

# Apply workflow to mtcars
eda_workflow(mtcars, "Motor Trend Car Road Tests")
```

### Documentation and Reporting

```r
# Function to create EDA documentation
create_eda_documentation <- function(data, dataset_name = "Dataset") {
  cat("=== EDA DOCUMENTATION TEMPLATE ===\n\n")
  
  cat("1. DATASET OVERVIEW\n")
  cat("   - Dataset name:", dataset_name, "\n")
  cat("   - Number of observations:", nrow(data), "\n")
  cat("   - Number of variables:", ncol(data), "\n")
  cat("   - Data collection period: [Specify]\n")
  cat("   - Data source: [Specify]\n\n")
  
  cat("2. VARIABLE DESCRIPTIONS\n")
  for (var in names(data)) {
    cat("   -", var, ":", class(data[[var]]), "\n")
    if (is.numeric(data[[var]])) {
      cat("     Range:", min(data[[var]], na.rm = TRUE), "to", 
          max(data[[var]], na.rm = TRUE), "\n")
      cat("     Mean:", round(mean(data[[var]], na.rm = TRUE), 3), "\n")
      cat("     SD:", round(sd(data[[var]], na.rm = TRUE), 3), "\n")
    } else {
      cat("     Levels:", paste(levels(data[[var]]), collapse = ", "), "\n")
    }
    cat("\n")
  }
  
  cat("3. DATA QUALITY ISSUES\n")
  missing_summary <- sapply(data, function(x) sum(is.na(x)))
  if (sum(missing_summary) > 0) {
    cat("   - Missing values found in:", 
        paste(names(missing_summary)[missing_summary > 0], collapse = ", "), "\n")
  } else {
    cat("   - No missing values\n")
  }
  
  cat("   - Duplicate rows:", sum(duplicated(data)), "\n")
  
  numeric_vars <- sapply(data, is.numeric)
  if (sum(numeric_vars) > 0) {
    outlier_counts <- sapply(data[, numeric_vars], function(x) {
      q1 <- quantile(x, 0.25, na.rm = TRUE)
      q3 <- quantile(x, 0.75, na.rm = TRUE)
      iqr <- q3 - q1
      sum(x < (q1 - 1.5 * iqr) | x > (q3 + 1.5 * iqr), na.rm = TRUE)
    })
    variables_with_outliers <- names(outlier_counts)[outlier_counts > 0]
    if (length(variables_with_outliers) > 0) {
      cat("   - Outliers found in:", paste(variables_with_outliers, collapse = ", "), "\n")
    } else {
      cat("   - No outliers detected\n")
    }
  }
  cat("\n")
  
  cat("4. KEY FINDINGS\n")
  cat("   - [Document main patterns and relationships]\n")
  cat("   - [Note any unexpected findings]\n")
  cat("   - [Highlight important insights]\n\n")
  
  cat("5. RECOMMENDATIONS\n")
  cat("   - [List specific actions to take]\n")
  cat("   - [Suggest next steps for analysis]\n")
  cat("   - [Note any limitations or concerns]\n\n")
  
  cat("6. APPENDIX\n")
  cat("   - [Include detailed statistical tests]\n")
  cat("   - [Add additional visualizations]\n")
  cat("   - [Provide code snippets]\n")
}

# Create documentation
create_eda_documentation(mtcars, "Motor Trend Car Road Tests")
```

## Exercises

### Exercise 1: Basic EDA
Perform exploratory data analysis on the `iris` dataset, including summary statistics, visualizations, and data quality checks.

```r
# Your solution here
data(iris)

# Perform comprehensive EDA
generate_eda_report(iris, "Iris Dataset")

# Create visualizations
create_univariate_plots(iris, "Sepal.Length")
create_scatter_analysis(iris, "Sepal.Length", "Sepal.Width")

# Group comparisons
create_group_comparison(iris, "Species", "Sepal.Length")

# Correlation analysis
numeric_cols <- sapply(iris, is.numeric)
correlation_matrix <- cor(iris[, numeric_cols], use = "complete.obs")
print(round(correlation_matrix, 3))
```

### Exercise 2: Missing Data Analysis
Create a dataset with missing values and develop strategies for handling them during EDA.

```r
# Your solution here
# Create dataset with missing values
set.seed(123)
n_obs <- 100

data_with_missing <- data.frame(
  id = 1:n_obs,
  age = rnorm(n_obs, mean = 35, sd = 10),
  income = rnorm(n_obs, mean = 50000, sd = 15000),
  education = sample(c("High School", "Bachelor", "Master", "PhD"), n_obs, replace = TRUE),
  satisfaction = sample(1:10, n_obs, replace = TRUE)
)

# Introduce missing values
data_with_missing$age[sample(1:n_obs, 10)] <- NA
data_with_missing$income[sample(1:n_obs, 15)] <- NA
data_with_missing$education[sample(1:n_obs, 5)] <- NA

# Analyze missing data patterns
missing_pattern <- is.na(data_with_missing)
missing_correlations <- cor(missing_pattern)
print(round(missing_correlations, 3))

# Strategies for handling missing data
# 1. Complete case analysis
complete_cases <- na.omit(data_with_missing)
cat("Complete cases:", nrow(complete_cases), "out of", nrow(data_with_missing), "\n")

# 2. Mean imputation
data_imputed <- data_with_missing
data_imputed$age[is.na(data_imputed$age)] <- mean(data_imputed$age, na.rm = TRUE)
data_imputed$income[is.na(data_imputed$income)] <- mean(data_imputed$income, na.rm = TRUE)

# 3. Multiple imputation (using mice package)
library(mice)
imp <- mice(data_with_missing, m = 5, method = "pmm")
data_imputed_mice <- complete(imp)
```

### Exercise 3: Outlier Detection
Implement multiple methods for outlier detection and compare their results on a dataset of your choice.

```r
# Your solution here
# Create dataset with outliers
set.seed(123)
n_obs <- 100

data_with_outliers <- data.frame(
  x = c(rnorm(95, mean = 0, sd = 1), rnorm(5, mean = 5, sd = 1)),
  y = c(rnorm(95, mean = 0, sd = 1), rnorm(5, mean = -5, sd = 1))
)

# Compare outlier detection methods
compare_outlier_methods <- function(data, variable) {
  cat("=== OUTLIER DETECTION COMPARISON FOR", variable, "===\n")
  
  # IQR method
  iqr_result <- detect_outliers(data[[variable]], "iqr")
  
  # Z-score method
  zscore_result <- detect_outliers(data[[variable]], "zscore")
  
  # Modified Z-score method
  modified_z_result <- detect_outliers(data[[variable]], "modified_zscore")
  
  # Print results
  cat("IQR method:", sum(iqr_result$outliers), "outliers\n")
  cat("Z-score method:", sum(zscore_result$outliers), "outliers\n")
  cat("Modified Z-score method:", sum(modified_z_result$outliers), "outliers\n")
  
  # Visualize outliers
  ggplot(data, aes_string(x = variable)) +
    geom_histogram(bins = 20, fill = "steelblue", alpha = 0.7) +
    geom_vline(data = data.frame(x = data[[variable]][iqr_result$outliers]), 
               aes(xintercept = x), color = "red", linetype = "dashed") +
    labs(title = paste("Outliers detected by IQR method in", variable)) +
    theme_minimal()
}

# Apply to dataset
compare_outlier_methods(data_with_outliers, "x")
```

### Exercise 4: Correlation Analysis
Analyze correlations between variables in a multivariate dataset and interpret the findings.

```r
# Your solution here
# Create multivariate dataset
set.seed(123)
n_obs <- 200

multivariate_data <- data.frame(
  height = rnorm(n_obs, mean = 170, sd = 10),
  weight = rnorm(n_obs, mean = 70, sd = 15),
  age = rnorm(n_obs, mean = 35, sd = 10),
  income = rnorm(n_obs, mean = 50000, sd = 20000),
  education_years = rnorm(n_obs, mean = 14, sd = 3)
)

# Add realistic correlations
multivariate_data$weight <- multivariate_data$weight + 0.5 * multivariate_data$height
multivariate_data$income <- multivariate_data$income + 5000 * multivariate_data$education_years

# Correlation analysis
correlation_matrix <- cor(multivariate_data, use = "complete.obs")
print(round(correlation_matrix, 3))

# Visualize correlation matrix
library(corrplot)
corrplot(correlation_matrix, method = "color", type = "upper", 
         order = "hclust", tl.col = "black", tl.srt = 45)

# Detailed correlation analysis
correlation_analysis(multivariate_data, "height", "weight")
correlation_analysis(multivariate_data, "income", "education_years")

# Test for significant correlations
significant_correlations <- which(abs(correlation_matrix) > 0.3 & abs(correlation_matrix) < 1, arr.ind = TRUE)
cat("Significant correlations (|r| > 0.3):\n")
for (i in 1:nrow(significant_correlations)) {
  var1 <- rownames(correlation_matrix)[significant_correlations[i, 1]]
  var2 <- colnames(correlation_matrix)[significant_correlations[i, 2]]
  cor_value <- correlation_matrix[significant_correlations[i, 1], significant_correlations[i, 2]]
  cat("  ", var1, "-", var2, ":", round(cor_value, 3), "\n")
}
```

### Exercise 5: Group Comparisons
Compare groups in a dataset using appropriate visualizations and statistical tests.

```r
# Your solution here
# Create dataset with groups
set.seed(123)
n_obs <- 150

group_data <- data.frame(
  value = c(rnorm(50, mean = 10, sd = 2), 
            rnorm(50, mean = 12, sd = 2),
            rnorm(50, mean = 8, sd = 2)),
  group = rep(c("A", "B", "C"), each = 50),
  category = sample(c("Low", "Medium", "High"), 150, replace = TRUE)
)

# Group comparisons
group_data$group <- factor(group_data$group)
create_group_comparison(group_data, "group", "value")

# Two-way ANOVA
two_way_anova <- aov(value ~ group * category, data = group_data)
print(summary(two_way_anova))

# Post-hoc tests
library(multcomp)
posthoc <- glht(two_way_anova, linfct = mcp(group = "Tukey"))
print(summary(posthoc))

# Effect size
library(effectsize)
eta_squared <- eta_squared(two_way_anova)
print(eta_squared)
```

## Next Steps

In the next chapter, we'll learn about measures of central tendency and their applications. We'll cover:

- **Measures of Central Tendency**: Mean, median, mode, and their properties
- **Measures of Variability**: Variance, standard deviation, and range
- **Measures of Shape**: Skewness and kurtosis
- **Robust Statistics**: Median absolute deviation and trimmed means
- **Statistical Inference**: Confidence intervals and hypothesis testing

---

**Key Takeaways:**
- EDA is the foundation of any statistical analysis
- Always check data quality before proceeding with analysis
- Use multiple visualization techniques to understand your data
- Consider both parametric and nonparametric approaches
- Document your findings systematically for reproducibility
- Be aware of data limitations and statistical assumptions
- EDA should guide your choice of statistical methods
- Transformations can help meet statistical assumptions
- Outliers and missing data require careful consideration
- Correlation does not imply causation 