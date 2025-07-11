# Exploratory Data Analysis

## Overview

Exploratory Data Analysis (EDA) is the first step in any statistical analysis. It involves examining and understanding your data through visualization and summary statistics before applying formal statistical methods.

## Data Exploration Fundamentals

### Initial Data Inspection

```r
# Load sample data
data(mtcars)

# Basic data structure
str(mtcars)
dim(mtcars)
names(mtcars)

# Data types
sapply(mtcars, class)

# First few observations
head(mtcars, 10)

# Last few observations
tail(mtcars, 5)
```

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
    n = n(),
    mean_mpg = mean(mpg, na.rm = TRUE),
    sd_mpg = sd(mpg, na.rm = TRUE),
    median_mpg = median(mpg, na.rm = TRUE),
    min_mpg = min(mpg, na.rm = TRUE),
    max_mpg = max(mpg, na.rm = TRUE)
  )
```

## Data Quality Assessment

### Missing Values

```r
# Check for missing values
missing_summary <- sapply(mtcars, function(x) sum(is.na(x)))
print(missing_summary)

# Visualize missing data
library(ggplot2)
missing_data <- data.frame(
  variable = names(missing_summary),
  missing_count = missing_summary,
  missing_percent = (missing_summary / nrow(mtcars)) * 100
)

ggplot(missing_data, aes(x = reorder(variable, missing_percent), y = missing_percent)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Missing Values by Variable",
       x = "Variable",
       y = "Percentage Missing") +
  theme_minimal()
```

### Duplicate Detection

```r
# Check for duplicate rows
duplicates <- sum(duplicated(mtcars))
cat("Number of duplicate rows:", duplicates, "\n")

# Check for duplicate values in specific columns
duplicate_mpg <- sum(duplicated(mtcars$mpg))
cat("Duplicate MPG values:", duplicate_mpg, "\n")
```

### Outlier Detection

```r
# Function to detect outliers using IQR method
detect_outliers <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  
  outliers <- x < lower_bound | x > upper_bound
  return(list(
    outliers = outliers,
    outlier_indices = which(outliers),
    outlier_values = x[outliers],
    bounds = c(lower_bound, upper_bound)
  ))
}

# Apply to numeric variables
numeric_vars <- sapply(mtcars, is.numeric)
outlier_summary <- lapply(mtcars[, numeric_vars], detect_outliers)

# Display outlier summary
for (var in names(outlier_summary)) {
  n_outliers <- sum(outlier_summary[[var]]$outliers)
  if (n_outliers > 0) {
    cat(var, ":", n_outliers, "outliers\n")
  }
}
```

## Univariate Analysis

### Distribution Analysis

```r
# Function to analyze distribution of a variable
analyze_distribution <- function(x, var_name) {
  cat("=== DISTRIBUTION ANALYSIS FOR", var_name, "===\n")
  
  # Basic statistics
  cat("Mean:", mean(x, na.rm = TRUE), "\n")
  cat("Median:", median(x, na.rm = TRUE), "\n")
  cat("Standard Deviation:", sd(x, na.rm = TRUE), "\n")
  cat("Skewness:", skewness(x, na.rm = TRUE), "\n")
  cat("Kurtosis:", kurtosis(x, na.rm = TRUE), "\n")
  
  # Quantiles
  cat("Quantiles:\n")
  print(quantile(x, probs = c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE))
  
  # Normality test
  shapiro_test <- shapiro.test(x)
  cat("Shapiro-Wilk normality test p-value:", shapiro_test$p.value, "\n")
  
  return(list(
    mean = mean(x, na.rm = TRUE),
    median = median(x, na.rm = TRUE),
    sd = sd(x, na.rm = TRUE),
    skewness = skewness(x, na.rm = TRUE),
    kurtosis = kurtosis(x, na.rm = TRUE),
    shapiro_p = shapiro_test$p.value
  ))
}

# Apply to MPG
mpg_analysis <- analyze_distribution(mtcars$mpg, "MPG")
```

### Visualization for Univariate Data

```r
# Create comprehensive univariate plots
create_univariate_plots <- function(data, variable) {
  # Histogram with density curve
  p1 <- ggplot(data, aes_string(x = variable)) +
    geom_histogram(aes(y = ..density..), bins = 15, fill = "steelblue", alpha = 0.7) +
    geom_density(color = "red", size = 1) +
    labs(title = paste("Distribution of", variable)) +
    theme_minimal()
  
  # Box plot
  p2 <- ggplot(data, aes_string(y = variable)) +
    geom_boxplot(fill = "lightgreen", alpha = 0.7) +
    labs(title = paste("Boxplot of", variable)) +
    theme_minimal()
  
  # Q-Q plot
  p3 <- ggplot(data, aes_string(sample = variable)) +
    stat_qq() +
    stat_qq_line() +
    labs(title = paste("Q-Q Plot of", variable)) +
    theme_minimal()
  
  # Combine plots
  library(gridExtra)
  grid.arrange(p1, p2, p3, ncol = 2)
}

# Apply to MPG
create_univariate_plots(mtcars, "mpg")
```

## Bivariate Analysis

### Correlation Analysis

```r
# Correlation matrix for numeric variables
numeric_data <- mtcars[, sapply(mtcars, is.numeric)]
correlation_matrix <- cor(numeric_data, use = "complete.obs")

# Visualize correlation matrix
library(corrplot)
corrplot(correlation_matrix, method = "color", type = "upper", 
         order = "hclust", tl.col = "black", tl.srt = 45)

# Detailed correlation analysis
correlation_analysis <- function(data, var1, var2) {
  cat("=== CORRELATION ANALYSIS ===\n")
  cat("Variables:", var1, "and", var2, "\n")
  
  # Pearson correlation
  pearson_cor <- cor.test(data[[var1]], data[[var2]], method = "pearson")
  cat("Pearson correlation:", round(pearson_cor$estimate, 3), "\n")
  cat("p-value:", pearson_cor$p.value, "\n")
  
  # Spearman correlation
  spearman_cor <- cor.test(data[[var1]], data[[var2]], method = "spearman")
  cat("Spearman correlation:", round(spearman_cor$estimate, 3), "\n")
  cat("p-value:", spearman_cor$p.value, "\n")
  
  return(list(
    pearson = pearson_cor,
    spearman = spearman_cor
  ))
}

# Analyze correlation between MPG and Weight
mpg_wt_correlation <- correlation_analysis(mtcars, "mpg", "wt")
```

### Scatter Plot Analysis

```r
# Create comprehensive scatter plots
create_scatter_analysis <- function(data, x_var, y_var) {
  # Basic scatter plot
  p1 <- ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", se = TRUE, color = "red") +
    labs(title = paste(y_var, "vs", x_var)) +
    theme_minimal()
  
  # Scatter plot with regression line and confidence interval
  p2 <- ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "loess", se = TRUE, color = "blue") +
    labs(title = paste(y_var, "vs", x_var, "(LOESS)")) +
    theme_minimal()
  
  # Residual plot
  model <- lm(as.formula(paste(y_var, "~", x_var)), data = data)
  residuals_data <- data.frame(
    fitted = fitted(model),
    residuals = residuals(model)
  )
  
  p3 <- ggplot(residuals_data, aes(x = fitted, y = residuals)) +
    geom_point(alpha = 0.7) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    labs(title = "Residual Plot") +
    theme_minimal()
  
  # Combine plots
  grid.arrange(p1, p2, p3, ncol = 2)
}

# Apply to MPG vs Weight
create_scatter_analysis(mtcars, "wt", "mpg")
```

## Multivariate Analysis

### Group Comparisons

```r
# Compare groups using box plots
create_group_comparison <- function(data, group_var, response_var) {
  # Box plot by group
  p1 <- ggplot(data, aes_string(x = group_var, y = response_var, fill = group_var)) +
    geom_boxplot(alpha = 0.7) +
    labs(title = paste(response_var, "by", group_var)) +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Violin plot
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
      max = max(!!sym(response_var), na.rm = TRUE)
    )
  
  print(group_summary)
  
  # Combine plots
  grid.arrange(p1, p2, ncol = 2)
}

# Compare MPG by transmission type
create_group_comparison(mtcars, "am", "mpg")
```

### Interaction Analysis

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
  
  # Combine plots
  grid.arrange(p1, p2, ncol = 1)
}

# Analyze interaction between weight and transmission on MPG
mtcars$am_factor <- factor(mtcars$am, labels = c("Automatic", "Manual"))
analyze_interactions(mtcars, "wt", "am_factor", "mpg")
```

## Data Transformation

### Variable Transformations

```r
# Function to apply common transformations
apply_transformations <- function(data, variable) {
  # Original variable
  original <- data[[variable]]
  
  # Log transformation
  log_transformed <- log(original)
  
  # Square root transformation
  sqrt_transformed <- sqrt(original)
  
  # Reciprocal transformation
  reciprocal_transformed <- 1 / original
  
  # Box-Cox transformation
  library(MASS)
  bc_result <- boxcox(original ~ 1)
  lambda <- bc_result$x[which.max(bc_result$y)]
  
  if (abs(lambda) < 0.001) {
    bc_transformed <- log(original)
  } else {
    bc_transformed <- (original^lambda - 1) / lambda
  }
  
  # Create comparison plots
  transformed_data <- data.frame(
    Original = original,
    Log = log_transformed,
    Sqrt = sqrt_transformed,
    Reciprocal = reciprocal_transformed,
    BoxCox = bc_transformed
  )
  
  # Plot distributions
  library(reshape2)
  melted_data <- melt(transformed_data)
  
  ggplot(melted_data, aes(x = value)) +
    geom_histogram(bins = 15, fill = "steelblue", alpha = 0.7) +
    facet_wrap(~variable, scales = "free") +
    labs(title = paste("Distribution of", variable, "under different transformations")) +
    theme_minimal()
  
  # Return transformation results
  return(list(
    original = original,
    log = log_transformed,
    sqrt = sqrt_transformed,
    reciprocal = reciprocal_transformed,
    boxcox = bc_transformed,
    lambda = lambda
  ))
}

# Apply transformations to MPG
mpg_transformations <- apply_transformations(mtcars, "mpg")
```

## Data Quality Checks

### Consistency Checks

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
    extreme_low <- sum(values < quantile(values, 0.01, na.rm = TRUE), na.rm = TRUE)
    extreme_high <- sum(values > quantile(values, 0.99, na.rm = TRUE), na.rm = TRUE)
    
    if (extreme_low > 0 || extreme_high > 0) {
      cat(var, ":", extreme_low, "extreme low values,", extreme_high, "extreme high values\n")
    }
  }
  
  # Check for data type consistency
  for (var in names(data)) {
    if (is.factor(data[[var]])) {
      n_levels <- nlevels(data[[var]])
      if (n_levels > 20) {
        cat("WARNING:", var, "has", n_levels, "levels (consider if this is correct)\n")
      }
    }
  }
}

# Apply consistency checks
check_data_consistency(mtcars)
```

## EDA Report Generation

```r
# Function to generate comprehensive EDA report
generate_eda_report <- function(data, dataset_name = "Dataset") {
  cat("=== EXPLORATORY DATA ANALYSIS REPORT ===\n\n")
  cat("Dataset:", dataset_name, "\n")
  cat("Dimensions:", nrow(data), "rows ×", ncol(data), "columns\n\n")
  
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
  }
  
  # Find variables with most variation
  if (sum(numeric_vars) > 0) {
    cv_values <- sapply(data[, numeric_vars], function(x) sd(x, na.rm = TRUE) / mean(x, na.rm = TRUE))
    most_variable <- names(cv_values)[which.max(cv_values)]
    cat("  Most variable numeric variable:", most_variable, 
        "(CV =", round(max(cv_values), 3), ")\n")
  }
  
  cat("\nRECOMMENDATIONS:\n")
  cat("1. Consider transformations for highly skewed variables\n")
  cat("2. Investigate outliers and their impact on analysis\n")
  cat("3. Check for multicollinearity in regression models\n")
  cat("4. Consider interaction effects in modeling\n")
}

# Generate EDA report
generate_eda_report(mtcars, "Motor Trend Car Road Tests")
```

## Practical Examples

### Example 1: Real Estate Data Analysis

```r
# Simulate real estate data
set.seed(123)
n_properties <- 100

real_estate_data <- data.frame(
  price = rnorm(n_properties, mean = 300000, sd = 75000),
  square_feet = rnorm(n_properties, mean = 2000, sd = 500),
  bedrooms = sample(1:5, n_properties, replace = TRUE),
  bathrooms = sample(1:4, n_properties, replace = TRUE),
  age = rnorm(n_properties, mean = 15, sd = 8),
  location_score = rnorm(n_properties, mean = 7, sd = 1)
)

# Perform EDA on real estate data
generate_eda_report(real_estate_data, "Real Estate Dataset")

# Create visualizations
create_univariate_plots(real_estate_data, "price")
create_scatter_analysis(real_estate_data, "square_feet", "price")
```

### Example 2: Customer Satisfaction Analysis

```r
# Simulate customer satisfaction data
set.seed(123)
n_customers <- 200

customer_data <- data.frame(
  satisfaction = sample(1:10, n_customers, replace = TRUE, 
                       prob = c(0.05, 0.08, 0.12, 0.15, 0.20, 0.18, 0.12, 0.08, 0.02, 0.01)),
  age_group = sample(c("18-25", "26-35", "36-45", "46+"), n_customers, replace = TRUE),
  income_level = sample(c("Low", "Medium", "High"), n_customers, replace = TRUE),
  purchase_amount = rnorm(n_customers, mean = 150, sd = 50)
)

# Perform EDA on customer data
generate_eda_report(customer_data, "Customer Satisfaction Dataset")

# Group comparisons
customer_data$age_group <- factor(customer_data$age_group, 
                                 levels = c("18-25", "26-35", "36-45", "46+"))
create_group_comparison(customer_data, "age_group", "satisfaction")
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
  cat("Memory usage:", format(object.size(data), units = "MB"), "\n\n")
  
  # Step 2: Data quality assessment
  cat("STEP 2: DATA QUALITY ASSESSMENT\n")
  missing_count <- sum(sapply(data, function(x) sum(is.na(x))))
  cat("Total missing values:", missing_count, "\n")
  
  duplicate_count <- sum(duplicated(data))
  cat("Duplicate rows:", duplicate_count, "\n\n")
  
  # Step 3: Univariate analysis
  cat("STEP 3: UNIVARIATE ANALYSIS\n")
  numeric_vars <- sapply(data, is.numeric)
  cat("Numeric variables:", sum(numeric_vars), "\n")
  cat("Categorical variables:", sum(!numeric_vars), "\n\n")
  
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
  cat("- Transform skewed variables if needed\n")
  cat("- Check for outliers and their impact\n")
}

# Apply workflow to mtcars
eda_workflow(mtcars, "Motor Trend Car Road Tests")
```

## Exercises

### Exercise 1: Basic EDA
Perform exploratory data analysis on the `iris` dataset, including summary statistics, visualizations, and data quality checks.

### Exercise 2: Missing Data Analysis
Create a dataset with missing values and develop strategies for handling them during EDA.

### Exercise 3: Outlier Detection
Implement multiple methods for outlier detection and compare their results on a dataset of your choice.

### Exercise 4: Correlation Analysis
Analyze correlations between variables in a multivariate dataset and interpret the findings.

### Exercise 5: Group Comparisons
Compare groups in a dataset using appropriate visualizations and statistical tests.

## Next Steps

In the next chapter, we'll learn about measures of central tendency and their applications.

---

**Key Takeaways:**
- EDA is the foundation of any statistical analysis
- Always check data quality before proceeding
- Use multiple visualization techniques
- Consider both parametric and nonparametric approaches
- Document your findings systematically
- Be aware of data limitations and assumptions
- EDA should guide your choice of statistical methods 