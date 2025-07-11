# Correlation Analysis

## Overview

Correlation analysis examines the strength and direction of relationships between two or more quantitative variables. It is foundational for understanding associations, prediction, and causality in statistics and data science.

### Key Concepts

- **Covariance**: Measures the joint variability of two variables.
- **Correlation**: Standardizes covariance to a scale from -1 to 1, indicating both strength and direction.
- **Types of Correlation**: Pearson (linear), Spearman (monotonic, rank-based), Kendall (ordinal, rank-based).
- **Interpretation**: Correlation does not imply causation.

## Mathematical Foundations

### Covariance

Covariance quantifies how two variables change together:

```math
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})
```

- $`\text{Cov}(X, Y) > 0`$: Variables tend to increase together
- $`\text{Cov}(X, Y) < 0`$: One increases as the other decreases
- $`\text{Cov}(X, Y) = 0`$: No linear relationship

### Pearson Correlation Coefficient ($`r`$)

Pearson's $`r`$ measures the strength and direction of a linear relationship:

```math
r = \frac{\text{Cov}(X, Y)}{s_X s_Y} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}
```

- $`r = 1`$: Perfect positive linear relationship
- $`r = -1`$: Perfect negative linear relationship
- $`r = 0`$: No linear relationship

### Spearman's Rank Correlation ($`\rho`$)

Spearman's $`\rho`$ assesses monotonic relationships using ranks:

```math
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
```

Where $`d_i`$ is the difference between the ranks of $`x_i`$ and $`y_i`$.

### Kendall's Tau ($`\tau`$)

Kendall's $`\tau`$ measures ordinal association:

```math
\tau = \frac{(\text{number of concordant pairs}) - (\text{number of discordant pairs})}{\frac{1}{2} n(n-1)}
```

## When to Use Each Correlation

| Correlation | Data Type | Assumptions | Use Case |
|------------|-----------|-------------|----------|
| Pearson    | Interval/Ratio | Linear, normality | Linear relationships |
| Spearman   | Ordinal/Interval | Monotonic | Nonlinear monotonic, outliers |
| Kendall    | Ordinal | Monotonic | Small samples, many ties |

## Calculating Correlation in R

### Pearson Correlation

```r
# Simulate data
set.seed(42)
x <- rnorm(100)
y <- 0.7 * x + rnorm(100, sd = 0.5)

# Pearson correlation
cor_pearson <- cor(x, y, method = "pearson")
cat("Pearson correlation:", round(cor_pearson, 3), "\n")

# Test significance
cor_test <- cor.test(x, y, method = "pearson")
print(cor_test)
```

### Spearman and Kendall Correlation

```r
# Spearman correlation
cor_spearman <- cor(x, y, method = "spearman")
cat("Spearman correlation:", round(cor_spearman, 3), "\n")
cor.test(x, y, method = "spearman")

# Kendall correlation
cor_kendall <- cor(x, y, method = "kendall")
cat("Kendall correlation:", round(cor_kendall, 3), "\n")
cor.test(x, y, method = "kendall")
```

### Correlation Matrix

```r
# Multiple variables
set.seed(123)
data <- data.frame(
  A = rnorm(100),
  B = rnorm(100),
  C = rnorm(100)
)

# Correlation matrix
cor_matrix <- cor(data)
print(cor_matrix)

# Significance matrix
library(Hmisc)
cor_results <- rcorr(as.matrix(data))
print(cor_results$r)  # Correlations
print(cor_results$P)  # p-values
```

## Visualization

### Scatter Plot with Correlation

```r
library(ggplot2)

# Scatter plot with regression line
plot_data <- data.frame(x = x, y = y)
ggplot(plot_data, aes(x = x, y = y)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "blue") +
  labs(title = "Scatter Plot with Regression Line",
       subtitle = paste("Pearson r =", round(cor_pearson, 2)),
       x = "X", y = "Y") +
  theme_minimal()
```

### Correlation Matrix Heatmap

```r
library(reshape2)
library(ggplot2)

cor_melt <- melt(cor_matrix)
ggplot(cor_melt, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 2)), color = "white", size = 4) +
  scale_fill_gradient2(low = "#4575B4", high = "#D73027", mid = "#FFFFBF", midpoint = 0) +
  labs(title = "Correlation Matrix Heatmap", x = "", y = "") +
  theme_minimal()
```

## Assumption Checking

### Pearson Correlation Assumptions
- Linearity: Relationship between variables is linear
- Normality: Both variables are normally distributed
- Homoscedasticity: Constant variance of residuals
- No significant outliers

#### Checking Linearity and Outliers

```r
# Scatter plot for linearity and outliers
plot(x, y, main = "Scatter Plot for Linearity and Outliers")
abline(lm(y ~ x), col = "red")
```

#### Checking Normality

```r
# Q-Q plots
qqnorm(x); qqline(x, col = "red")
qqnorm(y); qqline(y, col = "red")

# Shapiro-Wilk test
shapiro.test(x)
shapiro.test(y)
```

#### Checking Homoscedasticity

```r
# Residuals vs fitted
model <- lm(y ~ x)
plot(fitted(model), resid(model),
     main = "Residuals vs Fitted Values",
     xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red")
```

### Robust Alternatives
- Use Spearman or Kendall correlation if assumptions are violated
- Consider robust correlation methods (e.g., biweight midcorrelation)

## Effect Size and Confidence Intervals

### Interpreting Correlation Coefficient

| $`|r|`$      | Strength         |
|----------|------------------|
| 0.00-0.10 | Negligible       |
| 0.10-0.30 | Small            |
| 0.30-0.50 | Moderate         |
| 0.50-0.70 | Large            |
| 0.70-0.90 | Very large       |
| 0.90-1.00 | Nearly perfect   |

### Confidence Interval for $`r`$

```r
# Confidence interval for correlation
cor_test$conf.int
```

### Fisher's z-Transformation

To compare correlations or compute confidence intervals:

```math
z = \frac{1}{2} \ln\left(\frac{1 + r}{1 - r}\right)
```

The standard error of $`z`$ is $`\frac{1}{\sqrt{n-3}}`$.

## Practical Examples

### Example 1: Height and Weight

```r
# Simulate height and weight data
set.seed(1)
height <- rnorm(100, mean = 170, sd = 10)
weight <- 0.5 * height + rnorm(100, mean = 0, sd = 8)

# Pearson correlation
cor(height, weight)
cor.test(height, weight)
```

### Example 2: Nonlinear Relationship

```r
# Simulate nonlinear data
set.seed(2)
x <- rnorm(100)
y <- x^2 + rnorm(100)

# Pearson vs Spearman
cor(x, y, method = "pearson")
cor(x, y, method = "spearman")
```

## Best Practices

- Always visualize data before interpreting correlation
- Check assumptions for Pearson correlation
- Use Spearman or Kendall for non-normal or ordinal data
- Report effect size and confidence intervals
- Correlation does not imply causation
- Be cautious of outliers and influential points
- Use robust methods for non-normal data

## Reporting Guidelines

- Report the type of correlation, value, confidence interval, and p-value
- Example: "There was a large, positive correlation between X and Y, $`r = 0.65, 95\%\ CI [0.50, 0.77], p < .001`$."

## Exercises

### Exercise 1: Pearson Correlation
Simulate two variables with a linear relationship. Calculate and interpret the Pearson correlation, check assumptions, and visualize the data.

### Exercise 2: Spearman Correlation
Simulate two variables with a monotonic but nonlinear relationship. Calculate and interpret the Spearman correlation.

### Exercise 3: Correlation Matrix
Simulate a dataset with at least four variables. Compute and visualize the correlation matrix. Interpret the strongest and weakest relationships.

### Exercise 4: Robust Correlation
Simulate data with outliers. Compare Pearson, Spearman, and robust correlation methods. Discuss the impact of outliers.

### Exercise 5: Real-World Application
Find a real dataset (e.g., from R's datasets package). Perform a comprehensive correlation analysis, including visualization, assumption checking, and reporting.

---

**Key Takeaways:**
- Correlation quantifies the strength and direction of association
- Pearson for linear, normal data; Spearman/Kendall for ranks or non-normal data
- Always check assumptions and visualize
- Correlation ≠ causation
- Report effect size and confidence intervals 