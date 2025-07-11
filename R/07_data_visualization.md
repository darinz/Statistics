# Data Visualization

## Overview

Data visualization is a crucial component of statistical analysis. Good visualizations help us understand data patterns, identify outliers, and communicate findings effectively.

## Base R Graphics

### Basic Plot Types

```r
# Load sample data
data(mtcars)

# Scatter plot
plot(mtcars$wt, mtcars$mpg, 
     main = "MPG vs Weight",
     xlab = "Weight (1000 lbs)",
     ylab = "Miles per Gallon",
     pch = 16, col = "blue")

# Line plot
x <- 1:10
y <- x^2
plot(x, y, type = "l", 
     main = "Quadratic Function",
     xlab = "x", ylab = "y")

# Histogram
hist(mtcars$mpg, 
     main = "Distribution of MPG",
     xlab = "Miles per Gallon",
     col = "lightblue",
     breaks = 10)

# Box plot
boxplot(mtcars$mpg ~ mtcars$cyl,
        main = "MPG by Number of Cylinders",
        xlab = "Cylinders",
        ylab = "Miles per Gallon",
        col = c("red", "blue", "green"))
```

### Customizing Plots

```r
# Enhanced scatter plot
plot(mtcars$wt, mtcars$mpg,
     main = "Fuel Efficiency vs Weight",
     xlab = "Weight (1000 lbs)",
     ylab = "Miles per Gallon",
     pch = 16,
     col = ifelse(mtcars$am == 1, "red", "blue"),
     cex = 1.5)

# Add legend
legend("topright", 
       legend = c("Manual", "Automatic"),
       col = c("red", "blue"),
       pch = 16,
       title = "Transmission")

# Add grid
grid()

# Add regression line
abline(lm(mpg ~ wt, data = mtcars), 
       col = "green", lwd = 2)
```

## ggplot2 - Grammar of Graphics

### Installing and Loading ggplot2

```r
# Install and load ggplot2
install.packages("ggplot2")
library(ggplot2)

# Also load related packages
library(dplyr)
library(tidyr)
```

### Basic ggplot2 Syntax

```r
# Basic scatter plot
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point()

# Enhanced scatter plot
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(am))) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Fuel Efficiency vs Weight",
       x = "Weight (1000 lbs)",
       y = "Miles per Gallon",
       color = "Transmission") +
  theme_minimal()
```

### Different Plot Types

#### Scatter Plots

```r
# Basic scatter plot
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "MPG vs Weight with Regression Line")

# Scatter plot with multiple aesthetics
ggplot(mtcars, aes(x = wt, y = mpg, 
                   color = factor(cyl), 
                   size = hp)) +
  geom_point(alpha = 0.7) +
  labs(title = "MPG vs Weight by Cylinders and Horsepower",
       color = "Cylinders",
       size = "Horsepower")
```

#### Histograms

```r
# Basic histogram
ggplot(mtcars, aes(x = mpg)) +
  geom_histogram(bins = 10, fill = "lightblue", color = "black") +
  labs(title = "Distribution of MPG",
       x = "Miles per Gallon",
       y = "Frequency")

# Density plot
ggplot(mtcars, aes(x = mpg)) +
  geom_density(fill = "lightgreen", alpha = 0.5) +
  labs(title = "Density Plot of MPG",
       x = "Miles per Gallon",
       y = "Density")
```

#### Box Plots

```r
# Basic box plot
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "MPG Distribution by Cylinders",
       x = "Number of Cylinders",
       y = "Miles per Gallon")

# Box plot with individual points
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  geom_boxplot(fill = "lightblue") +
  geom_jitter(width = 0.2, alpha = 0.5) +
  labs(title = "MPG by Cylinders with Individual Points")
```

#### Bar Plots

```r
# Count bar plot
ggplot(mtcars, aes(x = factor(cyl))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Count of Cars by Cylinders",
       x = "Number of Cylinders",
       y = "Count")

# Summary bar plot
cyl_summary <- mtcars %>%
  group_by(cyl) %>%
  summarise(mean_mpg = mean(mpg))

ggplot(cyl_summary, aes(x = factor(cyl), y = mean_mpg)) +
  geom_bar(stat = "identity", fill = "lightcoral") +
  labs(title = "Average MPG by Cylinders",
       x = "Number of Cylinders",
       y = "Average MPG")
```

#### Line Plots

```r
# Create time series data
time_data <- data.frame(
  time = 1:20,
  value = cumsum(rnorm(20))
)

ggplot(time_data, aes(x = time, y = value)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 2) +
  labs(title = "Time Series Plot",
       x = "Time",
       y = "Value")
```

### Faceting

```r
# Facet by one variable
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  facet_wrap(~cyl) +
  labs(title = "MPG vs Weight by Cylinders")

# Facet by two variables
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  facet_grid(am ~ cyl) +
  labs(title = "MPG vs Weight by Transmission and Cylinders")
```

### Themes and Customization

```r
# Apply different themes
p <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  labs(title = "MPG vs Weight")

# Different themes
p + theme_minimal()
p + theme_classic()
p + theme_dark()
p + theme_bw()

# Custom theme
p + theme(
  plot.title = element_text(size = 16, face = "bold"),
  axis.title = element_text(size = 12),
  axis.text = element_text(size = 10),
  panel.background = element_rect(fill = "lightgray"),
  panel.grid = element_line(color = "white")
)
```

## Advanced Visualizations

### Correlation Matrix

```r
# Load required packages
library(corrplot)

# Calculate correlation matrix
cor_matrix <- cor(mtcars[, c("mpg", "cyl", "disp", "hp", "wt", "qsec")])

# Create correlation plot
corrplot(cor_matrix, 
         method = "color",
         type = "upper",
         addCoef.col = "black",
         tl.col = "black",
         tl.srt = 45)
```

### Violin Plots

```r
# Violin plot
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  geom_violin(fill = "lightblue", alpha = 0.7) +
  geom_boxplot(width = 0.2, fill = "white") +
  labs(title = "MPG Distribution by Cylinders",
       x = "Number of Cylinders",
       y = "Miles per Gallon")
```

### Heatmaps

```r
# Create heatmap data
heatmap_data <- mtcars %>%
  select(mpg, cyl, disp, hp, wt) %>%
  scale() %>%
  as.data.frame()

# Create heatmap
heatmap(as.matrix(heatmap_data), 
        main = "Variable Correlation Heatmap",
        col = colorRampPalette(c("blue", "white", "red"))(100))
```

### 3D Scatter Plots

```r
# Load required package
library(scatterplot3d)

# 3D scatter plot
scatterplot3d(mtcars$wt, mtcars$hp, mtcars$mpg,
              main = "3D Scatter Plot: Weight, Horsepower, MPG",
              xlab = "Weight",
              ylab = "Horsepower",
              zlab = "MPG",
              color = mtcars$cyl,
              pch = 16)
```

## Statistical Visualizations

### Q-Q Plots

```r
# Q-Q plot for normality
qqnorm(mtcars$mpg)
qqline(mtcars$mpg, col = "red")

# Using ggplot2
ggplot(mtcars, aes(sample = mpg)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "Q-Q Plot for MPG",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles")
```

### Residual Plots

```r
# Fit linear model
model <- lm(mpg ~ wt, data = mtcars)

# Residual plot
plot(fitted(model), residuals(model),
     main = "Residual Plot",
     xlab = "Fitted Values",
     ylab = "Residuals")
abline(h = 0, col = "red")

# Using ggplot2
residual_data <- data.frame(
  fitted = fitted(model),
  residuals = residuals(model)
)

ggplot(residual_data, aes(x = fitted, y = residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residual Plot",
       x = "Fitted Values",
       y = "Residuals")
```

## Interactive Visualizations

### Plotly

```r
# Install and load plotly
install.packages("plotly")
library(plotly)

# Create interactive scatter plot
p <- ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point() +
  labs(title = "Interactive MPG vs Weight")

ggplotly(p)
```

### Highcharter

```r
# Install and load highcharter
install.packages("highcharter")
library(highcharter)

# Create interactive chart
hchart(mtcars, "scatter", hcaes(wt, mpg, group = cyl)) %>%
  hc_title(text = "Interactive MPG vs Weight") %>%
  hc_xAxis(title = list(text = "Weight")) %>%
  hc_yAxis(title = list(text = "MPG"))
```

## Best Practices

### Color Choices

```r
# Color-blind friendly palette
colorblind_palette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", 
                       "#0072B2", "#D55E00", "#CC79A7")

# Use in plot
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point() +
  scale_color_manual(values = colorblind_palette) +
  labs(title = "MPG vs Weight with Colorblind-Friendly Colors")
```

### Saving Plots

```r
# Save plot as PNG
p <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  labs(title = "MPG vs Weight")

ggsave("mpg_vs_weight.png", p, width = 8, height = 6, dpi = 300)

# Save as PDF
ggsave("mpg_vs_weight.pdf", p, width = 8, height = 6)
```

## Practical Examples

### Example 1: Comprehensive Data Analysis

```r
# Load data
data(iris)

# Create multiple plots
p1 <- ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point() +
  labs(title = "Sepal Length vs Width")

p2 <- ggplot(iris, aes(x = Species, y = Sepal.Length)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Sepal Length by Species")

p3 <- ggplot(iris, aes(x = Sepal.Length)) +
  geom_histogram(bins = 20, fill = "lightgreen") +
  facet_wrap(~Species) +
  labs(title = "Sepal Length Distribution by Species")

# Display plots
print(p1)
print(p2)
print(p3)
```

### Example 2: Time Series Visualization

```r
# Create time series data
dates <- seq(as.Date("2023-01-01"), as.Date("2023-12-31"), by = "month")
values <- cumsum(rnorm(12))

ts_data <- data.frame(date = dates, value = values)

ggplot(ts_data, aes(x = date, y = value)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 2) +
  labs(title = "Time Series Plot",
       x = "Date",
       y = "Value") +
  theme_minimal()
```

## Exercises

### Exercise 1: Basic Plots
Create scatter plots, histograms, and box plots for the `mtcars` dataset.

### Exercise 2: Customization
Create a publication-ready plot with custom colors, themes, and annotations.

### Exercise 3: Faceting
Create faceted plots to show relationships across different groups.

### Exercise 4: Interactive Plots
Create an interactive visualization using plotly or highcharter.

### Exercise 5: Statistical Plots
Create Q-Q plots and residual plots for a linear regression model.

## Next Steps

In the next chapter, we'll learn about probability distributions and how to work with them in R.

---

**Key Takeaways:**
- Base R graphics are good for quick plots
- ggplot2 provides a consistent grammar of graphics
- Choose appropriate plot types for your data
- Use color and themes effectively
- Consider accessibility in color choices
- Save plots in appropriate formats
- Interactive plots enhance exploration
- Always include proper labels and titles 