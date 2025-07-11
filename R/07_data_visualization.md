# Data Visualization

## Overview

Data visualization is a crucial component of statistical analysis and data science. It serves as both an exploratory tool for understanding data patterns and a communication medium for presenting findings effectively. Good visualizations can reveal insights that might be missed in numerical summaries alone.

### The Importance of Data Visualization

Data visualization serves several key purposes:

1. **Exploration**: Discover patterns, trends, and relationships in data
2. **Communication**: Convey complex information clearly to different audiences
3. **Validation**: Verify assumptions and check for data quality issues
4. **Storytelling**: Guide viewers through a narrative about the data

### Visualization Principles

Effective data visualization follows these principles:

- **Accuracy**: Represent data truthfully without distortion
- **Clarity**: Make the message immediately understandable
- **Efficiency**: Maximize information-to-ink ratio
- **Aesthetics**: Use design elements to enhance rather than distract

## Mathematical Foundations

### Coordinate Systems

Most statistical visualizations use Cartesian coordinate systems where:

- **2D Cartesian**: Points defined by $(x, y)$ coordinates
- **3D Cartesian**: Points defined by $(x, y, z)$ coordinates
- **Polar**: Points defined by $(r, \theta)$ where $r$ is radius and $\theta$ is angle

### Statistical Concepts in Visualization

#### Density Estimation

For continuous data, we often estimate probability density functions:

```math
f(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)
```

Where:
- $K$ is the kernel function
- $h$ is the bandwidth parameter
- $n$ is the number of observations

#### Correlation Visualization

The correlation coefficient $\rho$ between variables $X$ and $Y$:

```math
\rho = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}
```

#### Quantile-Quantile (Q-Q) Plots

Q-Q plots compare sample quantiles to theoretical quantiles:

```math
Q(p) = F^{-1}(p)
```

Where $F^{-1}$ is the inverse cumulative distribution function.

## Base R Graphics

### Basic Plot Types

Base R provides fundamental plotting capabilities through the `graphics` package. Let's explore the core plot types with detailed explanations:

```r
# Load sample data
data(mtcars)

# Scatter plot - shows relationship between two continuous variables
plot(mtcars$wt, mtcars$mpg, 
     main = "MPG vs Weight",
     xlab = "Weight (1000 lbs)",
     ylab = "Miles per Gallon",
     pch = 16, col = "blue")

# Line plot - shows trends over time or ordered sequences
x <- 1:10
y <- x^2
plot(x, y, type = "l", 
     main = "Quadratic Function",
     xlab = "x", ylab = "y")

# Histogram - shows distribution of continuous data
hist(mtcars$mpg, 
     main = "Distribution of MPG",
     xlab = "Miles per Gallon",
     col = "lightblue",
     breaks = 10)

# Box plot - shows distribution and outliers by groups
boxplot(mtcars$mpg ~ mtcars$cyl,
        main = "MPG by Number of Cylinders",
        xlab = "Cylinders",
        ylab = "Miles per Gallon",
        col = c("red", "blue", "green"))
```

### Understanding Plot Parameters

#### Point Types and Symbols

```r
# Different point types (pch parameter)
plot_types <- data.frame(
  pch = 0:25,
  description = c("Square", "Circle", "Triangle", "Plus", "X", "Diamond",
                  "Triangle Down", "Square Cross", "Star", "Diamond Plus",
                  "Circle Plus", "Triangle Plus", "Square Plus", "Circle Cross",
                  "Square Cross", "Circle Cross", "Square Plus", "Circle Plus",
                  "Triangle Plus", "Diamond Plus", "Circle Plus", "Square Plus",
                  "Circle Cross", "Square Cross", "Circle Cross", "Square Cross")
)

# Visual demonstration of point types
plot(1:26, 1:26, pch = 0:25, cex = 2, 
     main = "Point Types in R",
     xlab = "Point Type Number",
     ylab = "Y Position")
text(1:26, 1:26, labels = 0:25, pos = 3)
```

#### Color Systems

R supports multiple color specifications:

```r
# Named colors
colors <- c("red", "blue", "green", "yellow", "purple", "orange")

# RGB values (0-1 scale)
rgb_colors <- c(rgb(1,0,0), rgb(0,1,0), rgb(0,0,1))

# Hexadecimal codes
hex_colors <- c("#FF0000", "#00FF00", "#0000FF")

# Color palette demonstration
plot(1:6, 1:6, col = colors, pch = 16, cex = 3,
     main = "Color Examples",
     xlab = "X", ylab = "Y")
```

### Customizing Plots

```r
# Enhanced scatter plot with multiple customization options
plot(mtcars$wt, mtcars$mpg,
     main = "Fuel Efficiency vs Weight",
     xlab = "Weight (1000 lbs)",
     ylab = "Miles per Gallon",
     pch = 16,
     col = ifelse(mtcars$am == 1, "red", "blue"),
     cex = 1.5,
     xlim = c(1.5, 5.5),
     ylim = c(10, 35))

# Add legend with positioning
legend("topright", 
       legend = c("Manual", "Automatic"),
       col = c("red", "blue"),
       pch = 16,
       title = "Transmission",
       bg = "white",
       box.lty = 1)

# Add grid for better readability
grid(nx = 10, ny = 10, col = "gray", lty = "dotted")

# Add regression line with confidence interval
model <- lm(mpg ~ wt, data = mtcars)
abline(model, col = "green", lwd = 2)

# Add text annotations
text(4, 30, "Strong negative correlation", 
     col = "darkgreen", font = 2)
```

### Advanced Base R Features

#### Multiple Plots on One Page

```r
# Set up multiple plots
par(mfrow = c(2, 2))  # 2 rows, 2 columns

# Plot 1: Scatter plot
plot(mtcars$wt, mtcars$mpg, main = "MPG vs Weight")

# Plot 2: Histogram
hist(mtcars$mpg, main = "MPG Distribution")

# Plot 3: Box plot
boxplot(mpg ~ cyl, data = mtcars, main = "MPG by Cylinders")

# Plot 4: Density plot
plot(density(mtcars$mpg), main = "MPG Density")

# Reset to single plot
par(mfrow = c(1, 1))
```

#### Interactive Plotting

```r
# Identify points interactively
plot(mtcars$wt, mtcars$mpg, pch = 16)
identify(mtcars$wt, mtcars$mpg, labels = rownames(mtcars))

# Locator function for adding elements
plot(1:10, 1:10)
points <- locator(n = 3)  # Click 3 points
points(points$x, points$y, col = "red", pch = 16)
```

## ggplot2 - Grammar of Graphics

### Installing and Loading ggplot2

```r
# Install and load ggplot2
install.packages("ggplot2")
library(ggplot2)

# Also load related packages for data manipulation
library(dplyr)
library(tidyr)
```

### The Grammar of Graphics Philosophy

ggplot2 implements Leland Wilkinson's Grammar of Graphics, which breaks down plots into components:

1. **Data**: The dataset being visualized
2. **Aesthetics**: Mappings from data to visual properties
3. **Geometries**: The actual marks used to represent data
4. **Scales**: Control how aesthetics are mapped to visual properties
5. **Facets**: Subdivision of data into multiple plots
6. **Themes**: Control of non-data elements

### Basic ggplot2 Syntax

```r
# Basic scatter plot
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point()

# Enhanced scatter plot with multiple aesthetics
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(am))) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Fuel Efficiency vs Weight",
       x = "Weight (1000 lbs)",
       y = "Miles per Gallon",
       color = "Transmission") +
  theme_minimal()
```

### Understanding Aesthetics

Aesthetics map data variables to visual properties:

```r
# Common aesthetics and their mathematical interpretation
aesthetics_demo <- data.frame(
  aesthetic = c("x", "y", "color", "size", "shape", "alpha"),
  data_type = c("continuous", "continuous", "discrete/continuous", 
                "continuous", "discrete", "continuous"),
  visual_property = c("horizontal position", "vertical position", 
                     "hue", "area/radius", "symbol type", "transparency"),
  mathematical_scale = c("linear", "linear", "categorical/continuous", 
                        "area/radius", "categorical", "opacity")
)

# Demonstrate aesthetic mapping
ggplot(mtcars, aes(x = wt, y = mpg, 
                   color = factor(cyl),    # Categorical color
                   size = hp,              # Continuous size
                   alpha = 0.7)) +        # Fixed transparency
  geom_point() +
  labs(title = "Multiple Aesthetics Mapping",
       color = "Cylinders",
       size = "Horsepower") +
  scale_color_brewer(type = "qual", palette = "Set1") +
  scale_size_continuous(range = c(2, 6))
```

### Different Plot Types

#### Scatter Plots

```r
# Basic scatter plot with regression line
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +  # se = TRUE shows confidence interval
  labs(title = "MPG vs Weight with Regression Line",
       subtitle = "95% confidence interval shown")

# Scatter plot with multiple aesthetics and mathematical annotations
ggplot(mtcars, aes(x = wt, y = mpg, 
                   color = factor(cyl), 
                   size = hp)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  labs(title = "MPG vs Weight by Cylinders and Horsepower",
       color = "Cylinders",
       size = "Horsepower") +
  annotate("text", x = 4, y = 30, 
           label = paste("r =", round(cor(mtcars$wt, mtcars$mpg), 3)),
           color = "red", fontface = "bold")
```

#### Histograms and Density Plots

```r
# Basic histogram with density overlay
ggplot(mtcars, aes(x = mpg)) +
  geom_histogram(aes(y = ..density..), bins = 10, 
                 fill = "lightblue", color = "black", alpha = 0.7) +
  geom_density(color = "red", size = 1) +
  labs(title = "Distribution of MPG with Density Overlay",
       x = "Miles per Gallon",
       y = "Density") +
  theme_minimal()

# Faceted histogram by groups
ggplot(mtcars, aes(x = mpg, fill = factor(cyl))) +
  geom_histogram(bins = 8, alpha = 0.7, position = "identity") +
  facet_wrap(~cyl, scales = "free_y") +
  labs(title = "MPG Distribution by Cylinders",
       fill = "Cylinders") +
  theme_bw()
```

#### Box Plots and Violin Plots

```r
# Box plot with individual points
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5, color = "darkblue") +
  labs(title = "MPG by Cylinders with Individual Points",
       x = "Number of Cylinders",
       y = "Miles per Gallon") +
  theme_classic()

# Violin plot showing full distribution shape
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  geom_violin(fill = "lightgreen", alpha = 0.7) +
  geom_boxplot(width = 0.2, fill = "white", alpha = 0.8) +
  labs(title = "MPG Distribution by Cylinders (Violin Plot)",
       x = "Number of Cylinders",
       y = "Miles per Gallon")
```

#### Bar Plots

```r
# Count bar plot
ggplot(mtcars, aes(x = factor(cyl))) +
  geom_bar(fill = "steelblue", alpha = 0.8) +
  labs(title = "Count of Cars by Cylinders",
       x = "Number of Cylinders",
       y = "Count") +
  theme_minimal()

# Summary bar plot with error bars
cyl_summary <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    mean_mpg = mean(mpg),
    se_mpg = sd(mpg) / sqrt(n()),
    .groups = 'drop'
  )

ggplot(cyl_summary, aes(x = factor(cyl), y = mean_mpg)) +
  geom_bar(stat = "identity", fill = "lightcoral", alpha = 0.8) +
  geom_errorbar(aes(ymin = mean_mpg - se_mpg, ymax = mean_mpg + se_mpg),
                width = 0.2, color = "darkred") +
  labs(title = "Average MPG by Cylinders with Standard Error",
       x = "Number of Cylinders",
       y = "Average MPG") +
  theme_bw()
```

#### Line Plots

```r
# Create time series data with mathematical functions
time_data <- data.frame(
  time = 1:100,
  linear = 1:100,
  quadratic = (1:100)^2,
  exponential = exp((1:100)/20),
  sine = sin((1:100)/10) * 50 + 50
)

# Multiple lines on same plot
ggplot(time_data, aes(x = time)) +
  geom_line(aes(y = linear, color = "Linear"), size = 1) +
  geom_line(aes(y = quadratic, color = "Quadratic"), size = 1) +
  geom_line(aes(y = exponential, color = "Exponential"), size = 1) +
  geom_line(aes(y = sine, color = "Sine"), size = 1) +
  labs(title = "Mathematical Functions Comparison",
       x = "Time",
       y = "Value",
       color = "Function Type") +
  scale_color_brewer(type = "qual", palette = "Set1") +
  theme_minimal()
```

### Faceting

Faceting creates multiple plots based on categorical variables:

```r
# Facet by one variable
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  facet_wrap(~cyl, scales = "free") +
  labs(title = "MPG vs Weight by Cylinders",
       subtitle = "Each panel shows relationship for different cylinder counts")

# Facet by two variables (transmission and cylinders)
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  facet_grid(am ~ cyl, labeller = label_both) +
  labs(title = "MPG vs Weight by Transmission and Cylinders",
       x = "Weight (1000 lbs)",
       y = "Miles per Gallon")
```

### Themes and Customization

```r
# Apply different themes
p <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  labs(title = "MPG vs Weight")

# Different built-in themes
p + theme_minimal()    # Clean, minimal design
p + theme_classic()    # Classic R style
p + theme_dark()       # Dark theme
p + theme_bw()         # Black and white
p + theme_void()       # No axes or grid

# Custom theme with mathematical precision
p + theme(
  plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
  axis.title = element_text(size = 12, face = "italic"),
  axis.text = element_text(size = 10),
  panel.background = element_rect(fill = "white"),
  panel.grid.major = element_line(color = "gray90"),
  panel.grid.minor = element_line(color = "gray95"),
  legend.position = "bottom",
  legend.title = element_text(size = 11, face = "bold")
)
```

## Advanced Visualizations

### Correlation Matrix

```r
# Load required packages
library(corrplot)

# Calculate correlation matrix with mathematical precision
cor_matrix <- cor(mtcars[, c("mpg", "cyl", "disp", "hp", "wt", "qsec")])

# Create correlation plot with different methods
corrplot(cor_matrix, 
         method = "color",
         type = "upper",
         addCoef.col = "black",
         tl.col = "black",
         tl.srt = 45,
         diag = FALSE)

# Alternative: Using ggplot2 for correlation heatmap
library(reshape2)
cor_data <- melt(cor_matrix)
names(cor_data) <- c("Var1", "Var2", "Correlation")

ggplot(cor_data, aes(x = Var1, y = Var2, fill = Correlation)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", Correlation)), 
            color = "white", size = 3) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Correlation Matrix Heatmap",
       x = "", y = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

### Violin Plots

```r
# Violin plot with statistical annotations
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  geom_violin(fill = "lightblue", alpha = 0.7) +
  geom_boxplot(width = 0.2, fill = "white", alpha = 0.8) +
  stat_summary(fun = mean, geom = "point", shape = 23, 
               size = 3, fill = "red") +
  labs(title = "MPG Distribution by Cylinders",
       subtitle = "Red diamond shows mean, box shows quartiles",
       x = "Number of Cylinders",
       y = "Miles per Gallon") +
  theme_bw()
```

### Heatmaps

```r
# Create heatmap data with scaling
heatmap_data <- mtcars %>%
  select(mpg, cyl, disp, hp, wt, qsec) %>%
  scale() %>%
  as.data.frame()

# Create heatmap with mathematical scaling
heatmap(as.matrix(heatmap_data), 
        main = "Variable Correlation Heatmap (Z-scores)",
        col = colorRampPalette(c("blue", "white", "red"))(100),
        scale = "none",
        Rowv = TRUE,
        Colv = TRUE)
```

### 3D Scatter Plots

```r
# Load required package
library(scatterplot3d)

# 3D scatter plot with mathematical perspective
scatterplot3d(mtcars$wt, mtcars$hp, mtcars$mpg,
              main = "3D Scatter Plot: Weight, Horsepower, MPG",
              xlab = "Weight",
              ylab = "Horsepower",
              zlab = "MPG",
              color = mtcars$cyl,
              pch = 16,
              angle = 45)  # Viewing angle
```

## Statistical Visualizations

### Q-Q Plots

Q-Q plots compare sample quantiles to theoretical quantiles:

```r
# Q-Q plot for normality testing
qqnorm(mtcars$mpg, main = "Normal Q-Q Plot for MPG")
qqline(mtcars$mpg, col = "red", lwd = 2)

# Using ggplot2 with mathematical annotations
ggplot(mtcars, aes(sample = mpg)) +
  stat_qq() +
  stat_qq_line(color = "red", size = 1) +
  labs(title = "Q-Q Plot for MPG",
       subtitle = "Points should follow red line for normality",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles") +
  theme_bw()

# Shapiro-Wilk test for normality
shapiro_result <- shapiro.test(mtcars$mpg)
cat("Shapiro-Wilk test p-value:", shapiro_result$p.value, "\n")
```

### Residual Plots

```r
# Fit linear model
model <- lm(mpg ~ wt, data = mtcars)

# Residual plot with mathematical diagnostics
plot(fitted(model), residuals(model),
     main = "Residual Plot",
     xlab = "Fitted Values",
     ylab = "Residuals",
     pch = 16)
abline(h = 0, col = "red", lwd = 2)

# Add confidence bands for residuals
residual_data <- data.frame(
  fitted = fitted(model),
  residuals = residuals(model)
)

# Calculate residual standard error
rse <- sqrt(sum(residuals(model)^2) / (length(residuals(model)) - 2)

ggplot(residual_data, aes(x = fitted, y = residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red", size = 1) +
  geom_hline(yintercept = c(-2*rse, 2*rse), color = "red", 
             linetype = "dashed", alpha = 0.7) +
  labs(title = "Residual Plot with 95% Confidence Bands",
       subtitle = paste("Residual Standard Error =", round(rse, 3)),
       x = "Fitted Values",
       y = "Residuals") +
  theme_bw()
```

### Distribution Comparison

```r
# Compare distributions across groups
ggplot(mtcars, aes(x = mpg, fill = factor(cyl))) +
  geom_density(alpha = 0.5) +
  labs(title = "MPG Density by Cylinders",
       x = "Miles per Gallon",
       y = "Density",
       fill = "Cylinders") +
  theme_minimal()

# Statistical summary by groups
mtcars %>%
  group_by(cyl) %>%
  summarise(
    n = n(),
    mean_mpg = mean(mpg),
    sd_mpg = sd(mpg),
    median_mpg = median(mpg),
    q25 = quantile(mpg, 0.25),
    q75 = quantile(mpg, 0.75),
    .groups = 'drop'
  ) %>%
  print()
```

## Interactive Visualizations

### Plotly

```r
# Install and load plotly
install.packages("plotly")
library(plotly)

# Create interactive scatter plot with mathematical annotations
p <- ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "Interactive MPG vs Weight",
       x = "Weight (1000 lbs)",
       y = "Miles per Gallon",
       color = "Cylinders") +
  theme_minimal()

ggplotly(p, tooltip = c("x", "y", "color"))
```

### Highcharter

```r
# Install and load highcharter
install.packages("highcharter")
library(highcharter)

# Create interactive chart with statistical features
hchart(mtcars, "scatter", hcaes(wt, mpg, group = cyl)) %>%
  hc_title(text = "Interactive MPG vs Weight") %>%
  hc_xAxis(title = list(text = "Weight")) %>%
  hc_yAxis(title = list(text = "MPG")) %>%
  hc_tooltip(pointFormat = "{point.x:.2f}, {point.y:.2f}") %>%
  hc_add_theme(hc_theme_flat())
```

## Best Practices

### Color Choices

```r
# Color-blind friendly palette (mathematically designed)
colorblind_palette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", 
                       "#0072B2", "#D55E00", "#CC79A7")

# Use in plot with mathematical color theory
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point(size = 3) +
  scale_color_manual(values = colorblind_palette) +
  labs(title = "MPG vs Weight with Colorblind-Friendly Colors",
       color = "Cylinders") +
  theme_bw()

# Sequential color palette for continuous data
ggplot(mtcars, aes(x = wt, y = mpg, color = hp)) +
  geom_point(size = 3) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "MPG vs Weight by Horsepower",
       color = "Horsepower") +
  theme_minimal()
```

### Saving Plots

```r
# Save plot as PNG with mathematical precision
p <- ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "MPG vs Weight with Regression",
       x = "Weight (1000 lbs)",
       y = "Miles per Gallon") +
  theme_bw()

# Save with high resolution for publication
ggsave("mpg_vs_weight.png", p, 
       width = 8, height = 6, dpi = 300, 
       units = "in")

# Save as PDF for vector graphics
ggsave("mpg_vs_weight.pdf", p, 
       width = 8, height = 6, 
       units = "in")
```

### Mathematical Annotations

```r
# Add mathematical formulas and statistical information
model <- lm(mpg ~ wt, data = mtcars)
r_squared <- summary(model)$r.squared
p_value <- summary(model)$coefficients[2, 4]

ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  annotate("text", x = 4, y = 30, 
           label = paste("R² =", round(r_squared, 3)), 
           color = "red", fontface = "bold") +
  annotate("text", x = 4, y = 28, 
           label = paste("p <", format.pval(p_value, digits = 3)), 
           color = "red", fontface = "bold") +
  labs(title = "MPG vs Weight with Statistical Information",
       x = "Weight (1000 lbs)",
       y = "Miles per Gallon") +
  theme_bw()
```

## Practical Examples

### Example 1: Comprehensive Data Analysis

```r
# Load data
data(iris)

# Create multiple plots with mathematical insights
p1 <- ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "Sepal Length vs Width by Species",
       subtitle = paste("Overall correlation:", 
                       round(cor(iris$Sepal.Length, iris$Sepal.Width), 3)))

p2 <- ggplot(iris, aes(x = Species, y = Sepal.Length)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7) +
  stat_summary(fun = mean, geom = "point", shape = 23, 
               size = 3, fill = "red") +
  labs(title = "Sepal Length by Species",
       subtitle = "Red diamond shows mean")

p3 <- ggplot(iris, aes(x = Sepal.Length, fill = Species)) +
  geom_density(alpha = 0.5) +
  labs(title = "Sepal Length Density by Species")

# Display plots
print(p1)
print(p2)
print(p3)
```

### Example 2: Time Series Visualization

```r
# Create time series data with mathematical functions
set.seed(123)
dates <- seq(as.Date("2023-01-01"), as.Date("2023-12-31"), by = "month")
trend <- 1:12
seasonal <- sin(2 * pi * (1:12) / 12) * 10
noise <- rnorm(12, 0, 2)
values <- trend + seasonal + noise

ts_data <- data.frame(
  date = dates, 
  value = values,
  trend = trend,
  seasonal = seasonal
)

# Time series plot with components
ggplot(ts_data, aes(x = date)) +
  geom_line(aes(y = value, color = "Observed"), size = 1) +
  geom_line(aes(y = trend, color = "Trend"), size = 1, linetype = "dashed") +
  geom_line(aes(y = seasonal, color = "Seasonal"), size = 1, linetype = "dotted") +
  labs(title = "Time Series with Trend and Seasonal Components",
       x = "Date",
       y = "Value",
       color = "Component") +
  scale_color_manual(values = c("Observed" = "black", 
                               "Trend" = "red", 
                               "Seasonal" = "blue")) +
  theme_minimal()
```

### Example 3: Statistical Process Control

```r
# Create control chart data
set.seed(456)
process_data <- data.frame(
  sample = 1:30,
  measurement = c(rnorm(20, 100, 2), rnorm(10, 105, 2))  # Process shift
)

# Calculate control limits
mean_val <- mean(process_data$measurement[1:20])
sd_val <- sd(process_data$measurement[1:20])

process_data$ucl <- mean_val + 3 * sd_val
process_data$lcl <- mean_val - 3 * sd_val
process_data$ucl_warning <- mean_val + 2 * sd_val
process_data$lcl_warning <- mean_val - 2 * sd_val

# Control chart
ggplot(process_data, aes(x = sample, y = measurement)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "blue", size = 2) +
  geom_line(aes(y = ucl), color = "red", linetype = "dashed") +
  geom_line(aes(y = lcl), color = "red", linetype = "dashed") +
  geom_line(aes(y = ucl_warning), color = "orange", linetype = "dotted") +
  geom_line(aes(y = lcl_warning), color = "orange", linetype = "dotted") +
  geom_hline(yintercept = mean_val, color = "green", linetype = "solid") +
  labs(title = "Statistical Process Control Chart",
       subtitle = "Green = Center line, Red = Control limits, Orange = Warning limits",
       x = "Sample Number",
       y = "Measurement") +
  theme_bw()
```

## Exercises

### Exercise 1: Basic Plots
Create scatter plots, histograms, and box plots for the `mtcars` dataset. Calculate and display correlation coefficients and summary statistics.

### Exercise 2: Customization
Create a publication-ready plot with custom colors, themes, and mathematical annotations including R² values and p-values.

### Exercise 3: Faceting
Create faceted plots to show relationships across different groups. Include statistical summaries for each facet.

### Exercise 4: Interactive Plots
Create an interactive visualization using plotly or highcharter with hover information and zoom capabilities.

### Exercise 5: Statistical Plots
Create Q-Q plots and residual plots for a linear regression model. Perform normality tests and interpret the results.

### Exercise 6: Advanced Visualization
Create a correlation matrix heatmap and a 3D scatter plot. Interpret the mathematical relationships shown.

## Next Steps

In the next chapter, we'll learn about probability distributions and how to work with them in R, building on the visualization concepts we've covered here.

---

**Key Takeaways:**
- Base R graphics are good for quick plots and mathematical precision
- ggplot2 provides a consistent grammar of graphics with mathematical foundations
- Choose appropriate plot types based on data characteristics and research questions
- Use color and themes effectively while considering accessibility
- Include mathematical annotations and statistical information
- Save plots in appropriate formats for different purposes
- Interactive plots enhance exploration and communication
- Always include proper labels, titles, and mathematical context
- Consider the mathematical relationships and statistical assumptions in your visualizations 