# Summarizing Data with R

## Introduction

Data summarization is a fundamental concept in statistics that involves condensing large datasets into meaningful, interpretable information. This process helps us understand patterns, trends, and characteristics of our data without having to examine every individual observation. R provides powerful packages for data summarization including base R, dplyr, ggplot2, and various statistical packages.

## Types of Data Summaries

### 1. Descriptive Statistics

Descriptive statistics provide numerical summaries of data characteristics:

#### Measures of Central Tendency
- **Mean**: The arithmetic average of all values
- **Median**: The middle value when data is ordered
- **Mode**: The most frequently occurring value

#### Measures of Variability
- **Range**: Difference between maximum and minimum values
- **Variance**: Average squared deviation from the mean
- **Standard Deviation**: Square root of variance
- **Interquartile Range (IQR)**: Range of the middle 50% of data

### 2. Graphical Summaries

Visual representations that help understand data distribution:

#### For Quantitative Data
- **Histograms**: Show frequency distribution of continuous data
- **Box Plots**: Display quartiles, median, and outliers
- **Stem-and-Leaf Plots**: Show data distribution with actual values
- **Scatter Plots**: Show relationship between two variables

#### For Categorical Data
- **Bar Charts**: Show frequency of categories
- **Pie Charts**: Show proportions of categories
- **Mosaic Plots**: Show relationships between categorical variables

## Key Concepts

### 1. Shape of Distribution
- **Symmetric**: Data evenly distributed around center
- **Skewed**: Data concentrated on one side
  - Right-skewed (positive): Long tail to the right
  - Left-skewed (negative): Long tail to the left
- **Unimodal**: Single peak in distribution
- **Bimodal**: Two distinct peaks

### 2. Outliers
- Data points that fall far from the main cluster
- Can significantly affect mean but not median
- Important to identify and understand their cause

### 3. Summary Statistics by Data Type

#### Quantitative Data
```r
# R example with comprehensive analysis
library(dplyr)
library(ggplot2)
library(moments)

# Sample dataset: Student test scores
scores <- c(65, 72, 68, 85, 90, 78, 82, 75, 88, 92, 70, 85, 79, 83, 91)
cat("Sample size:", length(scores), "\n")

# Calculate summary statistics
mean_score <- mean(scores)
median_score <- median(scores)
std_score <- sd(scores)
variance_score <- var(scores)
q1 <- quantile(scores, 0.25)
q3 <- quantile(scores, 0.75)
iqr <- IQR(scores)
range_score <- max(scores) - min(scores)

cat("Mean:", mean_score, "\n")
cat("Median:", median_score, "\n")
cat("Standard Deviation:", std_score, "\n")
cat("Variance:", variance_score, "\n")
cat("Q1:", q1, "\n")
cat("Q3:", q3, "\n")
cat("IQR:", iqr, "\n")
cat("Range:", range_score, "\n")

# Check for skewness
skewness <- skewness(scores)
cat("Skewness:", skewness, "\n")

# Comprehensive summary
cat("\nComprehensive Summary:\n")
print(summary(scores))

# Create comprehensive visualizations
par(mfrow=c(2, 3))

# Histogram with mean and median lines
hist(scores, main="Distribution of Test Scores", 
     xlab="Score", ylab="Frequency", col="lightblue", border="black")
abline(v=mean_score, col="red", lty=2, lwd=2)
abline(v=median_score, col="green", lty=2, lwd=2)
legend("topright", legend=c(paste("Mean:", round(mean_score, 1)), 
                           paste("Median:", round(median_score, 1))),
       col=c("red", "green"), lty=2, lwd=2)

# Box plot
boxplot(scores, main="Box Plot of Test Scores", 
        ylab="Score", col="lightgreen", border="darkgreen")

# Stem-and-leaf plot
stem(scores, main="Stem-and-Leaf Plot")

# Q-Q plot for normality
qqnorm(scores, main="Q-Q Plot for Normality")
qqline(scores, col="red")

# Cumulative distribution
plot(sort(scores), (1:length(scores))/length(scores), 
     type="l", main="Cumulative Distribution",
     xlab="Score", ylab="Cumulative Probability", col="blue", lwd=2)

# Summary statistics table
summary_df <- data.frame(
  Statistic = c("Count", "Mean", "Median", "Std Dev", "Variance", 
                "Min", "Q1", "Q2", "Q3", "Max", "IQR"),
  Value = c(length(scores), round(mean_score, 2), round(median_score, 2),
            round(std_score, 2), round(variance_score, 2), min(scores),
            round(q1, 2), round(median_score, 2), round(q3, 2), 
            max(scores), round(iqr, 2))
)
print(summary_df)

# Reset plot layout
par(mfrow=c(1, 1))

# Additional statistical measures
cat("\nAdditional Statistics:\n")
cat("Coefficient of Variation:", round((std_score/mean_score)*100, 2), "%\n")
cat("Standard Error:", round(std_score/sqrt(length(scores)), 2), "\n")
cat("95% Confidence Interval: [", 
    round(mean_score - 1.96*std_score/sqrt(length(scores)), 2), ", ",
    round(mean_score + 1.96*std_score/sqrt(length(scores)), 2), "]\n")

# Shapiro-Wilk test for normality
shapiro_test <- shapiro.test(scores)
cat("Shapiro-Wilk test p-value:", shapiro_test$p.value, "\n")
```

#### Categorical Data
```r
# R example with comprehensive categorical analysis
library(dplyr)
library(ggplot2)
library(vcd)

# Sample dataset: Student majors and grades
data <- data.frame(
  major = c("Math", "Physics", "Math", "Chemistry", "Physics", "Math", 
            "Chemistry", "Physics", "Math", "Chemistry", "Biology", "Physics"),
  grade = c("A", "B", "A", "C", "B", "A", "B", "A", "C", "B", "A", "B"),
  year = c("Freshman", "Sophomore", "Junior", "Senior", "Freshman",
           "Sophomore", "Junior", "Senior", "Freshman", "Sophomore", "Junior", "Senior"),
  gpa = c(3.8, 3.2, 3.9, 2.8, 3.4, 3.7, 3.1, 3.6, 2.9, 3.3, 3.5, 3.0)
)

# Frequency tables with percentages
cat("Major distribution:\n")
major_counts <- table(data$major)
major_percentages <- prop.table(major_counts) * 100
major_summary <- data.frame(
  Count = as.numeric(major_counts),
  Percentage = round(as.numeric(major_percentages), 2)
)
rownames(major_summary) <- names(major_counts)
print(major_summary)

cat("\nGrade distribution:\n")
grade_counts <- table(data$grade)
grade_percentages <- prop.table(grade_counts) * 100
grade_summary <- data.frame(
  Count = as.numeric(grade_counts),
  Percentage = round(as.numeric(grade_percentages), 2)
)
rownames(grade_summary) <- names(grade_counts)
print(grade_summary)

cat("\nYear distribution:\n")
year_counts <- table(data$year)
year_percentages <- prop.table(year_counts) * 100
year_summary <- data.frame(
  Count = as.numeric(year_counts),
  Percentage = round(as.numeric(year_percentages), 2)
)
rownames(year_summary) <- names(year_counts)
print(year_summary)

# Create comprehensive visualizations
par(mfrow=c(2, 3))

# Bar chart for majors
barplot(major_counts, main="Distribution of Majors", 
        col="skyblue", ylab="Count", border="darkblue")

# Pie chart for grades
pie(grade_counts, main="Grade Distribution", 
    col=rainbow(length(grade_counts)))

# Bar chart for years
barplot(year_counts, main="Distribution by Year", 
        col="lightgreen", ylab="Count", las=2, border="darkgreen")

# Cross-tabulation heatmap
cross_tab <- table(data$major, data$grade)
mosaicplot(cross_tab, main="Major vs Grade Cross-tabulation", 
           color=TRUE, shade=TRUE)

# GPA by major box plot
boxplot(gpa ~ major, data=data, main="GPA Distribution by Major",
        col="lightblue", border="darkblue", las=2)

# GPA by grade box plot
boxplot(gpa ~ grade, data=data, main="GPA Distribution by Grade",
        col="lightgreen", border="darkgreen")

# Reset plot layout
par(mfrow=c(1, 1))

# Cross-tabulation (contingency table)
cat("\nCross-tabulation of Major vs Grade:\n")
print(cross_tab)

# Chi-square test for independence
chi2_test <- chisq.test(cross_tab)
cat("\nChi-square test for independence:\n")
cat("Chi-square statistic:", chi2_test$statistic, "\n")
cat("p-value:", chi2_test$p.value, "\n")
cat("Degrees of freedom:", chi2_test$parameter, "\n")

# Summary statistics by group
cat("\nSummary statistics by major:\n")
print(data %>% group_by(major) %>% 
      summarise(count=n(), mean_gpa=mean(gpa), sd_gpa=sd(gpa), 
                min_gpa=min(gpa), max_gpa=max(gpa)) %>% 
      round(2))

cat("\nSummary statistics by grade:\n")
print(data %>% group_by(grade) %>% 
      summarise(count=n(), mean_gpa=mean(gpa), sd_gpa=sd(gpa), 
                min_gpa=min(gpa), max_gpa=max(gpa)) %>% 
      round(2))
```

## Best Practices

### 1. Always Start with Exploratory Data Analysis (EDA)
- Examine data structure and types
- Check for missing values
- Look for obvious patterns or anomalies

### 2. Choose Appropriate Summaries
- Use mean and standard deviation for symmetric data
- Use median and IQR for skewed data
- Consider the data type (quantitative vs categorical)

### 3. Visualize Before Summarizing
- Graphs often reveal patterns not apparent in numbers
- Help identify outliers and unusual patterns
- Provide context for numerical summaries

### 4. Report with Context
- Always include units of measurement
- Provide sample size
- Consider the audience and purpose

## Common Mistakes to Avoid

1. **Using mean for skewed data**: Median is more robust
2. **Ignoring outliers**: Can significantly affect summaries
3. **Overlooking data type**: Different summaries for different types
4. **Missing context**: Numbers without interpretation
5. **Over-summarizing**: Losing important details

## Real-World Examples

### Example 1: Car Fuel Efficiency Data
```r
# R example with real car data using built-in mtcars dataset
library(dplyr)
library(ggplot2)
library(corrplot)

# Load built-in mtcars dataset
data(mtcars)

cat("Car Fuel Efficiency Summary:\n")
cat("Number of cars:", nrow(mtcars), "\n")
cat("Mean MPG:", mean(mtcars$mpg), "\n")
cat("Median MPG:", median(mtcars$mpg), "\n")
cat("Standard Deviation:", sd(mtcars$mpg), "\n")

# Group by transmission type
manual_mpg <- mtcars$mpg[mtcars$am == 1]
auto_mpg <- mtcars$mpg[mtcars$am == 0]

cat("\nManual transmission cars:\n")
cat("  Mean MPG:", mean(manual_mpg), "\n")
cat("  Count:", length(manual_mpg), "\n")
cat("  Std Dev:", sd(manual_mpg), "\n")

cat("\nAutomatic transmission cars:\n")
cat("  Mean MPG:", mean(auto_mpg), "\n")
cat("  Count:", length(auto_mpg), "\n")
cat("  Std Dev:", sd(auto_mpg), "\n")

# Statistical test for difference in means
t_test <- t.test(manual_mpg, auto_mpg)
cat("\nT-test for difference in MPG between transmission types:\n")
cat("T-statistic:", t_test$statistic, "\n")
cat("p-value:", t_test$p.value, "\n")

# Create comprehensive visualizations
par(mfrow=c(2, 3))

# MPG distribution
hist(mtcars$mpg, main="Distribution of MPG", xlab="MPG", ylab="Frequency",
     col="lightblue", border="darkblue")
abline(v=mean(mtcars$mpg), col="red", lty=2, lwd=2)
abline(v=median(mtcars$mpg), col="green", lty=2, lwd=2)
legend("topright", legend=c(paste("Mean:", round(mean(mtcars$mpg), 1)), 
                           paste("Median:", round(median(mtcars$mpg), 1))),
       col=c("red", "green"), lty=2, lwd=2)

# MPG by transmission type
boxplot(mpg ~ am, data=mtcars, names=c("Automatic", "Manual"),
        main="MPG by Transmission Type", ylab="MPG",
        col=c("lightgreen", "lightblue"), border=c("darkgreen", "darkblue"))

# Scatter plot: MPG vs Weight
plot(mtcars$wt, mtcars$mpg, xlab="Weight (1000 lbs)", ylab="MPG",
     main="MPG vs Weight", pch=19, col="blue")

# Correlation heatmap
cor_matrix <- cor(mtcars[, c("mpg", "cyl", "disp", "hp", "wt")])
corrplot(cor_matrix, method="color", type="upper", order="hclust",
         tl.col="black", tl.srt=45, main="Correlation Matrix")

# MPG by number of cylinders
cyl_mpg <- mtcars %>% group_by(cyl) %>% 
           summarise(mean_mpg=mean(mpg), sd_mpg=sd(mpg), count=n())
barplot(cyl_mpg$mean_mpg, names.arg=cyl_mpg$cyl, 
        main="Mean MPG by Number of Cylinders",
        xlab="Number of Cylinders", ylab="Mean MPG",
        col="orange", border="darkorange")

# Pair plot
pairs(mtcars[, c("mpg", "wt", "hp", "disp")], main="Pair Plot of Key Variables")

# Reset plot layout
par(mfrow=c(1, 1))

# Additional analysis
cat("\nCorrelation with MPG:\n")
correlations <- cor(mtcars[, c("cyl", "disp", "hp", "wt")], mtcars$mpg)
for(i in 1:length(correlations)) {
  cat(names(correlations)[i], ":", round(correlations[i], 3), "\n")
}

# Multiple regression analysis
model <- lm(mpg ~ cyl + disp + hp + wt, data=mtcars)
cat("\nMultiple Regression Results:\n")
cat("R-squared:", round(summary(model)$r.squared, 3), "\n")
cat("Intercept:", round(coef(model)[1], 3), "\n")
for(i in 2:length(coef(model))) {
  cat(names(coef(model))[i], ":", round(coef(model)[i], 3), "\n")
}
```

### Example 2: Outlier Detection and Handling
```r
# R example: Comprehensive outlier detection and handling
library(dplyr)
library(ggplot2)

# Sample data with outliers
set.seed(42)
normal_data <- rnorm(95, mean=50, sd=10)
outlier_data <- c(normal_data, 150, 200, -50, 300, -100)

# Calculate statistics
mean_val <- mean(outlier_data)
median_val <- median(outlier_data)
std_val <- sd(outlier_data)
q1 <- quantile(outlier_data, 0.25)
q3 <- quantile(outlier_data, 0.75)
iqr <- IQR(outlier_data)

cat("Original data statistics:\n")
cat("Sample size:", length(outlier_data), "\n")
cat("Mean:", mean_val, "\n")
cat("Median:", median_val, "\n")
cat("Standard Deviation:", std_val, "\n")
cat("Q1:", q1, "\n")
cat("Q3:", q3, "\n")
cat("IQR:", iqr, "\n")

# Multiple outlier detection methods
# Method 1: Z-score method
z_scores <- abs(scale(outlier_data))
outliers_z <- outlier_data[z_scores > 2]

# Method 2: IQR method
lower_bound <- q1 - 1.5 * iqr
upper_bound <- q3 + 1.5 * iqr
outliers_iqr <- outlier_data[outlier_data < lower_bound | outlier_data > upper_bound]

# Method 3: Modified Z-score method (more robust)
median_abs_dev <- median(abs(outlier_data - median_val))
modified_z_scores <- 0.6745 * (outlier_data - median_val) / median_abs_dev
outliers_modified_z <- outlier_data[abs(modified_z_scores) > 3.5]

cat("\nOutlier detection results:\n")
cat("Z-score method (>2):", length(outliers_z), "outliers\n")
cat("IQR method:", length(outliers_iqr), "outliers\n")
cat("Modified Z-score method (>3.5):", length(outliers_modified_z), "outliers\n")

# Remove outliers and recalculate
data_clean <- outlier_data[outlier_data >= lower_bound & outlier_data <= upper_bound]
mean_clean <- mean(data_clean)
median_clean <- median(data_clean)
std_clean <- sd(data_clean)

cat("\nAfter removing outliers (IQR method):\n")
cat("Mean:", mean_clean, "\n")
cat("Median:", median_clean, "\n")
cat("Standard Deviation:", std_clean, "\n")

# Comprehensive visualization
par(mfrow=c(2, 3))

# Original data histogram
hist(outlier_data, main="Distribution with Outliers", 
     xlab="Values", ylab="Frequency", col="lightblue", border="darkblue")
abline(v=mean_val, col="red", lty=2, lwd=2)
abline(v=median_val, col="green", lty=2, lwd=2)
legend("topright", legend=c(paste("Mean:", round(mean_val, 1)), 
                           paste("Median:", round(median_val, 1))),
       col=c("red", "green"), lty=2, lwd=2)

# Box plot showing outliers
boxplot(outlier_data, main="Box Plot with Outliers", 
        ylab="Values", col="lightgreen", border="darkgreen")

# Q-Q plot
qqnorm(outlier_data, main="Q-Q Plot (with outliers)")
qqline(outlier_data, col="red")

# Clean data histogram
hist(data_clean, main="Distribution without Outliers", 
     xlab="Values", ylab="Frequency", col="lightgreen", border="darkgreen")
abline(v=mean_clean, col="red", lty=2, lwd=2)
abline(v=median_clean, col="green", lty=2, lwd=2)
legend("topright", legend=c(paste("Mean:", round(mean_clean, 1)), 
                           paste("Median:", round(median_clean, 1))),
       col=c("red", "green"), lty=2, lwd=2)

# Clean data box plot
boxplot(data_clean, main="Box Plot without Outliers", 
        ylab="Values", col="lightblue", border="darkblue")

# Q-Q plot of clean data
qqnorm(data_clean, main="Q-Q Plot (without outliers)")
qqline(data_clean, col="red")

# Reset plot layout
par(mfrow=c(1, 1))

# Comparison of statistics
comparison_df <- data.frame(
  Statistic = c("Mean", "Median", "Std Dev", "Q1", "Q3", "IQR"),
  With_Outliers = c(mean_val, median_val, std_val, q1, q3, iqr),
  Without_Outliers = c(mean_clean, median_clean, std_clean, 
                       quantile(data_clean, 0.25), quantile(data_clean, 0.75),
                       IQR(data_clean))
)
print(round(comparison_df, 2))

# Statistical tests for normality
cat("\nNormality tests:\n")
cat("Original data - Shapiro-Wilk p-value:", shapiro.test(outlier_data)$p.value, "\n")
cat("Clean data - Shapiro-Wilk p-value:", shapiro.test(data_clean)$p.value, "\n")
```

## Advanced Topics

### 1. Robust Statistics
- Statistics that are resistant to outliers
- Median, IQR, MAD (Median Absolute Deviation)
- Trimmed means

### 2. Multivariate Summaries
- Correlation matrices
- Covariance structures
- Principal component analysis

### 3. Time Series Summaries
- Trend analysis
- Seasonal patterns
- Autocorrelation

## Conclusion

Data summarization is both an art and a science. The key is to choose appropriate methods that accurately represent your data while being meaningful to your audience. Always remember that summaries are tools to help understand data, not replace the need for careful analysis and interpretation.

## Exercises

### Exercise 1: Basic Summary Statistics
Using the following dataset of student heights (in inches), calculate and interpret all relevant summary statistics:

```r
heights <- c(64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78)
```

**Tasks:**
1. Calculate mean, median, mode, standard deviation, variance, range, and IQR
2. Create a histogram and box plot
3. Determine if the data is symmetric, left-skewed, or right-skewed
4. Identify any outliers using both Z-score and IQR methods
5. Test for normality using appropriate statistical tests

### Exercise 2: Categorical Data Analysis
Analyze the following survey data about favorite programming languages:

```r
languages <- c("Python", "R", "Python", "JavaScript", "R", "Python", 
               "Java", "Python", "R", "JavaScript", "Python", "R")
```

**Tasks:**
1. Create a frequency table with percentages
2. Create a bar chart and pie chart
3. Identify the most and least popular languages
4. Calculate confidence intervals for proportions
5. Perform a chi-square test for uniform distribution

### Exercise 3: Real-World Dataset Analysis
Download and analyze a real dataset using R:

```r
# Example: Load iris dataset
data(iris)
```

**Tasks:**
1. Load the dataset and examine its structure
2. Calculate summary statistics for all numeric variables
3. Create appropriate visualizations for each variable
4. Identify any patterns or relationships between variables
5. Check for outliers and missing values
6. Perform statistical tests for normality and correlation
7. Write a brief report summarizing your findings

### Exercise 4: Outlier Analysis
Create a dataset with known outliers and practice outlier detection:

```r
set.seed(42)
normal_data <- rnorm(95, mean=50, sd=10)
outlier_data <- c(normal_data, 150, 200, -50)
```

**Tasks:**
1. Calculate summary statistics for both datasets
2. Compare the effect of outliers on mean vs median
3. Use different methods to detect outliers (Z-score, IQR, modified Z-score)
4. Create visualizations showing the outliers
5. Discuss the impact of outliers on data analysis
6. Implement robust statistical measures

### Exercise 5: Advanced Visualization
Create comprehensive visualizations for a multivariate dataset:

**Tasks:**
1. Create correlation matrices and heatmaps
2. Generate scatter plot matrices
3. Create grouped box plots
4. Develop interactive visualizations using plotly
5. Write code to automatically generate summary reports
6. Create publication-quality figures

### Exercise 6: Statistical Interpretation
Practice interpreting summary statistics in context:

**Scenario:** You're analyzing test scores from two different teaching methods.

**Data:**
- Method A: [75, 78, 82, 85, 88, 90, 92, 95]
- Method B: [65, 70, 75, 80, 85, 90, 95, 100]

**Tasks:**
1. Calculate and compare summary statistics for both methods
2. Create side-by-side box plots
3. Perform appropriate statistical tests
4. Determine which method appears more effective
5. Discuss the limitations of this analysis
6. Suggest additional analyses that might be helpful

### Exercise 7: Data Quality Assessment
Practice assessing data quality and handling common issues:

**Tasks:**
1. Identify and handle missing values using R functions
2. Detect and address data entry errors
3. Check for data consistency
4. Validate data types and ranges
5. Create data quality reports
6. Implement data cleaning pipelines

### Exercise 8: Reporting and Communication
Practice communicating statistical findings:

**Tasks:**
1. Write clear interpretations of summary statistics
2. Create professional visualizations
3. Develop executive summaries
4. Practice explaining technical concepts to non-technical audiences
5. Create reproducible reports with R Markdown
6. Use LaTeX for mathematical notation

## Resources

- **R Packages**: dplyr, ggplot2, tidyr, readr, corrplot, moments
- **Statistical Resources**: base R stats, car, MASS
- **Visualization**: ggplot2, plotly, lattice, base R graphics
- **Documentation**: R documentation, CRAN package documentation
- **Online Courses**: DataCamp, Coursera, edX
- **Books**: "R for Data Science" by Hadley Wickham, "The R Book" by Michael Crawley 