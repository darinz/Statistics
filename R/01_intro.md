# Introduction to R

## What is R?

R is a programming language and environment specifically designed for statistical computing and graphics. It was created by Ross Ihaka and Robert Gentleman at the University of Auckland, New Zealand, and is now maintained by the R Development Core Team.

### Key Features of R

- **Statistical Computing**: Built-in functions for statistical analysis, modeling, and testing
- **Graphics**: Comprehensive plotting capabilities with base R and packages like ggplot2
- **Data Manipulation**: Powerful tools for cleaning, transforming, and analyzing data
- **Reproducible Research**: Integration with R Markdown for creating reports
- **Extensibility**: Thousands of packages available through CRAN and other repositories
- **Open Source**: Free to use, modify, and distribute

## Getting Started with R

### Installing R and RStudio

1. **Install R**: Download from [r-project.org](https://www.r-project.org/)
2. **Install RStudio**: Download from [posit.co](https://posit.co/)

RStudio is an integrated development environment (IDE) that makes working with R much easier. It provides:
- Code editor with syntax highlighting
- Console for running R commands
- Environment pane to view objects
- Plot pane to display graphics
- File browser and help documentation

### Your First R Session

Let's start with some basic operations:

```r
# This is a comment
# Basic arithmetic
2 + 3
5 * 4
10 / 2

# Creating variables
x <- 5
y <- 10
z <- x + y
print(z)

# Vectors (basic data structure)
numbers <- c(1, 2, 3, 4, 5)
mean(numbers)
sd(numbers)
```

## R as a Calculator

R can perform all basic mathematical operations:

```r
# Addition
3 + 4

# Subtraction
10 - 5

# Multiplication
6 * 7

# Division
20 / 4

# Exponentiation
2^3

# Square root
sqrt(16)

# Natural logarithm
log(10)

# Exponential
exp(1)
```

## Variables and Assignment

In R, you assign values to variables using `<-` (preferred) or `=`:

```r
# Assignment
x <- 5
y = 10

# Variable names can contain letters, numbers, dots, and underscores
my_variable <- 42
data.2023 <- "some value"

# Check what variables exist
ls()

# Remove a variable
rm(x)
```

## Data Types in R

R has several basic data types:

### Numeric
```r
# Integer and double
age <- 25L        # integer
height <- 175.5   # double
```

### Character
```r
name <- "John"
city <- 'New York'
```

### Logical
```r
is_student <- TRUE
is_working <- FALSE
```

### Factor (for categorical data)
```r
gender <- factor(c("Male", "Female", "Male", "Female"))
levels(gender)
```

## Data Structures

### Vectors
Vectors are the most basic data structure in R:

```r
# Numeric vector
numbers <- c(1, 2, 3, 4, 5)

# Character vector
names <- c("Alice", "Bob", "Charlie")

# Logical vector
flags <- c(TRUE, FALSE, TRUE)

# Vector operations
numbers * 2
numbers + 10
```

### Data Frames
Data frames are like tables with rows and columns:

```r
# Create a data frame
students <- data.frame(
  name = c("Alice", "Bob", "Charlie"),
  age = c(20, 22, 21),
  grade = c(85, 92, 78)
)

# View the data frame
print(students)
str(students)
head(students)
```

### Lists
Lists can contain different types of data:

```r
person <- list(
  name = "John",
  age = 30,
  scores = c(85, 92, 78),
  is_student = TRUE
)

# Access list elements
person$name
person[[1]]
```

## Working with Data

### Reading Data
R can read data from various formats:

```r
# Read CSV file
data <- read.csv("filename.csv")

# Read Excel file (requires readxl package)
library(readxl)
data <- read_excel("filename.xlsx")

# Read text file
data <- read.table("filename.txt", header = TRUE)
```

### Basic Data Exploration

```r
# Load built-in dataset
data(mtcars)

# View first few rows
head(mtcars)

# Get dimensions
dim(mtcars)

# Get column names
names(mtcars)
colnames(mtcars)

# Get data types
str(mtcars)

# Summary statistics
summary(mtcars)

# Get help on dataset
?mtcars
```

## Basic Graphics

R has excellent plotting capabilities:

```r
# Scatter plot
plot(mtcars$wt, mtcars$mpg, 
     xlab = "Weight", ylab = "MPG",
     main = "MPG vs Weight")

# Histogram
hist(mtcars$mpg, 
     xlab = "MPG", 
     main = "Distribution of MPG")

# Box plot
boxplot(mpg ~ cyl, data = mtcars,
        xlab = "Number of Cylinders",
        ylab = "MPG",
        main = "MPG by Number of Cylinders")
```

## Getting Help

R has extensive help documentation:

```r
# Get help on a function
?mean
help(mean)

# Search for functions
??regression

# Get help on a package
help(package = "ggplot2")

# Get examples
example(plot)
```

## Best Practices

1. **Use meaningful variable names**
2. **Comment your code**
3. **Use spaces around operators**
4. **Keep lines under 80 characters**
5. **Use consistent indentation**
6. **Save your work regularly**

## Next Steps

After mastering these basics, you'll be ready to:
- Learn about the tidyverse packages
- Perform statistical analyses
- Create reproducible reports with R Markdown
- Build interactive applications with Shiny
- Work with larger datasets efficiently

## Resources

- [R Documentation](https://www.r-project.org/docs.html)
- [RStudio Cheat Sheets](https://www.rstudio.com/resources/cheatsheets/)
- [R for Data Science](https://r4ds.had.co.nz/)
- [R Markdown: The Definitive Guide](https://bookdown.org/yihui/rmarkdown/)

---

*This content is adapted from the Applied Statistics with R textbook and enhanced for learning purposes.* 