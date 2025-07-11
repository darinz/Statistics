# Introduction to R

## What is R?

R is a powerful programming language and environment for statistical computing and graphics. It was developed by Ross Ihaka and Robert Gentleman at the University of Auckland, New Zealand, and is now maintained by the R Development Core Team. R is both a programming language and a software environment designed specifically for statistical analysis, data manipulation, and graphical representation.

### Why Learn R?

- **Free and Open Source**: R is completely free to use and modify, making it accessible to everyone
- **Comprehensive**: Extensive statistical and graphical capabilities with thousands of packages
- **Active Community**: Large, active community of users and developers contributing packages and support
- **Reproducible Research**: Excellent tools for reproducible statistical analysis and reporting
- **Industry Standard**: Widely used in academia, research, data science, and industry
- **Statistical Power**: Built specifically for statistical analysis with advanced modeling capabilities
- **Data Visualization**: Superior plotting and visualization capabilities
- **Machine Learning**: Extensive libraries for machine learning and predictive modeling

### R vs. Other Statistical Software

| Feature | R | SPSS | SAS | Python |
|---------|---|------|-----|--------|
| Cost | Free | Expensive | Expensive | Free |
| Learning Curve | Moderate | Easy | Steep | Moderate |
| Statistical Capabilities | Excellent | Good | Excellent | Good |
| Graphics | Superior | Basic | Good | Good |
| Programming | Full language | Limited | Full language | Full language |
| Community Support | Excellent | Limited | Good | Excellent |

## Installing R and RStudio

### Step 1: Install R
1. Go to [CRAN](https://cran.r-project.org/) (Comprehensive R Archive Network)
2. Download R for your operating system (Windows, macOS, or Linux)
3. Install following the installation wizard
4. Verify installation by opening R console

### Step 2: Install RStudio
1. Go to [RStudio](https://posit.co/download/rstudio-desktop/)
2. Download RStudio Desktop (free version)
3. Install RStudio
4. RStudio will automatically detect your R installation

### Understanding the RStudio Interface

RStudio provides an integrated development environment with four main panes:

1. **Source Editor** (top-left): Where you write and edit R scripts
2. **Console** (bottom-left): Where R commands are executed and output is displayed
3. **Environment/History** (top-right): Shows variables, data, and command history
4. **Files/Plots/Packages/Help** (bottom-right): File browser, plot viewer, package manager, and help documentation

## Getting Started with R

### Your First R Session

```r
# Print "Hello, World!" - the traditional first program
print("Hello, World!")

# Basic arithmetic operations
2 + 3    # Addition: 2 + 3 = 5
5 * 4    # Multiplication: 5 × 4 = 20
10 / 2   # Division: 10 ÷ 2 = 5
2^3      # Exponentiation: 2³ = 8

# Create simple variables and perform operations
x <- 5   # Assign value 5 to variable x
y <- 10  # Assign value 10 to variable y
x + y    # Add x and y: 5 + 10 = 15

# Check the class (data type) of an object
class(x)  # Returns "numeric"
```

### Understanding R as a Calculator

R can perform all basic mathematical operations and much more:

```r
# Basic arithmetic operations
2 + 3    # Addition: 2 + 3 = 5
5 - 2    # Subtraction: 5 - 2 = 3
4 * 3    # Multiplication: 4 × 3 = 12
10 / 2   # Division: 10 ÷ 2 = 5
2^3      # Exponentiation: 2³ = 8

# Mathematical functions
sqrt(16)  # Square root: √16 = 4
log(10)   # Natural logarithm: ln(10) ≈ 2.302585
log10(100) # Base-10 logarithm: log₁₀(100) = 2
exp(1)    # Exponential function: e¹ ≈ 2.718282
abs(-5)   # Absolute value: |-5| = 5
round(3.14159, 2) # Round to 2 decimal places: 3.14
ceiling(3.2)  # Ceiling function: ⌈3.2⌉ = 4
floor(3.8)    # Floor function: ⌊3.8⌋ = 3

# Trigonometric functions (angles in radians)
sin(pi/2)  # Sine of 90° = 1
cos(pi)    # Cosine of 180° = -1
tan(pi/4)  # Tangent of 45° = 1
```

### Variables and Assignment

Variables in R are containers that store values. Understanding variable assignment is fundamental:

```r
# Assignment using <- (preferred) or =
x <- 5    # Assign 5 to variable x
y = 10    # Alternative assignment syntax (less preferred)

# Print variables
x         # Display value of x
y         # Display value of y

# Check variable type and structure
class(x)      # Returns "numeric"
typeof(x)     # Returns "double" (more specific than class)
is.numeric(x) # Returns TRUE if x is numeric
is.character(x) # Returns FALSE since x is numeric

# Remove variables from memory
rm(x)     # Remove variable x
ls()      # List all variables in current environment
rm(list = ls()) # Remove all variables
```

## Data Types in R

Understanding data types is crucial for statistical analysis. R has several fundamental data types:

### Numeric
Numeric data represents numbers, including integers and decimal numbers:

```r
# Numeric data examples
age <- 25        # Integer
height <- 175.5  # Decimal number
temperature <- -5.2 # Negative number
pi_value <- 3.14159 # Mathematical constant

# Check data types
class(age)       # "numeric"
class(height)    # "numeric"
is.numeric(age)  # TRUE
is.integer(age)  # FALSE (R stores integers as numeric by default)

# Integer type (explicit)
age_int <- as.integer(25)
class(age_int)   # "integer"
```

### Character
Character data represents text (strings):

```r
# Character (string) data
name <- "John Doe"
city <- 'New York'
email <- "john.doe@email.com"

# Check character data
class(name)      # "character"
is.character(name) # TRUE
nchar(name)      # Length of string: 8

# String operations
toupper(name)    # Convert to uppercase: "JOHN DOE"
tolower(name)    # Convert to lowercase: "john doe"
paste("Hello", name) # Concatenate strings: "Hello John Doe"
```

### Logical
Logical data represents TRUE/FALSE values (boolean):

```r
# Logical (boolean) data
is_student <- TRUE
is_working <- FALSE
is_adult <- age >= 18  # Logical expression

# Logical operations
TRUE && TRUE    # AND: TRUE
TRUE || FALSE   # OR: TRUE
!TRUE           # NOT: FALSE

# Comparison operators
5 > 3           # Greater than: TRUE
5 == 5          # Equal to: TRUE
5 != 3          # Not equal to: TRUE
5 >= 5          # Greater than or equal: TRUE
```

### Factor
Factors represent categorical data with predefined levels:

```r
# Factor (categorical) data
gender <- factor(c("Male", "Female", "Male", "Female"))
education <- factor(c("High School", "Bachelor", "Master", "PhD"),
                   levels = c("High School", "Bachelor", "Master", "PhD"))

# Factor properties
levels(gender)      # Show levels: "Female" "Male"
nlevels(gender)     # Number of levels: 2
table(gender)       # Frequency table

# Ordered factors
satisfaction <- factor(c("Low", "Medium", "High", "Medium", "High"),
                      levels = c("Low", "Medium", "High"),
                      ordered = TRUE)
```

## Working with Vectors

Vectors are the fundamental data structure in R. They are one-dimensional arrays that can hold multiple values of the same type.

### Creating Vectors

```r
# Numeric vector
numbers <- c(1, 2, 3, 4, 5)
numbers

# Character vector
names <- c("Alice", "Bob", "Charlie", "Diana")
names

# Logical vector
logical_vec <- c(TRUE, FALSE, TRUE, FALSE, TRUE)
logical_vec

# Using sequence functions
1:10              # Sequence from 1 to 10
seq(1, 10, by=2)  # Sequence from 1 to 10, step 2: 1,3,5,7,9
rep(5, 3)         # Repeat 5 three times: 5,5,5
```

### Vector Operations

R performs operations element-wise on vectors:

```r
# Create vectors
x <- c(1, 2, 3, 4, 5)
y <- c(10, 20, 30, 40, 50)

# Element-wise operations
x + 2        # Add 2 to each element: 3,4,5,6,7
x * 3        # Multiply each element by 3: 3,6,9,12,15
x^2          # Square each element: 1,4,9,16,25
sqrt(x)      # Square root of each element: 1,1.414,1.732,2,2.236

# Vector arithmetic
x + y        # Element-wise addition: 11,22,33,44,55
x * y        # Element-wise multiplication: 10,40,90,160,250
x / y        # Element-wise division: 0.1,0.1,0.1,0.1,0.1

# Logical operations on vectors
x > 3        # Compare each element: FALSE,FALSE,FALSE,TRUE,TRUE
x == 3       # Check equality: FALSE,FALSE,TRUE,FALSE,FALSE
```

### Vector Functions and Statistics

```r
# Create a sample data vector
data <- c(1, 3, 5, 7, 9, 2, 4, 6, 8, 10)

# Basic summary statistics
length(data)     # Number of elements: 10
sum(data)        # Sum of all elements: 55
mean(data)       # Arithmetic mean: 5.5
median(data)     # Median (middle value): 5.5
min(data)        # Minimum value: 1
max(data)        # Maximum value: 10
range(data)      # Range (min, max): 1 10

# Measures of variability
var(data)        # Variance: 9.166667
sd(data)         # Standard deviation: 3.027650
sqrt(var(data))  # Standard deviation (alternative): 3.027650

# Quantiles and percentiles
quantile(data)   # Quartiles: 0% 25% 50% 75% 100%
quantile(data, 0.9) # 90th percentile
```

### Vector Indexing and Subsetting

```r
# Create a vector
scores <- c(85, 92, 78, 96, 88, 91, 87, 94, 89, 93)

# Indexing (R uses 1-based indexing)
scores[1]        # First element: 85
scores[5]        # Fifth element: 88
scores[c(1,3,5)] # Multiple elements: 85, 78, 88

# Logical indexing
scores > 90      # Logical vector: FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE
scores[scores > 90] # Elements greater than 90: 92, 96, 91, 94, 93

# Negative indexing (exclude elements)
scores[-1]       # All elements except first
scores[-c(1,3,5)] # All elements except 1st, 3rd, and 5th
```

## Getting Help in R

R has excellent built-in help system and extensive online resources:

### Built-in Help
```r
# Get help for a function
?mean
help(mean)

# Search for functions
??regression
help.search("regression")

# Get information about objects
str(data)        # Structure of object
summary(data)    # Summary statistics
head(data)       # First few elements
tail(data)       # Last few elements

# Package information
library(help = "base")  # Help for base package
```

### Online Resources
- **R Documentation**: [r-project.org](https://www.r-project.org/)
- **RStudio Documentation**: [posit.co](https://posit.co/)
- **Stack Overflow**: [stackoverflow.com/questions/tagged/r](https://stackoverflow.com/questions/tagged/r)
- **R-bloggers**: [r-bloggers.com](https://www.r-bloggers.com/)
- **CRAN Task Views**: [cran.r-project.org/web/views/](https://cran.r-project.org/web/views/)
- **R Documentation**: [rdocumentation.org](https://www.rdocumentation.org/)

## Best Practices

### Code Style and Organization
```r
# Good: Use descriptive variable names
student_scores <- c(85, 92, 78, 96, 88)
class_average <- mean(student_scores)

# Good: Use spaces around operators for readability
x <- 5 + 3
y <- x * 2

# Good: Use comments to explain complex code
# Calculate the mean and standard deviation of student scores
mean_score <- mean(student_scores)
sd_score <- sd(student_scores)

# Good: Use consistent indentation for control structures
if (mean_score > 85) {
  print("High performing class")
  print(paste("Average score:", round(mean_score, 2)))
} else {
  print("Needs improvement")
  print(paste("Current average:", round(mean_score, 2)))
}

# Good: Use functions to organize code
calculate_stats <- function(data) {
  mean_val <- mean(data)
  sd_val <- sd(data)
  return(list(mean = mean_val, sd = sd_val))
}
```

### File Organization and Project Management
- **Organize scripts**: Keep related R scripts in organized folders
- **Meaningful names**: Use descriptive file names (e.g., `data_cleaning.R`, `analysis.R`)
- **Documentation**: Include comments and documentation in your code
- **Version control**: Use Git for tracking changes
- **Regular saves**: Save your work frequently
- **Project structure**: Organize projects with clear folder structure

### Error Handling and Debugging
```r
# Check for errors and handle them gracefully
tryCatch({
  result <- mean(non_existent_variable)
}, error = function(e) {
  print(paste("Error occurred:", e$message))
})

# Use print statements for debugging
x <- 5
print(paste("x =", x))
print(paste("x squared =", x^2))
```

## Mathematical Concepts in R

### Understanding Statistical Measures

**Mean (Arithmetic Average)**: The sum of all values divided by the number of values
```r
# Mathematical formula: μ = (Σxᵢ) / n
data <- c(2, 4, 6, 8, 10)
mean_value <- sum(data) / length(data)  # Manual calculation
mean_value_auto <- mean(data)           # R function
```

**Median**: The middle value when data is ordered
```r
# For odd number of values: middle value
# For even number of values: average of two middle values
data <- c(1, 3, 5, 7, 9)
median_value <- median(data)
```

**Variance**: Average squared deviation from the mean
```r
# Mathematical formula: σ² = Σ(xᵢ - μ)² / (n-1)
data <- c(2, 4, 6, 8, 10)
mean_data <- mean(data)
variance_manual <- sum((data - mean_data)^2) / (length(data) - 1)
variance_auto <- var(data)
```

**Standard Deviation**: Square root of variance
```r
# Mathematical formula: σ = √σ²
sd_manual <- sqrt(var(data))
sd_auto <- sd(data)
```

## Exercises

### Exercise 1: Basic Operations and Variables
Create variables for your age, height (in cm), and favorite color. Then perform some basic operations with the numeric variables.

```r
# Your solution here
age <- 25
height <- 175
favorite_color <- "blue"

# Basic operations
age_in_months <- age * 12
height_in_meters <- height / 100
bmi <- 70 / (height_in_meters^2)  # Assuming weight of 70kg
```

### Exercise 2: Vector Creation and Manipulation
Create vectors for:
- Your last 5 test scores
- Names of 3 friends
- Whether you like different foods (TRUE/FALSE)

```r
# Your solution here
test_scores <- c(85, 92, 78, 96, 88)
friend_names <- c("Alice", "Bob", "Charlie")
food_preferences <- c(TRUE, FALSE, TRUE, TRUE, FALSE)
```

### Exercise 3: Summary Statistics
Calculate the mean, median, and standard deviation of your test scores. Interpret what these values tell you about your performance.

```r
# Your solution here
mean_score <- mean(test_scores)
median_score <- median(test_scores)
sd_score <- sd(test_scores)

# Interpretation
cat("Mean score:", mean_score, "\n")
cat("Median score:", median_score, "\n")
cat("Standard deviation:", sd_score, "\n")
```

### Exercise 4: Data Types and Type Conversion
Create different types of data and practice converting between types.

```r
# Your solution here
numeric_var <- 42
character_var <- "42"
logical_var <- TRUE

# Type conversions
as.character(numeric_var)
as.numeric(character_var)
as.logical(1)  # 1 becomes TRUE
as.logical(0)  # 0 becomes FALSE
```

### Exercise 5: Vector Operations
Create two numeric vectors and perform various operations on them.

```r
# Your solution here
vector1 <- c(1, 2, 3, 4, 5)
vector2 <- c(10, 20, 30, 40, 50)

# Operations
sum_vectors <- vector1 + vector2
product_vectors <- vector1 * vector2
mean_vector1 <- mean(vector1)
sd_vector2 <- sd(vector2)
```

## Next Steps

In the next chapter, we'll explore more complex data structures like matrices, data frames, and lists, which are essential for statistical analysis in R. We'll also learn about:

- **Matrices**: Two-dimensional arrays for mathematical operations
- **Data Frames**: Tabular data structures (like spreadsheets)
- **Lists**: Flexible containers for different types of data
- **Importing Data**: Reading data from various file formats
- **Data Manipulation**: Cleaning and transforming data

---

**Key Takeaways:**
- R is a powerful language specifically designed for statistical computing
- Use `<-` for assignment (preferred over `=`)
- Vectors are fundamental data structures in R
- Always use descriptive variable names and add comments
- Get help using `?` and `help()` functions
- Practice regularly to build proficiency
- Understanding data types is crucial for proper analysis
- R performs element-wise operations on vectors automatically
- The built-in help system is comprehensive and useful 