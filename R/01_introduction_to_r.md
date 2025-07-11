# Introduction to R

## What is R?

R is a powerful programming language and environment for statistical computing and graphics. It was developed by Ross Ihaka and Robert Gentleman at the University of Auckland, New Zealand, and is now maintained by the R Development Core Team.

### Why Learn R?

- **Free and Open Source**: R is completely free to use and modify
- **Comprehensive**: Extensive statistical and graphical capabilities
- **Active Community**: Large, active community of users and developers
- **Reproducible Research**: Excellent tools for reproducible statistical analysis
- **Industry Standard**: Widely used in academia, research, and industry

## Installing R and RStudio

### Step 1: Install R
1. Go to [CRAN](https://cran.r-project.org/)
2. Download R for your operating system
3. Install following the installation wizard

### Step 2: Install RStudio
1. Go to [RStudio](https://posit.co/download/rstudio-desktop/)
2. Download RStudio Desktop (free version)
3. Install RStudio

## Getting Started with R

### Your First R Session

```r
# Print "Hello, World!"
print("Hello, World!")

# Basic arithmetic
2 + 3
5 * 4
10 / 2
2^3

# Create a simple variable
x <- 5
y <- 10
x + y

# Check the class of an object
class(x)
```

### R as a Calculator

```r
# Basic arithmetic operations
2 + 3    # Addition
5 - 2    # Subtraction
4 * 3    # Multiplication
10 / 2   # Division
2^3      # Exponentiation
sqrt(16) # Square root
log(10)  # Natural logarithm
exp(1)   # Exponential function
```

### Variables and Assignment

```r
# Assignment using <- (preferred) or =
x <- 5
y = 10

# Print variables
x
y

# Check variable type
class(x)
typeof(x)

# Remove variables
rm(x)
```

## Data Types in R

### Numeric
```r
# Numeric data
age <- 25
height <- 175.5
class(age)
class(height)
```

### Character
```r
# Character (string) data
name <- "John Doe"
city <- 'New York'
class(name)
```

### Logical
```r
# Logical (boolean) data
is_student <- TRUE
is_working <- FALSE
class(is_student)
```

### Factor
```r
# Factor (categorical) data
gender <- factor(c("Male", "Female", "Male", "Female"))
levels(gender)
```

## Working with Vectors

### Creating Vectors
```r
# Numeric vector
numbers <- c(1, 2, 3, 4, 5)
numbers

# Character vector
names <- c("Alice", "Bob", "Charlie")
names

# Logical vector
logical_vec <- c(TRUE, FALSE, TRUE, FALSE)
logical_vec
```

### Vector Operations
```r
# Create a vector
x <- c(1, 2, 3, 4, 5)

# Basic operations
x + 2        # Add 2 to each element
x * 3        # Multiply each element by 3
x^2          # Square each element
sqrt(x)      # Square root of each element

# Vector arithmetic
y <- c(10, 20, 30, 40, 50)
x + y        # Element-wise addition
x * y        # Element-wise multiplication
```

### Vector Functions
```r
# Create a vector
data <- c(1, 3, 5, 7, 9, 2, 4, 6, 8, 10)

# Summary statistics
length(data)     # Number of elements
sum(data)        # Sum of all elements
mean(data)       # Mean
median(data)     # Median
min(data)        # Minimum value
max(data)        # Maximum value
range(data)      # Range (min, max)
var(data)        # Variance
sd(data)         # Standard deviation
```

## Getting Help in R

### Built-in Help
```r
# Get help for a function
?mean
help(mean)

# Search for functions
??regression
help.search("regression")

# Get information about objects
str(data)
summary(data)
```

### Online Resources
- **R Documentation**: [r-project.org](https://www.r-project.org/)
- **RStudio Documentation**: [posit.co](https://posit.co/)
- **Stack Overflow**: [stackoverflow.com/questions/tagged/r](https://stackoverflow.com/questions/tagged/r)
- **R-bloggers**: [r-bloggers.com](https://www.r-bloggers.com/)

## Best Practices

### Code Style
```r
# Good: Use descriptive variable names
student_scores <- c(85, 92, 78, 96, 88)

# Good: Use spaces around operators
x <- 5 + 3

# Good: Use comments to explain code
# Calculate the mean of student scores
mean_score <- mean(student_scores)

# Good: Use consistent indentation
if (mean_score > 85) {
  print("High performing class")
} else {
  print("Needs improvement")
}
```

### File Organization
- Keep your R scripts organized in folders
- Use meaningful file names
- Include comments and documentation
- Save your work regularly

## Exercises

### Exercise 1: Basic Operations
Create variables for your age, height, and favorite color. Then perform some basic operations with the numeric variables.

### Exercise 2: Vector Creation
Create vectors for:
- Your last 5 test scores
- Names of 3 friends
- Whether you like different foods (TRUE/FALSE)

### Exercise 3: Summary Statistics
Calculate the mean, median, and standard deviation of your test scores.

### Exercise 4: Data Types
Create different types of data (numeric, character, logical, factor) and use the `class()` function to verify their types.

## Next Steps

In the next chapter, we'll explore more complex data structures like matrices, data frames, and lists, which are essential for statistical analysis in R.

---

**Key Takeaways:**
- R is a powerful language for statistical computing
- Use `<-` for assignment (preferred over `=`)
- Vectors are fundamental data structures in R
- Always use descriptive variable names
- Get help using `?` and `help()`
- Practice regularly to build proficiency 