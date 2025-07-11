# Data Types and Structures

## Overview

R provides several data structures to handle different types of data efficiently. Understanding these structures is crucial for effective data analysis and statistical computing.

## Matrices

A matrix is a two-dimensional array with rows and columns. All elements must be of the same data type.

### Creating Matrices

```r
# Create matrix from vector
numbers <- 1:12
matrix(numbers, nrow = 3, ncol = 4)

# Create matrix with specific dimensions
matrix(1:12, nrow = 3, ncol = 4, byrow = TRUE)

# Create matrix with row and column names
mat <- matrix(1:9, nrow = 3, ncol = 3)
rownames(mat) <- c("Row1", "Row2", "Row3")
colnames(mat) <- c("Col1", "Col2", "Col3")
mat
```

### Matrix Operations

```r
# Create two matrices
A <- matrix(1:4, nrow = 2, ncol = 2)
B <- matrix(5:8, nrow = 2, ncol = 2)

# Element-wise operations
A + B    # Addition
A - B    # Subtraction
A * B    # Element-wise multiplication
A / B    # Element-wise division

# Matrix multiplication
A %*% B

# Transpose
t(A)

# Matrix properties
dim(A)        # Dimensions
nrow(A)       # Number of rows
ncol(A)       # Number of columns
```

### Matrix Functions

```r
# Create a matrix
mat <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)

# Basic operations
rowSums(mat)      # Sum of each row
colSums(mat)      # Sum of each column
rowMeans(mat)     # Mean of each row
colMeans(mat)     # Mean of each column

# Diagonal operations
diag(mat)         # Extract diagonal
diag(3)           # Create identity matrix
```

## Data Frames

Data frames are the most commonly used data structure in R for statistical analysis. They are like tables with rows and columns, where each column can have a different data type.

### Creating Data Frames

```r
# Create data frame from vectors
name <- c("Alice", "Bob", "Charlie", "Diana")
age <- c(25, 30, 35, 28)
height <- c(165, 180, 175, 170)
is_student <- c(TRUE, FALSE, FALSE, TRUE)

# Create data frame
students <- data.frame(name, age, height, is_student)
students

# Create data frame directly
students <- data.frame(
  name = c("Alice", "Bob", "Charlie", "Diana"),
  age = c(25, 30, 35, 28),
  height = c(165, 180, 175, 170),
  is_student = c(TRUE, FALSE, FALSE, TRUE)
)
```

### Working with Data Frames

```r
# View data frame structure
str(students)
head(students)
tail(students)
dim(students)
names(students)
colnames(students)
rownames(students)

# Access columns
students$name
students[["age"]]
students[, "height"]

# Access rows
students[1, ]           # First row
students[1:3, ]         # First three rows
students[c(1, 3), ]     # Rows 1 and 3

# Access specific elements
students[1, 2]          # Row 1, Column 2
students[2, "age"]      # Row 2, Column "age"
```

### Modifying Data Frames

```r
# Add new column
students$bmi <- students$height / 100 * 2

# Remove column
students$bmi <- NULL

# Change column names
names(students)[1] <- "student_name"

# Add new row
new_student <- data.frame(
  name = "Eve",
  age = 27,
  height = 168,
  is_student = TRUE
)
students <- rbind(students, new_student)
```

## Lists

Lists are flexible data structures that can contain elements of different types and sizes.

### Creating Lists

```r
# Create a simple list
my_list <- list(
  name = "John",
  age = 30,
  scores = c(85, 92, 78),
  is_student = TRUE
)

# Access list elements
my_list$name
my_list[["age"]]
my_list[[3]]

# Nested lists
nested_list <- list(
  person = list(name = "Alice", age = 25),
  scores = c(90, 85, 88),
  courses = list("Statistics", "Programming", "Data Analysis")
)
```

### Working with Lists

```r
# List properties
length(my_list)
names(my_list)
str(my_list)

# Add elements
my_list$city <- "New York"

# Remove elements
my_list$city <- NULL

# Convert list to data frame (if possible)
as.data.frame(my_list)
```

## Arrays

Arrays are multi-dimensional generalizations of matrices.

```r
# Create 3D array
array_data <- array(1:24, dim = c(2, 3, 4))
array_data

# Access elements
array_data[1, 2, 3]    # Element at position (1, 2, 3)
array_data[1, , ]       # All elements in first row
array_data[, 2, ]       # All elements in second column
```

## Factors

Factors are used to represent categorical variables with predefined levels.

```r
# Create factor
gender <- factor(c("Male", "Female", "Male", "Female"))
gender

# Check levels
levels(gender)

# Create factor with specific levels
education <- factor(c("High School", "Bachelor", "Master", "PhD"),
                   levels = c("High School", "Bachelor", "Master", "PhD"))
education

# Ordered factors
satisfaction <- factor(c("Low", "Medium", "High", "Medium", "High"),
                      levels = c("Low", "Medium", "High"),
                      ordered = TRUE)
satisfaction
```

## Data Type Conversion

```r
# Numeric to character
x <- 123
as.character(x)

# Character to numeric
y <- "456"
as.numeric(y)

# Vector to factor
z <- c("A", "B", "A", "C")
as.factor(z)

# Data frame to matrix
df <- data.frame(x = 1:3, y = 4:6)
as.matrix(df)

# Matrix to data frame
mat <- matrix(1:6, nrow = 2)
as.data.frame(mat)
```

## Checking Data Types

```r
# Check data types
x <- 5
y <- "hello"
z <- TRUE

class(x)
class(y)
class(z)

is.numeric(x)
is.character(y)
is.logical(z)
is.factor(z)

# Check data structure
is.vector(x)
is.matrix(x)
is.data.frame(x)
is.list(x)
```

## Working with Missing Data

```r
# Create data with missing values
data_with_na <- c(1, 2, NA, 4, 5)

# Check for missing values
is.na(data_with_na)
any(is.na(data_with_na))
sum(is.na(data_with_na))

# Remove missing values
data_without_na <- na.omit(data_with_na)
data_without_na

# Replace missing values
data_filled <- data_with_na
data_filled[is.na(data_filled)] <- 0
data_filled
```

## Practical Examples

### Example 1: Student Grades

```r
# Create student data
students <- data.frame(
  name = c("Alice", "Bob", "Charlie", "Diana"),
  math = c(85, 92, 78, 96),
  science = c(88, 85, 92, 89),
  english = c(90, 87, 85, 92)
)

# Calculate average grade for each student
students$average <- rowMeans(students[, 2:4])
students

# Find students with average > 85
high_performers <- students[students$average > 85, ]
high_performers
```

### Example 2: Survey Data

```r
# Create survey data
survey <- data.frame(
  respondent_id = 1:10,
  age = c(25, 30, 35, 28, 42, 33, 29, 31, 27, 38),
  gender = factor(c("Female", "Male", "Male", "Female", "Female", 
                   "Male", "Female", "Male", "Female", "Male")),
  satisfaction = factor(c("High", "Medium", "Low", "High", "Medium",
                         "High", "Medium", "Low", "High", "Medium"),
                       levels = c("Low", "Medium", "High"),
                       ordered = TRUE)
)

# Summary statistics
summary(survey)
```

## Exercises

### Exercise 1: Matrix Operations
Create two 3x3 matrices and perform various operations (addition, multiplication, transpose).

### Exercise 2: Data Frame Creation
Create a data frame with information about 5 movies including title, year, rating, and genre.

### Exercise 3: List Manipulation
Create a list containing different types of data and practice accessing and modifying elements.

### Exercise 4: Factor Analysis
Create a factor variable for education levels and analyze the distribution of responses.

### Exercise 5: Missing Data
Create a dataset with missing values and practice different methods of handling them.

## Next Steps

In the next chapter, we'll learn how to import data from various sources and manipulate it for analysis.

---

**Key Takeaways:**
- Data frames are the most important structure for statistical analysis
- Matrices are useful for mathematical operations
- Lists provide flexibility for complex data structures
- Factors are essential for categorical variables
- Always check data types and structures before analysis
- Handle missing data appropriately 