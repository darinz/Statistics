# Data Types and Structures

## Overview

R provides several data structures to handle different types of data efficiently. Understanding these structures is crucial for effective data analysis and statistical computing. Each structure has specific characteristics that make it suitable for particular types of analysis.

### Why Different Data Structures Matter

- **Matrices**: Efficient for mathematical operations and linear algebra
- **Data Frames**: Ideal for statistical analysis and data manipulation
- **Lists**: Flexible containers for complex, heterogeneous data
- **Arrays**: Multi-dimensional data storage
- **Factors**: Essential for categorical variables in statistical models

## Matrices

A matrix is a two-dimensional array with rows and columns. All elements must be of the same data type (typically numeric). Matrices are fundamental for mathematical operations, linear algebra, and statistical computations.

### Mathematical Foundation

A matrix A with m rows and n columns is denoted as:
```
A = [aᵢⱼ] where i = 1,2,...,m and j = 1,2,...,n
```

### Creating Matrices

```r
# Create matrix from vector (fills by column by default)
numbers <- 1:12
matrix(numbers, nrow = 3, ncol = 4)
# Result:
#      [,1] [,2] [,3] [,4]
# [1,]    1    4    7   10
# [2,]    2    5    8   11
# [3,]    3    6    9   12

# Create matrix with specific dimensions (fill by row)
matrix(1:12, nrow = 3, ncol = 4, byrow = TRUE)
# Result:
#      [,1] [,2] [,3] [,4]
# [1,]    1    2    3    4
# [2,]    5    6    7    8
# [3,]    9   10   11   12

# Create matrix with row and column names
mat <- matrix(1:9, nrow = 3, ncol = 3)
rownames(mat) <- c("Row1", "Row2", "Row3")
colnames(mat) <- c("Col1", "Col2", "Col3")
mat
# Result:
#      Col1 Col2 Col3
# Row1    1    4    7
# Row2    2    5    8
# Row3    3    6    9

# Create identity matrix (diagonal matrix with 1s)
identity_matrix <- diag(3)
identity_matrix
# Result:
#      [,1] [,2] [,3]
# [1,]    1    0    0
# [2,]    0    1    0
# [3,]    0    0    1
```

### Matrix Operations

```r
# Create two matrices
A <- matrix(1:4, nrow = 2, ncol = 2)
B <- matrix(5:8, nrow = 2, ncol = 2)

# Display matrices
A
#      [,1] [,2]
# [1,]    1    3
# [2,]    2    4

B
#      [,1] [,2]
# [1,]    5    7
# [2,]    6    8

# Element-wise operations (Hadamard product)
A + B    # Addition: [1+5, 3+7; 2+6, 4+8] = [6, 10; 8, 12]
A - B    # Subtraction: [1-5, 3-7; 2-6, 4-8] = [-4, -4; -4, -4]
A * B    # Element-wise multiplication: [1*5, 3*7; 2*6, 4*8] = [5, 21; 12, 32]
A / B    # Element-wise division: [1/5, 3/7; 2/6, 4/8] = [0.2, 0.429; 0.333, 0.5]

# Matrix multiplication (dot product)
A %*% B
# Mathematical formula: Cᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ
# Result: [1*5+3*6, 1*7+3*8; 2*5+4*6, 2*7+4*8] = [23, 31; 34, 46]

# Transpose (swap rows and columns)
t(A)
# Result:
#      [,1] [,2]
# [1,]    1    2
# [2,]    3    4

# Matrix properties
dim(A)        # Dimensions: 2 2
nrow(A)       # Number of rows: 2
ncol(A)       # Number of columns: 2
```

### Matrix Functions and Statistical Operations

```r
# Create a sample matrix
mat <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, ncol = 3)
mat
#      [,1] [,2] [,3]
# [1,]    1    3    5
# [2,]    2    4    6

# Row and column operations
rowSums(mat)      # Sum of each row: [1+3+5, 2+4+6] = [9, 12]
colSums(mat)      # Sum of each column: [1+2, 3+4, 5+6] = [3, 7, 11]
rowMeans(mat)     # Mean of each row: [9/3, 12/3] = [3, 4]
colMeans(mat)     # Mean of each column: [3/2, 7/2, 11/2] = [1.5, 3.5, 5.5]

# Diagonal operations
diag(mat)         # Extract diagonal: [1, 4] (only for square matrices)
diag(3)           # Create 3x3 identity matrix
diag(c(1, 2, 3)) # Create diagonal matrix with values [1, 2, 3]

# Matrix determinant (for square matrices)
det(matrix(1:4, nrow = 2))  # Determinant of 2x2 matrix

# Matrix inverse (for square, non-singular matrices)
solve(matrix(1:4, nrow = 2))  # Inverse matrix
```

### Matrix Indexing and Subsetting

```r
# Create a matrix
mat <- matrix(1:9, nrow = 3, ncol = 3)
mat
#      [,1] [,2] [,3]
# [1,]    1    4    7
# [2,]    2    5    8
# [3,]    3    6    9

# Access elements
mat[1, 2]        # Element at row 1, column 2: 4
mat[2, ]         # Entire second row: [2, 5, 8]
mat[, 3]         # Entire third column: [7, 8, 9]
mat[1:2, 2:3]    # Submatrix: rows 1-2, columns 2-3

# Logical indexing
mat > 5          # Logical matrix: TRUE where elements > 5
mat[mat > 5]     # Extract elements > 5: [6, 7, 8, 9]
```

## Data Frames

Data frames are the most commonly used data structure in R for statistical analysis. They are like tables with rows and columns, where each column can have a different data type. Data frames are essentially lists of vectors of equal length.

### Structure of Data Frames

A data frame is a rectangular array where:
- Each column represents a variable
- Each row represents an observation
- Different columns can have different data types
- All columns must have the same length

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
# Result:
#      name age height is_student
# 1   Alice  25    165       TRUE
# 2     Bob  30    180      FALSE
# 3 Charlie  35    175      FALSE
# 4   Diana  28    170       TRUE

# Create data frame directly
students <- data.frame(
  name = c("Alice", "Bob", "Charlie", "Diana"),
  age = c(25, 30, 35, 28),
  height = c(165, 180, 175, 170),
  is_student = c(TRUE, FALSE, FALSE, TRUE),
  stringsAsFactors = FALSE  # Keep character columns as character
)

# Create data frame with row names
students <- data.frame(
  age = c(25, 30, 35, 28),
  height = c(165, 180, 175, 170),
  row.names = c("Alice", "Bob", "Charlie", "Diana")
)
```

### Working with Data Frames

```r
# View data frame structure and information
str(students)        # Structure of data frame
head(students)       # First 6 rows
tail(students)       # Last 6 rows
dim(students)        # Dimensions (rows, columns)
names(students)      # Column names
colnames(students)   # Column names (alternative)
rownames(students)   # Row names
nrow(students)       # Number of rows
ncol(students)       # Number of columns

# Access columns (multiple ways)
students$name        # Using $ operator
students[["age"]]    # Using double brackets
students[, "height"] # Using matrix notation
students[, 2]        # Using column index

# Access rows
students[1, ]        # First row
students[1:3, ]      # First three rows
students[c(1, 3), ]  # Rows 1 and 3
students[-2, ]       # All rows except row 2

# Access specific elements
students[1, 2]       # Row 1, Column 2
students[2, "age"]   # Row 2, Column "age"
students$name[3]     # Third element of name column
```

### Modifying Data Frames

```r
# Add new column
students$bmi <- students$height / 100 * 2  # Incorrect BMI calculation
# Correct BMI calculation: weight(kg) / height(m)²
students$bmi <- 70 / (students$height / 100)^2  # Assuming 70kg weight

# Remove column
students$bmi <- NULL

# Change column names
names(students)[1] <- "student_name"
colnames(students)[2] <- "student_age"

# Add new row
new_student <- data.frame(
  name = "Eve",
  age = 27,
  height = 168,
  is_student = TRUE
)
students <- rbind(students, new_student)

# Combine data frames (column-wise)
additional_info <- data.frame(
  gpa = c(3.8, 3.5, 3.9, 3.7, 3.6),
  major = c("Math", "Physics", "Chemistry", "Biology", "Computer Science")
)
students <- cbind(students, additional_info)
```

### Data Frame Operations and Analysis

```r
# Summary statistics
summary(students)    # Summary of all columns
summary(students$age) # Summary of specific column

# Subsetting based on conditions
young_students <- students[students$age < 30, ]
tall_students <- students[students$height > 175, ]
student_only <- students[students$is_student == TRUE, ]

# Multiple conditions
young_and_tall <- students[students$age < 30 & students$height > 170, ]

# Sorting
students_sorted_age <- students[order(students$age), ]
students_sorted_height_desc <- students[order(students$height, decreasing = TRUE), ]

# Aggregation
mean_age <- mean(students$age)
median_height <- median(students$height)
sd_age <- sd(students$age)
```

## Lists

Lists are flexible data structures that can contain elements of different types and sizes. They are essential for storing complex, heterogeneous data and are the foundation for many advanced R objects.

### Understanding Lists

Lists are recursive data structures that can contain:
- Vectors of different types
- Other lists (nested lists)
- Data frames
- Matrices
- Functions
- Any combination of the above

### Creating Lists

```r
# Create a simple list
my_list <- list(
  name = "John",
  age = 30,
  scores = c(85, 92, 78),
  is_student = TRUE
)

# Display list
my_list
# Result:
# $name
# [1] "John"
# $age
# [1] 30
# $scores
# [1] 85 92 78
# $is_student
# [1] TRUE

# Access list elements
my_list$name        # Using $ operator
my_list[["age"]]    # Using double brackets
my_list[[3]]        # Using index (scores)
my_list[["scores"]] # Using name with double brackets

# Nested lists
nested_list <- list(
  person = list(name = "Alice", age = 25),
  scores = c(90, 85, 88),
  courses = list("Statistics", "Programming", "Data Analysis"),
  contact = list(
    email = "alice@email.com",
    phone = "555-1234"
  )
)

# Access nested elements
nested_list$person$name
nested_list[["person"]][["age"]]
nested_list$contact$email
```

### Working with Lists

```r
# List properties
length(my_list)     # Number of elements: 4
names(my_list)      # Names of elements: "name" "age" "scores" "is_student"
str(my_list)        # Structure of list

# Add elements
my_list$city <- "New York"
my_list$hobbies <- c("reading", "swimming", "coding")

# Remove elements
my_list$city <- NULL

# Convert list to data frame (if possible)
# Only works if all elements are vectors of the same length
list_for_df <- list(
  name = c("Alice", "Bob", "Charlie"),
  age = c(25, 30, 35),
  city = c("NYC", "LA", "Chicago")
)
as.data.frame(list_for_df)

# List operations
lapply(my_list, class)  # Apply function to each element
sapply(my_list, length) # Simplified apply (returns vector)
```

### Advanced List Operations

```r
# Create a list of functions
math_functions <- list(
  mean = function(x) mean(x),
  median = function(x) median(x),
  sd = function(x) sd(x)
)

# Apply functions from list
data <- c(1, 2, 3, 4, 5)
results <- lapply(math_functions, function(f) f(data))

# List of data frames
df_list <- list(
  students = data.frame(name = c("Alice", "Bob"), age = c(25, 30)),
  teachers = data.frame(name = c("Dr. Smith", "Dr. Jones"), subject = c("Math", "Physics"))
)

# Access data frames in list
df_list$students
df_list[["teachers"]]
```

## Arrays

Arrays are multi-dimensional generalizations of matrices. They can have more than two dimensions and are useful for storing complex data structures.

### Understanding Arrays

An n-dimensional array has n indices. For example:
- 1D array: vector
- 2D array: matrix
- 3D array: cube of data
- 4D+ array: hypercube

### Creating and Working with Arrays

```r
# Create 3D array (2 rows × 3 columns × 4 layers)
array_data <- array(1:24, dim = c(2, 3, 4))
array_data
# Result: 3D array with 24 elements arranged in 2×3×4 structure

# Access elements
array_data[1, 2, 3]    # Element at position (1, 2, 3)
array_data[1, , ]       # All elements in first row (2D slice)
array_data[, 2, ]       # All elements in second column (2D slice)
array_data[, , 1]       # First layer (2D matrix)

# Array properties
dim(array_data)         # Dimensions: 2 3 4
length(array_data)      # Total elements: 24
dimnames(array_data)    # Dimension names (NULL if not set)

# Create array with dimension names
dimnames_array <- array(1:12, dim = c(2, 3, 2),
                       dimnames = list(
                         c("Row1", "Row2"),
                         c("Col1", "Col2", "Col3"),
                         c("Layer1", "Layer2")
                       ))
```

### Array Operations

```r
# Create sample arrays
arr1 <- array(1:8, dim = c(2, 2, 2))
arr2 <- array(9:16, dim = c(2, 2, 2))

# Element-wise operations
arr1 + arr2    # Addition
arr1 * arr2    # Element-wise multiplication

# Apply functions across dimensions
apply(arr1, 1, sum)    # Sum across first dimension
apply(arr1, 2, mean)   # Mean across second dimension
apply(arr1, 3, max)    # Maximum across third dimension
```

## Factors

Factors are used to represent categorical variables with predefined levels. They are essential for statistical analysis and modeling.

### Understanding Factors

Factors have two important properties:
1. **Levels**: The set of possible values
2. **Order**: Whether the levels have a natural order

### Creating and Working with Factors

```r
# Create factor
gender <- factor(c("Male", "Female", "Male", "Female"))
gender
# Result: [1] Male   Female Male   Female
# Levels: Female Male

# Check levels
levels(gender)      # Show levels: "Female" "Male"
nlevels(gender)     # Number of levels: 2

# Create factor with specific levels
education <- factor(c("High School", "Bachelor", "Master", "PhD"),
                   levels = c("High School", "Bachelor", "Master", "PhD"))
education
# Result: [1] High School Bachelor Master     PhD
# Levels: High School Bachelor Master PhD

# Ordered factors (for ordinal categorical variables)
satisfaction <- factor(c("Low", "Medium", "High", "Medium", "High"),
                      levels = c("Low", "Medium", "High"),
                      ordered = TRUE)
satisfaction
# Result: [1] Low    Medium High   Medium High
# Levels: Low < Medium < High

# Factor operations
table(gender)        # Frequency table
summary(education)   # Summary of factor
```

### Factor Analysis and Manipulation

```r
# Create factor with missing levels
survey_data <- factor(c("Yes", "No", "Yes", "Maybe", "No"),
                     levels = c("Yes", "No", "Maybe"))
survey_data

# Reorder levels
education_reordered <- factor(education, 
                            levels = c("PhD", "Master", "Bachelor", "High School"))

# Add new levels
education_extended <- factor(education, 
                           levels = c("High School", "Bachelor", "Master", "PhD", "PostDoc"))

# Convert between factor and character
as.character(education)
as.factor(c("A", "B", "A", "C"))

# Factor in data frames
df_with_factor <- data.frame(
  name = c("Alice", "Bob", "Charlie"),
  department = factor(c("IT", "HR", "IT"),
                     levels = c("IT", "HR", "Finance", "Marketing"))
)
```

## Data Type Conversion

Understanding how to convert between data types is crucial for data manipulation and analysis.

### Conversion Functions

```r
# Numeric to character
x <- 123
as.character(x)     # "123"
as.character(c(1, 2, 3))  # "1" "2" "3"

# Character to numeric
y <- "456"
as.numeric(y)       # 456
as.numeric(c("1", "2", "3"))  # 1 2 3

# Vector to factor
z <- c("A", "B", "A", "C")
as.factor(z)        # Factor with levels A B C

# Data frame to matrix
df <- data.frame(x = 1:3, y = 4:6)
as.matrix(df)       # Matrix with same data

# Matrix to data frame
mat <- matrix(1:6, nrow = 2)
as.data.frame(mat)  # Data frame with columns V1, V2, V3

# List to vector (if possible)
list_to_vector <- list(a = 1, b = 2, c = 3)
unlist(list_to_vector)  # 1 2 3

# Vector to list
vector_to_list <- as.list(c(1, 2, 3))
```

### Type Checking and Validation

```r
# Check data types
x <- 5
y <- "hello"
z <- TRUE
f <- factor(c("A", "B"))

class(x)        # "numeric"
class(y)        # "character"
class(z)        # "logical"
class(f)        # "factor"

# Type checking functions
is.numeric(x)   # TRUE
is.character(y) # TRUE
is.logical(z)   # TRUE
is.factor(f)    # TRUE

# Check data structure
is.vector(x)    # TRUE
is.matrix(x)    # FALSE
is.data.frame(x) # FALSE
is.list(x)      # FALSE

# Check for specific types
is.integer(x)   # FALSE (R stores integers as numeric by default)
is.double(x)    # TRUE
```

## Working with Missing Data

Missing data is common in real-world datasets and must be handled appropriately.

### Understanding Missing Data

In R, missing data is represented by `NA` (Not Available). Different data types have different NA representations:
- `NA` for logical, numeric, and character data
- `NA_character_` for character data
- `NA_integer_` for integer data
- `NA_real_` for numeric data

### Creating and Detecting Missing Data

```r
# Create data with missing values
data_with_na <- c(1, 2, NA, 4, 5)
character_with_na <- c("a", "b", NA, "d")

# Check for missing values
is.na(data_with_na)        # FALSE FALSE TRUE FALSE FALSE
any(is.na(data_with_na))   # TRUE
sum(is.na(data_with_na))   # 1 (count of missing values)

# Check for complete cases
complete.cases(data_with_na)  # TRUE TRUE FALSE TRUE TRUE

# Remove missing values
data_without_na <- na.omit(data_with_na)
data_without_na              # 1 2 4 5

# Replace missing values
data_filled <- data_with_na
data_filled[is.na(data_filled)] <- 0
data_filled                  # 1 2 0 4 5

# Replace with mean (for numeric data)
data_mean_filled <- data_with_na
data_mean_filled[is.na(data_mean_filled)] <- mean(data_with_na, na.rm = TRUE)
```

### Missing Data in Data Frames

```r
# Create data frame with missing values
df_with_na <- data.frame(
  name = c("Alice", "Bob", "Charlie", "Diana"),
  age = c(25, NA, 35, 28),
  height = c(165, 180, NA, 170),
  score = c(85, 92, 78, NA)
)

# Check for missing values
colSums(is.na(df_with_na))  # Missing values per column
rowSums(is.na(df_with_na))  # Missing values per row

# Remove rows with any missing values
df_complete <- na.omit(df_with_na)

# Remove rows with missing values in specific columns
df_partial <- df_with_na[!is.na(df_with_na$age), ]

# Fill missing values
df_filled <- df_with_na
df_filled$age[is.na(df_filled$age)] <- mean(df_filled$age, na.rm = TRUE)
df_filled$height[is.na(df_filled$height)] <- median(df_filled$height, na.rm = TRUE)
```

## Practical Examples

### Example 1: Student Grades Analysis

```r
# Create comprehensive student data
students <- data.frame(
  name = c("Alice", "Bob", "Charlie", "Diana", "Eve"),
  math = c(85, 92, 78, 96, 88),
  science = c(88, 85, 92, 89, 91),
  english = c(90, 87, 85, 92, 89),
  attendance = c(95, 88, 92, 96, 90)
)

# Calculate average grade for each student
students$average <- rowMeans(students[, 2:4])
students

# Find high performers (average > 85)
high_performers <- students[students$average > 85, ]
high_performers

# Calculate class statistics
class_stats <- data.frame(
  subject = c("Math", "Science", "English"),
  mean_score = c(mean(students$math), mean(students$science), mean(students$english)),
  median_score = c(median(students$math), median(students$science), median(students$english)),
  sd_score = c(sd(students$math), sd(students$science), sd(students$english))
)
class_stats

# Correlation analysis
correlation_matrix <- cor(students[, 2:5])
correlation_matrix
```

### Example 2: Survey Data Analysis

```r
# Create comprehensive survey data
survey <- data.frame(
  respondent_id = 1:15,
  age = c(25, 30, 35, 28, 42, 33, 29, 31, 27, 38, 26, 34, 39, 32, 36),
  gender = factor(c("Female", "Male", "Male", "Female", "Female", 
                   "Male", "Female", "Male", "Female", "Male",
                   "Female", "Male", "Female", "Male", "Female")),
  education = factor(c("Bachelor", "Master", "PhD", "Bachelor", "Master",
                      "Bachelor", "Master", "PhD", "Bachelor", "Master",
                      "Bachelor", "PhD", "Master", "Bachelor", "Master"),
                    levels = c("Bachelor", "Master", "PhD")),
  satisfaction = factor(c("High", "Medium", "Low", "High", "Medium",
                         "High", "Medium", "Low", "High", "Medium",
                         "High", "Medium", "Low", "High", "Medium"),
                       levels = c("Low", "Medium", "High"),
                       ordered = TRUE),
  income = c(45000, 65000, 85000, 52000, 72000, 48000, 68000, 90000, 
             55000, 75000, 50000, 95000, 78000, 58000, 82000)
)

# Summary statistics
summary(survey)

# Cross-tabulation
table(survey$gender, survey$education)
table(survey$education, survey$satisfaction)

# Income analysis by education
income_by_education <- aggregate(income ~ education, data = survey, FUN = mean)
income_by_education

# Age distribution
age_summary <- summary(survey$age)
age_summary

# Satisfaction analysis
satisfaction_summary <- table(survey$satisfaction)
satisfaction_summary
```

### Example 3: Matrix Operations for Statistical Analysis

```r
# Create correlation matrix
correlation_data <- matrix(c(1.0, 0.8, 0.6,
                           0.8, 1.0, 0.7,
                           0.6, 0.7, 1.0), nrow = 3, ncol = 3)
colnames(correlation_data) <- c("Math", "Science", "English")
rownames(correlation_data) <- c("Math", "Science", "English")
correlation_data

# Matrix operations
eigenvalues <- eigen(correlation_data)$values
eigenvectors <- eigen(correlation_data)$vectors

# Principal Component Analysis (simplified)
# Calculate eigenvalues and eigenvectors
eigen_decomp <- eigen(correlation_data)
principal_components <- eigen_decomp$vectors
explained_variance <- eigen_decomp$values / sum(eigen_decomp$values)
explained_variance
```

## Exercises

### Exercise 1: Matrix Operations
Create two 3x3 matrices and perform various operations (addition, multiplication, transpose, determinant).

```r
# Your solution here
A <- matrix(1:9, nrow = 3, ncol = 3)
B <- matrix(9:1, nrow = 3, ncol = 3)

# Operations
A_plus_B <- A + B
A_times_B <- A %*% B
A_transpose <- t(A)
det_A <- det(A)
```

### Exercise 2: Data Frame Creation and Manipulation
Create a data frame with information about 5 movies including title, year, rating, and genre. Then perform various operations.

```r
# Your solution here
movies <- data.frame(
  title = c("The Matrix", "Inception", "Interstellar", "The Dark Knight", "Pulp Fiction"),
  year = c(1999, 2010, 2014, 2008, 1994),
  rating = c(8.7, 8.8, 8.6, 9.0, 8.9),
  genre = factor(c("Sci-Fi", "Sci-Fi", "Sci-Fi", "Action", "Crime"))
)

# Operations
high_rated <- movies[movies$rating > 8.8, ]
recent_movies <- movies[movies$year > 2000, ]
```

### Exercise 3: List Manipulation
Create a list containing different types of data and practice accessing and modifying elements.

```r
# Your solution here
complex_list <- list(
  personal_info = list(name = "John", age = 30, city = "NYC"),
  scores = c(85, 92, 78, 96),
  courses = list("Statistics", "Programming", "Data Analysis"),
  grades = data.frame(subject = c("Math", "Science"), grade = c(85, 92))
)

# Access and modify
complex_list$personal_info$age
complex_list$scores[2]
complex_list$new_element <- "Additional data"
```

### Exercise 4: Factor Analysis
Create a factor variable for education levels and analyze the distribution of responses.

```r
# Your solution here
education_levels <- factor(c("High School", "Bachelor", "Master", "PhD", "Bachelor",
                            "Master", "High School", "PhD", "Bachelor", "Master"),
                          levels = c("High School", "Bachelor", "Master", "PhD"))

# Analysis
table(education_levels)
summary(education_levels)
levels(education_levels)
```

### Exercise 5: Missing Data Handling
Create a dataset with missing values and practice different methods of handling them.

```r
# Your solution here
data_with_missing <- data.frame(
  id = 1:10,
  age = c(25, 30, NA, 35, 28, 42, 33, NA, 27, 38),
  income = c(45000, 65000, 85000, NA, 72000, 48000, 68000, 90000, 55000, NA),
  education = c("Bachelor", "Master", "PhD", "Bachelor", "Master", 
                "Bachelor", "Master", "PhD", "Bachelor", "Master")
)

# Handle missing data
complete_cases <- na.omit(data_with_missing)
age_mean <- mean(data_with_missing$age, na.rm = TRUE)
income_median <- median(data_with_missing$income, na.rm = TRUE)
```

## Next Steps

In the next chapter, we'll learn how to import data from various sources and manipulate it for analysis. We'll cover:

- **Data Import**: Reading data from CSV, Excel, and other formats
- **Data Cleaning**: Handling missing values, outliers, and data quality issues
- **Data Transformation**: Reshaping and restructuring data
- **Data Validation**: Ensuring data integrity and consistency
- **Data Export**: Saving processed data in various formats

---

**Key Takeaways:**
- Data frames are the most important structure for statistical analysis
- Matrices are essential for mathematical operations and linear algebra
- Lists provide flexibility for complex, heterogeneous data structures
- Factors are crucial for categorical variables in statistical models
- Arrays handle multi-dimensional data efficiently
- Always check data types and structures before analysis
- Handle missing data appropriately based on the context
- Understanding data structure properties is key to effective manipulation 