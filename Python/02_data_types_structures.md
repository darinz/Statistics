# Data Types and Structures

## Overview

Python provides several data structures to handle different types of data efficiently. Understanding these structures is crucial for effective data analysis and scientific computing. Each structure has specific characteristics that make it suitable for particular types of analysis.

### Why Different Data Structures Matter

- **NumPy Arrays**: Efficient for mathematical operations and linear algebra
- **Pandas DataFrames**: Ideal for statistical analysis and data manipulation
- **Lists**: Flexible containers for complex, heterogeneous data
- **Tuples**: Immutable ordered collections
- **Dictionaries**: Key-value pairs for flexible data storage
- **Sets**: Unordered collections of unique elements

## NumPy Arrays (Matrices)

A NumPy array is a multi-dimensional, homogeneous data structure. Arrays are fundamental for mathematical operations, linear algebra, and scientific computations in Python.

### Mathematical Foundation

A matrix A with m rows and n columns is denoted as:
```
A = [a_ij] where i = 1,2,...,m and j = 1,2,...,n
```

### Creating Arrays and Matrices

```python
import numpy as np

# Create array from list
numbers = np.arange(1, 13)
mat1 = numbers.reshape(3, 4)  # 3 rows, 4 columns
print(mat1)

# Create matrix with specific dimensions (fill by row)
mat2 = np.arange(1, 13).reshape(3, 4)
print(mat2)

# Create matrix with row and column names (using pandas DataFrame)
import pandas as pd
mat_df = pd.DataFrame(np.arange(1, 10).reshape(3, 3),
                     index=["Row1", "Row2", "Row3"],
                     columns=["Col1", "Col2", "Col3"])
print(mat_df)

# Create identity matrix (diagonal matrix with 1s)
identity_matrix = np.eye(3)
print(identity_matrix)
```

### Matrix Operations

```python
A = np.array([[1, 3], [2, 4]])
B = np.array([[5, 7], [6, 8]])

# Display matrices
print("A:\n", A)
print("B:\n", B)

# Element-wise operations
print(A + B)    # Addition
print(A - B)    # Subtraction
print(A * B)    # Element-wise multiplication
print(A / B)    # Element-wise division

# Matrix multiplication (dot product)
print(A @ B)    # or np.dot(A, B)

# Transpose
print(A.T)

# Matrix properties
print(A.shape)  # Dimensions (rows, columns)
print(A.ndim)   # Number of dimensions
```

### Matrix Functions and Statistical Operations

```python
mat = np.array([[1, 3, 5], [2, 4, 6]])
print(mat)

# Row and column operations
print(np.sum(mat, axis=1))      # Sum of each row
print(np.sum(mat, axis=0))      # Sum of each column
print(np.mean(mat, axis=1))     # Mean of each row
print(np.mean(mat, axis=0))     # Mean of each column

# Diagonal operations
print(np.diag(mat))             # Extract diagonal (for square matrices)
print(np.eye(3))                # Create 3x3 identity matrix
print(np.diag([1, 2, 3]))       # Create diagonal matrix with values [1, 2, 3]

# Matrix determinant (for square matrices)
print(np.linalg.det(np.array([[1, 2], [3, 4]])))

# Matrix inverse (for square, non-singular matrices)
print(np.linalg.inv(np.array([[1, 2], [3, 4]])))
```

### Matrix Indexing and Slicing

```python
mat = np.arange(1, 10).reshape(3, 3)
print(mat)

# Access elements
print(mat[0, 1])        # Element at row 1, column 2
print(mat[1, :])        # Entire second row
print(mat[:, 2])        # Entire third column
print(mat[0:2, 1:3])    # Submatrix: rows 1-2, columns 2-3

# Logical indexing
print(mat > 5)          # Boolean matrix
print(mat[mat > 5])     # Extract elements > 5
```

## Pandas DataFrames

DataFrames are the most commonly used data structure in Python for statistical analysis. They are like tables with rows and columns, where each column can have a different data type. DataFrames are essentially collections of Series (columns) of equal length.

### Structure of DataFrames
- Each column represents a variable
- Each row represents an observation
- Different columns can have different data types
- All columns must have the same length

### Creating DataFrames

```python
import pandas as pd

# Create DataFrame from lists
name = ["Alice", "Bob", "Charlie", "Diana"]
age = [25, 30, 35, 28]
height = [165, 180, 175, 170]
is_student = [True, False, False, True]

students = pd.DataFrame({
    "name": name,
    "age": age,
    "height": height,
    "is_student": is_student
})
print(students)

# Create DataFrame directly
students = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "age": [25, 30, 35, 28],
    "height": [165, 180, 175, 170],
    "is_student": [True, False, False, True]
})
print(students)

# Create DataFrame with row names (index)
students = pd.DataFrame({
    "age": [25, 30, 35, 28],
    "height": [165, 180, 175, 170]
}, index=["Alice", "Bob", "Charlie", "Diana"])
print(students)
```

### Working with DataFrames

```python
# View DataFrame structure and information
print(students.info())        # Structure of DataFrame
print(students.head())        # First 5 rows
print(students.tail())        # Last 5 rows
print(students.shape)         # Dimensions (rows, columns)
print(students.columns)       # Column names
print(students.index)         # Row names (index)
print(students.dtypes)        # Data types

# Access columns
print(students["age"])       # Using column name
print(students.age)           # Using attribute access

# Access rows
print(students.iloc[0])       # First row
print(students.iloc[0:3])     # First three rows
print(students.iloc[[0, 2]])  # Rows 1 and 3
print(students.drop(students.index[1])) # All rows except row 2

# Access specific elements
print(students.iloc[0, 1])    # Row 1, Column 2
print(students.loc["Bob", "age"])   # Row 'Bob', Column 'age'
print(students["age"][2])    # Third element of age column
```

### Modifying DataFrames

```python
# Add new column
students["bmi"] = 70 / (students["height"] / 100) ** 2  # Assuming 70kg weight

# Remove column
students = students.drop(columns=["bmi"])

# Change column names
students = students.rename(columns={"age": "student_age"})

# Add new row
new_student = pd.DataFrame({
    "name": ["Eve"],
    "age": [27],
    "height": [168],
    "is_student": [True]
})
students = pd.concat([students.reset_index(drop=True), new_student], ignore_index=True)

# Combine DataFrames (column-wise)
additional_info = pd.DataFrame({
    "gpa": [3.8, 3.5, 3.9, 3.7, 3.6],
    "major": ["Math", "Physics", "Chemistry", "Biology", "Computer Science"]
})
students = pd.concat([students, additional_info], axis=1)
```

### DataFrame Operations and Analysis

```python
# Summary statistics
print(students.describe())    # Summary of numeric columns
print(students["age"].describe()) # Summary of specific column

# Subsetting based on conditions
young_students = students[students["age"] < 30]
tall_students = students[students["height"] > 175]
student_only = students[students["is_student"] == True]

# Multiple conditions
young_and_tall = students[(students["age"] < 30) & (students["height"] > 170)]

# Sorting
students_sorted_age = students.sort_values(by="age")
students_sorted_height_desc = students.sort_values(by="height", ascending=False)

# Aggregation
mean_age = students["age"].mean()
median_height = students["height"].median()
sd_age = students["age"].std()
```

## Lists

Lists are flexible data structures that can contain elements of different types and sizes. They are essential for storing complex, heterogeneous data and are the foundation for many advanced Python objects.

### Understanding Lists

Lists can contain:
- Numbers, strings, booleans
- Other lists (nested lists)
- Dictionaries
- DataFrames
- Any combination of the above

### Creating Lists

```python
# Create a simple list
my_list = [
    "John",
    30,
    [85, 92, 78],
    True
]
print(my_list)

# Access list elements
print(my_list[0])        # First element
print(my_list[2])        # Third element (list)

# Nested lists
nested_list = [
    {"name": "Alice", "age": 25},
    [90, 85, 88],
    ["Statistics", "Programming", "Data Analysis"],
    {"email": "alice@email.com", "phone": "555-1234"}
]

# Access nested elements
print(nested_list[0]["name"])
print(nested_list[0]["age"])
print(nested_list[3]["email"])
```

### Working with Lists

```python
# List properties
print(len(my_list))     # Number of elements

# Add elements
my_list.append("New York")
my_list.extend(["reading", "swimming", "coding"])

# Remove elements
my_list.remove("New York")

# Convert list of dicts to DataFrame
list_for_df = [
    {"name": "Alice", "age": 25, "city": "NYC"},
    {"name": "Bob", "age": 30, "city": "LA"},
    {"name": "Charlie", "age": 35, "city": "Chicago"}
]
df = pd.DataFrame(list_for_df)
print(df)
```

### Advanced List Operations

```python
# List of functions
import statistics
def mean(x): return statistics.mean(x)
def median(x): return statistics.median(x)
def stdev(x): return statistics.stdev(x)

math_functions = [mean, median, stdev]
data = [1, 2, 3, 4, 5]
results = [f(data) for f in math_functions]
print(results)

# List of DataFrames
df_list = [
    pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]}),
    pd.DataFrame({"name": ["Dr. Smith", "Dr. Jones"], "subject": ["Math", "Physics"]})
]
print(df_list[0])
print(df_list[1])
```

## Arrays (Multi-dimensional)

NumPy arrays can have more than two dimensions and are useful for storing complex data structures.

### Understanding Arrays

An n-dimensional array has n indices. For example:
- 1D array: vector
- 2D array: matrix
- 3D array: cube of data
- 4D+ array: hypercube

### Creating and Working with Arrays

```python
# Create 3D array (2 rows × 3 columns × 4 layers)
array_data = np.arange(1, 25).reshape(2, 3, 4)
print(array_data)

# Access elements
print(array_data[0, 1, 2])    # Element at position (0, 1, 2)
print(array_data[0, :, :])    # All elements in first row (2D slice)
print(array_data[:, 1, :])    # All elements in second column (2D slice)
print(array_data[:, :, 0])    # First layer (2D matrix)

# Array properties
print(array_data.shape)         # Dimensions
print(array_data.size)          # Total elements
```

### Array Operations

```python
arr1 = np.arange(1, 9).reshape(2, 2, 2)
arr2 = np.arange(9, 17).reshape(2, 2, 2)

# Element-wise operations
print(arr1 + arr2)    # Addition
print(arr1 * arr2)    # Element-wise multiplication

# Apply functions across dimensions
print(np.apply_over_axes(np.sum, arr1, axes=[0]))    # Sum across first dimension
print(np.apply_over_axes(np.mean, arr1, axes=[1]))   # Mean across second dimension
print(np.apply_over_axes(np.max, arr1, axes=[2]))    # Maximum across third dimension
```

## Categorical Data (Factors)

Categorical data is used to represent variables with a fixed number of possible values (categories). In Python, this is handled using pandas' `Categorical` type.

### Understanding Categorical Data

Categorical variables have two important properties:
1. **Categories**: The set of possible values
2. **Order**: Whether the categories have a natural order

### Creating and Working with Categorical Data

```python
# Create categorical variable
gender = pd.Categorical(["Male", "Female", "Male", "Female"])
print(gender)

# Check categories
print(gender.categories)      # Show categories
print(gender.codes)           # Integer codes for categories

# Create categorical with specific categories
education = pd.Categorical(
    ["High School", "Bachelor", "Master", "PhD"],
    categories=["High School", "Bachelor", "Master", "PhD"]
)
print(education)

# Ordered categorical (for ordinal variables)
satisfaction = pd.Categorical(
    ["Low", "Medium", "High", "Medium", "High"],
    categories=["Low", "Medium", "High"],
    ordered=True
)
print(satisfaction)

# Frequency table
print(pd.value_counts(gender))
print(pd.value_counts(education))
```

### Categorical Analysis and Manipulation

```python
# Create categorical with missing categories
survey_data = pd.Categorical(
    ["Yes", "No", "Yes", "Maybe", "No"],
    categories=["Yes", "No", "Maybe"]
)
print(survey_data)

# Reorder categories
education_reordered = pd.Categorical(
    education,
    categories=["PhD", "Master", "Bachelor", "High School"]
)
print(education_reordered)

# Add new categories
education_extended = pd.Categorical(
    education,
    categories=["High School", "Bachelor", "Master", "PhD", "PostDoc"]
)
print(education_extended)

# Convert between categorical and string
print(education.astype(str))
print(pd.Categorical(["A", "B", "A", "C"]))

# Categorical in DataFrames
df_with_cat = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "department": pd.Categorical(["IT", "HR", "IT"],
                                  categories=["IT", "HR", "Finance", "Marketing"])
})
print(df_with_cat)
```

## Data Type Conversion

Understanding how to convert between data types is crucial for data manipulation and analysis.

### Conversion Functions

```python
# Numeric to string
x = 123
print(str(x))     # "123"
print([str(i) for i in [1, 2, 3]])

# String to numeric
y = "456"
print(int(y))       # 456
print([int(i) for i in ["1", "2", "3"]])

# List to categorical
z = ["A", "B", "A", "C"]
print(pd.Categorical(z))

# DataFrame to NumPy array
df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
print(df.values)       # NumPy array with same data

# NumPy array to DataFrame
mat = np.array([[1, 2, 3], [4, 5, 6]])
df2 = pd.DataFrame(mat, columns=["V1", "V2", "V3"])
print(df2)

# List to NumPy array
list_to_array = np.array([1, 2, 3])
print(list_to_array)

# NumPy array to list
array_to_list = list(list_to_array)
print(array_to_list)
```

### Type Checking and Validation

```python
# Check data types
x = 5
y = "hello"
z = True
f = pd.Categorical(["A", "B"])

print(type(x))        # <class 'int'>
print(type(y))        # <class 'str'>
print(type(z))        # <class 'bool'>
print(type(f))        # <class 'pandas.core.arrays.categorical.Categorical'>

# Type checking functions
print(isinstance(x, int))   # True
print(isinstance(y, str))   # True
print(isinstance(z, bool))  # True
print(isinstance(f, pd.Categorical)) # True

# Check data structure
print(isinstance(x, list))    # False
print(isinstance(f, np.ndarray)) # False
print(isinstance(df, pd.DataFrame)) # True
print(isinstance(my_list, list))      # True

# Check for specific types
print(isinstance(x, int))   # True
print(isinstance(x, float)) # False
```

## Working with Missing Data

Missing data is common in real-world datasets and must be handled appropriately.

### Understanding Missing Data

In Python, missing data is represented by `None` or `np.nan` (for numeric data, especially in pandas and numpy).

### Creating and Detecting Missing Data

```python
import numpy as np
import pandas as pd

# Create data with missing values
data_with_na = [1, 2, np.nan, 4, 5]
character_with_na = ["a", "b", None, "d"]

# Check for missing values
print(pd.isna(data_with_na))        # Boolean mask
print(any(pd.isna(data_with_na)))   # True if any missing
print(np.sum(pd.isna(data_with_na)))   # Count of missing values

# Remove missing values
data_without_na = [x for x in data_with_na if not pd.isna(x)]
print(data_without_na)

# Replace missing values
data_filled = [0 if pd.isna(x) else x for x in data_with_na]
print(data_filled)

# Replace with mean (for numeric data)
mean_val = np.nanmean(data_with_na)
data_mean_filled = [mean_val if pd.isna(x) else x for x in data_with_na]
print(data_mean_filled)
```

### Missing Data in DataFrames

```python
df_with_na = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "age": [25, np.nan, 35, 28],
    "height": [165, 180, np.nan, 170],
    "score": [85, 92, 78, np.nan]
})

# Check for missing values
print(df_with_na.isna().sum())  # Missing values per column
print(df_with_na.isna().sum(axis=1))  # Missing values per row

# Remove rows with any missing values
df_complete = df_with_na.dropna()
print(df_complete)

# Remove rows with missing values in specific columns
df_partial = df_with_na[df_with_na["age"].notna()]
print(df_partial)

# Fill missing values
df_filled = df_with_na.copy()
df_filled["age"] = df_filled["age"].fillna(df_filled["age"].mean())
df_filled["height"] = df_filled["height"].fillna(df_filled["height"].median())
print(df_filled)
```

## Practical Examples

### Example 1: Student Grades Analysis

```python
students = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "math": [85, 92, 78, 96, 88],
    "science": [88, 85, 92, 89, 91],
    "english": [90, 87, 85, 92, 89],
    "attendance": [95, 88, 92, 96, 90]
})

# Calculate average grade for each student
students["average"] = students[["math", "science", "english"]].mean(axis=1)
print(students)

# Find high performers (average > 85)
high_performers = students[students["average"] > 85]
print(high_performers)

# Calculate class statistics
class_stats = pd.DataFrame({
    "subject": ["Math", "Science", "English"],
    "mean_score": [students["math"].mean(), students["science"].mean(), students["english"].mean()],
    "median_score": [students["math"].median(), students["science"].median(), students["english"].median()],
    "sd_score": [students["math"].std(), students["science"].std(), students["english"].std()]
})
print(class_stats)

# Correlation analysis
correlation_matrix = students[["math", "science", "english", "attendance"]].corr()
print(correlation_matrix)
```

### Example 2: Survey Data Analysis

```python
survey = pd.DataFrame({
    "respondent_id": range(1, 16),
    "age": [25, 30, 35, 28, 42, 33, 29, 31, 27, 38, 26, 34, 39, 32, 36],
    "gender": pd.Categorical(["Female", "Male", "Male", "Female", "Female", 
                              "Male", "Female", "Male", "Female", "Male",
                              "Female", "Male", "Female", "Male", "Female"]),
    "education": pd.Categorical([
        "Bachelor", "Master", "PhD", "Bachelor", "Master",
        "Bachelor", "Master", "PhD", "Bachelor", "Master",
        "Bachelor", "PhD", "Master", "Bachelor", "Master"
    ], categories=["Bachelor", "Master", "PhD"]),
    "satisfaction": pd.Categorical([
        "High", "Medium", "Low", "High", "Medium",
        "High", "Medium", "Low", "High", "Medium",
        "High", "Medium", "Low", "High", "Medium"
    ], categories=["Low", "Medium", "High"], ordered=True),
    "income": [45000, 65000, 85000, 52000, 72000, 48000, 68000, 90000, 
               55000, 75000, 50000, 95000, 78000, 58000, 82000]
})

# Summary statistics
print(survey.describe(include='all'))

# Cross-tabulation
print(pd.crosstab(survey["gender"], survey["education"]))
print(pd.crosstab(survey["education"], survey["satisfaction"]))

# Income analysis by education
income_by_education = survey.groupby("education")["income"].mean()
print(income_by_education)

# Age distribution
age_summary = survey["age"].describe()
print(age_summary)

# Satisfaction analysis
satisfaction_summary = survey["satisfaction"].value_counts()
print(satisfaction_summary)
```

### Example 3: Matrix Operations for Statistical Analysis

```python
correlation_data = np.array([
    [1.0, 0.8, 0.6],
    [0.8, 1.0, 0.7],
    [0.6, 0.7, 1.0]
])

# Matrix operations
eigenvalues, eigenvectors = np.linalg.eig(correlation_data)
print(eigenvalues)
print(eigenvectors)

# Principal Component Analysis (simplified)
explained_variance = eigenvalues / np.sum(eigenvalues)
print(explained_variance)
```

## Exercises

### Exercise 1: Matrix Operations
Create two 3x3 matrices and perform various operations (addition, multiplication, transpose, determinant).

```python
# Your solution here
import numpy as np
A = np.arange(1, 10).reshape(3, 3)
B = np.arange(9, 0, -1).reshape(3, 3)

# Operations
A_plus_B = A + B
A_times_B = A @ B
A_transpose = A.T
det_A = np.linalg.det(A)
```

### Exercise 2: DataFrame Creation and Manipulation
Create a DataFrame with information about 5 movies including title, year, rating, and genre. Then perform various operations.

```python
# Your solution here
import pandas as pd
movies = pd.DataFrame({
    "title": ["The Matrix", "Inception", "Interstellar", "The Dark Knight", "Pulp Fiction"],
    "year": [1999, 2010, 2014, 2008, 1994],
    "rating": [8.7, 8.8, 8.6, 9.0, 8.9],
    "genre": pd.Categorical(["Sci-Fi", "Sci-Fi", "Sci-Fi", "Action", "Crime"])
})

# Operations
high_rated = movies[movies["rating"] > 8.8]
recent_movies = movies[movies["year"] > 2000]
```

### Exercise 3: List Manipulation
Create a list containing different types of data and practice accessing and modifying elements.

```python
# Your solution here
complex_list = [
    {"personal_info": {"name": "John", "age": 30, "city": "NYC"}},
    [85, 92, 78, 96],
    ["Statistics", "Programming", "Data Analysis"],
    pd.DataFrame({"subject": ["Math", "Science"], "grade": [85, 92]})
]

# Access and modify
print(complex_list[0]["personal_info"]["age"])
print(complex_list[1][1])
complex_list.append("Additional data")
```

### Exercise 4: Categorical Analysis
Create a categorical variable for education levels and analyze the distribution of responses.

```python
# Your solution here
import pandas as pd
education_levels = pd.Categorical([
    "High School", "Bachelor", "Master", "PhD", "Bachelor",
    "Master", "High School", "PhD", "Bachelor", "Master"
], categories=["High School", "Bachelor", "Master", "PhD"])

# Analysis
print(pd.value_counts(education_levels))
print(education_levels.categories)
```

### Exercise 5: Missing Data Handling
Create a dataset with missing values and practice different methods of handling them.

```python
# Your solution here
import numpy as np
import pandas as pd
data_with_missing = pd.DataFrame({
    "id": range(1, 11),
    "age": [25, 30, np.nan, 35, 28, 42, 33, np.nan, 27, 38],
    "income": [45000, 65000, 85000, np.nan, 72000, 48000, 68000, 90000, 55000, np.nan],
    "education": ["Bachelor", "Master", "PhD", "Bachelor", "Master", 
                  "Bachelor", "Master", "PhD", "Bachelor", "Master"]
})

# Handle missing data
complete_cases = data_with_missing.dropna()
age_mean = data_with_missing["age"].mean()
income_median = data_with_missing["income"].median()
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
- Pandas DataFrames are the most important structure for statistical analysis
- NumPy arrays (matrices) are essential for mathematical operations and linear algebra
- Lists provide flexibility for complex, heterogeneous data structures
- Categorical data is crucial for representing variables in statistical models
- Arrays handle multi-dimensional data efficiently
- Always check data types and structures before analysis
- Handle missing data appropriately based on the context
- Understanding data structure properties is key to effective manipulation 