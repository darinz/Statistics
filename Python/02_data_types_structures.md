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

**📁 Code Reference**: See `creating_arrays_matrices()` function in `02_data_types_structures.py`

This section demonstrates:
- Creating arrays from lists using `np.arange()` and `reshape()`
- Creating matrices with specific dimensions
- Creating matrices with row and column names using pandas DataFrame
- Creating identity matrices using `np.eye()`

### Matrix Operations

**📁 Code Reference**: See `matrix_operations()` function in `02_data_types_structures.py`

Basic matrix operations include:
- Element-wise operations (addition, subtraction, multiplication, division)
- Matrix multiplication using `@` operator or `np.dot()`
- Matrix transpose using `.T` attribute
- Matrix properties (shape, dimensions)

### Matrix Functions and Statistical Operations

**📁 Code Reference**: See `matrix_functions_statistics()` function in `02_data_types_structures.py`

Statistical operations on matrices:
- Row and column operations (sum, mean across axes)
- Diagonal operations (extract diagonal, create diagonal matrices)
- Matrix determinant using `np.linalg.det()`
- Matrix inverse using `np.linalg.inv()`

### Matrix Indexing and Slicing

**📁 Code Reference**: See `matrix_indexing_slicing()` function in `02_data_types_structures.py`

Accessing matrix elements:
- Element access using `[row, column]` indexing
- Row and column slicing
- Submatrix extraction
- Logical indexing with boolean masks

## Pandas DataFrames

DataFrames are the most commonly used data structure in Python for statistical analysis. They are like tables with rows and columns, where each column can have a different data type. DataFrames are essentially collections of Series (columns) of equal length.

### Structure of DataFrames
- Each column represents a variable
- Each row represents an observation
- Different columns can have different data types
- All columns must have the same length

### Creating DataFrames

**📁 Code Reference**: See `creating_dataframes()` function in `02_data_types_structures.py`

Different ways to create DataFrames:
- From lists of data
- Direct creation with dictionaries
- With custom row indices
- From various data sources

### Working with DataFrames

**📁 Code Reference**: See `working_with_dataframes()` function in `02_data_types_structures.py`

DataFrame operations include:
- Viewing structure and information with `.info()`
- Accessing columns using names or attributes
- Accessing rows using `.iloc[]` for integer indexing
- Accessing specific elements using various indexing methods

### Modifying DataFrames

**📁 Code Reference**: See `modifying_dataframes()` function in `02_data_types_structures.py`

DataFrame modification techniques:
- Adding new columns with calculated values
- Removing columns using `.drop()`
- Renaming columns using `.rename()`
- Adding new rows using `pd.concat()`
- Combining DataFrames column-wise

### DataFrame Operations and Analysis

**📁 Code Reference**: See `dataframe_operations_analysis()` function in `02_data_types_structures.py`

Analysis operations:
- Summary statistics with `.describe()`
- Subsetting based on conditions
- Multiple condition filtering
- Sorting with `.sort_values()`
- Aggregation functions (mean, median, standard deviation)

## Lists

Lists are flexible data structures that can contain elements of different types and sizes. They are essential for storing complex, heterogeneous data and are the foundation for many advanced Python objects.

### Understanding Lists

**📁 Code Reference**: See `understanding_lists()` function in `02_data_types_structures.py`

Lists can contain:
- Numbers, strings, booleans
- Other lists (nested lists)
- Dictionaries
- DataFrames
- Any combination of the above

### Working with Lists

**📁 Code Reference**: See `working_with_lists()` function in `02_data_types_structures.py`

List operations include:
- Checking list properties (length)
- Adding elements with `.append()` and `.extend()`
- Removing elements with `.remove()`
- Converting lists of dictionaries to DataFrames

### Advanced List Operations

**📁 Code Reference**: See `advanced_list_operations()` function in `02_data_types_structures.py`

Advanced list techniques:
- Lists of functions for batch processing
- Lists of DataFrames for complex data structures
- Functional programming approaches with lists

## Arrays (Multi-dimensional)

NumPy arrays can have more than two dimensions and are useful for storing complex data structures.

### Understanding Arrays

An n-dimensional array has n indices. For example:
- 1D array: vector
- 2D array: matrix
- 3D array: cube of data
- 4D+ array: hypercube

### Creating and Working with Arrays

**📁 Code Reference**: See `multi_dimensional_arrays()` function in `02_data_types_structures.py`

Multi-dimensional array operations:
- Creating 3D arrays with `reshape()`
- Accessing elements using multiple indices
- Slicing across different dimensions
- Array properties (shape, size)

### Array Operations

**📁 Code Reference**: See `array_operations()` function in `02_data_types_structures.py`

Operations on multi-dimensional arrays:
- Element-wise operations between arrays
- Applying functions across dimensions using `np.apply_over_axes()`
- Statistical operations across different axes

## Categorical Data (Factors)

Categorical data is used to represent variables with a fixed number of possible values (categories). In Python, this is handled using pandas' `Categorical` type.

### Understanding Categorical Data

Categorical variables have two important properties:
1. **Categories**: The set of possible values
2. **Order**: Whether the categories have a natural order

### Creating and Working with Categorical Data

**📁 Code Reference**: See `categorical_data_basics()` function in `02_data_types_structures.py`

Basic categorical operations:
- Creating categorical variables with `pd.Categorical()`
- Checking categories and integer codes
- Creating ordered categorical variables
- Frequency tables with `pd.value_counts()`

### Categorical Analysis and Manipulation

**📁 Code Reference**: See `categorical_analysis_manipulation()` function in `02_data_types_structures.py`

Advanced categorical operations:
- Creating categorical variables with missing categories
- Reordering categories
- Adding new categories
- Converting between categorical and string types
- Using categorical data in DataFrames

## Data Type Conversion

Understanding how to convert between data types is crucial for data manipulation and analysis.

### Conversion Functions

**📁 Code Reference**: See `data_type_conversion()` function in `02_data_types_structures.py`

Type conversion operations:
- Numeric to string conversion
- String to numeric conversion
- List to categorical conversion
- DataFrame to NumPy array conversion
- NumPy array to DataFrame conversion
- List to NumPy array conversion

### Type Checking and Validation

**📁 Code Reference**: See `type_checking_validation()` function in `02_data_types_structures.py`

Type checking techniques:
- Using `type()` to check object types
- Using `isinstance()` for type validation
- Checking data structure types
- Validating specific data types

## Working with Missing Data

Missing data is common in real-world datasets and must be handled appropriately.

### Understanding Missing Data

In Python, missing data is represented by `None` or `np.nan` (for numeric data, especially in pandas and numpy).

### Creating and Detecting Missing Data

**📁 Code Reference**: See `missing_data_basics()` function in `02_data_types_structures.py`

Missing data operations:
- Creating data with missing values
- Detecting missing values with `pd.isna()`
- Counting missing values
- Removing missing values
- Replacing missing values with constants or statistics

### Missing Data in DataFrames

**📁 Code Reference**: See `missing_data_dataframes()` function in `02_data_types_structures.py`

DataFrame missing data handling:
- Checking missing values per column and row
- Removing rows with missing values using `.dropna()`
- Filtering rows with non-missing values
- Filling missing values with statistics using `.fillna()`

## Practical Examples

### Example 1: Student Grades Analysis

**📁 Code Reference**: See `student_grades_analysis()` function in `02_data_types_structures.py`

This example demonstrates:
- Creating a DataFrame with student information
- Calculating average grades for each student
- Finding high performers based on criteria
- Computing class statistics by subject
- Correlation analysis between variables

### Example 2: Survey Data Analysis

**📁 Code Reference**: See `survey_data_analysis()` function in `02_data_types_structures.py`

Survey analysis techniques:
- Creating a comprehensive survey DataFrame
- Summary statistics for all variables
- Cross-tabulation analysis
- Group-based analysis (income by education)
- Distribution analysis for categorical variables

### Example 3: Matrix Operations for Statistical Analysis

**📁 Code Reference**: See `matrix_operations_statistical()` function in `02_data_types_structures.py`

Statistical matrix operations:
- Working with correlation matrices
- Computing eigenvalues and eigenvectors
- Principal Component Analysis (simplified)
- Explained variance calculations

## Exercises

### Exercise 1: Matrix Operations

**📁 Code Reference**: See `exercise_1_matrix_operations()` function in `02_data_types_structures.py`

Create two 3x3 matrices and perform various operations:
- Element-wise addition and multiplication
- Matrix multiplication
- Matrix transpose
- Matrix determinant calculation

### Exercise 2: DataFrame Creation and Manipulation

**📁 Code Reference**: See `exercise_2_dataframe_creation()` function in `02_data_types_structures.py`

Create a DataFrame with movie information and perform operations:
- Creating DataFrame with title, year, rating, and genre
- Filtering high-rated movies
- Filtering recent movies
- Working with categorical genre data

### Exercise 3: List Manipulation

**📁 Code Reference**: See `exercise_3_list_manipulation()` function in `02_data_types_structures.py`

Create and manipulate complex lists:
- Lists containing dictionaries, numbers, strings, and DataFrames
- Accessing nested elements
- Adding and modifying list elements
- Working with heterogeneous data structures

### Exercise 4: Categorical Analysis

**📁 Code Reference**: See `exercise_4_categorical_analysis()` function in `02_data_types_structures.py`

Work with categorical education data:
- Creating categorical variables for education levels
- Analyzing distribution of responses
- Frequency table generation
- Category management

### Exercise 5: Missing Data Handling

**📁 Code Reference**: See `exercise_5_missing_data_handling()` function in `02_data_types_structures.py`

Handle missing data in a dataset:
- Creating data with missing values
- Identifying and counting missing data
- Removing complete cases
- Calculating statistics for imputation

## Running the Code Examples

To run all the code examples in this lesson:

1. **Run the entire file**: Execute `python 02_data_types_structures.py` in your terminal
2. **Run individual sections**: In a Python environment, import and call specific functions:
   ```python
   from data_types_structures import creating_arrays_matrices, creating_dataframes
   creating_arrays_matrices()
   creating_dataframes()
   ```
3. **Interactive learning**: Copy individual functions into Jupyter notebooks for interactive exploration

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
- All code examples are available in the companion Python file for hands-on practice 