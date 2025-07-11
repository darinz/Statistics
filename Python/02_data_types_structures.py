"""
Data Types and Structures - Code Examples

This file contains all the Python code examples from the Data Types and Structures lesson.
Each section corresponds to a topic covered in the markdown file.

To run this file:
1. Make sure you have Python installed with NumPy and Pandas
2. Run: python 02_data_types_structures.py

Or run individual sections in a Jupyter notebook by copying the relevant code blocks.
"""

import numpy as np
import pandas as pd

# =============================================================================
# SECTION 1: NumPy Arrays (Matrices) - Creating Arrays and Matrices
# =============================================================================

def creating_arrays_matrices():
    """Demonstrate creating NumPy arrays and matrices."""
    print("=== Creating Arrays and Matrices ===")
    
    # Create array from list
    numbers = np.arange(1, 13)
    mat1 = numbers.reshape(3, 4)  # 3 rows, 4 columns
    print("Matrix 1 (3x4):")
    print(mat1)

    # Create matrix with specific dimensions (fill by row)
    mat2 = np.arange(1, 13).reshape(3, 4)
    print("\nMatrix 2 (3x4):")
    print(mat2)

    # Create matrix with row and column names (using pandas DataFrame)
    mat_df = pd.DataFrame(np.arange(1, 10).reshape(3, 3),
                         index=["Row1", "Row2", "Row3"],
                         columns=["Col1", "Col2", "Col3"])
    print("\nMatrix with labels:")
    print(mat_df)

    # Create identity matrix (diagonal matrix with 1s)
    identity_matrix = np.eye(3)
    print("\nIdentity matrix (3x3):")
    print(identity_matrix)

# =============================================================================
# SECTION 2: NumPy Arrays (Matrices) - Matrix Operations
# =============================================================================

def matrix_operations():
    """Demonstrate basic matrix operations."""
    print("\n=== Matrix Operations ===")
    
    A = np.array([[1, 3], [2, 4]])
    B = np.array([[5, 7], [6, 8]])

    # Display matrices
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)

    # Element-wise operations
    print("\nElement-wise addition (A + B):")
    print(A + B)
    print("\nElement-wise subtraction (A - B):")
    print(A - B)
    print("\nElement-wise multiplication (A * B):")
    print(A * B)
    print("\nElement-wise division (A / B):")
    print(A / B)

    # Matrix multiplication (dot product)
    print("\nMatrix multiplication (A @ B):")
    print(A @ B)

    # Transpose
    print("\nTranspose of A (A.T):")
    print(A.T)

    # Matrix properties
    print(f"\nShape of A: {A.shape}")
    print(f"Number of dimensions of A: {A.ndim}")

# =============================================================================
# SECTION 3: NumPy Arrays (Matrices) - Matrix Functions and Statistical Operations
# =============================================================================

def matrix_functions_statistics():
    """Demonstrate matrix functions and statistical operations."""
    print("\n=== Matrix Functions and Statistical Operations ===")
    
    mat = np.array([[1, 3, 5], [2, 4, 6]])
    print("Matrix:")
    print(mat)

    # Row and column operations
    print(f"\nSum of each row: {np.sum(mat, axis=1)}")
    print(f"Sum of each column: {np.sum(mat, axis=0)}")
    print(f"Mean of each row: {np.mean(mat, axis=1)}")
    print(f"Mean of each column: {np.mean(mat, axis=0)}")

    # Diagonal operations
    print(f"\nDiagonal elements: {np.diag(mat)}")
    print("\nIdentity matrix (3x3):")
    print(np.eye(3))
    print("\nDiagonal matrix with values [1, 2, 3]:")
    print(np.diag([1, 2, 3]))

    # Matrix determinant (for square matrices)
    square_mat = np.array([[1, 2], [3, 4]])
    print(f"\nDeterminant of {square_mat}: {np.linalg.det(square_mat)}")

    # Matrix inverse (for square, non-singular matrices)
    print(f"\nInverse of {square_mat}:")
    print(np.linalg.inv(square_mat))

# =============================================================================
# SECTION 4: NumPy Arrays (Matrices) - Matrix Indexing and Slicing
# =============================================================================

def matrix_indexing_slicing():
    """Demonstrate matrix indexing and slicing operations."""
    print("\n=== Matrix Indexing and Slicing ===")
    
    mat = np.arange(1, 10).reshape(3, 3)
    print("Matrix:")
    print(mat)

    # Access elements
    print(f"\nElement at row 1, column 2: {mat[0, 1]}")
    print(f"Entire second row: {mat[1, :]}")
    print(f"Entire third column: {mat[:, 2]}")
    print("\nSubmatrix (rows 1-2, columns 2-3):")
    print(mat[0:2, 1:3])

    # Logical indexing
    print("\nBoolean matrix (elements > 5):")
    print(mat > 5)
    print(f"\nElements > 5: {mat[mat > 5]}")

# =============================================================================
# SECTION 5: Pandas DataFrames - Creating DataFrames
# =============================================================================

def creating_dataframes():
    """Demonstrate creating pandas DataFrames."""
    print("\n=== Creating DataFrames ===")
    
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
    print("DataFrame from lists:")
    print(students)

    # Create DataFrame directly
    students2 = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 28],
        "height": [165, 180, 175, 170],
        "is_student": [True, False, False, True]
    })
    print("\nDataFrame created directly:")
    print(students2)

    # Create DataFrame with row names (index)
    students3 = pd.DataFrame({
        "age": [25, 30, 35, 28],
        "height": [165, 180, 175, 170]
    }, index=["Alice", "Bob", "Charlie", "Diana"])
    print("\nDataFrame with custom index:")
    print(students3)

# =============================================================================
# SECTION 6: Pandas DataFrames - Working with DataFrames
# =============================================================================

def working_with_dataframes():
    """Demonstrate working with pandas DataFrames."""
    print("\n=== Working with DataFrames ===")
    
    students = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 28],
        "height": [165, 180, 175, 170],
        "is_student": [True, False, False, True]
    })

    # View DataFrame structure and information
    print("DataFrame info:")
    print(students.info())
    print(f"\nShape: {students.shape}")
    print(f"Columns: {list(students.columns)}")
    print(f"Index: {list(students.index)}")
    print(f"Data types:\n{students.dtypes}")

    # Access columns
    print(f"\nAge column (using name):\n{students['age']}")
    print(f"\nAge column (using attribute):\n{students.age}")

    # Access rows
    print(f"\nFirst row:\n{students.iloc[0]}")
    print(f"\nFirst three rows:\n{students.iloc[0:3]}")
    print(f"\nRows 1 and 3:\n{students.iloc[[0, 2]]}")

    # Access specific elements
    print(f"\nElement at row 1, column 2: {students.iloc[0, 1]}")
    print(f"Age of Bob: {students.loc['Bob', 'age'] if 'Bob' in students.index else 'Bob not in index'}")

# =============================================================================
# SECTION 7: Pandas DataFrames - Modifying DataFrames
# =============================================================================

def modifying_dataframes():
    """Demonstrate modifying pandas DataFrames."""
    print("\n=== Modifying DataFrames ===")
    
    students = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 28],
        "height": [165, 180, 175, 170],
        "is_student": [True, False, False, True]
    })

    # Add new column
    students["bmi"] = 70 / (students["height"] / 100) ** 2  # Assuming 70kg weight
    print("DataFrame with BMI column:")
    print(students)

    # Remove column
    students = students.drop(columns=["bmi"])
    print("\nDataFrame after removing BMI column:")
    print(students)

    # Change column names
    students = students.rename(columns={"age": "student_age"})
    print("\nDataFrame with renamed column:")
    print(students)

    # Add new row
    new_student = pd.DataFrame({
        "name": ["Eve"],
        "student_age": [27],
        "height": [168],
        "is_student": [True]
    })
    students = pd.concat([students.reset_index(drop=True), new_student], ignore_index=True)
    print("\nDataFrame with new student:")
    print(students)

    # Combine DataFrames (column-wise)
    additional_info = pd.DataFrame({
        "gpa": [3.8, 3.5, 3.9, 3.7, 3.6],
        "major": ["Math", "Physics", "Chemistry", "Biology", "Computer Science"]
    })
    students = pd.concat([students, additional_info], axis=1)
    print("\nDataFrame with additional info:")
    print(students)

# =============================================================================
# SECTION 8: Pandas DataFrames - DataFrame Operations and Analysis
# =============================================================================

def dataframe_operations_analysis():
    """Demonstrate DataFrame operations and analysis."""
    print("\n=== DataFrame Operations and Analysis ===")
    
    students = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 28, 27],
        "height": [165, 180, 175, 170, 168],
        "is_student": [True, False, False, True, True],
        "gpa": [3.8, 3.5, 3.9, 3.7, 3.6],
        "major": ["Math", "Physics", "Chemistry", "Biology", "Computer Science"]
    })

    # Summary statistics
    print("Summary statistics:")
    print(students.describe())

    # Subsetting based on conditions
    young_students = students[students["age"] < 30]
    print(f"\nYoung students (age < 30):\n{young_students}")

    tall_students = students[students["height"] > 175]
    print(f"\nTall students (height > 175):\n{tall_students}")

    student_only = students[students["is_student"] == True]
    print(f"\nStudents only:\n{student_only}")

    # Multiple conditions
    young_and_tall = students[(students["age"] < 30) & (students["height"] > 170)]
    print(f"\nYoung and tall students:\n{young_and_tall}")

    # Sorting
    students_sorted_age = students.sort_values(by="age")
    print(f"\nSorted by age:\n{students_sorted_age}")

    students_sorted_height_desc = students.sort_values(by="height", ascending=False)
    print(f"\nSorted by height (descending):\n{students_sorted_height_desc}")

    # Aggregation
    mean_age = students["age"].mean()
    median_height = students["height"].median()
    sd_age = students["age"].std()
    print(f"\nMean age: {mean_age:.2f}")
    print(f"Median height: {median_height}")
    print(f"Standard deviation of age: {sd_age:.2f}")

# =============================================================================
# SECTION 9: Lists - Understanding and Creating Lists
# =============================================================================

def understanding_lists():
    """Demonstrate understanding and creating lists."""
    print("\n=== Understanding Lists ===")
    
    # Create a simple list
    my_list = [
        "John",
        30,
        [85, 92, 78],
        True
    ]
    print("Simple list:")
    print(my_list)

    # Access list elements
    print(f"\nFirst element: {my_list[0]}")
    print(f"Third element (list): {my_list[2]}")

    # Nested lists
    nested_list = [
        {"name": "Alice", "age": 25},
        [90, 85, 88],
        ["Statistics", "Programming", "Data Analysis"],
        {"email": "alice@email.com", "phone": "555-1234"}
    ]
    print("\nNested list:")
    print(nested_list)

    # Access nested elements
    print(f"\nName from nested dict: {nested_list[0]['name']}")
    print(f"Age from nested dict: {nested_list[0]['age']}")
    print(f"Email from nested dict: {nested_list[3]['email']}")

# =============================================================================
# SECTION 10: Lists - Working with Lists
# =============================================================================

def working_with_lists():
    """Demonstrate working with lists."""
    print("\n=== Working with Lists ===")
    
    my_list = [
        "John",
        30,
        [85, 92, 78],
        True
    ]

    # List properties
    print(f"Number of elements: {len(my_list)}")

    # Add elements
    my_list.append("New York")
    print(f"\nAfter append: {my_list}")

    my_list.extend(["reading", "swimming", "coding"])
    print(f"After extend: {my_list}")

    # Remove elements
    my_list.remove("New York")
    print(f"After remove: {my_list}")

    # Convert list of dicts to DataFrame
    list_for_df = [
        {"name": "Alice", "age": 25, "city": "NYC"},
        {"name": "Bob", "age": 30, "city": "LA"},
        {"name": "Charlie", "age": 35, "city": "Chicago"}
    ]
    df = pd.DataFrame(list_for_df)
    print(f"\nDataFrame from list of dicts:\n{df}")

# =============================================================================
# SECTION 11: Lists - Advanced List Operations
# =============================================================================

def advanced_list_operations():
    """Demonstrate advanced list operations."""
    print("\n=== Advanced List Operations ===")
    
    # List of functions
    import statistics
    def mean(x): return statistics.mean(x)
    def median(x): return statistics.median(x)
    def stdev(x): return statistics.stdev(x)

    math_functions = [mean, median, stdev]
    data = [1, 2, 3, 4, 5]
    results = [f(data) for f in math_functions]
    print(f"Results from function list: {results}")

    # List of DataFrames
    df_list = [
        pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]}),
        pd.DataFrame({"name": ["Dr. Smith", "Dr. Jones"], "subject": ["Math", "Physics"]})
    ]
    print(f"\nFirst DataFrame in list:\n{df_list[0]}")
    print(f"\nSecond DataFrame in list:\n{df_list[1]}")

# =============================================================================
# SECTION 12: Arrays (Multi-dimensional) - Creating and Working with Arrays
# =============================================================================

def multi_dimensional_arrays():
    """Demonstrate creating and working with multi-dimensional arrays."""
    print("\n=== Multi-dimensional Arrays ===")
    
    # Create 3D array (2 rows × 3 columns × 4 layers)
    array_data = np.arange(1, 25).reshape(2, 3, 4)
    print("3D array:")
    print(array_data)

    # Access elements
    print(f"\nElement at position (0, 1, 2): {array_data[0, 1, 2]}")
    print(f"\nAll elements in first row (2D slice):")
    print(array_data[0, :, :])
    print(f"\nAll elements in second column (2D slice):")
    print(array_data[:, 1, :])
    print(f"\nFirst layer (2D matrix):")
    print(array_data[:, :, 0])

    # Array properties
    print(f"\nShape: {array_data.shape}")
    print(f"Total elements: {array_data.size}")

# =============================================================================
# SECTION 13: Arrays (Multi-dimensional) - Array Operations
# =============================================================================

def array_operations():
    """Demonstrate operations on multi-dimensional arrays."""
    print("\n=== Array Operations ===")
    
    arr1 = np.arange(1, 9).reshape(2, 2, 2)
    arr2 = np.arange(9, 17).reshape(2, 2, 2)

    print("Array 1:")
    print(arr1)
    print("\nArray 2:")
    print(arr2)

    # Element-wise operations
    print("\nElement-wise addition:")
    print(arr1 + arr2)
    print("\nElement-wise multiplication:")
    print(arr1 * arr2)

    # Apply functions across dimensions
    print("\nSum across first dimension:")
    print(np.apply_over_axes(np.sum, arr1, axes=[0]))
    print("\nMean across second dimension:")
    print(np.apply_over_axes(np.mean, arr1, axes=[1]))
    print("\nMaximum across third dimension:")
    print(np.apply_over_axes(np.max, arr1, axes=[2]))

# =============================================================================
# SECTION 14: Categorical Data - Creating and Working with Categorical Data
# =============================================================================

def categorical_data_basics():
    """Demonstrate creating and working with categorical data."""
    print("\n=== Categorical Data Basics ===")
    
    # Create categorical variable
    gender = pd.Categorical(["Male", "Female", "Male", "Female"])
    print("Categorical gender:")
    print(gender)

    # Check categories
    print(f"\nCategories: {gender.categories}")
    print(f"Integer codes: {gender.codes}")

    # Create categorical with specific categories
    education = pd.Categorical(
        ["High School", "Bachelor", "Master", "PhD"],
        categories=["High School", "Bachelor", "Master", "PhD"]
    )
    print(f"\nCategorical education:\n{education}")

    # Ordered categorical (for ordinal variables)
    satisfaction = pd.Categorical(
        ["Low", "Medium", "High", "Medium", "High"],
        categories=["Low", "Medium", "High"],
        ordered=True
    )
    print(f"\nOrdered categorical satisfaction:\n{satisfaction}")

    # Frequency table
    print(f"\nGender frequency table:\n{pd.value_counts(gender)}")
    print(f"\nEducation frequency table:\n{pd.value_counts(education)}")

# =============================================================================
# SECTION 15: Categorical Data - Categorical Analysis and Manipulation
# =============================================================================

def categorical_analysis_manipulation():
    """Demonstrate categorical analysis and manipulation."""
    print("\n=== Categorical Analysis and Manipulation ===")
    
    # Create categorical with missing categories
    survey_data = pd.Categorical(
        ["Yes", "No", "Yes", "Maybe", "No"],
        categories=["Yes", "No", "Maybe"]
    )
    print("Survey data:")
    print(survey_data)

    # Reorder categories
    education = pd.Categorical([
        "High School", "Bachelor", "Master", "PhD"
    ], categories=["High School", "Bachelor", "Master", "PhD"])
    
    education_reordered = pd.Categorical(
        education,
        categories=["PhD", "Master", "Bachelor", "High School"]
    )
    print(f"\nReordered education:\n{education_reordered}")

    # Add new categories
    education_extended = pd.Categorical(
        education,
        categories=["High School", "Bachelor", "Master", "PhD", "PostDoc"]
    )
    print(f"\nExtended education categories:\n{education_extended}")

    # Convert between categorical and string
    print(f"\nConvert to string:\n{education.astype(str)}")
    print(f"\nConvert from string:\n{pd.Categorical(['A', 'B', 'A', 'C'])}")

    # Categorical in DataFrames
    df_with_cat = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "department": pd.Categorical(["IT", "HR", "IT"],
                                      categories=["IT", "HR", "Finance", "Marketing"])
    })
    print(f"\nDataFrame with categorical:\n{df_with_cat}")

# =============================================================================
# SECTION 16: Data Type Conversion
# =============================================================================

def data_type_conversion():
    """Demonstrate data type conversion functions."""
    print("\n=== Data Type Conversion ===")
    
    # Numeric to string
    x = 123
    print(f"Numeric to string: {str(x)}")
    print(f"List of numbers to strings: {[str(i) for i in [1, 2, 3]]}")

    # String to numeric
    y = "456"
    print(f"\nString to integer: {int(y)}")
    print(f"List of strings to integers: {[int(i) for i in ['1', '2', '3']]}")

    # List to categorical
    z = ["A", "B", "A", "C"]
    print(f"\nList to categorical:\n{pd.Categorical(z)}")

    # DataFrame to NumPy array
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    print(f"\nDataFrame to NumPy array:\n{df.values}")

    # NumPy array to DataFrame
    mat = np.array([[1, 2, 3], [4, 5, 6]])
    df2 = pd.DataFrame(mat, columns=["V1", "V2", "V3"])
    print(f"\nNumPy array to DataFrame:\n{df2}")

    # List to NumPy array
    list_to_array = np.array([1, 2, 3])
    print(f"\nList to NumPy array: {list_to_array}")

    # NumPy array to list
    array_to_list = list(list_to_array)
    print(f"NumPy array to list: {array_to_list}")

# =============================================================================
# SECTION 17: Type Checking and Validation
# =============================================================================

def type_checking_validation():
    """Demonstrate type checking and validation."""
    print("\n=== Type Checking and Validation ===")
    
    # Check data types
    x = 5
    y = "hello"
    z = True
    f = pd.Categorical(["A", "B"])

    print(f"Type of x: {type(x)}")
    print(f"Type of y: {type(y)}")
    print(f"Type of z: {type(z)}")
    print(f"Type of f: {type(f)}")

    # Type checking functions
    print(f"\nisinstance(x, int): {isinstance(x, int)}")
    print(f"isinstance(y, str): {isinstance(y, str)}")
    print(f"isinstance(z, bool): {isinstance(z, bool)}")
    print(f"isinstance(f, pd.Categorical): {isinstance(f, pd.Categorical)}")

    # Check data structure
    df = pd.DataFrame({"x": [1, 2, 3]})
    my_list = [1, 2, 3]
    
    print(f"\nisinstance(x, list): {isinstance(x, list)}")
    print(f"isinstance(f, np.ndarray): {isinstance(f, np.ndarray)}")
    print(f"isinstance(df, pd.DataFrame): {isinstance(df, pd.DataFrame)}")
    print(f"isinstance(my_list, list): {isinstance(my_list, list)}")

    # Check for specific types
    print(f"\nisinstance(x, int): {isinstance(x, int)}")
    print(f"isinstance(x, float): {isinstance(x, float)}")

# =============================================================================
# SECTION 18: Working with Missing Data - Creating and Detecting Missing Data
# =============================================================================

def missing_data_basics():
    """Demonstrate creating and detecting missing data."""
    print("\n=== Missing Data Basics ===")
    
    # Create data with missing values
    data_with_na = [1, 2, np.nan, 4, 5]
    character_with_na = ["a", "b", None, "d"]

    print(f"Data with NaN: {data_with_na}")
    print(f"Character data with None: {character_with_na}")

    # Check for missing values
    print(f"\nBoolean mask for NaN: {pd.isna(data_with_na)}")
    print(f"Any missing values: {any(pd.isna(data_with_na))}")
    print(f"Count of missing values: {np.sum(pd.isna(data_with_na))}")

    # Remove missing values
    data_without_na = [x for x in data_with_na if not pd.isna(x)]
    print(f"\nData without NaN: {data_without_na}")

    # Replace missing values
    data_filled = [0 if pd.isna(x) else x for x in data_with_na]
    print(f"Data filled with 0: {data_filled}")

    # Replace with mean (for numeric data)
    mean_val = np.nanmean(data_with_na)
    data_mean_filled = [mean_val if pd.isna(x) else x for x in data_with_na]
    print(f"Data filled with mean: {data_mean_filled}")

# =============================================================================
# SECTION 19: Working with Missing Data - Missing Data in DataFrames
# =============================================================================

def missing_data_dataframes():
    """Demonstrate handling missing data in DataFrames."""
    print("\n=== Missing Data in DataFrames ===")
    
    df_with_na = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, np.nan, 35, 28],
        "height": [165, 180, np.nan, 170],
        "score": [85, 92, 78, np.nan]
    })

    print("DataFrame with missing values:")
    print(df_with_na)

    # Check for missing values
    print(f"\nMissing values per column:\n{df_with_na.isna().sum()}")
    print(f"\nMissing values per row:\n{df_with_na.isna().sum(axis=1)}")

    # Remove rows with any missing values
    df_complete = df_with_na.dropna()
    print(f"\nComplete cases only:\n{df_complete}")

    # Remove rows with missing values in specific columns
    df_partial = df_with_na[df_with_na["age"].notna()]
    print(f"\nRows with non-missing age:\n{df_partial}")

    # Fill missing values
    df_filled = df_with_na.copy()
    df_filled["age"] = df_filled["age"].fillna(df_filled["age"].mean())
    df_filled["height"] = df_filled["height"].fillna(df_filled["height"].median())
    print(f"\nDataFrame with filled missing values:\n{df_filled}")

# =============================================================================
# SECTION 20: Practical Examples - Student Grades Analysis
# =============================================================================

def student_grades_analysis():
    """Demonstrate student grades analysis example."""
    print("\n=== Student Grades Analysis ===")
    
    students = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "math": [85, 92, 78, 96, 88],
        "science": [88, 85, 92, 89, 91],
        "english": [90, 87, 85, 92, 89],
        "attendance": [95, 88, 92, 96, 90]
    })

    # Calculate average grade for each student
    students["average"] = students[["math", "science", "english"]].mean(axis=1)
    print("Students with average grades:")
    print(students)

    # Find high performers (average > 85)
    high_performers = students[students["average"] > 85]
    print(f"\nHigh performers (average > 85):\n{high_performers}")

    # Calculate class statistics
    class_stats = pd.DataFrame({
        "subject": ["Math", "Science", "English"],
        "mean_score": [students["math"].mean(), students["science"].mean(), students["english"].mean()],
        "median_score": [students["math"].median(), students["science"].median(), students["english"].median()],
        "sd_score": [students["math"].std(), students["science"].std(), students["english"].std()]
    })
    print(f"\nClass statistics:\n{class_stats}")

    # Correlation analysis
    correlation_matrix = students[["math", "science", "english", "attendance"]].corr()
    print(f"\nCorrelation matrix:\n{correlation_matrix}")

# =============================================================================
# SECTION 21: Practical Examples - Survey Data Analysis
# =============================================================================

def survey_data_analysis():
    """Demonstrate survey data analysis example."""
    print("\n=== Survey Data Analysis ===")
    
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
    print("Summary statistics:")
    print(survey.describe(include='all'))

    # Cross-tabulation
    print(f"\nGender vs Education cross-tabulation:\n{pd.crosstab(survey['gender'], survey['education'])}")
    print(f"\nEducation vs Satisfaction cross-tabulation:\n{pd.crosstab(survey['education'], survey['satisfaction'])}")

    # Income analysis by education
    income_by_education = survey.groupby("education")["income"].mean()
    print(f"\nMean income by education:\n{income_by_education}")

    # Age distribution
    age_summary = survey["age"].describe()
    print(f"\nAge distribution:\n{age_summary}")

    # Satisfaction analysis
    satisfaction_summary = survey["satisfaction"].value_counts()
    print(f"\nSatisfaction distribution:\n{satisfaction_summary}")

# =============================================================================
# SECTION 22: Practical Examples - Matrix Operations for Statistical Analysis
# =============================================================================

def matrix_operations_statistical():
    """Demonstrate matrix operations for statistical analysis."""
    print("\n=== Matrix Operations for Statistical Analysis ===")
    
    correlation_data = np.array([
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.7],
        [0.6, 0.7, 1.0]
    ])

    print("Correlation matrix:")
    print(correlation_data)

    # Matrix operations
    eigenvalues, eigenvectors = np.linalg.eig(correlation_data)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"\nEigenvectors:\n{eigenvectors}")

    # Principal Component Analysis (simplified)
    explained_variance = eigenvalues / np.sum(eigenvalues)
    print(f"\nExplained variance ratios: {explained_variance}")

# =============================================================================
# SECTION 23: Exercises
# =============================================================================

def exercise_1_matrix_operations():
    """Exercise 1: Matrix Operations"""
    print("\n=== Exercise 1: Matrix Operations ===")
    
    A = np.arange(1, 10).reshape(3, 3)
    B = np.arange(9, 0, -1).reshape(3, 3)

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)

    # Operations
    A_plus_B = A + B
    A_times_B = A @ B
    A_transpose = A.T
    det_A = np.linalg.det(A)
    
    print(f"\nA + B:\n{A_plus_B}")
    print(f"\nA @ B:\n{A_times_B}")
    print(f"\nA transpose:\n{A_transpose}")
    print(f"\nDeterminant of A: {det_A}")

def exercise_2_dataframe_creation():
    """Exercise 2: DataFrame Creation and Manipulation"""
    print("\n=== Exercise 2: DataFrame Creation and Manipulation ===")
    
    movies = pd.DataFrame({
        "title": ["The Matrix", "Inception", "Interstellar", "The Dark Knight", "Pulp Fiction"],
        "year": [1999, 2010, 2014, 2008, 1994],
        "rating": [8.7, 8.8, 8.6, 9.0, 8.9],
        "genre": pd.Categorical(["Sci-Fi", "Sci-Fi", "Sci-Fi", "Action", "Crime"])
    })

    print("Movies DataFrame:")
    print(movies)

    # Operations
    high_rated = movies[movies["rating"] > 8.8]
    recent_movies = movies[movies["year"] > 2000]
    
    print(f"\nHigh rated movies (rating > 8.8):\n{high_rated}")
    print(f"\nRecent movies (year > 2000):\n{recent_movies}")

def exercise_3_list_manipulation():
    """Exercise 3: List Manipulation"""
    print("\n=== Exercise 3: List Manipulation ===")
    
    complex_list = [
        {"personal_info": {"name": "John", "age": 30, "city": "NYC"}},
        [85, 92, 78, 96],
        ["Statistics", "Programming", "Data Analysis"],
        pd.DataFrame({"subject": ["Math", "Science"], "grade": [85, 92]})
    ]

    print("Complex list:")
    print(complex_list)

    # Access and modify
    print(f"\nAge from nested dict: {complex_list[0]['personal_info']['age']}")
    print(f"Second element of scores: {complex_list[1][1]}")
    
    complex_list.append("Additional data")
    print(f"\nAfter adding element: {complex_list}")

def exercise_4_categorical_analysis():
    """Exercise 4: Categorical Analysis"""
    print("\n=== Exercise 4: Categorical Analysis ===")
    
    education_levels = pd.Categorical([
        "High School", "Bachelor", "Master", "PhD", "Bachelor",
        "Master", "High School", "PhD", "Bachelor", "Master"
    ], categories=["High School", "Bachelor", "Master", "PhD"])

    print("Education levels:")
    print(education_levels)

    # Analysis
    print(f"\nFrequency table:\n{pd.value_counts(education_levels)}")
    print(f"Categories: {education_levels.categories}")

def exercise_5_missing_data_handling():
    """Exercise 5: Missing Data Handling"""
    print("\n=== Exercise 5: Missing Data Handling ===")
    
    data_with_missing = pd.DataFrame({
        "id": range(1, 11),
        "age": [25, 30, np.nan, 35, 28, 42, 33, np.nan, 27, 38],
        "income": [45000, 65000, 85000, np.nan, 72000, 48000, 68000, 90000, 55000, np.nan],
        "education": ["Bachelor", "Master", "PhD", "Bachelor", "Master", 
                      "Bachelor", "Master", "PhD", "Bachelor", "Master"]
    })

    print("Data with missing values:")
    print(data_with_missing)

    # Handle missing data
    complete_cases = data_with_missing.dropna()
    age_mean = data_with_missing["age"].mean()
    income_median = data_with_missing["income"].median()
    
    print(f"\nComplete cases only:\n{complete_cases}")
    print(f"Mean age: {age_mean:.2f}")
    print(f"Median income: {income_median:.2f}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """Run all examples when this file is executed directly."""
    print("Data Types and Structures - Code Examples")
    print("=" * 60)
    
    # Run all sections
    creating_arrays_matrices()
    matrix_operations()
    matrix_functions_statistics()
    matrix_indexing_slicing()
    creating_dataframes()
    working_with_dataframes()
    modifying_dataframes()
    dataframe_operations_analysis()
    understanding_lists()
    working_with_lists()
    advanced_list_operations()
    multi_dimensional_arrays()
    array_operations()
    categorical_data_basics()
    categorical_analysis_manipulation()
    data_type_conversion()
    type_checking_validation()
    missing_data_basics()
    missing_data_dataframes()
    student_grades_analysis()
    survey_data_analysis()
    matrix_operations_statistical()
    
    # Run exercises
    exercise_1_matrix_operations()
    exercise_2_dataframe_creation()
    exercise_3_list_manipulation()
    exercise_4_categorical_analysis()
    exercise_5_missing_data_handling()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("\nTo run individual sections, call the specific functions:")
    print("Example: creating_arrays_matrices()")
    print("Example: creating_dataframes()")
    print("Example: exercise_1_matrix_operations()") 