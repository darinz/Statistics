"""
Introduction to Python - Code Examples

This file contains all the Python code examples from the Introduction to Python lesson.
Each section corresponds to a topic covered in the markdown file.

To run this file:
1. Make sure you have Python installed
2. Run: python 01_introduction_to_python.py

Or run individual sections in a Jupyter notebook by copying the relevant code blocks.
"""

# =============================================================================
# SECTION 1: Your First Python Session
# =============================================================================

def first_python_session():
    """Demonstrate basic Python operations and variable assignment."""
    print("=== Your First Python Session ===")
    
    # Print "Hello, World!" - the traditional first program
    print("Hello, World!")

    # Basic arithmetic operations
    print(2 + 3)    # Addition: 2 + 3 = 5
    print(5 * 4)    # Multiplication: 5 × 4 = 20
    print(10 / 2)   # Division: 10 ÷ 2 = 5
    print(2 ** 3)   # Exponentiation: 2³ = 8

    # Create simple variables and perform operations
    x = 5   # Assign value 5 to variable x
    y = 10  # Assign value 10 to variable y
    print(x + y)    # Add x and y: 5 + 10 = 15

    # Check the type of an object
    print(type(x))  # Returns <class 'int'>

# =============================================================================
# SECTION 2: Python as a Calculator
# =============================================================================

def python_as_calculator():
    """Demonstrate Python's mathematical capabilities."""
    print("\n=== Python as a Calculator ===")
    
    # Basic arithmetic operations
    print(2 + 3)    # Addition: 2 + 3 = 5
    print(5 - 2)    # Subtraction: 5 - 2 = 3
    print(4 * 3)    # Multiplication: 4 × 3 = 12
    print(10 / 2)   # Division: 10 ÷ 2 = 5
    print(2 ** 3)   # Exponentiation: 2³ = 8

    # Mathematical functions
    import math
    print(math.sqrt(16))   # Square root: √16 = 4
    print(math.log(10))    # Natural logarithm: ln(10) ≈ 2.302585
    print(math.log10(100)) # Base-10 logarithm: log₁₀(100) = 2
    print(math.exp(1))     # Exponential function: e¹ ≈ 2.718282
    print(abs(-5))         # Absolute value: |-5| = 5
    print(round(3.14159, 2)) # Round to 2 decimal places: 3.14
    print(math.ceil(3.2))  # Ceiling function: ⌈3.2⌉ = 4
    print(math.floor(3.8)) # Floor function: ⌊3.8⌋ = 3

    # Trigonometric functions (angles in radians)
    print(math.sin(math.pi/2))  # Sine of 90° = 1
    print(math.cos(math.pi))    # Cosine of 180° = -1
    print(math.tan(math.pi/4))  # Tangent of 45° = 1

# =============================================================================
# SECTION 3: Variables and Assignment
# =============================================================================

def variables_and_assignment():
    """Demonstrate variable assignment and type checking."""
    print("\n=== Variables and Assignment ===")
    
    # Assignment using =
    x = 5    # Assign 5 to variable x
    y = 10   # Assign 10 to variable y

    # Print variables
    print(x)         # Display value of x
    print(y)         # Display value of y

    # Check variable type
    print(type(x))      # <class 'int'>
    print(isinstance(x, int)) # True if x is integer
    print(isinstance(x, float)) # False

    # Note: In Jupyter, use %reset to clear all variables
    # del x  # Remove variable x (uncomment if needed)

# =============================================================================
# SECTION 4: Data Types - Numeric
# =============================================================================

def numeric_data_types():
    """Demonstrate numeric data types in Python."""
    print("\n=== Numeric Data Types ===")
    
    # Numeric data examples
    age = 25        # Integer
    height = 175.5  # Decimal number
    temperature = -5.2 # Negative number
    pi_value = 3.14159 # Mathematical constant

    # Check data types
    print(type(age))       # <class 'int'>
    print(type(height))    # <class 'float'>
    print(isinstance(age, int))  # True
    print(isinstance(height, float))  # True

    # Integer type (explicit)
    age_int = int(25)
    print(type(age_int))   # <class 'int'>

# =============================================================================
# SECTION 5: Data Types - String
# =============================================================================

def string_data_types():
    """Demonstrate string data types and operations."""
    print("\n=== String Data Types ===")
    
    # String data
    name = "John Doe"
    city = 'New York'
    email = "john.doe@email.com"

    # Check string data
    print(type(name))      # <class 'str'>
    print(isinstance(name, str)) # True
    print(len(name))      # Length of string: 8

    # String operations
    print(name.upper())    # Convert to uppercase: "JOHN DOE"
    print(name.lower())    # Convert to lowercase: "john doe"
    print("Hello " + name) # Concatenate strings: "Hello John Doe"

# =============================================================================
# SECTION 6: Data Types - Boolean
# =============================================================================

def boolean_data_types():
    """Demonstrate boolean data types and logical operations."""
    print("\n=== Boolean Data Types ===")
    
    # Boolean data
    age = 25
    is_student = True
    is_working = False
    is_adult = age >= 18  # Logical expression

    # Logical operations
    print(True and True)    # AND: True
    print(True or False)    # OR: True
    print(not True)         # NOT: False

    # Comparison operators
    print(5 > 3)           # Greater than: True
    print(5 == 5)          # Equal to: True
    print(5 != 3)          # Not equal to: True
    print(5 >= 5)          # Greater than or equal: True

# =============================================================================
# SECTION 7: Data Types - List
# =============================================================================

def list_data_types():
    """Demonstrate list data types and operations."""
    print("\n=== List Data Types ===")
    
    # List data
    gender = ["Male", "Female", "Male", "Female"]
    education = ["High School", "Bachelor", "Master", "PhD"]

    # List properties
    print(set(gender))      # Unique values: {'Female', 'Male'}
    print(len(gender))      # Number of elements: 4
    print(gender.count("Male")) # Frequency of 'Male': 2

    # Ordered lists
    satisfaction = ["Low", "Medium", "High", "Medium", "High"]
    # To treat as ordered, use pandas Categorical if needed
    import pandas as pd
    satisfaction_cat = pd.Categorical(satisfaction, categories=["Low", "Medium", "High"], ordered=True)
    print(satisfaction_cat)

# =============================================================================
# SECTION 8: Working with Lists - Creating Lists
# =============================================================================

def creating_lists():
    """Demonstrate different ways to create lists."""
    print("\n=== Creating Lists ===")
    
    # Numeric list
    numbers = [1, 2, 3, 4, 5]
    print(numbers)

    # String list
    names = ["Alice", "Bob", "Charlie", "Diana"]
    print(names)

    # Boolean list
    logical_list = [True, False, True, False, True]
    print(logical_list)

    # Using range and list comprehensions
    print(list(range(1, 11)))              # Sequence from 1 to 10
    print(list(range(1, 11, 2)))           # Sequence from 1 to 10, step 2: 1,3,5,7,9
    print([5] * 3)                         # Repeat 5 three times: 5,5,5

# =============================================================================
# SECTION 9: Working with Lists - List Operations
# =============================================================================

def list_operations():
    """Demonstrate element-wise operations on lists."""
    print("\n=== List Operations ===")
    
    # Create lists
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]

    # Element-wise operations with list comprehensions
    print([i + 2 for i in x])        # Add 2 to each element
    print([i * 3 for i in x])        # Multiply each element by 3
    print([i ** 2 for i in x])       # Square each element
    import math
    print([math.sqrt(i) for i in x]) # Square root of each element

    # Element-wise operations between two lists
    print([a + b for a, b in zip(x, y)])        # Addition
    print([a * b for a, b in zip(x, y)])        # Multiplication
    print([a / b for a, b in zip(x, y)])        # Division

    # Logical operations on lists
    print([i > 3 for i in x])        # Compare each element
    print([i == 3 for i in x])       # Check equality

# =============================================================================
# SECTION 10: Working with Lists - List Functions and Statistics
# =============================================================================

def list_statistics():
    """Demonstrate statistical functions on lists."""
    print("\n=== List Functions and Statistics ===")
    
    # Create a sample data list
    data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]

    # Basic summary statistics
    print(len(data))     # Number of elements: 10
    print(sum(data))     # Sum of all elements: 55
    print(sum(data) / len(data))  # Arithmetic mean: 5.5
    import statistics
    print(statistics.mean(data))  # Mean
    print(statistics.median(data)) # Median
    print(min(data))     # Minimum value: 1
    print(max(data))     # Maximum value: 10
    print((min(data), max(data))) # Range

    # Measures of variability
    print(statistics.variance(data)) # Variance
    print(statistics.stdev(data))    # Standard deviation

    # Quantiles and percentiles
    import numpy as np
    print(np.percentile(data, [0, 25, 50, 75, 100])) # Quartiles
    print(np.percentile(data, 90)) # 90th percentile

# =============================================================================
# SECTION 11: Working with Lists - Indexing and Slicing
# =============================================================================

def list_indexing():
    """Demonstrate list indexing and slicing operations."""
    print("\n=== List Indexing and Slicing ===")
    
    # Create a list
    scores = [85, 92, 78, 96, 88, 91, 87, 94, 89, 93]

    # Indexing (Python uses 0-based indexing)
    print(scores[0])        # First element: 85
    print(scores[4])        # Fifth element: 88
    print([scores[i] for i in [0,2,4]]) # Multiple elements: 85, 78, 88

    # Logical indexing (using list comprehensions)
    print([score for score in scores if score > 90]) # Elements greater than 90: 92, 96, 91, 94, 93

    # Negative indexing (exclude elements)
    print(scores[1:])       # All elements except first
    print([scores[i] for i in range(len(scores)) if i not in [0,2,4]]) # Exclude 1st, 3rd, 5th

# =============================================================================
# SECTION 12: Getting Help in Python
# =============================================================================

def getting_help():
    """Demonstrate Python's help system."""
    print("\n=== Getting Help in Python ===")
    
    # Get help for a function
    # help(len)  # Uncomment to see help
    # help(print)  # Uncomment to see help

    # Get information about objects
    data = [1, 2, 3, 4, 5]
    print(type(data))        # Type of object
    print(dir(data))         # List available methods
    # print(vars())            # List all variables in current scope (uncomment if needed)

    # Package information
    # help(math)  # Help for math module (uncomment to see help)

# =============================================================================
# SECTION 13: Best Practices - Code Style and Organization
# =============================================================================

def code_style_examples():
    """Demonstrate good coding practices."""
    print("\n=== Code Style and Organization ===")
    
    # Good: Use descriptive variable names
    student_scores = [85, 92, 78, 96, 88]
    class_average = sum(student_scores) / len(student_scores)

    # Good: Use spaces around operators for readability
    x = 5 + 3
    y = x * 2

    # Good: Use comments to explain complex code
    # Calculate the mean and standard deviation of student scores
    import statistics
    mean_score = statistics.mean(student_scores)
    sd_score = statistics.stdev(student_scores)

    # Good: Use consistent indentation for control structures
    if mean_score > 85:
        print("High performing class")
        print(f"Average score: {mean_score:.2f}")
    else:
        print("Needs improvement")
        print(f"Current average: {mean_score:.2f}")

    # Good: Use functions to organize code
    def calculate_stats(data):
        mean_val = statistics.mean(data)
        sd_val = statistics.stdev(data)
        return {"mean": mean_val, "sd": sd_val}
    
    stats = calculate_stats(student_scores)
    print(f"Statistics: {stats}")

# =============================================================================
# SECTION 14: Best Practices - Error Handling and Debugging
# =============================================================================

def error_handling_examples():
    """Demonstrate error handling and debugging techniques."""
    print("\n=== Error Handling and Debugging ===")
    
    # Check for errors and handle them gracefully
    try:
        # This will cause an error since non_existent_variable is not defined
        # result = mean(non_existent_variable)
        result = statistics.mean([1, 2, 3])  # This will work
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error occurred: {e}")

    # Use print statements for debugging
    x = 5
    print(f"x = {x}")
    print(f"x squared = {x ** 2}")

# =============================================================================
# SECTION 15: Mathematical Concepts in Python
# =============================================================================

def mathematical_concepts():
    """Demonstrate mathematical and statistical concepts."""
    print("\n=== Mathematical Concepts in Python ===")
    
    # Understanding Statistical Measures
    
    # Mean (Arithmetic Average): The sum of all values divided by the number of values
    # Mathematical formula: μ = (Σxᵢ) / n
    data = [2, 4, 6, 8, 10]
    mean_value = sum(data) / len(data)  # Manual calculation
    import statistics
    mean_value_auto = statistics.mean(data)           # Python function
    print(f"Mean (manual): {mean_value}")
    print(f"Mean (auto): {mean_value_auto}")

    # Median: The middle value when data is ordered
    # For odd number of values: middle value
    # For even number of values: average of two middle values
    data = [1, 3, 5, 7, 9]
    median_value = statistics.median(data)
    print(f"Median: {median_value}")

    # Variance: Average squared deviation from the mean
    # Mathematical formula: σ² = Σ(xᵢ - μ)² / (n-1)
    data = [2, 4, 6, 8, 10]
    mean_data = sum(data) / len(data)
    variance_manual = sum((x - mean_data) ** 2 for x in data) / (len(data) - 1)
    variance_auto = statistics.variance(data)
    print(f"Variance (manual): {variance_manual}")
    print(f"Variance (auto): {variance_auto}")

    # Standard Deviation: Square root of variance
    # Mathematical formula: σ = √σ²
    import math
    sd_manual = math.sqrt(statistics.variance(data))
    sd_auto = statistics.stdev(data)
    print(f"Standard Deviation (manual): {sd_manual}")
    print(f"Standard Deviation (auto): {sd_auto}")

# =============================================================================
# SECTION 16: Exercises
# =============================================================================

def exercise_1_basic_operations():
    """Exercise 1: Basic Operations and Variables"""
    print("\n=== Exercise 1: Basic Operations and Variables ===")
    
    # Create variables for your age, height (in cm), and favorite color
    age = 25
    height = 175
    favorite_color = "blue"

    # Basic operations
    age_in_months = age * 12
    height_in_meters = height / 100
    bmi = 70 / (height_in_meters ** 2)  # Assuming weight of 70kg
    
    print(f"Age in months: {age_in_months}")
    print(f"Height in meters: {height_in_meters}")
    print(f"BMI: {bmi:.2f}")

def exercise_2_list_creation():
    """Exercise 2: List Creation and Manipulation"""
    print("\n=== Exercise 2: List Creation and Manipulation ===")
    
    # Create lists for different data types
    test_scores = [85, 92, 78, 96, 88]
    friend_names = ["Alice", "Bob", "Charlie"]
    food_preferences = [True, False, True, True, False]
    
    print(f"Test scores: {test_scores}")
    print(f"Friend names: {friend_names}")
    print(f"Food preferences: {food_preferences}")

def exercise_3_summary_statistics():
    """Exercise 3: Summary Statistics"""
    print("\n=== Exercise 3: Summary Statistics ===")
    
    test_scores = [85, 92, 78, 96, 88]
    
    import statistics
    mean_score = statistics.mean(test_scores)
    median_score = statistics.median(test_scores)
    sd_score = statistics.stdev(test_scores)

    # Interpretation
    print(f"Mean score: {mean_score:.2f}")
    print(f"Median score: {median_score}")
    print(f"Standard deviation: {sd_score:.2f}")

def exercise_4_data_types():
    """Exercise 4: Data Types and Type Conversion"""
    print("\n=== Exercise 4: Data Types and Type Conversion ===")
    
    numeric_var = 42
    character_var = "42"
    logical_var = True

    # Type conversions
    print(f"String from number: {str(numeric_var)}")
    print(f"Number from string: {int(character_var)}")
    print(f"Boolean from 1: {bool(1)}")  # 1 becomes True
    print(f"Boolean from 0: {bool(0)}")  # 0 becomes False

def exercise_5_list_operations():
    """Exercise 5: List Operations"""
    print("\n=== Exercise 5: List Operations ===")
    
    vector1 = [1, 2, 3, 4, 5]
    vector2 = [10, 20, 30, 40, 50]

    # Operations
    sum_vectors = [a + b for a, b in zip(vector1, vector2)]
    product_vectors = [a * b for a, b in zip(vector1, vector2)]
    mean_vector1 = sum(vector1) / len(vector1)
    import statistics
    sd_vector2 = statistics.stdev(vector2)
    
    print(f"Sum of vectors: {sum_vectors}")
    print(f"Product of vectors: {product_vectors}")
    print(f"Mean of vector1: {mean_vector1}")
    print(f"Standard deviation of vector2: {sd_vector2:.2f}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """Run all examples when this file is executed directly."""
    print("Introduction to Python - Code Examples")
    print("=" * 50)
    
    # Run all sections
    first_python_session()
    python_as_calculator()
    variables_and_assignment()
    numeric_data_types()
    string_data_types()
    boolean_data_types()
    list_data_types()
    creating_lists()
    list_operations()
    list_statistics()
    list_indexing()
    getting_help()
    code_style_examples()
    error_handling_examples()
    mathematical_concepts()
    
    # Run exercises
    exercise_1_basic_operations()
    exercise_2_list_creation()
    exercise_3_summary_statistics()
    exercise_4_data_types()
    exercise_5_list_operations()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("\nTo run individual sections, call the specific functions:")
    print("Example: first_python_session()")
    print("Example: python_as_calculator()")
    print("Example: exercise_1_basic_operations()") 