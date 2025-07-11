# Introduction to Python

## What is Python?

Python is a powerful, versatile, and easy-to-learn programming language widely used for data analysis, scientific computing, web development, automation, and more. Created by Guido van Rossum and first released in 1991, Python emphasizes readability and simplicity, making it an excellent choice for beginners and professionals alike.

### Why Learn Python?

- **Free and Open Source**: Python is free to use and has a massive open-source ecosystem
- **Versatile**: Used in data science, web development, automation, machine learning, and more
- **Active Community**: Large, active community of users and developers contributing packages and support
- **Reproducible Research**: Excellent tools for reproducible analysis and reporting (e.g., Jupyter, notebooks)
- **Industry Standard**: Widely used in academia, research, data science, and industry
- **Data Science Power**: Extensive libraries for data analysis, statistics, and machine learning (NumPy, pandas, scikit-learn)
- **Data Visualization**: Powerful plotting and visualization libraries (matplotlib, seaborn, plotly)
- **Machine Learning**: Leading libraries for machine learning and AI (scikit-learn, TensorFlow, PyTorch)

### Python vs. Other Statistical Software

| Feature | Python | R | SPSS | SAS |
|---------|--------|---|------|-----|
| Cost | Free | Free | Expensive | Expensive |
| Learning Curve | Moderate | Moderate | Easy | Steep |
| Statistical Capabilities | Excellent | Excellent | Good | Excellent |
| Graphics | Good | Superior | Basic | Good |
| Programming | Full language | Full language | Limited | Full language |
| Community Support | Excellent | Excellent | Limited | Good |

## Installing Python and Jupyter

### Step 1: Install Python
1. Go to [python.org](https://www.python.org/downloads/)
2. Download Python for your operating system (Windows, macOS, or Linux)
3. Install following the installation wizard
4. Verify installation by opening a terminal/command prompt and running `python --version`

### Step 2: Install Jupyter Notebook (Recommended)
1. Install pip (Python's package manager) if not already installed
2. Install Jupyter with: `pip install notebook`
3. Launch Jupyter Notebook with: `jupyter notebook`

### Understanding the Jupyter Interface

Jupyter provides an interactive environment for writing and running Python code in cells. You can mix code, text, equations, and visualizations in a single document (notebook).

- **Code Cells**: Where you write and execute Python code
- **Markdown Cells**: For formatted text, equations, and explanations
- **Output Cells**: Display results, plots, and errors

## Getting Started with Python

### Your First Python Session

```python
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
```

### Understanding Python as a Calculator

Python can perform all basic mathematical operations and much more:

```python
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
```

### Variables and Assignment

Variables in Python are containers that store values. Understanding variable assignment is fundamental:

```python
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

# Remove variables (not common, but possible)
del x  # Remove variable x
# Note: In Jupyter, use %reset to clear all variables
```

## Data Types in Python

Understanding data types is crucial for statistical analysis. Python has several fundamental data types:

### Numeric
Numeric data represents numbers, including integers and floating-point numbers:

```python
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
```

### String
String data represents text:

```python
# String data
name = "John Doe"
city = 'New York'
email = "john.doe@email.com"

# Check string data
type(name)      # <class 'str'>
isinstance(name, str) # True
len(name)      # Length of string: 8

# String operations
print(name.upper())    # Convert to uppercase: "JOHN DOE"
print(name.lower())    # Convert to lowercase: "john doe"
print("Hello " + name) # Concatenate strings: "Hello John Doe"
```

### Boolean
Boolean data represents True/False values:

```python
# Boolean data
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
```

### List
Lists represent ordered collections of items (can be of mixed types):

```python
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
```

## Working with Lists

Lists are a fundamental data structure in Python. They are ordered, mutable, and can hold multiple values of any type.

### Creating Lists

```python
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
```

### List Operations

Python performs operations element-wise using list comprehensions or with NumPy arrays:

```python
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
```

### List Functions and Statistics

```python
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
```

### List Indexing and Slicing

```python
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
```

## Getting Help in Python

Python has excellent built-in help system and extensive online resources:

### Built-in Help
```python
# Get help for a function
help(len)
help(print)

# Get information about objects
print(type(data))        # Type of object
print(dir(data))         # List available methods
print(vars())            # List all variables in current scope

# Package information
help(math)  # Help for math module
```

### Online Resources
- **Python Documentation**: [python.org/doc/](https://docs.python.org/3/)
- **Jupyter Documentation**: [jupyter.org](https://jupyter.org/)
- **Stack Overflow**: [stackoverflow.com/questions/tagged/python](https://stackoverflow.com/questions/tagged/python)
- **Real Python**: [realpython.com](https://realpython.com/)
- **PyPI (Python Package Index)**: [pypi.org](https://pypi.org/)

## Best Practices

### Code Style and Organization
```python
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
```

### File Organization and Project Management
- **Organize scripts**: Keep related Python scripts in organized folders
- **Meaningful names**: Use descriptive file names (e.g., `data_cleaning.py`, `analysis.py`)
- **Documentation**: Include comments and docstrings in your code
- **Version control**: Use Git for tracking changes
- **Regular saves**: Save your work frequently
- **Project structure**: Organize projects with clear folder structure

### Error Handling and Debugging
```python
# Check for errors and handle them gracefully
try:
    result = mean(non_existent_variable)
except Exception as e:
    print(f"Error occurred: {e}")

# Use print statements for debugging
x = 5
print(f"x = {x}")
print(f"x squared = {x ** 2}")
```

## Mathematical Concepts in Python

### Understanding Statistical Measures

**Mean (Arithmetic Average)**: The sum of all values divided by the number of values
```python
# Mathematical formula: μ = (Σxᵢ) / n
data = [2, 4, 6, 8, 10]
mean_value = sum(data) / len(data)  # Manual calculation
import statistics
mean_value_auto = statistics.mean(data)           # Python function
```

**Median**: The middle value when data is ordered
```python
# For odd number of values: middle value
# For even number of values: average of two middle values
data = [1, 3, 5, 7, 9]
import statistics
median_value = statistics.median(data)
```

**Variance**: Average squared deviation from the mean
```python
# Mathematical formula: σ² = Σ(xᵢ - μ)² / (n-1)
data = [2, 4, 6, 8, 10]
mean_data = sum(data) / len(data)
variance_manual = sum((x - mean_data) ** 2 for x in data) / (len(data) - 1)
import statistics
variance_auto = statistics.variance(data)
```

**Standard Deviation**: Square root of variance
```python
# Mathematical formula: σ = √σ²
import math
sd_manual = math.sqrt(statistics.variance(data))
sd_auto = statistics.stdev(data)
```

## Exercises

### Exercise 1: Basic Operations and Variables
Create variables for your age, height (in cm), and favorite color. Then perform some basic operations with the numeric variables.

```python
# Your solution here
age = 25
height = 175
favorite_color = "blue"

# Basic operations
age_in_months = age * 12
height_in_meters = height / 100
bmi = 70 / (height_in_meters ** 2)  # Assuming weight of 70kg
```

### Exercise 2: List Creation and Manipulation
Create lists for:
- Your last 5 test scores
- Names of 3 friends
- Whether you like different foods (True/False)

```python
# Your solution here
test_scores = [85, 92, 78, 96, 88]
friend_names = ["Alice", "Bob", "Charlie"]
food_preferences = [True, False, True, True, False]
```

### Exercise 3: Summary Statistics
Calculate the mean, median, and standard deviation of your test scores. Interpret what these values tell you about your performance.

```python
# Your solution here
import statistics
mean_score = statistics.mean(test_scores)
median_score = statistics.median(test_scores)
sd_score = statistics.stdev(test_scores)

# Interpretation
print("Mean score:", mean_score)
print("Median score:", median_score)
print("Standard deviation:", sd_score)
```

### Exercise 4: Data Types and Type Conversion
Create different types of data and practice converting between types.

```python
# Your solution here
numeric_var = 42
character_var = "42"
logical_var = True

# Type conversions
print(str(numeric_var))
print(int(character_var))
print(bool(1))  # 1 becomes True
print(bool(0))  # 0 becomes False
```

### Exercise 5: List Operations
Create two numeric lists and perform various operations on them.

```python
# Your solution here
vector1 = [1, 2, 3, 4, 5]
vector2 = [10, 20, 30, 40, 50]

# Operations
sum_vectors = [a + b for a, b in zip(vector1, vector2)]
product_vectors = [a * b for a, b in zip(vector1, vector2)]
mean_vector1 = sum(vector1) / len(vector1)
import statistics
sd_vector2 = statistics.stdev(vector2)
```

## Next Steps

In the next chapter, we'll explore more complex data structures like tuples, dictionaries, and sets, which are essential for data analysis in Python. We'll also learn about:

- **Tuples**: Immutable ordered collections
- **Dictionaries**: Key-value pairs for flexible data storage
- **Sets**: Unordered collections of unique elements
- **Importing Data**: Reading data from various file formats (CSV, Excel, etc.)
- **Data Manipulation**: Cleaning and transforming data with pandas

---

**Key Takeaways:**
- Python is a powerful, general-purpose language widely used for data analysis
- Use `=` for assignment
- Lists are fundamental data structures in Python
- Always use descriptive variable names and add comments
- Get help using `help()` and documentation
- Practice regularly to build proficiency
- Understanding data types is crucial for proper analysis
- Python performs element-wise operations using comprehensions or NumPy
- The built-in help system and online resources are comprehensive and useful 