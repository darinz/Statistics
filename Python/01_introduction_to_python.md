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

**📁 Code Reference**: See `first_python_session()` function in `01_introduction_to_python.py`

This section demonstrates:
- Printing "Hello, World!" - the traditional first program
- Basic arithmetic operations (addition, multiplication, division, exponentiation)
- Creating variables and performing operations
- Checking the type of objects

### Understanding Python as a Calculator

**📁 Code Reference**: See `python_as_calculator()` function in `01_introduction_to_python.py`

Python can perform all basic mathematical operations and much more:
- Basic arithmetic operations (addition, subtraction, multiplication, division, exponentiation)
- Mathematical functions (square root, logarithms, exponential, absolute value, rounding)
- Trigonometric functions (sine, cosine, tangent)

### Variables and Assignment

**📁 Code Reference**: See `variables_and_assignment()` function in `01_introduction_to_python.py`

Variables in Python are containers that store values. Understanding variable assignment is fundamental:
- Assignment using `=`
- Printing variables
- Checking variable types
- Variable management

## Data Types in Python

Understanding data types is crucial for statistical analysis. Python has several fundamental data types:

### Numeric

**📁 Code Reference**: See `numeric_data_types()` function in `01_introduction_to_python.py`

Numeric data represents numbers, including integers and floating-point numbers:
- Integer examples (age, counts)
- Decimal numbers (height, temperature)
- Type checking and conversion
- Mathematical constants

### String

**📁 Code Reference**: See `string_data_types()` function in `01_introduction_to_python.py`

String data represents text:
- String creation and assignment
- String properties and methods
- String operations (uppercase, lowercase, concatenation)
- Length and type checking

### Boolean

**📁 Code Reference**: See `boolean_data_types()` function in `01_introduction_to_python.py`

Boolean data represents True/False values:
- Boolean variables and expressions
- Logical operations (AND, OR, NOT)
- Comparison operators (greater than, equal to, not equal to)
- Conditional expressions

### List

**📁 Code Reference**: See `list_data_types()` function in `01_introduction_to_python.py`

Lists represent ordered collections of items (can be of mixed types):
- Creating lists with different data types
- List properties (unique values, length, frequency counts)
- Ordered categorical data with pandas Categorical

## Working with Lists

Lists are a fundamental data structure in Python. They are ordered, mutable, and can hold multiple values of any type.

### Creating Lists

**📁 Code Reference**: See `creating_lists()` function in `01_introduction_to_python.py`

Different ways to create lists:
- Numeric lists
- String lists
- Boolean lists
- Using range and list comprehensions
- Repeating elements

### List Operations

**📁 Code Reference**: See `list_operations()` function in `01_introduction_to_python.py`

Python performs operations element-wise using list comprehensions or with NumPy arrays:
- Element-wise operations with list comprehensions
- Operations between two lists (addition, multiplication, division)
- Logical operations on lists
- Mathematical transformations

### List Functions and Statistics

**📁 Code Reference**: See `list_statistics()` function in `01_introduction_to_python.py`

Statistical functions on lists:
- Basic summary statistics (length, sum, mean, median, min, max, range)
- Measures of variability (variance, standard deviation)
- Quantiles and percentiles
- Using both built-in functions and NumPy

### List Indexing and Slicing

**📁 Code Reference**: See `list_indexing()` function in `01_introduction_to_python.py`

Accessing and manipulating list elements:
- Indexing (0-based indexing in Python)
- Accessing multiple elements
- Logical indexing with list comprehensions
- Excluding elements and slicing

## Getting Help in Python

**📁 Code Reference**: See `getting_help()` function in `01_introduction_to_python.py`

Python has excellent built-in help system and extensive online resources:

### Built-in Help
- Getting help for functions with `help()`
- Object information with `type()` and `dir()`
- Variable scope with `vars()`
- Package information

### Online Resources
- **Python Documentation**: [python.org/doc/](https://docs.python.org/3/)
- **Jupyter Documentation**: [jupyter.org](https://jupyter.org/)
- **Stack Overflow**: [stackoverflow.com/questions/tagged/python](https://stackoverflow.com/questions/tagged/python)
- **Real Python**: [realpython.com](https://realpython.com/)
- **PyPI (Python Package Index)**: [pypi.org](https://pypi.org/)

## Best Practices

### Code Style and Organization

**📁 Code Reference**: See `code_style_examples()` function in `01_introduction_to_python.py`

Good coding practices:
- Using descriptive variable names
- Proper spacing around operators
- Adding comments to explain complex code
- Consistent indentation for control structures
- Organizing code into functions

### File Organization and Project Management
- **Organize scripts**: Keep related Python scripts in organized folders
- **Meaningful names**: Use descriptive file names (e.g., `data_cleaning.py`, `analysis.py`)
- **Documentation**: Include comments and docstrings in your code
- **Version control**: Use Git for tracking changes
- **Regular saves**: Save your work frequently
- **Project structure**: Organize projects with clear folder structure

### Error Handling and Debugging

**📁 Code Reference**: See `error_handling_examples()` function in `01_introduction_to_python.py`

Debugging techniques:
- Using try-except blocks for error handling
- Print statements for debugging
- Graceful error handling
- Understanding error messages

## Mathematical Concepts in Python

**📁 Code Reference**: See `mathematical_concepts()` function in `01_introduction_to_python.py`

### Understanding Statistical Measures

**Mean (Arithmetic Average)**: The sum of all values divided by the number of values
- Mathematical formula: μ = (Σxᵢ) / n
- Manual calculation vs. Python functions

**Median**: The middle value when data is ordered
- For odd number of values: middle value
- For even number of values: average of two middle values

**Variance**: Average squared deviation from the mean
- Mathematical formula: σ² = Σ(xᵢ - μ)² / (n-1)
- Manual calculation vs. Python functions

**Standard Deviation**: Square root of variance
- Mathematical formula: σ = √σ²
- Manual calculation vs. Python functions

## Exercises

### Exercise 1: Basic Operations and Variables

**📁 Code Reference**: See `exercise_1_basic_operations()` function in `01_introduction_to_python.py`

Create variables for your age, height (in cm), and favorite color. Then perform some basic operations with the numeric variables:
- Convert age to months
- Convert height to meters
- Calculate BMI (assuming a weight)

### Exercise 2: List Creation and Manipulation

**📁 Code Reference**: See `exercise_2_list_creation()` function in `01_introduction_to_python.py`

Create lists for:
- Your last 5 test scores
- Names of 3 friends
- Whether you like different foods (True/False)

### Exercise 3: Summary Statistics

**📁 Code Reference**: See `exercise_3_summary_statistics()` function in `01_introduction_to_python.py`

Calculate the mean, median, and standard deviation of your test scores. Interpret what these values tell you about your performance.

### Exercise 4: Data Types and Type Conversion

**📁 Code Reference**: See `exercise_4_data_types()` function in `01_introduction_to_python.py`

Create different types of data and practice converting between types:
- Converting numbers to strings
- Converting strings to numbers
- Converting numbers to booleans

### Exercise 5: List Operations

**📁 Code Reference**: See `exercise_5_list_operations()` function in `01_introduction_to_python.py`

Create two numeric lists and perform various operations on them:
- Element-wise addition and multiplication
- Calculating mean and standard deviation
- Working with multiple vectors

## Running the Code Examples

To run all the code examples in this lesson:

1. **Run the entire file**: Execute `python 01_introduction_to_python.py` in your terminal
2. **Run individual sections**: In a Python environment, import and call specific functions:
   ```python
   from introduction_to_python import first_python_session, python_as_calculator
   first_python_session()
   python_as_calculator()
   ```
3. **Interactive learning**: Copy individual functions into Jupyter notebooks for interactive exploration

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
- All code examples are available in the companion Python file for hands-on practice 