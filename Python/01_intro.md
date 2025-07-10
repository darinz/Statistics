# Introduction to Python

## What is Python?

Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum in 1991, Python has become one of the most popular languages for data science, machine learning, and statistical analysis.

### Key Features of Python

- **Readability**: Clean, simple syntax that's easy to learn and understand
- **Versatility**: Used for web development, data science, AI, automation, and more
- **Rich Ecosystem**: Extensive libraries for data analysis (pandas, numpy, scipy)
- **Statistical Computing**: Powerful packages for statistical analysis and modeling
- **Machine Learning**: Excellent libraries like scikit-learn, TensorFlow, and PyTorch
- **Open Source**: Free to use, modify, and distribute

## Getting Started with Python

### Installing Python and Jupyter

1. **Install Python**: Download from [python.org](https://www.python.org/)
2. **Install Jupyter**: Install via pip or conda

Jupyter Notebook is an interactive computing environment that makes working with Python much easier. It provides:
- Interactive code cells
- Rich text documentation with Markdown
- Inline plots and visualizations
- Easy sharing and collaboration
- Support for multiple programming languages

### Your First Python Session

Let's start with some basic operations:

```python
# This is a comment
# Basic arithmetic
2 + 3
5 * 4
10 / 2

# Creating variables
x = 5
y = 10
z = x + y
print(z)

# Lists (basic data structure)
numbers = [1, 2, 3, 4, 5]
import numpy as np
print(np.mean(numbers))
print(np.std(numbers))
```

## Python as a Calculator

Python can perform all basic mathematical operations:

```python
# Addition
3 + 4

# Subtraction
10 - 5

# Multiplication
6 * 7

# Division
20 / 4

# Exponentiation
2 ** 3

# Square root
import math
math.sqrt(16)

# Natural logarithm
math.log(10)

# Exponential
math.exp(1)
```

## Variables and Assignment

In Python, you assign values to variables using `=`:

```python
# Assignment
x = 5
y = 10

# Variable names can contain letters, numbers, and underscores
my_variable = 42
data_2023 = "some value"

# Check what variables exist (in interactive mode)
# Use dir() or locals() to see variables

# Remove a variable
del x
```

## Data Types in Python

Python has several basic data types:

### Numeric
```python
# Integer and float
age = 25        # integer
height = 175.5  # float
```

### String
```python
name = "John"
city = 'New York'
```

### Boolean
```python
is_student = True
is_working = False
```

### List (for collections)
```python
numbers = [1, 2, 3, 4, 5]
names = ["Alice", "Bob", "Charlie"]
```

## Data Structures

### Lists
Lists are one of the most basic data structures in Python:

```python
# Numeric list
numbers = [1, 2, 3, 4, 5]

# String list
names = ["Alice", "Bob", "Charlie"]

# Boolean list
flags = [True, False, True]

# List operations
[2 * x for x in numbers]
[x + 10 for x in numbers]
```

### Dictionaries
Dictionaries store key-value pairs:

```python
# Create a dictionary
person = {
    "name": "John",
    "age": 30,
    "scores": [85, 92, 78],
    "is_student": True
}

# Access dictionary elements
print(person["name"])
print(person.get("age"))
```

### Pandas DataFrames
DataFrames are like tables with rows and columns:

```python
import pandas as pd

# Create a DataFrame
students = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [20, 22, 21],
    'grade': [85, 92, 78]
})

# View the DataFrame
print(students)
print(students.info())
print(students.head())
```

## Working with Data

### Reading Data
Python can read data from various formats using pandas:

```python
import pandas as pd

# Read CSV file
data = pd.read_csv("filename.csv")

# Read Excel file
data = pd.read_excel("filename.xlsx")

# Read text file
data = pd.read_table("filename.txt", sep='\t')
```

### Basic Data Exploration

```python
import pandas as pd
import numpy as np

# Load sample dataset
from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# View first few rows
print(data.head())

# Get dimensions
print(data.shape)

# Get column names
print(data.columns)

# Get data types
print(data.dtypes)

# Summary statistics
print(data.describe())

# Get info about the dataset
print(data.info())
```

## Basic Graphics

Python has excellent plotting capabilities with matplotlib and seaborn:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Sample data
np.random.seed(42)
data = pd.DataFrame({
    'weight': np.random.normal(3, 0.5, 100),
    'mpg': np.random.normal(25, 5, 100),
    'cylinders': np.random.choice([4, 6, 8], 100)
})

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data['weight'], data['mpg'])
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('MPG vs Weight')
plt.show()

# Histogram
plt.figure(figsize=(8, 6))
plt.hist(data['mpg'], bins=20, alpha=0.7)
plt.xlabel('MPG')
plt.ylabel('Frequency')
plt.title('Distribution of MPG')
plt.show()

# Box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='cylinders', y='mpg', data=data)
plt.xlabel('Number of Cylinders')
plt.ylabel('MPG')
plt.title('MPG by Number of Cylinders')
plt.show()
```

## Getting Help

Python has extensive help documentation:

```python
# Get help on a function
help(len)
help(pd.DataFrame)

# Get help on an object
import pandas as pd
df = pd.DataFrame()
df.info?

# Search for functions (in IPython/Jupyter)
# Use ? or ?? for help
# len?
# pd.DataFrame??

# Get help on a module
help(pandas)

# Get examples
import matplotlib.pyplot as plt
plt.plot?
```

## Best Practices

1. **Use meaningful variable names**
2. **Comment your code**
3. **Use spaces around operators**
4. **Keep lines under 80 characters**
5. **Use consistent indentation**
6. **Save your work regularly**

## Next Steps

After mastering these basics, you'll be ready to:
- Learn about pandas, numpy, and scipy for data manipulation
- Perform statistical analyses with scipy and statsmodels
- Create reproducible reports with Jupyter notebooks
- Build machine learning models with scikit-learn
- Work with larger datasets efficiently
- Create interactive visualizations with plotly

## Resources

- [Python Documentation](https://docs.python.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Jupyter Notebook Documentation](https://jupyter.org/)
- [Python for Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

---

*This content is adapted for learning Python for statistical analysis and enhanced for educational purposes.* 