# Python Data Structures

This document summarizes the main data structures used in Python for statistical programming and data analysis.

## Lists

### List
- **list**: Ordered, mutable collection of items

```python
# Basic list
numbers = [1, 2, 3, 4, 5]
letters = ['a', 'b', 'c']

# Mixed types
mixed = [1, "hello", 3.14, True]

# Nested lists
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# List operations
numbers.append(6)      # Add element
numbers.remove(3)      # Remove element
len(numbers)           # Get length
```

## Arrays and Matrices

### NumPy Arrays
- **numpy.ndarray**: Multi-dimensional arrays for numerical computing

```python
import numpy as np

# 1D array
arr1d = np.array([1, 2, 3, 4, 5])

# 2D array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Array operations
print(matrix.shape)     # (3, 3)
print(matrix.ndim)      # 2
print(matrix.size)      # 9

# Mathematical operations
print(matrix + 1)       # Add 1 to all elements
print(matrix * 2)       # Multiply all elements by 2
```

## Dictionaries

### Dictionary (dict)
- **dict**: Key-value pairs for storing structured data

```python
# Basic dictionary
person = {'name': 'Alice', 'age': 30, 'city': 'NYC'}

# Nested dictionary
student = {
    'name': 'Bob',
    'grades': {'math': 95, 'science': 88, 'english': 92},
    'activities': ['soccer', 'chess']
}

# Dictionary operations
person['age'] = 31              # Update value
person['email'] = 'alice@email.com'  # Add new key
del person['city']              # Remove key
'name' in person               # Check if key exists
```

## DataFrames

### Pandas DataFrame
- **pandas.DataFrame**: Table-like structure for data analysis

```python
import pandas as pd
import numpy as np

# Create DataFrame
students = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [20, 21, 19],
    'grade': [95, 88, 92]
})

# DataFrame operations
print(students.head())           # View first few rows
print(students.shape)           # (rows, columns)
print(students.columns)         # Column names
print(students.dtypes)          # Data types

# Accessing data
print(students['name'])         # Single column
print(students.iloc[0])         # First row
print(students.loc[0, 'name'])  # Specific cell
```

## Sets

### Set
- **set**: Unordered collection of unique elements

```python
# Basic set
fruits = {'apple', 'banana', 'orange'}

# Set operations
fruits.add('grape')             # Add element
fruits.remove('banana')         # Remove element
'apple' in fruits              # Check membership

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
print(set1.union(set2))        # {1, 2, 3, 4, 5, 6}
print(set1.intersection(set2)) # {3, 4}
```

## Tuples

### Tuple
- **tuple**: Immutable ordered collection

```python
# Basic tuple
coordinates = (10, 20)
person = ('Alice', 25, 'Engineer')

# Tuple unpacking
name, age, job = person

# Named tuples (from collections)
from collections import namedtuple
Person = namedtuple('Person', ['name', 'age', 'city'])
alice = Person('Alice', 25, 'NYC')
```

## Best Practices

1. **Choose the right data structure for your use case**
2. **Use lists for ordered, mutable collections**
3. **Use dictionaries for key-value mappings**
4. **Use NumPy arrays for numerical computations**
5. **Use pandas DataFrames for tabular data**
6. **Use sets for unique collections**
7. **Use tuples for immutable data**

---

*This content is adapted for Python data science and statistical analysis.* 