# Python Data Types

This document summarizes the main data types used in Python for statistical programming and data analysis.

## Numeric Types

### Integer (int)
- **int**: Whole numbers (e.g., 3, -5, 1000)

```python
x = 3      # positive integer
y = -5     # negative integer
z = 1000   # large integer
```

### Float
- **float**: Numbers with decimals (e.g., 3.14, -2.5)

```python
x = 3.14   # positive float
y = -2.5   # negative float
z = 2.0    # float with decimal
```

### Complex
- **complex**: Complex numbers with real and imaginary parts

```python
x = 3 + 4j  # complex number
y = complex(3, 4)  # same as above
```

## String Types

### String (str)
- **str**: Text values, defined with single or double quotes

```python
name = "Alice"           # double quotes
city = 'New York'        # single quotes
message = """Multi-line
string with triple quotes"""
```

## Boolean Types

### Boolean (bool)
- **bool**: True or False (case sensitive)

```python
is_valid = True
is_empty = False
x = 5 > 3    # True
y = 5 == 3   # False
```

## Sequence Types

### List
- **list**: Ordered, mutable collection of items

```python
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
nested = [[1, 2], [3, 4]]
```

### Tuple
- **tuple**: Ordered, immutable collection of items

```python
coordinates = (10, 20)
person = ("Alice", 25, "Engineer")
```

### Range
- **range**: Immutable sequence of numbers

```python
numbers = range(5)        # 0, 1, 2, 3, 4
evens = range(0, 10, 2)  # 0, 2, 4, 6, 8
```

## Mapping Types

### Dictionary (dict)
- **dict**: Key-value pairs

```python
person = {"name": "Alice", "age": 25, "city": "NYC"}
scores = {"math": 95, "science": 88, "english": 92}
```

## Set Types

### Set
- **set**: Unordered collection of unique elements

```python
unique_numbers = {1, 2, 3, 4, 5}
fruits = {"apple", "banana", "orange"}
```

## Categorical Data

### Pandas Categorical
- **pandas.Categorical**: For categorical data analysis

```python
import pandas as pd
gender = pd.Categorical(["Male", "Female", "Male"])
print(gender.categories)
```

## Type Checking and Conversion

### Checking Types
```python
x = 5
print(type(x))        # <class 'int'>
print(isinstance(x, int))  # True
```

### Type Conversion
```python
# String to number
x = "123"
y = int(x)    # 123
z = float(x)  # 123.0

# Number to string
num = 42
text = str(num)  # "42"

# List to tuple
my_list = [1, 2, 3]
my_tuple = tuple(my_list)  # (1, 2, 3)
```

## Best Practices

1. **Use descriptive variable names**
2. **Choose appropriate data types for your use case**
3. **Use type hints for better code documentation**
4. **Be consistent with naming conventions**
5. **Consider memory usage for large datasets**

---

*This content is adapted for Python data science and statistical analysis.* 