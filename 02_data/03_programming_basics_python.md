# Python Programming Basics

This document summarizes basic programming concepts in Python for statistical analysis and data science.

## Control Flow

### If/Else Statements

```python
x = 5
if x > 0:
    print("Positive")
elif x == 0:
    print("Zero")
else:
    print("Negative")

# Ternary operator (conditional expression)
result = "Positive" if x > 0 else "Non-positive"
```

## Loops

### For Loops

```python
# Basic for loop with range
for i in range(1, 6):
    print(i)

# For loop with list
fruits = ['apple', 'banana', 'orange']
for fruit in fruits:
    print(fruit)

# For loop with enumerate
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# List comprehension (Pythonic way)
squares = [x**2 for x in range(1, 6)]
```

### While Loops

```python
# Basic while loop
i = 1
while i <= 5:
    print(i)
    i += 1

# While loop with break
count = 0
while True:
    count += 1
    if count > 5:
        break
    print(count)

# While loop with continue
i = 0
while i < 10:
    i += 1
    if i % 2 == 0:
        continue
    print(i)
```

## Functions

### Basic Functions

```python
# Simple function
def add(a, b):
    return a + b

# Function with default arguments
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Function with multiple return values
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)/len(numbers)

# Lambda functions (anonymous functions)
square = lambda x: x**2
```

### Function Examples

```python
# Statistical function
def calculate_mean(numbers):
    if not numbers:
        return None
    return sum(numbers) / len(numbers)

# Data processing function
def filter_data(data, threshold):
    return [x for x in data if x > threshold]

# Function with type hints
def multiply(a: float, b: float) -> float:
    return a * b
```

## Error Handling

### Try/Except

```python
# Basic error handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

# Multiple exception types
try:
    value = int("abc")
except ValueError:
    print("Invalid number")
except Exception as e:
    print(f"An error occurred: {e}")
```

## Best Practices

1. **Use descriptive function names**
2. **Write docstrings for functions**
3. **Use type hints for better code documentation**
4. **Handle exceptions appropriately**
5. **Use list comprehensions when appropriate**
6. **Follow PEP 8 style guidelines**

---

*This content is adapted for Python data science and statistical analysis.* 