# Data Structures

This document summarizes the main data structures used in statistical programming, with examples in both R and Python.

## Vectors (R) / Lists (Python)

### R
- **Vector**: Ordered collection of elements of the same type

```r
numbers <- c(1, 2, 3, 4, 5)
letters <- c("a", "b", "c")
```

### Python
- **List**: Ordered collection, can contain mixed types

```python
numbers = [1, 2, 3, 4, 5]
letters = ['a', 'b', 'c']
```

## Matrices

### R
- **Matrix**: 2D array of elements of the same type

```r
mat <- matrix(1:6, nrow=2, ncol=3)
```

### Python
- **numpy.ndarray**: 2D array (usually with numpy)

```python
import numpy as np
mat = np.array([[1, 2, 3], [4, 5, 6]])
```

## Lists (R) / Dictionaries (Python)

### R
- **List**: Collection of elements of different types

```r
person <- list(name="Alice", age=30, scores=c(90, 95, 88))
```

### Python
- **dict**: Key-value pairs

```python
person = {'name': 'Alice', 'age': 30, 'scores': [90, 95, 88]}
```

## Data Frames

### R
- **Data Frame**: Table-like structure with columns of different types

```r
students <- data.frame(name=c("Alice", "Bob"), age=c(20, 21))
```

### Python
- **pandas.DataFrame**: Table-like structure

```python
import pandas as pd
students = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [20, 21]})
```

---

*For more details, see the [Data and Programming](https://book.stat420.org/data-and-programming.html) chapter from Applied Statistics with R.* 