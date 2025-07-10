# Programming Basics

This document summarizes basic programming concepts for statistical analysis, with examples in both R and Python.

## Control Flow

### If/Else

#### R
```r
x <- 5
if (x > 0) {
  print("Positive")
} else {
  print("Non-positive")
}
```

#### Python
```python
x = 5
if x > 0:
    print("Positive")
else:
    print("Non-positive")
```

## For Loops

#### R
```r
for (i in 1:5) {
  print(i)
}
```

#### Python
```python
for i in range(1, 6):
    print(i)
```

## While Loops

#### R
```r
i <- 1
while (i <= 5) {
  print(i)
  i <- i + 1
}
```

#### Python
```python
i = 1
while i <= 5:
    print(i)
    i += 1
```

## Functions

#### R
```r
add <- function(a, b) {
  return(a + b)
}
add(2, 3)
```

#### Python
```python
def add(a, b):
    return a + b
add(2, 3)
```

---

*For more details, see the [Data and Programming](https://book.stat420.org/data-and-programming.html) chapter from Applied Statistics with R.* 