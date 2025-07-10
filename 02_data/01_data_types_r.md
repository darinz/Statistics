# R Data Types

This document summarizes the main data types used in R for statistical programming and data analysis.

## Numeric Types

### Numeric (double)
- **numeric**: Numbers with or without decimals (default numeric type in R)

```r
x <- 3      # numeric (double)
y <- 2.71   # numeric (double)
z <- -5.5   # numeric (double)
```

### Integer
- **integer**: Whole numbers (use L suffix)

```r
a <- 5L     # integer
b <- -10L   # integer
c <- 1000L  # integer
```

## Character Types

### Character
- **character**: Text values, defined with quotes

```r
name <- "Alice"
city <- 'New York'
message <- "Hello, World!"
```

## Logical Types

### Logical
- **logical**: TRUE or FALSE (case sensitive)

```r
is_valid <- TRUE
is_empty <- FALSE
x <- 5 > 3    # TRUE
y <- 5 == 3   # FALSE
```

## Factor Types

### Factor
- **factor**: Used for categorical data (e.g., gender, group, treatment)

```r
gender <- factor(c("Male", "Female", "Male"))
levels(gender)

# Create factor with specific levels
treatment <- factor(c("Control", "Treatment", "Control"), 
                   levels = c("Control", "Treatment"))
```

## Complex Types

### Complex
- **complex**: Numbers with real and imaginary parts

```r
z <- 3 + 4i
w <- complex(real = 2, imaginary = 5)
```

## Raw Types

### Raw
- **raw**: Raw bytes

```r
raw_data <- raw(10)
```

## Type Checking and Conversion

### Checking Types
```r
x <- 5
class(x)        # "numeric"
is.numeric(x)   # TRUE
is.integer(x)   # FALSE
```

### Type Conversion
```r
# String to number
x <- "123"
y <- as.numeric(x)    # 123

# Number to string
num <- 42
text <- as.character(num)  # "42"

# Character to factor
colors <- c("red", "blue", "red")
color_factor <- as.factor(colors)
```

## Best Practices

1. **Use meaningful variable names**
2. **Be consistent with data types**
3. **Use factors for categorical variables**
4. **Check data types before analysis**
5. **Use appropriate data types for memory efficiency**

---

*This content is adapted for R statistical analysis and data science.* 