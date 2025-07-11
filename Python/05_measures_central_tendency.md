# Measures of Central Tendency

## Overview

Measures of central tendency describe the center or typical value of a dataset. They help us understand what the "average" or "typical" value is in our data. These measures are fundamental to statistical analysis and provide the foundation for understanding data distributions.

### Why Central Tendency Matters

- **Data Summarization**: Provides a single value that represents the entire dataset
- **Comparison**: Allows comparison between different groups or datasets
- **Decision Making**: Helps in making informed decisions based on typical values
- **Statistical Inference**: Forms the basis for more advanced statistical methods
- **Communication**: Simplifies complex data for stakeholders

### Types of Central Tendency Measures

1. **Mean**: Arithmetic average, most common measure
2. **Median**: Middle value, robust to outliers
3. **Mode**: Most frequent value, useful for categorical data
4. **Geometric Mean**: For rates and ratios
5. **Harmonic Mean**: For rates and speeds
6. **Trimmed Mean**: Robust version of mean
7. **Winsorized Mean**: Another robust alternative

## Mean (Arithmetic Mean)

The arithmetic mean is the sum of all values divided by the number of values. It's the most commonly used measure of central tendency.

### Mathematical Foundation

**Population Mean**:
```math
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i
```

**Sample Mean**:
```math
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
```

**Properties of the Mean**:
- The sum of deviations from the mean equals zero: $\sum_{i=1}^{n} (x_i - \bar{x}) = 0$
- The mean minimizes the sum of squared deviations: $\sum_{i=1}^{n} (x_i - \bar{x})^2$ is minimized
- The mean is affected by every value in the dataset
- The mean is sensitive to outliers

### Basic Mean Calculation

See the function `basic_mean_calculation()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to demonstrate basic mean calculation with manual verification, handling missing values, and understanding the effect of outliers.

### Mean by Group

See the function `mean_by_group()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to calculate and visualize mean by group using the mtcars dataset, including standard error calculations and bar plots with error bars.

### Weighted Mean

The weighted mean is useful when different observations have different importance or reliability.

**Mathematical Formula**:
```math
\bar{x}_w = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}
```

See the function `weighted_mean_example()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to demonstrate weighted mean calculation with manual verification and course grades example.

## Median

The median is the middle value when data is ordered from smallest to largest. It's robust to outliers and represents the 50th percentile.

### Mathematical Foundation

**For odd number of observations**:
```math
\text{Median} = x_{\left(\frac{n+1}{2}\right)}
```

**For even number of observations**:
```math
\text{Median} = \frac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)}}{2}
```

**Properties of the Median**:
- Resistant to outliers
- Minimizes the sum of absolute deviations: $\sum_{i=1}^{n} |x_i - \text{median}|$ is minimized
- Represents the 50th percentile
- Not affected by extreme values

### Basic Median Calculation

See the function `basic_median_calculation()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to demonstrate basic median calculation with manual verification for odd and even sample sizes, and handling missing values.

### Median by Group

See the function `median_by_group()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to calculate median by group and compare mean vs median for the mtcars dataset, including visualizations.

## Mode

The mode is the most frequently occurring value in a dataset. It's the only measure of central tendency that can be used with categorical data.

### Mathematical Foundation

**Mode Definition**:
```math
\text{Mode} = \arg\max_{x} f(x)
```

where $f(x)$ is the frequency of value $x$.

**Properties of the Mode**:
- Can be used with any data type (numeric, categorical, ordinal)
- May not be unique (bimodal, multimodal)
- May not exist (uniform distribution)
- Not affected by extreme values
- Represents the most common value

### Finding the Mode

See the functions `find_mode()`, `find_all_modes()`, and `mode_examples()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to find the mode, handle multimodal data, and demonstrate mode calculation with various examples.

### Mode for Categorical Data

See the function `mode_categorical_data()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to demonstrate mode calculation for categorical data with frequency tables and visualizations.

## Geometric Mean

The geometric mean is useful for data that represents rates of change, ratios, or multiplicative relationships.

### Mathematical Foundation

**Geometric Mean Formula**:
```math
GM = \sqrt[n]{x_1 \times x_2 \times \cdots \times x_n} = \left(\prod_{i=1}^{n} x_i\right)^{\frac{1}{n}}
```

**Logarithmic Form**:
```math
GM = \exp\left(\frac{1}{n}\sum_{i=1}^{n} \ln(x_i)\right)
```

**Properties of Geometric Mean**:
- Always less than or equal to arithmetic mean
- Useful for rates of change and ratios
- Sensitive to zeros and negative values
- Multiplicative equivalent of arithmetic mean

See the functions `geometric_mean()` and `geometric_mean_examples()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to calculate geometric mean with growth rates and investment returns examples.

## Harmonic Mean

The harmonic mean is useful for rates, speeds, and situations involving reciprocals.

### Mathematical Foundation

**Harmonic Mean Formula**:
```math
HM = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}
```

**Properties of Harmonic Mean**:
- Always less than or equal to geometric mean
- Useful for rates and speeds
- Sensitive to small values
- Reciprocal of arithmetic mean of reciprocals

See the functions `harmonic_mean()` and `harmonic_mean_examples()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to calculate harmonic mean with speeds and parallel resistors examples.

## Trimmed Mean

The trimmed mean removes a percentage of extreme values before calculating the mean, making it more robust to outliers.

### Mathematical Foundation

**Trimmed Mean Formula**:
```math
\bar{x}_{\alpha} = \frac{1}{n(1-2\alpha)}\sum_{i=\alpha n + 1}^{(1-\alpha)n} x_{(i)}
```

where $\alpha$ is the trimming proportion and $x_{(i)}$ are the ordered values.

**Properties of Trimmed Mean**:
- More robust than arithmetic mean
- Less sensitive to outliers
- Maintains some efficiency for normal data
- Common trimming levels: 5%, 10%, 20%

See the function `trimmed_mean_examples()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to demonstrate trimmed mean calculation with different trimming levels and manual verification.

## Winsorized Mean

The winsorized mean replaces extreme values with less extreme values rather than removing them.

### Mathematical Foundation

**Winsorized Mean Formula**:
```math
\bar{x}_w = \frac{1}{n}\left(k \cdot x_{(k+1)} + \sum_{i=k+1}^{n-k} x_{(i)} + k \cdot x_{(n-k)}\right)
```

where $k$ is the number of values to be winsorized from each end.

See the functions `winsorized_mean()` and `winsorized_mean_examples()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to calculate winsorized mean and compare with other robust measures.

## Comparing Measures of Central Tendency

See the function `compare_all_measures()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to compare all measures of central tendency for a dataset with outliers.

## Mathematical Relationships

### Inequality Relationships

For positive data, the following relationship holds:
```math
\text{Harmonic Mean} \leq \text{Geometric Mean} \leq \text{Arithmetic Mean}
```

This is known as the **Arithmetic Mean-Geometric Mean-Harmonic Mean Inequality**.

See the function `demonstrate_inequality()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to demonstrate this inequality with examples.

## Practical Examples

### Example 1: Student Test Scores

See the function `student_test_scores_example()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to analyze student test scores with various central tendency measures, including visualizations and grade level analysis.

### Example 2: Income Data

See the function `income_data_example()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to analyze income data (right-skewed) with various central tendency measures, including percentiles and skewness indicators.

### Example 3: Temperature Data

See the function `temperature_data_example()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to analyze temperature data with various central tendency measures, including seasonal analysis and temperature range calculations.

## Using Built-in Functions

See the function `central_tendency_summary()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to create comprehensive summaries using built-in functions and robust summary statistics.

## Best Practices

### When to Use Each Measure

See the function `choose_central_tendency()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code implementing a decision tree to help choose the appropriate measure of central tendency based on data characteristics.

### Handling Missing Data

See the function `robust_central_tendency()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for code to handle missing data and calculate robust central tendency measures.

## Exercises

### Exercise 1: Basic Calculations
Create a dataset with 10 values and calculate the mean, median, and mode. Compare the results.

See the function `basic_mean_calculation()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for guidance on basic calculations.

### Exercise 2: Outlier Impact
Create a dataset, then add an outlier and observe how it affects the mean vs. median.

See the function `compare_all_measures()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for examples of outlier impact analysis.

### Exercise 3: Group Analysis
Using the `iris` dataset, calculate the mean and median of sepal length by species.

See the function `median_by_group()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for guidance on group analysis.

### Exercise 4: Robust Measures
Create a dataset with outliers and compare the regular mean, trimmed mean, and winsorized mean.

See the functions `trimmed_mean_examples()` and `winsorized_mean_examples()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for robust measure comparisons.

### Exercise 5: Real-world Application
Find a real dataset (or use built-in data) and perform a comprehensive central tendency analysis.

See the functions `student_test_scores_example()`, `income_data_example()`, and `temperature_data_example()` in [05_measures_central_tendency.py](05_measures_central_tendency.py) for real-world application examples.

## Next Steps

In the next chapter, we'll learn about measures of variability, which complement central tendency measures to give us a complete picture of our data. We'll cover:

- **Range and Interquartile Range**: Measures of spread
- **Variance and Standard Deviation**: Most common measures of variability
- **Coefficient of Variation**: Relative measure of variability
- **Skewness and Kurtosis**: Measures of distribution shape
- **Robust Measures of Variability**: Median absolute deviation

---

**Key Takeaways:**
- Mean is sensitive to outliers, median is robust
- Mode is best for categorical data
- Geometric mean for rates and ratios
- Harmonic mean for rates and speeds
- Always consider the data type and distribution
- Use multiple measures for comprehensive analysis
- Handle missing data appropriately
- Consider the context when choosing measures
- Outliers can significantly affect the mean
- Different measures serve different purposes 