# Data Import and Manipulation

## Overview

Data import and manipulation are fundamental skills in Python. Most real-world data analysis begins with importing data from external sources and cleaning it for analysis. This process is crucial because the quality of your analysis depends directly on the quality of your data.

### The Data Analysis Pipeline

1. **Data Import**: Reading data from various sources
2. **Data Inspection**: Understanding structure and quality
3. **Data Cleaning**: Handling missing values, outliers, and inconsistencies
4. **Data Transformation**: Reshaping and restructuring for analysis
5. **Data Validation**: Ensuring data integrity and consistency

### Why Data Import and Manipulation Matter

- **Data Quality**: Real-world data is often messy and requires cleaning
- **Format Compatibility**: Different sources provide data in various formats
- **Analysis Requirements**: Statistical analysis often requires specific data structures
- **Reproducibility**: Proper data handling ensures reproducible results
- **Efficiency**: Well-structured data enables faster and more accurate analysis

## Reading Data Files

### Understanding File Formats

Different file formats have different characteristics:

| Format | Extension | Pros | Cons | Best For |
|--------|-----------|------|------|----------|
| CSV | .csv | Simple, universal | No data types | Tabular data |
| Excel | .xlsx, .xls | Rich formatting | Proprietary | Complex tables |
| JSON | .json | Hierarchical | Complex parsing | Web APIs |
| XML | .xml | Structured | Verbose | Complex data |
| SPSS | .sav | Statistical | Proprietary | Survey data |
| SAS | .sas7bdat | Statistical | Proprietary | Clinical data |

### CSV Files

CSV (Comma-Separated Values) files are the most common format for tabular data:

```python
import pandas as pd

# Basic CSV reading
data = pd.read_csv("data.csv")

# Read CSV with specific options for better control
data = pd.read_csv(
    "data.csv",
    header=0,                # First row as column names
    sep=",",                # Comma separator
    na_values=["", "NA", "N/A", "NULL"],  # Missing value indicators
    encoding="utf-8",       # Character encoding
    quotechar='"',          # Quote character
    comment="#"             # Comment character
)

# Example with different separators
data_tsv = pd.read_csv("data.tsv", sep="\t")
data_semicolon = pd.read_csv("data.csv", sep=";")

# Write CSV file
data.to_csv("output.csv", index=False)

# Write with specific options
data.to_csv("output.csv", index=False, na_rep="", encoding="utf-8")
```

### Excel Files

Excel files can contain multiple sheets and complex formatting:

```python
# Install openpyxl if needed: pip install openpyxl
import pandas as pd

# Read Excel file (first sheet by default)
data = pd.read_excel("data.xlsx")

# Read specific sheet by name
data = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# Read specific sheet by position
data = pd.read_excel("data.xlsx", sheet_name=1)

# Read specific range (use nrows and usecols)
data = pd.read_excel("data.xlsx", usecols="A:D", nrows=10)

# List all sheet names
sheet_names = pd.ExcelFile("data.xlsx").sheet_names

# Read multiple sheets
all_sheets = pd.read_excel("data.xlsx", sheet_name=None)  # Returns dict of DataFrames

# Write Excel file
data.to_excel("output.xlsx", index=False)

# Write multiple sheets
with pd.ExcelWriter("output.xlsx") as writer:
    data1.to_excel(writer, sheet_name="Sheet1", index=False)
    data2.to_excel(writer, sheet_name="Sheet2", index=False)
```

### Other File Formats

```python
import pandas as pd

# SPSS files (Statistical Package for Social Sciences)
# pip install pyreadstat
spss_data, meta = pd.read_spss("data.sav", usecols=None, convert_categoricals=True, return_meta=True)

# SAS files (Statistical Analysis System)
sas_data = pd.read_sas("data.sas7bdat")

# Stata files
stata_data = pd.read_stata("data.dta")

# JSON files (JavaScript Object Notation)
import json
with open("data.json") as f:
    json_data = json.load(f)
# Or as DataFrame
df_json = pd.read_json("data.json")

# Read JSON from URL
df_json_url = pd.read_json("https://api.example.com/data")

# XML files (eXtensible Markup Language)
# pip install lxml
import xml.etree.ElementTree as ET
tree = ET.parse("data.xml")
root = tree.getroot()

# RDS and RData files are R-specific; in Python, use pickle for serialization
import pickle
# Save
with open("output.pkl", "wb") as f:
    pickle.dump(data, f)
# Load
with open("output.pkl", "rb") as f:
    data = pickle.load(f)
```

## Built-in Datasets

Python does not have as many built-in datasets as R, but several libraries provide sample datasets for practice and learning:

```python
from sklearn import datasets
import seaborn as sns

# Load iris dataset (as pandas DataFrame)
iris = sns.load_dataset("iris")
print(iris.head())

# Load other datasets
mtcars = sns.load_dataset("mpg")  # Similar to mtcars
print(mtcars.head())

# List available seaborn datasets
print(sns.get_dataset_names())

# scikit-learn datasets (as numpy arrays or Bunch objects)
boston = datasets.load_boston()
print(boston.data.shape)

# Popular built-in datasets
# iris - Fisher's Iris Data
# mpg - Car fuel economy data (similar to mtcars)
# tips - Restaurant tips data
# titanic - Titanic passenger data
```

## Data Inspection and Cleaning

### Understanding Data Quality Issues

Common data quality problems include:
- **Missing Values**: np.nan, None, or placeholder values
- **Inconsistent Formats**: Mixed date formats, inconsistent text
- **Outliers**: Extreme values that may be errors
- **Duplicates**: Repeated observations
- **Incorrect Data Types**: Numbers stored as text, dates as text
- **Inconsistent Naming**: Same values with different spellings

### Basic Data Inspection

```python
import pandas as pd
import numpy as np

# Load a dataset
import seaborn as sns
mtcars = sns.load_dataset("mpg")

# Basic information
print(mtcars.shape)           # Dimensions (rows, columns)
print(mtcars.columns)         # Column names
print(mtcars.info())          # Structure (data types and non-null counts)
print(mtcars.head())          # First 5 rows
print(mtcars.tail())          # Last 5 rows
print(mtcars.describe())      # Summary statistics

# Detailed inspection
print(type(mtcars))           # Object class
print(mtcars.dtypes)          # Data types of each column
print(mtcars.index)           # Index info
print(mtcars.values)          # Underlying numpy array

# Check for missing values
print(mtcars.isna())          # DataFrame of missing values
print(mtcars.isna().sum())    # Count of missing values per column
print(mtcars.isna().any().any()) # True if any missing values exist
print(mtcars.isna().sum().sum()) # Total number of missing values

# Check for duplicates
print(mtcars.duplicated())     # Boolean Series of duplicated rows
print(mtcars.duplicated().sum()) # Number of duplicate rows

# Check data types
print(mtcars.dtypes)           # Data types of each column

# Check for outliers (using IQR method)
def outlier_detection(x):
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (x < lower_bound) | (x > upper_bound)

# Apply to numeric columns
numeric_cols = mtcars.select_dtypes(include=[np.number])
outliers = numeric_cols.apply(outlier_detection)
```

### Data Cleaning Techniques

```python
# Remove rows with missing values
clean_data = mtcars.dropna()

# Remove rows with missing values in specific columns
clean_data = mtcars[mtcars["mpg"].notna()]

# Replace missing values with mean (for numeric data)
mtcars["mpg"] = mtcars["mpg"].fillna(mtcars["mpg"].mean())

# Replace missing values with median (more robust)
mtcars["mpg"] = mtcars["mpg"].fillna(mtcars["mpg"].median())

# Replace missing values with mode (for categorical data)
mode_value = mtcars["cylinders"].mode()[0]
mtcars["cylinders"] = mtcars["cylinders"].fillna(mode_value)

# Remove duplicate rows
unique_data = mtcars.drop_duplicates()

# Remove duplicate rows based on specific columns
unique_data = mtcars.drop_duplicates(subset=["mpg", "cylinders"])

# Convert data types
mtcars["cylinders"] = mtcars["cylinders"].astype("category")
mtcars["origin"] = mtcars["origin"].astype("category")

# Convert character to numeric (if appropriate)
mtcars["mpg"] = pd.to_numeric(mtcars["mpg"], errors="coerce")

# Standardize text (remove extra spaces, convert to lowercase)
mtcars["name"] = mtcars["name"].str.strip().str.lower()

# Handle outliers
# Method 1: Remove outliers
outlier_indices = outlier_detection(mtcars["mpg"])
mtcars_no_outliers = mtcars[~outlier_indices]

# Method 2: Cap outliers (winsorization)
def cap_outliers(x, lower_percentile=0.05, upper_percentile=0.95):
    lower_bound = x.quantile(lower_percentile)
    upper_bound = x.quantile(upper_percentile)
    return x.clip(lower=lower_bound, upper=upper_bound)

mtcars["mpg_capped"] = cap_outliers(mtcars["mpg"])
```

## Data Manipulation with pandas

The `pandas` library provides powerful tools for data manipulation using a consistent and expressive API:

### Selecting Columns

```python
# Select specific columns
mtcars_subset = mtcars[["mpg", "cylinders", "weight"]]

# Select columns by pattern
mtcars_numeric = mtcars.filter(regex="^m")  # Columns starting with "m"
mtcars_numeric = mtcars.filter(regex="t$")  # Columns ending with "t"
mtcars_numeric = mtcars.filter(like="p")    # Columns containing "p"

# Select columns by position
mtcars_first_three = mtcars.iloc[:, 0:3]

# Exclude columns
mtcars_no_mpg = mtcars.drop(columns=["mpg"])
mtcars_no_mpg_wt = mtcars.drop(columns=["mpg", "weight"])

# Rename columns while selecting
mtcars_renamed = mtcars.rename(columns={
    "mpg": "miles_per_gallon",
    "cylinders": "cyls",
    "weight": "wt"
})

# Select all columns except those matching a pattern
mtcars_no_m = mtcars.loc[:, ~mtcars.columns.str.startswith("m")]
```

### Filtering Rows

```python
# Filter by single condition
high_mpg = mtcars[mtcars["mpg"] > 20]
automatic = mtcars[mtcars["origin"] == "usa"]
heavy_cars = mtcars[mtcars["weight"] > 3500]

# Filter by multiple conditions
good_cars = mtcars[(mtcars["mpg"] > 20) & (mtcars["weight"] < 3000)]
efficient_or_light = mtcars[(mtcars["mpg"] > 25) | (mtcars["weight"] < 2500)]

# Filter with missing values
no_missing_mpg = mtcars[mtcars["mpg"].notna()]

# Filter with string matching
v8_cars = mtcars[mtcars["cylinders"] == 8]

# Filter with complex conditions
complex_filter = mtcars[(mtcars["mpg"] > 20) & (mtcars["weight"] < 3500) & (mtcars["cylinders"].isin([4, 6]))]

# Filter with between
medium_weight = mtcars[mtcars["weight"].between(2500, 3500)]
```

### Creating New Variables

```python
# Add single new column
mtcars["km_per_liter"] = mtcars["mpg"] * 0.425
mtcars["weight_kg"] = mtcars["weight"] * 0.453592

# Create multiple variables
mtcars["efficiency"] = mtcars["mpg"] / mtcars["weight"]
mtcars["size_category"] = mtcars["weight"].apply(lambda x: "Large" if x > 3500 else "Small")
mtcars["fuel_economy"] = pd.cut(mtcars["mpg"], bins=[0, 15, 20, 25, float('inf')], labels=["Poor", "Fair", "Good", "Excellent"])

# Create variables based on conditions
mtcars["is_efficient"] = mtcars["mpg"] > mtcars["mpg"].mean()
mtcars["weight_quartile"] = pd.qcut(mtcars["weight"], 4, labels=False)

# Apply function to multiple columns
mtcars[["mpg", "weight"]] = mtcars[["mpg", "weight"]].apply(lambda x: (x - x.mean()) / x.std())  # Standardize columns

# Create variables with row-wise operations
mtcars["row_mean"] = mtcars[["mpg", "weight", "acceleration"]].mean(axis=1)
```

### Summarizing Data

```python
# Overall summary
summary_stats = mtcars.agg(
    mean_mpg=("mpg", "mean"),
    median_mpg=("mpg", "median"),
    sd_mpg=("mpg", "std"),
    min_mpg=("mpg", "min"),
    max_mpg=("mpg", "max"),
    count=("mpg", "count"),
    missing_mpg=("mpg", lambda x: x.isna().sum())
)

# Grouped summary
cyl_summary = mtcars.groupby("cylinders").agg(
    mean_mpg=("mpg", "mean"),
    sd_mpg=("mpg", "std"),
    count=("mpg", "count")
).reset_index()

# Multiple grouping variables
detailed_summary = mtcars.groupby(["cylinders", "origin"]).agg(
    mean_mpg=("mpg", "mean"),
    count=("mpg", "count")
).reset_index()

# Summary with multiple functions
comprehensive_summary = mtcars.agg({
    col: ["mean", "std", "median"] for col in mtcars.select_dtypes(include=[np.number]).columns
})
```

### Arranging Data

```python
# Sort by single column (ascending)
sorted_mpg = mtcars.sort_values(by="mpg")

# Sort by single column (descending)
sorted_mpg_desc = mtcars.sort_values(by="mpg", ascending=False)

# Sort by multiple columns
sorted_multiple = mtcars.sort_values(by=["cylinders", "mpg"], ascending=[True, False])

# Sort with missing values (NaNs are always placed at the end by default)
sorted_with_na = mtcars.sort_values(by="mpg", ascending=False, na_position="last")

# Sort by computed values
mtcars["efficiency"] = mtcars["mpg"] / mtcars["weight"]
sorted_efficiency = mtcars.sort_values(by="efficiency", ascending=False)
```

### Method Chaining (Pipes)

Python uses method chaining (with `.`) instead of the R pipe operator `%>%`:

```python
# Without chaining (nested functions)
result = mtcars[mtcars["mpg"] > 20]
grouped = result.groupby("cylinders")
summary = grouped["mpg"].mean().reset_index(name="mean_mpg")

# With chaining (sequential operations)
result = (
    mtcars
    .loc[mtcars["mpg"] > 20]
    .groupby("cylinders")
    .agg(mean_mpg=("mpg", "mean"), count=("mpg", "count"))
    .reset_index()
)

# Complex pipeline example
analysis_result = (
    mtcars
    .dropna(subset=["mpg", "weight"])
    .assign(
        efficiency=lambda df: df["mpg"] / df["weight"],
        size_category=lambda df: np.where(df["weight"] > 3500, "Large", "Small")
    )
    .groupby(["cylinders", "size_category"])
    .agg(avg_efficiency=("efficiency", "mean"), count=("efficiency", "count"))
    .reset_index()
    .sort_values(by="avg_efficiency", ascending=False)
)
```

## Data Reshaping with pandas

The `pandas` library helps reshape data between wide and long formats:

### Understanding Data Shapes

- **Wide Format**: Each variable has its own column
- **Long Format**: Each observation has its own row

```python
import pandas as pd

# Create sample wide data
wide_data = pd.DataFrame({
    "id": [1, 2, 3],
    "math_2019": [85, 92, 78],
    "math_2020": [88, 95, 82],
    "science_2019": [90, 87, 85],
    "science_2020": [92, 89, 88]
})
```

### Wide to Long Format (melt)

```python
# Convert wide to long format
long_data = pd.melt(wide_data, id_vars=["id"], var_name="subject_year", value_name="score")

# More specific gathering with regex
long_data = wide_data.melt(id_vars=["id"], var_name="subject_year", value_name="score")

# Multiple value columns (use wide_to_long or custom split)
wide_data_multi = pd.DataFrame({
    "id": [1, 2, 3],
    "math_2019_score": [85, 92, 78],
    "math_2019_grade": ["A", "A", "B"],
    "math_2020_score": [88, 95, 82],
    "math_2020_grade": ["A", "A", "B"]
})

long_multi = pd.wide_to_long(
    wide_data_multi,
    stubnames=["math_2019", "math_2020"],
    i="id",
    j="measure",
    sep="_",
    suffix='score|grade'
).reset_index()
```

### Long to Wide Format (pivot)

```python
# Convert long to wide format
wide_data_again = long_data.pivot(index="id", columns="subject_year", values="score")

# Multiple value columns (pivot_table)
# (If you have multiple value columns, use pivot_table with aggfunc)
```

### Separating and Uniting Columns

```python
# Separate a column into multiple columns
long_data[["subject", "year"]] = long_data["subject_year"].str.split("_", expand=True)

# Unite columns
long_data["subject_year"] = long_data[["subject", "year"]].agg("_".join, axis=1)
```

## Working with Dates and Times

Date and time data requires special handling:

```python
import pandas as pd
import numpy as np

# Create date objects from different formats
dates_ymd = pd.to_datetime(["2023-01-15", "2023-02-20", "2023-03-10"])
dates_mdy = pd.to_datetime(["01/15/2023", "02/20/2023", "03/10/2023"], format="%m/%d/%Y")
dates_dmy = pd.to_datetime(["15-01-2023", "20-02-2023", "10-03-2023"], format="%d-%m-%Y")

# Extract components
print(dates_ymd.year)
print(dates_ymd.month)
print(dates_ymd.day)
print(dates_ymd.dayofweek)  # 0 = Monday
print(dates_ymd.dayofyear)

# Get month names
print(dates_ymd.month_name())
print(dates_ymd.day_name())

# Date arithmetic
print(dates_ymd + pd.Timedelta(days=7))
print(dates_ymd + pd.DateOffset(months=1))
print(dates_ymd + pd.DateOffset(years=1))

# Calculate differences
date_diff = dates_ymd[1] - dates_ymd[0]
print(date_diff.days)  # Difference in days

# Working with time zones
datetimes = pd.to_datetime(["2023-01-15 10:30:00", "2023-02-20 14:45:00"])
print(datetimes.tz_localize("UTC"))
print(datetimes.tz_localize("UTC").tz_convert("America/New_York"))

# Parse dates from character data
date_strings = ["2023-01-15", "Jan 15, 2023", "15/01/2023"]
parsed_dates = pd.to_datetime(date_strings, infer_datetime_format=True, errors="coerce")
```

## String Manipulation with pandas and re

String manipulation is essential for cleaning text data:

```python
import pandas as pd
import re

# Sample text data
text_data = pd.Series(["Hello World", "Python Programming", "Data Science", "  Extra Spaces  "])

# Basic operations
print(text_data.str.len())      # Count characters
print(text_data.str.upper())    # Convert to uppercase
print(text_data.str.lower())    # Convert to lowercase
print(text_data.str.strip())    # Remove leading/trailing whitespace
print(text_data.str.replace(r"\s+", " ", regex=True)) # Remove extra whitespace

# Pattern matching
print(text_data.str.contains("World"))     # TRUE FALSE FALSE FALSE
print(text_data.str.count("o"))            # Count occurrences of "o"
print(text_data.str.find("o"))             # Position of first "o"

# String replacement
print(text_data.str.replace("o", "0", n=1))           # Replace first occurrence
print(text_data.str.replace("o", "0"))                # Replace all occurrences

# String extraction
print(text_data.str.extract(r"(\b\w+)") )            # Extract first word
print(text_data.str.findall(r"(\b\w+)") )            # Extract all words
print(text_data.str.slice(0, 5))                        # Extract substring

# String splitting
print(text_data.str.split(" "))                        # Split by space

# Pattern matching with regular expressions
print(text_data.str.match(r"^[A-Z]"))                  # Starts with uppercase
print(text_data.str.contains(r"\d"))                  # Contains digit
print(text_data.str.contains(r"[aeiou]"))              # Contains vowel

# Case conversion functions
print(text_data.str.title())                            # Title case
```

## Practical Examples

### Example 1: Student Performance Analysis

```python
import pandas as pd

students = pd.DataFrame({
    "student_id": range(1, 11),
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve",
             "Frank", "Grace", "Henry", "Ivy", "Jack"],
    "math": [85, 92, 78, 96, 88, 75, 90, 82, 95, 87],
    "science": [88, 85, 92, 89, 90, 78, 92, 85, 88, 90],
    "english": [90, 87, 85, 92, 88, 80, 85, 88, 90, 85],
    "attendance": [95, 88, 92, 96, 90, 85, 94, 89, 93, 91]
})

# Calculate average and identify top performers
students["average"] = students[["math", "science", "english"]].mean(axis=1)
top_students = students[students["average"] > 85].sort_values(by="average", ascending=False)

# Analyze performance by subject
subject_analysis = pd.melt(students, id_vars=["student_id", "name", "attendance", "average"],
                          value_vars=["math", "science", "english"],
                          var_name="subject", value_name="score")
subject_stats = subject_analysis.groupby("subject").agg(
    mean_score=("score", "mean"),
    median_score=("score", "median"),
    sd_score=("score", "std"),
    count=("score", "count")
).reset_index()

# Correlation analysis
correlation_matrix = students[["math", "science", "english", "attendance"]].corr()
```

### Example 2: Sales Data Analysis

```python
import pandas as pd

sales_data = pd.DataFrame({
    "date": ["2023-01-15", "2023-01-16", "2023-01-17", "2023-01-18",
               "2023-01-19", "2023-01-20", "2023-01-21", "2023-01-22"],
    "product": ["A", "B", "A", "C", "B", "A", "C", "B"],
    "sales": [100, 150, 120, 200, 180, 110, 220, 160],
    "region": ["North", "South", "North", "East", "South", "North", "East", "South"],
    "customer_type": ["Retail", "Wholesale", "Retail", "Wholesale", 
                     "Retail", "Wholesale", "Retail", "Wholesale"]
})

# Convert date and analyze
sales_data["date"] = pd.to_datetime(sales_data["date"])
sales_data["month"] = sales_data["date"].dt.month
sales_data["day_of_week"] = sales_data["date"].dt.day_name()
sales_analysis = sales_data.groupby(["region", "customer_type"]).agg(
    total_sales=("sales", "sum"),
    avg_sales=("sales", "mean"),
    count=("sales", "count")
).reset_index().sort_values(by="total_sales", ascending=False)

# Time series analysis
daily_sales = sales_data.groupby("date").agg(
    total_daily_sales=("sales", "sum"),
    avg_daily_sales=("sales", "mean")
).reset_index().sort_values(by="date")

# Product performance
product_performance = sales_data.groupby("product").agg(
    total_sales=("sales", "sum"),
    avg_sales=("sales", "mean"),
    sales_count=("sales", "count")
).reset_index()
product_performance["sales_per_transaction"] = product_performance["total_sales"] / product_performance["sales_count"]
```

### Example 3: Data Quality Assessment

```python
import pandas as pd
import numpy as np

def assess_data_quality(data):
    print("=== DATA QUALITY ASSESSMENT ===")
    print("Dataset dimensions:", data.shape)
    print("\nMissing values per column:")
    print(data.isna().sum())
    print("\nData types:")
    print(data.dtypes)
    print("\nDuplicate rows:", data.duplicated().sum())
    print("Total missing values:", data.isna().sum().sum())
    print("\nPotential data quality issues:")
    for col in data.columns:
        if data[col].dtype == object:
            unique_vals = data[col].unique()
            if len(unique_vals) < data.shape[0] * 0.5:
                print(f"- Column '{col}' has many repeated values")

# Create dataset with various data quality issues
messy_data = pd.DataFrame({
    "id": range(1, 11),
    "name": ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown", "Charlie Wilson",
             "Diana Davis", "Eve Miller", "Frank Garcia", "Grace Martinez", "Henry Rodriguez"],
    "age": [25, 30, "N/A", 35, 28, 42, 33, "unknown", 27, 38],
    "income": [45000, 65000, 85000, "missing", 72000, 48000, 68000, 90000, 55000, "N/A"],
    "department": ["IT", "HR", "IT", "Finance", "Marketing", "IT", "HR", "Finance", "IT", "Marketing"],
    "hire_date": ["2020-01-15", "2019-03-20", "2021-06-10", "2018-11-05", "2020-09-12",
                  "2021-02-28", "2019-08-15", "2020-12-01", "2021-04-22", "2019-07-08"]
})

# Apply assessment
assess_data_quality(messy_data)

# Clean the data
clean_data = messy_data.copy()
clean_data["age"] = pd.to_numeric(clean_data["age"].replace(["N/A", "unknown"], np.nan))
clean_data["income"] = pd.to_numeric(clean_data["income"].replace(["missing", "N/A"], np.nan))
clean_data["hire_date"] = pd.to_datetime(clean_data["hire_date"])
clean_data = clean_data.dropna(subset=["age", "income"])
clean_data["department"] = clean_data["department"].str.upper()

# Verify cleaning
assess_data_quality(clean_data)
```

## Best Practices

### File Organization

```python
import os

# Set working directory
os.chdir("/path/to/your/project")

# Create organized folder structure
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/clean", exist_ok=True)
os.makedirs("scripts", exist_ok=True)
os.makedirs("results", exist_ok=True)
```

### Data Validation

```python
import pandas as pd
import numpy as np

def data_quality_check(data):
    print("=== COMPREHENSIVE DATA QUALITY CHECK ===")
    print("Dataset dimensions:", data.shape)
    print("Memory usage:", data.memory_usage(deep=True).sum() / (1024 ** 2), "MB")
    print("\nMissing values analysis:")
    missing_summary = pd.DataFrame({
        "Column": data.columns,
        "Missing_Count": data.isna().sum(),
        "Missing_Percent": round(data.isna().sum() / len(data) * 100, 2)
    })
    print(missing_summary)
    print("\nData type analysis:")
    type_summary = pd.DataFrame({
        "Column": data.columns,
        "Data_Type": data.dtypes,
        "Unique_Values": [data[col].nunique() for col in data.columns],
        "Sample_Values": [str(data[col].unique()[:3]) for col in data.columns]
    })
    print(type_summary)
    print("\nDuplicate analysis:")
    print("Duplicate rows:", data.duplicated().sum())
    print("Duplicate percentage:", round(data.duplicated().sum() / len(data) * 100, 2), "%\n")
    numeric_cols = data.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        print("Outlier analysis (using IQR method):")
        for col in numeric_cols.columns:
            Q1 = numeric_cols[col].quantile(0.25)
            Q3 = numeric_cols[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((numeric_cols[col] < (Q1 - 1.5 * IQR)) | (numeric_cols[col] > (Q3 + 1.5 * IQR))).sum()
            print(f"{col}: {outlier_count} outliers")

# Apply to your data
data_quality_check(mtcars)
```

### Reproducible Code

```python
import numpy as np
import pandas as pd
import pickle

# Set random seed for reproducibility
np.random.seed(123)

# Use relative paths
data = pd.read_csv("data/raw/sample_data.csv")

# Save processed data with metadata
processed_data = mtcars[mtcars["mpg"] > 20].copy()
processed_data["efficiency"] = processed_data["mpg"] / processed_data["weight"]

# Save with metadata
with open("data/clean/processed_data.pkl", "wb") as f:
    pickle.dump(processed_data, f)

# Create a data processing log
import datetime
processing_log = {
    "timestamp": datetime.datetime.now(),
    "original_rows": len(mtcars),
    "filtered_rows": len(processed_data),
    "variables_created": ["efficiency"],
    "filter_criteria": "mpg > 20"
}
with open("data/clean/processing_log.pkl", "wb") as f:
    pickle.dump(processing_log, f)

# Load processed data
with open("data/clean/processed_data.pkl", "rb") as f:
    processed_data = pickle.load(f)
```

## Exercises

### Exercise 1: Data Import and Inspection
Download a CSV file from the internet and import it into Python. Perform a comprehensive data quality assessment.

```python
# Your solution here
import pandas as pd
import numpy as np

# Download sample data (if available)
# data_url = "https://raw.githubusercontent.com/datasets/sample-data/master/data.csv"
# sample_data = pd.read_csv(data_url)

# For practice, create sample data
sample_data = pd.DataFrame({
    "id": range(1, 101),
    "value": np.random.normal(50, 10, 100),
    "category": np.random.choice(["A", "B", "C"], 100, replace=True),
    "date": pd.date_range("2023-01-01", periods=100, freq="D")
})

def data_quality_check(data):
    print("=== COMPREHENSIVE DATA QUALITY CHECK ===")
    print("Dataset dimensions:", data.shape)
    print("Memory usage:", data.memory_usage(deep=True).sum() / (1024 ** 2), "MB")
    print("\nMissing values analysis:")
    missing_summary = pd.DataFrame({
        "Column": data.columns,
        "Missing_Count": data.isna().sum(),
        "Missing_Percent": round(data.isna().sum() / len(data) * 100, 2)
    })
    print(missing_summary)
    print("\nData type analysis:")
    type_summary = pd.DataFrame({
        "Column": data.columns,
        "Data_Type": data.dtypes,
        "Unique_Values": [data[col].nunique() for col in data.columns],
        "Sample_Values": [str(data[col].unique()[:3]) for col in data.columns]
    })
    print(type_summary)
    print("\nDuplicate analysis:")
    print("Duplicate rows:", data.duplicated().sum())
    print("Duplicate percentage:", round(data.duplicated().sum() / len(data) * 100, 2), "%\n")
    numeric_cols = data.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        print("Outlier analysis (using IQR method):")
        for col in numeric_cols.columns:
            Q1 = numeric_cols[col].quantile(0.25)
            Q3 = numeric_cols[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((numeric_cols[col] < (Q1 - 1.5 * IQR)) | (numeric_cols[col] > (Q3 + 1.5 * IQR))).sum()
            print(f"{col}: {outlier_count} outliers")

data_quality_check(sample_data)
```

### Exercise 2: Data Manipulation with pandas
Using the `mpg` dataset, create a comprehensive analysis with multiple transformations.

```python
# Your solution here
import pandas as pd
import numpy as np
import seaborn as sns

mtcars = sns.load_dataset("mpg")
comprehensive_analysis = (
    mtcars
    .dropna(subset=["mpg", "weight"])
    .assign(
        efficiency=lambda df: df["mpg"] / df["weight"],
        size_category=lambda df: np.where(df["weight"] < 2500, "Small", np.where(df["weight"] < 3500, "Medium", "Large")),
        fuel_economy=lambda df: pd.cut(df["mpg"], bins=[0, 15, 20, 25, float('inf')], labels=["Poor", "Fair", "Good", "Excellent"])
    )
    .groupby(["cylinders", "size_category"])
    .agg(avg_efficiency=("efficiency", "mean"), avg_mpg=("mpg", "mean"), count=("mpg", "count"))
    .reset_index()
    .sort_values(by="avg_efficiency", ascending=False)
)
```

### Exercise 3: Data Reshaping
Create a wide dataset with multiple measurements and convert it to long format.

```python
# Your solution here
import pandas as pd
wide_data = pd.DataFrame({
    "id": range(1, 6),
    "temperature_jan": [10, 12, 8, 15, 11],
    "temperature_feb": [12, 14, 10, 16, 13],
    "temperature_mar": [15, 17, 13, 18, 16],
    "humidity_jan": [60, 65, 55, 70, 62],
    "humidity_feb": [62, 67, 57, 72, 64],
    "humidity_mar": [65, 70, 60, 75, 67]
})

# Convert to long format
long_data = pd.melt(
    wide_data,
    id_vars=["id"],
    var_name="measurement_month",
    value_name="value"
)
long_data[["measurement", "month"]] = long_data["measurement_month"].str.split("_", expand=True)
long_data = long_data.sort_values(by=["id", "month", "measurement"])
```

### Exercise 4: String Manipulation
Create a dataset with messy text data and clean it using pandas string functions.

```python
# Your solution here
import pandas as pd
import re
messy_text_data = pd.DataFrame({
    "id": range(1, 6),
    "name": ["  john doe  ", "JANE SMITH", "bob.johnson", "Alice-Brown", "CHARLIE_WILSON"],
    "email": ["john.doe@email.com", "jane.smith@email.com", "bob.johnson@email.com", 
              "alice.brown@email.com", "charlie.wilson@email.com"],
    "phone": ["555-123-4567", "(555) 234-5678", "555.345.6789", "555-456-7890", "555 567 8901"]
})

# Clean the data
clean_text_data = messy_text_data.copy()
clean_text_data["name"] = clean_text_data["name"].str.strip().str.title().str.replace(r"[._-]", " ", regex=True)
clean_text_data["email"] = clean_text_data["email"].str.lower()
clean_text_data["phone"] = clean_text_data["phone"].str.replace(r"[^0-9]", "", regex=True)
clean_text_data["phone"] = clean_text_data["phone"].str.replace(r"(\d{3})(\d{3})(\d{4})", r"\1-\2-\3", regex=True)
```

### Exercise 5: Date Analysis
Create a dataset with dates and perform various date-based analyses.

```python
# Your solution here
import pandas as pd
import numpy as np
date_data = pd.DataFrame({
    "id": range(1, 11),
    "event_date": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05", "2023-05-12",
                   "2023-06-18", "2023-07-25", "2023-08-30", "2023-09-14", "2023-10-22"],
    "value": np.random.normal(100, 20, 10)
})

date_data["event_date"] = pd.to_datetime(date_data["event_date"])
date_data["year"] = date_data["event_date"].dt.year
date_data["month"] = date_data["event_date"].dt.month_name()
date_data["day_of_week"] = date_data["event_date"].dt.day_name()
date_data["quarter"] = date_data["event_date"].dt.quarter
date_data["day_of_year"] = date_data["event_date"].dt.dayofyear

# Group by month and summarize
date_analysis = date_data.groupby("month").agg(
    avg_value=("value", "mean"),
    count=("value", "count")
).reset_index().sort_values(by="month")
```

## Next Steps

In the next chapter, we'll learn about exploratory data analysis techniques to understand and visualize our data. We'll cover:

- **Descriptive Statistics**: Understanding data distributions and summaries
- **Data Visualization**: Creating effective plots and charts
- **Correlation Analysis**: Understanding relationships between variables
- **Outlier Detection**: Identifying and handling unusual data points
- **Data Distribution Analysis**: Understanding the shape and characteristics of data

---

**Key Takeaways:**
- Always inspect your data after importing to understand its structure and quality
- Use appropriate packages for different file formats (pandas for CSV, Excel, JSON, etc.)
- Clean and validate data before analysis to ensure reliable results
- Use pandas for efficient and readable data manipulation
- Master method chaining for creating readable data processing pipelines
- Handle missing values appropriately based on the context and analysis needs
- Document your data processing steps for reproducibility
- Use pandas for reshaping data between wide and long formats
- Leverage pandas and datetime for date/time operations and string methods for text manipulation
- Create organized project structures with clear folder hierarchies 