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

**📁 Code Reference**: See `reading_csv_files()` function in `03_data_import_manipulation.py`

- Reading CSV files with pandas
- Handling different separators and missing value indicators
- Writing CSV files with various options

### Excel Files

**📁 Code Reference**: See `reading_excel_files()` function in `03_data_import_manipulation.py`

- Reading Excel files and specific sheets
- Writing Excel files and multiple sheets

### Other File Formats

**📁 Code Reference**: See `reading_other_formats()` function in `03_data_import_manipulation.py`

- Reading and writing JSON files
- Using pickle for serialization
- (SPSS, SAS, Stata, XML: see pandas documentation for more)

## Built-in Datasets

**📁 Code Reference**: See `built_in_datasets()` function in `03_data_import_manipulation.py`

- Loading datasets from seaborn and scikit-learn
- Listing available datasets

## Data Inspection and Cleaning

### Basic Data Inspection

**📁 Code Reference**: See `basic_data_inspection()` function in `03_data_import_manipulation.py`

- Checking shape, columns, info, head, tail, describe
- Checking for missing values, duplicates, outliers

### Data Cleaning Techniques

**📁 Code Reference**: See `data_cleaning_techniques()` function in `03_data_import_manipulation.py`

- Removing or filling missing values
- Removing duplicates
- Converting data types
- Standardizing text

## Data Manipulation with pandas

### Selecting Columns

**📁 Code Reference**: See `selecting_columns()` function in `03_data_import_manipulation.py`

- Selecting columns by name, pattern, or position
- Excluding columns
- Renaming columns

### Filtering Rows

**📁 Code Reference**: See `filtering_rows()` function in `03_data_import_manipulation.py`

- Filtering by single or multiple conditions
- Filtering with missing values or string matching
- Filtering with between

### Creating New Variables

**📁 Code Reference**: See `creating_new_variables()` function in `03_data_import_manipulation.py`

- Adding new columns
- Creating variables based on conditions
- Row-wise operations

### Summarizing Data

**📁 Code Reference**: See `summarizing_data()` function in `03_data_import_manipulation.py`

- Overall and grouped summary statistics
- Multiple grouping variables

### Arranging Data

**📁 Code Reference**: See `arranging_data()` function in `03_data_import_manipulation.py`

- Sorting by one or more columns
- Sorting by computed values

### Method Chaining (Pipes)

**📁 Code Reference**: See `method_chaining()` function in `03_data_import_manipulation.py`

- Using method chaining for readable data pipelines
- Complex pipeline examples

## Data Reshaping with pandas

### Understanding Data Shapes

**📁 Code Reference**: See `data_reshaping_basics()` function in `03_data_import_manipulation.py`

- Wide vs. long format
- Melting and pivoting data
- Separating and uniting columns

## Working with Dates and Times

**📁 Code Reference**: See `working_with_dates()` function in `03_data_import_manipulation.py`

- Creating and parsing dates
- Extracting date components
- Date arithmetic and differences

## String Manipulation with pandas and re

**📁 Code Reference**: See `string_manipulation()` function in `03_data_import_manipulation.py`

- Basic string operations
- Pattern matching and replacement
- String extraction and splitting
- Regular expressions

## Practical Examples

### Example 1: Student Performance Analysis

**📁 Code Reference**: See `student_performance_analysis()` function in `03_data_import_manipulation.py`

- Calculating averages and identifying top performers
- Subject-wise analysis and statistics
- Correlation analysis

### Example 2: Sales Data Analysis

**📁 Code Reference**: See `sales_data_analysis()` function in `03_data_import_manipulation.py`

- Grouped and time series analysis
- Product performance metrics

### Example 3: Data Quality Assessment

**📁 Code Reference**: See `data_quality_assessment()` function in `03_data_import_manipulation.py`

- Assessing and cleaning data quality issues

## Best Practices

### File Organization

**📁 Code Reference**: See `file_organization()` function in `03_data_import_manipulation.py`

- Creating organized folder structures

### Data Validation

**📁 Code Reference**: See `data_validation()` function in `03_data_import_manipulation.py`

- Comprehensive data quality checks

### Reproducible Code

**📁 Code Reference**: See `reproducible_code()` function in `03_data_import_manipulation.py`

- Setting random seeds
- Saving processed data and logs

## Exercises

### Exercise 1: Data Import and Inspection

**📁 Code Reference**: See `exercise_1_data_import_inspection()` function in `03_data_import_manipulation.py`

- Downloading or creating sample data
- Performing a comprehensive data quality assessment

### Exercise 2: Data Manipulation with pandas

**📁 Code Reference**: See `exercise_2_data_manipulation()` function in `03_data_import_manipulation.py`

- Comprehensive analysis with multiple transformations

### Exercise 3: Data Reshaping

**📁 Code Reference**: See `exercise_3_data_reshaping()` function in `03_data_import_manipulation.py`

- Creating wide and long datasets
- Reshaping and sorting data

### Exercise 4: String Manipulation

**📁 Code Reference**: See `exercise_4_string_manipulation()` function in `03_data_import_manipulation.py`

- Cleaning messy text data

### Exercise 5: Date Analysis

**📁 Code Reference**: See `exercise_5_date_analysis()` function in `03_data_import_manipulation.py`

- Creating and analyzing date-based data

## Running the Code Examples

To run all the code examples in this lesson:

1. **Run the entire file**: Execute `python 03_data_import_manipulation.py` in your terminal
2. **Run individual sections**: In a Python environment, import and call specific functions:
   ```python
   from data_import_manipulation import reading_csv_files, basic_data_inspection
   reading_csv_files()
   basic_data_inspection()
   ```
3. **Interactive learning**: Copy individual functions into Jupyter notebooks for interactive exploration

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
- All code examples are available in the companion Python file for hands-on practice 