# Data Import and Manipulation

## Overview

Data import and manipulation are fundamental skills in R. Most real-world data analysis begins with importing data from external sources and cleaning it for analysis. This process is crucial because the quality of your analysis depends directly on the quality of your data.

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

```r
# Basic CSV reading
data <- read.csv("data.csv")

# Read CSV with specific options for better control
data <- read.csv("data.csv", 
                 header = TRUE,           # First row as column names
                 sep = ",",              # Comma separator
                 na.strings = c("", "NA", "N/A", "NULL"),  # Missing value indicators
                 stringsAsFactors = FALSE,                  # Don't convert strings to factors
                 fileEncoding = "UTF-8",                    # Character encoding
                 quote = "\"",                             # Quote character
                 comment.char = "#")                       # Comment character

# Example with different separators
# Tab-separated file
data_tsv <- read.csv("data.tsv", sep = "\t")

# Semicolon-separated file (common in European countries)
data_semicolon <- read.csv("data.csv", sep = ";")

# Read CSV using readr package (faster and more robust)
library(readr)
data <- read_csv("data.csv")

# readr advantages:
# - Faster reading
# - Better error messages
# - Automatic type detection
# - Progress bars for large files

# Write CSV file
write.csv(data, "output.csv", row.names = FALSE)
write_csv(data, "output.csv")  # Using readr (no row names by default)

# Write with specific options
write.csv(data, "output.csv", 
          row.names = FALSE,
          na = "",              # How to represent missing values
          fileEncoding = "UTF-8")
```

### Excel Files

Excel files can contain multiple sheets and complex formatting:

```r
# Install and load readxl package
install.packages("readxl")
library(readxl)

# Read Excel file (first sheet by default)
data <- read_excel("data.xlsx")

# Read specific sheet by name
data <- read_excel("data.xlsx", sheet = "Sheet1")

# Read specific sheet by position
data <- read_excel("data.xlsx", sheet = 2)

# Read specific range (useful for large files)
data <- read_excel("data.xlsx", range = "A1:D10")

# Read with column types specification
data <- read_excel("data.xlsx", 
                   col_types = c("text", "numeric", "date", "text"))

# List all sheet names
excel_sheets("data.xlsx")

# Read multiple sheets
sheet_names <- excel_sheets("data.xlsx")
all_sheets <- lapply(sheet_names, function(sheet) {
  read_excel("data.xlsx", sheet = sheet)
})
names(all_sheets) <- sheet_names

# Write Excel file
library(writexl)
write_xlsx(data, "output.xlsx")

# Write multiple sheets
write_xlsx(list("Sheet1" = data1, "Sheet2" = data2), "output.xlsx")
```

### Other File Formats

```r
# SPSS files (Statistical Package for Social Sciences)
library(haven)
spss_data <- read_sav("data.sav")

# SAS files (Statistical Analysis System)
sas_data <- read_sas("data.sas7bdat")

# Stata files
stata_data <- read_dta("data.dta")

# JSON files (JavaScript Object Notation)
library(jsonlite)
json_data <- fromJSON("data.json")

# Read JSON from URL
json_from_url <- fromJSON("https://api.example.com/data")

# XML files (eXtensible Markup Language)
library(xml2)
xml_data <- read_xml("data.xml")

# Parse XML content
xml_content <- read_xml("<root><item>value</item></root>")

# RDS files (R Data Serialization - R's native format)
data <- readRDS("data.rds")
saveRDS(data, "output.rds")

# RData files (R's traditional format)
load("data.RData")
save(data, file = "output.RData")
```

## Built-in Datasets

R comes with many built-in datasets for practice and learning:

```r
# List all available datasets
data()

# List datasets in specific packages
data(package = "datasets")
data(package = "ggplot2")

# Load specific dataset
data(mtcars)
data(iris)
data(USArrests)

# View dataset information
?mtcars
head(mtcars)
str(mtcars)
dim(mtcars)
names(mtcars)

# Popular built-in datasets
# mtcars - Motor Trend Car Road Tests
# iris - Edgar Anderson's Iris Data
# USArrests - Violent Crime Rates by US State
# airquality - New York Air Quality Measurements
# faithful - Old Faithful Geyser Data
# Titanic - Survival of passengers on the Titanic
```

## Data Inspection and Cleaning

### Understanding Data Quality Issues

Common data quality problems include:
- **Missing Values**: NA, empty cells, or placeholder values
- **Inconsistent Formats**: Mixed date formats, inconsistent text
- **Outliers**: Extreme values that may be errors
- **Duplicates**: Repeated observations
- **Incorrect Data Types**: Numbers stored as text, dates as text
- **Inconsistent Naming**: Same values with different spellings

### Basic Data Inspection

```r
# Load a dataset
data(mtcars)

# Basic information
dim(mtcars)           # Dimensions (rows, columns)
names(mtcars)         # Column names
str(mtcars)           # Structure (data types and first few values)
head(mtcars)          # First 6 rows
tail(mtcars)          # Last 6 rows
summary(mtcars)       # Summary statistics

# Detailed inspection
class(mtcars)         # Object class
typeof(mtcars)        # Internal type
attributes(mtcars)    # All attributes

# Check for missing values
is.na(mtcars)         # Logical matrix of missing values
colSums(is.na(mtcars)) # Count of missing values per column
any(is.na(mtcars))    # TRUE if any missing values exist
sum(is.na(mtcars))    # Total number of missing values

# Check for duplicates
duplicated(mtcars)     # Logical vector of duplicated rows
sum(duplicated(mtcars)) # Number of duplicate rows

# Check data types
sapply(mtcars, class) # Class of each column
sapply(mtcars, typeof) # Internal type of each column

# Check for outliers (using IQR method)
outlier_detection <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  return(x < lower_bound | x > upper_bound)
}

# Apply to numeric columns
numeric_cols <- sapply(mtcars, is.numeric)
outliers <- sapply(mtcars[, numeric_cols], outlier_detection)
```

### Data Cleaning Techniques

```r
# Remove rows with missing values
clean_data <- na.omit(mtcars)

# Remove rows with missing values in specific columns
clean_data <- mtcars[!is.na(mtcars$mpg), ]

# Replace missing values with mean (for numeric data)
mtcars$mpg[is.na(mtcars$mpg)] <- mean(mtcars$mpg, na.rm = TRUE)

# Replace missing values with median (more robust)
mtcars$mpg[is.na(mtcars$mpg)] <- median(mtcars$mpg, na.rm = TRUE)

# Replace missing values with mode (for categorical data)
mode_value <- names(sort(table(mtcars$cyl), decreasing = TRUE))[1]
mtcars$cyl[is.na(mtcars$cyl)] <- mode_value

# Remove duplicate rows
unique_data <- unique(mtcars)

# Remove duplicate rows based on specific columns
unique_data <- mtcars[!duplicated(mtcars[, c("mpg", "cyl")]), ]

# Convert data types
mtcars$cyl <- as.factor(mtcars$cyl)
mtcars$am <- as.factor(mtcars$am)

# Convert character to numeric (if appropriate)
mtcars$mpg <- as.numeric(as.character(mtcars$mpg))

# Standardize text (remove extra spaces, convert to lowercase)
library(stringr)
mtcars$name <- str_trim(str_to_lower(mtcars$name))

# Handle outliers
# Method 1: Remove outliers
outlier_indices <- outlier_detection(mtcars$mpg)
mtcars_no_outliers <- mtcars[!outlier_indices, ]

# Method 2: Cap outliers (winsorization)
cap_outliers <- function(x, lower_percentile = 0.05, upper_percentile = 0.95) {
  lower_bound <- quantile(x, lower_percentile, na.rm = TRUE)
  upper_bound <- quantile(x, upper_percentile, na.rm = TRUE)
  x[x < lower_bound] <- lower_bound
  x[x > upper_bound] <- upper_bound
  return(x)
}

mtcars$mpg_capped <- cap_outliers(mtcars$mpg)
```

## Data Manipulation with dplyr

The `dplyr` package provides powerful tools for data manipulation using a consistent grammar:

### Understanding dplyr Philosophy

dplyr follows these principles:
- **Consistency**: All functions work similarly
- **Composability**: Functions can be combined easily
- **Expressiveness**: Code reads like natural language
- **Performance**: Optimized for speed and memory

```r
# Install and load dplyr
install.packages("dplyr")
library(dplyr)

# Load data
data(mtcars)
```

### Selecting Columns

```r
# Select specific columns
mtcars_subset <- select(mtcars, mpg, cyl, wt)

# Select columns by pattern
mtcars_numeric <- select(mtcars, starts_with("m"))  # Columns starting with "m"
mtcars_numeric <- select(mtcars, ends_with("t"))    # Columns ending with "t"
mtcars_numeric <- select(mtcars, contains("p"))     # Columns containing "p"

# Select columns by position
mtcars_first_three <- select(mtcars, 1:3)

# Exclude columns
mtcars_no_mpg <- select(mtcars, -mpg)
mtcars_no_mpg_wt <- select(mtcars, -c(mpg, wt))

# Rename columns while selecting
mtcars_renamed <- select(mtcars, 
                        miles_per_gallon = mpg,
                        cylinders = cyl,
                        weight = wt)

# Select all columns except those matching a pattern
mtcars_no_m <- select(mtcars, -starts_with("m"))
```

### Filtering Rows

```r
# Filter by single condition
high_mpg <- filter(mtcars, mpg > 20)
automatic <- filter(mtcars, am == 0)
heavy_cars <- filter(mtcars, wt > 3.5)

# Filter by multiple conditions
good_cars <- filter(mtcars, mpg > 20 & wt < 3.0)
efficient_or_light <- filter(mtcars, mpg > 25 | wt < 2.5)

# Filter with missing values
no_missing_mpg <- filter(mtcars, !is.na(mpg))

# Filter with string matching
v8_cars <- filter(mtcars, cyl == 8)

# Filter with complex conditions
complex_filter <- filter(mtcars, 
                        mpg > 20,
                        wt < 3.5,
                        cyl %in% c(4, 6))

# Filter with between
medium_weight <- filter(mtcars, between(wt, 2.5, 3.5))
```

### Creating New Variables

```r
# Add single new column
mtcars <- mutate(mtcars, 
                 km_per_liter = mpg * 0.425,
                 weight_kg = wt * 453.592)

# Create multiple variables
mtcars <- mutate(mtcars,
                 efficiency = mpg / wt,
                 size_category = ifelse(wt > 3.5, "Large", "Small"),
                 fuel_economy = case_when(
                   mpg >= 25 ~ "Excellent",
                   mpg >= 20 ~ "Good",
                   mpg >= 15 ~ "Fair",
                   TRUE ~ "Poor"
                 ))

# Create variables based on conditions
mtcars <- mutate(mtcars,
                 is_efficient = mpg > mean(mpg, na.rm = TRUE),
                 weight_quartile = ntile(wt, 4))

# Use across() for multiple columns
mtcars <- mutate(mtcars, across(c(mpg, wt), scale))  # Standardize columns

# Create variables with row-wise operations
mtcars <- mutate(mtcars,
                 row_mean = rowMeans(select(mtcars, mpg, wt, qsec), na.rm = TRUE))
```

### Summarizing Data

```r
# Overall summary
summary_stats <- summarise(mtcars,
                          mean_mpg = mean(mpg, na.rm = TRUE),
                          median_mpg = median(mpg, na.rm = TRUE),
                          sd_mpg = sd(mpg, na.rm = TRUE),
                          min_mpg = min(mpg, na.rm = TRUE),
                          max_mpg = max(mpg, na.rm = TRUE),
                          count = n(),
                          missing_mpg = sum(is.na(mpg)))

# Grouped summary
by_cyl <- group_by(mtcars, cyl)
cyl_summary <- summarise(by_cyl,
                        mean_mpg = mean(mpg, na.rm = TRUE),
                        sd_mpg = sd(mpg, na.rm = TRUE),
                        count = n(),
                        .groups = "drop")  # Remove grouping

# Multiple grouping variables
by_cyl_am <- group_by(mtcars, cyl, am)
detailed_summary <- summarise(by_cyl_am,
                             mean_mpg = mean(mpg, na.rm = TRUE),
                             count = n(),
                             .groups = "drop")

# Summary with multiple functions
comprehensive_summary <- summarise(mtcars,
                                 across(where(is.numeric), 
                                       list(mean = mean, sd = sd, median = median),
                                       na.rm = TRUE))
```

### Arranging Data

```r
# Sort by single column (ascending)
sorted_mpg <- arrange(mtcars, mpg)

# Sort by single column (descending)
sorted_mpg_desc <- arrange(mtcars, desc(mpg))

# Sort by multiple columns
sorted_multiple <- arrange(mtcars, cyl, desc(mpg))

# Sort with missing values
sorted_with_na <- arrange(mtcars, desc(mpg), na.last = TRUE)

# Sort by computed values
mtcars %>%
  mutate(efficiency = mpg / wt) %>%
  arrange(desc(efficiency))
```

### The Pipe Operator

The pipe operator (`%>%`) makes code more readable by chaining operations:

```r
# Without pipe (nested functions)
result <- summarise(
  group_by(
    filter(mtcars, mpg > 20),
    cyl
  ),
  mean_mpg = mean(mpg, na.rm = TRUE),
  count = n()
)

# With pipe (sequential operations)
result <- mtcars %>%
  filter(mpg > 20) %>%
  group_by(cyl) %>%
  summarise(mean_mpg = mean(mpg, na.rm = TRUE),
            count = n())

# Complex pipeline example
analysis_result <- mtcars %>%
  # Clean data
  filter(!is.na(mpg), !is.na(wt)) %>%
  # Create new variables
  mutate(efficiency = mpg / wt,
         size_category = ifelse(wt > 3.5, "Large", "Small")) %>%
  # Group and summarize
  group_by(cyl, size_category) %>%
  summarise(avg_efficiency = mean(efficiency, na.rm = TRUE),
            count = n(),
            .groups = "drop") %>%
  # Arrange results
  arrange(desc(avg_efficiency))
```

## Data Reshaping with tidyr

The `tidyr` package helps reshape data between wide and long formats:

### Understanding Data Shapes

- **Wide Format**: Each variable has its own column
- **Long Format**: Each observation has its own row

```r
# Install and load tidyr
install.packages("tidyr")
library(tidyr)

# Create sample wide data
wide_data <- data.frame(
  id = 1:3,
  math_2019 = c(85, 92, 78),
  math_2020 = c(88, 95, 82),
  science_2019 = c(90, 87, 85),
  science_2020 = c(92, 89, 88)
)
```

### Wide to Long Format (gather/pivot_longer)

```r
# Convert wide to long format (old tidyr syntax)
long_data <- wide_data %>%
  gather(key = "subject_year", value = "score", 
         -id)

# Modern tidyr syntax (pivot_longer)
long_data <- wide_data %>%
  pivot_longer(cols = -id,
               names_to = "subject_year",
               values_to = "score")

# More specific gathering
long_data <- wide_data %>%
  pivot_longer(cols = math_2019:science_2020,
               names_to = "subject_year",
               values_to = "score")

# Multiple value columns
wide_data_multi <- data.frame(
  id = 1:3,
  math_2019_score = c(85, 92, 78),
  math_2019_grade = c("A", "A", "B"),
  math_2020_score = c(88, 95, 82),
  math_2020_grade = c("A", "A", "B")
)

long_multi <- wide_data_multi %>%
  pivot_longer(cols = -id,
               names_to = c("subject", "year", "measure"),
               names_pattern = "(.*)_(.*)_(.*)",
               values_to = "value")
```

### Long to Wide Format (spread/pivot_wider)

```r
# Convert long to wide format (old tidyr syntax)
wide_data_again <- long_data %>%
  spread(key = subject_year, value = score)

# Modern tidyr syntax (pivot_wider)
wide_data_again <- long_data %>%
  pivot_wider(names_from = subject_year,
              values_from = score)

# Multiple value columns
long_multi %>%
  pivot_wider(names_from = measure,
              values_from = value)
```

### Separating and Uniting Columns

```r
# Separate a column into multiple columns
long_data <- long_data %>%
  separate(subject_year, into = c("subject", "year"))

# Separate with custom separator
data_with_sep <- data.frame(
  id = 1:3,
  name_location = c("John_NYC", "Jane_LA", "Bob_Chicago")
)

separated_data <- data_with_sep %>%
  separate(name_location, into = c("name", "location"), sep = "_")

# Unite columns
long_data <- long_data %>%
  unite(subject_year, subject, year, sep = "_")

# Unite with custom separator and remove original columns
long_data <- long_data %>%
  unite(subject_year, subject, year, sep = "_", remove = FALSE)
```

## Working with Dates and Times

Date and time data requires special handling:

```r
# Install and load lubridate
install.packages("lubridate")
library(lubridate)

# Create date objects from different formats
dates_ymd <- ymd(c("2023-01-15", "2023-02-20", "2023-03-10"))
dates_mdy <- mdy(c("01/15/2023", "02/20/2023", "03/10/2023"))
dates_dmy <- dmy(c("15-01-2023", "20-02-2023", "10-03-2023"))

# Create datetime objects
datetimes <- ymd_hms(c("2023-01-15 10:30:00", "2023-02-20 14:45:00"))

# Extract components
year(dates_ymd)
month(dates_ymd)
day(dates_ymd)
wday(dates_ymd)  # Day of week (1 = Sunday)
yday(dates_ymd)  # Day of year

# Get month names
month(dates_ymd, label = TRUE)
wday(dates_ymd, label = TRUE)

# Date arithmetic
dates_ymd + days(7)
dates_ymd + months(1)
dates_ymd + years(1)

# Calculate differences
date_diff <- dates_ymd[2] - dates_ymd[1]
as.numeric(date_diff)  # Difference in days

# Working with time zones
with_tz(datetimes, "UTC")
with_tz(datetimes, "America/New_York")

# Parse dates from character data
date_strings <- c("2023-01-15", "Jan 15, 2023", "15/01/2023")
parsed_dates <- parse_date_time(date_strings, c("ymd", "mdy", "dmy"))
```

## String Manipulation with stringr

String manipulation is essential for cleaning text data:

```r
# Install and load stringr
install.packages("stringr")
library(stringr)

# Sample text data
text_data <- c("Hello World", "R Programming", "Data Science", "  Extra Spaces  ")

# Basic operations
str_length(text_data)      # Count characters
str_to_upper(text_data)    # Convert to uppercase
str_to_lower(text_data)    # Convert to lowercase
str_trim(text_data)        # Remove leading/trailing whitespace
str_squish(text_data)      # Remove extra whitespace

# Pattern matching
str_detect(text_data, "World")     # TRUE FALSE FALSE FALSE
str_count(text_data, "o")          # Count occurrences of "o"
str_locate(text_data, "o")         # Position of first "o"
str_locate_all(text_data, "o")     # Positions of all "o"

# String replacement
str_replace(text_data, "o", "0")           # Replace first occurrence
str_replace_all(text_data, "o", "0")       # Replace all occurrences
str_replace_na(text_data, "MISSING")       # Replace NA with string

# String extraction
str_extract(text_data, "\\b\\w+")          # Extract first word
str_extract_all(text_data, "\\b\\w+")      # Extract all words
str_sub(text_data, 1, 5)                   # Extract substring

# String splitting
str_split(text_data, " ")                  # Split by space
str_split_fixed(text_data, " ", 2)         # Split into 2 parts

# Pattern matching with regular expressions
str_detect(text_data, "^[A-Z]")            # Starts with uppercase
str_detect(text_data, "\\d")               # Contains digit
str_detect(text_data, "[aeiou]")           # Contains vowel

# Case conversion functions
str_to_title(text_data)                    # Title case
str_to_sentence(text_data)                 # Sentence case
```

## Practical Examples

### Example 1: Student Performance Analysis

```r
# Create comprehensive student data
students <- data.frame(
  student_id = 1:10,
  name = c("Alice", "Bob", "Charlie", "Diana", "Eve",
           "Frank", "Grace", "Henry", "Ivy", "Jack"),
  math = c(85, 92, 78, 96, 88, 75, 90, 82, 95, 87),
  science = c(88, 85, 92, 89, 90, 78, 92, 85, 88, 90),
  english = c(90, 87, 85, 92, 88, 80, 85, 88, 90, 85),
  attendance = c(95, 88, 92, 96, 90, 85, 94, 89, 93, 91)
)

# Calculate average and identify top performers
top_students <- students %>%
  mutate(average = (math + science + english) / 3) %>%
  filter(average > 85) %>%
  arrange(desc(average))

# Analyze performance by subject
subject_analysis <- students %>%
  pivot_longer(cols = c(math, science, english),
               names_to = "subject",
               values_to = "score") %>%
  group_by(subject) %>%
  summarise(mean_score = mean(score, na.rm = TRUE),
            median_score = median(score, na.rm = TRUE),
            sd_score = sd(score, na.rm = TRUE),
            count = n())

# Correlation analysis
correlation_matrix <- students %>%
  select(math, science, english, attendance) %>%
  cor(use = "complete.obs")
```

### Example 2: Sales Data Analysis

```r
# Create comprehensive sales data
sales_data <- data.frame(
  date = c("2023-01-15", "2023-01-16", "2023-01-17", "2023-01-18",
           "2023-01-19", "2023-01-20", "2023-01-21", "2023-01-22"),
  product = c("A", "B", "A", "C", "B", "A", "C", "B"),
  sales = c(100, 150, 120, 200, 180, 110, 220, 160),
  region = c("North", "South", "North", "East", "South", "North", "East", "South"),
  customer_type = c("Retail", "Wholesale", "Retail", "Wholesale", 
                   "Retail", "Wholesale", "Retail", "Wholesale")
)

# Convert date and analyze
sales_analysis <- sales_data %>%
  mutate(date = ymd(date),
         month = month(date),
         day_of_week = wday(date, label = TRUE)) %>%
  group_by(region, customer_type) %>%
  summarise(total_sales = sum(sales),
            avg_sales = mean(sales),
            count = n(),
            .groups = "drop") %>%
  arrange(desc(total_sales))

# Time series analysis
daily_sales <- sales_data %>%
  mutate(date = ymd(date)) %>%
  group_by(date) %>%
  summarise(total_daily_sales = sum(sales),
            avg_daily_sales = mean(sales),
            .groups = "drop") %>%
  arrange(date)

# Product performance
product_performance <- sales_data %>%
  group_by(product) %>%
  summarise(total_sales = sum(sales),
            avg_sales = mean(sales),
            sales_count = n(),
            .groups = "drop") %>%
  mutate(sales_per_transaction = total_sales / sales_count)
```

### Example 3: Data Quality Assessment

```r
# Create dataset with various data quality issues
messy_data <- data.frame(
  id = 1:10,
  name = c("John Doe", "Jane Smith", "Bob Johnson", "Alice Brown", "Charlie Wilson",
           "Diana Davis", "Eve Miller", "Frank Garcia", "Grace Martinez", "Henry Rodriguez"),
  age = c(25, 30, "N/A", 35, 28, 42, 33, "unknown", 27, 38),
  income = c(45000, 65000, 85000, "missing", 72000, 48000, 68000, 90000, 55000, "N/A"),
  department = c("IT", "HR", "IT", "Finance", "Marketing", "IT", "HR", "Finance", "IT", "Marketing"),
  hire_date = c("2020-01-15", "2019-03-20", "2021-06-10", "2018-11-05", "2020-09-12",
                "2021-02-28", "2019-08-15", "2020-12-01", "2021-04-22", "2019-07-08")
)

# Data quality assessment function
assess_data_quality <- function(data) {
  cat("=== DATA QUALITY ASSESSMENT ===\n")
  cat("Dataset dimensions:", dim(data), "\n\n")
  
  cat("Missing values per column:\n")
  missing_counts <- colSums(is.na(data))
  print(missing_counts)
  cat("\n")
  
  cat("Data types:\n")
  data_types <- sapply(data, class)
  print(data_types)
  cat("\n")
  
  cat("Duplicate rows:", sum(duplicated(data)), "\n")
  cat("Total missing values:", sum(is.na(data)), "\n")
  
  # Check for inconsistent formats
  cat("\nPotential data quality issues:\n")
  for(col in names(data)) {
    if(is.character(data[[col]])) {
      # Check for mixed formats
      unique_vals <- unique(data[[col]])
      if(length(unique_vals) < nrow(data) * 0.5) {
        cat("- Column '", col, "' has many repeated values\n")
      }
    }
  }
}

# Apply assessment
assess_data_quality(messy_data)

# Clean the data
clean_data <- messy_data %>%
  # Convert age to numeric, handling non-numeric values
  mutate(age = as.numeric(ifelse(age %in% c("N/A", "unknown"), NA, age))) %>%
  # Convert income to numeric, handling non-numeric values
  mutate(income = as.numeric(ifelse(income %in% c("missing", "N/A"), NA, income))) %>%
  # Convert hire_date to proper date format
  mutate(hire_date = ymd(hire_date)) %>%
  # Remove rows with missing critical data
  filter(!is.na(age), !is.na(income)) %>%
  # Standardize department names
  mutate(department = str_to_upper(department))

# Verify cleaning
assess_data_quality(clean_data)
```

## Best Practices

### File Organization

```r
# Set working directory
setwd("/path/to/your/project")

# Create organized folder structure
# data/
#   raw/     - Original data files
#   clean/   - Cleaned data files
#   processed/ - Processed data files
# scripts/
#   import/  - Data import scripts
#   clean/   - Data cleaning scripts
#   analysis/ - Analysis scripts
# results/
#   tables/  - Output tables
#   figures/ - Output figures
#   reports/ - Generated reports

# Create directories if they don't exist
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
dir.create("data/clean", recursive = TRUE, showWarnings = FALSE)
dir.create("scripts", recursive = TRUE, showWarnings = FALSE)
dir.create("results", recursive = TRUE, showWarnings = FALSE)
```

### Data Validation

```r
# Comprehensive data quality check function
data_quality_check <- function(data, expected_types = NULL) {
  cat("=== COMPREHENSIVE DATA QUALITY CHECK ===\n")
  
  # Basic information
  cat("Dataset dimensions:", dim(data), "\n")
  cat("Memory usage:", format(object.size(data), units = "MB"), "\n\n")
  
  # Missing values analysis
  cat("Missing values analysis:\n")
  missing_summary <- data.frame(
    Column = names(data),
    Missing_Count = colSums(is.na(data)),
    Missing_Percent = round(colSums(is.na(data)) / nrow(data) * 100, 2)
  )
  print(missing_summary)
  cat("\n")
  
  # Data type analysis
  cat("Data type analysis:\n")
  type_summary <- data.frame(
    Column = names(data),
    Data_Type = sapply(data, class),
    Unique_Values = sapply(data, function(x) length(unique(x))),
    Sample_Values = sapply(data, function(x) {
      if(is.numeric(x)) {
        paste(round(range(x, na.rm = TRUE), 2), collapse = " to ")
      } else {
        paste(head(unique(x), 3), collapse = ", ")
      }
    })
  )
  print(type_summary)
  cat("\n")
  
  # Duplicate analysis
  cat("Duplicate analysis:\n")
  cat("Duplicate rows:", sum(duplicated(data)), "\n")
  cat("Duplicate percentage:", round(sum(duplicated(data)) / nrow(data) * 100, 2), "%\n\n")
  
  # Outlier analysis for numeric columns
  numeric_cols <- sapply(data, is.numeric)
  if(any(numeric_cols)) {
    cat("Outlier analysis (using IQR method):\n")
    outlier_summary <- data.frame(
      Column = names(data)[numeric_cols],
      Outlier_Count = sapply(data[numeric_cols], function(x) {
        Q1 <- quantile(x, 0.25, na.rm = TRUE)
        Q3 <- quantile(x, 0.75, na.rm = TRUE)
        IQR <- Q3 - Q1
        sum(x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR), na.rm = TRUE)
      })
    )
    print(outlier_summary)
  }
}

# Apply to your data
data_quality_check(mtcars)
```

### Reproducible Code

```r
# Set random seed for reproducibility
set.seed(123)

# Use relative paths
data <- read.csv("data/raw/sample_data.csv")

# Save processed data with metadata
processed_data <- mtcars %>%
  filter(mpg > 20) %>%
  mutate(efficiency = mpg / wt)

# Save with metadata
saveRDS(processed_data, "data/clean/processed_data.rds")

# Create a data processing log
processing_log <- list(
  timestamp = Sys.time(),
  original_rows = nrow(mtcars),
  filtered_rows = nrow(processed_data),
  variables_created = c("efficiency"),
  filter_criteria = "mpg > 20"
)

saveRDS(processing_log, "data/clean/processing_log.rds")

# Load processed data
processed_data <- readRDS("data/clean/processed_data.rds")
```

## Exercises

### Exercise 1: Data Import and Inspection
Download a CSV file from the internet and import it into R. Perform a comprehensive data quality assessment.

```r
# Your solution here
# Example: Download and analyze a sample dataset
library(readr)

# Download sample data (if available)
# data_url <- "https://raw.githubusercontent.com/datasets/sample-data/master/data.csv"
# sample_data <- read_csv(data_url)

# For practice, create sample data
sample_data <- data.frame(
  id = 1:100,
  value = rnorm(100, mean = 50, sd = 10),
  category = sample(c("A", "B", "C"), 100, replace = TRUE),
  date = seq(as.Date("2023-01-01"), by = "day", length.out = 100)
)

# Perform data quality assessment
data_quality_check(sample_data)
```

### Exercise 2: Data Manipulation with dplyr
Using the `mtcars` dataset, create a comprehensive analysis with multiple transformations.

```r
# Your solution here
comprehensive_analysis <- mtcars %>%
  # Clean data
  filter(!is.na(mpg), !is.na(wt)) %>%
  # Create new variables
  mutate(efficiency = mpg / wt,
         size_category = case_when(
           wt < 2.5 ~ "Small",
           wt < 3.5 ~ "Medium",
           TRUE ~ "Large"
         ),
         fuel_economy = case_when(
           mpg >= 25 ~ "Excellent",
           mpg >= 20 ~ "Good",
           mpg >= 15 ~ "Fair",
           TRUE ~ "Poor"
         )) %>%
  # Group and summarize
  group_by(cyl, size_category) %>%
  summarise(avg_efficiency = mean(efficiency, na.rm = TRUE),
            avg_mpg = mean(mpg, na.rm = TRUE),
            count = n(),
            .groups = "drop") %>%
  # Arrange results
  arrange(desc(avg_efficiency))
```

### Exercise 3: Data Reshaping
Create a wide dataset with multiple measurements and convert it to long format.

```r
# Your solution here
# Create wide dataset
wide_data <- data.frame(
  id = 1:5,
  temperature_jan = c(10, 12, 8, 15, 11),
  temperature_feb = c(12, 14, 10, 16, 13),
  temperature_mar = c(15, 17, 13, 18, 16),
  humidity_jan = c(60, 65, 55, 70, 62),
  humidity_feb = c(62, 67, 57, 72, 64),
  humidity_mar = c(65, 70, 60, 75, 67)
)

# Convert to long format
long_data <- wide_data %>%
  pivot_longer(cols = -id,
               names_to = c("measurement", "month"),
               names_pattern = "(.*)_(.*)",
               values_to = "value") %>%
  arrange(id, month, measurement)
```

### Exercise 4: String Manipulation
Create a dataset with messy text data and clean it using `stringr` functions.

```r
# Your solution here
messy_text_data <- data.frame(
  id = 1:5,
  name = c("  john doe  ", "JANE SMITH", "bob.johnson", "Alice-Brown", "CHARLIE_WILSON"),
  email = c("john.doe@email.com", "jane.smith@email.com", "bob.johnson@email.com", 
            "alice.brown@email.com", "charlie.wilson@email.com"),
  phone = c("555-123-4567", "(555) 234-5678", "555.345.6789", "555-456-7890", "555 567 8901")
)

# Clean the data
clean_text_data <- messy_text_data %>%
  mutate(
    # Clean names
    name = str_trim(str_to_title(name)),
    name = str_replace_all(name, "[._-]", " "),
    
    # Standardize email
    email = str_to_lower(email),
    
    # Standardize phone numbers
    phone = str_replace_all(phone, "[^0-9]", ""),
    phone = str_replace(phone, "(\\d{3})(\\d{3})(\\d{4})", "\\1-\\2-\\3")
  )
```

### Exercise 5: Date Analysis
Create a dataset with dates and perform various date-based analyses.

```r
# Your solution here
date_data <- data.frame(
  id = 1:10,
  event_date = c("2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05", "2023-05-12",
                 "2023-06-18", "2023-07-25", "2023-08-30", "2023-09-14", "2023-10-22"),
  value = rnorm(10, mean = 100, sd = 20)
)

# Perform date analysis
date_analysis <- date_data %>%
  mutate(
    event_date = ymd(event_date),
    year = year(event_date),
    month = month(event_date, label = TRUE),
    day_of_week = wday(event_date, label = TRUE),
    quarter = quarter(event_date),
    day_of_year = yday(event_date)
  ) %>%
  group_by(month) %>%
  summarise(
    avg_value = mean(value, na.rm = TRUE),
    count = n(),
    .groups = "drop"
  ) %>%
  arrange(month)
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
- Use appropriate packages for different file formats (readr for CSV, readxl for Excel)
- Clean and validate data before analysis to ensure reliable results
- Use dplyr for efficient and readable data manipulation
- Master the pipe operator (`%>%`) for creating readable data processing pipelines
- Handle missing values appropriately based on the context and analysis needs
- Document your data processing steps for reproducibility
- Use tidyr for reshaping data between wide and long formats
- Leverage lubridate for date/time operations and stringr for text manipulation
- Create organized project structures with clear folder hierarchies 