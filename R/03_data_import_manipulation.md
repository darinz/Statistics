# Data Import and Manipulation

## Overview

Data import and manipulation are fundamental skills in R. Most real-world data analysis begins with importing data from external sources and cleaning it for analysis.

## Reading Data Files

### CSV Files

```r
# Read CSV file
data <- read.csv("data.csv")

# Read CSV with specific options
data <- read.csv("data.csv", 
                 header = TRUE,           # First row as column names
                 sep = ",",              # Comma separator
                 na.strings = c("", "NA", "N/A"),  # Missing value indicators
                 stringsAsFactors = FALSE)          # Don't convert strings to factors

# Read CSV using readr package (faster and more robust)
library(readr)
data <- read_csv("data.csv")

# Write CSV file
write.csv(data, "output.csv", row.names = FALSE)
write_csv(data, "output.csv")  # Using readr
```

### Excel Files

```r
# Install and load readxl package
install.packages("readxl")
library(readxl)

# Read Excel file
data <- read_excel("data.xlsx", sheet = 1)

# Read specific sheet by name
data <- read_excel("data.xlsx", sheet = "Sheet1")

# Read specific range
data <- read_excel("data.xlsx", range = "A1:D10")

# Write Excel file
library(writexl)
write_xlsx(data, "output.xlsx")
```

### Other File Formats

```r
# SPSS files
library(haven)
spss_data <- read_sav("data.sav")

# SAS files
sas_data <- read_sas("data.sas7bdat")

# Stata files
stata_data <- read_dta("data.dta")

# JSON files
library(jsonlite)
json_data <- fromJSON("data.json")

# XML files
library(xml2)
xml_data <- read_xml("data.xml")
```

## Built-in Datasets

R comes with many built-in datasets for practice:

```r
# List available datasets
data()

# Load specific dataset
data(mtcars)
data(iris)
data(USArrests)

# View dataset information
?mtcars
head(mtcars)
str(mtcars)
```

## Data Inspection and Cleaning

### Basic Data Inspection

```r
# Load a dataset
data(mtcars)

# Basic information
dim(mtcars)           # Dimensions
names(mtcars)         # Column names
str(mtcars)           # Structure
head(mtcars)          # First 6 rows
tail(mtcars)          # Last 6 rows
summary(mtcars)       # Summary statistics

# Check for missing values
is.na(mtcars)
colSums(is.na(mtcars))
any(is.na(mtcars))
```

### Data Cleaning

```r
# Remove rows with missing values
clean_data <- na.omit(mtcars)

# Replace missing values
mtcars$mpg[is.na(mtcars$mpg)] <- mean(mtcars$mpg, na.rm = TRUE)

# Remove duplicate rows
unique_data <- unique(mtcars)

# Convert data types
mtcars$cyl <- as.factor(mtcars$cyl)
mtcars$am <- as.factor(mtcars$am)
```

## Data Manipulation with dplyr

The `dplyr` package provides powerful tools for data manipulation:

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
mtcars_numeric <- select(mtcars, starts_with("m"))
mtcars_numeric <- select(mtcars, ends_with("t"))

# Exclude columns
mtcars_no_mpg <- select(mtcars, -mpg)
```

### Filtering Rows

```r
# Filter by condition
high_mpg <- filter(mtcars, mpg > 20)
automatic <- filter(mtcars, am == 0)
heavy_cars <- filter(mtcars, wt > 3.5)

# Multiple conditions
good_cars <- filter(mtcars, mpg > 20 & wt < 3.0)
```

### Creating New Variables

```r
# Add new column
mtcars <- mutate(mtcars, 
                 km_per_liter = mpg * 0.425,
                 weight_kg = wt * 453.592)

# Create multiple variables
mtcars <- mutate(mtcars,
                 efficiency = mpg / wt,
                 size_category = ifelse(wt > 3.5, "Large", "Small"))
```

### Summarizing Data

```r
# Overall summary
summary_stats <- summarise(mtcars,
                          mean_mpg = mean(mpg),
                          median_mpg = median(mpg),
                          sd_mpg = sd(mpg),
                          count = n())

# Grouped summary
by_cyl <- group_by(mtcars, cyl)
cyl_summary <- summarise(by_cyl,
                        mean_mpg = mean(mpg),
                        count = n())
```

### Arranging Data

```r
# Sort by column
sorted_mpg <- arrange(mtcars, mpg)
sorted_mpg_desc <- arrange(mtcars, desc(mpg))

# Sort by multiple columns
sorted_multiple <- arrange(mtcars, cyl, desc(mpg))
```

### The Pipe Operator

```r
# Using pipe operator for cleaner code
library(magrittr)

# Without pipe
result <- summarise(
  group_by(
    filter(mtcars, mpg > 20),
    cyl
  ),
  mean_mpg = mean(mpg)
)

# With pipe
result <- mtcars %>%
  filter(mpg > 20) %>%
  group_by(cyl) %>%
  summarise(mean_mpg = mean(mpg))
```

## Data Reshaping with tidyr

The `tidyr` package helps reshape data:

```r
# Install and load tidyr
install.packages("tidyr")
library(tidyr)

# Create sample data
wide_data <- data.frame(
  id = 1:3,
  math_2019 = c(85, 92, 78),
  math_2020 = c(88, 95, 82),
  science_2019 = c(90, 87, 85),
  science_2020 = c(92, 89, 88)
)
```

### Wide to Long Format

```r
# Convert wide to long format
long_data <- wide_data %>%
  gather(key = "subject_year", value = "score", 
         -id)

# More specific gathering
long_data <- wide_data %>%
  gather(key = "subject_year", value = "score", 
         math_2019:science_2020)
```

### Long to Wide Format

```r
# Convert long to wide format
wide_data_again <- long_data %>%
  spread(key = subject_year, value = score)
```

### Separating and Uniting Columns

```r
# Separate a column
long_data <- long_data %>%
  separate(subject_year, into = c("subject", "year"))

# Unite columns
long_data <- long_data %>%
  unite(subject_year, subject, year, sep = "_")
```

## Working with Dates and Times

```r
# Install and load lubridate
install.packages("lubridate")
library(lubridate)

# Create date objects
dates <- c("2023-01-15", "2023-02-20", "2023-03-10")
date_objects <- ymd(dates)

# Extract components
year(date_objects)
month(date_objects)
day(date_objects)
wday(date_objects)  # Day of week

# Date arithmetic
date_objects + days(7)
date_objects + months(1)
```

## String Manipulation with stringr

```r
# Install and load stringr
install.packages("stringr")
library(stringr)

# Sample text data
text_data <- c("Hello World", "R Programming", "Data Science")

# Basic operations
str_length(text_data)
str_to_upper(text_data)
str_to_lower(text_data)
str_trim(text_data)

# Pattern matching
str_detect(text_data, "World")
str_count(text_data, "o")
str_locate(text_data, "o")

# String replacement
str_replace(text_data, "o", "0")
str_replace_all(text_data, "o", "0")
```

## Practical Examples

### Example 1: Student Performance Analysis

```r
# Create student data
students <- data.frame(
  student_id = 1:10,
  name = c("Alice", "Bob", "Charlie", "Diana", "Eve",
           "Frank", "Grace", "Henry", "Ivy", "Jack"),
  math = c(85, 92, 78, 96, 88, 75, 90, 82, 95, 87),
  science = c(88, 85, 92, 89, 90, 78, 92, 85, 88, 90),
  english = c(90, 87, 85, 92, 88, 80, 85, 88, 90, 85)
)

# Calculate average and identify top performers
top_students <- students %>%
  mutate(average = (math + science + english) / 3) %>%
  filter(average > 85) %>%
  arrange(desc(average))

top_students
```

### Example 2: Sales Data Analysis

```r
# Create sales data
sales_data <- data.frame(
  date = c("2023-01-15", "2023-01-16", "2023-01-17", "2023-01-18"),
  product = c("A", "B", "A", "C"),
  sales = c(100, 150, 120, 200),
  region = c("North", "South", "North", "East")
)

# Convert date and analyze
sales_analysis <- sales_data %>%
  mutate(date = ymd(date)) %>%
  group_by(region) %>%
  summarise(
    total_sales = sum(sales),
    avg_sales = mean(sales),
    count = n()
  )

sales_analysis
```

## Best Practices

### File Organization

```r
# Set working directory
setwd("/path/to/your/project")

# Create organized folder structure
# data/raw/     - Original data files
# data/clean/   - Cleaned data files
# scripts/      - R scripts
# results/      - Output files
```

### Data Validation

```r
# Check data quality
data_quality_check <- function(data) {
  cat("Dataset dimensions:", dim(data), "\n")
  cat("Missing values per column:\n")
  print(colSums(is.na(data)))
  cat("Data types:\n")
  print(sapply(data, class))
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

# Save processed data
saveRDS(processed_data, "data/clean/processed_data.rds")
```

## Exercises

### Exercise 1: Data Import
Download a CSV file from the internet and import it into R. Inspect the data structure and clean any issues.

### Exercise 2: Data Manipulation
Using the `mtcars` dataset, create a new variable for fuel efficiency (mpg/wt) and find the most efficient cars.

### Exercise 3: Data Reshaping
Create a wide dataset with multiple measurements and convert it to long format using `gather()`.

### Exercise 4: String Manipulation
Create a dataset with messy text data and clean it using `stringr` functions.

### Exercise 5: Date Analysis
Create a dataset with dates and perform various date-based analyses using `lubridate`.

## Next Steps

In the next chapter, we'll learn about exploratory data analysis techniques to understand and visualize our data.

---

**Key Takeaways:**
- Always inspect your data after importing
- Use appropriate packages for different file formats
- Clean and validate data before analysis
- Use dplyr for efficient data manipulation
- Master the pipe operator for readable code
- Handle missing values appropriately
- Document your data processing steps 