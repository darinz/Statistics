"""
Data Import and Manipulation

This module contains comprehensive examples of data import and manipulation
techniques in Python using pandas, numpy, and other libraries.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import json
import pickle
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def reading_csv_files():
    """
    Demonstrate reading and writing CSV files with pandas.
    
    Covers different separators, missing value indicators, and writing options.
    """
    print("=== Reading CSV Files ===\n")
    
    # Create sample data
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Age': [25, 30, 35, 28, 32],
        'Salary': [50000, 60000, 75000, 55000, 65000],
        'Department': ['HR', 'IT', 'Sales', 'HR', 'IT'],
        'Start_Date': ['2020-01-15', '2019-03-20', '2018-07-10', '2021-02-28', '2020-11-05']
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Save as CSV
    df.to_csv('sample_data.csv', index=False)
    print("Saved data to 'sample_data.csv'")
    
    # Read CSV file
    df_read = pd.read_csv('sample_data.csv')
    print("\nRead from CSV:")
    print(df_read.head())
    print("\n" + "="*50 + "\n")
    
    # Reading CSV with different options
    print("Reading with different options:")
    
    # Read with specific data types
    df_types = pd.read_csv('sample_data.csv', dtype={'Age': 'int64', 'Salary': 'float64'})
    print("\nWith specific data types:")
    print(df_types.dtypes)
    
    # Read with date parsing
    df_dates = pd.read_csv('sample_data.csv', parse_dates=['Start_Date'])
    print("\nWith date parsing:")
    print(df_dates.dtypes)
    
    # Read with missing value indicators
    df_missing = pd.read_csv('sample_data.csv', na_values=['', 'NA', 'null'])
    print("\nWith missing value indicators:")
    print(df_missing.isnull().sum())
    
    # Reading CSV with different separators (example with semicolon)
    df_semicolon = df.copy()
    df_semicolon.to_csv('sample_data_semicolon.csv', sep=';', index=False)
    df_read_semicolon = pd.read_csv('sample_data_semicolon.csv', sep=';')
    print("\nRead semicolon-separated file:")
    print(df_read_semicolon.head())


def reading_excel_files():
    """
    Demonstrate reading and writing Excel files with pandas.
    
    Covers reading specific sheets and writing multiple sheets.
    """
    print("=== Reading Excel Files ===\n")
    
    # Create sample data for multiple sheets
    employees = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Age': [25, 30, 35, 28, 32],
        'Salary': [50000, 60000, 75000, 55000, 65000],
        'Department': ['HR', 'IT', 'Sales', 'HR', 'IT']
    }
    
    departments = {
        'Department': ['HR', 'IT', 'Sales', 'Marketing'],
        'Manager': ['Alice', 'Bob', 'Charlie', 'Frank'],
        'Budget': [100000, 200000, 150000, 120000],
        'Employees': [2, 2, 1, 0]
    }
    
    # Create DataFrames
    df_employees = pd.DataFrame(employees)
    df_departments = pd.DataFrame(departments)
    
    print("Employees DataFrame:")
    print(df_employees)
    print("\nDepartments DataFrame:")
    print(df_departments)
    print("\n" + "="*50 + "\n")
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter('company_data.xlsx', engine='openpyxl') as writer:
        df_employees.to_excel(writer, sheet_name='Employees', index=False)
        df_departments.to_excel(writer, sheet_name='Departments', index=False)
    
    print("Saved data to 'company_data.xlsx' with multiple sheets")
    
    # Read specific sheet
    df_emp_read = pd.read_excel('company_data.xlsx', sheet_name='Employees')
    print("\nRead Employees sheet:")
    print(df_emp_read)
    
    # Read all sheets
    all_sheets = pd.read_excel('company_data.xlsx', sheet_name=None)
    print("\nAll sheets:")
    for sheet_name, df in all_sheets.items():
        print(f"\n{sheet_name}:")
        print(df)
    
    # Read with specific options
    df_emp_options = pd.read_excel('company_data.xlsx', 
                                  sheet_name='Employees',
                                  usecols=['Name', 'Age', 'Salary'])
    print("\nRead with column selection:")
    print(df_emp_options)


def reading_other_formats():
    """
    Demonstrate reading and writing other file formats.
    
    Covers JSON, pickle, and other formats.
    """
    print("=== Reading Other File Formats ===\n")
    
    # JSON format
    print("--- JSON Format ---")
    
    # Create sample data
    data_dict = {
        'employees': [
            {'name': 'Alice', 'age': 25, 'department': 'HR'},
            {'name': 'Bob', 'age': 30, 'department': 'IT'},
            {'name': 'Charlie', 'age': 35, 'department': 'Sales'}
        ],
        'company': {
            'name': 'TechCorp',
            'founded': 2010,
            'location': 'San Francisco'
        }
    }
    
    # Save as JSON
    with open('company_data.json', 'w') as f:
        json.dump(data_dict, f, indent=2)
    
    print("Saved data to 'company_data.json'")
    
    # Read JSON
    with open('company_data.json', 'r') as f:
        data_read = json.load(f)
    
    print("\nRead JSON data:")
    print("Employees:", len(data_read['employees']))
    print("Company:", data_read['company']['name'])
    
    # Convert to DataFrame
    df_json = pd.DataFrame(data_read['employees'])
    print("\nConverted to DataFrame:")
    print(df_json)
    
    # Pickle format
    print("\n--- Pickle Format ---")
    
    # Save DataFrame as pickle
    df_pickle = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Score': [85, 92, 78],
        'Grade': ['B', 'A', 'C']
    })
    
    df_pickle.to_pickle('student_scores.pkl')
    print("Saved DataFrame to 'student_scores.pkl'")
    
    # Read pickle
    df_pickle_read = pd.read_pickle('student_scores.pkl')
    print("\nRead pickle file:")
    print(df_pickle_read)
    
    # Using pandas for JSON
    print("\n--- Pandas JSON ---")
    
    # Save DataFrame as JSON
    df_json_pandas = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['NYC', 'LA', 'Chicago']
    })
    
    df_json_pandas.to_json('people.json', orient='records', indent=2)
    print("Saved DataFrame to 'people.json'")
    
    # Read JSON with pandas
    df_json_read = pd.read_json('people.json')
    print("\nRead JSON with pandas:")
    print(df_json_read)


def built_in_datasets():
    """
    Demonstrate loading and exploring built-in datasets.
    
    Covers datasets from seaborn and scikit-learn.
    """
    print("=== Built-in Datasets ===\n")
    
    # Seaborn datasets
    print("--- Seaborn Datasets ---")
    
    # Load iris dataset
    iris = sns.load_dataset('iris')
    print("Iris dataset shape:", iris.shape)
    print("Iris columns:", iris.columns.tolist())
    print("\nFirst few rows:")
    print(iris.head())
    
    # Load tips dataset
    tips = sns.load_dataset('tips')
    print(f"\nTips dataset shape: {tips.shape}")
    print("Tips columns:", tips.columns.tolist())
    print("\nFirst few rows:")
    print(tips.head())
    
    # Load titanic dataset
    titanic = sns.load_dataset('titanic')
    print(f"\nTitanic dataset shape: {titanic.shape}")
    print("Titanic columns:", titanic.columns.tolist())
    print("\nFirst few rows:")
    print(titanic.head())
    
    # Scikit-learn datasets
    print("\n--- Scikit-learn Datasets ---")
    
    try:
        from sklearn.datasets import load_breast_cancer, load_diabetes
        
        # Load breast cancer dataset
        cancer = load_breast_cancer()
        cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        cancer_df['target'] = cancer.target
        
        print("Breast cancer dataset shape:", cancer_df.shape)
        print("Target distribution:")
        print(cancer_df['target'].value_counts())
        
        # Load diabetes dataset
        diabetes = load_diabetes()
        diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        diabetes_df['target'] = diabetes.target
        
        print(f"\nDiabetes dataset shape: {diabetes_df.shape}")
        print("Target statistics:")
        print(diabetes_df['target'].describe())
        
    except ImportError:
        print("scikit-learn not available. Install with: pip install scikit-learn")
    
    # List available seaborn datasets
    print("\n--- Available Seaborn Datasets ---")
    try:
        import seaborn as sns
        datasets = sns.get_dataset_names()
        print("Available datasets:")
        for i, dataset in enumerate(datasets[:10], 1):  # Show first 10
            print(f"{i:2d}. {dataset}")
        if len(datasets) > 10:
            print(f"... and {len(datasets) - 10} more")
    except:
        print("Could not retrieve dataset list")


def basic_data_inspection():
    """
    Demonstrate basic data inspection techniques.
    
    Covers shape, columns, info, head, tail, describe, and data quality checks.
    """
    print("=== Basic Data Inspection ===\n")
    
    # Create sample dataset with various data types and issues
    np.random.seed(42)
    
    data = {
        'id': range(1, 101),
        'name': [f'Person_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'salary': np.random.normal(50000, 15000, 100),
        'department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing', 'Finance'], 100),
        'start_date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'performance_score': np.random.uniform(0, 100, 100),
        'is_manager': np.random.choice([True, False], 100, p=[0.2, 0.8]),
        'email': [f'person{i}@company.com' for i in range(1, 101)]
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some data quality issues
    df.loc[5:10, 'age'] = np.nan  # Missing values
    df.loc[15:20, 'salary'] = -1000  # Negative salary (invalid)
    df.loc[25:30, 'performance_score'] = 150  # Out of range
    df.loc[35:40, 'department'] = 'Unknown'  # Invalid category
    df.loc[45:50, 'name'] = ''  # Empty strings
    
    print("Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    print("\n" + "="*50 + "\n")
    
    # Basic information
    print("Dataset Info:")
    print(df.info())
    
    print("\n" + "="*50 + "\n")
    
    # First and last few rows
    print("First 5 rows:")
    print(df.head())
    
    print("\nLast 5 rows:")
    print(df.tail())
    
    print("\n" + "="*50 + "\n")
    
    # Descriptive statistics
    print("Descriptive Statistics:")
    print(df.describe())
    
    print("\nDescriptive Statistics (include all columns):")
    print(df.describe(include='all'))
    
    print("\n" + "="*50 + "\n")
    
    # Missing values
    print("Missing Values:")
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_percentages
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    print("\n" + "="*50 + "\n")
    
    # Duplicates
    print("Duplicate Rows:")
    duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicates}")
    
    if duplicates > 0:
        print("\nDuplicate rows:")
        print(df[df.duplicated()])
    
    print("\n" + "="*50 + "\n")
    
    # Unique values
    print("Unique Values per Column:")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")
    
    print("\n" + "="*50 + "\n")
    
    # Value counts for categorical variables
    print("Value Counts for Categorical Variables:")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head())
    
    print("\n" + "="*50 + "\n")
    
    # Data type information
    print("Data Type Information:")
    print("Numeric columns:", df.select_dtypes(include=[np.number]).columns.tolist())
    print("Categorical columns:", df.select_dtypes(include=['object', 'category']).columns.tolist())
    print("Datetime columns:", df.select_dtypes(include=['datetime']).columns.tolist())
    print("Boolean columns:", df.select_dtypes(include=['bool']).columns.tolist())


def data_cleaning_techniques():
    """
    Demonstrate various data cleaning techniques.
    
    Covers handling missing values, duplicates, data type conversion, and standardization.
    """
    print("=== Data Cleaning Techniques ===\n")
    
    # Create sample dataset with issues
    np.random.seed(42)
    
    data = {
        'id': range(1, 51),
        'name': [f'Person_{i}' for i in range(1, 51)],
        'age': np.random.randint(18, 80, 50),
        'salary': np.random.normal(50000, 15000, 50),
        'department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], 50),
        'performance_score': np.random.uniform(0, 100, 50),
        'email': [f'person{i}@company.com' for i in range(1, 51)]
    }
    
    df = pd.DataFrame(data)
    
    # Introduce data quality issues
    df.loc[5:10, 'age'] = np.nan
    df.loc[15:20, 'salary'] = -1000
    df.loc[25:30, 'performance_score'] = 150
    df.loc[35:40, 'department'] = 'Unknown'
    df.loc[45:50, 'name'] = ''
    df.loc[10:15, 'email'] = 'invalid_email'
    
    # Add duplicates
    df = pd.concat([df, df.iloc[0:3]], ignore_index=True)
    
    print("Original Dataset (with issues):")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    print("\n" + "="*50 + "\n")
    
    # 1. Handling Missing Values
    print("1. Handling Missing Values:")
    
    # Check missing values
    print("Missing values before cleaning:")
    print(df.isnull().sum())
    
    # Remove rows with missing values
    df_no_missing = df.dropna()
    print(f"\nShape after removing missing values: {df_no_missing.shape}")
    
    # Fill missing values
    df_filled = df.copy()
    df_filled['age'].fillna(df_filled['age'].median(), inplace=True)
    df_filled['salary'].fillna(df_filled['salary'].mean(), inplace=True)
    
    print("\nMissing values after filling:")
    print(df_filled.isnull().sum())
    
    print("\n" + "="*50 + "\n")
    
    # 2. Handling Duplicates
    print("2. Handling Duplicates:")
    
    print(f"Duplicates before cleaning: {df_filled.duplicated().sum()}")
    
    # Remove duplicates
    df_no_duplicates = df_filled.drop_duplicates()
    print(f"Shape after removing duplicates: {df_no_duplicates.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # 3. Data Type Conversion
    print("3. Data Type Conversion:")
    
    print("Data types before conversion:")
    print(df_no_duplicates.dtypes)
    
    # Convert data types
    df_clean = df_no_duplicates.copy()
    df_clean['age'] = df_clean['age'].astype('int64')
    df_clean['salary'] = df_clean['salary'].astype('float64')
    df_clean['department'] = df_clean['department'].astype('category')
    
    print("\nData types after conversion:")
    print(df_clean.dtypes)
    
    print("\n" + "="*50 + "\n")
    
    # 4. Handling Invalid Values
    print("4. Handling Invalid Values:")
    
    # Fix negative salary
    df_clean.loc[df_clean['salary'] < 0, 'salary'] = np.nan
    df_clean['salary'].fillna(df_clean['salary'].median(), inplace=True)
    
    # Fix out-of-range performance scores
    df_clean.loc[df_clean['performance_score'] > 100, 'performance_score'] = 100
    df_clean.loc[df_clean['performance_score'] < 0, 'performance_score'] = 0
    
    # Fix invalid departments
    valid_departments = ['HR', 'IT', 'Sales', 'Marketing']
    df_clean.loc[~df_clean['department'].isin(valid_departments), 'department'] = 'Other'
    
    # Fix empty names
    df_clean.loc[df_clean['name'] == '', 'name'] = 'Unknown'
    
    print("Invalid values fixed:")
    print(f"Negative salaries: {(df_clean['salary'] < 0).sum()}")
    print(f"Out-of-range scores: {((df_clean['performance_score'] < 0) | (df_clean['performance_score'] > 100)).sum()}")
    print(f"Invalid departments: {~df_clean['department'].isin(valid_departments).sum()}")
    
    print("\n" + "="*50 + "\n")
    
    # 5. String Standardization
    print("5. String Standardization:")
    
    # Standardize names
    df_clean['name'] = df_clean['name'].str.strip().str.title()
    
    # Standardize departments
    df_clean['department'] = df_clean['department'].str.upper()
    
    # Clean email addresses
    df_clean['email'] = df_clean['email'].str.lower().str.strip()
    
    print("String standardization applied:")
    print("Sample names:", df_clean['name'].head().tolist())
    print("Sample departments:", df_clean['department'].unique())
    print("Sample emails:", df_clean['email'].head().tolist())
    
    print("\n" + "="*50 + "\n")
    
    # 6. Final Data Quality Check
    print("6. Final Data Quality Check:")
    
    print("Final dataset shape:", df_clean.shape)
    print("\nMissing values:")
    print(df_clean.isnull().sum())
    print(f"\nDuplicates: {df_clean.duplicated().sum()}")
    
    print("\nData type summary:")
    print(df_clean.dtypes)
    
    print("\nValue ranges:")
    print(f"Age: {df_clean['age'].min()} - {df_clean['age'].max()}")
    print(f"Salary: {df_clean['salary'].min():.2f} - {df_clean['salary'].max():.2f}")
    print(f"Performance Score: {df_clean['performance_score'].min():.2f} - {df_clean['performance_score'].max():.2f}")
    
    return df_clean


def selecting_columns():
    """
    Demonstrate various techniques for selecting columns in pandas.
    
    Covers selecting by name, pattern, position, and excluding columns.
    """
    print("=== Selecting Columns ===\n")
    
    # Create sample dataset
    np.random.seed(42)
    
    data = {
        'id': range(1, 21),
        'name': [f'Person_{i}' for i in range(1, 21)],
        'age': np.random.randint(18, 80, 20),
        'salary': np.random.normal(50000, 15000, 20),
        'department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], 20),
        'performance_score': np.random.uniform(0, 100, 20),
        'start_date': pd.date_range('2020-01-01', periods=20, freq='D'),
        'is_manager': np.random.choice([True, False], 20, p=[0.2, 0.8]),
        'email': [f'person{i}@company.com' for i in range(1, 21)],
        'phone': [f'+1-555-{i:03d}-{i:04d}' for i in range(100, 120)]
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n" + "="*50 + "\n")
    
    # 1. Selecting single column
    print("1. Selecting Single Column:")
    
    # Using bracket notation
    names = df['name']
    print("Names column (Series):")
    print(names.head())
    print(f"Type: {type(names)}")
    
    # Using dot notation (only for valid Python identifiers)
    ages = df.age
    print("\nAges column (Series):")
    print(ages.head())
    
    print("\n" + "="*50 + "\n")
    
    # 2. Selecting multiple columns
    print("2. Selecting Multiple Columns:")
    
    # Using list of column names
    basic_info = df[['name', 'age', 'department']]
    print("Basic info columns:")
    print(basic_info.head())
    
    # Using loc
    contact_info = df.loc[:, ['name', 'email', 'phone']]
    print("\nContact info using loc:")
    print(contact_info.head())
    
    print("\n" + "="*50 + "\n")
    
    # 3. Selecting columns by position
    print("3. Selecting Columns by Position:")
    
    # First 3 columns
    first_three = df.iloc[:, :3]
    print("First 3 columns:")
    print(first_three.head())
    
    # Last 3 columns
    last_three = df.iloc[:, -3:]
    print("\nLast 3 columns:")
    print(last_three.head())
    
    # Specific positions
    specific_cols = df.iloc[:, [0, 2, 4, 6]]
    print("\nColumns at positions 0, 2, 4, 6:")
    print(specific_cols.head())
    
    print("\n" + "="*50 + "\n")
    
    # 4. Selecting columns by data type
    print("4. Selecting Columns by Data Type:")
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    print("Numeric columns:")
    print(numeric_cols.head())
    
    # Object (string) columns
    object_cols = df.select_dtypes(include=['object'])
    print("\nObject columns:")
    print(object_cols.head())
    
    # Datetime columns
    datetime_cols = df.select_dtypes(include=['datetime'])
    print("\nDatetime columns:")
    print(datetime_cols.head())
    
    # Boolean columns
    bool_cols = df.select_dtypes(include=['bool'])
    print("\nBoolean columns:")
    print(bool_cols.head())
    
    print("\n" + "="*50 + "\n")
    
    # 5. Selecting columns by pattern
    print("5. Selecting Columns by Pattern:")
    
    # Columns containing 'name'
    name_cols = df.filter(like='name')
    print("Columns containing 'name':")
    print(name_cols.head())
    
    # Columns starting with 'p'
    p_cols = df.filter(regex='^p')
    print("\nColumns starting with 'p':")
    print(p_cols.head())
    
    # Columns ending with 'e'
    e_cols = df.filter(regex='e$')
    print("\nColumns ending with 'e':")
    print(e_cols.head())
    
    print("\n" + "="*50 + "\n")
    
    # 6. Excluding columns
    print("6. Excluding Columns:")
    
    # Exclude specific columns
    exclude_cols = df.drop(['id', 'phone'], axis=1)
    print("Excluding 'id' and 'phone' columns:")
    print(exclude_cols.head())
    
    # Exclude by data type
    exclude_numeric = df.select_dtypes(exclude=[np.number])
    print("\nExcluding numeric columns:")
    print(exclude_numeric.head())
    
    print("\n" + "="*50 + "\n")
    
    # 7. Renaming columns
    print("7. Renaming Columns:")
    
    # Rename specific columns
    df_renamed = df.rename(columns={
        'name': 'full_name',
        'age': 'employee_age',
        'salary': 'annual_salary'
    })
    print("After renaming columns:")
    print(df_renamed.head())
    
    # Rename all columns to lowercase
    df_lower = df.copy()
    df_lower.columns = df_lower.columns.str.lower()
    print("\nAll columns to lowercase:")
    print(df_lower.head())


def filtering_rows():
    """
    Demonstrate various techniques for filtering rows in pandas.
    
    Covers filtering by conditions, missing values, string matching, and between.
    """
    print("=== Filtering Rows ===\n")
    
    # Create sample dataset
    np.random.seed(42)
    
    data = {
        'id': range(1, 31),
        'name': [f'Person_{i}' for i in range(1, 31)],
        'age': np.random.randint(18, 80, 30),
        'salary': np.random.normal(50000, 15000, 30),
        'department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing', 'Finance'], 30),
        'performance_score': np.random.uniform(0, 100, 30),
        'start_date': pd.date_range('2020-01-01', periods=30, freq='D'),
        'is_manager': np.random.choice([True, False], 30, p=[0.2, 0.8]),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], 30)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    df.loc[5:8, 'age'] = np.nan
    df.loc[12:15, 'salary'] = np.nan
    df.loc[20:22, 'department'] = np.nan
    
    print("Original DataFrame:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # 1. Simple filtering
    print("1. Simple Filtering:")
    
    # Filter by single condition
    high_performers = df[df['performance_score'] > 80]
    print("High performers (score > 80):")
    print(high_performers[['name', 'performance_score']].head())
    print(f"Count: {len(high_performers)}")
    
    # Filter by string condition
    it_employees = df[df['department'] == 'IT']
    print("\nIT employees:")
    print(it_employees[['name', 'department']].head())
    print(f"Count: {len(it_employees)}")
    
    print("\n" + "="*50 + "\n")
    
    # 2. Multiple conditions
    print("2. Multiple Conditions:")
    
    # Using AND (&)
    senior_it = df[(df['department'] == 'IT') & (df['age'] > 30)]
    print("Senior IT employees (IT dept AND age > 30):")
    print(senior_it[['name', 'department', 'age']].head())
    print(f"Count: {len(senior_it)}")
    
    # Using OR (|)
    high_salary_or_manager = df[(df['salary'] > 60000) | (df['is_manager'] == True)]
    print("\nHigh salary OR managers:")
    print(high_salary_or_manager[['name', 'salary', 'is_manager']].head())
    print(f"Count: {len(high_salary_or_manager)}")
    
    # Complex conditions
    complex_filter = df[
        (df['department'].isin(['IT', 'Sales'])) & 
        (df['performance_score'] > 70) & 
        (df['age'] < 50)
    ]
    print("\nComplex filter (IT/Sales, high performance, young):")
    print(complex_filter[['name', 'department', 'performance_score', 'age']].head())
    print(f"Count: {len(complex_filter)}")
    
    print("\n" + "="*50 + "\n")
    
    # 3. Filtering with missing values
    print("3. Filtering with Missing Values:")
    
    # Rows with missing age
    missing_age = df[df['age'].isnull()]
    print("Rows with missing age:")
    print(missing_age[['name', 'age']].head())
    print(f"Count: {len(missing_age)}")
    
    # Rows with complete data
    complete_data = df.dropna()
    print(f"\nRows with complete data: {len(complete_data)}")
    
    # Rows with any missing data
    any_missing = df[df.isnull().any(axis=1)]
    print(f"\nRows with any missing data: {len(any_missing)}")
    
    print("\n" + "="*50 + "\n")
    
    # 4. String filtering
    print("4. String Filtering:")
    
    # Contains
    nyc_employees = df[df['city'].str.contains('NYC', na=False)]
    print("NYC employees:")
    print(nyc_employees[['name', 'city']].head())
    
    # Starts with
    p_names = df[df['name'].str.startswith('Person_1')]
    print("\nNames starting with 'Person_1':")
    print(p_names[['name']].head())
    
    # Ends with
    even_ids = df[df['id'].astype(str).str.endswith(('0', '2', '4', '6', '8'))]
    print("\nEven IDs:")
    print(even_ids[['id', 'name']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 5. Between filtering
    print("5. Between Filtering:")
    
    # Using between
    middle_age = df[df['age'].between(30, 50)]
    print("Middle age employees (30-50):")
    print(middle_age[['name', 'age']].head())
    print(f"Count: {len(middle_age)}")
    
    # Using query
    salary_range = df.query('salary >= 40000 and salary <= 70000')
    print("\nSalary range 40k-70k:")
    print(salary_range[['name', 'salary']].head())
    print(f"Count: {len(salary_range)}")
    
    print("\n" + "="*50 + "\n")
    
    # 6. Using query method
    print("6. Using Query Method:")
    
    # Simple query
    managers = df.query('is_manager == True')
    print("Managers:")
    print(managers[['name', 'is_manager']].head())
    
    # Complex query
    complex_query = df.query('department in ["IT", "Sales"] and performance_score > 75')
    print("\nComplex query (IT/Sales with high performance):")
    print(complex_query[['name', 'department', 'performance_score']].head())
    
    # Query with variables
    min_salary = 45000
    max_age = 45
    query_with_vars = df.query(f'salary >= {min_salary} and age <= {max_age}')
    print(f"\nQuery with variables (salary >= {min_salary}, age <= {max_age}):")
    print(query_with_vars[['name', 'salary', 'age']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 7. Using loc for filtering
    print("7. Using loc for Filtering:")
    
    # Using loc with conditions
    high_perf_loc = df.loc[df['performance_score'] > 85, ['name', 'performance_score', 'department']]
    print("High performers using loc:")
    print(high_perf_loc.head())
    
    # Using loc with multiple conditions
    senior_managers = df.loc[
        (df['age'] > 35) & (df['is_manager'] == True), 
        ['name', 'age', 'is_manager', 'department']
    ]
    print("\nSenior managers using loc:")
    print(senior_managers.head())


def creating_new_variables():
    """
    Demonstrate techniques for creating new variables in pandas.
    
    Covers adding columns, conditional variables, and row-wise operations.
    """
    print("=== Creating New Variables ===\n")
    
    # Create sample dataset
    np.random.seed(42)
    
    data = {
        'id': range(1, 21),
        'name': [f'Person_{i}' for i in range(1, 21)],
        'age': np.random.randint(18, 80, 20),
        'salary': np.random.normal(50000, 15000, 20),
        'department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], 20),
        'performance_score': np.random.uniform(0, 100, 20),
        'years_experience': np.random.randint(0, 20, 20)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # 1. Simple arithmetic operations
    print("1. Simple Arithmetic Operations:")
    
    # Add new column with arithmetic
    df['salary_k'] = df['salary'] / 1000
    df['age_squared'] = df['age'] ** 2
    df['performance_percentage'] = df['performance_score'] / 100
    
    print("After adding arithmetic columns:")
    print(df[['name', 'salary', 'salary_k', 'age', 'age_squared', 'performance_score', 'performance_percentage']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 2. Conditional variables
    print("2. Conditional Variables:")
    
    # Using np.where
    df['age_group'] = np.where(df['age'] < 30, 'Young', 
                              np.where(df['age'] < 50, 'Middle', 'Senior'))
    
    # Using np.select for multiple conditions
    conditions = [
        df['performance_score'] >= 90,
        df['performance_score'] >= 80,
        df['performance_score'] >= 70,
        df['performance_score'] >= 60
    ]
    choices = ['Excellent', 'Good', 'Average', 'Below Average']
    df['performance_grade'] = np.select(conditions, choices, default='Poor')
    
    print("After adding conditional columns:")
    print(df[['name', 'age', 'age_group', 'performance_score', 'performance_grade']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 3. String operations
    print("3. String Operations:")
    
    # Extract first name (assuming format "Person_X")
    df['first_name'] = df['name'].str.split('_').str[0]
    df['person_number'] = df['name'].str.split('_').str[1].astype(int)
    
    # Create email
    df['email'] = df['first_name'].str.lower() + df['person_number'].astype(str) + '@company.com'
    
    # Create department code
    df['dept_code'] = df['department'].str[:2].str.upper()
    
    print("After adding string operation columns:")
    print(df[['name', 'first_name', 'person_number', 'email', 'department', 'dept_code']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 4. Date operations
    print("4. Date Operations:")
    
    # Create start date based on years of experience
    current_year = 2024
    df['start_year'] = current_year - df['years_experience']
    df['start_date'] = pd.to_datetime(df['start_year'].astype(str) + '-01-01')
    
    # Calculate tenure in months
    df['tenure_months'] = (current_year - df['start_year']) * 12
    
    print("After adding date operation columns:")
    print(df[['name', 'years_experience', 'start_year', 'start_date', 'tenure_months']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 5. Aggregated variables
    print("5. Aggregated Variables:")
    
    # Department average salary
    dept_avg_salary = df.groupby('department')['salary'].transform('mean')
    df['dept_avg_salary'] = dept_avg_salary
    
    # Salary percentile within department
    df['salary_percentile'] = df.groupby('department')['salary'].rank(pct=True)
    
    # Performance rank
    df['performance_rank'] = df['performance_score'].rank(ascending=False)
    
    print("After adding aggregated columns:")
    print(df[['name', 'department', 'salary', 'dept_avg_salary', 'salary_percentile', 'performance_rank']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 6. Row-wise operations
    print("6. Row-wise Operations:")
    
    # Calculate total compensation (salary + bonus based on performance)
    df['performance_bonus'] = df['performance_score'] * 100  # $100 per point
    df['total_compensation'] = df['salary'] + df['performance_bonus']
    
    # Calculate efficiency score (performance per year of experience)
    df['efficiency_score'] = df['performance_score'] / (df['years_experience'] + 1)  # +1 to avoid division by zero
    
    # Calculate age-adjusted performance
    df['age_adjusted_performance'] = df['performance_score'] * (1 + (df['age'] - 30) / 100)
    
    print("After adding row-wise operation columns:")
    print(df[['name', 'salary', 'performance_bonus', 'total_compensation', 'efficiency_score', 'age_adjusted_performance']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 7. Using apply for complex operations
    print("7. Using Apply for Complex Operations:")
    
    # Create a function for complex categorization
    def categorize_employee(row):
        if row['age'] < 30 and row['performance_score'] > 80:
            return 'Young High Performer'
        elif row['age'] >= 50 and row['years_experience'] > 15:
            return 'Senior Expert'
        elif row['salary'] > 70000:
            return 'High Earner'
        else:
            return 'Regular Employee'
    
    df['employee_category'] = df.apply(categorize_employee, axis=1)
    
    # Create a function for salary band
    def get_salary_band(salary):
        if salary < 40000:
            return 'Low'
        elif salary < 60000:
            return 'Medium'
        elif salary < 80000:
            return 'High'
        else:
            return 'Very High'
    
    df['salary_band'] = df['salary'].apply(get_salary_band)
    
    print("After adding apply-based columns:")
    print(df[['name', 'age', 'performance_score', 'salary', 'employee_category', 'salary_band']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 8. Summary of all new variables
    print("8. Summary of All New Variables:")
    
    new_columns = [col for col in df.columns if col not in ['id', 'name', 'age', 'salary', 'department', 'performance_score', 'years_experience']]
    print(f"Total new columns created: {len(new_columns)}")
    print("New columns:", new_columns)
    
    print("\nFinal DataFrame shape:", df.shape)
    print("\nSample of all columns:")
    print(df.head())


def summarizing_data():
    """
    Demonstrate techniques for summarizing data in pandas.
    
    Covers overall and grouped summary statistics.
    """
    print("=== Summarizing Data ===\n")
    
    # Create sample dataset
    np.random.seed(42)
    
    data = {
        'id': range(1, 51),
        'name': [f'Person_{i}' for i in range(1, 51)],
        'age': np.random.randint(18, 80, 50),
        'salary': np.random.normal(50000, 15000, 50),
        'department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], 50),
        'performance_score': np.random.uniform(0, 100, 50),
        'years_experience': np.random.randint(0, 20, 50),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 50),
        'is_manager': np.random.choice([True, False], 50, p=[0.2, 0.8])
    }
    
    df = pd.DataFrame(data)
    
    print("Sample Dataset:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # 1. Overall summary statistics
    print("1. Overall Summary Statistics:")
    
    # Basic describe
    print("Basic describe:")
    print(df.describe())
    
    print("\nDescribe including all columns:")
    print(df.describe(include='all'))
    
    print("\n" + "="*50 + "\n")
    
    # 2. Summary by data type
    print("2. Summary by Data Type:")
    
    # Numeric columns
    print("Numeric columns summary:")
    numeric_summary = df.select_dtypes(include=[np.number]).describe()
    print(numeric_summary)
    
    # Categorical columns
    print("\nCategorical columns summary:")
    categorical_summary = df.select_dtypes(include=['object', 'category']).describe()
    print(categorical_summary)
    
    print("\n" + "="*50 + "\n")
    
    # 3. Individual column summaries
    print("3. Individual Column Summaries:")
    
    # Value counts for categorical
    print("Department distribution:")
    print(df['department'].value_counts())
    
    print("\nDepartment distribution (with percentages):")
    print(df['department'].value_counts(normalize=True))
    
    print("\nCity distribution:")
    print(df['city'].value_counts())
    
    # Statistics for numeric columns
    print("\nSalary statistics:")
    print(f"Mean: ${df['salary'].mean():.2f}")
    print(f"Median: ${df['salary'].median():.2f}")
    print(f"Std: ${df['salary'].std():.2f}")
    print(f"Min: ${df['salary'].min():.2f}")
    print(f"Max: ${df['salary'].max():.2f}")
    
    print("\nAge statistics:")
    print(f"Mean: {df['age'].mean():.1f}")
    print(f"Median: {df['age'].median():.1f}")
    print(f"Std: {df['age'].std():.1f}")
    
    print("\n" + "="*50 + "\n")
    
    # 4. Grouped summaries
    print("4. Grouped Summaries:")
    
    # Group by department
    dept_summary = df.groupby('department').agg({
        'salary': ['mean', 'median', 'std', 'count'],
        'age': ['mean', 'min', 'max'],
        'performance_score': ['mean', 'min', 'max'],
        'is_manager': 'sum'
    }).round(2)
    
    print("Summary by department:")
    print(dept_summary)
    
    print("\n" + "="*50 + "\n")
    
    # 5. Multiple grouping variables
    print("5. Multiple Grouping Variables:")
    
    # Group by department and city
    dept_city_summary = df.groupby(['department', 'city']).agg({
        'salary': ['mean', 'count'],
        'performance_score': 'mean',
        'age': 'mean'
    }).round(2)
    
    print("Summary by department and city:")
    print(dept_city_summary)
    
    print("\n" + "="*50 + "\n")
    
    # 6. Custom aggregations
    print("6. Custom Aggregations:")
    
    # Custom functions
    def salary_range(x):
        return x.max() - x.min()
    
    def top_performer(x):
        return x.nlargest(1).iloc[0] if len(x) > 0 else np.nan
    
    custom_summary = df.groupby('department').agg({
        'salary': ['mean', salary_range, 'count'],
        'performance_score': ['mean', top_performer],
        'age': ['mean', 'std']
    }).round(2)
    
    print("Custom summary by department:")
    print(custom_summary)
    
    print("\n" + "="*50 + "\n")
    
    # 7. Cross-tabulation
    print("7. Cross-tabulation:")
    
    # Department vs City
    dept_city_crosstab = pd.crosstab(df['department'], df['city'])
    print("Department vs City:")
    print(dept_city_crosstab)
    
    # Department vs Manager status
    dept_manager_crosstab = pd.crosstab(df['department'], df['is_manager'], margins=True)
    print("\nDepartment vs Manager status (with totals):")
    print(dept_manager_crosstab)
    
    # With percentages
    dept_manager_pct = pd.crosstab(df['department'], df['is_manager'], normalize='index')
    print("\nDepartment vs Manager status (row percentages):")
    print(dept_manager_pct.round(3))
    
    print("\n" + "="*50 + "\n")
    
    # 8. Pivot tables
    print("8. Pivot Tables:")
    
    # Simple pivot
    salary_pivot = df.pivot_table(
        values='salary',
        index='department',
        columns='city',
        aggfunc='mean'
    ).round(2)
    
    print("Average salary by department and city:")
    print(salary_pivot)
    
    # Pivot with multiple aggregations
    performance_pivot = df.pivot_table(
        values=['salary', 'performance_score'],
        index='department',
        columns='is_manager',
        aggfunc=['mean', 'count']
    ).round(2)
    
    print("\nPerformance and salary by department and manager status:")
    print(performance_pivot)
    
    print("\n" + "="*50 + "\n")
    
    # 9. Summary statistics with conditions
    print("9. Summary Statistics with Conditions:")
    
    # Managers only
    managers_summary = df[df['is_manager'] == True].describe()
    print("Managers summary:")
    print(managers_summary)
    
    # High performers (top 25%)
    high_performers = df[df['performance_score'] >= df['performance_score'].quantile(0.75)]
    print(f"\nHigh performers summary (top 25%, n={len(high_performers)}):")
    print(high_performers[['salary', 'age', 'years_experience']].describe())
    
    # Young employees
    young_employees = df[df['age'] < 30]
    print(f"\nYoung employees summary (age < 30, n={len(young_employees)}):")
    print(young_employees[['salary', 'performance_score', 'years_experience']].describe())


def arranging_data():
    """
    Demonstrate techniques for arranging and sorting data in pandas.
    
    Covers sorting by columns and computed values.
    """
    print("=== Arranging Data ===\n")
    
    # Create sample dataset
    np.random.seed(42)
    
    data = {
        'id': range(1, 21),
        'name': [f'Person_{i}' for i in range(1, 21)],
        'age': np.random.randint(18, 80, 20),
        'salary': np.random.normal(50000, 15000, 20),
        'department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], 20),
        'performance_score': np.random.uniform(0, 100, 20),
        'years_experience': np.random.randint(0, 20, 20),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 20)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # 1. Basic sorting
    print("1. Basic Sorting:")
    
    # Sort by single column (ascending)
    df_sorted_age = df.sort_values('age')
    print("Sorted by age (ascending):")
    print(df_sorted_age[['name', 'age']].head())
    
    # Sort by single column (descending)
    df_sorted_salary_desc = df.sort_values('salary', ascending=False)
    print("\nSorted by salary (descending):")
    print(df_sorted_salary_desc[['name', 'salary']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 2. Sorting by multiple columns
    print("2. Sorting by Multiple Columns:")
    
    # Sort by department, then by salary within each department
    df_sorted_dept_salary = df.sort_values(['department', 'salary'], ascending=[True, False])
    print("Sorted by department (ascending), then salary (descending):")
    print(df_sorted_dept_salary[['name', 'department', 'salary']].head(10))
    
    # Sort by performance, then by age
    df_sorted_perf_age = df.sort_values(['performance_score', 'age'], ascending=[False, True])
    print("\nSorted by performance (descending), then age (ascending):")
    print(df_sorted_perf_age[['name', 'performance_score', 'age']].head(10))
    
    print("\n" + "="*50 + "\n")
    
    # 3. Sorting with missing values
    print("3. Sorting with Missing Values:")
    
    # Create copy with missing values
    df_missing = df.copy()
    df_missing.loc[5:8, 'salary'] = np.nan
    df_missing.loc[12:15, 'performance_score'] = np.nan
    
    # Sort with missing values at the end
    df_sorted_missing = df_missing.sort_values('salary', na_position='last')
    print("Sorted by salary with missing values at the end:")
    print(df_sorted_missing[['name', 'salary']].head(10))
    
    # Sort with missing values at the beginning
    df_sorted_missing_first = df_missing.sort_values('performance_score', na_position='first')
    print("\nSorted by performance with missing values at the beginning:")
    print(df_sorted_missing_first[['name', 'performance_score']].head(10))
    
    print("\n" + "="*50 + "\n")
    
    # 4. Sorting by computed values
    print("4. Sorting by Computed Values:")
    
    # Sort by salary per year of experience
    df['salary_per_year'] = df['salary'] / (df['years_experience'] + 1)
    df_sorted_efficiency = df.sort_values('salary_per_year', ascending=False)
    print("Sorted by salary per year of experience:")
    print(df_sorted_efficiency[['name', 'salary', 'years_experience', 'salary_per_year']].head())
    
    # Sort by performance percentile
    df['performance_percentile'] = df['performance_score'].rank(pct=True)
    df_sorted_percentile = df.sort_values('performance_percentile', ascending=False)
    print("\nSorted by performance percentile:")
    print(df_sorted_percentile[['name', 'performance_score', 'performance_percentile']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 5. Sorting by string columns
    print("5. Sorting by String Columns:")
    
    # Sort by name
    df_sorted_name = df.sort_values('name')
    print("Sorted by name:")
    print(df_sorted_name[['name', 'department']].head())
    
    # Sort by department, then by name
    df_sorted_dept_name = df.sort_values(['department', 'name'])
    print("\nSorted by department, then by name:")
    print(df_sorted_dept_name[['name', 'department']].head(10))
    
    print("\n" + "="*50 + "\n")
    
    # 6. Sorting with custom functions
    print("6. Sorting with Custom Functions:")
    
    # Sort by last name (assuming format "Person_X")
    df['person_number'] = df['name'].str.split('_').str[1].astype(int)
    df_sorted_number = df.sort_values('person_number')
    print("Sorted by person number:")
    print(df_sorted_number[['name', 'person_number']].head())
    
    # Sort by department length (shortest first)
    df['dept_length'] = df['department'].str.len()
    df_sorted_dept_length = df.sort_values('dept_length')
    print("\nSorted by department name length:")
    print(df_sorted_dept_length[['name', 'department', 'dept_length']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 7. Sorting with group operations
    print("7. Sorting with Group Operations:")
    
    # Sort within each department by salary
    df_sorted_within_dept = df.groupby('department').apply(
        lambda x: x.sort_values('salary', ascending=False)
    ).reset_index(drop=True)
    
    print("Sorted by salary within each department:")
    print(df_sorted_within_dept[['name', 'department', 'salary']].head(10))
    
    # Get top 3 earners from each department
    top_earners_by_dept = df.groupby('department').apply(
        lambda x: x.nlargest(3, 'salary')
    ).reset_index(drop=True)
    
    print("\nTop 3 earners from each department:")
    print(top_earners_by_dept[['name', 'department', 'salary']])
    
    print("\n" + "="*50 + "\n")
    
    # 8. Sorting with reset_index
    print("8. Sorting with Reset Index:")
    
    # Sort and reset index
    df_sorted_reset = df.sort_values('salary', ascending=False).reset_index(drop=True)
    print("Sorted by salary with reset index:")
    print(df_sorted_reset[['name', 'salary']].head())
    
    # Sort and keep original index
    df_sorted_keep_index = df.sort_values('performance_score', ascending=False)
    print("\nSorted by performance (keeping original index):")
    print(df_sorted_keep_index[['name', 'performance_score']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 9. Summary of sorting options
    print("9. Summary of Sorting Options:")
    
    print("Common sorting patterns:")
    print("- sort_values(column): Sort by single column (ascending)")
    print("- sort_values(column, ascending=False): Sort by single column (descending)")
    print("- sort_values([col1, col2]): Sort by multiple columns")
    print("- sort_values(column, na_position='last'): Handle missing values")
    print("- sort_values(column).reset_index(drop=True): Reset index after sorting")


def method_chaining():
    """
    Demonstrate method chaining (pipes) in pandas for readable data pipelines.
    
    Covers creating readable data processing pipelines using method chaining.
    """
    print("=== Method Chaining ===\n")
    
    # Create sample dataset
    np.random.seed(42)
    
    data = {
        'id': range(1, 31),
        'name': [f'Person_{i}' for i in range(1, 31)],
        'age': np.random.randint(18, 80, 30),
        'salary': np.random.normal(50000, 15000, 30),
        'department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], 30),
        'performance_score': np.random.uniform(0, 100, 30),
        'years_experience': np.random.randint(0, 20, 30),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 30)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some data quality issues
    df.loc[5:8, 'age'] = np.nan
    df.loc[12:15, 'salary'] = np.nan
    df.loc[20:22, 'department'] = np.nan
    
    print("Original DataFrame:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    print("\n" + "="*50 + "\n")
    
    # 1. Simple method chaining
    print("1. Simple Method Chaining:")
    
    # Clean and filter data
    clean_data = (df
                  .dropna(subset=['age', 'salary'])  # Remove rows with missing age or salary
                  .query('age >= 18 and salary > 0')  # Filter valid age and salary
                  .sort_values('salary', ascending=False)  # Sort by salary
                  .head(10))  # Get top 10
    
    print("Clean data (top 10 by salary):")
    print(clean_data[['name', 'age', 'salary', 'department']])
    
    print("\n" + "="*50 + "\n")
    
    # 2. Data transformation pipeline
    print("2. Data Transformation Pipeline:")
    
    # Create new variables and filter
    transformed_data = (df
                       .assign(
                           age_group=lambda x: pd.cut(x['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior']),
                           salary_k=lambda x: x['salary'] / 1000,
                           performance_grade=lambda x: pd.cut(x['performance_score'], 
                                                            bins=[0, 60, 80, 100], 
                                                            labels=['Poor', 'Good', 'Excellent'])
                       )
                       .dropna(subset=['age_group', 'performance_grade'])
                       .query('salary_k > 40')  # Salary > 40k
                       .sort_values(['department', 'salary'], ascending=[True, False]))
    
    print("Transformed data:")
    print(transformed_data[['name', 'department', 'age_group', 'salary_k', 'performance_grade']].head(10))
    
    print("\n" + "="*50 + "\n")
    
    # 3. Aggregation pipeline
    print("3. Aggregation Pipeline:")
    
    # Group and summarize
    summary_stats = (df
                    .dropna(subset=['department', 'salary'])
                    .groupby('department')
                    .agg({
                        'salary': ['mean', 'median', 'std', 'count'],
                        'age': ['mean', 'min', 'max'],
                        'performance_score': ['mean', 'min', 'max']
                    })
                    .round(2)
                    .sort_values(('salary', 'mean'), ascending=False))
    
    print("Summary statistics by department:")
    print(summary_stats)
    
    print("\n" + "="*50 + "\n")
    
    # 4. Complex data cleaning pipeline
    print("4. Complex Data Cleaning Pipeline:")
    
    # Comprehensive cleaning
    cleaned_data = (df
                   .copy()
                   .assign(
                       # Fill missing ages with median
                       age=lambda x: x['age'].fillna(x['age'].median()),
                       # Fill missing salaries with department mean
                       salary=lambda x: x.groupby('department')['salary'].transform(
                           lambda s: s.fillna(s.mean())
                       ),
                       # Fill missing departments with 'Unknown'
                       department=lambda x: x['department'].fillna('Unknown'),
                       # Create derived variables
                       salary_per_year=lambda x: x['salary'] / (x['years_experience'] + 1),
                       age_group=lambda x: pd.cut(x['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
                   )
                   .query('salary > 0 and age >= 18')  # Remove invalid data
                   .drop_duplicates(subset=['name'])  # Remove duplicates
                   .sort_values('salary', ascending=False))
    
    print("Comprehensive cleaning results:")
    print(f"Final shape: {cleaned_data.shape}")
    print(f"Missing values:\n{cleaned_data.isnull().sum()}")
    print("\nSample of cleaned data:")
    print(cleaned_data[['name', 'age', 'salary', 'department', 'salary_per_year', 'age_group']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 5. Analysis pipeline
    print("5. Analysis Pipeline:")
    
    # Complete analysis workflow
    analysis_results = (df
                       .dropna(subset=['department', 'salary', 'performance_score'])
                       .assign(
                           performance_quartile=lambda x: pd.qcut(x['performance_score'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4']),
                           salary_band=lambda x: pd.cut(x['salary'], bins=[0, 40000, 60000, 80000, 100000], 
                                                      labels=['Low', 'Medium', 'High', 'Very High'])
                       )
                       .groupby(['department', 'performance_quartile'])
                       .agg({
                           'salary': ['mean', 'count'],
                           'age': 'mean',
                           'years_experience': 'mean'
                       })
                       .round(2)
                       .sort_values(['department', ('salary', 'mean')], ascending=[True, False]))
    
    print("Analysis results by department and performance quartile:")
    print(analysis_results)
    
    print("\n" + "="*50 + "\n")
    
    # 6. Visualization pipeline
    print("6. Visualization Pipeline:")
    
    # Prepare data for visualization
    viz_data = (df
                .dropna(subset=['department', 'salary'])
                .groupby('department')
                .agg({
                    'salary': 'mean',
                    'performance_score': 'mean',
                    'age': 'mean'
                })
                .round(2)
                .reset_index()
                .melt(id_vars=['department'], 
                      value_vars=['salary', 'performance_score', 'age'],
                      var_name='metric', 
                      value_name='value'))
    
    print("Data prepared for visualization:")
    print(viz_data)
    
    print("\n" + "="*50 + "\n")
    
    # 7. Export pipeline
    print("7. Export Pipeline:")
    
    # Process and export data
    export_data = (df
                   .dropna(subset=['name', 'salary', 'department'])
                   .assign(
                       salary_k=lambda x: x['salary'] / 1000,
                       performance_grade=lambda x: np.where(x['performance_score'] >= 80, 'High', 
                                                          np.where(x['performance_score'] >= 60, 'Medium', 'Low'))
                   )
                   .filter(['name', 'department', 'salary_k', 'performance_grade', 'age'])
                   .sort_values(['department', 'salary_k'], ascending=[True, False])
                   .reset_index(drop=True))
    
    print("Data prepared for export:")
    print(export_data.head(10))
    
    # Save to CSV (commented out to avoid file creation)
    # export_data.to_csv('processed_employee_data.csv', index=False)
    print("\nData would be saved to 'processed_employee_data.csv'")


def data_reshaping_basics():
    """
    Demonstrate basic data reshaping techniques in pandas.
    
    Covers wide vs. long format, melting and pivoting data.
    """
    print("=== Data Reshaping Basics ===\n")
    
    # Create sample wide format data
    np.random.seed(42)
    
    # Wide format data (one row per subject, multiple measurements)
    wide_data = {
        'subject_id': range(1, 11),
        'name': [f'Person_{i}' for i in range(1, 11)],
        'age': np.random.randint(20, 60, 10),
        'test1_score': np.random.uniform(60, 100, 10),
        'test2_score': np.random.uniform(60, 100, 10),
        'test3_score': np.random.uniform(60, 100, 10),
        'test4_score': np.random.uniform(60, 100, 10)
    }
    
    df_wide = pd.DataFrame(wide_data)
    
    print("Wide Format Data (one row per subject):")
    print(df_wide)
    print(f"\nShape: {df_wide.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # 1. Melting data (wide to long)
    print("1. Melting Data (Wide to Long):")
    
    # Melt test scores
    df_long = df_wide.melt(
        id_vars=['subject_id', 'name', 'age'],
        value_vars=['test1_score', 'test2_score', 'test3_score', 'test4_score'],
        var_name='test',
        value_name='score'
    )
    
    print("Long Format Data (one row per test per subject):")
    print(df_long.head(10))
    print(f"\nShape: {df_long.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # 2. Pivoting data (long to wide)
    print("2. Pivoting Data (Long to Wide):")
    
    # Pivot back to wide format
    df_wide_again = df_long.pivot(
        index=['subject_id', 'name', 'age'],
        columns='test',
        values='score'
    ).reset_index()
    
    print("Back to Wide Format:")
    print(df_wide_again)
    print(f"\nShape: {df_wide_again.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # 3. Pivot table with aggregation
    print("3. Pivot Table with Aggregation:")
    
    # Create data with multiple measurements per subject
    repeated_data = []
    for i in range(1, 6):  # 5 subjects
        for test in ['Math', 'Science', 'English']:
            for week in [1, 2, 3, 4]:  # 4 weeks
                repeated_data.append({
                    'subject_id': i,
                    'name': f'Person_{i}',
                    'test': test,
                    'week': week,
                    'score': np.random.uniform(60, 100)
                })
    
    df_repeated = pd.DataFrame(repeated_data)
    
    print("Repeated Measures Data:")
    print(df_repeated.head(10))
    print(f"\nShape: {df_repeated.shape}")
    
    # Pivot table - average score by test and week
    pivot_avg = df_repeated.pivot_table(
        values='score',
        index='test',
        columns='week',
        aggfunc='mean'
    ).round(2)
    
    print("\nAverage Score by Test and Week:")
    print(pivot_avg)
    
    # Pivot table - multiple aggregations
    pivot_multi = df_repeated.pivot_table(
        values='score',
        index='test',
        columns='week',
        aggfunc=['mean', 'std', 'count']
    ).round(2)
    
    print("\nMultiple Aggregations by Test and Week:")
    print(pivot_multi)
    
    print("\n" + "="*50 + "\n")
    
    # 4. Separating and uniting columns
    print("4. Separating and Uniting Columns:")
    
    # Create data with combined columns
    combined_data = {
        'subject_id': range(1, 11),
        'name': [f'Person_{i}' for i in range(1, 11)],
        'location': ['NYC_Office_A', 'LA_Office_B', 'Chicago_Office_C', 'Houston_Office_D', 'Phoenix_Office_E'] * 2,
        'test_results': ['Math:85,Science:92,English:78', 'Math:92,Science:88,English:85', 
                        'Math:78,Science:85,English:90', 'Math:88,Science:90,English:82',
                        'Math:95,Science:87,English:88'] * 2
    }
    
    df_combined = pd.DataFrame(combined_data)
    
    print("Data with Combined Columns:")
    print(df_combined)
    
    # Separate location column
    df_separated = df_combined.copy()
    df_separated[['city', 'office_type', 'office_id']] = df_separated['location'].str.split('_', expand=True)
    
    print("\nAfter Separating Location:")
    print(df_separated[['subject_id', 'name', 'city', 'office_type', 'office_id', 'test_results']])
    
    # Unite columns back
    df_united = df_separated.copy()
    df_united['location_combined'] = df_united['city'] + '_' + df_united['office_type'] + '_' + df_united['office_id']
    
    print("\nAfter Uniting Location:")
    print(df_united[['subject_id', 'name', 'location_combined', 'test_results']])
    
    print("\n" + "="*50 + "\n")
    
    # 5. Stacking and unstacking
    print("5. Stacking and Unstacking:")
    
    # Create multi-index data
    multi_data = {
        'subject_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'test': ['Math', 'Science', 'English'] * 3,
        'score': np.random.uniform(60, 100, 9),
        'time': ['Pre', 'Post'] * 4 + ['Pre']
    }
    
    df_multi = pd.DataFrame(multi_data)
    
    print("Multi-index Data:")
    print(df_multi)
    
    # Set multi-index
    df_multi_index = df_multi.set_index(['subject_id', 'test'])
    
    print("\nMulti-index DataFrame:")
    print(df_multi_index)
    
    # Unstack
    df_unstacked = df_multi_index.unstack('test')
    
    print("\nUnstacked (test as columns):")
    print(df_unstacked)
    
    # Stack back
    df_stacked = df_unstacked.stack('test')
    
    print("\nStacked back:")
    print(df_stacked)
    
    print("\n" + "="*50 + "\n")
    
    # 6. Cross-tabulation for reshaping
    print("6. Cross-tabulation for Reshaping:")
    
    # Create categorical data
    survey_data = {
        'subject_id': range(1, 21),
        'gender': np.random.choice(['Male', 'Female'], 20),
        'age_group': np.random.choice(['18-25', '26-35', '36-50', '50+'], 20),
        'satisfaction': np.random.choice(['Low', 'Medium', 'High'], 20),
        'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], 20)
    }
    
    df_survey = pd.DataFrame(survey_data)
    
    print("Survey Data:")
    print(df_survey.head())
    
    # Cross-tabulation
    gender_age_crosstab = pd.crosstab(df_survey['gender'], df_survey['age_group'])
    
    print("\nGender vs Age Group:")
    print(gender_age_crosstab)
    
    # Cross-tabulation with multiple variables
    dept_satisfaction_crosstab = pd.crosstab(
        [df_survey['department'], df_survey['gender']], 
        df_survey['satisfaction'],
        margins=True
    )
    
    print("\nDepartment and Gender vs Satisfaction:")
    print(dept_satisfaction_crosstab)


def working_with_dates():
    """
    Demonstrate working with dates and times in pandas.
    
    Covers creating, parsing, extracting components, and date arithmetic.
    """
    print("=== Working with Dates ===\n")
    
    # Create sample data with dates
    np.random.seed(42)
    
    data = {
        'id': range(1, 21),
        'name': [f'Person_{i}' for i in range(1, 21)],
        'birth_date': pd.date_range('1980-01-01', periods=20, freq='D'),
        'hire_date': pd.date_range('2010-01-01', periods=20, freq='M'),
        'last_login': pd.date_range('2024-01-01', periods=20, freq='H'),
        'project_start': pd.date_range('2023-06-01', periods=20, freq='W'),
        'deadline': pd.date_range('2024-12-01', periods=20, freq='D')
    }
    
    df = pd.DataFrame(data)
    
    print("Sample Data with Dates:")
    print(df.head())
    print(f"\nData types:\n{df.dtypes}")
    
    print("\n" + "="*50 + "\n")
    
    # 1. Creating dates
    print("1. Creating Dates:")
    
    # From strings
    date_strings = ['2024-01-15', '2024-02-20', '2024-03-10']
    dates_from_strings = pd.to_datetime(date_strings)
    print("Dates from strings:")
    print(dates_from_strings)
    
    # From different formats
    different_formats = ['01/15/2024', '15-01-2024', '2024.01.15']
    dates_different = pd.to_datetime(different_formats, format='mixed')
    print("\nDates from different formats:")
    print(dates_different)
    
    # Date ranges
    date_range_daily = pd.date_range('2024-01-01', '2024-01-10', freq='D')
    date_range_monthly = pd.date_range('2024-01-01', periods=12, freq='M')
    date_range_yearly = pd.date_range('2020-01-01', periods=5, freq='Y')
    
    print("\nDate ranges:")
    print("Daily:", date_range_daily.tolist())
    print("Monthly:", date_range_monthly.tolist())
    print("Yearly:", date_range_yearly.tolist())
    
    print("\n" + "="*50 + "\n")
    
    # 2. Extracting date components
    print("2. Extracting Date Components:")
    
    # Extract various components
    df['birth_year'] = df['birth_date'].dt.year
    df['birth_month'] = df['birth_date'].dt.month
    df['birth_day'] = df['birth_date'].dt.day
    df['birth_weekday'] = df['birth_date'].dt.day_name()
    df['birth_quarter'] = df['birth_date'].dt.quarter
    df['birth_week'] = df['birth_date'].dt.isocalendar().week
    
    print("Date components extracted:")
    print(df[['name', 'birth_date', 'birth_year', 'birth_month', 'birth_day', 
              'birth_weekday', 'birth_quarter', 'birth_week']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 3. Date arithmetic
    print("3. Date Arithmetic:")
    
    # Calculate age
    current_date = pd.Timestamp('2024-01-01')
    df['age'] = (current_date - df['birth_date']).dt.days / 365.25
    
    # Calculate tenure
    df['tenure_days'] = (current_date - df['hire_date']).dt.days
    df['tenure_years'] = df['tenure_days'] / 365.25
    
    # Calculate time until deadline
    df['days_until_deadline'] = (df['deadline'] - current_date).dt.days
    
    print("Date arithmetic results:")
    print(df[['name', 'birth_date', 'age', 'hire_date', 'tenure_years', 
              'deadline', 'days_until_deadline']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 4. Date filtering
    print("4. Date Filtering:")
    
    # Filter by year
    recent_hires = df[df['hire_date'].dt.year >= 2020]
    print("Recent hires (2020 or later):")
    print(recent_hires[['name', 'hire_date']].head())
    
    # Filter by month
    january_births = df[df['birth_date'].dt.month == 1]
    print(f"\nJanuary births (n={len(january_births)}):")
    print(january_births[['name', 'birth_date']].head())
    
    # Filter by date range
    date_range_filter = df[
        (df['birth_date'] >= '1985-01-01') & 
        (df['birth_date'] <= '1995-12-31')
    ]
    print(f"\nBorn between 1985-1995 (n={len(date_range_filter)}):")
    print(date_range_filter[['name', 'birth_date']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 5. Date formatting
    print("5. Date Formatting:")
    
    # Different date formats
    df['birth_date_formatted'] = df['birth_date'].dt.strftime('%B %d, %Y')
    df['hire_date_short'] = df['hire_date'].dt.strftime('%Y-%m')
    df['last_login_time'] = df['last_login'].dt.strftime('%Y-%m-%d %H:%M')
    
    print("Different date formats:")
    print(df[['name', 'birth_date', 'birth_date_formatted', 'hire_date', 
              'hire_date_short', 'last_login', 'last_login_time']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 6. Working with timezones
    print("6. Working with Timezones:")
    
    # Create timezone-aware dates
    utc_dates = pd.date_range('2024-01-01', periods=5, freq='D', tz='UTC')
    print("UTC dates:")
    print(utc_dates)
    
    # Convert to different timezone
    est_dates = utc_dates.tz_convert('US/Eastern')
    print("\nEST dates:")
    print(est_dates)
    
    # Localize naive dates
    naive_dates = pd.date_range('2024-01-01', periods=5, freq='D')
    localized_dates = naive_dates.tz_localize('UTC')
    print("\nLocalized dates:")
    print(localized_dates)
    
    print("\n" + "="*50 + "\n")
    
    # 7. Date aggregation
    print("7. Date Aggregation:")
    
    # Group by year
    yearly_stats = df.groupby(df['birth_date'].dt.year).agg({
        'id': 'count',
        'age': 'mean'
    }).rename(columns={'id': 'count', 'age': 'avg_age'})
    
    print("Yearly birth statistics:")
    print(yearly_stats)
    
    # Group by month
    monthly_stats = df.groupby(df['birth_date'].dt.month).agg({
        'id': 'count'
    }).rename(columns={'id': 'birth_count'})
    
    print("\nMonthly birth statistics:")
    print(monthly_stats)
    
    # Group by weekday
    weekday_stats = df.groupby(df['birth_date'].dt.day_name()).agg({
        'id': 'count'
    }).rename(columns={'id': 'birth_count'})
    
    print("\nWeekday birth statistics:")
    print(weekday_stats)
    
    print("\n" + "="*50 + "\n")
    
    # 8. Business days and holidays
    print("8. Business Days and Holidays:")
    
    # Calculate business days between dates
    df['business_days_to_deadline'] = pd.bdate_range(
        current_date, 
        df['deadline'].max()
    ).shape[0]
    
    # Check if dates are business days
    df['is_business_day'] = df['birth_date'].dt.dayofweek < 5
    
    print("Business day calculations:")
    print(df[['name', 'birth_date', 'is_business_day', 'deadline', 'business_days_to_deadline']].head())


def string_manipulation():
    """
    Demonstrate string manipulation techniques in pandas.
    
    Covers basic operations, pattern matching, extraction, and regular expressions.
    """
    print("=== String Manipulation ===\n")
    
    # Create sample data with strings
    np.random.seed(42)
    
    data = {
        'id': range(1, 21),
        'name': [f'Person_{i}' for i in range(1, 21)],
        'email': [f'person{i}@company.com' for i in range(1, 21)],
        'phone': [f'+1-555-{i:03d}-{i:04d}' for i in range(100, 120)],
        'address': [
            '123 Main St, New York, NY 10001',
            '456 Oak Ave, Los Angeles, CA 90210',
            '789 Pine Rd, Chicago, IL 60601',
            '321 Elm St, Houston, TX 77001',
            '654 Maple Dr, Phoenix, AZ 85001'
        ] * 4,
        'product_code': [f'PROD-{chr(65+i%26)}-{i:03d}' for i in range(1, 21)],
        'description': [
            'High-quality product with excellent features',
            'Premium item with advanced technology',
            'Standard model with basic functionality',
            'Deluxe version with enhanced performance',
            'Economy option with essential features'
        ] * 4
    }
    
    df = pd.DataFrame(data)
    
    print("Sample Data with Strings:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # 1. Basic string operations
    print("1. Basic String Operations:")
    
    # Case conversion
    df['name_upper'] = df['name'].str.upper()
    df['name_lower'] = df['name'].str.lower()
    df['name_title'] = df['name'].str.title()
    
    # Length
    df['name_length'] = df['name'].str.len()
    df['description_length'] = df['description'].str.len()
    
    # Strip whitespace
    df['name_stripped'] = df['name'].str.strip()
    
    print("Basic string operations:")
    print(df[['name', 'name_upper', 'name_lower', 'name_title', 'name_length', 'name_stripped']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 2. String splitting and joining
    print("2. String Splitting and Joining:")
    
    # Split names
    df[['first_part', 'second_part']] = df['name'].str.split('_', expand=True)
    
    # Split addresses
    address_parts = df['address'].str.split(', ', expand=True)
    df['street'] = address_parts[0]
    df['city'] = address_parts[1]
    df['state_zip'] = address_parts[2]
    
    # Split state and zip
    df[['state', 'zip_code']] = df['state_zip'].str.split(' ', expand=True)
    
    # Join parts back
    df['full_address'] = df['street'] + ', ' + df['city'] + ', ' + df['state'] + ' ' + df['zip_code']
    
    print("String splitting and joining:")
    print(df[['name', 'first_part', 'second_part', 'address', 'street', 'city', 'state', 'zip_code']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 3. String replacement
    print("3. String Replacement:")
    
    # Simple replacement
    df['email_clean'] = df['email'].str.replace('@company.com', '@newcompany.com')
    
    # Multiple replacements
    df['description_clean'] = (df['description']
                              .str.replace('High-quality', 'Premium')
                              .str.replace('Standard', 'Basic')
                              .str.replace('Deluxe', 'Advanced'))
    
    # Replace with regex
    df['phone_clean'] = df['phone'].str.replace(r'[^\d]', '', regex=True)
    
    print("String replacement:")
    print(df[['email', 'email_clean', 'description', 'description_clean', 'phone', 'phone_clean']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 4. Pattern matching
    print("4. Pattern Matching:")
    
    # Contains
    df['has_quality'] = df['description'].str.contains('quality', case=False)
    df['has_technology'] = df['description'].str.contains('technology', case=False)
    
    # Starts with
    df['starts_with_prod'] = df['product_code'].str.startswith('PROD')
    
    # Ends with
    df['ends_with_com'] = df['email'].str.endswith('.com')
    
    # Match pattern
    df['valid_email'] = df['email'].str.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    print("Pattern matching:")
    print(df[['description', 'has_quality', 'has_technology', 'product_code', 'starts_with_prod', 
              'email', 'ends_with_com', 'valid_email']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 5. String extraction
    print("5. String Extraction:")
    
    # Extract numbers
    df['person_number'] = df['name'].str.extract(r'(\d+)').astype(int)
    
    # Extract letters
    df['product_letter'] = df['product_code'].str.extract(r'PROD-([A-Z])-')
    
    # Extract multiple groups
    df[['prod_prefix', 'prod_letter', 'prod_number']] = df['product_code'].str.extract(r'(PROD)-([A-Z])-(\d+)')
    
    # Extract with regex groups
    df['area_code'] = df['phone'].str.extract(r'\+1-555-(\d{3})')
    
    print("String extraction:")
    print(df[['name', 'person_number', 'product_code', 'product_letter', 'prod_prefix', 
              'prod_letter', 'prod_number', 'phone', 'area_code']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 6. Regular expressions
    print("6. Regular Expressions:")
    
    # Complex pattern matching
    df['has_digit'] = df['description'].str.contains(r'\d', regex=True)
    df['has_word_boundary'] = df['description'].str.contains(r'\bwith\b', regex=True)
    
    # Multiple patterns
    df['quality_indicator'] = df['description'].str.contains(r'quality|premium|deluxe', regex=True, case=False)
    
    # Pattern replacement
    df['description_simplified'] = df['description'].str.replace(r'\bwith\b', 'featuring', regex=True)
    
    # Complex extraction
    df['word_count'] = df['description'].str.count(r'\b\w+\b')
    
    print("Regular expressions:")
    print(df[['description', 'has_digit', 'has_word_boundary', 'quality_indicator', 
              'description_simplified', 'word_count']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 7. String formatting
    print("7. String Formatting:")
    
    # Format strings
    df['formatted_name'] = df['name'].str.format()
    df['formatted_product'] = df['product_code'].str.replace('PROD', 'Product')
    
    # Padding
    df['padded_id'] = df['id'].astype(str).str.zfill(3)
    df['padded_name'] = df['name'].str.pad(width=15, side='right', fillchar='-')
    
    # Truncation
    df['truncated_description'] = df['description'].str[:20] + '...'
    
    print("String formatting:")
    print(df[['id', 'padded_id', 'name', 'padded_name', 'product_code', 'formatted_product',
              'description', 'truncated_description']].head())
    
    print("\n" + "="*50 + "\n")
    
    # 8. String aggregation
    print("8. String Aggregation:")
    
    # Concatenate strings
    df['full_info'] = df['name'] + ' (' + df['email'] + ') - ' + df['product_code']
    
    # Join multiple columns
    df['contact_info'] = df['name'] + ' | ' + df['email'] + ' | ' + df['phone']
    
    # Conditional string creation
    df['status'] = np.where(df['has_quality'], 'High Quality', 'Standard')
    df['category'] = np.where(df['description'].str.contains('Premium'), 'Premium',
                             np.where(df['description'].str.contains('Economy'), 'Economy', 'Standard'))
    
    print("String aggregation:")
    print(df[['name', 'email', 'product_code', 'full_info', 'contact_info', 'status', 'category']].head())


def student_performance_analysis():
    """
    Practical Example 1: Student Performance Analysis
    - Calculates averages, identifies top performers, subject-wise analysis, and correlation.
    """
    print("=== Student Performance Analysis ===\n")
    np.random.seed(42)
    n_students = 30
    subjects = ['Math', 'Science', 'English', 'History']
    data = {
        'student_id': range(1, n_students + 1),
        'name': [f'Student_{i}' for i in range(1, n_students + 1)],
        'Math': np.random.randint(60, 100, n_students),
        'Science': np.random.randint(55, 100, n_students),
        'English': np.random.randint(65, 100, n_students),
        'History': np.random.randint(50, 100, n_students)
    }
    df = pd.DataFrame(data)
    print("Sample data:")
    print(df.head())
    print("\n" + "="*50 + "\n")
    # Calculate average score
    df['average'] = df[subjects].mean(axis=1)
    print("Averages calculated:")
    print(df[['name', 'average']].head())
    # Identify top performers
    top_students = df.nlargest(5, 'average')
    print("\nTop 5 students:")
    print(top_students[['name', 'average']])
    # Subject-wise statistics
    print("\nSubject-wise statistics:")
    print(df[subjects].describe())
    # Correlation analysis
    print("\nSubject score correlations:")
    print(df[subjects].corr())
    print("\n" + "="*50 + "\n")
    # Visualize
    import matplotlib.pyplot as plt
    df[subjects].plot(kind='box', title='Score Distribution by Subject')
    plt.ylabel('Score')
    plt.show()
    df.plot(x='name', y=subjects, kind='bar', figsize=(12, 5), title='Scores by Student')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.show()

def sales_data_analysis():
    """
    Practical Example 2: Sales Data Analysis
    - Grouped and time series analysis, product performance metrics.
    """
    print("=== Sales Data Analysis ===\n")
    np.random.seed(123)
    n_sales = 100
    months = pd.date_range('2023-01-01', periods=12, freq='M')
    data = {
        'sale_id': range(1, n_sales + 1),
        'date': np.random.choice(months, n_sales),
        'product': np.random.choice(['A', 'B', 'C', 'D'], n_sales),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_sales),
        'units_sold': np.random.poisson(20, n_sales),
        'unit_price': np.random.uniform(10, 50, n_sales)
    }
    df = pd.DataFrame(data)
    df['revenue'] = df['units_sold'] * df['unit_price']
    print("Sample sales data:")
    print(df.head())
    print("\n" + "="*50 + "\n")
    # Grouped analysis
    sales_by_product = df.groupby('product').agg({'units_sold': 'sum', 'revenue': 'sum'}).sort_values('revenue', ascending=False)
    print("Sales by product:")
    print(sales_by_product)
    # Time series analysis
    sales_by_month = df.groupby(df['date'].dt.to_period('M')).agg({'revenue': 'sum'})
    print("\nSales by month:")
    print(sales_by_month)
    # Product performance metrics
    print("\nProduct performance metrics:")
    print(df.groupby('product').agg({'unit_price': ['mean', 'std'], 'units_sold': 'mean'}))
    print("\n" + "="*50 + "\n")
    # Visualize
    import matplotlib.pyplot as plt
    sales_by_product['revenue'].plot(kind='bar', title='Revenue by Product')
    plt.ylabel('Revenue')
    plt.show()
    sales_by_month['revenue'].plot(kind='line', marker='o', title='Monthly Revenue')
    plt.ylabel('Revenue')
    plt.show()

def data_quality_assessment():
    """
    Practical Example 3: Data Quality Assessment
    - Assessing and cleaning data quality issues.
    """
    print("=== Data Quality Assessment ===\n")
    np.random.seed(456)
    n = 50
    data = {
        'id': range(1, n + 1),
        'age': np.random.choice([25, 30, 35, 40, np.nan], n),
        'income': np.random.choice([40000, 50000, 60000, np.nan], n),
        'gender': np.random.choice(['M', 'F', None], n),
        'score': np.random.uniform(0, 100, n)
    }
    df = pd.DataFrame(data)
    # Introduce duplicates
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    print("Sample data with issues:")
    print(df.head(10))
    print("\n" + "="*50 + "\n")
    # Assess missing values
    print("Missing values:")
    print(df.isnull().sum())
    # Assess duplicates
    print("\nDuplicates:")
    print(df.duplicated().sum())
    # Assess invalid values
    print("\nInvalid ages:")
    print(df['age'][df['age'] < 0])
    # Clean data
    df_clean = df.drop_duplicates().copy()
    df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
    df_clean['income'].fillna(df_clean['income'].mean(), inplace=True)
    df_clean['gender'].fillna('Unknown', inplace=True)
    print("\nCleaned data sample:")
    print(df_clean.head())
    print("\n" + "="*50 + "\n")
    # Visualize
    import matplotlib.pyplot as plt
    df_clean['age'].plot(kind='hist', bins=10, title='Age Distribution')
    plt.xlabel('Age')
    plt.show()
    df_clean['income'].plot(kind='box', title='Income Boxplot')
    plt.ylabel('Income')
    plt.show()


def file_organization():
    """
    Best Practice 1: File Organization
    - Creating organized folder structures for data projects.
    """
    print("=== File Organization Best Practices ===\n")
    
    # Create organized folder structure
    folders = [
        'data/raw',
        'data/processed', 
        'data/interim',
        'src',
        'notebooks',
        'reports',
        'logs'
    ]
    
    print("Recommended folder structure:")
    for folder in folders:
        print(f"  {folder}/")
    
    print("\n" + "="*50 + "\n")
    
    # Example file organization
    print("Example file organization:")
    print("data/raw/")
    print("  - original_data.csv")
    print("  - survey_responses.xlsx")
    print("data/processed/")
    print("  - cleaned_data.csv")
    print("  - aggregated_data.csv")
    print("src/")
    print("  - data_cleaning.py")
    print("  - analysis_functions.py")
    print("notebooks/")
    print("  - exploratory_analysis.ipynb")
    print("  - final_report.ipynb")
    print("reports/")
    print("  - analysis_report.pdf")
    print("  - data_quality_report.html")
    
    print("\n" + "="*50 + "\n")
    
    # Create sample project structure
    import os
    print("Creating sample project structure...")
    
    # Create directories (commented out to avoid actual creation)
    # for folder in folders:
    #     os.makedirs(folder, exist_ok=True)
    #     print(f"Created: {folder}/")
    
    print("Project structure would be created with:")
    print("- data/raw/ for original data files")
    print("- data/processed/ for cleaned and transformed data")
    print("- src/ for Python scripts and functions")
    print("- notebooks/ for Jupyter notebooks")
    print("- reports/ for output reports and visualizations")
    print("- logs/ for logging files")


def data_validation():
    """
    Best Practice 2: Data Validation
    - Comprehensive data quality checks and validation rules.
    """
    print("=== Data Validation Best Practices ===\n")
    
    # Create sample data for validation
    np.random.seed(789)
    n = 100
    
    data = {
        'id': range(1, n + 1),
        'name': [f'Person_{i}' for i in range(1, n + 1)],
        'age': np.random.randint(18, 80, n),
        'email': [f'person{i}@company.com' for i in range(1, n + 1)],
        'salary': np.random.normal(50000, 15000, n),
        'department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], n),
        'start_date': pd.date_range('2020-01-01', periods=n, freq='D')
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some data quality issues
    df.loc[5, 'age'] = -5  # Invalid age
    df.loc[10, 'age'] = 150  # Invalid age
    df.loc[15, 'salary'] = -1000  # Negative salary
    df.loc[20, 'email'] = 'invalid_email'  # Invalid email
    df.loc[25, 'department'] = 'Unknown'  # Invalid department
    df.loc[30:35, 'name'] = ''  # Empty names
    
    print("Sample data for validation:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # 1. Data type validation
    print("1. Data Type Validation:")
    
    expected_types = {
        'id': 'int64',
        'name': 'object',
        'age': 'int64',
        'email': 'object',
        'salary': 'float64',
        'department': 'object',
        'start_date': 'datetime64[ns]'
    }
    
    print("Expected data types:")
    for col, expected_type in expected_types.items():
        actual_type = str(df[col].dtype)
        status = "✓" if actual_type == expected_type else "✗"
        print(f"  {col}: {actual_type} {status}")
    
    print("\n" + "="*50 + "\n")
    
    # 2. Range validation
    print("2. Range Validation:")
    
    # Age validation
    age_violations = df[(df['age'] < 18) | (df['age'] > 100)]
    print(f"Age violations (n={len(age_violations)}):")
    if len(age_violations) > 0:
        print(age_violations[['id', 'name', 'age']])
    
    # Salary validation
    salary_violations = df[df['salary'] < 0]
    print(f"\nSalary violations (n={len(salary_violations)}):")
    if len(salary_violations) > 0:
        print(salary_violations[['id', 'name', 'salary']])
    
    print("\n" + "="*50 + "\n")
    
    # 3. Format validation
    print("3. Format Validation:")
    
    # Email format validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    invalid_emails = df[~df['email'].str.match(email_pattern, na=False)]
    print(f"Invalid email formats (n={len(invalid_emails)}):")
    if len(invalid_emails) > 0:
        print(invalid_emails[['id', 'name', 'email']])
    
    # Name format validation
    empty_names = df[df['name'].str.strip() == '']
    print(f"\nEmpty names (n={len(empty_names)}):")
    if len(empty_names) > 0:
        print(empty_names[['id', 'name']])
    
    print("\n" + "="*50 + "\n")
    
    # 4. Categorical validation
    print("4. Categorical Validation:")
    
    valid_departments = ['HR', 'IT', 'Sales', 'Marketing']
    invalid_departments = df[~df['department'].isin(valid_departments)]
    print(f"Invalid departments (n={len(invalid_departments)}):")
    if len(invalid_departments) > 0:
        print(invalid_departments[['id', 'name', 'department']])
    
    print("\n" + "="*50 + "\n")
    
    # 5. Completeness validation
    print("5. Completeness Validation:")
    
    missing_summary = df.isnull().sum()
    print("Missing values:")
    for col in df.columns:
        if missing_summary[col] > 0:
            print(f"  {col}: {missing_summary[col]}")
    
    # Duplicate validation
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    print("\n" + "="*50 + "\n")
    
    # 6. Consistency validation
    print("6. Consistency Validation:")
    
    # Check for logical inconsistencies
    # Example: Check if start dates are reasonable given age
    current_date = pd.Timestamp('2024-01-01')
    df['years_worked'] = (current_date - df['start_date']).dt.days / 365.25
    df['age_at_start'] = df['age'] - df['years_worked']
    
    # Flag if someone started working before age 16
    early_workers = df[df['age_at_start'] < 16]
    print(f"Started working before age 16 (n={len(early_workers)}):")
    if len(early_workers) > 0:
        print(early_workers[['id', 'name', 'age', 'start_date', 'age_at_start']])
    
    print("\n" + "="*50 + "\n")
    
    # 7. Summary report
    print("7. Validation Summary Report:")
    
    total_records = len(df)
    validation_issues = {
        'age_violations': len(age_violations),
        'salary_violations': len(salary_violations),
        'invalid_emails': len(invalid_emails),
        'empty_names': len(empty_names),
        'invalid_departments': len(invalid_departments),
        'duplicates': duplicates,
        'early_workers': len(early_workers)
    }
    
    total_issues = sum(validation_issues.values())
    data_quality_score = ((total_records - total_issues) / total_records) * 100
    
    print(f"Total records: {total_records}")
    print(f"Total issues found: {total_issues}")
    print(f"Data quality score: {data_quality_score:.1f}%")
    
    print("\nIssues breakdown:")
    for issue, count in validation_issues.items():
        if count > 0:
            print(f"  {issue}: {count}")


def reproducible_code():
    """
    Best Practice 3: Reproducible Code
    - Setting random seeds, saving processed data, and logging.
    """
    print("=== Reproducible Code Best Practices ===\n")
    
    # 1. Set random seeds
    print("1. Setting Random Seeds:")
    
    # Set seeds for reproducibility
    np.random.seed(42)
    import random
    random.seed(42)
    
    print("Random seeds set for numpy and random modules")
    print("This ensures reproducible results across runs")
    
    print("\n" + "="*50 + "\n")
    
    # 2. Create sample data processing pipeline
    print("2. Data Processing Pipeline:")
    
    # Generate sample data
    n = 50
    data = {
        'id': range(1, n + 1),
        'value': np.random.normal(100, 20, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'date': pd.date_range('2024-01-01', periods=n, freq='D')
    }
    
    df = pd.DataFrame(data)
    print("Original data shape:", df.shape)
    
    # Data processing steps
    df_processed = (df
                   .copy()
                   .assign(
                       value_scaled=lambda x: (x['value'] - x['value'].mean()) / x['value'].std(),
                       category_encoded=lambda x: pd.Categorical(x['category']).codes,
                       month=lambda x: x['date'].dt.month
                   )
                   .dropna()
                   .sort_values('date'))
    
    print("Processed data shape:", df_processed.shape)
    
    print("\n" + "="*50 + "\n")
    
    # 3. Save processed data
    print("3. Saving Processed Data:")
    
    # Save with timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f'processed_data_{timestamp}.csv'
    
    # Save data (commented out to avoid file creation)
    # df_processed.to_csv(f'data/processed/{filename}', index=False)
    print(f"Data would be saved as: {filename}")
    
    # Save metadata
    metadata = {
        'original_shape': df.shape,
        'processed_shape': df_processed.shape,
        'processing_date': timestamp,
        'random_seed': 42,
        'processing_steps': [
            'Scale values',
            'Encode categories', 
            'Extract month',
            'Remove missing values',
            'Sort by date'
        ]
    }
    
    # Save metadata (commented out to avoid file creation)
    # import json
    # with open(f'data/processed/metadata_{timestamp}.json', 'w') as f:
    #     json.dump(metadata, f, indent=2)
    print(f"Metadata would be saved as: metadata_{timestamp}.json")
    
    print("\n" + "="*50 + "\n")
    
    # 4. Logging
    print("4. Logging:")
    
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_processing.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Log processing steps
    logger.info("Starting data processing pipeline")
    logger.info(f"Original data shape: {df.shape}")
    logger.info("Applied data transformations")
    logger.info(f"Processed data shape: {df_processed.shape}")
    logger.info("Data processing completed successfully")
    
    print("Logging configured and messages written to data_processing.log")
    
    print("\n" + "="*50 + "\n")
    
    # 5. Version control
    print("5. Version Control:")
    
    print("Recommended version control practices:")
    print("- Use .gitignore to exclude data files and logs")
    print("- Commit code changes frequently")
    print("- Use descriptive commit messages")
    print("- Tag releases with version numbers")
    print("- Document dependencies in requirements.txt")
    
    # Example .gitignore content
    gitignore_content = """
# Data files
data/raw/
data/processed/
data/interim/

# Logs
*.log
logs/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.venv/
venv/

# IDE
.vscode/
.idea/
"""
    
    print("\nExample .gitignore content:")
    print(gitignore_content)
    
    print("\n" + "="*50 + "\n")
    
    # 6. Documentation
    print("6. Documentation:")
    
    print("Code documentation best practices:")
    print("- Use docstrings for functions and classes")
    print("- Include parameter descriptions and return values")
    print("- Add inline comments for complex logic")
    print("- Create README files for projects")
    print("- Document data sources and processing steps")
    
    # Example function documentation
    example_docstring = '''
def process_data(df, scaling=True, encoding=True):
    """
    Process raw data for analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data to be processed
    scaling : bool, default=True
        Whether to apply feature scaling
    encoding : bool, default=True
        Whether to encode categorical variables
    
    Returns:
    --------
    pandas.DataFrame
        Processed data ready for analysis
    
    Examples:
    ---------
    >>> processed_df = process_data(raw_df)
    >>> processed_df = process_data(raw_df, scaling=False)
    """
    pass
'''
    
    print("\nExample function documentation:")
    print(example_docstring)


def exercise_1_data_import_inspection():
    """
    Exercise 1: Data Import and Inspection
    - Download or create sample data and perform comprehensive data quality assessment.
    """
    print("=== Exercise 1: Data Import and Inspection ===\n")
    
    # Create sample dataset with various issues
    np.random.seed(123)
    n = 100
    
    data = {
        'customer_id': range(1, n + 1),
        'name': [f'Customer_{i}' for i in range(1, n + 1)],
        'email': [f'customer{i}@email.com' for i in range(1, n + 1)],
        'age': np.random.randint(18, 80, n),
        'income': np.random.normal(50000, 20000, n),
        'purchase_amount': np.random.exponential(100, n),
        'satisfaction_score': np.random.uniform(1, 10, n),
        'join_date': pd.date_range('2020-01-01', periods=n, freq='D'),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce data quality issues
    df.loc[5:10, 'age'] = np.nan
    df.loc[15:20, 'income'] = -1000
    df.loc[25:30, 'satisfaction_score'] = 15
    df.loc[35:40, 'email'] = 'invalid_email'
    df.loc[45:50, 'name'] = ''
    df.loc[55:60, 'region'] = 'Unknown'
    
    # Add duplicates
    df = pd.concat([df, df.iloc[0:5]], ignore_index=True)
    
    print("Sample customer data created with quality issues:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # Your tasks:
    print("TASKS TO COMPLETE:")
    print("1. Perform basic data exploration")
    print("2. Check for missing values and data quality issues")
    print("3. Create summary statistics for numeric variables")
    print("4. Analyze the distribution of each numeric variable")
    print("5. Create visualizations (histograms, box plots, scatter plots)")
    print("6. Examine relationships between variables")
    print("7. Compare regions and product categories")
    print("8. Document your findings")
    
    print("\n" + "="*50 + "\n")
    
    # Solution template
    print("SOLUTION TEMPLATE:")
    print("# 1. Basic data exploration")
    print("print('Dataset shape:', df.shape)")
    print("print('Data types:\\n', df.dtypes)")
    print("print('Missing values:\\n', df.isnull().sum())")
    print("print('Duplicate rows:', df.duplicated().sum())")
    
    print("\n# 2. Summary statistics")
    print("print('Numeric variables summary:')")
    print("print(df.describe())")
    
    print("\n# 3. Data quality assessment")
    print("# Check for invalid values")
    print("# Check for outliers")
    print("# Validate data types")
    
    print("\n# 4. Visualizations")
    print("# Create histograms, box plots, scatter plots")
    print("# Analyze distributions and relationships")
    
    print("\n# 5. Group analysis")
    print("# Compare regions")
    print("# Compare product categories")
    print("# Analyze relationships")


def exercise_2_data_manipulation():
    """
    Exercise 2: Data Manipulation with pandas
    - Comprehensive analysis with multiple transformations.
    """
    print("=== Exercise 2: Data Manipulation with pandas ===\n")
    
    # Create sample sales data
    np.random.seed(456)
    n = 200
    
    data = {
        'transaction_id': range(1, n + 1),
        'customer_id': np.random.randint(1, 51, n),
        'product_id': np.random.randint(1, 21, n),
        'quantity': np.random.poisson(3, n),
        'unit_price': np.random.uniform(10, 100, n),
        'date': pd.date_range('2023-01-01', periods=n, freq='D'),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'salesperson_id': np.random.randint(1, 11, n),
        'payment_method': np.random.choice(['Credit Card', 'Cash', 'Bank Transfer'], n)
    }
    
    df = pd.DataFrame(data)
    df['total_amount'] = df['quantity'] * df['unit_price']
    
    print("Sample sales transaction data:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # Your tasks:
    print("TASKS TO COMPLETE:")
    print("1. Calculate total sales by region")
    print("2. Find the top 10 customers by total spending")
    print("3. Calculate average transaction value by payment method")
    print("4. Create a new variable for transaction size category")
    print("5. Calculate daily sales totals")
    print("6. Find the most popular products")
    print("7. Calculate salesperson performance metrics")
    print("8. Create a summary report with key insights")
    
    print("\n" + "="*50 + "\n")
    
    # Solution hints
    print("SOLUTION HINTS:")
    print("# 1. Use groupby() and sum() for regional totals")
    print("regional_sales = df.groupby('region')['total_amount'].sum()")
    
    print("\n# 2. Use groupby() and sum(), then nlargest()")
    print("customer_totals = df.groupby('customer_id')['total_amount'].sum()")
    print("top_customers = customer_totals.nlargest(10)")
    
    print("\n# 3. Group by payment method and calculate mean")
    print("avg_by_payment = df.groupby('payment_method')['total_amount'].mean()")
    
    print("\n# 4. Use pd.cut() or np.where() for categories")
    print("df['transaction_category'] = pd.cut(df['total_amount'], bins=[0, 50, 200, 1000], labels=['Small', 'Medium', 'Large'])")
    
    print("\n# 5. Group by date and sum")
    print("daily_sales = df.groupby('date')['total_amount'].sum()")
    
    print("\n# 6. Count product occurrences")
    print("product_popularity = df['product_id'].value_counts()")
    
    print("\n# 7. Calculate multiple metrics by salesperson")
    print("salesperson_performance = df.groupby('salesperson_id').agg({")
    print("    'total_amount': ['sum', 'mean', 'count'],")
    print("    'customer_id': 'nunique'")
    print("})")


def exercise_3_data_reshaping():
    """
    Exercise 3: Data Reshaping
    - Creating wide and long datasets, reshaping and sorting data.
    """
    print("=== Exercise 3: Data Reshaping ===\n")
    
    # Create sample wide format data
    np.random.seed(789)
    n_students = 20
    subjects = ['Math', 'Science', 'English', 'History']
    
    wide_data = {
        'student_id': range(1, n_students + 1),
        'name': [f'Student_{i}' for i in range(1, n_students + 1)],
        'grade': np.random.choice(['9th', '10th', '11th', '12th'], n_students),
        'Math_Test1': np.random.randint(60, 100, n_students),
        'Math_Test2': np.random.randint(60, 100, n_students),
        'Science_Test1': np.random.randint(60, 100, n_students),
        'Science_Test2': np.random.randint(60, 100, n_students),
        'English_Test1': np.random.randint(60, 100, n_students),
        'English_Test2': np.random.randint(60, 100, n_students),
        'History_Test1': np.random.randint(60, 100, n_students),
        'History_Test2': np.random.randint(60, 100, n_students)
    }
    
    df_wide = pd.DataFrame(wide_data)
    
    print("Wide format data (one row per student):")
    print(df_wide.head())
    print(f"\nShape: {df_wide.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # Your tasks:
    print("TASKS TO COMPLETE:")
    print("1. Convert wide format to long format")
    print("2. Separate subject and test number from column names")
    print("3. Calculate average score by subject and test")
    print("4. Create a pivot table showing student performance")
    print("5. Reshape data to show test scores as columns")
    print("6. Calculate improvement between Test1 and Test2")
    print("7. Create a summary by grade level")
    print("8. Sort data by average performance")
    
    print("\n" + "="*50 + "\n")
    
    # Solution hints
    print("SOLUTION HINTS:")
    print("# 1. Use melt() to convert to long format")
    print("df_long = df_wide.melt(")
    print("    id_vars=['student_id', 'name', 'grade'],")
    print("    value_vars=['Math_Test1', 'Math_Test2', 'Science_Test1', ...],")
    print("    var_name='test', value_name='score'")
    print(")")
    
    print("\n# 2. Use str.split() to separate subject and test")
    print("df_long[['subject', 'test_number']] = df_long['test'].str.split('_', expand=True)")
    
    print("\n# 3. Group by subject and test_number")
    print("avg_by_subject_test = df_long.groupby(['subject', 'test_number'])['score'].mean()")
    
    print("\n# 4. Use pivot_table()")
    print("performance_pivot = df_long.pivot_table(")
    print("    values='score',")
    print("    index='name',")
    print("    columns=['subject', 'test_number']")
    print(")")
    
    print("\n# 5. Use pivot() to reshape")
    print("test_scores_wide = df_long.pivot(")
    print("    index=['student_id', 'name', 'grade'],")
    print("    columns='test',")
    print("    values='score'")
    print(").reset_index()")


def exercise_4_string_manipulation():
    """
    Exercise 4: String Manipulation
    - Cleaning messy text data.
    """
    print("=== Exercise 4: String Manipulation ===\n")
    
    # Create sample messy text data
    messy_data = {
        'id': range(1, 21),
        'name': [
            'john doe', 'JANE SMITH', 'mike.johnson', 'Sarah Wilson',
            'ROBERT BROWN', 'lisa.davis', 'DAVID MILLER', 'emily taylor',
            'CHRIS ANDERSON', 'amanda.garcia', 'JAMES MARTINEZ', 'rachel lee',
            'MICHAEL WHITE', 'jennifer.clark', 'DANIEL LEWIS', 'sophia hall',
            'CHRISTOPHER YOUNG', 'olivia.allen', 'MATTHEW KING', 'ava scott'
        ],
        'email': [
            'john.doe@email.com', 'JANE.SMITH@EMAIL.COM', 'mike.j@company.com',
            'sarah.wilson@email.com', 'robert.brown@COMPANY.COM', 'lisa.d@email.com',
            'david.miller@email.com', 'emily.taylor@COMPANY.COM', 'chris.a@email.com',
            'amanda.garcia@email.com', 'james.martinez@COMPANY.COM', 'rachel.lee@email.com',
            'michael.white@email.com', 'jennifer.clark@COMPANY.COM', 'daniel.lewis@email.com',
            'sophia.hall@email.com', 'christopher.young@COMPANY.COM', 'olivia.allen@email.com',
            'matthew.king@email.com', 'ava.scott@COMPANY.COM'
        ],
        'phone': [
            '+1-555-123-4567', '(555) 234-5678', '555.345.6789',
            '+1-555-456-7890', '(555) 567-8901', '555.678.9012',
            '+1-555-789-0123', '(555) 890-1234', '555.901.2345',
            '+1-555-012-3456', '(555) 123-4567', '555.234.5678',
            '+1-555-345-6789', '(555) 456-7890', '555.567.8901',
            '+1-555-678-9012', '(555) 789-0123', '555.890.1234',
            '+1-555-901-2345', '(555) 012-3456'
        ],
        'address': [
            '123 Main St, New York, NY 10001',
            '456 Oak Ave, Los Angeles, CA 90210',
            '789 Pine Rd, Chicago, IL 60601',
            '321 Elm St, Houston, TX 77001',
            '654 Maple Dr, Phoenix, AZ 85001',
            '987 Cedar Ln, Philadelphia, PA 19101',
            '147 Birch Blvd, San Antonio, TX 78201',
            '258 Spruce Way, San Diego, CA 92101',
            '369 Willow Ct, Dallas, TX 75201',
            '741 Aspen Pl, San Jose, CA 95101',
            '852 Poplar St, Austin, TX 73301',
            '963 Sycamore Ave, Jacksonville, FL 32201',
            '159 Magnolia Dr, Fort Worth, TX 76101',
            '357 Dogwood Rd, Columbus, OH 43201',
            '468 Redwood Ln, Charlotte, NC 28201',
            '579 Sequoia Blvd, San Francisco, CA 94101',
            '681 Cypress Way, Indianapolis, IN 46201',
            '792 Juniper Ct, Seattle, WA 98101',
            '813 Hemlock Pl, Denver, CO 80201',
            '924 Fir St, Washington, DC 20001'
        ]
    }
    
    df = pd.DataFrame(messy_data)
    
    print("Messy text data:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # Your tasks:
    print("TASKS TO COMPLETE:")
    print("1. Standardize name format (Title Case)")
    print("2. Clean and standardize email addresses")
    print("3. Extract and format phone numbers consistently")
    print("4. Parse address into separate columns (street, city, state, zip)")
    print("5. Extract domain from email addresses")
    print("6. Create a standardized contact format")
    print("7. Identify and handle duplicate names")
    print("8. Create a summary of data cleaning results")
    
    print("\n" + "="*50 + "\n")
    
    # Solution hints
    print("SOLUTION HINTS:")
    print("# 1. Use str.title() for name formatting")
    print("df['name_clean'] = df['name'].str.title()")
    
    print("\n# 2. Use str.lower() and str.strip() for emails")
    print("df['email_clean'] = df['email'].str.lower().str.strip()")
    
    print("\n# 3. Use regex to extract and format phone numbers")
    print("df['phone_clean'] = df['phone'].str.replace(r'[^\d]', '', regex=True)")
    print("df['phone_formatted'] = df['phone_clean'].str.replace(r'(\\d{3})(\\d{3})(\\d{4})', r'(\\1) \\2-\\3', regex=True)")
    
    print("\n# 4. Use str.split() to parse addresses")
    print("address_parts = df['address'].str.split(', ', expand=True)")
    print("df['street'] = address_parts[0]")
    print("df['city'] = address_parts[1]")
    print("df['state_zip'] = address_parts[2]")
    
    print("\n# 5. Use str.extract() to get domain")
    print("df['email_domain'] = df['email'].str.extract(r'@([^.]+\\.[^.]+)')")
    
    print("\n# 6. Create standardized contact format")
    print("df['contact_info'] = df['name_clean'] + ' | ' + df['email_clean'] + ' | ' + df['phone_formatted']")


def exercise_5_date_analysis():
    """
    Exercise 5: Date Analysis
    - Creating and analyzing date-based data.
    """
    print("=== Exercise 5: Date Analysis ===\n")
    
    # Create sample date-based data
    np.random.seed(321)
    n = 100
    
    # Generate random dates over the past year
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-12-31')
    random_dates = pd.date_range(start_date, end_date, periods=n)
    
    data = {
        'id': range(1, n + 1),
        'customer_id': np.random.randint(1, 21, n),
        'order_date': random_dates,
        'delivery_date': random_dates + pd.Timedelta(days=np.random.randint(1, 15, n)),
        'return_date': random_dates + pd.Timedelta(days=np.random.randint(30, 90, n)),
        'amount': np.random.uniform(10, 500, n),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing delivery and return dates
    df.loc[np.random.choice(df.index, 10), 'delivery_date'] = pd.NaT
    df.loc[np.random.choice(df.index, 20), 'return_date'] = pd.NaT
    
    print("Sample order data with dates:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    
    print("\n" + "="*50 + "\n")
    
    # Your tasks:
    print("TASKS TO COMPLETE:")
    print("1. Calculate delivery time (days between order and delivery)")
    print("2. Calculate return rate by month")
    print("3. Analyze seasonal patterns in orders")
    print("4. Find the busiest day of the week for orders")
    print("5. Calculate average order value by quarter")
    print("6. Identify peak ordering periods")
    print("7. Analyze delivery performance by region")
    print("8. Create a time series analysis of order trends")
    
    print("\n" + "="*50 + "\n")
    
    # Solution hints
    print("SOLUTION HINTS:")
    print("# 1. Calculate delivery time")
    print("df['delivery_time'] = (df['delivery_date'] - df['order_date']).dt.days")
    
    print("\n# 2. Calculate return rate by month")
    print("df['order_month'] = df['order_date'].dt.to_period('M')")
    print("return_rate_by_month = df.groupby('order_month').agg({")
    print("    'id': 'count',")
    print("    'return_date': lambda x: x.notna().sum()")
    print("})")
    print("return_rate_by_month['return_rate'] = return_rate_by_month['return_date'] / return_rate_by_month['id']")
    
    print("\n# 3. Analyze seasonal patterns")
    print("df['season'] = df['order_date'].dt.quarter")
    print("seasonal_orders = df.groupby('season')['amount'].sum()")
    
    print("\n# 4. Find busiest day of week")
    print("df['day_of_week'] = df['order_date'].dt.day_name()")
    print("orders_by_day = df['day_of_week'].value_counts()")
    
    print("\n# 5. Average order value by quarter")
    print("quarterly_avg = df.groupby(df['order_date'].dt.quarter)['amount'].mean()")
    
    print("\n# 6. Peak ordering periods")
    print("daily_orders = df.groupby(df['order_date'].dt.date)['id'].count()")
    print("peak_days = daily_orders.nlargest(10)")
    
    print("\n# 7. Delivery performance by region")
    print("delivery_performance = df.groupby('region')['delivery_time'].agg(['mean', 'std', 'count'])")
    
    print("\n# 8. Time series analysis")
    print("monthly_orders = df.groupby(df['order_date'].dt.to_period('M')).agg({")
    print("    'id': 'count',")
    print("    'amount': 'sum'")
    print("})")


if __name__ == "__main__":
    print("Data Import and Manipulation Examples")
    print("=" * 50)
    
    # Run all functions
    reading_csv_files()
    print("\n" + "="*80 + "\n")
    
    reading_excel_files()
    print("\n" + "="*80 + "\n")
    
    reading_other_formats()
    print("\n" + "="*80 + "\n")
    
    built_in_datasets()
    print("\n" + "="*80 + "\n") 