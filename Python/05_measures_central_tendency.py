"""
Measures of Central Tendency - Python Code Examples
Corresponds to: 05_measures_central_tendency.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from scipy.stats import mode, gmean, hmean, trim_mean
from scipy.stats.mstats import winsorize

# Load sample data
def load_mtcars():
    """
    Load the mtcars dataset from OpenML as a pandas DataFrame.
    """
    mtcars = fetch_openml(name='mtcars', as_frame=True).frame
    return mtcars

def basic_mean_calculation():
    """
    Demonstrate basic mean calculation with manual verification and outlier effects.
    """
    # Create sample data
    data = [2, 4, 6, 8, 10]
    
    # Calculate mean
    mean_value = np.mean(data)
    print(f"Mean: {mean_value}")
    
    # Verify calculation manually
    manual_mean = sum(data) / len(data)
    print(f"Manual calculation: {manual_mean}")
    
    # Mean with missing values
    data_with_na = [2, 4, np.nan, 8, 10]
    mean_with_na = np.nanmean(data_with_na)
    print(f"Mean with NaN removed: {mean_with_na}")
    
    # Understanding the effect of outliers
    data_with_outlier = [2, 4, 6, 8, 10, 100]
    print(f"Original mean: {np.mean(data)}")
    print(f"Mean with outlier: {np.mean(data_with_outlier)}")

def mean_by_group():
    """
    Calculate and visualize mean by group using the mtcars dataset.
    """
    mtcars = load_mtcars()
    
    # Calculate mean by group
    # Mean MPG by number of cylinders
    mpg_by_cyl = mtcars.groupby('cyl').agg({
        'mpg': ['count', 'mean', 'std']
    }).round(3)
    mpg_by_cyl.columns = ['count', 'mean_mpg', 'sd_mpg']
    mpg_by_cyl['se_mpg'] = mpg_by_cyl['sd_mpg'] / np.sqrt(mpg_by_cyl['count'])
    print(mpg_by_cyl)
    
    # Visualize means by group
    plt.figure(figsize=(8, 6))
    sns.barplot(data=mpg_by_cyl.reset_index(), x='cyl', y='mean_mpg', color='steelblue', alpha=0.7)
    plt.errorbar(x=range(len(mpg_by_cyl)), y=mpg_by_cyl['mean_mpg'], 
                 yerr=mpg_by_cyl['se_mpg'], fmt='none', color='black', capsize=5)
    plt.title('Mean MPG by Number of Cylinders')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('Mean MPG')
    plt.show()

def weighted_mean_example():
    """
    Demonstrate weighted mean calculation with manual verification and course grades example.
    """
    # Create data with weights
    values = [85, 92, 78, 96, 88]
    weights = [0.2, 0.3, 0.1, 0.25, 0.15]
    
    # Calculate weighted mean
    weighted_mean = np.average(values, weights=weights)
    print(f"Weighted mean: {weighted_mean}")
    
    # Verify calculation manually
    manual_weighted_mean = sum(v * w for v, w in zip(values, weights)) / sum(weights)
    print(f"Manual calculation: {manual_weighted_mean}")
    
    # Example: Course grades with different weights
    course_grades = pd.DataFrame({
        'assignment': ['Homework', 'Midterm', 'Final', 'Participation'],
        'grade': [85, 78, 92, 95],
        'weight': [0.3, 0.25, 0.35, 0.1]
    })
    
    course_weighted_mean = np.average(course_grades['grade'], weights=course_grades['weight'])
    print(f"Course weighted mean: {course_weighted_mean}")

def basic_median_calculation():
    """
    Demonstrate basic median calculation with manual verification for odd and even sample sizes.
    """
    # Calculate median
    data = [1, 3, 5, 7, 9]
    median_value = np.median(data)
    print(f"Median: {median_value}")
    
    # Median with even number of observations
    data_even = [1, 3, 5, 7, 9, 11]
    median_even = np.median(data_even)
    print(f"Median (even n): {median_even}")
    
    # Verify calculation for odd n
    sorted_data = sorted(data)
    n = len(data)
    manual_median_odd = sorted_data[(n + 1) // 2 - 1]  # Python uses 0-based indexing
    print(f"Manual median (odd): {manual_median_odd}")
    
    # Verify calculation for even n
    sorted_even = sorted(data_even)
    n_even = len(data_even)
    manual_median_even = (sorted_even[n_even//2 - 1] + sorted_even[n_even//2]) / 2
    print(f"Manual median (even): {manual_median_even}")
    
    # Median with missing values
    data_with_na = [1, 3, np.nan, 7, 9]
    median_with_na = np.nanmedian(data_with_na)
    print(f"Median with NaN removed: {median_with_na}")

def median_by_group():
    """
    Calculate median by group and compare mean vs median for the mtcars dataset.
    """
    mtcars = load_mtcars()
    
    # Median MPG by cylinders
    median_by_cyl = mtcars.groupby('cyl').agg({
        'mpg': ['count', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    }).round(3)
    median_by_cyl.columns = ['count', 'median_mpg', 'q25', 'q75']
    print(median_by_cyl)
    
    # Compare mean vs median by group
    comparison_by_cyl = mtcars.groupby('cyl').agg({
        'mpg': ['mean', 'median']
    }).round(3)
    comparison_by_cyl.columns = ['mean_mpg', 'median_mpg']
    comparison_by_cyl['difference'] = comparison_by_cyl['mean_mpg'] - comparison_by_cyl['median_mpg']
    comparison_by_cyl['count'] = mtcars.groupby('cyl').size()
    print(comparison_by_cyl)
    
    # Visualize mean vs median
    plt.figure(figsize=(8, 6))
    x_pos = range(len(comparison_by_cyl))
    plt.plot(x_pos, comparison_by_cyl['mean_mpg'], 'ro-', label='Mean', markersize=8)
    plt.plot(x_pos, comparison_by_cyl['median_mpg'], 'bs-', label='Median', markersize=8)
    plt.xticks(x_pos, comparison_by_cyl.index)
    plt.title('Mean vs Median MPG by Cylinders')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('MPG')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def find_mode(x):
    """
    Find the mode (most frequent value) in a dataset.
    """
    return mode(x, keepdims=False)[0]

def find_all_modes(x):
    """
    Find all modes for multimodal data.
    """
    freq_table = pd.Series(x).value_counts()
    max_freq = freq_table.max()
    modes = freq_table[freq_table == max_freq].index.tolist()
    return modes

def mode_examples():
    """
    Demonstrate mode calculation with various examples including multimodal data.
    """
    # Test the function
    data = [1, 2, 2, 3, 4, 4, 4, 5]
    mode_value = find_mode(data)
    print(f"Mode: {mode_value}")
    
    # Using value_counts to see frequency
    freq_table = pd.Series(data).value_counts()
    print(freq_table)
    
    # Find all modes (for multimodal data)
    multimodal_data = [1, 2, 2, 3, 4, 4, 5]
    all_modes = find_all_modes(multimodal_data)
    print(f"All modes: {all_modes}")

def mode_categorical_data():
    """
    Demonstrate mode calculation for categorical data with visualization.
    """
    # Create categorical data
    colors = ['red', 'blue', 'red', 'green', 'blue', 'red', 'yellow']
    mode_color = find_mode(colors)
    print(f"Mode color: {mode_color}")
    
    # Frequency table
    color_freq = pd.Series(colors).value_counts()
    print(color_freq)
    
    # Visualize frequency distribution
    plt.figure(figsize=(8, 6))
    color_freq.plot(kind='bar', color='steelblue', alpha=0.7)
    plt.title('Frequency Distribution of Colors')
    plt.xlabel('Color')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def geometric_mean(x):
    """
    Calculate geometric mean for a dataset, handling NaN values and non-positive values.
    """
    x = np.array(x)
    x = x[~np.isnan(x)]  # Remove NaN values
    if np.any(x <= 0):
        print("Warning: Geometric mean requires positive values")
        return np.nan
    return gmean(x)

def geometric_mean_examples():
    """
    Demonstrate geometric mean calculation with growth rates and investment returns.
    """
    # Example: Growth rates
    growth_rates = [1.05, 1.08, 1.12, 1.06]
    gm_growth = geometric_mean(growth_rates)
    print(f"Geometric mean of growth rates: {gm_growth}")
    
    # Verify with built-in function
    manual_gm = np.exp(np.mean(np.log(growth_rates)))
    print(f"Manual calculation: {manual_gm}")
    
    # Compare with arithmetic mean
    am_growth = np.mean(growth_rates)
    print(f"Arithmetic mean: {am_growth}")
    print(f"Geometric mean: {gm_growth}")
    
    # Example: Investment returns
    returns = [0.05, 0.08, -0.02, 0.12, 0.06]
    # Convert to growth factors
    growth_factors = [1 + r for r in returns]
    gm_return = geometric_mean(growth_factors) - 1
    print(f"Geometric mean return: {gm_return}")

def harmonic_mean(x):
    """
    Calculate harmonic mean for a dataset, handling NaN values and non-positive values.
    """
    x = np.array(x)
    x = x[~np.isnan(x)]  # Remove NaN values
    if np.any(x <= 0):
        print("Warning: Harmonic mean requires positive values")
        return np.nan
    return hmean(x)

def harmonic_mean_examples():
    """
    Demonstrate harmonic mean calculation with speeds and parallel resistors.
    """
    # Example: Average speed
    speeds = [60, 40, 80]  # km/h
    hm_speed = harmonic_mean(speeds)
    print(f"Harmonic mean speed: {hm_speed} km/h")
    
    # Verify calculation
    manual_hm = 3 / (1/60 + 1/40 + 1/80)
    print(f"Manual calculation: {manual_hm} km/h")
    
    # Compare with arithmetic mean
    am_speed = np.mean(speeds)
    print(f"Arithmetic mean speed: {am_speed} km/h")
    print(f"Harmonic mean speed: {hm_speed} km/h")
    
    # Example: Parallel resistors
    resistors = [10, 20, 30]  # ohms
    hm_resistance = harmonic_mean(resistors)
    print(f"Equivalent parallel resistance: {hm_resistance} ohms")

def trimmed_mean_examples():
    """
    Demonstrate trimmed mean calculation with different trimming levels.
    """
    # Calculate trimmed mean
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    
    # 10% trimmed mean (removes 10% from each end)
    trimmed_mean_val = trim_mean(data, 0.1)
    print(f"10% trimmed mean: {trimmed_mean_val}")
    
    # Compare with regular mean
    regular_mean = np.mean(data)
    print(f"Regular mean: {regular_mean}")
    
    # Manual calculation of 10% trimmed mean
    sorted_data = sorted(data)
    n = len(data)
    trim_n = round(n * 0.1)
    trimmed_data = sorted_data[trim_n:(n - trim_n)]
    manual_trimmed_mean = np.mean(trimmed_data)
    print(f"Manual trimmed mean: {manual_trimmed_mean}")
    
    # Compare different trimming levels
    trim_levels = [0, 0.05, 0.1, 0.2]
    trimmed_means = [trim_mean(data, level) for level in trim_levels]
    for level, mean_val in zip(trim_levels, trimmed_means):
        print(f"{level*100}% trimmed mean: {mean_val}")

def winsorized_mean(x, k=1):
    """
    Calculate winsorized mean by replacing extreme values with less extreme values.
    """
    x_winsorized = winsorize(x, limits=(k/len(x), k/len(x)))
    return np.mean(x_winsorized)

def winsorized_mean_examples():
    """
    Demonstrate winsorized mean calculation and compare with other robust measures.
    """
    # Example
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    winsorized_mean_val = winsorized_mean(data, k=1)
    print(f"Winsorized mean (k=1): {winsorized_mean_val}")
    
    # Compare different robust measures
    print(f"Original data: {data}")
    print(f"Arithmetic mean: {np.mean(data)}")
    print(f"Median: {np.median(data)}")
    print(f"10% trimmed mean: {trim_mean(data, 0.1)}")
    print(f"Winsorized mean (k=1): {winsorized_mean_val}")

def compare_all_measures():
    """
    Compare all measures of central tendency for a dataset with outliers.
    """
    # Create data with outliers
    data_with_outliers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    
    # Calculate different measures
    mean_val = np.mean(data_with_outliers)
    median_val = np.median(data_with_outliers)
    mode_val = find_mode(data_with_outliers)
    geom_mean = geometric_mean(data_with_outliers)
    harm_mean = harmonic_mean(data_with_outliers)
    trimmed_mean_val = trim_mean(data_with_outliers, 0.1)
    winsorized_mean_val = winsorized_mean(data_with_outliers, k=1)
    
    # Compare results
    print("Measures of Central Tendency:")
    print(f"Mean: {mean_val}")
    print(f"Median: {median_val}")
    print(f"Mode: {mode_val}")
    print(f"Geometric Mean: {geom_mean}")
    print(f"Harmonic Mean: {harm_mean}")
    print(f"10% Trimmed Mean: {trimmed_mean_val}")
    print(f"Winsorized Mean: {winsorized_mean_val}")

def central_tendency_summary(x):
    """
    Create a comprehensive summary of all central tendency measures for a dataset.
    """
    print("=== CENTRAL TENDENCY SUMMARY ===")
    print(f"Data length: {len(x)}")
    print(f"Missing values: {np.sum(np.isnan(x))}\n")
    
    print("Measures of Central Tendency:")
    print(f"Mean: {np.nanmean(x)}")
    print(f"Median: {np.nanmedian(x)}")
    print(f"Mode: {find_mode(x[~np.isnan(x)])}")
    print(f"Geometric Mean: {geometric_mean(x)}")
    print(f"Harmonic Mean: {harmonic_mean(x)}")
    print(f"10% Trimmed Mean: {trim_mean(x, 0.1)}")
    print(f"Winsorized Mean: {winsorized_mean(x, k=1)}\n")
    
    # Compare mean and median
    mean_val = np.nanmean(x)
    median_val = np.nanmedian(x)
    print(f"Mean - Median: {mean_val - median_val}")
    
    if abs(mean_val - median_val) > 0.1 * mean_val:
        print("Note: Large difference suggests skewed distribution")

def demonstrate_inequality():
    """
    Demonstrate the Arithmetic Mean-Geometric Mean-Harmonic Mean Inequality.
    """
    # Demonstrate the inequality
    positive_data = [2, 4, 8, 16]
    
    hm = harmonic_mean(positive_data)
    gm = geometric_mean(positive_data)
    am = np.mean(positive_data)
    
    print(f"Harmonic Mean: {hm}")
    print(f"Geometric Mean: {gm}")
    print(f"Arithmetic Mean: {am}")
    
    print(f"Inequality holds: {hm <= gm and gm <= am}")

def student_test_scores_example():
    """
    Analyze student test scores with various central tendency measures.
    """
    # Create student test scores
    test_scores = [65, 72, 78, 85, 88, 90, 92, 95, 98, 100]
    
    # Calculate various measures
    print("Test Score Analysis:")
    central_tendency_summary(test_scores)
    
    # Visualize the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(test_scores, bins=10, color='lightblue', edgecolor='white', alpha=0.7)
    plt.axvline(np.mean(test_scores), color='red', linestyle='--', linewidth=2, label='Mean')
    plt.axvline(np.median(test_scores), color='blue', linestyle='--', linewidth=2, label='Median')
    plt.title('Distribution of Test Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Analyze by grade level
    def grade_analysis(scores):
        print("Grade Level Analysis:")
        scores = np.array(scores)
        print(f"A (90+): {np.mean(scores[scores >= 90])}")
        print(f"B (80-89): {np.mean(scores[(scores >= 80) & (scores < 90)])}")
        print(f"C (70-79): {np.mean(scores[(scores >= 70) & (scores < 80)])}")
        print(f"D (60-69): {np.mean(scores[(scores >= 60) & (scores < 70)])}")
    
    grade_analysis(test_scores)

def income_data_example():
    """
    Analyze income data (right-skewed) with various central tendency measures.
    """
    # Simulate income data (right-skewed)
    np.random.seed(123)
    income = np.random.lognormal(mean=10, sigma=0.5, size=1000)
    
    # Calculate measures
    print("Income Analysis:")
    central_tendency_summary(income)
    
    # Compare mean and median for skewed data
    print(f"Mean vs Median difference: {np.mean(income) - np.median(income)}")
    print(f"Skewness indicator: {'Right-skewed' if np.mean(income) > np.median(income) else 'Left-skewed'}")
    
    # Visualize with log scale
    plt.figure(figsize=(10, 6))
    plt.hist(income, bins=50, color='lightgreen', edgecolor='white', alpha=0.7)
    plt.axvline(np.mean(income), color='red', linestyle='--', linewidth=2, label='Mean')
    plt.axvline(np.median(income), color='blue', linestyle='--', linewidth=2, label='Median')
    plt.title('Distribution of Income (Log Scale)')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Income percentiles
    percentiles = np.percentile(income, [25, 50, 75, 90, 95, 99])
    print("Income Percentiles:")
    for p, val in zip([25, 50, 75, 90, 95, 99], percentiles):
        print(f"{p}th percentile: {val:.2f}")

def temperature_data_example():
    """
    Analyze temperature data with various central tendency measures.
    """
    # Create temperature data
    temperatures = [15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42]
    
    # Calculate measures
    print("Temperature Analysis:")
    central_tendency_summary(temperatures)
    
    # Seasonal analysis
    winter_temps = temperatures[:3]
    summer_temps = temperatures[-3:]
    
    print("\nWinter temperatures:")
    central_tendency_summary(winter_temps)
    
    print("\nSummer temperatures:")
    central_tendency_summary(summer_temps)
    
    # Temperature range analysis
    temp_range = max(temperatures) - min(temperatures)
    print(f"Temperature range: {temp_range} degrees")
    print(f"Mean temperature: {np.mean(temperatures)} degrees")
    print(f"Temperature variability: {np.std(temperatures)} degrees")

def choose_central_tendency(x):
    """
    Decision tree function to help choose the appropriate measure of central tendency.
    """
    print("=== CENTRAL TENDENCY DECISION TREE ===")
    
    # Check for missing values
    if np.sum(np.isnan(x)) > 0:
        print("Data contains missing values - use nan functions")
    
    # Check data type
    if hasattr(x, 'dtype') and x.dtype == 'object':
        print("Recommendation: Use MODE (categorical data)")
        return "Mode"
    
    # Check for outliers
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    outliers = np.sum((x < (q1 - 1.5 * iqr)) | (x > (q3 + 1.5 * iqr)))
    
    if outliers > 0:
        print(f"Outliers detected: {outliers}")
        print("Recommendation: Use MEDIAN or TRIMMED MEAN")
        return "Median"
    
    # Check for skewness
    mean_val = np.mean(x)
    median_val = np.median(x)
    skewness_indicator = abs(mean_val - median_val) / mean_val
    
    if skewness_indicator > 0.1:
        print("Skewed distribution detected")
        print("Recommendation: Use MEDIAN")
        return "Median"
    else:
        print("Symmetric distribution")
        print("Recommendation: Use MEAN")
        return "Mean"

def robust_central_tendency(x):
    """
    Function to handle missing data and calculate robust central tendency measures.
    """
    # Remove missing values
    x_clean = x[~np.isnan(x)]
    
    if len(x_clean) == 0:
        return np.nan
    
    # Calculate measures
    result = {
        'mean': np.mean(x_clean),
        'median': np.median(x_clean),
        'mode': find_mode(x_clean),
        'geometric_mean': geometric_mean(x_clean),
        'harmonic_mean': harmonic_mean(x_clean),
        'trimmed_mean': trim_mean(x_clean, 0.1),
        'winsorized_mean': winsorized_mean(x_clean, k=1),
        'n': len(x_clean),
        'missing': np.sum(np.isnan(x)),
        'missing_percent': np.sum(np.isnan(x)) / len(x) * 100
    }
    
    return result

if __name__ == "__main__":
    # Example usage
    basic_mean_calculation()
    mean_by_group()
    weighted_mean_example()
    basic_median_calculation()
    median_by_group()
    mode_examples()
    mode_categorical_data()
    geometric_mean_examples()
    harmonic_mean_examples()
    trimmed_mean_examples()
    winsorized_mean_examples()
    compare_all_measures()
    mtcars = load_mtcars()
    central_tendency_summary(mtcars['mpg'].values)
    demonstrate_inequality()
    student_test_scores_example()
    income_data_example()
    temperature_data_example()
    choose_central_tendency(mtcars['mpg'].values)
    data_with_missing = [1, 2, np.nan, 4, 5, np.nan, 7, 8]
    result = robust_central_tendency(data_with_missing)
    for key, value in result.items():
        print(f"{key}: {value}") 