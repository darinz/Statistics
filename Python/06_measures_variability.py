"""
Measures of Variability - Python Code Examples
Corresponds to: 06_measures_variability.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from scipy.stats import iqr, median_abs_deviation, levene

# Load sample data
def load_mtcars():
    """
    Load the mtcars dataset from OpenML as a pandas DataFrame.
    """
    mtcars = fetch_openml(name='mtcars', as_frame=True).frame
    return mtcars

def basic_range_calculation():
    """
    Demonstrate basic range calculation with manual verification and outlier effects.
    """
    # Create sample data
    data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # Calculate range
    data_range = np.ptp(data)  # peak to peak (max - min)
    print(f"Range: {data_range}")
    print(f"Min: {np.min(data)}")
    print(f"Max: {np.max(data)}")
    
    # Using built-in function
    data_range_alt = np.max(data) - np.min(data)
    print(f"Range (alternative): {data_range_alt}")
    
    # Understanding the effect of outliers
    data_with_outlier = data + [100]
    print(f"Original range: {np.max(data) - np.min(data)}")
    print(f"Range with outlier: {np.max(data_with_outlier) - np.min(data_with_outlier)}")
    
    # Range as percentage of mean
    range_percent = ((np.max(data) - np.min(data)) / np.mean(data)) * 100
    print(f"Range as % of mean: {range_percent:.2f}%")

def range_by_group():
    """
    Calculate and visualize range by group using the mtcars dataset.
    """
    mtcars = load_mtcars()
    
    # Calculate range by cylinders
    range_by_cyl = mtcars.groupby('cyl').agg({
        'mpg': ['max', 'min', 'mean']
    }).round(3)
    range_by_cyl.columns = ['max_mpg', 'min_mpg', 'mean_mpg']
    range_by_cyl['range_mpg'] = range_by_cyl['max_mpg'] - range_by_cyl['min_mpg']
    range_by_cyl['range_percent'] = (range_by_cyl['range_mpg'] / range_by_cyl['mean_mpg']) * 100
    print(range_by_cyl)
    
    # Visualize ranges by group
    plt.figure(figsize=(8, 6))
    sns.barplot(data=range_by_cyl.reset_index(), x='cyl', y='range_mpg', color='steelblue', alpha=0.7)
    plt.title('MPG Range by Number of Cylinders')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('Range (Max - Min)')
    plt.show()

def basic_iqr_calculation():
    """
    Demonstrate basic IQR calculation with manual verification and quartile understanding.
    """
    # Create sample data
    data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # Calculate IQR
    data_iqr = iqr(data)
    print(f"IQR: {data_iqr}")
    
    # Manual calculation
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    manual_iqr = q3 - q1
    print(f"Manual IQR: {manual_iqr}")
    
    # Verify with built-in function
    print(f"Built-in IQR: {iqr(data)}")
    
    # Understanding quartiles
    print(f"Q1 (25th percentile): {q1}")
    print(f"Q2 (50th percentile/median): {np.percentile(data, 50)}")
    print(f"Q3 (75th percentile): {q3}")
    
    # IQR as percentage of median
    iqr_percent = (data_iqr / np.median(data)) * 100
    print(f"IQR as % of median: {iqr_percent:.2f}%")

def iqr_by_group():
    """
    Calculate IQR by group and compare IQR vs Range for the mtcars dataset.
    """
    mtcars = load_mtcars()
    
    # IQR by cylinders
    iqr_by_cyl = mtcars.groupby('cyl').agg({
        'mpg': ['count', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    }).round(3)
    iqr_by_cyl.columns = ['count', 'median_mpg', 'q1_mpg', 'q3_mpg']
    iqr_by_cyl['iqr_mpg'] = iqr_by_cyl['q3_mpg'] - iqr_by_cyl['q1_mpg']
    iqr_by_cyl['iqr_percent'] = (iqr_by_cyl['iqr_mpg'] / iqr_by_cyl['median_mpg']) * 100
    print(iqr_by_cyl)
    
    # Compare IQR vs Range
    comparison_by_cyl = mtcars.groupby('cyl').agg({
        'mpg': ['min', 'max', lambda x: iqr(x)]
    }).round(3)
    comparison_by_cyl.columns = ['min_mpg', 'max_mpg', 'iqr_mpg']
    comparison_by_cyl['range_mpg'] = comparison_by_cyl['max_mpg'] - comparison_by_cyl['min_mpg']
    comparison_by_cyl['ratio'] = comparison_by_cyl['range_mpg'] / comparison_by_cyl['iqr_mpg']
    print(comparison_by_cyl)

def population_variance(x):
    """
    Calculate population variance using n denominator.
    """
    n = len(x)
    mean_x = np.mean(x)
    return np.sum((x - mean_x)**2) / n

def sample_variance(x):
    """
    Calculate sample variance using n-1 denominator.
    """
    n = len(x)
    mean_x = np.mean(x)
    return np.sum((x - mean_x)**2) / (n - 1)

def variance_comparison():
    """
    Compare population vs sample variance and understand the difference.
    """
    # Create sample data
    data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # Compare with built-in function
    print(f"Population variance: {population_variance(data)}")
    print(f"Sample variance: {sample_variance(data)}")
    print(f"Built-in var(): {np.var(data, ddof=1)}")
    
    # Understanding the difference
    n = len(data)
    print(f"Sample size (n): {n}")
    print(f"Degrees of freedom (n-1): {n-1}")
    print(f"Ratio n/(n-1): {n/(n-1):.3f}")
    print(f"Population variance * (n/(n-1)) = Sample variance: {population_variance(data) * (n/(n-1))}")

def variance_by_group():
    """
    Calculate variance by group and visualize using the mtcars dataset.
    """
    mtcars = load_mtcars()
    
    # Variance by cylinders
    variance_by_cyl = mtcars.groupby('cyl').agg({
        'mpg': ['var', 'mean', 'std']
    }).round(3)
    variance_by_cyl.columns = ['variance_mpg', 'mean_mpg', 'sd_mpg']
    variance_by_cyl['cv_mpg'] = (variance_by_cyl['sd_mpg'] / variance_by_cyl['mean_mpg']) * 100
    print(variance_by_cyl)
    
    # Visualize variance by group
    plt.figure(figsize=(8, 6))
    sns.barplot(data=variance_by_cyl.reset_index(), x='cyl', y='variance_mpg', color='lightgreen', alpha=0.7)
    plt.title('MPG Variance by Number of Cylinders')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('Variance')
    plt.show()

def population_sd(x):
    """
    Calculate population standard deviation.
    """
    return np.sqrt(population_variance(x))

def basic_standard_deviation():
    """
    Demonstrate basic standard deviation calculation with manual verification and empirical rule.
    """
    # Create sample data
    data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # Calculate standard deviation
    data_sd = np.std(data, ddof=1)
    print(f"Standard deviation: {data_sd}")
    
    # Manual calculation
    manual_sd = np.sqrt(np.var(data, ddof=1))
    print(f"Manual SD: {manual_sd}")
    
    print(f"Population SD: {population_sd(data)}")
    
    # Understanding the empirical rule
    mean_data = np.mean(data)
    sd_data = np.std(data, ddof=1)
    print(f"Mean: {mean_data}")
    print(f"Standard deviation: {sd_data}")
    print(f"68% of data within: {mean_data - sd_data:.2f} to {mean_data + sd_data:.2f}")
    print(f"95% of data within: {mean_data - 2*sd_data:.2f} to {mean_data + 2*sd_data:.2f}")
    print(f"99.7% of data within: {mean_data - 3*sd_data:.2f} to {mean_data + 3*sd_data:.2f}")

def standard_deviation_by_group():
    """
    Calculate standard deviation by group and visualize mean ± SD using the mtcars dataset.
    """
    mtcars = load_mtcars()
    
    # SD by cylinders
    sd_by_cyl = mtcars.groupby('cyl').agg({
        'mpg': ['std', 'mean', 'count']
    }).round(3)
    sd_by_cyl.columns = ['sd_mpg', 'mean_mpg', 'count']
    sd_by_cyl['cv_mpg'] = sd_by_cyl['sd_mpg'] / sd_by_cyl['mean_mpg']  # Coefficient of variation
    sd_by_cyl['se_mpg'] = sd_by_cyl['sd_mpg'] / np.sqrt(sd_by_cyl['count'])  # Standard error
    print(sd_by_cyl)
    
    # Visualize mean ± SD by group
    plt.figure(figsize=(8, 6))
    plt.errorbar(x=sd_by_cyl.index, y=sd_by_cyl['mean_mpg'], 
                 yerr=sd_by_cyl['sd_mpg'], fmt='o', color='red', capsize=5)
    plt.title('Mean ± SD by Cylinders')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('MPG')
    plt.grid(True, alpha=0.3)
    plt.show()

def cv(x):
    """
    Calculate coefficient of variation (CV) for a dataset.
    """
    return (np.std(x, ddof=1) / np.mean(x)) * 100

def coefficient_of_variation_examples():
    """
    Demonstrate coefficient of variation calculation and comparison across different variables.
    """
    mtcars = load_mtcars()
    
    # Apply to different variables
    print(f"CV for MPG: {cv(mtcars['mpg']):.2f}%")
    print(f"CV for Weight: {cv(mtcars['wt']):.2f}%")
    print(f"CV for Horsepower: {cv(mtcars['hp']):.2f}%")
    
    # Compare variability across different scales
    cv_comparison = pd.DataFrame({
        'Variable': ['MPG', 'Weight', 'Horsepower'],
        'Mean': [mtcars['mpg'].mean(), mtcars['wt'].mean(), mtcars['hp'].mean()],
        'SD': [mtcars['mpg'].std(ddof=1), mtcars['wt'].std(ddof=1), mtcars['hp'].std(ddof=1)],
        'CV': [cv(mtcars['mpg']), cv(mtcars['wt']), cv(mtcars['hp'])]
    })
    print(cv_comparison)
    
    # Visualize CV comparison
    plt.figure(figsize=(8, 6))
    sns.barplot(data=cv_comparison, x='Variable', y='CV', color='orange', alpha=0.7)
    plt.title('Coefficient of Variation Comparison')
    plt.xlabel('Variable')
    plt.ylabel('CV (%)')
    plt.show()
    
    # Understanding CV interpretation
    print("CV Interpretation:")
    print("CV < 15%: Low variability")
    print("CV 15-35%: Moderate variability")
    print("CV > 35%: High variability")

def mad_calculation(x):
    """
    Calculate Mean Absolute Deviation (MAD) around the mean.
    """
    return np.mean(np.abs(x - np.mean(x)))

def mad_examples():
    """
    Demonstrate MAD calculation and comparison with standard deviation.
    """
    # Create sample data
    data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # Compare with built-in function
    print(f"Manual MAD: {mad_calculation(data)}")
    print(f"Built-in MAD: {median_abs_deviation(data)}")
    
    # Note: scipy's median_abs_deviation uses median by default
    print(f"MAD around median: {median_abs_deviation(data)}")
    print(f"MAD around mean: {median_abs_deviation(data, center=np.mean(data))}")
    
    # Compare MAD vs SD
    print(f"Standard deviation: {np.std(data, ddof=1)}")
    print(f"MAD around mean: {median_abs_deviation(data, center=np.mean(data))}")
    print(f"Ratio SD/MAD: {np.std(data, ddof=1) / median_abs_deviation(data, center=np.mean(data)):.3f}")
    
    # For normal distribution, SD ≈ 1.253 × MAD
    print("Expected ratio for normal distribution: 1.253")

def percentiles_examples():
    """
    Demonstrate percentile calculations and five-number summary.
    """
    mtcars = load_mtcars()
    
    # Calculate percentiles
    percentiles = np.percentile(mtcars['mpg'], [10, 25, 50, 75, 90])
    print("Percentiles:")
    for p, val in zip([10, 25, 50, 75, 90], percentiles):
        print(f"{p}th percentile: {val:.2f}")
    
    # Calculate deciles (10th, 20th, ..., 90th percentiles)
    deciles = np.percentile(mtcars['mpg'], np.arange(10, 100, 10))
    print("\nDeciles:")
    for i, val in enumerate(deciles, 1):
        print(f"{i*10}th percentile: {val:.2f}")
    
    # Five-number summary
    five_number = [np.min(mtcars['mpg']), 
                   np.percentile(mtcars['mpg'], 25),
                   np.median(mtcars['mpg']),
                   np.percentile(mtcars['mpg'], 75),
                   np.max(mtcars['mpg'])]
    print("\nFive-number summary:")
    print(f"Min: {five_number[0]:.2f}")
    print(f"Q1: {five_number[1]:.2f}")
    print(f"Median: {five_number[2]:.2f}")
    print(f"Q3: {five_number[3]:.2f}")
    print(f"Max: {five_number[4]:.2f}")

def percentile_rank(x, value):
    """
    Calculate percentile rank of a value in a dataset.
    """
    return (np.sum(x <= value) / len(x)) * 100

def percentile_rank_examples():
    """
    Demonstrate percentile rank calculations.
    """
    mtcars = load_mtcars()
    
    print(f"Percentile rank of 20 MPG: {percentile_rank(mtcars['mpg'], 20):.1f}%")
    print(f"Percentile rank of 25 MPG: {percentile_rank(mtcars['mpg'], 25):.1f}%")

def robust_variability_measures():
    """
    Demonstrate robust measures of variability including MAD and quartile coefficient of dispersion.
    """
    mtcars = load_mtcars()
    
    # MAD around median
    mad_median = median_abs_deviation(mtcars['mpg'])
    print(f"MAD around median: {mad_median:.3f}")
    
    # MAD around mean
    mad_mean = median_abs_deviation(mtcars['mpg'], center=np.mean(mtcars['mpg']))
    print(f"MAD around mean: {mad_mean:.3f}")
    
    # Compare MAD with IQR
    print(f"IQR: {iqr(mtcars['mpg']):.3f}")
    print(f"MAD around median: {mad_median:.3f}")
    print(f"Ratio IQR/MAD: {iqr(mtcars['mpg']) / mad_median:.3f}")
    
    # For normal distribution, IQR ≈ 1.349 × MAD
    print("Expected ratio for normal distribution: 1.349")

def qcd(x):
    """
    Calculate quartile coefficient of dispersion.
    """
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    return (q3 - q1) / (q3 + q1)

def quartile_coefficient_examples():
    """
    Demonstrate quartile coefficient of dispersion and compare with CV.
    """
    mtcars = load_mtcars()
    
    print(f"Quartile coefficient of dispersion: {qcd(mtcars['mpg']):.3f}")
    
    # Compare with CV
    print(f"Coefficient of variation: {cv(mtcars['mpg']):.2f}%")
    print(f"Quartile coefficient of dispersion: {qcd(mtcars['mpg']):.3f}")

def detect_outliers_iqr(x):
    """
    Detect outliers using IQR method.
    """
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr_val = q3 - q1
    lower_bound = q1 - 1.5 * iqr_val
    upper_bound = q3 + 1.5 * iqr_val
    
    outliers = (x < lower_bound) | (x > upper_bound)
    return {
        'outliers': x[outliers],
        'outlier_indices': np.where(outliers)[0],
        'bounds': [lower_bound, upper_bound],
        'n_outliers': np.sum(outliers)
    }

def detect_outliers_zscore(x, threshold=3):
    """
    Detect outliers using z-score method.
    """
    z_scores = np.abs((x - np.mean(x)) / np.std(x, ddof=1))
    outliers = z_scores > threshold
    
    return {
        'outliers': x[outliers],
        'outlier_indices': np.where(outliers)[0],
        'z_scores': z_scores,
        'n_outliers': np.sum(outliers)
    }

def outlier_detection_examples():
    """
    Demonstrate outlier detection using IQR and z-score methods.
    """
    mtcars = load_mtcars()
    
    # Detect outliers in MPG
    mpg_outliers_iqr = detect_outliers_iqr(mtcars['mpg'])
    mpg_outliers_zscore = detect_outliers_zscore(mtcars['mpg'])
    
    print("=== OUTLIER DETECTION FOR MPG ===")
    print(f"IQR method outliers: {mpg_outliers_iqr['outliers'].tolist()}")
    print(f"IQR method count: {mpg_outliers_iqr['n_outliers']}")
    print(f"Z-score method outliers: {mpg_outliers_zscore['outliers'].tolist()}")
    print(f"Z-score method count: {mpg_outliers_zscore['n_outliers']}")

def variability_summary(x, variable_name="Variable"):
    """
    Create a comprehensive summary of all variability measures for a dataset.
    """
    print(f"=== VARIABILITY SUMMARY FOR {variable_name} ===")
    print(f"Data length: {len(x)}")
    print(f"Missing values: {np.sum(np.isnan(x))}\n")
    
    print("Basic Measures:")
    print(f"Range: {np.max(x) - np.min(x):.3f}")
    print(f"IQR: {iqr(x):.3f}")
    print(f"Variance: {np.var(x, ddof=1):.3f}")
    print(f"Standard Deviation: {np.std(x, ddof=1):.3f}")
    print(f"Coefficient of Variation: {(np.std(x, ddof=1) / np.mean(x)) * 100:.2f}%")
    
    print("\nRobust Measures:")
    print(f"MAD (around median): {median_abs_deviation(x):.3f}")
    print(f"MAD (around mean): {median_abs_deviation(x, center=np.mean(x)):.3f}")
    print(f"Quartile coefficient of dispersion: {qcd(x):.3f}")
    
    print("\nPercentiles:")
    percentiles = np.percentile(x, [5, 25, 50, 75, 95])
    print(f"5th percentile: {percentiles[0]:.3f}")
    print(f"25th percentile: {percentiles[1]:.3f}")
    print(f"50th percentile (median): {percentiles[2]:.3f}")
    print(f"75th percentile: {percentiles[3]:.3f}")
    print(f"95th percentile: {percentiles[4]:.3f}")

def comprehensive_variability_examples():
    """
    Apply comprehensive variability summary to different variables.
    """
    mtcars = load_mtcars()
    
    # Apply to different variables
    variability_summary(mtcars['mpg'], "MPG")
    print()
    variability_summary(mtcars['wt'], "Weight")

def compare_variability_across_groups():
    """
    Compare MPG variability across transmission types with statistical testing.
    """
    mtcars = load_mtcars()
    
    # Compare MPG variability across transmission types
    auto_mpg = mtcars[mtcars['am'] == 0]['mpg']
    manual_mpg = mtcars[mtcars['am'] == 1]['mpg']
    
    print("=== MPG VARIABILITY BY TRANSMISSION ===")
    print("Automatic transmission:")
    variability_summary(auto_mpg, "Auto MPG")
    
    print("\nManual transmission:")
    variability_summary(manual_mpg, "Manual MPG")
    
    # Test for equal variances
    var_test = levene(auto_mpg, manual_mpg)
    print(f"\nLevene's test for equal variances:")
    print(f"Statistic: {var_test.statistic:.3f}")
    print(f"p-value: {var_test.pvalue:.4f}")
    
    # Visualize variability comparison
    transmission_data = pd.DataFrame({
        'transmission': ['Automatic'] * len(auto_mpg) + ['Manual'] * len(manual_mpg),
        'mpg': list(auto_mpg) + list(manual_mpg)
    })
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=transmission_data, x='transmission', y='mpg', color='lightblue', alpha=0.7)
    plt.title('MPG Variability by Transmission Type')
    plt.xlabel('Transmission')
    plt.ylabel('MPG')
    plt.show()

def rolling_sd(x, window=10):
    """
    Calculate rolling standard deviation for time series data.
    """
    n = len(x)
    result = np.zeros(n - window + 1)
    
    for i in range(n - window + 1):
        result[i] = np.std(x[i:(i + window)], ddof=1)
    
    return result

def time_series_variability():
    """
    Demonstrate time series variability analysis with rolling standard deviation.
    """
    # Simulate time series data
    np.random.seed(123)
    time_series = np.cumsum(np.random.normal(0, 1, 100))
    
    # Apply rolling SD
    rolling_volatility = rolling_sd(time_series, window=10)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_volatility, 'b-')
    plt.title('Rolling Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Standard Deviation')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate overall variability measures
    print("=== TIME SERIES VARIABILITY ===")
    print(f"Overall SD: {np.std(time_series, ddof=1):.3f}")
    print(f"Mean rolling SD: {np.mean(rolling_volatility):.3f}")
    print(f"SD of rolling SD: {np.std(rolling_volatility, ddof=1):.3f}")

def financial_data_analysis():
    """
    Demonstrate financial data variability analysis with volatility measures.
    """
    # Simulate stock returns
    np.random.seed(123)
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
    
    # Calculate volatility measures
    print("=== STOCK RETURN VOLATILITY ===")
    print(f"Daily volatility (SD): {np.std(returns, ddof=1):.4f}")
    print(f"Annualized volatility: {np.std(returns, ddof=1) * np.sqrt(252):.4f}")
    print(f"Downside deviation: {np.std(returns[returns < 0], ddof=1):.4f}")
    
    # Value at Risk (VaR)
    var_95 = np.percentile(returns, 5)
    print(f"95% VaR: {var_95:.4f}")
    
    # Expected Shortfall (Conditional VaR)
    es_95 = np.mean(returns[returns <= var_95])
    print(f"95% Expected Shortfall: {es_95:.4f}")
    
    # Rolling volatility
    rolling_vol = rolling_sd(returns, window=20)
    print(f"Mean 20-day rolling volatility: {np.mean(rolling_vol):.4f}")
    print(f"Volatility of volatility: {np.std(rolling_vol, ddof=1):.4f}")

def visualize_variability():
    """
    Create comprehensive visualizations of variability measures.
    """
    mtcars = load_mtcars()
    
    # Box plots to show variability
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(data=mtcars, x='cyl', y='mpg', color='lightblue')
    plt.title('MPG Variability by Cylinders')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('Miles per Gallon')
    
    # Violin plots for density and variability
    plt.subplot(2, 2, 2)
    sns.violinplot(data=mtcars, x='cyl', y='mpg', color='lightgreen', alpha=0.7)
    sns.boxplot(data=mtcars, x='cyl', y='mpg', width=0.2, color='white')
    plt.title('MPG Distribution by Cylinders')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('Miles per Gallon')
    
    # Histogram with variability measures
    plt.subplot(2, 2, 3)
    plt.hist(mtcars['mpg'], bins=15, color='lightblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(mtcars['mpg']), color='red', linestyle='--', label='Mean')
    plt.axvline(np.mean(mtcars['mpg']) + np.std(mtcars['mpg'], ddof=1), color='blue', linestyle=':', label='+1 SD')
    plt.axvline(np.mean(mtcars['mpg']) - np.std(mtcars['mpg'], ddof=1), color='blue', linestyle=':', label='-1 SD')
    plt.title('MPG Distribution with Mean and ±1 SD')
    plt.xlabel('Miles per Gallon')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Scatter plot with error bars
    plt.subplot(2, 2, 4)
    plt.scatter(mtcars['wt'], mtcars['mpg'], alpha=0.7)
    plt.xlabel('Weight')
    plt.ylabel('MPG')
    plt.title('MPG vs Weight')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def choose_variability_measure(x):
    """
    Decision tree function to help choose the appropriate measure of variability.
    """
    print("=== VARIABILITY MEASURE DECISION TREE ===")
    
    # Check for missing values
    if np.sum(np.isnan(x)) > 0:
        print("Data contains missing values - use nan functions")
    
    # Check for outliers
    outliers_iqr = detect_outliers_iqr(x)
    outliers_zscore = detect_outliers_zscore(x)
    
    if outliers_iqr['n_outliers'] > 0 or outliers_zscore['n_outliers'] > 0:
        print("Outliers detected")
        print("Recommendation: Use IQR or MAD (robust measures)")
        return "IQR"
    
    # Check for skewness
    mean_val = np.mean(x)
    median_val = np.median(x)
    skewness_indicator = abs(mean_val - median_val) / mean_val
    
    if skewness_indicator > 0.1:
        print("Skewed distribution detected")
        print("Recommendation: Use IQR or MAD")
        return "IQR"
    else:
        print("Symmetric distribution")
        print("Recommendation: Use Standard Deviation")
        return "SD"

def robust_variability(x):
    """
    Function to handle missing data in variability calculations.
    """
    # Remove missing values
    x_clean = x[~np.isnan(x)]
    
    if len(x_clean) == 0:
        return np.nan
    
    # Calculate measures
    result = {
        'range': np.max(x_clean) - np.min(x_clean),
        'iqr': iqr(x_clean),
        'variance': np.var(x_clean, ddof=1),
        'sd': np.std(x_clean, ddof=1),
        'cv': (np.std(x_clean, ddof=1) / np.mean(x_clean)) * 100,
        'mad': median_abs_deviation(x_clean),
        'qcd': qcd(x_clean),
        'n': len(x_clean),
        'missing': np.sum(np.isnan(x)),
        'missing_percent': np.sum(np.isnan(x)) / len(x) * 100
    }
    
    return result

if __name__ == "__main__":
    # Example usage
    basic_range_calculation()
    range_by_group()
    basic_iqr_calculation()
    iqr_by_group()
    variance_comparison()
    variance_by_group()
    basic_standard_deviation()
    standard_deviation_by_group()
    coefficient_of_variation_examples()
    mad_examples()
    percentiles_examples()
    percentile_rank_examples()
    robust_variability_measures()
    quartile_coefficient_examples()
    outlier_detection_examples()
    comprehensive_variability_examples()
    compare_variability_across_groups()
    time_series_variability()
    financial_data_analysis()
    visualize_variability()
    mtcars = load_mtcars()
    choose_variability_measure(mtcars['mpg'])
    data_with_na = [1, 2, np.nan, 4, 5, np.nan, 7, 8]
    result = robust_variability(data_with_na)
    for key, value in result.items():
        print(f"{key}: {value}") 