"""
Exploratory Data Analysis (EDA) - Python Code Examples
Corresponds to: 04_exploratory_data_analysis.md
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from scipy.stats import skew, kurtosis, shapiro
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import boxcox, yeojohnson

# Load sample data
def load_mtcars():
    """
    Load the mtcars dataset from OpenML as a pandas DataFrame.
    """
    mtcars = fetch_openml(name='mtcars', as_frame=True).frame
    return mtcars

def initial_data_inspection():
    """
    Perform initial inspection of the mtcars dataset: structure, types, and memory usage.
    """
    mtcars = load_mtcars()
    print("--- DataFrame Info ---")
    print(mtcars.info())
    print("\n--- Shape (rows, columns) ---")
    print(mtcars.shape)
    print("\n--- Columns ---")
    print(mtcars.columns)
    print("\n--- Data Types ---")
    print(mtcars.dtypes)
    print("\n--- First 10 Rows ---")
    print(mtcars.head(10))
    print("\n--- Last 5 Rows ---")
    print(mtcars.tail(5))
    print("\n--- Memory Usage ---")
    print(mtcars.memory_usage(deep=True).sum(), 'bytes')
    print(f"{mtcars.memory_usage(deep=True).sum() / 1024 ** 2:.3f} MB")

def summary_statistics():
    """
    Print comprehensive summary statistics for the mtcars dataset, including group summaries and coefficient of variation.
    """
    mtcars = load_mtcars()
    print("--- Comprehensive Summary (all variables) ---")
    print(mtcars.describe(include='all'))
    print("\n--- Numeric Variables Summary ---")
    print(mtcars.select_dtypes(include=[np.number]).describe())
    print("\n--- Summary by Transmission (am) ---")
    summary_by_am = mtcars.groupby('am').agg(
        n=('mpg', 'count'),
        mean_mpg=('mpg', 'mean'),
        sd_mpg=('mpg', 'std'),
        median_mpg=('mpg', 'median'),
        min_mpg=('mpg', 'min'),
        max_mpg=('mpg', 'max'),
        q25=('mpg', lambda x: x.quantile(0.25)),
        q75=('mpg', lambda x: x.quantile(0.75)),
        iqr=('mpg', lambda x: x.quantile(0.75) - x.quantile(0.25))
    )
    print(summary_by_am)
    cv_mpg = mtcars['mpg'].std() / mtcars['mpg'].mean()
    print(f"\nCoefficient of variation for MPG: {cv_mpg:.3f}")

def missing_values_analysis():
    """
    Analyze and visualize missing values in the mtcars dataset.
    """
    mtcars = load_mtcars()
    missing_summary = mtcars.isnull().sum()
    print("--- Missing Values Summary ---")
    print(missing_summary)
    missing_percent = (missing_summary / len(mtcars)) * 100
    missing_data = pd.DataFrame({
        'variable': missing_summary.index,
        'missing_count': missing_summary.values,
        'missing_percent': missing_percent.values,
        'data_type': mtcars.dtypes.values
    })
    print("\n--- Variables with Missing Data ---")
    print(missing_data[missing_data['missing_count'] > 0])
    # Visualize missing data
    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=missing_data,
        y='variable', x='missing_percent', color='steelblue',
        order=missing_data.sort_values('missing_percent', ascending=False)['variable']
    )
    plt.title('Missing Values by Variable')
    plt.xlabel('Percentage Missing')
    plt.ylabel('Variable')
    plt.show()
    # Check for patterns in missing data
    missing_pattern = mtcars.isnull()
    missing_corr = missing_pattern.corr()
    print("\n--- Correlation of Missingness ---")
    print(missing_corr.round(3))

def duplicate_detection():
    """
    Detect duplicate rows and values in the mtcars dataset, and provide indices of duplicates.
    """
    mtcars = load_mtcars()
    duplicates = mtcars.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    duplicate_mpg = mtcars['mpg'].duplicated().sum()
    print(f"Duplicate MPG values: {duplicate_mpg}")
    if duplicates > 0:
        duplicate_indices = mtcars[mtcars.duplicated()].index.tolist()
        print(f"Duplicate row indices: {duplicate_indices}")
        print(mtcars.loc[duplicate_indices])
    # Near-duplicates (simple similarity)
    def near_duplicates(df, threshold=0.95):
        n_rows = df.shape[0]
        high_similarity = []
        for i in range(n_rows):
            for j in range(i+1, n_rows):
                sim = (df.iloc[i] == df.iloc[j]).sum() / df.shape[1]
                if sim > threshold:
                    high_similarity.append((i, j))
        return high_similarity
    # Example usage (commented out for performance):
    # print(near_duplicates(mtcars))

def detect_outliers(x, method="iqr"):
    """
    Detect outliers in a numeric array using IQR, z-score, or modified z-score methods.
    """
    x = np.array(x)
    if method == "iqr":
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (x < lower_bound) | (x > upper_bound)
    elif method == "zscore":
        z_scores = np.abs((x - np.mean(x)) / np.std(x))
        outliers = z_scores > 3
    elif method == "modified_zscore":
        median_val = np.median(x)
        mad_val = np.median(np.abs(x - median_val))
        modified_z_scores = 0.6745 * (x - median_val) / mad_val
        outliers = np.abs(modified_z_scores) > 3.5
    else:
        raise ValueError("Unknown method")
    return {
        'outliers': outliers,
        'outlier_indices': np.where(outliers)[0],
        'outlier_values': x[outliers],
        'method': method
    }

def outlier_summary_all_numeric():
    """
    Apply outlier detection to all numeric variables in mtcars and print summary.
    """
    mtcars = load_mtcars()
    numeric_vars = mtcars.select_dtypes(include=[np.number]).columns
    outlier_summary = {var: detect_outliers(mtcars[var].dropna()) for var in numeric_vars}
    for var, summary in outlier_summary.items():
        n_outliers = summary['outliers'].sum()
        if n_outliers > 0:
            print(f"{var}: {n_outliers} outliers detected")
            print(f"  Outlier values: {summary['outlier_values']}")

def compare_outlier_methods(x, var_name):
    """
    Compare IQR, z-score, and modified z-score outlier detection methods for a variable.
    """
    iqr_result = detect_outliers(x, "iqr")
    zscore_result = detect_outliers(x, "zscore")
    modified_z_result = detect_outliers(x, "modified_zscore")
    print(f"=== OUTLIER DETECTION COMPARISON FOR {var_name} ===")
    print(f"IQR method: {iqr_result['outliers'].sum()} outliers")
    print(f"Z-score method: {zscore_result['outliers'].sum()} outliers")
    print(f"Modified Z-score method: {modified_z_result['outliers'].sum()} outliers")
    return {
        'iqr': iqr_result,
        'zscore': zscore_result,
        'modified_zscore': modified_z_result
    }

def analyze_distribution(x, var_name):
    """
    Analyze distribution characteristics (mean, median, sd, skewness, kurtosis, normality) for a variable.
    """
    mean_val = np.mean(x)
    median_val = np.median(x)
    sd_val = np.std(x, ddof=1)
    print(f"=== DISTRIBUTION ANALYSIS FOR {var_name} ===")
    print(f"Mean: {mean_val:.3f}")
    print(f"Median: {median_val:.3f}")
    print(f"Standard Deviation: {sd_val:.3f}")
    skewness_val = skew(x, nan_policy='omit')
    kurtosis_val = kurtosis(x, nan_policy='omit', fisher=False)
    print(f"Skewness: {skewness_val:.3f}")
    print(f"Kurtosis: {kurtosis_val:.3f}")
    if abs(skewness_val) < 0.5:
        print("Distribution is approximately symmetric")
    elif skewness_val > 0.5:
        print("Distribution is right-skewed (positive skew)")
    else:
        print("Distribution is left-skewed (negative skew)")
    if kurtosis_val < 3:
        print("Distribution has lighter tails than normal (platykurtic)")
    elif kurtosis_val > 3:
        print("Distribution has heavier tails than normal (leptokurtic)")
    else:
        print("Distribution has normal-like tails (mesokurtic)")
    quantiles = np.percentile(x, [5, 25, 50, 75, 95])
    print("Quantiles:")
    print(np.round(quantiles, 3))
    shapiro_test = shapiro(x)
    print(f"Shapiro-Wilk normality test p-value: {shapiro_test.pvalue:.4f}")
    if shapiro_test.pvalue < 0.05:
        print("Data is NOT normally distributed (p < 0.05)")
    else:
        print("Data appears to be normally distributed (p >= 0.05)")
    return {
        'mean': mean_val,
        'median': median_val,
        'sd': sd_val,
        'skewness': skewness_val,
        'kurtosis': kurtosis_val,
        'shapiro_p': shapiro_test.pvalue
    }

def create_univariate_plots(data, variable):
    """
    Create histogram, boxplot, Q-Q plot, and CDF for a univariate variable.
    """
    x = data[variable].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Histogram with density
    sns.histplot(x, kde=True, ax=axes[0, 0], color='steelblue', bins=15)
    axes[0, 0].axvline(x.mean(), color='green', linestyle='dashed', linewidth=1, label='Mean')
    axes[0, 0].axvline(x.median(), color='orange', linestyle='dashed', linewidth=1, label='Median')
    axes[0, 0].set_title(f"Distribution of {variable}")
    axes[0, 0].legend()
    # Box plot
    sns.boxplot(y=x, ax=axes[0, 1], color='lightgreen')
    sns.stripplot(y=x, ax=axes[0, 1], color='black', alpha=0.5)
    axes[0, 1].set_title(f"Boxplot of {variable}")
    # Q-Q plot
    stats.probplot(x, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f"Q-Q Plot of {variable}")
    # CDF
    sns.ecdfplot(x, ax=axes[1, 1])
    axes[1, 1].set_title(f"Cumulative Distribution of {variable}")
    plt.tight_layout()
    plt.show()

def correlation_matrix_and_heatmap(data):
    """
    Compute and visualize the correlation matrix for numeric variables in the dataset.
    """
    correlation_matrix = data.select_dtypes(include=[np.number]).corr()
    print(correlation_matrix.round(3))
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Matrix')
    plt.show()

def correlation_analysis(data, var1, var2):
    """
    Print Pearson, Spearman, and Kendall correlation between two variables, with p-values and strength interpretation.
    """
    from scipy.stats import pearsonr, spearmanr, kendalltau
    print(f"=== CORRELATION ANALYSIS ===")
    print(f"Variables: {var1} and {var2}")
    pearson_cor, pearson_p = pearsonr(data[var1], data[var2])
    print(f"Pearson correlation: {pearson_cor:.3f}")
    print(f"p-value: {pearson_p:.4f}")
    spearman_cor, spearman_p = spearmanr(data[var1], data[var2])
    print(f"Spearman correlation: {spearman_cor:.3f}")
    print(f"p-value: {spearman_p:.4f}")
    kendall_cor, kendall_p = kendalltau(data[var1], data[var2])
    print(f"Kendall's tau: {kendall_cor:.3f}")
    print(f"p-value: {kendall_p:.4f}")
    pearson_abs = abs(pearson_cor)
    if pearson_abs < 0.1:
        print("Correlation strength: Negligible")
    elif pearson_abs < 0.3:
        print("Correlation strength: Weak")
    elif pearson_abs < 0.5:
        print("Correlation strength: Moderate")
    elif pearson_abs < 0.7:
        print("Correlation strength: Strong")
    else:
        print("Correlation strength: Very strong")
    return {
        'pearson': pearson_cor,
        'spearman': spearman_cor,
        'kendall': kendall_cor
    }

def create_scatter_analysis(data, x_var, y_var):
    """
    Create scatter plots (with regression and LOWESS), residual plots, and print OLS summary for two variables.
    """
    x = data[x_var].values.reshape(-1, 1)
    y = data[y_var].values
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Scatter plot with regression line
    sns.regplot(x=x.flatten(), y=y, ax=axes[0, 0], line_kws={'color': 'red'}, scatter_kws={'alpha': 0.7})
    axes[0, 0].set_title(f"{y_var} vs {x_var} (Linear Regression)")
    # Scatter plot with LOWESS curve
    sns.regplot(x=x.flatten(), y=y, ax=axes[0, 1], lowess=True, line_kws={'color': 'blue'}, scatter_kws={'alpha': 0.7})
    axes[0, 1].set_title(f"{y_var} vs {x_var} (LOWESS)")
    # Residual plot
    model = LinearRegression().fit(x, y)
    fitted = model.predict(x)
    residuals = y - fitted
    axes[1, 0].scatter(fitted, residuals, alpha=0.7)
    axes[1, 0].axhline(0, color='red', linestyle='dashed')
    axes[1, 0].set_title("Residual Plot")
    axes[1, 0].set_xlabel("Fitted values")
    axes[1, 0].set_ylabel("Residuals")
    # Residuals vs predictor
    axes[1, 1].scatter(x, residuals, alpha=0.7)
    axes[1, 1].axhline(0, color='red', linestyle='dashed')
    axes[1, 1].set_title("Residuals vs Predictor")
    axes[1, 1].set_xlabel(x_var)
    axes[1, 1].set_ylabel("Residuals")
    plt.tight_layout()
    plt.show()
    # Print model summary
    x_with_const = sm.add_constant(x)
    ols_model = sm.OLS(y, x_with_const).fit()
    print(ols_model.summary())
    return ols_model

def create_group_comparison(data, group_var, response_var):
    """
    Create boxplot and violin plot for group comparisons, print group summary, and perform ANOVA or t-test as appropriate.
    """
    import scipy.stats as stats
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    # Box plot by group
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=group_var, y=response_var, data=data, palette='Set2')
    sns.stripplot(x=group_var, y=response_var, data=data, color='black', alpha=0.5)
    plt.title(f"{response_var} by {group_var}")
    # Violin plot
    plt.subplot(1, 2, 2)
    sns.violinplot(x=group_var, y=response_var, data=data, palette='Set2')
    sns.boxplot(x=group_var, y=response_var, data=data, width=0.2, showcaps=False, boxprops={'facecolor':'None'}, showfliers=False, whiskerprops={'linewidth':0}, color='k')
    plt.title(f"{response_var} by {group_var} (Violin Plot)")
    plt.tight_layout()
    plt.show()
    # Summary statistics by group
    group_summary = data.groupby(group_var)[response_var].agg(['count', 'mean', 'std', 'median', 'min', 'max', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    group_summary.columns = ['n', 'mean', 'sd', 'median', 'min', 'max', 'q25', 'q75']
    print(group_summary)
    # Statistical test for group differences
    unique_groups = data[group_var].nunique()
    if unique_groups > 2:
        # ANOVA
        model = smf.ols(f"{response_var} ~ C({group_var})", data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print("\n=== ANOVA RESULTS ===")
        print(anova_table)
    elif unique_groups == 2:
        # t-test
        groups = data[group_var].unique()
        group1 = data[data[group_var] == groups[0]][response_var]
        group2 = data[data[group_var] == groups[1]][response_var]
        t_stat, p_val = stats.ttest_ind(group1, group2, nan_policy='omit')
        print("\n=== T-TEST RESULTS ===")
        print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")

def apply_transformations(data, variable):
    """
    Apply various transformations (log, sqrt, reciprocal, Box-Cox, Yeo-Johnson) to a variable and test normality.
    """
    x = data[variable].dropna()
    # Log transformation
    log_transformed = np.log(x)
    # Square root transformation
    sqrt_transformed = np.sqrt(x)
    # Reciprocal transformation
    reciprocal_transformed = 1 / x
    # Box-Cox transformation (only for positive values)
    if (x > 0).all():
        bc_transformed, lambda_bc = boxcox(x)
    else:
        bc_transformed, lambda_bc = np.full_like(x, np.nan), np.nan
    # Yeo-Johnson transformation (can handle zero/negative)
    yj_transformed, lambda_yj = yeojohnson(x)
    # Plot distributions
    transformed_data = pd.DataFrame({
        'Original': x,
        'Log': log_transformed,
        'Sqrt': sqrt_transformed,
        'Reciprocal': reciprocal_transformed,
        'BoxCox': bc_transformed,
        'YeoJohnson': yj_transformed
    })
    transformed_data_melted = transformed_data.melt(var_name='Transformation', value_name='Value')
    g = sns.FacetGrid(transformed_data_melted, col='Transformation', col_wrap=3, sharex=False, sharey=False)
    g.map(sns.histplot, 'Value', bins=15, color='steelblue', alpha=0.7)
    g.fig.suptitle(f"Distribution of {variable} under different transformations", y=1.05)
    plt.show()
    # Test normality for each transformation
    print("=== NORMALITY TESTS ===")
    for col in transformed_data.columns:
        vals = transformed_data[col].dropna()
        if len(vals) > 3:
            stat, p = shapiro(vals)
            print(f"{col} Shapiro-Wilk p-value: {p:.4f}")
    return {
        'original': x,
        'log': log_transformed,
        'sqrt': sqrt_transformed,
        'reciprocal': reciprocal_transformed,
        'boxcox': bc_transformed,
        'yeojohnson': yj_transformed,
        'lambda_boxcox': lambda_bc,
        'lambda_yeojohnson': lambda_yj
    }

def check_data_consistency(data):
    """
    Check for logical inconsistencies, extreme values, data type issues, and missing value patterns in the dataset.
    """
    print("=== DATA CONSISTENCY CHECKS ===")
    # Check for logical inconsistencies
    if 'mpg' in data.columns and 'wt' in data.columns:
        correlation = data['mpg'].corr(data['wt'])
        print(f"MPG-Weight correlation: {correlation:.3f}")
        if correlation > 0:
            print("WARNING: Positive correlation between weight and MPG (unexpected)")
    # Check for extreme values
    numeric_vars = data.select_dtypes(include=[np.number]).columns
    for var in numeric_vars:
        values = data[var].dropna()
        extreme_low = (values < np.percentile(values, 1)).sum()
        extreme_high = (values > np.percentile(values, 99)).sum()
        if extreme_low > 0 or extreme_high > 0:
            print(f"{var}: {extreme_low} extreme low values, {extreme_high} extreme high values")
        if var == 'mpg' and (values < 0).any():
            print("WARNING: mpg contains negative values (impossible for MPG)")
        if var == 'wt' and (values <= 0).any():
            print("WARNING: wt contains zero or negative values (impossible for weight)")
    # Check for data type consistency
    for var in data.columns:
        if pd.api.types.is_categorical_dtype(data[var]) or data[var].dtype == object:
            n_levels = data[var].nunique()
            if n_levels > 20:
                print(f"WARNING: {var} has {n_levels} levels (consider if this is correct)")
            level_counts = data[var].value_counts()
            rare_levels = level_counts[level_counts < 3]
            if len(rare_levels) > 0:
                print(f"WARNING: {var} has rare levels: {list(rare_levels.index)}")
    # Check for duplicate observations
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate rows found")
    # Check for missing value patterns
    missing_pattern = data.isnull()
    if missing_pattern.values.sum() > 0:
        missing_corr = missing_pattern.corr()
        high_missing_corr = ((abs(missing_corr) > 0.7) & (abs(missing_corr) < 1)).sum().sum()
        if high_missing_corr > 0:
            print("WARNING: Missing values are correlated between variables")

def create_validation_rules(data):
    """
    Create and apply validation rules for data quality checks (range, valid values, etc.).
    """
    print("=== DATA VALIDATION RULES ===")
    rules = {
        'mpg': {'min': 0, 'max': 50, 'description': 'MPG should be between 0 and 50'},
        'wt': {'min': 0.5, 'max': 10, 'description': 'Weight should be between 0.5 and 10 tons'},
        'cyl': {'valid_values': [4, 6, 8], 'description': 'Cylinders should be 4, 6, or 8'}
    }
    violations = {}
    for var, rule in rules.items():
        if var in data.columns:
            values = data[var]
            if 'min' in rule and 'max' in rule:
                violation = (values < rule['min']) | (values > rule['max'])
                n_violations = violation.sum()
                if n_violations > 0:
                    print(f"{var}: {n_violations} values outside range [{rule['min']}, {rule['max']}]")
                violations[var] = violation
            elif 'valid_values' in rule:
                violation = ~values.isin(rule['valid_values'])
                n_violations = violation.sum()
                if n_violations > 0:
                    print(f"{var}: {n_violations} invalid values")
                violations[var] = violation
    return violations

def generate_eda_report(data, dataset_name="Dataset"):
    """
    Generate a comprehensive EDA report including dataset info, data quality, summary statistics, and recommendations.
    """
    print("=== EXPLORATORY DATA ANALYSIS REPORT ===\n")
    print(f"Dataset: {dataset_name}")
    print(f"Dimensions: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024 ** 2:.3f} MB\n")
    # Data types summary
    print("DATA TYPES:")
    print(data.dtypes)
    print()
    # Missing values summary
    missing_summary = data.isnull().sum()
    if missing_summary.sum() > 0:
        print("MISSING VALUES:")
        for var in missing_summary.index:
            if missing_summary[var] > 0:
                print(f"  {var}: {missing_summary[var]} ({missing_summary[var]/len(data)*100:.1f}%)")
        print()
    else:
        print("No missing values found.\n")
    # Summary statistics for numeric variables
    numeric_vars = data.select_dtypes(include=[np.number]).columns
    if len(numeric_vars) > 0:
        print("NUMERIC VARIABLES SUMMARY:")
        print(data[numeric_vars].describe())
        print()
        # Distribution characteristics
        print("DISTRIBUTION CHARACTERISTICS:")
        for var in numeric_vars:
            values = data[var].dropna()
            skewness_val = skew(values)
            kurtosis_val = kurtosis(values, fisher=False)
            print(f"  {var}:")
            print(f"    Skewness: {skewness_val:.3f}", end='')
            if abs(skewness_val) < 0.5:
                print(" (symmetric)")
            elif skewness_val > 0.5:
                print(" (right-skewed)")
            else:
                print(" (left-skewed)")
            print(f"    Kurtosis: {kurtosis_val:.3f}", end='')
            if kurtosis_val < 3:
                print(" (light tails)")
            elif kurtosis_val > 3:
                print(" (heavy tails)")
            else:
                print(" (normal-like)")
        print()
    # Key findings
    print("KEY FINDINGS:")
    # Find strongest correlations
    if len(numeric_vars) > 1:
        cor_matrix = data[numeric_vars].corr()
        np.fill_diagonal(cor_matrix.values, 0)
        max_cor = np.unravel_index(np.abs(cor_matrix.values).argmax(), cor_matrix.shape)
        var1 = cor_matrix.index[max_cor[0]]
        var2 = cor_matrix.columns[max_cor[1]]
        max_cor_value = cor_matrix.values[max_cor]
        print(f"  Strongest correlation: {var1} and {var2} ({max_cor_value:.3f})")
        # Find high correlations
        high_correlations = np.where((np.abs(cor_matrix.values) > 0.7) & (np.abs(cor_matrix.values) < 1))
        if len(high_correlations[0]) > 0:
            print("  High correlations (|r| > 0.7):")
            for i, j in zip(*high_correlations):
                print(f"    {cor_matrix.index[i]} - {cor_matrix.columns[j]}: {cor_matrix.values[i, j]:.3f}")
    # Find variables with most variation
    if len(numeric_vars) > 0:
        cv_values = data[numeric_vars].std() / data[numeric_vars].mean()
        most_variable = cv_values.idxmax()
        print(f"  Most variable numeric variable: {most_variable} (CV = {cv_values.max():.3f})")
    # Outlier summary
    if len(numeric_vars) > 0:
        outlier_counts = {}
        for var in numeric_vars:
            x = data[var].dropna()
            q1 = np.percentile(x, 25)
            q3 = np.percentile(x, 75)
            iqr = q3 - q1
            outlier_counts[var] = ((x < (q1 - 1.5 * iqr)) | (x > (q3 + 1.5 * iqr))).sum()
        variables_with_outliers = [var for var, count in outlier_counts.items() if count > 0]
        if variables_with_outliers:
            print("  Variables with outliers:")
            for var in variables_with_outliers:
                print(f"    {var}: {outlier_counts[var]} outliers")
    print("\nRECOMMENDATIONS:")
    print("1. Consider transformations for highly skewed variables")
    print("2. Investigate outliers and their impact on analysis")
    print("3. Check for multicollinearity in regression models")
    print("4. Consider interaction effects in modeling")
    print("5. Validate data quality before proceeding with analysis")

def automated_eda_pipeline(data, dataset_name="Dataset"):
    """
    Run an automated EDA pipeline with systematic steps and recommendations.
    """
    print("=== AUTOMATED EDA PIPELINE ===\n")
    # Step 1: Data overview
    print(f"STEP 1: DATA OVERVIEW\nDimensions: {data.shape[0]} × {data.shape[1]}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024 ** 2:.3f} MB")
    print(f"Data types: {', '.join(data.dtypes.unique().astype(str))}\n")
    # Step 2: Data quality assessment
    missing_count = data.isnull().sum().sum()
    print(f"STEP 2: DATA QUALITY ASSESSMENT\nTotal missing values: {missing_count}")
    duplicate_count = data.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")
    # Step 3: Univariate analysis
    numeric_vars = data.select_dtypes(include=[np.number]).columns
    print(f"STEP 3: UNIVARIATE ANALYSIS\nNumeric variables: {len(numeric_vars)}")
    print(f"Categorical variables: {data.shape[1] - len(numeric_vars)}")
    if len(numeric_vars) > 0:
        print("Distribution analysis:")
        for var in numeric_vars:
            values = data[var].dropna()
            skewness_val = skew(values)
            print(f"  {var}: skewness = {skewness_val:.3f}", end='')
            if abs(skewness_val) > 1:
                print(" (highly skewed - consider transformation)")
            else:
                print(" (moderate skew)")
    print()
    # Step 4: Bivariate analysis
    if len(numeric_vars) > 1:
        cor_matrix = data[numeric_vars].corr()
        high_correlations = np.where((np.abs(cor_matrix.values) > 0.7) & (np.abs(cor_matrix.values) < 1))
        if len(high_correlations[0]) > 0:
            print("STEP 4: BIVARIATE ANALYSIS\nHigh correlations detected:")
            for i, j in zip(*high_correlations):
                print(f"  {cor_matrix.index[i]} - {cor_matrix.columns[j]}: {cor_matrix.values[i, j]:.3f}")
        else:
            print("STEP 4: BIVARIATE ANALYSIS\nNo high correlations detected")
    print()
    # Step 5: Recommendations
    print("STEP 5: RECOMMENDATIONS")
    if missing_count > 0:
        print("- Address missing values before analysis")
    if duplicate_count > 0:
        print("- Remove or investigate duplicate records")
    if len(numeric_vars) > 1 and len(high_correlations[0]) > 0:
        print("- Consider multicollinearity in regression models")
    if len(numeric_vars) > 0:
        skewed_vars = [var for var in numeric_vars if abs(skew(data[var].dropna())) > 1]
        if len(skewed_vars) > 0:
            print("- Transform highly skewed variables")
    print("- Check for outliers and their impact")
    print("- Validate statistical assumptions")
    print("- Consider interaction effects in modeling")

def create_customer_satisfaction_data():
    """
    Create simulated customer satisfaction data for EDA practice.
    """
    np.random.seed(123)
    n_customers = 200
    customer_data = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'age': np.random.normal(45, 15, n_customers).round(0),
        'satisfaction_score': np.random.normal(7.5, 1.5, n_customers).round(1),
        'purchase_amount': np.random.exponential(100, n_customers).round(2),
        'service_quality': np.random.choice([1, 2, 3, 4, 5], n_customers, p=[0.05, 0.1, 0.2, 0.4, 0.25]),
        'loyalty_program': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], n_customers, p=[0.4, 0.3, 0.2, 0.1]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_customers)
    })
    # Add some realistic patterns
    customer_data.loc[customer_data['loyalty_program'] == 'Platinum', 'satisfaction_score'] += 0.5
    customer_data.loc[customer_data['service_quality'] == 5, 'satisfaction_score'] += 0.3
    customer_data.loc[customer_data['age'] > 60, 'purchase_amount'] *= 1.2
    # Add some missing values
    customer_data.loc[np.random.choice(customer_data.index, 10), 'satisfaction_score'] = np.nan
    customer_data.loc[np.random.choice(customer_data.index, 5), 'purchase_amount'] = np.nan
    return customer_data

def analyze_customer_data(data):
    """
    Perform comprehensive EDA on customer satisfaction data.
    """
    print("=== CUSTOMER SATISFACTION ANALYSIS ===\n")
    
    # 1. Data quality assessment
    print("1. DATA QUALITY ASSESSMENT:")
    missing_summary = data.isnull().sum()
    if missing_summary.sum() > 0:
        print("Missing values found:")
        for var in missing_summary.index:
            if missing_summary[var] > 0:
                print(f"  {var}: {missing_summary[var]} ({missing_summary[var]/len(data)*100:.1f}%)")
    else:
        print("No missing values found.")
    print()
    
    # 2. Univariate analysis
    print("2. UNIVARIATE ANALYSIS:")
    numeric_vars = ['age', 'satisfaction_score', 'purchase_amount']
    for var in numeric_vars:
        values = data[var].dropna()
        print(f"\n{var.upper()}:")
        print(f"  Mean: {values.mean():.2f}")
        print(f"  Median: {values.median():.2f}")
        print(f"  Std Dev: {values.std():.2f}")
        print(f"  Range: [{values.min():.2f}, {values.max():.2f}]")
        print(f"  Skewness: {skew(values):.3f}")
    
    # 3. Categorical analysis
    print("\n3. CATEGORICAL ANALYSIS:")
    categorical_vars = ['service_quality', 'loyalty_program', 'region']
    for var in categorical_vars:
        print(f"\n{var.upper()}:")
        value_counts = data[var].value_counts()
        for level, count in value_counts.items():
            print(f"  {level}: {count} ({count/len(data)*100:.1f}%)")
    
    # 4. Bivariate analysis
    print("\n4. BIVARIATE ANALYSIS:")
    # Satisfaction vs Service Quality
    print("\nSatisfaction Score by Service Quality:")
    satisfaction_by_quality = data.groupby('service_quality')['satisfaction_score'].agg(['count', 'mean', 'std'])
    print(satisfaction_by_quality)
    
    # Satisfaction vs Loyalty Program
    print("\nSatisfaction Score by Loyalty Program:")
    satisfaction_by_loyalty = data.groupby('loyalty_program')['satisfaction_score'].agg(['count', 'mean', 'std'])
    print(satisfaction_by_loyalty)
    
    # Correlation analysis
    numeric_data = data[numeric_vars].dropna()
    corr_matrix = numeric_data.corr()
    print(f"\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # 5. Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Satisfaction distribution
    sns.histplot(data['satisfaction_score'].dropna(), kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Satisfaction Score Distribution')
    
    # Age distribution
    sns.histplot(data['age'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Age Distribution')
    
    # Purchase amount distribution
    sns.histplot(data['purchase_amount'].dropna(), kde=True, ax=axes[0, 2])
    axes[0, 2].set_title('Purchase Amount Distribution')
    
    # Satisfaction by service quality
    sns.boxplot(x='service_quality', y='satisfaction_score', data=data, ax=axes[1, 0])
    axes[1, 0].set_title('Satisfaction by Service Quality')
    
    # Satisfaction by loyalty program
    sns.boxplot(x='loyalty_program', y='satisfaction_score', data=data, ax=axes[1, 1])
    axes[1, 1].set_title('Satisfaction by Loyalty Program')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Age vs Satisfaction scatter
    sns.scatterplot(x='age', y='satisfaction_score', data=data, ax=axes[1, 2])
    axes[1, 2].set_title('Age vs Satisfaction Score')
    
    plt.tight_layout()
    plt.show()
    
    # 6. Key insights
    print("\n5. KEY INSIGHTS:")
    print("- Average satisfaction score:", f"{data['satisfaction_score'].mean():.2f}")
    print("- Highest satisfaction by service quality:", 
          f"{satisfaction_by_quality['mean'].idxmax()} ({satisfaction_by_quality['mean'].max():.2f})")
    print("- Highest satisfaction by loyalty program:", 
          f"{satisfaction_by_loyalty['mean'].idxmax()} ({satisfaction_by_loyalty['mean'].max():.2f})")
    print("- Age-satisfaction correlation:", f"{data['age'].corr(data['satisfaction_score']):.3f}")

if __name__ == "__main__":
    # Example usage
    initial_data_inspection()
    summary_statistics()
    missing_values_analysis()
    duplicate_detection()
    outlier_summary_all_numeric()
    mtcars = load_mtcars()
    compare_outlier_methods(mtcars['mpg'], "MPG")
    analyze_distribution(mtcars['mpg'].dropna(), "MPG")
    create_univariate_plots(mtcars, "mpg")
    correlation_matrix_and_heatmap(mtcars)
    correlation_analysis(mtcars, "mpg", "wt")
    create_scatter_analysis(mtcars, "wt", "mpg")
    mtcars['am_factor'] = mtcars['am'].map({0: 'Automatic', 1: 'Manual'})
    create_group_comparison(mtcars, 'am_factor', 'mpg')
    apply_transformations(mtcars, "mpg")
    check_data_consistency(mtcars)
    create_validation_rules(mtcars)
    generate_eda_report(mtcars, "Motor Trend Car Road Tests")
    automated_eda_pipeline(mtcars, "Motor Trend Car Road Tests")
    # Customer satisfaction example
    customer_data = create_customer_satisfaction_data()
    analyze_customer_data(customer_data) 