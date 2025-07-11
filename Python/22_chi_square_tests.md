# Chi-Square Tests

## Introduction

Chi-square tests are fundamental statistical methods for analyzing categorical data. They are nonparametric tests that examine relationships between categorical variables, test for independence, and assess goodness of fit. Chi-square tests are widely used in research across disciplines including psychology, medicine, social sciences, and business.

**Key Applications:**
- Testing independence between categorical variables
- Comparing observed frequencies to expected frequencies
- Assessing homogeneity across groups
- Analyzing survey responses and contingency tables
- Quality control and process monitoring

**When to Use Chi-Square Tests:**
- Data are categorical (nominal or ordinal)
- Observations are independent
- Expected frequencies meet minimum requirements
- Testing for relationships or differences in proportions

## Mathematical Foundations

### Chi-Square Distribution

The chi-square distribution is a continuous probability distribution with $`k`$ degrees of freedom. It is the distribution of the sum of squares of $`k`$ independent standard normal random variables.

**Probability Density Function:**
```math
f(x; k) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{k/2-1} e^{-x/2}, \quad x > 0
```

**Properties:**
- Always positive and right-skewed
- Mean = $`k`$ (degrees of freedom)
- Variance = $`2k`$
- As $`k`$ increases, approaches normal distribution

### Chi-Square Test Statistic

The general form of the chi-square test statistic is:
```math
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
```

where:
- $`O_i`$ = observed frequency for category $`i`$
- $`E_i`$ = expected frequency for category $`i`$
- $`k`$ = number of categories

Under the null hypothesis, this statistic follows a chi-square distribution with appropriate degrees of freedom.

## Types of Chi-Square Tests

### 1. Chi-Square Goodness of Fit Test

**Purpose:** Test whether observed frequencies match expected frequencies from a theoretical distribution.

**Null Hypothesis:** $`H_0: O_i = E_i`$ for all categories
**Alternative Hypothesis:** $`H_1: O_i \neq E_i`$ for at least one category

**Test Statistic:**
```math
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
```

**Degrees of Freedom:** $`df = k - 1 - p`$ where $`p`$ is the number of parameters estimated from the data.

**Python Implementation:**
```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Basic goodness of fit test
observed = np.array([45, 52, 38, 65])
expected = np.array([50, 50, 50, 50])  # Equal expected frequencies

# Perform chi-square goodness of fit test
chi_square_test = stats.chisquare(observed, f_exp=expected)

# Extract and display results
print("Chi-Square Goodness of Fit Test Results:")
print(f"Chi-square statistic: {chi_square_test.statistic:.3f}")
print(f"p-value: {chi_square_test.pvalue:.4f}")
print(f"Expected frequencies: {expected}")
print(f"Observed frequencies: {observed}")

# Manual calculation for understanding
def manual_chi_square(observed, expected):
    chi_square = np.sum((observed - expected)**2 / expected)
    df = len(observed) - 1
    p_value = 1 - stats.chi2.cdf(chi_square, df)
    residuals = (observed - expected) / np.sqrt(expected)
    
    return {
        'chi_square': chi_square,
        'df': df,
        'p_value': p_value,
        'residuals': residuals
    }

manual_result = manual_chi_square(observed, expected)
print("\nManual Calculation:")
print(f"Chi-square statistic: {manual_result['chi_square']:.3f}")
print(f"Standardized residuals: {manual_result['residuals']:.3f}")

# Visualize goodness of fit
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, observed, width, label='Observed', alpha=0.8)
plt.bar(x + width/2, expected, width, label='Expected', alpha=0.8)
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.title('Observed vs Expected Frequencies')
plt.xticks(x, categories)
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(categories, manual_result['residuals'])
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Categories')
plt.ylabel('Standardized Residuals')
plt.title('Standardized Residuals')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

### 2. Chi-Square Test of Independence

**Purpose:** Test whether two categorical variables are independent (not associated).

**Null Hypothesis:** $`H_0: \text{Variables are independent}`$
**Alternative Hypothesis:** $`H_1: \text{Variables are dependent}`$

**Test Statistic:**
```math
\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
```

where $`E_{ij} = \frac{R_i \times C_j}{N}`$ is the expected frequency for cell $`(i,j)`$.

**Degrees of Freedom:** $`df = (r-1) \times (c-1)`$ where $`r`$ and $`c`$ are the number of rows and columns.

**Python Implementation:**
```python
# Create contingency table
contingency_table = np.array([
    [45, 35, 25],
    [55, 40, 30],
    [30, 25, 15]
])
row_names = ['Low', 'Medium', 'High']
col_names = ['Group A', 'Group B', 'Group C']

# Create DataFrame for better display
df_table = pd.DataFrame(contingency_table, index=row_names, columns=col_names)
print("Contingency Table:")
print(df_table)

# Perform chi-square test of independence
independence_test = stats.chi2_contingency(contingency_table)

# Display results
print("\nChi-Square Test of Independence Results:")
print(f"Chi-square statistic: {independence_test.statistic:.3f}")
print(f"Degrees of freedom: {independence_test.dof}")
print(f"p-value: {independence_test.pvalue:.4f}")

# Expected frequencies and residuals
print("\nExpected Frequencies:")
expected_df = pd.DataFrame(independence_test.expected, index=row_names, columns=col_names)
print(expected_df.round(2))

# Calculate standardized residuals
observed = contingency_table
expected = independence_test.expected
residuals = (observed - expected) / np.sqrt(expected)

print("\nStandardized Residuals:")
residuals_df = pd.DataFrame(residuals, index=row_names, columns=col_names)
print(residuals_df.round(3))

# Identify significant cells
significant_cells = np.where(np.abs(residuals) > 2)
if len(significant_cells[0]) > 0:
    print("\nSignificant cells (|residual| > 2):")
    for i, j in zip(significant_cells[0], significant_cells[1]):
        row_name = row_names[i]
        col_name = col_names[j]
        residual = residuals[i, j]
        print(f"  {row_name} - {col_name}: {residual:.3f}")

# Visualize contingency table
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(df_table, annot=True, fmt='d', cmap='Blues')
plt.title('Observed Frequencies')

plt.subplot(1, 3, 2)
sns.heatmap(expected_df, annot=True, fmt='.1f', cmap='Greens')
plt.title('Expected Frequencies')

plt.subplot(1, 3, 3)
sns.heatmap(residuals_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
plt.title('Standardized Residuals')

plt.tight_layout()
plt.show()
```

### 3. Chi-Square Test of Homogeneity

**Purpose:** Test whether the distribution of a categorical variable is the same across different groups.

**Null Hypothesis:** $`H_0: \text{Proportions are equal across groups}`$
**Alternative Hypothesis:** $`H_1: \text{Proportions differ across groups}`$

**Test Statistic:** Same as independence test, but interpretation differs.

**Python Implementation:**
```python
# Create homogeneity test data
homogeneity_data = np.array([
    [20, 30, 25],
    [15, 25, 20],
    [10, 15, 10]
])
row_names = ['Treatment A', 'Treatment B', 'Treatment C']
col_names = ['Success', 'Partial', 'Failure']

df_homogeneity = pd.DataFrame(homogeneity_data, index=row_names, columns=col_names)
print("Homogeneity Test Data:")
print(df_homogeneity)

# Perform chi-square test of homogeneity
homogeneity_test = stats.chi2_contingency(homogeneity_data)

print("\nChi-Square Test of Homogeneity Results:")
print(f"Chi-square statistic: {homogeneity_test.statistic:.3f}")
print(f"Degrees of freedom: {homogeneity_test.dof}")
print(f"p-value: {homogeneity_test.pvalue:.4f}")

# Visualize homogeneity test
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(df_homogeneity, annot=True, fmt='d', cmap='Blues')
plt.title('Observed Frequencies')

plt.subplot(1, 2, 2)
# Stacked bar chart
df_homogeneity.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Stacked Bar Chart by Treatment')
plt.xlabel('Treatment')
plt.ylabel('Count')
plt.legend(title='Outcome')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

## Effect Size Measures

### Cramér's V

**Formula:**
```math
V = \sqrt{\frac{\chi^2}{N(k-1)}}
```

where $`k = \min(r, c)`$ is the smaller of the number of rows or columns.

**Interpretation:**
- $`V < 0.1`$: Negligible effect
- $`0.1 \leq V < 0.3`$: Small effect
- $`0.3 \leq V < 0.5`$: Medium effect
- $`V \geq 0.5`$: Large effect

### Phi Coefficient (2x2 tables)

**Formula:**
```math
\phi = \sqrt{\frac{\chi^2}{N}}
```

### Contingency Coefficient

**Formula:**
```math
C = \sqrt{\frac{\chi^2}{\chi^2 + N}}
```

**Python Implementation:**
```python
# Function to calculate effect sizes
def calculate_chi_square_effect_sizes(chi_square_result, n):
    # Cramer's V
    df = chi_square_result.dof
    min_dim = min(chi_square_result.expected.shape)
    cramers_v = np.sqrt(chi_square_result.statistic / (n * (min_dim - 1)))
    
    # Phi coefficient (for 2x2 tables)
    if chi_square_result.expected.shape == (2, 2):
        phi = np.sqrt(chi_square_result.statistic / n)
    else:
        phi = np.nan
    
    # Contingency coefficient
    contingency_coef = np.sqrt(chi_square_result.statistic / (chi_square_result.statistic + n))
    
    return {
        'cramers_v': cramers_v,
        'phi': phi,
        'contingency_coef': contingency_coef
    }

# Apply to independence test
effect_sizes = calculate_chi_square_effect_sizes(independence_test, np.sum(contingency_table))

print("Effect Size Analysis:")
print(f"Cramer's V: {effect_sizes['cramers_v']:.3f}")
if not np.isnan(effect_sizes['phi']):
    print(f"Phi coefficient: {effect_sizes['phi']:.3f}")
print(f"Contingency coefficient: {effect_sizes['contingency_coef']:.3f}")

# Interpretation function
def interpret_cramers_v(v):
    if v < 0.1:
        return "Negligible effect"
    elif v < 0.3:
        return "Small effect"
    elif v < 0.5:
        return "Medium effect"
    else:
        return "Large effect"

print(f"Effect size interpretation: {interpret_cramers_v(effect_sizes['cramers_v'])}")
```

## Assumptions and Violations

### Key Assumptions

1. **Independence:** Observations must be independent
2. **Expected Frequencies:** All expected frequencies should be ≥ 5
3. **Random Sampling:** Data should come from random sampling
4. **Mutually Exclusive Categories:** Each observation belongs to exactly one category

### Checking Assumptions

```python
# Function to check chi-square assumptions
def check_chi_square_assumptions(contingency_table):
    print("=== CHI-SQUARE ASSUMPTIONS CHECK ===")
    
    # Calculate expected frequencies
    chi_square_result = stats.chi2_contingency(contingency_table)
    expected_freq = chi_square_result.expected
    
    print("Expected frequencies:")
    print(pd.DataFrame(expected_freq, 
                      index=[f'Row {i+1}' for i in range(expected_freq.shape[0])],
                      columns=[f'Col {j+1}' for j in range(expected_freq.shape[1])]).round(2))
    
    # Check minimum expected frequency
    min_expected = np.min(expected_freq)
    print(f"Minimum expected frequency: {min_expected:.2f}")
    
    # Check for cells with expected frequency < 5
    low_expected_cells = np.where(expected_freq < 5)
    
    if len(low_expected_cells[0]) > 0:
        print("WARNING: Cells with expected frequency < 5:")
        for i, j in zip(low_expected_cells[0], low_expected_cells[1]):
            expected_val = expected_freq[i, j]
            print(f"  Row {i+1} - Col {j+1}: {expected_val:.2f}")
        print("Consider using Fisher's exact test or combining categories")
    else:
        print("All expected frequencies ≥ 5. Chi-square test is appropriate.")
    
    # Check for independence
    print("\nIndependence assumption: Data should be from independent observations.")
    
    return {
        'expected_frequencies': expected_freq,
        'min_expected': min_expected,
        'low_expected_cells': low_expected_cells
    }

# Check assumptions
assumption_results = check_chi_square_assumptions(contingency_table)
```

## Alternative Tests

### Fisher's Exact Test

**When to Use:**
- Small expected frequencies (< 5)
- 2x2 contingency tables
- Exact p-values needed

**Python Implementation:**
```python
# Fisher's exact test for small expected frequencies
fisher_test = stats.fisher_exact(contingency_table)

print("Fisher's Exact Test Results:")
print(f"p-value: {fisher_test.pvalue:.4f}")
print(f"Odds ratio: {fisher_test.statistic:.3f}")

# Compare with chi-square results
print("\nComparison:")
print(f"Chi-square p-value: {independence_test.pvalue:.4f}")
print(f"Fisher's exact p-value: {fisher_test.pvalue:.4f}")

# Visualize 2x2 table for Fisher's test
if contingency_table.shape == (2, 2):
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(contingency_table, 
                            index=['Row 1', 'Row 2'], 
                            columns=['Col 1', 'Col 2']), 
               annot=True, fmt='d', cmap='Reds')
    plt.title('Fisher\'s Exact Test Table')
    plt.show()
```

### Likelihood Ratio Test

**Formula:**
```math
G = 2 \sum_{i=1}^{r} \sum_{j=1}^{c} O_{ij} \ln\left(\frac{O_{ij}}{E_{ij}}\right)
```

**Python Implementation:**
```python
# Likelihood ratio test
def likelihood_ratio_test(observed):
    # Calculate expected frequencies under independence
    row_totals = np.sum(observed, axis=1)
    col_totals = np.sum(observed, axis=0)
    total = np.sum(observed)
    
    expected = np.outer(row_totals, col_totals) / total
    
    # Calculate likelihood ratio statistic
    lr_statistic = 2 * np.sum(observed * np.log(observed / expected))
    
    # Degrees of freedom
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    
    # p-value
    p_value = 1 - stats.chi2.cdf(lr_statistic, df)
    
    return {
        'statistic': lr_statistic,
        'df': df,
        'p_value': p_value
    }

# Apply likelihood ratio test
lr_result = likelihood_ratio_test(contingency_table)

print("Likelihood Ratio Test Results:")
print(f"G-statistic: {lr_result['statistic']:.3f}")
print(f"Degrees of freedom: {lr_result['df']}")
print(f"p-value: {lr_result['p_value']:.4f}")
```

## Post Hoc Analysis

### Pairwise Chi-Square Tests

```python
# Function to perform pairwise chi-square tests
def pairwise_chi_square(contingency_table, alpha=0.05):
    n_rows = contingency_table.shape[0]
    n_cols = contingency_table.shape[1]
    
    # Calculate number of pairwise comparisons
    n_comparisons = int(n_rows * (n_rows - 1) / 2)
    
    # Bonferroni correction
    alpha_corrected = alpha / n_comparisons
    
    results = []
    pair_count = 0
    
    for i in range(n_rows - 1):
        for j in range(i + 1, n_rows):
            # Extract 2x2 subtable
            subtable = contingency_table[[i, j], :]
            
            # Perform chi-square test
            test_result = stats.chi2_contingency(subtable)
            
            results.append({
                'comparison': f"Row {i+1} vs Row {j+1}",
                'chi_square': test_result.statistic,
                'p_value': test_result.pvalue,
                'significant': test_result.pvalue < alpha_corrected
            })
            
            pair_count += 1
    
    return results

# Apply pairwise tests
pairwise_results = pairwise_chi_square(contingency_table)

print(f"Pairwise Chi-Square Tests (Bonferroni-corrected α = {0.05/len(pairwise_results):.3f}):")
for result in pairwise_results:
    print(f"{result['comparison']}:")
    print(f"  Chi-square: {result['chi_square']:.3f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Significant: {result['significant']}\n")
```

## Power Analysis

### Power Analysis for Chi-Square

```python
# Power analysis for chi-square test
def power_analysis_chi_square(n, w, df, alpha=0.05):
    # Calculate power using chi-square distribution
    # This is a simplified approximation
    critical_value = stats.chi2.ppf(1 - alpha, df)
    power = 1 - stats.chi2.cdf(critical_value, df, ncp=n * w**2)
    
    # Estimate required sample size for 80% power
    # This is a rough approximation
    target_power = 0.8
    required_n = int(critical_value / (w**2 * (1 - target_power)))
    
    return {
        'power': power,
        'required_n': required_n,
        'effect_size': w,
        'alpha': alpha
    }

# Apply power analysis
# For 2x2 table, df = 1, w = 0.3 (medium effect)
power_result = power_analysis_chi_square(n=100, w=0.3, df=1)

print("Power Analysis Results:")
print(f"Current power: {power_result['power']:.3f}")
print(f"Required sample size for 80% power: {power_result['required_n']}")
```

## Practical Examples

### Example 1: Survey Analysis

```python
# Simulate survey data
np.random.seed(123)
n_responses = 200

# Generate survey responses
age_group = np.random.choice(['18-25', '26-35', '36-45', '46+'], 
                           n_responses, p=[0.3, 0.35, 0.25, 0.1])
satisfaction = np.random.choice(['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied'], 
                              n_responses, p=[0.4, 0.3, 0.2, 0.1])

# Create contingency table
survey_data = pd.DataFrame({'Age Group': age_group, 'Satisfaction': satisfaction})
survey_table = pd.crosstab(survey_data['Age Group'], survey_data['Satisfaction'])
print("Survey Results:")
print(survey_table)

# Perform chi-square test
survey_test = stats.chi2_contingency(survey_table.values)

# Effect size
survey_effect = calculate_chi_square_effect_sizes(survey_test, n_responses)
print(f"Cramer's V: {survey_effect['cramers_v']:.3f}")
print(f"Interpretation: {interpret_cramers_v(survey_effect['cramers_v'])}")

# Visualize survey results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(survey_table, annot=True, fmt='d', cmap='Blues')
plt.title('Survey Results')

plt.subplot(1, 2, 2)
survey_table.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Satisfaction by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Satisfaction')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

### Example 2: Clinical Trial

```python
# Simulate clinical trial data
np.random.seed(123)

# Create 2x2 contingency table
clinical_data = np.array([[45, 15], [30, 25]])
row_names = ['Treatment', 'Control']
col_names = ['Improved', 'No Improvement']

df_clinical = pd.DataFrame(clinical_data, index=row_names, columns=col_names)
print("Clinical Trial Results:")
print(df_clinical)

# Chi-square test
clinical_test = stats.chi2_contingency(clinical_data)

# Fisher's exact test
clinical_fisher = stats.fisher_exact(clinical_data)

# Effect size
clinical_effect = calculate_chi_square_effect_sizes(clinical_test, np.sum(clinical_data))
print(f"Phi coefficient: {clinical_effect['phi']:.3f}")

# Visualize clinical trial results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(df_clinical, annot=True, fmt='d', cmap='Blues')
plt.title('Clinical Trial Results')

plt.subplot(1, 2, 2)
df_clinical.plot(kind='bar', ax=plt.gca())
plt.title('Improvement by Treatment Group')
plt.xlabel('Group')
plt.ylabel('Count')
plt.legend(title='Outcome')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()
```

### Example 3: Quality Control

```python
# Simulate quality control data
np.random.seed(123)

# Create 3x3 contingency table
quality_data = np.array([
    [85, 10, 5],
    [70, 20, 10],
    [60, 25, 15]
])
row_names = ['Machine A', 'Machine B', 'Machine C']
col_names = ['Excellent', 'Good', 'Poor']

df_quality = pd.DataFrame(quality_data, index=row_names, columns=col_names)
print("Quality Control Results:")
print(df_quality)

# Chi-square test of homogeneity
quality_test = stats.chi2_contingency(quality_data)

# Check assumptions
quality_assumptions = check_chi_square_assumptions(quality_data)

# Effect size
quality_effect = calculate_chi_square_effect_sizes(quality_test, np.sum(quality_data))
print(f"Cramer's V: {quality_effect['cramers_v']:.3f}")

# Visualize quality control results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(df_quality, annot=True, fmt='d', cmap='Blues')
plt.title('Quality Control Results')

plt.subplot(1, 2, 2)
df_quality.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Quality by Machine')
plt.xlabel('Machine')
plt.ylabel('Count')
plt.legend(title='Quality')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

## Best Practices

### Test Selection Guidelines

```python
# Function to help choose appropriate chi-square test
def choose_chi_square_test(contingency_table):
    print("=== CHI-SQUARE TEST SELECTION ===")
    
    # Check expected frequencies
    chi_square_result = stats.chi2_contingency(contingency_table)
    expected_freq = chi_square_result.expected
    min_expected = np.min(expected_freq)
    
    print(f"Minimum expected frequency: {min_expected:.2f}")
    
    # Check table dimensions
    n_rows, n_cols = contingency_table.shape
    print(f"Table dimensions: {n_rows} x {n_cols}")
    
    print("\nRECOMMENDATIONS:")
    
    if min_expected >= 5:
        print("- Use chi-square test of independence")
        print("- All expected frequencies ≥ 5")
    elif min_expected >= 1 and n_rows == 2 and n_cols == 2:
        print("- Use Fisher's exact test")
        print("- Small expected frequencies in 2x2 table")
    else:
        print("- Use Fisher's exact test or combine categories")
        print("- Very small expected frequencies")
    
    # Effect size calculation
    effect_sizes = calculate_chi_square_effect_sizes(chi_square_result, np.sum(contingency_table))
    print(f"- Effect size (Cramer's V): {effect_sizes['cramers_v']:.3f}")
    print(f"- Interpretation: {interpret_cramers_v(effect_sizes['cramers_v'])}")
    
    return {
        'min_expected': min_expected,
        'table_dimensions': (n_rows, n_cols),
        'effect_sizes': effect_sizes
    }

# Apply to contingency table
test_selection = choose_chi_square_test(contingency_table)
```

### Reporting Guidelines

```python
# Function to generate comprehensive chi-square report
def generate_chi_square_report(chi_square_result, contingency_table, test_type="independence"):
    print("=== CHI-SQUARE TEST REPORT ===\n")
    
    print("CONTINGENCY TABLE:")
    print(pd.DataFrame(contingency_table))
    print()
    
    print("TEST RESULTS:")
    print(f"Test type: {test_type}")
    print(f"Chi-square statistic: {chi_square_result.statistic:.3f}")
    print(f"Degrees of freedom: {chi_square_result.dof}")
    print(f"p-value: {chi_square_result.pvalue:.4f}")
    
    # Effect size
    effect_sizes = calculate_chi_square_effect_sizes(chi_square_result, np.sum(contingency_table))
    print(f"Cramer's V: {effect_sizes['cramers_v']:.3f}")
    print(f"Effect size interpretation: {interpret_cramers_v(effect_sizes['cramers_v'])}\n")
    
    # Expected frequencies
    print("EXPECTED FREQUENCIES:")
    print(pd.DataFrame(chi_square_result.expected).round(2))
    print()
    
    # Standardized residuals
    observed = contingency_table
    expected = chi_square_result.expected
    residuals = (observed - expected) / np.sqrt(expected)
    print("STANDARDIZED RESIDUALS:")
    print(pd.DataFrame(residuals).round(3))
    print()
    
    # Conclusion
    alpha = 0.05
    if chi_square_result.pvalue < alpha:
        print("CONCLUSION:")
        print(f"Reject the null hypothesis (p < {alpha})")
        if test_type == "independence":
            print("There is a significant relationship between the variables")
        elif test_type == "homogeneity":
            print("The proportions are significantly different across groups")
        else:
            print("The observed frequencies differ significantly from expected")
    else:
        print("CONCLUSION:")
        print(f"Fail to reject the null hypothesis (p >= {alpha})")
        print("There is insufficient evidence of a relationship")

# Generate report
generate_chi_square_report(independence_test, contingency_table, "independence")
```

## Exercises

### Exercise 1: Goodness of Fit Test
- **Objective:** Test whether observed frequencies match expected frequencies in a categorical variable.
- **Data:** Create a dataset with observed frequencies and test against equal expected frequencies.
- **Hint:** Use `stats.chisquare()` with the `f_exp` parameter for expected frequencies.

### Exercise 2: Independence Test
- **Objective:** Analyze the relationship between two categorical variables using chi-square test of independence.
- **Data:** Create a contingency table and test for independence.
- **Hint:** Use `stats.chi2_contingency()` on a numpy array.

### Exercise 3: Homogeneity Test
- **Objective:** Test whether proportions are the same across different groups.
- **Data:** Create data with multiple groups and test for homogeneity.
- **Hint:** Use `stats.chi2_contingency()` and interpret as homogeneity test.

### Exercise 4: Assumption Checking
- **Objective:** Check chi-square assumptions and recommend appropriate alternatives when violated.
- **Data:** Create data with small expected frequencies.
- **Hint:** Use `check_chi_square_assumptions()` and `stats.fisher_exact()`.

### Exercise 5: Effect Size Analysis
- **Objective:** Calculate and interpret different effect size measures for chi-square tests.
- **Data:** Use any contingency table from previous exercises.
- **Hint:** Use `calculate_chi_square_effect_sizes()` function.

### Exercise 6: Post Hoc Analysis
- **Objective:** Perform pairwise chi-square tests with multiple comparisons correction.
- **Data:** Use a 3x3 or larger contingency table.
- **Hint:** Use `pairwise_chi_square()` function.

### Exercise 7: Power Analysis
- **Objective:** Conduct power analysis for a chi-square test.
- **Data:** Determine required sample size for desired power.
- **Hint:** Use `power_analysis_chi_square()` function.

### Exercise 8: Comprehensive Analysis
- **Objective:** Perform a complete chi-square analysis including assumption checking, test selection, and reporting.
- **Data:** Create a realistic dataset (e.g., survey responses, clinical trial results).
- **Hint:** Use all the functions developed in this chapter.

## Next Steps

In the next chapter, we'll learn about time series analysis for analyzing temporal data.

---

**Key Takeaways:**
- Chi-square tests are essential for categorical data analysis
- Three main types: goodness of fit, independence, and homogeneity
- Always check expected frequency requirements before using chi-square tests
- Effect sizes (Cramer's V, Phi) provide important information about practical significance
- Fisher's exact test is preferred for small expected frequencies
- Post hoc analyses help identify specific differences when overall test is significant
- Power analysis helps determine appropriate sample sizes
- Proper reporting includes contingency tables, test results, effect sizes, and conclusions
- Assumption violations require alternative approaches or data modifications 