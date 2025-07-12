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
See `chi_square_goodness_of_fit()` in `22_chi_square_tests.py` for code and visualization.

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
See `chi_square_independence()` in `22_chi_square_tests.py` for code and visualization.

### 3. Chi-Square Test of Homogeneity

**Purpose:** Test whether the distribution of a categorical variable is the same across different groups.

**Null Hypothesis:** $`H_0: \text{Proportions are equal across groups}`$
**Alternative Hypothesis:** $`H_1: \text{Proportions differ across groups}`$

**Test Statistic:** Same as independence test, but interpretation differs.

**Python Implementation:**
See `chi_square_homogeneity()` in `22_chi_square_tests.py` for code and visualization.

## Effect Size Measures

- **Cramér's V**: See `calculate_chi_square_effect_sizes()` in `22_chi_square_tests.py`.
- **Phi Coefficient**: See `calculate_chi_square_effect_sizes()` in `22_chi_square_tests.py`.
- **Contingency Coefficient**: See `calculate_chi_square_effect_sizes()` in `22_chi_square_tests.py`.
- **Interpretation**: See `interpret_cramers_v()` in `22_chi_square_tests.py`.

## Assumptions and Violations

- **Assumption Checking**: See `check_chi_square_assumptions()` in `22_chi_square_tests.py` for code and output.

## Alternative Tests

- **Fisher's Exact Test**: See `fishers_exact()` in `22_chi_square_tests.py` for code and visualization.
- **Likelihood Ratio Test (G-test)**: See `likelihood_ratio_test()` in `22_chi_square_tests.py`.

## Post Hoc Analysis

- **Pairwise Chi-Square Tests**: See `pairwise_chi_square()` in `22_chi_square_tests.py` for code and output.

## Power Analysis

- **Power Analysis for Chi-Square**: See `power_analysis_chi_square()` in `22_chi_square_tests.py` for code and output.

## Practical Examples

- **Survey Analysis**: See `survey_analysis_example()` in `22_chi_square_tests.py` for a full demonstration.
- **Clinical Trial**: See `clinical_trial_example()` in `22_chi_square_tests.py` for a full demonstration.
- **Quality Control**: See `quality_control_example()` in `22_chi_square_tests.py` for a full demonstration.

## Best Practices

- **Test Selection Guidelines**: See `check_chi_square_assumptions()` and `fishers_exact()` for guidance on which test to use.
- **Reporting Guidelines**: See `generate_chi_square_report()` in `22_chi_square_tests.py` for a comprehensive reporting template.

## Python Implementation Reference

**Function Mapping:**
- Goodness of Fit: `chi_square_goodness_of_fit()`
- Independence: `chi_square_independence()`
- Homogeneity: `chi_square_homogeneity()`
- Effect Sizes: `calculate_chi_square_effect_sizes()`, `interpret_cramers_v()`
- Assumption Checking: `check_chi_square_assumptions()`
- Fisher's Exact: `fishers_exact()`
- Likelihood Ratio: `likelihood_ratio_test()`
- Post Hoc: `pairwise_chi_square()`
- Power: `power_analysis_chi_square()`
- Reporting: `generate_chi_square_report()`
- Practical Examples: `survey_analysis_example()`, `clinical_trial_example()`, `quality_control_example()`

**Usage:**
Import the functions from `22_chi_square_tests.py` and use them as demonstrated in the file's main block. Each function prints results and produces relevant plots for interpretation.

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