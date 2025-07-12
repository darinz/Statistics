# Repeated Measures ANOVA

## Overview

Repeated Measures Analysis of Variance (ANOVA) is a powerful statistical technique used to analyze data where the same subjects are measured multiple times under different conditions. This within-subjects design is more powerful than between-subjects designs because it controls for individual differences and reduces error variance, leading to increased statistical power and more precise effect estimates.

### Key Concepts

**Within-Subjects Design**: A research design where each participant is measured under all experimental conditions, allowing for direct comparison of conditions within the same individuals.

**Individual Differences**: The natural variation between participants that is controlled for in repeated measures designs, reducing error variance.

**Sphericity**: The assumption that the variances of the differences between all pairs of conditions are equal, which is crucial for the validity of repeated measures ANOVA.

**Compound Symmetry**: A stronger assumption than sphericity, requiring equal variances and equal covariances between all conditions.

### Mathematical Foundation

The repeated measures ANOVA model can be expressed as:

```math
Y_{ij} = \mu + \alpha_i + \pi_j + \epsilon_{ij}
```

Where:
- $`Y_{ij}`$ is the observed value for the jth subject in the ith condition
- $`\mu`$ is the overall population mean
- $`\alpha_i`$ is the effect of the ith condition (treatment effect)
- $`\pi_j`$ is the effect of the jth subject (individual differences)
- $`\epsilon_{ij}`$ is the random error term

### Sum of Squares Decomposition

The total variability in repeated measures data is partitioned into three components:

```math
SS_{Total} = SS_{Between Subjects} + SS_{Within Subjects}
```

Where:
- $`SS_{Total} = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{..})^2`$
- $`SS_{Between Subjects} = k\sum_{j=1}^{n}(\bar{Y}_{.j} - \bar{Y}_{..})^2`$ (individual differences)
- $`SS_{Within Subjects} = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{.j})^2`$ (within-subject variation)

The within-subjects sum of squares is further partitioned into:

```math
SS_{Within Subjects} = SS_{Conditions} + SS_{Error}
```

Where:
- $`SS_{Conditions} = n\sum_{i=1}^{k}(\bar{Y}_{i.} - \bar{Y}_{..})^2`$ (treatment effect)
- $`SS_{Error} = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{i.} - \bar{Y}_{.j} + \bar{Y}_{..})^2`$ (residual error)

### Degrees of Freedom

```math
df_{Total} = nk - 1
df_{Between Subjects} = n - 1
df_{Within Subjects} = n(k - 1)
df_{Conditions} = k - 1
df_{Error} = (n - 1)(k - 1)
```

Where $`n`$ is the number of subjects and $`k`$ is the number of conditions.

### F-Statistic

```math
F = \frac{MS_{Conditions}}{MS_{Error}} = \frac{SS_{Conditions}/df_{Conditions}}{SS_{Error}/df_{Error}}
```

### Effect Size Measures

**Partial Eta-Squared**:
```math
\eta_p^2 = \frac{SS_{Conditions}}{SS_{Conditions} + SS_{Error}}
```

**Eta-Squared**:
```math
\eta^2 = \frac{SS_{Conditions}}{SS_{Total}}
```

**Omega-Squared** (unbiased estimate):
```math
\omega^2 = \frac{SS_{Conditions} - df_{Conditions} \times MS_{Error}}{SS_{Total} + MS_{Error}}
```

### Sphericity Assumption

The sphericity assumption requires that the variance of the differences between any two conditions is constant:

```math
Var(Y_{i} - Y_{j}) = \sigma^2 \text{ for all } i \neq j
```

This can be tested using Mauchly's test of sphericity, which examines the correlation matrix of the conditions.

## Basic Repeated Measures ANOVA

### Manual Calculation

The manual calculation helps understand the underlying mathematics and provides insight into how the statistical software performs the analysis.

**Code Reference**: See `simulate_repeated_measures_data()` and `manual_repeated_anova()` functions in `16_repeated_measures_anova.py`.

The manual calculation involves:
1. **Data Simulation**: Generate repeated measures data with individual differences and condition effects
2. **Sum of Squares Calculation**: Compute SS Total, SS Between Subjects, SS Within Subjects, SS Conditions, and SS Error
3. **Degrees of Freedom**: Calculate df for conditions, error, and total
4. **F-statistic**: Compute F = MS Conditions / MS Error
5. **Effect Size**: Calculate partial eta-squared
6. **Sphericity Measures**: Estimate Greenhouse-Geisser and Huynh-Feldt epsilon values

The manual calculation provides the foundation for understanding how repeated measures ANOVA works and validates the results from statistical software.

### Using Python's Built-in Repeated Measures ANOVA

Python's `statsmodels` provides a convenient way to perform repeated measures ANOVA.

**Code Reference**: See `builtin_repeated_anova()` function in `16_repeated_measures_anova.py`.

The built-in function uses `AnovaRM` from statsmodels to perform the analysis automatically, providing:
- F-statistic and p-value
- Degrees of freedom for conditions and error
- ANOVA table with all relevant statistics
- Automatic handling of the repeated measures design

This approach is more convenient for routine analyses while providing the same statistical rigor as manual calculations.

## Descriptive Statistics

### Understanding Repeated Measures Descriptive Statistics

In repeated measures designs, we need to understand several types of statistics:

1. **Condition Means**: The average performance across all subjects for each condition
2. **Subject Means**: The average performance across all conditions for each subject
3. **Individual Differences**: The variation between subjects
4. **Within-Subject Variation**: The variation within each subject across conditions

### Mathematical Definitions

**Condition Mean**: $`\bar{Y}_{i.} = \frac{1}{n}\sum_{j=1}^{n}Y_{ij}`$

**Subject Mean**: $`\bar{Y}_{.j} = \frac{1}{k}\sum_{i=1}^{k}Y_{ij}`$

**Grand Mean**: $`\bar{Y}_{..} = \frac{1}{nk}\sum_{i=1}^{k}\sum_{j=1}^{n}Y_{ij}`$

**Individual Differences**: $`SS_{Between Subjects} = k\sum_{j=1}^{n}(\bar{Y}_{.j} - \bar{Y}_{..})^2`$

**Within-Subject Variation**: $`SS_{Within Subjects} = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{.j})^2`$

### Condition and Subject Statistics

**Code Reference**: See `calculate_descriptive_statistics()` function in `16_repeated_measures_anova.py`.

The descriptive statistics function calculates:
1. **Condition Statistics**: Mean, standard deviation, standard error, and confidence intervals for each condition
2. **Subject Statistics**: Mean, standard deviation, range, and other summary statistics for each subject
3. **Individual Differences**: How each subject's mean performance differs from the grand mean
4. **Condition Differences**: How each condition's mean differs from the grand mean

This comprehensive analysis helps understand both the treatment effects (condition differences) and individual variability (subject differences) in the repeated measures design.

### Understanding Individual Differences

**Code Reference**: See `analyze_individual_differences()` function in `16_repeated_measures_anova.py`.

The individual differences analysis provides:
1. **Performance Level Categorization**: Subjects are classified as Low, Average, or High performers based on their deviation from the grand mean
2. **Within-Subject Variability**: Measures how much each subject's performance varies across conditions
3. **Individual Difference Metrics**: Standard deviation and range of individual differences across subjects
4. **Performance Distribution**: Summary of how many subjects fall into each performance category

This analysis is crucial for understanding the extent of individual differences and how they relate to treatment effects in repeated measures designs.

### Visualization

Visualization is crucial for understanding repeated measures data, as it helps identify patterns, individual differences, and potential violations of assumptions.

#### Individual Subject Profiles

The individual subject profiles plot is the most important visualization for repeated measures data as it shows how each subject changes across conditions.

**Code Reference**: See `plot_individual_profiles()` function in `16_repeated_measures_anova.py`.

This visualization shows:
- **Individual Subject Lines**: Each subject's performance trajectory across conditions
- **Group Mean Line**: The average performance across all subjects
- **Confidence Intervals**: Standard error bars around the group mean
- **Pattern Interpretation**: 
  - Parallel lines indicate consistent individual differences
  - Non-parallel lines indicate individual differences in treatment effects
  - Steep slopes indicate strong treatment effects
  - Flat lines indicate no treatment effect

#### Enhanced Distribution Plots

**Code Reference**: See `plot_distribution()` function in `16_repeated_measures_anova.py`.

This function creates two complementary distribution plots:
1. **Box Plot with Individual Points**: Shows the distribution, median, quartiles, and individual data points for each condition
2. **Violin Plot with Box Plot Overlay**: Shows the density distribution and summary statistics for each condition

Both plots include the mean values as red dots, making it easy to compare central tendency across conditions while seeing the full distribution of the data.

#### Individual Differences Visualization

**Code Reference**: See `plot_individual_differences()` function in `16_repeated_measures_anova.py`.

This function creates two bar plots to visualize individual differences:
1. **Individual Differences from Grand Mean**: Shows how each subject's average performance differs from the overall mean, with reference lines at ±1 standard deviation
2. **Within-Subject Variability**: Shows the standard deviation of each subject's performance across conditions, with a reference line at the mean variability

These plots help identify subjects who are consistently high or low performers and those who show more or less variability across conditions.

#### Correlation Matrix Visualization

**Code Reference**: See `plot_correlation_matrix()` function in `16_repeated_measures_anova.py`.

The correlation matrix heatmap shows the correlations between conditions, which is important for:
- **Sphericity Assessment**: High correlations suggest sphericity assumption may be met
- **Pattern Recognition**: Identifying which conditions are most similar
- **Assumption Checking**: Understanding the covariance structure of the repeated measures

The correlation matrix is a key diagnostic tool for repeated measures ANOVA, as it helps assess whether the sphericity assumption is reasonable.

#### Trend Analysis Visualization

**Code Reference**: See `trend_analysis()` function in `16_repeated_measures_anova.py`.

The trend analysis plot shows:
- **Individual Subject Lines**: Each subject's performance trajectory across conditions
- **Group Mean Line**: Average performance across all subjects
- **Confidence Intervals**: Standard error bars around the group mean
- **Trend Line**: Linear regression line showing the overall trend across conditions

This visualization helps identify whether there's a systematic linear trend in the data, which can be tested statistically using polynomial contrasts.

## Effect Size Analysis

Effect sizes are crucial for understanding the practical significance of repeated measures ANOVA results, as they provide information about the magnitude of effects independent of sample size.

### Understanding Effect Size Measures in Repeated Measures

#### Partial Eta-Squared ($`\eta_p^2`$)

Partial eta-squared is the most commonly used effect size measure in repeated measures ANOVA. It represents the proportion of variance in the dependent variable that is explained by the condition effect, controlling for individual differences.

```math
\eta_p^2 = \frac{SS_{Conditions}}{SS_{Conditions} + SS_{Error}}
```

**Advantages:**
- Controls for individual differences
- Ranges from 0 to 1
- Easy to interpret
- Most commonly reported in literature

**Disadvantages:**
- Can be biased upward in small samples
- Values can be larger than in between-subjects designs

#### Eta-Squared ($`\eta^2`$)

Eta-squared represents the proportion of total variance explained by the condition effect.

```math
\eta^2 = \frac{SS_{Conditions}}{SS_{Total}}
```

**Advantages:**
- Intuitive interpretation
- Values sum to 1 across all effects

**Disadvantages:**
- Does not control for individual differences
- Can be misleading in repeated measures designs

#### Omega-Squared ($`\omega^2`$)

Omega-squared is an unbiased estimate of the population effect size.

```math
\omega^2 = \frac{SS_{Conditions} - df_{Conditions} \times MS_{Error}}{SS_{Total} + MS_{Error}}
```

**Advantages:**
- Unbiased estimate
- More conservative than eta-squared

**Disadvantages:**
- Can be negative for small effects
- Less commonly reported

### Comprehensive Effect Size Calculation

**Code Reference**: See `calculate_repeated_effect_sizes()` function in `16_repeated_measures_anova.py`.

The comprehensive effect size calculation provides multiple measures of effect size:

1. **Partial η²**: Most commonly used effect size for repeated measures ANOVA, controlling for individual differences
2. **η²**: Total variance explained by the condition effect
3. **ω²**: Unbiased estimate of population effect size
4. **Cohen's f**: Effect size for power analysis
5. **Individual Differences Effect Sizes**: How much variance is explained by individual differences
6. **Condition-Specific Effects**: Cohen's d for each condition compared to the grand mean
7. **Bootstrap Confidence Intervals**: 95% CI for partial η² using resampling

This comprehensive approach provides multiple perspectives on the magnitude and practical significance of the treatment effect.

### Effect Size Interpretation

**Code Reference**: See `interpret_effect_size()` function in `16_repeated_measures_anova.py`.

The effect size interpretation function provides:
1. **Multiple Effect Size Measures**: Interpretation guidelines for partial η², η², Cohen's f, and Cohen's d
2. **Standardized Categories**: Small, medium, and large effect size classifications
3. **Practical Significance**: Assessment of whether effects are practically meaningful
4. **Comprehensive Reporting**: Clear interpretation of both condition effects and individual differences

This function helps researchers understand not just statistical significance but also the practical importance of their findings.

### Effect Size Visualization

**Code Reference**: See `plot_effect_sizes()` function in `16_repeated_measures_anova.py`.

The effect size visualization function creates two complementary plots:
1. **Effect Size Comparison**: Bar plot comparing different effect size measures (partial η², η², ω²) for condition effects and individual differences
2. **Condition-Specific Effects**: Bar plot showing Cohen's d for each condition compared to the grand mean

Both plots include reference lines for effect size interpretation (small, medium, large) to help researchers quickly assess the practical significance of their findings.

## Post Hoc Tests

### Pairwise Comparisons

**Code Reference**: See `post_hoc_pairwise_tests()` function in `16_repeated_measures_anova.py`.

The post hoc analysis provides two approaches for pairwise comparisons:

1. **Tukey's HSD Test**: Controls family-wise error rate for all pairwise comparisons
2. **Bonferroni-Corrected Paired t-tests**: More conservative approach that divides alpha by the number of comparisons

The function returns:
- **Estimated Marginal Means**: Mean scores for each condition
- **Tukey Results**: Complete Tukey HSD test results
- **Significant Pairs**: Identification of which condition pairs differ significantly
- **Bonferroni Results**: Paired t-test results with Bonferroni correction

This comprehensive approach ensures that multiple comparisons are handled appropriately while maintaining statistical rigor.

### Trend Analysis

**Code Reference**: See `trend_analysis()` function in `16_repeated_measures_anova.py`.

The trend analysis examines whether there are systematic patterns across conditions:

1. **Linear Trend**: Tests whether there's a systematic increase or decrease across conditions
2. **Quadratic Trend**: Tests whether there's a U-shaped or inverted U-shaped pattern
3. **Polynomial Contrasts**: Uses predefined contrast coefficients to test specific trends
4. **F-statistics**: Calculates F-values and p-values for each trend component

This analysis is particularly useful when conditions have a natural order (e.g., time points, dosage levels) and you want to understand the nature of the relationship between conditions and the dependent variable.

## Assumption Checking

Repeated measures ANOVA relies on several key assumptions that must be verified before interpreting results. Violations of these assumptions can lead to incorrect conclusions and inflated Type I error rates.

### Key Assumptions

1. **Sphericity**: The variances of the differences between all pairs of conditions are equal
2. **Normality**: Residuals should be normally distributed
3. **Independence**: Observations should be independent (within-subjects correlation is expected)
4. **Linearity**: The relationship between conditions and the dependent variable should be linear

### Comprehensive Sphericity Testing

The sphericity assumption is the most critical assumption for repeated measures ANOVA. It requires that the variance of the differences between any two conditions is constant.

**Code Reference**: See `check_sphericity()` function in `16_repeated_measures_anova.py`.

The sphericity testing function provides:
1. **Correlation Matrix Analysis**: Examines correlations between conditions
2. **Variance Ratio Assessment**: Compares variances across conditions
3. **Difference Variance Analysis**: Tests equality of variances of condition differences
4. **Epsilon Calculations**: Greenhouse-Geisser, Huynh-Feldt, and Lower-bound epsilon values
5. **Sphericity Indicators**: Multiple diagnostic criteria for sphericity assessment
6. **Recommendations**: Clear guidance on whether to use corrections or alternatives

This comprehensive approach helps determine whether the sphericity assumption is met and provides appropriate recommendations for analysis.

### Comprehensive Normality Testing

**Code Reference**: See `check_normality_repeated()` function in `16_repeated_measures_anova.py`.

The normality testing function provides:
1. **Multiple Normality Tests**: Shapiro-Wilk, Kolmogorov-Smirnov, and Anderson-Darling tests on residuals
2. **Residual Analysis**: Tests normality of residuals (deviations from condition means)
3. **Condition-Specific Testing**: Tests normality within each condition
4. **Diagnostic Plots**: Q-Q plots, histograms with normal curves, and residuals vs fitted plots
5. **Comprehensive Assessment**: Summary of normality test results with clear recommendations
6. **Alternative Suggestions**: Guidance on nonparametric alternatives when normality is violated

This approach ensures that the normality assumption is thoroughly evaluated before proceeding with parametric repeated measures ANOVA.

## Mixed ANOVA

### Between-Subjects and Within-Subjects Factors

**Code Reference**: See `mixed_anova_example()` function in `16_repeated_measures_anova.py`.

Mixed ANOVA designs combine both between-subjects and within-subjects factors:
- **Between-Subjects Factor**: Different groups of participants (e.g., treatment vs. control)
- **Within-Subjects Factor**: Same participants measured across conditions (e.g., time points)
- **Interaction Effects**: Tests whether the effect of the within-subjects factor differs between groups
- **Main Effects**: Tests for overall differences between groups and across time points

This design is particularly useful in clinical trials and intervention studies where you want to compare different treatments over time.

## Nonparametric Alternatives

### Friedman Test

**Code Reference**: See `friedman_test()` function in `16_repeated_measures_anova.py`.

The Friedman test is a nonparametric alternative to repeated measures ANOVA that:
- Tests for differences between conditions without assuming normality
- Uses rank-based analysis instead of mean-based analysis
- Is robust to violations of normality and sphericity assumptions
- Provides chi-squared statistic and p-value for overall condition differences

This test is particularly useful when the data are ordinal or when the assumptions of parametric repeated measures ANOVA are violated.

### Wilcoxon Signed-Rank Test for Pairwise Comparisons

**Code Reference**: See `wilcoxon_pairwise_tests()` function in `16_repeated_measures_anova.py`.

The Wilcoxon signed-rank test provides nonparametric pairwise comparisons that:
- Test for differences between each pair of conditions
- Use rank-based analysis instead of t-tests
- Are robust to violations of normality assumptions
- Can be corrected for multiple comparisons using Bonferroni or other methods
- Provide V-statistic and p-value for each comparison

This approach is particularly useful when the data are not normally distributed or when you want to be conservative about the assumptions.

## Power Analysis

### Power Analysis for Repeated Measures ANOVA

**Code Reference**: See `power_analysis()` function in `16_repeated_measures_anova.py`.

The power analysis function provides:
1. **Current Power**: Calculates the statistical power of the current analysis
2. **Required Sample Size**: Determines the sample size needed for 80% power
3. **Effect Size Integration**: Uses Cohen's f from the effect size analysis
4. **Multiple Conditions**: Handles power analysis for designs with multiple conditions

This analysis helps researchers understand whether their study has adequate power to detect meaningful effects and plan future studies with appropriate sample sizes.

## Practical Examples

### Example 1: Clinical Trial

**Code Reference**: See `clinical_trial_example()` function in `16_repeated_measures_anova.py`.

This example demonstrates repeated measures ANOVA in a clinical trial context:
- **Data Structure**: 25 patients measured at 4 time points (Baseline, Week 4, Week 8, Week 12)
- **Research Question**: Do symptom scores change significantly over time?
- **Analysis**: Repeated measures ANOVA to test for time effects
- **Visualization**: Individual patient trajectories with group means and standard errors

This example shows how repeated measures ANOVA can be used to analyze treatment effectiveness over time in clinical research.

### Example 2: Learning Study

**Code Reference**: See `learning_study_example()` function in `16_repeated_measures_anova.py`.

This example demonstrates repeated measures ANOVA in an educational context:
- **Data Structure**: 30 students measured across 3 learning sessions
- **Research Question**: Does student performance improve across learning sessions?
- **Analysis**: Repeated measures ANOVA to test for session effects
- **Visualization**: Individual student learning curves with group means

This example shows how repeated measures ANOVA can be used to analyze learning progression and educational interventions.

### Example 3: Exercise Study

**Code Reference**: See `exercise_study_example()` function in `16_repeated_measures_anova.py`.

This example demonstrates mixed ANOVA in a sports science context:
- **Data Structure**: 20 participants per exercise type (Aerobic vs. Strength) measured at 3 time points
- **Research Question**: Do different exercise types lead to different fitness improvements over time?
- **Analysis**: Mixed ANOVA to test for exercise type effects, time effects, and their interaction
- **Visualization**: Interaction plot showing fitness scores by exercise type and time

This example shows how mixed ANOVA can be used to analyze the effectiveness of different interventions over time.

## Best Practices

### Test Selection Guidelines

**Code Reference**: See `choose_repeated_test()` function in `16_repeated_measures_anova.py`.

The test selection function provides a systematic approach to choosing the appropriate analysis method:

1. **Assumption Checking**: Automatically tests normality and sphericity assumptions
2. **Sample Size Assessment**: Evaluates whether the sample size is adequate
3. **Decision Logic**: Provides clear recommendations based on assumption test results
4. **Multiple Options**: Suggests standard ANOVA, corrected ANOVA, or nonparametric alternatives

This function helps researchers make informed decisions about which statistical test to use based on their data characteristics and assumption violations.

### Reporting Guidelines

**Code Reference**: See `generate_repeated_report()` function in `16_repeated_measures_anova.py`.

The reporting function provides a comprehensive template for reporting repeated measures ANOVA results:

1. **Descriptive Statistics**: Summary statistics for each condition
2. **ANOVA Results**: F-statistic, degrees of freedom, and p-value
3. **Effect Size**: Partial eta-squared with interpretation
4. **Post Hoc Results**: Placeholder for multiple comparison results
5. **Conclusion**: Clear statement of statistical decision and practical interpretation

This function helps ensure that all essential information is included in research reports and follows standard reporting conventions for repeated measures ANOVA.

## Comprehensive Exercises

### Exercise 1: Basic Repeated Measures ANOVA Analysis

**Objective**: Perform a complete repeated measures ANOVA analysis from data preparation to interpretation.

**Scenario**: A researcher is studying the effectiveness of a new learning intervention. 15 students are tested on their problem-solving skills at three time points: before the intervention (baseline), immediately after (post-test), and 3 months later (follow-up).

**Tasks**:
1. Generate realistic data for this scenario
2. Perform descriptive statistics and visualization
3. Conduct repeated measures ANOVA with manual calculations
4. Check assumptions thoroughly
5. Calculate and interpret effect sizes
6. Perform post hoc analysis if needed
7. Write a comprehensive report

**Requirements**:
- Use 15 subjects and 3 time points
- Include individual differences and learning effects
- Generate data with realistic effect sizes
- Provide detailed interpretation of results

**Expected Learning Outcomes**:
- Understanding of repeated measures ANOVA workflow
- Ability to interpret individual differences
- Knowledge of assumption checking procedures
- Skills in effect size calculation and interpretation

**Solution Framework**:
```python
# Exercise 1 Solution Framework
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM

np.random.seed(123)
n_subjects = 15
n_timepoints = 3

# Generate data with learning effects
# ... (students will implement this)

# Perform comprehensive analysis
# ... (students will implement this)

# Check assumptions
# ... (students will implement this)

# Calculate effect sizes
# ... (students will implement this)

# Write report
# ... (students will implement this)
```

### Exercise 2: Mixed ANOVA with Interaction Effects

**Objective**: Analyze a mixed ANOVA design with both between-subjects and within-subjects factors.

**Scenario**: A clinical trial comparing two treatments (A and B) for anxiety reduction. Patients are randomly assigned to treatments and measured at baseline, week 4, and week 8.

**Tasks**:
1. Create a 2×3 mixed design dataset
2. Analyze main effects and interaction effects
3. Perform simple effects analysis
4. Create interaction plots
5. Conduct post hoc tests for significant effects
6. Report results comprehensively

**Requirements**:
- 20 patients per treatment group
- 3 measurement time points
- Include treatment × time interaction
- Realistic anxiety scores (0-100 scale)

**Expected Learning Outcomes**:
- Understanding of mixed ANOVA designs
- Ability to interpret interaction effects
- Skills in simple effects analysis
- Knowledge of appropriate post hoc procedures

### Exercise 3: Advanced Assumption Checking and Remedies

**Objective**: Practice comprehensive assumption checking and implement appropriate remedies for violations.

**Scenario**: You have repeated measures data that may violate sphericity and normality assumptions.

**Tasks**:
1. Generate data with known assumption violations
2. Perform comprehensive assumption checking
3. Apply appropriate corrections (Greenhouse-Geisser, Huynh-Feldt)
4. Compare results with and without corrections
5. Implement nonparametric alternatives
6. Compare parametric and nonparametric results

**Requirements**:
- Create data with sphericity violations
- Create data with normality violations
- Apply multiple correction methods
- Use Friedman test and Wilcoxon tests
- Provide recommendations for each scenario

**Expected Learning Outcomes**:
- Understanding of assumption violations
- Knowledge of correction methods
- Skills in nonparametric alternatives
- Ability to make informed decisions about analysis methods

### Exercise 4: Effect Size Analysis and Power

**Objective**: Conduct comprehensive effect size analysis and power calculations.

**Scenario**: You need to design a study to detect different effect sizes with adequate power.

**Tasks**:
1. Calculate effect sizes for existing data
2. Perform power analysis for different scenarios
3. Determine required sample sizes
4. Analyze effect size confidence intervals
5. Compare different effect size measures
6. Make recommendations for study design

**Requirements**:
- Use multiple effect size measures
- Calculate bootstrap confidence intervals
- Perform power analysis for different effect sizes
- Consider practical significance
- Provide sample size recommendations

**Expected Learning Outcomes**:
- Understanding of effect size interpretation
- Knowledge of power analysis procedures
- Skills in sample size determination
- Ability to assess practical significance

### Exercise 5: Real-World Application

**Objective**: Apply repeated measures ANOVA to a real-world research scenario.

**Scenario**: Choose one of the following:
- **Educational Research**: Student performance across multiple exams
- **Clinical Research**: Patient outcomes over treatment period
- **Sports Science**: Athlete performance across training phases
- **Psychology**: Cognitive task performance under different conditions

**Tasks**:
1. Design a realistic research study
2. Generate appropriate data
3. Perform complete analysis workflow
4. Create publication-ready visualizations
5. Write a research report
6. Address potential limitations

**Requirements**:
- Realistic research design
- Appropriate sample sizes
- Meaningful dependent variables
- Comprehensive analysis
- Professional reporting

**Expected Learning Outcomes**:
- Application of repeated measures ANOVA to real problems
- Skills in research design
- Ability to communicate results effectively
- Understanding of research limitations

### Exercise 6: Advanced Topics

**Objective**: Explore advanced topics in repeated measures analysis.

**Tasks**:
1. **Unbalanced Designs**: Handle missing data and unbalanced designs
2. **Robust Methods**: Implement robust repeated measures ANOVA
3. **Multivariate Approaches**: Use MANOVA for multiple dependent variables
4. **Trend Analysis**: Perform polynomial contrasts and trend analysis
5. **Bootstrap Methods**: Use bootstrap for confidence intervals and hypothesis testing

**Requirements**:
- Implement multiple advanced techniques
- Compare results across methods
- Understand when to use each approach
- Provide practical recommendations

**Expected Learning Outcomes**:
- Knowledge of advanced repeated measures techniques
- Understanding of robust methods
- Skills in multivariate analysis
- Ability to choose appropriate methods for different situations

## Best Practices and Guidelines

### Test Selection Guidelines

**Code Reference**: See `choose_repeated_test()` function in `16_repeated_measures_anova.py`.

The test selection function provides a comprehensive decision-making framework for choosing the appropriate repeated measures analysis method based on:

1. **Sample Size Assessment**: Evaluates whether the sample size is adequate for parametric tests
2. **Assumption Testing**: Automatically checks normality and sphericity assumptions
3. **Decision Matrix**: Provides clear recommendations based on assumption test results
4. **Additional Considerations**: Offers guidance for special cases (2 conditions, many conditions, large samples)

This function helps researchers make informed decisions about which statistical test to use based on their data characteristics.

### Data Preparation Best Practices

**Code Reference**: See `prepare_repeated_data()` function in `16_repeated_measures_anova.py`.

The data preparation function provides a comprehensive checklist for preparing repeated measures data:

1. **Missing Data Assessment**: Identifies and quantifies missing data patterns
2. **Data Structure Check**: Verifies balanced vs. unbalanced designs
3. **Outlier Detection**: Identifies potential outliers that may affect analysis
4. **Data Quality Recommendations**: Provides specific guidance for data quality issues
5. **Cleaned Dataset**: Returns a dataset ready for analysis

This function ensures that data quality issues are identified and addressed before conducting repeated measures ANOVA.

### Reporting Guidelines

**Code Reference**: See `generate_comprehensive_report()` function in `16_repeated_measures_anova.py`.

The comprehensive reporting function provides a complete template for reporting repeated measures ANOVA results:

1. **Study Description**: Clear description of the research design and sample
2. **Descriptive Statistics**: Summary statistics with confidence intervals for each condition
3. **Assumption Checking Summary**: Summary of normality and sphericity test results
4. **ANOVA Results**: F-statistic, degrees of freedom, and p-value
5. **Effect Sizes**: Multiple effect size measures with interpretations
6. **Interpretation**: Clear statements about statistical and practical significance
7. **Recommendations**: Guidance on result reliability and future analyses

This function ensures that all essential information is included in research reports and follows standard reporting conventions for repeated measures ANOVA.

## Next Steps

In the next chapter, we'll learn about correlation analysis for examining relationships between variables.

---

**Key Takeaways:**
- Repeated measures ANOVA is more powerful than between-subjects designs
- Always check sphericity and normality assumptions
- Effect sizes provide important information about practical significance
- Post hoc tests are necessary when the main effect is significant
- Nonparametric alternatives exist for non-normal data
- Mixed ANOVA combines between-subjects and within-subjects factors
- Proper reporting includes descriptive statistics, test results, and effect sizes 