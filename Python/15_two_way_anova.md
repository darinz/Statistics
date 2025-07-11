# Two-Way ANOVA

> **Note:** All Python code for this lesson has been moved to [15_two_way_anova.py](15_two_way_anova.py). Code blocks in this markdown are now replaced with references to the relevant functions and sections in the Python file. Use both files together for a complete learning experience.

## Overview

Two-way Analysis of Variance (ANOVA) is a powerful statistical technique used to analyze the effects of two independent variables (factors) on a dependent variable, including their interaction effects. This design is more sophisticated than one-way ANOVA as it allows researchers to examine both main effects and interaction effects simultaneously, providing a more complete understanding of the relationships between variables.

### Key Concepts

**Main Effects**: The individual effect of each factor on the dependent variable, ignoring the other factor.

**Interaction Effects**: The combined effect of both factors that cannot be explained by their individual main effects alone. An interaction occurs when the effect of one factor depends on the level of the other factor.

**Factorial Design**: A research design where all combinations of factor levels are included in the study.

### Mathematical Foundation

The two-way ANOVA model can be expressed as:

```math
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}
```

Where:
- $`Y_{ijk}`$ is the observed value for the kth observation in the ith level of factor A and jth level of factor B
- $`\mu`$ is the overall population mean
- $`\alpha_i`$ is the effect of the ith level of factor A (main effect of factor A)
- $`\beta_j`$ is the effect of the jth level of factor B (main effect of factor B)
- $`(\alpha\beta)_{ij}`$ is the interaction effect between the ith level of factor A and jth level of factor B
- $`\epsilon_{ijk}`$ is the random error term

### Sum of Squares Decomposition

The total variability in the data is partitioned into four components:

```math
SS_{Total} = SS_A + SS_B + SS_{AB} + SS_{Error}
```

Where:
- $`SS_{Total} = \sum_{i=1}^{a}\sum_{j=1}^{b}\sum_{k=1}^{n_{ij}}(Y_{ijk} - \bar{Y}_{...})^2`$
- $`SS_A = \sum_{i=1}^{a}bn_i(\bar{Y}_{i..} - \bar{Y}_{...})^2`$ (Factor A main effect)
- $`SS_B = \sum_{j=1}^{b}an_j(\bar{Y}_{.j.} - \bar{Y}_{...})^2`$ (Factor B main effect)
- $`SS_{AB} = \sum_{i=1}^{a}\sum_{j=1}^{b}n_{ij}(\bar{Y}_{ij.} - \bar{Y}_{i..} - \bar{Y}_{.j.} + \bar{Y}_{...})^2`$ (Interaction effect)
- $`SS_{Error} = \sum_{i=1}^{a}\sum_{j=1}^{b}\sum_{k=1}^{n_{ij}}(Y_{ijk} - \bar{Y}_{ij.})^2`$ (Error term)

### Degrees of Freedom

```math
df_A = a - 1
df_B = b - 1
df_{AB} = (a - 1)(b - 1)
df_{Error} = N - ab
df_{Total} = N - 1
```

Where $`a`$ and $`b`$ are the number of levels for factors A and B respectively, and $`N`$ is the total sample size.

### F-Statistics

```math
F_A = \frac{MS_A}{MS_{Error}} = \frac{SS_A/df_A}{SS_{Error}/df_{Error}}
```

```math
F_B = \frac{MS_B}{MS_{Error}} = \frac{SS_B/df_B}{SS_{Error}/df_{Error}}
```

```math
F_{AB} = \frac{MS_{AB}}{MS_{Error}} = \frac{SS_{AB}/df_{AB}}{SS_{Error}/df_{Error}}
```

### Effect Size Measures

**Partial Eta-Squared** (most commonly used):
```math
\eta_p^2 = \frac{SS_{Effect}}{SS_{Effect} + SS_{Error}}
```

**Eta-Squared**:
```math
\eta^2 = \frac{SS_{Effect}}{SS_{Total}}
```

**Omega-Squared** (unbiased estimate):
```math
\omega^2 = \frac{SS_{Effect} - df_{Effect} \times MS_{Error}}{SS_{Total} + MS_{Error}}
```

## Basic Two-Way ANOVA

### Manual Calculation

The manual calculation helps understand the underlying mathematics and provides insight into how the statistical software performs the analysis.

> **Python Implementation:** See `manual_two_way_anova()` function in [15_two_way_anova.py](15_two_way_anova.py) for a complete implementation of manual two-way ANOVA calculations, including sum of squares, degrees of freedom, F-statistics, p-values, and effect sizes.

### Using Python's Built-in Two-Way ANOVA

Python's `statsmodels` provides a convenient way to perform two-way ANOVA with automatic calculation of all statistics.

> **Python Implementation:** See `builtin_two_way_anova()` function in [15_two_way_anova.py](15_two_way_anova.py) for using statsmodels to perform two-way ANOVA with automatic calculation of all statistics.

## Descriptive Statistics

### Understanding Cell Means and Marginal Means

In two-way ANOVA, we need to understand three types of means:

1. **Cell Means**: The mean of the dependent variable for each combination of factor levels
2. **Marginal Means**: The mean of the dependent variable for each level of a factor, averaged across all levels of the other factor
3. **Grand Mean**: The overall mean of the dependent variable across all observations

### Mathematical Definitions

**Cell Mean**: $`\bar{Y}_{ij.} = \frac{1}{n_{ij}}\sum_{k=1}^{n_{ij}}Y_{ijk}`$

**Marginal Mean for Factor A**: $`\bar{Y}_{i..} = \frac{1}{bn_i}\sum_{j=1}^{b}\sum_{k=1}^{n_{ij}}Y_{ijk}`$

**Marginal Mean for Factor B**: $`\bar{Y}_{.j.} = \frac{1}{an_j}\sum_{i=1}^{a}\sum_{k=1}^{n_{ij}}Y_{ijk}`$

**Grand Mean**: $`\bar{Y}_{...} = \frac{1}{N}\sum_{i=1}^{a}\sum_{j=1}^{b}\sum_{k=1}^{n_{ij}}Y_{ijk}`$

### Cell Means and Marginal Means

> **Python Implementation:** See `calculate_descriptive_stats()` function in [15_two_way_anova.py](15_two_way_anova.py) for calculating cell means, marginal means, standard errors, and confidence intervals.

### Understanding Interaction Effects

Interaction effects can be calculated as the difference between the observed cell mean and what would be expected based on the main effects alone:

```math
(\alpha\beta)_{ij} = \bar{Y}_{ij.} - \bar{Y}_{i..} - \bar{Y}_{.j.} + \bar{Y}_{...}
```

> **Python Implementation:** See `calculate_interaction_effects()` function in [15_two_way_anova.py](15_two_way_anova.py) for calculating and interpreting interaction effects matrix.

### Visualization

Visualization is crucial for understanding two-way ANOVA results, especially for detecting interaction effects and patterns in the data.

#### Interaction Plot

The interaction plot is the most important visualization for two-way ANOVA as it shows how the effect of one factor changes across the levels of the other factor.

> **Python Implementation:** See `create_interaction_plot()` function in [15_two_way_anova.py](15_two_way_anova.py) for creating interaction plots with confidence intervals.

#### Box Plot with Individual Points

Box plots show the distribution of data within each cell, while individual points show the actual data distribution.

> **Python Implementation:** See `create_box_plot()` function in [15_two_way_anova.py](15_two_way_anova.py) for creating box plots with individual data points.

#### Heatmap of Cell Means

A heatmap provides a visual representation of cell means, making it easy to identify patterns and interactions.

> **Python Implementation:** See `create_heatmap()` function in [15_two_way_anova.py](15_two_way_anova.py) for creating heatmaps of cell means.

#### Marginal Effects Plot

This plot shows the main effects of each factor independently.

> **Python Implementation:** See `create_marginal_effects_plot()` function in [15_two_way_anova.py](15_two_way_anova.py) for creating marginal effects plots.

#### Residuals Plot

Checking residuals is important for assumption validation.

> **Python Implementation:** See `create_residuals_plot()` function in [15_two_way_anova.py](15_two_way_anova.py) for creating residuals plots for model diagnostics.

## Effect Size Analysis

Effect sizes are crucial for understanding the practical significance of statistical results, as they provide information about the magnitude of effects independent of sample size.

### Understanding Effect Size Measures

#### Partial Eta-Squared ($`\eta_p^2`$)

Partial eta-squared is the most commonly used effect size measure in ANOVA. It represents the proportion of variance in the dependent variable that is explained by a specific effect, controlling for other effects in the model.

```math
\eta_p^2 = \frac{SS_{Effect}}{SS_{Effect} + SS_{Error}}
```

**Advantages:**
- Controls for other effects in the model
- Ranges from 0 to 1
- Easy to interpret

**Disadvantages:**
- Can be biased upward in small samples
- Values can sum to more than 1

#### Eta-Squared ($`\eta^2`$)

Eta-squared represents the proportion of total variance explained by an effect.

```math
\eta^2 = \frac{SS_{Effect}}{SS_{Total}}
```

**Advantages:**
- Values sum to 1 across all effects
- Intuitive interpretation

**Disadvantages:**
- Does not control for other effects
- Can be misleading in factorial designs

#### Omega-Squared ($`\omega^2`$)

Omega-squared is an unbiased estimate of the population effect size.

```math
\omega^2 = \frac{SS_{Effect} - df_{Effect} \times MS_{Error}}{SS_{Total} + MS_{Error}}
```

**Advantages:**
- Unbiased estimate
- More conservative than eta-squared

**Disadvantages:**
- Can be negative for small effects
- Less commonly reported

### Comprehensive Effect Size Calculation

> **Python Implementation:** See `calculate_two_way_effect_sizes()` function in [15_two_way_anova.py](15_two_way_anova.py) for comprehensive effect size calculations including partial eta-squared, eta-squared, omega-squared, Cohen's f, and bootstrap confidence intervals.

### Effect Size Interpretation

> **Python Implementation:** See `interpret_effect_size()` function in [15_two_way_anova.py](15_two_way_anova.py) for interpreting effect size values and assessing practical significance.

### Effect Size Visualization

> **Python Implementation:** See `create_effect_size_plot()` function in [15_two_way_anova.py](15_two_way_anova.py) for creating effect size comparison plots.

## Simple Effects Analysis

When a significant interaction is found in two-way ANOVA, the main effects cannot be interpreted independently. Instead, we must examine **simple effects** - the effect of one factor at each level of the other factor.

### Understanding Simple Effects

Simple effects analysis answers questions like:
- "Does the effect of teaching method differ across class sizes?"
- "Is there a difference between automatic and manual transmission for 4-cylinder engines?"
- "Do the effects of drug treatment vary by dosage level?"

### Mathematical Foundation

The simple effect of Factor A at level j of Factor B is:

```math
SS_{A|B_j} = \sum_{i=1}^{a} n_{ij}(\bar{Y}_{ij.} - \bar{Y}_{.j.})^2
```

With degrees of freedom: $`df_{A|B_j} = a - 1`$

The F-statistic for the simple effect is:

```math
F_{A|B_j} = \frac{MS_{A|B_j}}{MS_{Error}} = \frac{SS_{A|B_j}/df_{A|B_j}}{SS_{Error}/df_{Error}}
```

### Comprehensive Simple Effects Analysis

> **Python Implementation:** See `simple_effects_analysis()` function in [15_two_way_anova.py](15_two_way_anova.py) for comprehensive simple effects analysis with effect sizes and descriptive statistics.

### Simple Effects Visualization

> **Python Implementation:** See `plot_simple_effects()` function in [15_two_way_anova.py](15_two_way_anova.py) for creating visualizations of simple effects.

### Post Hoc Tests for Simple Effects

> **Python Implementation:** See `simple_effects_posthoc()` function in [15_two_way_anova.py](15_two_way_anova.py) for performing post hoc tests on significant simple effects.

## Post Hoc Tests

### Post Hoc for Main Effects

> **Python Implementation:** See `perform_posthoc_tests()` function in [15_two_way_anova.py](15_two_way_anova.py) for performing post hoc tests on significant main effects and interactions.

### Post Hoc for Interaction Effects

> **Python Implementation:** See the interaction post hoc section in `perform_posthoc_tests()` function in [15_two_way_anova.py](15_two_way_anova.py) for pairwise comparisons of all cells when interaction is significant.

## Assumption Checking

Two-way ANOVA relies on several key assumptions that must be verified before interpreting results. Violations of these assumptions can lead to incorrect conclusions.

### Key Assumptions

1. **Normality**: Residuals should be normally distributed
2. **Homogeneity of Variance**: Variances should be equal across all groups
3. **Independence**: Observations should be independent
4. **Linearity**: The relationship between factors and the dependent variable should be linear

### Comprehensive Normality Testing

> **Python Implementation:** See `check_normality_two_way()` function in [15_two_way_anova.py](15_two_way_anova.py) for comprehensive normality testing using Shapiro-Wilk, Kolmogorov-Smirnov, and Anderson-Darling tests with diagnostic plots.

### Comprehensive Homogeneity of Variance Testing

> **Python Implementation:** See `check_homogeneity_two_way()` function in [15_two_way_anova.py](15_two_way_anova.py) for comprehensive homogeneity testing using Levene's, Bartlett's, and Fligner-Killeen tests with diagnostic plots.

## Practical Examples

### Example 1: Educational Research

> **Python Implementation:** See `create_educational_example()` function in [15_two_way_anova.py](15_two_way_anova.py) for simulating educational intervention data with teaching method and class size factors.

### Example 2: Clinical Trial

> **Python Implementation:** See `create_clinical_example()` function in [15_two_way_anova.py](15_two_way_anova.py) for simulating clinical trial data with treatment and dosage factors.

### Example 3: Manufacturing Quality

> **Python Implementation:** See `create_manufacturing_example()` function in [15_two_way_anova.py](15_two_way_anova.py) for simulating manufacturing quality data with machine type and shift factors.

## Best Practices and Guidelines

Following best practices ensures reliable, reproducible, and interpretable two-way ANOVA results. This section provides comprehensive guidelines for conducting and reporting two-way ANOVA analyses.

### Test Selection Guidelines

> **Python Implementation:** The main execution block in [15_two_way_anova.py](15_two_way_anova.py) demonstrates a complete workflow including data preparation, assumption checking, analysis, and interpretation.

### Data Preparation Best Practices

> **Python Implementation:** See `load_sample_data()` function in [15_two_way_anova.py](15_two_way_anova.py) for data preparation and factor creation.

### Reporting Guidelines

> **Python Implementation:** The main execution block in [15_two_way_anova.py](15_two_way_anova.py) shows how to generate comprehensive reports including descriptive statistics, ANOVA results, effect sizes, and interpretation.

## Comprehensive Exercises

The following exercises are designed to help you master two-way ANOVA concepts, from basic applications to advanced analyses.

### Exercise 1: Basic Two-Way ANOVA Analysis

**Objective**: Perform a complete two-way ANOVA analysis on the mtcars dataset.

**Task**: Analyze the effects of cylinder count (4, 6, 8) and transmission type (automatic, manual) on horsepower.

**Requirements**:
1. Create appropriate factor variables
2. Perform manual calculations for all components (SS, df, MS, F, p-values)
3. Use Python's built-in functions
4. Calculate descriptive statistics for all cells
5. Create interaction plots and box plots
6. Calculate effect sizes (partial η², η², ω²)
7. Interpret results comprehensively

**Expected Learning Outcomes**:
- Understanding of two-way ANOVA mathematical foundations
- Ability to perform both manual and automated calculations
- Skills in creating informative visualizations
- Competence in effect size interpretation

**Hints**:
- Use the functions in [15_two_way_anova.py](15_two_way_anova.py)
- Remember to check assumptions before interpretation
- Consider the practical significance of results

### Exercise 2: Interaction Effects Analysis

**Objective**: Create and analyze a dataset with significant interaction effects.

**Task**: Generate a simulated dataset with a 2×3 factorial design where:
- Factor A has 2 levels
- Factor B has 3 levels
- There is a significant interaction effect
- Sample size is 20 per cell

**Requirements**:
1. Generate data with known interaction effects
2. Perform two-way ANOVA
3. Conduct simple effects analysis
4. Create interaction plots
5. Perform post hoc tests for significant simple effects
6. Calculate and interpret effect sizes for all effects

**Expected Learning Outcomes**:
- Understanding of interaction effects
- Ability to perform simple effects analysis
- Skills in interpreting complex factorial designs
- Competence in post hoc testing for interactions

**Hints**:
- Use the example functions in [15_two_way_anova.py](15_two_way_anova.py) as templates
- Consider using different means for different cells to create interactions
- Remember that interactions can be ordinal or disordinal

### Exercise 3: Comprehensive Assumption Checking

**Objective**: Develop expertise in assumption checking and alternative methods.

**Task**: Using the mtcars dataset, perform comprehensive assumption checking and recommend appropriate analyses.

**Requirements**:
1. Test normality using multiple methods (Shapiro-Wilk, KS, Anderson-Darling)
2. Test homogeneity of variance using multiple methods (Levene's, Bartlett's, Fligner-Killeen)
3. Create diagnostic plots (Q-Q plots, residual plots, variance plots)
4. If assumptions are violated, perform appropriate alternatives:
   - Data transformations (log, square root, etc.)
   - Nonparametric alternatives
   - Robust methods
5. Compare results between parametric and nonparametric approaches

**Expected Learning Outcomes**:
- Comprehensive understanding of ANOVA assumptions
- Ability to choose appropriate alternative methods
- Skills in diagnostic plotting and interpretation
- Competence in data transformation

**Hints**:
- Use the assumption checking functions in [15_two_way_anova.py](15_two_way_anova.py)
- Consider robust alternatives when assumptions are violated
- Document your decision-making process

### Exercise 4: Advanced Effect Size Analysis

**Objective**: Master effect size calculations and interpretation.

**Task**: Perform comprehensive effect size analysis for a two-way ANOVA.

**Requirements**:
1. Calculate all effect size measures (partial η², η², ω², Cohen's f)
2. Compute bootstrap confidence intervals for effect sizes
3. Create effect size comparison plots
4. Assess practical significance
5. Perform power analysis
6. Compare effect sizes across different datasets

**Expected Learning Outcomes**:
- Deep understanding of effect size measures
- Ability to interpret practical significance
- Skills in power analysis
- Competence in bootstrap methods

**Hints**:
- Use the effect size functions in [15_two_way_anova.py](15_two_way_anova.py)
- Consider using the boot package for bootstrap confidence intervals
- Remember that effect sizes are sample size independent

### Exercise 5: Real-World Application

**Objective**: Apply two-way ANOVA to a real-world research scenario.

**Task**: Design and analyze a research study using two-way ANOVA.

**Scenario Options**:
- **Educational Research**: Teaching method × Class size on student performance
- **Clinical Trial**: Drug treatment × Dosage level on patient outcomes
- **Manufacturing**: Machine type × Shift on product quality
- **Marketing**: Advertisement type × Target audience on purchase behavior

**Requirements**:
1. Design the study with appropriate sample sizes
2. Generate realistic data based on the scenario
3. Perform complete two-way ANOVA analysis
4. Create publication-ready visualizations
5. Write a comprehensive results section
6. Discuss practical implications

**Expected Learning Outcomes**:
- Ability to design factorial research studies
- Skills in data generation and analysis
- Competence in results presentation
- Understanding of practical applications

**Hints**:
- Use the example functions in [15_two_way_anova.py](15_two_way_anova.py) as starting points
- Consider effect sizes when determining sample sizes
- Use realistic means and standard deviations
- Include interaction effects that make sense for the scenario

### Exercise 6: Advanced Topics

**Objective**: Explore advanced two-way ANOVA topics.

**Task**: Investigate one or more advanced topics:

**Options**:
1. **Unbalanced Designs**: Analyze data with unequal cell sizes
2. **Mixed Models**: Use mixed-effects ANOVA
3. **Robust ANOVA**: Implement robust alternatives
4. **Bootstrap ANOVA**: Perform bootstrap-based ANOVA
5. **Bayesian ANOVA**: Use Bayesian ANOVA

**Requirements**:
1. Implement the chosen advanced method
2. Compare results with standard two-way ANOVA
3. Discuss advantages and limitations
4. Create appropriate visualizations
5. Provide recommendations for when to use each method

**Expected Learning Outcomes**:
- Understanding of advanced ANOVA methods
- Ability to choose appropriate methods for different situations
- Skills in implementing specialized packages
- Competence in method comparison

**Hints**:
- For unbalanced designs, consider Type I vs Type III SS
- For mixed models, understand random vs fixed effects
- For robust methods, understand when they're most useful

### Exercise Solutions Framework

For each exercise, follow this systematic approach:

1. **Data Preparation**:
   - Check data structure and quality
   - Create appropriate factor variables
   - Handle missing values appropriately

2. **Exploratory Analysis**:
   - Calculate descriptive statistics
   - Create initial visualizations
   - Identify potential issues

3. **Assumption Checking**:
   - Test normality and homogeneity
   - Create diagnostic plots
   - Decide on appropriate analysis method

4. **Statistical Analysis**:
   - Perform the chosen analysis
   - Calculate effect sizes
   - Conduct post hoc tests if needed

5. **Results Interpretation**:
   - Interpret statistical significance
   - Assess practical significance
   - Consider limitations and assumptions

6. **Reporting**:
   - Create publication-ready tables and figures
   - Write clear interpretations
   - Provide recommendations

**Learning Progression**:
- Start with Exercise 1 to build foundational skills
- Progress through exercises 2-4 to develop advanced competencies
- Complete Exercise 5 to apply skills to real-world scenarios
- Attempt Exercise 6 to explore cutting-edge methods

**Assessment Criteria**:
- Correct implementation of statistical methods
- Appropriate interpretation of results
- Quality of visualizations and reporting
- Understanding of underlying concepts
- Ability to make practical recommendations

## Next Steps

In the next chapter, we'll learn about repeated measures ANOVA for analyzing within-subject designs.

---

**Key Takeaways:**
- Two-way ANOVA analyzes effects of two independent variables and their interaction
- Always check assumptions before interpreting results
- Effect sizes provide important information about practical significance
- Simple effects analysis is needed when interactions are significant
- Post hoc tests are necessary for significant main effects
- Proper reporting includes descriptive statistics, test results, and effect sizes
- Interaction effects can modify the interpretation of main effects