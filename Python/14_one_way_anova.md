# One-Way ANOVA

> **Note:** All Python code for this lesson has been moved to [14_one_way_anova.py](14_one_way_anova.py). Code blocks in this markdown are now replaced with references to the relevant functions and sections in the Python file. Use both files together for a complete learning experience.

## Overview

One-way Analysis of Variance (ANOVA) is a powerful statistical technique used to compare means across three or more groups simultaneously. It extends the two-sample t-test to multiple groups and is fundamental to experimental design and observational studies.

### When to Use One-Way ANOVA

One-way ANOVA is appropriate when you want to:
- Compare means across three or more independent groups
- Test the null hypothesis that all group means are equal
- Determine if there are significant differences between any groups
- Analyze experimental data with one categorical independent variable
- Compare multiple treatments, conditions, or categories

### Key Concepts

**Null Hypothesis (H₀):** All population means are equal ($\mu_1 = \mu_2 = \mu_3 = ... = \mu_k$)
**Alternative Hypothesis (H₁):** At least one population mean differs from the others

The test determines whether observed differences between group means are statistically significant or due to random sampling variation.

### Mathematical Foundation

One-way ANOVA is based on partitioning the total variance in the data into two components:

**Total Sum of Squares (SST):**
```math
SST = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_{..})^2
```

**Between-Groups Sum of Squares (SSB):**
```math
SSB = \sum_{i=1}^{k} n_i(\bar{X}_{i.} - \bar{X}_{..})^2
```

**Within-Groups Sum of Squares (SSW):**
```math
SSW = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_{i.})^2
```

where:
- $X_{ij}$ = observation $j$ in group $i$
- $\bar{X}_{i.}$ = mean of group $i$
- $\bar{X}_{..}$ = overall mean
- $n_i$ = sample size of group $i$
- $k$ = number of groups

**F-Statistic:**
```math
F = \frac{MSB}{MSW} = \frac{SSB/(k-1)}{SSW/(N-k)}
```

where $N = \sum_{i=1}^{k} n_i$ is the total sample size.

The F-statistic follows an F-distribution with $(k-1, N-k)$ degrees of freedom under the null hypothesis.

## Basic One-Way ANOVA

The basic one-way ANOVA procedure involves calculating the F-statistic by comparing the variance between groups to the variance within groups. Understanding the manual calculation helps clarify the underlying principles.

### Mathematical Foundation

**Variance Partitioning:**
The total variance in the data is partitioned into two components:
1. **Between-groups variance:** Measures differences between group means
2. **Within-groups variance:** Measures variability within each group

**Expected Values Under Null Hypothesis:**
- $E(MSB) = \sigma^2 + \frac{\sum_{i=1}^{k} n_i(\mu_i - \bar{\mu})^2}{k-1}$
- $E(MSW) = \sigma^2$

where $\bar{\mu} = \frac{\sum_{i=1}^{k} n_i\mu_i}{N}$ is the weighted grand mean.

Under the null hypothesis ($\mu_1 = \mu_2 = ... = \mu_k$), both expected values equal $\sigma^2$, so $E(F) = 1$.

**F-Distribution Properties:**
- $F \sim F(k-1, N-k)$ under $H_0$
- $F \geq 0$ (always positive)
- Reject $H_0$ if $F > F_{\alpha, k-1, N-k}$

### Manual Calculation

**See:** `manual_anova()` in [14_one_way_anova.py](14_one_way_anova.py)

*This function demonstrates a step-by-step manual calculation of one-way ANOVA, including sums of squares, mean squares, F-statistic, p-value, and effect sizes.*

---

### Using Python's Built-in ANOVA

**See:** `builtin_anova()` and `statsmodels_anova()` in [14_one_way_anova.py](14_one_way_anova.py)

*These functions show how to perform ANOVA using scipy and statsmodels, including model diagnostics and confidence intervals.*

---

## Descriptive Statistics

**See:** `descriptive_stats()` and `pooled_variance()` in [14_one_way_anova.py](14_one_way_anova.py)

*These functions provide group-wise and overall descriptive statistics, including means, standard deviations, standard errors, coefficients of variation, skewness, and kurtosis.*

---

## Visualization

**See:** `anova_visualizations()` in [14_one_way_anova.py](14_one_way_anova.py)

*This function creates boxplots, violin plots, histograms, Q-Q plots, residuals vs fitted plots, and mean comparison plots for ANOVA diagnostics and group comparison.*

---

## Effect Size

**See:** `anova_effect_sizes()` in [14_one_way_anova.py](14_one_way_anova.py)

*Calculates eta-squared, omega-squared, Cohen's f, and interprets effect size for ANOVA.*

---

## Post Hoc Tests

**See:** `posthoc_tests()` in [14_one_way_anova.py](14_one_way_anova.py)

*Runs Tukey's HSD, Bonferroni, Holm, and FDR post hoc tests for multiple comparisons after ANOVA.*

---

## Assumption Checking

**See:** `check_assumptions()` in [14_one_way_anova.py](14_one_way_anova.py)

*Checks normality, homogeneity of variance, and outliers for ANOVA assumptions.*

---

## Nonparametric Alternatives

**See:** `kruskal_wallis()` in [14_one_way_anova.py](14_one_way_anova.py)

*Performs Kruskal-Wallis test and pairwise Mann-Whitney U tests as nonparametric alternatives to ANOVA.*

---

## Power Analysis

**See:** `power_analysis_anova()` in [14_one_way_anova.py](14_one_way_anova.py)

*Calculates statistical power for one-way ANOVA given effect size and sample size.*

---

## Practical Examples

**See:** `example_educational()`, `example_clinical()`, and `example_quality()` in [14_one_way_anova.py](14_one_way_anova.py)

*These functions simulate and analyze real-world scenarios for educational research, clinical trials, and quality control using one-way ANOVA.*

---

## Exercises

The exercises below reference the code in [14_one_way_anova.py](14_one_way_anova.py). For each exercise, use the relevant functions as described above.

---

## Next Steps

In the next chapter, we'll learn about two-way ANOVA for analyzing the effects of two independent variables.

---

**Key Takeaways:**
- One-way ANOVA compares means across three or more groups
- Always check assumptions before interpreting results
- Effect sizes provide important information about practical significance
- Post hoc tests are necessary when ANOVA is significant
- Nonparametric alternatives exist for non-normal data
- Power analysis helps determine appropriate sample sizes
- Proper reporting includes descriptive statistics, test results, and effect sizes