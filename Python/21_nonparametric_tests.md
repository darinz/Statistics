# Nonparametric Tests

## Introduction

Nonparametric tests are statistical methods that do not assume a specific distribution for the data (such as normality). They are especially useful when data are ordinal, not normally distributed, have outliers, or when sample sizes are small. Nonparametric tests are often based on ranks rather than raw data values.

**When to Use Nonparametric Tests:**
- Data are ordinal or nominal
- Assumptions of parametric tests (e.g., normality, homogeneity of variance) are violated
- Small sample sizes
- Presence of outliers or skewed distributions

## Key Nonparametric Tests

### 1. Wilcoxon Signed-Rank Test

**Purpose:** Compare the median of a single sample to a known value, or compare paired samples (analogous to the paired t-test).

**Mathematical Foundation:**
Given paired differences $`d_i = x_i - y_i`$, the test ranks the absolute differences, assigns signs, and sums the signed ranks.

**Test Statistic:**
```math
W = \sum_{i=1}^n \text{sgn}(d_i) R_i
```
where $`R_i`$ is the rank of $`|d_i|`$.

**Python Implementation:**
See `wilcoxon_signed_rank()` in `21_nonparametric_tests.py` for code and visualization.

**Assumptions:**
- Paired samples
- Differences are symmetrically distributed

**Interpretation:**
A small p-value suggests a significant difference in medians.

---

### 2. Mann-Whitney U Test (Wilcoxon Rank-Sum Test)

**Purpose:** Compare medians of two independent samples (analogous to the independent t-test).

**Mathematical Foundation:**
Ranks all observations, sums ranks for each group, and computes the U statistic:
```math
U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1
```
where $`R_1`$ is the sum of ranks for group 1.

**Python Implementation:**
See `mann_whitney_u()` in `21_nonparametric_tests.py` for code and visualization.

**Assumptions:**
- Independent samples
- Ordinal or continuous data
- Similar shapes of distributions

**Interpretation:**
A small p-value suggests a difference in distributions/medians.

---

### 3. Kruskal-Wallis Test

**Purpose:** Compare medians across more than two independent groups (analogous to one-way ANOVA).

**Mathematical Foundation:**
Ranks all data, computes the test statistic:
```math
H = \frac{12}{N(N+1)} \sum_{i=1}^k n_i (\bar{R}_i - \bar{R})^2
```
where $`n_i`$ is the sample size of group $`i`$, $`\bar{R}_i`$ is the mean rank of group $`i`$, and $`N`$ is the total sample size.

**Python Implementation:**
See `kruskal_wallis()` in `21_nonparametric_tests.py` for code and visualization.

**Post Hoc:** Use pairwise Wilcoxon tests with p-value adjustment. See the practical example and `kruskal_posthoc_example()` in the Python file.

---

### 4. Friedman Test

**Purpose:** Compare more than two related groups (analogous to repeated measures ANOVA).

**Mathematical Foundation:**
Ranks data within each block/subject, then computes:
```math
Q = \frac{12}{nk(k+1)} \sum_{j=1}^k R_j^2 - 3n(k+1)
```
where $`R_j`$ is the sum of ranks for treatment $`j`$, $`n`$ is the number of blocks, $`k`$ is the number of treatments.

**Python Implementation:**
See `friedman_test()` in `21_nonparametric_tests.py` for code and visualization.

---

### 5. Sign Test

**Purpose:** Test for median difference in paired data (less powerful than Wilcoxon).

**Mathematical Foundation:**
Counts the number of positive and negative differences. Under the null, the number of positives follows a binomial distribution $`B(n, 0.5)`$.

**Python Implementation:**
See `sign_test()` in `21_nonparametric_tests.py` for code and visualization.

---

### 6. McNemar's Test

**Purpose:** Test for changes in paired binary data (e.g., pre/post intervention).

**Mathematical Foundation:**
For a 2x2 table:
```math
\chi^2 = \frac{(b-c)^2}{b+c}
```
where $`b`$ and $`c`$ are discordant pairs.

**Python Implementation:**
See `mcnemar_test()` in `21_nonparametric_tests.py` for code and visualization.

---

### 7. Chi-Square Test of Independence

**Purpose:** Test association between two categorical variables.

**Mathematical Foundation:**
```math
\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
```
where $`O_{ij}`$ is observed count, $`E_{ij}`$ is expected count.

**Python Implementation:**
See `chi_square_independence()` in `21_nonparametric_tests.py` for code and visualization.

---

### 8. Fisher's Exact Test

**Purpose:** Test association in small 2x2 tables (when chi-square assumptions are not met).

**Python Implementation:**
See `fishers_exact()` in `21_nonparametric_tests.py` for code and visualization.

---

## Effect Size for Nonparametric Tests

- **Rank-biserial correlation** for Wilcoxon/Mann-Whitney: $`r = 1 - \frac{2U}{n_1 n_2}`$
- **Eta squared** for Kruskal-Wallis: $`\eta^2 = \frac{H - k + 1}{n - k}`$
- **Cramér's V** for chi-square: $`V = \sqrt{\frac{\chi^2}{n(k-1)}}`$

**Python Implementation:**
See `wilcox_effsize()` and `cramers_v()` in `21_nonparametric_tests.py` for effect size calculations.

---

## Best Practices

- Check assumptions (independence, scale, shape)
- Use exact tests for small samples
- Report effect sizes and confidence intervals
- Use appropriate post hoc tests with adjustment
- Visualize data (boxplots, barplots, mosaic plots)
- Clearly state hypotheses and interpret results in context

---

## Practical Example: Kruskal-Wallis and Post Hoc

See `kruskal_posthoc_example()` in `21_nonparametric_tests.py` for a full demonstration of Kruskal-Wallis test, post hoc pairwise comparisons, and visualization.

---

## Python Implementation Reference

**Function Mapping:**
- Wilcoxon Signed-Rank: `wilcoxon_signed_rank()`
- Mann-Whitney U: `mann_whitney_u()`
- Kruskal-Wallis: `kruskal_wallis()`
- Friedman: `friedman_test()`
- Sign Test: `sign_test()`
- McNemar's Test: `mcnemar_test()`
- Chi-Square Test: `chi_square_independence()`
- Fisher's Exact Test: `fishers_exact()`
- Effect Sizes: `wilcox_effsize()`, `cramers_v()`
- Practical Example: `kruskal_posthoc_example()`

**Usage:**
Import the functions from `21_nonparametric_tests.py` and use them as demonstrated in the file's main block. Each function prints results and produces relevant plots for interpretation.

---

**Key Takeaways:**
- Nonparametric tests are robust alternatives when parametric assumptions are violated
- They are based on ranks or counts, not means
- Always check assumptions and report effect sizes
- Use appropriate post hoc tests and visualize results
- Interpret findings in the context of the research question 