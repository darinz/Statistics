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

**Python Example:**
```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Paired data example
before = np.array([120, 115, 130, 140, 125])
after = np.array([118, 117, 128, 135, 130])

# Wilcoxon signed-rank test
statistic, p_value = stats.wilcoxon(before, after)
print(f"Wilcoxon signed-rank test:")
print(f"Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Visualize paired differences
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(before, after)
plt.plot([110, 150], [110, 150], 'r--', alpha=0.7)
plt.xlabel('Before')
plt.ylabel('After')
plt.title('Paired Data')

plt.subplot(1, 2, 2)
differences = before - after
plt.hist(differences, bins=5, alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel('Difference (Before - After)')
plt.ylabel('Frequency')
plt.title('Distribution of Differences')

plt.tight_layout()
plt.show()
```

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

**Python Example:**
```python
# Independent groups data
group1 = np.array([12, 15, 14, 10, 13])
group2 = np.array([18, 20, 17, 16, 19])

# Mann-Whitney U test
statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
print(f"Mann-Whitney U test:")
print(f"Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Visualize group comparison
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.boxplot([group1, group2], labels=['Group 1', 'Group 2'])
plt.ylabel('Values')
plt.title('Boxplot Comparison')

plt.subplot(1, 2, 2)
all_data = np.concatenate([group1, group2])
all_labels = ['Group 1'] * len(group1) + ['Group 2'] * len(group2)
sns.violinplot(x=all_labels, y=all_data)
plt.ylabel('Values')
plt.title('Violin Plot Comparison')

plt.tight_layout()
plt.show()
```

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

**Python Example:**
```python
# Create data for 3 groups
np.random.seed(42)
group1 = np.random.normal(10, 2, 5)
group2 = np.random.normal(12, 2, 5)
group3 = np.random.normal(15, 2, 5)

# Combine data and create group labels
values = np.concatenate([group1, group2, group3])
groups = np.repeat([1, 2, 3], 5)

# Kruskal-Wallis test
statistic, p_value = stats.kruskal(group1, group2, group3)
print(f"Kruskal-Wallis test:")
print(f"Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Visualize group comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.boxplot([group1, group2, group3], labels=['Group 1', 'Group 2', 'Group 3'])
plt.ylabel('Values')
plt.title('Boxplot Comparison')

plt.subplot(1, 2, 2)
group_labels = ['Group 1'] * len(group1) + ['Group 2'] * len(group2) + ['Group 3'] * len(group3)
sns.violinplot(x=group_labels, y=values)
plt.ylabel('Values')
plt.title('Violin Plot Comparison')

plt.tight_layout()
plt.show()
```

**Assumptions:**
- Independent samples
- Ordinal or continuous data
- Similar shapes of distributions

**Post Hoc:** Use pairwise Wilcoxon tests with p-value adjustment.

---

### 4. Friedman Test

**Purpose:** Compare more than two related groups (analogous to repeated measures ANOVA).

**Mathematical Foundation:**
Ranks data within each block/subject, then computes:
```math
Q = \frac{12}{nk(k+1)} \sum_{j=1}^k R_j^2 - 3n(k+1)
```
where $`R_j`$ is the sum of ranks for treatment $`j`$, $`n`$ is the number of blocks, $`k`$ is the number of treatments.

**Python Example:**
```python
# Data: 3 treatments, 5 subjects
values = np.array([
    [10, 12, 11, 13, 14],
    [20, 22, 19, 21, 23],
    [30, 28, 27, 29, 31]
]).T  # Transpose to get subjects as rows

# Friedman test
statistic, p_value = stats.friedmanchisquare(values[:, 0], values[:, 1], values[:, 2])
print(f"Friedman test:")
print(f"Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Visualize repeated measures
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.boxplot([values[:, 0], values[:, 1], values[:, 2]], 
           labels=['Treatment 1', 'Treatment 2', 'Treatment 3'])
plt.ylabel('Values')
plt.title('Boxplot by Treatment')

plt.subplot(1, 2, 2)
# Individual profiles
for i in range(len(values)):
    plt.plot([1, 2, 3], values[i], 'o-', alpha=0.7, label=f'Subject {i+1}')
plt.xlabel('Treatment')
plt.ylabel('Values')
plt.title('Individual Profiles')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
```

**Assumptions:**
- Related samples (blocks/subjects)
- Ordinal or continuous data

**Post Hoc:** Use pairwise Wilcoxon signed-rank tests with adjustment.

---

### 5. Sign Test

**Purpose:** Test for median difference in paired data (less powerful than Wilcoxon).

**Mathematical Foundation:**
Counts the number of positive and negative differences. Under the null, the number of positives follows a binomial distribution $`B(n, 0.5)`$.

**Python Example:**
```python
# Paired data
diff = before - after
n_pos = np.sum(diff > 0)
n_total = np.sum(diff != 0)

# Binomial test
statistic, p_value = stats.binomtest(n_pos, n_total, p=0.5).proportions_ci()
print(f"Sign test:")
print(f"Positive differences: {n_pos}")
print(f"Total non-zero differences: {n_total}")
print(f"P-value: {p_value:.4f}")

# Visualize sign test
plt.figure(figsize=(8, 6))
signs = ['Positive' if d > 0 else 'Negative' for d in diff if d != 0]
sign_counts = pd.Series(signs).value_counts()
plt.bar(sign_counts.index, sign_counts.values)
plt.ylabel('Count')
plt.title('Sign Test Results')
plt.show()
```

**Assumptions:**
- Paired samples
- No assumption about distribution shape

---

### 6. McNemar's Test

**Purpose:** Test for changes in paired binary data (e.g., pre/post intervention).

**Mathematical Foundation:**
For a 2x2 table:
```math
\chi^2 = \frac{(b-c)^2}{b+c}
```
where $`b`$ and $`c`$ are discordant pairs.

**Python Example:**
```python
# Create 2x2 contingency table
# Rows: Before, Columns: After
# [a b]
# [c d]
tab = np.array([[30, 10], [5, 55]])

# McNemar's test
statistic, p_value = stats.mcnemar(tab, exact=False)
print(f"McNemar's test:")
print(f"Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Visualize contingency table
plt.figure(figsize=(8, 6))
sns.heatmap(tab, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['After: No', 'After: Yes'],
           yticklabels=['Before: No', 'Before: Yes'])
plt.title('McNemar Test Contingency Table')
plt.show()
```

**Assumptions:**
- Paired binary data

---

### 7. Chi-Square Test of Independence

**Purpose:** Test association between two categorical variables.

**Mathematical Foundation:**
```math
\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
```
where $`O_{ij}`$ is observed count, $`E_{ij}`$ is expected count.

**Python Example:**
```python
# Create 2x2 contingency table
tab = np.array([[20, 15], [30, 35]])

# Chi-square test
statistic, p_value, dof, expected = stats.chi2_contingency(tab)
print(f"Chi-square test of independence:")
print(f"Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

# Visualize contingency table
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(tab, annot=True, fmt='d', cmap='Blues')
plt.title('Observed Frequencies')

plt.subplot(1, 2, 2)
sns.heatmap(expected, annot=True, fmt='.1f', cmap='Greens')
plt.title('Expected Frequencies')

plt.tight_layout()
plt.show()
```

**Assumptions:**
- Categorical data
- Expected cell counts > 5 (for validity)

---

### 8. Fisher's Exact Test

**Purpose:** Test association in small 2x2 tables (when chi-square assumptions are not met).

**Python Example:**
```python
# Small 2x2 table
tab = np.array([[2, 3], [8, 7]])

# Fisher's exact test
odds_ratio, p_value = stats.fisher_exact(tab)
print(f"Fisher's exact test:")
print(f"Odds ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.4f}")

# Visualize small table
plt.figure(figsize=(6, 6))
sns.heatmap(tab, annot=True, fmt='d', cmap='Reds')
plt.title('Fisher\'s Exact Test Table')
plt.show()
```

**Assumptions:**
- Categorical data
- Small sample sizes

---

## Effect Size for Nonparametric Tests

- **Rank-biserial correlation** for Wilcoxon/Mann-Whitney:
  $`r = 1 - \frac{2U}{n_1 n_2}`$
- **Eta squared** for Kruskal-Wallis:
  $`\eta^2 = \frac{H - k + 1}{n - k}`$
- **Cramér's V** for chi-square:
  $`V = \sqrt{\frac{\chi^2}{n(k-1)}}`$

**Python Example:**
```python
def wilcox_effsize(x, y):
    """Calculate effect size for Mann-Whitney U test"""
    statistic, _ = stats.mannwhitneyu(x, y, alternative='two-sided')
    n1, n2 = len(x), len(y)
    r = 1 - (2 * statistic) / (n1 * n2)
    return r

def cramers_v(contingency_table):
    """Calculate Cramér's V for chi-square test"""
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = np.sum(contingency_table)
    k = min(contingency_table.shape)
    v = np.sqrt(chi2 / (n * (k - 1)))
    return v

# Effect size for Mann-Whitney
eff_size = wilcox_effsize(group1, group2)
print(f"Rank-biserial correlation: {eff_size:.4f}")

# Cramér's V for chi-square
tab = np.array([[20, 15], [30, 35]])
cramers_v_value = cramers_v(tab)
print(f"Cramér's V: {cramers_v_value:.4f}")
```

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

```python
# Simulate data for 3 groups
np.random.seed(42)
g1 = np.random.normal(10, 2, 20)
g2 = np.random.normal(12, 2, 20)
g3 = np.random.normal(15, 2, 20)
values = np.concatenate([g1, g2, g3])
group = np.repeat([1, 2, 3], 20)

# Kruskal-Wallis test
statistic, p_value = stats.kruskal(g1, g2, g3)
print(f"Kruskal-Wallis test:")
print(f"Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Post hoc pairwise Wilcoxon tests with Bonferroni correction
from itertools import combinations
groups = [g1, g2, g3]
group_names = ['Group 1', 'Group 2', 'Group 3']

print("\nPost hoc pairwise comparisons (Bonferroni corrected):")
for i, (g1_name, g2_name) in enumerate(combinations(group_names, 2)):
    idx1, idx2 = i, i + 1
    statistic, p_value = stats.mannwhitneyu(groups[idx1], groups[idx2], alternative='two-sided')
    # Bonferroni correction for 3 comparisons
    p_corrected = min(p_value * 3, 1.0)
    print(f"{g1_name} vs {g2_name}: p = {p_corrected:.4f}")

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.boxplot([g1, g2, g3], labels=['Group 1', 'Group 2', 'Group 3'])
plt.ylabel('Values')
plt.title('Group Comparison')

plt.subplot(1, 2, 2)
group_labels = ['Group 1'] * len(g1) + ['Group 2'] * len(g2) + ['Group 3'] * len(g3)
sns.violinplot(x=group_labels, y=values)
plt.ylabel('Values')
plt.title('Distribution Comparison')

plt.tight_layout()
plt.show()
```

---

## Exercises

### Exercise 1: Wilcoxon Signed-Rank Test
- **Objective:** Test whether a treatment changes median blood pressure in paired samples.
- **Hint:** Use `stats.wilcoxon()` for paired data.

### Exercise 2: Mann-Whitney U Test
- **Objective:** Compare two independent groups on a non-normal outcome.
- **Hint:** Use `stats.mannwhitneyu()`.

### Exercise 3: Kruskal-Wallis and Post Hoc
- **Objective:** Compare three or more groups and identify which differ.
- **Hint:** Use `stats.kruskal()` and pairwise `stats.mannwhitneyu()` with correction.

### Exercise 4: Friedman Test
- **Objective:** Analyze repeated measures data from three time points.
- **Hint:** Use `stats.friedmanchisquare()`.

### Exercise 5: Chi-Square and Fisher's Exact Test
- **Objective:** Test association between two categorical variables in a contingency table.
- **Hint:** Use `stats.chi2_contingency()` and `stats.fisher_exact()` for small samples.

---

**Key Takeaways:**
- Nonparametric tests are robust alternatives when parametric assumptions are violated
- They are based on ranks or counts, not means
- Always check assumptions and report effect sizes
- Use appropriate post hoc tests and visualize results
- Interpret findings in the context of the research question 