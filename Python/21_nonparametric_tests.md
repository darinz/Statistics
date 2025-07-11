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

**R Example:**
```r
# Paired data example
before <- c(120, 115, 130, 140, 125)
after <- c(118, 117, 128, 135, 130)

# Wilcoxon signed-rank test
wilcox.test(before, after, paired = TRUE)
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

**R Example:**
```r
group1 <- c(12, 15, 14, 10, 13)
group2 <- c(18, 20, 17, 16, 19)

wilcox.test(group1, group2)
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

**R Example:**
```r
group <- factor(rep(1:3, each = 5))
values <- c(10, 12, 11, 13, 14, 20, 22, 19, 21, 23, 30, 28, 27, 29, 31)

kruskal.test(values ~ group)
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

**R Example:**
```r
# Data: 3 treatments, 5 subjects
values <- matrix(c(10, 12, 11, 13, 14,
                  20, 22, 19, 21, 23,
                  30, 28, 27, 29, 31), nrow = 5, byrow = FALSE)

friedman.test(values)
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

**R Example:**
```r
# Paired data
diff <- before - after
n_pos <- sum(diff > 0)
n <- sum(diff != 0)

# Binomial test
binom.test(n_pos, n, p = 0.5)
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

**R Example:**
```r
tab <- matrix(c(30, 10, 5, 55), nrow = 2)
mcnemar.test(tab)
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

**R Example:**
```r
tab <- matrix(c(20, 15, 30, 35), nrow = 2)
chisq.test(tab)
```

**Assumptions:**
- Categorical data
- Expected cell counts > 5 (for validity)

---

### 8. Fisher's Exact Test

**Purpose:** Test association in small 2x2 tables (when chi-square assumptions are not met).

**R Example:**
```r
tab <- matrix(c(2, 3, 8, 7), nrow = 2)
fisher.test(tab)
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

**R Example:**
```r
# Effect size for Mann-Whitney
wilcox_effsize <- function(x, y) {
  res <- wilcox.test(x, y)
  U <- res$statistic
  n1 <- length(x)
  n2 <- length(y)
  r <- 1 - (2 * U) / (n1 * n2)
  return(r)
}

# Cramér's V for chi-square
tab <- matrix(c(20, 15, 30, 35), nrow = 2)
chisq <- chisq.test(tab)
cramers_v <- sqrt(chisq$statistic / (sum(tab) * (min(dim(tab)) - 1)))
cramers_v
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

```r
# Simulate data for 3 groups
set.seed(42)
g1 <- rnorm(20, mean = 10)
g2 <- rnorm(20, mean = 12)
g3 <- rnorm(20, mean = 15)
values <- c(g1, g2, g3)
group <- factor(rep(1:3, each = 20))

# Kruskal-Wallis test
kruskal.test(values ~ group)

# Post hoc pairwise Wilcoxon tests
pairwise.wilcox.test(values, group, p.adjust.method = "bonferroni")

# Boxplot
boxplot(values ~ group, main = "Group Comparison", ylab = "Value")
```

---

## Exercises

### Exercise 1: Wilcoxon Signed-Rank Test
- **Objective:** Test whether a treatment changes median blood pressure in paired samples.
- **Hint:** Use `wilcox.test()` with `paired = TRUE`.

### Exercise 2: Mann-Whitney U Test
- **Objective:** Compare two independent groups on a non-normal outcome.
- **Hint:** Use `wilcox.test()`.

### Exercise 3: Kruskal-Wallis and Post Hoc
- **Objective:** Compare three or more groups and identify which differ.
- **Hint:** Use `kruskal.test()` and `pairwise.wilcox.test()`.

### Exercise 4: Friedman Test
- **Objective:** Analyze repeated measures data from three time points.
- **Hint:** Use `friedman.test()`.

### Exercise 5: Chi-Square and Fisher's Exact Test
- **Objective:** Test association between two categorical variables in a contingency table.
- **Hint:** Use `chisq.test()` and `fisher.test()` for small samples.

---

**Key Takeaways:**
- Nonparametric tests are robust alternatives when parametric assumptions are violated
- They are based on ranks or counts, not means
- Always check assumptions and report effect sizes
- Use appropriate post hoc tests and visualize results
- Interpret findings in the context of the research question 