# Hypothesis Testing

## Overview

Hypothesis testing is a fundamental concept in statistics that allows us to make decisions about populations based on sample data. It provides a framework for testing claims about parameters or relationships in data through a systematic process of statistical inference.

### What is Hypothesis Testing?

Hypothesis testing is a statistical method used to determine whether there is enough evidence in a sample to support a particular claim about a population parameter. It involves making an initial assumption (the null hypothesis) and then determining whether the observed data provides sufficient evidence to reject this assumption.

### The Scientific Method and Hypothesis Testing

Hypothesis testing follows the scientific method:
1. **Observation**: Collect data from a sample
2. **Hypothesis**: Formulate null and alternative hypotheses
3. **Prediction**: Determine what we expect to observe if the null hypothesis is true
4. **Testing**: Calculate a test statistic and compare it to a critical value
5. **Conclusion**: Make a decision based on the evidence

### Key Components of Hypothesis Testing

1. **Null Hypothesis ($H_0$)**: The default assumption about the population parameter
2. **Alternative Hypothesis ($H_1$ or $H_a$)**: The claim we want to support
3. **Test Statistic**: A calculated value that measures the strength of evidence
4. **Significance Level ($\alpha$)**: The probability of rejecting $H_0$ when it's true
5. **P-value**: The probability of observing the test statistic or more extreme values
6. **Decision Rule**: Criteria for rejecting or failing to reject $H_0$

### Mathematical Foundation

The general framework for hypothesis testing involves:

```math
\text{Test Statistic} = \frac{\text{Sample Statistic} - \text{Parameter under } H_0}{\text{Standard Error}}
```

The decision rule is:
- Reject $H_0$ if $|T| > t_{\alpha/2, df}$ (two-tailed test)
- Reject $H_0$ if $T > t_{\alpha, df}$ (one-tailed test, upper)
- Reject $H_0$ if $T < -t_{\alpha, df}$ (one-tailed test, lower)

Where $T$ is the test statistic and $t_{\alpha, df}$ is the critical value from the appropriate distribution.

## Basic Concepts

### Null and Alternative Hypotheses

The foundation of hypothesis testing lies in formulating two competing hypotheses about a population parameter.

#### Mathematical Foundation

**Null Hypothesis ($H_0$)**: The default assumption that there is no effect, no difference, or no relationship. It represents the status quo or the claim we want to test against.

**Alternative Hypothesis ($H_1$ or $H_a$)**: The claim we want to support. It represents the research hypothesis or the effect we're trying to detect.

For a population parameter $\theta$, the hypotheses can be:

**Two-tailed test:**
```math
H_0: \theta = \theta_0 \quad \text{vs.} \quad H_1: \theta \neq \theta_0
```

**One-tailed test (upper):**
```math
H_0: \theta \leq \theta_0 \quad \text{vs.} \quad H_1: \theta > \theta_0
```

**One-tailed test (lower):**
```math
H_0: \theta \geq \theta_0 \quad \text{vs.} \quad H_1: \theta < \theta_0
```

#### Python Example

See `coin_fairness_test` in `11_hypothesis_testing.py` for a function that demonstrates basic hypothesis testing concepts using a coin fairness test. This function shows how to formulate null and alternative hypotheses, calculate test statistics, and make decisions based on p-values.

---

### Type I and Type II Errors

Understanding the types of errors that can occur in hypothesis testing is crucial for interpreting results and making informed decisions.

#### Mathematical Foundation

**Type I Error ($\alpha$)**: The probability of rejecting $H_0$ when it's actually true.
```math
\alpha = P(\text{Reject } H_0 | H_0 \text{ is true})
```

**Type II Error ($\beta$)**: The probability of failing to reject $H_0$ when it's actually false.
```math
\beta = P(\text{Fail to reject } H_0 | H_1 \text{ is true})
```

**Power ($1 - \beta$)**: The probability of correctly rejecting $H_0$ when it's false.
```math
\text{Power} = 1 - \beta = P(\text{Reject } H_0 | H_1 \text{ is true})
```

The relationship between these concepts:
```math
\alpha + \text{Power} \leq 1
```

#### Decision Matrix

| Decision | $H_0$ True | $H_0$ False |
|----------|-------------|-------------|
| Reject $H_0$ | Type I Error ($\alpha$) | Correct Decision (Power) |
| Fail to reject $H_0$ | Correct Decision ($1-\alpha$) | Type II Error ($\beta$) |

#### Python Example

See `calculate_power` and `power_analysis_demo` in `11_hypothesis_testing.py` for functions that demonstrate power analysis and the trade-off between Type I and Type II errors. These functions show how power varies with effect size, sample size, and significance level.

---

## One-Sample Tests

One-sample tests are used to compare a sample statistic to a hypothesized population parameter.

### One-Sample t-Test

The one-sample t-test is used to determine whether the mean of a single sample differs from a hypothesized population mean.

#### Mathematical Foundation

For a sample of size $n$ with mean $\bar{x}$ and standard deviation $s$, the test statistic is:

```math
t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}
```

Where $\mu_0$ is the hypothesized population mean under $H_0$.

The test statistic follows a t-distribution with $n-1$ degrees of freedom.

**Hypotheses:**
```math
H_0: \mu = \mu_0 \quad \text{vs.} \quad H_1: \mu \neq \mu_0
```

#### Assumptions

1. **Independence**: Observations are independent
2. **Normality**: The population is normally distributed (or sample size is large)
3. **Random Sampling**: Data comes from a random sample

#### Python Example

See `one_sample_t_test` in `11_hypothesis_testing.py` for a function that performs a one-sample t-test with assumption checking, effect size calculation, and confidence interval estimation.

---

### One-Sample Proportion Test

The one-sample proportion test is used to determine whether a sample proportion differs from a hypothesized population proportion.

#### Mathematical Foundation

For a sample of size $n$ with $x$ successes, the sample proportion is $\hat{p} = x/n$.

**Large Sample Approximation (Normal):**
When $np \geq 10$ and $n(1-p) \geq 10$, the test statistic is:

```math
z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}
```

Where $p_0$ is the hypothesized population proportion under $H_0$.

**Exact Test (Binomial):**
For small samples, use the exact binomial test:

```math
P(X \geq x) = \sum_{k=x}^n \binom{n}{k} p_0^k (1-p_0)^{n-k}
```

**Hypotheses:**
```math
H_0: p = p_0 \quad \text{vs.} \quad H_1: p \neq p_0
```

#### Python Example

See `one_sample_proportion_test` in `11_hypothesis_testing.py` for a function that performs both exact binomial and normal approximation tests for proportions, with large sample condition checking.

---

### One-Sample Variance Test

The one-sample variance test (chi-square test for variance) is used to determine whether a sample variance differs from a hypothesized population variance.

#### Mathematical Foundation

For a sample of size $n$ with variance $s^2$, the test statistic is:

```math
\chi^2 = \frac{(n-1)s^2}{\sigma_0^2}
```

Where $\sigma_0^2$ is the hypothesized population variance under $H_0$.

The test statistic follows a chi-square distribution with $n-1$ degrees of freedom.

**Hypotheses:**
```math
H_0: \sigma^2 = \sigma_0^2 \quad \text{vs.} \quad H_1: \sigma^2 \neq \sigma_0^2
```

**Assumptions:**
1. **Normality**: The population is normally distributed
2. **Independence**: Observations are independent
3. **Random Sampling**: Data comes from a random sample

#### Python Example

See `one_sample_variance_test` in `11_hypothesis_testing.py` for a function that performs a chi-square test for variance with confidence interval calculation and effect size estimation.

---

## Two-Sample Tests

Two-sample tests are used to compare parameters between two independent or paired groups.

### Independent t-Test

The independent t-test (two-sample t-test) is used to compare the means of two independent groups.

#### Mathematical Foundation

For two independent samples with means $\bar{x}_1$ and $\bar{x}_2$, the test statistic is:

**Equal Variances (Pooled):**
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
```

Where the pooled standard deviation is:
```math
s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
```

**Unequal Variances (Welch's):**
```math
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
```

With degrees of freedom (Welch's approximation):
```math
df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
```

**Hypotheses:**
```math
H_0: \mu_1 = \mu_2 \quad \text{vs.} \quad H_1: \mu_1 \neq \mu_2
```

#### Assumptions

1. **Independence**: Observations are independent within and between groups
2. **Normality**: Both populations are normally distributed (or sample sizes are large)
3. **Equal Variances**: Population variances are equal (for pooled t-test)
4. **Random Sampling**: Data comes from random samples

#### Python Example

See `independent_t_test` in `11_hypothesis_testing.py` for a function that performs both pooled and Welch's t-tests with comprehensive assumption checking, effect size calculation, and confidence interval estimation.

---

### Paired t-Test

The paired t-test is used to compare the means of two related groups, such as before/after measurements on the same subjects.

#### Mathematical Foundation

For paired observations $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$, we work with the differences $d_i = x_i - y_i$.

The test statistic is:
```math
t = \frac{\bar{d}}{s_d/\sqrt{n}}
```

Where:
- $\bar{d}$ is the mean of the differences
- $s_d$ is the standard deviation of the differences
- $n$ is the number of pairs

The test statistic follows a t-distribution with $n-1$ degrees of freedom.

**Hypotheses:**
```math
H_0: \mu_d = 0 \quad \text{vs.} \quad H_1: \mu_d \neq 0
```

#### Advantages of Paired Tests

1. **Reduced Variability**: Removes between-subject variability
2. **Increased Power**: More sensitive to detect treatment effects
3. **Control for Confounding**: Each subject serves as their own control

#### Python Example

See `paired_t_test` in `11_hypothesis_testing.py` for a function that performs a paired t-test with effect size calculation, normality checking, and correlation analysis between paired measurements.

---

### Two-Sample Proportion Test

The two-sample proportion test is used to compare proportions between two independent groups.

#### Mathematical Foundation

For two independent samples with proportions $\hat{p}_1$ and $\hat{p}_2$, the test statistic is:

**Large Sample Approximation:**
```math
z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}
```

Where the pooled proportion is:
```math
\hat{p} = \frac{x_1 + x_2}{n_1 + n_2}
```

**Hypotheses:**
```math
H_0: p_1 = p_2 \quad \text{vs.} \quad H_1: p_1 \neq p_2
```

#### Assumptions

1. **Independence**: Observations are independent within and between groups
2. **Large Sample**: $n_1p_1 \geq 10$, $n_1(1-p_1) \geq 10$, $n_2p_2 \geq 10$, $n_2(1-p_2) \geq 10$
3. **Random Sampling**: Data comes from random samples

#### Python Example

See `two_sample_proportion_test` in `11_hypothesis_testing.py` for a function that performs chi-square tests for proportions with large sample condition checking, confidence intervals, and effect size measures (risk ratio, odds ratio).

---

## Nonparametric Tests

Nonparametric tests make fewer assumptions about the underlying population distribution and are often used when parametric assumptions are violated.

### Wilcoxon Rank-Sum Test (Mann-Whitney U)

The Wilcoxon rank-sum test is a nonparametric alternative to the independent t-test that tests whether two independent samples come from populations with the same distribution.

#### Mathematical Foundation

The test statistic is based on the sum of ranks. For two samples of sizes $n_1$ and $n_2$:

```math
W = \sum_{i=1}^{n_1} R_i
```

Where $R_i$ are the ranks of the first sample when all observations are ranked together.

The test statistic follows approximately a normal distribution:
```math
Z = \frac{W - \mu_W}{\sigma_W}
```

Where:
```math
\mu_W = \frac{n_1(n_1 + n_2 + 1)}{2}
```

```math
\sigma_W = \sqrt{\frac{n_1 n_2(n_1 + n_2 + 1)}{12}}
```

**Hypotheses:**
```math
H_0: \text{The two populations have the same distribution}
```

```math
H_1: \text{The two populations have different distributions}
```

#### Assumptions

1. **Independence**: Observations are independent within and between groups
2. **Random Sampling**: Data comes from random samples
3. **Ordinal Data**: Data can be ranked (no normality assumption)

#### Python Example

See `wilcoxon_rank_sum_test` in `11_hypothesis_testing.py` for a function that performs the Wilcoxon rank-sum test with comparison to parametric t-tests and rank-biserial correlation effect size calculation.

---

### Wilcoxon Signed-Rank Test

The Wilcoxon signed-rank test is a nonparametric alternative to the paired t-test that tests whether the median of the differences is zero.

#### Mathematical Foundation

For paired differences $d_i = x_i - y_i$, the test statistic is:

```math
W = \sum_{i=1}^{n} \text{sgn}(d_i) R_i
```

Where:
- $\text{sgn}(d_i)$ is the sign of the difference
- $R_i$ is the rank of $|d_i|$ when all absolute differences are ranked

The test statistic follows approximately a normal distribution:
```math
Z = \frac{W}{\sqrt{\frac{n(n+1)(2n+1)}{6}}}
```

**Hypotheses:**
```math
H_0: \text{The median of differences is zero}
```

```math
H_1: \text{The median of differences is not zero}
```

#### Python Example

See `wilcoxon_signed_rank_test` in `11_hypothesis_testing.py` for a function that performs the Wilcoxon signed-rank test with comparison to parametric paired t-tests and rank-biserial correlation effect size calculation.

---

### Kruskal-Wallis Test

The Kruskal-Wallis test is a nonparametric alternative to one-way ANOVA that tests whether multiple independent groups have the same distribution.

#### Mathematical Foundation

For $k$ groups with sample sizes $n_1, n_2, \ldots, n_k$, the test statistic is:

```math
H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)
```

Where:
- $N = n_1 + n_2 + \ldots + n_k$ is the total sample size
- $R_i$ is the sum of ranks for group $i$

The test statistic follows approximately a chi-square distribution with $k-1$ degrees of freedom.

**Hypotheses:**
```math
H_0: \text{All groups have the same distribution}
```

```math
H_1: \text{At least one group has a different distribution}
```

#### Python Example

See `kruskal_wallis_test` in `11_hypothesis_testing.py` for a function that performs the Kruskal-Wallis test with epsilon-squared effect size calculation, comparison to parametric ANOVA, and post-hoc pairwise Mann-Whitney U tests with Bonferroni correction.

---

## Multiple Comparison Tests

### One-Way ANOVA

See `one_way_anova` in `11_hypothesis_testing.py` for a function that performs one-way ANOVA with Tukey's HSD post-hoc tests.

---

### Chi-Square Test

See `chi_square_test` in `11_hypothesis_testing.py` for a function that performs chi-square tests of independence with contingency table analysis.

---

## Effect Size

### Cohen's d

See `cohens_d_two_sample` in `11_hypothesis_testing.py` for a function that calculates Cohen's d effect size for two independent samples.

---

## Power Analysis

See `power_analysis_t_test` in `11_hypothesis_testing.py` for a function that performs power analysis for t-tests, calculating required sample sizes and power for different scenarios.

---

## Multiple Testing Correction

### Bonferroni Correction

See `multiple_testing_correction` in `11_hypothesis_testing.py` for a function that applies multiple testing corrections including Bonferroni and False Discovery Rate methods.

---

## Best Practices

### Assumption Checking

See `check_t_test_assumptions` in `11_hypothesis_testing.py` for a function that checks normality and variance assumptions for t-tests.

---

### Reporting Results

See `format_test_results` in `11_hypothesis_testing.py` for a function that formats test results for proper reporting with significance levels.

---

## Practical Examples

See `practical_examples` in `11_hypothesis_testing.py` for functions that demonstrate real-world applications including drug efficacy studies, customer satisfaction surveys, and A/B testing.

---

## Common Mistakes to Avoid

1. **Multiple testing without correction**: Always use appropriate correction methods when conducting multiple tests
2. **Ignoring effect size**: Report both p-values and effect sizes for complete interpretation
3. **Data dredging**: Pre-specify hypotheses and analysis plans
4. **Ignoring assumptions**: Always verify test assumptions before conducting analyses

---

## Exercises

1. **Basic Hypothesis Testing**: Test if the mean of a dataset differs from a hypothesized value using the one-sample t-test functions.
2. **Two-Sample Comparison**: Compare two groups using both parametric and nonparametric tests.
3. **Multiple Testing**: Perform multiple tests and apply correction methods.
4. **Power Analysis**: Calculate required sample sizes for different effect sizes and power levels.
5. **Real-world Application**: Design and conduct hypothesis tests for practical scenarios.

Refer to the corresponding functions in `11_hypothesis_testing.py` for implementation guidance.

---

## Next Steps

In the next chapter, we'll learn about confidence intervals and their relationship with hypothesis testing.

---

**Key Takeaways:**
- Always state null and alternative hypotheses clearly
- Choose appropriate tests based on data type and assumptions
- Report both p-values and effect sizes
- Check assumptions before conducting tests
- Use correction methods for multiple testing
- Consider power analysis for study design
- Interpret results in context of practical significance
- Avoid common pitfalls like data dredging 