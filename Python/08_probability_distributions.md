# Probability Distributions

## Overview

Probability distributions are fundamental to statistical analysis and data science. They describe how probabilities are distributed over the possible values of a random variable, providing the mathematical foundation for statistical inference, hypothesis testing, and predictive modeling.

### The Role of Probability Distributions

Probability distributions serve several critical functions:

1. **Modeling Random Phenomena**: Describe the behavior of random variables in real-world scenarios
2. **Statistical Inference**: Provide the basis for confidence intervals and hypothesis tests
3. **Risk Assessment**: Quantify uncertainty in decision-making processes
4. **Predictive Modeling**: Enable forecasting and simulation studies

### Key Concepts

- **Random Variable**: A variable whose values depend on outcomes of a random phenomenon
- **Probability Mass Function (PMF)**: For discrete variables, gives $P(X = x)$
- **Probability Density Function (PDF)**: For continuous variables, gives the density at point $x$
- **Cumulative Distribution Function (CDF)**: Gives $P(X \leq x)$ for any value $x$

## Mathematical Foundations

### Probability Axioms

The foundation of probability theory rests on three axioms:

```math
\begin{align}
1. & \quad P(A) \geq 0 \text{ for any event } A \\
2. & \quad P(\Omega) = 1 \text{ where } \Omega \text{ is the sample space} \\
3. & \quad P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i) \text{ for disjoint events}
\end{align}
```

### Expected Value and Variance

For any random variable $X$ with PDF $f(x)$ or PMF $p(x)$:

```math
\begin{align}
\text{Expected Value:} & \quad E[X] = \int_{-\infty}^{\infty} x f(x) dx \text{ (continuous)} \\
& \quad E[X] = \sum_{x} x p(x) \text{ (discrete)} \\
\text{Variance:} & \quad \text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
\end{align}
```

## Normal Distribution

The normal (Gaussian) distribution is the most important continuous distribution in statistics, characterized by its bell-shaped curve and mathematical properties.

### Mathematical Definition

The normal distribution with mean $\mu$ and standard deviation $\sigma$ has PDF:

```math
f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
```

### Properties of Normal Distribution

1. **Symmetry**: The distribution is symmetric around the mean
2. **68-95-99.7 Rule**: Approximately 68%, 95%, and 99.7% of data fall within 1, 2, and 3 standard deviations of the mean
3. **Central Limit Theorem**: Sums of independent random variables approach normality
4. **Maximum Entropy**: Among all distributions with given mean and variance, normal has maximum entropy

### Basic Normal Distribution Functions

**Python Code Reference:** `basic_normal_distribution()` in `08_probability_distributions.py`

This function demonstrates:
- Generating normal random variables with mathematical precision
- Calculating and plotting the normal density function
- Adding mathematical annotations showing mean (μ) and standard deviation (σ)
- Visualizing the standard normal distribution N(0,1) with proper labeling

### Normal Distribution Properties

**Python Code Reference:** `normal_distribution_properties()` in `08_probability_distributions.py`

This function demonstrates:
- Calculating cumulative probabilities using the standard normal CDF Φ(x)
- Computing quantiles using the inverse CDF Φ⁻¹(p)
- Verifying the 68-95-99.7 rule for normal distributions
- Calculating interval probabilities using CDF differences

### Checking Normality

**Python Code Reference:** `checking_normality()` in `08_probability_distributions.py`

This function demonstrates:
- Comprehensive normality assessment with multiple diagnostic plots
- Histogram with normal overlay comparison
- Q-Q plot for normality testing
- Kernel density estimation vs theoretical normal PDF
- Box plot for symmetry assessment
- Statistical tests: Shapiro-Wilk and Kolmogorov-Smirnov tests
- Interpretation of normality test results

## Binomial Distribution

The binomial distribution models the number of successes in a fixed number of independent Bernoulli trials, where each trial has the same probability of success.

### Mathematical Definition

For $n$ independent trials with success probability $p$, the probability of $k$ successes is:

```math
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
```

Where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the binomial coefficient.

### Properties

- **Mean**: $E[X] = np$
- **Variance**: $\text{Var}(X) = np(1-p)$
- **Range**: $X \in \{0, 1, 2, \ldots, n\}$

### Basic Binomial Functions

**Python Code Reference:** `basic_binomial_functions()` in `08_probability_distributions.py`

This function demonstrates:
- Setting up binomial distribution parameters (n trials, p success probability)
- Generating binomial random variables
- Calculating exact probabilities using PMF and CDF
- Verifying mathematical properties: E[X] = np, Var(X) = np(1-p)
- Plotting binomial distribution with probability mass function
- Adding mathematical annotations showing expected value

### Binomial Distribution Examples

**Python Code Reference:** `binomial_distribution_examples()` in `08_probability_distributions.py`

This function demonstrates:
- Coin flipping example: calculating probability of exactly 7 heads in 10 flips
- Using binomial formula: P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
- Calculating cumulative probabilities for quality control scenarios
- Computing expected values using E[X] = np formula
- Practical applications in quality control and defect analysis

## Poisson Distribution

The Poisson distribution models the number of events occurring in a fixed interval, where events occur independently at a constant average rate.

### Mathematical Definition

For events occurring at rate $\lambda$ per unit time/space:

```math
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
```

### Properties

- **Mean**: $E[X] = \lambda$
- **Variance**: $\text{Var}(X) = \lambda$ (equal to mean)
- **Range**: $X \in \{0, 1, 2, \ldots\}$

### Basic Poisson Functions

**Python Code Reference:** `basic_poisson_functions()` in `08_probability_distributions.py`

This function demonstrates:
- Setting up Poisson distribution with rate parameter λ
- Generating Poisson random variables
- Calculating exact probabilities using PMF and CDF
- Verifying mathematical properties: E[X] = Var(X) = λ
- Plotting Poisson distribution with probability mass function
- Adding mathematical annotations showing mean equals variance

### Poisson Distribution Examples

**Python Code Reference:** `poisson_distribution_examples()` in `08_probability_distributions.py`

This function demonstrates:
- Customer arrivals example: calculating probability of exactly 3 customers per hour
- Using Poisson formula: P(X = k) = (λ^k × e^(-λ))/k!
- Calculating cumulative probabilities for customer service scenarios
- Manufacturing defects example with practical interpretation
- Computing probabilities for defect-free and limited defect scenarios

## Exponential Distribution

The exponential distribution models the time between events in a Poisson process, characterized by the memoryless property.

### Mathematical Definition

For rate parameter $\lambda$, the PDF is:

```math
f(x) = \lambda e^{-\lambda x} \text{ for } x \geq 0
```

### Properties

- **Mean**: $E[X] = \frac{1}{\lambda}$
- **Variance**: $\text{Var}(X) = \frac{1}{\lambda^2}$
- **Memoryless Property**: $P(X > s+t | X > s) = P(X > t)$

### Basic Exponential Functions

**Python Code Reference:** `basic_exponential_functions()` in `08_probability_distributions.py`

This function demonstrates:
- Setting up exponential distribution with rate parameter λ
- Generating exponential random variables
- Calculating probabilities using CDF
- Verifying mathematical properties: E[X] = 1/λ, Var(X) = 1/λ²
- Plotting exponential distribution with probability density function
- Adding mathematical annotations showing expected value

## Chi-Square Distribution

The chi-square distribution is important for variance testing and goodness-of-fit tests, arising from sums of squared standard normal variables.

### Mathematical Definition

If $Z_1, Z_2, \ldots, Z_k$ are independent standard normal variables, then:

```math
X = \sum_{i=1}^{k} Z_i^2 \sim \chi^2(k)
```

### Properties

- **Mean**: $E[X] = k$ (degrees of freedom)
- **Variance**: $\text{Var}(X) = 2k$
- **Range**: $X \geq 0$

### Basic Chi-Square Functions

**Python Code Reference:** `basic_chi_square_functions()` in `08_probability_distributions.py`

This function demonstrates:
- Plotting chi-square distributions with different degrees of freedom
- Calculating critical values for hypothesis testing
- Verifying mathematical properties: E[X] = k, Var(X) = 2k
- Visualizing how shape changes with degrees of freedom
- Computing critical values for significance level α = 0.05

## t-Distribution

The t-distribution is used for small sample inference and approaches the normal distribution as degrees of freedom increase.

### Mathematical Definition

If $Z \sim N(0,1)$ and $V \sim \chi^2(k)$ are independent, then:

```math
T = \frac{Z}{\sqrt{V/k}} \sim t(k)
```

### Properties

- **Mean**: $E[T] = 0$ (for $k > 1$)
- **Variance**: $\text{Var}(T) = \frac{k}{k-2}$ (for $k > 2$)
- **Heavier tails** than normal distribution

### Basic t-Distribution Functions

**Python Code Reference:** `basic_t_distribution_functions()` in `08_probability_distributions.py`

This function demonstrates:
- Plotting t-distributions with different degrees of freedom
- Comparing t-distributions to normal distribution
- Calculating two-tailed critical values for hypothesis testing
- Showing how t-distribution approaches normal as df increases
- Computing critical values for significance level α = 0.05

## F-Distribution

The F-distribution is used for comparing variances and in ANOVA, arising from ratios of chi-square variables.

### Mathematical Definition

If $U \sim \chi^2(k_1)$ and $V \sim \chi^2(k_2)$ are independent, then:

```math
F = \frac{U/k_1}{V/k_2} \sim F(k_1, k_2)
```

### Properties

- **Mean**: $E[F] = \frac{k_2}{k_2-2}$ (for $k_2 > 2$)
- **Variance**: Complex function of $k_1$ and $k_2$
- **Range**: $F \geq 0$

### Basic F-Distribution Functions

**Python Code Reference:** `basic_f_distribution_functions()` in `08_probability_distributions.py`

This function demonstrates:
- Plotting F-distributions with different degrees of freedom pairs
- Calculating critical values for variance ratio tests
- Visualizing F-distribution shapes for different parameter combinations
- Computing critical values for significance level α = 0.05
- Understanding F-distribution as ratio of chi-square variables

## Distribution Fitting

### Fitting Distributions to Data

**Python Code Reference:** `distribution_fitting()` in `08_probability_distributions.py`

This function demonstrates:
- Loading real data (mtcars MPG) for distribution fitting
- Fitting normal distribution to data with maximum likelihood estimation
- Plotting histogram with fitted normal distribution overlay
- Adding mathematical annotations showing fitted parameters
- Performing Kolmogorov-Smirnov goodness-of-fit test
- Interpreting fit quality based on p-values

### Comparing Multiple Distributions

**Python Code Reference:** `comparing_multiple_distributions()` in `08_probability_distributions.py`

This function demonstrates:
- Fitting multiple distributions (normal, exponential, gamma) to the same data
- Computing goodness-of-fit statistics for each distribution
- Calculating Akaike Information Criterion (AIC) for model comparison
- Comparing Kolmogorov-Smirnov test results across distributions
- Identifying the best-fitting distribution based on statistical criteria
- Providing comprehensive interpretation of fit quality

## Practical Examples

### Example 1: Quality Control with Mathematical Precision

**Python Code Reference:** `practical_example_quality_control()` in `08_probability_distributions.py`

This function demonstrates:
- Simulating defect data using Poisson distribution with λ = 2.5
- Calculating exact probabilities for different defect counts
- Plotting histogram with theoretical Poisson PMF overlay
- Adding mathematical annotations showing E[X] = Var(X) = λ
- Practical application in manufacturing quality control
- Interpreting probabilities for decision-making in quality management

### Example 2: Customer Service Times with Exponential Model

**Python Code Reference:** `practical_example_customer_service()` in `08_probability_distributions.py`

This function demonstrates:
- Simulating customer service times using exponential distribution
- Calculating probabilities for different time intervals
- Plotting histogram with theoretical exponential PDF overlay
- Adding mathematical annotations showing E[X] = 1/λ
- Practical application in service operations management
- Using exponential distribution for time-between-events modeling

### Example 3: Investment Returns with Normal Model

**Python Code Reference:** `practical_example_investment_returns()` in `08_probability_distributions.py`

This function demonstrates:
- Simulating daily investment returns using normal distribution
- Calculating probabilities for positive returns and extreme events
- Computing Value at Risk (VaR) for risk management
- Plotting return distribution with theoretical normal PDF overlay
- Adding mathematical annotations showing distribution parameters
- Practical application in financial risk assessment and portfolio management

## Best Practices

### Choosing the Right Distribution

**Python Code Reference:** `best_practices_distribution_selection()` in `08_probability_distributions.py`

This function provides comprehensive guidelines for distribution selection, including:
- Normal distribution: symmetric, continuous data (heights, weights, measurement errors)
- Binomial distribution: counts of successes in fixed trials (coin flips, defect counts)
- Poisson distribution: counts of rare events (customer arrivals, accidents)
- Exponential distribution: time between events (service times, equipment lifetimes)
- Chi-square distribution: variance testing, goodness-of-fit
- t-distribution: small sample inference (confidence intervals, hypothesis tests)
- F-distribution: variance comparisons, ANOVA

### Distribution Validation

**Python Code Reference:** `validate_distribution()` in `08_probability_distributions.py`

This function demonstrates comprehensive distribution validation including:
- Visual checks: histogram with fitted curve, Q-Q plots, empirical vs theoretical CDF, P-P plots
- Statistical tests: Kolmogorov-Smirnov and Shapiro-Wilk tests for normality
- Support for normal, exponential, and Poisson distributions
- Comprehensive diagnostic plots for distribution assessment
- Statistical interpretation of fit quality

## Exercises

### Exercise 1: Normal Distribution
Generate 1000 random numbers from a normal distribution with mean 50 and standard deviation 10. Calculate P(X < 45) and P(X > 60). Verify the 68-95-99.7 rule.

### Exercise 2: Binomial Distribution
Calculate the probability of getting exactly 8 heads in 15 coin flips. Also calculate the probability of getting 8 or more heads. Verify that E[X] = np and Var(X) = np(1-p).

### Exercise 3: Poisson Distribution
If customers arrive at a rate of 3 per hour, calculate the probability of exactly 5 customers arriving in 2 hours. Verify that E[X] = Var(X) = λ.

### Exercise 4: Distribution Fitting
Fit different distributions to the iris sepal length data and determine which fits best using both visual and statistical methods.

### Exercise 5: Real-world Application
Find a real dataset and identify which probability distribution best describes it. Validate your choice using appropriate tests.

### Exercise 6: Mathematical Properties
For each distribution covered, verify the mathematical properties (mean, variance) using both theoretical formulas and simulation.

## Next Steps

In the next chapter, we'll learn about sampling and sampling distributions, which build upon our understanding of probability distributions and provide the foundation for statistical inference.

---

**Key Takeaways:**
- Normal distribution is fundamental for statistical inference and the Central Limit Theorem
- Choose distributions based on data characteristics and underlying processes
- Always validate distribution assumptions using both visual and statistical methods
- Understand the mathematical properties and parameters of each distribution
- Consider the context and real-world interpretation when selecting distributions
- Use appropriate goodness-of-fit tests to validate distribution choices
- Mathematical foundations provide insight into when and why distributions are appropriate 