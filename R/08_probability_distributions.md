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

```r
# Generate normal random variables with mathematical precision
set.seed(123)
normal_data <- rnorm(1000, mean = 0, sd = 1)

# Density function calculation
x <- seq(-4, 4, by = 0.1)
normal_density <- dnorm(x, mean = 0, sd = 1)

# Plot normal distribution with mathematical annotations
plot(x, normal_density, type = "l", 
     main = "Standard Normal Distribution N(0,1)",
     xlab = "x", ylab = "Density f(x)",
     col = "blue", lwd = 2)
abline(v = 0, col = "red", lty = 2, lwd = 2)
abline(v = c(-1, 1), col = "green", lty = 3, lwd = 1)
abline(v = c(-2, 2), col = "orange", lty = 3, lwd = 1)

# Add mathematical annotations
text(0, 0.3, expression(mu == 0), col = "red", font = 2)
text(1, 0.2, expression(sigma == 1), col = "blue", font = 2)
```

### Normal Distribution Properties

```r
# Calculate probabilities with mathematical interpretation
cat("P(X < 0) = Φ(0):", pnorm(0), "\n")
cat("P(X < 1) = Φ(1):", pnorm(1), "\n")
cat("P(X < 2) = Φ(2):", pnorm(2), "\n")

# Calculate quantiles (inverse CDF)
cat("95th percentile: Φ⁻¹(0.95) =", qnorm(0.95), "\n")
cat("99th percentile: Φ⁻¹(0.99) =", qnorm(0.99), "\n")

# Calculate probabilities for intervals
cat("P(-1 < X < 1) = Φ(1) - Φ(-1):", pnorm(1) - pnorm(-1), "\n")
cat("P(-2 < X < 2) = Φ(2) - Φ(-2):", pnorm(2) - pnorm(-2), "\n")

# Verify 68-95-99.7 rule
cat("P(-1 < X < 1):", pnorm(1) - pnorm(-1), "(≈ 68%)\n")
cat("P(-2 < X < 2):", pnorm(2) - pnorm(-2), "(≈ 95%)\n")
cat("P(-3 < X < 3):", pnorm(3) - pnorm(-3), "(≈ 99.7%)\n")
```

### Checking Normality

```r
# Comprehensive normality assessment
par(mfrow = c(2, 2))

# Histogram with normal overlay
hist(normal_data, main = "Histogram with Normal Overlay", 
     xlab = "Value", col = "lightblue", freq = FALSE)
curve(dnorm(x, mean = 0, sd = 1), add = TRUE, col = "red", lwd = 2)

# Q-Q plot for normality
qqnorm(normal_data, main = "Q-Q Plot for Normality")
qqline(normal_data, col = "red", lwd = 2)

# Density plot comparison
plot(density(normal_data), main = "Density Plot vs Normal")
curve(dnorm(x, mean = 0, sd = 1), add = TRUE, col = "red", lty = 2, lwd = 2)

# Box plot for symmetry
boxplot(normal_data, main = "Box Plot (Check for Symmetry)")

par(mfrow = c(1, 1))

# Statistical normality tests with interpretation
shapiro_test <- shapiro.test(normal_data)
cat("Shapiro-Wilk Test:\n")
cat("  W =", shapiro_test$statistic, "\n")
cat("  p-value =", shapiro_test$p.value, "\n")
cat("  Interpretation:", ifelse(shapiro_test$p.value > 0.05, 
                               "Data appears normal", "Data may not be normal"), "\n\n")

# Kolmogorov-Smirnov test
ks_test <- ks.test(normal_data, "pnorm")
cat("Kolmogorov-Smirnov Test:\n")
cat("  D =", ks_test$statistic, "\n")
cat("  p-value =", ks_test$p.value, "\n")
cat("  Interpretation:", ifelse(ks_test$p.value > 0.05, 
                               "Data appears normal", "Data may not be normal"), "\n")
```

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

```r
# Binomial distribution parameters with mathematical interpretation
n <- 10  # number of trials (fixed)
p <- 0.5  # probability of success (constant across trials)

# Generate binomial random variables
set.seed(123)
binomial_data <- rbinom(1000, size = n, prob = p)

# Calculate exact probabilities
cat("P(X = 5):", dbinom(5, size = n, prob = p), "\n")
cat("P(X ≤ 5):", pbinom(5, size = n, prob = p), "\n")
cat("P(X ≥ 5):", 1 - pbinom(4, size = n, prob = p), "\n")

# Verify mathematical properties
cat("Theoretical mean (np):", n * p, "\n")
cat("Sample mean:", mean(binomial_data), "\n")
cat("Theoretical variance (np(1-p)):", n * p * (1-p), "\n")
cat("Sample variance:", var(binomial_data), "\n")

# Plot binomial distribution with mathematical annotations
x <- 0:n
binomial_probs <- dbinom(x, size = n, prob = p)

plot(x, binomial_probs, type = "h", 
     main = paste("Binomial Distribution B(", n, ",", p, ")"),
     xlab = "Number of Successes (k)", ylab = "P(X = k)",
     col = "blue", lwd = 2)
points(x, binomial_probs, col = "red", pch = 16)

# Add mathematical annotations
text(n/2, max(binomial_probs), 
     expression(paste("E[X] = ", n, "p = ", n*p)), 
     pos = 3, col = "darkgreen", font = 2)
```

### Binomial Distribution Examples

```r
# Example 1: Coin flipping with mathematical interpretation
# Probability of getting exactly 7 heads in 10 flips
p_7_heads <- dbinom(7, size = 10, prob = 0.5)
cat("P(X = 7) = C(10,7) × 0.5⁷ × 0.5³ =", p_7_heads, "\n")

# Probability of getting 7 or more heads
p_7_or_more <- 1 - pbinom(6, size = 10, prob = 0.5)
cat("P(X ≥ 7) = 1 - P(X ≤ 6) =", p_7_or_more, "\n")

# Example 2: Quality control with practical interpretation
# Probability of 2 or fewer defective items in 20 (10% defect rate)
p_defective <- dbinom(0:2, size = 20, prob = 0.1)
cat("P(X = 0):", p_defective[1], "\n")
cat("P(X = 1):", p_defective[2], "\n")
cat("P(X = 2):", p_defective[3], "\n")
cat("P(X ≤ 2):", sum(p_defective), "\n")

# Expected number of defects
cat("Expected defects (np):", 20 * 0.1, "\n")
```

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

```r
# Poisson distribution parameter with mathematical interpretation
lambda <- 3  # average number of events per unit

# Generate Poisson random variables
set.seed(123)
poisson_data <- rpois(1000, lambda = lambda)

# Calculate exact probabilities
cat("P(X = 3):", dpois(3, lambda = lambda), "\n")
cat("P(X ≤ 3):", ppois(3, lambda = lambda), "\n")
cat("P(X ≥ 3):", 1 - ppois(2, lambda = lambda), "\n")

# Verify mathematical properties
cat("Theoretical mean (λ):", lambda, "\n")
cat("Sample mean:", mean(poisson_data), "\n")
cat("Theoretical variance (λ):", lambda, "\n")
cat("Sample variance:", var(poisson_data), "\n")

# Plot Poisson distribution with mathematical annotations
x <- 0:10
poisson_probs <- dpois(x, lambda = lambda)

plot(x, poisson_probs, type = "h", 
     main = paste("Poisson Distribution Poi(", lambda, ")"),
     xlab = "Number of Events (k)", ylab = "P(X = k)",
     col = "green", lwd = 2)
points(x, poisson_probs, col = "red", pch = 16)

# Add mathematical annotations
text(lambda, max(poisson_probs), 
     expression(paste("E[X] = Var(X) = ", lambda)), 
     pos = 3, col = "darkgreen", font = 2)
```

### Poisson Distribution Examples

```r
# Example 1: Customer arrivals with practical interpretation
# Average 5 customers per hour (λ = 5)
lambda_customers <- 5

# Probability of exactly 3 customers in an hour
p_3_customers <- dpois(3, lambda = lambda_customers)
cat("P(X = 3) = (5³ × e⁻⁵)/3! =", p_3_customers, "\n")

# Probability of 3 or fewer customers
p_3_or_fewer <- ppois(3, lambda = lambda_customers)
cat("P(X ≤ 3) =", p_3_or_fewer, "\n")

# Example 2: Defects in manufacturing
# Average 2 defects per 100 items (λ = 2)
lambda_defects <- 2

# Probability of no defects
p_no_defects <- dpois(0, lambda = lambda_defects)
cat("P(X = 0) = e⁻² =", p_no_defects, "\n")

# Probability of 1 or 2 defects
p_1_or_2_defects <- dpois(1, lambda = lambda_defects) + 
                     dpois(2, lambda = lambda_defects)
cat("P(1 ≤ X ≤ 2) =", p_1_or_2_defects, "\n")
```

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

```r
# Exponential distribution parameter with mathematical interpretation
rate <- 2  # rate parameter λ

# Generate exponential random variables
set.seed(123)
exponential_data <- rexp(1000, rate = rate)

# Calculate probabilities
cat("P(X < 1):", pexp(1, rate = rate), "\n")
cat("P(X > 2):", 1 - pexp(2, rate = rate), "\n")

# Verify mathematical properties
cat("Theoretical mean (1/λ):", 1/rate, "\n")
cat("Sample mean:", mean(exponential_data), "\n")
cat("Theoretical variance (1/λ²):", 1/rate^2, "\n")
cat("Sample variance:", var(exponential_data), "\n")

# Plot exponential distribution with mathematical annotations
x <- seq(0, 5, by = 0.1)
exponential_density <- dexp(x, rate = rate)

plot(x, exponential_density, type = "l", 
     main = paste("Exponential Distribution Exp(", rate, ")"),
     xlab = "x", ylab = "f(x)",
     col = "red", lwd = 2)

# Add mathematical annotations
text(1, max(exponential_density), 
     expression(paste("E[X] = 1/", lambda, " = ", 1/rate)), 
     pos = 3, col = "darkred", font = 2)
```

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

```r
# Chi-square distribution with different degrees of freedom
df_values <- c(1, 3, 5, 10)
x <- seq(0, 20, by = 0.1)

# Plot chi-square distributions with mathematical annotations
plot(x, dchisq(x, df = df_values[1]), type = "l", 
     main = "Chi-Square Distributions",
     xlab = "x", ylab = "f(x)",
     col = "blue", lwd = 2)

for (i in 2:length(df_values)) {
  lines(x, dchisq(x, df = df_values[i]), 
        col = c("red", "green", "purple")[i-1], lwd = 2)
}

legend("topright", legend = paste("df =", df_values), 
       col = c("blue", "red", "green", "purple"), lwd = 2)

# Calculate critical values with mathematical interpretation
alpha <- 0.05
critical_values <- qchisq(1 - alpha, df = df_values)
cat("Critical values for α = 0.05:\n")
for (i in 1:length(df_values)) {
  cat("χ²(", df_values[i], ") =", critical_values[i], "\n")
}

# Verify mathematical properties
for (df in df_values) {
  cat("df =", df, ": E[X] =", df, ", Var(X) =", 2*df, "\n")
}
```

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

```r
# t-distribution with different degrees of freedom
df_t <- c(1, 5, 10, 30)
x <- seq(-4, 4, by = 0.1)

# Plot t-distributions with normal comparison
plot(x, dt(x, df = df_t[1]), type = "l", 
     main = "t-Distributions vs Normal",
     xlab = "x", ylab = "f(x)",
     col = "blue", lwd = 2)

for (i in 2:length(df_t)) {
  lines(x, dt(x, df = df_t[i]), 
        col = c("red", "green", "purple")[i-1], lwd = 2)
}

# Add normal distribution for comparison
lines(x, dnorm(x), col = "black", lty = 2, lwd = 2)

legend("topright", legend = c(paste("t(df =", df_t, ")"), "N(0,1)"), 
       col = c("blue", "red", "green", "purple", "black"), 
       lty = c(rep(1, 4), 2), lwd = 2)

# Calculate critical values with mathematical interpretation
alpha <- 0.05
t_critical <- qt(1 - alpha/2, df = df_t)
cat("Two-tailed critical values for α = 0.05:\n")
for (i in 1:length(df_t)) {
  cat("t(", df_t[i], ") =", t_critical[i], "\n")
}

# Compare with normal critical value
cat("Normal critical value (z):", qnorm(1 - alpha/2), "\n")
```

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

```r
# F-distribution with different degrees of freedom
df1_values <- c(5, 10, 20)
df2_values <- c(10, 20, 30)
x <- seq(0, 5, by = 0.1)

# Plot F-distributions with mathematical annotations
plot(x, df(x, df1 = df1_values[1], df2 = df2_values[1]), type = "l", 
     main = "F-Distributions",
     xlab = "x", ylab = "f(x)",
     col = "blue", lwd = 2)

for (i in 2:length(df1_values)) {
  lines(x, df(x, df1 = df1_values[i], df2 = df2_values[i]), 
        col = c("red", "green")[i-1], lwd = 2)
}

legend("topright", legend = paste("F(", df1_values, ",", df2_values, ")"), 
       col = c("blue", "red", "green"), lwd = 2)

# Calculate critical values with mathematical interpretation
alpha <- 0.05
f_critical <- qf(1 - alpha, df1 = df1_values, df2 = df2_values)
cat("Critical values for α = 0.05:\n")
for (i in 1:length(df1_values)) {
  cat("F(", df1_values[i], ",", df2_values[i], ") =", f_critical[i], "\n")
}
```

## Distribution Fitting

### Fitting Distributions to Data

```r
# Load sample data for distribution fitting
data(mtcars)

# Fit normal distribution to MPG with mathematical precision
mpg_data <- mtcars$mpg
mpg_mean <- mean(mpg_data)
mpg_sd <- sd(mpg_data)

# Plot histogram with fitted normal
hist(mpg_data, freq = FALSE, main = "MPG with Fitted Normal Distribution",
     xlab = "MPG", col = "lightblue")
curve(dnorm(x, mean = mpg_mean, sd = mpg_sd), add = TRUE, 
      col = "red", lwd = 2)

# Add mathematical annotations
text(mpg_mean, 0.1, 
     expression(paste("N(", mu, "=", mpg_mean, ", ", sigma, "=", mpg_sd, ")")), 
     col = "red", font = 2)

# Goodness-of-fit test with interpretation
ks_test_mpg <- ks.test(mpg_data, "pnorm", mean = mpg_mean, sd = mpg_sd)
cat("Kolmogorov-Smirnov Test for Normal Fit:\n")
cat("  D =", ks_test_mpg$statistic, "\n")
cat("  p-value =", ks_test_mpg$p.value, "\n")
cat("  Interpretation:", ifelse(ks_test_mpg$p.value > 0.05, 
                               "Normal distribution is a good fit", 
                               "Normal distribution may not be appropriate"), "\n")
```

### Comparing Multiple Distributions

```r
# Function to fit and compare distributions with mathematical rigor
fit_distributions <- function(data, distributions = c("normal", "exponential", "gamma")) {
  results <- list()
  
  for (dist in distributions) {
    if (dist == "normal") {
      # Fit normal distribution
      params <- list(mean = mean(data), sd = sd(data))
      ks_result <- ks.test(data, "pnorm", mean = params$mean, sd = params$sd)
      aic <- -2 * sum(log(dnorm(data, mean = params$mean, sd = params$sd))) + 2 * 2
    } else if (dist == "exponential") {
      # Fit exponential distribution
      rate <- 1/mean(data)
      ks_result <- ks.test(data, "pexp", rate = rate)
      aic <- -2 * sum(log(dexp(data, rate = rate))) + 2 * 1
    } else if (dist == "gamma") {
      # Fit gamma distribution
      library(MASS)
      fit_gamma <- fitdistr(data, "gamma")
      ks_result <- ks.test(data, "pgamma", shape = fit_gamma$estimate["shape"], 
                          rate = fit_gamma$estimate["rate"])
      aic <- -2 * sum(log(dgamma(data, shape = fit_gamma$estimate["shape"], 
                                 rate = fit_gamma$estimate["rate"]))) + 2 * 2
    }
    
    results[[dist]] <- list(
      distribution = dist,
      ks_statistic = ks_result$statistic,
      p_value = ks_result$p.value,
      aic = aic
    )
  }
  
  return(results)
}

# Apply to MPG data with comprehensive analysis
library(MASS)
mpg_fits <- fit_distributions(mpg_data)

# Compare results with mathematical interpretation
cat("Distribution Comparison for MPG Data:\n")
cat("=====================================\n")
for (dist_name in names(mpg_fits)) {
  cat(dist_name, "distribution:\n")
  cat("  KS statistic:", mpg_fits[[dist_name]]$ks_statistic, "\n")
  cat("  p-value:", mpg_fits[[dist_name]]$p_value, "\n")
  cat("  AIC:", mpg_fits[[dist_name]]$aic, "\n")
  cat("  Interpretation:", ifelse(mpg_fits[[dist_name]]$p_value > 0.05, 
                                 "Good fit", "Poor fit"), "\n\n")
}

# Find best fit based on AIC
best_fit <- names(mpg_fits)[which.min(sapply(mpg_fits, function(x) x$aic))]
cat("Best fitting distribution (lowest AIC):", best_fit, "\n")
```

## Practical Examples

### Example 1: Quality Control with Mathematical Precision

```r
# Simulate defect data with Poisson distribution
set.seed(123)
defects_per_batch <- rpois(100, lambda = 2.5)

# Calculate probabilities with mathematical interpretation
cat("Quality Control Analysis:\n")
cat("========================\n")
cat("Average defects per batch (λ):", 2.5, "\n")
cat("P(X = 0):", dpois(0, lambda = 2.5), "\n")
cat("P(X = 1):", dpois(1, lambda = 2.5), "\n")
cat("P(X ≤ 2):", ppois(2, lambda = 2.5), "\n")
cat("P(X ≥ 5):", 1 - ppois(4, lambda = 2.5), "\n")

# Plot defect distribution with mathematical annotations
hist(defects_per_batch, main = "Defects per Batch (Poisson λ=2.5)",
     xlab = "Number of Defects", col = "lightgreen", freq = FALSE)
curve(dpois(x, lambda = 2.5), add = TRUE, col = "red", lwd = 2, type = "s")

# Add mathematical annotations
text(2.5, 0.25, expression(paste("E[X] = Var(X) = ", lambda, " = 2.5")), 
     col = "red", font = 2)
```

### Example 2: Customer Service Times with Exponential Model

```r
# Simulate service times (exponential distribution)
set.seed(123)
service_times <- rexp(100, rate = 0.5)  # mean = 2 minutes

# Calculate probabilities with mathematical interpretation
cat("Customer Service Time Analysis:\n")
cat("==============================\n")
cat("Average service time (1/λ):", 1/0.5, "minutes\n")
cat("P(X < 1 minute):", pexp(1, rate = 0.5), "\n")
cat("P(X > 5 minutes):", 1 - pexp(5, rate = 0.5), "\n")
cat("P(1 < X < 3 minutes):", pexp(3, rate = 0.5) - pexp(1, rate = 0.5), "\n")

# Plot service time distribution with mathematical annotations
hist(service_times, main = "Service Times (Exponential λ=0.5)",
     xlab = "Time (minutes)", col = "lightcoral", freq = FALSE)
curve(dexp(x, rate = 0.5), add = TRUE, col = "red", lwd = 2)

# Add mathematical annotations
text(2, 0.4, expression(paste("E[X] = 1/", lambda, " = 2 minutes")), 
     col = "red", font = 2)
```

### Example 3: Investment Returns with Normal Model

```r
# Simulate investment returns (normal distribution)
set.seed(123)
returns <- rnorm(252, mean = 0.001, sd = 0.02)  # Daily returns

# Calculate probabilities with mathematical interpretation
cat("Investment Return Analysis:\n")
cat("==========================\n")
cat("Daily mean return (μ):", 0.001, "\n")
cat("Daily volatility (σ):", 0.02, "\n")
cat("P(positive return):", 1 - pnorm(0, mean = 0.001, sd = 0.02), "\n")
cat("P(return > 0.05):", 1 - pnorm(0.05, mean = 0.001, sd = 0.02), "\n")

# Value at Risk (VaR) calculations
var_95 <- qnorm(0.05, mean = 0.001, sd = 0.02)
var_99 <- qnorm(0.01, mean = 0.001, sd = 0.02)
cat("95% VaR:", var_95, "\n")
cat("99% VaR:", var_99, "\n")

# Plot return distribution with mathematical annotations
hist(returns, main = "Daily Investment Returns (Normal μ=0.001, σ=0.02)",
     xlab = "Return", col = "lightblue", freq = FALSE)
curve(dnorm(x, mean = 0.001, sd = 0.02), add = TRUE, col = "red", lwd = 2)

# Add mathematical annotations
text(0.001, 20, expression(paste("N(", mu, "=0.001, ", sigma, "=0.02)")), 
     col = "red", font = 2)
```

## Best Practices

### Choosing the Right Distribution

```r
# Comprehensive guidelines for distribution selection
cat("GUIDELINES FOR DISTRIBUTION SELECTION:\n")
cat("=====================================\n")
cat("1. Normal Distribution N(μ,σ²):\n")
cat("   - Use for: Symmetric, continuous data\n")
cat("   - Examples: Heights, weights, measurement errors\n")
cat("   - Properties: Bell-shaped, symmetric around mean\n\n")

cat("2. Binomial Distribution B(n,p):\n")
cat("   - Use for: Counts of successes in fixed trials\n")
cat("   - Examples: Coin flips, defect counts, survey responses\n")
cat("   - Properties: Discrete, bounded [0,n]\n\n")

cat("3. Poisson Distribution Poi(λ):\n")
cat("   - Use for: Counts of rare events\n")
cat("   - Examples: Customer arrivals, defect counts, accidents\n")
cat("   - Properties: Discrete, unbounded, E[X] = Var(X) = λ\n\n")

cat("4. Exponential Distribution Exp(λ):\n")
cat("   - Use for: Time between events\n")
cat("   - Examples: Service times, equipment lifetimes\n")
cat("   - Properties: Memoryless, E[X] = 1/λ\n\n")

cat("5. Chi-square Distribution χ²(k):\n")
cat("   - Use for: Variance testing, goodness-of-fit\n")
cat("   - Examples: Sample variance, contingency tables\n")
cat("   - Properties: E[X] = k, Var(X) = 2k\n\n")

cat("6. t-Distribution t(k):\n")
cat("   - Use for: Small sample inference\n")
cat("   - Examples: Confidence intervals, hypothesis tests\n")
cat("   - Properties: Heavier tails than normal\n\n")

cat("7. F-Distribution F(k₁,k₂):\n")
cat("   - Use for: Variance comparisons, ANOVA\n")
cat("   - Examples: Comparing group variances\n")
cat("   - Properties: Ratio of chi-square variables\n")
```

### Distribution Validation

```r
# Function to validate distribution fit with comprehensive checks
validate_distribution <- function(data, distribution, params) {
  cat("Distribution Validation for", distribution, "distribution:\n")
  cat("=====================================================\n")
  
  # Visual checks
  par(mfrow = c(2, 2))
  
  # Histogram with fitted curve
  hist(data, freq = FALSE, main = paste("Histogram with Fitted", distribution))
  if (distribution == "normal") {
    curve(dnorm(x, mean = params$mean, sd = params$sd), add = TRUE, col = "red")
  } else if (distribution == "exponential") {
    curve(dexp(x, rate = params$rate), add = TRUE, col = "red")
  } else if (distribution == "poisson") {
    curve(dpois(x, lambda = params$lambda), add = TRUE, col = "red", type = "s")
  }
  
  # Q-Q plot
  if (distribution == "normal") {
    qqnorm(data, main = "Q-Q Plot for Normality")
    qqline(data, col = "red")
  }
  
  # Empirical vs theoretical CDF
  plot(ecdf(data), main = "Empirical vs Theoretical CDF")
  if (distribution == "normal") {
    curve(pnorm(x, mean = params$mean, sd = params$sd), add = TRUE, col = "red")
  } else if (distribution == "exponential") {
    curve(pexp(x, rate = params$rate), add = TRUE, col = "red")
  }
  
  # P-P plot
  if (distribution == "normal") {
    theoretical_quantiles <- qnorm(ppoints(length(data)), mean = params$mean, sd = params$sd)
    plot(sort(data), theoretical_quantiles, main = "P-P Plot")
    abline(0, 1, col = "red")
  }
  
  par(mfrow = c(1, 1))
  
  # Statistical tests
  if (distribution == "normal") {
    ks_result <- ks.test(data, "pnorm", mean = params$mean, sd = params$sd)
    shapiro_result <- shapiro.test(data)
    cat("Kolmogorov-Smirnov test p-value:", ks_result$p.value, "\n")
    cat("Shapiro-Wilk test p-value:", shapiro_result$p.value, "\n")
  } else if (distribution == "exponential") {
    ks_result <- ks.test(data, "pexp", rate = params$rate)
    cat("Kolmogorov-Smirnov test p-value:", ks_result$p.value, "\n")
  }
  
  return(list(ks_result = ks_result, shapiro_result = shapiro_result))
}
```

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