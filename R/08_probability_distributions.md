# Probability Distributions

## Overview

Probability distributions are fundamental to statistical analysis. They describe how probabilities are distributed over the possible values of a random variable. Understanding distributions is crucial for hypothesis testing, confidence intervals, and statistical modeling.

## Normal Distribution

The normal (Gaussian) distribution is the most important continuous distribution in statistics.

### Basic Normal Distribution Functions

```r
# Generate normal random variables
set.seed(123)
normal_data <- rnorm(1000, mean = 0, sd = 1)

# Density function
x <- seq(-4, 4, by = 0.1)
normal_density <- dnorm(x, mean = 0, sd = 1)

# Plot normal distribution
plot(x, normal_density, type = "l", 
     main = "Standard Normal Distribution",
     xlab = "x", ylab = "Density",
     col = "blue", lwd = 2)
abline(v = 0, col = "red", lty = 2)
```

### Normal Distribution Properties

```r
# Calculate probabilities
cat("P(X < 0):", pnorm(0), "\n")
cat("P(X < 1):", pnorm(1), "\n")
cat("P(X < 2):", pnorm(2), "\n")

# Calculate quantiles
cat("95th percentile:", qnorm(0.95), "\n")
cat("99th percentile:", qnorm(0.99), "\n")

# Calculate probabilities for intervals
cat("P(-1 < X < 1):", pnorm(1) - pnorm(-1), "\n")
cat("P(-2 < X < 2):", pnorm(2) - pnorm(-2), "\n")
```

### Checking Normality

```r
# Visual normality checks
par(mfrow = c(2, 2))

# Histogram
hist(normal_data, main = "Histogram", xlab = "Value", col = "lightblue")

# Q-Q plot
qqnorm(normal_data, main = "Q-Q Plot")
qqline(normal_data, col = "red")

# Density plot
plot(density(normal_data), main = "Density Plot")
curve(dnorm(x, mean = 0, sd = 1), add = TRUE, col = "red", lty = 2)

# Box plot
boxplot(normal_data, main = "Box Plot")

par(mfrow = c(1, 1))

# Statistical normality tests
shapiro_test <- shapiro.test(normal_data)
print(shapiro_test)

# Kolmogorov-Smirnov test
ks_test <- ks.test(normal_data, "pnorm")
print(ks_test)
```

## Binomial Distribution

The binomial distribution models the number of successes in a fixed number of independent trials.

### Basic Binomial Functions

```r
# Binomial distribution parameters
n <- 10  # number of trials
p <- 0.5  # probability of success

# Generate binomial random variables
set.seed(123)
binomial_data <- rbinom(1000, size = n, prob = p)

# Calculate probabilities
cat("P(X = 5):", dbinom(5, size = n, prob = p), "\n")
cat("P(X ≤ 5):", pbinom(5, size = n, prob = p), "\n")
cat("P(X ≥ 5):", 1 - pbinom(4, size = n, prob = p), "\n")

# Plot binomial distribution
x <- 0:n
binomial_probs <- dbinom(x, size = n, prob = p)

plot(x, binomial_probs, type = "h", 
     main = "Binomial Distribution (n=10, p=0.5)",
     xlab = "Number of Successes", ylab = "Probability",
     col = "blue", lwd = 2)
```

### Binomial Distribution Examples

```r
# Example 1: Coin flipping
# Probability of getting exactly 7 heads in 10 flips
p_7_heads <- dbinom(7, size = 10, prob = 0.5)
cat("P(7 heads in 10 flips):", p_7_heads, "\n")

# Probability of getting 7 or more heads
p_7_or_more <- 1 - pbinom(6, size = 10, prob = 0.5)
cat("P(7 or more heads):", p_7_or_more, "\n")

# Example 2: Quality control
# Probability of 2 or fewer defective items in 20
p_defective <- dbinom(0:2, size = 20, prob = 0.1)
cat("P(0 defective):", p_defective[1], "\n")
cat("P(1 defective):", p_defective[2], "\n")
cat("P(2 defective):", p_defective[3], "\n")
cat("P(2 or fewer defective):", sum(p_defective), "\n")
```

## Poisson Distribution

The Poisson distribution models the number of events occurring in a fixed interval.

### Basic Poisson Functions

```r
# Poisson distribution parameter
lambda <- 3  # average number of events

# Generate Poisson random variables
set.seed(123)
poisson_data <- rpois(1000, lambda = lambda)

# Calculate probabilities
cat("P(X = 3):", dpois(3, lambda = lambda), "\n")
cat("P(X ≤ 3):", ppois(3, lambda = lambda), "\n")
cat("P(X ≥ 3):", 1 - ppois(2, lambda = lambda), "\n")

# Plot Poisson distribution
x <- 0:10
poisson_probs <- dpois(x, lambda = lambda)

plot(x, poisson_probs, type = "h", 
     main = "Poisson Distribution (λ=3)",
     xlab = "Number of Events", ylab = "Probability",
     col = "green", lwd = 2)
```

### Poisson Distribution Examples

```r
# Example 1: Customer arrivals
# Average 5 customers per hour
lambda_customers <- 5

# Probability of exactly 3 customers in an hour
p_3_customers <- dpois(3, lambda = lambda_customers)
cat("P(3 customers in 1 hour):", p_3_customers, "\n")

# Probability of 3 or fewer customers
p_3_or_fewer <- ppois(3, lambda = lambda_customers)
cat("P(3 or fewer customers):", p_3_or_fewer, "\n")

# Example 2: Defects in manufacturing
# Average 2 defects per 100 items
lambda_defects <- 2

# Probability of no defects
p_no_defects <- dpois(0, lambda = lambda_defects)
cat("P(no defects):", p_no_defects, "\n")

# Probability of 1 or 2 defects
p_1_or_2_defects <- dpois(1, lambda = lambda_defects) + 
                     dpois(2, lambda = lambda_defects)
cat("P(1 or 2 defects):", p_1_or_2_defects, "\n")
```

## Exponential Distribution

The exponential distribution models the time between events in a Poisson process.

### Basic Exponential Functions

```r
# Exponential distribution parameter
rate <- 2  # rate parameter

# Generate exponential random variables
set.seed(123)
exponential_data <- rexp(1000, rate = rate)

# Calculate probabilities
cat("P(X < 1):", pexp(1, rate = rate), "\n")
cat("P(X > 2):", 1 - pexp(2, rate = rate), "\n")

# Plot exponential distribution
x <- seq(0, 5, by = 0.1)
exponential_density <- dexp(x, rate = rate)

plot(x, exponential_density, type = "l", 
     main = "Exponential Distribution (rate=2)",
     xlab = "x", ylab = "Density",
     col = "red", lwd = 2)
```

## Chi-Square Distribution

The chi-square distribution is important for variance testing and goodness-of-fit tests.

### Basic Chi-Square Functions

```r
# Chi-square distribution with different degrees of freedom
df_values <- c(1, 3, 5, 10)
x <- seq(0, 20, by = 0.1)

# Plot chi-square distributions
plot(x, dchisq(x, df = df_values[1]), type = "l", 
     main = "Chi-Square Distributions",
     xlab = "x", ylab = "Density",
     col = "blue", lwd = 2)

for (i in 2:length(df_values)) {
  lines(x, dchisq(x, df = df_values[i]), 
        col = c("red", "green", "purple")[i-1], lwd = 2)
}

legend("topright", legend = paste("df =", df_values), 
       col = c("blue", "red", "green", "purple"), lwd = 2)

# Calculate critical values
alpha <- 0.05
critical_values <- qchisq(1 - alpha, df = df_values)
cat("Critical values for α = 0.05:\n")
for (i in 1:length(df_values)) {
  cat("df =", df_values[i], ":", critical_values[i], "\n")
}
```

## t-Distribution

The t-distribution is used for small sample inference and is similar to the normal distribution.

### Basic t-Distribution Functions

```r
# t-distribution with different degrees of freedom
df_t <- c(1, 5, 10, 30)
x <- seq(-4, 4, by = 0.1)

# Plot t-distributions
plot(x, dt(x, df = df_t[1]), type = "l", 
     main = "t-Distributions",
     xlab = "x", ylab = "Density",
     col = "blue", lwd = 2)

for (i in 2:length(df_t)) {
  lines(x, dt(x, df = df_t[i]), 
        col = c("red", "green", "purple")[i-1], lwd = 2)
}

# Add normal distribution for comparison
lines(x, dnorm(x), col = "black", lty = 2, lwd = 2)

legend("topright", legend = c(paste("t(df =", df_t, ")"), "Normal"), 
       col = c("blue", "red", "green", "purple", "black"), 
       lty = c(rep(1, 4), 2), lwd = 2)

# Calculate critical values
alpha <- 0.05
t_critical <- qt(1 - alpha/2, df = df_t)
cat("Two-tailed critical values for α = 0.05:\n")
for (i in 1:length(df_t)) {
  cat("df =", df_t[i], ":", t_critical[i], "\n")
}
```

## F-Distribution

The F-distribution is used for comparing variances and in ANOVA.

### Basic F-Distribution Functions

```r
# F-distribution with different degrees of freedom
df1_values <- c(5, 10, 20)
df2_values <- c(10, 20, 30)
x <- seq(0, 5, by = 0.1)

# Plot F-distributions
plot(x, df(x, df1 = df1_values[1], df2 = df2_values[1]), type = "l", 
     main = "F-Distributions",
     xlab = "x", ylab = "Density",
     col = "blue", lwd = 2)

for (i in 2:length(df1_values)) {
  lines(x, df(x, df1 = df1_values[i], df2 = df2_values[i]), 
        col = c("red", "green")[i-1], lwd = 2)
}

legend("topright", legend = paste("F(", df1_values, ",", df2_values, ")"), 
       col = c("blue", "red", "green"), lwd = 2)

# Calculate critical values
alpha <- 0.05
f_critical <- qf(1 - alpha, df1 = df1_values, df2 = df2_values)
cat("Critical values for α = 0.05:\n")
for (i in 1:length(df1_values)) {
  cat("F(", df1_values[i], ",", df2_values[i], "):", f_critical[i], "\n")
}
```

## Distribution Fitting

### Fitting Distributions to Data

```r
# Load sample data
data(mtcars)

# Fit normal distribution to MPG
mpg_data <- mtcars$mpg
mpg_mean <- mean(mpg_data)
mpg_sd <- sd(mpg_data)

# Plot histogram with fitted normal
hist(mpg_data, freq = FALSE, main = "MPG with Fitted Normal",
     xlab = "MPG", col = "lightblue")
curve(dnorm(x, mean = mpg_mean, sd = mpg_sd), add = TRUE, 
      col = "red", lwd = 2)

# Goodness-of-fit test
ks_test_mpg <- ks.test(mpg_data, "pnorm", mean = mpg_mean, sd = mpg_sd)
print(ks_test_mpg)
```

### Comparing Multiple Distributions

```r
# Function to fit and compare distributions
fit_distributions <- function(data, distributions = c("normal", "exponential", "gamma")) {
  results <- list()
  
  for (dist in distributions) {
    if (dist == "normal") {
      # Fit normal distribution
      params <- list(mean = mean(data), sd = sd(data))
      ks_result <- ks.test(data, "pnorm", mean = params$mean, sd = params$sd)
    } else if (dist == "exponential") {
      # Fit exponential distribution
      rate <- 1/mean(data)
      ks_result <- ks.test(data, "pexp", rate = rate)
    } else if (dist == "gamma") {
      # Fit gamma distribution
      fit_gamma <- fitdistr(data, "gamma")
      ks_result <- ks.test(data, "pgamma", shape = fit_gamma$estimate["shape"], 
                          rate = fit_gamma$estimate["rate"])
    }
    
    results[[dist]] <- list(
      distribution = dist,
      ks_statistic = ks_result$statistic,
      p_value = ks_result$p.value
    )
  }
  
  return(results)
}

# Apply to MPG data
library(MASS)
mpg_fits <- fit_distributions(mpg_data)

# Compare results
for (dist_name in names(mpg_fits)) {
  cat(dist_name, "distribution:\n")
  cat("  KS statistic:", mpg_fits[[dist_name]]$ks_statistic, "\n")
  cat("  p-value:", mpg_fits[[dist_name]]$p_value, "\n\n")
}
```

## Practical Examples

### Example 1: Quality Control

```r
# Simulate defect data
set.seed(123)
defects_per_batch <- rpois(100, lambda = 2.5)

# Calculate probabilities
cat("Probability of 0 defects:", dpois(0, lambda = 2.5), "\n")
cat("Probability of 1 defect:", dpois(1, lambda = 2.5), "\n")
cat("Probability of 2 or fewer defects:", ppois(2, lambda = 2.5), "\n")

# Plot defect distribution
hist(defects_per_batch, main = "Defects per Batch",
     xlab = "Number of Defects", col = "lightgreen")
```

### Example 2: Customer Service Times

```r
# Simulate service times (exponential distribution)
set.seed(123)
service_times <- rexp(100, rate = 0.5)  # mean = 2 minutes

# Calculate probabilities
cat("Probability of service time < 1 minute:", pexp(1, rate = 0.5), "\n")
cat("Probability of service time > 5 minutes:", 1 - pexp(5, rate = 0.5), "\n")

# Plot service time distribution
hist(service_times, main = "Service Times",
     xlab = "Time (minutes)", col = "lightcoral", freq = FALSE)
curve(dexp(x, rate = 0.5), add = TRUE, col = "red", lwd = 2)
```

### Example 3: Investment Returns

```r
# Simulate investment returns (normal distribution)
set.seed(123)
returns <- rnorm(252, mean = 0.001, sd = 0.02)  # Daily returns

# Calculate probabilities
cat("Probability of positive return:", 1 - pnorm(0, mean = 0.001, sd = 0.02), "\n")
cat("Probability of return > 0.05:", 1 - pnorm(0.05, mean = 0.001, sd = 0.02), "\n")

# Value at Risk (VaR)
var_95 <- qnorm(0.05, mean = 0.001, sd = 0.02)
cat("95% VaR:", var_95, "\n")

# Plot return distribution
hist(returns, main = "Daily Investment Returns",
     xlab = "Return", col = "lightblue", freq = FALSE)
curve(dnorm(x, mean = 0.001, sd = 0.02), add = TRUE, col = "red", lwd = 2)
```

## Best Practices

### Choosing the Right Distribution

```r
# Guidelines for distribution selection
cat("GUIDELINES FOR DISTRIBUTION SELECTION:\n")
cat("1. Normal: Use for symmetric, continuous data\n")
cat("2. Binomial: Use for counts of successes in fixed trials\n")
cat("3. Poisson: Use for counts of rare events\n")
cat("4. Exponential: Use for time between events\n")
cat("5. Chi-square: Use for variance testing\n")
cat("6. t-distribution: Use for small sample inference\n")
cat("7. F-distribution: Use for variance comparisons\n")
```

### Distribution Validation

```r
# Function to validate distribution fit
validate_distribution <- function(data, distribution, params) {
  # Visual checks
  par(mfrow = c(2, 2))
  
  # Histogram with fitted curve
  hist(data, freq = FALSE, main = "Histogram with Fitted Distribution")
  if (distribution == "normal") {
    curve(dnorm(x, mean = params$mean, sd = params$sd), add = TRUE, col = "red")
  } else if (distribution == "exponential") {
    curve(dexp(x, rate = params$rate), add = TRUE, col = "red")
  }
  
  # Q-Q plot
  if (distribution == "normal") {
    qqnorm(data, main = "Q-Q Plot")
    qqline(data, col = "red")
  }
  
  # Empirical vs theoretical CDF
  plot(ecdf(data), main = "Empirical vs Theoretical CDF")
  if (distribution == "normal") {
    curve(pnorm(x, mean = params$mean, sd = params$sd), add = TRUE, col = "red")
  }
  
  # Residual plot
  if (distribution == "normal") {
    theoretical_quantiles <- qnorm(ppoints(length(data)), mean = params$mean, sd = params$sd)
    plot(sort(data), theoretical_quantiles, main = "P-P Plot")
    abline(0, 1, col = "red")
  }
  
  par(mfrow = c(1, 1))
  
  # Statistical tests
  ks_result <- ks.test(data, distribution, params)
  return(ks_result)
}
```

## Exercises

### Exercise 1: Normal Distribution
Generate 1000 random numbers from a normal distribution with mean 50 and standard deviation 10. Calculate P(X < 45) and P(X > 60).

### Exercise 2: Binomial Distribution
Calculate the probability of getting exactly 8 heads in 15 coin flips. Also calculate the probability of getting 8 or more heads.

### Exercise 3: Poisson Distribution
If customers arrive at a rate of 3 per hour, calculate the probability of exactly 5 customers arriving in 2 hours.

### Exercise 4: Distribution Fitting
Fit different distributions to the iris sepal length data and determine which fits best.

### Exercise 5: Real-world Application
Find a real dataset and identify which probability distribution best describes it.

## Next Steps

In the next chapter, we'll learn about sampling and sampling distributions, which build upon our understanding of probability distributions.

---

**Key Takeaways:**
- Normal distribution is fundamental for statistical inference
- Choose distributions based on data characteristics
- Always validate distribution assumptions
- Use appropriate tests for goodness-of-fit
- Consider the context when selecting distributions
- Visual and statistical checks are both important
- Understand the parameters and properties of each distribution 