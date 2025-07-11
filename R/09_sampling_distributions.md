# Sampling and Sampling Distributions

## Overview

Sampling distributions are fundamental to statistical inference and form the bridge between probability theory and practical data analysis. They describe how sample statistics vary across different samples from the same population, providing the mathematical foundation for hypothesis testing, confidence intervals, and statistical modeling.

### The Importance of Sampling Distributions

Sampling distributions serve several critical functions:

1. **Statistical Inference**: Enable us to make probabilistic statements about population parameters
2. **Hypothesis Testing**: Provide the theoretical basis for testing statistical hypotheses
3. **Confidence Intervals**: Allow us to quantify uncertainty in our estimates
4. **Sample Size Planning**: Help determine appropriate sample sizes for desired precision

### Key Concepts

- **Population**: The complete set of individuals or objects of interest
- **Sample**: A subset of the population selected for study
- **Parameter**: A numerical characteristic of the population (e.g., population mean $\mu$)
- **Statistic**: A numerical characteristic of the sample (e.g., sample mean $\bar{X}$)
- **Sampling Distribution**: The probability distribution of a statistic across all possible samples

## Mathematical Foundations

### Expected Value and Variance of Sample Statistics

For a random sample of size $n$ from a population with mean $\mu$ and variance $\sigma^2$:

```math
\begin{align}
\text{Expected Value of Sample Mean:} & \quad E[\bar{X}] = \mu \\
\text{Variance of Sample Mean:} & \quad \text{Var}(\bar{X}) = \frac{\sigma^2}{n} \\
\text{Standard Error of Sample Mean:} & \quad \text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}
\end{align}
```

### Sampling Distribution Properties

The sampling distribution of the mean has these key properties:

```math
\begin{align}
\text{Mean:} & \quad \mu_{\bar{X}} = \mu \\
\text{Variance:} & \quad \sigma^2_{\bar{X}} = \frac{\sigma^2}{n} \\
\text{Standard Error:} & \quad \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}
\end{align}
```

### Central Limit Theorem

If $X_1, X_2, \ldots, X_n$ are independent and identically distributed random variables with mean $\mu$ and variance $\sigma^2$, then:

```math
\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i
```

As $n \to \infty$, the sampling distribution of $\bar{X}_n$ approaches a normal distribution:

```math
\bar{X}_n \xrightarrow{d} N\left(\mu, \frac{\sigma^2}{n}\right)
```

## Basic Sampling Concepts

### Simple Random Sampling

Simple random sampling ensures that every possible sample of size $n$ has an equal probability of being selected.

```r
# Load sample data for demonstration
data(mtcars)

# Simple random sampling with mathematical precision
set.seed(123)
sample_size <- 10
random_sample <- mtcars[sample(nrow(mtcars), sample_size), ]

# View sample with mathematical context
cat("Simple Random Sample (n =", sample_size, "):\n")
print(random_sample)

# Calculate sample statistics with mathematical interpretation
sample_mean <- mean(random_sample$mpg)
sample_sd <- sd(random_sample$mpg)
sample_se <- sample_sd / sqrt(sample_size)

cat("\nSample Statistics:\n")
cat("Sample mean (x̄):", sample_mean, "\n")
cat("Sample standard deviation (s):", sample_sd, "\n")
cat("Standard error of the mean (SE):", sample_se, "\n")

# Compare with population parameters
population_mean <- mean(mtcars$mpg)
population_sd <- sd(mtcars$mpg)
population_se <- population_sd / sqrt(sample_size)

cat("\nPopulation Parameters:\n")
cat("Population mean (μ):", population_mean, "\n")
cat("Population standard deviation (σ):", population_sd, "\n")
cat("Theoretical standard error:", population_se, "\n")

# Calculate sampling error
sampling_error <- sample_mean - population_mean
cat("Sampling error (x̄ - μ):", sampling_error, "\n")
```

### Systematic Sampling

Systematic sampling selects every $k$th element from the population, where $k = N/n$ and $N$ is the population size.

```r
# Systematic sampling with mathematical explanation
n <- nrow(mtcars)
k <- floor(n / sample_size)  # Sampling interval
cat("Systematic sampling interval (k = N/n):", k, "\n")

# Perform systematic sampling
systematic_sample <- mtcars[seq(1, n, by = k)[1:sample_size], ]

cat("\nSystematic Sample (n =", sample_size, "):\n")
print(systematic_sample)

# Calculate systematic sample statistics
sys_mean <- mean(systematic_sample$mpg)
sys_sd <- sd(systematic_sample$mpg)
sys_se <- sys_sd / sqrt(sample_size)

cat("\nSystematic Sample Statistics:\n")
cat("Sample mean:", sys_mean, "\n")
cat("Sample standard deviation:", sys_sd, "\n")
cat("Standard error:", sys_se, "\n")
```

### Stratified Sampling

Stratified sampling divides the population into homogeneous subgroups (strata) and samples from each stratum.

```r
# Stratified sampling by cylinders with mathematical precision
library(dplyr)

# Calculate sample size per stratum (proportional allocation)
strata_sizes <- mtcars %>%
  group_by(cyl) %>%
  summarise(
    n = n(),
    proportion = n / nrow(mtcars),
    sample_size = ceiling(n * sample_size / nrow(mtcars))
  )

cat("Stratified Sampling Allocation:\n")
print(strata_sizes)

# Perform stratified sampling
stratified_sample <- mtcars %>%
  group_by(cyl) %>%
  slice_sample(n = first(strata_sizes$sample_size)) %>%
  ungroup()

cat("\nStratified Sample:\n")
print(stratified_sample)

# Calculate stratified sample statistics
strat_mean <- mean(stratified_sample$mpg)
strat_sd <- sd(stratified_sample$mpg)
strat_se <- strat_sd / sqrt(nrow(stratified_sample))

cat("\nStratified Sample Statistics:\n")
cat("Sample mean:", strat_mean, "\n")
cat("Sample standard deviation:", strat_sd, "\n")
cat("Standard error:", strat_se, "\n")

# Compare sampling methods
cat("\nComparison of Sampling Methods:\n")
cat("Simple Random Sample mean:", sample_mean, "\n")
cat("Systematic Sample mean:", sys_mean, "\n")
cat("Stratified Sample mean:", strat_mean, "\n")
cat("Population mean:", population_mean, "\n")
```

## Sampling Distribution of the Mean

### Mathematical Foundation

The sampling distribution of the mean is the probability distribution of sample means from all possible samples of size $n$ from the population.

For a population with mean $\mu$ and standard deviation $\sigma$:

```math
\begin{align}
\text{Mean of sampling distribution:} & \quad \mu_{\bar{X}} = \mu \\
\text{Standard error:} & \quad \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}} \\
\text{Shape:} & \quad \text{Approximately normal for large } n \text{ (CLT)}
\end{align}
```

### Simulating Sampling Distribution

```r
# Simulate sampling distribution of the mean with mathematical precision
set.seed(123)
population_mean <- mean(mtcars$mpg)
population_sd <- sd(mtcars$mpg)
n_samples <- 1000
sample_size <- 10

cat("Sampling Distribution Simulation:\n")
cat("Population mean (μ):", population_mean, "\n")
cat("Population standard deviation (σ):", population_sd, "\n")
cat("Sample size (n):", sample_size, "\n")
cat("Number of samples:", n_samples, "\n")

# Generate sampling distribution
sample_means <- numeric(n_samples)
for (i in 1:n_samples) {
  sample_data <- sample(mtcars$mpg, size = sample_size, replace = TRUE)
  sample_means[i] <- mean(sample_data)
}

# Analyze sampling distribution with mathematical interpretation
sampling_mean <- mean(sample_means)
sampling_sd <- sd(sample_means)
theoretical_se <- population_sd / sqrt(sample_size)

cat("\nSampling Distribution Analysis:\n")
cat("Mean of sampling distribution:", sampling_mean, "\n")
cat("Standard deviation of sampling distribution:", sampling_sd, "\n")
cat("Theoretical standard error (σ/√n):", theoretical_se, "\n")
cat("Empirical vs Theoretical SE ratio:", sampling_sd / theoretical_se, "\n")

# Verify Central Limit Theorem properties
cat("\nCentral Limit Theorem Verification:\n")
cat("Sampling distribution mean ≈ Population mean:", abs(sampling_mean - population_mean) < 0.1, "\n")
cat("Sampling distribution SD ≈ σ/√n:", abs(sampling_sd - theoretical_se) < 0.1, "\n")
```

### Visualizing Sampling Distribution

```r
# Comprehensive visualization of sampling distribution
par(mfrow = c(2, 2))

# Histogram of sample means with normal overlay
hist(sample_means, main = "Sampling Distribution of Mean",
     xlab = "Sample Mean (x̄)", ylab = "Frequency", 
     col = "lightblue", freq = FALSE)

# Add theoretical normal curve
x <- seq(min(sample_means), max(sample_means), length.out = 100)
curve(dnorm(x, mean = population_mean, sd = theoretical_se), 
      add = TRUE, col = "red", lwd = 2)

# Add mathematical annotations
text(population_mean, max(density(sample_means)$y), 
     expression(paste("μ = ", population_mean)), 
     col = "red", font = 2, pos = 3)

# Q-Q plot for normality assessment
qqnorm(sample_means, main = "Q-Q Plot of Sample Means")
qqline(sample_means, col = "red", lwd = 2)

# Box plot for distribution shape
boxplot(sample_means, main = "Box Plot of Sample Means", 
        col = "lightgreen", ylab = "Sample Mean")

# Density plot comparison
plot(density(sample_means), main = "Density Plot vs Normal",
     xlab = "Sample Mean", ylab = "Density")
curve(dnorm(x, mean = population_mean, sd = theoretical_se), 
      add = TRUE, col = "red", lty = 2, lwd = 2)

par(mfrow = c(1, 1))

# Statistical normality test
shapiro_test <- shapiro.test(sample_means)
cat("\nNormality Test for Sampling Distribution:\n")
cat("Shapiro-Wilk test p-value:", shapiro_test$p.value, "\n")
cat("Interpretation:", ifelse(shapiro_test$p.value > 0.05, 
                             "Sampling distribution appears normal", 
                             "Sampling distribution may not be normal"), "\n")
```

## Central Limit Theorem

### Mathematical Statement

The Central Limit Theorem states that for independent and identically distributed random variables $X_1, X_2, \ldots, X_n$ with mean $\mu$ and variance $\sigma^2$:

```math
\text{As } n \to \infty, \quad \frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0,1)
```

This means that regardless of the original population distribution, the sampling distribution of the mean approaches normality as sample size increases.

### Demonstrating CLT

```r
# Demonstrate Central Limit Theorem with different distributions
set.seed(123)

# Function to simulate sampling distribution with mathematical precision
simulate_sampling_dist <- function(population, sample_size, n_samples = 1000) {
  sample_means <- numeric(n_samples)
  for (i in 1:n_samples) {
    sample_data <- sample(population, size = sample_size, replace = TRUE)
    sample_means[i] <- mean(sample_data)
  }
  return(sample_means)
}

# Different population distributions with mathematical properties
n <- 1000

# Uniform distribution: U(0,10)
uniform_pop <- runif(n, 0, 10)
uniform_means <- simulate_sampling_dist(uniform_pop, 30)
cat("Uniform Distribution Properties:\n")
cat("Population mean:", mean(uniform_pop), "(theoretical: 5)\n")
cat("Population variance:", var(uniform_pop), "(theoretical: 8.33)\n")
cat("Sampling distribution mean:", mean(uniform_means), "\n")
cat("Sampling distribution SE:", sd(uniform_means), "\n")
cat("Theoretical SE:", sqrt(var(uniform_pop)/30), "\n\n")

# Exponential distribution: Exp(0.5)
exp_pop <- rexp(n, rate = 0.5)
exp_means <- simulate_sampling_dist(exp_pop, 30)
cat("Exponential Distribution Properties:\n")
cat("Population mean:", mean(exp_pop), "(theoretical: 2)\n")
cat("Population variance:", var(exp_pop), "(theoretical: 4)\n")
cat("Sampling distribution mean:", mean(exp_means), "\n")
cat("Sampling distribution SE:", sd(exp_means), "\n")
cat("Theoretical SE:", sqrt(var(exp_pop)/30), "\n\n")

# Skewed distribution: Gamma(2,1)
skewed_pop <- rgamma(n, shape = 2, rate = 1)
skewed_means <- simulate_sampling_dist(skewed_pop, 30)
cat("Gamma Distribution Properties:\n")
cat("Population mean:", mean(skewed_pop), "(theoretical: 2)\n")
cat("Population variance:", var(skewed_pop), "(theoretical: 2)\n")
cat("Sampling distribution mean:", mean(skewed_means), "\n")
cat("Sampling distribution SE:", sd(skewed_means), "\n")
cat("Theoretical SE:", sqrt(var(skewed_pop)/30), "\n")

# Plot results with mathematical annotations
par(mfrow = c(3, 2))

# Uniform distribution
hist(uniform_pop, main = "Uniform Population U(0,10)", 
     xlab = "Value", col = "lightblue", freq = FALSE)
curve(dunif(x, 0, 10), add = TRUE, col = "red", lwd = 2)

hist(uniform_means, main = "Sampling Distribution (n=30)", 
     xlab = "Sample Mean", col = "lightgreen", freq = FALSE)
curve(dnorm(x, mean = mean(uniform_means), sd = sd(uniform_means)), 
      add = TRUE, col = "red", lwd = 2)

# Exponential distribution
hist(exp_pop, main = "Exponential Population Exp(0.5)", 
     xlab = "Value", col = "lightcoral", freq = FALSE)
curve(dexp(x, rate = 0.5), add = TRUE, col = "red", lwd = 2)

hist(exp_means, main = "Sampling Distribution (n=30)", 
     xlab = "Sample Mean", col = "lightyellow", freq = FALSE)
curve(dnorm(x, mean = mean(exp_means), sd = sd(exp_means)), 
      add = TRUE, col = "red", lwd = 2)

# Skewed distribution
hist(skewed_pop, main = "Gamma Population Γ(2,1)", 
     xlab = "Value", col = "lightpink", freq = FALSE)
curve(dgamma(x, shape = 2, rate = 1), add = TRUE, col = "red", lwd = 2)

hist(skewed_means, main = "Sampling Distribution (n=30)", 
     xlab = "Sample Mean", col = "lightcyan", freq = FALSE)
curve(dnorm(x, mean = mean(skewed_means), sd = sd(skewed_means)), 
      add = TRUE, col = "red", lwd = 2)

par(mfrow = c(1, 1))
```

### Effect of Sample Size

```r
# Study effect of sample size on sampling distribution with mathematical precision
sample_sizes <- c(5, 10, 30, 50)
n_samples <- 1000

cat("Effect of Sample Size on Sampling Distribution:\n")
cat("==============================================\n")

# Simulate for different sample sizes
sampling_distributions <- list()
for (size in sample_sizes) {
  sample_means <- numeric(n_samples)
  for (i in 1:n_samples) {
    sample_data <- sample(mtcars$mpg, size = size, replace = TRUE)
    sample_means[i] <- mean(sample_data)
  }
  sampling_distributions[[as.character(size)]] <- sample_means
  
  # Calculate theoretical and empirical properties
  theoretical_se <- population_sd / sqrt(size)
  empirical_se <- sd(sample_means)
  
  cat("Sample size n =", size, ":\n")
  cat("  Theoretical SE (σ/√n):", theoretical_se, "\n")
  cat("  Empirical SE:", empirical_se, "\n")
  cat("  Ratio (Empirical/Theoretical):", empirical_se / theoretical_se, "\n\n")
}

# Plot results with mathematical annotations
par(mfrow = c(2, 2))
for (i in 1:length(sample_sizes)) {
  size <- sample_sizes[i]
  means <- sampling_distributions[[as.character(size)]]
  
  hist(means, main = paste("Sample Size n =", size),
       xlab = "Sample Mean", col = "lightblue", freq = FALSE)
  
  # Add theoretical normal curve
  x <- seq(min(means), max(means), length.out = 100)
  theoretical_mean <- population_mean
  theoretical_se <- population_sd / sqrt(size)
  curve(dnorm(x, mean = theoretical_mean, sd = theoretical_se), 
        add = TRUE, col = "red", lwd = 2)
  
  # Add mathematical annotations
  text(theoretical_mean, max(density(means)$y), 
       expression(paste("SE = ", sigma, "/√n")), 
       col = "red", font = 2, pos = 3)
}

par(mfrow = c(1, 1))

# Compare standard errors with mathematical precision
theoretical_ses <- population_sd / sqrt(sample_sizes)
empirical_ses <- sapply(sampling_distributions, sd)

comparison <- data.frame(
  Sample_Size = sample_sizes,
  Theoretical_SE = theoretical_ses,
  Empirical_SE = empirical_ses,
  Ratio = empirical_ses / theoretical_ses
)

cat("Standard Error Comparison:\n")
print(comparison)

# Verify that SE decreases as 1/√n
cat("\nVerification of SE ∝ 1/√n:\n")
for (i in 2:length(sample_sizes)) {
  ratio <- theoretical_ses[1] / theoretical_ses[i]
  expected_ratio <- sqrt(sample_sizes[i] / sample_sizes[1])
  cat("n =", sample_sizes[1], "to n =", sample_sizes[i], 
      ": SE ratio =", ratio, "(expected:", expected_ratio, ")\n")
}
```

## Sampling Distribution of the Proportion

### Mathematical Foundation

For a population with proportion $p$ of successes, the sampling distribution of the sample proportion $\hat{p}$ has these properties:

```math
\begin{align}
\text{Mean:} & \quad \mu_{\hat{p}} = p \\
\text{Variance:} & \quad \sigma^2_{\hat{p}} = \frac{p(1-p)}{n} \\
\text{Standard Error:} & \quad \sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}
\end{align}
```

For large samples ($np \geq 10$ and $n(1-p) \geq 10$), the sampling distribution is approximately normal:

```math
\hat{p} \sim N\left(p, \sqrt{\frac{p(1-p)}{n}}\right)
```

### Simulating Proportion Sampling

```r
# Simulate sampling distribution of proportion with mathematical precision
set.seed(123)
population_prop <- mean(mtcars$am)  # Proportion of manual transmissions
n_samples <- 1000
sample_size <- 20

cat("Sampling Distribution of Proportion:\n")
cat("==================================\n")
cat("Population proportion (p):", population_prop, "\n")
cat("Sample size (n):", sample_size, "\n")
cat("Number of samples:", n_samples, "\n")

# Verify normal approximation conditions
cat("np =", sample_size * population_prop, "(should be ≥ 10)\n")
cat("n(1-p) =", sample_size * (1 - population_prop), "(should be ≥ 10)\n")

# Generate sampling distribution of proportion
sample_proportions <- numeric(n_samples)
for (i in 1:n_samples) {
  sample_data <- sample(mtcars$am, size = sample_size, replace = TRUE)
  sample_proportions[i] <- mean(sample_data)
}

# Analyze sampling distribution with mathematical interpretation
sampling_prop_mean <- mean(sample_proportions)
sampling_prop_sd <- sd(sample_proportions)
theoretical_prop_se <- sqrt(population_prop * (1 - population_prop) / sample_size)

cat("\nSampling Distribution Analysis:\n")
cat("Mean of sampling distribution:", sampling_prop_mean, "\n")
cat("Standard deviation of sampling distribution:", sampling_prop_sd, "\n")
cat("Theoretical standard error √[p(1-p)/n]:", theoretical_prop_se, "\n")
cat("Empirical vs Theoretical SE ratio:", sampling_prop_sd / theoretical_prop_se, "\n")

# Verify CLT for proportions
cat("\nCentral Limit Theorem Verification for Proportions:\n")
cat("Sampling distribution mean ≈ Population proportion:", 
    abs(sampling_prop_mean - population_prop) < 0.01, "\n")
cat("Sampling distribution SD ≈ √[p(1-p)/n]:", 
    abs(sampling_prop_sd - theoretical_prop_se) < 0.01, "\n")

# Plot sampling distribution with mathematical annotations
hist(sample_proportions, main = "Sampling Distribution of Proportion",
     xlab = "Sample Proportion (p̂)", ylab = "Frequency", 
     col = "lightgreen", freq = FALSE)

# Add theoretical normal curve
x <- seq(min(sample_proportions), max(sample_proportions), length.out = 100)
curve(dnorm(x, mean = population_prop, sd = theoretical_prop_se), 
      add = TRUE, col = "red", lwd = 2)

# Add mathematical annotations
text(population_prop, max(density(sample_proportions)$y), 
     expression(paste("p = ", population_prop)), 
     col = "red", font = 2, pos = 3)

# Q-Q plot for normality assessment
qqnorm(sample_proportions, main = "Q-Q Plot of Sample Proportions")
qqline(sample_proportions, col = "red", lwd = 2)

# Statistical normality test
shapiro_test_prop <- shapiro.test(sample_proportions)
cat("\nNormality Test for Proportion Sampling Distribution:\n")
cat("Shapiro-Wilk test p-value:", shapiro_test_prop$p.value, "\n")
cat("Interpretation:", ifelse(shapiro_test_prop$p.value > 0.05, 
                             "Sampling distribution appears normal", 
                             "Sampling distribution may not be normal"), "\n")
```

## Sampling Distribution of the Variance

### Mathematical Foundation

For a sample of size $n$ from a normal population with variance $\sigma^2$, the sampling distribution of the sample variance $S^2$ follows a chi-square distribution:

```math
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)
```

This means:

```math
\begin{align}
\text{Mean of } S^2: & \quad E[S^2] = \sigma^2 \\
\text{Variance of } S^2: & \quad \text{Var}(S^2) = \frac{2\sigma^4}{n-1} \\
\text{Standard Error of } S^2: & \quad \text{SE}(S^2) = \sigma^2\sqrt{\frac{2}{n-1}}
\end{align}
```

### Simulating Variance Sampling

```r
# Simulate sampling distribution of variance with mathematical precision
set.seed(123)
population_var <- var(mtcars$mpg)
population_sd <- sd(mtcars$mpg)
n_samples <- 1000
sample_size <- 10

cat("Sampling Distribution of Variance:\n")
cat("=================================\n")
cat("Population variance (σ²):", population_var, "\n")
cat("Population standard deviation (σ):", population_sd, "\n")
cat("Sample size (n):", sample_size, "\n")
cat("Degrees of freedom (n-1):", sample_size - 1, "\n")

# Generate sampling distribution of variance
sample_variances <- numeric(n_samples)
for (i in 1:n_samples) {
  sample_data <- sample(mtcars$mpg, size = sample_size, replace = TRUE)
  sample_variances[i] <- var(sample_data)
}

# Analyze sampling distribution with mathematical interpretation
sampling_var_mean <- mean(sample_variances)
sampling_var_sd <- sd(sample_variances)
theoretical_var_se <- population_var * sqrt(2 / (sample_size - 1))

cat("\nSampling Distribution Analysis:\n")
cat("Mean of sampling distribution:", sampling_var_mean, "\n")
cat("Standard deviation of sampling distribution:", sampling_var_sd, "\n")
cat("Theoretical standard error σ²√[2/(n-1)]:", theoretical_var_se, "\n")
cat("Empirical vs Theoretical SE ratio:", sampling_var_sd / theoretical_var_se, "\n")

# Verify chi-square distribution properties
cat("\nChi-Square Distribution Verification:\n")
cat("Sampling distribution mean ≈ Population variance:", 
    abs(sampling_var_mean - population_var) < 1, "\n")
cat("Sampling distribution SD ≈ σ²√[2/(n-1)]:", 
    abs(sampling_var_sd - theoretical_var_se) < 1, "\n")

# Plot sampling distribution with mathematical annotations
hist(sample_variances, main = "Sampling Distribution of Variance",
     xlab = "Sample Variance (S²)", ylab = "Frequency", 
     col = "lightcoral", freq = FALSE)

# Add theoretical chi-square distribution
df <- sample_size - 1
theoretical_var_dist <- population_var * df / qchisq(ppoints(length(sample_variances)), df)
hist(theoretical_var_dist, add = TRUE, col = "red", alpha = 0.3, freq = FALSE)

# Add mathematical annotations
text(population_var, max(density(sample_variances)$y), 
     expression(paste("σ² = ", population_var)), 
     col = "red", font = 2, pos = 3)

# Q-Q plot for chi-square distribution
chi_square_quantiles <- qchisq(ppoints(length(sample_variances)), df)
scaled_variances <- (sample_variances * df) / population_var

plot(chi_square_quantiles, scaled_variances, 
     main = "Q-Q Plot for Chi-Square Distribution",
     xlab = "Theoretical Chi-Square Quantiles", 
     ylab = "Sample Chi-Square Values")
abline(0, 1, col = "red", lwd = 2)

# Statistical test for chi-square distribution
ks_test_var <- ks.test(scaled_variances, "pchisq", df = df)
cat("\nChi-Square Distribution Test:\n")
cat("Kolmogorov-Smirnov test p-value:", ks_test_var$p.value, "\n")
cat("Interpretation:", ifelse(ks_test_var$p.value > 0.05, 
                             "Variance sampling distribution appears chi-square", 
                             "Variance sampling distribution may not be chi-square"), "\n")
```

## Bootstrap Sampling

### Mathematical Foundation

Bootstrap sampling is a resampling technique that estimates the sampling distribution of a statistic by repeatedly sampling with replacement from the original data.

For a sample $X_1, X_2, \ldots, X_n$, bootstrap samples are created by sampling with replacement:

```math
X^*_1, X^*_2, \ldots, X^*_n \sim \text{Uniform}(X_1, X_2, \ldots, X_n)
```

The bootstrap estimate of the standard error is:

```math
\text{SE}_{boot}(\hat{\theta}) = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}(\hat{\theta}^*_b - \bar{\hat{\theta}}^*)^2}
```

Where $\hat{\theta}^*_b$ is the statistic calculated from the $b$th bootstrap sample.

### Basic Bootstrap

```r
# Bootstrap sampling with mathematical precision
library(boot)

# Bootstrap function for mean with mathematical interpretation
boot_mean <- function(data, indices) {
  return(mean(data[indices]))
}

# Perform bootstrap with comprehensive analysis
set.seed(123)
boot_results <- boot(mtcars$mpg, boot_mean, R = 1000)

cat("Bootstrap Analysis for Sample Mean:\n")
cat("==================================\n")
cat("Original sample mean:", mean(mtcars$mpg), "\n")
cat("Bootstrap mean:", mean(boot_results$t), "\n")
cat("Bootstrap standard error:", sd(boot_results$t), "\n")
cat("Theoretical standard error:", sd(mtcars$mpg) / sqrt(length(mtcars$mpg)), "\n")

# Bootstrap confidence interval with mathematical interpretation
boot_ci <- boot.ci(boot_results, type = "perc", conf = 0.95)
cat("\nBootstrap 95% Confidence Interval:\n")
cat("Lower bound:", boot_ci$percent[4], "\n")
cat("Upper bound:", boot_ci$percent[5], "\n")

# Plot bootstrap distribution with mathematical annotations
hist(boot_results$t, main = "Bootstrap Distribution of Mean",
     xlab = "Bootstrap Mean (x̄*)", ylab = "Frequency", 
     col = "lightblue", freq = FALSE)

# Add confidence interval lines
abline(v = boot_ci$percent[4:5], col = "red", lty = 2, lwd = 2)
abline(v = mean(mtcars$mpg), col = "green", lwd = 2)

# Add mathematical annotations
text(mean(mtcars$mpg), max(density(boot_results$t)$y), 
     "Original Mean", col = "green", font = 2, pos = 3)

# Q-Q plot for bootstrap distribution
qqnorm(boot_results$t, main = "Q-Q Plot of Bootstrap Means")
qqline(boot_results$t, col = "red", lwd = 2)

# Statistical analysis of bootstrap distribution
cat("\nBootstrap Distribution Analysis:\n")
cat("Skewness:", skewness(boot_results$t), "\n")
cat("Kurtosis:", kurtosis(boot_results$t), "\n")
cat("Normality test p-value:", shapiro.test(boot_results$t)$p.value, "\n")
```

### Bootstrap for Different Statistics

```r
# Bootstrap for median with mathematical interpretation
boot_median <- function(data, indices) {
  return(median(data[indices]))
}

boot_median_results <- boot(mtcars$mpg, boot_median, R = 1000)
boot_median_ci <- boot.ci(boot_median_results, type = "perc", conf = 0.95)

cat("Bootstrap Analysis for Sample Median:\n")
cat("====================================\n")
cat("Original sample median:", median(mtcars$mpg), "\n")
cat("Bootstrap median:", median(boot_median_results$t), "\n")
cat("Bootstrap standard error:", sd(boot_median_results$t), "\n")
cat("95% Confidence interval:", boot_median_ci$percent[4:5], "\n\n")

# Bootstrap for standard deviation with mathematical interpretation
boot_sd <- function(data, indices) {
  return(sd(data[indices]))
}

boot_sd_results <- boot(mtcars$mpg, boot_sd, R = 1000)
boot_sd_ci <- boot.ci(boot_sd_results, type = "perc", conf = 0.95)

cat("Bootstrap Analysis for Sample Standard Deviation:\n")
cat("===============================================\n")
cat("Original sample SD:", sd(mtcars$mpg), "\n")
cat("Bootstrap SD:", median(boot_sd_results$t), "\n")
cat("Bootstrap standard error:", sd(boot_sd_results$t), "\n")
cat("95% Confidence interval:", boot_sd_ci$percent[4:5], "\n\n")

# Compare bootstrap results with mathematical precision
comparison_table <- data.frame(
  Statistic = c("Mean", "Median", "Standard Deviation"),
  Original_Value = c(mean(mtcars$mpg), median(mtcars$mpg), sd(mtcars$mpg)),
  Bootstrap_Mean = c(mean(boot_results$t), mean(boot_median_results$t), mean(boot_sd_results$t)),
  Bootstrap_SE = c(sd(boot_results$t), sd(boot_median_results$t), sd(boot_sd_results$t)),
  CI_Lower = c(boot_ci$percent[4], boot_median_ci$percent[4], boot_sd_ci$percent[4]),
  CI_Upper = c(boot_ci$percent[5], boot_median_ci$percent[5], boot_sd_ci$percent[5])
)

cat("Bootstrap Comparison Summary:\n")
print(comparison_table)

# Plot comparison of bootstrap distributions
par(mfrow = c(1, 3))

hist(boot_results$t, main = "Bootstrap: Mean", xlab = "Value", col = "lightblue")
abline(v = mean(mtcars$mpg), col = "red", lwd = 2)

hist(boot_median_results$t, main = "Bootstrap: Median", xlab = "Value", col = "lightgreen")
abline(v = median(mtcars$mpg), col = "red", lwd = 2)

hist(boot_sd_results$t, main = "Bootstrap: SD", xlab = "Value", col = "lightcoral")
abline(v = sd(mtcars$mpg), col = "red", lwd = 2)

par(mfrow = c(1, 1))
```

## Sampling Methods Comparison

### Mathematical Comparison

Different sampling methods have different statistical properties:

```math
\begin{align}
\text{Simple Random Sampling:} & \quad \text{Var}(\bar{X}_{SRS}) = \frac{\sigma^2}{n} \\
\text{Systematic Sampling:} & \quad \text{Var}(\bar{X}_{SYS}) \approx \frac{\sigma^2}{n} \text{ (if no periodicity)} \\
\text{Stratified Sampling:} & \quad \text{Var}(\bar{X}_{STR}) = \sum_{h=1}^{H} \frac{N_h^2}{N^2} \frac{\sigma_h^2}{n_h}
\end{align}
```

### Comparing Different Sampling Methods

```r
# Function to compare sampling methods with mathematical precision
compare_sampling_methods <- function(data, variable, sample_size, n_simulations = 100) {
  # Simple random sampling
  srs_means <- numeric(n_simulations)
  srs_ses <- numeric(n_simulations)
  
  for (i in 1:n_simulations) {
    srs_sample <- sample(data[[variable]], size = sample_size, replace = TRUE)
    srs_means[i] <- mean(srs_sample)
    srs_ses[i] <- sd(srs_sample) / sqrt(sample_size)
  }
  
  # Systematic sampling
  sys_means <- numeric(n_simulations)
  sys_ses <- numeric(n_simulations)
  n <- length(data[[variable]])
  k <- floor(n / sample_size)
  
  for (i in 1:n_simulations) {
    start <- sample(1:k, 1)
    indices <- seq(start, n, by = k)[1:sample_size]
    sys_sample <- data[[variable]][indices]
    sys_means[i] <- mean(sys_sample)
    sys_ses[i] <- sd(sys_sample) / sqrt(sample_size)
  }
  
  # Stratified sampling (if grouping variable available)
  if ("cyl" %in% names(data)) {
    strat_means <- numeric(n_simulations)
    strat_ses <- numeric(n_simulations)
    
    for (i in 1:n_simulations) {
      strat_sample <- data %>%
        group_by(cyl) %>%
        slice_sample(n = ceiling(sample_size / length(unique(data$cyl)))) %>%
        pull(variable)
      strat_means[i] <- mean(strat_sample)
      strat_ses[i] <- sd(strat_sample) / sqrt(length(strat_sample))
    }
  } else {
    strat_means <- NULL
    strat_ses <- NULL
  }
  
  # Calculate efficiency measures
  pop_mean <- mean(data[[variable]])
  pop_var <- var(data[[variable]])
  
  srs_efficiency <- pop_var / var(srs_means)
  sys_efficiency <- pop_var / var(sys_means)
  
  if (!is.null(strat_means)) {
    strat_efficiency <- pop_var / var(strat_means)
  } else {
    strat_efficiency <- NULL
  }
  
  # Return comprehensive results
  results <- list(
    SRS = list(means = srs_means, ses = srs_ses, efficiency = srs_efficiency),
    Systematic = list(means = sys_means, ses = sys_ses, efficiency = sys_efficiency),
    Stratified = list(means = strat_means, ses = strat_ses, efficiency = strat_efficiency)
  )
  
  return(results)
}

# Apply comparison with mathematical analysis
sampling_comparison <- compare_sampling_methods(mtcars, "mpg", 10)

cat("Sampling Methods Comparison:\n")
cat("===========================\n")
cat("Population mean:", mean(mtcars$mpg), "\n")
cat("Population variance:", var(mtcars$mpg), "\n\n")

cat("Simple Random Sampling:\n")
cat("  Mean of sample means:", mean(sampling_comparison$SRS$means), "\n")
cat("  Variance of sample means:", var(sampling_comparison$SRS$means), "\n")
cat("  Efficiency:", sampling_comparison$SRS$efficiency, "\n\n")

cat("Systematic Sampling:\n")
cat("  Mean of sample means:", mean(sampling_comparison$Systematic$means), "\n")
cat("  Variance of sample means:", var(sampling_comparison$Systematic$means), "\n")
cat("  Efficiency:", sampling_comparison$Systematic$efficiency, "\n\n")

if (!is.null(sampling_comparison$Stratified$means)) {
  cat("Stratified Sampling:\n")
  cat("  Mean of sample means:", mean(sampling_comparison$Stratified$means), "\n")
  cat("  Variance of sample means:", var(sampling_comparison$Stratified$means), "\n")
  cat("  Efficiency:", sampling_comparison$Stratified$efficiency, "\n\n")
}

# Plot comparison with mathematical annotations
par(mfrow = c(2, 2))

hist(sampling_comparison$SRS$means, main = "Simple Random Sampling",
     xlab = "Sample Mean", col = "lightblue", freq = FALSE)
abline(v = mean(mtcars$mpg), col = "red", lwd = 2)

hist(sampling_comparison$Systematic$means, main = "Systematic Sampling",
     xlab = "Sample Mean", col = "lightgreen", freq = FALSE)
abline(v = mean(mtcars$mpg), col = "red", lwd = 2)

if (!is.null(sampling_comparison$Stratified$means)) {
  hist(sampling_comparison$Stratified$means, main = "Stratified Sampling",
       xlab = "Sample Mean", col = "lightcoral", freq = FALSE)
  abline(v = mean(mtcars$mpg), col = "red", lwd = 2)
}

# Efficiency comparison
efficiencies <- c(
  SRS = sampling_comparison$SRS$efficiency,
  Systematic = sampling_comparison$Systematic$efficiency
)

if (!is.null(sampling_comparison$Stratified$efficiency)) {
  efficiencies <- c(efficiencies, Stratified = sampling_comparison$Stratified$efficiency)
}

barplot(efficiencies, main = "Sampling Efficiency Comparison",
        ylab = "Efficiency (Higher is Better)", col = c("lightblue", "lightgreen", "lightcoral"))

par(mfrow = c(1, 1))

# Statistical comparison
cat("Statistical Comparison of Sampling Methods:\n")
cat("==========================================\n")

# Bias comparison
srs_bias <- mean(sampling_comparison$SRS$means) - mean(mtcars$mpg)
sys_bias <- mean(sampling_comparison$Systematic$means) - mean(mtcars$mpg)

cat("SRS bias:", srs_bias, "\n")
cat("Systematic bias:", sys_bias, "\n")

if (!is.null(sampling_comparison$Stratified$means)) {
  strat_bias <- mean(sampling_comparison$Stratified$means) - mean(mtcars$mpg)
  cat("Stratified bias:", strat_bias, "\n")
}

# Precision comparison
cat("\nPrecision Comparison (Standard Error):\n")
cat("SRS SE:", sd(sampling_comparison$SRS$means), "\n")
cat("Systematic SE:", sd(sampling_comparison$Systematic$means), "\n")

if (!is.null(sampling_comparison$Stratified$means)) {
  cat("Stratified SE:", sd(sampling_comparison$Stratified$means), "\n")
}
```

## Practical Examples

### Example 1: Quality Control Sampling with Mathematical Precision

```r
# Simulate quality control scenario with comprehensive analysis
set.seed(123)
production_batch <- rnorm(1000, mean = 100, sd = 5)  # Product weights

cat("Quality Control Sampling Analysis:\n")
cat("=================================\n")
cat("Population mean (μ):", mean(production_batch), "\n")
cat("Population standard deviation (σ):", sd(production_batch), "\n")
cat("Population size (N):", length(production_batch), "\n")

# Quality control sampling with mathematical interpretation
n_samples <- 100
sample_size <- 20

qc_means <- numeric(n_samples)
qc_sds <- numeric(n_samples)

for (i in 1:n_samples) {
  sample_data <- sample(production_batch, size = sample_size)
  qc_means[i] <- mean(sample_data)
  qc_sds[i] <- sd(sample_data)
}

# Calculate quality control limits with mathematical precision
population_mean <- mean(production_batch)
population_sd <- sd(production_batch)
theoretical_se <- population_sd / sqrt(sample_size)

ucl_mean <- population_mean + 2 * theoretical_se
lcl_mean <- population_mean - 2 * theoretical_se

cat("\nQuality Control Limits (2σ limits):\n")
cat("Upper Control Limit (UCL):", ucl_mean, "\n")
cat("Lower Control Limit (LCL):", lcl_mean, "\n")
cat("Theoretical standard error:", theoretical_se, "\n")

# Analyze sampling distribution
cat("\nSampling Distribution Analysis:\n")
cat("Mean of sample means:", mean(qc_means), "\n")
cat("Standard deviation of sample means:", sd(qc_means), "\n")
cat("Empirical vs Theoretical SE ratio:", sd(qc_means) / theoretical_se, "\n")

# Count out-of-control samples
out_of_control <- sum(qc_means < lcl_mean | qc_means > ucl_mean)
cat("Out-of-control samples:", out_of_control, "(", out_of_control/n_samples*100, "%)\n")

# Plot control chart with mathematical annotations
plot(qc_means, type = "l", main = "Quality Control Chart (x̄ Chart)",
     xlab = "Sample Number", ylab = "Sample Mean (x̄)")
abline(h = population_mean, col = "blue", lty = 2, lwd = 2)
abline(h = c(lcl_mean, ucl_mean), col = "red", lty = 2, lwd = 2)

# Add mathematical annotations
text(n_samples/2, population_mean + 2*theoretical_se, 
     expression(paste("UCL = μ + 2σ/√n")), col = "red", font = 2, pos = 3)
text(n_samples/2, population_mean - 2*theoretical_se, 
     expression(paste("LCL = μ - 2σ/√n")), col = "red", font = 2, pos = 1)

# Histogram of sample means with normal overlay
hist(qc_means, main = "Distribution of Sample Means",
     xlab = "Sample Mean", col = "lightblue", freq = FALSE)
curve(dnorm(x, mean = population_mean, sd = theoretical_se), 
      add = TRUE, col = "red", lwd = 2)

# Add mathematical annotations
text(population_mean, max(density(qc_means)$y), 
     expression(paste("μ = ", population_mean)), col = "red", font = 2, pos = 3)
```

### Example 2: Survey Sampling with Statistical Precision

```r
# Simulate survey data with comprehensive analysis
set.seed(123)
population_size <- 10000
satisfaction_scores <- rnorm(population_size, mean = 7.5, sd = 1.2)

cat("Survey Sampling Analysis:\n")
cat("=======================\n")
cat("Population size (N):", population_size, "\n")
cat("Population mean (μ):", mean(satisfaction_scores), "\n")
cat("Population standard deviation (σ):", sd(satisfaction_scores), "\n")

# Different sampling strategies with mathematical comparison
sample_size <- 100

# Simple random sample with mathematical analysis
srs_sample <- sample(satisfaction_scores, size = sample_size)
srs_mean <- mean(srs_sample)
srs_se <- sd(srs_sample) / sqrt(sample_size)
srs_ci_lower <- srs_mean - 1.96 * srs_se
srs_ci_upper <- srs_mean + 1.96 * srs_se

cat("\nSimple Random Sampling Results:\n")
cat("Sample mean (x̄):", srs_mean, "\n")
cat("Standard error (SE):", srs_se, "\n")
cat("95% Confidence interval: [", srs_ci_lower, ",", srs_ci_upper, "]\n")
cat("Margin of error: ±", 1.96 * srs_se, "\n")

# Stratified sample with mathematical analysis
satisfaction_levels <- cut(satisfaction_scores, breaks = 3, labels = c("Low", "Medium", "High"))
stratified_data <- data.frame(score = satisfaction_scores, level = satisfaction_levels)

# Calculate stratum sizes and proportions
stratum_info <- stratified_data %>%
  group_by(level) %>%
  summarise(
    n = n(),
    proportion = n / nrow(stratified_data),
    stratum_mean = mean(score),
    stratum_sd = sd(score)
  )

cat("\nStratum Information:\n")
print(stratum_info)

# Perform stratified sampling
stratified_sample <- stratified_data %>%
  group_by(level) %>%
  slice_sample(n = ceiling(sample_size / 3)) %>%
  pull(score)

strat_mean <- mean(stratified_sample)
strat_se <- sd(stratified_sample) / sqrt(sample_size)
strat_ci_lower <- strat_mean - 1.96 * strat_se
strat_ci_upper <- strat_mean + 1.96 * strat_se

cat("\nStratified Sampling Results:\n")
cat("Sample mean (x̄):", strat_mean, "\n")
cat("Standard error (SE):", strat_se, "\n")
cat("95% Confidence interval: [", strat_ci_lower, ",", strat_ci_upper, "]\n")
cat("Margin of error: ±", 1.96 * strat_se, "\n")

# Compare sampling methods with mathematical precision
cat("\nSampling Method Comparison:\n")
cat("==========================\n")
cat("Population mean:", mean(satisfaction_scores), "\n")
cat("SRS estimate:", srs_mean, "(error:", srs_mean - mean(satisfaction_scores), ")\n")
cat("Stratified estimate:", strat_mean, "(error:", strat_mean - mean(satisfaction_scores), ")\n")
cat("SRS precision (SE):", srs_se, "\n")
cat("Stratified precision (SE):", strat_se, "\n")
cat("Relative efficiency (SRS/Stratified):", srs_se / strat_se, "\n")

# Plot comparison with mathematical annotations
par(mfrow = c(2, 2))

# Population distribution
hist(satisfaction_scores, main = "Population Distribution",
     xlab = "Satisfaction Score", col = "lightblue", freq = FALSE)
abline(v = mean(satisfaction_scores), col = "red", lwd = 2)

# SRS sample distribution
hist(srs_sample, main = "Simple Random Sample",
     xlab = "Satisfaction Score", col = "lightgreen", freq = FALSE)
abline(v = srs_mean, col = "red", lwd = 2)

# Stratified sample distribution
hist(stratified_sample, main = "Stratified Sample",
     xlab = "Satisfaction Score", col = "lightcoral", freq = FALSE)
abline(v = strat_mean, col = "red", lwd = 2)

# Confidence intervals comparison
ci_comparison <- data.frame(
  Method = c("SRS", "Stratified"),
  Estimate = c(srs_mean, strat_mean),
  SE = c(srs_se, strat_se),
  CI_Lower = c(srs_ci_lower, strat_ci_lower),
  CI_Upper = c(srs_ci_upper, strat_ci_upper)
)

plot(1:2, ci_comparison$Estimate, ylim = c(7, 8),
     main = "Confidence Intervals Comparison",
     xlab = "Sampling Method", ylab = "Satisfaction Score",
     pch = 16, col = c("blue", "red"))
arrows(1:2, ci_comparison$CI_Lower, 1:2, ci_comparison$CI_Upper,
       angle = 90, code = 3, length = 0.1, col = c("blue", "red"))
abline(h = mean(satisfaction_scores), col = "black", lty = 2)
axis(1, at = 1:2, labels = c("SRS", "Stratified"))

par(mfrow = c(1, 1))
```

## Best Practices

### Sample Size Determination with Mathematical Precision

```r
# Function to determine sample size for mean estimation with mathematical foundation
determine_sample_size <- function(population_sd, margin_of_error, confidence_level = 0.95) {
  z_score <- qnorm(1 - (1 - confidence_level) / 2)
  sample_size <- ceiling((z_score * population_sd / margin_of_error)^2)
  
  cat("Sample Size Calculation for Mean Estimation:\n")
  cat("===========================================\n")
  cat("Population standard deviation (σ):", population_sd, "\n")
  cat("Margin of error (E):", margin_of_error, "\n")
  cat("Confidence level:", confidence_level, "\n")
  cat("Z-score (z_{α/2}):", z_score, "\n")
  cat("Required sample size (n):", sample_size, "\n")
  cat("Formula: n = (z_{α/2} × σ / E)²\n\n")
  
  return(sample_size)
}

# Function to determine sample size for proportion estimation
determine_prop_sample_size <- function(expected_prop, margin_of_error, confidence_level = 0.95) {
  z_score <- qnorm(1 - (1 - confidence_level) / 2)
  sample_size <- ceiling((z_score^2 * expected_prop * (1 - expected_prop)) / margin_of_error^2)
  
  cat("Sample Size Calculation for Proportion Estimation:\n")
  cat("================================================\n")
  cat("Expected proportion (p):", expected_prop, "\n")
  cat("Margin of error (E):", margin_of_error, "\n")
  cat("Confidence level:", confidence_level, "\n")
  cat("Z-score (z_{α/2}):", z_score, "\n")
  cat("Required sample size (n):", sample_size, "\n")
  cat("Formula: n = (z_{α/2}² × p(1-p)) / E²\n\n")
  
  return(sample_size)
}

# Example calculations with mathematical interpretation
population_sd <- sd(mtcars$mpg)
margin_of_error <- 2  # MPG units

required_sample_size <- determine_sample_size(population_sd, margin_of_error)

# Sample size for proportion
expected_prop <- 0.5  # Conservative estimate
margin_of_error_prop <- 0.05  # 5 percentage points

required_prop_sample_size <- determine_prop_sample_size(expected_prop, margin_of_error_prop)

# Finite population correction
finite_pop_correction <- function(sample_size, population_size) {
  corrected_size <- ceiling(sample_size / (1 + sample_size / population_size))
  
  cat("Finite Population Correction:\n")
  cat("============================\n")
  cat("Original sample size:", sample_size, "\n")
  cat("Population size (N):", population_size, "\n")
  cat("Corrected sample size:", corrected_size, "\n")
  cat("Reduction:", sample_size - corrected_size, "samples\n")
  cat("Formula: n' = n / (1 + n/N)\n\n")
  
  return(corrected_size)
}

# Apply finite population correction
population_size_example <- 1000
corrected_sample_size <- finite_pop_correction(required_sample_size, population_size_example)
```

### Sampling Bias and Errors with Mathematical Analysis

```r
# Function to check for sampling bias with comprehensive analysis
check_sampling_bias <- function(population, sample_data, variable) {
  pop_mean <- mean(population[[variable]])
  pop_sd <- sd(population[[variable]])
  
  sample_mean <- mean(sample_data[[variable]])
  sample_sd <- sd(sample_data[[variable]])
  
  # Calculate bias with mathematical interpretation
  bias <- sample_mean - pop_mean
  relative_bias <- (bias / pop_mean) * 100
  
  # Calculate efficiency with mathematical foundation
  theoretical_se <- pop_sd / sqrt(length(sample_data[[variable]]))
  empirical_se <- sample_sd / sqrt(length(sample_data[[variable]]))
  efficiency <- (theoretical_se^2) / (empirical_se^2)
  
  # Calculate precision
  precision <- 1 / (empirical_se^2)
  
  cat("Sampling Bias Analysis:\n")
  cat("======================\n")
  cat("Population mean (μ):", pop_mean, "\n")
  cat("Sample mean (x̄):", sample_mean, "\n")
  cat("Absolute bias (x̄ - μ):", bias, "\n")
  cat("Relative bias (%):", relative_bias, "%\n")
  cat("Population standard deviation (σ):", pop_sd, "\n")
  cat("Sample standard deviation (s):", sample_sd, "\n")
  cat("Theoretical standard error (σ/√n):", theoretical_se, "\n")
  cat("Empirical standard error (s/√n):", empirical_se, "\n")
  cat("Efficiency (σ²/s²):", efficiency, "\n")
  cat("Precision (1/SE²):", precision, "\n")
  
  # Statistical significance of bias
  t_statistic <- bias / empirical_se
  p_value <- 2 * (1 - pt(abs(t_statistic), df = length(sample_data[[variable]]) - 1))
  
  cat("t-statistic for bias test:", t_statistic, "\n")
  cat("p-value for bias test:", p_value, "\n")
  cat("Bias statistically significant:", p_value < 0.05, "\n")
  
  return(list(
    bias = bias,
    relative_bias = relative_bias,
    efficiency = efficiency,
    precision = precision,
    t_statistic = t_statistic,
    p_value = p_value
  ))
}

# Example bias check with mathematical analysis
population_data <- mtcars
sample_data <- mtcars[sample(nrow(mtcars), 15), ]

bias_check <- check_sampling_bias(population_data, sample_data, "mpg")

# Visual analysis of bias
par(mfrow = c(2, 2))

# Population vs sample distribution
hist(population_data$mpg, main = "Population Distribution",
     xlab = "MPG", col = "lightblue", freq = FALSE)
abline(v = mean(population_data$mpg), col = "red", lwd = 2)

hist(sample_data$mpg, main = "Sample Distribution",
     xlab = "MPG", col = "lightgreen", freq = FALSE)
abline(v = mean(sample_data$mpg), col = "red", lwd = 2)

# Bias visualization
bias_data <- data.frame(
  Type = c("Population", "Sample"),
  Mean = c(mean(population_data$mpg), mean(sample_data$mpg)),
  SE = c(sd(population_data$mpg)/sqrt(nrow(population_data)), 
         sd(sample_data$mpg)/sqrt(nrow(sample_data)))
)

barplot(bias_data$Mean, names.arg = bias_data$Type,
        main = "Mean Comparison",
        ylab = "Mean MPG", col = c("lightblue", "lightgreen"))

# Precision comparison
barplot(c(1/bias_data$SE[1]^2, 1/bias_data$SE[2]^2), 
        names.arg = c("Population", "Sample"),
        main = "Precision Comparison",
        ylab = "Precision (1/SE²)", col = c("lightblue", "lightgreen"))

par(mfrow = c(1, 1))
```

## Exercises

### Exercise 1: Sampling Distribution Simulation with Mathematical Verification
Simulate the sampling distribution of the mean for sample sizes 5, 10, 20, and 50. Compare the standard errors and verify that they follow the theoretical relationship $\text{SE} = \sigma/\sqrt{n}$.

### Exercise 2: Central Limit Theorem Demonstration with Mathematical Analysis
Demonstrate the Central Limit Theorem using different population distributions (uniform, exponential, chi-square). Verify that the sampling distribution approaches normality and that the mean and standard error match theoretical predictions.

### Exercise 3: Bootstrap Confidence Intervals with Mathematical Interpretation
Use bootstrap sampling to estimate confidence intervals for the median and standard deviation of the mtcars MPG data. Compare bootstrap intervals with theoretical intervals and interpret the differences.

### Exercise 4: Sampling Methods Comparison with Statistical Analysis
Compare simple random sampling, systematic sampling, and stratified sampling for estimating the mean MPG. Calculate efficiency measures and determine which method provides the most precise estimates.

### Exercise 5: Sample Size Determination with Mathematical Foundation
Determine the required sample size for estimating the mean MPG with a margin of error of ±1 MPG at 95% confidence. Verify your calculation using simulation.

### Exercise 6: Sampling Bias Analysis with Statistical Testing
Generate a biased sample and analyze the bias using statistical tests. Calculate the magnitude and significance of the bias and suggest methods to reduce it.

### Exercise 7: Advanced Bootstrap Applications
Use bootstrap sampling to estimate the sampling distribution of the correlation coefficient between MPG and weight in the mtcars dataset. Compare with parametric methods.

### Exercise 8: Stratified Sampling Optimization
Design an optimal stratified sampling plan for the mtcars dataset using different stratification variables. Compare the efficiency of different stratification schemes.

## Next Steps

In the next chapter, we'll learn about confidence intervals, which build upon our understanding of sampling distributions and provide the foundation for statistical inference and hypothesis testing.

---

**Key Takeaways:**
- Sampling distributions show how sample statistics vary across different samples
- Central Limit Theorem applies to sample means from any distribution with finite variance
- Bootstrap sampling is useful for non-parametric inference and complex statistics
- Sample size affects the precision of estimates according to the square root law
- Different sampling methods have different statistical properties and efficiency
- Always consider potential sampling bias and use appropriate statistical tests
- Use mathematical foundations to determine appropriate sample sizes and validate results
- Understanding sampling distributions is crucial for statistical inference and decision-making 