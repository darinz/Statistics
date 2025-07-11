# Sampling and Sampling Distributions

## Overview

Sampling distributions are fundamental to statistical inference. They describe how sample statistics vary across different samples from the same population. Understanding sampling distributions is crucial for hypothesis testing and confidence intervals.

## Basic Sampling Concepts

### Simple Random Sampling

```r
# Load sample data
data(mtcars)

# Simple random sampling
set.seed(123)
sample_size <- 10
random_sample <- mtcars[sample(nrow(mtcars), sample_size), ]

# View sample
print(random_sample)

# Calculate sample statistics
sample_mean <- mean(random_sample$mpg)
sample_sd <- sd(random_sample$mpg)
cat("Sample mean:", sample_mean, "\n")
cat("Sample SD:", sample_sd, "\n")
```

### Systematic Sampling

```r
# Systematic sampling
n <- nrow(mtcars)
k <- floor(n / sample_size)  # Sampling interval
systematic_sample <- mtcars[seq(1, n, by = k)[1:sample_size], ]

# View systematic sample
print(systematic_sample)
```

### Stratified Sampling

```r
# Stratified sampling by cylinders
library(dplyr)

# Calculate sample size per stratum
strata_sizes <- mtcars %>%
  group_by(cyl) %>%
  summarise(n = n()) %>%
  mutate(sample_size = ceiling(n * sample_size / nrow(mtcars)))

# Perform stratified sampling
stratified_sample <- mtcars %>%
  group_by(cyl) %>%
  slice_sample(n = first(strata_sizes$sample_size)) %>%
  ungroup()

print(stratified_sample)
```

## Sampling Distribution of the Mean

### Simulating Sampling Distribution

```r
# Simulate sampling distribution of the mean
set.seed(123)
population_mean <- mean(mtcars$mpg)
population_sd <- sd(mtcars$mpg)
n_samples <- 1000
sample_size <- 10

# Generate sampling distribution
sample_means <- numeric(n_samples)
for (i in 1:n_samples) {
  sample_data <- sample(mtcars$mpg, size = sample_size, replace = TRUE)
  sample_means[i] <- mean(sample_data)
}

# Analyze sampling distribution
sampling_mean <- mean(sample_means)
sampling_sd <- sd(sample_means)
theoretical_se <- population_sd / sqrt(sample_size)

cat("Population mean:", population_mean, "\n")
cat("Population SD:", population_sd, "\n")
cat("Sampling distribution mean:", sampling_mean, "\n")
cat("Sampling distribution SD:", sampling_sd, "\n")
cat("Theoretical standard error:", theoretical_se, "\n")
```

### Visualizing Sampling Distribution

```r
# Plot sampling distribution
par(mfrow = c(2, 2))

# Histogram of sample means
hist(sample_means, main = "Sampling Distribution of Mean",
     xlab = "Sample Mean", col = "lightblue", freq = FALSE)

# Add normal curve
x <- seq(min(sample_means), max(sample_means), length.out = 100)
curve(dnorm(x, mean = sampling_mean, sd = sampling_sd), 
      add = TRUE, col = "red", lwd = 2)

# Q-Q plot
qqnorm(sample_means, main = "Q-Q Plot of Sample Means")
qqline(sample_means, col = "red")

# Box plot
boxplot(sample_means, main = "Box Plot of Sample Means", col = "lightgreen")

# Density plot
plot(density(sample_means), main = "Density Plot of Sample Means")
curve(dnorm(x, mean = sampling_mean, sd = sampling_sd), 
      add = TRUE, col = "red", lty = 2)

par(mfrow = c(1, 1))
```

## Central Limit Theorem

### Demonstrating CLT

```r
# Demonstrate Central Limit Theorem with different distributions
set.seed(123)

# Function to simulate sampling distribution
simulate_sampling_dist <- function(population, sample_size, n_samples = 1000) {
  sample_means <- numeric(n_samples)
  for (i in 1:n_samples) {
    sample_data <- sample(population, size = sample_size, replace = TRUE)
    sample_means[i] <- mean(sample_data)
  }
  return(sample_means)
}

# Different population distributions
n <- 1000

# Uniform distribution
uniform_pop <- runif(n, 0, 10)
uniform_means <- simulate_sampling_dist(uniform_pop, 30)

# Exponential distribution
exp_pop <- rexp(n, rate = 0.5)
exp_means <- simulate_sampling_dist(exp_pop, 30)

# Skewed distribution
skewed_pop <- rgamma(n, shape = 2, rate = 1)
skewed_means <- simulate_sampling_dist(skewed_pop, 30)

# Plot results
par(mfrow = c(3, 2))

# Uniform distribution
hist(uniform_pop, main = "Uniform Population", col = "lightblue")
hist(uniform_means, main = "Sampling Distribution (n=30)", col = "lightgreen")

# Exponential distribution
hist(exp_pop, main = "Exponential Population", col = "lightcoral")
hist(exp_means, main = "Sampling Distribution (n=30)", col = "lightyellow")

# Skewed distribution
hist(skewed_pop, main = "Skewed Population", col = "lightpink")
hist(skewed_means, main = "Sampling Distribution (n=30)", col = "lightcyan")

par(mfrow = c(1, 1))
```

### Effect of Sample Size

```r
# Study effect of sample size on sampling distribution
sample_sizes <- c(5, 10, 30, 50)
n_samples <- 1000

# Simulate for different sample sizes
sampling_distributions <- list()
for (size in sample_sizes) {
  sample_means <- numeric(n_samples)
  for (i in 1:n_samples) {
    sample_data <- sample(mtcars$mpg, size = size, replace = TRUE)
    sample_means[i] <- mean(sample_data)
  }
  sampling_distributions[[as.character(size)]] <- sample_means
}

# Plot results
par(mfrow = c(2, 2))
for (i in 1:length(sample_sizes)) {
  size <- sample_sizes[i]
  means <- sampling_distributions[[as.character(size)]]
  
  hist(means, main = paste("Sample Size =", size),
       xlab = "Sample Mean", col = "lightblue", freq = FALSE)
  
  # Add normal curve
  x <- seq(min(means), max(means), length.out = 100)
  theoretical_mean <- mean(mtcars$mpg)
  theoretical_se <- sd(mtcars$mpg) / sqrt(size)
  curve(dnorm(x, mean = theoretical_mean, sd = theoretical_se), 
        add = TRUE, col = "red", lwd = 2)
}

par(mfrow = c(1, 1))

# Compare standard errors
theoretical_ses <- sd(mtcars$mpg) / sqrt(sample_sizes)
empirical_ses <- sapply(sampling_distributions, sd)

comparison <- data.frame(
  Sample_Size = sample_sizes,
  Theoretical_SE = theoretical_ses,
  Empirical_SE = empirical_ses
)

print(comparison)
```

## Sampling Distribution of the Proportion

### Simulating Proportion Sampling

```r
# Simulate sampling distribution of proportion
set.seed(123)
population_prop <- mean(mtcars$am)  # Proportion of manual transmissions
n_samples <- 1000
sample_size <- 20

# Generate sampling distribution of proportion
sample_proportions <- numeric(n_samples)
for (i in 1:n_samples) {
  sample_data <- sample(mtcars$am, size = sample_size, replace = TRUE)
  sample_proportions[i] <- mean(sample_data)
}

# Analyze sampling distribution
sampling_prop_mean <- mean(sample_proportions)
sampling_prop_sd <- sd(sample_proportions)
theoretical_prop_se <- sqrt(population_prop * (1 - population_prop) / sample_size)

cat("Population proportion:", population_prop, "\n")
cat("Sampling distribution mean:", sampling_prop_mean, "\n")
cat("Sampling distribution SD:", sampling_prop_sd, "\n")
cat("Theoretical standard error:", theoretical_prop_se, "\n")

# Plot sampling distribution
hist(sample_proportions, main = "Sampling Distribution of Proportion",
     xlab = "Sample Proportion", col = "lightgreen", freq = FALSE)

# Add normal curve
x <- seq(min(sample_proportions), max(sample_proportions), length.out = 100)
curve(dnorm(x, mean = sampling_prop_mean, sd = sampling_prop_sd), 
      add = TRUE, col = "red", lwd = 2)
```

## Sampling Distribution of the Variance

### Simulating Variance Sampling

```r
# Simulate sampling distribution of variance
set.seed(123)
population_var <- var(mtcars$mpg)
n_samples <- 1000
sample_size <- 10

# Generate sampling distribution of variance
sample_variances <- numeric(n_samples)
for (i in 1:n_samples) {
  sample_data <- sample(mtcars$mpg, size = sample_size, replace = TRUE)
  sample_variances[i] <- var(sample_data)
}

# Analyze sampling distribution
sampling_var_mean <- mean(sample_variances)
sampling_var_sd <- sd(sample_variances)

cat("Population variance:", population_var, "\n")
cat("Sampling distribution mean:", sampling_var_mean, "\n")
cat("Sampling distribution SD:", sampling_var_sd, "\n")

# Plot sampling distribution
hist(sample_variances, main = "Sampling Distribution of Variance",
     xlab = "Sample Variance", col = "lightcoral", freq = FALSE)

# Add theoretical chi-square distribution
df <- sample_size - 1
theoretical_var <- population_var * df / qchisq(ppoints(length(sample_variances)), df)
hist(theoretical_var, add = TRUE, col = "red", alpha = 0.3, freq = FALSE)
```

## Bootstrap Sampling

### Basic Bootstrap

```r
# Bootstrap sampling
library(boot)

# Bootstrap function for mean
boot_mean <- function(data, indices) {
  return(mean(data[indices]))
}

# Perform bootstrap
set.seed(123)
boot_results <- boot(mtcars$mpg, boot_mean, R = 1000)

# Bootstrap confidence interval
boot_ci <- boot.ci(boot_results, type = "perc")
print(boot_ci)

# Plot bootstrap distribution
hist(boot_results$t, main = "Bootstrap Distribution of Mean",
     xlab = "Bootstrap Mean", col = "lightblue", freq = FALSE)

# Add confidence interval lines
abline(v = boot_ci$percent[4:5], col = "red", lty = 2, lwd = 2)
abline(v = mean(mtcars$mpg), col = "green", lwd = 2)
```

### Bootstrap for Different Statistics

```r
# Bootstrap for median
boot_median <- function(data, indices) {
  return(median(data[indices]))
}

boot_median_results <- boot(mtcars$mpg, boot_median, R = 1000)
boot_median_ci <- boot.ci(boot_median_results, type = "perc")

# Bootstrap for standard deviation
boot_sd <- function(data, indices) {
  return(sd(data[indices]))
}

boot_sd_results <- boot(mtcars$mpg, boot_sd, R = 1000)
boot_sd_ci <- boot.ci(boot_sd_results, type = "perc")

# Compare results
cat("Mean bootstrap CI:", boot_ci$percent[4:5], "\n")
cat("Median bootstrap CI:", boot_median_ci$percent[4:5], "\n")
cat("SD bootstrap CI:", boot_sd_ci$percent[4:5], "\n")
```

## Sampling Methods Comparison

### Comparing Different Sampling Methods

```r
# Function to compare sampling methods
compare_sampling_methods <- function(data, variable, sample_size, n_simulations = 100) {
  # Simple random sampling
  srs_means <- numeric(n_simulations)
  for (i in 1:n_simulations) {
    srs_sample <- sample(data[[variable]], size = sample_size, replace = TRUE)
    srs_means[i] <- mean(srs_sample)
  }
  
  # Systematic sampling
  sys_means <- numeric(n_simulations)
  n <- length(data[[variable]])
  k <- floor(n / sample_size)
  
  for (i in 1:n_simulations) {
    start <- sample(1:k, 1)
    indices <- seq(start, n, by = k)[1:sample_size]
    sys_sample <- data[[variable]][indices]
    sys_means[i] <- mean(sys_sample)
  }
  
  # Stratified sampling (if grouping variable available)
  if ("cyl" %in% names(data)) {
    strat_means <- numeric(n_simulations)
    for (i in 1:n_simulations) {
      strat_sample <- data %>%
        group_by(cyl) %>%
        slice_sample(n = ceiling(sample_size / length(unique(data$cyl)))) %>%
        pull(variable)
      strat_means[i] <- mean(strat_sample)
    }
  } else {
    strat_means <- NULL
  }
  
  # Compare results
  results <- list(
    SRS = srs_means,
    Systematic = sys_means,
    Stratified = strat_means
  )
  
  return(results)
}

# Apply comparison
sampling_comparison <- compare_sampling_methods(mtcars, "mpg", 10)

# Plot comparison
par(mfrow = c(2, 2))

hist(sampling_comparison$SRS, main = "Simple Random Sampling",
     xlab = "Sample Mean", col = "lightblue")
hist(sampling_comparison$Systematic, main = "Systematic Sampling",
     xlab = "Sample Mean", col = "lightgreen")
if (!is.null(sampling_comparison$Stratified)) {
  hist(sampling_comparison$Stratified, main = "Stratified Sampling",
       xlab = "Sample Mean", col = "lightcoral")
}

# Compare standard errors
cat("SRS SE:", sd(sampling_comparison$SRS), "\n")
cat("Systematic SE:", sd(sampling_comparison$Systematic), "\n")
if (!is.null(sampling_comparison$Stratified)) {
  cat("Stratified SE:", sd(sampling_comparison$Stratified), "\n")
}

par(mfrow = c(1, 1))
```

## Practical Examples

### Example 1: Quality Control Sampling

```r
# Simulate quality control scenario
set.seed(123)
production_batch <- rnorm(1000, mean = 100, sd = 5)  # Product weights

# Quality control sampling
n_samples <- 100
sample_size <- 20

qc_means <- numeric(n_samples)
qc_sds <- numeric(n_samples)

for (i in 1:n_samples) {
  sample_data <- sample(production_batch, size = sample_size)
  qc_means[i] <- mean(sample_data)
  qc_sds[i] <- sd(sample_data)
}

# Quality control limits
ucl_mean <- 100 + 2 * (5 / sqrt(sample_size))
lcl_mean <- 100 - 2 * (5 / sqrt(sample_size))

# Plot control chart
plot(qc_means, type = "l", main = "Quality Control Chart",
     xlab = "Sample Number", ylab = "Sample Mean")
abline(h = 100, col = "blue", lty = 2)
abline(h = c(lcl_mean, ucl_mean), col = "red", lty = 2)
```

### Example 2: Survey Sampling

```r
# Simulate survey data
set.seed(123)
population_size <- 10000
satisfaction_scores <- rnorm(population_size, mean = 7.5, sd = 1.2)

# Different sampling strategies
sample_size <- 100

# Simple random sample
srs_sample <- sample(satisfaction_scores, size = sample_size)
srs_mean <- mean(srs_sample)
srs_se <- sd(srs_sample) / sqrt(sample_size)

# Stratified sample (by satisfaction level)
satisfaction_levels <- cut(satisfaction_scores, breaks = 3, labels = c("Low", "Medium", "High"))
stratified_data <- data.frame(score = satisfaction_scores, level = satisfaction_levels)

stratified_sample <- stratified_data %>%
  group_by(level) %>%
  slice_sample(n = ceiling(sample_size / 3)) %>%
  pull(score)

strat_mean <- mean(stratified_sample)
strat_se <- sd(stratified_sample) / sqrt(sample_size)

# Compare results
cat("Population mean:", mean(satisfaction_scores), "\n")
cat("SRS mean:", srs_mean, "±", 1.96 * srs_se, "\n")
cat("Stratified mean:", strat_mean, "±", 1.96 * strat_se, "\n")
```

## Best Practices

### Sample Size Determination

```r
# Function to determine sample size for mean estimation
determine_sample_size <- function(population_sd, margin_of_error, confidence_level = 0.95) {
  z_score <- qnorm(1 - (1 - confidence_level) / 2)
  sample_size <- ceiling((z_score * population_sd / margin_of_error)^2)
  return(sample_size)
}

# Example calculations
population_sd <- sd(mtcars$mpg)
margin_of_error <- 2  # MPG units

required_sample_size <- determine_sample_size(population_sd, margin_of_error)
cat("Required sample size for margin of error ±2 MPG:", required_sample_size, "\n")

# Sample size for proportion
determine_prop_sample_size <- function(expected_prop, margin_of_error, confidence_level = 0.95) {
  z_score <- qnorm(1 - (1 - confidence_level) / 2)
  sample_size <- ceiling((z_score^2 * expected_prop * (1 - expected_prop)) / margin_of_error^2)
  return(sample_size)
}

expected_prop <- 0.5  # Conservative estimate
margin_of_error_prop <- 0.05  # 5 percentage points

required_prop_sample_size <- determine_prop_sample_size(expected_prop, margin_of_error_prop)
cat("Required sample size for proportion with ±5% margin:", required_prop_sample_size, "\n")
```

### Sampling Bias and Errors

```r
# Function to check for sampling bias
check_sampling_bias <- function(population, sample_data, variable) {
  pop_mean <- mean(population[[variable]])
  pop_sd <- sd(population[[variable]])
  
  sample_mean <- mean(sample_data[[variable]])
  sample_sd <- sd(sample_data[[variable]])
  
  # Calculate bias
  bias <- sample_mean - pop_mean
  
  # Calculate relative bias
  relative_bias <- (bias / pop_mean) * 100
  
  # Calculate efficiency
  efficiency <- (pop_sd^2 / length(sample_data[[variable]])) / 
                (sample_sd^2 / length(sample_data[[variable]]))
  
  return(list(
    bias = bias,
    relative_bias = relative_bias,
    efficiency = efficiency
  ))
}

# Example bias check
population_data <- mtcars
sample_data <- mtcars[sample(nrow(mtcars), 15), ]

bias_check <- check_sampling_bias(population_data, sample_data, "mpg")
print(bias_check)
```

## Exercises

### Exercise 1: Sampling Distribution Simulation
Simulate the sampling distribution of the mean for sample sizes 5, 10, 20, and 50. Compare the standard errors.

### Exercise 2: Central Limit Theorem
Demonstrate the Central Limit Theorem using different population distributions (uniform, exponential, chi-square).

### Exercise 3: Bootstrap Confidence Intervals
Use bootstrap sampling to estimate confidence intervals for the median and standard deviation of the mtcars MPG data.

### Exercise 4: Sampling Methods Comparison
Compare simple random sampling, systematic sampling, and stratified sampling for estimating the mean MPG.

### Exercise 5: Sample Size Determination
Determine the required sample size for estimating the mean MPG with a margin of error of ±1 MPG at 95% confidence.

## Next Steps

In the next chapter, we'll learn about confidence intervals, which build upon our understanding of sampling distributions.

---

**Key Takeaways:**
- Sampling distributions show how sample statistics vary
- Central Limit Theorem applies to sample means from any distribution
- Bootstrap sampling is useful for non-parametric inference
- Sample size affects the precision of estimates
- Different sampling methods have different properties
- Always consider potential sampling bias
- Use appropriate sample size calculations for your study 