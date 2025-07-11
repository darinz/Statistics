# Time Series Analysis

## Overview

Time series analysis involves studying data points collected over time to identify patterns, trends, and make predictions. This field is crucial for forecasting, trend analysis, and understanding temporal dependencies in data.

## Basic Time Series Concepts

### Creating Time Series Objects

```r
# Load required packages
library(tseries)
library(forecast)
library(ggplot2)

# Create a simple time series
set.seed(123)
n_periods <- 100
time_index <- 1:n_periods

# Generate time series with trend and noise
trend <- 0.1 * time_index
seasonal <- 5 * sin(2 * pi * time_index / 12)  # Annual seasonality
noise <- rnorm(n_periods, 0, 2)
time_series_data <- 50 + trend + seasonal + noise

# Create time series object
ts_data <- ts(time_series_data, frequency = 12, start = c(2010, 1))
print(ts_data)

# Basic time series properties
cat("Time Series Properties:\n")
cat("Length:", length(ts_data), "\n")
cat("Frequency:", frequency(ts_data), "\n")
cat("Start:", start(ts_data), "\n")
cat("End:", end(ts_data), "\n")
cat("Mean:", round(mean(ts_data), 2), "\n")
cat("Standard deviation:", round(sd(ts_data), 2), "\n")
```

### Time Series Visualization

```r
# Basic time series plot
plot(ts_data, main = "Time Series Plot", ylab = "Value", xlab = "Time")

# Enhanced plot with ggplot2
ts_df <- data.frame(
  time = time(ts_data),
  value = as.numeric(ts_data)
)

ggplot(ts_df, aes(x = time, y = value)) +
  geom_line(color = "steelblue", size = 0.8) +
  geom_smooth(method = "loess", color = "red", se = FALSE) +
  labs(title = "Time Series with Trend Line",
       x = "Time", y = "Value") +
  theme_minimal()

# Seasonal decomposition plot
decomposed_ts <- decompose(ts_data)
plot(decomposed_ts)
```

## Trend Analysis

### Linear Trend Analysis

```r
# Linear trend analysis
trend_analysis <- function(time_series) {
  # Extract time index
  time_index <- 1:length(time_series)
  
  # Fit linear trend
  trend_model <- lm(as.numeric(time_series) ~ time_index)
  
  # Extract coefficients
  intercept <- coef(trend_model)[1]
  slope <- coef(trend_model)[2]
  
  # Calculate trend line
  trend_line <- intercept + slope * time_index
  
  # R-squared
  r_squared <- summary(trend_model)$r.squared
  
  # Statistical significance
  p_value <- summary(trend_model)$coefficients[2, 4]
  
  return(list(
    intercept = intercept,
    slope = slope,
    trend_line = trend_line,
    r_squared = r_squared,
    p_value = p_value,
    model = trend_model
  ))
}

# Apply trend analysis
trend_result <- trend_analysis(ts_data)

cat("Linear Trend Analysis Results:\n")
cat("Intercept:", round(trend_result$intercept, 3), "\n")
cat("Slope:", round(trend_result$slope, 3), "\n")
cat("R-squared:", round(trend_result$r_squared, 3), "\n")
cat("p-value:", round(trend_result$p_value, 4), "\n")

# Visualize trend
ggplot(ts_df, aes(x = time, y = value)) +
  geom_line(color = "steelblue", alpha = 0.7) +
  geom_line(aes(y = trend_result$trend_line), color = "red", size = 1) +
  labs(title = "Time Series with Linear Trend",
       x = "Time", y = "Value") +
  theme_minimal()
```

### Non-linear Trend Analysis

```r
# Polynomial trend analysis
polynomial_trend <- function(time_series, degree = 2) {
  time_index <- 1:length(time_series)
  
  # Fit polynomial trend
  poly_model <- lm(as.numeric(time_series) ~ poly(time_index, degree))
  
  # Calculate fitted values
  fitted_values <- fitted(poly_model)
  
  # R-squared
  r_squared <- summary(poly_model)$r.squared
  
  return(list(
    model = poly_model,
    fitted_values = fitted_values,
    r_squared = r_squared
  ))
}

# Apply polynomial trend analysis
poly_result <- polynomial_trend(ts_data, degree = 2)

cat("Polynomial Trend Analysis Results:\n")
cat("R-squared:", round(poly_result$r_squared, 3), "\n")

# Compare linear vs polynomial
cat("Model Comparison:\n")
cat("Linear R-squared:", round(trend_result$r_squared, 3), "\n")
cat("Polynomial R-squared:", round(poly_result$r_squared, 3), "\n")
```

## Seasonality Analysis

### Seasonal Decomposition

```r
# Seasonal decomposition
seasonal_decomposition <- function(time_series) {
  # Classical decomposition
  classical_decomp <- decompose(time_series)
  
  # STL decomposition (more robust)
  stl_decomp <- stl(time_series, s.window = "periodic")
  
  return(list(
    classical = classical_decomp,
    stl = stl_decomp
  ))
}

# Apply seasonal decomposition
decomp_result <- seasonal_decomposition(ts_data)

# Plot decomposition components
par(mfrow = c(2, 2))
plot(decomp_result$classical$trend, main = "Trend Component")
plot(decomp_result$classical$seasonal, main = "Seasonal Component")
plot(decomp_result$classical$random, main = "Random Component")
plot(ts_data, main = "Original Time Series")
par(mfrow = c(1, 1))

# Extract seasonal strength
seasonal_strength <- function(time_series) {
  decomp <- decompose(time_series)
  
  # Calculate seasonal strength
  seasonal_var <- var(decomp$seasonal, na.rm = TRUE)
  random_var <- var(decomp$random, na.rm = TRUE)
  
  strength <- seasonal_var / (seasonal_var + random_var)
  
  return(strength)
}

seasonal_strength_value <- seasonal_strength(ts_data)
cat("Seasonal Strength:", round(seasonal_strength_value, 3), "\n")
```

### Seasonal Adjustment

```r
# Seasonal adjustment
seasonal_adjustment <- function(time_series) {
  # Classical seasonal adjustment
  decomp <- decompose(time_series)
  seasonally_adjusted <- time_series - decomp$seasonal
  
  # STL seasonal adjustment
  stl_decomp <- stl(time_series, s.window = "periodic")
  stl_adjusted <- seasadj(stl_decomp)
  
  return(list(
    classical_adjusted = seasonally_adjusted,
    stl_adjusted = stl_adjusted
  ))
}

# Apply seasonal adjustment
adjusted_result <- seasonal_adjustment(ts_data)

# Visualize seasonal adjustment
adjusted_df <- data.frame(
  time = time(ts_data),
  original = as.numeric(ts_data),
  classical_adjusted = as.numeric(adjusted_result$classical_adjusted),
  stl_adjusted = as.numeric(adjusted_result$stl_adjusted)
)

ggplot(adjusted_df, aes(x = time)) +
  geom_line(aes(y = original, color = "Original"), alpha = 0.7) +
  geom_line(aes(y = classical_adjusted, color = "Classically Adjusted"), alpha = 0.7) +
  geom_line(aes(y = stl_adjusted, color = "STL Adjusted"), alpha = 0.7) +
  labs(title = "Seasonal Adjustment Comparison",
       x = "Time", y = "Value", color = "Series") +
  theme_minimal()
```

## Stationarity Analysis

### Stationarity Tests

```r
# Augmented Dickey-Fuller test
adf_test <- adf.test(ts_data)
print(adf_test)

# Kwiatkowski-Phillips-Schmidt-Shin test
kpss_test <- kpss.test(ts_data)
print(kpss_test)

# Phillips-Perron test
pp_test <- pp.test(ts_data)
print(pp_test)

cat("Stationarity Test Results:\n")
cat("ADF p-value:", round(adf_test$p.value, 4), "\n")
cat("KPSS p-value:", round(kpss_test$p.value, 4), "\n")
cat("PP p-value:", round(pp_test$p.value, 4), "\n")

# Interpretation
cat("\nInterpretation:\n")
if (adf_test$p.value < 0.05) {
  cat("- ADF test: Series is stationary\n")
} else {
  cat("- ADF test: Series is non-stationary\n")
}

if (kpss_test$p.value > 0.05) {
  cat("- KPSS test: Series is stationary\n")
} else {
  cat("- KPSS test: Series is non-stationary\n")
}
```

### Differencing for Stationarity

```r
# First differencing
first_diff <- diff(ts_data)
adf_diff_test <- adf.test(first_diff)
print(adf_diff_test)

# Second differencing
second_diff <- diff(first_diff)
adf_diff2_test <- adf.test(second_diff)
print(adf_diff2_test)

# Seasonal differencing
seasonal_diff <- diff(ts_data, lag = 12)
adf_seasonal_test <- adf.test(seasonal_diff)
print(adf_seasonal_test)

cat("Differencing Results:\n")
cat("First difference ADF p-value:", round(adf_diff_test$p.value, 4), "\n")
cat("Second difference ADF p-value:", round(adf_diff2_test$p.value, 4), "\n")
cat("Seasonal difference ADF p-value:", round(adf_seasonal_test$p.value, 4), "\n")
```

## Autocorrelation Analysis

### Autocorrelation Functions

```r
# Autocorrelation function (ACF)
acf_result <- acf(ts_data, lag.max = 24, plot = FALSE)
print(acf_result)

# Partial autocorrelation function (PACF)
pacf_result <- pacf(ts_data, lag.max = 24, plot = FALSE)
print(pacf_result)

# Plot ACF and PACF
par(mfrow = c(2, 1))
acf(ts_data, lag.max = 24, main = "Autocorrelation Function")
pacf(ts_data, lag.max = 24, main = "Partial Autocorrelation Function")
par(mfrow = c(1, 1))

# Identify significant lags
significant_acf <- which(abs(acf_result$acf) > 2/sqrt(length(ts_data)))
significant_pacf <- which(abs(pacf_result$acf) > 2/sqrt(length(ts_data)))

cat("Significant ACF lags:", significant_acf, "\n")
cat("Significant PACF lags:", significant_pacf, "\n")
```

## ARIMA Modeling

### ARIMA Model Identification

```r
# Automatic ARIMA model selection
auto_arima <- auto.arima(ts_data, seasonal = TRUE)
print(auto_arima)

# Extract model parameters
arima_order <- auto_arima$arma
cat("ARIMA Model Parameters:\n")
cat("p (AR order):", arima_order[1], "\n")
cat("d (differencing):", arima_order[6], "\n")
cat("q (MA order):", arima_order[2], "\n")
cat("P (seasonal AR):", arima_order[3], "\n")
cat("D (seasonal differencing):", arima_order[7], "\n")
cat("Q (seasonal MA):", arima_order[4], "\n")

# Model diagnostics
checkresiduals(auto_arima)

# AIC and BIC
cat("Model Selection Criteria:\n")
cat("AIC:", round(AIC(auto_arima), 2), "\n")
cat("BIC:", round(BIC(auto_arima), 2), "\n")
```

### Manual ARIMA Model Fitting

```r
# Manual ARIMA model fitting
manual_arima <- arima(ts_data, order = c(1, 1, 1), seasonal = list(order = c(1, 1, 1), period = 12))
print(manual_arima)

# Compare models
cat("Model Comparison:\n")
cat("Auto ARIMA AIC:", round(AIC(auto_arima), 2), "\n")
cat("Manual ARIMA AIC:", round(AIC(manual_arima), 2), "\n")

# Residual analysis
residuals_auto <- residuals(auto_arima)
residuals_manual <- residuals(manual_arima)

# Ljung-Box test for residuals
ljung_auto <- Box.test(residuals_auto, type = "Ljung-Box")
ljung_manual <- Box.test(residuals_manual, type = "Ljung-Box")

cat("Ljung-Box Test Results:\n")
cat("Auto ARIMA p-value:", round(ljung_auto$p.value, 4), "\n")
cat("Manual ARIMA p-value:", round(ljung_manual$p.value, 4), "\n")
```

## Forecasting

### Point Forecasts

```r
# Generate forecasts
forecast_periods <- 12
forecast_result <- forecast(auto_arima, h = forecast_periods)
print(forecast_result)

# Extract forecast components
point_forecast <- forecast_result$mean
lower_ci <- forecast_result$lower[, 2]  # 95% CI
upper_ci <- forecast_result$upper[, 2]

# Create forecast data frame
forecast_df <- data.frame(
  time = seq(length(ts_data) + 1, length(ts_data) + forecast_periods),
  forecast = as.numeric(point_forecast),
  lower = as.numeric(lower_ci),
  upper = as.numeric(upper_ci)
)

# Visualize forecast
ggplot() +
  geom_line(data = ts_df, aes(x = time, y = value), color = "steelblue") +
  geom_line(data = forecast_df, aes(x = time, y = forecast), color = "red", size = 1) +
  geom_ribbon(data = forecast_df, aes(x = time, ymin = lower, ymax = upper), 
              alpha = 0.3, fill = "red") +
  labs(title = "Time Series Forecast",
       x = "Time", y = "Value") +
  theme_minimal()
```

### Forecast Accuracy

```r
# Split data for forecast evaluation
train_size <- floor(0.8 * length(ts_data))
train_data <- ts_data[1:train_size]
test_data <- ts_data[(train_size + 1):length(ts_data)]

# Fit model on training data
train_arima <- auto.arima(train_data, seasonal = TRUE)

# Generate forecasts
forecast_test <- forecast(train_arima, h = length(test_data))

# Calculate accuracy measures
accuracy_measures <- accuracy(forecast_test, test_data)
print(accuracy_measures)

# Mean Absolute Error (MAE)
mae <- mean(abs(forecast_test$mean - test_data))
cat("Mean Absolute Error:", round(mae, 2), "\n")

# Root Mean Square Error (RMSE)
rmse <- sqrt(mean((forecast_test$mean - test_data)^2))
cat("Root Mean Square Error:", round(rmse, 2), "\n")

# Mean Absolute Percentage Error (MAPE)
mape <- mean(abs((test_data - forecast_test$mean) / test_data)) * 100
cat("Mean Absolute Percentage Error:", round(mape, 2), "%\n")
```

## Advanced Time Series Models

### Exponential Smoothing

```r
# Simple exponential smoothing
ses_model <- ses(ts_data, h = 12)
print(ses_model)

# Holt's method (trend)
holt_model <- holt(ts_data, h = 12)
print(holt_model)

# Holt-Winters method (trend + seasonality)
hw_model <- hw(ts_data, h = 12)
print(hw_model)

# Compare models
cat("Model Comparison (AIC):\n")
cat("SES:", round(AIC(ses_model), 2), "\n")
cat("Holt:", round(AIC(holt_model), 2), "\n")
cat("Holt-Winters:", round(AIC(hw_model), 2), "\n")
```

### Vector Autoregression (VAR)

```r
# Create multivariate time series
set.seed(123)
n_periods <- 100
var_data <- data.frame(
  y1 = cumsum(rnorm(n_periods, 0, 1)),
  y2 = cumsum(rnorm(n_periods, 0, 1)) + 0.5 * cumsum(rnorm(n_periods, 0, 1))
)

# Convert to time series
var_ts <- ts(var_data, frequency = 12, start = c(2010, 1))

# Fit VAR model
library(vars)
var_model <- VAR(var_ts, p = 2, type = "const")
print(var_model)

# Granger causality test
granger_test <- causality(var_model, cause = "y1")
print(granger_test)

# Impulse response function
irf_result <- irf(var_model, impulse = "y1", response = "y2", n.ahead = 10)
plot(irf_result)
```

## Practical Examples

### Example 1: Economic Data Analysis

```r
# Simulate economic time series
set.seed(123)
n_periods <- 120

# Generate GDP data with trend and seasonality
time_index <- 1:n_periods
trend <- 0.02 * time_index
seasonal <- 2 * sin(2 * pi * time_index / 4)  # Quarterly seasonality
noise <- rnorm(n_periods, 0, 1)
gdp_data <- 100 + trend + seasonal + noise

# Create time series
gdp_ts <- ts(gdp_data, frequency = 4, start = c(2010, 1))

# Analyze GDP data
gdp_decomp <- decompose(gdp_ts)
plot(gdp_decomp)

# Fit ARIMA model
gdp_arima <- auto.arima(gdp_ts)
print(gdp_arima)

# Generate forecasts
gdp_forecast <- forecast(gdp_arima, h = 8)
plot(gdp_forecast)
```

### Example 2: Sales Data Analysis

```r
# Simulate sales data
set.seed(123)
n_periods <- 60

# Generate sales with trend, seasonality, and cycles
time_index <- 1:n_periods
trend <- 0.5 * time_index
seasonal <- 10 * sin(2 * pi * time_index / 12)  # Monthly seasonality
cycle <- 5 * sin(2 * pi * time_index / 24)  # Biennial cycle
noise <- rnorm(n_periods, 0, 3)
sales_data <- 50 + trend + seasonal + cycle + noise

# Create time series
sales_ts <- ts(sales_data, frequency = 12, start = c(2015, 1))

# Seasonal decomposition
sales_decomp <- stl(sales_ts, s.window = "periodic")
plot(sales_decomp)

# Fit multiple models
sales_ses <- ses(sales_ts, h = 12)
sales_holt <- holt(sales_ts, h = 12)
sales_hw <- hw(sales_ts, h = 12)
sales_arima <- auto.arima(sales_ts)

# Compare forecast accuracy
accuracy_comparison <- data.frame(
  Model = c("SES", "Holt", "Holt-Winters", "ARIMA"),
  AIC = c(AIC(sales_ses), AIC(sales_holt), AIC(sales_hw), AIC(sales_arima))
)
print(accuracy_comparison)
```

### Example 3: Weather Data Analysis

```r
# Simulate temperature data
set.seed(123)
n_periods <- 365

# Generate daily temperature with annual seasonality
time_index <- 1:n_periods
annual_trend <- 0.001 * time_index  # Slight warming trend
seasonal <- 15 * sin(2 * pi * time_index / 365)  # Annual seasonality
noise <- rnorm(n_periods, 0, 2)
temperature_data <- 20 + annual_trend + seasonal + noise

# Create time series
temp_ts <- ts(temperature_data, frequency = 365, start = c(2020, 1))

# Analyze temperature data
temp_decomp <- decompose(temp_ts)
plot(temp_decomp)

# Fit seasonal ARIMA model
temp_arima <- auto.arima(temp_ts, seasonal = TRUE)
print(temp_arima)

# Generate seasonal forecasts
temp_forecast <- forecast(temp_arima, h = 30)
plot(temp_forecast)
```

## Best Practices

### Model Selection Guidelines

```r
# Function to help choose appropriate time series model
choose_time_series_model <- function(time_series) {
  cat("=== TIME SERIES MODEL SELECTION ===\n")
  
  # Check stationarity
  adf_result <- adf.test(time_series)
  cat("ADF test p-value:", round(adf_result$p.value, 4), "\n")
  
  # Check seasonality
  decomp <- decompose(time_series)
  seasonal_strength <- var(decomp$seasonal, na.rm = TRUE) / 
                      (var(decomp$seasonal, na.rm = TRUE) + var(decomp$random, na.rm = TRUE))
  cat("Seasonal strength:", round(seasonal_strength, 3), "\n")
  
  # Check trend
  trend_model <- lm(as.numeric(time_series) ~ time(time_series))
  trend_p_value <- summary(trend_model)$coefficients[2, 4]
  cat("Trend significance p-value:", round(trend_p_value, 4), "\n")
  
  cat("\nRECOMMENDATIONS:\n")
  
  if (adf_result$p.value < 0.05) {
    cat("- Data is stationary\n")
    if (seasonal_strength > 0.1) {
      cat("- Use seasonal ARIMA or Holt-Winters\n")
    } else {
      cat("- Use ARIMA or exponential smoothing\n")
    }
  } else {
    cat("- Data is non-stationary\n")
    cat("- Use differencing or trend-stationary models\n")
  }
  
  if (trend_p_value < 0.05) {
    cat("- Significant trend detected\n")
    cat("- Consider Holt's method or trend-stationary ARIMA\n")
  }
  
  return(list(
    stationary = adf_result$p.value < 0.05,
    seasonal_strength = seasonal_strength,
    trend_significant = trend_p_value < 0.05
  ))
}

# Apply model selection
model_selection <- choose_time_series_model(ts_data)
```

### Reporting Guidelines

```r
# Function to generate comprehensive time series report
generate_time_series_report <- function(time_series, model, forecast_result) {
  cat("=== TIME SERIES ANALYSIS REPORT ===\n\n")
  
  # Data summary
  cat("DATA SUMMARY:\n")
  cat("Length:", length(time_series), "\n")
  cat("Frequency:", frequency(time_series), "\n")
  cat("Start:", start(time_series), "\n")
  cat("End:", end(time_series), "\n")
  cat("Mean:", round(mean(time_series), 2), "\n")
  cat("Standard deviation:", round(sd(time_series), 2), "\n\n")
  
  # Model summary
  cat("MODEL SUMMARY:\n")
  if (inherits(model, "Arima")) {
    cat("Model type: ARIMA\n")
    cat("Order:", paste(model$arma[1:3], collapse = ","), "\n")
    cat("Seasonal order:", paste(model$arma[3:5], collapse = ","), "\n")
  } else if (inherits(model, "ets")) {
    cat("Model type: Exponential Smoothing\n")
    cat("Method:", model$method, "\n")
  }
  cat("AIC:", round(AIC(model), 2), "\n")
  cat("BIC:", round(BIC(model), 2), "\n\n")
  
  # Forecast summary
  cat("FORECAST SUMMARY:\n")
  cat("Forecast periods:", length(forecast_result$mean), "\n")
  cat("Point forecast range:", round(range(forecast_result$mean), 2), "\n")
  cat("95% CI range:", round(range(forecast_result$lower[, 2]), 2), "to", 
      round(range(forecast_result$upper[, 2]), 2), "\n\n")
  
  # Residual analysis
  residuals_model <- residuals(model)
  cat("RESIDUAL ANALYSIS:\n")
  cat("Mean residual:", round(mean(residuals_model), 4), "\n")
  cat("Residual standard deviation:", round(sd(residuals_model), 4), "\n")
  
  # Ljung-Box test
  ljung_test <- Box.test(residuals_model, type = "Ljung-Box")
  cat("Ljung-Box test p-value:", round(ljung_test$p.value, 4), "\n")
  
  if (ljung_test$p.value > 0.05) {
    cat("Residuals appear to be white noise\n")
  } else {
    cat("Residuals may not be white noise\n")
  }
}

# Generate report
generate_time_series_report(ts_data, auto_arima, forecast_result)
```

## Exercises

### Exercise 1: Trend Analysis
Analyze the trend component of a time series and compare linear vs non-linear trend models.

### Exercise 2: Seasonality Detection
Identify and analyze seasonal patterns in time series data using decomposition methods.

### Exercise 3: Stationarity Testing
Perform comprehensive stationarity tests and apply appropriate transformations.

### Exercise 4: ARIMA Modeling
Fit ARIMA models to time series data and evaluate model diagnostics.

### Exercise 5: Forecasting
Generate forecasts using different models and evaluate forecast accuracy.

## Next Steps

In the next chapter, we'll learn about multivariate analysis techniques.

---

**Key Takeaways:**
- Time series analysis requires understanding of trend, seasonality, and stationarity
- Proper model selection depends on data characteristics
- ARIMA models are powerful for univariate time series forecasting
- Exponential smoothing is useful for trend and seasonal data
- Always check model diagnostics and forecast accuracy
- Seasonal decomposition helps understand data components
- Stationarity is crucial for many time series models
- Proper reporting includes data summary, model diagnostics, and forecast evaluation 