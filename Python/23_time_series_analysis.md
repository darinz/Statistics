# Time Series Analysis

## Introduction

Time series analysis is a statistical methodology for analyzing data points collected sequentially over time. It involves identifying patterns, trends, seasonality, and dependencies to understand temporal behavior and make predictions about future values.

**Key Applications:**
- Economic forecasting (GDP, inflation, stock prices)
- Weather prediction and climate analysis
- Sales forecasting and demand planning
- Quality control and process monitoring
- Financial market analysis
- Epidemiology and disease modeling

**Components of Time Series:**
- **Trend:** Long-term systematic change (linear, polynomial, exponential)
- **Seasonality:** Regular periodic patterns (daily, weekly, monthly, yearly)
- **Cycles:** Irregular periodic patterns with varying lengths
- **Random/Noise:** Unpredictable fluctuations

## Mathematical Foundations

### Time Series Components

A time series $`Y_t`$ can be decomposed into its components:

```math
Y_t = T_t + S_t + C_t + R_t
```

where:
- $`T_t`$ = Trend component at time $`t`$
- $`S_t`$ = Seasonal component at time $`t`$
- $`C_t`$ = Cyclical component at time $`t`$
- $`R_t`$ = Random/Noise component at time $`t`$

### Stationarity

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) remain constant over time.

**Weak Stationarity Conditions:**
1. **Constant Mean:** $`E[Y_t] = \mu`$ for all $`t`$
2. **Constant Variance:** $`Var[Y_t] = \sigma^2`$ for all $`t`$
3. **Constant Autocovariance:** $`Cov[Y_t, Y_{t-k}] = \gamma_k`$ depends only on lag $`k`$

### Autocorrelation Function (ACF)

The autocorrelation at lag $`k`$ is defined as:

```math
\rho_k = \frac{Cov[Y_t, Y_{t-k}]}{\sqrt{Var[Y_t] \cdot Var[Y_{t-k}]}} = \frac{\gamma_k}{\gamma_0}
```

**Properties:**
- $`\rho_0 = 1`$
- $`\rho_k = \rho_{-k}`$
- $`|\rho_k| \leq 1`$ for all $`k`$

### Partial Autocorrelation Function (PACF)

The partial autocorrelation $`\phi_{kk}`$ measures the correlation between $`Y_t`$ and $`Y_{t-k}`$ after removing the effects of intermediate observations:

```math
\phi_{kk} = Corr[Y_t - \hat{Y}_t^{(k-1)}, Y_{t-k} - \hat{Y}_{t-k}^{(k-1)}]
```

where $`\hat{Y}_t^{(k-1)}`$ is the linear prediction of $`Y_t`$ based on $`Y_{t-1}, Y_{t-2}, \ldots, Y_{t-k+1}`$.

---

# Practical Implementation

All Python code for this chapter has been moved to the companion file: `23_time_series_analysis.py`.

- For each theoretical section, refer to the corresponding function in the Python file.
- The Python file contains modular, well-documented functions and a main demonstration block.
- See the end of this document for a summary mapping theory sections to code functions.

---

## Code Reference Guide

| Theory Section                        | Python Function/Section                |
|---------------------------------------|----------------------------------------|
| Creating Time Series Objects          | `create_example_time_series()`         |
| Time Series Visualization             | `plot_time_series()`                   |
| Linear Trend Analysis                 | `trend_analysis()`                     |
| Polynomial Trend Analysis             | `polynomial_trend()`                   |
| Seasonal Decomposition                | `seasonal_decomposition()`             |
| Seasonal Strength                     | `seasonal_strength()`                  |
| Seasonal Adjustment                   | `seasonal_adjustment()`                |
| Stationarity Tests                    | `stationarity_tests()`                 |
| Differencing for Stationarity         | `difference_series()`                  |
| Autocorrelation Analysis              | `autocorrelation_analysis()`           |
| ARIMA Modeling                        | `fit_auto_arima()`, `fit_manual_arima()`|
| Forecasting                           | `forecast_arima()`                     |
| Forecast Accuracy                     | `calculate_accuracy_measures()`        |
| Exponential Smoothing                 | `fit_exponential_smoothing()`          |
| Vector Autoregression (VAR)           | `fit_var_model()`                      |
| Model Selection Guidelines            | `choose_time_series_model()`           |
| Reporting Guidelines                  | `generate_time_series_report()`        |
| Practical Examples                    | See main block in `.py` file           |

For exercises, use the functions above as building blocks for your analysis.

---

## Best Practices

### Model Selection Guidelines

```python
# Function to help choose appropriate time series model
def choose_time_series_model(time_series):
    print("=== TIME SERIES MODEL SELECTION ===")
    
    # Check stationarity
    adf_result = adfuller(time_series)
    print(f"ADF test p-value: {adf_result[1]:.4f}")
    
    # Check seasonality
    decomp = seasonal_decompose(time_series, model='additive', period=12)
    seasonal_strength = np.var(decomp.seasonal.dropna()) / (np.var(decomp.seasonal.dropna()) + np.var(decomp.resid.dropna()))
    print(f"Seasonal strength: {seasonal_strength:.3f}")
    
    # Check trend
    time_index = np.arange(len(time_series))
    trend_model = np.polyfit(time_index, time_series.values, 1)
    trend_slope = trend_model[0]
    print(f"Trend slope: {trend_slope:.4f}")
    
    print("\nRECOMMENDATIONS:")
    
    if adf_result[1] < 0.05:
        print("- Data is stationary")
        if seasonal_strength > 0.1:
            print("- Use seasonal ARIMA or Holt-Winters")
        else:
            print("- Use ARIMA or exponential smoothing")
    else:
        print("- Data is non-stationary")
        print("- Use differencing or trend-stationary models")
    
    if abs(trend_slope) > 0.01:
        print("- Significant trend detected")
        print("- Consider Holt's method or trend-stationary ARIMA")
    
    return {
        'stationary': adf_result[1] < 0.05,
        'seasonal_strength': seasonal_strength,
        'trend_significant': abs(trend_slope) > 0.01
    }

# Apply model selection
model_selection = choose_time_series_model(ts_data)
```

### Reporting Guidelines

```python
# Function to generate comprehensive time series report
def generate_time_series_report(time_series, model, forecast_result):
    print("=== TIME SERIES ANALYSIS REPORT ===\n")
    
    # Data summary
    print("DATA SUMMARY:")
    print(f"Length: {len(time_series)}")
    print(f"Frequency: {time_series.index.freq}")
    print(f"Start: {time_series.index[0]}")
    print(f"End: {time_series.index[-1]}")
    print(f"Mean: {time_series.mean():.2f}")
    print(f"Standard deviation: {time_series.std():.2f}\n")
    
    # Model summary
    print("MODEL SUMMARY:")
    if hasattr(model, 'order'):
        print("Model type: ARIMA")
        print(f"Order: {model.order}")
        if hasattr(model, 'seasonal_order'):
            print(f"Seasonal order: {model.seasonal_order}")
    elif hasattr(model, 'params'):
        print("Model type: Exponential Smoothing")
        print(f"Method: {model.params}")
    
    if hasattr(model, 'aic'):
        print(f"AIC: {model.aic():.2f}")
    if hasattr(model, 'bic'):
        print(f"BIC: {model.bic():.2f}")
    print()
    
    # Forecast summary
    print("FORECAST SUMMARY:")
    print(f"Forecast periods: {len(forecast_result)}")
    print(f"Point forecast range: {forecast_result.min():.2f} to {forecast_result.max():.2f}")
    print()
    
    # Residual analysis
    if hasattr(model, 'resid'):
        residuals_model = model.resid()
    else:
        residuals_model = model.resid
    
    print("RESIDUAL ANALYSIS:")
    print(f"Mean residual: {np.mean(residuals_model):.4f}")
    print(f"Residual standard deviation: {np.std(residuals_model):.4f}")
    
    # Ljung-Box test
    ljung_test = acorr_ljungbox(residuals_model, lags=10, return_df=True)
    print(f"Ljung-Box test p-value: {ljung_test['lb_pvalue'].iloc[-1]:.4f}")
    
    if ljung_test['lb_pvalue'].iloc[-1] > 0.05:
        print("Residuals appear to be white noise")
    else:
        print("Residuals may not be white noise")

# Generate report
generate_time_series_report(ts_data, auto_arima_model, forecast_result)
```

## Exercises

### Exercise 1: Trend Analysis
- **Objective:** Analyze the trend component of a time series and compare linear vs non-linear trend models.
- **Data:** Create a time series with known trend and analyze using different trend models.
- **Hint:** Use `trend_analysis()` and `polynomial_trend()` functions.

### Exercise 2: Seasonality Detection
- **Objective:** Identify and analyze seasonal patterns in time series data using decomposition methods.
- **Data:** Create a time series with known seasonality and perform decomposition.
- **Hint:** Use `seasonal_decomposition()` and `seasonal_strength()` functions.

### Exercise 3: Stationarity Testing
- **Objective:** Perform comprehensive stationarity tests and apply appropriate transformations.
- **Data:** Create non-stationary time series and test various differencing approaches.
- **Hint:** Use `adfuller()`, `kpss()`, and `diff()` functions.

### Exercise 4: ARIMA Modeling
- **Objective:** Fit ARIMA models to time series data and evaluate model diagnostics.
- **Data:** Use real or simulated time series data.
- **Hint:** Use `auto_arima()` and residual analysis functions.

### Exercise 5: Forecasting
- **Objective:** Generate forecasts using different models and evaluate forecast accuracy.
- **Data:** Split time series into training and test sets.
- **Hint:** Use `predict()` and `calculate_accuracy_measures()` functions.

### Exercise 6: Exponential Smoothing
- **Objective:** Compare different exponential smoothing methods for time series forecasting.
- **Data:** Create time series with trend and/or seasonality.
- **Hint:** Use `ExponentialSmoothing()` with different parameters.

### Exercise 7: Model Selection
- **Objective:** Choose the best time series model based on data characteristics and performance metrics.
- **Data:** Use a complex time series with multiple components.
- **Hint:** Use `choose_time_series_model()` function and compare AIC/BIC values.

### Exercise 8: Comprehensive Analysis
- **Objective:** Perform a complete time series analysis including decomposition, modeling, and forecasting.
- **Data:** Use a realistic dataset (e.g., economic, sales, or weather data).
- **Hint:** Use all the functions and techniques covered in this chapter.

## Next Steps

In the next chapter, we'll learn about multivariate analysis techniques.

---

**Key Takeaways:**
- Time series analysis requires understanding of trend, seasonality, and stationarity
- Proper model selection depends on data characteristics and statistical tests
- ARIMA models are powerful for univariate time series forecasting
- Exponential smoothing is useful for trend and seasonal data
- Always check model diagnostics and forecast accuracy
- Seasonal decomposition helps understand data components
- Stationarity is crucial for many time series models
- Proper reporting includes data summary, model diagnostics, and forecast evaluation
- Multiple models should be compared using information criteria
- Forecast uncertainty should be quantified with confidence intervals