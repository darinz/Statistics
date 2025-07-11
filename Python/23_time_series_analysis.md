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

## Basic Time Series Concepts

### Creating Time Series Objects

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Create a simple time series
np.random.seed(123)
n_periods = 100
time_index = np.arange(1, n_periods + 1)

# Generate time series with trend and noise
trend = 0.1 * time_index  # Linear trend: β₀ + β₁t
seasonal = 5 * np.sin(2 * np.pi * time_index / 12)  # Annual seasonality
noise = np.random.normal(0, 2, n_periods)  # White noise: εₜ ~ N(0, σ²)
time_series_data = 50 + trend + seasonal + noise

# Create time series object with pandas
dates = pd.date_range(start='2010-01-01', periods=n_periods, freq='M')
ts_data = pd.Series(time_series_data, index=dates)
print(ts_data.head())

# Basic time series properties
print("Time Series Properties:")
print(f"Length: {len(ts_data)}")
print(f"Frequency: {ts_data.index.freq}")
print(f"Start: {ts_data.index[0]}")
print(f"End: {ts_data.index[-1]}")
print(f"Mean: {ts_data.mean():.2f}")
print(f"Standard deviation: {ts_data.std():.2f}")

# Mathematical summary
print("\nMathematical Summary:")
print(f"Trend coefficient (β₁): {0.1:.3f}")
print(f"Seasonal amplitude: {5:.2f}")
print(f"Noise standard deviation (σ): {2:.2f}")
```

### Time Series Visualization

```python
# Basic time series plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(ts_data.index, ts_data.values, 'b-', linewidth=0.8)
plt.title('Time Series Plot')
plt.ylabel('Value')
plt.xlabel('Time')
plt.grid(True, alpha=0.3)

# Enhanced plot with trend line
plt.subplot(2, 2, 2)
plt.plot(ts_data.index, ts_data.values, 'steelblue', linewidth=0.8, alpha=0.7)
# Add trend line using rolling mean
trend_line = ts_data.rolling(window=12, center=True).mean()
plt.plot(ts_data.index, trend_line, 'red', linewidth=2, label='Trend')
plt.title('Time Series with Trend Line')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)

# Seasonal decomposition plot
plt.subplot(2, 2, 3)
decomposed_ts = seasonal_decompose(ts_data, model='additive', period=12)
plt.plot(decomposed_ts.trend, 'red', linewidth=2)
plt.title('Trend Component (T_t)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(decomposed_ts.seasonal, 'green', linewidth=2)
plt.title('Seasonal Component (S_t)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# STL decomposition (more robust)
stl_decomp = STL(ts_data, period=12).fit()

plt.figure(figsize=(15, 10))
plt.subplot(4, 1, 1)
plt.plot(ts_data.index, ts_data.values, 'b-')
plt.title('Original Time Series (Y_t)')
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 2)
plt.plot(ts_data.index, stl_decomp.trend, 'red')
plt.title('Trend Component (T_t)')
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 3)
plt.plot(ts_data.index, stl_decomp.seasonal, 'green')
plt.title('Seasonal Component (S_t)')
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 4)
plt.plot(ts_data.index, stl_decomp.resid, 'purple')
plt.title('Residual Component (R_t)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Trend Analysis

### Linear Trend Analysis

**Mathematical Model:**
```math
Y_t = \beta_0 + \beta_1 t + \varepsilon_t
```

where:
- $`\beta_0`$ = intercept
- $`\beta_1`$ = trend coefficient (slope)
- $`\varepsilon_t`$ = random error term

**Estimation:**
The parameters are estimated using ordinary least squares (OLS):
```math
\hat{\beta}_1 = \frac{\sum_{t=1}^n (t - \bar{t})(Y_t - \bar{Y})}{\sum_{t=1}^n (t - \bar{t})^2}
```

```math
\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{t}
```

**Python Implementation:**
```python
# Linear trend analysis
def trend_analysis(time_series):
    # Extract time index
    time_index = np.arange(1, len(time_series) + 1)
    
    # Fit linear trend: Y_t = β₀ + β₁t + ε_t
    from sklearn.linear_model import LinearRegression
    X = time_index.reshape(-1, 1)
    y = time_series.values
    
    trend_model = LinearRegression()
    trend_model.fit(X, y)
    
    # Extract coefficients
    intercept = trend_model.intercept_  # β₀
    slope = trend_model.coef_[0]        # β₁
    
    # Calculate trend line
    trend_line = trend_model.predict(X)
    
    # R-squared: proportion of variance explained
    r_squared = trend_model.score(X, y)
    
    # Statistical significance using statsmodels
    import statsmodels.api as sm
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    p_value = model.pvalues[1]  # p-value for slope
    se_slope = model.bse[1]     # standard error of slope
    ci_slope = model.conf_int()[1]  # 95% confidence interval for slope
    
    return {
        'intercept': intercept,
        'slope': slope,
        'trend_line': trend_line,
        'r_squared': r_squared,
        'p_value': p_value,
        'se_slope': se_slope,
        'ci_slope': ci_slope,
        'model': model
    }

# Apply trend analysis
trend_result = trend_analysis(ts_data)

print("Linear Trend Analysis Results:")
print("Model: Y_t = β₀ + β₁t + ε_t")
print(f"Intercept (β₀): {trend_result['intercept']:.3f}")
print(f"Slope (β₁): {trend_result['slope']:.3f}")
print(f"Standard Error of Slope: {trend_result['se_slope']:.4f}")
print(f"95% CI for Slope: {trend_result['ci_slope'][0]:.4f} to {trend_result['ci_slope'][1]:.4f}")
print(f"R-squared: {trend_result['r_squared']:.3f}")
print(f"p-value: {trend_result['p_value']:.4f}")

# Visualize trend
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ts_data.index, ts_data.values, 'steelblue', alpha=0.7, linewidth=0.8)
plt.plot(ts_data.index, trend_result['trend_line'], 'red', linewidth=2)
plt.title('Time Series with Linear Trend')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Residuals plot
residuals = ts_data.values - trend_result['trend_line']
plt.scatter(ts_data.index, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals from Linear Trend')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Non-linear Trend Analysis

**Polynomial Trend Model:**
```math
Y_t = \beta_0 + \beta_1 t + \beta_2 t^2 + \ldots + \beta_p t^p + \varepsilon_t
```

**Python Implementation:**
```python
# Polynomial trend analysis
def polynomial_trend(time_series, degree=2):
    time_index = np.arange(1, len(time_series) + 1)
    
    # Fit polynomial trend: Y_t = β₀ + β₁t + β₂t² + ... + βₚtᵖ + ε_t
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    X = time_index.reshape(-1, 1)
    y = time_series.values
    
    poly_model.fit(X, y)
    
    # Calculate fitted values
    fitted_values = poly_model.predict(X)
    
    # R-squared
    r_squared = poly_model.score(X, y)
    
    # AIC and BIC for model comparison
    n = len(y)
    k = degree + 1  # number of parameters
    mse = np.mean((y - fitted_values) ** 2)
    aic_value = n * np.log(mse) + 2 * k
    bic_value = n * np.log(mse) + k * np.log(n)
    
    return {
        'model': poly_model,
        'fitted_values': fitted_values,
        'r_squared': r_squared,
        'aic': aic_value,
        'bic': bic_value
    }

# Apply polynomial trend analysis
poly_result = polynomial_trend(ts_data, degree=2)

print("Polynomial Trend Analysis Results:")
print("Model: Y_t = β₀ + β₁t + β₂t² + ε_t")
print(f"R-squared: {poly_result['r_squared']:.3f}")
print(f"AIC: {poly_result['aic']:.2f}")
print(f"BIC: {poly_result['bic']:.2f}")

# Compare linear vs polynomial
print("\nModel Comparison:")
print(f"Linear R-squared: {trend_result['r_squared']:.3f}")
print(f"Polynomial R-squared: {poly_result['r_squared']:.3f}")
print(f"Improvement: {poly_result['r_squared'] - trend_result['r_squared']:.3f}")

# Visualize polynomial trend
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ts_data.index, ts_data.values, 'steelblue', alpha=0.7, linewidth=0.8)
plt.plot(ts_data.index, trend_result['trend_line'], 'red', linewidth=2, label='Linear')
plt.plot(ts_data.index, poly_result['fitted_values'], 'green', linewidth=2, label='Polynomial')
plt.title('Linear vs Polynomial Trend')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Residuals comparison
linear_residuals = ts_data.values - trend_result['trend_line']
poly_residuals = ts_data.values - poly_result['fitted_values']

plt.scatter(ts_data.index, linear_residuals, alpha=0.6, label='Linear', s=20)
plt.scatter(ts_data.index, poly_residuals, alpha=0.6, label='Polynomial', s=20)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals Comparison')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Seasonality Analysis

### Seasonal Decomposition

**Classical Decomposition:**
The time series is decomposed using moving averages:

1. **Trend Estimation:** $`T_t = MA(Y_t)`$ using centered moving average
2. **Detrended Series:** $`Y_t - T_t`$
3. **Seasonal Component:** $`S_t = \text{Seasonal Average}(Y_t - T_t)`$
4. **Random Component:** $`R_t = Y_t - T_t - S_t`$

**STL Decomposition:**
More robust decomposition using LOESS smoothing:

```math
Y_t = T_t + S_t + R_t
```

**Python Implementation:**
```python
# Seasonal decomposition
def seasonal_decomposition(time_series):
    # Classical decomposition
    classical_decomp = seasonal_decompose(time_series, model='additive', period=12)
    
    # STL decomposition (more robust)
    stl_decomp = STL(time_series, period=12).fit()
    
    return {
        'classical': classical_decomp,
        'stl': stl_decomp
    }

# Apply seasonal decomposition
decomp_result = seasonal_decomposition(ts_data)

# Plot decomposition components
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
axes[0].plot(decomp_result['classical'].trend, 'red')
axes[0].set_title('Trend Component (T_t)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(decomp_result['classical'].seasonal, 'green')
axes[1].set_title('Seasonal Component (S_t)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(decomp_result['classical'].resid, 'purple')
axes[2].set_title('Random Component (R_t)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(ts_data.index, ts_data.values, 'blue')
axes[3].set_title('Original Time Series (Y_t)')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Extract seasonal strength
def seasonal_strength(time_series):
    decomp = seasonal_decompose(time_series, model='additive', period=12)
    
    # Calculate seasonal strength: Var(S_t) / [Var(S_t) + Var(R_t)]
    seasonal_var = np.var(decomp.seasonal.dropna())
    random_var = np.var(decomp.resid.dropna())
    
    strength = seasonal_var / (seasonal_var + random_var)
    
    return strength

seasonal_strength_value = seasonal_strength(ts_data)
print(f"Seasonal Strength: {seasonal_strength_value:.3f}")
print(f"Interpretation: {'Strong seasonality' if seasonal_strength_value > 0.1 else 'Weak seasonality'}")
```

### Seasonal Adjustment

**Seasonally Adjusted Series:**
```math
Y_t^{SA} = Y_t - S_t = T_t + R_t
```

**Python Implementation:**
```python
# Seasonal adjustment
def seasonal_adjustment(time_series):
    # Classical seasonal adjustment
    decomp = seasonal_decompose(time_series, model='additive', period=12)
    seasonally_adjusted = time_series - decomp.seasonal
    
    # STL seasonal adjustment
    stl_decomp = STL(time_series, period=12).fit()
    stl_adjusted = time_series - stl_decomp.seasonal
    
    return {
        'classical_adjusted': seasonally_adjusted,
        'stl_adjusted': stl_adjusted
    }

# Apply seasonal adjustment
adjusted_result = seasonal_adjustment(ts_data)

# Visualize seasonal adjustment
plt.figure(figsize=(12, 8))
plt.plot(ts_data.index, ts_data.values, 'blue', alpha=0.7, label='Original (Y_t)')
plt.plot(adjusted_result['classical_adjusted'].index, 
         adjusted_result['classical_adjusted'].values, 
         'red', alpha=0.7, label='Classically Adjusted (Y_t - S_t)')
plt.plot(adjusted_result['stl_adjusted'].index, 
         adjusted_result['stl_adjusted'].values, 
         'green', alpha=0.7, label='STL Adjusted')
plt.title('Seasonal Adjustment Comparison')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Stationarity Analysis

### Stationarity Tests

**Augmented Dickey-Fuller (ADF) Test:**
Tests the null hypothesis that a unit root is present in the time series.

**Test Equation:**
```math
\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^p \delta_i \Delta Y_{t-i} + \varepsilon_t
```

**Null Hypothesis:** $`H_0: \gamma = 0`$ (unit root exists, series is non-stationary)
**Alternative Hypothesis:** $`H_1: \gamma < 0`$ (no unit root, series is stationary)

**Python Implementation:**
```python
# Augmented Dickey-Fuller test
adf_test = adfuller(ts_data)
print("Augmented Dickey-Fuller Test:")
print(f"ADF Statistic: {adf_test[0]:.4f}")
print(f"p-value: {adf_test[1]:.4f}")
print(f"Critical values:")
for key, value in adf_test[4].items():
    print(f"\t{key}: {value:.3f}")

# Kwiatkowski-Phillips-Schmidt-Shin test
kpss_test = kpss(ts_data)
print("\nKwiatkowski-Phillips-Schmidt-Shin Test:")
print(f"KPSS Statistic: {kpss_test[0]:.4f}")
print(f"p-value: {kpss_test[1]:.4f}")

print("\nStationarity Test Results:")
print(f"ADF p-value: {adf_test[1]:.4f}")
print(f"KPSS p-value: {kpss_test[1]:.4f}")

# Interpretation
print("\nInterpretation:")
if adf_test[1] < 0.05:
    print("- ADF test: Reject H₀, series is stationary")
else:
    print("- ADF test: Fail to reject H₀, series is non-stationary")

if kpss_test[1] > 0.05:
    print("- KPSS test: Fail to reject H₀, series is stationary")
else:
    print("- KPSS test: Reject H₀, series is non-stationary")
```

### Differencing for Stationarity

**First Difference:**
```math
\Delta Y_t = Y_t - Y_{t-1}
```

**Second Difference:**
```math
\Delta^2 Y_t = \Delta Y_t - \Delta Y_{t-1} = Y_t - 2Y_{t-1} + Y_{t-2}
```

**Seasonal Difference:**
```math
\Delta_s Y_t = Y_t - Y_{t-s}
```

**Python Implementation:**
```python
# First differencing
first_diff = ts_data.diff().dropna()
adf_diff_test = adfuller(first_diff)
print("First Difference ADF Test:")
print(f"ADF Statistic: {adf_diff_test[0]:.4f}")
print(f"p-value: {adf_diff_test[1]:.4f}")

# Second differencing
second_diff = first_diff.diff().dropna()
adf_diff2_test = adfuller(second_diff)
print("\nSecond Difference ADF Test:")
print(f"ADF Statistic: {adf_diff2_test[0]:.4f}")
print(f"p-value: {adf_diff2_test[1]:.4f}")

# Seasonal differencing
seasonal_diff = ts_data.diff(periods=12).dropna()
adf_seasonal_test = adfuller(seasonal_diff)
print("\nSeasonal Difference ADF Test:")
print(f"ADF Statistic: {adf_seasonal_test[0]:.4f}")
print(f"p-value: {adf_seasonal_test[1]:.4f}")

print("\nDifferencing Results:")
print(f"First difference ADF p-value: {adf_diff_test[1]:.4f}")
print(f"Second difference ADF p-value: {adf_diff2_test[1]:.4f}")
print(f"Seasonal difference ADF p-value: {adf_seasonal_test[1]:.4f}")

# Visualize differenced series
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(ts_data.index, ts_data.values)
axes[0, 0].set_title('Original Series')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(first_diff.index, first_diff.values)
axes[0, 1].set_title('First Difference')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(second_diff.index, second_diff.values)
axes[1, 0].set_title('Second Difference')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(seasonal_diff.index, seasonal_diff.values)
axes[1, 1].set_title('Seasonal Difference')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Autocorrelation Analysis

### Autocorrelation Functions

**Sample Autocorrelation:**
```math
\hat{\rho}_k = \frac{\sum_{t=k+1}^n (Y_t - \bar{Y})(Y_{t-k} - \bar{Y})}{\sum_{t=1}^n (Y_t - \bar{Y})^2}
```

**Python Implementation:**
```python
# Autocorrelation function (ACF)
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(ts_data, lags=24, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')

# Partial autocorrelation function (PACF)
plot_pacf(ts_data, lags=24, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# Get ACF and PACF values
from statsmodels.tsa.stattools import acf, pacf
acf_result = acf(ts_data, nlags=24)
pacf_result = pacf(ts_data, nlags=24)

print("ACF values (first 10 lags):")
for i, val in enumerate(acf_result[1:11]):
    print(f"Lag {i+1}: {val:.3f}")

print("\nPACF values (first 10 lags):")
for i, val in enumerate(pacf_result[1:11]):
    print(f"Lag {i+1}: {val:.3f}")

# Identify significant lags (beyond ±2/√n confidence bands)
n = len(ts_data)
confidence_band = 2 / np.sqrt(n)
significant_acf = np.where(np.abs(acf_result) > confidence_band)[0]
significant_pacf = np.where(np.abs(pacf_result) > confidence_band)[0]

print(f"\nSignificant ACF lags: {significant_acf}")
print(f"Significant PACF lags: {significant_pacf}")
print(f"Confidence band (±2/√n): {confidence_band:.3f}")
```

## ARIMA Modeling

### ARIMA Model Structure

**ARIMA(p,d,q) Model:**
```math
(1 - \phi_1 B - \phi_2 B^2 - \ldots - \phi_p B^p)(1 - B)^d Y_t = (1 + \theta_1 B + \theta_2 B^2 + \ldots + \theta_q B^q)\varepsilon_t
```

where:
- $`B`$ is the backshift operator: $`BY_t = Y_{t-1}`$
- $`p`$ = order of autoregressive (AR) terms
- $`d`$ = degree of differencing
- $`q`$ = order of moving average (MA) terms
- $`\phi_i`$ = AR coefficients
- $`\theta_i`$ = MA coefficients
- $`\varepsilon_t`$ = white noise error term

**Seasonal ARIMA(p,d,q)(P,D,Q)s:**
```math
\phi_p(B)\Phi_P(B^s)(1 - B)^d(1 - B^s)^D Y_t = \theta_q(B)\Theta_Q(B^s)\varepsilon_t
```

**Python Implementation:**
```python
# Automatic ARIMA model selection
from pmdarima import auto_arima

auto_arima_model = auto_arima(ts_data, seasonal=True, m=12, 
                             suppress_warnings=True, error_action='ignore')
print(auto_arima_model.summary())

# Extract model parameters
arima_order = auto_arima_model.order
seasonal_order = auto_arima_model.seasonal_order
print("ARIMA Model Parameters:")
print(f"p (AR order): {arima_order[0]}")
print(f"d (differencing): {arima_order[1]}")
print(f"q (MA order): {arima_order[2]}")
print(f"P (seasonal AR): {seasonal_order[0]}")
print(f"D (seasonal differencing): {seasonal_order[1]}")
print(f"Q (seasonal MA): {seasonal_order[2]}")

# Model diagnostics
residuals_auto = auto_arima_model.resid()

# Plot residuals
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(residuals_auto)
axes[0, 0].set_title('Residuals')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(residuals_auto, bins=20, alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Residuals Histogram')
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot
from scipy import stats
stats.probplot(residuals_auto, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot of Residuals')

# ACF of residuals
plot_acf(residuals_auto, lags=20, ax=axes[1, 1])
axes[1, 1].set_title('ACF of Residuals')

plt.tight_layout()
plt.show()

# AIC and BIC
print("Model Selection Criteria:")
print(f"AIC: {auto_arima_model.aic():.2f}")
print(f"BIC: {auto_arima_model.bic():.2f}")

# Residual analysis
print("Residual Analysis:")
print(f"Mean: {np.mean(residuals_auto):.4f}")
print(f"Standard deviation: {np.std(residuals_auto):.4f}")
print(f"Skewness: {stats.skew(residuals_auto):.3f}")
print(f"Kurtosis: {stats.kurtosis(residuals_auto):.3f}")

# Ljung-Box test for residual autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
ljung_auto = acorr_ljungbox(residuals_auto, lags=10, return_df=True)
print(f"Ljung-Box test p-value: {ljung_auto['lb_pvalue'].iloc[-1]:.4f}")
```

### Manual ARIMA Model Fitting

```python
# Manual ARIMA model fitting
manual_arima = ARIMA(ts_data, order=(1, 1, 1))
manual_arima_fitted = manual_arima.fit()
print(manual_arima_fitted.summary())

# Seasonal ARIMA
seasonal_arima = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
seasonal_arima_fitted = seasonal_arima.fit(disp=False)
print(seasonal_arima_fitted.summary())

# Compare models
print("Model Comparison:")
print(f"Auto ARIMA AIC: {auto_arima_model.aic():.2f}")
print(f"Manual ARIMA AIC: {manual_arima_fitted.aic:.2f}")
print(f"Seasonal ARIMA AIC: {seasonal_arima_fitted.aic:.2f}")

# Residual analysis
residuals_manual = manual_arima_fitted.resid
residuals_seasonal = seasonal_arima_fitted.resid

# Ljung-Box test for residuals
ljung_manual = acorr_ljungbox(residuals_manual, lags=10, return_df=True)
ljung_seasonal = acorr_ljungbox(residuals_seasonal, lags=10, return_df=True)

print("Ljung-Box Test Results:")
print(f"Auto ARIMA p-value: {ljung_auto['lb_pvalue'].iloc[-1]:.4f}")
print(f"Manual ARIMA p-value: {ljung_manual['lb_pvalue'].iloc[-1]:.4f}")
print(f"Seasonal ARIMA p-value: {ljung_seasonal['lb_pvalue'].iloc[-1]:.4f}")
```

## Forecasting

### Point Forecasts

**Forecast Equation:**
For an ARIMA(p,d,q) model, the forecast at time $`t+h`$ is:
```math
\hat{Y}_{t+h} = E[Y_{t+h} | Y_1, Y_2, \ldots, Y_t]
```

**Python Implementation:**
```python
# Generate forecasts
forecast_periods = 12
forecast_result = auto_arima_model.predict(n_periods=forecast_periods)

# Get confidence intervals
forecast_ci = auto_arima_model.predict(n_periods=forecast_periods, return_conf_int=True)
point_forecast = forecast_result
lower_ci = forecast_ci[1][:, 0]  # 95% CI lower
upper_ci = forecast_ci[1][:, 1]  # 95% CI upper

# Create forecast data frame
forecast_dates = pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), 
                              periods=forecast_periods, freq='M')
forecast_df = pd.DataFrame({
    'forecast': point_forecast,
    'lower': lower_ci,
    'upper': upper_ci
}, index=forecast_dates)

print("Forecast Results:")
print(forecast_df.head())

# Visualize forecast
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data.values, 'steelblue', label='Historical Data')
plt.plot(forecast_df.index, forecast_df['forecast'], 'red', linewidth=2, label='Forecast')
plt.fill_between(forecast_df.index, forecast_df['lower'], forecast_df['upper'], 
                alpha=0.3, color='red', label='95% Confidence Interval')
plt.title('Time Series Forecast with 95% Confidence Intervals')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Forecast Accuracy

**Accuracy Measures:**

1. **Mean Absolute Error (MAE):**
```math
MAE = \frac{1}{n} \sum_{i=1}^n |Y_i - \hat{Y}_i|
```

2. **Root Mean Square Error (RMSE):**
```math
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (Y_i - \hat{Y}_i)^2}
```

3. **Mean Absolute Percentage Error (MAPE):**
```math
MAPE = \frac{100}{n} \sum_{i=1}^n \left|\frac{Y_i - \hat{Y}_i}{Y_i}\right|
```

**Python Implementation:**
```python
# Split data for forecast evaluation
train_size = int(0.8 * len(ts_data))
train_data = ts_data[:train_size]
test_data = ts_data[train_size:]

# Fit model on training data
train_arima = auto_arima(train_data, seasonal=True, m=12, 
                        suppress_warnings=True, error_action='ignore')

# Generate forecasts
forecast_test = train_arima.predict(n_periods=len(test_data))

# Calculate accuracy measures
def calculate_accuracy_measures(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

mae, rmse, mape = calculate_accuracy_measures(test_data.values, forecast_test)

print("Forecast Accuracy Measures:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Visualize forecast accuracy
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data.values, 'blue', label='Training Data')
plt.plot(test_data.index, test_data.values, 'green', label='Actual Test Data')
plt.plot(test_data.index, forecast_test, 'red', linewidth=2, label='Forecast')
plt.title('Forecast Accuracy Evaluation')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Advanced Time Series Models

### Exponential Smoothing

**Simple Exponential Smoothing (SES):**
```math
\hat{Y}_{t+1} = \alpha Y_t + (1 - \alpha) \hat{Y}_t
```

where $`\alpha`$ is the smoothing parameter (0 < α < 1).

**Holt's Method (Trend):**
```math
\hat{Y}_{t+h} = l_t + h b_t
```

where:
- $`l_t = \alpha Y_t + (1 - \alpha)(l_{t-1} + b_{t-1})`$ (level)
- $`b_t = \beta(l_t - l_{t-1}) + (1 - \beta)b_{t-1}`$ (trend)

**Holt-Winters Method (Trend + Seasonality):**
```math
\hat{Y}_{t+h} = l_t + h b_t + s_{t+h-s}
```

where $`s_t`$ is the seasonal component.

**Python Implementation:**
```python
# Simple exponential smoothing
ses_model = ExponentialSmoothing(ts_data, seasonal_periods=None).fit()
ses_forecast = ses_model.forecast(12)

# Holt's method (trend)
holt_model = ExponentialSmoothing(ts_data, trend='add', seasonal_periods=None).fit()
holt_forecast = holt_model.forecast(12)

# Holt-Winters method (trend + seasonality)
hw_model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=12).fit()
hw_forecast = hw_model.forecast(12)

# Compare models
print("Model Comparison (AIC):")
print(f"SES: {ses_model.aic:.2f}")
print(f"Holt: {holt_model.aic:.2f}")
print(f"Holt-Winters: {hw_model.aic:.2f}")

# Extract smoothing parameters
print("Smoothing Parameters:")
print(f"SES alpha: {ses_model.params['smoothing_level']:.3f}")
print(f"Holt alpha: {holt_model.params['smoothing_level']:.3f}")
print(f"Holt beta: {holt_model.params['smoothing_trend']:.3f}")
print(f"HW alpha: {hw_model.params['smoothing_level']:.3f}")
print(f"HW beta: {hw_model.params['smoothing_trend']:.3f}")
print(f"HW gamma: {hw_model.params['smoothing_seasonal']:.3f}")

# Visualize exponential smoothing forecasts
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data.values, 'blue', label='Historical Data')
plt.plot(ses_forecast.index, ses_forecast.values, 'red', label='SES Forecast')
plt.plot(holt_forecast.index, holt_forecast.values, 'green', label='Holt Forecast')
plt.plot(hw_forecast.index, hw_forecast.values, 'orange', label='Holt-Winters Forecast')
plt.title('Exponential Smoothing Forecasts')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Vector Autoregression (VAR)

**VAR(p) Model:**
```math
Y_t = c + \Phi_1 Y_{t-1} + \Phi_2 Y_{t-2} + \ldots + \Phi_p Y_{t-p} + \varepsilon_t
```

where $`Y_t`$ is a vector of variables, $`\Phi_i`$ are coefficient matrices.

**Python Implementation:**
```python
# Create multivariate time series
np.random.seed(123)
n_periods = 100
var_data = pd.DataFrame({
    'y1': np.cumsum(np.random.normal(0, 1, n_periods)),
    'y2': np.cumsum(np.random.normal(0, 1, n_periods)) + 0.5 * np.cumsum(np.random.normal(0, 1, n_periods))
}, index=pd.date_range(start='2010-01-01', periods=n_periods, freq='M'))

# Fit VAR model
from statsmodels.tsa.vector_ar.var_model import VAR
var_model = VAR(var_data)
var_fitted = var_model.fit(maxlags=2)
print(var_fitted.summary())

# Granger causality test
from statsmodels.tsa.vector_ar.granger_causality import grangercausalitytests
gc_result = grangercausalitytests(var_data, maxlag=2, verbose=False)
print("Granger Causality Test Results:")
for lag, result in gc_result.items():
    print(f"Lag {lag}: p-value = {result[0]['ssr_chi2test'][1]:.4f}")

# Impulse response function
irf_result = var_fitted.irf(periods=10)
irf_result.plot()
plt.show()
```

## Practical Examples

### Example 1: Economic Data Analysis

```python
# Simulate economic time series
np.random.seed(123)
n_periods = 120

# Generate GDP data with trend and seasonality
time_index = np.arange(1, n_periods + 1)
trend = 0.02 * time_index  # 2% annual growth
seasonal = 2 * np.sin(2 * np.pi * time_index / 4)  # Quarterly seasonality
noise = np.random.normal(0, 1, n_periods)
gdp_data = 100 + trend + seasonal + noise

# Create time series
gdp_dates = pd.date_range(start='2010-01-01', periods=n_periods, freq='Q')
gdp_ts = pd.Series(gdp_data, index=gdp_dates)

# Analyze GDP data
gdp_decomp = seasonal_decompose(gdp_ts, model='additive', period=4)
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
axes[0].plot(gdp_decomp.trend)
axes[0].set_title('GDP Trend Component')
axes[0].grid(True, alpha=0.3)

axes[1].plot(gdp_decomp.seasonal)
axes[1].set_title('GDP Seasonal Component')
axes[1].grid(True, alpha=0.3)

axes[2].plot(gdp_decomp.resid)
axes[2].set_title('GDP Random Component')
axes[2].grid(True, alpha=0.3)

axes[3].plot(gdp_ts.index, gdp_ts.values)
axes[3].set_title('Original GDP Time Series')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Fit ARIMA model
gdp_arima = auto_arima(gdp_ts, seasonal=True, m=4, 
                      suppress_warnings=True, error_action='ignore')
print("GDP ARIMA Model:")
print(gdp_arima.summary())

# Generate forecasts
gdp_forecast = gdp_arima.predict(n_periods=8)
gdp_forecast_dates = pd.date_range(start=gdp_ts.index[-1] + pd.DateOffset(months=3), 
                                  periods=8, freq='Q')
gdp_forecast_series = pd.Series(gdp_forecast, index=gdp_forecast_dates)

plt.figure(figsize=(12, 6))
plt.plot(gdp_ts.index, gdp_ts.values, 'blue', label='Historical GDP')
plt.plot(gdp_forecast_series.index, gdp_forecast_series.values, 'red', linewidth=2, label='GDP Forecast')
plt.title('GDP Forecast')
plt.xlabel('Time')
plt.ylabel('GDP')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Example 2: Sales Data Analysis

```python
# Simulate sales data
np.random.seed(123)
n_periods = 60

# Generate sales with trend, seasonality, and cycles
time_index = np.arange(1, n_periods + 1)
trend = 0.5 * time_index
seasonal = 10 * np.sin(2 * np.pi * time_index / 12)  # Monthly seasonality
cycle = 5 * np.sin(2 * np.pi * time_index / 24)  # Biennial cycle
noise = np.random.normal(0, 3, n_periods)
sales_data = 50 + trend + seasonal + cycle + noise

# Create time series
sales_dates = pd.date_range(start='2015-01-01', periods=n_periods, freq='M')
sales_ts = pd.Series(sales_data, index=sales_dates)

# Seasonal decomposition
sales_decomp = STL(sales_ts, period=12).fit()
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
axes[0].plot(sales_decomp.trend)
axes[0].set_title('Sales Trend Component')
axes[0].grid(True, alpha=0.3)

axes[1].plot(sales_decomp.seasonal)
axes[1].set_title('Sales Seasonal Component')
axes[1].grid(True, alpha=0.3)

axes[2].plot(sales_decomp.resid)
axes[2].set_title('Sales Residual Component')
axes[2].grid(True, alpha=0.3)

axes[3].plot(sales_ts.index, sales_ts.values)
axes[3].set_title('Original Sales Time Series')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Fit multiple models
sales_ses = ExponentialSmoothing(sales_ts, seasonal_periods=None).fit()
sales_holt = ExponentialSmoothing(sales_ts, trend='add', seasonal_periods=None).fit()
sales_hw = ExponentialSmoothing(sales_ts, trend='add', seasonal='add', seasonal_periods=12).fit()
sales_arima = auto_arima(sales_ts, seasonal=True, m=12, 
                        suppress_warnings=True, error_action='ignore')

# Compare forecast accuracy
accuracy_comparison = pd.DataFrame({
    'Model': ['SES', 'Holt', 'Holt-Winters', 'ARIMA'],
    'AIC': [sales_ses.aic, sales_holt.aic, sales_hw.aic, sales_arima.aic()]
})
print("Model Comparison:")
print(accuracy_comparison)

# Generate forecasts
forecast_periods = 12
ses_forecast = sales_ses.forecast(forecast_periods)
holt_forecast = sales_holt.forecast(forecast_periods)
hw_forecast = sales_hw.forecast(forecast_periods)
arima_forecast = sales_arima.predict(n_periods=forecast_periods)

# Visualize forecasts
forecast_dates = pd.date_range(start=sales_ts.index[-1] + pd.DateOffset(months=1), 
                              periods=forecast_periods, freq='M')

plt.figure(figsize=(12, 6))
plt.plot(sales_ts.index, sales_ts.values, 'blue', label='Historical Sales')
plt.plot(forecast_dates, ses_forecast.values, 'red', label='SES Forecast')
plt.plot(forecast_dates, holt_forecast.values, 'green', label='Holt Forecast')
plt.plot(forecast_dates, hw_forecast.values, 'orange', label='Holt-Winters Forecast')
plt.plot(forecast_dates, arima_forecast, 'purple', label='ARIMA Forecast')
plt.title('Sales Forecast Comparison')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Example 3: Weather Data Analysis

```python
# Simulate temperature data
np.random.seed(123)
n_periods = 365

# Generate daily temperature with annual seasonality
time_index = np.arange(1, n_periods + 1)
annual_trend = 0.001 * time_index  # Slight warming trend
seasonal = 15 * np.sin(2 * np.pi * time_index / 365)  # Annual seasonality
noise = np.random.normal(0, 2, n_periods)
temperature_data = 20 + annual_trend + seasonal + noise

# Create time series
temp_dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
temp_ts = pd.Series(temperature_data, index=temp_dates)

# Analyze temperature data
temp_decomp = seasonal_decompose(temp_ts, model='additive', period=365)
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
axes[0].plot(temp_decomp.trend)
axes[0].set_title('Temperature Trend Component')
axes[0].grid(True, alpha=0.3)

axes[1].plot(temp_decomp.seasonal)
axes[1].set_title('Temperature Seasonal Component')
axes[1].grid(True, alpha=0.3)

axes[2].plot(temp_decomp.resid)
axes[2].set_title('Temperature Random Component')
axes[2].grid(True, alpha=0.3)

axes[3].plot(temp_ts.index, temp_ts.values)
axes[3].set_title('Original Temperature Time Series')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Fit seasonal ARIMA model
temp_arima = auto_arima(temp_ts, seasonal=True, m=365, 
                       suppress_warnings=True, error_action='ignore')
print("Temperature ARIMA Model:")
print(temp_arima.summary())

# Generate seasonal forecasts
temp_forecast = temp_arima.predict(n_periods=30)
temp_forecast_dates = pd.date_range(start=temp_ts.index[-1] + pd.DateOffset(days=1), 
                                   periods=30, freq='D')
temp_forecast_series = pd.Series(temp_forecast, index=temp_forecast_dates)

plt.figure(figsize=(12, 6))
plt.plot(temp_ts.index, temp_ts.values, 'blue', label='Historical Temperature')
plt.plot(temp_forecast_series.index, temp_forecast_series.values, 'red', linewidth=2, label='Temperature Forecast')
plt.title('Temperature Forecast')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

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