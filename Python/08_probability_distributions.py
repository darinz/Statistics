"""
Probability Distributions - Python Code Examples
Corresponds to: 08_probability_distributions.md
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm, binom, poisson, expon, chi2, t, f
import seaborn as sns
from scipy.stats import shapiro, kstest, gaussian_kde
from sklearn.datasets import fetch_openml

def basic_normal_distribution():
    """
    Demonstrate basic normal distribution functions and properties.
    """
    # Generate normal random variables with mathematical precision
    np.random.seed(123)
    normal_data = np.random.normal(0, 1, 1000)
    
    # Density function calculation
    x = np.arange(-4, 4.1, 0.1)
    normal_density = norm.pdf(x, 0, 1)
    
    # Plot normal distribution with mathematical annotations
    plt.figure(figsize=(10, 6))
    plt.plot(x, normal_density, color='blue', linewidth=2, label='N(0,1)')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='μ = 0')
    plt.axvline(x=-1, color='green', linestyle=':', linewidth=1, alpha=0.7)
    plt.axvline(x=1, color='green', linestyle=':', linewidth=1, alpha=0.7)
    plt.axvline(x=-2, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    plt.axvline(x=2, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    
    # Add mathematical annotations
    plt.text(0, 0.3, r'$\mu = 0$', color='red', fontsize=12, fontweight='bold', ha='center')
    plt.text(1, 0.2, r'$\sigma = 1$', color='blue', fontsize=12, fontweight='bold', ha='center')
    
    plt.title('Standard Normal Distribution N(0,1)', fontsize=14, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Density f(x)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def normal_distribution_properties():
    """
    Calculate and display normal distribution properties and probabilities.
    """
    # Calculate probabilities with mathematical interpretation
    print(f"P(X < 0) = Φ(0): {norm.cdf(0):.6f}")
    print(f"P(X < 1) = Φ(1): {norm.cdf(1):.6f}")
    print(f"P(X < 2) = Φ(2): {norm.cdf(2):.6f}")
    
    # Calculate quantiles (inverse CDF)
    print(f"95th percentile: Φ⁻¹(0.95) = {norm.ppf(0.95):.6f}")
    print(f"99th percentile: Φ⁻¹(0.99) = {norm.ppf(0.99):.6f}")
    
    # Calculate probabilities for intervals
    print(f"P(-1 < X < 1) = Φ(1) - Φ(-1): {norm.cdf(1) - norm.cdf(-1):.6f}")
    print(f"P(-2 < X < 2) = Φ(2) - Φ(-2): {norm.cdf(2) - norm.cdf(-2):.6f}")
    
    # Verify 68-95-99.7 rule
    print(f"P(-1 < X < 1): {norm.cdf(1) - norm.cdf(-1):.6f} (≈ 68%)")
    print(f"P(-2 < X < 2): {norm.cdf(2) - norm.cdf(-2):.6f} (≈ 95%)")
    print(f"P(-3 < X < 3): {norm.cdf(3) - norm.cdf(-3):.6f} (≈ 99.7%)")

def checking_normality():
    """
    Comprehensive normality assessment with multiple diagnostic plots and tests.
    """
    # Generate normal data for demonstration
    np.random.seed(123)
    normal_data = np.random.normal(0, 1, 1000)
    
    # Comprehensive normality assessment
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram with normal overlay
    axes[0, 0].hist(normal_data, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    x_range = np.linspace(normal_data.min(), normal_data.max(), 100)
    axes[0, 0].plot(x_range, norm.pdf(x_range, 0, 1), 'r-', linewidth=2, label='Normal PDF')
    axes[0, 0].set_title('Histogram with Normal Overlay')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].legend()
    
    # Q-Q plot for normality
    stats.probplot(normal_data, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot for Normality')
    
    # Density plot comparison
    kde = gaussian_kde(normal_data)
    x_dense = np.linspace(normal_data.min(), normal_data.max(), 200)
    axes[1, 0].plot(x_dense, kde(x_dense), 'b-', linewidth=2, label='KDE')
    axes[1, 0].plot(x_dense, norm.pdf(x_dense, 0, 1), 'r--', linewidth=2, label='Normal PDF')
    axes[1, 0].set_title('Density Plot vs Normal')
    axes[1, 0].legend()
    
    # Box plot for symmetry
    axes[1, 1].boxplot(normal_data, vert=True)
    axes[1, 1].set_title('Box Plot (Check for Symmetry)')
    axes[1, 1].set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical normality tests with interpretation
    shapiro_test = shapiro(normal_data)
    print("Shapiro-Wilk Test:")
    print(f"  W = {shapiro_test.statistic:.6f}")
    print(f"  p-value = {shapiro_test.pvalue:.6f}")
    print(f"  Interpretation: {'Data appears normal' if shapiro_test.pvalue > 0.05 else 'Data may not be normal'}\n")
    
    # Kolmogorov-Smirnov test
    ks_test = kstest(normal_data, 'norm')
    print("Kolmogorov-Smirnov Test:")
    print(f"  D = {ks_test.statistic:.6f}")
    print(f"  p-value = {ks_test.pvalue:.6f}")
    print(f"  Interpretation: {'Data appears normal' if ks_test.pvalue > 0.05 else 'Data may not be normal'}")

def basic_binomial_functions():
    """
    Demonstrate basic binomial distribution functions and properties.
    """
    # Binomial distribution parameters with mathematical interpretation
    n = 10  # number of trials (fixed)
    p = 0.5  # probability of success (constant across trials)
    
    # Generate binomial random variables
    np.random.seed(123)
    binomial_data = np.random.binomial(n, p, 1000)
    
    # Calculate exact probabilities
    print(f"P(X = 5): {binom.pmf(5, n, p):.6f}")
    print(f"P(X ≤ 5): {binom.cdf(5, n, p):.6f}")
    print(f"P(X ≥ 5): {1 - binom.cdf(4, n, p):.6f}")
    
    # Verify mathematical properties
    print(f"Theoretical mean (np): {n * p}")
    print(f"Sample mean: {np.mean(binomial_data):.6f}")
    print(f"Theoretical variance (np(1-p)): {n * p * (1-p)}")
    print(f"Sample variance: {np.var(binomial_data):.6f}")
    
    # Plot binomial distribution with mathematical annotations
    x = np.arange(0, n + 1)
    binomial_probs = binom.pmf(x, n, p)
    
    plt.figure(figsize=(10, 6))
    plt.vlines(x, 0, binomial_probs, colors='blue', linewidth=2, label=f'B({n},{p})')
    plt.plot(x, binomial_probs, 'ro', markersize=8, label='Probability Mass')
    
    # Add mathematical annotations
    plt.text(n/2, max(binomial_probs), 
             f'E[X] = np = {n*p}', 
             fontsize=12, fontweight='bold', ha='center', color='darkgreen')
    
    plt.title(f'Binomial Distribution B({n},{p})', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Successes (k)', fontsize=12)
    plt.ylabel('P(X = k)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def binomial_distribution_examples():
    """
    Demonstrate practical examples of binomial distribution applications.
    """
    # Example 1: Coin flipping with mathematical interpretation
    # Probability of getting exactly 7 heads in 10 flips
    p_7_heads = binom.pmf(7, 10, 0.5)
    print(f"P(X = 7) = C(10,7) × 0.5⁷ × 0.5³ = {p_7_heads:.6f}")
    
    # Probability of getting 7 or more heads
    p_7_or_more = 1 - binom.cdf(6, 10, 0.5)
    print(f"P(X ≥ 7) = 1 - P(X ≤ 6) = {p_7_or_more:.6f}")
    
    # Example 2: Quality control with practical interpretation
    # Probability of 2 or fewer defective items in 20 (10% defect rate)
    p_defective = [binom.pmf(k, 20, 0.1) for k in range(3)]
    print(f"P(X = 0): {p_defective[0]:.6f}")
    print(f"P(X = 1): {p_defective[1]:.6f}")
    print(f"P(X = 2): {p_defective[2]:.6f}")
    print(f"P(X ≤ 2): {sum(p_defective):.6f}")
    
    # Expected number of defects
    print(f"Expected defects (np): {20 * 0.1}")

def basic_poisson_functions():
    """
    Demonstrate basic Poisson distribution functions and properties.
    """
    # Poisson distribution parameter with mathematical interpretation
    lambda_val = 3  # average number of events per unit
    
    # Generate Poisson random variables
    np.random.seed(123)
    poisson_data = np.random.poisson(lambda_val, 1000)
    
    # Calculate exact probabilities
    print(f"P(X = 3): {poisson.pmf(3, lambda_val):.6f}")
    print(f"P(X ≤ 3): {poisson.cdf(3, lambda_val):.6f}")
    print(f"P(X ≥ 3): {1 - poisson.cdf(2, lambda_val):.6f}")
    
    # Verify mathematical properties
    print(f"Theoretical mean (λ): {lambda_val}")
    print(f"Sample mean: {np.mean(poisson_data):.6f}")
    print(f"Theoretical variance (λ): {lambda_val}")
    print(f"Sample variance: {np.var(poisson_data):.6f}")
    
    # Plot Poisson distribution with mathematical annotations
    x = np.arange(0, 11)
    poisson_probs = poisson.pmf(x, lambda_val)
    
    plt.figure(figsize=(10, 6))
    plt.vlines(x, 0, poisson_probs, colors='green', linewidth=2, label=f'Poi({lambda_val})')
    plt.plot(x, poisson_probs, 'ro', markersize=8, label='Probability Mass')
    
    # Add mathematical annotations
    plt.text(lambda_val, max(poisson_probs), 
             f'E[X] = Var(X) = {lambda_val}', 
             fontsize=12, fontweight='bold', ha='center', color='darkgreen')
    
    plt.title(f'Poisson Distribution Poi({lambda_val})', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Events (k)', fontsize=12)
    plt.ylabel('P(X = k)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def poisson_distribution_examples():
    """
    Demonstrate practical examples of Poisson distribution applications.
    """
    # Example 1: Customer arrivals with practical interpretation
    # Average 5 customers per hour (λ = 5)
    lambda_customers = 5
    
    # Probability of exactly 3 customers in an hour
    p_3_customers = poisson.pmf(3, lambda_customers)
    print(f"P(X = 3) = (5³ × e⁻⁵)/3! = {p_3_customers:.6f}")
    
    # Probability of 3 or fewer customers
    p_3_or_fewer = poisson.cdf(3, lambda_customers)
    print(f"P(X ≤ 3) = {p_3_or_fewer:.6f}")
    
    # Example 2: Defects in manufacturing
    # Average 2 defects per 100 items (λ = 2)
    lambda_defects = 2
    
    # Probability of no defects
    p_no_defects = poisson.pmf(0, lambda_defects)
    print(f"P(X = 0) = e⁻² = {p_no_defects:.6f}")
    
    # Probability of 1 or 2 defects
    p_1_or_2_defects = poisson.pmf(1, lambda_defects) + poisson.pmf(2, lambda_defects)
    print(f"P(1 ≤ X ≤ 2) = {p_1_or_2_defects:.6f}")

def basic_exponential_functions():
    """
    Demonstrate basic exponential distribution functions and properties.
    """
    # Exponential distribution parameter with mathematical interpretation
    rate = 2  # rate parameter λ
    
    # Generate exponential random variables
    np.random.seed(123)
    exponential_data = np.random.exponential(1/rate, 1000)
    
    # Calculate probabilities
    print(f"P(X < 1): {expon.cdf(1, scale=1/rate):.6f}")
    print(f"P(X > 2): {1 - expon.cdf(2, scale=1/rate):.6f}")
    
    # Verify mathematical properties
    print(f"Theoretical mean (1/λ): {1/rate}")
    print(f"Sample mean: {np.mean(exponential_data):.6f}")
    print(f"Theoretical variance (1/λ²): {1/rate**2}")
    print(f"Sample variance: {np.var(exponential_data):.6f}")
    
    # Plot exponential distribution with mathematical annotations
    x = np.arange(0, 5.1, 0.1)
    exponential_density = expon.pdf(x, scale=1/rate)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, exponential_density, color='red', linewidth=2, label=f'Exp({rate})')
    
    # Add mathematical annotations
    plt.text(1, max(exponential_density), 
             f'E[X] = 1/λ = {1/rate}', 
             fontsize=12, fontweight='bold', ha='center', color='darkred')
    
    plt.title(f'Exponential Distribution Exp({rate})', fontsize=14, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def basic_chi_square_functions():
    """
    Demonstrate basic chi-square distribution functions and properties.
    """
    # Chi-square distribution with different degrees of freedom
    df_values = [1, 3, 5, 10]
    x = np.arange(0, 20.1, 0.1)
    
    # Plot chi-square distributions with mathematical annotations
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, df in enumerate(df_values):
        plt.plot(x, chi2.pdf(x, df), color=colors[i], linewidth=2, label=f'df = {df}')
    
    plt.title('Chi-Square Distributions', fontsize=14, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate critical values with mathematical interpretation
    alpha = 0.05
    critical_values = [chi2.ppf(1 - alpha, df) for df in df_values]
    print("Critical values for α = 0.05:")
    for i, df in enumerate(df_values):
        print(f"χ²({df}) = {critical_values[i]:.6f}")
    
    # Verify mathematical properties
    for df in df_values:
        print(f"df = {df}: E[X] = {df}, Var(X) = {2*df}")

def basic_t_distribution_functions():
    """
    Demonstrate basic t-distribution functions and properties.
    """
    # t-distribution with different degrees of freedom
    df_t = [1, 5, 10, 30]
    x = np.arange(-4, 4.1, 0.1)
    
    # Plot t-distributions with normal comparison
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, df in enumerate(df_t):
        plt.plot(x, t.pdf(x, df), color=colors[i], linewidth=2, label=f't(df = {df})')
    
    # Add normal distribution for comparison
    plt.plot(x, norm.pdf(x), color='black', linestyle='--', linewidth=2, label='N(0,1)')
    
    plt.title('t-Distributions vs Normal', fontsize=14, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate critical values with mathematical interpretation
    alpha = 0.05
    t_critical = [t.ppf(1 - alpha/2, df) for df in df_t]
    print("Two-tailed critical values for α = 0.05:")
    for i, df in enumerate(df_t):
        print(f"t({df}) = {t_critical[i]:.6f}")
    
    # Compare with normal critical value
    print(f"Normal critical value (z): {norm.ppf(1 - alpha/2):.6f}")

def basic_f_distribution_functions():
    """
    Demonstrate basic F-distribution functions and properties.
    """
    # F-distribution with different degrees of freedom
    df1_values = [5, 10, 20]
    df2_values = [10, 20, 30]
    x = np.arange(0, 5.1, 0.1)
    
    # Plot F-distributions with mathematical annotations
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green']
    
    for i in range(len(df1_values)):
        plt.plot(x, f.pdf(x, df1_values[i], df2_values[i]), 
                 color=colors[i], linewidth=2, 
                 label=f'F({df1_values[i]},{df2_values[i]})')
    
    plt.title('F-Distributions', fontsize=14, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate critical values with mathematical interpretation
    alpha = 0.05
    f_critical = [f.ppf(1 - alpha, df1, df2) for df1, df2 in zip(df1_values, df2_values)]
    print("Critical values for α = 0.05:")
    for i in range(len(df1_values)):
        print(f"F({df1_values[i]},{df2_values[i]}) = {f_critical[i]:.6f}")

def distribution_fitting():
    """
    Demonstrate fitting distributions to data with mathematical precision.
    """
    # Load sample data for distribution fitting
    mtcars = fetch_openml(name='mtcars', as_frame=True).frame
    
    # Fit normal distribution to MPG with mathematical precision
    mpg_data = mtcars['mpg'].values
    mpg_mean = np.mean(mpg_data)
    mpg_sd = np.std(mpg_data, ddof=1)
    
    # Plot histogram with fitted normal
    plt.figure(figsize=(10, 6))
    plt.hist(mpg_data, bins=15, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    x_range = np.linspace(mpg_data.min(), mpg_data.max(), 100)
    plt.plot(x_range, norm.pdf(x_range, mpg_mean, mpg_sd), 'r-', linewidth=2, label='Fitted Normal')
    
    # Add mathematical annotations
    plt.text(mpg_mean, 0.1, 
             f'N(μ={mpg_mean:.2f}, σ={mpg_sd:.2f})', 
             fontsize=12, fontweight='bold', ha='center', color='red')
    
    plt.title('MPG with Fitted Normal Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('MPG', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Goodness-of-fit test with interpretation
    ks_test_mpg = kstest(mpg_data, 'norm', args=(mpg_mean, mpg_sd))
    print("Kolmogorov-Smirnov Test for Normal Fit:")
    print(f"  D = {ks_test_mpg.statistic:.6f}")
    print(f"  p-value = {ks_test_mpg.pvalue:.6f}")
    print(f"  Interpretation: {'Normal distribution is a good fit' if ks_test_mpg.pvalue > 0.05 else 'Normal distribution may not be appropriate'}")

def comparing_multiple_distributions():
    """
    Function to fit and compare distributions with mathematical rigor.
    """
    # Load data for comparison
    mtcars = fetch_openml(name='mtcars', as_frame=True).frame
    mpg_data = mtcars['mpg'].values
    
    def fit_distributions(data, distributions=["normal", "exponential", "gamma"]):
        results = {}
        
        for dist in distributions:
            if dist == "normal":
                # Fit normal distribution
                params = {"mean": np.mean(data), "std": np.std(data, ddof=1)}
                ks_result = kstest(data, 'norm', args=(params["mean"], params["std"]))
                aic = -2 * np.sum(np.log(norm.pdf(data, params["mean"], params["std"]))) + 2 * 2
            elif dist == "exponential":
                # Fit exponential distribution
                rate = 1/np.mean(data)
                ks_result = kstest(data, 'expon', args=(0, 1/rate))
                aic = -2 * np.sum(np.log(expon.pdf(data, scale=1/rate))) + 2 * 1
            elif dist == "gamma":
                # Fit gamma distribution
                from scipy.stats import gamma
                shape, loc, scale = gamma.fit(data)
                ks_result = kstest(data, 'gamma', args=(shape, loc, scale))
                aic = -2 * np.sum(np.log(gamma.pdf(data, shape, loc, scale))) + 2 * 3
            
            results[dist] = {
                "distribution": dist,
                "ks_statistic": ks_result.statistic,
                "p_value": ks_result.pvalue,
                "aic": aic
            }
        
        return results
    
    # Apply to MPG data with comprehensive analysis
    mpg_fits = fit_distributions(mpg_data)
    
    # Compare results with mathematical interpretation
    print("Distribution Comparison for MPG Data:")
    print("=====================================")
    for dist_name, result in mpg_fits.items():
        print(f"{dist_name} distribution:")
        print(f"  KS statistic: {result['ks_statistic']:.6f}")
        print(f"  p-value: {result['p_value']:.6f}")
        print(f"  AIC: {result['aic']:.6f}")
        print(f"  Interpretation: {'Good fit' if result['p_value'] > 0.05 else 'Poor fit'}\n")
    
    # Find best fit based on AIC
    best_fit = min(mpg_fits.keys(), key=lambda x: mpg_fits[x]['aic'])
    print(f"Best fitting distribution (lowest AIC): {best_fit}")

def practical_example_quality_control():
    """
    Quality control example using Poisson distribution with mathematical precision.
    """
    # Simulate defect data with Poisson distribution
    np.random.seed(123)
    defects_per_batch = np.random.poisson(2.5, 100)
    
    # Calculate probabilities with mathematical interpretation
    print("Quality Control Analysis:")
    print("========================")
    print(f"Average defects per batch (λ): {2.5}")
    print(f"P(X = 0): {poisson.pmf(0, 2.5):.6f}")
    print(f"P(X = 1): {poisson.pmf(1, 2.5):.6f}")
    print(f"P(X ≤ 2): {poisson.cdf(2, 2.5):.6f}")
    print(f"P(X ≥ 5): {1 - poisson.cdf(4, 2.5):.6f}")
    
    # Plot defect distribution with mathematical annotations
    plt.figure(figsize=(10, 6))
    plt.hist(defects_per_batch, bins=range(max(defects_per_batch)+2), 
             density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    x_range = np.arange(0, max(defects_per_batch)+1)
    plt.plot(x_range, poisson.pmf(x_range, 2.5), 'ro-', linewidth=2, markersize=8, label='Poisson PMF')
    
    # Add mathematical annotations
    plt.text(2.5, 0.25, f'E[X] = Var(X) = λ = 2.5', 
             fontsize=12, fontweight='bold', ha='center', color='red')
    
    plt.title('Defects per Batch (Poisson λ=2.5)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Defects', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def practical_example_customer_service():
    """
    Customer service times example using exponential distribution.
    """
    # Simulate service times (exponential distribution)
    np.random.seed(123)
    service_times = np.random.exponential(1/0.5, 100)  # mean = 2 minutes
    
    # Calculate probabilities with mathematical interpretation
    print("Customer Service Time Analysis:")
    print("==============================")
    print(f"Average service time (1/λ): {1/0.5} minutes")
    print(f"P(X < 1 minute): {expon.cdf(1, scale=1/0.5):.6f}")
    print(f"P(X > 5 minutes): {1 - expon.cdf(5, scale=1/0.5):.6f}")
    print(f"P(1 < X < 3 minutes): {expon.cdf(3, scale=1/0.5) - expon.cdf(1, scale=1/0.5):.6f}")
    
    # Plot service time distribution with mathematical annotations
    plt.figure(figsize=(10, 6))
    plt.hist(service_times, bins=20, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
    x_range = np.linspace(0, service_times.max(), 100)
    plt.plot(x_range, expon.pdf(x_range, scale=1/0.5), 'r-', linewidth=2, label='Exponential PDF')
    
    # Add mathematical annotations
    plt.text(2, 0.4, f'E[X] = 1/λ = 2 minutes', 
             fontsize=12, fontweight='bold', ha='center', color='red')
    
    plt.title('Service Times (Exponential λ=0.5)', fontsize=14, fontweight='bold')
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def practical_example_investment_returns():
    """
    Investment returns example using normal distribution.
    """
    # Simulate investment returns (normal distribution)
    np.random.seed(123)
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
    
    # Calculate probabilities with mathematical interpretation
    print("Investment Return Analysis:")
    print("==========================")
    print(f"Daily mean return (μ): {0.001}")
    print(f"Daily volatility (σ): {0.02}")
    print(f"P(positive return): {1 - norm.cdf(0, 0.001, 0.02):.6f}")
    print(f"P(return > 0.05): {1 - norm.cdf(0.05, 0.001, 0.02):.6f}")
    
    # Value at Risk (VaR) calculations
    var_95 = norm.ppf(0.05, 0.001, 0.02)
    var_99 = norm.ppf(0.01, 0.001, 0.02)
    print(f"95% VaR: {var_95:.6f}")
    print(f"99% VaR: {var_99:.6f}")
    
    # Plot return distribution with mathematical annotations
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=30, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    x_range = np.linspace(returns.min(), returns.max(), 100)
    plt.plot(x_range, norm.pdf(x_range, 0.001, 0.02), 'r-', linewidth=2, label='Normal PDF')
    
    # Add mathematical annotations
    plt.text(0.001, 20, f'N(μ=0.001, σ=0.02)', 
             fontsize=12, fontweight='bold', ha='center', color='red')
    
    plt.title('Daily Investment Returns (Normal μ=0.001, σ=0.02)', fontsize=14, fontweight='bold')
    plt.xlabel('Return', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def best_practices_distribution_selection():
    """
    Comprehensive guidelines for distribution selection.
    """
    print("GUIDELINES FOR DISTRIBUTION SELECTION:")
    print("=====================================")
    print("1. Normal Distribution N(μ,σ²):")
    print("   - Use for: Symmetric, continuous data")
    print("   - Examples: Heights, weights, measurement errors")
    print("   - Properties: Bell-shaped, symmetric around mean\n")
    
    print("2. Binomial Distribution B(n,p):")
    print("   - Use for: Counts of successes in fixed trials")
    print("   - Examples: Coin flips, defect counts, survey responses")
    print("   - Properties: Discrete, bounded [0,n]\n")
    
    print("3. Poisson Distribution Poi(λ):")
    print("   - Use for: Counts of rare events")
    print("   - Examples: Customer arrivals, defect counts, accidents")
    print("   - Properties: Discrete, unbounded, E[X] = Var(X) = λ\n")
    
    print("4. Exponential Distribution Exp(λ):")
    print("   - Use for: Time between events")
    print("   - Examples: Service times, equipment lifetimes")
    print("   - Properties: Memoryless, E[X] = 1/λ\n")
    
    print("5. Chi-square Distribution χ²(k):")
    print("   - Use for: Variance testing, goodness-of-fit")
    print("   - Examples: Sample variance, contingency tables")
    print("   - Properties: E[X] = k, Var(X) = 2k\n")
    
    print("6. t-Distribution t(k):")
    print("   - Use for: Small sample inference")
    print("   - Examples: Confidence intervals, hypothesis tests")
    print("   - Properties: Heavier tails than normal\n")
    
    print("7. F-Distribution F(k₁,k₂):")
    print("   - Use for: Variance comparisons, ANOVA")
    print("   - Examples: Comparing group variances")
    print("   - Properties: Ratio of chi-square variables")

def validate_distribution(data, distribution, params):
    """
    Function to validate distribution fit with comprehensive checks.
    """
    print(f"Distribution Validation for {distribution} distribution:")
    print("=====================================================")
    
    # Visual checks
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram with fitted curve
    axes[0, 0].hist(data, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    x_range = np.linspace(data.min(), data.max(), 100)
    
    if distribution == "normal":
        axes[0, 0].plot(x_range, norm.pdf(x_range, params['mean'], params['std']), 'r-', linewidth=2)
    elif distribution == "exponential":
        axes[0, 0].plot(x_range, expon.pdf(x_range, scale=1/params['rate']), 'r-', linewidth=2)
    elif distribution == "poisson":
        x_discrete = np.arange(0, int(data.max())+1)
        axes[0, 0].plot(x_discrete, poisson.pmf(x_discrete, params['lambda']), 'ro-', linewidth=2)
    
    axes[0, 0].set_title(f"Histogram with Fitted {distribution}")
    
    # Q-Q plot
    if distribution == "normal":
        stats.probplot(data, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Q-Q Plot for Normality")
    
    # Empirical vs theoretical CDF
    from scipy.stats import ecdf
    ecdf_result = ecdf(data)
    axes[1, 0].step(ecdf_result.x, ecdf_result.y, where='post', label='Empirical CDF')
    
    if distribution == "normal":
        axes[1, 0].plot(x_range, norm.cdf(x_range, params['mean'], params['std']), 'r-', linewidth=2, label='Theoretical CDF')
    elif distribution == "exponential":
        axes[1, 0].plot(x_range, expon.cdf(x_range, scale=1/params['rate']), 'r-', linewidth=2, label='Theoretical CDF')
    
    axes[1, 0].set_title("Empirical vs Theoretical CDF")
    axes[1, 0].legend()
    
    # P-P plot
    if distribution == "normal":
        theoretical_quantiles = norm.ppf(np.linspace(0.01, 0.99, len(data)), params['mean'], params['std'])
        axes[1, 1].scatter(np.sort(data), theoretical_quantiles)
        axes[1, 1].plot([data.min(), data.max()], [data.min(), data.max()], 'r--')
        axes[1, 1].set_title("P-P Plot")
        axes[1, 1].set_xlabel("Sample Quantiles")
        axes[1, 1].set_ylabel("Theoretical Quantiles")
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    if distribution == "normal":
        ks_result = kstest(data, 'norm', args=(params['mean'], params['std']))
        shapiro_result = shapiro(data)
        print(f"Kolmogorov-Smirnov test p-value: {ks_result.pvalue:.6f}")
        print(f"Shapiro-Wilk test p-value: {shapiro_result.pvalue:.6f}")
    elif distribution == "exponential":
        ks_result = kstest(data, 'expon', args=(0, 1/params['rate']))
        print(f"Kolmogorov-Smirnov test p-value: {ks_result.pvalue:.6f}")
    
    return {"ks_result": ks_result, "shapiro_result": shapiro_result if distribution == "normal" else None}

def example_validation():
    """
    Example of using the validation function with normal data.
    """
    # Generate normal data for validation example
    np.random.seed(123)
    normal_data = np.random.normal(0, 1, 1000)
    
    # Validate normal distribution fit
    params = {"mean": np.mean(normal_data), "std": np.std(normal_data, ddof=1)}
    results = validate_distribution(normal_data, "normal", params)
    
    return results

if __name__ == "__main__":
    # Example usage
    basic_normal_distribution()
    normal_distribution_properties()
    checking_normality()
    basic_binomial_functions()
    binomial_distribution_examples()
    basic_poisson_functions()
    poisson_distribution_examples()
    basic_exponential_functions()
    basic_chi_square_functions()
    basic_t_distribution_functions()
    basic_f_distribution_functions()
    distribution_fitting()
    comparing_multiple_distributions()
    practical_example_quality_control()
    practical_example_customer_service()
    practical_example_investment_returns()
    best_practices_distribution_selection()
    example_validation() 