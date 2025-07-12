"""
Variable Selection and Model Building Toolkit
============================================

This module provides a comprehensive set of functions and workflows for variable selection 
and model building, including:
- Stepwise selection methods (forward, backward, RFE)
- Regularization methods (Ridge, Lasso, Elastic Net)
- Principal Component Regression (PCR) and Partial Least Squares (PLS)
- Model diagnostics and validation
- Cross-validation and model comparison
- Multicollinearity detection

Each function is documented and can be referenced from the corresponding theory in the markdown file.
"""

# === Imports ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_sample_data(n_samples=200, n_features=10, seed=42):
    """
    Create sample data with known correlation structure and true coefficients.
    Corresponds to: 'Python Implementation' in the markdown.
    
    Parameters:
    -----------
    n_samples : int
        Number of observations
    n_features : int
        Number of features
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X, y, beta_true, feature_names, df)
        X: feature matrix, y: target variable, beta_true: true coefficients,
        feature_names: list of feature names, df: complete DataFrame
    """
    np.random.seed(seed)
    
    # Create correlated features
    X = np.random.randn(n_samples, n_features)
    # Add some correlation between features
    X[:, 2] = X[:, 0] * 0.8 + X[:, 1] * 0.2 + np.random.randn(n_samples) * 0.1
    X[:, 3] = X[:, 1] * 0.7 + np.random.randn(n_samples) * 0.3
    
    # Create target variable (only some features are important)
    beta_true = np.array([2.0, -1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ beta_true + np.random.randn(n_samples) * 0.5
    
    # Create DataFrame
    feature_names = [f'X{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y
    
    return X, y, beta_true, feature_names, df

def perform_stepwise_selection(X, y, feature_names, n_features_to_select=5, cv=5):
    """
    Perform stepwise selection methods including forward selection, backward elimination, and RFE.
    Corresponds to: 'Python Implementation' in the Stepwise Selection Methods section.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    feature_names : list
        List of feature names
    n_features_to_select : int
        Number of features to select
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    dict : Dictionary containing selection results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Forward Selection
    forward_selector = SequentialFeatureSelector(
        LinearRegression(), 
        n_features_to_select=n_features_to_select, 
        direction='forward',
        scoring='neg_mean_squared_error',
        cv=cv
    )
    forward_selector.fit(X_train, y_train)
    forward_selected = forward_selector.get_support()
    forward_features = [feature_names[i] for i in range(len(feature_names)) if forward_selected[i]]
    
    # Backward Elimination
    backward_selector = SequentialFeatureSelector(
        LinearRegression(), 
        n_features_to_select=n_features_to_select, 
        direction='backward',
        scoring='neg_mean_squared_error',
        cv=cv
    )
    backward_selector.fit(X_train, y_train)
    backward_selected = backward_selector.get_support()
    backward_features = [feature_names[i] for i in range(len(feature_names)) if backward_selected[i]]
    
    # Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    rfe_selected = rfe.get_support()
    rfe_features = [feature_names[i] for i in range(len(feature_names)) if rfe_selected[i]]
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'forward_features': forward_features,
        'backward_features': backward_features,
        'rfe_features': rfe_features,
        'forward_selected': forward_selected,
        'backward_selected': backward_selected,
        'rfe_selected': rfe_selected
    }

def evaluate_model(X_train, X_test, y_train, y_test, feature_names, method_name):
    """
    Evaluate a model and calculate performance metrics including AIC and BIC.
    Corresponds to: 'Python Implementation' in the Stepwise Selection Methods section.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test feature matrices
    y_train, y_test : array-like
        Training and test target variables
    feature_names : list
        List of feature names
    method_name : str
        Name of the method for reporting
        
    Returns:
    --------
    tuple : (model, mse, r2, aic, bic)
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate AIC and BIC
    n = len(y_train)
    p = X_train.shape[1]
    rss = np.sum((y_train - model.predict(X_train))**2)
    sigma2_hat = rss / n
    aic = n * np.log(sigma2_hat) + 2 * p
    bic = n * np.log(sigma2_hat) + p * np.log(n)
    
    print(f"\n{method_name}:")
    print(f"Features: {feature_names}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"AIC: {aic:.4f}")
    print(f"BIC: {bic:.4f}")
    
    return model, mse, r2, aic, bic

def perform_regularization_analysis(X_train, X_test, y_train, y_test, feature_names, beta_true):
    """
    Perform regularization analysis using Ridge, Lasso, and Elastic Net.
    Corresponds to: 'Python Implementation' in the Regularization Methods section.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test feature matrices
    y_train, y_test : array-like
        Training and test target variables
    feature_names : list
        List of feature names
    beta_true : array-like
        True coefficients for comparison
        
    Returns:
    --------
    dict : Dictionary containing regularization results
    """
    # Standardize features for regularization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    ridge_r2 = r2_score(y_test, ridge_pred)
    
    # Lasso Regression
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_scaled, y_train)
    lasso_pred = lasso.predict(X_test_scaled)
    lasso_mse = mean_squared_error(y_test, lasso_pred)
    lasso_r2 = r2_score(y_test, lasso_pred)
    
    # Elastic Net
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic_net.fit(X_train_scaled, y_train)
    elastic_pred = elastic_net.predict(X_test_scaled)
    elastic_mse = mean_squared_error(y_test, elastic_pred)
    elastic_r2 = r2_score(y_test, elastic_pred)
    
    # Compare coefficients
    coef_comparison = pd.DataFrame({
        'True': beta_true,
        'Ridge': ridge.coef_,
        'Lasso': lasso.coef_,
        'Elastic Net': elastic_net.coef_
    }, index=feature_names)
    
    return {
        'ridge': ridge,
        'lasso': lasso,
        'elastic_net': elastic_net,
        'ridge_mse': ridge_mse,
        'lasso_mse': lasso_mse,
        'elastic_mse': elastic_mse,
        'ridge_r2': ridge_r2,
        'lasso_r2': lasso_r2,
        'elastic_r2': elastic_r2,
        'coef_comparison': coef_comparison,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled
    }

def plot_coefficient_paths(X_train_scaled, y_train, feature_names):
    """
    Plot coefficient paths for Ridge and Lasso regression.
    Corresponds to: 'Python Implementation' in the Regularization Methods section.
    
    Parameters:
    -----------
    X_train_scaled : array-like
        Standardized training features
    y_train : array-like
        Training target variable
    feature_names : list
        List of feature names
    """
    alphas = np.logspace(-3, 1, 50)
    ridge_coefs = []
    lasso_coefs = []
    
    for alpha in alphas:
        ridge_temp = Ridge(alpha=alpha)
        ridge_temp.fit(X_train_scaled, y_train)
        ridge_coefs.append(ridge_temp.coef_)
        
        lasso_temp = Lasso(alpha=alpha)
        lasso_temp.fit(X_train_scaled, y_train)
        lasso_coefs.append(lasso_temp.coef_)
    
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)
    
    # Plot coefficient paths
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    for i in range(len(feature_names)):
        plt.plot(alphas, ridge_coefs[:, i], label=feature_names[i])
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficients')
    plt.title('Ridge Regression Coefficient Paths')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for i in range(len(feature_names)):
        plt.plot(alphas, lasso_coefs[:, i], label=feature_names[i])
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficients')
    plt.title('Lasso Regression Coefficient Paths')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def perform_cv_regularization(X_train_scaled, y_train):
    """
    Perform cross-validation for optimal regularization parameters.
    Corresponds to: 'Python Implementation' in the Regularization Methods section.
    
    Parameters:
    -----------
    X_train_scaled : array-like
        Standardized training features
    y_train : array-like
        Training target variable
        
    Returns:
    --------
    dict : Dictionary containing CV results
    """
    alphas = np.logspace(-3, 1, 50)
    
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X_train_scaled, y_train)
    
    lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=1000)
    lasso_cv.fit(X_train_scaled, y_train)
    
    elastic_cv = ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5, max_iter=1000)
    elastic_cv.fit(X_train_scaled, y_train)
    
    return {
        'ridge_cv': ridge_cv,
        'lasso_cv': lasso_cv,
        'elastic_cv': elastic_cv,
        'optimal_ridge_alpha': ridge_cv.alpha_,
        'optimal_lasso_alpha': lasso_cv.alpha_,
        'optimal_elastic_alpha': elastic_cv.alpha_,
        'optimal_elastic_l1_ratio': elastic_cv.l1_ratio_
    }

def perform_pcr_pls_analysis(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """
    Perform Principal Component Regression (PCR) and Partial Least Squares (PLS) analysis.
    Corresponds to: 'Python Implementation' in the PCR and PLS section.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test feature matrices
    y_train, y_test : array-like
        Training and test target variables
    X_train_scaled, X_test_scaled : array-like
        Standardized training and test features
        
    Returns:
    --------
    dict : Dictionary containing PCR and PLS results
    """
    # Principal Component Regression (PCR)
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Fit PCR with different numbers of components
    n_components_range = range(1, min(X_train.shape[1] + 1, 11))
    pcr_scores = []
    
    for n_comp in n_components_range:
        pcr_model = LinearRegression()
        pcr_model.fit(X_train_pca[:, :n_comp], y_train)
        pcr_pred = pcr_model.predict(X_test_pca[:, :n_comp])
        pcr_mse = mean_squared_error(y_test, pcr_pred)
        pcr_scores.append(pcr_mse)
    
    # Partial Least Squares (PLS)
    pls_scores = []
    for n_comp in n_components_range:
        pls_model = PLSRegression(n_components=n_comp)
        pls_model.fit(X_train_scaled, y_train)
        pls_pred = pls_model.predict(X_test_scaled)
        pls_mse = mean_squared_error(y_test, pls_pred)
        pls_scores.append(pls_mse)
    
    # Find optimal number of components
    optimal_pcr_components = n_components_range[np.argmin(pcr_scores)]
    optimal_pls_components = n_components_range[np.argmin(pls_scores)]
    
    # Fit final models
    pcr_final = LinearRegression()
    pcr_final.fit(X_train_pca[:, :optimal_pcr_components], y_train)
    pcr_final_pred = pcr_final.predict(X_test_pca[:, :optimal_pcr_components])
    pcr_final_mse = mean_squared_error(y_test, pcr_final_pred)
    
    pls_final = PLSRegression(n_components=optimal_pls_components)
    pls_final.fit(X_train_scaled, y_train)
    pls_final_pred = pls_final.predict(X_test_scaled)
    pls_final_mse = mean_squared_error(y_test, pls_final_pred)
    
    return {
        'pcr_scores': pcr_scores,
        'pls_scores': pls_scores,
        'n_components_range': n_components_range,
        'optimal_pcr_components': optimal_pcr_components,
        'optimal_pls_components': optimal_pls_components,
        'pcr_final_mse': pcr_final_mse,
        'pls_final_mse': pls_final_mse,
        'pcr_final_pred': pcr_final_pred,
        'pls_final_pred': pls_final_pred
    }

def plot_pcr_pls_comparison(pcr_pls_results):
    """
    Plot comparison of PCR and PLS performance.
    Corresponds to: 'Python Implementation' in the PCR and PLS section.
    
    Parameters:
    -----------
    pcr_pls_results : dict
        Results from perform_pcr_pls_analysis()
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(pcr_pls_results['n_components_range'], pcr_pls_results['pcr_scores'], 'bo-', label='PCR')
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Squared Error')
    plt.title('Principal Component Regression')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(pcr_pls_results['n_components_range'], pcr_pls_results['pls_scores'], 'ro-', label='PLS')
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Squared Error')
    plt.title('Partial Least Squares')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def perform_model_diagnostics(best_model, X_test_scaled, y_test, X_train_scaled, feature_names):
    """
    Perform comprehensive model diagnostics including residual analysis and multicollinearity detection.
    Corresponds to: 'Python Implementation' in the Model Diagnostics section.
    
    Parameters:
    -----------
    best_model : sklearn estimator
        The best performing model
    X_test_scaled : array-like
        Standardized test features
    y_test : array-like
        Test target variable
    X_train_scaled : array-like
        Standardized training features
    feature_names : list
        List of feature names
        
    Returns:
    --------
    dict : Dictionary containing diagnostic results
    """
    y_pred_best = best_model.predict(X_test_scaled)
    residuals = y_test - y_pred_best
    
    # Diagnostic plots
    plt.figure(figsize=(15, 10))
    
    # Residuals vs fitted
    plt.subplot(2, 3, 1)
    plt.scatter(y_pred_best, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.grid(True)
    
    # Q-Q plot
    plt.subplot(2, 3, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    # Histogram of residuals
    plt.subplot(2, 3, 3)
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.grid(True)
    
    # Residuals vs index
    plt.subplot(2, 3, 4)
    plt.plot(residuals, 'o-', alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Observation Index')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Index')
    plt.grid(True)
    
    # Scale-location plot
    plt.subplot(2, 3, 5)
    plt.scatter(y_pred_best, np.sqrt(np.abs(residuals)), alpha=0.6)
    plt.xlabel('Fitted Values')
    plt.ylabel('√|Residuals|')
    plt.title('Scale-Location Plot')
    plt.grid(True)
    
    # Leverage plot (simplified)
    plt.subplot(2, 3, 6)
    leverage = np.diag(X_train_scaled @ np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T)
    plt.scatter(leverage, residuals[:len(leverage)], alpha=0.6)
    plt.xlabel('Leverage')
    plt.ylabel('Residuals')
    plt.title('Leverage Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Multicollinearity detection
    def calculate_vif(X):
        vif_data = pd.DataFrame()
        vif_data["Variable"] = feature_names
        vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        return vif_data
    
    vif_results = calculate_vif(X_train_scaled)
    
    # Condition number
    condition_number = np.linalg.cond(X_train_scaled)
    
    # Correlation matrix
    correlation_matrix = pd.DataFrame(X_train_scaled, columns=feature_names).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return {
        'residuals': residuals,
        'vif_results': vif_results,
        'condition_number': condition_number,
        'correlation_matrix': correlation_matrix
    }

def create_model_summary(stepwise_results, regularization_results, pcr_pls_results, 
                        full_model, full_mse, full_r2, feature_names):
    """
    Create a comprehensive summary of all model performances.
    Corresponds to: 'Python Implementation' in the Model Diagnostics section.
    
    Parameters:
    -----------
    stepwise_results : dict
        Results from stepwise selection
    regularization_results : dict
        Results from regularization analysis
    pcr_pls_results : dict
        Results from PCR and PLS analysis
    full_model : sklearn estimator
        Full model (OLS)
    full_mse, full_r2 : float
        Performance metrics for full model
    feature_names : list
        List of feature names
        
    Returns:
    --------
    pandas.DataFrame : Summary of all model performances
    """
    # Evaluate stepwise models
    forward_model, forward_mse, forward_r2, forward_aic, forward_bic = evaluate_model(
        stepwise_results['X_train'], stepwise_results['X_test'], 
        stepwise_results['y_train'], stepwise_results['y_test'], 
        stepwise_results['forward_features'], "Forward Selection"
    )
    
    backward_model, backward_mse, backward_r2, backward_aic, backward_bic = evaluate_model(
        stepwise_results['X_train'], stepwise_results['X_test'], 
        stepwise_results['y_train'], stepwise_results['y_test'], 
        stepwise_results['backward_features'], "Backward Elimination"
    )
    
    rfe_model, rfe_mse, rfe_r2, rfe_aic, rfe_bic = evaluate_model(
        stepwise_results['X_train'], stepwise_results['X_test'], 
        stepwise_results['y_train'], stepwise_results['y_test'], 
        stepwise_results['rfe_features'], "RFE"
    )
    
    # Create summary DataFrame
    models_summary = pd.DataFrame({
        'Model': ['OLS', 'Forward Selection', 'Backward Elimination', 'RFE', 
                  'Ridge', 'Lasso', 'Elastic Net', 'PCR', 'PLS'],
        'MSE': [full_mse, forward_mse, backward_mse, rfe_mse, 
                regularization_results['ridge_mse'], regularization_results['lasso_mse'], 
                regularization_results['elastic_mse'], pcr_pls_results['pcr_final_mse'], 
                pcr_pls_results['pls_final_mse']],
        'R²': [full_r2, forward_r2, backward_r2, rfe_r2, 
               regularization_results['ridge_r2'], regularization_results['lasso_r2'], 
               regularization_results['elastic_r2'], 
               r2_score(stepwise_results['y_test'], pcr_pls_results['pcr_final_pred']), 
               r2_score(stepwise_results['y_test'], pcr_pls_results['pls_final_pred'])]
    })
    
    return models_summary.sort_values('MSE')

if __name__ == "__main__":
    """
    Main demonstration block showing how to use all variable selection and model building 
    functions in a coherent workflow.
    """
    print("=== VARIABLE SELECTION AND MODEL BUILDING DEMONSTRATION ===\n")
    
    # 1. Create sample data
    print("1. Creating sample data...")
    X, y, beta_true, feature_names, df = create_sample_data()
    print(f"Dataset shape: {df.shape}")
    print(f"True coefficients: {beta_true}")
    print(f"Feature names: {feature_names}\n")
    
    # 2. Stepwise Selection
    print("2. Performing stepwise selection...")
    stepwise_results = perform_stepwise_selection(X, y, feature_names)
    print(f"Forward Selection Features: {stepwise_results['forward_features']}")
    print(f"Backward Elimination Features: {stepwise_results['backward_features']}")
    print(f"RFE Features: {stepwise_results['rfe_features']}\n")
    
    # 3. Regularization Analysis
    print("3. Performing regularization analysis...")
    regularization_results = perform_regularization_analysis(
        stepwise_results['X_train'], stepwise_results['X_test'],
        stepwise_results['y_train'], stepwise_results['y_test'],
        feature_names, beta_true
    )
    print("Coefficient Comparison:")
    print(regularization_results['coef_comparison'])
    
    # Plot coefficient paths
    plot_coefficient_paths(regularization_results['X_train_scaled'], 
                          stepwise_results['y_train'], feature_names)
    
    # Cross-validation for optimal parameters
    cv_results = perform_cv_regularization(regularization_results['X_train_scaled'], 
                                         stepwise_results['y_train'])
    print(f"\nOptimal Ridge alpha: {cv_results['optimal_ridge_alpha']:.4f}")
    print(f"Optimal Lasso alpha: {cv_results['optimal_lasso_alpha']:.4f}")
    print(f"Optimal Elastic Net alpha: {cv_results['optimal_elastic_alpha']:.4f}, "
          f"l1_ratio: {cv_results['optimal_elastic_l1_ratio']:.1f}\n")
    
    # 4. PCR and PLS Analysis
    print("4. Performing PCR and PLS analysis...")
    pcr_pls_results = perform_pcr_pls_analysis(
        stepwise_results['X_train'], stepwise_results['X_test'],
        stepwise_results['y_train'], stepwise_results['y_test'],
        regularization_results['X_train_scaled'], regularization_results['X_test_scaled']
    )
    print(f"Optimal PCR components: {pcr_pls_results['optimal_pcr_components']}")
    print(f"Optimal PLS components: {pcr_pls_results['optimal_pls_components']}")
    print(f"PCR MSE: {pcr_pls_results['pcr_final_mse']:.4f}")
    print(f"PLS MSE: {pcr_pls_results['pls_final_mse']:.4f}\n")
    
    # Plot PCR/PLS comparison
    plot_pcr_pls_comparison(pcr_pls_results)
    
    # 5. Model Diagnostics
    print("5. Performing model diagnostics...")
    # Use the best model (Lasso with CV) for diagnostics
    best_model = cv_results['lasso_cv']
    diagnostic_results = perform_model_diagnostics(
        best_model, regularization_results['X_test_scaled'], 
        stepwise_results['y_test'], regularization_results['X_train_scaled'], 
        feature_names
    )
    
    print("Variance Inflation Factors:")
    print(diagnostic_results['vif_results'])
    print(f"\nCondition Number: {diagnostic_results['condition_number']:.2f}")
    
    # 6. Model Summary
    print("6. Creating model performance summary...")
    # Evaluate full model for comparison
    full_model, full_mse, full_r2, full_aic, full_bic = evaluate_model(
        stepwise_results['X_train'], stepwise_results['X_test'],
        stepwise_results['y_train'], stepwise_results['y_test'],
        feature_names, "Full Model"
    )
    
    models_summary = create_model_summary(
        stepwise_results, regularization_results, pcr_pls_results,
        full_model, full_mse, full_r2, feature_names
    )
    
    print("\nModel Performance Summary (sorted by MSE):")
    print(models_summary)
    
    print("\n=== DEMONSTRATION COMPLETE ===")
    print("All variable selection and model building techniques have been demonstrated.")
    print("Refer to the markdown file for theoretical explanations and interpretation guidelines.") 