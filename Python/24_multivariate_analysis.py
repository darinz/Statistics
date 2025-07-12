"""
Multivariate Analysis Toolkit
============================

This module provides a comprehensive set of functions and workflows for multivariate analysis, including:
- Principal Component Analysis (PCA)
- Factor Analysis
- Cluster Analysis (K-means, Hierarchical)
- Discriminant Analysis (LDA, QDA)
- Canonical Correlation Analysis (CCA)
- Data generation and preprocessing
- Visualization and interpretation tools

Each function is documented and can be referenced from the corresponding theory in the markdown file.
"""

# === Imports ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cross_decomposition import CCA
from scipy.stats import multivariate_normal, chi2
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
import warnings
warnings.filterwarnings('ignore')

def create_multivariate_data(n_samples=100, n_variables=5, seed=123):
    """
    Create synthetic multivariate data with known correlation structure.
    Corresponds to: 'Python Implementation and Interpretation' in the markdown.
    
    Parameters:
    -----------
    n_samples : int
        Number of observations
    n_variables : int
        Number of variables
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    df : pandas.DataFrame
        Multivariate dataset with correlated variables
    """
    np.random.seed(seed)
    
    # Create correlated variables
    correlation_matrix = np.array([
        [1.0, 0.8, 0.6, 0.4, 0.2],
        [0.8, 1.0, 0.7, 0.5, 0.3],
        [0.6, 0.7, 1.0, 0.6, 0.4],
        [0.4, 0.5, 0.6, 1.0, 0.5],
        [0.2, 0.3, 0.4, 0.5, 1.0]
    ])
    
    # Generate multivariate normal data
    multivariate_data = multivariate_normal.rvs(
        mean=np.zeros(n_variables), 
        cov=correlation_matrix, 
        size=n_samples
    )
    
    # Create DataFrame
    df = pd.DataFrame(multivariate_data, columns=['Var1', 'Var2', 'Var3', 'Var4', 'Var5'])
    return df

def perform_pca_analysis(data, n_components=None):
    """
    Perform Principal Component Analysis on the data.
    Corresponds to: 'Python Implementation and Interpretation' in the markdown.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data for PCA
    n_components : int, optional
        Number of components to retain
        
    Returns:
    --------
    dict : Dictionary containing PCA results
    """
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit(data_scaled)
    
    # Transform data to principal components
    pca_scores = pca.transform(data_scaled)
    pca_df = pd.DataFrame(pca_scores, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    
    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=data.columns
    )
    
    return {
        'pca': pca,
        'scores': pca_df,
        'loadings': loadings,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'data_scaled': data_scaled
    }

def plot_pca_results(pca_results):
    """
    Create visualization plots for PCA results.
    Corresponds to: 'Python Implementation and Interpretation' in the markdown.
    
    Parameters:
    -----------
    pca_results : dict
        Results from perform_pca_analysis()
    """
    # Create scree plot
    plt.figure(figsize=(15, 5))
    
    # Scree plot
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(pca_results['explained_variance_ratio']) + 1), 
             pca_results['explained_variance_ratio'], 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.grid(True)
    
    # Cumulative variance plot
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(pca_results['explained_variance_ratio']) + 1), 
             pca_results['cumulative_variance'], 'ro-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance Plot')
    plt.grid(True)
    
    # Biplot (first two components)
    plt.subplot(1, 3, 3)
    pca_scores = pca_results['scores'].values
    plt.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.6)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Biplot')
    
    # Add variable loadings as arrows
    pca = pca_results['pca']
    for i, var in enumerate(pca_results['loadings'].index):
        plt.arrow(0, 0, pca.components_[0, i] * 3, pca.components_[1, i] * 3, 
                  color='red', alpha=0.7, head_width=0.1)
        plt.text(pca.components_[0, i] * 3.2, pca.components_[1, i] * 3.2, var, 
                 color='red', fontsize=12)
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def perform_factor_analysis(data, n_factors=2, rotation=None):
    """
    Perform Factor Analysis on the data.
    Corresponds to: 'Python Implementation' in the Factor Analysis section.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data for factor analysis
    n_factors : int
        Number of factors to extract
    rotation : str, optional
        Rotation method ('varimax', 'promax', etc.)
        
    Returns:
    --------
    dict : Dictionary containing factor analysis results
    """
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # KMO test
    kmo_all, kmo_model = calculate_kmo(data_scaled)
    
    # Bartlett's test of sphericity
    def bartlett_sphericity(data):
        n = data.shape[0]
        p = data.shape[1]
        chi_square = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(np.corrcoef(data.T)))
        df = p * (p - 1) / 2
        p_value = chi2.sf(chi_square, df)
        return chi_square, p_value
    
    chi_square, p_value = bartlett_sphericity(data_scaled)
    
    # Factor analysis
    fa = FactorAnalyzer(rotation=rotation, n_factors=n_factors)
    fa.fit(data_scaled)
    
    # Get results
    loadings = fa.loadings_
    communalities = fa.get_communalities()
    variance = fa.get_factor_variance()
    
    return {
        'factor_analyzer': fa,
        'loadings': pd.DataFrame(loadings, index=data.columns, 
                                columns=[f'Factor{i+1}' for i in range(n_factors)]),
        'communalities': communalities,
        'variance_explained': variance[0],
        'proportional_variance': variance[1],
        'cumulative_variance': variance[2],
        'kmo': kmo_model,
        'bartlett_chi_square': chi_square,
        'bartlett_p_value': p_value
    }

def perform_clustering_analysis(data, max_clusters=10):
    """
    Perform cluster analysis using K-means and hierarchical clustering.
    Corresponds to: 'Python Implementation' in the Cluster Analysis section.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data for clustering
    max_clusters : int
        Maximum number of clusters to test
        
    Returns:
    --------
    dict : Dictionary containing clustering results
    """
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Find optimal number of clusters using elbow method
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))
    
    # Find optimal k (simplified: using silhouette score)
    optimal_k = K_range[np.argmax(silhouette_scores)]
    
    # Perform K-means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data_scaled)
    
    # Hierarchical clustering
    linkage_matrix = linkage(data_scaled, method='ward')
    
    # Cluster validation
    silhouette = silhouette_score(data_scaled, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(data_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(data_scaled, cluster_labels)
    
    return {
        'kmeans': kmeans,
        'cluster_labels': cluster_labels,
        'optimal_k': optimal_k,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'linkage_matrix': linkage_matrix,
        'validation_scores': {
            'silhouette': silhouette,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin
        },
        'data_scaled': data_scaled
    }

def plot_clustering_results(clustering_results, pca_scores=None):
    """
    Create visualization plots for clustering results.
    Corresponds to: 'Python Implementation' in the Cluster Analysis section.
    
    Parameters:
    -----------
    clustering_results : dict
        Results from perform_clustering_analysis()
    pca_scores : array-like, optional
        PCA scores for 2D visualization
    """
    plt.figure(figsize=(15, 5))
    
    # Plot elbow method
    plt.subplot(1, 3, 1)
    K_range = range(2, len(clustering_results['inertias']) + 2)
    plt.plot(K_range, clustering_results['inertias'], 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    
    # Plot silhouette scores
    plt.subplot(1, 3, 2)
    plt.plot(K_range, clustering_results['silhouette_scores'], 'ro-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.grid(True)
    
    # Plot clusters (using PCA if available)
    plt.subplot(1, 3, 3)
    if pca_scores is not None:
        scatter = plt.scatter(pca_scores[:, 0], pca_scores[:, 1], 
                             c=clustering_results['cluster_labels'], cmap='viridis')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
    else:
        scatter = plt.scatter(clustering_results['data_scaled'][:, 0], 
                             clustering_results['data_scaled'][:, 1], 
                             c=clustering_results['cluster_labels'], cmap='viridis')
        plt.xlabel('First Variable')
        plt.ylabel('Second Variable')
    
    plt.title(f'K-means Clustering (k={clustering_results["optimal_k"]})')
    plt.colorbar(scatter)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(clustering_results['linkage_matrix'], 
               labels=range(len(clustering_results['data_scaled'])), 
               leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

def create_discriminant_data(n_samples_per_group=50, seed=42):
    """
    Create sample data with known groups for discriminant analysis.
    Corresponds to: 'Python Implementation' in the Discriminant Analysis section.
    
    Parameters:
    -----------
    n_samples_per_group : int
        Number of samples per group
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X, y) where X is features and y is group labels
    """
    np.random.seed(seed)
    
    # Generate three groups with different means
    group1 = multivariate_normal.rvs(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=n_samples_per_group)
    group2 = multivariate_normal.rvs(mean=[3, 3], cov=[[1, 0.5], [0.5, 1]], size=n_samples_per_group)
    group3 = multivariate_normal.rvs(mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=n_samples_per_group)
    
    # Combine data
    X = np.vstack([group1, group2, group3])
    y = np.hstack([np.zeros(n_samples_per_group), 
                   np.ones(n_samples_per_group), 
                   np.ones(n_samples_per_group) * 2])
    
    return X, y

def perform_discriminant_analysis(X, y, test_size=0.3):
    """
    Perform Linear and Quadratic Discriminant Analysis.
    Corresponds to: 'Python Implementation' in the Discriminant Analysis section.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Group labels
    test_size : float
        Proportion of data for testing
        
    Returns:
    --------
    dict : Dictionary containing discriminant analysis results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=42, stratify=y)
    
    # Linear Discriminant Analysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred_lda = lda.predict(X_test)
    
    # Quadratic Discriminant Analysis
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    y_pred_qda = qda.predict(X_test)
    
    # Cross-validation comparison
    lda_scores = cross_val_score(lda, X, y, cv=5)
    qda_scores = cross_val_score(qda, X, y, cv=5)
    
    return {
        'lda': lda,
        'qda': qda,
        'y_test': y_test,
        'y_pred_lda': y_pred_lda,
        'y_pred_qda': y_pred_qda,
        'lda_cv_scores': lda_scores,
        'qda_cv_scores': qda_scores,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train
    }

def plot_decision_boundaries(X, y, discriminant_results):
    """
    Plot decision boundaries for LDA and QDA.
    Corresponds to: 'Python Implementation' in the Discriminant Analysis section.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Group labels
    discriminant_results : dict
        Results from perform_discriminant_analysis()
    """
    def plot_decision_boundary(X, y, model, title):
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_decision_boundary(X, y, discriminant_results['lda'], 'Linear Discriminant Analysis')
    plt.subplot(1, 2, 2)
    plot_decision_boundary(X, y, discriminant_results['qda'], 'Quadratic Discriminant Analysis')
    plt.tight_layout()
    plt.show()

def perform_canonical_correlation(X_set, Y_set, n_components=2):
    """
    Perform Canonical Correlation Analysis between two sets of variables.
    Corresponds to: 'Python Implementation' in the Canonical Correlation Analysis section.
    
    Parameters:
    -----------
    X_set : array-like
        First set of variables
    Y_set : array-like
        Second set of variables
    n_components : int
        Number of canonical components
        
    Returns:
    --------
    dict : Dictionary containing CCA results
    """
    # Perform CCA
    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X_set, Y_set)
    
    # Canonical correlations
    canonical_correlations = np.corrcoef(X_c.T, Y_c.T)[:n_components, n_components:]
    
    # Canonical loadings
    X_loadings = cca.x_weights_
    Y_loadings = cca.y_weights_
    
    return {
        'cca': cca,
        'X_canonical': X_c,
        'Y_canonical': Y_c,
        'canonical_correlations': np.diag(canonical_correlations),
        'X_loadings': pd.DataFrame(X_loadings, 
                                  columns=[f'CC{i+1}' for i in range(n_components)],
                                  index=[f'X{i+1}' for i in range(X_set.shape[1])]),
        'Y_loadings': pd.DataFrame(Y_loadings, 
                                  columns=[f'CC{i+1}' for i in range(n_components)],
                                  index=[f'Y{i+1}' for i in range(Y_set.shape[1])])
    }

def plot_canonical_correlation(cca_results):
    """
    Plot canonical correlation results.
    Corresponds to: 'Python Implementation' in the Canonical Correlation Analysis section.
    
    Parameters:
    -----------
    cca_results : dict
        Results from perform_canonical_correlation()
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(cca_results['X_canonical'][:, 0], cca_results['Y_canonical'][:, 0], alpha=0.6)
    plt.xlabel('First Canonical Variate (X)')
    plt.ylabel('First Canonical Variate (Y)')
    plt.title(f'First Canonical Pair (r = {cca_results["canonical_correlations"][0]:.3f})')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(cca_results['X_canonical'][:, 1], cca_results['Y_canonical'][:, 1], alpha=0.6)
    plt.xlabel('Second Canonical Variate (X)')
    plt.ylabel('Second Canonical Variate (Y)')
    plt.title(f'Second Canonical Pair (r = {cca_results["canonical_correlations"][1]:.3f})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_cca_data(n_samples=100, seed=42):
    """
    Create sample data for Canonical Correlation Analysis.
    Corresponds to: 'Python Implementation' in the Canonical Correlation Analysis section.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X_set, Y_set) two sets of correlated variables
    """
    np.random.seed(seed)
    
    # First set of variables
    X_set = multivariate_normal.rvs(mean=[0, 0, 0], 
                                   cov=[[1, 0.7, 0.5], [0.7, 1, 0.6], [0.5, 0.6, 1]], 
                                   size=n_samples)
    
    # Second set of variables (correlated with first set)
    Y_set = multivariate_normal.rvs(mean=[0, 0], 
                                   cov=[[1, 0.8], [0.8, 1]], 
                                   size=n_samples)
    
    return X_set, Y_set

if __name__ == "__main__":
    """
    Main demonstration block showing how to use all multivariate analysis functions
    in a coherent workflow.
    """
    print("=== MULTIVARIATE ANALYSIS DEMONSTRATION ===\n")
    
    # 1. Create sample data
    print("1. Creating multivariate data...")
    df = create_multivariate_data()
    print(f"Data shape: {df.shape}")
    print(f"Data head:\n{df.head()}\n")
    
    # 2. Principal Component Analysis
    print("2. Performing Principal Component Analysis...")
    pca_results = perform_pca_analysis(df)
    print(f"Explained variance ratio: {pca_results['explained_variance_ratio']}")
    print(f"Cumulative variance: {pca_results['cumulative_variance']}")
    print(f"Component loadings:\n{pca_results['loadings']}\n")
    
    # Plot PCA results
    plot_pca_results(pca_results)
    
    # 3. Factor Analysis
    print("3. Performing Factor Analysis...")
    fa_results = perform_factor_analysis(df)
    print(f"KMO Test: {fa_results['kmo']:.3f}")
    print(f"Bartlett's test p-value: {fa_results['bartlett_p_value']:.6f}")
    print(f"Factor loadings:\n{fa_results['loadings']}\n")
    
    # 4. Cluster Analysis
    print("4. Performing Cluster Analysis...")
    clustering_results = perform_clustering_analysis(df)
    print(f"Optimal number of clusters: {clustering_results['optimal_k']}")
    print(f"Silhouette Score: {clustering_results['validation_scores']['silhouette']:.3f}")
    print(f"Calinski-Harabasz Score: {clustering_results['validation_scores']['calinski_harabasz']:.3f}")
    print(f"Davies-Bouldin Score: {clustering_results['validation_scores']['davies_bouldin']:.3f}\n")
    
    # Plot clustering results
    plot_clustering_results(clustering_results, pca_results['scores'].values)
    
    # 5. Discriminant Analysis
    print("5. Performing Discriminant Analysis...")
    X, y = create_discriminant_data()
    discriminant_results = perform_discriminant_analysis(X, y)
    
    print("Linear Discriminant Analysis:")
    print(classification_report(discriminant_results['y_test'], discriminant_results['y_pred_lda']))
    print("Quadratic Discriminant Analysis:")
    print(classification_report(discriminant_results['y_test'], discriminant_results['y_pred_qda']))
    
    print(f"LDA Cross-validation accuracy: {discriminant_results['lda_cv_scores'].mean():.3f} "
          f"(+/- {discriminant_results['lda_cv_scores'].std() * 2:.3f})")
    print(f"QDA Cross-validation accuracy: {discriminant_results['qda_cv_scores'].mean():.3f} "
          f"(+/- {discriminant_results['qda_cv_scores'].std() * 2:.3f})\n")
    
    # Plot decision boundaries
    plot_decision_boundaries(X, y, discriminant_results)
    
    # 6. Canonical Correlation Analysis
    print("6. Performing Canonical Correlation Analysis...")
    X_set, Y_set = create_cca_data()
    cca_results = perform_canonical_correlation(X_set, Y_set)
    
    print("Canonical Correlations:")
    for i, corr in enumerate(cca_results['canonical_correlations']):
        print(f"Canonical pair {i+1}: {corr:.3f}")
    
    print(f"\nX-set canonical loadings:\n{cca_results['X_loadings']}")
    print(f"\nY-set canonical loadings:\n{cca_results['Y_loadings']}\n")
    
    # Plot canonical correlation results
    plot_canonical_correlation(cca_results)
    
    print("=== DEMONSTRATION COMPLETE ===")
    print("All multivariate analysis techniques have been demonstrated.")
    print("Refer to the markdown file for theoretical explanations and interpretation guidelines.") 