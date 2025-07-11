# Multivariate Analysis

## Introduction

Multivariate analysis is a collection of statistical techniques for examining relationships among multiple variables simultaneously. It is essential for understanding complex data structures, reducing dimensionality, identifying patterns, classifying observations, and modeling latent constructs.

**Key Goals:**
- Dimensionality reduction (e.g., PCA, factor analysis)
- Pattern recognition and clustering
- Classification and discrimination
- Understanding latent variables and structure
- Exploring relationships between variable sets

**When to Use Multivariate Analysis:**
- Data with more than two variables
- Interest in joint relationships, not just pairwise
- Need to reduce complexity or visualize high-dimensional data
- Classification, segmentation, or prediction tasks

## Mathematical Foundations

### Multivariate Data Matrix

A dataset with $`n`$ observations and $`p`$ variables is represented as a matrix $`\mathbf{X}`$:

```math
\mathbf{X} = \begin{bmatrix}
  x_{11} & x_{12} & \cdots & x_{1p} \\
  x_{21} & x_{22} & \cdots & x_{2p} \\
  \vdots & \vdots & \ddots & \vdots \\
  x_{n1} & x_{n2} & \cdots & x_{np}
\end{bmatrix}
```

### Covariance and Correlation Matrices

- **Covariance matrix:** $`\mathbf{S} = \frac{1}{n-1}(\mathbf{X} - \bar{\mathbf{X}})^T(\mathbf{X} - \bar{\mathbf{X}})`$
- **Correlation matrix:** Standardized version of $`\mathbf{S}`$

These matrices summarize the relationships among variables and are the basis for most multivariate techniques.

---

## Principal Component Analysis (PCA)

### Conceptual Overview

PCA is a technique for reducing the dimensionality of a dataset while retaining as much variance as possible. It transforms the original variables into a new set of uncorrelated variables (principal components) ordered by the amount of variance they explain.

### Mathematical Foundation

- **Principal components** are linear combinations:

```math
Z_1 = a_{11}X_1 + a_{12}X_2 + \cdots + a_{1p}X_p
```

- The first principal component maximizes variance:

```math
\max_{\mathbf{a}_1} Var(Z_1) \text{ subject to } \|\mathbf{a}_1\| = 1
```

- **Eigenvalues and eigenvectors:**
  - The eigenvectors of the covariance/correlation matrix give the directions (loadings) of the principal components.
  - The eigenvalues give the variance explained by each component.

- **Total variance explained:**

```math
\text{Proportion explained by } k \text{ components} = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^p \lambda_i}
```

where $`\lambda_i`$ are the eigenvalues.

### Python Implementation and Interpretation

```python
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(123)

# Generate multivariate data
n_samples = 100
n_variables = 5

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

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Perform PCA
pca = PCA()
pca_result = pca.fit(data_scaled)

# Print PCA results
print("PCA Results:")
print(f"Number of components: {len(pca.explained_variance_)}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")

# Create scree plot
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)

# Cumulative variance plot
plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance Plot')
plt.grid(True)
plt.tight_layout()
plt.show()

# Component loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=df.columns
)
print("\nComponent Loadings:")
print(loadings)

# Transform data to principal components
pca_scores = pca.transform(data_scaled)
pca_df = pd.DataFrame(pca_scores, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

# Biplot (first two components)
plt.figure(figsize=(12, 8))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], alpha=0.6)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Biplot')

# Add variable loadings as arrows
for i, var in enumerate(df.columns):
    plt.arrow(0, 0, pca.components_[0, i] * 3, pca.components_[1, i] * 3, 
              color='red', alpha=0.7, head_width=0.1)
    plt.text(pca.components_[0, i] * 3.2, pca.components_[1, i] * 3.2, var, 
             color='red', fontsize=12)

plt.grid(True)
plt.show()
```

- **Scree plot:** Visualizes eigenvalues to help select the number of components.
- **Biplot:** Shows observations and variable loadings in the space of the first two components.
- **Component loadings:** Indicate how much each variable contributes to each component.

**Assumptions:**
- Linearity among variables
- Large sample size
- Variables measured on comparable scales (standardize if not)

**Best Practices:**
- Standardize variables before PCA if scales differ
- Use scree plot, cumulative variance, and interpretability to select components
- Interpret loadings to understand component meaning

---

## Factor Analysis

### Conceptual Overview

Factor analysis models observed variables as linear combinations of a smaller number of unobserved latent factors plus error.

### Mathematical Foundation

- **Factor model:**

```math
X_j = \lambda_{j1}F_1 + \lambda_{j2}F_2 + \cdots + \lambda_{jm}F_m + \varepsilon_j
```

where $`\lambda_{jk}`$ are factor loadings, $`F_k`$ are factors, and $`\varepsilon_j`$ is unique variance.

- **Communality:** Proportion of variance in $`X_j`$ explained by the factors.

- **Rotation:** Orthogonal (varimax) or oblique (promax) rotation improves interpretability.

**Assumptions:**
- Sufficient correlations among variables
- Multivariate normality (for maximum likelihood)
- No perfect multicollinearity

**Diagnostics:**
- **KMO test:** Measures sampling adequacy
- **Bartlett's test:** Tests if correlation matrix is an identity matrix
- **Parallel analysis:** Helps select number of factors

**Best Practices:**
- Use KMO and Bartlett's test before factor analysis
- Choose number of factors based on parallel analysis and interpretability
- Report factor loadings, communalities, and model fit

### Python Implementation

```python
# Import factor analysis libraries
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats import chi2

# KMO test
kmo_all, kmo_model = calculate_kmo(data_scaled)
print(f"KMO Test: {kmo_model:.3f}")

# Bartlett's test of sphericity
def bartlett_sphericity(data):
    n = data.shape[0]
    p = data.shape[1]
    chi_square = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(np.corrcoef(data.T)))
    df = p * (p - 1) / 2
    p_value = chi2.sf(chi_square, df)
    return chi_square, p_value

chi_square, p_value = bartlett_sphericity(data_scaled)
print(f"Bartlett's test: chi-square = {chi_square:.3f}, p-value = {p_value:.6f}")

# Factor analysis
fa = FactorAnalyzer(rotation=None, n_factors=2)
fa.fit(data_scaled)

# Get factor loadings
loadings = fa.loadings_
print("\nFactor Loadings:")
print(pd.DataFrame(loadings, index=df.columns, columns=['Factor1', 'Factor2']))

# Get communalities
communalities = fa.get_communalities()
print(f"\nCommunalities: {communalities}")

# Get variance explained
variance = fa.get_factor_variance()
print(f"Variance explained: {variance[0]}")
print(f"Proportional variance: {variance[1]}")
print(f"Cumulative variance: {variance[2]}")
```

---

## Cluster Analysis

### Conceptual Overview

Cluster analysis groups observations into clusters so that those within a cluster are more similar to each other than to those in other clusters.

### Mathematical Foundation

- **Distance measures:**
  - Euclidean: $`d_{ij} = \sqrt{\sum_{k=1}^p (x_{ik} - x_{jk})^2}`$
  - Manhattan, Mahalanobis, etc.

- **Hierarchical clustering:** Builds a tree (dendrogram) by successively merging or splitting clusters.
- **K-means clustering:** Partitions data into $`K`$ clusters by minimizing within-cluster sum of squares:

```math
\text{WCSS} = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
```

where $`\boldsymbol{\mu}_k`$ is the centroid of cluster $`k`$.

**Cluster validation:**
- Silhouette width, Calinski-Harabasz, Davies-Bouldin indices

**Best Practices:**
- Standardize variables before clustering
- Use multiple methods to determine optimal number of clusters
- Validate clusters with internal and external indices

### Python Implementation

```python
# Import clustering libraries
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# K-means clustering
# Find optimal number of clusters using elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))

# Plot elbow method
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)

# Plot silhouette scores
plt.subplot(1, 3, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.grid(True)

# Perform K-means with optimal k
optimal_k = 3  # Based on elbow and silhouette analysis
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(data_scaled)

# Plot clusters
plt.subplot(1, 3, 3)
scatter = plt.scatter(pca_scores[:, 0], pca_scores[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title(f'K-means Clustering (k={optimal_k})')
plt.colorbar(scatter)
plt.grid(True)
plt.tight_layout()
plt.show()

# Hierarchical clustering
linkage_matrix = linkage(data_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=range(len(data_scaled)), leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Cluster validation
print(f"Silhouette Score: {silhouette_score(data_scaled, cluster_labels):.3f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz_score(data_scaled, cluster_labels):.3f}")
print(f"Davies-Bouldin Score: {davies_bouldin_score(data_scaled, cluster_labels):.3f}")
```

---

## Discriminant Analysis

### Conceptual Overview

Discriminant analysis classifies observations into predefined groups based on predictor variables.

### Mathematical Foundation

- **Linear Discriminant Analysis (LDA):** Finds linear combinations of variables that best separate groups.

```math
Y = \mathbf{a}^T \mathbf{X}
```

- **LDA rule:** Assign $`\mathbf{x}`$ to group $`g`$ that maximizes:

```math
\delta_g(\mathbf{x}) = \mathbf{x}^T \Sigma^{-1} \boldsymbol{\mu}_g - \frac{1}{2} \boldsymbol{\mu}_g^T \Sigma^{-1} \boldsymbol{\mu}_g + \log \pi_g
```

where $`\boldsymbol{\mu}_g`$ is the mean vector for group $`g`$, $`\Sigma`$ is the pooled covariance, $`\pi_g`$ is the prior probability.

- **Quadratic Discriminant Analysis (QDA):** Allows group-specific covariance matrices.

**Assumptions:**
- Multivariate normality within groups
- Equal covariance matrices for LDA
- Independence of observations

**Best Practices:**
- Check group means and covariance matrices
- Use cross-validation to assess classification accuracy
- Compare LDA and QDA for linear vs. nonlinear boundaries

### Python Implementation

```python
# Import discriminant analysis libraries
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Create sample data with known groups
np.random.seed(42)
n_samples_per_group = 50

# Generate three groups with different means
group1 = multivariate_normal.rvs(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=n_samples_per_group)
group2 = multivariate_normal.rvs(mean=[3, 3], cov=[[1, 0.5], [0.5, 1]], size=n_samples_per_group)
group3 = multivariate_normal.rvs(mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=n_samples_per_group)

# Combine data
X = np.vstack([group1, group2, group3])
y = np.hstack([np.zeros(n_samples_per_group), np.ones(n_samples_per_group), np.ones(n_samples_per_group) * 2])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)

# Compare results
print("Linear Discriminant Analysis:")
print(classification_report(y_test, y_pred_lda))
print("\nQuadratic Discriminant Analysis:")
print(classification_report(y_test, y_pred_qda))

# Cross-validation comparison
lda_scores = cross_val_score(lda, X, y, cv=5)
qda_scores = cross_val_score(qda, X, y, cv=5)

print(f"\nLDA Cross-validation accuracy: {lda_scores.mean():.3f} (+/- {lda_scores.std() * 2:.3f})")
print(f"QDA Cross-validation accuracy: {qda_scores.mean():.3f} (+/- {qda_scores.std() * 2:.3f})")

# Visualize decision boundaries
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
plot_decision_boundary(X, y, lda, 'Linear Discriminant Analysis')
plt.subplot(1, 2, 2)
plot_decision_boundary(X, y, qda, 'Quadratic Discriminant Analysis')
plt.tight_layout()
plt.show()
```

---

## Canonical Correlation Analysis (CCA)

### Conceptual Overview

CCA explores relationships between two sets of variables by finding linear combinations (canonical variates) that are maximally correlated.

### Mathematical Foundation

- **Canonical variates:**

```math
U = \mathbf{a}^T \mathbf{X}, \quad V = \mathbf{b}^T \mathbf{Y}
```

- **Canonical correlation:**

```math
\rho = \max_{\mathbf{a}, \mathbf{b}} Corr(U, V)
```

- **Interpretation:**
  - The first pair of canonical variates has the highest possible correlation between the two sets.
  - Subsequent pairs are uncorrelated with previous ones.

**Assumptions:**
- Linearity and multivariate normality
- Sufficient sample size

**Best Practices:**
- Standardize variables before CCA
- Interpret canonical loadings and cross-loadings
- Report canonical correlations and significance

### Python Implementation

```python
# Import CCA library
from sklearn.cross_decomposition import CCA

# Create two sets of variables
np.random.seed(42)
n_samples = 100

# First set of variables
X_set = multivariate_normal.rvs(mean=[0, 0, 0], cov=[[1, 0.7, 0.5], [0.7, 1, 0.6], [0.5, 0.6, 1]], size=n_samples)

# Second set of variables (correlated with first set)
Y_set = multivariate_normal.rvs(mean=[0, 0], cov=[[1, 0.8], [0.8, 1]], size=n_samples)

# Perform CCA
cca = CCA(n_components=2)
X_c, Y_c = cca.fit_transform(X_set, Y_set)

# Canonical correlations
canonical_correlations = np.corrcoef(X_c.T, Y_c.T)[:2, 2:]
print("Canonical Correlations:")
for i, corr in enumerate(np.diag(canonical_correlations)):
    print(f"Canonical pair {i+1}: {corr:.3f}")

# Canonical loadings
X_loadings = cca.x_weights_
Y_loadings = cca.y_weights_

print("\nX-set canonical loadings:")
print(pd.DataFrame(X_loadings, columns=['CC1', 'CC2'], index=[f'X{i+1}' for i in range(X_set.shape[1])]))

print("\nY-set canonical loadings:")
print(pd.DataFrame(Y_loadings, columns=['CC1', 'CC2'], index=[f'Y{i+1}' for i in range(Y_set.shape[1])]))

# Plot canonical variates
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_c[:, 0], Y_c[:, 0], alpha=0.6)
plt.xlabel('First Canonical Variate (X)')
plt.ylabel('First Canonical Variate (Y)')
plt.title(f'First Canonical Pair (r = {canonical_correlations[0, 0]:.3f})')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_c[:, 1], Y_c[:, 1], alpha=0.6)
plt.xlabel('Second Canonical Variate (X)')
plt.ylabel('Second Canonical Variate (Y)')
plt.title(f'Second Canonical Pair (r = {canonical_correlations[1, 1]:.3f})')
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Practical Examples

- **Customer segmentation:** Use PCA and clustering to identify customer groups.
- **Survey analysis:** Use factor analysis to uncover latent constructs.
- **Medical diagnosis:** Use LDA/QDA for disease classification.
- **Marketing:** Use CCA to relate customer attitudes and behaviors.

---

## Best Practices

- Always check assumptions (normality, multicollinearity, sample size)
- Standardize variables when appropriate
- Use scree plots, parallel analysis, and validation indices for model selection
- Report loadings, scores, and fit statistics
- Visualize results for interpretation
- Use cross-validation for classification accuracy
- Profile and interpret clusters and components

---

## Exercises

### Exercise 1: PCA Analysis
- **Objective:** Perform PCA on a dataset and interpret the components and loadings.
- **Hint:** Use `sklearn.decomposition.PCA` and scree plot for component selection.

### Exercise 2: Factor Analysis
- **Objective:** Conduct factor analysis to identify latent variables in survey data.
- **Hint:** Use `factor_analyzer.FactorAnalyzer` and interpret factor loadings.

### Exercise 3: Cluster Analysis
- **Objective:** Apply different clustering methods and validate the results.
- **Hint:** Use `sklearn.cluster.KMeans`, `sklearn.cluster.AgglomerativeClustering`, and silhouette analysis.

### Exercise 4: Discriminant Analysis
- **Objective:** Use LDA and QDA for classification and compare their performance.
- **Hint:** Use `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` and `QuadraticDiscriminantAnalysis` and compare confusion matrices.

### Exercise 5: Canonical Correlation
- **Objective:** Analyze relationships between two sets of variables using canonical correlation.
- **Hint:** Use `sklearn.cross_decomposition.CCA` and interpret canonical correlations.

---

**Key Takeaways:**
- Multivariate analysis provides tools for understanding complex relationships
- PCA is useful for dimensionality reduction and data visualization
- Factor analysis identifies latent variables in observed data
- Cluster analysis groups similar observations together
- Discriminant analysis is effective for classification problems
- Canonical correlation explores relationships between variable sets
- Always validate results and check assumptions
- Choose appropriate techniques based on analysis goals
- Proper reporting includes data summary, results interpretation, and recommendations 