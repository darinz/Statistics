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

# Practical Implementation

All Python code for this chapter has been moved to the companion file: `24_multivariate_analysis.py`.

- For each theoretical section, refer to the corresponding function in the Python file.
- The Python file contains modular, well-documented functions and a main demonstration block.
- See the end of this document for a summary mapping theory sections to code functions.

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

**Reference:** Use `create_multivariate_data()`, `perform_pca_analysis()`, and `plot_pca_results()` functions in the Python file.

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

**Reference:** Use `perform_factor_analysis()` function in the Python file.

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

**Reference:** Use `perform_clustering_analysis()` and `plot_clustering_results()` functions in the Python file.

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

**Reference:** Use `create_discriminant_data()`, `perform_discriminant_analysis()`, and `plot_decision_boundaries()` functions in the Python file.

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

**Reference:** Use `create_cca_data()`, `perform_canonical_correlation()`, and `plot_canonical_correlation()` functions in the Python file.

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

## Code Reference Guide

| Theory Section                        | Python Function/Section                |
|---------------------------------------|----------------------------------------|
| Data Creation                          | `create_multivariate_data()`           |
| Principal Component Analysis          | `perform_pca_analysis()`, `plot_pca_results()` |
| Factor Analysis                       | `perform_factor_analysis()`            |
| Cluster Analysis                      | `perform_clustering_analysis()`, `plot_clustering_results()` |
| Discriminant Analysis                 | `create_discriminant_data()`, `perform_discriminant_analysis()`, `plot_decision_boundaries()` |
| Canonical Correlation Analysis        | `create_cca_data()`, `perform_canonical_correlation()`, `plot_canonical_correlation()` |
| Complete Workflow                     | See main block in `.py` file           |

For exercises, use the functions above as building blocks for your analysis.

---

## Exercises

### Exercise 1: PCA Analysis
- **Objective:** Perform PCA on a dataset and interpret the components and loadings.
- **Hint:** Use `create_multivariate_data()` and `perform_pca_analysis()` functions.

### Exercise 2: Factor Analysis
- **Objective:** Conduct factor analysis to identify latent variables in survey data.
- **Hint:** Use `perform_factor_analysis()` and interpret factor loadings.

### Exercise 3: Cluster Analysis
- **Objective:** Apply different clustering methods and validate the results.
- **Hint:** Use `perform_clustering_analysis()` and silhouette analysis.

### Exercise 4: Discriminant Analysis
- **Objective:** Use LDA and QDA for classification and compare their performance.
- **Hint:** Use `perform_discriminant_analysis()` and compare confusion matrices.

### Exercise 5: Canonical Correlation
- **Objective:** Analyze relationships between two sets of variables using canonical correlation.
- **Hint:** Use `perform_canonical_correlation()` and interpret canonical correlations.

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