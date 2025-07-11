# Multivariate Analysis

## Overview

Multivariate analysis involves examining relationships between multiple variables simultaneously. This field includes techniques for dimensionality reduction, pattern recognition, classification, and understanding complex relationships in high-dimensional data.

## Principal Component Analysis (PCA)

### Basic PCA Implementation

```r
# Load required packages
library(stats)
library(ggplot2)
library(ggfortify)

# Generate multivariate data
set.seed(123)
n_samples <- 100
n_variables <- 5

# Create correlated variables
correlation_matrix <- matrix(c(
  1.0, 0.8, 0.6, 0.4, 0.2,
  0.8, 1.0, 0.7, 0.5, 0.3,
  0.6, 0.7, 1.0, 0.6, 0.4,
  0.4, 0.5, 0.6, 1.0, 0.5,
  0.2, 0.3, 0.4, 0.5, 1.0
), nrow = 5, ncol = 5)

# Generate multivariate normal data
library(MASS)
multivariate_data <- mvrnorm(n_samples, mu = rep(0, n_variables), Sigma = correlation_matrix)
colnames(multivariate_data) <- c("Var1", "Var2", "Var3", "Var4", "Var5")

# Perform PCA
pca_result <- prcomp(multivariate_data, center = TRUE, scale = TRUE)
print(pca_result)

# Summary of PCA results
summary_pca <- summary(pca_result)
print(summary_pca)

# Extract key information
cat("PCA Results Summary:\n")
cat("Number of components:", length(pca_result$sdev), "\n")
cat("Proportion of variance explained:\n")
print(round(summary_pca$importance[2, ], 3))
cat("Cumulative proportion of variance:\n")
print(round(summary_pca$importance[3, ], 3))
```

### PCA Visualization

```r
# Scree plot
scree_data <- data.frame(
  Component = 1:length(pca_result$sdev),
  Variance = pca_result$sdev^2,
  Proportion = summary_pca$importance[2, ],
  Cumulative = summary_pca$importance[3, ]
)

# Scree plot
ggplot(scree_data, aes(x = Component, y = Variance)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point(color = "red", size = 3) +
  labs(title = "Scree Plot",
       x = "Principal Component",
       y = "Eigenvalue (Variance)") +
  theme_minimal()

# Cumulative variance plot
ggplot(scree_data, aes(x = Component, y = Cumulative)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point(color = "red", size = 3) +
  geom_hline(yintercept = 0.8, linetype = "dashed", color = "orange") +
  labs(title = "Cumulative Proportion of Variance Explained",
       x = "Principal Component",
       y = "Cumulative Proportion") +
  theme_minimal()

# Biplot
autoplot(pca_result, data = multivariate_data, 
         colour = "steelblue", loadings = TRUE, 
         loadings.label = TRUE, loadings.label.size = 3) +
  labs(title = "PCA Biplot") +
  theme_minimal()
```

### Component Selection

```r
# Function to determine optimal number of components
select_components <- function(pca_result, method = "cumulative", threshold = 0.8) {
  summary_pca <- summary(pca_result)
  eigenvalues <- pca_result$sdev^2
  
  if (method == "cumulative") {
    # Cumulative variance method
    cumulative_var <- summary_pca$importance[3, ]
    n_components <- which(cumulative_var >= threshold)[1]
    cat("Cumulative variance method (threshold =", threshold, "):\n")
    cat("Number of components:", n_components, "\n")
    cat("Cumulative variance explained:", round(cumulative_var[n_components], 3), "\n")
  } else if (method == "eigenvalue") {
    # Kaiser criterion (eigenvalue > 1)
    n_components <- sum(eigenvalues > 1)
    cat("Kaiser criterion (eigenvalue > 1):\n")
    cat("Number of components:", n_components, "\n")
    cat("Eigenvalues > 1:", sum(eigenvalues > 1), "\n")
  } else if (method == "scree") {
    # Scree plot method (elbow)
    diff_eigenvalues <- diff(eigenvalues)
    n_components <- which.max(diff_eigenvalues) + 1
    cat("Scree plot method (elbow):\n")
    cat("Number of components:", n_components, "\n")
  }
  
  return(n_components)
}

# Apply different selection methods
n_comp_cumulative <- select_components(pca_result, "cumulative", 0.8)
n_comp_eigenvalue <- select_components(pca_result, "eigenvalue")
n_comp_scree <- select_components(pca_result, "scree")
```

### Component Interpretation

```r
# Extract loadings
loadings_matrix <- pca_result$rotation
print("Component Loadings:")
print(round(loadings_matrix, 3))

# Function to interpret components
interpret_components <- function(pca_result, threshold = 0.5) {
  loadings <- pca_result$rotation
  n_components <- ncol(loadings)
  
  cat("Component Interpretation (loadings >", threshold, "):\n")
  
  for (i in 1:n_components) {
    component_loadings <- loadings[, i]
    significant_vars <- which(abs(component_loadings) > threshold)
    
    cat("\nComponent", i, ":\n")
    if (length(significant_vars) > 0) {
      for (j in significant_vars) {
        loading_value <- component_loadings[j]
        var_name <- names(component_loadings)[j]
        cat("  ", var_name, ":", round(loading_value, 3), "\n")
      }
    } else {
      cat("  No variables with loadings >", threshold, "\n")
    }
  }
}

# Interpret components
interpret_components(pca_result, threshold = 0.5)

# Calculate component scores
component_scores <- pca_result$x
print("First few component scores:")
print(head(round(component_scores, 3)))
```

## Factor Analysis

### Exploratory Factor Analysis (EFA)

```r
# Load required packages
library(psych)
library(GPArotation)

# Perform factor analysis
fa_result <- fa(multivariate_data, nfactors = 2, rotate = "varimax")
print(fa_result)

# Extract factor loadings
factor_loadings <- fa_result$loadings
print("Factor Loadings:")
print(round(factor_loadings, 3))

# Factor scores
factor_scores <- fa_result$scores
print("First few factor scores:")
print(head(round(factor_scores, 3)))

# Model fit statistics
cat("Factor Analysis Model Fit:\n")
cat("RMSEA:", round(fa_result$RMSEA, 3), "\n")
cat("TLI:", round(fa_result$TLI, 3), "\n")
cat("CFI:", round(fa_result$CFI, 3), "\n")
```

### Factor Analysis Diagnostics

```r
# Kaiser-Meyer-Olkin (KMO) test
kmo_result <- KMO(multivariate_data)
cat("KMO Overall:", round(kmo_result$MSA, 3), "\n")
print("KMO for individual variables:")
print(round(kmo_result$MSAi, 3))

# Bartlett's test of sphericity
bartlett_result <- cortest.bartlett(multivariate_data)
cat("Bartlett's test p-value:", round(bartlett_result$p.value, 4), "\n")

# Parallel analysis for factor selection
parallel_result <- fa.parallel(multivariate_data, fm = "ml", fa = "fa")
cat("Parallel analysis suggests", parallel_result$nfact, "factors\n")

# Factor analysis with different rotation methods
fa_varimax <- fa(multivariate_data, nfactors = 2, rotate = "varimax")
fa_promax <- fa(multivariate_data, nfactors = 2, rotate = "promax")

cat("Rotation Comparison:\n")
cat("Varimax - RMSEA:", round(fa_varimax$RMSEA, 3), "\n")
cat("Promax - RMSEA:", round(fa_promax$RMSEA, 3), "\n")
```

## Cluster Analysis

### Hierarchical Clustering

```r
# Calculate distance matrix
distance_matrix <- dist(multivariate_data, method = "euclidean")

# Perform hierarchical clustering
hclust_result <- hclust(distance_matrix, method = "ward.D2")
print(hclust_result)

# Dendrogram
plot(hclust_result, main = "Hierarchical Clustering Dendrogram",
     xlab = "Observations", ylab = "Distance")

# Determine optimal number of clusters
library(factoextra)

# Elbow method
elbow_plot <- fviz_nbclust(multivariate_data, FUN = hcut, method = "wss")
print(elbow_plot)

# Silhouette method
silhouette_plot <- fviz_nbclust(multivariate_data, FUN = hcut, method = "silhouette")
print(silhouette_plot)

# Gap statistic
gap_plot <- fviz_nbclust(multivariate_data, FUN = hcut, method = "gap_stat")
print(gap_plot)

# Cut dendrogram to get clusters
optimal_clusters <- 3
cluster_assignments <- cutree(hclust_result, k = optimal_clusters)
print("Cluster assignments:")
print(table(cluster_assignments))
```

### K-Means Clustering

```r
# Perform k-means clustering
set.seed(123)
kmeans_result <- kmeans(multivariate_data, centers = 3, nstart = 25)
print(kmeans_result)

# Cluster centers
cluster_centers <- kmeans_result$centers
print("Cluster Centers:")
print(round(cluster_centers, 3))

# Cluster assignments
cluster_assignments_km <- kmeans_result$cluster
print("Cluster assignments:")
print(table(cluster_assignments_km))

# Within-cluster sum of squares
within_ss <- kmeans_result$withinss
total_ss <- kmeans_result$totss
between_ss <- kmeans_result$betweenss

cat("Cluster Analysis Results:\n")
cat("Total sum of squares:", round(total_ss, 2), "\n")
cat("Between-cluster sum of squares:", round(between_ss, 2), "\n")
cat("Within-cluster sum of squares:", round(sum(within_ss), 2), "\n")
cat("Explained variance:", round(between_ss / total_ss, 3), "\n")

# Visualize clusters
cluster_data <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  Cluster = factor(cluster_assignments_km)
)

ggplot(cluster_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "K-Means Clustering Results",
       x = "First Principal Component",
       y = "Second Principal Component") +
  theme_minimal()
```

### Cluster Validation

```r
# Silhouette analysis
library(cluster)
silhouette_result <- silhouette(cluster_assignments_km, distance_matrix)
silhouette_avg <- mean(silhouette_result[, 3])
cat("Average silhouette width:", round(silhouette_avg, 3), "\n")

# Calinski-Harabasz index
library(fpc)
ch_index <- calinhara(multivariate_data, cluster_assignments_km)
cat("Calinski-Harabasz index:", round(ch_index, 2), "\n")

# Davies-Bouldin index
db_index <- index.DB(multivariate_data, cluster_assignments_km)$DB
cat("Davies-Bouldin index:", round(db_index, 3), "\n")

# Compare different numbers of clusters
cluster_evaluation <- function(data, max_clusters = 10) {
  results <- data.frame(
    n_clusters = 2:max_clusters,
    within_ss = numeric(max_clusters - 1),
    silhouette = numeric(max_clusters - 1),
    ch_index = numeric(max_clusters - 1)
  )
  
  for (k in 2:max_clusters) {
    # K-means
    km_result <- kmeans(data, centers = k, nstart = 25)
    results$within_ss[k-1] <- sum(km_result$withinss)
    
    # Silhouette
    sil_result <- silhouette(km_result$cluster, dist(data))
    results$silhouette[k-1] <- mean(sil_result[, 3])
    
    # Calinski-Harabasz
    results$ch_index[k-1] <- calinhara(data, km_result$cluster)
  }
  
  return(results)
}

# Apply cluster evaluation
eval_results <- cluster_evaluation(multivariate_data, max_clusters = 8)
print("Cluster Evaluation Results:")
print(round(eval_results, 3))
```

## Discriminant Analysis

### Linear Discriminant Analysis (LDA)

```r
# Load required packages
library(MASS)
library(caret)

# Create classification data
set.seed(123)
n_samples_per_class <- 50
n_variables <- 4

# Generate three classes with different means
class1_data <- mvrnorm(n_samples_per_class, mu = c(0, 0, 0, 0), Sigma = diag(4))
class2_data <- mvrnorm(n_samples_per_class, mu = c(2, 2, 0, 0), Sigma = diag(4))
class3_data <- mvrnorm(n_samples_per_class, mu = c(0, 0, 2, 2), Sigma = diag(4))

# Combine data
lda_data <- rbind(class1_data, class2_data, class3_data)
lda_labels <- factor(rep(c("Class1", "Class2", "Class3"), each = n_samples_per_class))

# Perform LDA
lda_result <- lda(lda_data, lda_labels)
print(lda_result)

# Extract discriminant functions
discriminant_functions <- lda_result$scaling
print("Discriminant Functions:")
print(round(discriminant_functions, 3))

# Prior probabilities
prior_probs <- lda_result$prior
print("Prior Probabilities:")
print(prior_probs)

# Group means
group_means <- lda_result$means
print("Group Means:")
print(round(group_means, 3))
```

### LDA Classification

```r
# Predict class assignments
lda_predictions <- predict(lda_result, lda_data)
predicted_classes <- lda_predictions$class
posterior_probs <- lda_predictions$posterior

# Confusion matrix
confusion_matrix <- table(Actual = lda_labels, Predicted = predicted_classes)
print("Confusion Matrix:")
print(confusion_matrix)

# Classification accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Classification Accuracy:", round(accuracy, 3), "\n")

# Per-class accuracy
per_class_accuracy <- diag(confusion_matrix) / rowSums(confusion_matrix)
print("Per-class Accuracy:")
print(round(per_class_accuracy, 3))

# Cross-validation
library(caret)
cv_lda <- train(lda_data, lda_labels, method = "lda", 
                trControl = trainControl(method = "cv", number = 5))
print(cv_lda)

cat("Cross-validation Accuracy:", round(cv_lda$results$Accuracy, 3), "\n")
```

### Quadratic Discriminant Analysis (QDA)

```r
# Perform QDA
qda_result <- qda(lda_data, lda_labels)
print(qda_result)

# QDA predictions
qda_predictions <- predict(qda_result, lda_data)
qda_predicted_classes <- qda_predictions$class

# QDA confusion matrix
qda_confusion_matrix <- table(Actual = lda_labels, Predicted = qda_predicted_classes)
print("QDA Confusion Matrix:")
print(qda_confusion_matrix)

# QDA accuracy
qda_accuracy <- sum(diag(qda_confusion_matrix)) / sum(qda_confusion_matrix)
cat("QDA Classification Accuracy:", round(qda_accuracy, 3), "\n")

# Compare LDA vs QDA
cat("Model Comparison:\n")
cat("LDA Accuracy:", round(accuracy, 3), "\n")
cat("QDA Accuracy:", round(qda_accuracy, 3), "\n")
```

## Canonical Correlation Analysis

### Basic Canonical Correlation

```r
# Create two sets of variables
set.seed(123)
n_samples <- 100

# First set of variables
set1_data <- mvrnorm(n_samples, mu = rep(0, 3), Sigma = diag(3))
colnames(set1_data) <- c("X1", "X2", "X3")

# Second set of variables (correlated with first set)
set2_data <- mvrnorm(n_samples, mu = rep(0, 3), Sigma = diag(3))
# Add correlation between sets
set2_data[, 1] <- 0.7 * set1_data[, 1] + 0.3 * rnorm(n_samples)
set2_data[, 2] <- 0.6 * set1_data[, 2] + 0.4 * rnorm(n_samples)
set2_data[, 3] <- 0.5 * set1_data[, 3] + 0.5 * rnorm(n_samples)
colnames(set2_data) <- c("Y1", "Y2", "Y3")

# Perform canonical correlation analysis
library(CCA)
cca_result <- cc(set1_data, set2_data)
print(cca_result)

# Canonical correlations
canonical_correlations <- cca_result$cor
print("Canonical Correlations:")
print(round(canonical_correlations, 3))

# Canonical coefficients
x_coefficients <- cca_result$xcoef
y_coefficients <- cca_result$ycoef

print("X Canonical Coefficients:")
print(round(x_coefficients, 3))
print("Y Canonical Coefficients:")
print(round(y_coefficients, 3))

# Canonical variates
x_scores <- cca_result$scores$xscores
y_scores <- cca_result$scores$yscores

print("First few canonical variates:")
print(head(round(x_scores, 3)))
print(head(round(y_scores, 3)))
```

## Practical Examples

### Example 1: Customer Segmentation

```r
# Simulate customer data
set.seed(123)
n_customers <- 200

customer_data <- data.frame(
  Age = rnorm(n_customers, 45, 10),
  Income = rnorm(n_customers, 60000, 15000),
  Spending = rnorm(n_customers, 5000, 1000),
  Frequency = rpois(n_customers, 12),
  Satisfaction = rnorm(n_customers, 7, 1)
)

# Standardize data
customer_scaled <- scale(customer_data)

# PCA for dimensionality reduction
customer_pca <- prcomp(customer_scaled)
print(summary(customer_pca))

# Cluster analysis
customer_clusters <- kmeans(customer_scaled, centers = 4, nstart = 25)
customer_data$Cluster <- factor(customer_clusters$cluster)

# Analyze clusters
cluster_summary <- aggregate(customer_data[, -6], by = list(customer_data$Cluster), FUN = mean)
print("Cluster Profiles:")
print(round(cluster_summary, 2))

# Visualize clusters
cluster_pca <- data.frame(
  PC1 = customer_pca$x[, 1],
  PC2 = customer_pca$x[, 2],
  Cluster = customer_data$Cluster
)

ggplot(cluster_pca, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Customer Segmentation",
       x = "First Principal Component",
       y = "Second Principal Component") +
  theme_minimal()
```

### Example 2: Factor Analysis of Survey Data

```r
# Simulate survey responses
set.seed(123)
n_respondents <- 300

# Generate correlated survey items
survey_data <- data.frame(
  Q1 = rnorm(n_respondents, 3.5, 1),  # Satisfaction
  Q2 = rnorm(n_respondents, 3.5, 1),  # Satisfaction
  Q3 = rnorm(n_respondents, 3.5, 1),  # Satisfaction
  Q4 = rnorm(n_respondents, 4.0, 1),  # Quality
  Q5 = rnorm(n_respondents, 4.0, 1),  # Quality
  Q6 = rnorm(n_respondents, 4.0, 1),  # Quality
  Q7 = rnorm(n_respondents, 2.5, 1),  # Price
  Q8 = rnorm(n_respondents, 2.5, 1),  # Price
  Q9 = rnorm(n_respondents, 2.5, 1)   # Price
)

# Add correlations within factors
survey_data$Q2 <- survey_data$Q2 + 0.6 * survey_data$Q1
survey_data$Q3 <- survey_data$Q3 + 0.5 * survey_data$Q1
survey_data$Q5 <- survey_data$Q5 + 0.6 * survey_data$Q4
survey_data$Q6 <- survey_data$Q6 + 0.5 * survey_data$Q4
survey_data$Q8 <- survey_data$Q8 + 0.6 * survey_data$Q7
survey_data$Q9 <- survey_data$Q9 + 0.5 * survey_data$Q7

# Factor analysis
survey_fa <- fa(survey_data, nfactors = 3, rotate = "varimax")
print(survey_fa)

# Factor loadings
print("Factor Loadings:")
print(round(survey_fa$loadings, 3))

# Factor scores
survey_scores <- survey_fa$scores
colnames(survey_scores) <- c("Satisfaction", "Quality", "Price")
print("First few factor scores:")
print(head(round(survey_scores, 3)))
```

### Example 3: Classification Analysis

```r
# Simulate medical diagnosis data
set.seed(123)
n_patients <- 150

# Generate features for three disease types
healthy_data <- mvrnorm(50, mu = c(70, 120, 37, 5), Sigma = diag(4))
disease1_data <- mvrnorm(50, mu = c(85, 140, 38.5, 7), Sigma = diag(4))
disease2_data <- mvrnorm(50, mu = c(90, 160, 39, 9), Sigma = diag(4))

# Combine data
diagnosis_data <- rbind(healthy_data, disease1_data, disease2_data)
diagnosis_labels <- factor(rep(c("Healthy", "Disease1", "Disease2"), each = 50))
colnames(diagnosis_data) <- c("Temperature", "BloodPressure", "HeartRate", "Inflammation")

# LDA classification
diagnosis_lda <- lda(diagnosis_data, diagnosis_labels)
print(diagnosis_lda)

# Predictions
diagnosis_predictions <- predict(diagnosis_lda, diagnosis_data)
diagnosis_confusion <- table(Actual = diagnosis_labels, 
                           Predicted = diagnosis_predictions$class)
print("Diagnosis Classification Results:")
print(diagnosis_confusion)

# Classification accuracy
diagnosis_accuracy <- sum(diag(diagnosis_confusion)) / sum(diagnosis_confusion)
cat("Classification Accuracy:", round(diagnosis_accuracy, 3), "\n")

# Cross-validation
cv_diagnosis <- train(diagnosis_data, diagnosis_labels, method = "lda",
                     trControl = trainControl(method = "cv", number = 5))
cat("Cross-validation Accuracy:", round(cv_diagnosis$results$Accuracy, 3), "\n")
```

## Best Practices

### Model Selection Guidelines

```r
# Function to help choose appropriate multivariate technique
choose_multivariate_technique <- function(data, goal = "exploration") {
  cat("=== MULTIVARIATE TECHNIQUE SELECTION ===\n")
  
  n_variables <- ncol(data)
  n_observations <- nrow(data)
  
  cat("Data dimensions:", n_observations, "observations,", n_variables, "variables\n")
  
  if (goal == "exploration") {
    cat("\nEXPLORATORY ANALYSIS:\n")
    if (n_variables > 3) {
      cat("- Use PCA for dimensionality reduction\n")
      cat("- Use factor analysis for latent variable identification\n")
    } else {
      cat("- Use correlation analysis\n")
    }
  } else if (goal == "clustering") {
    cat("\nCLUSTERING ANALYSIS:\n")
    if (n_variables > 5) {
      cat("- Use PCA first for dimensionality reduction\n")
      cat("- Then apply k-means or hierarchical clustering\n")
    } else {
      cat("- Use k-means or hierarchical clustering directly\n")
    }
  } else if (goal == "classification") {
    cat("\nCLASSIFICATION ANALYSIS:\n")
    if (n_variables > 10) {
      cat("- Use LDA for linear classification\n")
      cat("- Consider QDA for non-linear boundaries\n")
    } else {
      cat("- Use LDA or QDA directly\n")
    }
  }
  
  # Data quality checks
  cat("\nDATA QUALITY CHECKS:\n")
  missing_data <- sum(is.na(data))
  cat("Missing values:", missing_data, "\n")
  
  if (missing_data > 0) {
    cat("Consider imputation before analysis\n")
  }
  
  # Multicollinearity check
  cor_matrix <- cor(data, use = "complete.obs")
  high_correlations <- which(abs(cor_matrix) > 0.8 & cor_matrix != 1, arr.ind = TRUE)
  
  if (nrow(high_correlations) > 0) {
    cat("High correlations detected. Consider PCA or factor analysis.\n")
  }
  
  return(list(
    n_variables = n_variables,
    n_observations = n_observations,
    missing_data = missing_data,
    high_correlations = nrow(high_correlations)
  ))
}

# Apply technique selection
technique_selection <- choose_multivariate_technique(multivariate_data, "exploration")
```

### Reporting Guidelines

```r
# Function to generate comprehensive multivariate analysis report
generate_multivariate_report <- function(data, pca_result = NULL, cluster_result = NULL, 
                                       lda_result = NULL, analysis_type = "pca") {
  cat("=== MULTIVARIATE ANALYSIS REPORT ===\n\n")
  
  # Data summary
  cat("DATA SUMMARY:\n")
  cat("Number of observations:", nrow(data), "\n")
  cat("Number of variables:", ncol(data), "\n")
  cat("Missing values:", sum(is.na(data)), "\n")
  cat("Variable names:", paste(colnames(data), collapse = ", "), "\n\n")
  
  if (analysis_type == "pca" && !is.null(pca_result)) {
    cat("PRINCIPAL COMPONENT ANALYSIS:\n")
    summary_pca <- summary(pca_result)
    cat("Number of components:", length(pca_result$sdev), "\n")
    cat("Proportion of variance explained by first 2 components:", 
        round(sum(summary_pca$importance[2, 1:2]), 3), "\n")
    cat("Cumulative proportion explained by first 3 components:", 
        round(summary_pca$importance[3, 3], 3), "\n\n")
  }
  
  if (analysis_type == "clustering" && !is.null(cluster_result)) {
    cat("CLUSTER ANALYSIS:\n")
    cat("Number of clusters:", length(unique(cluster_result$cluster)), "\n")
    cat("Cluster sizes:", table(cluster_result$cluster), "\n")
    cat("Within-cluster sum of squares:", round(sum(cluster_result$withinss), 2), "\n")
    cat("Between-cluster sum of squares:", round(cluster_result$betweenss, 2), "\n")
    cat("Explained variance:", round(cluster_result$betweenss / cluster_result$totss, 3), "\n\n")
  }
  
  if (analysis_type == "classification" && !is.null(lda_result)) {
    cat("DISCRIMINANT ANALYSIS:\n")
    cat("Number of classes:", length(lda_result$lev), "\n")
    cat("Prior probabilities:", round(lda_result$prior, 3), "\n")
    cat("Number of discriminant functions:", length(lda_result$svd), "\n\n")
  }
  
  # Recommendations
  cat("RECOMMENDATIONS:\n")
  if (analysis_type == "pca") {
    cat("- Consider using first 2-3 components for further analysis\n")
    cat("- Interpret loadings to understand component meaning\n")
  } else if (analysis_type == "clustering") {
    cat("- Validate cluster solution with different methods\n")
    cat("- Profile clusters to understand characteristics\n")
  } else if (analysis_type == "classification") {
    cat("- Use cross-validation to assess classification accuracy\n")
    cat("- Consider ensemble methods for improved performance\n")
  }
}

# Generate report
generate_multivariate_report(multivariate_data, pca_result = pca_result, 
                           analysis_type = "pca")
```

## Exercises

### Exercise 1: PCA Analysis
Perform PCA on a dataset and interpret the components and loadings.

### Exercise 2: Factor Analysis
Conduct factor analysis to identify latent variables in survey data.

### Exercise 3: Cluster Analysis
Apply different clustering methods and validate the results.

### Exercise 4: Discriminant Analysis
Use LDA and QDA for classification and compare their performance.

### Exercise 5: Canonical Correlation
Analyze relationships between two sets of variables using canonical correlation.

## Next Steps

In the next chapter, we'll learn about survival analysis for time-to-event data.

---

**Key Takeaways:**
- Multivariate analysis provides tools for understanding complex relationships
- PCA is useful for dimensionality reduction and data visualization
- Factor analysis identifies latent variables in observed data
- Cluster analysis groups similar observations together
- Discriminant analysis is effective for classification problems
- Always validate results and check assumptions
- Consider data quality and preprocessing requirements
- Choose appropriate techniques based on analysis goals
- Proper reporting includes data summary, results interpretation, and recommendations 