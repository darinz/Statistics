# Statistics with R

[![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)](https://www.r-project.org/)
[![RStudio](https://img.shields.io/badge/RStudio-75AADB?style=for-the-badge&logo=rstudio&logoColor=white)](https://posit.co/)
[![Markdown](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white)](https://markdown.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Enhanced-brightgreen)](https://github.com/yourusername/statistics-with-r)

A comprehensive, in-depth guide to learning statistics using R programming language, featuring mathematical foundations, detailed code explanations, practical applications, and hands-on exercises. This enhanced curriculum provides clear conceptual understanding, step-by-step implementations, and real-world examples for each statistical method.

## Table of Contents

### Fundamentals
- [Introduction to R](01_introduction_to_r.md) - R basics, environment setup, and programming fundamentals
- [Data Types and Structures](02_data_types_structures.md) - Vectors, matrices, data frames, and lists with practical examples
- [Data Import and Manipulation](03_data_import_manipulation.md) - Data cleaning, transformation, and preprocessing techniques
- [Exploratory Data Analysis](04_exploratory_data_analysis.md) - Comprehensive EDA with visualization and summary statistics

### Descriptive Statistics
- [Measures of Central Tendency](05_measures_central_tendency.md) - Mean, median, mode with mathematical foundations and applications
- [Measures of Variability](06_measures_variability.md) - Variance, standard deviation, range, and coefficient of variation
- [Data Visualization](07_data_visualization.md) - Advanced plotting with ggplot2, publication-ready graphics, and interactive visualizations
- [Probability Distributions](08_probability_distributions.md) - Normal, binomial, Poisson, and other distributions with applications

### Statistical Inference
- [Sampling and Sampling Distributions](09_sampling_distributions.md) - Central limit theorem, sampling methods, and distribution properties
- [Confidence Intervals](10_confidence_intervals.md) - CI construction, interpretation, and sample size determination
- [Hypothesis Testing](11_hypothesis_testing.md) - Test statistics, p-values, significance levels, and decision making
- [One-Sample Tests](12_one_sample_tests.md) - t-tests, z-tests, and nonparametric alternatives with effect sizes
- [Two-Sample Tests](13_two_sample_tests.md) - Independent and paired samples, parametric and nonparametric approaches

### Analysis of Variance
- [One-Way ANOVA](14_one_way_anova.md) - Single-factor ANOVA with post-hoc tests, effect sizes, and assumptions
- [Two-Way ANOVA](15_two_way_anova.md) - Factorial designs, interaction effects, and main effects analysis
- [Repeated Measures ANOVA](16_repeated_measures_anova.md) - Within-subjects designs, sphericity, and mixed models

### Correlation and Regression
- [Correlation Analysis](17_correlation_analysis.md) - Pearson, Spearman, Kendall correlations with confidence intervals and significance testing
- [Simple Linear Regression](18_simple_linear_regression.md) - Model fitting, diagnostics, prediction, and interpretation
- [Multiple Linear Regression](19_multiple_linear_regression.md) - Multivariate regression, model selection, and multicollinearity
- [Model Diagnostics](20_model_diagnostics.md) - Residual analysis, influence diagnostics, and model validation for linear, logistic, and GLMs

### Advanced Topics
- [Nonparametric Tests](21_nonparametric_tests.md) - Wilcoxon, Mann-Whitney, Kruskal-Wallis, and other rank-based methods
- [Chi-Square Tests](22_chi_square_tests.md) - Goodness-of-fit, independence, and homogeneity tests with effect sizes
- [Time Series Analysis](23_time_series_analysis.md) - Decomposition, stationarity, ARIMA models, and forecasting
- [Multivariate Analysis](24_multivariate_analysis.md) - PCA, factor analysis, clustering, and discriminant analysis
- [Variable Selection and Model Building](25_variable_selection_model_building.md) - Stepwise selection, regularization, PCR, and PLS methods
- [Logistic Regression](26_logistic_regression.md) - Binary, multinomial, and ordinal logistic regression with comprehensive diagnostics

## Getting Started

1. **Prerequisites**: Basic understanding of mathematics and statistics concepts
2. **Software**: Install R and RStudio (latest versions recommended)
3. **Learning Path**: Follow the chapters in order for best results
4. **Practice**: Complete all exercises and work through the practical examples

## Required R Packages

```r
# Core packages for this enhanced course
install.packages(c(
  # Data manipulation and visualization
  "tidyverse",    # Comprehensive data science toolkit
  "ggplot2",      # Advanced plotting and visualization
  "dplyr",        # Data manipulation and transformation
  "readr",        # Fast data import
  "haven",        # Import SPSS, SAS, Stata files
  "tidyr",        # Data tidying and reshaping
  
  # Statistical analysis
  "car",          # Companion to Applied Regression
  "rstatix",      # Pipe-friendly statistical tests
  "broom",        # Tidy statistical output
  "emmeans",      # Estimated marginal means
  "effects",      # Effect displays for regression models
  
  # Visualization and reporting
  "ggpubr",       # Publication-ready plots
  "corrplot",     # Correlation matrix visualization
  "plotly",       # Interactive plots
  "knitr",        # Dynamic report generation
  "rmarkdown",    # R Markdown documents
  
  # Advanced modeling
  "MASS",         # Support functions and datasets
  "nnet",         # Neural networks and multinomial models
  "pROC",         # ROC curve analysis
  "ResourceSelection", # Hosmer-Lemeshow test
  
  # Time series and multivariate analysis
  "forecast",     # Time series forecasting
  "tseries",      # Time series analysis
  "psych",        # Psychometric analysis
  "cluster",      # Clustering algorithms
  
  # Model diagnostics and validation
  "leaps",        # Variable selection
  "glmnet",       # Regularized regression
  "pls",          # Partial least squares
  "caret"         # Classification and regression training
))
```

## Enhanced Learning Features

### Mathematical Foundations
- **LaTeX Formatting**: All mathematical concepts use proper LaTeX notation
- **Step-by-step Derivations**: Clear explanations of statistical formulas and their applications
- **Conceptual Understanding**: Deep dive into the "why" behind statistical methods

### Comprehensive Code Examples
- **Detailed Explanations**: Every code block includes explanations of what it does and why
- **Best Practices**: Industry-standard coding practices and R programming conventions
- **Error Handling**: Common pitfalls and how to avoid them
- **Performance Optimization**: Efficient R code and memory management

### Practical Applications
- **Real-world Datasets**: Examples using realistic data scenarios
- **Industry Context**: Applications from healthcare, finance, social sciences, and more
- **Case Studies**: Complete analysis workflows from data import to final reporting

### Hands-on Learning
- **Progressive Exercises**: From basic to advanced, with clear learning objectives
- **Hints and Solutions**: Guided learning with helpful hints for challenging problems
- **Learning Outcomes**: Clear expectations for what you'll learn from each exercise

### Model Diagnostics and Validation
- **Assumption Checking**: Comprehensive testing of statistical assumptions
- **Diagnostic Plots**: Interpretation of residual plots, Q-Q plots, and influence diagnostics
- **Model Comparison**: AIC, BIC, cross-validation, and other model selection criteria

## Learning Objectives

By the end of this enhanced course, you will be able to:

### Technical Skills
- Import, clean, and preprocess data efficiently in R
- Perform comprehensive exploratory data analysis with advanced visualizations
- Conduct various statistical tests with proper assumption checking
- Build and validate regression models (linear, logistic, GLMs)
- Create publication-ready visualizations and reports
- Apply advanced statistical methods (time series, multivariate analysis)

### Analytical Thinking
- Choose appropriate statistical methods for different research questions
- Interpret statistical results with confidence intervals and effect sizes
- Diagnose and address model violations and data issues
- Compare and select optimal models using multiple criteria
- Communicate statistical findings effectively

### Practical Application
- Apply statistical concepts to real-world problems across domains
- Implement reproducible research workflows
- Generate comprehensive statistical reports
- Use R for both exploratory and confirmatory analysis
- Understand the limitations and assumptions of statistical methods

## How to Use This Repository

Each enhanced markdown file contains:

### Theory and Concepts
- **Mathematical Foundations**: Clear explanations with LaTeX formatting
- **Statistical Background**: Historical context and theoretical underpinnings
- **Assumptions and Limitations**: When and why methods work or fail

### Practical Implementation
- **Step-by-step R Code**: Complete examples with detailed explanations
- **Data Generation**: Realistic simulated datasets for practice
- **Best Practices**: Industry-standard approaches and coding conventions

### Learning Reinforcement
- **Progressive Exercises**: Structured practice problems with increasing difficulty
- **Learning Objectives**: Clear goals for each exercise
- **Hints and Guidance**: Helpful tips without giving away solutions

### Real-world Applications
- **Case Studies**: Complete analysis workflows
- **Domain-specific Examples**: Healthcare, finance, social sciences, etc.
- **Reporting Guidelines**: How to present results professionally

## Course Structure

### Beginner Level (Chapters 1-8)
- R fundamentals and data manipulation
- Descriptive statistics and visualization
- Probability distributions and basic concepts

### Intermediate Level (Chapters 9-16)
- Statistical inference and hypothesis testing
- Analysis of variance and experimental design
- Correlation analysis and simple regression

### Advanced Level (Chapters 17-26)
- Multiple regression and model diagnostics
- Nonparametric methods and categorical data
- Time series, multivariate analysis, and advanced modeling

## Contributing

We welcome contributions to enhance this learning resource:

### Content Improvements
- Add more real-world examples and case studies
- Improve mathematical explanations and derivations
- Expand exercise sets with additional problems
- Add new statistical methods and techniques

### Technical Enhancements
- Optimize R code for better performance
- Add interactive visualizations and Shiny apps
- Include additional diagnostic tools and validation methods
- Create supplementary materials (videos, slides, etc.)

### Quality Assurance
- Fix errors in code or explanations
- Update package dependencies and compatibility
- Improve clarity and readability
- Add more comprehensive testing and validation

## Acknowledgments

This enhanced curriculum builds upon the Applied Statistics with R framework, incorporating:
- Modern R programming practices and tidyverse workflows
- Comprehensive mathematical foundations with proper notation
- Industry-standard diagnostic and validation procedures
- Real-world applications and case studies
- Progressive learning with clear objectives and outcomes

---

*Happy learning and statistical discovery!*