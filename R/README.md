# Statistics with R

[![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)](https://www.r-project.org/)
[![RStudio](https://img.shields.io/badge/RStudio-75AADB?style=for-the-badge&logo=rstudio&logoColor=white)](https://posit.co/)
[![Markdown](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white)](https://markdown.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)](https://github.com/yourusername/statistics-with-r)

A comprehensive guide to learning statistics using R programming language, based on the Applied Statistics with R curriculum.

## Table of Contents

### Fundamentals
- [Introduction to R](01_introduction_to_r.md)
- [Data Types and Structures](02_data_types_structures.md)
- [Data Import and Manipulation](03_data_import_manipulation.md)
- [Exploratory Data Analysis](04_exploratory_data_analysis.md)

### Descriptive Statistics
- [Measures of Central Tendency](05_measures_central_tendency.md)
- [Measures of Variability](06_measures_variability.md)
- [Data Visualization](07_data_visualization.md)
- [Probability Distributions](08_probability_distributions.md)

### Statistical Inference
- [Sampling and Sampling Distributions](09_sampling_distributions.md)
- [Confidence Intervals](10_confidence_intervals.md)
- [Hypothesis Testing](11_hypothesis_testing.md)
- [One-Sample Tests](12_one_sample_tests.md)
- [Two-Sample Tests](13_two_sample_tests.md)

### Analysis of Variance
- [One-Way ANOVA](14_one_way_anova.md)
- [Two-Way ANOVA](15_two_way_anova.md)
- [Repeated Measures ANOVA](16_repeated_measures_anova.md)

### Correlation and Regression
- [Correlation Analysis](17_correlation_analysis.md)
- [Simple Linear Regression](18_simple_linear_regression.md)
- [Multiple Linear Regression](19_multiple_linear_regression.md)
- [Model Diagnostics](20_model_diagnostics.md)

### Advanced Topics
- [Nonparametric Tests](21_nonparametric_tests.md)
- [Chi-Square Tests](22_chi_square_tests.md)
- [Time Series Analysis](23_time_series_analysis.md)
- [Multivariate Analysis](24_multivariate_analysis.md)
- [Variable Selection and Model Building](25_variable_selection_model_building.md)
- [Logistic Regression](26_logistic_regression.md)

## Getting Started

1. **Prerequisites**: Basic understanding of mathematics and statistics concepts
2. **Software**: Install R and RStudio
3. **Learning Path**: Follow the chapters in order for best results

## Required R Packages

```r
# Core packages for this course
install.packages(c(
  "tidyverse",    # Data manipulation and visualization
  "ggplot2",      # Advanced plotting
  "dplyr",        # Data manipulation
  "readr",        # Data import
  "haven",        # Import SPSS, SAS, Stata files
  "car",          # Companion to Applied Regression
  "rstatix",      # Pipe-friendly statistical tests
  "ggpubr",       # Publication ready plots
  "corrplot",     # Correlation plots
  "plotly",       # Interactive plots
  "knitr",        # Dynamic report generation
  "rmarkdown"     # R Markdown
))
```

## Learning Objectives

By the end of this course, you will be able to:

- Import and clean data in R
- Perform exploratory data analysis
- Conduct various statistical tests
- Create publication-ready visualizations
- Interpret statistical results
- Apply statistical concepts to real-world problems

## How to Use This Repository

Each markdown file contains:
- **Theory**: Statistical concepts and background
- **R Code**: Practical examples with code
- **Exercises**: Practice problems to reinforce learning
- **Real-world Applications**: Case studies and examples

## Contributing

Feel free to contribute by:
- Adding more examples
- Improving explanations
- Fixing errors
- Adding new topics

---

*Happy learning!* 