# Statistics

[![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)](https://www.r-project.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![RStudio](https://img.shields.io/badge/RStudio-75AADB?style=for-the-badge&logo=rstudio&logoColor=white)](https://posit.co/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Learning Objectives](#learning-objectives)
- [Prerequisites](#prerequisites)
- [Course Structure](#course-structure)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project contains comprehensive materials for learning both theoretical and practical techniques for applying statistical models to data. The focus is on regression models, which are used to model a variable of interest as a function of explanatory variables.

**Example Applications:**
- Which variables are significant predictors for the success of a restaurant?
- What factors make for a fuel efficient car?
- Can you make accurate predictions of the opening weekend gross of a new film?

You will learn the mathematical fundamentals of linear models, a broad range of models that are the first line of defense in numerous application areas. By the end of the course, you will be able to critique and distinguish variables and models that are useful for predicting and explaining the behavior of a response variable of interest.

## Learning Objectives

Upon successful completion of this course, you will be able to:

### R Programming Skills
- Interact with data using R and RStudio
- Create reproducible reports with RMarkdown to communicate results
- Use regression models to make predictions and explain relationships
- Interpret modeling results in the context of real-world problems
- Utilize simulation to explore statistical properties of models
- Interpret regression models that use categorical predictors and interactions
- Identify and diagnose violations of the assumptions of linear models
- Add complexity to regression models using transformations and interactions
- Use variable selection techniques to select a model
- Perform regression analyses for a binary response
- Conduct comprehensive data summarization and exploratory analysis
- Perform statistical testing and hypothesis testing
- Create publication-quality visualizations

### Python Programming Skills
- Work with data using pandas, numpy, and scipy
- Create reproducible analyses with Jupyter notebooks
- Implement statistical models using scikit-learn and statsmodels
- Visualize data and model results with matplotlib and seaborn
- Perform hypothesis testing and statistical inference
- Build and evaluate machine learning models
- Conduct exploratory data analysis (EDA)
- Create interactive visualizations with plotly
- Deploy models and create reproducible workflows
- Perform comprehensive data summarization and statistical analysis
- Handle outliers and data quality issues
- Create advanced statistical visualizations

## Prerequisites

Before taking this course, you should be familiar with probability and statistics at a level that requires calculus as a prerequisite. In particular, you should know how to:

- Perform basic probability calculations for discrete and continuous distributions, especially the normal distribution
- Given data, calculate various summary statistics
- Perform a one-sample t-test

We will recap the most important concepts needed for this course, but some familiarity will be helpful. You should also have some prior exposure to programming. Previous experience with R or Python is not necessary.

## Course Structure

### Available Modules

#### Introduction (01_intro/)
- **Introduction to R**: Basic R concepts, installation, and first steps
- **Introduction to Python**: Python fundamentals for data science
- **Learning Paths**: Guidance on choosing between R and Python

#### Data Fundamentals (02_data/)
- **R Data Types**: Comprehensive coverage of R data types and structures
- **Python Data Types**: Python data types for statistical analysis
- **Python Data Structures**: Lists, arrays, dictionaries, DataFrames, sets, tuples
- **Python Programming Basics**: Control flow, loops, functions, error handling
- **R Data Structures**: Vectors, matrices, lists, data frames
- **R Programming Basics**: Control flow, loops, functions in R

#### Language-Specific Tracks

##### R Track (R/)
- **Introduction to R** (`01_intro.md`): Basic R concepts and setup
- **R Data Types** (`02_data_types.md`): Comprehensive R data types
- **R Data Structures** (`03_data_structures.md`): R-specific data structures
- **R Programming Basics** (`04_programming_basics.md`): R programming fundamentals
- **Summarizing Data with R** (`05_summarizing_data.md`): Comprehensive statistical analysis

##### Python Track (Python/)
- **Introduction to Python** (`01_intro.md`): Python fundamentals for data science
- **Python Data Types** (`02_data_types.md`): Complete Python data types guide
- **Python Data Structures** (`03_data_structures.md`): Python data structures for statistics
- **Python Programming Basics** (`04_programming_basics.md`): Python programming fundamentals
- **Summarizing Data with Python** (`05_summarizing_data.md`): Comprehensive statistical analysis

#### Summarizing Data (03_summarizing/)
- **Summarizing Data** (`01_summarizing_data.md`): Comprehensive guide to data summarization

### Planned Modules

#### R Track
- **Module 6**: Data Manipulation with dplyr and tidyr
- **Module 7**: Linear Regression Models
- **Module 8**: Model Diagnostics and Assumptions
- **Module 9**: Multiple Regression and Variable Selection
- **Module 10**: Categorical Predictors and Interactions
- **Module 11**: Logistic Regression
- **Module 12**: Reproducible Research with RMarkdown

#### Python Track
- **Module 6**: Statistical Analysis with scipy and statsmodels
- **Module 7**: Linear Regression with scikit-learn
- **Module 8**: Model Evaluation and Diagnostics
- **Module 9**: Feature Engineering and Selection
- **Module 10**: Logistic Regression and Classification
- **Module 11**: Advanced Visualization and Reporting

## Technologies Used

### R Ecosystem
- **R**: Statistical computing and graphics
- **RStudio**: Integrated development environment
- **tidyverse**: Data science packages (dplyr, ggplot2, tidyr, etc.)
- **rmarkdown**: Reproducible reports
- **caret**: Machine learning framework
- **shiny**: Interactive web applications
- **corrplot**: Correlation matrix visualization
- **moments**: Higher-order moments and skewness
- **vcd**: Visualization of categorical data

### Python Ecosystem
- **Python**: Programming language
- **Jupyter**: Interactive computing environment
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning library
- **matplotlib/seaborn**: Data visualization
- **statsmodels**: Statistical modeling
- **plotly**: Interactive visualizations
- **scipy.stats**: Statistical functions and tests

## Getting Started

### Quick Start

1. **Choose Your Path**: Start with the [Introduction section](01_intro/README.md) to decide between R and Python
2. **Follow the Modules**: Work through the available modules in order
3. **Practice**: Use the code examples and exercises in each module

### Setting up R Environment

1. **Install R**: Download from [r-project.org](https://www.r-project.org/)
2. **Install RStudio**: Download from [posit.co](https://posit.co/)
3. **Install required packages**:
   ```r
   install.packages(c("tidyverse", "rmarkdown", "caret", "shiny", "corrplot", "moments", "vcd"))
   ```

### Setting up Python Environment

1. **Install Python**: Download from [python.org](https://www.python.org/)
2. **Install Jupyter**: 
   ```bash
   pip install jupyter
   ```
3. **Install required packages**:
   ```bash
   pip install pandas numpy scipy scikit-learn matplotlib seaborn statsmodels plotly
   ```

### Alternative: Using Conda

```bash
conda create -n statistics python=3.9
conda activate statistics
conda install jupyter pandas numpy scipy scikit-learn matplotlib seaborn statsmodels plotly
```

## Project Structure

```
Statistics/
├── 01_intro/             # Introduction materials
│   ├── README.md         # Overview and learning paths
│   ├── 01_intro_to_r.md  # Introduction to R
│   └── 01_intro_to_python.md # Introduction to Python
├── 02_data/              # Data types and structures
│   ├── README.md         # Overview and learning paths
│   ├── 01_data_types_r.md      # R data types
│   ├── 01_data_types_python.md # Python data types
│   ├── 02_data_structures_r.md # R data structures
│   ├── 02_data_structures_python.md # Python data structures
│   ├── 03_programming_basics_r.md # R programming basics
│   └── 03_programming_basics_python.md # Python programming basics
├── 03_summarizing/       # Data summarization
│   └── 01_summarizing_data.md # Comprehensive data summarization guide
├── R/                    # R-specific track
│   ├── README.md         # R track overview and learning path
│   ├── 01_intro.md       # Introduction to R
│   ├── 02_data_types.md  # R data types
│   ├── 03_data_structures.md # R data structures
│   ├── 04_programming_basics.md # R programming basics
│   └── 05_summarizing_data.md # Summarizing data with R
├── Python/               # Python-specific track
│   ├── README.md         # Python track overview and learning path
│   ├── 01_intro.md       # Introduction to Python
│   ├── 02_data_types.md  # Python data types
│   ├── 03_data_structures.md # Python data structures
│   ├── 04_programming_basics.md # Python programming basics
│   └── 05_summarizing_data.md # Summarizing data with Python
├── docs/                 # Documentation
├── examples/             # Example projects
├── LICENSE               # MIT License
└── README.md            # This file
```

## Learning Paths

### For Beginners
1. **Start with Introduction**: Choose between R and Python based on your goals
2. **Learn Data Fundamentals**: Understand data types and structures
3. **Master Programming Basics**: Learn control flow, functions, and best practices
4. **Apply Statistical Analysis**: Use the language-specific tracks for comprehensive learning

### For Intermediate Users
- Skip to the language-specific tracks (R/ or Python/)
- Focus on the advanced statistical analysis materials
- Practice with real-world examples and datasets

### For Advanced Users
- Use materials as reference for specific techniques
- Focus on the comprehensive statistical analysis modules
- Implement exercises with your own datasets

## Key Features

### Comprehensive Coverage
- **Complete language tracks** for both R and Python
- **Progressive learning** from basics to advanced statistical analysis
- **Real-world examples** with actual datasets
- **Comprehensive exercises** and practice problems

### Statistical Focus
- **Data summarization** and exploratory analysis
- **Statistical testing** and hypothesis testing
- **Outlier detection** and data quality assessment
- **Advanced visualizations** and reporting

### Practical Applications
- **Car fuel efficiency analysis** with real data
- **Student performance analysis** with test scores
- **Categorical data analysis** with survey data
- **Outlier detection** and handling techniques

## Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Additional statistical techniques and examples
- More real-world datasets and case studies
- Enhanced visualizations and reporting
- Additional programming exercises
- Documentation improvements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This project is designed to support learning statistics with both R and Python. The repository contains comprehensive materials for both languages, with language-specific tracks that provide deep coverage of statistical analysis techniques. Choose the track that best fits your needs, or explore both to understand the strengths of each language in statistical analysis.

**Current Status**: 
- ✅ Introduction and Data Fundamentals modules complete
- ✅ Language-specific tracks (R/ and Python/) complete with comprehensive content
- ✅ Data summarization module complete with real-world examples
- 🔄 Advanced statistical modeling modules planned for future development

**Recent Updates**:
- Added comprehensive R and Python language-specific tracks
- Enhanced data summarization with real-world examples
- Added outlier detection and statistical testing
- Included advanced visualizations and reporting techniques