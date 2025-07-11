# Data Visualization

## Overview

Data visualization is a crucial component of statistical analysis and data science. It serves as both an exploratory tool for understanding data patterns and a communication medium for presenting findings effectively. Good visualizations can reveal insights that might be missed in numerical summaries alone.

### The Importance of Data Visualization

Data visualization serves several key purposes:

1. **Exploration**: Discover patterns, trends, and relationships in data
2. **Communication**: Convey complex information clearly to different audiences
3. **Validation**: Verify assumptions and check for data quality issues
4. **Storytelling**: Guide viewers through a narrative about the data

### Visualization Principles

Effective data visualization follows these principles:

- **Accuracy**: Represent data truthfully without distortion
- **Clarity**: Make the message immediately understandable
- **Efficiency**: Maximize information-to-ink ratio
- **Aesthetics**: Use design elements to enhance rather than distract

## Mathematical Foundations

### Coordinate Systems

Most statistical visualizations use Cartesian coordinate systems where:

- **2D Cartesian**: Points defined by $(x, y)$ coordinates
- **3D Cartesian**: Points defined by $(x, y, z)$ coordinates
- **Polar**: Points defined by $(r, \theta)$ where $r$ is radius and $\theta$ is angle

### Statistical Concepts in Visualization

#### Density Estimation

For continuous data, we often estimate probability density functions:

```math
f(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)
```

Where:
- $K$ is the kernel function
- $h$ is the bandwidth parameter
- $n$ is the number of observations

#### Correlation Visualization

The correlation coefficient $\rho$ between variables $X$ and $Y$:

```math
\rho = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}
```

#### Quantile-Quantile (Q-Q) Plots

Q-Q plots compare sample quantiles to theoretical quantiles:

```math
Q(p) = F^{-1}(p)
```

Where $F^{-1}$ is the inverse cumulative distribution function.

## Matplotlib and Seaborn

### Basic Plot Types

Python provides powerful plotting capabilities through `matplotlib` and `seaborn`. Let's explore the core plot types with detailed explanations:

**Python Code Reference:** See `basic_plot_types()` function in `07_data_visualization.py` for examples of scatter plots, line plots, histograms, and box plots.

The function demonstrates:
- Scatter plots showing relationships between continuous variables
- Line plots for trends and mathematical functions
- Histograms for distribution visualization
- Box plots for grouped data analysis

### Understanding Plot Parameters

#### Point Types and Symbols

**Python Code Reference:** See `understanding_plot_parameters()` function in `07_data_visualization.py` for demonstrations of different point types and symbols.

The function shows:
- Various marker types available in matplotlib
- Visual demonstration of each marker type
- Legend positioning and customization

#### Color Systems

Python supports multiple color specifications:

**Python Code Reference:** See `understanding_plot_parameters()` function in `07_data_visualization.py` for color system demonstrations.

The function demonstrates:
- Named colors in matplotlib
- RGB color specifications
- Hexadecimal color codes
- Color palette visualization

### Customizing Plots

**Python Code Reference:** See `customizing_plots()` function in `07_data_visualization.py` for enhanced scatter plot customization.

The function demonstrates:
- Multiple customization options for scatter plots
- Legend positioning and custom legend elements
- Grid customization
- Regression line with statistical annotations
- Text annotations and mathematical formulas

### Advanced Matplotlib Features

#### Multiple Plots on One Page

**Python Code Reference:** See `advanced_matplotlib_features()` function in `07_data_visualization.py` for multiple plots on one page.

The function demonstrates:
- Creating subplots with different layouts
- Multiple plot types on the same figure
- Density estimation with kernel density estimation
- Proper layout management with tight_layout()

#### Interactive Plotting

**Python Code Reference:** See `interactive_visualizations()` function in `07_data_visualization.py` for interactive plotting examples.

The function demonstrates:
- Interactive scatter plots with plotly
- Hover information and data exploration
- Integration with statistical models
- Alternative interactive libraries like bokeh

## Seaborn - Statistical Data Visualization

### Installing and Loading Seaborn

**Python Code Reference:** See `seaborn_setup()` function in `07_data_visualization.py` for seaborn installation and setup.

The function demonstrates:
- Proper seaborn installation and import
- Style configuration for better-looking plots
- Palette setup for consistent color schemes

### The Grammar of Graphics Philosophy

Seaborn builds on matplotlib and implements many concepts from Leland Wilkinson's Grammar of Graphics, which breaks down plots into components:

1. **Data**: The dataset being visualized
2. **Aesthetics**: Mappings from data to visual properties
3. **Geometries**: The actual marks used to represent data
4. **Scales**: Control how aesthetics are mapped to visual properties
5. **Facets**: Subdivision of data into multiple plots
6. **Themes**: Control of non-data elements

### Basic Seaborn Syntax

**Python Code Reference:** See `basic_seaborn_syntax()` function in `07_data_visualization.py` for basic seaborn syntax examples.

The function demonstrates:
- Basic seaborn scatter plots
- Enhanced plots with multiple aesthetics (hue, size, alpha)
- Proper legend handling and customization

### Understanding Aesthetics

Aesthetics map data variables to visual properties:

**Python Code Reference:** See `understanding_aesthetics()` function in `07_data_visualization.py` for aesthetic mapping demonstrations.

The function demonstrates:
- Common aesthetics and their mathematical interpretation
- Data type considerations for different aesthetics
- Visual property mappings
- Mathematical scale transformations

### Different Plot Types

#### Scatter Plots

**Python Code Reference:** See `seaborn_scatter_plots()` function in `07_data_visualization.py` for seaborn scatter plot examples.

The function demonstrates:
- Basic scatter plots with regression lines
- Multiple aesthetics mapping (hue, size, alpha)
- Mathematical annotations and correlation coefficients
- Overlaying regression lines on scatter plots

#### Histograms and Density Plots

**Python Code Reference:** See `seaborn_histograms_density()` function in `07_data_visualization.py` for histogram and density plot examples.

The function demonstrates:
- Histograms with density overlays
- Kernel density estimation (KDE) plots
- Faceted histograms by groups
- Proper statistical visualization techniques

#### Box Plots and Violin Plots

**Python Code Reference:** See `seaborn_box_violin_plots()` function in `07_data_visualization.py` for box plot and violin plot examples.

The function demonstrates:
- Box plots with individual data points (stripplots)
- Violin plots showing full distribution shapes
- Combining box plots and violin plots
- Statistical distribution visualization

#### Bar Plots

**Python Code Reference:** See `seaborn_bar_plots()` function in `07_data_visualization.py` for bar plot examples.

The function demonstrates:
- Count bar plots for categorical data
- Summary bar plots with error bars
- Statistical aggregation and standard error calculations
- Proper error bar visualization

#### Line Plots

**Python Code Reference:** See `seaborn_line_plots()` function in `07_data_visualization.py` for line plot examples.

The function demonstrates:
- Time series data creation with mathematical functions
- Multiple lines on the same plot
- Mathematical function comparisons (linear, quadratic, exponential, sine)
- Proper legend and grid formatting

### Faceting

Faceting creates multiple plots based on categorical variables:

**Python Code Reference:** See `seaborn_faceting()` function in `07_data_visualization.py` for faceting examples.

The function demonstrates:
- Faceting by one variable with multiple panels
- Faceting by two variables (rows and columns)
- Adding regression lines to faceted plots
- Proper title and annotation formatting

### Themes and Customization

**Python Code Reference:** See `themes_and_customization()` function in `07_data_visualization.py` for theme and customization examples.

The function demonstrates:
- Different built-in matplotlib themes
- Custom theme creation with mathematical precision
- Font styling and sizing
- Grid and background customization

## Advanced Visualizations

### Correlation Matrix

**Python Code Reference:** See `correlation_matrix()` function in `07_data_visualization.py` for correlation matrix heatmap examples.

The function demonstrates:
- Correlation matrix calculation with mathematical precision
- Seaborn heatmap with masked upper triangle
- Matplotlib heatmap with custom annotations
- Proper color mapping and text formatting

### Violin Plots

**Python Code Reference:** See `violin_plots_advanced()` function in `07_data_visualization.py` for advanced violin plot examples.

The function demonstrates:
- Violin plots with statistical annotations
- Combining violin plots with box plots
- Adding mean points with custom markers
- Statistical interpretation annotations

### Heatmaps

**Python Code Reference:** See `heatmaps_advanced()` function in `07_data_visualization.py` for advanced heatmap examples.

The function demonstrates:
- Data scaling with StandardScaler
- Z-score transformation for heatmap visualization
- Mathematical scaling considerations
- Proper heatmap formatting and labeling

### 3D Scatter Plots

**Python Code Reference:** See `three_d_scatter_plots()` function in `07_data_visualization.py` for 3D scatter plot examples.

The function demonstrates:
- 3D scatter plots with mathematical perspective
- Color mapping for additional variables
- Viewing angle control
- Proper axis labeling and colorbar integration

## Statistical Visualizations

### Q-Q Plots

Q-Q plots compare sample quantiles to theoretical quantiles:

**Python Code Reference:** See `qq_plots()` function in `07_data_visualization.py` for Q-Q plot examples.

The function demonstrates:
- Q-Q plots for normality testing
- Shapiro-Wilk test for normality
- Mathematical annotations and interpretations
- Multiple plotting approaches (matplotlib and scipy)

### Residual Plots

**Python Code Reference:** See `residual_plots()` function in `07_data_visualization.py` for residual plot examples.

The function demonstrates:
- Linear model fitting and residual calculation
- Basic residual plots with mathematical diagnostics
- Confidence bands for residuals
- Residual standard error calculations

### Distribution Comparison

**Python Code Reference:** See `distribution_comparison()` function in `07_data_visualization.py` for distribution comparison examples.

The function demonstrates:
- Comparing distributions across groups using KDE plots
- Statistical summary calculations by groups
- Density visualization techniques
- Group-wise statistical analysis

## Interactive Visualizations

### Plotly

**Python Code Reference:** See `interactive_visualizations()` function in `07_data_visualization.py` for interactive visualization examples.

The function demonstrates:
- Interactive scatter plots with plotly
- Hover information and data exploration
- Integration with statistical models
- Alternative interactive libraries like bokeh

### Bokeh (Alternative to Highcharter)

**Python Code Reference:** See `interactive_visualizations()` function in `07_data_visualization.py` for bokeh interactive visualization examples.

The function demonstrates:
- Interactive charts with bokeh
- Hover tools and data exploration
- Statistical features integration
- Alternative to plotly for interactive visualizations

## Best Practices

### Color Choices

**Python Code Reference:** See `best_practices_color_choices()` function in `07_data_visualization.py` for color choice best practices.

The function demonstrates:
- Color-blind friendly palettes
- Mathematical color theory applications
- Sequential color palettes for continuous data
- Accessibility considerations in visualization

### Saving Plots

**Python Code Reference:** See `saving_plots()` function in `07_data_visualization.py` for plot saving examples.

The function demonstrates:
- Saving plots in different formats (PNG, PDF)
- High resolution settings for publication
- Vector graphics for scalability
- Mathematical precision in saved plots

### Mathematical Annotations

**Python Code Reference:** See `mathematical_annotations()` function in `07_data_visualization.py` for mathematical annotation examples.

The function demonstrates:
- Adding mathematical formulas and statistical information to plots
- R-squared and p-value calculations
- Text annotations with statistical context
- Integration of statistical tests with visualizations

## Practical Examples

### Example 1: Comprehensive Data Analysis

**Python Code Reference:** See `practical_example_comprehensive_analysis()` function in `07_data_visualization.py` for comprehensive data analysis example.

The function demonstrates:
- Loading and preparing the iris dataset
- Creating multiple plots with mathematical insights
- Scatter plots with regression lines by groups
- Box plots with mean annotations
- Density plots for distribution comparison

### Example 2: Time Series Visualization

**Python Code Reference:** See `practical_example_time_series()` function in `07_data_visualization.py` for time series visualization example.

The function demonstrates:
- Creating time series data with mathematical functions
- Trend and seasonal component decomposition
- Time series plotting with multiple components
- Proper date formatting and axis rotation

### Example 3: Statistical Process Control

**Python Code Reference:** See `practical_example_statistical_process_control()` function in `07_data_visualization.py` for statistical process control example.

The function demonstrates:
- Creating control chart data with process shifts
- Calculating control limits and warning limits
- Statistical process control chart visualization
- Proper annotation and interpretation guidelines

## Exercises

### Exercise 1: Basic Plots
Create scatter plots, histograms, and box plots for the `mtcars` dataset. Calculate and display correlation coefficients and summary statistics.

### Exercise 2: Customization
Create a publication-ready plot with custom colors, themes, and mathematical annotations including R² values and p-values.

### Exercise 3: Faceting
Create faceted plots to show relationships across different groups. Include statistical summaries for each facet.

### Exercise 4: Interactive Plots
Create an interactive visualization using plotly or bokeh with hover information and zoom capabilities.

### Exercise 5: Statistical Plots
Create Q-Q plots and residual plots for a linear regression model. Perform normality tests and interpret the results.

### Exercise 6: Advanced Visualization
Create a correlation matrix heatmap and a 3D scatter plot. Interpret the mathematical relationships shown.

## Next Steps

In the next chapter, we'll learn about probability distributions and how to work with them in Python, building on the visualization concepts we've covered here.

---

**Key Takeaways:**
- Matplotlib provides fundamental plotting capabilities with mathematical precision
- Seaborn builds on matplotlib and provides a consistent grammar of graphics with mathematical foundations
- Choose appropriate plot types based on data characteristics and research questions
- Use color and themes effectively while considering accessibility
- Include mathematical annotations and statistical information
- Save plots in appropriate formats for different purposes
- Interactive plots enhance exploration and communication
- Always include proper labels, titles, and mathematical context
- Consider the mathematical relationships and statistical assumptions in your visualizations 