"""
Data Visualization - Python Code Examples
Corresponds to: 07_data_visualization.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from scipy import stats
from scipy.stats import gaussian_kde, probplot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load sample data
def load_mtcars():
    """
    Load the mtcars dataset from OpenML as a pandas DataFrame.
    """
    mtcars = fetch_openml(name='mtcars', as_frame=True).frame
    return mtcars

def basic_plot_types():
    """
    Demonstrate basic plot types: scatter plot, line plot, histogram, and box plot.
    """
    mtcars = load_mtcars()
    
    # Scatter plot - shows relationship between two continuous variables
    plt.figure(figsize=(8, 6))
    plt.scatter(mtcars['wt'], mtcars['mpg'], c='blue', s=50, alpha=0.7)
    plt.title('MPG vs Weight')
    plt.xlabel('Weight (1000 lbs)')
    plt.ylabel('Miles per Gallon')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Line plot - shows trends over time or ordered sequences
    x = np.arange(1, 11)
    y = x**2
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.title('Quadratic Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Histogram - shows distribution of continuous data
    plt.figure(figsize=(8, 6))
    plt.hist(mtcars['mpg'], bins=10, color='lightblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of MPG')
    plt.xlabel('Miles per Gallon')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Box plot - shows distribution and outliers by groups
    plt.figure(figsize=(8, 6))
    mtcars.boxplot(column='mpg', by='cyl', ax=plt.gca())
    plt.title('MPG by Number of Cylinders')
    plt.suptitle('')  # Remove default suptitle
    plt.xlabel('Cylinders')
    plt.ylabel('Miles per Gallon')
    plt.show()

def understanding_plot_parameters():
    """
    Demonstrate different point types, symbols, and color systems.
    """
    # Different point types (marker parameter)
    markers = ['o', 's', '^', '+', 'x', 'D', 'v', 'p', '*', 'h', 'H', 'd', '|', '_']
    marker_names = ['Circle', 'Square', 'Triangle Up', 'Plus', 'X', 'Diamond', 
                    'Triangle Down', 'Pentagon', 'Star', 'Hexagon', 'Hexagon2', 
                    'Thin Diamond', 'Vertical Line', 'Horizontal Line']
    
    # Visual demonstration of point types
    plt.figure(figsize=(12, 8))
    for i, (marker, name) in enumerate(zip(markers, marker_names)):
        plt.scatter(i, i, marker=marker, s=200, label=f'{marker}: {name}')
    
    plt.title('Point Types in Python')
    plt.xlabel('Point Type Number')
    plt.ylabel('Y Position')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Named colors
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    # Color palette demonstration
    plt.figure(figsize=(8, 6))
    for i, color in enumerate(colors):
        plt.scatter(i, i, c=color, s=300, alpha=0.7)
    plt.title('Color Examples')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.show()

def customizing_plots():
    """
    Demonstrate enhanced scatter plot with multiple customization options.
    """
    mtcars = load_mtcars()
    
    # Enhanced scatter plot with multiple customization options
    plt.figure(figsize=(10, 8))
    colors = ['red' if am == 1 else 'blue' for am in mtcars['am']]
    plt.scatter(mtcars['wt'], mtcars['mpg'], c=colors, s=100, alpha=0.7)
    plt.title('Fuel Efficiency vs Weight')
    plt.xlabel('Weight (1000 lbs)')
    plt.ylabel('Miles per Gallon')
    plt.xlim(1.5, 5.5)
    plt.ylim(10, 35)
    
    # Add legend with positioning
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Manual'),
                       Patch(facecolor='blue', label='Automatic')]
    plt.legend(handles=legend_elements, title='Transmission', loc='upper right')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add regression line with confidence interval
    slope, intercept, r_value, p_value, std_err = stats.linregress(mtcars['wt'], mtcars['mpg'])
    x_line = np.array([mtcars['wt'].min(), mtcars['wt'].max()])
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color='green', linewidth=2, label=f'R² = {r_value**2:.3f}')
    
    # Add text annotations
    plt.text(4, 30, 'Strong negative correlation', color='darkgreen', fontweight='bold')
    plt.legend()
    plt.show()

def advanced_matplotlib_features():
    """
    Demonstrate multiple plots on one page and interactive plotting capabilities.
    """
    mtcars = load_mtcars()
    
    # Set up multiple plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Scatter plot
    axes[0, 0].scatter(mtcars['wt'], mtcars['mpg'])
    axes[0, 0].set_title('MPG vs Weight')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram
    axes[0, 1].hist(mtcars['mpg'], bins=10, color='lightblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('MPG Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Box plot
    mtcars.boxplot(column='mpg', by='cyl', ax=axes[1, 0])
    axes[1, 0].set_title('MPG by Cylinders')
    axes[1, 0].set_xlabel('Cylinders')
    
    # Plot 4: Density plot
    kde = gaussian_kde(mtcars['mpg'])
    x_range = np.linspace(mtcars['mpg'].min(), mtcars['mpg'].max(), 100)
    axes[1, 1].plot(x_range, kde(x_range))
    axes[1, 1].set_title('MPG Density')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def seaborn_setup():
    """
    Set up seaborn with style and palette configuration.
    """
    # Set the style for better-looking plots
    sns.set_style("whitegrid")
    sns.set_palette("husl")

def basic_seaborn_syntax():
    """
    Demonstrate basic seaborn syntax and enhanced scatter plots.
    """
    mtcars = load_mtcars()
    
    # Basic scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=mtcars, x='wt', y='mpg')
    plt.title('MPG vs Weight')
    plt.show()
    
    # Enhanced scatter plot with multiple aesthetics
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=mtcars, x='wt', y='mpg', 
                    hue='am', size='hp', alpha=0.7, sizes=(50, 200))
    plt.title('Fuel Efficiency vs Weight')
    plt.xlabel('Weight (1000 lbs)')
    plt.ylabel('Miles per Gallon')
    plt.legend(title='Transmission')
    plt.show()

def understanding_aesthetics():
    """
    Demonstrate aesthetic mapping and mathematical interpretation.
    """
    mtcars = load_mtcars()
    
    # Common aesthetics and their mathematical interpretation
    aesthetics_demo = pd.DataFrame({
        'aesthetic': ['x', 'y', 'color', 'size', 'shape', 'alpha'],
        'data_type': ['continuous', 'continuous', 'discrete/continuous', 
                      'continuous', 'discrete', 'continuous'],
        'visual_property': ['horizontal position', 'vertical position', 
                           'hue', 'area/radius', 'symbol type', 'transparency'],
        'mathematical_scale': ['linear', 'linear', 'categorical/continuous', 
                              'area/radius', 'categorical', 'opacity']
    })
    
    print(aesthetics_demo)
    
    # Demonstrate aesthetic mapping
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=mtcars, x='wt', y='mpg', 
                    hue='cyl',           # Categorical color
                    size='hp',           # Continuous size
                    alpha=0.7,           # Fixed transparency
                    sizes=(50, 300))     # Size range
    plt.title('Multiple Aesthetics Mapping')
    plt.xlabel('Weight (1000 lbs)')
    plt.ylabel('Miles per Gallon')
    plt.legend(title='Cylinders', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def seaborn_scatter_plots():
    """
    Demonstrate seaborn scatter plots with regression lines and annotations.
    """
    mtcars = load_mtcars()
    
    # Basic scatter plot with regression line
    plt.figure(figsize=(10, 8))
    sns.regplot(data=mtcars, x='wt', y='mpg', 
                scatter_kws={'alpha': 0.7}, line_kws={'color': 'red'})
    plt.title('MPG vs Weight with Regression Line')
    plt.xlabel('Weight (1000 lbs)')
    plt.ylabel('Miles per Gallon')
    plt.show()
    
    # Scatter plot with multiple aesthetics and mathematical annotations
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=mtcars, x='wt', y='mpg', 
                    hue='cyl', size='hp', alpha=0.7, sizes=(50, 300))
    sns.regplot(data=mtcars, x='wt', y='mpg', 
                scatter=False, line_kws={'color': 'black', 'linewidth': 2})
    
    # Add correlation coefficient annotation
    correlation = mtcars['wt'].corr(mtcars['mpg'])
    plt.text(4, 30, f'r = {correlation:.3f}', 
             color='red', fontweight='bold', fontsize=12)
    
    plt.title('MPG vs Weight by Cylinders and Horsepower')
    plt.xlabel('Weight (1000 lbs)')
    plt.ylabel('Miles per Gallon')
    plt.legend(title='Cylinders', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def seaborn_histograms_density():
    """
    Demonstrate seaborn histograms and density plots.
    """
    mtcars = load_mtcars()
    
    # Basic histogram with density overlay
    plt.figure(figsize=(10, 8))
    sns.histplot(data=mtcars, x='mpg', stat='density', bins=10, 
                 color='lightblue', alpha=0.7, edgecolor='black')
    sns.kdeplot(data=mtcars, x='mpg', color='red', linewidth=2)
    plt.title('Distribution of MPG with Density Overlay')
    plt.xlabel('Miles per Gallon')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Faceted histogram by groups
    g = sns.FacetGrid(mtcars, col='cyl', col_wrap=2, height=4, aspect=1.5)
    g.map_dataframe(sns.histplot, x='mpg', bins=8, alpha=0.7)
    g.set_titles(col_template='Cylinders: {col_name}')
    g.fig.suptitle('MPG Distribution by Cylinders', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

def seaborn_box_violin_plots():
    """
    Demonstrate seaborn box plots and violin plots.
    """
    mtcars = load_mtcars()
    
    # Box plot with individual points
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=mtcars, x='cyl', y='mpg', color='lightblue', alpha=0.7)
    sns.stripplot(data=mtcars, x='cyl', y='mpg', color='darkblue', alpha=0.5, size=4)
    plt.title('MPG by Cylinders with Individual Points')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('Miles per Gallon')
    plt.show()
    
    # Violin plot showing full distribution shape
    plt.figure(figsize=(10, 8))
    sns.violinplot(data=mtcars, x='cyl', y='mpg', color='lightgreen', alpha=0.7)
    sns.boxplot(data=mtcars, x='cyl', y='mpg', width=0.2, color='white', alpha=0.8)
    plt.title('MPG Distribution by Cylinders (Violin Plot)')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('Miles per Gallon')
    plt.show()

def seaborn_bar_plots():
    """
    Demonstrate seaborn bar plots with count and summary statistics.
    """
    mtcars = load_mtcars()
    
    # Count bar plot
    plt.figure(figsize=(8, 6))
    sns.countplot(data=mtcars, x='cyl', color='steelblue', alpha=0.8)
    plt.title('Count of Cars by Cylinders')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('Count')
    plt.show()
    
    # Summary bar plot with error bars
    cyl_summary = mtcars.groupby('cyl').agg({
        'mpg': ['mean', 'std', 'count']
    }).round(3)
    cyl_summary.columns = ['mean_mpg', 'std_mpg', 'count']
    cyl_summary['se_mpg'] = cyl_summary['std_mpg'] / np.sqrt(cyl_summary['count'])
    
    plt.figure(figsize=(10, 8))
    bars = plt.bar(cyl_summary.index, cyl_summary['mean_mpg'], 
                   color='lightcoral', alpha=0.8)
    plt.errorbar(cyl_summary.index, cyl_summary['mean_mpg'], 
                 yerr=cyl_summary['se_mpg'], fmt='none', color='darkred', capsize=5)
    plt.title('Average MPG by Cylinders with Standard Error')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('Average MPG')
    plt.grid(True, alpha=0.3)
    plt.show()

def seaborn_line_plots():
    """
    Demonstrate seaborn line plots with mathematical functions.
    """
    # Create time series data with mathematical functions
    time_data = pd.DataFrame({
        'time': range(1, 101),
        'linear': range(1, 101),
        'quadratic': [x**2 for x in range(1, 101)],
        'exponential': [np.exp(x/20) for x in range(1, 101)],
        'sine': [np.sin(x/10) * 50 + 50 for x in range(1, 101)]
    })
    
    # Multiple lines on same plot
    plt.figure(figsize=(12, 8))
    plt.plot(time_data['time'], time_data['linear'], label='Linear', linewidth=2)
    plt.plot(time_data['time'], time_data['quadratic'], label='Quadratic', linewidth=2)
    plt.plot(time_data['time'], time_data['exponential'], label='Exponential', linewidth=2)
    plt.plot(time_data['time'], time_data['sine'], label='Sine', linewidth=2)
    
    plt.title('Mathematical Functions Comparison')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def seaborn_faceting():
    """
    Demonstrate seaborn faceting capabilities for creating multiple plots.
    """
    mtcars = load_mtcars()
    
    # Facet by one variable
    g = sns.FacetGrid(mtcars, col='cyl', col_wrap=2, height=4, aspect=1.5)
    g.map_dataframe(sns.scatterplot, x='wt', y='mpg')
    g.map_dataframe(sns.regplot, x='wt', y='mpg', scatter=False, color='red')
    g.set_titles(col_template='Cylinders: {col_name}')
    g.fig.suptitle('MPG vs Weight by Cylinders', y=1.02, fontsize=16)
    g.fig.text(0.5, 0.02, 'Each panel shows relationship for different cylinder counts', 
               ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Facet by two variables (transmission and cylinders)
    g = sns.FacetGrid(mtcars, row='am', col='cyl', height=3, aspect=1.2)
    g.map_dataframe(sns.scatterplot, x='wt', y='mpg')
    g.map_dataframe(sns.regplot, x='wt', y='mpg', scatter=False, color='red')
    g.set_titles(row_template='Transmission: {row_name}', col_template='Cylinders: {col_name}')
    g.fig.suptitle('MPG vs Weight by Transmission and Cylinders', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

def themes_and_customization():
    """
    Demonstrate different themes and customization options.
    """
    mtcars = load_mtcars()
    
    # Apply different themes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Different built-in themes
    themes = ['default', 'classic', 'bmh', 'ggplot', 'fivethirtyeight', 'seaborn-v0_8']
    
    for i, theme in enumerate(themes):
        row = i // 3
        col = i % 3
        with plt.style.context(theme):
            axes[row, col].scatter(mtcars['wt'], mtcars['mpg'])
            axes[row, col].set_title(f'{theme} theme')
            axes[row, col].set_xlabel('Weight')
            axes[row, col].set_ylabel('MPG')
    
    plt.tight_layout()
    plt.show()
    
    # Custom theme with mathematical precision
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(mtcars['wt'], mtcars['mpg'], alpha=0.7)
    ax.set_title('MPG vs Weight', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Weight (1000 lbs)', fontsize=12, style='italic')
    ax.set_ylabel('Miles per Gallon', fontsize=12, style='italic')
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()

def correlation_matrix():
    """
    Create correlation matrix heatmaps with mathematical precision.
    """
    mtcars = load_mtcars()
    
    # Calculate correlation matrix with mathematical precision
    cor_matrix = mtcars[['mpg', 'cyl', 'disp', 'hp', 'wt', 'qsec']].corr()
    
    # Create correlation heatmap using seaborn
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(cor_matrix, dtype=bool))
    sns.heatmap(cor_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Alternative: Using matplotlib for more control
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cor_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cor_matrix.columns)))
    ax.set_yticks(range(len(cor_matrix.columns)))
    ax.set_xticklabels(cor_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(cor_matrix.columns)
    
    # Add text annotations
    for i in range(len(cor_matrix.columns)):
        for j in range(len(cor_matrix.columns)):
            text = ax.text(j, i, f'{cor_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black")
    
    plt.colorbar(im)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()

def violin_plots_advanced():
    """
    Create violin plots with statistical annotations.
    """
    mtcars = load_mtcars()
    
    # Violin plot with statistical annotations
    plt.figure(figsize=(10, 8))
    sns.violinplot(data=mtcars, x='cyl', y='mpg', color='lightblue', alpha=0.7)
    sns.boxplot(data=mtcars, x='cyl', y='mpg', width=0.2, color='white', alpha=0.8)
    
    # Add mean points
    means = mtcars.groupby('cyl')['mpg'].mean()
    plt.scatter(range(len(means)), means, color='red', s=100, marker='D', zorder=5)
    
    plt.title('MPG Distribution by Cylinders')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('Miles per Gallon')
    plt.text(0.02, 0.98, 'Red diamond shows mean, box shows quartiles', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.show()

def heatmaps_advanced():
    """
    Create heatmaps with mathematical scaling.
    """
    mtcars = load_mtcars()
    
    # Create heatmap data with scaling
    scaler = StandardScaler()
    heatmap_data = pd.DataFrame(
        scaler.fit_transform(mtcars[['mpg', 'cyl', 'disp', 'hp', 'wt', 'qsec']]),
        columns=['mpg', 'cyl', 'disp', 'hp', 'wt', 'qsec']
    )
    
    # Create heatmap with mathematical scaling
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data.T, cmap='RdBu_r', center=0, 
                xticklabels=False, yticklabels=True)
    plt.title('Variable Correlation Heatmap (Z-scores)')
    plt.show()

def three_d_scatter_plots():
    """
    Create 3D scatter plots with mathematical perspective.
    """
    mtcars = load_mtcars()
    
    # 3D scatter plot with mathematical perspective
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(mtcars['wt'], mtcars['hp'], mtcars['mpg'], 
                         c=mtcars['cyl'], cmap='viridis', s=50, alpha=0.7)
    
    ax.set_xlabel('Weight')
    ax.set_ylabel('Horsepower')
    ax.set_zlabel('MPG')
    ax.set_title('3D Scatter Plot: Weight, Horsepower, MPG')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cylinders')
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    plt.show()

def qq_plots():
    """
    Create Q-Q plots for normality testing.
    """
    mtcars = load_mtcars()
    
    # Q-Q plot for normality testing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Using matplotlib
    stats.probplot(mtcars['mpg'], dist="norm", plot=ax1)
    ax1.set_title('Normal Q-Q Plot for MPG')
    ax1.grid(True, alpha=0.3)
    
    # Using seaborn with mathematical annotations
    probplot(mtcars['mpg'], dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot for MPG')
    ax2.text(0.05, 0.95, 'Points should follow red line for normality', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Shapiro-Wilk test for normality
    shapiro_result = stats.shapiro(mtcars['mpg'])
    print(f"Shapiro-Wilk test p-value: {shapiro_result.pvalue:.4f}")

def residual_plots():
    """
    Create residual plots with mathematical diagnostics.
    """
    mtcars = load_mtcars()
    
    # Fit linear model
    X = mtcars[['wt']]
    y = mtcars['mpg']
    model = LinearRegression()
    model.fit(X, y)
    
    # Get predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Residual plot with mathematical diagnostics
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='red', linewidth=2)
    plt.title('Residual Plot')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Add confidence bands for residuals
    residual_data = pd.DataFrame({
        'fitted': y_pred,
        'residuals': residuals
    })
    
    # Calculate residual standard error
    rse = np.sqrt(np.sum(residuals**2) / (len(residuals) - 2))
    
    plt.figure(figsize=(10, 8))
    plt.scatter(residual_data['fitted'], residual_data['residuals'], alpha=0.7)
    plt.axhline(y=0, color='red', linewidth=2)
    plt.axhline(y=2*rse, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=-2*rse, color='red', linestyle='--', alpha=0.7)
    plt.title(f'Residual Plot with 95% Confidence Bands\nResidual Standard Error = {rse:.3f}')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    plt.show()

def distribution_comparison():
    """
    Compare distributions across groups with statistical summaries.
    """
    mtcars = load_mtcars()
    
    # Compare distributions across groups
    plt.figure(figsize=(10, 8))
    for cyl in mtcars['cyl'].unique():
        subset = mtcars[mtcars['cyl'] == cyl]
        sns.kdeplot(data=subset['mpg'], label=f'Cylinders: {cyl}', alpha=0.7)
    
    plt.title('MPG Density by Cylinders')
    plt.xlabel('Miles per Gallon')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Statistical summary by groups
    summary = mtcars.groupby('cyl').agg({
        'mpg': ['count', 'mean', 'std', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    }).round(3)
    summary.columns = ['n', 'mean_mpg', 'sd_mpg', 'median_mpg', 'q25', 'q75']
    print(summary)

def interactive_visualizations():
    """
    Demonstrate interactive visualizations using plotly and bokeh.
    """
    mtcars = load_mtcars()
    
    # Note: These require plotly and bokeh to be installed
    # pip install plotly bokeh
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Create interactive scatter plot with mathematical annotations
        fig = px.scatter(mtcars, x='wt', y='mpg', color='cyl',
                         title='Interactive MPG vs Weight',
                         labels={'wt': 'Weight (1000 lbs)', 'mpg': 'Miles per Gallon'},
                         hover_data=['hp', 'disp'])
        
        # Add regression line
        X = mtcars[['wt']]
        y = mtcars['mpg']
        model = LinearRegression()
        model.fit(X, y)
        
        fig.add_trace(go.Scatter(x=mtcars['wt'], y=model.predict(mtcars[['wt']]),
                                 mode='lines', name='Regression Line',
                                 line=dict(color='red', dash='dash')))
        
        fig.show()
        
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
    
    try:
        from bokeh.plotting import figure, show
        from bokeh.io import output_notebook
        from bokeh.models import ColumnDataSource, HoverTool
        
        # Create interactive chart with statistical features
        output_notebook()
        
        source = ColumnDataSource(mtcars)
        p = figure(title='Interactive MPG vs Weight', 
                   x_axis_label='Weight', y_axis_label='MPG')
        
        p.scatter('wt', 'mpg', source=source, color='cyl', size=8, alpha=0.7)
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ('Weight', '@wt'),
            ('MPG', '@mpg'),
            ('Cylinders', '@cyl'),
            ('Horsepower', '@hp')
        ])
        p.add_tools(hover)
        
        show(p)
        
    except ImportError:
        print("Bokeh not installed. Install with: pip install bokeh")

def best_practices_color_choices():
    """
    Demonstrate color-blind friendly palettes and sequential color schemes.
    """
    mtcars = load_mtcars()
    
    # Color-blind friendly palette (mathematically designed)
    colorblind_palette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", 
                         "#0072B2", "#D55E00", "#CC79A7"]
    
    # Use in plot with mathematical color theory
    plt.figure(figsize=(10, 8))
    for i, cyl in enumerate(mtcars['cyl'].unique()):
        subset = mtcars[mtcars['cyl'] == cyl]
        plt.scatter(subset['wt'], subset['mpg'], 
                   c=colorblind_palette[i], s=50, alpha=0.7, label=f'Cylinders: {cyl}')
    
    plt.title('MPG vs Weight with Colorblind-Friendly Colors')
    plt.xlabel('Weight (1000 lbs)')
    plt.ylabel('Miles per Gallon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Sequential color palette for continuous data
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(mtcars['wt'], mtcars['mpg'], c=mtcars['hp'], 
                         cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Horsepower')
    plt.title('MPG vs Weight by Horsepower')
    plt.xlabel('Weight (1000 lbs)')
    plt.ylabel('Miles per Gallon')
    plt.grid(True, alpha=0.3)
    plt.show()

def saving_plots():
    """
    Demonstrate saving plots in different formats with mathematical precision.
    """
    mtcars = load_mtcars()
    
    # Save plot as PNG with mathematical precision
    plt.figure(figsize=(10, 8))
    sns.regplot(data=mtcars, x='wt', y='mpg', 
                scatter_kws={'alpha': 0.7}, line_kws={'color': 'red'})
    plt.title('MPG vs Weight with Regression')
    plt.xlabel('Weight (1000 lbs)')
    plt.ylabel('Miles per Gallon')
    plt.grid(True, alpha=0.3)
    
    # Save with high resolution for publication
    plt.savefig('mpg_vs_weight.png', dpi=300, bbox_inches='tight')
    
    # Save as PDF for vector graphics
    plt.savefig('mpg_vs_weight.pdf', bbox_inches='tight')
    
    plt.show()

def mathematical_annotations():
    """
    Add mathematical formulas and statistical information to plots.
    """
    mtcars = load_mtcars()
    
    X = mtcars[['wt']]
    y = mtcars['mpg']
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R-squared and p-value
    y_pred = model.predict(X)
    r_squared = model.score(X, y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(mtcars['wt'], mtcars['mpg'])
    
    plt.figure(figsize=(10, 8))
    sns.regplot(data=mtcars, x='wt', y='mpg', 
                scatter_kws={'alpha': 0.7}, line_kws={'color': 'red'})
    
    plt.text(4, 30, f'R² = {r_squared:.3f}', 
             color='red', fontweight='bold', fontsize=12)
    plt.text(4, 28, f'p < {p_value:.3f}', 
             color='red', fontweight='bold', fontsize=12)
    
    plt.title('MPG vs Weight with Statistical Information')
    plt.xlabel('Weight (1000 lbs)')
    plt.ylabel('Miles per Gallon')
    plt.grid(True, alpha=0.3)
    plt.show()

def practical_example_comprehensive_analysis():
    """
    Comprehensive data analysis example using iris dataset.
    """
    # Load data
    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target_names[iris.target]
    
    # Create multiple plots with mathematical insights
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Scatter plot with regression
    for species in iris_df['species'].unique():
        subset = iris_df[iris_df['species'] == species]
        axes[0].scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], 
                       alpha=0.7, label=species)
        # Add regression line for each species
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            subset['sepal length (cm)'], subset['sepal width (cm)'])
        x_range = np.array([subset['sepal length (cm)'].min(), subset['sepal length (cm)'].max()])
        axes[0].plot(x_range, slope * x_range + intercept, alpha=0.7)
    
    overall_corr = iris_df['sepal length (cm)'].corr(iris_df['sepal width (cm)'])
    axes[0].set_title(f'Sepal Length vs Width by Species\nOverall correlation: {overall_corr:.3f}')
    axes[0].set_xlabel('Sepal Length (cm)')
    axes[0].set_ylabel('Sepal Width (cm)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Box plot with mean
    sns.boxplot(data=iris_df, x='species', y='sepal length (cm)', ax=axes[1])
    means = iris_df.groupby('species')['sepal length (cm)'].mean()
    axes[1].scatter(range(len(means)), means, color='red', s=100, marker='D', zorder=5)
    axes[1].set_title('Sepal Length by Species\nRed diamond shows mean')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Density plot
    for species in iris_df['species'].unique():
        subset = iris_df[iris_df['species'] == species]
        sns.kdeplot(data=subset['sepal length (cm)'], ax=axes[2], label=species, alpha=0.7)
    
    axes[2].set_title('Sepal Length Density by Species')
    axes[2].set_xlabel('Sepal Length (cm)')
    axes[2].set_ylabel('Density')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def practical_example_time_series():
    """
    Time series visualization example with mathematical functions.
    """
    # Create time series data with mathematical functions
    np.random.seed(123)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
    trend = np.arange(1, 13)
    seasonal = np.sin(2 * np.pi * np.arange(1, 13) / 12) * 10
    noise = np.random.normal(0, 2, 12)
    values = trend + seasonal + noise
    
    ts_data = pd.DataFrame({
        'date': dates,
        'value': values,
        'trend': trend,
        'seasonal': seasonal
    })
    
    # Time series plot with components
    plt.figure(figsize=(12, 8))
    plt.plot(ts_data['date'], ts_data['value'], 'o-', label='Observed', linewidth=2)
    plt.plot(ts_data['date'], ts_data['trend'], '--', label='Trend', linewidth=2, color='red')
    plt.plot(ts_data['date'], ts_data['seasonal'], ':', label='Seasonal', linewidth=2, color='blue')
    
    plt.title('Time Series with Trend and Seasonal Components')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def practical_example_statistical_process_control():
    """
    Statistical process control chart example.
    """
    # Create control chart data
    np.random.seed(456)
    process_data = pd.DataFrame({
        'sample': range(1, 31),
        'measurement': np.concatenate([np.random.normal(100, 2, 20), 
                                      np.random.normal(105, 2, 10)])  # Process shift
    })
    
    # Calculate control limits
    mean_val = np.mean(process_data['measurement'][:20])
    sd_val = np.std(process_data['measurement'][:20], ddof=1)
    
    process_data['ucl'] = mean_val + 3 * sd_val
    process_data['lcl'] = mean_val - 3 * sd_val
    process_data['ucl_warning'] = mean_val + 2 * sd_val
    process_data['lcl_warning'] = mean_val - 2 * sd_val
    
    # Control chart
    plt.figure(figsize=(12, 8))
    plt.plot(process_data['sample'], process_data['measurement'], 'o-', color='blue', linewidth=1)
    plt.plot(process_data['sample'], process_data['ucl'], '--', color='red', linewidth=2)
    plt.plot(process_data['sample'], process_data['lcl'], '--', color='red', linewidth=2)
    plt.plot(process_data['sample'], process_data['ucl_warning'], ':', color='orange', linewidth=2)
    plt.plot(process_data['sample'], process_data['lcl_warning'], ':', color='orange', linewidth=2)
    plt.axhline(y=mean_val, color='green', linewidth=2)
    
    plt.title('Statistical Process Control Chart')
    plt.xlabel('Sample Number')
    plt.ylabel('Measurement')
    plt.text(0.02, 0.98, 'Green = Center line, Red = Control limits, Orange = Warning limits', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Example usage
    basic_plot_types()
    understanding_plot_parameters()
    customizing_plots()
    advanced_matplotlib_features()
    seaborn_setup()
    basic_seaborn_syntax()
    understanding_aesthetics()
    seaborn_scatter_plots()
    seaborn_histograms_density()
    seaborn_box_violin_plots()
    seaborn_bar_plots()
    seaborn_line_plots()
    seaborn_faceting()
    themes_and_customization()
    correlation_matrix()
    violin_plots_advanced()
    heatmaps_advanced()
    three_d_scatter_plots()
    qq_plots()
    residual_plots()
    distribution_comparison()
    interactive_visualizations()
    best_practices_color_choices()
    saving_plots()
    mathematical_annotations()
    practical_example_comprehensive_analysis()
    practical_example_time_series()
    practical_example_statistical_process_control() 