import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib

def _apply_common_plot_style():
    """Applies common styling to plots."""
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['figure.titlesize'] = 26

def _finalize_plot(fig, ax, title, output_filename, xlabel, ylabel):
    """Finalizes and saves the plot."""
    ax.set_xlabel(xlabel, weight='bold', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel(ylabel, weight='bold', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_title(title, weight='bold', fontsize=plt.rcParams['figure.titlesize'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', direction='out', length=6, width=1)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def visualize_fit_rates():
    """Creates a grouped bar chart for FIT rates across different models and configurations."""
    _apply_common_plot_style()
    
    # Define the data
    data = {
        'Model': ['ResNet-50', 'ResNet-50', 'ResNet-50', 'ResNet-50', 'ResNet-50', 'ResNet-50',
                  'ResNet-18', 'ResNet-18', 'ResNet-18', 'ResNet-18', 'ResNet-18', 'ResNet-18',
                  'LeNet-5', 'LeNet-5', 'LeNet-5', 'LeNet-5', 'LeNet-5', 'LeNet-5'],
        'Configuration': ['FP32', 'FP32 Activation Pruned', 'FP32 Magnitude Pruned', 
                         'W8A8', 'W8A8 Activation Pruned', 'W8A8 Magnitude Pruned',
                         'FP32', 'FP32 Activation Pruned', 'FP32 Magnitude Pruned', 
                         'W8A8', 'W8A8 Activation Pruned', 'W8A8 Magnitude Pruned',
                         'FP32', 'FP32 Activation Pruned', 'FP32 Magnitude Pruned', 
                         'W8A8', 'W8A8 Activation Pruned', 'W8A8 Magnitude Pruned'],
        'FIT_Rate': [12.67, 12.71, 12.84, 0.86, 0.89, 0.94,
                     12.92, 13.33, 13.45, 0.99, 1.01, 1.09,
                     18.54, 20.65, 21.62, 0.87, 0.88, 0.96]
    }
    
    df = pd.DataFrame(data)
    
    # Create pivot table for grouped bar chart
    pivot_df = df.pivot(index='Model', columns='Configuration', values='FIT_Rate')
    
    # Define colors for different configurations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Create grouped bar chart
    pivot_df.plot(kind='bar', ax=ax, width=0.8, color=colors, edgecolor='black')
    
    # Customize the plot
    ax.legend(title='Configuration', title_fontsize=22, fontsize=18, 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=14, rotation=0, padding=3)
    
    # Set x-axis labels rotation
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=plt.rcParams['xtick.labelsize'])
    
    _finalize_plot(fig, ax, 'FIT Rate Comparison Across Models and Configurations',
                   'fit_rate_comparison.png',
                   'Model', 'FIT Rate (%)')
    
    print("Generated FIT rate comparison visualization: fit_rate_comparison.png")

if __name__ == '__main__':
    visualize_fit_rates() 