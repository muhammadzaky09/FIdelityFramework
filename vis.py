import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib

def visualize_fault_injection(csv_path):
    # Read CSV data
    df = pd.read_csv(csv_path)
    
    # Convert classification_changed to numeric if needed
    if df['classification_changed'].dtype == 'object':
        df['classification_changed'] = df['classification_changed'].apply(
            lambda x: 1 if str(x).lower() == 'true' else 0)
    
    # Filter out RD fault models
    df = df[~df['fault_model'].str.contains('RD')]
    
    # 1. Vulnerability by Bit Position
    plt.figure(figsize=(14, 8))
    bit_vuln = df.groupby('bit_position')['classification_changed'].mean() * 100
    
    plt.bar(bit_vuln.index, bit_vuln.values, 
            color='steelblue', edgecolor='black', width=0.7)
    
    plt.xlabel('Bit Position', fontsize=14)
    plt.ylabel('Misclassification Rate (%)', fontsize=14)
    plt.title('Vulnerability by Bit Position', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(range(len(bit_vuln)), fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(bit_vuln.values):
        plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('bit_position_vulnerability.png', dpi=300)
    
    # 2. Vulnerability by Fault Model
    plt.figure(figsize=(10, 8))
    model_vuln = df.groupby('fault_model')['classification_changed'].mean() * 100
    
    plt.bar(model_vuln.index, model_vuln.values, 
            color='indianred', edgecolor='black', width=0.6)
    
    plt.xlabel('Fault Model', fontsize=14)
    plt.ylabel('Misclassification Rate (%)', fontsize=14)
    plt.title('Vulnerability by Fault Model', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(model_vuln.values):
        plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('fault_model_vulnerability.png', dpi=300)
    
    # 3. Vulnerability by Layer
    plt.figure(figsize=(16, 8))
    layer_vuln = df.groupby('layer_name')['classification_changed'].mean() * 100
    
    plt.bar(layer_vuln.index, layer_vuln.values, 
            color='forestgreen', edgecolor='black', width=0.7)
    
    plt.xlabel('Layer Name', fontsize=14)
    plt.ylabel('Misclassification Rate (%)', fontsize=14)
    plt.title('Vulnerability by Layer', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(layer_vuln.values):
        plt.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('layer_vulnerability.png', dpi=300)
    
    # 4. Layer-Bit Position Heatmap
    plt.figure(figsize=(18, 10))
    heatmap_data = pd.pivot_table(
        df, values='classification_changed', 
        index='layer_name', columns='bit_position', aggfunc='mean'
    ) * 100
    
    # Plot heatmap using pcolormesh for better control
    plt.pcolormesh(heatmap_data, cmap='YlOrRd')
    plt.colorbar(label='Misclassification Rate (%)')
    
    # Configure axes
    plt.xticks(np.arange(0.5, len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(np.arange(0.5, len(heatmap_data.index)), heatmap_data.index)
    plt.xlabel('Bit Position', fontsize=14)
    plt.ylabel('Layer Name', fontsize=14)
    plt.title('Vulnerability Heatmap: Layer vs Bit Position', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('layer_bit_heatmap.png', dpi=300)
    
    print(f"Generated 4 visualization plots")
    return "Visualization complete"

# To use:
visualize_fault_injection('pt_fi/cnn/results/ResNet18/fp32/resnet18_fp32_fault_injection_results.csv')