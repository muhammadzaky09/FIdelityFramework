import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns # Add seaborn import
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib

def _apply_common_plot_style():
    """Applies common styling to plots."""
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 16

def _finalize_plot(fig, ax, title, output_filename, xlabel, ylabel):
    """Finalizes and saves the plot."""
    ax.set_xlabel(xlabel, weight='bold', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel(ylabel, weight='bold', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_title(title, weight='bold', fontsize=plt.rcParams['figure.titlesize'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', direction='out', length=6, width=1)
    ax.grid(True, linestyle='--', alpha=0.7) # Ensure grid is applied

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def _load_and_preprocess_data(csv_path):
    """Loads and preprocesses fault injection data from a CSV file."""
    df = pd.read_csv(csv_path)
    if df['classification_changed'].dtype == 'object':
        df['classification_changed'] = df['classification_changed'].apply(
            lambda x: 1 if str(x).lower() == 'true' else 0
        )
    df['original_confidence'] = pd.to_numeric(df['original_confidence'], errors='coerce')
    df['faulty_confidence'] = pd.to_numeric(df['faulty_confidence'], errors='coerce')
    df.dropna(subset=['original_confidence', 'faulty_confidence'], inplace=True)
    df['confidence_drop'] = df['original_confidence'] - df['faulty_confidence']
    return df

def _group_layers(df):
    """
    Groups ResNet layers into logical categories for better visualization.
    
    Args:
        df: DataFrame containing layer_name column
        
    Returns:
        DataFrame with an added layer_group column
    """
    df = df.copy()
    
    # Define grouping function
    def assign_group(layer_name):
        if not isinstance(layer_name, str):
            return "Other"
            
        if layer_name.startswith(('conv1', 'bn1', 'relu', 'maxpool')):
            return "Initial Layers"
        elif 'layer1.' in layer_name:
            return "Layer1 (Block 1)"
        elif 'layer2.' in layer_name:
            return "Layer2 (Block 2)"
        elif 'layer3.' in layer_name:
            return "Layer3 (Block 3)"
        elif 'layer4.' in layer_name:
            return "Layer4 (Block 4)"
        elif layer_name in ('fc', 'avgpool'):
            return "Final Layers"
        else:
            return "Other"
    
    df['layer_group'] = df['layer_name'].apply(assign_group)
    return df

def visualize_fault_injection(csv_path):
    _apply_common_plot_style()
    # Read CSV data
    df = pd.read_csv(csv_path)
    
    # Convert classification_changed to numeric if needed
    if df['classification_changed'].dtype == 'object':
        df['classification_changed'] = df['classification_changed'].apply(
            lambda x: 1 if str(x).lower() == 'true' else 0)
            
    # Ensure confidence columns are numeric
    df['original_confidence'] = pd.to_numeric(df['original_confidence'], errors='coerce')
    df['faulty_confidence'] = pd.to_numeric(df['faulty_confidence'], errors='coerce')
    df.dropna(subset=['original_confidence', 'faulty_confidence'], inplace=True) # Drop rows where conversion failed

    # Calculate confidence change
    df['confidence_drop'] = df['original_confidence'] - df['faulty_confidence']
    
    # Filter out RD fault models (if this is still desired, though not explicitly part of the new focus)
    # df = df[~df['fault_model'].str.contains('RD')] # Commenting out as RD models are not in the new focus
    
    # 1. Vulnerability by Bit Position
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    bit_vuln = df.groupby('bit_position')['classification_changed'].mean() * 100
    
    # Use a seaborn color palette
    # colors = sns.color_palette("viridis", n_colors=len(bit_vuln) if len(bit_vuln) > 0 else 1)

    ax1.bar(bit_vuln.index.astype(str), bit_vuln.values, 
            color='skyblue', # Use first color or default
            edgecolor='black', width=0.7)
    
    # Add value labels on bars
    # for i, v in enumerate(bit_vuln.values):
    #     ax1.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
    
    _finalize_plot(fig1, ax1, 'Vulnerability by Bit Position', 
                   'vulnerability_by_bit_position.png', 
                   'Bit Position', 'Misclassification Rate (%)')
    
    # 2. Vulnerability by Layer
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    layer_vuln = df.groupby('layer_name')['classification_changed'].mean() * 100
    layer_vuln = layer_vuln.sort_values(ascending=False) # Optional: sort for better viz

    # colors_layer = sns.color_palette("mako", n_colors=len(layer_vuln) if len(layer_vuln) > 0 else 1)

    ax2.bar(layer_vuln.index, layer_vuln.values, 
            color='lightgreen', # Use first color or default
            edgecolor='black', width=0.7)
    
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=plt.rcParams['xtick.labelsize'])

    # Add value labels on bars
    # for i, v in enumerate(layer_vuln.values):
    #     ax2.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10)
        
    _finalize_plot(fig2, ax2, 'Vulnerability by Layer', 
                   'vulnerability_by_layer.png',
                   'Layer Name', 'Misclassification Rate (%)')

    # 3. Average Confidence Drop by Bit Position
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    bit_conf_drop = df.groupby('bit_position')['confidence_drop'].mean()
    
    # colors_conf_bit = sns.color_palette("crest", n_colors=len(bit_conf_drop) if len(bit_conf_drop) > 0 else 1)

    ax3.bar(bit_conf_drop.index.astype(str), bit_conf_drop.values,
            color='lightcoral',
            edgecolor='black', width=0.7)

    # for i, v in enumerate(bit_conf_drop.values):
    #     ax3.text(i, v + (0.01 * bit_conf_drop.values.max() if bit_conf_drop.values.max() else 0.01), # Dynamic offset
    #              f'{v:.3f}', ha='center', fontsize=10)

    _finalize_plot(fig3, ax3, 'Average Confidence Drop by Bit Position',
                   'confidence_drop_by_bit_position.png',
                   'Bit Position', 'Average Confidence Drop')

    # 4. Average Confidence Drop by Layer
    fig4, ax4 = plt.subplots(figsize=(16, 8))
    layer_conf_drop = df.groupby('layer_name')['confidence_drop'].mean()
    layer_conf_drop = layer_conf_drop.sort_values(ascending=True) # Optional: sort for better viz

    # colors_conf_layer = sns.color_palette("rocket", n_colors=len(layer_conf_drop) if len(layer_conf_drop) > 0 else 1)
    
    ax4.bar(layer_conf_drop.index, layer_conf_drop.values,
            color='gold',
            edgecolor='black', width=0.7)
            
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=plt.rcParams['xtick.labelsize'])

    # for i, v in enumerate(layer_conf_drop.values):
    #     ax4.text(i, v + (0.01 * layer_conf_drop.values.max() if layer_conf_drop.values.max() else 0.01), # Dynamic offset
    #              f'{v:.3f}', ha='center', fontsize=10)

    _finalize_plot(fig4, ax4, 'Average Confidence Drop by Layer',
                   'confidence_drop_by_layer.png',
                   'Layer Name', 'Average Confidence Drop')
    
    print(f"Generated 4 visualization plots with new styling.")
    return "Visualization complete"

def visualize_comparison_fault_injection(original_csv_path, pruned_act_csv_path, pruned_mag_csv_path, group_layers=True):
    """
    Visualizes and compares fault injection results from an original model and two pruned versions.
    Generates grouped bar charts for misclassification rate and confidence drop,
    by bit position and by layer.
    
    Args:
        original_csv_path: Path to CSV with original model results
        pruned_act_csv_path: Path to CSV with activation-pruned model results
        pruned_mag_csv_path: Path to CSV with magnitude-pruned model results
        group_layers: If True, groups layers into categories for better visualization
    """
    _apply_common_plot_style()

    try:
        df_orig = _load_and_preprocess_data(original_csv_path)
        df_act = _load_and_preprocess_data(pruned_act_csv_path)
        df_mag = _load_and_preprocess_data(pruned_mag_csv_path)
    except Exception as e:
        print(f"Error loading or preprocessing CSV files: {e}")
        return

    if group_layers:
        df_orig = _group_layers(df_orig)
        df_act = _group_layers(df_act)
        df_mag = _group_layers(df_mag)

    model_labels = ['Original', 'Pruned (Activation)', 'Pruned (Magnitude)']
    colors = ['skyblue', 'lightgreen', 'lightcoral']

    # 1. Comparison: Vulnerability by Bit Position
    bit_vuln_orig = df_orig.groupby('bit_position')['classification_changed'].mean() * 100
    bit_vuln_act = df_act.groupby('bit_position')['classification_changed'].mean() * 100
    bit_vuln_mag = df_mag.groupby('bit_position')['classification_changed'].mean() * 100

    all_bit_positions = sorted(list(set(bit_vuln_orig.index) | set(bit_vuln_act.index) | set(bit_vuln_mag.index)))
    
    plot_data_bit_vuln = pd.DataFrame({
        model_labels[0]: bit_vuln_orig,
        model_labels[1]: bit_vuln_act,
        model_labels[2]: bit_vuln_mag
    }).reindex(all_bit_positions).fillna(0)

    if not plot_data_bit_vuln.empty:
        fig1, ax1 = plt.subplots(figsize=(18, 10))
        plot_data_bit_vuln.plot(kind='bar', ax=ax1, width=0.8, color=colors, edgecolor='black')
        ax1.legend(title='Model Type')
        _finalize_plot(fig1, ax1, 'Comparison: Vulnerability by Bit Position',
                       'comparison_vulnerability_by_bit_position.png',
                       'Bit Position', 'Misclassification Rate (%)')
    else:
        print("No data available for 'Comparison: Vulnerability by Bit Position' plot.")

    # 2. Comparison: Vulnerability by Layer (using layer_group if enabled)
    group_column = 'layer_group' if group_layers else 'layer_name'
    
    layer_vuln_orig = df_orig.groupby(group_column)['classification_changed'].mean() * 100
    layer_vuln_act = df_act.groupby(group_column)['classification_changed'].mean() * 100
    layer_vuln_mag = df_mag.groupby(group_column)['classification_changed'].mean() * 100

    all_layers = sorted(list(set(layer_vuln_orig.index) | set(layer_vuln_act.index) | set(layer_vuln_mag.index)))
    
    plot_data_layer_vuln = pd.DataFrame({
        model_labels[0]: layer_vuln_orig,
        model_labels[1]: layer_vuln_act,
        model_labels[2]: layer_vuln_mag
    }).reindex(all_layers).fillna(0)

    if not plot_data_layer_vuln.empty:
        fig2, ax2 = plt.subplots(figsize=(20, 12))
        plot_data_layer_vuln.plot(kind='bar', ax=ax2, width=0.8, color=colors, edgecolor='black')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=plt.rcParams['xtick.labelsize'])
        ax2.legend(title='Model Type')
        title_suffix = "by Layer Group" if group_layers else "by Layer"
        _finalize_plot(fig2, ax2, f'Comparison: Vulnerability {title_suffix}',
                       'comparison_vulnerability_by_layer.png',
                       'Layer' if group_layers else 'Layer Name', 'Misclassification Rate (%)')
    else:
        print("No data available for 'Comparison: Vulnerability by Layer' plot.")

    # 3. Comparison: Average Confidence Drop by Bit Position
    bit_conf_drop_orig = df_orig.groupby('bit_position')['confidence_drop'].mean()
    bit_conf_drop_act = df_act.groupby('bit_position')['confidence_drop'].mean()
    bit_conf_drop_mag = df_mag.groupby('bit_position')['confidence_drop'].mean()
    
    plot_data_bit_conf = pd.DataFrame({
        model_labels[0]: bit_conf_drop_orig,
        model_labels[1]: bit_conf_drop_act,
        model_labels[2]: bit_conf_drop_mag
    }).reindex(all_bit_positions).fillna(0)

    if not plot_data_bit_conf.empty:
        fig3, ax3 = plt.subplots(figsize=(18, 10))
        plot_data_bit_conf.plot(kind='bar', ax=ax3, width=0.8, color=colors, edgecolor='black')
        ax3.legend(title='Model Type')
        _finalize_plot(fig3, ax3, 'Comparison: Avg. Confidence Drop by Bit Position',
                       'comparison_confidence_drop_by_bit_position.png',
                       'Bit Position', 'Average Confidence Drop')
    else:
        print("No data available for 'Comparison: Avg. Confidence Drop by Bit Position' plot.")

    # 4. Comparison: Average Confidence Drop by Layer
    layer_conf_drop_orig = df_orig.groupby(group_column)['confidence_drop'].mean()
    layer_conf_drop_act = df_act.groupby(group_column)['confidence_drop'].mean()
    layer_conf_drop_mag = df_mag.groupby(group_column)['confidence_drop'].mean()

    plot_data_layer_conf = pd.DataFrame({
        model_labels[0]: layer_conf_drop_orig,
        model_labels[1]: layer_conf_drop_act,
        model_labels[2]: layer_conf_drop_mag
    }).reindex(all_layers).fillna(0)

    if not plot_data_layer_conf.empty:
        fig4, ax4 = plt.subplots(figsize=(20, 12))
        plot_data_layer_conf.plot(kind='bar', ax=ax4, width=0.8, color=colors, edgecolor='black')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=plt.rcParams['xtick.labelsize'])
        ax4.legend(title='Model Type')
        title_suffix = "by Layer Group" if group_layers else "by Layer"
        _finalize_plot(fig4, ax4, f'Comparison: Avg. Confidence Drop {title_suffix}',
                       'comparison_confidence_drop_by_layer.png',
                       'Layer' if group_layers else 'Layer Name', 'Average Confidence Drop')
    else:
        print("No data available for 'Comparison: Avg. Confidence Drop by Layer' plot.")
        
    print(f"Generated comparison visualization plots (up to 4) {'with grouped layers' if group_layers else ''}.")

# To use:
# Ensure the CSV path is correct for your setup.
# Example:
# visualize_fault_injection('pt_fi/cnn/results/ResNet18/fp32/resnet18_fp32_fault_injection_results.csv')
# Make sure you have a CSV file at the specified path. For testing, you might use:
if __name__ == '__main__':
    # Create a dummy CSV for testing if the actual one isn't available
    # This is just for making the script runnable for a quick check.
    # Replace with your actual CSV path.
    
    # --- Original Single Model Visualization (Example) ---
    dummy_single_model_csv_path = 'dummy_fault_injection_results.csv'
    actual_single_model_csv_path = 'pt_fi/cnn/results/ResNet18/fp32/resnet18_fp32_fault_injection_results.csv' # Example path
    
    run_single_model_visualization = False # Set to True to run the original visualization part

    if run_single_model_visualization:
        try:
            pd.read_csv(actual_single_model_csv_path) 
            print(f"Attempting to use actual CSV for single model viz: {actual_single_model_csv_path}")
            visualize_fault_injection(actual_single_model_csv_path)
        except FileNotFoundError:
            print(f"Actual CSV for single model viz not found at {actual_single_model_csv_path}. Creating and using a dummy CSV.")
            data = {
                'model': ['resnet18'] * 20,
                'precision': ['fp32'] * 20,
                'layer_name': ['conv1', 'layer1.0.conv1', 'conv1', 'layer1.0.conv1'] * 5,
                'layer_type': ['Conv2d'] * 20,
                'fault_model': ['INPUT'] * 20,
                'bit_position': list(range(10)) + list(range(10)),
                'experiment_id': list(range(20)),
                'image_label': [0]*20,
                'original_class': [0]*10 + [1]*10,
                'original_confidence': np.random.rand(20) * 0.5 + 0.5,
                'faulty_class': [1]*5 + [0]*5 + [0]*5 + [1]*5,
                'faulty_confidence': np.random.rand(20) * 0.5,
                'classification_changed': [True]*5 + [False]*5 + [True]*5 + [False]*5,
                'injection_position': ['(0,0,0,0)']*20
            }
            dummy_df = pd.DataFrame(data)
            dummy_df.to_csv(dummy_single_model_csv_path, index=False)
            visualize_fault_injection(dummy_single_model_csv_path)
            print(f"Dummy CSV used for single model viz: {dummy_single_model_csv_path}")

    # --- Comparison Visualization ---
    ORIGINAL_CSV = 'pt_fi/cnn/results/ResNet50/fp32/resnet50_fp32_fault_injection_results.csv'
    PRUNED_ACT_CSV = 'pt_fi/cnn/results/ResNet50ActivationStep10/fp32/resnet50_fp32_fault_injection_results.csv'
    PRUNED_MAG_CSV = 'pt_fi/cnn/results/ResNet50MagnitudeStep10/fp32/resnet50_fp32_fault_injection_results.csv'



    def create_dummy_comparison_csv(file_path, model_name_suffix):
        data = {
            'model': [f'resnet18_{model_name_suffix}'] * 40, # More data points
            'precision': ['fp32'] * 40,
            'layer_name': (['conv1', 'layer1.0.conv1', 'fc', 'layer2.0.conv1'] * 10),
            'layer_type': ['Conv2d', 'Conv2d', 'Linear', 'Conv2d'] * 10,
            'fault_model': ['INPUT'] * 40,
            'bit_position': np.random.choice(range(16), 40), # Realistic bit positions 0-15
            'experiment_id': list(range(40)),
            'image_label': np.random.randint(0, 5, 40),
            'original_class': np.random.randint(0, 5, 40),
            'original_confidence': np.random.uniform(0.6, 1.0, 40),
            'faulty_class': np.random.randint(0, 5, 40),
            'faulty_confidence': np.random.uniform(0.0, 0.7, 40),
            'classification_changed': np.random.choice([True, False], 40, p=[0.3, 0.7]), # 30% misclassification
            'injection_position': ['(0,0,0,0)']*40
        }
        # Simulate some differences for pruned models, e.g. slightly different confidence or misclassification rates
        if 'pruned' in model_name_suffix:
            data['original_confidence'] = np.random.uniform(0.5, 0.9, 40) # Slightly lower original confidence for pruned
            data['classification_changed'] = np.random.choice([True, False], 40, p=[0.35, 0.65]) # Potentially higher misclassification
            if 'act' in model_name_suffix: # Activation pruning might remove some layers
                 data['layer_name'] = (['conv1', 'fc', 'layer2.0.conv1'] * (40//3 +1))[:40]


        dummy_df = pd.DataFrame(data)
        dummy_df.to_csv(file_path, index=False)
        print(f"Created dummy CSV for comparison: {file_path}")

    try:
        # Check if all three actual files exist by trying to load them
        # This check is implicitly done by _load_and_preprocess_data if we call directly
        print(f"Attempting to use actual CSVs for comparison:")
        print(f"  Original: {ORIGINAL_CSV}")
        print(f"  Pruned (Activation): {PRUNED_ACT_CSV}")
        print(f"  Pruned (Magnitude): {PRUNED_MAG_CSV}")
        # Test read to trigger FileNotFoundError early if any is missing
        pd.read_csv(ORIGINAL_CSV, nrows=1)
        pd.read_csv(PRUNED_ACT_CSV, nrows=1)
        pd.read_csv(PRUNED_MAG_CSV, nrows=1)
        visualize_comparison_fault_injection(ORIGINAL_CSV, PRUNED_ACT_CSV, PRUNED_MAG_CSV, group_layers=True)
    except Exception as e:
        print(f"An error occurred during comparison visualization: {e}")
        print("If using actual files, ensure they are valid and paths are correct.")
        print("If dummy files were generated, there might be an issue in the dummy data generation or plotting logic itself.")

