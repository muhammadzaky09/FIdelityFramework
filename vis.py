import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns # Add seaborn import
import re # Import re for parsing layer names
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
    ax.grid(True, linestyle='--', alpha=0.7) # Ensure grid is applied

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def _get_layer_sort_key(layer_name):
    """Generates a sort key for layer names like 'Conv_X' or 'Gemm_X'."""
    if isinstance(layer_name, str):
        match = re.match(r"([a-zA-Z]+)_(\d+)", layer_name)
        if match:
            prefix = match.group(1)
            number = int(match.group(2))
            return (prefix, number)
    return (str(layer_name), 0) # Fallback for unexpected names

def _get_layer_number(layer_name):
    """Extracts the layer number from layer names like 'Conv_X' or 'Gemm_X'."""
    if isinstance(layer_name, str):
        # Try to match the pattern like 'Conv_5', 'Gemm_10', etc.
        match = re.match(r"([a-zA-Z]+)_(\d+)", layer_name)
        if match:
            return int(match.group(2))
        
        # Try to extract any number from the layer name as fallback
        numbers = re.findall(r'\d+', layer_name)
        if numbers:
            return int(numbers[-1])  # Use the last number found
    
    # If no number found, return a default value
    return 999  # Put unmatched layers at the end

def _convert_pytorch_to_onnx_names(df):
    """Convert PyTorch layer names to simple sequential ONNX-style names."""
    df_result = df.copy()
    
    # Filter to only include conv and fc/linear layers
    essential_mask = (
        df_result['layer_name'].str.contains('conv', case=False, na=False) |
        df_result['layer_name'].str.contains('layer', case=False, na=False) |
        df_result['layer_name'].str.contains('fc', case=False, na=False) |
        df_result['layer_name'].str.contains('linear', case=False, na=False)
    )
    
    df_result = df_result[essential_mask].copy()
    
    if df_result.empty:
        print("Warning: No essential layers found after filtering")
        return df_result
    
    # Get unique layer names
    unique_layers = df_result['layer_name'].unique()
    
    # Separate conv and fc layers
    conv_layers = []
    fc_layers = []
    
    for layer in unique_layers:
        if 'conv' in layer.lower() or ('layer' in layer.lower() and ('conv' in layer.lower() or 'shortcut' in layer.lower() or 'downsample' in layer.lower())):
            # Include ALL conv layers including shortcut/downsample convs
            conv_layers.append(layer)
        elif 'fc' in layer.lower() or 'linear' in layer.lower():
            fc_layers.append(layer)
    
    # Simple sorting: conv1 first, then layer blocks in order, including shortcuts
    def simple_sort_key(layer_name):
        layer_lower = layer_name.lower()
        
        # conv1 gets position 0
        if layer_lower == 'conv1':
            return (0, 0, 0, 0)
        
        # layer blocks get positions based on layer number, block number, conv number
        if 'layer' in layer_lower:
            parts = layer_name.split('.')
            if len(parts) >= 2:
                # Extract layer1, layer2, etc.
                layer_match = re.search(r'layer(\d+)', parts[0])
                layer_num = int(layer_match.group(1)) if layer_match else 999
                
                # Extract block number (0, 1, 2, etc.)
                block_num = int(parts[1]) if parts[1].isdigit() else 999
                
                # Handle different types within the block
                if len(parts) >= 3:
                    if 'conv' in parts[2]:
                        # Regular conv layers (conv1, conv2, conv3)
                        conv_match = re.search(r'conv(\d+)', parts[2])
                        conv_num = int(conv_match.group(1)) if conv_match else 999
                        return (layer_num, block_num, 0, conv_num)  # Regular convs first
                    elif 'shortcut' in parts[2] or 'downsample' in parts[2]:
                        # Shortcut/downsample layers come after regular convs in the same block
                        if len(parts) >= 4 and 'conv' in parts[3]:
                            # shortcut.0 or downsample.0 (the conv layer)
                            return (layer_num, block_num, 1, 0)  # Shortcuts after regular convs
                        else:
                            return (layer_num, block_num, 1, 999)  # Other shortcut components
                else:
                    return (layer_num, block_num, 999, 999)
        
        return (999, 999, 999, 999)  # Fallback
    
    # Sort conv layers
    conv_layers_sorted = sorted(conv_layers, key=simple_sort_key)
    
    # Debug: Show what layers we found
    print(f"Debug: Found conv layers including shortcuts:")
    for i, layer in enumerate(conv_layers_sorted[:10]):  # Show first 10
        print(f"  {i}: {layer}")
    if len(conv_layers_sorted) > 10:
        print(f"  ... and {len(conv_layers_sorted) - 10} more")
        print(f"  Last few:")
        for i, layer in enumerate(conv_layers_sorted[-3:], len(conv_layers_sorted)-3):
            print(f"  {i}: {layer}")
    
    # Create simple sequential mapping
    name_mapping = {}
    
    # Map conv layers to Conv_0, Conv_1, Conv_2, ...
    for i, layer_name in enumerate(conv_layers_sorted):
        name_mapping[layer_name] = f"Conv_{i}"
    
    # Map FC layers to Gemm_0, Gemm_1, ...
    for i, layer_name in enumerate(fc_layers):
        name_mapping[layer_name] = f"Gemm_{i}"
    
    print(f"Debug: INCLUDING shortcuts - {len(conv_layers_sorted)} conv layers → Conv_0 to Conv_{len(conv_layers_sorted)-1}")
    if fc_layers:
        print(f"Debug: {len(fc_layers)} FC layers → Gemm_0 to Gemm_{len(fc_layers)-1}")
    
    # Apply the mapping
    df_result['layer_name'] = df_result['layer_name'].map(name_mapping).fillna(df_result['layer_name'])
    
    return df_result

def _group_layers_by_range(df, group_size=5):
    """Groups layers into ranges and aggregates the data - NO OTHER CATEGORIES."""
    # Convert PyTorch names to ONNX-style names first
    df = _convert_pytorch_to_onnx_names(df)
    
    # AGGRESSIVE FILTERING: Only keep layers that start with Conv_ or Gemm_
    df = df[df['layer_name'].str.startswith(('Conv_', 'Gemm_'))].copy()
    
    if df.empty:
        print("Warning: No Conv_ or Gemm_ layers found after filtering")
        return df
    
    # Extract layer numbers using FIdelity-Q's simple approach
    df['layer_number'] = df['layer_name'].apply(_get_layer_number)
    
    # Use FIdelity-Q's simple layer detection - ONLY Conv and Gemm
    df_conv = df[df['layer_name'].str.startswith('Conv_')].copy()
    df_gemm = df[df['layer_name'].str.startswith('Gemm_')].copy()
    
    result_dfs = []
    
    # Process Conv layers - exactly like FIdelity-Q
    if not df_conv.empty:
        # Get the actual layer numbers from Conv layer names
        conv_numbers = []
        for name in df_conv['layer_name'].unique():
            match = re.findall(r'\d+', name)
            if match:
                conv_numbers.append(int(match[0]))
        
        if conv_numbers:
            conv_numbers = sorted(conv_numbers)
            min_conv = min(conv_numbers)
            max_conv = max(conv_numbers)
            
            print(f"Debug: Conv layers found: {min_conv} to {max_conv} (total: {len(conv_numbers)} unique)")
            
            # Create groups based on actual layer numbers - exactly like FIdelity-Q
            for i, name in enumerate(df_conv['layer_name'].unique()):
                layer_num = int(re.findall(r'\d+', name)[0])
                # Calculate which group this layer belongs to
                group_start = (layer_num // group_size) * group_size
                group_end = group_start + group_size - 1
                
                # Make sure we don't go beyond the actual max layer number
                if group_end > max_conv:
                    group_end = max_conv
                
                if group_start == group_end:
                    range_name = f"Conv_{group_start}"
                else:
                    range_name = f"Conv_{group_start}-{group_end}"
                
                df_conv.loc[df_conv['layer_name'] == name, 'layer_range'] = range_name
        
        result_dfs.append(df_conv)
    
    # Process Gemm layers - exactly like FIdelity-Q
    if not df_gemm.empty:
        # Get the actual layer numbers from Gemm layer names
        gemm_numbers = []
        for name in df_gemm['layer_name'].unique():
            match = re.findall(r'\d+', name)
            if match:
                gemm_numbers.append(int(match[0]))
        
        if gemm_numbers:
            gemm_numbers = sorted(gemm_numbers)
            
            # Create groups for Gemm layers
            for i, name in enumerate(df_gemm['layer_name'].unique()):
                layer_num = int(re.findall(r'\d+', name)[0])
                # Calculate which group this layer belongs to
                group_start = (layer_num // group_size) * group_size
                group_end = group_start + group_size - 1
                
                # Make sure we don't go beyond the actual max layer number
                max_gemm = max(gemm_numbers)
                if group_end > max_gemm:
                    group_end = max_gemm
                
                if group_start == group_end:
                    range_name = f"Gemm_{group_start}"
                else:
                    range_name = f"Gemm_{group_start}-{group_end}"
                
                df_gemm.loc[df_gemm['layer_name'] == name, 'layer_range'] = range_name
        
        result_dfs.append(df_gemm)
    
    # NO OTHER LAYERS - PERIOD!
    
    # Combine all dataframes
    if result_dfs:
        result_df = pd.concat(result_dfs, ignore_index=True)
    else:
        result_df = pd.DataFrame()  # Return empty DataFrame if nothing found
    
    print(f"Debug: Final result - {len(result_df)} entries, NO OTHER CATEGORIES")
    
    return result_df

def _sort_layer_ranges(layer_ranges):
    """Sort layer ranges - ONLY Conv and Gemm, NO OTHER categories."""
    def sort_key(x):
        # Handle Conv layers
        if x.startswith('Conv_'):
            try:
                if '-' in x:
                    start_num = int(x.split('_')[1].split('-')[0])
                else:
                    start_num = int(x.split('_')[1])
                return (1, start_num)  # Conv layers first
            except (ValueError, IndexError):
                return (1, 999)
        
        # Handle Gemm layers
        elif x.startswith('Gemm_'):
            try:
                if '-' in x:
                    start_num = int(x.split('_')[1].split('-')[0])
                else:
                    start_num = int(x.split('_')[1])
                return (2, start_num)  # Gemm layers second
            except (ValueError, IndexError):
                return (2, 999)
        
        # Should never reach here with the new filtering
        else:
            return (999, 999)  # This should never happen now
    
    return sorted(layer_ranges, key=sort_key)

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
    
    # Ensure bit_position is integer
    df['bit_position'] = pd.to_numeric(df['bit_position'], errors='coerce')
    df.dropna(subset=['bit_position'], inplace=True)
    df['bit_position'] = df['bit_position'].astype(int)
    
    df['confidence_drop'] = df['original_confidence'] - df['faulty_confidence']
    # Removed RD and RD_BITFLIP filtering to match FIdelity-Q behavior
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
    
    # Removed RD and RD_BITFLIP filtering to match FIdelity-Q behavior
    
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
    df_grouped = _group_layers_by_range(df.copy(), group_size=5)
    layer_vuln = df_grouped.groupby('layer_range')['classification_changed'].mean() * 100
    layer_vuln = layer_vuln.reindex(_sort_layer_ranges(layer_vuln.index))

    # colors_layer = sns.color_palette("mako", n_colors=len(layer_vuln) if len(layer_vuln) > 0 else 1)

    ax2.bar(layer_vuln.index, layer_vuln.values, 
            color='lightgreen', # Use first color or default
            edgecolor='black', width=0.7)
    
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=plt.rcParams['xtick.labelsize'])

    # Add value labels on bars
    # for i, v in enumerate(layer_vuln.values):
    #     ax2.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10)
        
    _finalize_plot(fig2, ax2, 'Vulnerability by Layer Range', 
                   'vulnerability_by_layer.png',
                   'Layer Range', 'Misclassification Rate (%)')

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
    layer_conf_drop = df_grouped.groupby('layer_range')['confidence_drop'].mean()
    layer_conf_drop = layer_conf_drop.reindex(_sort_layer_ranges(layer_conf_drop.index))

    # colors_conf_layer = sns.color_palette("rocket", n_colors=len(layer_conf_drop) if len(layer_conf_drop) > 0 else 1)
    
    ax4.bar(layer_conf_drop.index, layer_conf_drop.values,
            color='gold',
            edgecolor='black', width=0.7)
            
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=plt.rcParams['xtick.labelsize'])

    # for i, v in enumerate(layer_conf_drop.values):
    #     ax4.text(i, v + (0.01 * layer_conf_drop.values.max() if layer_conf_drop.values.max() else 0.01), # Dynamic offset
    #              f'{v:.3f}', ha='center', fontsize=10)

    _finalize_plot(fig4, ax4, 'Average Confidence Drop by Layer Range',
                   'confidence_drop_by_layer.png',
                   'Layer Range', 'Average Confidence Drop')
    
    print(f"Generated 4 visualization plots with new styling.")
    return "Visualization complete"

def visualize_comparison_fault_injection(original_csv_path, pruned_act_csv_path, pruned_mag_csv_path):
    """
    Visualizes and compares fault injection results from an original model and two pruned versions.
    Generates grouped bar charts for misclassification rate and confidence drop,
    by bit position and by layer.
    
    Args:
        original_csv_path: Path to CSV with original model results
        pruned_act_csv_path: Path to CSV with activation-pruned model results
        pruned_mag_csv_path: Path to CSV with magnitude-pruned model results
    """
    _apply_common_plot_style()

    try:
        df_orig = _load_and_preprocess_data(original_csv_path)
        df_act = _load_and_preprocess_data(pruned_act_csv_path)
        df_mag = _load_and_preprocess_data(pruned_mag_csv_path)
    except Exception as e:
        print(f"Error loading or preprocessing CSV files: {e}")
        return

    # Group layers by ranges for better visualization
    df_orig_grouped = _group_layers_by_range(df_orig.copy(), group_size=5)
    df_act_grouped = _group_layers_by_range(df_act.copy(), group_size=5)
    df_mag_grouped = _group_layers_by_range(df_mag.copy(), group_size=5)

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
        ax1.legend(title='Model Type', title_fontsize=22, fontsize=20)
        _finalize_plot(fig1, ax1, 'Comparison: Vulnerability by Bit Position',
                       'comparison_vulnerability_by_bit_position.png',
                       'Bit Position', 'Misclassification Rate (%)')
    else:
        print("No data available for 'Comparison: Vulnerability by Bit Position' plot.")

    # 2. Comparison: Vulnerability by Layer Range
    layer_vuln_orig = df_orig_grouped.groupby('layer_range')['classification_changed'].mean() * 100
    layer_vuln_act = df_act_grouped.groupby('layer_range')['classification_changed'].mean() * 100
    layer_vuln_mag = df_mag_grouped.groupby('layer_range')['classification_changed'].mean() * 100

    all_layer_ranges = _sort_layer_ranges(list(set(layer_vuln_orig.index) | set(layer_vuln_act.index) | set(layer_vuln_mag.index)))
    
    plot_data_layer_vuln = pd.DataFrame({
        model_labels[0]: layer_vuln_orig,
        model_labels[1]: layer_vuln_act,
        model_labels[2]: layer_vuln_mag
    }).reindex(all_layer_ranges).fillna(0)

    if not plot_data_layer_vuln.empty:
        fig2, ax2 = plt.subplots(figsize=(20, 12))
        plot_data_layer_vuln.plot(kind='bar', ax=ax2, width=0.8, color=colors, edgecolor='black')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=plt.rcParams['xtick.labelsize'])
        ax2.legend(title='Model Type', title_fontsize=22, fontsize=20)
        _finalize_plot(fig2, ax2, 'Comparison: Vulnerability by Layer Range',
                       'comparison_vulnerability_by_layer.png',
                       'Layer Range', 'Misclassification Rate (%)')
    else:
        print("No data available for 'Comparison: Vulnerability by Layer Range' plot.")

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
        ax3.legend(title='Model Type', title_fontsize=22, fontsize=20)
        _finalize_plot(fig3, ax3, 'Comparison: Avg. Confidence Drop by Bit Position',
                       'comparison_confidence_drop_by_bit_position.png',
                       'Bit Position', 'Average Confidence Drop')
    else:
        print("No data available for 'Comparison: Avg. Confidence Drop by Bit Position' plot.")

    # 4. Comparison: Average Confidence Drop by Layer Range
    layer_conf_drop_orig = df_orig_grouped.groupby('layer_range')['confidence_drop'].mean()
    layer_conf_drop_act = df_act_grouped.groupby('layer_range')['confidence_drop'].mean()
    layer_conf_drop_mag = df_mag_grouped.groupby('layer_range')['confidence_drop'].mean()

    plot_data_layer_conf = pd.DataFrame({
        model_labels[0]: layer_conf_drop_orig,
        model_labels[1]: layer_conf_drop_act,
        model_labels[2]: layer_conf_drop_mag
    }).reindex(all_layer_ranges).fillna(0)

    if not plot_data_layer_conf.empty:
        fig4, ax4 = plt.subplots(figsize=(20, 12))
        plot_data_layer_conf.plot(kind='bar', ax=ax4, width=0.8, color=colors, edgecolor='black')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=plt.rcParams['xtick.labelsize'])
        ax4.legend(title='Model Type', title_fontsize=22, fontsize=20)
        _finalize_plot(fig4, ax4, 'Comparison: Avg. Confidence Drop by Layer Range',
                       'comparison_confidence_drop_by_layer.png',
                       'Layer Range', 'Average Confidence Drop')
    else:
        print("No data available for 'Comparison: Avg. Confidence Drop by Layer Range' plot.")
        
    print(f"Generated comparison visualization plots with layer grouping.")

# To use:
# Ensure the CSV path is correct for your setup.
# Example:
# visualize_fault_injection('pt_fi/cnn/results/ResNet18/fp32/resnet18_fp32_fault_injection_results.csv')
# Make sure you have a CSV file at the specified path. For testing, you might use:
if __name__ == '__main__':

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
        visualize_comparison_fault_injection(ORIGINAL_CSV, PRUNED_ACT_CSV, PRUNED_MAG_CSV)
    except Exception as e:
        print(f"An error occurred during comparison visualization: {e}")
        print("If using actual files, ensure they are valid and paths are correct.")
        print("If dummy files were generated, there might be an issue in the dummy data generation or plotting logic itself.")

