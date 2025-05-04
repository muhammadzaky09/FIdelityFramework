import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import json
import argparse
import time
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST

# Import our modified fault injection modules
from inj_layers import register_fault_hooks, remove_fault_hooks
from inj_util import get_bit_flip_perturbation

def discover_layers(model, input_shape, target_types=(nn.Conv2d, nn.Linear)):
    """Discover all injectable layers in a model with their shapes"""
    model.eval()
    layer_info = OrderedDict()
    
    # Register hooks to capture output shapes
    hooks = []
    
    def hook_fn(name):
        def forward_hook(module, input, output):
            layer_info[name] = {
                'module': module,
                'output_shape': output.shape,
                'type': module.__class__.__name__
            }
            if hasattr(module, 'weight'):
                layer_info[name]['weight_shape'] = module.weight.shape
        return forward_hook
    
    for name, module in model.named_modules():
        if name and isinstance(module, target_types):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run a forward pass with dummy input
    dummy_input = torch.randn(*input_shape)
    with torch.no_grad():
        model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return layer_info

def run_fault_injection(model, layer_name, bit_position, inj_type, test_loader, 
                       num_experiments=1, num_images=100, precision='fp32',
                       device='cpu'):
    """
    Run fault injection experiments for a specific layer, bit position, and type
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer to inject
        bit_position: Bit position to flip (0-31 for FP32)
        inj_type: Injection type ('INPUT', 'WEIGHT', 'RD_BFLIP')
        test_loader: DataLoader for test images
        num_experiments: Number of experiments per configuration
        num_images: Number of images to test
        precision: Numerical precision ('fp32', 'fp16', 'int16', 'int8')
        device: Device to run on
        
    Returns:
        DataFrame with results
    """
    model.to(device)
    model.eval()
    
    results = []
    
    # Get the module corresponding to the layer name
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
    
    if target_module is None:
        print(f"Layer {layer_name} not found!")
        return pd.DataFrame()
    
    # Default quant_min_max for FP32
    quant_min_max = [-3.4028235e38, 3.4028235e38]
    
    for exp_id in range(num_experiments):
        # For each experiment, we'll inject at different random positions
        
        # Run on a subset of test images
        image_count = 0
        correct_golden = 0
        correct_faulty = 0
        classification_changes = 0
        
        for images, labels in test_loader:
            if image_count >= num_images:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)
            
            # First get golden results (without fault injection)
            with torch.no_grad():
                golden_outputs = model(images)
                golden_preds = torch.argmax(golden_outputs, dim=1)
                correct_golden += (golden_preds == labels).sum().item()
            
            # Now run with fault injection
            # Register hooks for the specific layer and bit position
            hooks = register_fault_hooks(
                model, inj_type, [layer_name], 
                quant_min_max=quant_min_max, 
                precision=precision,
                bit_position=bit_position  # Pass specific bit position
            )
            
            with torch.no_grad():
                faulty_outputs = model(images)
                faulty_preds = torch.argmax(faulty_outputs, dim=1)
                correct_faulty += (faulty_preds == labels).sum().item()
                classification_changes += (faulty_preds != golden_preds).sum().item()
            
            # Remove hooks to restore model
            remove_fault_hooks(model, hooks)
            
            image_count += batch_size
        
        # Calculate metrics
        golden_accuracy = correct_golden / image_count
        faulty_accuracy = correct_faulty / image_count
        sdc_rate = classification_changes / image_count  # Silent Data Corruption rate
        
        results.append({
            'layer': layer_name,
            'layer_type': target_module.__class__.__name__,
            'bit_position': bit_position,
            'inj_type': inj_type,
            'experiment_id': exp_id,
            'golden_accuracy': golden_accuracy,
            'faulty_accuracy': faulty_accuracy,
            'accuracy_drop': golden_accuracy - faulty_accuracy,
            'sdc_rate': sdc_rate,
            'images_tested': image_count
        })
    
    return pd.DataFrame(results)

def worker_function(job, model_class, checkpoint_path, dataset_type, transform, 
                   num_experiments, num_images, batch_size, precision, device_id):
    """
    Worker function to run a set of fault injection experiments
    
    Args:
        job: Dictionary with job details (layer, bit positions, inj types)
        model_class: Class to instantiate the model
        checkpoint_path: Path to model checkpoint
        dataset_type: 'mnist' or 'cifar10'
        transform: Torchvision transform
        num_experiments: Number of experiments per configuration
        num_images: Number of images to test
        batch_size: Batch size for testing
        precision: Numerical precision
        device_id: GPU ID or 'cpu'
    
    Returns:
        DataFrame with results
    """
    # Select device
    if device_id == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device_id}')
    
    # Create model instance
    model = model_class()
    
    # Load checkpoint if provided
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    # Load dataset
    if dataset_type == 'mnist':
        testset = MNIST(root='./data', train=False, download=True, transform=transform)
    else:  # cifar10
        testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Run experiments
    all_results = []
    layer_name = job['layer']
    
    for bit_pos in job['bit_positions']:
        for inj_type in job['inj_types']:
            try:
                results = run_fault_injection(
                    model, layer_name, bit_pos, inj_type, test_loader,
                    num_experiments=num_experiments, num_images=num_images,
                    precision=precision, device=device
                )
                all_results.append(results)
            except Exception as e:
                print(f"Error in worker (layer={layer_name}, bit={bit_pos}, type={inj_type}): {str(e)}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

def run_comprehensive_campaign(model_name, precision='fp32', checkpoint_path=None,
                              output_dir='fi_results', num_experiments=1, 
                              num_images=100, batch_size=64, num_processes=None,
                              skip_layer_discovery=False, layer_info_path=None,
                              gpu_ids=None):
    """
    Run a comprehensive fault injection campaign using multiprocessing
    
    Args:
        model_name: 'lenet5', 'resnet18', or 'resnet50'
        precision: 'fp32', 'fp16', 'int16', 'int8'
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save results
        num_experiments: Number of experiments per configuration
        num_images: Number of images to test
        batch_size: Batch size for testing
        num_processes: Number of processes to use (default: number of CPU cores)
        skip_layer_discovery: Skip layer discovery if layer info is available
        layer_info_path: Path to saved layer info
        gpu_ids: List of GPU IDs to use (default: use CPU only)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of processes
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Setup device mapping
    if gpu_ids:
        # Check available GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if any(gpu_id >= num_gpus for gpu_id in gpu_ids):
                print(f"Warning: Requested GPU IDs {gpu_ids} exceed available GPUs ({num_gpus})")
                gpu_ids = [i % num_gpus for i in gpu_ids]
        else:
            print("Warning: CUDA not available, using CPU instead")
            gpu_ids = None
    
    # Device assignment function
    def get_device_id(process_idx):
        if gpu_ids:
            return gpu_ids[process_idx % len(gpu_ids)]
        else:
            return 'cpu'
    
    # 1. Load model class
    if model_name == 'lenet5':
        from LeNet5 import LeNet5
        model_class = LeNet5
        input_shape = (1, 1, 28, 28)
        dataset_type = 'mnist'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif model_name == 'resnet18':
        from resnet import ResNet18
        model_class = ResNet18
        input_shape = (1, 3, 32, 32)
        dataset_type = 'cifar10'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    elif model_name == 'resnet50':
        from resnet import ResNet50
        model_class = ResNet50
        input_shape = (1, 3, 32, 32)
        dataset_type = 'cifar10'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create a temporary model instance for layer discovery
    temp_model = model_class()
    
    # 2. Discover layers or load layer info
    if skip_layer_discovery and layer_info_path:
        with open(layer_info_path, 'r') as f:
            layer_info = json.load(f)
        # Convert layer info from JSON to dict with actual layer names
        layers = list(layer_info.keys())
    else:
        print("Discovering layers...")
        discovered_info = discover_layers(temp_model, input_shape)
        layers = list(discovered_info.keys())
        
        # Save layer info for future use
        layer_info = {k: {
            'type': discovered_info[k]['type'],
            'output_shape': list(discovered_info[k]['output_shape']),
            'weight_shape': list(discovered_info[k]['weight_shape']) if 'weight_shape' in discovered_info[k] else None
        } for k in layers}
        
        with open(os.path.join(output_dir, 'layer_info.json'), 'w') as f:
            json.dump(layer_info, f, indent=2)
    
    # Print layer summary
    print(f"Found {len(layers)} layers:")
    for i, name in enumerate(layers):
        print(f"  {i+1}. {name} ({layer_info[name]['type']})")
    
    # 3. Define fault models and bit positions
    fault_models = ['INPUT', 'WEIGHT', 'RD_BFLIP']
    
    # Define bit positions based on precision
    if precision == 'fp32':
        bit_positions = list(range(32))  # 0-31 
    elif precision == 'fp16':
        bit_positions = list(range(16))  # 0-15
    elif precision == 'int8':
        bit_positions = list(range(8))   # 0-7
    else:  # int16
        bit_positions = list(range(16))  # 0-15
    
    # Calculate total experiments
    total_exps = len(layers) * len(bit_positions) * len(fault_models) * num_experiments
    print(f"Running {total_exps} experiments across {num_processes} processes...")
    
    # 4. Prepare jobs for multiprocessing
    jobs = []
    for layer in layers:
        jobs.append({
            'layer': layer,
            'bit_positions': bit_positions,
            'inj_types': fault_models
        })
    
    # Balance jobs across processes
    balanced_jobs = []
    for i in range(num_processes):
        process_jobs = jobs[i::num_processes]
        if process_jobs:
            balanced_jobs.append(process_jobs)
    
    # 5. Setup multiprocessing
    # Initialize the multiprocessing context with 'spawn' method (safer for PyTorch)
    ctx = mp.get_context('spawn')
    
    # Create results directory
    results_dir = os.path.join(output_dir, f"{model_name}_{precision}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # 6. Run the jobs in parallel
    results = []
    
    # Create a pool of worker processes
    with ctx.Pool(processes=num_processes) as pool:
        # Create worker tasks
        worker_tasks = []
        
        for process_id, process_jobs in enumerate(balanced_jobs):
            for job in process_jobs:
                device_id = get_device_id(process_id)
                worker_tasks.append(
                    pool.apply_async(
                        worker_function,
                        args=(job, model_class, checkpoint_path, dataset_type, transform,
                             num_experiments, num_images, batch_size, precision, device_id)
                    )
                )
        
        # Create a progress bar to monitor tasks
        with tqdm(total=len(worker_tasks), desc="Fault injection progress") as pbar:
            for task in worker_tasks:
                # Wait for task to complete and collect results
                task_result = task.get()
                results.append(task_result)
                pbar.update(1)
    
    # 7. Combine and save results
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(os.path.join(results_dir, "all_results.csv"), index=False)
    
    # 8. Generate summary
    generate_summary(final_df, results_dir)
    
    # Done
    elapsed_time = time.time() - start_time
    print(f"Campaign completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to {results_dir}")

def generate_summary(results_df, output_dir):
    """Generate summary statistics and visualizations"""
    if len(results_df) == 0:
        print("No results to summarize")
        return
        
    # By layer
    layer_summary = results_df.groupby('layer')['sdc_rate'].mean().reset_index()
    layer_summary.sort_values('sdc_rate', ascending=False, inplace=True)
    layer_summary.to_csv(os.path.join(output_dir, 'layer_vulnerability.csv'), index=False)
    
    # By bit position
    bit_summary = results_df.groupby('bit_position')['sdc_rate'].mean().reset_index()
    bit_summary.to_csv(os.path.join(output_dir, 'bit_vulnerability.csv'), index=False)
    
    # By injection type
    type_summary = results_df.groupby('inj_type')['sdc_rate'].mean().reset_index()
    type_summary.to_csv(os.path.join(output_dir, 'type_vulnerability.csv'), index=False)
    
    # Layer × bit position heatmap
    heatmap = results_df.pivot_table(
        index='layer', columns='bit_position', values='sdc_rate', aggfunc='mean'
    )
    heatmap.to_csv(os.path.join(output_dir, 'layer_bit_heatmap.csv'))
    
    # Overall statistics
    with open(os.path.join(output_dir, 'summary_stats.txt'), 'w') as f:
        f.write(f"Total experiments: {len(results_df)}\n")
        f.write(f"Average SDC rate: {results_df['sdc_rate'].mean():.4f}\n")
        f.write(f"Average accuracy drop: {results_df['accuracy_drop'].mean():.4f}\n")
        f.write(f"Most vulnerable layer: {layer_summary.iloc[0]['layer']} (SDC rate: {layer_summary.iloc[0]['sdc_rate']:.4f})\n")
        f.write(f"Most vulnerable bit: {bit_summary.loc[bit_summary['sdc_rate'].idxmax()]['bit_position']} (SDC rate: {bit_summary['sdc_rate'].max():.4f})\n")
        f.write(f"Most vulnerable injection type: {type_summary.loc[type_summary['sdc_rate'].idxmax()]['inj_type']} (SDC rate: {type_summary['sdc_rate'].max():.4f})\n")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Fault Injection Campaign')
    parser.add_argument('--model', required=True, choices=['lenet5', 'resnet18', 'resnet50'],
                       help='Model architecture')
    parser.add_argument('--precision', default='fp32', choices=['fp32', 'fp16', 'int8', 'int16'],
                       help='Numerical precision')
    parser.add_argument('--checkpoint', default=None, help='Path to model checkpoint')
    parser.add_argument('--output_dir', default='fi_results', help='Directory to save results')
    parser.add_argument('--num_experiments', type=int, default=1, 
                       help='Number of experiments per configuration')
    parser.add_argument('--num_images', type=int, default=100,
                       help='Number of images to test per experiment')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--num_processes', type=int, default=None, 
                       help='Number of processes to use (default: CPU count)')
    parser.add_argument('--gpu_ids', type=int, nargs='*', default=None,
                       help='List of GPU IDs to use (default: use CPU only)')
    parser.add_argument('--layer_info', default=None, help='Path to saved layer info')
    
    args = parser.parse_args()
    
    run_comprehensive_campaign(
        model_name=args.model,
        precision=args.precision,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_experiments=args.num_experiments,
        num_images=args.num_images,
        batch_size=args.batch_size,
        num_processes=args.num_processes,
        gpu_ids=args.gpu_ids,
        skip_layer_discovery=args.layer_info is not None,
        layer_info_path=args.layer_info
    )

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()