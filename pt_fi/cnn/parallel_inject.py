# comprehensive_fault_injection.py
import os
import sys
import csv
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import torch.nn as nn
import gc

from models.resnet import ResNet18, ResNet50
from models.LeNet5 import LeNet5
# Import the PyTorch-adapted injection functions
from inj_layers import register_fault_hooks, remove_fault_hooks
from inj_util import get_bit_flip_perturbation

def load_model_and_dataset(model_name, ckpt_path=None, device='cuda'):
    if model_name == 'lenet5':
        #model = LeNet5()
        model = torch.load('../../../results/lenet5-magnitude/pruned_model_step_10.pth')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif model_name == 'resnet18':
        #model = torch.load('../../../results/resnet18-act/pruned_model_step_10.pth')
        model = ResNet18()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif model_name == 'resnet50':
        model = ResNet50()
        #model = torch.load('../../../results/resnet50-act/pruned_model_step_10.pth')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    #checkpoint= torch.load(ckpt_path, map_location=device)
    #model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model, dataset, num_classes

def discover_layers(model, input_shape, target_layer_types=(nn.Linear, nn.Conv2d)):
    layer_info = {}
    model.eval()
    outputs = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            outputs[name] = {
                'output': output,
                'module': module,
                'input': input[0] if isinstance(input, tuple) and len(input) > 0 else input
            }
        return hook
    for name, module in model.named_modules():
        if name == '':  # Skip the model itself
            continue
        if isinstance(module, target_layer_types):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
    with torch.no_grad():
        model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    for name, output_data in outputs.items():
        module = output_data['module']
        output = output_data['output']
        input_tensor = output_data['input']
        
        layer_info[name] = {
            'name': name,
            'type': module.__class__.__name__,
            'module': module,
            'output_shape': list(output.shape) if isinstance(output, torch.Tensor) else None,
            'input_shape': list(input_tensor.shape) if isinstance(input_tensor, torch.Tensor) else None
        }
        if hasattr(module, 'weight'):
            layer_info[name]['weight_shape'] = list(module.weight.shape)
    
    return layer_info

def get_random_tensor_position(tensor_shape):
    position = []
    for dim_size in tensor_shape:
        position.append(random.randint(0, dim_size-1))
    return tuple(position)

def run_fault_injection_campaign(args):
    device = torch.device( "cpu")
    print(f"Using device: {device}")
    
    model, dataset, num_classes = load_model_and_dataset(args.model, args.ckpt_path, device)
    
    if args.model == 'lenet5':
        input_shape = (1, 1, 28, 28)  
    else:
        input_shape = (1, 3, 32, 32)  
    
    print("Discovering layers...")
    layer_info = discover_layers(model, input_shape, target_layer_types=(nn.Linear,nn.Conv2d,))
    print(f"Found {len(layer_info)} target layers")
    print("Layer names:", list(layer_info.keys()))
    print(layer_info)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = args.output_dir if args.output_dir else f"fi_results_{args.model}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    csv_path = os.path.join(result_dir, f"{args.model}_{args.precision}_fault_injection_results.csv")
    fieldnames = [
                'model', 'precision', 'layer_name', 'layer_type', 'fault_model', 
                'bit_position', 'experiment_id', 'image_label', 'original_class', 
                'original_confidence', 'faulty_class', 'faulty_confidence', 
                'classification_changed', 'injection_position'
            ]
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    if args.precision == 'fp32':
        bit_positions = range(32)
    elif args.precision == 'fp16':
        bit_positions = range(16)
    elif args.precision == 'int8':
        bit_positions = range(8)
    else:  # int16
        bit_positions = range(16)
    
    fault_models = ['INPUT','WEIGHT','INPUT16','WEIGHT16','RD','RD_BFLIP']
    
    # Calculate total experiments considering different loop structures for different fault models
    total_experiments = 0
    for fault_model in fault_models:
        if fault_model == 'RD':
            experiments_per_layer = 2 * args.experiments_per_config  # range(2) * experiments_per_config
        elif fault_model == 'RD_BFLIP':
            experiments_per_layer = 1 * args.experiments_per_config  # range(1) * experiments_per_config
        else:  # For future INPUT, WEIGHT, etc.
            experiments_per_layer = len(bit_positions) * args.experiments_per_config
        total_experiments += len(layer_info) * experiments_per_layer
    
    print(f"Starting fault injection campaign with {total_experiments} total experiments")
    
    progress_bar = tqdm(total=total_experiments, desc="Progress")
    
    for layer_name, layer_data in layer_info.items():
        for fault_model in fault_models:
            current_bit_positions = bit_positions  # Use a different variable name
            if 'RD' in fault_model:  # Only apply range(2) for pure RD, not RD_BFLIP
                current_bit_positions = range(2)  # Don't overwrite the original bit_positions
            for bit_position in current_bit_positions:
                for exp_id in range(args.experiments_per_config):
                    print("--------------------------------")
                    
                    # Generate random bit position for RD_BFLIP
                    if fault_model == 'RD_BFLIP':
                        if args.precision == 'fp32':
                            actual_bit_position = random.randint(0, 31)
                        elif args.precision == 'fp16':
                            actual_bit_position = random.randint(0, 15)
                        elif args.precision == 'int8':
                            actual_bit_position = random.randint(0, 7)
                        else:  # int16
                            actual_bit_position = random.randint(0, 15)
                    else:
                        actual_bit_position = bit_position
                    
                    print("layer_name: ",layer_name,"bit_position: ",actual_bit_position,"fault_model: ",fault_model,"exp_id: ",exp_id)
                    # Get random image from dataset
                    image_idx = random.randint(0, len(dataset)-1)
                    image, label = dataset[image_idx]
                    image = image.unsqueeze(0).to(device)  
                    
                    model.eval()
                    with torch.no_grad():
                        original_output = model(image)
                        original_probs = torch.nn.functional.softmax(original_output, dim=1)
                        original_class = torch.argmax(original_probs, dim=1).item()
                        original_confidence = original_probs[0, original_class].item()
                    
                    if 'INPUT' in fault_model:
                        tensor_shape = layer_data['input_shape']
                        inj_position = (0,) + get_random_tensor_position(tensor_shape[1:])
                    elif 'WEIGHT' in fault_model:
                        tensor_shape = layer_data['weight_shape']
                        inj_position = get_random_tensor_position(tensor_shape)
                    else:  # RD_BFLIP
                        tensor_shape = layer_data['output_shape']
                        inj_position = (0,) + get_random_tensor_position(tensor_shape[1:])
                    
                    # Set up fault injection
                    quant_min_max = [-3.402823e38, 3.402823e38] if args.precision == 'fp32' else [-65504, 65504]
                    if args.precision in ['int8', 'int16']:
                        quant_min_max = [-128, 127] if args.precision == 'int8' else [-32768, 32767]
                    
                    # Configure hooks for targeted injection
                    inj_pos = {layer_name: [inj_position]}
                    
                    # Register fault injection hooks
                    hooks = register_fault_hooks(
                        model,
                        inj_type=fault_model,
                        inj_layer=layer_name,
                        inj_pos=inj_pos,
                        quant_min_max=quant_min_max,
                        precision=args.precision,
                        bit_position=actual_bit_position  # Pass the actual bit position (random for RD_BFLIP)
                    )
                    
                    model.eval()
                    try:
                        with torch.no_grad():
                            faulty_output = model(image)
                            faulty_probs = torch.nn.functional.softmax(faulty_output, dim=1)
                            faulty_class = torch.argmax(faulty_probs, dim=1).item()
                            faulty_confidence = faulty_probs[0, faulty_class].item()
                        
                        # Remove hooks to restore model
                        remove_fault_hooks(model, hooks)
                        
                        # Store the bit position for recording
                        recorded_bit_position = actual_bit_position 
                        
                        # Record results
                        result = {
                            'model': args.model,
                            'precision': args.precision,
                            'layer_name': layer_name,
                            'layer_type': layer_data['type'],
                            'fault_model': fault_model,
                            'bit_position': recorded_bit_position,
                            'experiment_id': exp_id,
                            'image_label': label,
                            'original_class': original_class,
                            'original_confidence': original_confidence,
                            'faulty_class': faulty_class,
                            'faulty_confidence': faulty_confidence,
                            'classification_changed': bool(original_class != faulty_class),
                            'injection_position': str(inj_position)
                        }
                        
                        # Append to CSV
                        with open(csv_path, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writerow(result)
                        gc.collect()
                        
                    except Exception as e:
                        print(f"\nError in experiment: layer={layer_name}, bit={actual_bit_position}, fault={fault_model}, exp={exp_id}")
                        print(f"Exception: {str(e)}")
                        # Ensure hooks are removed even if there's an error
                        try:
                            remove_fault_hooks(model, hooks)
                        except:
                            pass
                    
                    progress_bar.update(1)
    
    progress_bar.close()
    print(f"Fault injection campaign completed. Results saved to {csv_path}")
    return csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive Fault Injection Framework')
    parser.add_argument('--model', type=str, required=True, choices=['lenet5', 'resnet18', 'resnet50'],
                        help='Model architecture to test')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'int16', 'int8'],
                        help='Numerical precision for fault injection')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: auto-generated)')
    parser.add_argument('--experiments_per_config', type=int, default=2,
                        help='Number of experiments per configuration (layer/bit/fault)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disable CUDA acceleration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    run_fault_injection_campaign(args)