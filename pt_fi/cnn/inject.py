import sys
import csv
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.models as models

# Import our modified fault injection modules
from inj_util import get_bit_flip_perturbation, delta_generator
from inj_util import get_network_inj_type, get_pytorch_conv_layers
from inj_layers import register_fault_hooks, remove_fault_hooks

def str2list(inp, do_float=False):
    str_list = inp.strip("][").split(',')
    return [float(i) if do_float else int(i) for i in str_list]

def run_one_cnn(args):
    num_inj_per_layer = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Obtain target layer information from file
    layer_dict = {}
    with open(args.layer_path) as layer_csv:
        layer_reader = csv.reader(layer_csv, delimiter='\t')
        for row in layer_reader:
            layer_dict[row[0]] = row[1]
    print(layer_dict)
    
    layer_name = layer_dict["Layer name:"]
    layer_dims = str2list(layer_dict["Output shape:"])
    quant_min_max = str2list(layer_dict["Quant min max:"], True) if "Quant min max:" in layer_dict else None
    
    # Load PyTorch model
    if args.network == 'rs':
        model = models.resnet50(pretrained=False)
    elif args.network == 'mb':
        model = models.mobilenet_v2(pretrained=False)
    elif args.network == 'ic':
        model = models.inception_v3(pretrained=False)
    else:
        print(f"Unsupported network: {args.network}")
        exit(1)
    
    # Load pre-trained weights
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Print available layers for debugging
    if args.list_layers:
        print("Available layers:")
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                print(f"Layer: {name}, Shape: {list(module.weight.shape)}")
        return
    
    # Generate injection positions
    delta_set, inj_pos = delta_generator(
        args.network, args.precision, args.inj_type, [layer_name], layer_dims, quant_min_max
    )
    
    # Load test data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 dataset
    dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Get specific image
    image, label = dataset[args.image_id]
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # First run without injection to get golden output
    with torch.no_grad():
        original_output = model(image)
        original_pred = torch.argmax(original_output, dim=1).item()
        original_prob = torch.softmax(original_output, dim=1)[0, original_pred].item()
    
    print(f"Original prediction: {original_pred}, probability: {original_prob:.4f}")
    
    # Register fault injection hooks
    hooks = register_fault_hooks(
        model, 
        args.inj_type, 
        [layer_name], 
        inj_pos=inj_pos, 
        quant_min_max=quant_min_max, 
        precision=args.precision
    )
    
    # Run with fault injection
    with torch.no_grad():
        faulty_output = model(image)
        faulty_pred = torch.argmax(faulty_output, dim=1).item()
        faulty_prob = torch.softmax(faulty_output, dim=1)[0, faulty_pred].item()
    
    # Remove hooks to restore model
    remove_fault_hooks(model, hooks)
    
    print(f"After injection, prediction: {faulty_pred}, probability: {faulty_prob:.4f}")
    
    # Print change statistics
    print(f"Classification {'changed' if original_pred != faulty_pred else 'unchanged'}")
    
    return {
        'original_pred': original_pred,
        'original_prob': original_prob,
        'faulty_pred': faulty_pred,
        'faulty_prob': faulty_prob,
        'changed': original_pred != faulty_pred,
        'invalid': original_pred != label
    }

def main():
    parser = argparse.ArgumentParser(description='PyTorch Fault Injection')
    parser.add_argument('--network', required=True, help="The target network, can be ic, mb or rs.")
    parser.add_argument('--precision', required=True, help="The data precision, can be fp16, int16 or int8")
    parser.add_argument('--inj_type', required=True, help="The injection type, can be INPUT, INPUT16, WEIGHT, WEIGHT16, RD_BFLIP or RD")
    parser.add_argument('--layer_path', required=True, help="Path to file that stores layer information")
    parser.add_argument('--image_id', type=int, required=True, help="The image ID to perform injection")
    parser.add_argument('--ckpt_path', required=True, help="The path to the network checkpoint")
    parser.add_argument('--list_layers', action='store_true', help="List available layers and exit")
    
    args = parser.parse_args()
    result = run_one_cnn(args)
    
    # Optionally save results to CSV
    if args.output_csv:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        file_exists = os.path.isfile(args.output_csv)
        with open(args.output_csv, 'a', newline='') as csvfile:
            fieldnames = ['network', 'precision', 'inj_type', 'layer',  
                         'original_pred', 'original_prob', 'faulty_pred', 'faulty_prob', 'changed','invalid']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'network': args.network,
                'precision': args.precision,
                'inj_type': args.inj_type,
                'layer': args.layer_path,
                'original_pred': result['original_pred'],
                'original_prob': result['original_prob'],
                'faulty_pred': result['faulty_pred'],
                'faulty_prob': result['faulty_prob'],
                'changed': result['changed'],
                'invalid': result['invalid']
            })

# if __name__ == '__main__':
#     main()