import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from inj_util import bin2fp32, fp322bin, bin2fp16, fp162bin
from inj_util import bin2int16, int162bin, bin2int8, int82bin
from inj_util import get_bit_flip_perturbation

# Function to register hooks for fault injection
def register_fault_hooks(model, inj_type, inj_layer, inj_pos=None, quant_min_max=None, precision='fp32', bit_position=None):
    """
    Register appropriate hooks to model layers based on injection type
    
    Args:
        model: PyTorch model
        inj_type: Type of injection (INPUT, WEIGHT, RD_BFLIP)
        inj_layer: List of layer names to inject
        inj_pos: Dictionary mapping layer names to injection positions
        quant_min_max: Min/max values for quantized types
        precision: Numerical precision (fp32, fp16, int16, int8)
        bit_position: Specific bit position to inject (if applicable)
    
    Returns:
        handles: List of hook handles (for later removal)
    """
    handles = []
    
    # Create storage for delta values
    if not hasattr(model, 'delta_values'):
        model.delta_values = {}
    
    for name, module in model.named_modules():
        if name in inj_layer:
            if 'INPUT' in inj_type:
                # Register pre-forward hook for input injection
                handle = module.register_forward_pre_hook(
                    lambda mod, inp, name=name, precision=precision, quant_min_max=quant_min_max, inj_type=inj_type, inj_pos=inj_pos: 
                    input_injection_hook(mod, inp, name, precision, quant_min_max, inj_type, inj_pos, bit_position)
                )
                handles.append(handle)
            
            elif 'WEIGHT' in inj_type:
                # Register pre-forward hook for weight injection
                handle = module.register_forward_pre_hook(
                    lambda mod, inp, name=name, precision=precision, quant_min_max=quant_min_max, inj_type=inj_type, inj_pos=inj_pos: 
                    weight_injection_hook(mod, inp, name, precision, quant_min_max, inj_type, inj_pos, bit_position)
                )
                handles.append(handle)
            
            elif 'RD_BFLIP' in inj_type or 'RD' in inj_type:
                # Register forward hook for output injection
                handle = module.register_forward_hook(
                    lambda mod, inp, output, name=name, precision=precision, quant_min_max=quant_min_max, inj_type=inj_type, inj_pos=inj_pos: 
                    output_injection_hook(mod, inp, output, name, precision, quant_min_max, inj_type, inj_pos, bit_position)
                )
                handles.append(handle)
    
    return handles

def input_injection_hook(module, inputs, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    """Hook for injecting faults into input tensors"""
    
    # Get the input tensor
    inp = inputs[0]
    batch_size = inp.shape[0]
    
    # If we already have prepared delta for this layer, apply it
    if hasattr(module, 'input_delta'):
        delta = module.input_delta
        # Apply delta to input
        modified_input = inp + delta
        
        # Apply bounding based on precision
        if precision == 'fp32':
            modified_input = torch.clamp(modified_input, -3.402823e38, 3.402823e38)
        elif precision == 'fp16':
            modified_input = torch.clamp(modified_input, -65504, 65504)
        elif precision == 'int16' or precision == 'int8':
            if quant_min_max is not None:
                q_min, q_max = quant_min_max
                modified_input = torch.clamp(modified_input, q_min, q_max)
        
        # Replace NaN values with zeros
        modified_input = torch.where(torch.isnan(modified_input), 
                                    torch.zeros_like(modified_input), 
                                    modified_input)
        
        return (modified_input,) + inputs[1:] if len(inputs) > 1 else modified_input
    
    # If this is the first call, prepare delta
    if 'INPUT16' in inj_type:
        # Handle grouped injection (16 values at once)
        # Implementation similar to the original code but adapted for PyTorch tensors
        pass
    else:
        # Handle single value injection
        # Select random position if not specified
        if inj_pos is None or layer_name not in inj_pos:
            # For BCHW format
            c = torch.randint(0, inp.shape[1], (1,)).item()
            h = torch.randint(0, inp.shape[2], (1,)).item()
            w = torch.randint(0, inp.shape[3], (1,)).item()
            positions = [(0, c, h, w)]  # Assuming we inject in first batch item
        else:
            positions = inj_pos[layer_name]
        
        # Create delta tensor
        delta = torch.zeros_like(inp)
        
        for pos in positions:
            b, c, h, w = pos
            golden_val = inp[b, c, h, w].item()
            
            # Get bit flip perturbation
            _, perturb = get_bit_flip_perturbation(
                'default', precision, golden_val, layer_name, 'INPUT', quant_min_max, bit_position
            )
            
            # Apply perturbation
            delta[b, c, h, w] = perturb
        
        # Store delta for future calls
        module.input_delta = delta
        
        # Apply delta to input
        modified_input = inp + delta
        
        # Apply bounding
        if precision == 'fp32':
            modified_input = torch.clamp(modified_input, -3.402823e38, 3.402823e38)
        elif precision == 'fp16':
            modified_input = torch.clamp(modified_input, -65504, 65504)
        elif precision == 'int16' or precision == 'int8':
            if quant_min_max is not None:
                q_min, q_max = quant_min_max
                modified_input = torch.clamp(modified_input, q_min, q_max)
        
        # Replace NaN values
        modified_input = torch.where(torch.isnan(modified_input), 
                                    torch.zeros_like(modified_input), 
                                    modified_input)
        
        return (modified_input,) + inputs[1:] if len(inputs) > 1 else modified_input

def weight_injection_hook(module, inputs, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    """Hook for injecting faults into weight tensors"""
    
    # Store original weights for restoration
    if not hasattr(module, 'original_weight'):
        module.original_weight = module.weight.data.clone()
    
    # If we already have prepared delta for this layer, apply it
    if hasattr(module, 'weight_delta'):
        # Apply stored delta to weights
        module.weight.data = module.original_weight + module.weight_delta
        return
    
    # Create delta for weights
    if isinstance(module, nn.Conv2d):
        weights = module.weight.data
        
        # Select random position if not specified
        if inj_pos is None or layer_name not in inj_pos:
            # For conv weights: out_channels, in_channels, kernel_h, kernel_w
            out_c = torch.randint(0, weights.shape[0], (1,)).item()
            in_c = torch.randint(0, weights.shape[1], (1,)).item()
            k_h = torch.randint(0, weights.shape[2], (1,)).item()
            k_w = torch.randint(0, weights.shape[3], (1,)).item()
            positions = [(out_c, in_c, k_h, k_w)]
        else:
            positions = inj_pos[layer_name]
        
        # Create delta tensor
        delta = torch.zeros_like(weights)
        
        for pos in positions:
            o, i, h, w = pos
            golden_val = weights[o, i, h, w].item()
            
            # Get bit flip perturbation
            _, perturb = get_bit_flip_perturbation(
                'default', precision, golden_val, layer_name, 'WEIGHT', quant_min_max, bit_position
            )
            
            # Apply perturbation
            delta[o, i, h, w] = perturb
        
        # Store delta for future calls
        module.weight_delta = delta
        
        # Apply delta to weights
        module.weight.data = module.original_weight + delta

def output_injection_hook(module, inputs, output, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    """Hook for injecting faults into output tensors"""
    
    # For RD_BFLIP (random bit flip) or general RD injection
    if 'RD_BFLIP' in inj_type or 'RD' in inj_type:
        # Handle direct output modification
        if not hasattr(module, 'output_delta'):
            # Create delta for output
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            
            # Select random position if not specified
            if inj_pos is None or layer_name not in inj_pos:
                # For BCHW format
                b = 0  # Assuming we inject in first batch item
                c = torch.randint(0, out.shape[1], (1,)).item()
                h = torch.randint(0, out.shape[2], (1,)).item()
                w = torch.randint(0, out.shape[3], (1,)).item()
                positions = [(b, c, h, w)]
            else:
                positions = inj_pos[layer_name]
            
            # Create delta tensor
            delta = torch.zeros_like(out)
            
            for pos in positions:
                b, c, h, w = pos
                golden_val = out[b, c, h, w].item()
                
                # Get bit flip perturbation
                _, perturb = get_bit_flip_perturbation(
                    'default', precision, golden_val, layer_name, 'RD_BFLIP', quant_min_max, bit_position
                )
                
                # Apply perturbation
                delta[b, c, h, w] = perturb
            
            # Store delta for future calls
            module.output_delta = delta
        
        # Apply delta to output
        if isinstance(output, tuple):
            modified_out = output[0] + module.output_delta
            
            # Apply bounding
            if precision == 'fp32':
                modified_out = torch.clamp(modified_out, -3.402823e38, 3.402823e38)
            elif precision == 'fp16':
                modified_out = torch.clamp(modified_out, -65504, 65504)
            elif precision == 'int16' or precision == 'int8':
                if quant_min_max is not None:
                    q_min, q_max = quant_min_max
                    modified_out = torch.clamp(modified_out, q_min, q_max)
            
            # Replace NaN values
            modified_out = torch.where(torch.isnan(modified_out), 
                                      torch.zeros_like(modified_out), 
                                      modified_out)
            
            return (modified_out,) + output[1:]
        else:
            modified_out = output + module.output_delta
            
            # Apply bounding
            if precision == 'fp32':
                modified_out = torch.clamp(modified_out, -3.402823e38, 3.402823e38)
            elif precision == 'fp16':
                modified_out = torch.clamp(modified_out, -65504, 65504)
            elif precision == 'int16' or precision == 'int8':
                if quant_min_max is not None:
                    q_min, q_max = quant_min_max
                    modified_out = torch.clamp(modified_out, q_min, q_max)
            
            # Replace NaN values
            modified_out = torch.where(torch.isnan(modified_out), 
                                      torch.zeros_like(modified_out), 
                                      modified_out)
            
            return modified_out
    
    return output

# Function to clean up hooks after testing
def remove_fault_hooks(model, handles):
    """Remove all registered hooks"""
    for handle in handles:
        handle.remove()
    
    # Restore original weights if modified
    for name, module in model.named_modules():
        if hasattr(module, 'original_weight'):
            module.weight.data = module.original_weight
            delattr(module, 'original_weight')
        
        if hasattr(module, 'weight_delta'):
            delattr(module, 'weight_delta')
        
        if hasattr(module, 'input_delta'):
            delattr(module, 'input_delta')
        
        if hasattr(module, 'output_delta'):
            delattr(module, 'output_delta')