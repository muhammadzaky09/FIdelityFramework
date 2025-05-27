import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

from inj_util import bin2fp32, fp322bin, bin2fp16, fp162bin
from inj_util import bin2int16, int162bin, bin2int8, int82bin
from inj_util import get_bit_flip_perturbation, perturb_conv, apply_precision_bounds

def log_to_file(message, filename="log.txt", mode="a"):
    with open(filename, mode) as f:
        f.write(f"{message}\n")

def calculate_conv_output_position(input_h, input_w, kernel_h, kernel_w, stride, padding, output_h, output_w):
    """
    Calculate which output positions would be affected by an input fault at (input_h, input_w)
    Following FIdelity-Q's approach
    """
    # Calculate the range of output positions that could be affected
    min_out_h = max(0, ((input_h + padding) // stride - kernel_h + 1))
    max_out_h = min(((input_h + padding) // stride + 1), output_h)
    
    min_out_w = max(0, ((input_w + padding) // stride - kernel_w + 1))
    max_out_w = min(((input_w + padding) // stride + 1), output_w)
    
    # Randomly select within the valid range
    if min_out_h < max_out_h:
        start_h = torch.randint(min_out_h, max_out_h, (1,)).item()
    else:
        start_h = min_out_h
        
    if min_out_w < max_out_w:
        start_w = torch.randint(min_out_w, max_out_w, (1,)).item()
    else:
        start_w = min_out_w
    
    return start_h, start_w

def input_fault_hook(module, inputs, output, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    """Hook function for input fault injection"""
    inp = inputs[0]
    weights = module.weight
    
    # Create perturbation and inject fault based on layer type
    if isinstance(module, nn.Conv2d):
        # Define stride and padding
        stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
        padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
        groups = module.groups if hasattr(module, 'groups') else 1
        
        # Get positions to inject faults
        if inj_pos is not None and layer_name in inj_pos:
            b, c, h, w = inj_pos[layer_name][0]
        else:
            # Single random position
            b = 0
            c = torch.randint(0, inp.shape[1], (1,)).item()
            h = torch.randint(0, inp.shape[2], (1,)).item()
            w = torch.randint(0, inp.shape[3], (1,)).item()

        inp_perturb = torch.zeros_like(inp)
        golden_val = inp[b, c, h, w].item()
        
        _, perturb = get_bit_flip_perturbation(
            'default', precision, golden_val, layer_name, 'INPUT', quant_min_max, bit_position
        )
        
        inp_perturb[b, c, h, w] = perturb
        delta = perturb_conv(inp_perturb, weights, stride, padding, groups)
        
    elif isinstance(module, nn.Linear):
        # Handle Linear layer input fault
        if inj_pos is not None and layer_name in inj_pos:
            b, f = inj_pos[layer_name][0]
        else:
            # Single random position for Linear input
            b = 0
            # For flattened input, select one feature
            f = torch.randint(0, inp.shape[1] if len(inp.shape) == 2 else inp.numel() // inp.shape[0], (1,)).item()
        
        # Reshape input if needed
        inp_flat = inp if len(inp.shape) == 2 else inp.view(inp.shape[0], -1)
        inp_perturb = torch.zeros_like(inp_flat)
        
        golden_val = inp_flat[b, f].item()
        
        _, perturb = get_bit_flip_perturbation(
            'default', precision, golden_val, layer_name, 'INPUT', quant_min_max, bit_position
        )
        
        inp_perturb[b, f] = perturb
        
        delta = F.linear(inp_perturb, weights, bias=None)
    
    # Apply INPUT16 logic - FIdelity-Q style
    if 'INPUT16' in inj_type:
        if isinstance(module, nn.Conv2d):
            if np.count_nonzero(delta) == 0:
                return output
            
            delta_16 = torch.zeros_like(delta)
            
            # FIdelity-Q approach: Calculate output position using convolution math
            if inj_pos is not None and layer_name in inj_pos:
                # Use the original fault position to calculate output position
                _, _, input_h, input_w = inj_pos[layer_name][0]
                kernel_h, kernel_w = weights.shape[2], weights.shape[3]
                output_h, output_w = delta.shape[2], delta.shape[3]
                
                start_h, start_w = calculate_conv_output_position(
                    input_h, input_w, kernel_h, kernel_w, stride, padding, output_h, output_w
                )
            else:
                # Fallback to random position
                start_h = torch.randint(0, delta.shape[2], (1,)).item()
                start_w = torch.randint(0, delta.shape[3], (1,)).item()
            
            # Select 16 consecutive channels at the calculated position
            total_channels = delta.shape[1]
            if total_channels >= 16:
                start_channel = torch.randint(0, total_channels - 15, (1,)).item()
                num_channels = 16
            else:
                start_channel = 0
                num_channels = total_channels
            
            for channel in range(num_channels):
                delta_16[0, start_channel + channel, start_h, start_w] = delta[0, start_channel + channel, start_h, start_w]
            
            delta = delta_16
            
        elif isinstance(module, nn.Linear):
            if np.count_nonzero(delta) == 0:
                return output
            delta_16 = torch.zeros_like(delta)
            # For Linear layer, select 16 consecutive outputs
            total_features = delta.shape[1]
            max_start = max(0, total_features - 16)
            f_start = torch.randint(0, max_start + 1, (1,)).item()
            # Limit to actual size or 16, whichever is smaller
            num_features = min(16, total_features)
            for i in range(num_features):
                delta_16[0, f_start + i] = delta[0, f_start + i]
            delta = delta_16
    
    # Add delta to output and apply bounds
    modified_output = output + delta
    modified_output = modified_output.float() 
    modified_output = apply_precision_bounds(modified_output, precision, quant_min_max)

    return modified_output

def weight_fault_hook(module, inputs, output, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    """Hook function for weight fault injection"""
    inp = inputs[0]
    weights = module.weight
    
    # Handle different layer types
    if isinstance(module, nn.Conv2d):
        # Get positions to inject faults
        if inj_pos is not None and layer_name in inj_pos:
            out_c, in_c, h, w = inj_pos[layer_name][0]
        else:
            # Single random position
            out_c = torch.randint(0, weights.shape[0], (1,)).item()
            in_c = torch.randint(0, weights.shape[1], (1,)).item()
            h = torch.randint(0, weights.shape[2], (1,)).item()
            w = torch.randint(0, weights.shape[3], (1,)).item()
        
        # Create perturbation tensor
        wt_perturb = torch.zeros_like(weights)
        
        # Get original value
        golden_val = weights[out_c, in_c, h, w].item()
        
        # Calculate perturbation via bit flip
        _, perturb = get_bit_flip_perturbation(
            'default', precision, golden_val, layer_name, 'WEIGHT', quant_min_max, bit_position
        )
        
        # Apply perturbation
        wt_perturb[out_c, in_c, h, w] = perturb
        
        # Define stride and padding
        stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
        padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
        groups = module.groups if hasattr(module, 'groups') else 1
        
        # Propagate through convolution
        delta = perturb_conv(inp, wt_perturb, stride, padding, groups)
    
    elif isinstance(module, nn.Linear):
        # Get positions to inject faults - ONLY in weight matrix (no bias)
        if inj_pos is not None and layer_name in inj_pos:
            out_f, in_f = inj_pos[layer_name][0]
        else:
            # Only target weight matrix - exclude bias completely
            out_f = torch.randint(0, weights.shape[0], (1,)).item()
            in_f = torch.randint(0, weights.shape[1], (1,)).item()
        
        # Weight matrix fault: affects one output neuron, scaled by corresponding input
        wt_perturb = torch.zeros_like(weights)
        golden_val = weights[out_f, in_f].item()
        
        _, perturb = get_bit_flip_perturbation(
            'default', precision, golden_val, layer_name, 'WEIGHT', quant_min_max, bit_position
        )
        
        wt_perturb[out_f, in_f] = perturb
        
        # Use F.linear to properly propagate the weight fault
        # Reshape input if needed
        reshaped_inp = inp.view(inp.size(0), -1)
        delta = F.linear(reshaped_inp, wt_perturb, bias=None)
    
    # Apply WEIGHT16 logic - FIdelity-Q style
    if 'WEIGHT16' in inj_type:
        if isinstance(module, nn.Conv2d):
            if np.count_nonzero(delta) == 0:
                print("delta is all zeros")
                return output
                
            delta_16 = torch.zeros_like(delta)
            
            # FIdelity-Q approach: Use predetermined channel and raster-scan positions
            if inj_pos is not None and layer_name in inj_pos:
                # Use the affected channel from the original fault
                start_channel = inj_pos[layer_name][0][0]  # out_c from weight fault
            else:
                # Fallback to random channel
                start_channel = torch.randint(0, delta.shape[1], (1,)).item()
            
            dim_height, dim_width = delta.shape[2], delta.shape[3]
            total_positions = dim_height * dim_width
            
            # Select random spatial block for 16 consecutive positions
            if total_positions >= 16:
                start_position = torch.randint(0, total_positions // 16, (1,)).item()
                
                # Map to 16 consecutive spatial positions in raster-scan order
                for inject_index in range(16):
                    linear_pos = start_position * 16 + inject_index
                    if linear_pos >= total_positions:
                        break
                    
                    inject_height = linear_pos // dim_width
                    inject_width = linear_pos % dim_width
                    
                    delta_16[0, start_channel, inject_height, inject_width] = delta[0, start_channel, inject_height, inject_width]
            else:
                # If less than 16 positions total, use all available
                for h in range(dim_height):
                    for w in range(dim_width):
                        delta_16[0, start_channel, h, w] = delta[0, start_channel, h, w]
            
            delta = delta_16
            
        elif isinstance(module, nn.Linear):
            # WEIGHT16 for FC: "One out of 16 output neurons are faulty"
            # This means every 16th neuron gets the same fault value
            if torch.count_nonzero(delta) == 0:
                return output
                
            # Find the original faulty neuron and get its fault value
            original_faulty_neuron = torch.nonzero(delta).flatten()
            if len(original_faulty_neuron) == 0:
                return output
                
            faulty_neuron_idx = original_faulty_neuron[0].item()
            fault_value = delta[0, faulty_neuron_idx].item()
            
            # Create delta_16 with strided fault pattern
            delta_16 = torch.zeros_like(delta)
            total_neurons = delta.shape[1]
            neurons_affected = 0
            
            # Apply fault every 16 neurons starting from the original position
            current_neuron = faulty_neuron_idx
            while current_neuron < total_neurons and neurons_affected < 16:
                delta_16[0, current_neuron] = fault_value
                current_neuron += 16
                neurons_affected += 1
            
            delta = delta_16
    
    # Add delta to output and apply bounds
    modified_output = output + delta
    modified_output = modified_output.float()
    modified_output = apply_precision_bounds(modified_output, precision, quant_min_max)
    
    return modified_output

def output_fault_hook(module, inputs, output, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    """Directly inject faults into output tensors (RD and RD_BFLIP models)"""
    if isinstance(output, tuple):
        orig_output = output[0]
        is_tuple = True
    else:
        orig_output = output
        is_tuple = False
    
    # Make a copy to avoid modifying the original in-place
    modified_output = orig_output.clone()
    
    # Determine if output is from Linear layer based on tensor shape
    is_linear_output = len(orig_output.shape) == 2
    
    # Get positions to inject faults
    if is_linear_output:
        # Handle Linear layer output (2D: batch, features)
        if inj_pos is not None and layer_name in inj_pos:
            b, f = inj_pos[layer_name][0]
        else:
            # Single random position
            b = 0
            f = torch.randint(0, orig_output.shape[1], (1,)).item()
            
        if 'RD_BFLIP' in inj_type:
            golden_val = orig_output[b, f].item()
            # Use the bit position passed from the main loop (already randomized there)
            _, perturb = get_bit_flip_perturbation(
                'default', precision, golden_val, layer_name, 'RD_BFLIP', quant_min_max, bit_position
            )
            modified_output[b, f] += perturb
        else:
            # Random replacement with appropriate precision
            if precision == 'fp32':
                random_bin = ''.join(str(torch.randint(0, 2, (1,)).item()) for _ in range(32))
                random_val = bin2fp32(random_bin)
            elif precision == 'fp16':
                random_bin = ''.join(str(torch.randint(0, 2, (1,)).item()) for _ in range(16))
                random_val = bin2fp16(random_bin)
            elif precision == 'int16':
                if quant_min_max is not None:
                    q_min, q_max = quant_min_max
                    granu = (q_max - q_min) / 65535
                    delta_int = torch.randint(0, 2**16, (1,)).item()
                    random_val = delta_int * granu + q_min
                else:
                    random_val = torch.randint(-32768, 32767 + 1, (1,)).item()
            elif precision == 'int8':
                if quant_min_max is not None:
                    q_min, q_max = quant_min_max
                    granu = (q_max - q_min) / 256
                    delta_int = torch.randint(0, 2**8, (1,)).item()
                    random_val = delta_int * granu + q_min
                else:
                    random_val = torch.randint(-128, 127 + 1, (1,)).item()
            
            modified_output[b, f] = random_val
    else:
        # Original Conv2d output handling (4D: batch, channel, height, width)
        if inj_pos is not None and layer_name in inj_pos:
            b, c, h, w = inj_pos[layer_name][0] 
        else:
            # Single random position
            b = 0
            c = torch.randint(0, orig_output.shape[1], (1,)).item()
            h = torch.randint(0, orig_output.shape[2], (1,)).item()
            w = torch.randint(0, orig_output.shape[3], (1,)).item()

        if 'RD_BFLIP' in inj_type:
            golden_val = orig_output[b, c, h, w].item()
            # Use the bit position passed from the main loop (already randomized there)
            _, perturb = get_bit_flip_perturbation(
                'default', precision, golden_val, layer_name, 'RD_BFLIP', quant_min_max, bit_position
            )
            modified_output[b, c, h, w] += perturb
        else:
            # Random replacement with appropriate precision
            if precision == 'fp32':
                random_bin = ''.join(str(torch.randint(0, 2, (1,)).item()) for _ in range(32))
                random_val = bin2fp32(random_bin)
            elif precision == 'fp16':
                random_bin = ''.join(str(torch.randint(0, 2, (1,)).item()) for _ in range(16))
                random_val = bin2fp16(random_bin)
            elif precision == 'int16':
                if quant_min_max is not None:
                    q_min, q_max = quant_min_max
                    granu = (q_max - q_min) / 65535
                    delta_int = torch.randint(0, 2**16, (1,)).item()
                    random_val = delta_int * granu + q_min
                else:
                    random_val = torch.randint(-32768, 32767 + 1, (1,)).item()
            elif precision == 'int8':
                if quant_min_max is not None:
                    q_min, q_max = quant_min_max
                    granu = (q_max - q_min) / 256
                    delta_int = torch.randint(0, 2**8, (1,)).item()
                    random_val = delta_int * granu + q_min
                else:
                    random_val = torch.randint(-128, 127 + 1, (1,)).item()
            
            modified_output[b, c, h, w] = random_val

    # Apply bounds
    modified_output = apply_precision_bounds(modified_output, precision, quant_min_max)

    # Return the modified output with the same structure as the original
    if is_tuple:
        return (modified_output,) + output[1:]
    else:
        return modified_output

def register_fault_hooks(model, inj_type, inj_layer, inj_pos=None, quant_min_max=None, precision='fp32', bit_position=None):
    """Register fault injection hooks based on injection type"""
    handles = []
    
    # Create appropriate hook based on injection type
    for name, module in model.named_modules():
        if name == inj_layer and (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
            if 'INPUT' in inj_type:
                print(f"registering input fault hook for layer {name}")
                handle = module.register_forward_hook(
                    lambda mod, inp, output, name=name: input_fault_hook(
                        mod, inp, output, name, precision, quant_min_max, inj_type, inj_pos, bit_position
                    )
                )
                handles.append(handle)
                break
            elif 'WEIGHT' in inj_type:
                print(f"registering weight fault hook for layer {name}")
                handle = module.register_forward_hook(
                    lambda mod, inp, output, name=name: weight_fault_hook(
                        mod, inp, output, name, precision, quant_min_max, inj_type, inj_pos, bit_position
                    )
                )
                handles.append(handle)
                break
            elif 'RD' in inj_type:
                handle = module.register_forward_hook(
                    lambda mod, inp, output, name=name: output_fault_hook(
                        mod, inp, output, name, precision, quant_min_max, inj_type, inj_pos, bit_position
                    )
                )
                handles.append(handle)
                break
    return handles

def remove_fault_hooks(model, handles):
    if handles is None:
        return
    for i, handle in enumerate(handles):
        handle.remove()
    handles.clear()