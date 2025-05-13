import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from inj_util import bin2fp32, fp322bin, bin2fp16, fp162bin
from inj_util import bin2int16, int162bin, bin2int8, int82bin
from inj_util import get_bit_flip_perturbation, perturb_conv, apply_precision_bounds





def input_fault_hook(module, inputs, output, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    """Hook function for input fault injection"""
    inp = inputs[0]
    weights = module.weight
    
    # Define stride and padding
    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
    groups = module.groups if hasattr(module, 'groups') else 1
    
    # Get positions to inject faults
    positions = []
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
    
    print("perturb: ",np.count_nonzero(inp_perturb))
    if np.count_nonzero(inp_perturb) > 0:
        nonzero_val = inp_perturb[inp_perturb != 0][0]
        print("Single nonzero perturb value:", nonzero_val)
        
    delta = perturb_conv(inp_perturb, weights, stride, padding, groups)
    if 'INPUT' == inj_type:
        deltaprint = delta.cpu().numpy()
        print("number of non-zero values in layer ",layer_name,"injection type ",inj_type,"is ",np.count_nonzero(deltaprint))
    
    # For INPUT16, modify the delta tensor to affect only 16 channels at one position
    if 'INPUT16' in inj_type:
        delta_16 = torch.zeros_like(delta)
        nonzero_positions = torch.nonzero(delta[0], as_tuple=True)
        idx = torch.randint(0, nonzero_positions[0].size(0), (1,)).item()
        c, h_pos, w_pos = nonzero_positions[0][idx], nonzero_positions[1][idx], nonzero_positions[2][idx]
        

        total_channels = delta.shape[1]
        if total_channels < 16:
            for i in range(total_channels):
                delta_16[0, i, h_pos, w_pos] = delta[0, i, h_pos, w_pos]
        else:
            max_start = total_channels - 16
            c_start = torch.randint(0, max_start + 1, (1,)).item()
            for i in range(16):
                delta_16[0, c_start + i, h_pos, w_pos] = delta[0, c_start + i, h_pos, w_pos]
       
        
        delta = delta_16
        if 'INPUT16' in inj_type:
            deltaprint = delta.cpu().numpy()
            print("number of non-zero values in layer ",layer_name,"injection type ",inj_type,"is ",np.count_nonzero(deltaprint))
        
    
    modified_output = output + delta
    modified_output = modified_output.float()
    modified_output = apply_precision_bounds(modified_output, precision, quant_min_max)
    return modified_output

def weight_fault_hook(module, inputs, output, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    """Hook function for weight fault injection"""
    inp = inputs[0]
    weights = module.weight
    
    # Define stride and padding
    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
    groups = module.groups if hasattr(module, 'groups') else 1
    
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
    print("perturb: ",perturb)
    # Apply perturbation
    wt_perturb[out_c, in_c, h, w] = perturb

    print("wt_perturb: ",np.count_nonzero(wt_perturb))
    # Propagate through convolution
    delta = perturb_conv(inp, wt_perturb, stride, padding, groups)
    print("delta before mask: ",np.count_nonzero(delta))
    # For WEIGHT16, modify the delta tensor to affect only 16 spatial positions in one channel
    if 'WEIGHT16' in inj_type:
        # Create a zero tensor with same shape as delta
        delta_16 = torch.zeros_like(delta)
        
        
        # Find non-zero indices by channel
        non_zero_by_channel = {}
        for c in range(delta.shape[1]):
            # Get linear indices of non-zero elements in this channel
            channel_mask = delta[0, c] != 0
            if torch.any(channel_mask):
                non_zero_positions = torch.nonzero(channel_mask, as_tuple=True)
                h_indices, w_indices = non_zero_positions
                # Store as (h, w) tuples
                positions = [(h.item(), w.item()) for h, w in zip(h_indices, w_indices)]
                non_zero_by_channel[c] = positions
        
        # Select a channel that has non-zero values
        valid_channels = list(non_zero_by_channel.keys())
        c = valid_channels[torch.randint(0, len(valid_channels), (1,)).item()]
        positions = non_zero_by_channel[c]
        
        # Determine how many positions to affect (random from 1-16, but limited by available positions)
        num_positions = min(torch.randint(1, 17, (1,)).item(), len(positions))
        
        # If we have fewer than 16 positions, use all of them
        if len(positions) <= 16:
            for h_pos, w_pos in positions:
                delta_16[0, c, h_pos, w_pos] = delta[0, c, h_pos, w_pos]
        else:
            # Choose a random starting point for contiguous positions
            max_start = len(positions) - num_positions
            start_pos = torch.randint(0, max_start + 1, (1,)).item()
            
            # Select num_positions contiguous positions
            for i in range(num_positions):
                h_pos, w_pos = positions[start_pos + i]
                delta_16[0, c, h_pos, w_pos] = delta[0, c, h_pos, w_pos]
        
        # Use this modified delta
        delta = delta_16
    print("delta: ",np.count_nonzero(delta))
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
    
    # Get positions to inject faults
    if inj_pos is not None and layer_name in inj_pos:
        b, c, h, w = inj_pos[layer_name][0] 
    else:
        # Single random position
        b = 0
        c = torch.randint(0, orig_output.shape[1], (1,)).item()
        h = torch.randint(0, orig_output.shape[2], (1,)).item()
        w = torch.randint(0, orig_output.shape[3], (1,)).item()

    

                
    if 'RD_BFLIP' in inj_type:
        # Bit flip fault - flip a specific bit in the original value
        golden_val = orig_output[b, c, h, w].item()
        _, perturb = get_bit_flip_perturbation(
            'default', precision, golden_val, layer_name, 'RD_BFLIP', quant_min_max, bit_position
        )
        modified_output[b, c, h, w] += perturb
    else:
        # Random replacement - generate a completely random value
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
        if name in inj_layer and isinstance(module, nn.Conv2d):
            if 'INPUT' in inj_type:
                # Important: Use a default parameter to capture the current value of 'name'
                handle = module.register_forward_hook(
                    lambda mod, inp, output, name=name: input_fault_hook(
                        mod, inp, output, name, precision, quant_min_max, inj_type, inj_pos, bit_position
                    )
                )
                handles.append(handle)
            elif 'WEIGHT' in inj_type:
                handle = module.register_forward_hook(
                    lambda mod, inp, output, name=name: weight_fault_hook(
                        mod, inp, output, name, precision, quant_min_max, inj_type, inj_pos, bit_position
                    )
                )
                handles.append(handle)
            elif 'RD' in inj_type:
                handle = module.register_forward_hook(
                    lambda mod, inp, output, name=name: output_fault_hook(
                        mod, inp, output, name, precision, quant_min_max, inj_type, inj_pos, bit_position
                    )
                )
                handles.append(handle)
            
    return handles

def remove_fault_hooks(model, handles):
    if handles is None:
        return
    for i, handle in enumerate(handles):
        handle.remove()
    handles.clear()