import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

from inj_util import *

def log_to_file(message, filename="log.txt", mode="a"):
    with open(filename, mode) as f:
        f.write(f"{message}\n")


def input_fault_hook(module, inputs, output, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    """Hook function for input fault injection"""
    inp = inputs[0]
    weights = module.weight
    
    if isinstance(module, nn.Conv2d):
        stride, padding, groups = extract_conv_params(module)
        b, c, h, w = get_conv_position(inp, inj_pos, layer_name)
        
        inp_perturb = torch.zeros_like(inp)
        golden_val = inp[b, c, h, w].item()
        
        _, perturb = get_bit_flip_perturbation(
             precision, golden_val, layer_name, 'INPUT', quant_min_max, bit_position
        )
        
        inp_perturb[b, c, h, w] = perturb
        delta = perturb_conv(inp_perturb, weights, stride, padding, groups)
        
        if 'INPUT16' in inj_type:
            delta_16 = apply_input16_conv(delta, weights, inj_pos, layer_name, stride, padding)
            if delta_16 is None:
                return output
            delta = delta_16
        
    elif isinstance(module, nn.Linear):
        b, f = get_linear_position(inp, inj_pos, layer_name)
        
        inp_flat = inp if len(inp.shape) == 2 else inp.view(inp.shape[0], -1)
        inp_perturb = torch.zeros_like(inp_flat)
        
        golden_val = inp_flat[b, f].item()
        
        _, perturb = get_bit_flip_perturbation(
            precision, golden_val, layer_name, 'INPUT', quant_min_max, bit_position
        )
        
        inp_perturb[b, f] = perturb
        delta = F.linear(inp_perturb, weights, bias=None)
        
        if 'INPUT16' in inj_type:
            delta_16 = apply_input16_linear(delta)
            if delta_16 is None:
                return output
            delta = delta_16
    
    modified_output = output + delta
    modified_output = apply_precision_bounds(modified_output, precision, quant_min_max)

    return modified_output

def weight_fault_hook(module, inputs, output, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    inp = inputs[0]
    weights = module.weight
    
    if isinstance(module, nn.Conv2d):
        out_c, in_c, h, w = get_weight_conv_position(weights, inj_pos, layer_name)
        
        wt_perturb = torch.zeros_like(weights)
        golden_val = weights[out_c, in_c, h, w].item()
        
        _, perturb = get_bit_flip_perturbation(
             precision, golden_val, layer_name, 'WEIGHT', quant_min_max, bit_position
        )
        
        wt_perturb[out_c, in_c, h, w] = perturb
        
        stride, padding, groups = extract_conv_params(module)
        delta = perturb_conv(inp, wt_perturb, stride, padding, groups)
        
        if 'WEIGHT16' in inj_type:
            delta_16 = apply_weight16_conv(delta, inj_pos, layer_name)
            if delta_16 is None:
                return output
            delta = delta_16
    
    elif isinstance(module, nn.Linear):
        out_f, in_f = get_weight_linear_position(weights, inj_pos, layer_name)
        
        wt_perturb = torch.zeros_like(weights)
        golden_val = weights[out_f, in_f].item()
        
        _, perturb = get_bit_flip_perturbation(
             precision, golden_val, layer_name, 'WEIGHT', quant_min_max, bit_position
        )
        
        wt_perturb[out_f, in_f] = perturb
        
        reshaped_inp = inp.view(inp.size(0), -1)
        delta = F.linear(reshaped_inp, wt_perturb, bias=None)
        
        if 'WEIGHT16' in inj_type:
            delta_16 = apply_weight16_linear(delta)
            if delta_16 is None:
                return output
            delta = delta_16
    
    modified_output = output + delta
    modified_output = apply_precision_bounds(modified_output, precision, quant_min_max)
    
    return modified_output

def output_fault_hook(module, inputs, output, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    if isinstance(output, tuple):
        orig_output = output[0]
        is_tuple = True
    else:
        orig_output = output
        is_tuple = False
    
    modified_output = orig_output.clone()
    is_linear_output = len(orig_output.shape) == 2
    
    if is_linear_output:
        if inj_pos is not None and layer_name in inj_pos:
            b, f = inj_pos[layer_name][0]
        else:
            b = 0
            f = torch.randint(0, orig_output.shape[1], (1,)).item()
            
        if 'RD_BFLIP' in inj_type:
            golden_val = orig_output[b, f].item()
            _, perturb = get_bit_flip_perturbation(
                 precision, golden_val, layer_name, 'RD_BFLIP', quant_min_max, bit_position
            )
            modified_output[b, f] += perturb
        else:
            random_val = delta_init(precision)
            modified_output[b, f] = random_val
    else:
        if inj_pos is not None and layer_name in inj_pos:
            b, c, h, w = inj_pos[layer_name][0]
        else:
            b = 0
            c = torch.randint(0, orig_output.shape[1], (1,)).item()
            h = torch.randint(0, orig_output.shape[2], (1,)).item()
            w = torch.randint(0, orig_output.shape[3], (1,)).item()

        if 'RD_BFLIP' in inj_type:
            golden_val = orig_output[b, c, h, w].item()
            _, perturb = get_bit_flip_perturbation(
                 precision, golden_val, layer_name, 'RD_BFLIP', quant_min_max, bit_position
            )
            modified_output[b, c, h, w] += perturb
        else:
            random_val = delta_init(precision)
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