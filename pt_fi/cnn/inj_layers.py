import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from inj_util import bin2fp32, fp322bin, bin2fp16, fp162bin
from inj_util import bin2int16, int162bin, bin2int8, int82bin
from inj_util import get_bit_flip_perturbation, perturb_conv

def register_fault_hooks(model, inj_type, inj_layer, inj_pos=None, quant_min_max=None, precision='fp32', bit_position=None):
    """
    Register appropriate hooks to model layers based on injection type - FIXED VERSION
    
    This corrected version registers both pre-hooks (for calculation) and forward hooks
    (for application) for INPUT and WEIGHT fault models.
    """
    handles = []
    
    if not hasattr(model, 'delta_values'):
        model.delta_values = {}
    
    # Define hook creator functions that properly capture all parameters
    def create_input_hook(name, precision, quant_min_max, inj_type, inj_pos, bit_pos):
        """Create an input injection hook with fixed parameters"""
        def hook(mod, inp):
            return input_injection_hook(mod, inp, name, precision, quant_min_max, inj_type, inj_pos, bit_pos)
        return hook
    
    def create_weight_hook(name, precision, quant_min_max, inj_type, inj_pos, bit_pos):
        """Create a weight injection hook with fixed parameters"""
        def hook(mod, inp):
            return weight_injection_hook(mod, inp, name, precision, quant_min_max, inj_type, inj_pos, bit_pos)
        return hook
    
    def create_output_hook(name, precision, quant_min_max, inj_type, inj_pos, bit_pos):
        """Create an output injection hook with fixed parameters"""
        def hook(mod, inp, output):
            return output_injection_hook(mod, inp, output, name, precision, quant_min_max, inj_type, inj_pos, bit_pos)
        return hook
    
    def create_propagation_hook(name):
        """Create a hook to apply the propagated delta to the output"""
        def hook(mod, inp, output):
            if hasattr(mod, 'propagated_delta'):
                if isinstance(output, tuple):
                    modified_out = output[0] + mod.propagated_delta
                    
                    # Apply bounding based on precision
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
                    modified_out = output + mod.propagated_delta
                    
                    # Apply bounding based on precision
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
        return hook  # FIXED: return the hook function, not 'output'
    
    for name, module in model.named_modules():
        if name in inj_layer:
            if 'INPUT' in inj_type:
                # Register pre-forward hook for input injection calculation
                handle = module.register_forward_pre_hook(
                    create_input_hook(name, precision, quant_min_max, inj_type, inj_pos, bit_position)
                )
                handles.append(handle)
                
                # ALSO register forward hook to apply the calculated perturbation
                handle = module.register_forward_hook(
                    create_propagation_hook(name)
                )
                handles.append(handle)
            
            elif 'WEIGHT' in inj_type:
                # Register pre-forward hook for weight injection calculation
                handle = module.register_forward_pre_hook(
                    create_weight_hook(name, precision, quant_min_max, inj_type, inj_pos, bit_position)
                )
                handles.append(handle)
                
                # ALSO register forward hook to apply the calculated perturbation
                handle = module.register_forward_hook(
                    create_propagation_hook(name)
                )
                handles.append(handle)
            
            elif 'RD_BFLIP' in inj_type or 'RD' in inj_type:
                # Register forward hook for direct output injection
                handle = module.register_forward_hook(
                    create_output_hook(name, precision, quant_min_max, inj_type, inj_pos, bit_position)
                )
                handles.append(handle)
    
    return handles

def input_injection_hook(module, inputs, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    """Hook for calculating input perturbations"""
    
    inp = inputs[0]
    
    if not isinstance(module, nn.Conv2d):
        return inputs
    
    weights = module.weight.data
    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
    same_padding = padding > 0
    
    if not hasattr(module, 'propagated_delta'):
        if 'INPUT16' in inj_type:
            b, c, h, w = inp.shape
            
            if inj_pos is None or layer_name not in inj_pos:
                batch_idx = 0
                
                # FIXED: Handle case where channel count < 16
                # If we have fewer than 16 channels, just start at 0
                if c < 16:
                    start_channel = 0
                    max_channels = c  # Only use as many channels as available
                else:
                    max_start_channel = max(0, c - 16)
                    start_channel = torch.randint(0, max_start_channel + 1, (1,)).item()
                    max_channels = 16  # Can use all 16 channels
                
                h_pos = torch.randint(0, h, (1,)).item()
                w_pos = torch.randint(0, w, (1,)).item()
                
                inp_perturb = torch.zeros_like(inp)
                
                # Only loop through as many channels as we can
                for i in range(max_channels):
                    channel_idx = start_channel + i
                    if channel_idx >= c:  # Double check to be safe
                        break
                        
                    golden_val = inp[batch_idx, channel_idx, h_pos, w_pos].item()
                    
                    _, perturb = get_bit_flip_perturbation(
                        'default', precision, golden_val, layer_name, 'INPUT', quant_min_max, bit_position
                    )
                    
                    inp_perturb[batch_idx, channel_idx, h_pos, w_pos] = perturb
            else:
                positions = inj_pos[layer_name]
                inp_perturb = torch.zeros_like(inp)
                
                for pos in positions:
                    b, c, h, w = pos
                    if c >= inp.shape[1]:  # Check bounds
                        continue
                    golden_val = inp[b, c, h, w].item()
                    
                    _, perturb = get_bit_flip_perturbation(
                        'default', precision, golden_val, layer_name, 'INPUT', quant_min_max, bit_position
                    )
                    
                    inp_perturb[b, c, h, w] = perturb
            
            propagated_delta = perturb_conv(
                inp_perturb, weights, stride, same_padding
            )
            
        else:  # Regular INPUT fault model
            if inj_pos is None or layer_name not in inj_pos:
                b = 0  
                c = torch.randint(0, inp.shape[1], (1,)).item()
                h = torch.randint(0, inp.shape[2], (1,)).item()
                w = torch.randint(0, inp.shape[3], (1,)).item()
                positions = [(b, c, h, w)]
            else:
                positions = inj_pos[layer_name]
            
            inp_perturb = torch.zeros_like(inp)
            
            for pos in positions:
                b, c, h, w = pos
                # Bounds check
                if c >= inp.shape[1] or h >= inp.shape[2] or w >= inp.shape[3]:
                    continue
                
                golden_val = inp[b, c, h, w].item()
                
                _, perturb = get_bit_flip_perturbation(
                    'default', precision, golden_val, layer_name, 'INPUT', quant_min_max, bit_position
                )
                
                inp_perturb[b, c, h, w] = perturb
            
            propagated_delta = perturb_conv(
                inp_perturb, weights, stride, same_padding
            )
        
        module.propagated_delta = propagated_delta
    
    return inputs  # Return unmodified inputs - perturbations applied in forward hook

def weight_injection_hook(module, inputs, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    """Hook for calculating weight perturbations"""
    
    inp = inputs[0]
    
    if not isinstance(module, nn.Conv2d):
        return inputs
    
    if not hasattr(module, 'original_weight'):
        module.original_weight = module.weight.data.clone()
    
    weights = module.weight.data
    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
    same_padding = padding > 0
    
    if not hasattr(module, 'propagated_delta'):
        if 'WEIGHT16' in inj_type:
            out_c, in_c, k_h, k_w = weights.shape
            wt_perturb = torch.zeros_like(weights)
            if inj_pos is None or layer_name not in inj_pos:
                out_channel = torch.randint(0, out_c, (1,)).item()
                
                # FIXED: Handle case where input channels < 16
                # Check if we have enough input channels
                if in_c < 16:
                    in_channel = 0  # Start at the first one
                    spatial_elements = k_h * k_w
                    # If we don't have 16 input channels, we'll use what we have
                    # and distribute the rest across spatial elements
                    num_spatial_elements = min(16 // in_c + (16 % in_c > 0), spatial_elements)
                    
                    count = 0
                    for ic in range(in_c):
                        for i in range(min(num_spatial_elements, 16 - count)):
                            if count >= 16:
                                break
                                
                            spatial_idx = i
                            h_idx = spatial_idx // k_w
                            w_idx = spatial_idx % k_w
                            
                            if h_idx >= k_h or w_idx >= k_w:
                                continue
                                
                            golden_val = weights[out_channel, ic, h_idx, w_idx].item()
                            
                            _, perturb = get_bit_flip_perturbation(
                                'default', precision, golden_val, layer_name, 'WEIGHT', quant_min_max, bit_position
                            )
                            
                            wt_perturb[out_channel, ic, h_idx, w_idx] = perturb
                            count += 1
                else:
                    # We have enough input channels, just pick one and do 16 spatial locations
                    in_channel = torch.randint(0, in_c, (1,)).item()
                    spatial_elements = k_h * k_w
                    
                    for i in range(min(16, spatial_elements)):
                        spatial_idx = i
                        h_idx = spatial_idx // k_w
                        w_idx = spatial_idx % k_w
                        
                        golden_val = weights[out_channel, in_channel, h_idx, w_idx].item()
                        
                        _, perturb = get_bit_flip_perturbation(
                            'default', precision, golden_val, layer_name, 'WEIGHT', quant_min_max, bit_position
                        )
                        
                        wt_perturb[out_channel, in_channel, h_idx, w_idx] = perturb
            else:
                positions = inj_pos[layer_name]
                
                for pos in positions:
                    o, i, h, w = pos
                    # Bounds check
                    if o >= weights.shape[0] or i >= weights.shape[1] or h >= weights.shape[2] or w >= weights.shape[3]:
                        continue
                        
                    golden_val = weights[o, i, h, w].item()
                    
                    _, perturb = get_bit_flip_perturbation(
                        'default', precision, golden_val, layer_name, 'WEIGHT', quant_min_max, bit_position
                    )
                    wt_perturb[o, i, h, w] = perturb
        
            propagated_delta = perturb_conv(
                inp, wt_perturb, stride, same_padding
            )
            
        else:  # Regular WEIGHT fault model
           
            if inj_pos is None or layer_name not in inj_pos:
                out_c = torch.randint(0, weights.shape[0], (1,)).item()
                in_c = torch.randint(0, weights.shape[1], (1,)).item()
                k_h = torch.randint(0, weights.shape[2], (1,)).item()
                k_w = torch.randint(0, weights.shape[3], (1,)).item()
                positions = [(out_c, in_c, k_h, k_w)]
            else:
                positions = inj_pos[layer_name]
            
            wt_perturb = torch.zeros_like(weights)
            
            for pos in positions:
                o, i, h, w = pos
                # Bounds check
                if o >= weights.shape[0] or i >= weights.shape[1] or h >= weights.shape[2] or w >= weights.shape[3]:
                    continue
                    
                golden_val = weights[o, i, h, w].item()
                
                _, perturb = get_bit_flip_perturbation(
                    'default', precision, golden_val, layer_name, 'WEIGHT', quant_min_max, bit_position
                )
                
                wt_perturb[o, i, h, w] = perturb
            
            propagated_delta = perturb_conv(
                inp, wt_perturb, stride, same_padding
            )
        
        module.propagated_delta = propagated_delta
    
    return inputs

def output_injection_hook(module, inputs, output, layer_name, precision, quant_min_max, inj_type, inj_pos, bit_position):
    """Hook for injecting faults directly into output tensors (for RD and RD_BFLIP)"""
    
    # Only handle RD and RD_BFLIP in this hook
    if 'RD_BFLIP' in inj_type:
        if not hasattr(module, 'output_delta'):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            
            if inj_pos is None or layer_name not in inj_pos:
                b = 0  
                c = torch.randint(0, out.shape[1], (1,)).item()
                h = torch.randint(0, out.shape[2], (1,)).item()
                w = torch.randint(0, out.shape[3], (1,)).item()
                positions = [(b, c, h, w)]
            else:
                positions = inj_pos[layer_name]
            
            delta = torch.zeros_like(out)
            
            for pos in positions:
                b, c, h, w = pos
                golden_val = out[b, c, h, w].item()
                
                _, perturb = get_bit_flip_perturbation(
                    'default', precision, golden_val, layer_name, 'RD_BFLIP', quant_min_max, bit_position
                )
                
                delta[b, c, h, w] = perturb
            
            module.output_delta = delta
        
        if isinstance(output, tuple):
            modified_out = output[0] + module.output_delta
            
            if precision == 'fp32':
                modified_out = torch.clamp(modified_out, -3.402823e38, 3.402823e38)
            elif precision == 'fp16':
                modified_out = torch.clamp(modified_out, -65504, 65504)
            elif precision == 'int16' or precision == 'int8':
                if quant_min_max is not None:
                    q_min, q_max = quant_min_max
                    modified_out = torch.clamp(modified_out, q_min, q_max)
            
            modified_out = torch.where(torch.isnan(modified_out), 
                                      torch.zeros_like(modified_out), 
                                      modified_out)
            
            return (modified_out,) + output[1:]
        else:
            modified_out = output + module.output_delta
            
            if precision == 'fp32':
                modified_out = torch.clamp(modified_out, -3.402823e38, 3.402823e38)
            elif precision == 'fp16':
                modified_out = torch.clamp(modified_out, -65504, 65504)
            elif precision == 'int16' or precision == 'int8':
                if quant_min_max is not None:
                    q_min, q_max = quant_min_max
                    modified_out = torch.clamp(modified_out, q_min, q_max)
            
            modified_out = torch.where(torch.isnan(modified_out), 
                                      torch.zeros_like(modified_out), 
                                      modified_out)
            
            return modified_out
    
    elif 'RD' in inj_type:
        if not hasattr(module, 'output_random_values'):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            
            if inj_pos is None or layer_name not in inj_pos:
                b = 0  # Assuming we inject in first batch item
                c = torch.randint(0, out.shape[1], (1,)).item()
                h = torch.randint(0, out.shape[2], (1,)).item()
                w = torch.randint(0, out.shape[3], (1,)).item()
                positions = [(b, c, h, w)]
            else:
                positions = inj_pos[layer_name]
            
            # Create a tensor of random values
            random_values = torch.zeros_like(out)
            mask = torch.zeros_like(out, dtype=torch.bool)
            
            for pos in positions:
                b, c, h, w = pos
                
                # Generate random value based on precision
                if precision == 'fp32':
                    # Random 32-bit binary string to fp32
                    random_bin = ''.join(str(torch.randint(0, 2, (1,)).item()) for _ in range(32))
                    # Convert to float (using our utility function)
                    random_val = bin2fp32(random_bin)
                elif precision == 'fp16':
                    # Random 16-bit binary string to fp16
                    random_bin = ''.join(str(torch.randint(0, 2, (1,)).item()) for _ in range(16))
                    # Convert to float (using our utility function)
                    random_val = bin2fp16(random_bin)
                elif precision == 'int16':
                    # Random integer within quantization range
                    if quant_min_max is not None:
                        q_min, q_max = quant_min_max
                        granu = (q_max - q_min) / 65535
                        delta_int = torch.randint(0, 2**16, (1,)).item()
                        random_val = delta_int * granu + q_min
                    else:
                        # Default int16 range
                        random_val = torch.randint(-32768, 32767 + 1, (1,)).item()
                elif precision == 'int8':
                    if quant_min_max is not None:
                        q_min, q_max = quant_min_max
                        granu = (q_max - q_min) / 256
                        delta_int = torch.randint(0, 2**8, (1,)).item()
                        random_val = delta_int * granu + q_min
                    else:
                        # Default int8 range
                        random_val = torch.randint(-128, 127 + 1, (1,)).item()
                
                # Store random value
                random_values[b, c, h, w] = random_val
                # Mark this position in the mask
                mask[b, c, h, w] = True
            
            module.output_random_values = random_values
            module.output_random_mask = mask
        
        if isinstance(output, tuple):
            # Create a copy of the original output
            modified_out = output[0].clone()
            # Replace values at masked positions with stored random values
            modified_out = torch.where(module.output_random_mask, 
                                       module.output_random_values, 
                                       modified_out)
            
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
            modified_out = output.clone()
            modified_out = torch.where(module.output_random_mask, 
                                       module.output_random_values, 
                                       modified_out)
            
            if precision == 'fp32':
                modified_out = torch.clamp(modified_out, -3.402823e38, 3.402823e38)
            elif precision == 'fp16':
                modified_out = torch.clamp(modified_out, -65504, 65504)
            elif precision == 'int16' or precision == 'int8':
                if quant_min_max is not None:
                    q_min, q_max = quant_min_max
                    modified_out = torch.clamp(modified_out, q_min, q_max)
                    
            modified_out = torch.where(torch.isnan(modified_out), 
                                      torch.zeros_like(modified_out), 
                                      modified_out)
            
            return modified_out
    
    # If not RD or RD_BFLIP, return original output
    return output

# Function to clean up hooks after testing
def remove_fault_hooks(model, handles):
    """Remove all registered hooks and clean up attributes"""
    for handle in handles:
        handle.remove()
    
    # Restore original weights if modified and clean up attributes
    for name, module in model.named_modules():
        # Clean up weight-related attributes
        if hasattr(module, 'original_weight'):
            module.weight.data = module.original_weight
            delattr(module, 'original_weight')
        
        if hasattr(module, 'weight_delta'):
            delattr(module, 'weight_delta')
        
        # Clean up input-related attributes
        if hasattr(module, 'input_delta'):
            delattr(module, 'input_delta')
        
        # Clean up output-related attributes
        if hasattr(module, 'output_delta'):
            delattr(module, 'output_delta')
            
        # Clean up propagated delta
        if hasattr(module, 'propagated_delta'):
            delattr(module, 'propagated_delta')
        
        # Clean up RD-specific attributes
        if hasattr(module, 'output_random_values'):
            delattr(module, 'output_random_values')
            
        if hasattr(module, 'output_random_mask'):
            delattr(module, 'output_random_mask')
            
        # Clean up any other attributes we might have added
        if hasattr(module, 'fault_calculated'):
            delattr(module, 'fault_calculated')