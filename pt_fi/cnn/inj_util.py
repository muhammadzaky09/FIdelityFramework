import sys
import random
import re
import numpy as np
import struct
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def bin2fp32(bin_str):
    """Convert 32-bit binary string to float32"""
    assert len(bin_str) == 32
    # Convert binary string to integer
    int_val = int(bin_str, 2)
    # Pack as unsigned int and unpack as float
    float_val = struct.unpack('!f', struct.pack('!I', int_val))[0]
    return float_val

def fp322bin(fp):
    """Convert float32 to 32-bit binary string"""
    # Pack float as binary and unpack as unsigned int
    int_val = struct.unpack('!I', struct.pack('!f', fp))[0]
    # Convert to binary string
    bin_str = bin(int_val)[2:].zfill(32)
    return bin_str

def bin2fp16(bin_str):
    assert len(bin_str) == 16
    sign_bin = bin_str[0]
    if sign_bin == '0':
        sign_val = 1.0
    else:
        sign_val = -1.0
    exponent_bin = bin_str[1:6]
    mantissa_bin = bin_str[6:]
    assert len(mantissa_bin) == 10
    exponent_val = int(exponent_bin,2)
    mantissa_val = 0.0
    for i in range(10):
        if mantissa_bin[i] == '1':
            mantissa_val += pow(2,-i-1)
    # Handling subnormal numbers
    if exponent_val == 0:
        return sign_val * pow(2,-14) * mantissa_val
    # Handling normal numbers
    else:
        value = sign_val * pow(2,exponent_val-15) * (1 + mantissa_val)
        # Handling NaNs and INFs
        if value == 65536:
            return 65535
        elif value == -65536:
            return -65535
        elif value > 65536 or value < -65536:
            return 0
        else:
            return value

def fp162bin(fp):
    sign = math.copysign(1,fp)
    abs_fp = abs(fp)
    # Handling subnormal numbers
    if abs_fp < pow(2,-14):
        target_fp = abs_fp * pow(2,14)
        exponent_bin = '00000'
        frac_bin = ''
        frac_mid = target_fp
        for i in range(25):
            frac_mid *= 2
            if frac_mid >= 1.0:
                frac_bin += '1'
                frac_mid -= 1.0
            else:
                frac_bin += '0'
        mantissa_bin = frac_bin
    # Handling normal numbers
    else:
        int_part = int(np.fix(abs_fp))
        frac_part = abs_fp - int_part
        int_bin = bin(int_part)[2:]
        frac_bin = ''
        frac_mid = frac_part
        for i in range(25):
            frac_mid *= 2
            if frac_mid >= 1.0:
                frac_bin += '1'
                frac_mid -= 1.0
            else:
                frac_bin += '0'
        int_frac_bin = int_bin + frac_bin
        # Decimal point is at the back of variable decimal_point
        decimal_point = len(int_bin)-1
        # Looking for the first 1
        first_one = int_frac_bin.find('1')
        # Special case: 0
        if first_one < 0:
            return ('0x00', '0x00')
        exponent_val = decimal_point - first_one + 15
        assert exponent_val <= 31
        assert exponent_val >= 0
        exponent_bin = bin(exponent_val)[2:].zfill(5)
        mantissa_bin = int_frac_bin[first_one+1:]
        if len(mantissa_bin) < 10:
            mantissa_bin = mantissa_bin.zfill(10)
    if sign == 1.0:
        sign_bin = '0'
    else:
        sign_bin = '1'
    total_bin = (sign_bin + exponent_bin + mantissa_bin)[:16]
    return total_bin

def bin2int16(text):
    assert len(text) == 16
    us_int = int(text,2)
    if us_int > 32767:
        return -(65536 - us_int)
    else:
        return us_int

def int162bin(val):
    assert val <= 32767 and val >= -32768
    if val < 0:
        us_val = 65536 + val
    else:
        us_val = val
    return bin(us_val)[2:].zfill(16)

def bin2int8(text):
    assert len(text) == 8
    us_int = int(text,2)
    if us_int > 127:
        return -(256 - us_int)
    else:
        return us_int

def int82bin(val):
    assert val <= 127 and val >= -128
    if val < 0:
        us_val = 256 + val
    else:
        us_val = val
    return bin(us_val)[2:].zfill(8)

def get_bit_flip_perturbation(precision, golden_d, layer, typ=None, quant_min_max=None, bit_position=None):
    # Convert tensor value to Python scalar if needed
    if isinstance(golden_d, torch.Tensor):
        golden_d = golden_d.item()

    if 'fp32' in precision:
        golden_b = fp322bin(golden_d)
        assert len(golden_b) == 32
        flip_bit = bit_position
        if golden_b[31-flip_bit] == '1':
            inj_b = golden_b[:31-flip_bit] + '0' + golden_b[31-flip_bit+1:]
        else:
            inj_b = golden_b[:31-flip_bit] + '1' + golden_b[31-flip_bit+1:]
        inj_d = bin2fp32(inj_b)
        perturb = inj_d - golden_d
    elif 'fp16' in precision:
        golden_b = fp162bin(golden_d)
        assert len(golden_b) == 16
        flip_bit = bit_position
        if golden_b[15-flip_bit] == '1':
            inj_b = golden_b[:15-flip_bit] + '0' + golden_b[15-flip_bit+1:]
        else:
            inj_b = golden_b[:15-flip_bit] + '1' + golden_b[15-flip_bit+1:]
        inj_d = bin2fp16(inj_b)
        perturb = inj_d - golden_d
    elif 'int16' in precision:
        q_min, q_max = quant_min_max
        granu = (q_max - q_min)/65535
        golden_b = int162bin(max(-32768,min(32767,int(round((golden_d - q_min)/granu)) - 32768)))
        assert len(golden_b) == 16
        flip_bit = bit_position
        if golden_b[15-flip_bit] == '1':
            inj_b = golden_b[:15-flip_bit] + '0' + golden_b[15-flip_bit+1:]
        else:
            inj_b = golden_b[:15-flip_bit] + '1' + golden_b[15-flip_bit+1:]
        inj_d = bin2int16(inj_b) + 32768
        perturb = (inj_d * granu + q_min) - golden_d
    elif 'int8' in precision:
        q_min, q_max = quant_min_max
        granu = (q_max - q_min)/256
        golden_b = int82bin(max(-128,min(127,int(round((golden_d - q_min)/granu)) - 128)))
        assert len(golden_b) == 8
        flip_bit = bit_position
        if golden_b[7-flip_bit] == '1':
            inj_b = golden_b[:7-flip_bit] + '0' + golden_b[7-flip_bit+1:]
        else:
            inj_b = golden_b[:7-flip_bit] + '1' + golden_b[7-flip_bit+1:]
        inj_d = bin2int8(inj_b) + 128
        perturb = (inj_d * granu + q_min) - golden_d
    else:
        print('Wrong precision!')
        exit(15)
    return flip_bit, perturb

def perturb_conv(inp, weight, stride, padding, groups=1):

    if inp.device != weight.device:
        weight = weight.to(inp.device)
    
    # Handle stride and padding formats
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    delta = F.conv2d(inp, weight, bias=None, stride=stride, padding=padding, groups=groups)
    
    return delta

def get_network_inj_type(precision, inj_type):
    assert precision in ['fp32', 'fp16', 'int16', 'int8']
    prec_dict = {
        'fp32': 'F32',
        'fp16': 'F16',
        'int16': 'I16',
        'int8': 'I8'
    }
    return inj_type + prec_dict[precision]

def apply_precision_bounds(tensor, precision, quant_min_max=None):
    if precision == 'fp32':
        # For fp32, just handle NaN/Inf cases
        result = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        return result
    elif precision == 'fp16':
        bounded = torch.clamp(tensor, min=torch.finfo(torch.float16).min, max=torch.finfo(torch.float16).max)
        result = torch.where(torch.isnan(bounded), torch.zeros_like(bounded), bounded)
        return result
    
    elif 'int' in precision and quant_min_max is not None:
        q_min, q_max = quant_min_max
        temp_tensor = tensor
        if torch.is_floating_point(tensor):
            is_nan = torch.isnan(tensor)
            is_posinf = torch.isposinf(tensor)
            is_neginf = torch.isneginf(tensor)

            temp_tensor = torch.where(is_nan, torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device), tensor)
            temp_tensor = torch.where(is_posinf, torch.tensor(q_max, dtype=tensor.dtype, device=tensor.device), temp_tensor)
            temp_tensor = torch.where(is_neginf, torch.tensor(q_min, dtype=tensor.dtype, device=tensor.device), temp_tensor)

        target_dtype = tensor.dtype if not torch.is_floating_point(tensor) else torch.int32 # Default target int if input was float
        return torch.clamp(temp_tensor, q_min, q_max).to(target_dtype)
    else:
        # Default case - just return the tensor
        return tensor


def calculate_conv_output_position(input_h, input_w, kernel_h, kernel_w, stride, padding, output_h, output_w):
    min_out_h = max(0, ((input_h + padding) // stride - kernel_h + 1))
    max_out_h = min(((input_h + padding) // stride + 1), output_h)
    
    min_out_w = max(0, ((input_w + padding) // stride - kernel_w + 1))
    max_out_w = min(((input_w + padding) // stride + 1), output_w)
    
    if min_out_h < max_out_h:
        start_h = torch.randint(min_out_h, max_out_h, (1,)).item()
    else:
        start_h = min_out_h
        
    if min_out_w < max_out_w:
        start_w = torch.randint(min_out_w, max_out_w, (1,)).item()
    else:
        start_w = min_out_w
    
    return start_h, start_w

# Helper functions to reduce code duplication
def extract_conv_params(module):
    """Extract conv2d parameters in a consistent way"""
    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
    groups = module.groups if hasattr(module, 'groups') else 1
    return stride, padding, groups

def get_conv_position(inp, inj_pos, layer_name):
    """Get or generate random position for conv layers"""
    if inj_pos is not None and layer_name in inj_pos:
        return inj_pos[layer_name][0]
    else:
        b = 0
        c = torch.randint(0, inp.shape[1], (1,)).item()
        h = torch.randint(0, inp.shape[2], (1,)).item()
        w = torch.randint(0, inp.shape[3], (1,)).item()
        return b, c, h, w

def get_linear_position(inp, inj_pos, layer_name):
    """Get or generate random position for linear layers"""
    if inj_pos is not None and layer_name in inj_pos:
        return inj_pos[layer_name][0]
    else:
        b = 0
        f = torch.randint(0, inp.shape[1] if len(inp.shape) == 2 else inp.numel() // inp.shape[0], (1,)).item()
        return b, f

def get_weight_conv_position(weights, inj_pos, layer_name):
    """Get or generate random position for conv weight injection"""
    if inj_pos is not None and layer_name in inj_pos:
        return inj_pos[layer_name][0]
    else:
        out_c = torch.randint(0, weights.shape[0], (1,)).item()
        in_c = torch.randint(0, weights.shape[1], (1,)).item()
        h = torch.randint(0, weights.shape[2], (1,)).item()
        w = torch.randint(0, weights.shape[3], (1,)).item()
        return out_c, in_c, h, w

def get_weight_linear_position(weights, inj_pos, layer_name):
    """Get or generate random position for linear weight injection"""
    if inj_pos is not None and layer_name in inj_pos:
        return inj_pos[layer_name][0]
    else:
        out_f = torch.randint(0, weights.shape[0], (1,)).item()
        in_f = torch.randint(0, weights.shape[1], (1,)).item()
        return out_f, in_f

def apply_input16_conv(delta, weights, inj_pos, layer_name, stride, padding):
    """Apply 16-channel logic for conv layers"""
    if np.count_nonzero(delta) == 0:
        return None
        
    delta_16 = torch.zeros_like(delta)
    
    if inj_pos is not None and layer_name in inj_pos:
        _, _, input_h, input_w = inj_pos[layer_name][0]
        kernel_h, kernel_w = weights.shape[2], weights.shape[3]
        output_h, output_w = delta.shape[2], delta.shape[3]
        
        start_h, start_w = calculate_conv_output_position(
            input_h, input_w, kernel_h, kernel_w, stride, padding, output_h, output_w
        )
    else:   
        start_h = torch.randint(0, delta.shape[2], (1,)).item()
        start_w = torch.randint(0, delta.shape[3], (1,)).item()
    
    total_channels = delta.shape[1]
    if total_channels >= 16:
        start_channel = torch.randint(0, total_channels - 15, (1,)).item()
        num_channels = 16
    else:
        start_channel = 0
        num_channels = total_channels
    
    for channel in range(num_channels):
        delta_16[0, start_channel + channel, start_h, start_w] = delta[0, start_channel + channel, start_h, start_w]
    
    return delta_16

def apply_input16_linear(delta):
    """Apply 16-feature logic for linear layers"""
    if np.count_nonzero(delta) == 0:
        return None
        
    delta_16 = torch.zeros_like(delta)
    total_features = delta.shape[1]
    max_start = max(0, total_features - 16)
    f_start = torch.randint(0, max_start + 1, (1,)).item()
    num_features = min(16, total_features)
    for i in range(num_features):
        delta_16[0, f_start + i] = delta[0, f_start + i]
    return delta_16

def apply_weight16_conv(delta, inj_pos, layer_name):
    """Apply 16-weight logic for conv layers"""
    if np.count_nonzero(delta) == 0:
        print("delta is all zeros")
        return None
        
    delta_16 = torch.zeros_like(delta)
    
    if inj_pos is not None and layer_name in inj_pos:
        start_channel = inj_pos[layer_name][0][0]  # out_c from weight fault
    else:
        start_channel = torch.randint(0, delta.shape[1], (1,)).item()
    
    dim_height, dim_width = delta.shape[2], delta.shape[3]
    total_positions = dim_height * dim_width
    
    if total_positions >= 16:
        start_position = torch.randint(0, total_positions // 16, (1,)).item()
        
        for inject_index in range(16):
            linear_pos = start_position * 16 + inject_index
            if linear_pos >= total_positions:
                break
            
            inject_height = linear_pos // dim_width
            inject_width = linear_pos % dim_width
            
            delta_16[0, start_channel, inject_height, inject_width] = delta[0, start_channel, inject_height, inject_width]
    else:
        for h in range(dim_height):
            for w in range(dim_width):
                delta_16[0, start_channel, h, w] = delta[0, start_channel, h, w]
    
    return delta_16

def apply_weight16_linear(delta):
    """Apply 16-weight logic for linear layers"""
    if torch.count_nonzero(delta) == 0:
        return None

    original_faulty_neuron = torch.nonzero(delta).flatten()
    if len(original_faulty_neuron) == 0:
        return None
        
    faulty_neuron_idx = original_faulty_neuron[0].item()
    fault_value = delta[0, faulty_neuron_idx].item()
    
    delta_16 = torch.zeros_like(delta)
    total_neurons = delta.shape[1]
    neurons_affected = 0
    
    current_neuron = faulty_neuron_idx
    while current_neuron < total_neurons and neurons_affected < 16:
        delta_16[0, current_neuron] = fault_value
        current_neuron += 16
        neurons_affected += 1
    
    return delta_16

def delta_init(precision):
    if precision == 'fp32':
        random_bin = ''.join(str(torch.randint(0, 2, (1,)).item()) for _ in range(32))
        return bin2fp32(random_bin)
    elif precision == 'fp16':
        random_bin = ''.join(str(torch.randint(0, 2, (1,)).item()) for _ in range(16))
        return bin2fp16(random_bin)
    elif precision == 'int16':
        return torch.randint(-32768, 32767 + 1, (1,)).item()
    elif precision == 'int8':
        return torch.randint(-128, 127 + 1, (1,)).item()

