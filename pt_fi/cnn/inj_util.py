import sys
import random
import re
import numpy as np
import struct
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# Keep these functions unchanged as they're framework-agnostic
def bin2fp32(bin_str):
    assert len(bin_str) == 32
    data = struct.unpack('!f',struct.pack('!I', int(bin_str, 2)))[0]
    if np.isnan(data):
        return 0
    else:
        return data

def fp322bin(value):
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', value))

def bin2fp16(bin_str):
    # Keep existing implementation...
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
    # Keep existing implementation...
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

# Keep int8/int16 conversion functions unchanged
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

# Keep bit flip perturbation logic, with slight tensor adjustments
def get_bit_flip_perturbation(network, precision, golden_d, layer, typ=None, quant_min_max=None, bit_position=None):
    # Convert tensor value to Python scalar if needed
    if isinstance(golden_d, torch.Tensor):
        golden_d = golden_d.item()
        
    if 'fp32' in precision:
        golden_b = fp322bin(golden_d)
        assert len(golden_b) == 32
        

        # Use the bit_position passed from the experiment loop
        flip_bit = bit_position # This was the original variable name used for indexing
        
        # Actual index in the binary string (MSB is golden_b[0], LSB is golden_b[31])
        # string_idx_to_flip refers to the index in golden_b if bit_position means (0=LSB, 31=MSB)
        # The original code used `31-flip_bit` implying flip_bit was 0 for LSB, 31 for MSB.
        # If bit_position is 0 for LSB, 31 for MSB as per your loop.
        string_idx_to_flip = 31 - flip_bit

        original_char_at_flip_idx = golden_b[string_idx_to_flip]

        if original_char_at_flip_idx == '1':
            inj_b = golden_b[:string_idx_to_flip] + '0' + golden_b[string_idx_to_flip+1:]
        else:
            inj_b = golden_b[:string_idx_to_flip] + '1' + golden_b[string_idx_to_flip+1:]
        
        inj_d = bin2fp32(inj_b) # This will convert NaN to 0
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

# Modify delta_init for PyTorch
def delta_init(network, precision, layer, quant_min_max):
    if 'fp32' in precision:
        one_bin = ''
        for _ in range(32):
            one_bin += str(np.random.randint(0,2))
        return bin2fp32(one_bin)
    elif 'fp16' in precision:
        one_bin = ''
        for _ in range(16):
            one_bin += str(np.random.randint(0,2))
        return bin2fp16(one_bin)
    elif 'int8' in precision:
        quant_min, quant_max = quant_min_max
        delta_int = np.random.randint(0,pow(2,8))
        return delta_int * ((quant_max - quant_min) / 255) + quant_min
    elif 'int16' in precision:
        quant_min, quant_max = quant_min_max
        delta_int = np.random.randint(0,pow(2,16))
        return delta_int * ((quant_max - quant_min) / 65535) + quant_min

# Modified delta_generator for PyTorch tensors
def delta_generator(network, precision, inj_type, layer_list, layer_dim, quant_min_max=None):
    num_inj_per_layer = 1
    delta_set = {}
    inj_pos = {}
    inj_h_set = []
    inj_w_set = []
    inj_c_set = []
    
    if 'INPUT' not in inj_type and 'WEIGHT' not in inj_type and 'RD_BFLIP' not in inj_type:
        for layer in layer_list: 
            tup_set = []
            layer_delta_set = []
            # Adjust to PyTorch tensor dimensions - batch, channel, height, width (BCHW)
            _, max_c, max_h, max_w = layer_dim
            while len(tup_set) < num_inj_per_layer:
                # Position format adjusted for PyTorch tensors (c, h, w)
                tup = (random.randint(0, max_c-1), random.randint(0, max_h-1), random.randint(0, max_w-1))
                if tup not in tup_set:
                    tup_set.append(tup)
                    inj_c_set.append(tup[0])
                    inj_h_set.append(tup[1])
                    inj_w_set.append(tup[2])

                    # Initialize delta
                    delta_val = delta_init(network, precision, layer, quant_min_max)
                    layer_delta_set.append(delta_val)

            inj_pos[layer] = tup_set
            delta_set[layer] = layer_delta_set

    return delta_set, inj_pos

def perturb_conv(inp, weight, stride, padding, groups=1):
    """
    Propagate a perturbation through convolution.
    """
    # Ensure inputs are tensors with proper device
    if not isinstance(inp, torch.Tensor):
        inp = torch.tensor(inp, dtype=torch.float32)
    if not isinstance(weight, torch.Tensor):
        weight = torch.tensor(weight, dtype=torch.float32)
    
    # Ensure weight and input are on the same device
    if inp.device != weight.device:
        weight = weight.to(inp.device)
    
    # Handle stride and padding formats
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    # Perform convolution
    delta = F.conv2d(inp.double(), weight.double(), bias=None, stride=stride, padding=padding, groups=groups)
    
    return delta

# Function to get appropriate injection type string for PyTorch
def get_network_inj_type(precision, inj_type):
    assert precision in ['fp32', 'fp16', 'int16', 'int8']
    prec_dict = {
        'fp32': 'F32',
        'fp16': 'F16',
        'int16': 'I16',
        'int8': 'I8'
    }
    return inj_type + prec_dict[precision]

# New PyTorch-specific function to find all Conv2d modules in a model
def get_pytorch_conv_layers(model):
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layers[name] = module
    return layers

def apply_precision_bounds(tensor, precision, quant_min_max=None):
    """
    Apply precision-specific bounds to tensor and handle NaN values.
    Modified for FP32 to allow NaN/Inf propagation like TensorFlow default behavior.
    """
    if precision == 'fp32':
        # For FP32, allow NaN and Inf to propagate to better match TensorFlow's default op behavior.
        # PyTorch operations will handle these values according to IEEE 754 standards.
        return tensor 
    elif precision == 'fp16':
        # Original FP16 handling: clamp Inf to max/min_fp16, then NaN to 0.
        # To be more TF-like for FP16 (propagate NaN/Inf):
        # return tensor # Option 1: Fully TF-like, let NaN/Inf propagate
        # Option 2: Keep original clamping for FP16 for now, as it's a smaller range.
        bounded = torch.clamp(tensor, min=torch.finfo(torch.float16).min, max=torch.finfo(torch.float16).max)
        result = torch.where(torch.isnan(bounded), torch.zeros_like(bounded), bounded)
        return result
    elif 'int' in precision and quant_min_max is not None:
        q_min, q_max = quant_min_max
        # For integer types, inputs ideally should not be NaN/Inf.
        # If they are, casting to int is problematic. Defaulting NaN to 0 and clamping Inf.
        # This is a reasonable approach for int conversion.
        temp_tensor = tensor
        if torch.is_floating_point(tensor):
            # Ensure finite values before int clamping/casting if coming from float
            # Replace NaN with 0, Inf with int range bounds (q_max, q_min)
            # This attempts to make problematic floats convertible to int
            is_nan = torch.isnan(tensor)
            is_posinf = torch.isposinf(tensor)
            is_neginf = torch.isneginf(tensor)

            temp_tensor = torch.where(is_nan, torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device), tensor)
            temp_tensor = torch.where(is_posinf, torch.tensor(q_max, dtype=tensor.dtype, device=tensor.device), temp_tensor)
            temp_tensor = torch.where(is_neginf, torch.tensor(q_min, dtype=tensor.dtype, device=tensor.device), temp_tensor)
        
        # Clamp and cast to the original integer tensor's dtype if it was int, or a default int otherwise
        # If original tensor was float, this will cast to a default int type (e.g. torch.int64 for .int())
        # To preserve specific int type if tensor was already int but somehow went through float path:
        target_dtype = tensor.dtype if not torch.is_floating_point(tensor) else torch.int32 # Default target int if input was float
        return torch.clamp(temp_tensor, q_min, q_max).to(target_dtype)
    else:
        # If no specific precision rule (e.g. already int and no quant_min_max)
        return tensor