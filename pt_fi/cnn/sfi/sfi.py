"""
Paper: Assessing Convolutional Neural Networks
Reliability through Statistical Fault Injections

Method: Data-aware statistical fault injection

Authors: Ruospo et al.
Published in: 2023
IEEE Design, Automation, and Test in Europe Conference (DATE)
"""

import torch
import numpy
from typing import Tuple, List, Dict, Any
from collections import defaultdict
import math
from sfi_utils import *

def count_bit_frequency(weights: np.ndarray, precision: str) -> Dict[int, Dict[str, int]]:
    
    
    if precision == 'fp16':
        bits = 16
        weight_bins = convert_fp16_to_bits(weights.flatten())
    elif precision == 'int8':
        bits = 8
        weight_bins = convert_int8_to_bits(weights.flatten())
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use 'fp16' or 'int8'")
    
    bit_patterns = {}
    
    for bit_pos in range(bits):
        f0_count = 0 
        f1_count = 0  
        
        for weight_bin in weight_bins:
            bit_value = (weight_bin >> bit_pos) & 1
            if bit_value == 0:
                f0_count += 1
            else:
                f1_count += 1
        
        bit_patterns[bit_pos] = {
            'f0': f0_count,
            'f1': f1_count
        }
    
    return bit_patterns

def calc_bitflip_distance(weights: np.ndarray, precision: str) -> Dict[int, Dict[str, float]]:
    if precision == 'fp16':
        bits = 16
        single_convert = convert_fp16_to_bits
        bits_to_value = convert_bits_to_fp16
    elif precision == 'int8':
        bits = 8
        single_convert = convert_int8_to_bits
        bits_to_value = convert_bits_to_int8
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use 'fp16' or 'int8'")
    
    weights_flat = weights.flatten()
    
    running_means = {}
    for bit_pos in range(bits):
        running_means[bit_pos] = {
            'mean_0_1': 0.0,    
            'mean_1_0': 0.0,    
            'count_0_1': 0,     
            'count_1_0': 0      
        }
    
    for weight in weights_flat:
        weight_bin = single_convert(weight)
        
        for bit_pos in range(bits):
            current_bit = (weight_bin >> bit_pos) & 1
            
       
            if current_bit == 0:
                
                flipped_bits = weight_bin | (1 << bit_pos)
                flipped_weight = bits_to_value(flipped_bits)
                distance = abs(float(weight) - float(flipped_weight))
                
                
                tracker = running_means[bit_pos]
                count = tracker['count_0_1']
                old_mean = tracker['mean_0_1']
                
                
                new_mean = old_mean + (distance - old_mean) / (count + 1)
                
                tracker['mean_0_1'] = new_mean
                tracker['count_0_1'] = count + 1
                
            else:
                
                flipped_bits = weight_bin & ~(1 << bit_pos)
                flipped_weight = bits_to_value(flipped_bits)
                distance = abs(float(weight) - float(flipped_weight))
                
                
                tracker = running_means[bit_pos]
                count = tracker['count_1_0']
                old_mean = tracker['mean_1_0']
                
                
                new_mean = old_mean + (distance - old_mean) / (count + 1)
                
                tracker['mean_1_0'] = new_mean
                tracker['count_1_0'] = count + 1
    
    
    distances = {}
    for bit_pos in range(bits):
        distances[bit_pos] = {
            'D0_1': running_means[bit_pos]['mean_0_1'],
            'D1_0': running_means[bit_pos]['mean_1_0']
        }
    
    return distances

def compute_criticality_probabilities(bit_patterns: Dict[int, Dict[str, int]], 
                                    distances: Dict[int, Dict[str, float]],
                                    precision: str) -> Dict[int, float]:
    
    bits = 16 if precision == 'fp16' else 8
    
    
    davg_values = {}
    
    for bit_pos in range(bits):
        f0 = bit_patterns[bit_pos]['f0']
        f1 = bit_patterns[bit_pos]['f1']
        total_weights = f0 + f1
        
        if total_weights == 0:
            davg_values[bit_pos] = 0.0
            continue
        
        
        f0_prob = f0 / total_weights
        f1_prob = f1 / total_weights
        
        
        davg = (distances[bit_pos]['D0_1'] * f0_prob + 
                distances[bit_pos]['D1_0'] * f1_prob)
        davg_values[bit_pos] = davg
    
    
    all_davg = list(davg_values.values())
    davg_min = min(all_davg)
    davg_max = max(all_davg)
    
    probabilities = {}
    
    for bit_pos in range(bits):
        if davg_max == davg_min:
            p_i = 0.5
        else:
            
            p_i = 0.0 + (davg_values[bit_pos] - davg_min) * 0.5 / (davg_max - davg_min)
        
        probabilities[bit_pos] = p_i
    
    return probabilities

def generate_sfi_plan(model_weights: Dict[str, np.ndarray], 
                     precision: str,
                     error_margin: float = 0.01,
                     confidence_level: float = 0.99) -> Dict: 
    
    all_weights = []
    layer_info = []
    
    for layer_name, weights in model_weights.items():
        all_weights.append(weights.flatten())
        layer_info.append({
            'name': layer_name,
            'shape': weights.shape,
            'num_params': weights.size
        })
        print(f"Layer {layer_name}: {weights.shape} ({weights.size:,} parameters)")
    
    combined_weights = np.concatenate(all_weights)
    bit_patterns = count_bit_frequency(combined_weights, precision)
    distances = calc_bitflip_distance(combined_weights, precision)
    probabilities = compute_criticality_probabilities(bit_patterns, distances, precision)
    

    bits = 16 if precision == 'fp16' else 8
    t_value = 2.576 if confidence_level == 0.99 else 1.96 
    
    total_sample_size = 0
    layer_samples = []
    
    for layer in layer_info:
        layer_samples_dict = {}
        layer_total = 0
        
        for bit_pos in range(bits):
            N_il = layer['num_params'] * 2 
            print(f"N_il: {N_il}")
            
            p = probabilities[bit_pos]
            e = error_margin
            
  
            if p == 0 or p == 1:
                n_il = min(10, N_il)
            else:
                denominator = 1 + (e**2 * (N_il - 1)) / (t_value**2 * p * (1 - p))
                
                n_il = max(1, int(np.ceil(N_il / denominator)))
                n_il = min(n_il, N_il) 
            
            layer_samples_dict[bit_pos] = n_il
            layer_total += n_il
        
        layer_samples.append({
            'layer_name': layer['name'],
            'layer_shape': layer['shape'],
            'total_params': layer['num_params'],
            'samples_per_bit': layer_samples_dict,
            'total_samples': layer_total
        })
        total_sample_size += layer_total
    
    total_exhaustive = sum(layer['num_params'] for layer in layer_info) * 2 * bits
    reduction_percentage = (1 - total_sample_size / total_exhaustive) * 100
    
    plan = {
        'precision': precision,
        'total_parameters': len(combined_weights),
        'total_sample_size': total_sample_size,
        'total_exhaustive_size': total_exhaustive,
        'reduction_percentage': reduction_percentage,
        'error_margin': error_margin,
        'confidence_level': confidence_level,
        'bit_patterns': bit_patterns,
        'distances': distances,
        'probabilities': probabilities,
        'layer_samples': layer_samples
    }
    
    return plan

def main():
    model_weights = {
        'conv1': np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float16),
        'conv2': np.array([5.0, 6.0, 7.0, 8.0]).astype(np.float16)
    }
    precision = 'fp16'
    error_margin = 0.01
    confidence_level = 0.99

    plan = generate_sfi_plan(model_weights, precision, error_margin, confidence_level)
    print(plan)

if __name__ == "__main__":
    main()
            
            
        
        
        
