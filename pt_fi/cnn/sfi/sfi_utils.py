import numpy as np

def convert_fp16_to_bits(weights: np.ndarray) -> list:
    weight_bits = weights.view(np.uint16).tolist()
    return weight_bits

def convert_int8_to_bits(weights: np.ndarray) -> list:
    weight_bits = []
    for weight in weights:
        if weight < 0:
            bits_int = (256 + weight) & 0xFF
        else:
            bits_int = weight & 0xFF
        weight_bits.append(bits_int)
    return weight_bits

def convert_bits_to_fp16(bits: int) -> float:
    return np.uint16(bits).view(np.float16).item()

def convert_bits_to_int8(bits: int) -> int:
    if bits > 127:
        return bits - 256
    else:
        return bits