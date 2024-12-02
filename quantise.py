import torch
import torch.nn as nn

def quantise_model(model, bits):
    """
    Quantize a single model using specified number of bits per weight.
    Each row of model.W is quantized independently. Bias is copied without quantization.
    """
    # Calculate number of levels from bits
    num_levels = 2 ** bits
    
    # Create a copy of the model to avoid modifying the original
    quantized_model = type(model)(model.W.shape[0], model.W.shape[1])
    
    # Process each row independently
    for i in range(model.W.shape[0]):
        row = model.W.data[i]
        
        # Calculate min and max for this row
        row_min = torch.min(row)
        row_max = torch.max(row)
        
        if row_min == row_max:
            quantized_model.W.data[i] = row
            continue
            
        step_size = (row_max - row_min) / (num_levels - 1)
        
        normalized = (row - row_min) / step_size
        quantized = torch.round(normalized) * step_size + row_min
        
        quantized_model.W.data[i] = quantized
    
    # Copy bias without quantizing
    quantized_model.b.data = model.b.data.clone()
    
    return quantized_model

def quantise_models(models, bits_list):
    """Quantize multiple models using different bit widths."""
    quantized_models = []
    
    for model in models:
        model_variants = []
        for bits in bits_list:
            quantized = quantise_model(model, bits)
            model_variants.append(quantized)
        quantized_models.append(model_variants)
            
    return quantized_models
