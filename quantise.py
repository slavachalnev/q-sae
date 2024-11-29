import torch

def quantise_model(model, bits):
    """
    Quantize a single model using specified number of bits per weight.
    Each row of model.W is quantized independently. Bias is copied without quantization.
    """
    # Calculate number of levels from bits
    num_levels = 2 ** bits
    
    # Create a copy of the model to avoid modifying the original
    quantized_model = type(model)(model.W.shape[0], model.W.shape[1])
    quantized_model.W = torch.zeros_like(model.W)
    quantized_model.b = model.b.clone()  # Copy bias without quantizing
    
    # Process each row independently
    for i in range(model.W.shape[0]):
        row = model.W[i]
        
        # Calculate min and max for this row
        row_min = torch.min(row)
        row_max = torch.max(row)
        
        if row_min == row_max:
            quantized_model.W[i] = row
            continue
            
        step_size = (row_max - row_min) / (num_levels - 1)
        
        normalized = (row - row_min) / step_size
        quantized = torch.round(normalized) * step_size + row_min
        
        quantized_model.W[i] = quantized
    
    return quantized_model

def quantise_models(models, bits_list):
    """Quantize multiple models using different bit widths."""
    quantized_models = []
    
    for model in models:
        for bits in bits_list:
            quantized = quantise_model(model, bits)
            quantized_models.append(quantized)
            
    return quantized_models
