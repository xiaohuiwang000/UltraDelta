import torch
from collections import OrderedDict


def calculate_sparsity(matrix):
    """Calculate the sparsity of a delta weight by determining the ratio of zero elements to total elements."""
    num_zeros = torch.sum(matrix == 0).item()
    total_elements = matrix.numel()
    sparsity = num_zeros / total_elements
    return sparsity


def quantize_delta(vec, bit):
    """Quantize the delta to a lower precision based on the specified bit."""
    min_val = torch.min(vec)
    max_val = torch.max(vec)
    new_vec = (vec - min_val) / (max_val - min_val)
    new_vec = new_vec * (2 ** bit - 1)
    new_vec = torch.round(new_vec)
    new_vec = new_vec / (2 ** bit - 1)
    new_vec = new_vec * (max_val - min_val) + min_val
    return new_vec


def assign_sparsity_rates(deltas, args):
    """Assign sparsity rates across layers based on their standard deviation."""
    layer_mask_rates_list = []
    
    for i, delta in enumerate(deltas):
        layer_stds = {}
        layer_sizes = {}
        total_params = 0
        
        # Calculate standard deviation and size for each layer
        for key, value in delta.items():
            size = value.numel()
            layer_sizes[key] = size
            total_params += size
            
            if size > 1:
                mean = torch.mean(value)
                var = torch.mean((value - mean) ** 2)
                layer_stds[key] = torch.sqrt(var).item()
            else:
                layer_stds[key] = 0.0

        keys_by_std = sorted(layer_stds.keys(), key=lambda k: layer_stds[k])
        
        target_rate = args.mask_rate
        high_rate = target_rate + args.step_size
        mid_rate = target_rate
        
        first_third_params = total_params / 3
        second_third_params = 2 * total_params / 3
        
        layer_mask_rates = {}
        accumulated_params = 0
        std_min_params = std_mid_params = std_max_params = 0
        
        # Assign sparsity
        for key in keys_by_std:
            size = layer_sizes[key]
            accumulated_params += size
            
            if accumulated_params <= first_third_params:
                layer_mask_rates[key] = high_rate
                std_min_params += size
            elif accumulated_params <= second_third_params:
                layer_mask_rates[key] = mid_rate
                std_mid_params += size
            else:
                std_max_params += size
        
        if std_max_params > 0:
            max_rate = (target_rate * total_params - mid_rate * std_mid_params - high_rate * std_min_params) / std_max_params
            max_rate = min(0.99, max(0.0, max_rate))
        else:
            max_rate = mid_rate
        
        for key in keys_by_std:
            if key not in layer_mask_rates:
                layer_mask_rates[key] = max_rate
        
        layer_mask_rates_list.append(layer_mask_rates)

    return layer_mask_rates_list


def apply_sparsity_mask(deltas, layer_mask_rates_list, args):
    """Apply group-wise pruning to the deltas using the assigned sparsity rates."""
    for i, (delta, layer_mask_rates) in enumerate(zip(deltas, layer_mask_rates_list)):
        for key, value in delta.items():
            mask_rate_original = layer_mask_rates[key]
            current_sparsity = calculate_sparsity(delta[key])
            current_mask_rate = (mask_rate_original - current_sparsity) / (1 - current_sparsity)

            if current_mask_rate < 0:
                current_mask_rate = 0
            
            assert 0.0 <= current_mask_rate <= 1.0, f"Wrong range of mask_rate {mask_rate_original}, should be [0.0, 1.0]!"
            delta[key] = mask_unique_values(delta[key], current_mask_rate)
        
        # Apply global rescaling
        weighted_rate = args.mask_rate
        rescaler = 1 / (1 - weighted_rate)
        rescaler = rescaler * args.additional_factor
        for key in delta:
            delta[key] = delta[key] * rescaler
    
    return deltas

    
def mask_unique_values(input_tensor, mask_rate):
    """Mask unique values in the input tensor by randomly zeroing out a proportion of elements based on mask_rate."""
    unique_values = torch.unique(input_tensor)
    masked_tensor = input_tensor.clone()
    for value in unique_values:

        value_indices = (input_tensor == value).nonzero(as_tuple=True)
        num_value_elements = len(value_indices[0])
        num_elements_to_remove = round(num_value_elements * mask_rate)
        indices_to_remove = torch.randperm(num_value_elements)[:num_elements_to_remove]
        
        if input_tensor.dim() == 1:
            masked_tensor[value_indices[0][indices_to_remove]] = 0
        else:
            masked_tensor[value_indices[0][indices_to_remove], value_indices[1][indices_to_remove]] = 0
    
    return masked_tensor