# %%
import torch
import plotly.graph_objects as go
from train import train_model_variations, TrainConfig, Model, eval_loss_per_feature
from quantise import quantise_models
import numpy as np
import torch.nn.functional as F
from dataclasses import replace
import math
from typing import List
from plotly.subplots import make_subplots

# %%
# cfg = TrainConfig(
#     num_epochs=30000, 
#     hidden_dim=64, 
#     input_dim=32768,
# )
# sparsities = [0.95, 0.99, 0.995, 0.999, 0.9995, 0.9998, 0.9999]

cfg = TrainConfig(
    num_epochs=300, 
    hidden_dim=32, 
    input_dim=8192,
)
sparsities = [0.9, 0.99]
sparsities, models, losses = train_model_variations(
    base_config=cfg,
    sparsities=sparsities,
)

# %%
def plot_feature_norms(models, sparsities):
    fig = go.Figure()

    for sparsity, model in zip(sparsities, models):
        ft_norms = torch.norm(model.W, dim=1)
        sorted_norms, _ = torch.sort(ft_norms, descending=True)
        fig.add_trace(go.Scatter(
            y=sorted_norms.tolist(),
            name=f'Sparsity {sparsity}',
            mode='lines'
        ))
    fig.update_layout(
        title='Feature Norms by Sparsity Level',
        xaxis_title='Feature Index (sorted by norm)',
        yaxis_title='Feature Norm',
    )
    return fig

def dims_per_ft(model: Model):
    """Calculate the ratio of hidden dimensions to the Frobenius norm of W."""
    return model.W.shape[1] / (torch.norm(model.W).item() ** 2)

def dims_per_active_ft(model: Model, threshold: float = 0.5):
    """Calculate the ratio of hidden dimensions to the number of active features."""
    ft_norms = torch.norm(model.W, dim=1)
    active_features = torch.sum(ft_norms > threshold).item()
    if active_features == 0:
        return float('nan')
    return model.W.shape[1] / active_features

def plot_dims_per_ft(models, sparsities):
    dims_per_ft_values = [dims_per_ft(model) for model in models]
    inverse_density = [1/(1-s) for s in sparsities]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=inverse_density,
        y=dims_per_ft_values,
        mode='lines+markers',
        name='Dims per Feature'
    ))
    
    fig.update_layout(
        title='Dimensions per Feature vs Inverse Density (1/1-sparsity)',
        xaxis_title='Inverse Density (1/1-sparsity)',
        yaxis_title='Dimensions per Feature',
        xaxis=dict(
            type='log',
        )
    )
    return fig

def plot_dims_per_active_ft(models, sparsities, threshold: float = 0.5):
    dims_per_active_values = [dims_per_active_ft(model, threshold) for model in models]
    inverse_density = [1/(1-s) for s in sparsities]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=inverse_density,
        y=dims_per_active_values,
        mode='lines+markers',
        name='Dims per Active Feature'
    ))
    
    fig.update_layout(
        title=f'Dimensions per Active Feature (norm > {threshold}) vs Inverse Density',
        xaxis_title='Inverse Density (1/1-sparsity)',
        yaxis_title='Dimensions per Active Feature',
        xaxis=dict(
            type='log',
        ),
        yaxis=dict(
            type='log',
        )
    )
    return fig

def plot_feature_losses(models, sparsities, scale_by_density=False):
    fig = go.Figure()

    for sparsity, model in zip(sparsities, models):
        config = replace(cfg, sparsity=sparsity)
        feature_losses = eval_loss_per_feature(model, config=config)
        
        if scale_by_density:
            # Multiply losses by inverse density
            inverse_density = 1 / (1 - sparsity)
            feature_losses = feature_losses * inverse_density
        
        ft_norms = torch.norm(model.W, dim=1)
        _, norm_indices = torch.sort(ft_norms, descending=True)
        
        sorted_losses = feature_losses[norm_indices]
        
        fig.add_trace(go.Scatter(
            y=sorted_losses.tolist(),
            name=f'Sparsity {sparsity}',
            mode='lines'
        ))

    title_prefix = 'Density-Scaled ' if scale_by_density else ''
    fig.update_layout(
        title=f'{title_prefix}Feature Losses (sorted by feature norm)',
        xaxis_title='Feature Index (sorted by decreasing norm)',
        yaxis_title=f'{title_prefix}Loss per Feature',
        yaxis=dict(
            type='log'
        )
    )
    return fig

def compute_active_features_loss(model: Model, config: TrainConfig, threshold: float = 0.5, scale_by_density=False):
    """Compute average loss of active features (features with norm > threshold)"""
    ft_norms = torch.norm(model.W, dim=1)
    feature_losses = eval_loss_per_feature(model, config=config)
    if scale_by_density:
        # Multiply losses by inverse density
        inverse_density = 1 / (1 - config.sparsity)
        feature_losses = feature_losses * inverse_density
    active_mask = ft_norms > threshold
    if not torch.any(active_mask):
        return float('nan')
    active_losses = feature_losses[active_mask]
    return active_losses.mean().item()

def plot_active_features_loss(models, sparsities, threshold: float = 0.5, scale_by_density=False):
    """Plot average loss of active features vs inverse density"""
    inverse_density = [1/(1-s) for s in sparsities]
    
    active_losses = []
    for sparsity, model in zip(sparsities, models):
        config = replace(cfg, sparsity=sparsity)
        avg_loss = compute_active_features_loss(model, config, threshold, scale_by_density)
        active_losses.append(avg_loss)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=inverse_density,
        y=active_losses,
        mode='lines+markers',
        name='Avg Loss of Active Features'
    ))
    
    title_prefix = 'Density-Scaled ' if scale_by_density else ''
    fig.update_layout(
        title=f'{title_prefix}Average Loss of Active Features (norm > {threshold}) vs Inverse Density',
        xaxis_title='Inverse Density (1/1-sparsity)',
        yaxis_title=f'{title_prefix}Average Loss of Active Features',
        xaxis=dict(type='log'),
        yaxis=dict(type='log')
    )
    return fig

def compute_power_law(s1, s2, m1, m2, threshold=0.5, scale_by_density=False):
    """
    Compute the power law exponent 'k' given two sparsities and two models.
    """
    x1 = 1 / (1 - s1)
    x2 = 1 / (1 - s2)
    y1 = compute_active_features_loss(m1, replace(cfg, sparsity=s1), threshold, scale_by_density)
    y2 = compute_active_features_loss(m2, replace(cfg, sparsity=s2), threshold, scale_by_density)
    k = (math.log(y2) - math.log(y1)) / (math.log(x2) - math.log(x1))
    return k

# %%
fig = plot_feature_norms(models, sparsities)
fig.show()

fig_dims = plot_dims_per_ft(models, sparsities)
fig_dims.show()

fig_dims_active = plot_dims_per_active_ft(models, sparsities)
fig_dims_active.show()

# Plot both regular and density-scaled losses
fig_losses = plot_feature_losses(models, sparsities, scale_by_density=False)
fig_losses.show()
fig_losses_scaled = plot_feature_losses(models, sparsities, scale_by_density=True)
fig_losses_scaled.show()

# Plot both regular and density-scaled active feature losses
fig_active_loss = plot_active_features_loss(models, sparsities, scale_by_density=False)
fig_active_loss.show()
fig_active_loss_scaled = plot_active_features_loss(models, sparsities, scale_by_density=True)
fig_active_loss_scaled.show()

# # Run compute_power_law on models with sparsities 0.95 and 0.99
# s1 = 0.95
# s2 = 0.99
# m1, m2 = [models[sparsities.index(s)] for s in (s1, s2)]
# k = compute_power_law(s1, s2, m1, m2, scale_by_density=False)
# k_scaled = compute_power_law(s1, s2, m1, m2, scale_by_density=True)
# print(f"Computed power law exponent k between sparsities {s1} and {s2}:")
# print(f"  Without density scaling: {k:.3f}")
# print(f"  With density scaling: {k_scaled:.3f}")

# %%
print('now quantising')
bits = [2, 2.5, 3, 4, 8]
quantised_models: List[List[Model]] = quantise_models(models, bits)

# %%
############### quantised plots ###############

# Define color palette once
QUANTISED_COLORS = [
    'rgb(102, 197, 204)',   # Soft blue-green
    'rgb(246, 207, 113)',   # Soft yellow
    'rgb(248, 156, 116)',   # Soft coral
    'rgb(220, 176, 242)',   # Soft purple
    'rgb(135, 197, 95)',    # Soft green
    'rgb(158, 185, 243)',   # Soft blue
    'rgb(254, 136, 177)',   # Soft pink
    'rgb(201, 219, 116)',   # Soft lime
]


def plot_quantised_feature_losses(quantised_models, sparsities, bits, scale_by_density=False):
    from plotly.subplots import make_subplots
    num_sparsities = len(sparsities)
    title_prefix = 'Density-Scaled ' if scale_by_density else ''
    fig = make_subplots(rows=num_sparsities, cols=1, subplot_titles=[f'Sparsity {s}' for s in sparsities])

    colors = QUANTISED_COLORS[:len(bits)]  # Take only as many colors as we have bits

    for row, (models_per_bit, sparsity) in enumerate(zip(quantised_models, sparsities), start=1):
        config = replace(cfg, sparsity=sparsity)
        for model, bit, color in zip(models_per_bit, bits, colors):
            feature_losses = eval_loss_per_feature(model, config=config)
            if scale_by_density:
                inverse_density = 1 / (1 - sparsity)
                feature_losses = feature_losses * inverse_density
            ft_norms = torch.norm(model.W, dim=1)
            _, norm_indices = torch.sort(ft_norms, descending=True)
            sorted_losses = feature_losses[norm_indices]
            fig.add_trace(go.Scatter(
                y=sorted_losses.tolist(),
                mode='lines',
                name=f'{bit}-bit',
                line=dict(color=color),
            ), row=row, col=1)
    
    for i in range(num_sparsities):
        fig.update_xaxes(title_text='Feature Index (sorted by decreasing norm)', row=i+1, col=1)
        if i == num_sparsities // 2:
            fig.update_yaxes(title_text=f'{title_prefix}Loss per Feature', row=i+1, col=1)
        fig.update_yaxes(type='log', row=i+1, col=1)
    fig.update_layout(
        title=f'{title_prefix}Feature Losses (Quantised Models)',
        showlegend=True,
        height=200 * num_sparsities
    )
    fig.show()


# %%

def plot_quantised_overall_loss(quantised_models, sparsities, bits, scale_by_density=False, only_active=False):
    """Plot overall loss vs inverse density for quantised models"""
    inverse_density = [1/(1-s) for s in sparsities]
    fig = go.Figure()
    threshold = 0.5  # Threshold for active features
    
    for bit_index, bit in enumerate(bits):
        losses = []
        for sparsity, models_per_bit in zip(sparsities, quantised_models):
            model = models_per_bit[bit_index]
            config = replace(cfg, sparsity=sparsity)
            feature_losses = eval_loss_per_feature(model, config=config)
            
            if only_active:
                ft_norms = torch.norm(model.W, dim=1)
                active_mask = ft_norms > threshold
                loss = torch.mean(feature_losses[active_mask])
            else:
                loss = torch.mean(feature_losses)
                
            if scale_by_density:
                loss = loss * (1 / (1 - sparsity))
            
            losses.append(loss.item())
            
        fig.add_trace(go.Scatter(
            x=inverse_density,
            y=losses,
            mode='lines+markers',
            name=f'{bit}-bit'
        ))
    
    # Build title components
    title_parts = []
    if scale_by_density:
        title_parts.append("Density-Scaled")
    if only_active:
        title_parts.append("Active Features")
    title_parts.append("Overall Loss vs Inverse Density")
    title = " ".join(title_parts) + " (Quantised Models)"
    
    fig.update_layout(
        title=title,
        xaxis_title='Inverse Density (1/1-sparsity)',
        yaxis_title='Overall Loss' if not scale_by_density else 'Density-Scaled Overall Loss',
        xaxis=dict(type='log'),
        yaxis=dict(type='log')
    )
    fig.show()

plot_quantised_feature_losses(quantised_models, sparsities, bits, scale_by_density=False)

plot_quantised_overall_loss(quantised_models, sparsities, bits, scale_by_density=True, only_active=False)
plot_quantised_overall_loss(quantised_models, sparsities, bits, scale_by_density=True, only_active=True)

# %%
