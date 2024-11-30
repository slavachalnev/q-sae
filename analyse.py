# %%
import torch
import plotly.graph_objects as go
from train import train_models_with_sparsities, TrainConfig, Model, eval_loss_per_feature
import numpy as np
import torch.nn.functional as F
from dataclasses import replace
import math

# %%
sparsities = [0.7, 0.9, 0.95, 0.99, 0.995]
cfg = TrainConfig(
    num_epochs=20000,
    # input_dim=4096, # default is 2048
)
# sparsities = [0.9, 0.99]
sparsities, models, losses = train_models_with_sparsities(
    sparsities,
    base_config=cfg,
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

def compute_active_features_loss(model: Model, config: TrainConfig, threshold: float = 0.8, scale_by_density=False):
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

def plot_active_features_loss(models, sparsities, threshold: float = 0.8, scale_by_density=False):
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

def compute_power_law(s1, s2, m1, m2, threshold=0.8, scale_by_density=False):
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

# Run compute_power_law on models with sparsities 0.95 and 0.99
s1 = 0.95
s2 = 0.99
m1, m2 = [models[sparsities.index(s)] for s in (s1, s2)]
k = compute_power_law(s1, s2, m1, m2, scale_by_density=False)
k_scaled = compute_power_law(s1, s2, m1, m2, scale_by_density=True)
print(f"Computed power law exponent k between sparsities {s1} and {s2}:")
print(f"  Without density scaling: {k:.3f}")
print(f"  With density scaling: {k_scaled:.3f}")

# %%
