# %%
import torch
import plotly.graph_objects as go
from train import train_models_with_sparsities, TrainConfig, Model, eval_loss_per_feature
import numpy as np
import torch.nn.functional as F
from dataclasses import replace

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


def plot_feature_losses(models, sparsities):
    fig = go.Figure()

    for sparsity, model in zip(sparsities, models):
        config = replace(cfg, sparsity=sparsity)
        feature_losses = eval_loss_per_feature(model, config=config)
        
        ft_norms = torch.norm(model.W, dim=1)
        _, norm_indices = torch.sort(ft_norms, descending=True)
        
        sorted_losses = feature_losses[norm_indices]
        
        fig.add_trace(go.Scatter(
            y=sorted_losses.tolist(),
            name=f'Sparsity {sparsity}',
            mode='lines'
        ))

    fig.update_layout(
        title='Feature Losses (sorted by feature norm)',
        xaxis_title='Feature Index (sorted by decreasing norm)',
        yaxis_title='Loss per Feature',
        yaxis=dict(
            type='log'
        )
    )
    return fig

def compute_active_features_loss(model: Model, config: TrainConfig, threshold: float = 0.8):
    """Compute average loss of active features (features with norm > threshold)"""
    ft_norms = torch.norm(model.W, dim=1)
    feature_losses = eval_loss_per_feature(model, config=config)
    active_mask = ft_norms > threshold
    if not torch.any(active_mask):
        return float('nan')
    active_losses = feature_losses[active_mask]
    return active_losses.mean().item()

def plot_active_features_loss(models, sparsities, threshold: float = 0.8):
    """Plot average loss of active features vs inverse density"""
    inverse_density = [1/(1-s) for s in sparsities]
    
    active_losses = []
    for sparsity, model in zip(sparsities, models):
        config = replace(cfg, sparsity=sparsity)
        avg_loss = compute_active_features_loss(model, config, threshold)
        active_losses.append(avg_loss)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=inverse_density,
        y=active_losses,
        mode='lines+markers',
        name='Avg Loss of Active Features'
    ))
    
    fig.update_layout(
        title=f'Average Loss of Active Features (norm > {threshold}) vs Inverse Density',
        xaxis_title='Inverse Density (1/1-sparsity)',
        yaxis_title='Average Loss of Active Features',
        xaxis=dict(type='log'),
        yaxis=dict(type='log')
    )
    return fig

# %%
fig = plot_feature_norms(models, sparsities)
fig.show()

fig_dims = plot_dims_per_ft(models, sparsities)
fig_dims.show()

fig_losses = plot_feature_losses(models, sparsities)
fig_losses.show()

fig_active_loss = plot_active_features_loss(models, sparsities)
fig_active_loss.show()

# %%
