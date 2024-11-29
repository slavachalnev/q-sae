# %%
import torch
import plotly.graph_objects as go
from train import train_models_with_sparsities, TrainConfig, Model, generate_sparse_data #, eval_loss_per_feature
import numpy as np
import torch.nn.functional as F

# %%
sparsities = [0.7, 0.9, 0.95, 0.99, 0.995]
# sparsities = [0.9, 0.99]
sparsities, models, losses = train_models_with_sparsities(
    sparsities,
    base_config=TrainConfig(num_epochs=20000)
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

@torch.no_grad()
def eval_loss_per_feature(model: Model, config: TrainConfig, num_batches: int = 100) -> torch.Tensor:
    original_device = model.W.device
    model.to(config.device)
    
    # Initialize accumulator for losses
    total_loss = torch.zeros(config.input_dim, device=config.device)
    
    # Evaluate over multiple batches
    for _ in range(num_batches):
        x = generate_sparse_data(config.batch_size, config.input_dim, config.sparsity, config.device)
        output = model(x)
        loss = ((output - x) ** 2).mean(dim=0)
        total_loss += loss
    
    # Calculate average loss across batches
    avg_loss = total_loss / num_batches
    
    model.to(original_device)
    return avg_loss.to(original_device)

def plot_feature_losses(models, sparsities):
    fig = go.Figure()

    for sparsity, model in zip(sparsities, models):
        # Calculate loss per feature
        feature_losses = eval_loss_per_feature(model, config=TrainConfig())
        
        # Get feature norms and their indices
        ft_norms = torch.norm(model.W, dim=1)
        _, norm_indices = torch.sort(ft_norms, descending=True)
        
        # Sort losses according to norm ordering
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
    )
    return fig

# %%
fig = plot_feature_norms(models, sparsities)
fig.show()

fig_dims = plot_dims_per_ft(models, sparsities)
fig_dims.show()

fig_losses = plot_feature_losses(models, sparsities)
fig_losses.show()

# %%
