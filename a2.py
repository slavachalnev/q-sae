# analysis in the case of exponential decay
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
from utils import plot_feature_norms

# %%

# cfg = TrainConfig(
#     num_epochs=20000, 
#     hidden_dim=64, 
#     input_dim=2048,
#     initial_prob=0.005,
#     decay_rate=0.002,
# )
# cutoffs = [256, 512, 768, 1024, 1536, 2048]
# sparsities, models, losses = train_model_variations(
#     base_config=cfg,
#     cutoffs=cutoffs,
# )

cfg = TrainConfig(
    num_epochs=20000, 
    hidden_dim=100, 
    input_dim=2048,
    initial_prob=0.005,
    decay_rate=0.002,
)
cutoffs = [256, 512, 768, 1024, 1536, 2048]
sparsities, models, losses = train_model_variations(
    base_config=cfg,
    cutoffs=cutoffs,
)

# %%
plot_feature_norms(models, cutoffs, sort_by_norm=False).show()
plot_feature_norms(models, cutoffs, sort_by_norm=True).show()

# %%
# Add new code to plot eval loss per feature
def plot_eval_loss_per_feature(models, cutoffs, base_cfg):
    fig = go.Figure()
    
    # Evaluate loss per feature for each model
    for model, cutoff in zip(models, cutoffs):
        # Create a new config with the correct cutoff
        cfg = replace(base_cfg, feature_cutoff=cutoff, use_exponential_decay=True)
        losses = eval_loss_per_feature(model, cfg)
        
        # Create line plot for this model
        fig.add_trace(go.Scatter(
            x=list(range(base_cfg.input_dim)),
            y=losses.cpu().numpy(),
            name=f'Cutoff {cutoff}',
            mode='lines',
        ))
    
    fig.update_layout(
        title='Loss per Feature for Different Cutoffs',
        xaxis_title='Feature Index',
        yaxis_title='Mean Squared Error',
        yaxis_type='log',  # Use log scale for better visualization
        showlegend=True
    )
    
    return fig

# Plot the eval loss per feature
plot_eval_loss_per_feature(models, cutoffs, cfg).show()

# def plot_avg_loss_over_active_features(models, cutoffs, base_cfg):
#     avg_losses = []
#     for model, cutoff in zip(models, cutoffs):
#         # Get feature norms from the model
#         feature_norms = model.W.norm(dim=1).detach().cpu().numpy()
#         # Identify active features (norm > 0.5)
#         active_features = feature_norms > 0.5
#         print(f"Active features shape: {active_features.shape}") # shape (input_dim,)
#         # Ensure there are active features to prevent division by zero
#         if np.sum(active_features) == 0:
#             avg_loss = np.nan  # Handle case with no active features
#         else:
#             # Create a new config with the current cutoff
#             cfg = replace(base_cfg, feature_cutoff=cutoff, use_exponential_decay=True)
#             # Evaluate loss per feature
#             losses = eval_loss_per_feature(model, cfg).cpu().numpy()
#             # Compute average loss over active features
#             print(f"Losses shape: {losses.shape}") # shape (input_dim,)
#             avg_loss = losses[active_features].mean()
#         avg_losses.append(avg_loss)
    
#     # Plot average loss over active features vs. cutoff
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=cutoffs,
#         y=avg_losses,
#         mode='lines+markers',
#         name='Avg Loss over Active Features'
#     ))
#     fig.update_layout(
#         title='Average Loss over Active Features vs Cutoff',
#         xaxis_title='Cutoff',
#         yaxis_title='Average Loss',
#         yaxis_type='log'  # Use log scale if needed
#     )
#     fig.show()

# # Plot the average loss over active features
# plot_avg_loss_over_active_features(models, cutoffs, cfg)

# After the existing plotting functions, add:

def plot_avg_loss_first_256(models, cutoffs, base_cfg):
    avg_losses = []
    for model, cutoff in zip(models, cutoffs):
        # Create a new config with the current cutoff
        cfg = replace(base_cfg, feature_cutoff=cutoff, use_exponential_decay=True)
        # Evaluate loss per feature
        losses = eval_loss_per_feature(model, cfg).cpu().numpy()
        # Compute average loss over first 256 features
        avg_loss = losses[:256].mean()
        avg_losses.append(avg_loss)
    
    # Plot average loss vs. cutoff
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cutoffs,
        y=avg_losses,
        mode='lines+markers',
        name='Avg Loss (First 256 Features)'
    ))
    
    fig.update_layout(
        title='Average Loss over First 256 Features vs Cutoff',
        xaxis_title='Model Cutoff',
        yaxis_title='Average Loss',
        yaxis_type='log'  # Using log scale for better visualization
    )
    return fig

# Call the plotting function
plot_avg_loss_first_256(models, cutoffs, cfg).show()

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

# After the existing plotting functions, add:

def plot_avg_loss_first_256_quantised(quantised_models, cutoffs, bits, base_cfg):
    fig = go.Figure()
    
    # For each quantisation level
    for bit_idx, bit in enumerate(bits):
        avg_losses = []
        # For each cutoff
        for model_idx, cutoff in enumerate(cutoffs):
            # Get the model with this bit width and cutoff
            model = quantised_models[model_idx][bit_idx]
            # Create config with current cutoff
            cfg = replace(base_cfg, feature_cutoff=cutoff, use_exponential_decay=True)
            # Evaluate loss per feature
            losses = eval_loss_per_feature(model, cfg).cpu().numpy()
            # Compute average loss over first 256 features
            avg_loss = losses[:256].mean()
            avg_losses.append(avg_loss)
        
        # Add trace for this bit width
        fig.add_trace(go.Scatter(
            x=cutoffs,
            y=avg_losses,
            mode='lines+markers',
            name=f'{bit}-bit',
            line=dict(color=QUANTISED_COLORS[bit_idx % len(QUANTISED_COLORS)])
        ))
    
    fig.update_layout(
        title='Average Loss over First 256 Features vs Cutoff (Quantised Models)',
        xaxis_title='Model Cutoff',
        yaxis_title='Average Loss',
        yaxis_type='log',  # Using log scale for better visualization
        showlegend=True
    )
    return fig

# Call the plotting function after quantisation
plot_avg_loss_first_256_quantised(quantised_models, cutoffs, bits, cfg).show()

# %%

def plot_loss_per_feature_quantised(quantised_models, cutoffs, bits, base_cfg):
    # Create subplots - one for each cutoff
    fig = make_subplots(
        rows=len(cutoffs), 
        cols=1,
        subplot_titles=[f'Cutoff {cutoff}' for cutoff in cutoffs],
        vertical_spacing=0.05
    )
    
    # For each cutoff (subplot)
    for cutoff_idx, cutoff in enumerate(cutoffs):
        # For each quantisation level
        for bit_idx, bit in enumerate(bits):
            # Get the model with this bit width and cutoff
            model = quantised_models[cutoff_idx][bit_idx]
            
            # Create config with current cutoff
            cfg = replace(base_cfg, feature_cutoff=cutoff, use_exponential_decay=True)
            
            # Evaluate loss per feature
            losses = eval_loss_per_feature(model, cfg).cpu().numpy()
            
            # Add trace for this bit width
            fig.add_trace(
                go.Scatter(
                    x=list(range(base_cfg.input_dim)),
                    y=losses,
                    mode='lines',
                    name=f'{bit}-bit',
                    line=dict(color=QUANTISED_COLORS[bit_idx % len(QUANTISED_COLORS)]),
                    showlegend=(cutoff_idx == 0)  # Only show legend for first subplot
                ),
                row=cutoff_idx + 1,
                col=1
            )
    
    # Update layout
    fig.update_layout(
        height=300 * len(cutoffs),  # Adjust height based on number of subplots
        title='Loss per Feature for Different Quantisation Levels',
        showlegend=True
    )
    
    # Update all subplot y-axes to log scale and add labels
    for i in range(len(cutoffs)):
        fig.update_yaxes(
            type='log',
            title='Loss per Feature' if i == len(cutoffs)//2 else None,  # Add y-label in middle subplot
            row=i+1,
            col=1
        )
        fig.update_xaxes(
            title='Feature Index' if i == len(cutoffs)-1 else None,  # Add x-label only on bottom subplot
            row=i+1,
            col=1
        )
    
    return fig

# Call the plotting function after quantisation
plot_loss_per_feature_quantised(quantised_models, cutoffs, bits, cfg).show()

# %%

# After the existing plotting functions, add:
def plot_avg_loss_all_features_quantised(models, quantised_models, cutoffs, bits, base_cfg):
    fig = go.Figure()
    
    # Plot unquantized model first
    avg_losses = []
    for model, cutoff in zip(models, cutoffs):
        cfg = replace(base_cfg, feature_cutoff=None, use_exponential_decay=True)
        losses = eval_loss_per_feature(model, cfg).cpu().numpy()
        avg_loss = losses.mean()
        avg_losses.append(avg_loss)
    
    fig.add_trace(go.Scatter(
        x=cutoffs,
        y=avg_losses,
        mode='lines+markers',
        name='Unquantized',
        line=dict(color='black', width=2)
    ))
    
    # For each quantisation level
    for bit_idx, bit in enumerate(bits):
        avg_losses = []
        # For each cutoff
        for model_idx, cutoff in enumerate(cutoffs):
            # Get the model with this bit width and cutoff
            model = quantised_models[model_idx][bit_idx]
            cfg = replace(base_cfg, feature_cutoff=None, use_exponential_decay=True)
            losses = eval_loss_per_feature(model, cfg).cpu().numpy()
            avg_loss = losses.mean()
            avg_losses.append(avg_loss)
        
        # Add trace for this bit width
        fig.add_trace(go.Scatter(
            x=cutoffs,
            y=avg_losses,
            mode='lines+markers',
            name=f'{bit}-bit',
            line=dict(color=QUANTISED_COLORS[bit_idx % len(QUANTISED_COLORS)])
        ))
    
    fig.update_layout(
        title='Average Loss over All Features vs Model Cutoff (Quantised Models)',
        xaxis_title='Model Cutoff',
        yaxis_title='Average Loss',
        yaxis_type='log',  # Using log scale for better visualization
        showlegend=True
    )
    return fig

# Call the plotting function
plot_avg_loss_all_features_quantised(models, quantised_models, cutoffs, bits, cfg).show()

# %%

def compute_cosine_sim_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix using PyTorch."""
    # Normalize the features
    normalized = F.normalize(features, p=2, dim=1)
    # Compute cosine similarity matrix
    cosine_sim = torch.mm(normalized, normalized.t())
    return cosine_sim

def plot_cosine_sim_to_closest_feature(models, cutoffs, base_cfg):
    fig = go.Figure()
    
    for model, cutoff in zip(models, cutoffs):
        # Get the feature vectors (rows of W)
        features = model.W.detach()
        
        # Calculate cosine similarity matrix
        cosine_sim = compute_cosine_sim_matrix(features)
        
        # For each feature, find the highest cosine similarity 
        # excluding self-similarity (which would always be 1)
        cosine_sim.fill_diagonal_(-1)  # Replace self-similarity with -1
        max_similarities = torch.max(cosine_sim, dim=1).values
        
        # Create line plot for this model
        fig.add_trace(go.Scatter(
            x=list(range(base_cfg.input_dim)),
            y=max_similarities.cpu().numpy(),
            name=f'Cutoff {cutoff}',
            mode='lines',
        ))
    
    fig.update_layout(
        title='Cosine Similarity to Closest Feature',
        xaxis_title='Feature Index',
        yaxis_title='Cosine Similarity',
        showlegend=True
    )
    
    return fig

# Add after the other plotting calls
plot_cosine_sim_to_closest_feature(models, cutoffs, cfg).show()

# Also add a quantized version of the same plot
def plot_cosine_sim_to_closest_feature_quantised(quantised_models, cutoffs, bits, base_cfg):
    fig = make_subplots(
        rows=len(cutoffs), 
        cols=1,
        subplot_titles=[f'Cutoff {cutoff}' for cutoff in cutoffs],
        vertical_spacing=0.05
    )
    
    for cutoff_idx, cutoff in enumerate(cutoffs):
        for bit_idx, bit in enumerate(bits):
            # Get the model with this bit width and cutoff
            model = quantised_models[cutoff_idx][bit_idx]
            
            # Get the feature vectors and compute similarity
            features = model.W.detach()
            cosine_sim = compute_cosine_sim_matrix(features)
            
            # Find highest similarity excluding self
            cosine_sim.fill_diagonal_(-1)
            max_similarities = torch.max(cosine_sim, dim=1).values
            
            # Add trace for this bit width
            fig.add_trace(
                go.Scatter(
                    x=list(range(base_cfg.input_dim)),
                    y=max_similarities.cpu().numpy(),
                    mode='lines',
                    name=f'{bit}-bit',
                    line=dict(color=QUANTISED_COLORS[bit_idx % len(QUANTISED_COLORS)]),
                    showlegend=(cutoff_idx == 0)
                ),
                row=cutoff_idx + 1,
                col=1
            )
    
    fig.update_layout(
        height=300 * len(cutoffs),
        title='Cosine Similarity to Closest Feature (Quantised Models)',
        showlegend=True
    )
    
    for i in range(len(cutoffs)):
        fig.update_yaxes(
            title='Cosine Similarity' if i == len(cutoffs)//2 else None,
            row=i+1,
            col=1
        )
        fig.update_xaxes(
            title='Feature Index' if i == len(cutoffs)-1 else None,
            row=i+1,
            col=1
        )
    
    return fig

# Call the quantised plotting function after the quantisation section
plot_cosine_sim_to_closest_feature_quantised(quantised_models, cutoffs, bits, cfg).show()

# %%
