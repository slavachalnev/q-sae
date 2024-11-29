# %%
import torch
import plotly.graph_objects as go
from train import train_models_with_sparsities, TrainConfig

# %%
sparsities = [0.5, 0.7, 0.9, 0.95, 0.99]
sparsities, models, losses = train_models_with_sparsities(
    sparsities,
    base_config=TrainConfig(
        num_epochs=10000,
    )
)


# %%

fig = go.Figure()

for sparsity, model in zip(sparsities, models):
    # Calculate row norms
    ft_norms = torch.norm(model.W, dim=1)
    
    # Sort row norms in descending order
    sorted_norms, _ = torch.sort(ft_norms, descending=True)
    
    # Add line to plot
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

fig.show()

# %%
