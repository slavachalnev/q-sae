import torch
import plotly.graph_objects as go


def plot_feature_norms(models, names):
    fig = go.Figure()

    for name, model in zip(names, models):
        ft_norms = torch.norm(model.W, dim=1)
        sorted_norms, _ = torch.sort(ft_norms, descending=True)
        fig.add_trace(go.Scatter(
            y=sorted_norms.tolist(),
            name=name,
            mode='lines'
        ))
    fig.update_layout(
        title='Feature Norms',
        xaxis_title='Feature Index (sorted by norm)',
        yaxis_title='Feature Norm',
    )
    return fig