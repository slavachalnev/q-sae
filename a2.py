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
from utils import plot_feature_norms, plot_dims_per_active_ft

# %%

cfg = TrainConfig(
    num_epochs=20000, 
    hidden_dim=64, 
    input_dim=1024, # 2048
    initial_prob=0.01,
    decay_rate=0.002,
)
cutoffs = [256, 512, 768, 1024]
sparsities, models, losses = train_model_variations(
    base_config=cfg,
    cutoffs=cutoffs,
)

# %%
plot_feature_norms(models, cutoffs, sort_by_norm=False).show()
plot_feature_norms(models, cutoffs, sort_by_norm=True).show()

plot_dims_per_active_ft(models, sparsities).show()
# %%
