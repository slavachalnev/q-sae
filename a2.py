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

cfg = TrainConfig(
    num_epochs=20000, 
    hidden_dim=32, 
    input_dim=4096,
)
cutoffs = [512, 1024, 1536, 2048, None]
sparsities, models, losses = train_model_variations(
    base_config=cfg,
    cutoffs=cutoffs,
)

# %%
plot_feature_norms(models, cutoffs)
# %%
