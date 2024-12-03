import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import Model
from dataclasses import dataclass
from typing import List, Tuple
import copy
from tqdm import tqdm

@dataclass
class TrainConfig:
    input_dim: int = 2048
    hidden_dim: int = 32
    batch_size: int = 8192
    num_epochs: int = 10000
    sparsity: float = 0.9
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    min_lr: float = 1e-6  # Minimum learning rate for cosine decay
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_sparse_data(batch_size, input_dim, sparsity, device):
    """Generate synthetic sparse data where active features are uniform[0,1]."""
    active_mask = (torch.rand(batch_size, input_dim, device=device) > sparsity).float()
    values = torch.rand(batch_size, input_dim, device=device)
    return values * active_mask

def train_model(config: TrainConfig):
    model = Model(config.input_dim, config.hidden_dim).to(config.device)
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config.learning_rate, 
                           weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, 
                                 T_max=config.num_epochs,
                                 eta_min=config.min_lr)
    criterion = nn.MSELoss()

    pbar = tqdm(range(config.num_epochs), desc='Training')
    for epoch in pbar:
        x = generate_sparse_data(config.batch_size, config.input_dim, config.sparsity, config.device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
    return model, loss.item()


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



def train_models_with_sparsities(sparsities: List[float], base_config: TrainConfig = None) -> Tuple[List[float], List[Model], List[float]]:
    if base_config is None:
        base_config = TrainConfig()
    
    trained_sparsities = []
    trained_models = []
    final_losses = []
    
    for sparsity in sparsities:
        config = copy.copy(base_config)
        config.sparsity = sparsity

        model, final_loss = train_model(config)
        model.to('cpu')

        trained_sparsities.append(sparsity)
        trained_models.append(model)
        final_losses.append(final_loss)
        
        print(f"Completed training for sparsity {sparsity:.2f}, final loss: {final_loss:.6f}")
    return trained_sparsities, trained_models, final_losses

if __name__ == "__main__":
    # Example usage with multiple sparsity levels
    sparsities = [0.5, 0.7, 0.9, 0.95, 0.99]
    base_config = TrainConfig(
        num_epochs=500,
        weight_decay=0.01
    )
    
    print(f"Using device: {base_config.device}")
    sparsities, models, losses = train_models_with_sparsities(sparsities, base_config)
    
    # Print summary of results
    print("\nTraining Results Summary:")
    for sparsity, loss in zip(sparsities, losses):
        print(f"Sparsity: {sparsity:.2f}, Final Loss: {loss:.6f}")
    