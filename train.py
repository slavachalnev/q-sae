import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from dataclasses import dataclass
from typing import List, Tuple
import copy

@dataclass
class TrainConfig:
    input_dim: int = 1024
    hidden_dim: int = 32
    batch_size: int = 4096
    num_epochs: int = 1000
    sparsity: float = 0.9
    learning_rate: float = 1e-3
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_sparse_data(batch_size, input_dim, sparsity, device):
    """Generate synthetic sparse data where active features are uniform[0,1]."""
    active_mask = (torch.rand(batch_size, input_dim, device=device) > sparsity).float()
    values = torch.rand(batch_size, input_dim, device=device)
    return values * active_mask

def train_model(config: TrainConfig, silent=False):
    model = Model(config.input_dim, config.hidden_dim).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(config.num_epochs):
        x = generate_sparse_data(config.batch_size, config.input_dim, config.sparsity, config.device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()

        if not silent and (epoch + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
    return model, loss.item()

def train_models_with_sparsities(sparsities: List[float], base_config: TrainConfig = None) -> List[Tuple[float, Model, float]]:
    if base_config is None:
        base_config = TrainConfig()
    
    results = []
    for sparsity in sparsities:
        config = copy.copy(base_config)
        config.sparsity = sparsity

        model, final_loss = train_model(config, silent=True)

        results.append((sparsity, model, final_loss))
        print(f"Completed training for sparsity {sparsity:.2f}, final loss: {final_loss:.6f}")
    return results

if __name__ == "__main__":
    # Example usage with multiple sparsity levels
    sparsities = [0.5, 0.7, 0.9, 0.95, 0.99]
    base_config = TrainConfig(num_epochs=500)  # Reduced epochs for example
    
    print(f"Using device: {base_config.device}")
    results = train_models_with_sparsities(sparsities, base_config)
    
    # Print summary of results
    print("\nTraining Results Summary:")
    for sparsity, _, loss in results:
        print(f"Sparsity: {sparsity:.2f}, Final Loss: {loss:.6f}")
    