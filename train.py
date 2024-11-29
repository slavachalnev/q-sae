import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from dataclasses import dataclass

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

def train_model(config: TrainConfig):
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

        if (epoch + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}')

    return model, loss.item()


if __name__ == "__main__":
    # Create config with default values
    config = TrainConfig()
    print(f"Using device: {config.device}")
    
    # Train model
    model, final_loss = train_model(config)
    