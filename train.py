import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import Model
from dataclasses import dataclass, asdict
from typing import List, Tuple
import copy
from tqdm import tqdm
import json
import hashlib
import os
from pathlib import Path

# Add these constants at the top of the file
TRAINED_MODELS_DIR = Path("trained_models")

@dataclass
class TrainConfig:
    input_dim: int = 2048
    hidden_dim: int = 32
    batch_size: int = 8192
    num_epochs: int = 10000
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    min_lr: float = 1e-6  # Minimum learning rate for cosine decay
    device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Sparsity configuration
    use_exponential_decay: bool = False  # whether to use exponential decay for feature probabilities
    sparsity: float = 0.9  # used when use_exponential_decay=False
    initial_prob: float = 0.01  # probability of first feature being active when using decay
    decay_rate: float = 0.002    # controls how quickly probabilities decay
    feature_cutoff: int = None  # if set, features beyond this index have 0 probability


def generate_sparse_data(batch_size: int, config: TrainConfig) -> torch.Tensor:
    """Generate synthetic sparse data where active features are uniform[0,1].
    
    Args:
        batch_size: Number of samples to generate
        config: TrainConfig object containing all necessary parameters
    """
    if not config.use_exponential_decay:
        # Uniform sparsity
        active_mask = (torch.rand(batch_size, config.input_dim, device=config.device) > config.sparsity).float()
    else:
        # Exponentially decaying activation probabilities
        probs = config.initial_prob * torch.exp(-torch.arange(config.input_dim, device=config.device) * config.decay_rate)
        
        # Apply cutoff if specified
        if config.feature_cutoff is not None:
            probs[config.feature_cutoff:] = 0.0
            
        # Broadcast probabilities to batch size
        probs = probs.expand(batch_size, -1)
        # Generate active mask using the varying probabilities
        active_mask = (torch.rand(batch_size, config.input_dim, device=config.device) < probs).float()
    
    values = torch.rand(batch_size, config.input_dim, device=config.device)
    return values * active_mask

def train_model(config: TrainConfig):
    model = Model(config.input_dim, config.hidden_dim, 
                 feature_cutoff=config.feature_cutoff).to(config.device)
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config.learning_rate, 
                           weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, 
                                 T_max=config.num_epochs,
                                 eta_min=config.min_lr)
    criterion = nn.MSELoss()

    pbar = tqdm(range(config.num_epochs), desc='Training')
    for epoch in pbar:
        x = generate_sparse_data(config.batch_size, config)

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
        x = generate_sparse_data(config.batch_size, config)
        output = model(x)
        loss = ((output - x) ** 2).mean(dim=0)
        total_loss += loss
    
    # Calculate average loss across batches
    avg_loss = total_loss / num_batches
    
    model.to(original_device)
    return avg_loss.to(original_device)

# Add these new helper functions
def hash_config(config: TrainConfig) -> str:
    """Create a deterministic hash of the config."""
    # Convert config to dict and create a deterministic string representation
    config_dict = asdict(config)
    # Remove device from hashing as it's not relevant to the model's training
    config_dict.pop('device', None)
    # Sort keys to ensure deterministic ordering
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]

def save_trained_model(model: Model, config: TrainConfig, final_loss: float):
    """Save the model weights, config, and final loss."""
    config_hash = hash_config(config)
    save_dir = TRAINED_MODELS_DIR / config_hash
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), save_dir / "model.pt")
    
    # Save config and loss
    config_dict = asdict(config)
    config_dict['final_loss'] = final_loss
    with open(save_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_trained_model(config: TrainConfig) -> Tuple[Model, float]:
    """Load a previously trained model if it exists."""
    config_hash = hash_config(config)
    model_dir = TRAINED_MODELS_DIR / config_hash
    
    if not model_dir.exists():
        return None, None
    
    # Load model with feature_cutoff parameter
    model = Model(config.input_dim, config.hidden_dim, 
                 feature_cutoff=config.feature_cutoff)
    model.load_state_dict(torch.load(model_dir / "model.pt"))
    
    # Load config and get final loss
    with open(model_dir / "config.json", 'r') as f:
        saved_config = json.load(f)
    
    return model, saved_config['final_loss']

# Modify the train_models_with_sparsities function
def train_model_variations(base_config: TrainConfig, sparsities: List[float] = None, cutoffs: List[int] = None):
    """Train models with different sparsities or feature cutoffs.
    
    Args:
        base_config: Base configuration to use
        sparsities: List of sparsity values to train with, or None
        feature_cutoffs: List of feature cutoff values to train with, or None
    
    Returns:
        Tuple of (variation_values, trained_models, final_losses) where variation_values
        are either sparsities or cutoffs depending on which was specified
    """
    if sparsities is None and cutoffs is None:
        raise ValueError("Must specify either sparsities or cutoffs")
    if sparsities is not None and cutoffs is not None:
        raise ValueError("Cannot specify both sparsities and cutoffs")
        
    variations = sparsities if sparsities is not None else cutoffs
    trained_variations = []
    trained_models = []
    final_losses = []
    
    for value in variations:
        config = copy.copy(base_config)
        if sparsities is not None:
            config.sparsity = value
            variation_name = f"sparsity {value}"
        else:
            config.feature_cutoff = value
            config.use_exponential_decay = True  # Feature cutoffs only make sense with exponential decay
            variation_name = f"cutoff {value}"
        
        # Try to load existing model
        model, final_loss = load_trained_model(config)
        
        if model is not None:
            print(f"Loading existing model for {variation_name}, final loss: {final_loss:.6f}")
        else:
            model, final_loss = train_model(config)
            model.to('cpu')
            print(f"Training new model for {variation_name}, final loss: {final_loss:.6f}")
            # Save the trained model
            save_trained_model(model, config, final_loss)

        trained_variations.append(value)
        trained_models.append(model)
        final_losses.append(final_loss)
        
        print(f"Completed processing for {variation_name}, final loss: {final_loss:.6f}")
    
    return trained_variations, trained_models, final_losses

# Update the main block to show both usage patterns
if __name__ == "__main__":
    base_config = TrainConfig(
        num_epochs=500,
        weight_decay=0.01
    )
    print(f"Using device: {base_config.device}")
    
    # Example 1: Training with different sparsities
    print("\nTraining models with different sparsities:")
    sparsities = [0.5, 0.7, 0.9, 0.95, 0.99]
    variations, models, losses = train_model_variations(base_config, sparsities=sparsities)
    print("\nSparsity Training Results Summary:")
    for var, loss in zip(variations, losses):
        print(f"Sparsity: {var}, Final Loss: {loss:.6f}")
    
    # Example 2: Training with different feature cutoffs
    print("\nTraining models with different feature cutoffs:")
    cutoffs = [512, 1024, 1536, 2048]
    base_config_cutoffs = copy.copy(base_config)
    base_config_cutoffs.use_exponential_decay = True
    base_config_cutoffs.initial_prob = 0.1
    base_config_cutoffs.decay_rate = 0.002
    variations, models, losses = train_model_variations(base_config_cutoffs, feature_cutoffs=cutoffs)
    print("\nFeature Cutoff Training Results Summary:")
    for var, loss in zip(variations, losses):
        print(f"Feature Cutoff: {var}, Final Loss: {loss:.6f}")
    