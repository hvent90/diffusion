import math
import torch
import torch.nn as nn
from src.lib.get_device import get_device

def sinusoidal_embedding(timesteps: torch.Tensor, dimensions: int) -> torch.Tensor:
    """
    timesteps: (Batch size,) integers
    dimensions: embedding dimension (64)
    returns: (Batch size, dimensions)
    """
    half_dimensions = dimensions // 2

    # Create frequencies: fast,  medium slow, very slow...
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half_dimensions, device=timesteps.device) / half_dimensions
    )

    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0) # (Batch size, 32)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class TrajectoryDiffusionModel(nn.Module):
    def __init__(self, num_steps: int = 32, time_embed_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.num_steps = num_steps
        self.time_embed_dim = time_embed_dim

        trajectory_dim = num_steps * 2 # 32 * 2 = 64
        obs_dim = 4
        input_dim = trajectory_dim + time_embed_dim + obs_dim # 132

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, trajectory_dim),
        )

    def forward(
            self,
            noisy_trajectory: torch.Tensor, # (Batch size, 32, 2)
            timestep: torch.Tensor, # (Batch size,)
            observation: torch.Tensor # (Batch size, 4)
    ) -> torch.Tensor:
        batch = noisy_trajectory.shape[0]

        # Flatten: (B, 32, 2) -> (B, 64)
        flattened_trajectory = noisy_trajectory.view(batch, -1)

        # Time embed: (B,) -> (B, 64)
        time_embed = sinusoidal_embedding(timestep, self.time_embed_dim)

        # Concat: (B, 132)
        x = torch.cat([flattened_trajectory, time_embed, observation], dim=-1)

        # MLP: (B, 132) -> (B, 64)
        out = self.net(x)

        # Reshape: (B, 64) -> (B, 32, 2)
        return out.view(batch, self.num_steps, 2)

def get_model() -> TrajectoryDiffusionModel:
    return TrajectoryDiffusionModel().to(get_device())