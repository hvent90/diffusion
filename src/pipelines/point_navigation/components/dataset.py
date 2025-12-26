import torch.utils.data
from typing import TypedDict
import torch

class TrajectorySample(TypedDict):
    observation: torch.Tensor # (4,)
    trajectory: torch.Tensor  # (32, 2)

class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples: int = 10000, num_steps: int = 32):
        self.data: list[TrajectorySample] = []
        # Generate and store all data here
        for _ in range(n_samples):
            # Random start and target in [-1, 1]
            start = torch.rand(2) * 2 - 1
            target = torch.rand(2) * 2 - 1

            # Lerp
            t = torch.linspace(0, 1, num_steps).unsqueeze(1) # (32, 1)
            trajectory = start + t * (target - start) # (32, 1)

            # Add small noise
            trajectory = trajectory + torch.randn_like(trajectory) * 0.02 # (32, 1)

            # Store
            observation = torch.cat([start, target]) # (4,)
            self.data.append({
                "observation": observation,
                "trajectory": trajectory
            })

    def __len__(self) -> int:
        # How many samples?
        return len(self.data)

    def __getitem__(self, idx):
        # Return one sample by index
        return self.data[idx]
    

def get_dataloader() -> torch.utils.data.DataLoader:
    dataset = TrajectoryDataset()
    return torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True
    )
