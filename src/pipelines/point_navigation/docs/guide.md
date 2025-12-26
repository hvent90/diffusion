# Diffusion Policy: 2D Point Navigation Tutorial

Welcome! In this tutorial, you'll build a **diffusion policy** from scratch—a model that generates trajectories to navigate from a start position to a target position.

By the end, you'll understand:
- How diffusion models work for action/trajectory generation (not just images)
- How conditioning works (telling the model where to go)
- The connection between image diffusion and robotics diffusion

---

## Table of Contents

1. [Conceptual Overview](#1-conceptual-overview)
2. [Project Structure](#2-project-structure)
3. [Step 1: The Scheduler](#3-step-1-the-scheduler)
4. [Step 2: The Dataset](#4-step-2-the-dataset)
5. [Step 3: The Model](#5-step-3-the-model)
6. [Step 4: Training](#6-step-4-training)
7. [Step 5: Inference & Visualization](#7-step-5-inference--visualization)
8. [Experiments to Try](#8-experiments-to-try)

---

## 1. Conceptual Overview

### What is Diffusion Policy?

In image diffusion (like your butterfly model), we learn to denoise random noise into images:

```
noise → butterfly image
```

In **diffusion policy**, we denoise random noise into **action trajectories**:

```
noise → trajectory (sequence of positions)
```

### Our Task

```
Given:  start position (x1, y1) and target position (x2, y2)
Output: trajectory of 32 (x, y) positions connecting them
```

Visually:
```
    (target)
       *
      /
     /  <-- trajectory (32 points)
    /
   o
(start)
```

### Why Diffusion for Trajectories?

You might ask: "Why not just draw a straight line?"

For this simple case, yes—a straight line works. But diffusion shines when:
- There are **multiple valid paths** (around obstacles)
- The trajectory needs to be **smooth and natural**
- You're learning from **demonstrations** (imitation learning)

We start simple to understand the mechanics, then you can add complexity.

### The Key Insight: Conditioning

Your butterfly model is **unconditional**—it only knows "denoise into butterfly" because that's all it saw.

Our model is **conditional**—we tell it "denoise into a trajectory that goes from HERE to THERE" by passing the start/target as input.

```python
# Butterfly (unconditional)
noise_pred = model(noisy_image, timestep)

# Point navigation (conditional)
noise_pred = model(noisy_trajectory, timestep, observation)
#                                              ^
#                            [start_x, start_y, target_x, target_y]
```

The model learns: "given this observation, what trajectory should I produce?"

---

## 2. Project Structure

Create this structure:

```
src/pipelines/point_navigation/
├── __init__.py
├── docs/
│   └── guide.md          <-- you are here
├── components/
│   ├── __init__.py
│   ├── scheduler.py      <-- Step 1
│   ├── dataset.py        <-- Step 2
│   └── model.py          <-- Step 3
├── training.py           <-- Step 4
└── visualize.py          <-- Step 5
```

---

## 3. Step 1: The Scheduler

**File:** `components/scheduler.py`

The scheduler controls how noise is added during training and removed during inference. We reuse the same `DDPMScheduler` from your butterfly model.

### Your Task

Create a function `get_scheduler()` that returns a `DDPMScheduler` with:
- 1000 training timesteps
- "squaredcos_cap_v2" beta schedule (smooth noise schedule)

### Concepts to Understand

**Q: What does the scheduler do during training?**

It adds noise to clean data at various "timesteps". Timestep 0 = nearly clean, timestep 999 = pure noise.

```python
noisy = scheduler.add_noise(clean, noise, timestep)
```

**Q: What does it do during inference?**

It removes noise step-by-step, going from timestep 999 to 0:

```python
for t in reversed(range(1000)):
    noise_pred = model(noisy, t, observation)
    noisy = scheduler.step(noise_pred, t, noisy).prev_sample
# noisy is now clean!
```

### Implementation Notes

- Import `DDPMScheduler` from diffusers
- The scheduler works on any tensor shape—images, trajectories, whatever
- No changes needed from the butterfly version

### Reference

Look at your butterfly scheduler: `src/pipelines/butterfly/components/scheduler.py`

---

## 4. Step 2: The Dataset

**File:** `components/dataset.py`

We need training data: pairs of (observation, trajectory) where:
- `observation` = [start_x, start_y, target_x, target_y]
- `trajectory` = 32 points from start to target

Since we don't have real robot data, we **generate synthetic expert demonstrations**—simple straight-line paths with a bit of noise.

### Your Task

Create a `TrajectoryDataset` class that:

1. In `__init__`: Generate `n_samples` random start/target pairs in the [-1, 1] square
2. For each pair, generate a trajectory using linear interpolation
3. Add small Gaussian noise for variation
4. Return dictionaries with "observation" and "trajectory" keys

### The Math: Linear Interpolation

To create a smooth path from start to target:

```python
# t goes from 0 to 1 in 32 steps
t = torch.linspace(0, 1, num_steps).unsqueeze(1)  # shape: (32, 1)

# Linear interpolation formula:
# trajectory = start + t * (target - start)
#
# When t=0: trajectory[0] = start
# When t=1: trajectory[-1] = target

trajectory = start + t * (target - start)  # shape: (32, 2)
```

### Why Add Noise?

Pure straight lines are "too perfect". The model might learn shortcuts instead of truly understanding the denoising process. Adding small random perturbations:

```python
trajectory = trajectory + torch.randn_like(trajectory) * 0.02
```

This creates slightly wiggly paths while still going from start to target.

### Complete Implementation

```python
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    """Dataset of synthetic trajectories from random start to random target."""

    def __init__(self, n_samples: int = 10000, num_steps: int = 32):
        self.num_steps = num_steps
        self.data = []

        for _ in range(n_samples):
            # Random start and target in [-1, 1] x [-1, 1]
            start = torch.rand(2) * 2 - 1   # (2,)
            target = torch.rand(2) * 2 - 1  # (2,)

            # Linear interpolation
            t = torch.linspace(0, 1, num_steps).unsqueeze(1)  # (32, 1)
            trajectory = start + t * (target - start)         # (32, 2)

            # Add small noise for variation
            trajectory = trajectory + torch.randn_like(trajectory) * 0.02

            # Store as dict
            observation = torch.cat([start, target])  # (4,)
            self.data.append({
                "observation": observation,
                "trajectory": trajectory,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(batch_size: int = 256) -> DataLoader:
    dataset = TrajectoryDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

### Validation

After implementing, test it:

```python
dataset = TrajectoryDataset(n_samples=5)
sample = dataset[0]
print(f"Observation shape: {sample['observation'].shape}")  # Should be torch.Size([4])
print(f"Trajectory shape: {sample['trajectory'].shape}")    # Should be torch.Size([32, 2])
print(f"Start: {sample['observation'][:2]}")
print(f"Target: {sample['observation'][2:]}")
print(f"First point: {sample['trajectory'][0]}")   # Should be close to start
print(f"Last point: {sample['trajectory'][-1]}")   # Should be close to target
```

---

## 5. Step 3: The Model

**File:** `components/model.py`

This is the brain—a neural network that predicts noise given:
- A noisy trajectory
- The current diffusion timestep
- The observation (where to go)

### Why Not Use UNet2D?

Your butterfly model uses `UNet2DModel` because images have 2D spatial structure—nearby pixels are related. UNet's convolutions exploit this.

Trajectories are **1D sequences**, not 2D grids. For this hello-world, a simple MLP works fine. (Real robotics systems often use 1D temporal convolutions or transformers.)

### Architecture Overview

```
Inputs:
  noisy_trajectory: (B, 32, 2) -> flatten -> (B, 64)
  timestep: (B,) -> sinusoidal embedding -> (B, 64)
  observation: (B, 4) -> keep as-is -> (B, 4)

Concatenate: (B, 64 + 64 + 4) = (B, 132)

MLP:
  Linear(132 -> 256) -> ReLU
  Linear(256 -> 256) -> ReLU
  Linear(256 -> 64)

Reshape output: (B, 64) -> (B, 32, 2)
```

### Sinusoidal Time Embedding

This is how we tell the network "what noise level are we at?"

The same technique from the original Transformer paper—encode a scalar (timestep) as a vector using sin/cos at different frequencies. This gives the model a rich representation of the timestep:

```python
import math
import torch


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal positional embeddings for timesteps.

    Args:
        timesteps: (B,) tensor of integer timestep values
        dim: embedding dimension (should be even)

    Returns:
        (B, dim) tensor of embeddings
    """
    half_dim = dim // 2

    # Create frequency bands: 1, 1/10^(1/half), 1/10^(2/half), ..., 1/10000
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device) / half_dim
    )

    # Compute angles: timestep * frequency for each frequency band
    # (B,) x (half_dim,) -> (B, half_dim) via broadcasting
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)

    # Concatenate sin and cos to get full embedding
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

**Why this works:** Different frequencies capture different scales. Low frequencies change slowly (distinguish t=0 from t=500), high frequencies change fast (distinguish t=100 from t=101).

### Complete Implementation

```python
import math
import torch
import torch.nn as nn

from src.lib.get_device import get_device


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal embeddings for diffusion timesteps."""
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device) / half_dim
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TrajectoryDiffusionModel(nn.Module):
    """
    Simple MLP that predicts noise in trajectories.

    Takes:
      - noisy_trajectory: (B, num_steps, 2) - noisy x,y positions
      - timestep: (B,) - diffusion timestep (0 to 999)
      - observation: (B, 4) - [start_x, start_y, target_x, target_y]

    Returns:
      - noise_pred: (B, num_steps, 2) - predicted noise
    """

    def __init__(
        self,
        num_steps: int = 32,
        time_embed_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.time_embed_dim = time_embed_dim

        trajectory_dim = num_steps * 2  # 32 * 2 = 64
        obs_dim = 4  # start_x, start_y, target_x, target_y

        input_dim = trajectory_dim + time_embed_dim + obs_dim  # 64 + 64 + 4 = 132

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, trajectory_dim),
        )

    def forward(
        self,
        noisy_trajectory: torch.Tensor,  # (B, 32, 2)
        timestep: torch.Tensor,          # (B,)
        observation: torch.Tensor,       # (B, 4)
    ) -> torch.Tensor:
        B = noisy_trajectory.shape[0]

        # Flatten trajectory: (B, 32, 2) -> (B, 64)
        flat_traj = noisy_trajectory.view(B, -1)

        # Time embedding: (B,) -> (B, 64)
        time_emb = sinusoidal_embedding(timestep, self.time_embed_dim)

        # Concatenate all inputs: (B, 64 + 64 + 4) = (B, 132)
        x = torch.cat([flat_traj, time_emb, observation], dim=-1)

        # MLP forward pass: (B, 132) -> (B, 64)
        out = self.net(x)

        # Reshape to trajectory: (B, 64) -> (B, 32, 2)
        noise_pred = out.view(B, self.num_steps, 2)

        return noise_pred


def get_model() -> TrajectoryDiffusionModel:
    """Factory function to create model on the appropriate device."""
    model = TrajectoryDiffusionModel()
    return model.to(get_device())
```

### Testing Your Model

Run this to verify shapes are correct:

```python
from src.lib.get_device import get_device

device = get_device()
model = get_model()

B = 4  # batch size
noisy_traj = torch.randn(B, 32, 2).to(device)
timesteps = torch.randint(0, 1000, (B,)).to(device)
obs = torch.randn(B, 4).to(device)

output = model(noisy_traj, timesteps, obs)
print(f"Output shape: {output.shape}")  # Should be torch.Size([4, 32, 2])

# Check parameter count
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")  # Should be around 100k
```

---

## 6. Step 4: Training

**File:** `training.py`

This follows the exact same pattern as your butterfly training! The only differences:
- We work with trajectories instead of images
- We pass observation to the model

### The Training Loop Explained

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Get data
        obs = batch["observation"].to(device)        # (B, 4)
        clean_traj = batch["trajectory"].to(device)  # (B, 32, 2)

        # 2. Sample random noise (same shape as trajectory)
        noise = torch.randn_like(clean_traj)

        # 3. Sample random timesteps for each item in batch
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps,
            (clean_traj.shape[0],), device=device
        ).long()

        # 4. Add noise to clean trajectories
        # At timestep 0: noisy_traj ≈ clean_traj
        # At timestep 999: noisy_traj ≈ pure noise
        noisy_traj = scheduler.add_noise(clean_traj, noise, timesteps)

        # 5. Model predicts what noise was added
        noise_pred = model(noisy_traj, timesteps, obs)

        # 6. Loss = how wrong was the noise prediction?
        loss = F.mse_loss(noise_pred, noise)

        # 7. Gradient descent
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Why This Works

The model learns: "given a noisy trajectory at timestep t, and the observation (start, target), what noise was added?"

Once trained, we can reverse the process: start with pure noise, repeatedly ask the model "what noise is in this?", subtract it, and end up with a clean trajectory that goes from start to target.

### Complete Implementation

```python
import os
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from src.lib.get_device import get_device
from src.pipelines.point_navigation.components.dataset import get_dataloader
from src.pipelines.point_navigation.components.model import get_model
from src.pipelines.point_navigation.components.scheduler import get_scheduler


def train(num_epochs: int = 100) -> None:
    device = get_device()
    print(f"Training on {device}")

    # Initialize components
    scheduler = get_scheduler()
    model = get_model()
    dataloader = get_dataloader(batch_size=256)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch in dataloader:
            obs = batch["observation"].to(device)
            clean_traj = batch["trajectory"].to(device)

            # Sample noise
            noise = torch.randn_like(clean_traj)

            # Sample timesteps
            B = clean_traj.shape[0]
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # Add noise
            noisy_traj = scheduler.add_noise(clean_traj, noise, timesteps)

            # Predict noise
            noise_pred = model(noisy_traj, timesteps, obs)

            # Compute loss
            loss = F.mse_loss(noise_pred, noise)

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} | Loss: {avg_loss:.6f}")

    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss (Log)")

    plt.tight_layout()
    plt.show()

    # Save model
    os.makedirs("dist/point_navigation", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "scheduler_config": scheduler.config,
    }, "dist/point_navigation/model.pt")
    print("Model saved to dist/point_navigation/model.pt")


if __name__ == "__main__":
    train()
```

### Expected Behavior

- **First few epochs:** Loss around 0.3-0.5 (model is random)
- **After 20 epochs:** Loss around 0.05-0.1
- **After 100 epochs:** Loss around 0.01-0.02
- **Training time:** ~30 seconds to 2 minutes depending on hardware

If loss doesn't decrease, check:
- Is the scheduler adding noise correctly?
- Is observation being passed to the model?
- Are you computing loss between noise_pred and noise (not clean_traj)?

---

## 7. Step 5: Inference & Visualization

**File:** `visualize.py`

Now the fun part—generate trajectories and watch them emerge from noise!

### The Inference Loop (Reverse Diffusion)

During training, we added noise. During inference, we remove it:

```python
def generate_trajectory(model, scheduler, observation, device):
    """Generate a trajectory given start/target observation."""
    model.eval()

    with torch.no_grad():
        # Start with pure random noise
        trajectory = torch.randn(1, 32, 2).to(device)

        # Set up scheduler for inference
        scheduler.set_timesteps(1000)  # Use all 1000 steps

        # Denoise step by step: 999 -> 998 -> ... -> 0
        for t in scheduler.timesteps:
            # Model predicts noise at this timestep
            noise_pred = model(
                trajectory,
                torch.tensor([t], device=device),
                observation.unsqueeze(0).to(device),
            )

            # Scheduler removes some noise based on prediction
            trajectory = scheduler.step(noise_pred, t, trajectory).prev_sample

    return trajectory.squeeze(0).cpu()  # (32, 2)
```

### Simple Visualization

```python
import torch
import matplotlib.pyplot as plt

from src.lib.get_device import get_device
from src.pipelines.point_navigation.components.model import TrajectoryDiffusionModel
from src.pipelines.point_navigation.components.scheduler import get_scheduler


def load_model():
    """Load trained model from checkpoint."""
    device = get_device()
    checkpoint = torch.load("dist/point_navigation/model.pt", map_location=device)

    model = TrajectoryDiffusionModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


def generate_trajectory(model, scheduler, observation, device):
    """Generate trajectory via reverse diffusion."""
    with torch.no_grad():
        trajectory = torch.randn(1, 32, 2).to(device)
        scheduler.set_timesteps(1000)

        for t in scheduler.timesteps:
            noise_pred = model(
                trajectory,
                torch.tensor([t], device=device),
                observation.unsqueeze(0).to(device),
            )
            trajectory = scheduler.step(noise_pred, t, trajectory).prev_sample

    return trajectory.squeeze(0).cpu()


def visualize():
    model, device = load_model()
    scheduler = get_scheduler()

    # Generate a few trajectories with random start/target
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax in axes:
        # Random start and target
        start = torch.rand(2) * 2 - 1
        target = torch.rand(2) * 2 - 1
        observation = torch.cat([start, target])

        # Generate trajectory
        trajectory = generate_trajectory(model, scheduler, observation, device)

        # Plot
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Start (green) and target (red)
        ax.scatter(*start.numpy(), c="green", s=100, zorder=5, label="Start")
        ax.scatter(*target.numpy(), c="red", s=100, zorder=5, label="Target")

        # Trajectory
        traj_np = trajectory.numpy()
        ax.plot(traj_np[:, 0], traj_np[:, 1], "b.-", alpha=0.7, markersize=3)

        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize()
```

### Animated Visualization (Watch Denoising Happen)

This shows the trajectory emerging from noise step by step:

```python
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.lib.get_device import get_device
from src.pipelines.point_navigation.components.model import TrajectoryDiffusionModel
from src.pipelines.point_navigation.components.scheduler import get_scheduler


def animate_denoising():
    """Animate the reverse diffusion process."""
    device = get_device()

    # Load model
    checkpoint = torch.load("dist/point_navigation/model.pt", map_location=device)
    model = TrajectoryDiffusionModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scheduler = get_scheduler()

    # Random start/target
    start = torch.rand(2) * 2 - 1
    target = torch.rand(2) * 2 - 1
    observation = torch.cat([start, target]).to(device)

    # Collect trajectories at each denoising step
    trajectories = []
    trajectory = torch.randn(1, 32, 2).to(device)

    scheduler.set_timesteps(1000)

    with torch.no_grad():
        for t in scheduler.timesteps:
            # Save current state every 10 steps (100 frames total)
            if t % 10 == 0:
                trajectories.append(trajectory.squeeze(0).cpu().numpy())

            noise_pred = model(
                trajectory,
                torch.tensor([t], device=device),
                observation.unsqueeze(0),
            )
            trajectory = scheduler.step(noise_pred, t, trajectory).prev_sample

    # Add final result
    trajectories.append(trajectory.squeeze(0).cpu().numpy())

    # Animate
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Static elements
    ax.scatter(*start.numpy(), c="green", s=100, zorder=5, label="Start")
    ax.scatter(*target.numpy(), c="red", s=100, zorder=5, label="Target")
    ax.legend()

    # Animated line
    (line,) = ax.plot([], [], "b.-", alpha=0.7, markersize=4)
    title = ax.set_title("")

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        traj = trajectories[frame]
        line.set_data(traj[:, 0], traj[:, 1])
        progress = frame / len(trajectories) * 100
        title.set_text(f"Denoising: {progress:.0f}%")
        return (line,)

    ani = FuncAnimation(
        fig, update, init_func=init,
        frames=len(trajectories), interval=50, blit=True
    )

    plt.show()


if __name__ == "__main__":
    animate_denoising()
```

---

## 8. Experiments to Try

Once you have the basic version working, try these extensions:

### Easy Experiments

- [ ] **Change trajectory length:** Try 16 or 64 steps instead of 32
- [ ] **Vary noise amount:** Change the 0.02 noise in dataset to 0.05 or 0.0
- [ ] **Different learning rates:** Try 1e-4, 5e-4, 5e-3
- [ ] **Fewer diffusion steps at inference:** Use scheduler.set_timesteps(100) instead of 1000

### Medium Experiments

- [ ] **Add obstacles:** Modify the dataset to generate curved paths around a circular obstacle at (0, 0)
- [ ] **Deeper network:** Add more hidden layers (256 -> 256 -> 256 -> 256)
- [ ] **Larger hidden dim:** Try hidden_dim=512
- [ ] **Dropout:** Add nn.Dropout(0.1) between layers for regularization

### Advanced Experiments

- [ ] **1D ConvNet:** Replace MLP with temporal convolutions over the trajectory
- [ ] **Predict velocity:** Output both position and velocity at each step
- [ ] **Classifier-free guidance:** Train with observation dropout, use guidance scale at inference
- [ ] **Variable-length trajectories:** Pad shorter trajectories, mask the loss

---

## Quick Reference

### Tensor Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| observation | (B, 4) | [start_x, start_y, target_x, target_y] |
| trajectory | (B, 32, 2) | 32 timesteps, each with (x, y) |
| timesteps | (B,) | Diffusion timestep per sample (0-999) |
| time_embedding | (B, 64) | Sinusoidal encoding of timestep |
| noise | (B, 32, 2) | Gaussian noise, same shape as trajectory |
| noise_pred | (B, 32, 2) | Model's prediction of the noise |

### Key Equations

**Training (forward diffusion):**
```
noisy_traj = sqrt(alpha_t) * clean_traj + sqrt(1 - alpha_t) * noise
```

**Inference (reverse diffusion, simplified):**
```
traj_{t-1} = (traj_t - noise_pred * some_factor) / another_factor + small_noise
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| num_epochs | 100 | More is usually better |
| batch_size | 256 | Larger = more stable gradients |
| learning_rate | 1e-3 | Standard for Adam |
| hidden_dim | 256 | Model capacity |
| time_embed_dim | 64 | Timestep representation |
| num_train_timesteps | 1000 | Noise schedule granularity |
| trajectory_length | 32 | Points per trajectory |
| dataset_size | 10000 | Training examples |

---

## Troubleshooting

### Loss doesn't decrease
- Check that `scheduler.add_noise()` is being called with correct arguments
- Verify observation is passed to model (not just noisy_traj and timestep)
- Make sure you're computing loss on noise, not clean trajectory
- Try a smaller learning rate

### Generated trajectories are random noise
- Train for more epochs (at least 50)
- Check that inference uses `scheduler.set_timesteps()` before the loop
- Verify `model.eval()` is called before inference
- Make sure you're loading the saved weights correctly

### Trajectories don't start at start or end at target
- The model only learns from data—it doesn't "know" the trajectory should hit these points exactly
- For stronger conditioning: train longer, use a larger model, or add an auxiliary loss
- The start/end will be close but not exact; this is normal for diffusion

### Training is very slow
- Reduce dataset size to 1000 for debugging
- Reduce num_epochs to 20
- Use a smaller batch_size if GPU memory is limited

---

## Summary: The Big Picture

You've built a minimal diffusion policy that:

1. **Learns from demonstrations:** Trajectories from start to target
2. **Uses conditioning:** The observation tells it where to go
3. **Generates via denoising:** Start with noise, iteratively clean it up

This is the same core loop used in:
- Stable Diffusion (text → image)
- Real robot manipulation (observation → action trajectory)
- Motion planning (start/goal → path)

The only differences are:
- What you're denoising (images vs. trajectories vs. robot actions)
- What you condition on (text vs. observations vs. goals)
- The architecture (UNet vs. MLP vs. Transformer)

The math is the same. Now go build something cool!
