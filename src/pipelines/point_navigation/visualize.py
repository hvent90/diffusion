import torch
import matplotlib.pyplot as plt

from src.lib.get_device import get_device
from src.pipelines.point_navigation.components.model import TrajectoryDiffusionModel
from src.pipelines.point_navigation.components.scheduler import get_scheduler


def load_model():
    """Load trained model from checkpoint."""
    device = get_device()
    checkpoint = torch.load("dist/point_navigation/model.pt", map_location=device, weights_only=False)
    model = TrajectoryDiffusionModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def generate_trajectory(model, scheduler, observation, device):
    """Generate a trajectory via reverse diffusion."""
    with torch.no_grad():
        # Start with pure noise
        trajectory = torch.randn(1, 32, 2).to(device)

        # Set up scheduler for inference
        scheduler.set_timesteps(1000)

        # Denoise: 999 -> 998 -> ... -> 0
        for t in scheduler.timesteps:
            noise_pred = model(
                trajectory,
                torch.tensor([t], device=device),
                observation.unsqueeze(0).to(device),
            )
            trajectory = scheduler.step(noise_pred, t, trajectory).prev_sample

    return trajectory.squeeze(0).cpu()  # (32, 2)


def visualize():
    """Generate and plot several trajectories."""
    model, device = load_model()
    scheduler = get_scheduler()

    # Generate 6 trajectories in a grid
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
