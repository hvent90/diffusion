import torch
import matplotlib.pyplot as plt

from src.pipelines.point_navigation.pipeline import PointNavigationPipeline


def visualize():
    """Generate and plot several trajectories."""
    pipeline = PointNavigationPipeline.from_pretrained("dist/point_navigation")

    # Generate 6 trajectories in a grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax in axes:
        # Random start and target
        start = torch.rand(2) * 2 - 1
        target = torch.rand(2) * 2 - 1

        # Generate trajectory
        result = pipeline(start=start.tolist(), target=target.tolist())
        trajectory = result.trajectories[0].cpu()

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
