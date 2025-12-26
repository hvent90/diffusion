import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from matplotlib import pyplot as plt

from src.lib.get_device import get_device
from src.pipelines.point_navigation.components.dataset import get_dataloader
from src.pipelines.point_navigation.components.model import get_model
from src.pipelines.point_navigation.components.scheduler import get_scheduler


def train() -> None:
    scheduler = get_scheduler()
    model = get_model()
    train_dataloader = get_dataloader()
    device = get_device()

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []

    for epoch in range(150):
        for step, batch in enumerate(train_dataloader):
            # Get the input data
            observation = batch["observation"].to(device)
            clean_trajectory = batch["trajectory"].to(device)

            # Sample noise to add to the images
            noise = torch.randn_like(clean_trajectory)

            # Sample random timesteps
            B = clean_trajectory.shape[0]
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (B,),
                device=device
            ).long()

            # Add noises to clean trajectories
            noisy_trajectory = scheduler.add_noise(clean_trajectory, noise, timesteps)

            # Predict noise
            noise_prediction = model(noisy_trajectory, timesteps, observation)

            # Calculate the loss
            loss = F.mse_loss(noise_prediction, noise)
            losses.append(loss.item())

            # Back propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 5 == 0:
            loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
            print(f"Epoch: {epoch+1}, loss {loss_last_epoch}")

    # Draw plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(losses)
    axs[1].plot(np.log(losses))
    plt.show()

    # Save
    os.makedirs("dist/point_navigation", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "scheduler_config": scheduler.config,
    }, "dist/point_navigation/model.pt")
    print("Model saved to dist/point_navigation/model.pt")

if __name__ == '__main__':
    train()