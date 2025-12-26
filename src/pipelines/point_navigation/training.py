import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt
from diffusers import EMAModel

from src.lib.get_device import get_device
from src.pipelines.point_navigation.components.dataset import get_dataloader
from src.pipelines.point_navigation.components.model import TrajectoryDiffusionModel
from src.pipelines.point_navigation.components.scheduler import get_scheduler
from src.pipelines.point_navigation.pipeline import PointNavigationPipeline


def train() -> None:
    epochs = 150

    train_dataloader = get_dataloader(batch_size=64)
    scheduler = get_scheduler()
    device = get_device()
    model = TrajectoryDiffusionModel().to(device)
    ema = EMAModel(model.parameters(), decay=0.999)
    ema.to(device)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    losses = []

    for epoch in range(epochs):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            ema.step(model.parameters())

        if (epoch + 1) % 5 == 0:
            loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
            print(f"Epoch: {epoch+1}, loss {loss_last_epoch}")

        lr_scheduler.step()

    # Draw plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(losses)
    axs[1].plot(np.log(losses))
    plt.show()

    # Save as diffusers pipeline
    ema.copy_to(model.parameters())
    pipeline = PointNavigationPipeline(model=model, scheduler=scheduler)
    pipeline.save_pretrained("dist/point_navigation")
    print("Pipeline saved to dist/point_navigation")

if __name__ == '__main__':
    train()