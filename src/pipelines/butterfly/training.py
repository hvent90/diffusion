import numpy as np
import torch.nn.functional as F
import torch.optim
from diffusers import DDPMPipeline
from matplotlib import pyplot as plt

from src.lib.get_device import get_device
from src.pipelines.butterfly.components.dataset import get_dataloader
from src.pipelines.butterfly.components.model import get_model
from src.pipelines.butterfly.components.scheduler import get_scheduler


def train() -> None:
    scheduler = get_scheduler()
    model = get_model()
    train_dataloader = get_dataloader()
    device = get_device()

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

    losses = []

    for epoch in range(30):
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Calculate the loss
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)
            losses.append(loss.item())

            # Update the model parameters with the optimizer
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
    image_pipe = DDPMPipeline(unet=model, scheduler=scheduler)
    image_pipe.save_pretrained("dist/butterfly")

if __name__ == '__main__':
    train()