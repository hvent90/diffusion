import torch
from diffusers import DDPMPipeline

from src.lib.get_device import get_device
from src.lib.make_grid import make_grid


def pipeline() -> None:
    device = get_device()

    # Load the butterfly pipeline
    butterfly_pipeline = DDPMPipeline.from_pretrained(
        "dist/butterfly"
    ).to(device)

    # Create 8 images
    images = butterfly_pipeline(batch_size=8).images

    # View the result
    grid = make_grid(images)
    grid.show()

if __name__ == "__main__":
    pipeline()