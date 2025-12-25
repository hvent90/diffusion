import numpy as np
import torch
import torchvision
from PIL import Image
from jaxtyping import Float


def show_images(x: Float[torch.Tensor, "batch channels height width"]) -> Image.Image:
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5 # map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im