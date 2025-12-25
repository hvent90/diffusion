import torch
from PIL import Image
from collections.abc import Sequence

def make_grid(images: Sequence[Image.Image], size: int = 64) -> Image.Image:
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im