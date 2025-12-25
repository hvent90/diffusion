from diffusers import UNet2DModel

from src.lib.get_device import get_device


def get_model(image_size: int = 32) -> UNet2DModel:
    model = UNet2DModel(
        sample_size=image_size, # the target image resolution
        in_channels=3, # the number of input channels, 3 for RGB images
        out_channels=3, # the number of output channels
        layers_per_block=2, # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 128, 256), # More channels -> more parameters
        down_block_types=(
            "DownBlock2D", # a regular ResNet downsampling block
            "DownBlock2D",
            "AttnDownBlock2D", # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        )
    )
    model.to(get_device())
    return model