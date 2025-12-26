from huggingface_hub import get_full_repo_name, HfApi, create_repo, ModelCard
import os
from dotenv import load_dotenv
from src.lib.publish import publish

load_dotenv()

def publish_butterfly() -> None:
    model_name = "butterflies"
    token = os.getenv("HF_TOKEN")
    hub_model_id = get_full_repo_name(model_name, token=token)
    publish(model_name, f"""
    ---
    license: mit
    tags:
    - pytorch
    - diffusers
    - unconditional-image-generation
    - diffusion-models-class
    ---

    # Model Card for Unit 1 of the [Diffusion Models Class 🧨](https://github.com/huggingface/diffusion-models-class)

    This model is a diffusion model for unconditional image generation of cute 🦋.

    ## Usage

    ```python
    from diffusers import DDPMPipeline

    pipeline = DDPMPipeline.from_pretrained('{hub_model_id}')
    image = pipeline().images[0]
    image
    ```
    """)

if __name__ == "__main__":
    publish_butterfly()