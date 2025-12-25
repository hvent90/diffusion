from huggingface_hub import get_full_repo_name, HfApi, create_repo, ModelCard
import os
from dotenv import load_dotenv

load_dotenv()

def publish() -> None:
    token = os.getenv("HF_TOKEN")
    model_name = "butterflies"
    hub_model_id = get_full_repo_name(model_name, token=token)

    create_repo(hub_model_id, token=token)
    api = HfApi()
    api.upload_folder(
        folder_path="dist/butterfly/scheduler", path_in_repo="", repo_id=hub_model_id, token=token
    )
    api.upload_folder(folder_path="dist/butterfly/unet", path_in_repo="", repo_id=hub_model_id, token=token)
    api.upload_file(
        token=token,
        path_or_fileobj="dist/butterfly/model_index.json",
        path_in_repo="model_index.json",
        repo_id=hub_model_id,
    )

    content = f"""
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
    """

    card = ModelCard(content)
    card.push_to_hub(hub_model_id, token=token)

if __name__ == "__main__":
    publish()