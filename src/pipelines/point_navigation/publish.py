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
    - conditional-navigation
    - diffusion-models-class
    ---

    # Simple Point Navigation

    This generates a path between two points in a 2D grid. Basically AGI.

    ## Usage

    ```python
    from diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained("{hub_model_id}", trust_remote_code=True)
    trajectory = pipeline(start=[0, 0], target=[0.8, 0.8]).trajectories[0]
    print(trajectory)  # 32 waypoints from start to target
    ```
    """)

if __name__ == "__main__":
    publish_butterfly()