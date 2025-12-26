from huggingface_hub import get_full_repo_name, HfApi, create_repo, ModelCard
import os
from dotenv import load_dotenv

load_dotenv()

def publish_point_navigation() -> None:
    model_name = "point_navigation"
    token = os.getenv("HF_TOKEN")
    hub_model_id = get_full_repo_name(model_name, token=token)

    create_repo(hub_model_id, token=token, exist_ok=True)
    api = HfApi()

    # Upload model and scheduler folders
    api.upload_folder(
        folder_path=f"dist/{model_name}/model", path_in_repo="model", repo_id=hub_model_id, token=token
    )
    api.upload_folder(
        folder_path=f"dist/{model_name}/scheduler", path_in_repo="scheduler", repo_id=hub_model_id, token=token
    )

    # Upload model_index.json
    api.upload_file(
        token=token,
        path_or_fileobj=f"dist/{model_name}/model_index.json",
        path_in_repo="model_index.json",
        repo_id=hub_model_id,
    )

    # Upload pipeline.py for trust_remote_code
    api.upload_file(
        token=token,
        path_or_fileobj="src/pipelines/point_navigation/pipeline.py",
        path_in_repo="pipeline.py",
        repo_id=hub_model_id,
    )

    # Model card
    card = ModelCard(f"""
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
    card.push_to_hub(hub_model_id, token=token)

if __name__ == "__main__":
    publish_point_navigation()
