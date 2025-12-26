import string

from huggingface_hub import get_full_repo_name, HfApi, create_repo, ModelCard
import os
from dotenv import load_dotenv

load_dotenv()

def publish(model_name: string, content: string) -> None:
    token = os.getenv("HF_TOKEN")
    hub_model_id = get_full_repo_name(model_name, token=token)

    create_repo(hub_model_id, token=token)
    api = HfApi()
    api.upload_folder(
        folder_path=f"dist/{model_name}/scheduler", path_in_repo="", repo_id=hub_model_id, token=token
    )
    api.upload_folder(folder_path=f"dist/{model_name}/unet", path_in_repo="", repo_id=hub_model_id, token=token)
    api.upload_file(
        token=token,
        path_or_fileobj=f"dist/{model_name}/model_index.json",
        path_in_repo="model_index.json",
        repo_id=hub_model_id,
    )

    card = ModelCard(content)
    card.push_to_hub(hub_model_id, token=token)