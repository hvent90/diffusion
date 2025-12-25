import torch.utils.data
from datasets import load_dataset
from torchvision import transforms

def get_dataloader() -> torch.utils.data.DataLoader:
    dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

    # Or load images from a local folder
    # dataset = load_dataset("imagefolder", data_dir"path/to/folder")

    # We'll train on 32-pixel square images, but you can try larger sizes too
    image_size = 32
    # You can lower your batch size if you're running out of GPU memory
    batch_size = 64

    # Define data augmentations
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize
            transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
            transforms.ToTensor(),  # Convert to tesnor (0, 1)
            transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
        ]
    )

    def transform(examples: dict[str, list]) -> dict[str, list]:
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )