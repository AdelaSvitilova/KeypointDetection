import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.models.factory import get_model
from src.datasets.factory import get_dataset
from src.train.factory import get_trainer


# ---------------------------
# Helper functions
# ---------------------------
def visualize_keypoints(image, keypoints, fname=None):
    """
    Visualize a single image with keypoints overlay.
    image: np.ndarray or torch.Tensor [C, H, W] in [0,1]
    keypoints: np.ndarray [num_keypoints, 2]
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image = np.clip(image, 0, 1)

    plt.imshow(image)

    # keypoints
    for x, y, *rest in keypoints:
        plt.scatter(x, y, c="red", s=20)

    if fname:
        plt.title(fname)
    plt.axis("off")
    plt.show()


def numpy_loader(dataset, batch_size=1):
    """
    Simple generator for NumPy-based dataset.
    Yields batches of size `batch_size`.
    """
    for i in range(0, len(dataset), batch_size):
        batch_samples = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        images = np.stack([s["image"] for s in batch_samples])
        filenames = [s["filename"] for s in batch_samples]
        yield {"image": images, "filename": filenames}

# ---------------------------
# Main
# ---------------------------
def main():
    # Load configuration
    cfg = load_config("configs/config_list.yaml")
    print("Configuration loaded.")

    # Create model
    model = get_model(cfg["model"]["name"], **cfg["model"]["params"])
    print(f"Model '{cfg['model']['name']}' created.")

    # Load dataset (NumPy-based)
    dataset = get_dataset(
        name="images",
        load=cfg["predict"]["images_list"],
        num_samples=cfg["predict"]["num_samples"],
        **cfg["predict"]["params"]
    )
    print(f"Dataset loaded: {len(dataset)} images.")

    # Wrap dataset in simple NumPy loader
    loader = numpy_loader(dataset, batch_size=cfg["predict"]["batch_size"])

    # Create trainer (needed if you have special predict logic)
    trainer = get_trainer(
        backend=cfg["model"]["framework"],
        model=model,
        train_dataset=None,
        val_dataset=None,
        loss_fn=None,
        metrics=None,
        experiment_name=cfg["experiment"]["name"],
        special_mode=cfg["model"].get("special_mode"),
        **cfg["train"]
    )
    print("Trainer initialized.")

    # Predict & visualize
    for batch, preds in trainer.predict(loader):
        for img, pred, fname in zip(batch["image"], preds, batch["filename"]):
            visualize_keypoints(img, pred, fname)


if __name__ == "__main__":
    main()
