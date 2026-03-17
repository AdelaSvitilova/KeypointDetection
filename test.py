import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import csv

from src.utils.config import load_config
from src.models.factory import get_model
from src.datasets.factory import get_dataset
from src.metrics.factory import get_metrics
from src.train.factory import get_trainer


# ---------------------------
# Visualization functions
# ---------------------------

def visualize_heatmaps_grid(image, heatmaps, path_to_save, fname="heatmaps_grid.png",
                            annotations_label=None, keypoints=None):
    """Visualize individual heatmaps overlaid on an image in a grid.

    Args:
        image (np.ndarray or torch.Tensor): Input image [C,H,W] (C=1 or 3).
        heatmaps (np.ndarray or torch.Tensor): Heatmaps [N,64,64].
        path_to_save (str): Directory to save the figure.
        fname (str, optional): Filename for saved figure. Defaults to "heatmaps_grid.png".
        annotations_label (list of str, optional): Titles for each subplot. Defaults to None.
        keypoints (list of np.ndarray, optional): Keypoints to overlay [num_points, 2]. Defaults to None.
    """
    # Convert torch tensors to numpy
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(heatmaps):
        heatmaps = heatmaps.cpu().numpy()

    # Convert image to HWC format
    image = image.transpose(1, 2, 0) if image.ndim == 3 and image.shape[0] in (1, 3) else image
    image = np.clip(image, 0, 1)
    H, W = image.shape[:2]

    N = heatmaps.shape[0]
    cols = min(5, N)
    rows = int(np.ceil(N / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(N):
        ax = axes[i]
        ax.imshow(image)

        # Always one heatmap [64,64]
        hm = heatmaps[i]
        hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
        hm = np.clip(hm, 0, None)
        if hm.max() > 0:
            hm = hm / hm.max()

        ax.imshow(hm, cmap="turbo", alpha=0.4)

        # Overlay keypoints if provided
        if keypoints is not None:
            kp_i = np.atleast_2d(keypoints[i]).copy()
            kp_i[:, 0] = kp_i[:, 0] * W / 64  # scale x
            kp_i[:, 1] = kp_i[:, 1] * H / 64  # scale y
            ax.scatter(kp_i[:, 0], kp_i[:, 1], c="green", s=6,
                       label="Keypoint" if i == 0 else None)

        # Set subplot title
        if annotations_label is not None:
            ax.set_title(annotations_label[i])

        # Create legend (once per figure)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

        ax.axis("off")

    # Turn off unused axes
    for j in range(N, len(axes)):
        axes[j].axis("off")

    os.makedirs(path_to_save, exist_ok=True)
    plt.savefig(os.path.join(path_to_save, f"g_{fname}"), bbox_inches="tight")
    plt.close(fig)


def visualize_heatmaps_combined(image, heatmaps, path_to_save, fname="heatmaps_combined.png",
                                show_points=True, threshold=0.05, keypoints=None):
    """Visualize combined heatmaps overlaid on an image.

    Args:
        image (np.ndarray or torch.Tensor): Input image [C,H,W] or [H,W,3].
        heatmaps (np.ndarray or torch.Tensor): Heatmaps [N,64,64].
        path_to_save (str): Directory to save figure.
        fname (str, optional): Filename for saved figure. Defaults to "heatmaps_combined.png".
        show_points (bool, optional): Whether to show maxima and keypoints. Defaults to True.
        threshold (float, optional): Minimum heatmap max to include. Defaults to 0.05.
        keypoints (list of np.ndarray, optional): Keypoints [num_points, 2]. Defaults to None.
    """
    # Convert torch tensors to numpy
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(heatmaps):
        heatmaps = heatmaps.cpu().numpy()

    # Convert image to HWC if needed
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    H, W = image.shape[:2]

    # Combine heatmaps
    combined = np.zeros((H, W), dtype=np.float32)
    for hm in heatmaps:
        hm_resized = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
        hm_resized = np.clip(hm_resized, 0, None)
        if hm_resized.max() < threshold:
            continue
        if hm_resized.max() > 0:
            hm_resized /= hm_resized.max()
        combined += hm_resized

    if combined.max() > 0:
        combined /= combined.max()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.imshow(combined, cmap="turbo", alpha=0.4)

    if show_points:
        # Mark maxima of each heatmap
        for hm in heatmaps:
            hm_resized = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
            y, x = np.unravel_index(np.argmax(hm_resized), hm_resized.shape)
            ax.scatter(x, y, c="red", s=6, label="Heatmap maxima")

        # Overlay keypoints if provided
        if keypoints is not None:
            for i, kp_set in enumerate(keypoints):
                kp_set = np.atleast_2d(kp_set).copy()
                kp_set[:, 0] = kp_set[:, 0] * W / 64
                kp_set[:, 1] = kp_set[:, 1] * H / 64
                ax.scatter(kp_set[:, 0], kp_set[:, 1], c="green", s=6, marker="x",
                           label="Keypoint" if i == 0 else None)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.axis("off")
    os.makedirs(path_to_save, exist_ok=True)
    plt.savefig(os.path.join(path_to_save, fname), bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# Dataset helper
# ---------------------------

def batch_loader(dataset, batch_size=1):
    """Generator to load dataset in batches, stacking NumPy arrays.

    Args:
        dataset (list or Dataset): Dataset with __getitem__ method.
        batch_size (int, optional): Number of samples per batch. Defaults to 1.

    Yields:
        dict: Batch dictionary with stacked arrays or list of values.
    """
    n = len(dataset)
    for i in range(0, n, batch_size):
        batch_samples = [dataset[j] for j in range(i, min(i + batch_size, n))]

        batch = {}
        for key in batch_samples[0]:
            values = [s[key] for s in batch_samples]
            if isinstance(values[0], np.ndarray):
                batch[key] = np.stack(values)
            else:
                batch[key] = values  # metadata, filename, etc.

        yield batch


# ---------------------------
# Main prediction pipeline
# ---------------------------

def main():
    """Main function to load model, dataset, run predictions, visualize and save metrics."""
    # Load configuration
    cfg = load_config("configs/config_list.yaml")
    print("Configuration loaded.")
    print(cfg)

    # Create model
    model = get_model(cfg["model"]["name"], **cfg["model"]["params"])
    print(f"Model '{cfg['model']['name']}' created.")

    # Load dataset
    dataset = get_dataset(
        name=cfg["predict"]["name"],
        load=cfg["predict"]["images_list"],
        num_samples=cfg["predict"]["num_samples"],
        **cfg["predict"]["params"]
    )
    print(f"Dataset loaded: {len(dataset)} images.")

    # Load annotation labels from text file
    txt_path = os.path.join(cfg["predict"]["params"]["root_dir"],
                            cfg["predict"]["annotation_label"])
    annotations_label = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                annotations_label.append(line)
    print("Annotation labels:", annotations_label)

    # Wrap dataset in batch loader
    loader = batch_loader(dataset, batch_size=cfg["predict"]["batch_size"])

    # Metrics
    metrics = get_metrics(cfg["predict"]["metrics"])
    print(f"Metrics initialized: {cfg['predict']['metrics']}")

    # Trainer (for prediction logic)
    trainer = get_trainer(
        backend=cfg["model"]["backend"],
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

    path_to_save = os.path.join("results", cfg["experiment"]["name"], "prediction")
    os.makedirs(path_to_save, exist_ok=True)

    results = []

    # === Prediction loop ===
    for batch, preds in trainer.predict_image(loader):
        for img, pred, fname, keypoints in zip(
            batch["image"], preds, batch["filename"], batch["keypoints"]
        ):
            # Visualize individual heatmaps
            visualize_heatmaps_grid(img, pred, path_to_save, fname,
                                    annotations_label=annotations_label, keypoints=keypoints)
            # Visualize combined heatmap
            visualize_heatmaps_combined(img, pred, path_to_save, fname,
                                        show_points=True, keypoints=keypoints)

            # Compute metrics
            metrics_values = {}
            for m in metrics:
                m.reset()
                m.update(np.expand_dims(pred, axis=0), np.expand_dims(keypoints, axis=0))
                metrics_values[type(m).__name__] = m.compute()

            # Add filename to results
            row = {"filename": fname}
            row.update(metrics_values)
            results.append(row)

    # Sort results by metric
    metrics = list(metrics_values.keys())

    results_sorted = sorted(
        results,
        key=lambda x: (x[metrics[0]], -x[metrics[1]]),
        reverse=True
    )

    # Save results to CSV
    csv_path = os.path.join(path_to_save, "metrics_sorted.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename"] + list(metrics_values.keys())
        writer.writerow(header)
        for row in results_sorted:
            writer.writerow([row[h] for h in header])

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
