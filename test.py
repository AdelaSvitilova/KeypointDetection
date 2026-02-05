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
# Helper functions
# ---------------------------
def visualize_heatmaps_grid(image, heatmaps, path_to_save, fname="heatmaps_grid.png", annotations_label=None):
    """
    image: [C, 256, 256]
    heatmaps: [N, 64, 64]
    """

    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(heatmaps):
        heatmaps = heatmaps.cpu().numpy()

    # image -> HWC
    image = image.transpose(1, 2, 0)
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

        # 🔑 tady je fix: VŽDY jen jedna heatmapa [64,64]
        hm = heatmaps[i]

        # resize 64x64 → 256x256
        hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)

        hm = np.clip(hm, 0, None)
        if hm.max() > 0:
            hm = hm / hm.max()

        ax.imshow(hm, cmap="turbo", alpha=0.4)
        ax.set_title(annotations_label[i])
        ax.axis("off")

    # vypnout prázdné subploty
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    os.makedirs(path_to_save, exist_ok=True)
    plt.savefig(os.path.join(path_to_save, f"g_{fname}"), bbox_inches="tight")
    plt.close(fig)

def visualize_heatmaps_combined(
    image,
    heatmaps,
    path_to_save,
    fname="heatmaps_combined.png",
    show_points=True,
    threshold=0.05
):
    """
    image:    [C,H,W] nebo [H,W,3]
    heatmaps: [N,64,64]
    """

    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(heatmaps):
        heatmaps = heatmaps.cpu().numpy()

    # image -> HWC
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    H, W = image.shape[:2]

    # složení heatmap
    combined = np.zeros((H, W), dtype=np.float32)

    for hm in heatmaps:
        hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
        hm = np.clip(hm, 0, None)
        if hm.max() < threshold:
            continue  # přeskoč heatmapy, kde není žádný signál
        if hm.max() > 0:
            hm = hm / hm.max()
        combined += hm

    # normalizace výsledku
    if combined.max() > 0:
        combined /= combined.max()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.imshow(combined, cmap="turbo", alpha=0.4)

    # vykreslení bodů maxima
    if show_points:
        for hm in heatmaps:
            hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            ax.scatter(x, y, c="red", s=6)

    ax.axis("off")

    os.makedirs(path_to_save, exist_ok=True)
    plt.savefig(os.path.join(path_to_save, fname), bbox_inches="tight")
    plt.close(fig)


def batch_loader(dataset, batch_size=1):
    """
    Framework-agnostic generator for any dataset with __getitem__.
    Yields dictionaries with batch_size samples stacked.
    """
    n = len(dataset)
    for i in range(0, n, batch_size):
        batch_samples = [dataset[j] for j in range(i, min(i + batch_size, n))]

        batch = {}
        # stack all keys that are np.ndarray
        for key in batch_samples[0]:
            values = [s[key] for s in batch_samples]
            if isinstance(values[0], np.ndarray):
                batch[key] = np.stack(values)
            else:
                batch[key] = values  # e.g., filename or metadata

        yield batch

# ---------------------------
# Main
# ---------------------------
def main():
    # Load configuration
    cfg = load_config("configs/config_list.yaml")
    print("Configuration loaded.")
    print(cfg)

    # Create model
    model = get_model(cfg["model"]["name"], **cfg["model"]["params"])
    print(f"Model '{cfg['model']['name']}' created.")

    # Load dataset (NumPy-based)
    dataset = get_dataset(
        name=cfg["predict"]["name"],
        load=cfg["predict"]["images_list"],
        num_samples=cfg["predict"]["num_samples"],
        **cfg["predict"]["params"]
    )
    print(f"Dataset loaded: {len(dataset)} images.")

    # cesta k txt souboru
    txt_path = os.path.join(cfg["predict"]["params"]["root_dir"],cfg["predict"]["annotation_label"])

    # vytvoříme list pro uložení
    annotations_label = []

    # otevření souboru a čtení řádek po řádku
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()        # odstraní \n a bílé znaky
            if line:                   # ignoruje prázdné řádky
                annotations_label.append(line)

    print(annotations_label)

    # Wrap dataset in simple NumPy loader
    loader = batch_loader(dataset, batch_size=cfg["predict"]["batch_size"])

    # === Loss and metrics ===
    metrics = get_metrics(cfg["predict"]["metrics"])
    print(f"Metrics: {cfg['predict']['metrics']}")

    # Create trainer (needed if you have special predict logic)
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

    path_to_save = os.path.join("results",cfg["experiment"]["name"], "prediction")
    os.makedirs(path_to_save, exist_ok=True)

    results = []

    for batch, preds in trainer.predict(loader):
        for img, pred, fname, keypoints in zip(
            batch["image"], preds, batch["filename"], batch["keypoints"]
        ):
            # Vizualizace
            visualize_heatmaps_grid(img, pred[0], path_to_save, fname, annotations_label=annotations_label)
            visualize_heatmaps_combined(img, pred[0], path_to_save, fname, show_points=True)

            # Výpočet metrik
            metrics_values = {}
            for m in metrics:
                m.reset()
                m.update(np.expand_dims(pred[0], axis=0), np.expand_dims(keypoints, axis=0))
                metrics_values[type(m).__name__] = m.compute()

            # Přidáme filename
            row = {"filename": fname}
            row.update(metrics_values)
            results.append(row)

    # Seřazení podle první metriky
    first_metric = list(metrics_values.keys())[0]
    results_sorted = sorted(results, key=lambda x: x[first_metric], reverse=True)  # nejvyšší nahoře

    # Uložení do CSV
    csv_path = f"{path_to_save}/metrics_sorted.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # hlavička
        header = ["filename"] + list(metrics_values.keys())
        writer.writerow(header)

        # řádky
        for row in results_sorted:
            writer.writerow([row[h] for h in header])

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
