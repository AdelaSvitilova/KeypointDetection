from src.utils.config import load_config
from src.models.factory import get_model
from src.datasets.factory import get_dataset
from src.train.factory import get_trainer
import numpy as np
import os
import csv


def main():
    """Main entry point for the prediction pipeline."""

    # === Config ===
    cfg = load_config("configs/config_list.yaml")
    print("Loaded configuration:", cfg)

    # === Model ===
    model = get_model(cfg["model"]["name"], **cfg["model"]["params"])
    print(f"Model '{cfg['model']['name']}' created.")

    # === Dataset (upraveno) ===
    dataset = get_dataset(
        name=cfg["predict"]["name"], 
        load=cfg["predict"]["images_list"],
        num_samples=cfg["predict"]["num_samples"],
        keypoint_format=cfg["keypoint_format"],
        **cfg["predict"]["params"]
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # === Labels ===
    txt_path = os.path.join(
        cfg["predict"]["params"]["root_dir"],
        cfg["predict"]["annotation_label"]
    )

    annotations_label = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                annotations_label.append(line)

    print("Annotation labels:", annotations_label)

    # === Trainer ===
    trainer = get_trainer(
        backend=cfg["model"]["backend"],
        model=model,
        train_dataset=None,
        val_dataset=None,
        loss_fn=None,
        metrics=None,
        experiment_name=cfg["experiment"]["name"],
        keypoint_format=cfg["keypoint_format"],
        special_mode=cfg["model"]["special_mode"],
        **cfg["train"]
    )

    # === Output path ===
    output_dir = os.path.join("results", cfg["experiment"]["name"])
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "predictions_vs_annotations.csv")

    # === CSV writing ===
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # === HEADER (dynamický) ===
        header = ["filename"]

        for label in annotations_label:
            header.extend([
                f"{label}_ann_x", f"{label}_ann_y",
                f"{label}_pred_x", f"{label}_pred_y"
            ])

        writer.writerow(header)

        # === Prediction loop ===
        for item, heatmaps_np in trainer.predict(dataset, cfg["predict"]["batch_size"]):

            preds_np = heatmaps_np  # (B, K, H, W)
            filenames = item["filename"]
            annotations = item["keypoints"]  # (B, K, 2)

            for batch_idx in range(preds_np.shape[0]):
                row = []

                filename = filenames[batch_idx]
                row.append(filename)

                for keypoint_idx in range(preds_np.shape[1]):
                    heatmap = preds_np[batch_idx, keypoint_idx]

                    # prediction
                    y_pred, x_pred = np.unravel_index(
                        np.argmax(heatmap), heatmap.shape
                    )

                    # annotation
                    x_ann = annotations[batch_idx][keypoint_idx][0]
                    y_ann = annotations[batch_idx][keypoint_idx][1]

                    row.extend([
                        int(x_ann), int(y_ann),
                        int(x_pred), int(y_pred)
                    ])

                writer.writerow(row)

    print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    main()