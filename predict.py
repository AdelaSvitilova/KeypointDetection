from src.utils.config import load_config
from src.models.factory import get_model
from src.datasets.factory import get_dataset
from src.train.factory import get_trainer
import json
import numpy as np
import os


def main():
    """Main entry point for the training pipeline."""
    # Load configuration from YAML
    cfg = load_config("configs/config_list.yaml")
    print("Loaded configuration:", cfg)

    # === Model creation ===
    model = get_model(cfg["model"]["name"], **cfg["model"]["params"])
    print(f"Model '{cfg['model']['name']}' created.")

    # === Dataset loading ===
    # Training dataset
    dataset = get_dataset(
        name="images",
        load=cfg["predict"]["images_list"],
        num_samples=cfg["predict"]["num_samples"],
        keypoint_format=cfg["keypoint_format"],
        **cfg["predict"]["params"]
    )
    print(f"Training dataset loaded: {len(dataset)} samples")

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

    # === Trainer creation ===
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
    print(f"Trainer initialized for backend '{cfg['model']['backend']}'")

    results = []

    for item, heatmaps_np in trainer.predict(dataset, checkpoint=cfg["predict"]["model"], batch_size=cfg["predict"]["batch_size"]):

        preds_np = heatmaps_np  # (B, K, H, W)

        filenames = item["filename"]  # většinou list délky B

        for batch_idx in range(preds_np.shape[0]):
            sample_result = {}

            filename = filenames[batch_idx]

            for keypoint_idx in range(preds_np.shape[1]):
                heatmap = preds_np[batch_idx, keypoint_idx]

                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

                label_name = annotations_label[keypoint_idx]

                sample_result[label_name] = {
                    "x": int(x),
                    "y": int(y),
                    # "value": float(heatmap[y, x])
                }

            results.append({
                "filename": filename,
                "keypoints": sample_result
            })

    # Uložení do JSON
    path_for_json=os.path.join("results",cfg["experiment"]["name"],"keypoints.json")
    with open(path_for_json, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
