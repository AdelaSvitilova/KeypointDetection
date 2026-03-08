"""
Run the full training pipeline for keypoint detection.

This script performs the following steps:
1. Load the configuration from a YAML file.
2. Create the model using a factory function.
3. Load the training and validation datasets.
4. Create the loss function and evaluation metrics.
5. Instantiate the trainer for the selected backend.
6. Start the training process (optionally from a checkpoint).

The pipeline is framework-agnostic and relies on factory functions for
models, datasets, losses, metrics, and trainers. This design allows
easy switching between different backends (e.g., PyTorch, Keras).

Features:
    - Supports multiple keypoint formats (raw coordinates or heatmaps).
    - Supports continuing experiments from checkpoints.
    - Prints the configuration for quick verification.
"""

from src.utils.config import load_config
from src.models.factory import get_model
from src.datasets.factory import get_dataset
from src.losses.factory import get_loss
from src.metrics.factory import get_metrics
from src.train.factory import get_trainer


def main():
    """Main entry point for the training pipeline."""

    # === Load configuration ===
    cfg = load_config("configs/config_list.yaml")
    print("Configuration loaded:")
    print(cfg)

    # === Model creation ===
    model = get_model(cfg["model"]["name"], **cfg["model"]["params"])
    print(f"Model '{cfg['model']['name']}' initialized.")

    # === Dataset loading ===

    # Training dataset
    train_dataset = get_dataset(
        name=cfg["dataset"]["name"],
        load=cfg["dataset"]["train"],
        num_samples=cfg["dataset"]["train_num_samples"],
        keypoint_format=cfg["keypoint_format"],
        **cfg["dataset"]["params"],
    )
    print(f"Training dataset loaded ({len(train_dataset)} samples).")

    # Validation dataset
    val_dataset = get_dataset(
        name=cfg["dataset"]["name"],
        load=cfg["dataset"]["val"],
        num_samples=cfg["dataset"]["val_num_samples"],
        keypoint_format=cfg["keypoint_format"],
        **cfg["dataset"]["params"],
    )
    print(f"Validation dataset loaded ({len(val_dataset)} samples).")

    # === Loss function and metrics ===
    loss_fn = get_loss(cfg["loss"]["name"], cfg["model"]["backend"])
    metrics = get_metrics(cfg["metrics"]["names"])

    print(f"Loss function: {cfg['loss']['name']}")
    print(f"Metrics: {cfg['metrics']['names']}")

    # === Trainer creation ===
    trainer = get_trainer(
        backend=cfg["model"]["backend"],
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=loss_fn,
        metrics=metrics,
        experiment_name=cfg["experiment"]["name"],
        keypoint_format=cfg["keypoint_format"],
        special_mode=cfg["model"]["special_mode"],
        **cfg["train"],
    )

    print(f"Trainer initialized for backend '{cfg['model']['backend']}'.")

    # === Start training ===
    if cfg["experiment"]["continue_from"] is None:
        print("Starting training from scratch...")
        trainer.train()
    else:
        checkpoint = cfg["experiment"]["continue_from"]
        print(f"Resuming training from checkpoint: {checkpoint}")
        trainer.continue_train(checkpoint)


if __name__ == "__main__":
    main()