"""
Main script to run a full training pipeline for keypoint detection.

This script performs the following steps:
1. Loads the configuration from a YAML file.
2. Creates the model using the factory function based on the configuration.
3. Loads the training and validation datasets.
4. Creates the loss function and evaluation metrics.
5. Instantiates the trainer for the specified backend.
6. Starts the training process, optionally continuing from a checkpoint.

The script is fully framework-agnostic and uses factory functions for models,
datasets, losses, metrics, and trainers to allow easy switching between
different backends (e.g., PyTorch, Keras).

Additional features:
- Supports custom dataset keypoint formats (raw or heatmaps).
- Supports experiment continuation from checkpoints.
- Prints configuration for quick verification.
"""

from src.utils.config import load_config
from src.models.factory import get_model
from src.datasets.factory import get_dataset
from src.losses.factory import get_loss
from src.metrics.factory import get_metrics
from src.train.factory import get_trainer


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
    train_dataset = get_dataset(
        name=cfg["dataset"]["name"],
        load=cfg["dataset"]["train"],
        num_samples=cfg["dataset"]["train_num_samples"],
        keypoint_format=cfg["keypoint_format"],
        **cfg["dataset"]["params"]
    )
    print(f"Training dataset loaded: {len(train_dataset)} samples")

    # Validation dataset
    val_dataset = get_dataset(
        name=cfg["dataset"]["name"],
        load=cfg["dataset"]["val"],
        num_samples=cfg["dataset"]["val_num_samples"],
        keypoint_format=cfg["keypoint_format"],
        **cfg["dataset"]["params"]
    )
    print(f"Validation dataset loaded: {len(val_dataset)} samples")

    # === Loss and metrics ===
    loss_fn = get_loss(cfg["loss"]["name"], cfg["model"]["backend"])
    metrics = get_metrics(cfg["metrics"]["names"])
    print(f"Loss function: {cfg['loss']['name']}, Metrics: {cfg['metrics']['names']}")

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
        **cfg["train"]
    )
    print(f"Trainer initialized for backend '{cfg['model']['backend']}'")

    # === Start training ===
    if cfg["experiment"]["continue_from"] is None:
        print("Starting training from scratch...")
        trainer.train()
    else:
        print(f"Continuing training from checkpoint: {cfg['experiment']['continue_from']}")
        trainer.continue_train(cfg["experiment"]["continue_from"])


if __name__ == "__main__":
    main()
