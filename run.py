"""
Main script to run a full training pipeline.

This script performs the following steps:
1. Loads the configuration from a YAML file.
2. Creates the model using the factory function based on the configuration.
3. Loads the training and validation datasets.
4. Creates the loss function and metrics.
5. Instantiates the trainer for the specified backend.
6. Starts the training process.

The script is fully framework-agnostic and uses factory functions for models,
datasets, losses, metrics, and trainers to allow easy switching between
different backends (e.g., PyTorch, Keras).
"""

from src.utils.config import load_config
from src.models.factory import get_model
from src.datasets.factory import get_dataset
from src.losses.factory import get_loss
from src.metrics.factory import get_metrics
from src.train.factory import get_trainer

def main():
    cfg = load_config("configs/config_list.yaml")
    print(cfg)

    # Create model
    model = get_model(cfg["model"]["name"], **cfg["model"]["params"])

    # Load datasets
    train_dataset, val_dataset = get_dataset(
        cfg["dataset"]["name"], cfg["dataset"]["split"], **cfg["dataset"]["params"]
    )

    # Create loss function and metrics
    loss_fn = get_loss(cfg["loss"]["name"], cfg["model"]["framework"])
    metrics = get_metrics(cfg["metrics"]["names"])

    # Create trainer and start training
    trainer = get_trainer(
        backend=cfg["model"]["framework"],
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=loss_fn,
        metrics=metrics,
        keypoint_format=cfg["keypoint_format"],
        special_mode=cfg["model"]["special_mode"],
        **cfg["train"]
    )
    trainer.train()

if __name__ == "__main__":
    main()