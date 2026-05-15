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

import optuna
from pathlib import Path

from src.utils.seed import set_global_seed
from src.utils.config import load_config, save_config
from src.utils.optuna_config import apply_optuna, apply_optuna_best

cfg = load_config("configs/config_list.yaml")
set_global_seed(cfg["experiment"]["seed"])

from src.models.factory import get_model
from src.datasets.transforms.factory import get_transform
from src.datasets.factory import get_dataset
from src.losses.factory import get_loss
from src.metrics.factory import get_metrics
from src.train.factory import get_trainer

def load_cfg():
    cfg = load_config("configs/config_list.yaml")
    print("Configuration loaded:")
    print(cfg)
    save_config(cfg, experiment_name=cfg["experiment"]["name"])
    return cfg

def train_model(cfg):
    # === Model creation ===
    model = get_model(cfg["model"]["name"], cfg=cfg, **cfg["model"]["params"])
    print(f"Model '{cfg['model']['name']}' initialized.")

    # === Dataset loading ===

    # Training dataset
    transform = get_transform(cfg["dataset"]["augmentation"], cfg["keypoint_format"], cfg["dataset"]["params"]["input_size"])

    train_dataset = get_dataset(
        name=cfg["dataset"]["name"],
        load=cfg["dataset"]["train"],
        num_samples=cfg["dataset"]["train_num_samples"],
        keypoint_format=cfg["keypoint_format"],
        transform=transform,
        **cfg["dataset"]["params"],
    )
    print(f"Training dataset loaded ({len(train_dataset)} samples).")

    # Validation dataset
    transform = get_transform(None, cfg["keypoint_format"], cfg["dataset"]["params"]["input_size"])

    val_dataset = get_dataset(
        name=cfg["dataset"]["name"],
        load=cfg["dataset"]["val"],
        num_samples=cfg["dataset"]["val_num_samples"],
        keypoint_format=cfg["keypoint_format"],
        transform=transform,
        **cfg["dataset"]["params"],
    )
    print(f"Validation dataset loaded ({len(val_dataset)} samples).")

    # === Loss function and metrics ===
    loss_fn = get_loss(cfg["loss"]["name"], cfg["model"]["backend"], **cfg["loss"]["params"])
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

    train_loss = float('inf')
    # === Start training ===
    if cfg["experiment"]["continue_from"] is None:
        print("Starting training from scratch...")
        train_loss = trainer.train()
    else:
        checkpoint = cfg["experiment"]["continue_from"]
        print(f"Resuming training from checkpoint: {checkpoint}")
        train_loss = trainer.continue_train(checkpoint)
    
    return train_loss

def objective(trial, cfg):
    cfg_optuna = apply_optuna(cfg, trial)

    print(cfg_optuna)

    score = train_model(cfg_optuna)

    return score

def main():
    """Main entry point for the training pipeline."""

    cfg = load_cfg()

    import uuid
    if cfg["experiment"]["optuna"]:
        print("Optimalization by optuna")
        study = optuna.create_study(
            direction="minimize",
            study_name=f"optuna_experiment_{uuid.uuid4()}",
            storage=None, #f"sqlite:///results/{cfg["experiment"]["name"]}/optuna_experiment.db",
            load_if_exists=False
        )
        study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg["experiment"]["optuna_trials"])

        path = Path("results", cfg["experiment"]["name"], "optuna_results.txt")
        with open(path, "w", encoding="utf-8") as f:
            print("Best score:", study.best_value, file=f)

            print("Best params:", file=f)

            for k, v in study.best_params.items():
                print(f"{k}: {v}", file=f)

        cfg_final = apply_optuna_best(cfg, study.best_params)
        save_config(cfg_final, experiment_name=cfg["experiment"]["name"], file_name="config_final.yaml")

    else:
        loss = train_model(cfg)


if __name__ == "__main__":
    main()