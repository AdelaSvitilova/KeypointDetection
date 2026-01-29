from src.utils.config import load_config
from src.models.factory import get_model
from src.datasets.factory import get_dataset
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
    dataset = get_dataset(
        name="images",
        load=cfg["predict"]["images_list"],
        num_samples=cfg["predict"]["num_samples"],
        keypoint_format=cfg["keypoint_format"],
        **cfg["predict"]["params"]
    )
    print(f"Training dataset loaded: {len(dataset)} samples")

    # === Trainer creation ===
    trainer = get_trainer(
        backend=cfg["model"]["framework"],
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
    print(f"Trainer initialized for backend '{cfg['model']['framework']}'")

    for preds in trainer.predict(dataset, cfg["predict"]["batch_size"]):
        print(preds)


if __name__ == "__main__":
    main()
