from src.utils.config import load_config
from src.models.factory import get_model
from src.datasets.factory import get_dataset
from src.losses.factory import get_loss
from src.metrics.factory import get_metrics
from src.train.factory import get_trainer

# cfg = load_config(["configs/base.yaml"])
# cfg = load_config(["configs/base.yaml", "configs/stacked_hourglass_pytorch.yaml"])
cfg = load_config("configs/config_list.yaml")
print(cfg)

model = get_model(cfg["model"]["name"], **cfg["model"]["params"])
train_dataset, val_dataset = get_dataset(cfg["dataset"]["name"], cfg["dataset"]["split"], **cfg["dataset"]["params"])
loss_fn = get_loss(cfg["loss"]["name"], cfg["model"]["backend"],)
metrics = get_metrics(cfg["metrics"]["names"])

trainer = get_trainer(
    backend=cfg["model"]["backend"],
    model=model, train_dataset=train_dataset,
    val_dataset=val_dataset, loss_fn=loss_fn,
    metrics=metrics, **cfg["train"]
)
trainer.train()




















# # --- Ověření načtení datasetu ---
# print(f"Dataset type: {type(dataset)}")
# if hasattr(dataset, '__len__'):
#     print(f"Dataset length: {len(dataset)}")

# try:
#     sample = dataset[0]
#     if isinstance(sample, (list, tuple)):
#         image, keypoints = sample
#         print("Sample image shape:", image.shape)
#         print("Sample keypoints:", keypoints.shape)
#     else:
#         print("Sample type:", type(sample))
# except Exception as e:
#     print("❌ Chyba při načítání vzorku z datasetu:", e)
#     raise
# # --- konec ověření ---