import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from .utils import collate_fn
from pathlib import Path
import time

class PytorchTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        loss_fn,
        metrics=None,
        batch_size=16,
        epochs=10,
        lr=0.01,
        save_every_epoch=10,
        experiment_name="exp0",
        keypoint_format="keypoints",
        special_mode=None,
        device=None
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            loss_fn=loss_fn,
            metrics=metrics,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            framework="pytorch",
            keypoint_format=keypoint_format
        )
        
        self.special_mode = special_mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)

        self.checkpoint_dir = Path("results") / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_every_epoch = save_every_epoch

    def train(self, start_epoch=0, best_val_loss=float('inf')):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        best_val_loss = best_val_loss

        log_file = self.checkpoint_dir / "training.log"

        for epoch in range(start_epoch + 1, self.epochs + 1):
            start = time.time()
            self.model.train()
            for metric in self.metrics:
                metric.reset()
            
            running_loss = 0.0

            for item in train_loader:
                # převod na torch.Tensor až tady
                x = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                y = torch.as_tensor(item["keypoints"], dtype=torch.float32, device=self.device)
                y_h = y if item["heatmaps"] is None else torch.as_tensor(item["heatmaps"], dtype=torch.float32, device=self.device)

                self.optimizer.zero_grad()
                preds = self.model(x)

                # loss = torch-based → backprop funguje
                if self.special_mode=="cut_five_dim" and preds.ndim == 5:
                    preds_tmp = preds[:, -1]
                else:
                    preds_tmp = preds
                    
                    
                loss = self.loss_fn(preds, y_h)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # metriku počítáme na detachnutých tensorech
                for metric in self.metrics:
                    metric.update(preds_tmp.detach().cpu().numpy(), y.detach().cpu().numpy())

            end = time.time()

            hours, remainder = divmod(end-start, 3600)
            minutes, seconds = divmod(remainder, 60)

            train_time = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

            train_loss = running_loss / len(self.train_dataset)
            train_metrics = {type(m).__name__: f"{m.compute():.4f}" for m in self.metrics}

            self.model.eval()
            for metric in self.metrics:
                metric.reset()

            val_loss = 0.0
            start = time.time()
            with torch.no_grad():
                for item in val_loader:
                    x_val = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                    y_val = torch.as_tensor(item["keypoints"], dtype=torch.float32, device=self.device)
                    y_h_val = y_val if item["heatmaps"] is None else torch.as_tensor(item["heatmaps"], dtype=torch.float32, device=self.device)
                    
                    preds_val = self.model(x_val)

                    if self.special_mode=="cut_five_dim" and preds_val.ndim == 5:
                        preds_val_tmp = preds_val[:, -1, :, :, :]
                    else:
                        preds_val_tmp = preds_val

                    
                    loss_val = self.loss_fn(preds_val, y_h_val)
                    val_loss += loss_val.item()

                    for metric in self.metrics:
                        metric.update(preds_val_tmp.detach().cpu().numpy(), y_val.detach().cpu().numpy())

            end = time.time()

            hours, remainder = divmod(end-start, 3600)
            minutes, seconds = divmod(remainder, 60)

            val_time = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

            val_loss /= len(self.val_dataset)
            val_metrics = {type(m).__name__: f"{m.compute():.4f}" for m in self.metrics}

            self.scheduler.step()
            
            if (epoch) % self.save_every_epoch == 0:
                self.save_checkpoint(
                    epoch,
                    best_val_loss, 
                    path = self.checkpoint_dir / f"epoch_{epoch}.pt",
                    )
            
            if val_loss < best_val_loss:
                self.save_checkpoint(
                    epoch,
                    best_val_loss, 
                    path = self.checkpoint_dir / f"best.pt",
                    )
                best_val_loss = val_loss

            print(f"Epoch {epoch} | Train Loss: {train_loss:.3e} | Train Metrics: {train_metrics} | Train Time: {train_time} | Val Loss: {val_loss:.3e} | Val Metrics: {val_metrics} | Val Time: {val_time} | lr: {self.optimizer.param_groups[0]['lr']:.3e}")

            with open(log_file, "a") as f:
                f.write(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.3e} | "
                    f"train_metrics={train_metrics} | "
                    f"train_time={train_time} | "
                    f"val_loss={val_loss:.3e} | "
                    f"val_time={val_time} | "
                    f"val_metrics={val_metrics} | "
                    f"lr={self.optimizer.param_groups[0]['lr']:.3e}\n"
                )

    def save_checkpoint(self, epoch, best_val_loss, path, scheduler=None):
        checkpoint = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_model_from_checkpoint(self, checkpoint_name):
        if checkpoint_name is None:
            raise ValueError("Checkpoint path was not specified.")

        path = self.checkpoint_dir / checkpoint_name

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if hasattr(self, "scheduler") and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint.get('epoch', 0)
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]

        return epoch, start_epoch, best_val_loss


    def continue_train(self, checkpoint_name):
        epoch, start_epoch, best_val_loss = self.load_model_from_checkpoint(checkpoint_name)

        print(f"Checkpoint načten: epoch {epoch}. Pokračuji od epochy {start_epoch + 1}.")

        self.train(start_epoch=start_epoch, best_val_loss=best_val_loss)

    def predict_image(self, data_loader, batch_size=1): 
        self.load_model_from_checkpoint("best.pt")
        # data_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        with torch.no_grad():
            for item in data_loader:
                images = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                images = images.to(self.device)
                preds = self.model(images)
                if self.special_mode == "cut_five_dim" and preds.ndim == 5:
                    # (B, 2, K, H, W) → (B, K, H, W)
                    preds = preds[:, -1, :, :, :]
                yield item, preds.cpu().numpy()

    def predict(self, data_loader, batch_size=1): 
        self.load_model_from_checkpoint("best.pt")
        data_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        with torch.no_grad():
            for item in data_loader:
                images = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                images = images.to(self.device)
                preds = self.model(images)
                if self.special_mode == "cut_five_dim" and preds.ndim == 5:
                    # (B, 2, K, H, W) → (B, K, H, W)
                    preds = preds[:, -1, :, :, :]
                yield item, preds.cpu().numpy()

