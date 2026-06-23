import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from .utils import collate_fn
from pathlib import Path
import time
import math
import optuna

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
        trial=None,
        device=None,
        use_tensorboard=True,
        scheduler=None,
        CosineAnnealingLR=None,
        CosineAnnealingWarmRestarts=None,
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

        if scheduler=="CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=int(CosineAnnealingLR["T_max"]), eta_min=float(CosineAnnealingLR["eta_min"]))
        elif scheduler=="CosineAnnealingWarmRestarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=int(CosineAnnealingWarmRestarts["T_0"]),
                                                                                    T_mult=int(CosineAnnealingWarmRestarts["T_mult"]),
                                                                                    eta_min=float(CosineAnnealingWarmRestarts["eta_min"]))

        self.checkpoint_dir = Path("results") / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.trial = trial
        if self.trial:
            self.log_dir = Path("results") / experiment_name / Path("tensorboard") / f"trial_{self.trial.number}"
        else:
            self.log_dir = Path("results") / experiment_name / Path("tensorboard")

        self.save_every_epoch = save_every_epoch
        self.use_tensorboard = use_tensorboard

    def train(self, start_epoch=0, best_val_loss=float('inf')):
        if self.use_tensorboard:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(self.log_dir))

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        best_val_loss = best_val_loss

        train_loss = float('inf')

        log_file = self.checkpoint_dir / "training.log"

        for epoch in range(start_epoch + 1, self.epochs + 1):
            start = time.time()
            self.model.train()
            for metric in self.metrics:
                metric.reset()
            
            running_loss = 0.0

            for item in train_loader:
                # # převod na torch.Tensor až tady
                # x = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                # y = torch.as_tensor(item["keypoints"], dtype=torch.float32, device=self.device)
                # y_h = y if item["heatmaps"] is None else torch.as_tensor(item["heatmaps"], dtype=torch.float32, device=self.device)

                # self.optimizer.zero_grad()
                try:
                    # převod na torch.Tensor až tady
                    x = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                    y = torch.as_tensor(item["keypoints"], dtype=torch.float32, device=self.device)
                    y_h = y if item["heatmaps"] is None else torch.as_tensor(item["heatmaps"], dtype=torch.float32, device=self.device)

                    self.optimizer.zero_grad()

                    preds = self.model(x)
                except torch.cuda.OutOfMemoryError:
                    if self.trial:
                        raise optuna.TrialPruned()
                    else:
                        raise 

                # loss = torch-based → backprop funguje
                if self.special_mode=="cut_five_dim" and preds.ndim == 5:
                    preds_tmp = preds[:, -1, :, :, :]
                else:
                    preds_tmp = preds
                    
                loss = self.loss_fn(preds, targets=y_h, keypoint_targets=y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # metriku počítáme na detachnutých tensorech
                for metric in self.metrics:
                    metric.update(preds_tmp.detach().cpu().numpy(), 
                        y.detach().cpu().numpy(), 
                        norm_coefficient=item["norm_coefficient"].detach().cpu().numpy(),
                        orig_height=item["orig_height"].detach().cpu().numpy(),
                        orig_width=item["orig_width"].detach().cpu().numpy(),
                    )

            end = time.time()
            train_seconds = end - start

            hours, remainder = divmod(end-start, 3600)
            minutes, seconds = divmod(remainder, 60)

            train_time = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

            train_loss = running_loss / len(self.train_dataset)
            train_metrics_text = {type(m).__name__: f"{m.compute():.4f}" for m in self.metrics}
            train_metrics = {type(m).__name__: m.compute() for m in self.metrics}

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

                    loss_val = self.loss_fn(preds_val, targets=y_h_val, keypoint_targets=y_val)

                    val_loss += loss_val.item()

                    for metric in self.metrics:
                        metric.update(preds_val_tmp.detach().cpu().numpy(), 
                            y_val.detach().cpu().numpy(), 
                            norm_coefficient=item["norm_coefficient"].detach().cpu().numpy(),
                            orig_height=item["orig_height"].detach().cpu().numpy(),
                            orig_width=item["orig_width"].detach().cpu().numpy(),
                        )

            end = time.time()
            val_seconds = end - start

            hours, remainder = divmod(end-start, 3600)
            minutes, seconds = divmod(remainder, 60)

            val_time = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

            val_loss /= len(self.val_dataset)
            val_metrics_text = {type(m).__name__: f"{m.compute():.4f}" for m in self.metrics}
            val_metrics = {type(m).__name__: m.compute() for m in self.metrics}

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

            print(f"Epoch {epoch} | Train Loss: {train_loss:.3e} | Train Metrics: {train_metrics_text} | Train Time: {train_time} | Val Loss: {val_loss:.3e} | Val Metrics: {val_metrics_text} | Val Time: {val_time} | lr: {self.optimizer.param_groups[0]['lr']:.3e}")

            with open(log_file, "a") as f:
                f.write(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.3e} | "
                    f"train_metrics={train_metrics_text} | "
                    f"train_time={train_time} | "
                    f"val_loss={val_loss:.3e} | "
                    f"val_time={val_time} | "
                    f"val_metrics={val_metrics_text} | "
                    f"lr={self.optimizer.param_groups[0]['lr']:.3e}\n"
                )
                
            if self.use_tensorboard:
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Validation", val_loss, epoch)
                writer.add_scalar("Time/Train", train_seconds, epoch)
                writer.add_scalar("Time/Validation", val_seconds, epoch)
                writer.add_scalar("Learning_Rate", self.optimizer.param_groups[0]['lr'], epoch)
                # logování metrik do TensorBoardu
                for name, value in train_metrics.items():
                    writer.add_scalar(f"Metrics/Train/{name}", value, epoch)

                for name, value in val_metrics.items():
                    writer.add_scalar(f"Metrics/Validation/{name}", value, epoch)
                writer.flush()

            if self.trial:
                self.trial.report(train_loss, epoch)

                if self.trial.should_prune():
                    raise optuna.TrialPruned()

        if self.use_tensorboard:
            writer.close()

        return train_loss

    def save_checkpoint(self, epoch, best_val_loss, path):
        checkpoint = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_model_from_checkpoint(self, checkpoint_name):
        if checkpoint_name is None:
            raise ValueError("Checkpoint path was not specified.")

        path = self.checkpoint_dir / checkpoint_name

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if hasattr(self, "scheduler") and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint.get('epoch', 0)
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]

        return epoch, start_epoch, best_val_loss


    def continue_train(self, checkpoint_name):
        epoch, start_epoch, best_val_loss = self.load_model_from_checkpoint(checkpoint_name)

        print(f"Loaded checkpoint from epoch {epoch}. Resuming training at epoch {start_epoch + 1}.")

        self.train(start_epoch=start_epoch, best_val_loss=best_val_loss)


    def predict(
        self, 
        data_loader, 
        method=None,
        checkpoint="best.pt", 
        batch_size=1, 
        create_data_loader=False, 
        window_size=256, 
        window_stride=128, 
        sigma_scale=0.25,
        **kwargs
    ): 
        self.load_model_from_checkpoint(checkpoint)
        
        if create_data_loader:
            data_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=False)

        if method == "sliding_window":
            yield from self._sliding_window_predict(data_loader, window_size, window_stride, sigma_scale)
        else:
            yield from self._predict(data_loader)


    def _predict(self, data_loader):
        print("predict")

        self.model.eval()
        with torch.no_grad():
            for item in data_loader:
                images = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                images = images.to(self.device)
                preds = self.model(images)
                if self.special_mode == "cut_five_dim" and preds.ndim == 5:
                    preds = preds[:, -1, :, :, :]
                yield item, preds.cpu().numpy()

    def _create_gaussian_window(self, window_size=256, sigma_scale=0.25):
        # Pokud by window_size byl náhodou list/tuple, vezmeme první prvek
        if isinstance(window_size, (list, tuple)):
            window_size = window_size[0]
            
        sigma = window_size * sigma_scale
        coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        gaussian_2d = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
        
        return gaussian_2d / gaussian_2d.max()

    def _sliding_window_predict(
        self, 
        data_loader, 
        window_size=256, 
        window_stride=128, 
        sigma_scale=0.25, 
        min_peak_threshold=0.05,  # Prahování pro prázdná okna (0 = vypnuto)
    ):
        if isinstance(window_size, (list, tuple)):
            window_size = window_size[0]

        if isinstance(window_stride, (list, tuple)):
            window_stride = window_stride[0]

        self.model.eval()
        
        with torch.no_grad():
            for item in data_loader:
                images = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                B, C, H, W = images.shape
                
                if H < window_size:
                    pad_H = window_size - H
                    num_patches_H = 1
                else:
                    num_patches_H = math.ceil((H - window_size) / window_stride) + 1
                    pad_H = (num_patches_H - 1) * window_stride + window_size - H
                    
                if W < window_size:
                    pad_W = window_size - W
                    num_patches_W = 1
                else:
                    num_patches_W = math.ceil((W - window_size) / window_stride) + 1
                    pad_W = (num_patches_W - 1) * window_stride + window_size - W
                
                padded_images = F.pad(images, (0, pad_W, 0, pad_H), mode='constant', value=0)
                H_pad, W_pad = padded_images.shape[2], padded_images.shape[3]
                
                patches = F.unfold(padded_images, kernel_size=window_size, stride=window_stride)
                _, _, L = patches.shape
                patches_for_model = patches.transpose(1, 2).reshape(B * L, C, window_size, window_size)
                
                preds = self.model(patches_for_model)
                
                if self.special_mode == "cut_five_dim" and preds.ndim == 5:
                    preds = preds[:, -1, :, :, :]
                
                num_keypoints = preds.shape[1]
                out_window_size = preds.shape[2]
                
                scale_factor = float(out_window_size) / float(window_size)
                out_stride = int(window_stride * scale_factor)
                out_H_pad = int(H_pad * scale_factor)
                out_W_pad = int(W_pad * scale_factor)
                out_H = int(H * scale_factor)
                out_W = int(W * scale_factor)
                
                gaussian_window = self._create_gaussian_window(
                    window_size=out_window_size, 
                    sigma_scale=sigma_scale
                ).to(self.device)
                
                gaussian_flat = gaussian_window.view(-1, 1)
                
                preds_reshaped = preds.view(B, L, num_keypoints, out_window_size * out_window_size)
                
                gauss_4d = gaussian_flat.view(1, 1, 1, -1)
                
                preds_weighted = preds_reshaped * gauss_4d
                
                preds_flat = preds_weighted.view(B, L, num_keypoints * out_window_size * out_window_size).transpose(1, 2)
                
                reconstructed_sum = F.fold(
                    preds_flat, 
                    output_size=(out_H_pad, out_W_pad), 
                    kernel_size=out_window_size, 
                    stride=out_stride
                )
                
                gauss_map_src = gaussian_flat.view(1, -1, 1).expand(B, -1, L)
                
                overlap_weight_map = F.fold(
                    gauss_map_src, 
                    output_size=(out_H_pad, out_W_pad), 
                    kernel_size=out_window_size, 
                    stride=out_stride
                )
                
                safe_weight_map = torch.where(overlap_weight_map > 1e-6, overlap_weight_map, torch.ones_like(overlap_weight_map))
                final_heatmaps = reconstructed_sum / safe_weight_map
                
                final_heatmaps = final_heatmaps[:, :, :out_H, :out_W]
                                        
                yield item, final_heatmaps.cpu().numpy()