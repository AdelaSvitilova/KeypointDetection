"""PyTorch trainer module for training and evaluating keypoint detection models."""

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
    """Trainer class for PyTorch models, extending BaseTrainer.

    Handles the training loop, validation, checkpointing, and prediction.
    Supports TensorBoard logging, Optuna trial pruning, and sliding window inference.
    """
    
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
        """Initializes the PytorchTrainer.

        Args:
            model (torch.nn.Module): The neural network model to train.
            train_dataset (Dataset): Dataset used for training.
            val_dataset (Dataset): Dataset used for validation.
            loss_fn (callable): Loss function used for optimization.
            metrics (list, optional): List of metric objects to track.
            batch_size (int, optional): Batch size for data loaders. Defaults to 16.
            epochs (int, optional): Number of total epochs to train. Defaults to 10.
            lr (float, optional): Learning rate. Defaults to 0.01.
            save_every_epoch (int, optional): Interval for saving checkpoints. Defaults to 10.
            experiment_name (str, optional): Name of the experiment for saving paths. Defaults to "exp0".
            Format of the keypoints. Must be either 'keypoints' or 'heatmaps'. Defaults to "keypoints".
            special_mode (str, optional): Special handling mode (e.g., "cut_five_dim").
            trial (optuna.Trial, optional): Optuna trial for hyperparameter optimization.
            device (str, optional): Device string ('cuda' or 'cpu'). Auto-detected if None.
            use_tensorboard (bool, optional): Enable TensorBoard logging. Defaults to True.
            scheduler (str, optional): Type of learning rate scheduler.
            CosineAnnealingLR (dict, optional): Parameters for CosineAnnealingLR.
            CosineAnnealingWarmRestarts (dict, optional): Parameters for CosineAnnealingWarmRestarts.
        """
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

        # Initialize learning rate scheduler based on the provided configuration
        if scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=int(CosineAnnealingLR["T_max"]), 
                eta_min=float(CosineAnnealingLR["eta_min"])
            )
        elif scheduler == "CosineAnnealingWarmRestarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=int(CosineAnnealingWarmRestarts["T_0"]),
                T_mult=int(CosineAnnealingWarmRestarts["T_mult"]),
                eta_min=float(CosineAnnealingWarmRestarts["eta_min"])
            )

        self.checkpoint_dir = Path("results") / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.trial = trial
        # Isolate TensorBoard logs per Optuna trial if applicable
        if self.trial:
            self.log_dir = Path("results") / experiment_name / Path("tensorboard") / f"trial_{self.trial.number}"
        else:
            self.log_dir = Path("results") / experiment_name / Path("tensorboard")

        self.save_every_epoch = save_every_epoch
        self.use_tensorboard = use_tensorboard

    def train(self, start_epoch=0, best_val_loss=float('inf')):
        """Executes the training and validation loops.

        Args:
            start_epoch (int, optional): Epoch to resume training from. Defaults to 0.
            best_val_loss (float, optional): Best validation loss recorded so far. Defaults to infinity.

        Returns:
            float: The final training loss of the last epoch.

        Raises:
            optuna.TrialPruned: If the trial is pruned by Optuna during training due to poor performance.
        """
        if self.use_tensorboard:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(self.log_dir))

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        train_loss = float('inf')
        log_file = self.checkpoint_dir / "training.log"

        for epoch in range(start_epoch + 1, self.epochs + 1):
            start = time.time()
            self.model.train()
            
            for metric in self.metrics:
                metric.reset()
            
            running_loss = 0.0

            for item in train_loader:
                try:
                    # Convert inputs to torch.Tensor strictly at this point to avoid device mismatch
                    x = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                    y = torch.as_tensor(item["keypoints"], dtype=torch.float32, device=self.device)
                    y_h = y if item["heatmaps"] is None else torch.as_tensor(item["heatmaps"], dtype=torch.float32, device=self.device)

                    self.optimizer.zero_grad()
                    preds = self.model(x)
                    
                except torch.cuda.OutOfMemoryError:
                    # Prune the trial if it runs out of memory, otherwise reraise
                    if self.trial:
                        raise optuna.TrialPruned()
                    else:
                        raise 

                # Process predictions based on special network architectures
                if self.special_mode == "cut_five_dim" and preds.ndim == 5:
                    preds_tmp = preds[:, -1, :, :, :]
                else:
                    preds_tmp = preds
                    
                # Loss calculation and backpropagation
                loss = self.loss_fn(preds, targets=y_h, keypoint_targets=y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Calculate metrics on detached tensors to free up computational graph memory
                for metric in self.metrics:
                    metric.update(
                        preds_tmp.detach().cpu().numpy(), 
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

            # --- Validation Phase ---
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

                    if self.special_mode == "cut_five_dim" and preds_val.ndim == 5:
                        preds_val_tmp = preds_val[:, -1, :, :, :]
                    else:
                        preds_val_tmp = preds_val

                    loss_val = self.loss_fn(preds_val, targets=y_h_val, keypoint_targets=y_val)
                    val_loss += loss_val.item()

                    for metric in self.metrics:
                        metric.update(
                            preds_val_tmp.detach().cpu().numpy(), 
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

            # Step the learning rate scheduler
            self.scheduler.step()
            
            # Save periodic checkpoint
            if (epoch) % self.save_every_epoch == 0:
                self.save_checkpoint(
                    epoch,
                    best_val_loss, 
                    path = self.checkpoint_dir / f"epoch_{epoch}.pt",
                )
            
            # Save best checkpoint based on validation loss
            if val_loss < best_val_loss:
                self.save_checkpoint(
                    epoch,
                    best_val_loss, 
                    path = self.checkpoint_dir / f"best.pt",
                )
                best_val_loss = val_loss

            print(f"Epoch {epoch} | Train Loss: {train_loss:.3e} | Train Metrics: {train_metrics_text} | Train Time: {train_time} | Val Loss: {val_loss:.3e} | Val Metrics: {val_metrics_text} | Val Time: {val_time} | lr: {self.optimizer.param_groups[0]['lr']:.3e}")

            # Append epoch results to the log file
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
                
            # Log metrics to TensorBoard
            if self.use_tensorboard:
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Validation", val_loss, epoch)
                writer.add_scalar("Time/Train", train_seconds, epoch)
                writer.add_scalar("Time/Validation", val_seconds, epoch)
                writer.add_scalar("Learning_Rate", self.optimizer.param_groups[0]['lr'], epoch)
                
                for name, value in train_metrics.items():
                    writer.add_scalar(f"Metrics/Train/{name}", value, epoch)

                for name, value in val_metrics.items():
                    writer.add_scalar(f"Metrics/Validation/{name}", value, epoch)
                writer.flush()

            # Report to Optuna and check for pruning
            if self.trial:
                self.trial.report(train_loss, epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

        # Save final model
        self.save_checkpoint(
            self.epochs,
            best_val_loss, 
            path=self.checkpoint_dir / "final.pt",
        )
        print(f"Training completed. Final model saved to {self.checkpoint_dir / 'final.pt'}")

        if self.use_tensorboard:
            writer.close()

        return train_loss

    def save_checkpoint(self, epoch, best_val_loss, path):
        """Saves the model and optimizer states to a file.

        Args:
            epoch (int): Current epoch number.
            best_val_loss (float): Best validation loss recorded up to this point.
            path (Path): File path where the checkpoint will be saved.
        """
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
        """Loads model weights, optimizer, and scheduler states from a checkpoint.

        Args:
            checkpoint_name (str): The filename of the checkpoint located in `checkpoint_dir`.

        Returns:
            tuple: A tuple containing:
                - epoch (int): The epoch at which the checkpoint was saved.
                - start_epoch (int): Identical to epoch, representing where to start training.
                - best_val_loss (float): The validation loss at the checkpoint.

        Raises:
            ValueError: If `checkpoint_name` is not provided.
        """
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
        """Resumes training from a previously saved checkpoint.

        Args:
            checkpoint_name (str): Filename of the checkpoint to load.
        """
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
        """Generates predictions on a given dataset.

        Args:
            data_loader (Dataset or DataLoader): The data to predict on.
            method (str, optional): Inference method. Set to "sliding_window" to predict using the sliding window method. Defaults to None.
            checkpoint (str, optional): Checkpoint filename to load weights from. Defaults to "best.pt".
            batch_size (int, optional): Batch size if creating a new DataLoader. Defaults to 1.
            create_data_loader (bool, optional): Whether to wrap the input data_loader in a torch DataLoader.
            window_size (int, optional): Size of the sliding window crop. Defaults to 256.
            window_stride (int, optional): Step size for the sliding window. Defaults to 128.
            sigma_scale (float, optional): Scaling factor for the Gaussian smoothing window. Defaults to 0.25.

        Yields:
            tuple: A tuple containing the input item dictionary and the resulting predictions (numpy array).
        """ 
        self.load_model_from_checkpoint(checkpoint)
        
        if create_data_loader:
            data_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=False)

        if method == "sliding_window":
            yield from self._sliding_window_predict(data_loader, window_size, window_stride, sigma_scale)
        else:
            yield from self._predict(data_loader)

    def _predict(self, data_loader):
        """Standard batch-by-batch prediction generator.

        Args:
            data_loader (DataLoader): The DataLoader providing batches.

        Yields:
            tuple: (input item dict, model predictions as numpy array).
        """
        print("predict")
        self.model.eval()
        with torch.no_grad():
            for item in data_loader:
                images = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                preds = self.model(images)
                
                if self.special_mode == "cut_five_dim" and preds.ndim == 5:
                    preds = preds[:, -1, :, :, :]
                yield item, preds.cpu().numpy()

    def _create_gaussian_window(self, window_size=256, sigma_scale=0.25):
        """Creates a 2D Gaussian window used to weight sliding window predictions.
        
        This prevents harsh edge artifacts by prioritizing the center of each cropped window.

        Args:
            window_size (int or tuple): The dimension of the window.
            sigma_scale (float): Determines the spread of the Gaussian curve.

        Returns:
            torch.Tensor: Normalized 2D Gaussian tensor.
        """
        # If window_size happens to be a list/tuple, extract the first element
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
        min_peak_threshold=0.05,
    ):
        """Generates predictions using a sliding window strategy.

        Args:
            data_loader (DataLoader): The DataLoader providing the images.
            window_size (int or tuple): Size of the cropped patches.
            window_stride (int or tuple): Stride between subsequent crops.
            sigma_scale (float): Controls the Gaussian weighting used for patch overlap.
            min_peak_threshold (float): Thresholding for empty windows (0 = disabled).

        Yields:
            tuple: (input item dict, final stitched heatmap predictions as a numpy array).
        """
        print("sliding window predict")
        if isinstance(window_size, (list, tuple)):
            window_size = window_size[0]

        if isinstance(window_stride, (list, tuple)):
            window_stride = window_stride[0]

        self.model.eval()
        
        with torch.no_grad():
            for item in data_loader:
                images = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                B, C, H, W = images.shape
                
                # Determine necessary padding so the sliding window covers the entire image
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
                
                # Apply padding to the original image tensor
                padded_images = F.pad(images, (0, pad_W, 0, pad_H), mode='constant', value=0)
                H_pad, W_pad = padded_images.shape[2], padded_images.shape[3]
                
                # Extract patches across the padded image
                patches = F.unfold(padded_images, kernel_size=window_size, stride=window_stride)
                _, _, L = patches.shape
                patches_for_model = patches.transpose(1, 2).reshape(B * L, C, window_size, window_size)
                
                # Run the model on all extracted patches
                preds = self.model(patches_for_model)
                
                if self.special_mode == "cut_five_dim" and preds.ndim == 5:
                    preds = preds[:, -1, :, :, :]
                
                num_keypoints = preds.shape[1]
                out_window_size = preds.shape[2]
                
                # Setup scaling factors in case the model's output size differs from the input size
                scale_factor = float(out_window_size) / float(window_size)
                out_stride = int(window_stride * scale_factor)
                out_H_pad = int(H_pad * scale_factor)
                out_W_pad = int(W_pad * scale_factor)
                out_H = int(H * scale_factor)
                out_W = int(W * scale_factor)
                
                # Generate Gaussian mask to smooth the overlaps during stitching
                gaussian_window = self._create_gaussian_window(
                    window_size=out_window_size, 
                    sigma_scale=sigma_scale
                ).to(self.device)
                
                gaussian_flat = gaussian_window.view(-1, 1)
                preds_reshaped = preds.view(B, L, num_keypoints, out_window_size * out_window_size)
                gauss_4d = gaussian_flat.view(1, 1, 1, -1)
                
                # Apply the Gaussian weights to the predicted patches
                preds_weighted = preds_reshaped * gauss_4d
                preds_flat = preds_weighted.view(B, L, num_keypoints * out_window_size * out_window_size).transpose(1, 2)
                
                # Fold the patches back together to reconstruct the full heatmap
                reconstructed_sum = F.fold(
                    preds_flat, 
                    output_size=(out_H_pad, out_W_pad), 
                    kernel_size=out_window_size, 
                    stride=out_stride
                )
                
                # Keep track of overlapping weights to properly normalize the stitched regions
                gauss_map_src = gaussian_flat.view(1, -1, 1).expand(B, -1, L)
                overlap_weight_map = F.fold(
                    gauss_map_src, 
                    output_size=(out_H_pad, out_W_pad), 
                    kernel_size=out_window_size, 
                    stride=out_stride
                )
                
                # Normalize by accumulated weights, using ones_like as fallback to avoid division by zero
                safe_weight_map = torch.where(overlap_weight_map > 1e-6, overlap_weight_map, torch.ones_like(overlap_weight_map))
                final_heatmaps = reconstructed_sum / safe_weight_map
                
                # Crop back to the original (scaled) image dimensions to remove padding
                final_heatmaps = final_heatmaps[:, :, :out_H, :out_W]
                                        
                yield item, final_heatmaps.cpu().numpy()