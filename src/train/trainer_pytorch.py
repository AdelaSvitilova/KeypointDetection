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
    Supports TensorBoard logging, Optuna trial pruning, and sliding window inference
    specifically designed for high-resolution medical imaging (e.g., X-rays).
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
            train_dataset (torch.utils.data.Dataset): Dataset used for training.
            val_dataset (torch.utils.data.Dataset): Dataset used for validation.
            loss_fn (callable): Loss function used for optimization.
            metrics (list, optional): List of metric objects to track. Defaults to None.
            batch_size (int, optional): Batch size for data loaders. Defaults to 16.
            epochs (int, optional): Number of total epochs to train. Defaults to 10.
            lr (float, optional): Initial learning rate. Defaults to 0.01.
            save_every_epoch (int, optional): Interval for saving checkpoints. Defaults to 10.
            experiment_name (str, optional): Name of the experiment for saving paths. Defaults to "exp0".
            keypoint_format (str, optional): Format of the targets. Must be either 'keypoints' or 'heatmaps'. Defaults to "keypoints".
            special_mode (str, optional): Special handling mode (e.g., "extract_last_stack" for Stacked Hourglass). Defaults to None.
            trial (optuna.Trial, optional): Optuna trial for hyperparameter optimization pruning. Defaults to None.
            device (str, optional): Device string ('cuda' or 'cpu'). Auto-detected if None.
            use_tensorboard (bool, optional): Enable TensorBoard logging. Defaults to True.
            scheduler (str, optional): Type of learning rate scheduler. Defaults to None.
            CosineAnnealingLR (dict, optional): Parameters for CosineAnnealingLR scheduler. Defaults to None.
            CosineAnnealingWarmRestarts (dict, optional): Parameters for CosineAnnealingWarmRestarts. Defaults to None.
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
        self.scheduler = None        
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
        
        # Isolate TensorBoard logs per Optuna trial if applicable to prevent overlapping plots
        if self.trial:
            self.log_dir = self.checkpoint_dir / "tensorboard" / f"trial_{self.trial.number}"
        else:
            self.log_dir = self.checkpoint_dir / "tensorboard"

        self.save_every_epoch = save_every_epoch
        self.use_tensorboard = use_tensorboard

    # ------------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------------

    def _extract_final_predictions(self, preds):
        """Extracts the final prediction stack for metrics calculation and inference.
        
        Certain architectures (like Stacked Hourglass) return intermediate predictions 
        from all stacks during the forward pass for deep supervision. This method isolates 
        only the final stack output for actual evaluation.

        Args:
            preds (torch.Tensor): Raw predictions directly from the model output.

        Returns:
            torch.Tensor: The final prediction stack if special_mode requires it, 
                          otherwise returns the unmodified predictions.
        """
        if self.special_mode == "extract_last_stack" and preds.ndim in (4, 5):
            return preds[:, -1, ...]
        return preds

    def _compute_loss(self, preds, y_keypoints, y_heatmaps=None):
        """Computes the loss dynamically based on the expected keypoint format.

        Args:
            preds (torch.Tensor): Model predictions.
            y_keypoints (torch.Tensor): Ground truth (x, y) keypoint coordinates.
            y_heatmaps (torch.Tensor, optional): Ground truth heatmaps. Defaults to None.

        Returns:
            torch.Tensor: A scalar tensor representing the computed loss.
        """
        if self.keypoint_format == "keypoints":
            return self.loss_fn(preds, targets=y_keypoints, keypoint_targets=y_keypoints)
        else:
            # Fallback to keypoints if heatmaps are not provided but format demands them
            targets = y_keypoints if y_heatmaps is None else y_heatmaps
            return self.loss_fn(preds, targets=targets, keypoint_targets=y_keypoints)

    def _format_time(self, seconds):
        """Formats time in seconds to a human-readable HHh MMm SSs format.

        Args:
            seconds (float): Elapsed time in seconds.

        Returns:
            str: Formatted time string (e.g., '1h 23m 45.67s').
        """
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"

    # ------------------------------------------------------------------------
    # Core Training Loop
    # ------------------------------------------------------------------------

    def train(self, start_epoch=0, best_val_loss=float('inf')):
        """Executes the complete training and validation loops across epochs.

        Iterates over the dataset, computes loss, updates model weights via backpropagation, 
        evaluates on the validation set, steps the learning rate scheduler, and saves checkpoints.

        Args:
            start_epoch (int, optional): The epoch number to start/resume training from. Defaults to 0.
            best_val_loss (float, optional): The best validation loss recorded so far. Defaults to infinity.

        Returns:
            float: The final validation loss achieved at the end of training.

        Raises:
            optuna.TrialPruned: If the trial is pruned by Optuna due to poor performance 
                                or out-of-memory errors during hyperparameter tuning.
        """
        if self.use_tensorboard:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(self.log_dir))

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        log_file = self.checkpoint_dir / "training.log"

        for epoch in range(start_epoch + 1, self.epochs + 1):
            # --- Training Phase ---
            start = time.time()
            self.model.train()
            
            for metric in self.metrics:
                metric.reset()
            
            running_loss = 0.0

            for item in train_loader:
                try:
                    # Convert inputs strictly at this point to avoid device mismatch
                    x = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                    y = torch.as_tensor(item["keypoints"], dtype=torch.float32, device=self.device)
                    y_h = None if item["heatmaps"] is None else torch.as_tensor(item["heatmaps"], dtype=torch.float32, device=self.device)

                    self.optimizer.zero_grad()
                    preds = self.model(x)
                    
                except torch.cuda.OutOfMemoryError:
                    # Prune the trial automatically if it runs out of memory during Optuna search
                    if self.trial:
                        raise optuna.TrialPruned()
                    raise 

                # Compute loss on raw preds (allowing intermediate loss supervision for architectures like CPN or Stacked Hourglass)
                loss = self._compute_loss(preds, y, y_h)
                
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * x.size(0)

                # Extract final stack strictly for metric calculation on CPU to save VRAM
                preds_final = self._extract_final_predictions(preds).detach().cpu().numpy()
                y_cpu = y.detach().cpu().numpy()
                norm_coeff = item["norm_coefficient"].detach().cpu().numpy()

                for metric in self.metrics:
                    metric.update(preds_final, y_cpu, norm_coefficient=norm_coeff)

            train_seconds = time.time() - start
            train_time_str = self._format_time(train_seconds)

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
                    y_h_val = None if item["heatmaps"] is None else torch.as_tensor(item["heatmaps"], dtype=torch.float32, device=self.device)
                    
                    preds_val = self.model(x_val)

                    loss_val = self._compute_loss(preds_val, y_val, y_h_val)
                    val_loss += loss_val.item() * x_val.size(0)

                    preds_val_final = self._extract_final_predictions(preds_val).detach().cpu().numpy()
                    y_val_cpu = y_val.detach().cpu().numpy()
                    norm_coeff_val = item["norm_coefficient"].detach().cpu().numpy()

                    for metric in self.metrics:
                        metric.update(preds_val_final, y_val_cpu, norm_coefficient=norm_coeff_val)

            val_seconds = time.time() - start
            val_time_str = self._format_time(val_seconds)

            val_loss /= len(self.val_dataset)
            val_metrics_text = {type(m).__name__: f"{m.compute():.4f}" for m in self.metrics}
            val_metrics = {type(m).__name__: m.compute() for m in self.metrics}

            # Step the learning rate scheduler if it is active
            if self.scheduler is not None:
                self.scheduler.step()
            
            # --- Checkpointing & Logging ---
            if epoch % self.save_every_epoch == 0:
                self.save_checkpoint(epoch, best_val_loss, self.checkpoint_dir / f"epoch_{epoch}.pt")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, best_val_loss, self.checkpoint_dir / "best.pt")

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} | Train Loss: {train_loss:.3e} | Train Metrics: {train_metrics_text} | Train Time: {train_time_str} | Val Loss: {val_loss:.3e} | Val Metrics: {val_metrics_text} | Val Time: {val_time_str} | lr: {current_lr:.3e}")

            # Append results to text log
            with open(log_file, "a") as f:
                f.write(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.3e} | train_metrics={train_metrics_text} | train_time={train_time_str} | "
                    f"val_loss={val_loss:.3e} | val_time={val_time_str} | val_metrics={val_metrics_text} | lr={current_lr:.3e}\n"
                )
                
            # Log metrics to TensorBoard
            if self.use_tensorboard:
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Validation", val_loss, epoch)
                writer.add_scalar("Time/Train", train_seconds, epoch)
                writer.add_scalar("Time/Validation", val_seconds, epoch)
                writer.add_scalar("Learning_Rate", current_lr, epoch)
                
                for name, value in train_metrics.items():
                    writer.add_scalar(f"Metrics/Train/{name}", value, epoch)
                for name, value in val_metrics.items():
                    writer.add_scalar(f"Metrics/Validation/{name}", value, epoch)
                writer.flush()

            # Optuna pruning check
            if self.trial:
                self.trial.report(val_loss, epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

        # Save the final model state after all epochs are done
        self.save_checkpoint(self.epochs, best_val_loss, self.checkpoint_dir / "final.pt")
        print(f"Training completed. Final model saved to {self.checkpoint_dir / 'final.pt'}")

        if self.use_tensorboard:
            writer.close()

        return val_loss

    # ------------------------------------------------------------------------
    # Checkpoint Management
    # ------------------------------------------------------------------------

    def save_checkpoint(self, epoch, best_val_loss, path):
        """Saves the current state of the model, optimizer, and scheduler to a file.

        Args:
            epoch (int): The current training epoch.
            best_val_loss (float): The best validation loss recorded up to this point.
            path (pathlib.Path): The filesystem path where the checkpoint dictionary will be saved.
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
        """Loads model weights, optimizer, and scheduler states from a specified checkpoint.

        Args:
            checkpoint_name (str): The filename of the checkpoint located within `self.checkpoint_dir`.

        Returns:
            tuple: A tuple containing:
                - epoch (int): The epoch at which the checkpoint was originally saved (fallback 0).
                - start_epoch (int): The epoch to resume training from.
                - best_val_loss (float): The recorded best validation loss at the time of saving.

        Raises:
            ValueError: If `checkpoint_name` is None or not provided.
        """
        if checkpoint_name is None:
            raise ValueError("Checkpoint path was not specified.")

        path = self.checkpoint_dir / checkpoint_name
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if it exists in the checkpoint and a scheduler is initialized
        if hasattr(self, "scheduler") and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint.get('epoch', 0), checkpoint["epoch"], checkpoint["best_val_loss"]

    def continue_train(self, checkpoint_name):
        """Resumes the training process from a previously saved checkpoint.

        Args:
            checkpoint_name (str): Filename of the checkpoint to load (e.g., 'epoch_50.pt').

        Returns:
            float: The final validation loss achieved after resuming and completing training.
        """
        epoch, start_epoch, best_val_loss = self.load_model_from_checkpoint(checkpoint_name)
        print(f"Loaded checkpoint from epoch {epoch}. Resuming training at epoch {start_epoch + 1}.")
        return self.train(start_epoch=start_epoch, best_val_loss=best_val_loss)

    # ------------------------------------------------------------------------
    # Inference & Sliding Window Logic
    # ------------------------------------------------------------------------

    def predict(
        self, data_loader, method=None, checkpoint="best.pt", 
        batch_size=1, create_data_loader=False, 
        window_size=256, window_stride=128, sigma_scale=0.25, **kwargs
    ):
        """Generates predictions for a given dataset, supporting standard and sliding window inference.

        Args:
            data_loader (Dataset or DataLoader): The dataset or data loader to evaluate.
            method (str, optional): Inference strategy. Use "sliding_window" for large images. Defaults to None.
            checkpoint (str, optional): Filename of the model weights to load. Defaults to "best.pt".
            batch_size (int, optional): Batch size if creating a new DataLoader internally. Defaults to 1.
            create_data_loader (bool, optional): If True, wraps the input `data_loader` inside a PyTorch DataLoader. Defaults to False.
            window_size (int or tuple, optional): Size of the sliding window patches. Defaults to 256.
            window_stride (int or tuple, optional): Step size for moving the sliding window. Defaults to 128.
            sigma_scale (float, optional): Scaling factor for the Gaussian smoothing matrix. Defaults to 0.25.

        Yields:
            tuple: (item, predictions), where `item` is the original data dictionary and `predictions` is a numpy array.
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

        Processes the entire image at once. Suitable when images fit comfortably in VRAM.

        Args:
            data_loader (DataLoader): PyTorch DataLoader providing batches of data.

        Yields:
            tuple: (item dictionary, model predictions as a numpy array).
        """
        print("Starting standard batch prediction...")
        self.model.eval()
        with torch.no_grad():
            for item in data_loader:
                images = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                preds = self.model(images)
                
                # Use the clean helper method for final extraction
                preds_final = self._extract_final_predictions(preds)
                yield item, preds_final.cpu().numpy()

    def _create_gaussian_window(self, window_size=(256, 256), sigma_scale=0.25):
        """Creates a 2D Gaussian window to smoothly blend overlapping sliding window patches.
        
        Assigns higher weights to the center of a patch and lower weights to the edges, 
        drastically reducing border artifacts during reconstruction.

        Args:
            window_size (int or tuple, optional): Dimensions of the window (H, W). Defaults to (256, 256).
            sigma_scale (float, optional): Standard deviation scaling factor. Defaults to 0.25.

        Returns:
            torch.Tensor: A normalized 2D Gaussian tensor matching the window size.
        """
        if isinstance(window_size, (list, tuple)):
            height, width = window_size
        else:
            height = width = window_size
            
        sigma_y, sigma_x = height * sigma_scale, width * sigma_scale
        
        coords_y = torch.arange(height, dtype=torch.float32) - (height - 1) / 2.0
        coords_x = torch.arange(width, dtype=torch.float32) - (width - 1) / 2.0
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')
        
        # Calculate 2D Gaussian distribution
        gaussian_2d = torch.exp(-(grid_x**2 / (2 * max(sigma_x**2, 1e-6)) + grid_y**2 / (2 * max(sigma_y**2, 1e-6))))
        
        # Normalize so the maximum value at the center is 1.0
        return gaussian_2d / gaussian_2d.max()

    def _sliding_window_predict(
        self, data_loader, window_size=256, window_stride=128, sigma_scale=0.25, 
        min_peak_threshold=0.05, window_output_size=64
    ):
        """Orchestrates the sliding window inference pipeline.

        Pads the input image, extracts crops (patches) using `F.unfold`, processes them 
        through the model, and delegates the merging logic to format-specific methods.

        Args:
            data_loader (DataLoader): PyTorch DataLoader providing the original high-res images.
            window_size (int or tuple, optional): Crop dimensions (H, W). Defaults to 256.
            window_stride (int or tuple, optional): Step size between crops (H, W). Defaults to 128.
            sigma_scale (float, optional): Gaussian smoothing factor. Defaults to 0.25.
            min_peak_threshold (float, optional): Threshold for filtering weak peaks (unused natively). Defaults to 0.05.
            window_output_size (int or tuple, optional): Expected spatial output size per patch. Defaults to 64.

        Yields:
            tuple: (item dictionary, fully reconstructed predictions as a numpy array).
        """
        window_h = window_size[0] if isinstance(window_size, (list, tuple)) else window_size
        window_w = window_size[1] if isinstance(window_size, (list, tuple)) else window_size
        stride_h = window_stride[0] if isinstance(window_stride, (list, tuple)) else window_stride
        stride_w = window_stride[1] if isinstance(window_stride, (list, tuple)) else window_stride

        self.model.eval()

        with torch.no_grad():
            for item in data_loader:
                images = torch.as_tensor(item["image"], dtype=torch.float32, device=self.device)
                B, C, H, W = images.shape

                # Calculate how many patches are needed across height and width
                num_patches_h = max(1, math.ceil(max(0, H - window_h) / stride_h) + 1)
                num_patches_w = max(1, math.ceil(max(0, W - window_w) / stride_w) + 1)

                # Calculate padding required to cover the entire image space
                pad_h = (num_patches_h - 1) * stride_h + window_h - H
                pad_w = (num_patches_w - 1) * stride_w + window_w - W

                padded_images = F.pad(images, (0, pad_w, 0, pad_h), mode="constant", value=0)
                
                # F.unfold acts like a sliding window extractor
                patches = F.unfold(padded_images, kernel_size=(window_h, window_w), stride=(stride_h, stride_w))
                
                # L is the total number of extracted patches per image
                L = patches.shape[-1]
                
                # Reshape to treat each patch as a separate item in the batch for the model
                patches_for_model = patches.transpose(1, 2).reshape(B * L, C, window_h, window_w)

                preds = self.model(patches_for_model)
                
                # Uniformly extract the final stack for inference
                preds = self._extract_final_predictions(preds)

                # Route to specific merge logic based on what the network predicts
                if self.keypoint_format == "keypoints":
                    final_out = self._merge_sliding_window_keypoints(
                        preds, B, L, H, W, window_h, window_w, stride_h, stride_w, 
                        num_patches_h, num_patches_w, window_output_size, sigma_scale
                    )
                else:
                    final_out = self._merge_sliding_window_heatmaps(
                        preds, B, L, H, W, window_h, window_w, stride_h, stride_w, 
                        num_patches_h, num_patches_w, sigma_scale
                    )

                yield item, final_out.cpu().numpy()

    def _merge_sliding_window_keypoints(
        self, preds, B, L, H, W, window_h, window_w, stride_h, stride_w, 
        num_patches_h, num_patches_w, window_output_size, sigma_scale
    ):
        """Processes and merges raw (x, y) keypoint coordinates from overlapping patches.
        
        Maps local patch coordinates back to the global image coordinate system, 
        filters out out-of-bound predictions, evaluates density to cluster multiple 
        predictions for the same keypoint, and computes a Gaussian-weighted average.

        Args:
            preds (torch.Tensor): Raw predictions tensor of shape (B*L, K, 2).
            B (int): Batch size (original).
            L (int): Number of patches per image.
            H (int): Original image height.
            W (int): Original image width.
            window_h (int): Patch height.
            window_w (int): Patch width.
            stride_h (int): Patch stride height.
            stride_w (int): Patch stride width.
            num_patches_h (int): Number of patches along the y-axis.
            num_patches_w (int): Number of patches along the x-axis.
            window_output_size (int or tuple): Expected network spatial output size per patch.
            sigma_scale (float): Scale for the Gaussian weighting.

        Returns:
            torch.Tensor: Weighted and clustered final keypoint coordinates of shape (B, K, 2).
        """
        if preds.ndim != 3 or preds.shape[-1] != 2:
            raise ValueError(f"Expected keypoint output (B*L, K, 2), got {tuple(preds.shape)}")

        _, K, _ = preds.shape
        # Reshape back to separate batch elements and their respective patches
        preds = preds.reshape(B, L, K, 2).clone()

        output_h = window_output_size[0] if isinstance(window_output_size, (list, tuple)) else window_output_size
        output_w = window_output_size[1] if isinstance(window_output_size, (list, tuple)) else window_output_size

        # Scaling factors between original image patch size and network output size
        scale_x = output_w / window_w
        scale_y = output_h / window_h

        # Generate a grid that tracks which row and column each patch belongs to
        grid_y, grid_x = torch.meshgrid(
            torch.arange(num_patches_h, device=self.device),
            torch.arange(num_patches_w, device=self.device),
            indexing="ij"
        )

        # Calculate the top-left global origin coordinates for each patch
        origin_x = (grid_x.reshape(-1) * stride_w * scale_x).to(preds.dtype)
        origin_y = (grid_y.reshape(-1) * stride_h * scale_y).to(preds.dtype)

        # Save local coordinates for weight calculation before shifting them to global
        local_x, local_y = preds[..., 0].clone(), preds[..., 1].clone()

        # Shift local predictions to global image coordinates
        preds[..., 0] += origin_x.view(1, L, 1)
        preds[..., 1] += origin_y.view(1, L, 1)

        # Determine which predictions fall physically outside their patch boundaries or the global image
        valid = (
            (local_x >= 0) & (local_x < output_w) & 
            (local_y >= 0) & (local_y < output_h) & 
            (preds[..., 0] < W * scale_x) & (preds[..., 1] < H * scale_y)
        )

        # Prepare Gaussian weighting based on how close the point is to the patch center
        center_x, center_y = (output_w - 1) / 2, (output_h - 1) / 2
        sigma_x, sigma_y = max(output_w * sigma_scale, 1e-6), max(output_h * sigma_scale, 1e-6)
        weights = torch.exp(-0.5 * (((local_x - center_x) / sigma_x) ** 2 + ((local_y - center_y) / sigma_y) ** 2))

        # Transpose arrays to shape [Batch, Keypoints, Patches] for clustering calculation
        p = preds.transpose(1, 2)
        v = valid.transpose(1, 2).float()
        w = weights.transpose(1, 2) * v
        
        # Compute pairwise Euclidean distance squared between all patches predicting the same keypoint
        dist_sq = ((p.unsqueeze(3) - p.unsqueeze(2)) ** 2).sum(dim=-1)
        # Push invalid points infinitely far away so they are ignored
        dist_sq += (1 - v.unsqueeze(2)) * 1e6

        # Clustering: Two predictions belong to the same anatomical point if distance squared < 25
        is_close = (dist_sq < 25.0).float()
        # Density score determines which cluster of predictions has the highest confidence (weight sum)
        density_score = (is_close * w.unsqueeze(2)).sum(dim=-1)

        # Find the index of the highest density cluster for each keypoint
        best_idx = density_score.argmax(dim=-1)
        best_idx_exp = best_idx.view(B, K, 1, 1).expand(B, K, 1, L)
        best_cluster_mask = torch.gather(is_close, dim=2, index=best_idx_exp).squeeze(2).transpose(1, 2)

        # Zero out weights for predictions that do not belong to the winning cluster
        cluster_weights = weights * best_cluster_mask * valid.float()

        # Compute the final Gaussian-weighted average of the valid clustered coordinates
        weighted_sum = (preds * cluster_weights.unsqueeze(-1)).sum(dim=1)
        weight_sum = cluster_weights.sum(dim=1, keepdim=False).unsqueeze(-1)

        final_keypoints = weighted_sum / weight_sum.clamp_min(1e-8)
        
        # Replace empty clusters (weight_sum == 0) with NaNs
        return torch.where(weight_sum > 0, final_keypoints, torch.full_like(final_keypoints, float("nan")))

    def _merge_sliding_window_heatmaps(
        self, preds, B, L, H, W, window_h, window_w, stride_h, stride_w, 
        num_patches_h, num_patches_w, sigma_scale
    ):
        """Reconstructs full global heatmaps from sliding window patches.
        
        Applies a Gaussian weight to each predicted patch so edges are down-weighted, 
        then maps them back onto a global canvas using `F.fold`.

        Args:
            preds (torch.Tensor): Raw heatmap predictions of shape (B*L, K, out_h, out_w).
            B (int): Batch size (original).
            L (int): Number of patches per image.
            H (int): Original image height.
            W (int): Original image width.
            window_h (int): Patch height.
            window_w (int): Patch width.
            stride_h (int): Patch stride height.
            stride_w (int): Patch stride width.
            num_patches_h (int): Number of patches along the y-axis.
            num_patches_w (int): Number of patches along the x-axis.
            sigma_scale (float): Scale for the Gaussian weighting.

        Returns:
            torch.Tensor: Global reconstructed heatmap of shape (B, K, H_scaled, W_scaled).
        """
        if preds.ndim != 4:
            raise ValueError(f"Expected heatmap output (B*L, K, H, W), got {tuple(preds.shape)}")

        _, K, out_h, out_w = preds.shape
        scale_y, scale_x = out_h / window_h, out_w / window_w
        
        # Calculate scaled stride size to match the network's output dimensions
        out_stride_h, out_stride_w = max(1, round(stride_h * scale_y)), max(1, round(stride_w * scale_x))

        # Canvas size including padded regions
        out_h_pad = (num_patches_h - 1) * out_stride_h + out_h
        out_w_pad = (num_patches_w - 1) * out_stride_w + out_w
        
        # Actual size of the global image in output scale space
        out_h_full, out_w_full = math.ceil(H * scale_y), math.ceil(W * scale_x)

        # Utilize native non-square support in _create_gaussian_window
        gaussian = self._create_gaussian_window(
            window_size=(out_h, out_w), 
            sigma_scale=sigma_scale
        ).to(self.device)

        # Flatten spatial dimensions for efficient multiplication
        gaussian_flat = gaussian.reshape(-1, 1)
        preds_reshaped = preds.reshape(B, L, K, out_h * out_w)
        
        # Apply Gaussian weights to the heatmap patch
        preds_weighted = preds_reshaped * gaussian_flat.view(1, 1, 1, -1)
        preds_flat = preds_weighted.reshape(B, L, K * out_h * out_w).transpose(1, 2)

        # F.fold performs the inverse of F.unfold, stitching the patches back together
        reconstructed_sum = F.fold(
            preds_flat, output_size=(out_h_pad, out_w_pad),
            kernel_size=(out_h, out_w), stride=(out_stride_h, out_stride_w)
        )

        # Track the accumulated Gaussian weights at each pixel to normalize overlapping areas
        weight_src = gaussian_flat.view(1, -1, 1).expand(B, -1, L)
        overlap_weight_map = F.fold(
            weight_src, output_size=(out_h_pad, out_w_pad),
            kernel_size=(out_h, out_w), stride=(out_stride_h, out_stride_w)
        )

        # Divide by sum of weights to get the final blended heatmap
        final_heatmaps = reconstructed_sum / overlap_weight_map.clamp_min(1e-8)
        
        # Crop the padded regions out to return strictly the original image bounds
        return final_heatmaps[:, :, :out_h_full, :out_w_full]