import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .base_trainer import BaseTrainer

class PytorchTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        loss_fn,
        metrics=None,
        batch_size=16,
        lr=0.01,
        epochs=10,
        device=None,
        val_every_epochs=1
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.val_every_epochs = val_every_epochs

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for metric in self.metrics:
                metric.reset()
            
            running_loss = 0.0

            for x, y in train_loader:
                # převod na torch.Tensor až tady
                x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
                y = torch.as_tensor(y, dtype=torch.float32, device=self.device)

                self.optimizer.zero_grad()
                preds = self.model(x)

                # loss = torch-based → backprop funguje
                if preds.ndim == 5:
                    preds = preds[:, 0]
                    
                if y.shape[-2:] != preds.shape[2:]:  # porovnáváme poslední dvě dimenze (H, W)
                    y_small = F.interpolate(y, size=preds.shape[2:], mode='bilinear', align_corners=False)
                else:
                    y_small = y
                    
                loss = self.loss_fn(preds, y_small)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * x.size(0)

                # metriku počítáme na detachnutých tensorech
                for metric in self.metrics:
                    metric.update(preds.detach(), y_small.detach())

            epoch_loss = running_loss / len(self.train_dataset)
            epoch_metrics = {type(m).__name__: m.compute() for m in self.metrics}
            print(f"Epoch {epoch}/{self.epochs} | Train Loss: {epoch_loss:.4f} | Train Metrics: {epoch_metrics}")

            if self.val_dataset is not None and (epoch % self.val_every_epochs == 0):
                self.validate(val_loader, epoch)

    def validate(self, val_loader, epoch):
        self.model.eval()
        for metric in self.metrics:
            metric.reset()

        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = torch.as_tensor(x_val, dtype=torch.float32, device=self.device)
                y_val = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
                
                preds_val = self.model(x_val)

                if preds_val.ndim == 5:
                    preds_val = preds_val[:, 0]
                    
                if y_val.shape[-2:] != preds_val.shape[2:]:  # porovnáváme poslední dvě dimenze (H, W)
                    y_val_small = F.interpolate(y_val, size=preds_val.shape[2:], mode='bilinear', align_corners=False)
                else:
                    y_val_small = y_val

                
                loss_val = self.loss_fn(preds_val, y_val_small)
                val_loss += loss_val.item() * x_val.size(0)

                for metric in self.metrics:
                    metric.update(preds_val.detach(), y_val_small.detach())

        val_loss /= len(self.val_dataset)
        val_metrics = {type(m).__name__: m.compute() for m in self.metrics}
        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Val Metrics: {val_metrics}")
