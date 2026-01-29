from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
    Abstract base class for trainers (framework-agnostic).

    Subclasses for specific frameworks (e.g., PyTorch, Keras, TensorFlow) 
    must implement the `train` and `validate` methods. This class provides 
    a common interface and stores shared training parameters.

    Parameters:
        model: The model to train.
        train_dataset: Dataset used for training.
        val_dataset: Optional dataset for validation.
        loss_fn: Loss function to optimize.
        metrics: List of metric instances to track during training/validation.
        batch_size (int): Number of samples per batch.
        epochs (int): Number of training epochs.
        framework (str): Framework name, used for informational purposes.
    """

    def __init__(self, model, train_dataset, val_dataset=None, loss_fn=None, metrics=None, batch_size=16, epochs=10, lr=0.01, framework='pytorch', keypoint_format="keypoints"):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.framework = framework
        self.keypoint_format = keypoint_format

    @abstractmethod
    def train(self):
        """
        Main training loop.

        Must be implemented by a framework-specific trainer subclass.
        """
        pass

    @abstractmethod
    def predict(self, val_loader, epoch):
        """
        Run inference on a dataset and return model predictions.

        This method performs forward passes only (no loss computation,
        no backpropagation, no metric updates).

        Must be implemented by a framework-specific trainer subclass.

        Parameters:
            data_loader: DataLoader or iterable providing input batches.
            epoch (int, optional): Current epoch number (for logging or tracking).

        Returns:
            predictions: Model outputs aggregated over the entire dataset.
        """
        pass

    @abstractmethod
    def continue_train(self):
        pass
