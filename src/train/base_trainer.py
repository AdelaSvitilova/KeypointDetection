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

    def __init__(self, model, train_dataset, val_dataset=None, loss_fn=None, metrics=None, batch_size=16, epochs=10, framework='numpy'):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.batch_size = batch_size
        self.epochs = epochs
        self.framework = framework

    @abstractmethod
    def train(self):
        """
        Main training loop.

        Must be implemented by a framework-specific trainer subclass.
        """
        pass

    @abstractmethod
    def validate(self, val_loader, epoch):
        """
        Validate the model on the validation dataset.

        Must be implemented by a framework-specific trainer subclass.

        Parameters:
            val_loader: DataLoader or iterator over the validation dataset.
            epoch (int): Current epoch number (optional, for logging purposes).
        """
        pass
