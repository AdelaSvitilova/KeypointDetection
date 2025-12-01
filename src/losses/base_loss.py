from abc import ABC, abstractmethod

class BaseLoss(ABC):
    """
    Abstract base class for all loss functions.
    
    Subclasses must implement the __call__ method.
    Note: The implementation of the loss is framework-dependent (e.g., PyTorch, TensorFlow),
    because different frameworks handle tensors and gradients differently. Using a 
    framework-agnostic implementation may lead to incorrect results.
    """

    @abstractmethod
    def __call__(self, preds, targets):
        """
        Compute the loss from predictions and targets.
        """
        pass
