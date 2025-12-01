import keras
from keras.losses import MeanSquaredError
from .base_loss import BaseLoss

class KeypointMSELossKeras(BaseLoss):
    """
    Keras implementation of Mean Squared Error (MSE) loss for keypoints.

    This class wraps Keras' built-in MeanSquaredError loss function and provides
    a consistent interface through the BaseLoss abstract class. It can be used
    wherever a framework-specific loss function is required.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = MeanSquaredError()

    def __call__(self, preds, targets):
        """
        Compute the MSE loss between predictions and target keypoints.

        Parameters:
            preds: Predicted keypoints (tensor or array compatible with Keras).
            targets: Ground truth keypoints (tensor or array compatible with Keras).

        Returns:
            Tensor: Computed MSE loss.
        """
        return self.loss_fn(preds, targets)
