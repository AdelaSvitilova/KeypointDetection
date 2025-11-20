import keras
from keras.losses import MeanSquaredError
from .base_loss import BaseLoss


class KeypointMSELossKeras(BaseLoss):

    def __init__(self):
        super().__init__()
        self.loss_fn = MeanSquaredError()

    def __call__(self, preds, targets):
        return self.loss_fn(preds, targets)
