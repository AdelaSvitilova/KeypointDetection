from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """Framework-agnostic rozhraní."""

    @abstractmethod
    def predict(self, x):
        """Vrací výstup modelu jako numpy nebo tensor podle implementace."""
        pass

    @property
    @abstractmethod
    def input_channels(self):
        pass

    @property
    @abstractmethod
    def num_keypoints(self):
        pass

import torch
import torch.nn as nn

class TorchModel(nn.Module, ModelInterface):
    def __init__(self):
        nn.Module.__init__(self)
        ModelInterface.__init__(self)

    @abstractmethod
    def forward(self, x):
        pass

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        
import tensorflow as tf

class TFModel(tf.keras.Model, ModelInterface):
    def predict(self, x):
        return super().predict(x)
    
from keras import Model, layers

class KerasModel(Model, ModelInterface):

    def __init__(self):
        super().__init__()

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def num_keypoints(self):
        return self._num_keypoints

    def call(self, inputs, training=False):
        raise NotImplementedError("Implementujte call() v konkrétním modelu.")

    def predict(self, x):
        return super().predict(x)

