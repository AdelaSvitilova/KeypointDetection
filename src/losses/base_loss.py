# src/losses/base_loss.py
from abc import ABC, abstractmethod

class BaseLoss(ABC):
    @abstractmethod
    def __call__(self, preds, targets):
        pass
