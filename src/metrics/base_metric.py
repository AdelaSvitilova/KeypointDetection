from abc import ABC, abstractmethod

class BaseMetric(ABC):
    @abstractmethod
    def update(self, preds, targets):
        pass

    @abstractmethod
    def compute(self):
        pass

    def reset(self):
        pass
