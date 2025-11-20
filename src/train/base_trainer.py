from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
    Abstraktní třída pro trénery framework-agnostic.
    Každý konkrétní trainer (PyTorch, Keras, TensorFlow) musí implementovat tyto metody.
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
        Hlavní tréninkový loop.
        Musí být implementován ve framework-specifickém traineru.
        """
        pass

    @abstractmethod
    def validate(self, val_loader, epoch):
        """
        Validace modelu.
        Musí být implementováno ve framework-specifickém traineru.
        """
        pass
