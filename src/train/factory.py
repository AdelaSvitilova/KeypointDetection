from .trainer_pytorch import PytorchTrainer
from .trainer_keras import KerasTrainer

# tato funkce dostane nízev modelu, který má použít a odpovídající parametry a vytvoří instanci třídy s danámi parametry
def get_trainer(backend, **kwargs):
    backends = {
        "pytorch": PytorchTrainer,
        "keras": KerasTrainer,
    }
    if backend not in backends:
        raise ValueError(f"Unknown model name: {backend}")
    return backends[backend](**kwargs)