from .trainer_pytorch import PytorchTrainer
from .trainer_keras import KerasTrainer

def get_trainer(backend, **kwargs):
    """
    Factory function to create a trainer instance based on the backend/framework.

    This function selects the appropriate framework-specific trainer class
    (e.g., PyTorch or Keras) and initializes it with the provided keyword arguments.
    It allows switching between different training backends using a unified interface.

    Parameters:
        backend (str): Name of the backend/framework to use ("pytorch", "keras", etc.).
        **kwargs: Additional keyword arguments to pass to the trainer constructor.

    Returns:
        An instance of the selected trainer class.

    Raises:
        ValueError: If the specified backend is not recognized.
    """
    backends = {
        "pytorch": PytorchTrainer,
        "keras": KerasTrainer,
    }
    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}")
    return backends[backend](**kwargs)
