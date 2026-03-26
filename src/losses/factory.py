from .mse_loss_pytorch import MSELossPytorch
from .mse_loss_keras import MSELossKeras
from .l2_norm_loss_pytorch import L2NormLossPytorch
from .aed_loss_pytorch import AEDLossPytorch
from .huber_loss_pytorch import HuberLossPytorch
from .multi_mse_loss_pytorch import MultiMSELossPytorch
from .multi_l2_norm_loss_pytorch import MultiL2NormLossPytorch
from .multi_aed_loss_pytorch import MultiAEDLossPytorch
from .multi_huber_loss_pytorch import MultiHuberLossPytorch

def get_loss(name, backend, **kwargs):
    """
    Factory function to create a loss instance based on the loss name and backend.

    This function selects the appropriate loss class for the specified backend
    (e.g., PyTorch or Keras) and initializes it with any additional keyword arguments.
    It ensures that the loss is compatible with the framework being used.

    Parameters:
        name (str): Name of the loss function (e.g., "MSE", "multi_MSE").
        backend (str): Backend/framework to use ("pytorch" or "keras").
        **kwargs: Additional keyword arguments to pass to the loss constructor.

    Returns:
        An instance of the selected loss class.

    Raises:
        ValueError: If the loss name or backend is not supported.
    """
    backend = backend.lower()

    losses = {
        "MSE": {
            "pytorch": MSELossPytorch,
            "keras": MSELossKeras,
        },
        "multi_MSE": {
            "pytorch": MultiMSELossPytorch
        }, 
        "L2Norm": {
            "pytorch": L2NormLossPytorch
        },
        "multi_L2Norm": {
            "pytorch": MultiL2NormLossPytorch
        },
        "AED": {
            "pytorch": AEDLossPytorch
        },
        "multi_AED": {
            "pytorch": MultiAEDLossPytorch
        },
        "Huber": {
            "pytorch": HuberLossPytorch
        },
        "multi_Huber": {
            "pytorch": MultiHuberLossPytorch
        },
    }

    if name not in losses:
        raise ValueError(f"Unknown loss name: {name}")
    if backend not in losses[name]:
        raise ValueError(f"Loss '{name}' not implemented for backend '{backend}'")

    loss_cls = losses[name][backend]
    return loss_cls(**kwargs)
