from .mse_loss_pytorch import KeypointMSELossPytorch
from .mse_loss_keras import KeypointMSELossKeras
from .multi_mse_loss_pytorch import MultiMSELossPytorch

def get_loss(name, backend, **kwargs):
    backend = backend.lower()

    losses = {
        "MSE": {
            "pytorch": KeypointMSELossPytorch,
            "keras": KeypointMSELossKeras,
            # "numpy": KeypointMSELossNumpy,
        },
        "multi_MSE": {
            "pytorch": MultiMSELossPytorch
        }
    }

    if name not in losses:
        raise ValueError(f"Unknown loss name: {name}")
    if backend not in losses[name]:
        raise ValueError(f"Loss '{name}' not implemented for backend '{backend}'")

    loss_cls = losses[name][backend]
    return loss_cls(**kwargs)
