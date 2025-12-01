from .stacked_hourglass_keras import StackedHourglassKeras
from .stacked_hourglass_pytorch import StackedHourglassPytorch
from .test_model_pytorch import TestModelPytorch
from .test_model_keras import TestModelKeras
from .test_model_heatmap_pytorch import TestModelHeatmapPytorch
from .test_model_heatmap_keras import TestModelHeatmapKeras

# tato funkce dostane nízev modelu, který má použít a odpovídající parametry a vytvoří instanci třídy s danámi parametry
def get_model(name, **kwargs):
    """
    Factory function to create a model instance based on its name.

    This function selects the appropriate model class and initializes it with
    the provided keyword arguments. It allows switching between different
    model architectures and frameworks using a unified interface.

    Parameters:
        name (str): Name of the model to create. Should match one of the 
                    registered model names in the factory.
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        An instance of the selected model class.

    Raises:
        ValueError: If the specified model name is not recognized.
    """
    models = {
        "stacked_hourglass_keras": StackedHourglassKeras,
        "stacked_hourglass_pytorch": StackedHourglassPytorch,
        "test_model_pytorch": TestModelPytorch,
        "test_model_keras": TestModelKeras,
        "test_model_heatmap_pytorch": TestModelHeatmapPytorch,
        "test_model_heatmap_keras": TestModelHeatmapKeras,
    }
    if name not in models:
        raise ValueError(f"Unknown model name: {name}")
    return models[name](**kwargs)