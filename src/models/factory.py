from .stacked_hourglass_keras import StackedHourglassKeras
from .stacked_hourglass_pytorch import StackedHourglassPytorch
from .test_model_pytorch import TestModelPytorch
from .test_model_keras import TestModelKeras

# tato funkce dostane nízev modelu, který má použít a odpovídající parametry a vytvoří instanci třídy s danámi parametry
def get_model(name, **kwargs):
    models = {
        "stacked_hourglass_keras": StackedHourglassKeras,
        "stacked_hourglass_pytorch": StackedHourglassPytorch,
        "test_model_pytorch": TestModelPytorch,
        "test_model_keras": TestModelKeras,
    }
    if name not in models:
        raise ValueError(f"Unknown model name: {name}")
    return models[name](**kwargs)