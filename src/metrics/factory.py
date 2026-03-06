from .pck import PCK
from .pck_heatmaps import PCKHeatmaps
from .aed import AED
from .aed_heatmaps import AEDHeatmaps

def get_metrics(names, **kwargs):
    """
    Factory function to create metric instances based on their names.

    This function selects the appropriate metric class and initializes it with
    any additional keyword arguments. All metrics are framework-agnostic.

    Parameters:
        names (list of str): List of metric names to create (e.g., ["PCK"]).
        **kwargs: Additional keyword arguments to pass to the metric constructors.

    Returns:
        list: Instances of the selected metric classes.

    Raises:
        ValueError: If a requested metric name is not recognized.
    """
    metrics = {
        "PCK": PCK,
        "PCK_heatmaps": PCKHeatmaps,
        "AED": AED,
        "AED_heatmaps": AEDHeatmaps,
    }
    selected = []
    for name in names:
        if name not in metrics:
            raise ValueError(f"Unknown metric: {name}")
        selected.append(metrics[name](**kwargs))
    return selected
