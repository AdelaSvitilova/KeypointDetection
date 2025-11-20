from .pck import PCK

def get_metrics(names, **kwargs):
    metrics = {
        "PCK": PCK,
    }
    selected = []
    for name in names:
        if name not in metrics:
            raise ValueError(f"Unknown metric: {name}")
        selected.append(metrics[name](**kwargs))
    return selected
