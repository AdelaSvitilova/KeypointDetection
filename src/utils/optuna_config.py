import copy
import optuna


def set_nested(config, path, value):
    """
    Set a value in a nested dictionary using dot-separated path.

    Args:
        config (dict): Target configuration dictionary.
        path (str): Dot-separated path (e.g. "model.lr").
        value: Value to assign.
    """
    keys = path.split(".")
    d = config

    # Traverse nested dictionary structure
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]

    # Set final value
    d[keys[-1]] = value


def apply_optuna(cfg, trial):
    """
    Apply Optuna hyperparameter sampling to a configuration dictionary.

    Args:
        cfg (dict): Base configuration.
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        dict: Config with sampled hyperparameters.
    """
    cfg_optuna = copy.deepcopy(cfg)

    # Iterate over all Optuna-defined parameters
    for path, spec in cfg_optuna["optuna"].items():

        print(path)

        param_type = spec["type"]

        if param_type == "int":
            value = trial.suggest_int(
                path,
                spec["low"],
                spec["high"]
            )

        elif param_type == "float":
            value = trial.suggest_float(
                path,
                spec["low"],
                spec["high"]
            )

        elif param_type == "cat":
            value = trial.suggest_categorical(
                path,
                spec["choices"]
            )

        elif param_type == "list":
            # Sample two categorical components independently
            ch1 = trial.suggest_categorical(
                f"{path}_1",
                spec["choices1"]
            )

            ch2 = trial.suggest_categorical(
                f"{path}_2",
                spec["choices2"]
            )

            value = [ch1, ch2]

        elif param_type == "weights":
            # Sample weight and derive complementary value
            w1 = trial.suggest_float(
                path,
                spec["low"],
                spec["high"]
            )

            value = [w1, 1 - w1]

        else:
            raise ValueError(f"Unknown Optuna type: {param_type}")

        # Apply sampled value into nested config
        set_nested(cfg_optuna, path, value)

    return cfg_optuna


def apply_optuna_best(cfg, best):
    """
    Apply best hyperparameters (from Optuna study) to configuration.

    Args:
        cfg (dict): Base configuration.
        best (dict): Dictionary of best parameters (dot-separated keys).

    Returns:
        dict: Updated configuration with best hyperparameters.
    """
    cfg_optuna = copy.deepcopy(cfg)

    # Inject best parameters into config
    for path, value in best.items():
        if path in cfg_optuna["optuna"]:
            param_type = cfg_optuna["optuna"][path]["type"]

            # Weights are stored in Optuna as a single value w1.
            # The second weight is derived as 1 - w1.
            if param_type == "weights":
                value = [value, 1 - value]

        else:
            # Parameters stored as separate list elements:
            # <path>_1, <path>_2
            base_path = path.rsplit("_", 1)[0]

            if (
                path.endswith("_1")
                and base_path in cfg_optuna["optuna"]
                and cfg_optuna["optuna"][base_path]["type"] == "list"
            ):
                path = base_path

                value = [
                    best[f"{path}_1"],
                    best[f"{path}_2"],
                ]

            else:
                continue

        set_nested(cfg_optuna, path, value)


    return cfg_optuna