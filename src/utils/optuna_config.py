import copy
import optuna

def set_nested(config, path, value):

    keys = path.split(".")
    d = config

    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]

    d[keys[-1]] = value

def apply_optuna(cfg, trial):
    cfg_optuna = copy.deepcopy(cfg)

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
            w1 = trial.suggest_float(
                path,
                spec["low"],
                spec["high"]
            )

            value = [w1, 1-w1]

        else:
            raise ValueError(f"Unknown Optuna type: {param_type}")

        set_nested(cfg_optuna, path, value)

    return cfg_optuna

def apply_optuna_best(cfg, best):
    cfg_optuna = copy.deepcopy(cfg)

    for path, value in best.items():
        set_nested(cfg_optuna, path, value)

    return cfg_optuna