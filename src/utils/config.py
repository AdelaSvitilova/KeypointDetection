import yaml
import os
from pathlib import Path

def merge_configs(base_cfg, override_cfg):
    """Recursively merge two dictionaries.

    Nested dictionaries are merged. Values in `override_cfg` 
    overwrite values in `base_cfg`.

    Args:
        base_cfg (dict): The base dictionary to merge into.
        override_cfg (dict): The dictionary whose values overwrite base_cfg.

    Returns:
        dict: The merged dictionary.
    """
    for key, val in override_cfg.items():
        # If both values are dictionaries, merge recursively
        if key in base_cfg and isinstance(base_cfg[key], dict) and isinstance(val, dict):
            merge_configs(base_cfg[key], val)
        else:
            # Otherwise, override the value
            base_cfg[key] = val
    return base_cfg


def load_config(config_stack_path):
    """Load and merge YAML configuration files listed in a stack YAML.

    The YAML file specified by `config_stack_path` must contain a list of
    configuration file paths under the key 'configs'. Files are merged
    in order, with later files overriding earlier ones.

    Args:
        config_stack_path (str): Path to the YAML file containing the list of configs.

    Returns:
        dict: The final merged configuration.

    Raises:
        FileNotFoundError: If any listed config file does not exist.
        ValueError: If 'configs' key is missing or empty.
    """
    if not os.path.isfile(config_stack_path):
        raise FileNotFoundError(
            f"Configuration stack file not found: {config_stack_path}"
        )

    # Load the stack file containing the list of configs
    with open(config_stack_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    if "configs" not in meta or not isinstance(meta["configs"], list):
        raise ValueError(
            "The YAML stack file must contain a 'configs' key with a list of YAML files."
        )

    paths = meta["configs"]
    if not paths:
        raise ValueError("The 'configs' list is empty.")

    # Load the first config as the base
    base_path = paths[0]
    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"Base config file not found: {base_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Merge the remaining config files
    for path in paths[1:]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Override config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            override_cfg = yaml.safe_load(f) or {}

        merge_configs(cfg, override_cfg)

    return cfg

def save_config(config, experiment_name, file_name="config.yaml"):
    config_path = Path("results", experiment_name, file_name)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
