import yaml
import os

def merge_configs(base_cfg, override_cfg):
    """
    Rekurzivně sloučí konfigurace:
    - slovníky se mergují
    - ostatní hodnoty se přepisují override hodnotami
    """
    for key, val in override_cfg.items():
        if (
            key in base_cfg
            and isinstance(base_cfg[key], dict)
            and isinstance(val, dict)
        ):
            merge_configs(base_cfg[key], val)
        else:
            base_cfg[key] = val
    return base_cfg


def load_config(config_stack_path):
    """
    Načte YAML soubor, který obsahuje seznam konfigurací pod klíčem 'configs',
    následně je v uvedeném pořadí rekurzivně sloučí.

    Parametry:
        config_stack_path (str): cesta k YAML souboru se seznamem konfiguračních souborů

    Návrat:
        dict: výsledná konfigurace
    """
    if not os.path.isfile(config_stack_path):
        raise FileNotFoundError(f"Soubor se seznamem konfigurací nenalezen: {config_stack_path}")

    # Načti meta-config s listem 'configs'
    with open(config_stack_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    if "configs" not in meta or not isinstance(meta["configs"], list):
        raise ValueError("Soubor musí obsahovat klíč 'configs' se seznamem YAML souborů.")

    paths = meta["configs"]
    if not paths:
        raise ValueError("Seznam 'configs' je prázdný.")
    print(paths)

    # --- původní logika sloučení ---
    base_path = paths[0]
    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"Základní soubor nenalezen: {base_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Přepis dalšími soubory
    for path in paths[1:]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Soubor pro přepis nenalezen: {path}")

        with open(path, "r", encoding="utf-8") as f:
            override_cfg = yaml.safe_load(f) or {}

        merge_configs(cfg, override_cfg)

    return cfg
