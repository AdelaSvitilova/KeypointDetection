import yaml
import os

def merge_configs(base_cfg, override_cfg):
    """
    Rekurzivně spojí slovníky: hodnoty v override_cfg přepíšou base_cfg.
    """
    for k, v in override_cfg.items():
        if k in base_cfg and isinstance(base_cfg[k], dict) and isinstance(v, dict):
            merge_configs(base_cfg[k], v)
        else:
            base_cfg[k] = v
    return base_cfg

def load_config(paths):
    """
    Načte libovolný počet YAML configů, spojí je a vrátí výsledný slovník.
    První soubor je považován za základní, následující přepíší hodnoty.
    Vyhodí výjimku, pokud seznam cest je prázdný nebo soubor neexistuje.
    
    Parametry:
        paths (list[str]): seznam cest k YAML souborům

    Návrat:
        dict: výsledná konfigurace
    """
    if not paths:
        raise ValueError("Nebyl zadán žádný YAML soubor.")

    # Načtení základního configu
    base_path = paths[0]
    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"Základní soubor nenalezen: {base_path}")

    with open(base_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Postupné přepisování dalšími soubory
    for path in paths[1:]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Soubory pro přepis nenalezeny: {path}")
        with open(path, "r") as f:
            override_cfg = yaml.safe_load(f) or {}
        cfg = merge_configs(cfg, override_cfg)

    return cfg