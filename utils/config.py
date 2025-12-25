import yaml
import os

def load_config(config_path="configs/config.yaml"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, config_path):
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

