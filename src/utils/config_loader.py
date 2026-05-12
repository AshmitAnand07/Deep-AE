import os
import yaml
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads a YAML configuration file.
    
    Args:
        config_path (str): Path to the config file. Defaults to "config.yaml".
        
    Returns:
        dict: Parsed configuration dictionary.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[CONFIG_ERROR] Configuration file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"[CONFIG_ERROR] Failed to parse YAML file at {config_path}: {e}")
