import yaml
import torch
from .model import ESpecFormer  

def load_config(yaml_path: str) -> dict:
    """Reads a YAML configuration file."""
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def build_model_from_config(config: dict, device: str = 'cpu') -> torch.nn.Module:
    """Instantiates the ESpecFormer model directly from a config dictionary."""
    
    model_kwargs = {k: v for k, v in config['model'].items() if k != 'variant'}
    
    # Extract num_classes from the dataset block
    num_classes = config['dataset']['num_classes']
    
    # Combine and unpack all arguments into the model
    model = ESpecFormer(num_classes=num_classes, **model_kwargs).to(device)
    
    return model