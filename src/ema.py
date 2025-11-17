"""
Exponential Moving Average (EMA) for model parameters.

EMA maintains a smoothed copy of model parameters that updates more slowly
than the training model, leading to more stable and better-performing models.
"""

import torch
from collections import OrderedDict
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Maintains a separate copy of model parameters that are updated using
    exponential moving average: ema = ema * rate + model * (1 - rate)
    
    Args:
        model: The model to create EMA for
        ema_rate: EMA decay rate (default: 0.999)
        device: Device to store EMA parameters on
    """
    
    def __init__(self, model, ema_rate=0.999, device=None):
        """
        Initialize EMA with a copy of model parameters.
        
        Args:
            model: PyTorch model to create EMA for
            ema_rate: Decay rate for EMA (0.999 means 99.9% old, 0.1% new)
            device: Device to store EMA parameters (defaults to model's device)
        """
        self.ema_rate = ema_rate
        self.device = device if device is not None else next(model.parameters()).device
        
        # Create EMA state dict (copy of model parameters)
        self.ema_state = OrderedDict()
        model_state = model.state_dict()
        
        for key, value in model_state.items():
            # Copy parameter data to EMA state
            self.ema_state[key] = deepcopy(value.data).to(self.device)
        
        # Parameters to ignore (not trainable, should be copied directly)
        self.ignore_keys = [
            x for x in self.ema_state.keys() 
            if ('running_' in x or 'num_batches_tracked' in x)
        ]
    
    def update(self, model):
        """
        Update EMA state with current model parameters.
        
        Should be called after optimizer.step() to update EMA with the
        newly optimized model weights.
        
        Args:
            model: The model to read parameters from
        """
        with torch.no_grad():
            source_state = model.state_dict()
            
            for key, value in self.ema_state.items():
                if key in self.ignore_keys:
                    # For non-trainable parameters (e.g., BatchNorm stats), copy directly
                    self.ema_state[key] = source_state[key].to(self.device)
                else:
                    # EMA update: ema = ema * rate + model * (1 - rate)
                    source_param = source_state[key].detach().to(self.device)
                    self.ema_state[key].mul_(self.ema_rate).add_(source_param, alpha=1 - self.ema_rate)
    
    def apply_to_model(self, model):
        """
        Load EMA state into model.
        
        This replaces model parameters with EMA parameters. Useful for
        validation or inference using the EMA model.
        
        Args:
            model: Model to load EMA state into
        """
        model.load_state_dict(self.ema_state)
    
    def state_dict(self):
        """
        Get EMA state dict for saving.
        
        Returns:
            OrderedDict: EMA state dictionary
        """
        return self.ema_state
    
    def load_state_dict(self, state_dict):
        """
        Load EMA state from saved checkpoint.
        
        Args:
            state_dict: EMA state dictionary to load
        """
        self.ema_state = OrderedDict(state_dict)
    
    def add_ignore_key(self, key_pattern):
        """
        Add a key pattern to ignore list.
        
        Parameters matching this pattern will be copied directly instead
        of using EMA update.
        
        Args:
            key_pattern: String pattern to match (e.g., 'relative_position_index')
        """
        matching_keys = [x for x in self.ema_state.keys() if key_pattern in x]
        self.ignore_keys.extend(matching_keys)
        # Remove duplicates
        self.ignore_keys = list(set(self.ignore_keys))

