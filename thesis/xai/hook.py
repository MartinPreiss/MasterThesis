import torch
from torch import nn

def get_layer_hooks(model,layer_ids=None):
    try:
        if not layer_ids:
            layer_ids = range(len(model.model.layers))
        hooks = []
        for id,layer in enumerate(model.model.layers): 
            if id in layer_ids:
                hooks.append(Hook(layer))
    except:
        if not layer_ids:
            layer_ids = range(len(model.language_model.model.layers))
        hooks = []
        for id,layer in enumerate(model.language_model.model.layers): 
            if id in layer_ids:
                hooks.append(Hook(layer))

    return hooks

class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        self.module_name = module.__class__.__name__

    def save_grad(self, module, input, output):
        self.data = output[0]
        #output.requires_grad_(True)
        #output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    def clear_hook(self):
        del self.data

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad

#optional print hook
def print_hook(module, input, output):
    print(output.shape)
    return None