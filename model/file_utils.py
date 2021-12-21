import importlib.util

def is_torch_available():
    return importlib.util.find_spec("torch") is not None
    
def is_flax_available():
    return importlib.util.find_spec("jax") is not None and importlib.util.find_spec("flax") is not None