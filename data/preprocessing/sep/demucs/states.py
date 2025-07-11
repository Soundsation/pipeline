from contextlib import contextmanager

import functools
import hashlib
import inspect
import io
from pathlib import Path
import warnings

from omegaconf import OmegaConf
from diffq import DiffQuantizer, UniformQuantizer, restore_quantized_state
import torch


# Function to get the appropriate quantizer based on the given arguments
# Supports DiffQuantizer and UniformQuantizer
# Optionally sets up optimizer for DiffQuantizer
def get_quantizer(model, args, optimizer=None):
    """Return the quantizer given the XP quantization args."""
    quantizer = None
    if args.diffq:
        quantizer = DiffQuantizer(
            model, min_size=args.min_size, group_size=args.group_size)
        if optimizer is not None:
            quantizer.setup_optimizer(optimizer)
    elif args.qat:
        quantizer = UniformQuantizer(
                model, bits=args.qat, min_size=args.min_size)
    return quantizer


# Function to load a model from a serialized package or file path
# Supports loading from dict or file path
# Handles strict and non-strict loading of model parameters
def load_model(path_or_package, strict=False):
    """Load a model from the given serialized model, either given as a dict (already loaded)
    or a path to a file on disk."""
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            package = torch.load(path, 'cpu')
    else:
        raise ValueError(f"Invalid type for {path_or_package}.")

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    # If strict, instantiate model with all args and kwargs
    # Otherwise, drop kwargs that are not in the model's __init__ signature
    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]

    # Set the model state
    set_state(model, state)
    return model


# Function to get the state dictionary from a model
# Supports quantized state if quantizer is provided
# Optionally converts model parameters to half precision to save space
def get_state(model, quantizer, half=False):
    """Get the state from a model, potentially with quantization applied.
    If `half` is True, model are stored as half precision, which shouldn't impact performance
    but half the state size."""
    if quantizer is None:
        dtype = torch.half if half else None
        state = {k: p.data.to(device='cpu', dtype=dtype) for k, p in model.state_dict().items()}
    else:
        state = quantizer.get_quantized_state()
        state['__quantized'] = True
    return state


# Function to set the state dictionary on a model
# Supports restoring quantized state if quantizer is provided
def set_state(model, state, quantizer=None):
    """Set the state on a given model."""
    if state.get('__quantized'):
        if quantizer is not None:
            quantizer.restore_quantized_state(model, state['quantized'])
        else:
            restore_quantized_state(model, state)
    else:
        model.load_state_dict(state)
    return state


# Function to save content to disk with a SHA256 checksum appended to the filename
# Useful for saving serialized models or states with integrity verification
def save_with_checksum(content, path):
    """Save the given value on disk, along with a sha256 hash.
    Should be used with the output of either `serialize_model` or `get_state`."""
    buf = io.BytesIO()
    torch.save(content, buf)
    sig = hashlib.sha256(buf.getvalue()).hexdigest()[:8]

    path = path.parent / (path.stem + "-" + sig + path.suffix)
    path.write_bytes(buf.getvalue())


# Function to serialize a model along with its training arguments and optional quantizer
# Returns a dictionary containing class info, init args, state, and training args
def serialize_model(model, training_args, quantizer=None, half=True):
    args, kwargs = model._init_args_kwargs
    klass = model.__class__

    state = get_state(model, quantizer, half)
    return {
        'klass': klass,
        'args': args,
        'kwargs': kwargs,
        'state': state,
        'training_args': OmegaConf.to_container(training_args, resolve=True),
    }


# Function to create a deep copy of a model state dictionary
# Copies tensors to CPU and clones them to avoid shared references
def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


# Context manager to temporarily swap the state of a model
# Useful for evaluating or testing a model with a different state without permanent changes
@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state, strict=False)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


# Decorator to capture the __init__ arguments and keyword arguments of a class instance
# Stores them in _init_args_kwargs attribute for later use (e.g., serialization)
def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__
