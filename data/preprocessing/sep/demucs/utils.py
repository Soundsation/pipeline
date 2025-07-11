from collections import defaultdict
from contextlib import contextmanager
import math
import os
import tempfile
import typing as tp

import errno
import functools
import hashlib
import inspect
import io
import os
import random
import socket
import tempfile
import warnings
import zlib
# import tkinter as tk - commented out, likely not needed in current implementation

# Deep learning quantization libraries for model compression
from diffq import UniformQuantizer, DiffQuantizer
import torch as th
import tqdm
from torch import distributed
from torch.nn import functional as F

import torch

def unfold(a, kernel_size, stride):
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.

    This will pad the input so that `F = ceil(T / K)`.

    This function implements a sliding window operation on the last dimension of a tensor,
    similar to the one-dimensional variant of torch.nn.Unfold. It's useful for processing
    sequential data such as audio in overlapping chunks.
    
    Args:
        a: Input tensor of shape [*OT, T] where T is the sequence length
        kernel_size: Size of each sliding window
        stride: Step size between windows
        
    Returns:
        Unfolded tensor of shape [*OT, F, K] where F is the number of frames
        
    see https://github.com/pytorch/pytorch/issues/60466
    """
    *shape, length = a.shape  # Extract the sequence length and other dimensions
    n_frames = math.ceil(length / stride)  # Calculate number of frames
    tgt_length = (n_frames - 1) * stride + kernel_size  # Calculate required padded length
    
    # Pad the input tensor to ensure we can extract the required number of frames
    a = F.pad(a, (0, tgt_length - length))
    
    # Calculate the memory strides for the output tensor
    strides = list(a.stride())
    assert strides[-1] == 1, 'data should be contiguous'  # Ensure data is contiguous in memory
    strides = strides[:-1] + [stride, 1]  # Create new strides for the output tensor
    
    # Use as_strided to create a view without copying data
    return a.as_strided([*shape, n_frames, kernel_size], strides)


def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    
    This function trims a tensor symmetrically around its center to match the size of
    a reference tensor (or specified length). This is often used in audio processing
    to ensure outputs match the expected size after processing.
    
    Args:
        tensor: Tensor to be trimmed
        reference: Either a tensor whose last dimension size will be matched, or an integer length
        
    Returns:
        Center-trimmed tensor with last dimension matching reference
        
    Note:
        If the size difference is odd, the extra sample is removed from the right side.
    """
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    
    # Calculate how much to trim
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    
    # Perform the trimming if needed
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


def pull_metric(history: tp.List[dict], name: str):
    """
    Extract a specific metric from a list of metric dictionaries.
    
    This function navigates through nested dictionaries to extract values
    associated with the specified metric name.
    
    Args:
        history: List of metric dictionaries from training history
        name: Metric name to extract, can use dot notation for nested dicts (e.g., 'train.loss')
    
    Returns:
        List of values for the specified metric across all history entries
    """
    out = []
    for metrics in history:
        metric = metrics
        # Navigate through nested dictionary using the dot notation
        for part in name.split("."):
            metric = metric[part]
        out.append(metric)
    return out


def EMA(beta: float = 1):
    """
    Exponential Moving Average callback.
    
    This function creates a stateful callback that can be used to update
    an exponential moving average of metrics over time. This is useful for
    smoothing metrics during training to reduce noise.
    
    Args:
        beta: Smoothing factor (1 = simple average, close to 0 = recent values weighted higher)
    
    Returns:
        A function that can be called repeatedly to update and return the EMA
        
    Note:
        For `beta=1`, this is equivalent to a simple average.
    """
    # Initialize dictionaries to store the weighted sum and normalization factor
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: dict, weight: float = 1) -> dict:
        """
        Update the EMA with new metrics.
        
        Args:
            metrics: Dictionary of new metric values to incorporate
            weight: Weight to apply to the new metrics (default: 1)
            
        Returns:
            Updated EMA for all metrics
        """
        nonlocal total, fix
        for key, value in metrics.items():
            # Update the weighted sum and normalization factor
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        
        # Return the normalized values
        return {key: tot / fix[key] for key, tot in total.items()}
    
    return _update


def sizeof_fmt(num: float, suffix: str = 'B'):
    """
    Format bytes into a human-readable size representation.
    
    Converts a byte size into a human-readable string with appropriate unit prefix
    (KB, MB, GB, etc.)
    
    Args:
        num: Size in bytes
        suffix: Unit suffix (default: 'B' for bytes)
        
    Returns:
        String representation of the size with appropriate unit
        
    Example:
        sizeof_fmt(1024) returns "1.0KiB"
        
    Taken from https://stackoverflow.com/a/1094933
    """
    # Iterate through binary size units
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    # For extremely large values
    return "%.1f%s%s" % (num, 'Yi', suffix)


@contextmanager
def temp_filenames(count: int, delete=True):
    """
    Context manager that provides temporary filenames.
    
    This is useful when you need temporary files that should be automatically
    cleaned up after use.
    
    Args:
        count: Number of temporary filenames to generate
        delete: Whether to delete the files when exiting the context (default: True)
        
    Yields:
        List of temporary filenames
        
    Example:
        with temp_filenames(2) as names:
            # use names[0] and names[1] as temporary file paths
        # files are automatically deleted when exiting the with block
    """
    names = []
    try:
        # Create the requested number of temporary files
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        # Clean up by deleting the files if requested
        if delete:
            for name in names:
                os.unlink(name)

def average_metric(metric, count=1.):
    """
    Average a metric across all distributed processes.
    
    This function takes a local metric value and its weight (count), and
    computes the weighted average across all distributed processes. This
    is useful for distributed training to aggregate metrics from all nodes.
    
    Args:
        metric: The metric value to average
        count: Weight for this value (default: 1.0), typically number of examples
        
    Returns:
        Weighted average of the metric across all processes
    """
    # Create a tensor with count and weighted metric
    metric = th.tensor([count, count * metric], dtype=th.float32, device='cuda')
    
    # Sum across all processes
    distributed.all_reduce(metric, op=distributed.ReduceOp.SUM)
    
    # Return the weighted average
    return metric[1].item() / metric[0].item()


def free_port(host='', low=20000, high=40000):
    """
    Find and return a free network port.
    
    This function attempts to find an available port in the specified range
    by trying to bind to random ports until it succeeds.
    
    Args:
        host: Host address to bind to (default: '' which means all interfaces)
        low: Lower bound of the port range (default: 20000)
        high: Upper bound of the port range (default: 40000)
        
    Returns:
        A port number that is likely to be free
        
    Note:
        There is a potential race condition if another process binds to the
        port between when this function checks it and when you use it.
    """
    sock = socket.socket()
    while True:
        port = random.randint(low, high)
        try:
            # Try to bind to the randomly selected port
            sock.bind((host, port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                # Port is in use, try another one
                continue
            raise
        # Port is available
        return port


# Note: This is a duplicate of the earlier sizeof_fmt function
# Keeping for backward compatibility, but should be consolidated
def sizeof_fmt(num, suffix='B'):
    """
    Given `num` bytes, return human readable size.
    Taken from https://stackoverflow.com/a/1094933
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def human_seconds(seconds, display='.2f'):
    """
    Convert seconds to a human-readable duration string.
    
    This function takes a duration in seconds and returns a formatted string
    with appropriate time units (us, ms, s, min, hrs, days).
    
    Args:
        seconds: Time in seconds to format
        display: Format string for the numeric part (default: '.2f')
        
    Returns:
        Formatted duration string with appropriate units
        
    Example:
        human_seconds(0.001) returns "1.00 ms"
        human_seconds(3600) returns "1.00 hrs"
    """
    # Start with microseconds (seconds * 1e6)
    value = seconds * 1e6
    ratios = [1e3, 1e3, 60, 60, 24]  # Conversion factors between units
    names = ['us', 'ms', 's', 'min', 'hrs', 'days']  # Unit names
    
    # Start with the smallest unit
    last = names.pop(0)
    
    # Convert to larger units as needed
    for name, ratio in zip(names, ratios):
        if value / ratio < 0.3:  # Stop when conversion would be less than 0.3
            break
        value /= ratio
        last = name
    
    # Format and return the result
    return f"{format(value, display)} {last}"


class TensorChunk:
    """
    Represents a chunk/slice of a tensor without copying the data.
    
    This class allows working with a portion of a tensor as if it were
    a separate tensor, while avoiding memory copies. It's useful for
    processing large tensors in smaller chunks.
    
    Attributes:
        tensor: The original complete tensor
        offset: Starting position of the chunk in the last dimension
        length: Length of the chunk in the last dimension
        device: Device where the tensor is stored
    """
    
    def __init__(self, tensor, offset=0, length=None):
        """
        Initialize a TensorChunk.
        
        Args:
            tensor: The source tensor to create a chunk from
            offset: Starting position in the last dimension (default: 0)
            length: Length of the chunk (default: remaining length after offset)
        """
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        self.tensor = tensor
        self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        """
        Get the shape of the chunk.
        
        Returns:
            Shape of the chunk (same as original tensor but with modified last dimension)
        """
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        """
        Create a padded version of the chunk to match the target length.
        
        This method centers the chunk and adds padding on both sides to
        reach the target length.
        
        Args:
            target_length: Desired length of the output tensor
            
        Returns:
            Padded tensor of the specified length
        """
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        # Calculate start and end positions with padding
        start = self.offset - delta // 2
        end = start + target_length

        # Correct for boundaries of the original tensor
        correct_start = max(0, start)
        correct_end = min(total_length, end)

        # Calculate actual padding needed
        pad_left = correct_start - start
        pad_right = end - correct_end

        # Extract and pad the chunk
        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    """
    Convert a tensor or TensorChunk to a TensorChunk.
    
    This helper function ensures the input is a TensorChunk, converting
    a regular tensor if necessary.
    
    Args:
        tensor_or_chunk: Either a torch.Tensor or a TensorChunk
        
    Returns:
        A TensorChunk representing the input
    """
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, th.Tensor)
        return TensorChunk(tensor_or_chunk)


def apply_model_v1(model, mix, shifts=None, split=False, progress=False, set_progress_bar=None):
    """
    Apply model to a given mixture - Version 1 implementation.
    
    This function applies a source separation model to an audio mixture,
    with options for time shifting (for time equivariance) and processing
    in chunks (for memory efficiency).
    
    Args:
        model: The source separation model to apply
        mix: The input mixture tensor [channels, length]
        shifts: If > 0, will shift the input in time randomly and average predictions
        split: If True, process the input in chunks (useful for memory constraints)
        progress: If True, show a progress bar when processing chunks
        set_progress_bar: Optional callback function to update an external progress bar
        
    Returns:
        Separated sources as a tensor
    """

    channels, length = mix.size()
    device = mix.device
    progress_value = 0
    
    if split:
        # Process in chunks to save memory
        out = th.zeros(4, channels, length, device=device)  # Pre-allocate output tensor
        shift = model.samplerate * 10  # 10-second chunks
        offsets = range(0, length, shift)
        scale = 10
        
        # Set up progress tracking if requested
        if progress:
            offsets = tqdm.tqdm(offsets, unit_scale=scale, ncols=120, unit='seconds')
            
        # Process each chunk
        for offset in offsets:
            chunk = mix[..., offset:offset + shift]
            
            # Process the chunk, updating progress if needed
            if set_progress_bar:
                progress_value += 1
                set_progress_bar(0.1, (0.8/len(offsets)*progress_value))
                chunk_out = apply_model_v1(model, chunk, shifts=shifts, set_progress_bar=set_progress_bar)
            else:
                chunk_out = apply_model_v1(model, chunk, shifts=shifts)
                
            # Add the processed chunk to the output tensor
            out[..., offset:offset + shift] = chunk_out
            offset += shift
        return out
    elif shifts:
        # Apply time shifts for better time equivariance
        max_shift = int(model.samplerate / 2)  # Maximum shift of 0.5 seconds
        mix = F.pad(mix, (max_shift, max_shift))  # Pad to allow for shifting
        
        # Generate random shifts
        offsets = list(range(max_shift))
        random.shuffle(offsets)
        out = 0
        
        # Process the input with different shifts and average results
        for offset in offsets[:shifts]:
            shifted = mix[..., offset:offset + length + max_shift]
            
            if set_progress_bar:
                shifted_out = apply_model_v1(model, shifted, set_progress_bar=set_progress_bar)
            else:
                shifted_out = apply_model_v1(model, shifted)
                
            # Correct for the shift in the output
            out += shifted_out[..., max_shift - offset:max_shift - offset + length]
            
        # Average the results
        out /= shifts
        return out
    else:
        # Simple case: direct model application
        # Pad the input to a valid length for the model
        valid_length = model.valid_length(length)
        delta = valid_length - length
        padded = F.pad(mix, (delta // 2, delta - delta // 2))
        
        # Apply the model
        with th.no_grad():
            out = model(padded.unsqueeze(0))[0]
            
        # Trim the output to match the input length
        return center_trim(out, mix)

def apply_model_v2(model, mix, shifts=None, split=False,
                overlap=0.25, transition_power=1., progress=False, set_progress_bar=None): 
    """
    Apply model to a given mixture - Version 2 implementation with improved chunking.
    
    This version adds overlap between chunks and smooth transitions using a
    weighted average approach, producing better results at chunk boundaries.
    
    Args:
        model: The source separation model to apply
        mix: The input mixture tensor [channels, length]
        shifts: If > 0, will shift the input in time randomly and average predictions
        split: If True, process the input in chunks with overlap
        overlap: Fraction of overlap between adjacent chunks (default: 0.25 = 25%)
        transition_power: Power to apply to the transition weights (higher = sharper transitions)
        progress: If True, show a progress bar when processing chunks
        set_progress_bar: Optional callback function to update an external progress bar
        
    Returns:
        Separated sources as a tensor
    """
    
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    device = mix.device
    channels, length = mix.shape
    progress_value = 0
    
    if split:
        # Process in overlapping chunks with weighted averaging
        out = th.zeros(len(model.sources), channels, length, device=device)
        sum_weight = th.zeros(length, device=device)  # Track weights for normalization
        
        segment = model.segment_length
        stride = int((1 - overlap) * segment)  # Calculate stride based on overlap
        offsets = range(0, length, stride)
        scale = stride / model.samplerate
        
        # Set up progress tracking if requested
        if progress:
            offsets = tqdm.tqdm(offsets, unit_scale=scale, ncols=120, unit='seconds')
            
        # Create a weight function for smooth transitions between chunks
        # Triangle shape with maximum weight in the middle of the segment
        weight = th.cat([th.arange(1, segment // 2 + 1),
                         th.arange(segment - segment // 2, 0, -1)]).to(device)
        assert len(weight) == segment
        
        # Normalize and apply transition power
        weight = (weight / weight.max())**transition_power
        
        # Process each chunk
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment)
            
            # Process the chunk, updating progress if needed
            if set_progress_bar:
                progress_value += 1
                set_progress_bar(0.1, (0.8/len(offsets)*progress_value))
                chunk_out = apply_model_v2(model, chunk, shifts=shifts, set_progress_bar=set_progress_bar)
            else:
                chunk_out = apply_model_v2(model, chunk, shifts=shifts)
                
            # Apply weighted average for smooth transitions
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment] += weight[:chunk_length] * chunk_out
            sum_weight[offset:offset + segment] += weight[:chunk_length]
            offset += segment
            
        # Ensure all positions have been weighted
        assert sum_weight.min() > 0
        
        # Normalize by the weights
        out /= sum_weight
        return out
    elif shifts:
        # Apply time shifts for better time equivariance
        max_shift = int(0.5 * model.samplerate)  # Maximum shift of 0.5 seconds
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0
        
        # Process the input with different shifts and average results
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            
            if set_progress_bar:
                progress_value += 1
                shifted_out = apply_model_v2(model, shifted, set_progress_bar=set_progress_bar)
            else:
                shifted_out = apply_model_v2(model, shifted)
                
            # Add the processed output, correcting for the shift
            out += shifted_out[..., max_shift - offset:]
            
        # Average the results
        out /= shifts
        return out
    else:
        # Simple case: direct model application
        valid_length = model.valid_length(length)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(valid_length)
        
        # Apply the model
        with th.no_grad():
            out = model(padded_mix.unsqueeze(0))[0]
            
        # Trim the output to match the input length
        return center_trim(out, length)


# Note: This is a duplicate of the earlier temp_filenames function
# Should probably be consolidated
@contextmanager
def temp_filenames(count, delete=True):
    """
    Context manager that provides temporary filenames.
    
    Args:
        count: Number of temporary filenames to generate
        delete: Whether to delete the files when exiting the context
        
    Yields:
        List of temporary filenames
    """
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)


def get_quantizer(model, args, optimizer=None):
    """
    Create and configure a quantizer for model compression.
    
    This function sets up a quantizer based on command-line arguments,
    which can be used to compress the model weights for efficient storage
    or deployment.
    
    Args:
        model: The model to quantize
        args: Command-line arguments containing quantization settings
        optimizer: Optional optimizer to configure for differentiable quantization
        
    Returns:
        Configured quantizer or None if quantization is not requested
    """
    quantizer = None
    
    # DiffQ (differentiable quantization)
    if args.diffq:
        quantizer = DiffQuantizer(
            model, min_size=args.q_min_size, group_size=8)
        if optimizer is not None:
            # Configure optimizer for quantization-aware training
            quantizer.setup_optimizer(optimizer)
    
    # QAT (quantization-aware training)
    elif args.qat:
        quantizer = UniformQuantizer(
                model, bits=args.qat, min_size=args.q_min_size)
                
    return quantizer


def load_model(path, strict=False):
    """
    Load a model from a saved checkpoint.
    
    This function restores a model from a saved state, handling
    compatibility issues and quantization if needed.
    
    Args:
        path: Path to the saved model file
        strict: If True, require exact parameter match; if False, allow dropping parameters
        
    Returns:
        The loaded model
    """
    # Suppress irrelevant warnings during loading
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_from = path
        package = th.load(load_from, 'cpu')  # Load on CPU to avoid OOM on GPU

    # Extract model information
    klass = package["klass"]  # Model class
    args = package["args"]    # Positional arguments for initialization
    kwargs = package["kwargs"]  # Keyword arguments for initialization

    if strict:
        # Create model with exact parameters
        model = klass(*args, **kwargs)
    else:
        # Handle backward compatibility by dropping unknown parameters
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    # Extract model state and training arguments
    state = package["state"]
    training_args = package["training_args"]
    
    # Set up quantizer if needed
    quantizer = get_quantizer(model, training_args)

    # Restore model state
    set_state(model, quantizer, state)
    return model


def get_state(model, quantizer):
    """
    Get the current state of a model, considering quantization if applied.
    
    This function extracts model parameters and handles potential
    quantization for efficient storage.
    
    Args:
        model: The model to get state from
        quantizer: Optional quantizer that may be applied to the state
        
    Returns:
        Dictionary containing the model state
    """
    if quantizer is None:
        # Regular state dict - move to CPU to avoid OOM issues
        state = {k: p.data.to('cpu') for k, p in model.state_dict().items()}
    else:
        # Get quantized parameters and compress them
        state = quantizer.get_quantized_state()
        buf = io.BytesIO()
        th.save(state, buf)
        state = {'compressed': zlib.compress(buf.getvalue())}
    return state


def set_state(model, quantizer, state):
    """
    Set the state of a model, handling quantization if needed.
    
    This function restores model parameters from a saved state,
    including decompression and dequantization if applicable.
    
    Args:
        model: The model to restore state to
        quantizer: Optional quantizer to use for restoring quantized weights
        state: The state dictionary to restore from
        
    Returns:
        The restored state
    """
    if quantizer is None:
        # Direct loading of state dict
        model.load_state_dict(state)
    else:
        # Decompress and restore quantized state
        buf = io.BytesIO(zlib.decompress(state["compressed"]))
        state = th.load(buf, "cpu")
        quantizer.restore_quantized_state(state)

    return state


def save_state(state, path):
    """
    Save a model state with a signature in the filename.
    
    This function saves a model state and adds a hash signature to the
    filename to identify the specific version.
    
    Args:
        state: Model state to save
        path: Path object where the state should be saved
        
    Note:
        The final filename will include a hash signature for versioning
    """
    # Serialize the state
    buf = io.BytesIO()
    th.save(state, buf)
    
    # Generate a short hash signature for the state
    sig = hashlib.sha256(buf.getvalue()).hexdigest()[:8]

    # Create filename with signature
    path = path.parent / (path.stem + "-" + sig + path.suffix)
    
    # Write the serialized state to the file
    path.write_bytes(buf.getvalue())


def save_model(model, quantizer, training_args, path):
    """
    Save a complete model including class information and state.
    
    This function saves everything needed to recreate the model:
    - Class definition
    - Initialization arguments
    - Model parameters
    - Training arguments
    
    Args:
        model: The model to save
        quantizer: Optional quantizer used with the model
        training_args: Arguments used during training
        path: Path where the model should be saved
    """
    # Get initialization information
    args, kwargs = model._init_args_kwargs
    klass = model.__class__

    # Get model state (potentially quantized)
    state = get_state(model, quantizer)

    # Create the complete package
    save_to = path
    package = {
        'klass': klass,
        'args': args,
        'kwargs': kwargs,
        'state': state,
        'training_args': training_args,
    }
    
    # Save the package
    th.save(package, save_to)


def capture_init(init):
    """
    Decorator to capture initialization arguments for later model saving.
    
    This decorator wraps a class's __init__ method to store the arguments
    used to initialize it, which allows later recreation of the instance.
    
    Args:
        init: The original __init__ method to wrap
        
    Returns:
        Wrapped __init__ method that captures its arguments
    """
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        # Store the initialization arguments
        self._init_args_kwargs = (args, kwargs)
        # Call the original init
        init(self, *args, **kwargs)

    return __init__

class DummyPoolExecutor:
    """
    A simple replacement for concurrent.futures.ProcessPoolExecutor or ThreadPoolExecutor.
    
    This class provides the same interface as the standard library executors
    but executes tasks synchronously in the current thread. This is useful
    for debugging or when parallel execution is problematic.
    """
    
    class DummyResult:
        """
        A simple replacement for the Future objects returned by executors.
        
        This class mimics the result() method of Future but executes the
        function immediately when result() is called.
        """
        def __init__(self, func, *args, **kwargs):
            """
            Initialize a DummyResult with a function and its arguments.
            
            Args:
                func: The function to execute
                *args: Positional arguments for the function
                **kwargs: Keyword arguments for the function
            """
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            """
            Execute the function and return its result.
            
            Unlike real Future objects which would execute asynchronously and
            wait for completion when result() is called, this executes the
            function immediately when result() is called.
            
            Returns:
                The result of calling the function with the stored arguments
            """
            return self.func(*self.args, **self.kwargs)

    def __init__(self, workers=0):
        """
        Initialize a DummyPoolExecutor.
        
        Args:
            workers: Number of workers (ignored, included for compatibility)
        """
        pass

    def submit(self, func, *args, **kwargs):
        """
        Submit a function for execution.
        
        Instead of executing asynchronously, this creates a DummyResult object
        that will execute the function when its result() method is called.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            A DummyResult that will execute the function on demand
        """
        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)

    def __enter__(self):
        """
        Enter the context manager.
        
        Returns:
            Self, to be used as the context manager
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        Exit the context manager.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_value: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
            
        Returns:
            False, to propagate any exceptions
        """
        return