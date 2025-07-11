import math

import julius  # Julius library for audio resampling operations
from torch import nn
from .tasnet_v2 import ConvTasNet  # Alternative architecture for source separation
from .utils import capture_init, center_trim  # Utility functions for model initialization and tensor trimming


class BLSTM(nn.Module):
    """Bidirectional LSTM with linear projection.
    
    This module processes temporal sequences in both forward and backward directions,
    concatenates the results, and projects them back to the original dimension.
    
    Bidirectional processing allows the network to use both past and future context
    at each time step, which is crucial for audio processing where both directions
    contain valuable information.
    """
    def __init__(self, dim, layers=1):
        """Initialize the Bidirectional LSTM module.
        
        Args:
            dim (int): The input feature dimension. This will also be the output dimension
                       after projection through the linear layer.
            layers (int): Number of stacked LSTM layers. Deeper networks can model
                          more complex temporal relationships but are harder to train.
        """
        super().__init__()
        # Create bidirectional LSTM with specified number of layers
        # - bidirectional=True: Process sequence in both directions
        # - num_layers=layers: Stack multiple LSTM layers
        # - hidden_size=dim: Size of the hidden state in each direction
        # - input_size=dim: Dimensionality of input features
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        
        # Linear projection to reduce dimensionality back to original input size
        # LSTM output will be 2*dim because we concatenate outputs from both directions
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x):
        """Process sequences through the BLSTM.
        
        Args:
            x: Input tensor of shape [batch, channels, time]
            
        Returns:
            Processed tensor of same shape [batch, channels, time]
        """
        # PyTorch LSTM expects input as [time, batch, channels]
        # So we need to permute the dimensions before processing
        x = x.permute(2, 0, 1)  # [batch, channels, time] -> [time, batch, channels]
        
        # Apply LSTM and get only the output sequence (ignoring hidden states)
        # Output shape will be [time, batch, 2*channels] due to bidirectional
        x = self.lstm(x)[0]
        
        # Project back to original dimension via linear layer
        # [time, batch, 2*channels] -> [time, batch, channels]
        x = self.linear(x)
        
        # Restore original tensor shape
        # [time, batch, channels] -> [batch, channels, time]
        x = x.permute(1, 2, 0)
        return x


def rescale_conv(conv, reference):
    """Rescale weights of a convolutional layer to have a specific standard deviation.
    
    This helps stabilize the initial training phase by ensuring weights start with
    a consistent scale. Without proper scaling, some layers might have gradients that
    are too small or too large, causing training issues.
    
    Args:
        conv: A convolutional layer (nn.Conv1d or nn.ConvTranspose1d) with weights to rescale
        reference: Target standard deviation for the weights
    """
    # Compute current standard deviation of weights
    std = conv.weight.std().detach()
    
    # Calculate rescaling factor (using square root to account for variance propagation)
    scale = (std / reference)**0.5
    
    # Apply scaling factor to weights
    conv.weight.data /= scale
    
    # Also scale bias if it exists
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    """Recursively rescale weights in all convolutional layers of a module.
    
    This function walks through all components of a module and rescales
    the weights of any convolutional layers it finds.
    
    Args:
        module: A PyTorch module containing convolutional layers
        reference: Target standard deviation for the weights
    """
    # Iterate through all submodules
    for sub in module.modules():
        # Check if the submodule is a convolutional layer
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            # Rescale the weights of the convolutional layer
            rescale_conv(sub, reference)


def auto_load_demucs_model_v2(sources, demucs_model_name):
    """Automatically select and initialize the appropriate model architecture based on the model name.
    
    This factory function analyzes the model name string to determine:
    1. The architecture type (Demucs or ConvTasNet)
    2. The appropriate hyperparameters (e.g., number of channels)
    
    Args:
        sources (list): List of source names to separate (e.g., ['drums', 'bass', 'vocals', 'other'])
        demucs_model_name (str): Name of the model configuration to use
    
    Returns:
        nn.Module: Initialized model instance (either Demucs or ConvTasNet)
    """
    # Determine number of channels based on model name
    if '48' in demucs_model_name:
        channels = 48  # Smaller model variant with fewer parameters
    elif 'unittest' in demucs_model_name:
        channels = 4   # Minimal model for testing purposes
    else:
        channels = 64  # Default full-size model
    
    # Select architecture based on model name
    if 'tasnet' in demucs_model_name:
        # ConvTasNet is an alternative time-domain separation architecture
        # X=10 parameter controls the number of convolutional blocks
        init_demucs_model = ConvTasNet(sources, X=10)
    else:
        # Use standard Demucs architecture with determined channel count
        init_demucs_model = Demucs(sources, channels=channels)
        
    return init_demucs_model


class Demucs(nn.Module):
    """Demucs model for music source separation.
    
    Demucs is a waveform-to-waveform model that directly processes and outputs
    audio signals. It uses a U-Net-like architecture with:
    
    1. An encoder that progressively reduces the temporal resolution while 
       increasing the channel dimension through strided convolutions
    2. Optional BiLSTM for temporal modeling
    3. A decoder with transposed convolutions that restores the temporal resolution
       and separates the mixture into multiple sources
    4. Skip connections between encoder and decoder layers to preserve fine-grained details
    
    This architecture is especially suited for music source separation as it 
    operates directly on waveforms without requiring spectrograms or other
    time-frequency representations.
    """
    
    @capture_init  # Decorator to capture initialization arguments for later reference
    def __init__(self,
                 sources,
                 audio_channels=2,
                 channels=64,
                 depth=6,
                 rewrite=True,
                 glu=True,
                 rescale=0.1,
                 resample=True,
                 kernel_size=8,
                 stride=4,
                 growth=2.,
                 lstm_layers=2,
                 context=3,
                 normalize=False,
                 samplerate=44100,
                 segment_length=4 * 10 * 44100):
        """Initialize the Demucs model with the specified configuration.
        
        Args:
            sources (list[str]): List of source names to separate (e.g., ['drums', 'bass', 'vocals'])
            audio_channels (int): Number of audio channels (1=mono, 2=stereo)
            channels (int): Base number of channels for the first encoder layer
            depth (int): Number of encoder/decoder layers (controls model capacity and receptive field)
            rewrite (bool): Whether to add extra 1x1 convolutions after each encoder layer and
                            context-sized convolutions before each decoder layer for increased capacity
            glu (bool): Whether to use Gated Linear Units instead of ReLU for activations
            rescale (float): Target standard deviation for weight initialization (0=disable rescaling)
            resample (bool): Whether to upsample input by 2x and downsample output by 2x,
                             which can help the model capture finer temporal details
            kernel_size (int): Size of convolutional kernels (affects receptive field)
            stride (int): Stride of convolutions (controls downsampling/upsampling factor at each layer)
            growth (float): Channel growth factor between layers (e.g., 2.0 doubles channels each layer)
            lstm_layers (int): Number of stacked BiLSTM layers (0=no LSTM)
            context (int): Kernel size for the additional context convolution in decoder
            normalize (bool): Whether to normalize the input by subtracting mean and dividing by std
            samplerate (int): Audio sample rate (stored as metadata)
            segment_length (int): Training segment length in samples (stored as metadata)
        """

        super().__init__()
        
        # Store configuration parameters as instance attributes
        self.audio_channels = audio_channels  # Number of input/output audio channels
        self.sources = sources                # List of sources to separate
        self.kernel_size = kernel_size        # Convolution kernel size
        self.context = context                # Context kernel size for decoder
        self.stride = stride                  # Stride for convolutions
        self.depth = depth                    # Number of encoder/decoder layers
        self.resample = resample              # Whether to use resampling
        self.channels = channels              # Initial number of channels
        self.normalize = normalize            # Whether to normalize input
        self.samplerate = samplerate          # Audio sample rate (metadata)
        self.segment_length = segment_length  # Segment length (metadata)

        # Create lists to store encoder and decoder modules
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Set up activation function: GLU or ReLU
        if glu:
            # Gated Linear Unit provides learnable gating mechanism
            activation = nn.GLU(dim=1)  # Operates on channel dimension
            ch_scale = 2  # GLU halves the channels, so we need to double input channels
        else:
            # Standard ReLU activation
            activation = nn.ReLU()
            ch_scale = 1  # No channel reduction with ReLU
        
        # Start with audio_channels as input for first layer
        in_channels = audio_channels
        
        # Build the encoder and decoder structure
        for index in range(depth):
            # === ENCODER LAYER ===
            encode = []
            
            # Main strided convolution (downsamples by 'stride' factor)
            encode += [nn.Conv1d(in_channels, channels, kernel_size, stride), nn.ReLU()]
            
            # Optional extra 1x1 convolution to increase model capacity
            if rewrite:
                encode += [nn.Conv1d(channels, ch_scale * channels, 1), activation]
                
            # Add this encoder layer to the module list
            self.encoder.append(nn.Sequential(*encode))

            # === DECODER LAYER (built in reverse order) ===
            decode = []
            
            # Determine output channels for this decoder layer
            if index > 0:
                # For intermediate layers, output matches corresponding encoder input
                out_channels = in_channels
            else:
                # For the final layer, output needs to have channels for all sources
                out_channels = len(self.sources) * audio_channels
                
            # Optional context convolution before transposed convolution
            if rewrite:
                decode += [nn.Conv1d(channels, ch_scale * channels, context), activation]
                
            # Transposed convolution to upsample by 'stride' factor
            decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)]
            
            # Add activation except for the final layer (source outputs)
            if index > 0:
                decode.append(nn.ReLU())
                
            # Insert at beginning to create reverse order (first decoder = last encoder)
            self.decoder.insert(0, nn.Sequential(*decode))
            
            # Update channel counts for next layer
            in_channels = channels  # Current layer's output becomes next layer's input
            channels = int(growth * channels)  # Increase channels by growth factor

        # Store final encoder output channels for LSTM
        channels = in_channels

        # Optional BiLSTM for temporal modeling
        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None

        # Optionally rescale initial weights for better training dynamics
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """Calculate a valid input length that will not cause size mismatches in the model.
        
        Due to strided convolutions and transposed convolutions, not all input
        lengths will produce outputs of the expected size. This function determines
        the nearest valid length to ensure consistent behavior.
        
        Args:
            length (int): Desired input length
            
        Returns:
            int: Nearest valid length that will work with the model's architecture
        """
        # Account for input resampling (doubles length if enabled)
        if self.resample:
            length *= 2
            
        # Forward pass through encoder layers
        for _ in range(self.depth):
            # Formula for convolutional layer output length with stride
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)  # Ensure length is at least 1
            
            # Account for context convolution in decoder
            length += self.context - 1
            
        # Backward pass through decoder layers
        for _ in range(self.depth):
            # Formula for transposed convolutional layer output length
            length = (length - 1) * self.stride + self.kernel_size

        # Account for output resampling (halves length if enabled)
        if self.resample:
            length = math.ceil(length / 2)
            
        return int(length)

    def forward(self, mix):
        """Forward pass of the Demucs model.
        
        This function processes an audio mixture to separate it into
        individual sources (e.g., vocals, drums, bass, etc.).
        
        Args:
            mix: Input audio mixture tensor of shape [batch, channels, time]
            
        Returns:
            Separated sources tensor of shape [batch, sources, channels, time]
        """
        # Store original input
        x = mix

        # Optional normalization to center and scale the input
        if self.normalize:
            # Convert to mono by averaging channels for computing statistics
            mono = mix.mean(dim=1, keepdim=True)
            
            # Calculate mean and standard deviation along time dimension
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
        else:
            # No normalization
            mean = 0
            std = 1

        # Apply normalization (if enabled)
        x = (x - mean) / (1e-5 + std)  # Small epsilon prevents division by zero

        # Optional upsampling of the input by factor of 2
        if self.resample:
            x = julius.resample_frac(x, 1, 2)  # Resample from rate 1 to rate 2

        # Encoder forward pass with skip connection storage
        saved = []  # Store encoder outputs for skip connections
        for encode in self.encoder:
            x = encode(x)  # Apply encoder layer
            saved.append(x)  # Save output for corresponding decoder layer
            
        # Optional BiLSTM temporal modeling
        if self.lstm:
            x = self.lstm(x)
            
        # Decoder forward pass with skip connections
        for decode in self.decoder:
            # Get corresponding encoder output for skip connection
            # Use center_trim to ensure compatible dimensions
            skip = center_trim(saved.pop(-1), x)
            
            # Add skip connection (residual connection)
            x = x + skip
            
            # Apply decoder layer
            x = decode(x)

        # Optional downsampling of the output by factor of 2
        if self.resample:
            x = julius.resample_frac(x, 2, 1)  # Resample from rate 2 to rate 1
            
        # Restore original scale (undo normalization)
        x = x * std + mean
        
        # Reshape to separate sources dimension
        # [batch, sources*channels, time] -> [batch, sources, channels, time]
        x = x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))
        
        return x
