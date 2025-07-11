import math

import julius  # Audio resampling library
from torch import nn
from .tasnet_v2 import ConvTasNet

from .utils import capture_init, center_trim


class BLSTM(nn.Module):
    """Bidirectional LSTM with linear projection.
    
    This module processes features in both forward and backward directions,
    then projects the combined outputs back to the original dimension.
    """
    def __init__(self, dim, layers=1):
        """Initialize the BLSTM module.
        
        Args:
            dim (int): Dimensionality of the input and output features
            layers (int): Number of LSTM layers to stack
        """
        super().__init__()
        # Bidirectional LSTM maintains same hidden size as input dimension
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        # Linear projection to reduce dimensionality back to input size
        self.linear = nn.Linear(2 * dim, dim)  # 2*dim because of bidirectional

    def forward(self, x):
        """Process input features with bidirectional LSTM.
        
        Args:
            x: Input tensor of shape [batch, channels, time]
            
        Returns:
            Processed tensor of same shape as input
        """
        # Permute to time-first format expected by LSTM: [time, batch, channels]
        x = x.permute(2, 0, 1)
        # Apply LSTM and take the output sequence (ignore hidden states)
        x = self.lstm(x)[0]
        # Project concatenated bidirectional outputs back to original dimension
        x = self.linear(x)
        # Restore original dimensions: [batch, channels, time]
        x = x.permute(1, 2, 0)
        return x


def rescale_conv(conv, reference):
    """Rescale convolutional layer weights to have a specific standard deviation.
    
    This helps stabilize training by ensuring consistent initial weight scales.
    
    Args:
        conv: Convolutional layer to rescale
        reference: Target standard deviation
    """
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    """Recursively rescale all convolutional layers in a module.
    
    Args:
        module: PyTorch module containing convolutional layers
        reference: Target standard deviation for weights
    """
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


def auto_load_demucs_model_v2(sources, demucs_model_name):
    """Automatically create the appropriate model based on the model name.
    
    Args:
        sources (list): List of source names to separate (e.g., ['drums', 'bass', 'vocals'])
        demucs_model_name (str): Name of the model configuration to load
    
    Returns:
        nn.Module: Initialized model (either Demucs or ConvTasNet)
    """
    # Determine number of channels based on model name
    if '48' in demucs_model_name:
        channels = 48  # Lighter model with fewer channels
    elif 'unittest' in demucs_model_name:
        channels = 4   # Minimal model for testing
    else:
        channels = 64  # Standard model size
    
    # Load appropriate architecture based on model name
    if 'tasnet' in demucs_model_name:
        # ConvTasNet is an alternative architecture (TasNet variant)
        init_demucs_model = ConvTasNet(sources, X=10)
    else:
        # Default to standard Demucs architecture
        init_demucs_model = Demucs(sources, channels=channels)
        
    return init_demucs_model


class Demucs(nn.Module):
    """Demucs model for music source separation.
    
    Demucs uses a U-Net-like architecture with an encoder/decoder structure,
    optional LSTM, and skip connections. It's designed to separate a mixture
    into individual sources (like vocals, drums, bass, etc.).
    
    Architecture overview:
    1. Optional input resampling (upsample by x2)
    2. Series of strided 1D convolutions (encoder)
    3. Optional BLSTM
    4. Series of transposed convolutions with skip connections (decoder)
    5. Optional output resampling (downsample by x2)
    """
    
    @capture_init
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
        """Initialize the Demucs model.
        
        Args:
            sources (list[str]): List of source names to separate (e.g., ['drums', 'bass', 'vocals'])
            audio_channels (int): Number of audio channels (1=mono, 2=stereo)
            channels (int): Initial number of hidden channels for the first encoder layer
            depth (int): Number of encoder/decoder layers
            rewrite (bool): Whether to add extra 1x1 convolution to encoder layers and
                            context convolution to decoder layers
            glu (bool): Whether to use Gated Linear Units instead of ReLU activations
            rescale (float): Target standard deviation for weight initialization
            resample (bool): Whether to upsample input x2 and downsample output /2
            kernel_size (int): Size of the convolutional kernels
            stride (int): Stride of the convolutions, controls downsampling/upsampling factor
            growth (float): Channel growth factor for each layer depth
            lstm_layers (int): Number of LSTM layers (0 = no LSTM)
            context (int): Kernel size for the context convolution in decoder
            normalize (bool): Whether to normalize the input by mean and standard deviation
            samplerate (int): Audio sample rate, stored as metadata
            segment_length (int): Training segment length in samples, stored as metadata
        """

        super().__init__()
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.resample = resample
        self.channels = channels
        self.normalize = normalize
        self.samplerate = samplerate
        self.segment_length = segment_length

        # Create encoder and decoder module lists
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Set up activation function - GLU or ReLU
        if glu:
            activation = nn.GLU(dim=1)  # Gated Linear Unit (provides learnable gating)
            ch_scale = 2  # GLU reduces channels by half, so we double initial channels
        else:
            activation = nn.ReLU()
            ch_scale = 1
            
        # Initial number of input channels is the number of audio channels
        in_channels = audio_channels
        
        # Build encoder and decoder layers
        for index in range(depth):
            # === Encoder layer ===
            encode = []
            # Main convolution: downsamples by stride and increases channels
            encode += [nn.Conv1d(in_channels, channels, kernel_size, stride), nn.ReLU()]
            
            # Optional 1x1 "rewrite" convolution
            if rewrite:
                encode += [nn.Conv1d(channels, ch_scale * channels, 1), activation]
                
            self.encoder.append(nn.Sequential(*encode))

            # === Decoder layer (in reverse order) ===
            decode = []
            
            # For the output layer, we need to generate all sources
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = len(self.sources) * audio_channels
                
            # Optional context convolution with larger kernel
            if rewrite:
                decode += [nn.Conv1d(channels, ch_scale * channels, context), activation]
                
            # Transposed convolution to upsample
            decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)]
            
            # Add ReLU except for the final layer
            if index > 0:
                decode.append(nn.ReLU())
                
            # Insert at the beginning to reverse the order
            self.decoder.insert(0, nn.Sequential(*decode))
            
            # Update channel dimensions for next layer
            in_channels = channels
            channels = int(growth * channels)  # Increase channels by growth factor

        # Final number of channels after all encoder layers
        channels = in_channels

        # Optional LSTM layer for temporal modeling
        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None

        # Optionally rescale initial weights
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """Calculate a valid input length that will not cause size mismatches.
        
        Due to strided convolutions and transposed convolutions, not all input
        lengths will produce outputs of the expected size. This function computes
        a valid length to avoid these issues.
        
        Args:
            length (int): Desired length
            
        Returns:
            int: Nearest valid length that will work with the model
        """
        # Account for optional input resampling
        if self.resample:
            length *= 2
            
        # Forward pass through encoder (each layer decreases length)
        for _ in range(self.depth):
            # Formula for convolutional layer output length
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            # Account for context convolution
            length += self.context - 1
            
        # Backward pass through decoder (each layer increases length)
        for _ in range(self.depth):
            # Formula for transposed convolutional layer output length
            length = (length - 1) * self.stride + self.kernel_size

        # Account for output resampling
        if self.resample:
            length = math.ceil(length / 2)
            
        return int(length)

    def forward(self, mix):
        """Forward pass of the model.
        
        Args:
            mix: Input mixture tensor of shape [batch, channels, time]
            
        Returns:
            Separated sources tensor of shape [batch, sources, channels, time]
        """
        x = mix

        # Optional normalization by mean and std of the mixture
        if self.normalize:
            # Convert to mono for computing statistics
            mono = mix.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
        else:
            mean = 0
            std = 1

        # Apply normalization
        x = (x - mean) / (1e-5 + std)  # Small epsilon for numerical stability

        # Optional upsampling of the input
        if self.resample:
            x = julius.resample_frac(x, 1, 2)  # Resample from rate 1 to rate 2

        # Encoder - save intermediate outputs for skip connections
        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)
            
        # Optional LSTM for temporal modeling
        if self.lstm:
            x = self.lstm(x)
            
        # Decoder with skip connections from encoder
        for decode in self.decoder:
            # Get matching encoder output and trim to the right size
            skip = center_trim(saved.pop(-1), x)
            # Add skip connection
            x = x + skip
            # Apply decoder layer
            x = decode(x)

        # Optional downsampling of the output
        if self.resample:
            x = julius.resample_frac(x, 2, 1)  # Resample from rate 2 to rate 1
            
        # Restore original scale
        x = x * std + mean
        
        # Reshape to separate the sources dimension: [batch, sources*channels, time] 
        #
