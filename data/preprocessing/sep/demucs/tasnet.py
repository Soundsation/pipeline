import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import capture_init

# Small constant to prevent division by zero
EPS = 1e-8


def overlap_and_add(signal, frame_step):
    """
    Reconstructs a signal from a framed representation using overlap-add method.
    
    Args:
        signal: Framed signal with shape (*, frames, frame_length)
        frame_step: Step size between adjacent frames
        
    Returns:
        Reconstructed signal with proper overlapping
    """
    # Get dimensions
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    # Find greatest common divisor for efficient computation
    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    
    # Calculate output size
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    # Reshape signal for overlap-add operation
    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    # Create frame indices for proper positioning
    frame = torch.arange(0, output_subframes,
                      device=signal.device).unfold(0, subframes_per_frame, subframe_step)
    frame = frame.long()  # Ensure indices are long type for indexing
    frame = frame.contiguous().view(-1)

    # Initialize result tensor and add subframes at their positions
    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    
    # Reshape to final output
    result = result.view(*outer_dimensions, -1)
    return result


class ConvTasNet(nn.Module):
    """
    Conv-TasNet: A fully-convolutional time-domain audio separation network.
    
    This network consists of three components:
    1. Encoder: Converts time-domain signals to feature representation
    2. Separator: Estimates masks for each source using temporal convolution
    3. Decoder: Converts masked features back to time-domain signals
    """
    
    @capture_init
    def __init__(self,
                 N=256,           # Number of filters in autoencoder
                 L=20,            # Length of the filters (in samples)
                 B=256,           # Number of channels in bottleneck 1×1-conv
                 H=512,           # Number of channels in convolutional blocks
                 P=3,             # Kernel size in convolutional blocks
                 X=8,             # Number of blocks in each repeat
                 R=4,             # Number of repeats
                 C=4,             # Number of sources to separate
                 audio_channels=1, # Number of audio channels (1=mono, 2=stereo)
                 samplerate=44100, # Sample rate of the audio
                 norm_type="gLN", # Type of normalization (gLN, cLN, BN)
                 causal=False,    # Whether the network is causal
                 mask_nonlinear='relu'): # Non-linear function for masks
        """
        Initialize the Conv-TasNet model with specified parameters.
        """
        super(ConvTasNet, self).__init__()
        
        # Store hyperparameters
        self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = N, L, B, H, P, X, R, C
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.audio_channels = audio_channels
        self.samplerate = samplerate
        
        # Build network components
        self.encoder = Encoder(L, N, audio_channels)
        self.separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear)
        self.decoder = Decoder(N, L, audio_channels)
        
        # Initialize weights using Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def valid_length(self, length):
        """Returns valid length of output (handles padding/stride effects)"""
        return length

    def forward(self, mixture):
        """
        Forward pass of the Conv-TasNet model.
        
        Args:
            mixture: Input mixture signal [batch_size, audio_channels, time_samples]
            
        Returns:
            est_source: Estimated sources [batch_size, num_sources, audio_channels, time_samples]
        """
        # Convert to feature representation
        mixture_w = self.encoder(mixture)
        
        # Estimate masks for each source
        est_mask = self.separator(mixture_w)
        
        # Apply masks and decode to time-domain
        est_source = self.decoder(mixture_w, est_mask)

        # Handle potential length mismatch due to convolution
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source


class Encoder(nn.Module):
    """
    Encoder that converts time-domain signals to feature representations.
    
    Uses 1D convolution with ReLU activation to transform waveforms into
    a higher-dimensional feature space.
    """
    
    def __init__(self, L, N, audio_channels):
        """
        Initialize the encoder.
        
        Args:
            L: Length of filters (in samples)
            N: Number of filters
            audio_channels: Number of audio channels (1=mono, 2=stereo)
        """
        super(Encoder, self).__init__()
        
        # Store parameters
        self.L, self.N = L, N
        
        # Create 1D convolution with 50% overlap between frames
        # Input: [batch, audio_channels, time]
        # Output: [batch, N, feature_time]
        self.conv1d_U = nn.Conv1d(audio_channels, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Forward pass of encoder.
        
        Args:
            mixture: Input signal [batch, audio_channels, time]
            
        Returns:
            mixture_w: Feature representation [batch, N, feature_time]
                where feature_time = (time-L)/(L/2)+1 = 2*time/L-1
        """
        # Apply convolution with ReLU activation
        mixture_w = F.relu(self.conv1d_U(mixture))
        return mixture_w


class Decoder(nn.Module):
    """
    Decoder that converts feature representations back to time-domain signals.
    
    Uses basis transformation followed by overlap-and-add to reconstruct waveforms.
    """
    
    def __init__(self, N, L, audio_channels):
        """
        Initialize the decoder.
        
        Args:
            N: Number of filters in autoencoder
            L: Length of filters (in samples)
            audio_channels: Number of audio channels (1=mono, 2=stereo)
        """
        super(Decoder, self).__init__()
        
        # Store parameters
        self.N, self.L = N, L
        self.audio_channels = audio_channels
        
        # Create linear layer for basis transformation
        # Maps each feature to a time-domain segment
        self.basis_signals = nn.Linear(N, audio_channels * L, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Forward pass of decoder.
        
        Args:
            mixture_w: Feature representation [batch, N, feature_time]
            est_mask: Estimated masks [batch, sources, N, feature_time]
            
        Returns:
            est_source: Estimated sources [batch, sources, audio_channels, time]
        """
        # Apply masks to features (D = W * M)
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [batch, sources, N, feature_time]
        source_w = torch.transpose(source_w, 2, 3)  # [batch, sources, feature_time, N]
        
        # Convert to time-domain segments (S = DV)
        est_source = self.basis_signals(source_w)  # [batch, sources, feature_time, audio_channels * L]
        
        # Reshape and prepare for overlap-add
        m, c, k, _ = est_source.size()
        est_source = est_source.view(m, c, k, self.audio_channels, -1).transpose(2, 3).contiguous()
        
        # Reconstruct full waveform using overlap-add
        est_source = overlap_and_add(est_source, self.L // 2)
        return est_source


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network that estimates masks for source separation.
    
    Uses a series of dilated convolutional blocks to model long-term dependencies.
    """
    
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear='relu'):
        """
        Initialize the Temporal Convolutional Network.
        
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1×1-conv
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of blocks in each repeat
            R: Number of repeats
            C: Number of sources
            norm_type: Type of normalization (gLN, cLN, BN)
            causal: Whether the network is causal
            mask_nonlinear: Non-linear function for masks
        """
        super(TemporalConvNet, self).__init__()
        
        # Store parameters
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        
        # Network components
        # Layer normalization along channel dimension
        layer_norm = ChannelwiseLayerNorm(N)
        
        # Bottleneck convolution to reduce channels
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        
        # Create stacked dilated convolutional blocks
        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                # Exponentially increasing dilation
                dilation = 2**x
                
                # Calculate appropriate padding
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                
                # Add temporal block
                blocks += [
                    TemporalBlock(B,
                                 H,
                                 P,
                                 stride=1,
                                 padding=padding,
                                 dilation=dilation,
                                 norm_type=norm_type,
                                 causal=causal)
                ]
            repeats += [nn.Sequential(*blocks)]
            
        # Create temporal convolutional network
        temporal_conv_net = nn.Sequential(*repeats)
        
        # Final 1x1 convolution for mask generation
        mask_conv1x1 = nn.Conv1d(B, C * N, 1, bias=False)
        
        # Combine all components
        self.network = nn.Sequential(layer_norm, bottleneck_conv1x1, temporal_conv_net,
                                   mask_conv1x1)

    def forward(self, mixture_w):
        """
        Forward pass of separator.
        
        Args:
            mixture_w: Feature representation [batch, N, feature_time]
            
        Returns:
            est_mask: Estimated masks [batch, sources, N, feature_time]
        """
        M, N, K = mixture_w.size()
        
        # Generate mask scores
        score = self.network(mixture_w)  # [batch, C*N, feature_time]
        
        # Reshape to get individual source masks
        score = score.view(M, self.C, N, K)  # [batch, sources, N, feature_time]
        
        # Apply non-linearity to generate final masks
        if self.mask_nonlinear == 'softmax':
            # Softmax ensures masks sum to 1 across sources
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            # ReLU for non-negative masks
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
            
        return est_mask


class TemporalBlock(nn.Module):
    """
    Temporal Block with residual connection.
    
    Consists of a 1x1 convolution, PReLU, normalization,
    and a depthwise separable convolution.
    """
    
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                norm_type="gLN",
                causal=False):
        """
        Initialize a Temporal Block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Convolution stride
            padding: Padding size
            dilation: Dilation factor
            norm_type: Normalization type
            causal: Whether block is causal
        """
        super(TemporalBlock, self).__init__()
        
        # 1x1 convolution to change channel dimension
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
        # PReLU activation (learnable parameter for negative slope)
        prelu = nn.PReLU()
        
        # Normalization layer
        norm = chose_norm(norm_type, out_channels)
        
        # Depthwise separable convolution
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size, stride, padding,
                                      dilation, norm_type, causal)
        
        # Combine components
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor [batch, channels, feature_time]
            
        Returns:
            Output with residual connection added
        """
        # Store input for residual connection
        residual = x
        
        # Pass through network
        out = self.net(x)
        
        # Add residual connection (no ReLU after addition works better)
        return out + residual


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution.
    
    Consists of depthwise convolution (one filter per input channel)
    followed by pointwise convolution (1x1 conv to change channel count).
    Reduces parameters compared to standard convolution.
    """
    
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                norm_type="gLN",
                causal=False):
        """
        Initialize Depthwise Separable Convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Convolution stride
            padding: Padding size
            dilation: Dilation factor
            norm_type: Normalization type
            causal: Whether convolution is causal
        """
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution: one filter per input channel
        # groups=in_channels ensures depthwise operation
        depthwise_conv = nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=in_channels,
                                 bias=False)
        
        # For causal convolution, remove future information
        if causal:
            chomp = Chomp1d(padding)
            
        # PReLU activation
        prelu = nn.PReLU()
        
        # Normalization layer
        norm = chose_norm(norm_type, in_channels)
        
        # Pointwise convolution: 1x1 conv to change channel count
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
        # Combine components based on causality
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm, pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        """
        Forward pass of Depthwise Separable Convolution.
        
        Args:
            x: Input tensor [batch, in_channels, feature_time]
            
        Returns:
            Output tensor [batch, out_channels, feature_time]
        """
        return self.net(x)


class Chomp1d(nn.Module):
    """
    Removes padding from the end of the sequence for causal convolution.
    
    Ensures no information leakage from future timesteps.
    """
    
    def __init__(self, chomp_size):
        """
        Initialize the Chomp1d module.
        
        Args:
            chomp_size: Number of timesteps to remove from the end
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Remove padding from the end of the sequence.
        
        Args:
            x: Input tensor [batch, channels, time+padding]
            
        Returns:
            Output with padding removed [batch, channels, time]
        """
        # Remove chomp_size elements from the end
        return x[:, :, :-self.chomp_size].contiguous()


def chose_norm(norm_type, channel_size):
    """
    Factory function to create appropriate normalization layer.
    
    Args:
        norm_type: Type of normalization (gLN, cLN, BN, id)
        channel_size: Number of channels
        
    Returns:
        Normalization layer instance
    """
    if norm_type == "gLN":
        # Global Layer Normalization: normalize across all timesteps
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        # Channel-wise Layer Normalization: normalize each channel independently
        return ChannelwiseLayerNorm(channel_size)
    elif norm_type == "id":
        # Identity: no normalization
        return nn.Identity()
    else:  # norm_type == "BN":
        # Batch Normalization: normalize across batch and timesteps
        return nn.BatchNorm1d(channel_size)


class ChannelwiseLayerNorm(nn.Module):
    """
    Channel-wise Layer Normalization (cLN).
    
    Normalizes each channel independently across time dimension.
    """
    
    def __init__(self, channel_size):
        """
        Initialize Channel-wise Layer Normalization.
        
        Args:
            channel_size: Number of channels
        """
        super(ChannelwiseLayerNorm, self).__init__()
        
        # Learnable parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, channels, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, channels, 1]
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize gamma to 1 and beta to 0"""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Forward pass of Channel-wise Layer Normalization.
        
        Args:
            y: Input tensor [batch, channels, time]
            
        Returns:
            Normalized tensor [batch, channels, time]
        """
        # Calculate mean and variance along time dimension
        mean = torch.mean(y, dim=1, keepdim=True)  # [batch, 1, time]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [batch, 1, time]
        
        # Normalize, scale, and shift
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """
    Global Layer Normalization (gLN).
    
    Normalizes each channel across all time steps for each sample.
    """
    
    def __init__(self, channel_size):
        """
        Initialize Global Layer Normalization.
        
        Args:
            channel_size: Number of channels
        """
        super(GlobalLayerNorm, self).__init__()
        
        # Learnable parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, channels, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, channels, 1]
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize gamma to 1 and beta to 0"""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Forward pass of Global Layer Normalization.
        
        Args:
            y: Input tensor [batch, channels, time]
            
        Returns:
            Normalized tensor [batch, channels, time]
        """
        # Calculate mean across both channel and time dimensions
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [batch, 1, 1]
        
        # Calculate variance across both channel and time dimensions
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        
        # Normalize, scale, and shift
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


# Test code for when script is run directly
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(123)
    
    # Define test parameters
    M, N, L, T = 2, 3, 4, 12  # Batch size, filters, filter length, time samples
    K = 2 * T // L - 1        # Time features after encoding
    B, H, P, X, R, C = 2, 3, 3, 3, 2, 2  # Network architecture parameters
    norm_type, causal = "gLN", False      # Normalization and causality settings
    
    # Create random input mixture
    mixture = torch.randint(3, (M, T))
    
    # Test Encoder
    encoder = Encoder(L, N)
    encoder.conv1d_U.weight.data = torch.randint(2, encoder.conv1d_U.weight.size())
    mixture_w = encoder(mixture)
    print('mixture', mixture)
    print('U', encoder.conv1d_U.weight)
    print('mixture_w', mixture_w)
    print('mixture_w size', mixture_w.size())

    # Test Separator
    separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type=norm_type, causal=causal)
    est_mask = separator(mixture_w)
    print('est_mask', est_mask)

    # Test Decoder
    decoder = Decoder(N, L)
    est_mask = torch.randint(2, (B, K, C, N))
    est_source = decoder(mixture_w, est_mask)
    print('est_source', est_source)

    # Test full Conv-TasNet model
    conv_tasnet = ConvTasNet(N, L, B, H, P, X, R, C, norm_type=norm_type)
    est_source = conv_tasnet(mixture)
    print('est_source', est_source)
    print('est_source size', est_source.size())
