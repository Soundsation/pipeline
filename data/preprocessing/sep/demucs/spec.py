import torch as th

# Function to compute the spectrogram of a signal tensor
# x: input tensor with shape (..., length)
# n_fft: number of FFT points
# hop_length: hop length between frames, default is n_fft // 4
# pad: padding factor for FFT size
# Returns: complex spectrogram tensor with shape (..., freqs, frames)
def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape  # unpack all dimensions except the last (length)
    x = x.reshape(-1, length)  # flatten all leading dimensions
    
    device_type = x.device.type
    is_other_gpu = not device_type in ["cuda", "cpu"]  # check if device is not cuda or cpu

    if is_other_gpu:
        x = x.cpu()  # move tensor to CPU if on unsupported device
    z = th.stft(x,
                n_fft * (1 + pad),  # FFT size with padding
                hop_length or n_fft // 4,  # hop length default
                window=th.hann_window(n_fft).to(x),  # Hann window
                win_length=n_fft,
                normalized=True,
                center=True,
                return_complex=True,
                pad_mode='reflect')
    _, freqs, frame = z.shape  # get frequency bins and frame count
    return z.view(*other, freqs, frame)  # reshape back to original leading dims


# Function to compute the inverse spectrogram (reconstruct signal from spectrogram)
# z: complex spectrogram tensor with shape (..., freqs, frames)
# hop_length: hop length between frames
# length: length of the output signal
# pad: padding factor used in spectrogram
# Returns: reconstructed signal tensor with shape (..., length)
def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape  # unpack all dims except last two
    n_fft = 2 * freqs - 2  # calculate FFT size from frequency bins
    z = z.view(-1, freqs, frames)  # flatten leading dims
    win_length = n_fft // (1 + pad)  # window length
    
    device_type = z.device.type
    is_other_gpu = not device_type in ["cuda", "cpu"]  # check device
    
    if is_other_gpu:
        z = z.cpu()  # move to CPU if unsupported device
    x = th.istft(z,
                 n_fft,
                 hop_length,
                 window=th.hann_window(win_length).to(z.real),
                 win_length=win_length,
                 normalized=True,
                 length=length,
                 center=True)
    _, length = x.shape  # get output length
    return x.view(*other, length)  # reshape to original leading dims
