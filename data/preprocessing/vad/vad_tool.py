"""
Audio Segmentation Utility for Voice Activity Detection (VAD)

This module provides utilities for:
1. Analyzing audio files to detect voice activity
2. Segmenting audio based on voice activity
3. Generating timestamps and cut points for audio processing
4. Creating silence segments for padding audio clips
"""
import collections
import sys
import librosa
import numpy
import random


# Mean value for Gaussian distribution in random frame generation - impacts segment length
MU = 1800  


def read_wave_to_frames(path, sr=16000, frame_duration=10):
    """
    Load an audio file and convert it to frames with specified sample rate.
    
    Args:
        path: Path to the audio file
        sr: Sample rate to use when loading the file (default: 16000Hz)
        frame_duration: Frame duration in milliseconds (default: 10ms)
        
    Returns:
        Tuple of (frames, raw_waveform)
    """
    # Load audio file with librosa and convert to mono with specified sample rate
    wav, orig_sr = librosa.load(path, sr=sr, mono=True, res_type='polyphase')
    
    # Convert floating point audio to 16-bit PCM format
    wav = (wav * 2**15).astype(numpy.int16)
    
    # Convert numpy array to bytes for frame processing
    wav_bytes = wav.tobytes()
    
    # Generate frames from the audio bytes
    frames = frame_generator(frame_duration, wav_bytes, sr)
    
    return frames, wav


class Frame(object):
    """
    Represents a fixed-duration frame of audio data.
    
    Attributes:
        bytes: Raw audio data bytes
        timestamp: Start time of the frame in seconds
        duration: Duration of the frame in seconds
    """
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes        # Raw audio data
        self.timestamp = timestamp  # Start time in seconds
        self.duration = duration    # Duration in seconds


def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    Generates audio frames from PCM audio data.
    
    Args:
        frame_duration_ms: Frame duration in milliseconds
        audio: Raw PCM audio data as bytes
        sample_rate: Audio sample rate in Hz
        
    Yields:
        Frame objects of the requested duration
    """
    # Calculate bytes per frame (2 bytes per sample in 16-bit audio)
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    
    offset = 0
    timestamp = 0.0
    # Calculate actual frame duration in seconds
    duration = (float(n) / sample_rate) / 2.0
    
    # Generate frames until we run out of audio data
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_generator(frames, sr, vad):
    """
    Apply voice activity detection to audio frames.
    
    Args:
        frames: List of Frame objects
        sr: Sample rate of the audio
        vad: Voice Activity Detector object
        
    Returns:
        List of boolean values indicating speech/non-speech for each frame
    """
    vad_info = []
    for frame in frames:
        # Apply VAD to each frame and store result (True=speech, False=silence)
        vad_info.append(vad.is_speech(frame.bytes, sr))
    return vad_info


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """
    Filters out non-voiced audio frames using a padded sliding window algorithm.
    
    Args:
        sample_rate: Audio sample rate in Hz
        frame_duration_ms: Frame duration in milliseconds
        padding_duration_ms: Amount of padding to add around speech segments
        vad: Voice Activity Detector object
        frames: List of Frame objects
        
    Yields:
        Bytes containing continuous segments of detected speech with padding
    """
    # Calculate number of padding frames based on desired padding duration
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    
    # Use deque as a sliding window/ring buffer with fixed max length
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    
    # Start in non-triggered state (not detecting speech)
    triggered = False

    # Storage for frames identified as speech
    voiced_frames = []
    
    for frame in frames:
        # Check if current frame contains speech
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # Visual output for debugging (1=speech, 0=silence)
        sys.stdout.write('1' if is_speech else '0')
        
        if not triggered:
            # In non-triggered state, add frames to ring buffer
            ring_buffer.append((frame, is_speech))
            
            # Count number of speech frames in buffer
            num_voiced = len([f for f, speech in ring_buffer if speech])
            
            # Enter triggered state if >90% of frames in buffer contain speech
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # Mark start of speech segment with timestamp
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                
                # Add all frames from buffer to voiced_frames
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # In triggered state, collect all frames as speech
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            
            # Count number of non-speech frames in buffer
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            
            # Exit triggered state if >90% of frames in buffer are non-speech
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # Mark end of speech segment with timestamp
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                
                # Yield all collected speech frames as one continuous byte segment
                yield b''.join([f.bytes for f in voiced_frames])
                
                # Reset buffers for next segment
                ring_buffer.clear()
                voiced_frames = []
                
    # Handle case where audio ends while still in triggered state
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    
    # Yield any remaining speech frames
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


class ActivateInfo:
    """
    Stores information about active (speech) or inactive (silence) audio segments.
    
    Attributes:
        active: Boolean indicating if segment contains speech
        duration: Duration in frames
        start_pos: Start position in frames
        end_pos: End position in frames
        keep: Flag indicating if segment should be kept
    """
    def __init__(self, active, duration, start_pos, end_pos, keep=True):
        self.active = active      # True=speech, False=silence
        self.duration = duration   # Duration in frames
        self.start_pos = start_pos  # Start position in frames
        self.end_pos = end_pos      # End position in frames
        self.keep = keep           # Flag to keep or discard segment
    
    def __add__(self, x):
        """Addition operator overload to support duration calculations"""
        return x + self.duration
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"{self.active} {self.start_pos}, {self.end_pos}"


class SegmentInfo:
    """
    Represents audio segment information with type and positioning details.
    
    Attributes:
        type: Segment type ('raw' for actual audio, 'pad' for silence padding)
        duration: Duration in frames
        start_pos: Start position in frames
        end_pos: End position in frames
        frame_duration: Duration of each frame in milliseconds
    """
    def __init__(self, type="raw", duration=0, start_pos=0, end_pos=0, frame_duration=10):
        self.type = type          # 'raw' or 'pad'
        self.duration = duration   # Duration in frames
        self.start_pos = start_pos  # Start position in frames
        self.end_pos = end_pos      # End position in frames
        self.frame_duration = frame_duration  # Frame duration in ms

    def get_wav_seg(self, wav: numpy.array, sr: int, frame_duration: int=None):
        """
        Extract segment from audio array based on position.
        
        Args:
            wav: Raw audio waveform as numpy array
            sr: Sample rate in Hz
            frame_duration: Optional override for frame duration
            
        Returns:
            Numpy array containing the audio segment or zeros for padding
        """
        # Use provided frame duration or default
        fd = frame_duration if frame_duration is not None else self.frame_duration
        
        # Calculate samples per frame
        sample_pre_frame = fd*sr/1000
        
        if self.type == "pad":
            # For padding segments, return zeros
            return numpy.zeros((int(sample_pre_frame*self.duration), ), dtype=numpy.int16)
        
        # For raw segments, extract portion from original waveform
        return wav[int(self.start_pos*sample_pre_frame):int((self.end_pos*sample_pre_frame))]

    def __repr__(self) -> str:
        """String representation showing time range or padding duration"""
        if self.type == "raw":
            # For raw segments, show start:end time in ms
            text = f"{self.start_pos*self.frame_duration}:{self.end_pos*self.frame_duration}"
        else:
            # For padding segments, show duration in ms in brackets
            text = f"[{self.duration*self.frame_duration}]"
        return text


def get_sil_segments(active_info: ActivateInfo, sil_frame: int, attach_pos: str="mid") -> list:
    """
    Generate silence segments based on active info and attachment position.
    
    Args:
        active_info: ActivateInfo object describing segment
        sil_frame: Duration of silence to generate in frames
        attach_pos: Where to attach silence ('head', 'tail', or 'mid')
        
    Returns:
        List of SegmentInfo objects representing silence
    """
    # If segment is longer than requested silence duration
    if active_info.duration >= sil_frame:
        if attach_pos == "tail":
            # Attach silence at beginning of segment
            seg = [SegmentInfo(start_pos=active_info.start_pos, end_pos=active_info.start_pos+sil_frame)]
        elif attach_pos == "head":
            # Attach silence at end of segment
            seg = [SegmentInfo(start_pos=active_info.end_pos-sil_frame, end_pos=active_info.end_pos)]
        elif attach_pos == "mid":
            # Split silence between beginning and end of segment
            seg = [
                SegmentInfo(start_pos=active_info.start_pos, end_pos=active_info.start_pos+sil_frame // 2-1),
                SegmentInfo(start_pos=active_info.end_pos-sil_frame // 2+1, end_pos=active_info.end_pos),
            ]
        else:
            raise NotImplementedError
    else:
        # If segment is shorter than requested silence, use what's available and pad if needed
        if attach_pos == "tail":
            seg = [
                SegmentInfo(start_pos=active_info.start_pos, end_pos=active_info.end_pos),
                # Add padding to reach desired silence duration
                SegmentInfo(type="pad", duration=sil_frame-active_info.duration),
            ]
        elif attach_pos == "head":
            seg = [
                # Add padding to reach desired silence duration
                SegmentInfo(type="pad", duration=sil_frame-active_info.duration),
                SegmentInfo(start_pos=active_info.start_pos, end_pos=active_info.end_pos),
            ]
        elif attach_pos == "mid":
            # Just use segment as is if attaching to middle
            seg = [
                SegmentInfo(start_pos=active_info.start_pos, end_pos=active_info.end_pos),
            ]
        else:
            raise NotImplementedError
    return seg


def merge_segment(segment: list) -> list:
    """
    Optimize segments by merging adjacent ones for efficiency.
    
    Args:
        segment: List of SegmentInfo objects
        
    Returns:
        List of merged SegmentInfo objects
    """
    new_segment = []
    last_s = None
    
    for s in segment:
        s: SegmentInfo
        
        # Handle padding segments separately (can't be merged)
        if s.type == "pad":
            if last_s is not None:
                new_segment.append(last_s)
                last_s = None
            new_segment.append(s)
            continue
            
        if last_s is None:
            # First segment or after padding
            last_s = s
        else:
            # If current segment starts right after previous one, merge them
            if last_s.end_pos+1 == s.start_pos:
                last_s.end_pos = s.end_pos
            else:
                # Otherwise add previous segment and start tracking current one
                new_segment.append(last_s)
                last_s = s
                
    # Add final segment if there is one
    if last_s is not None:
        new_segment.append(last_s)
        
    return new_segment


def random_frame(min_frame, max_frame):
    """
    Generate random frame length using Gaussian distribution.
    
    The distribution is centered around MU (1800) and scaled so that
    the lower tail is approximately at min_frame.
    
    Args:
        min_frame: Minimum frame length
        max_frame: Maximum frame length
        
    Returns:
        Random integer frame length between min_frame and max_frame
    """
    mu = MU  # Center of distribution
    
    # Set sigma so that min_frame is approximately 3 standard deviations below mean
    sigma = (mu - min_frame) / 3
    
    # Generate random value from Gaussian distribution
    length = random.gauss(mu, sigma)
    
    # Clamp value to allowed range
    length = int(min(max(length, min_frame), max_frame))
    
    return length


def cut_points_generator(
        vad_info, 
        min_active_frame=20, 
        sil_frame=50, 
        sil_mid_frame=100, 
        cut_min_frame=8 * 100, 
        cut_max_frame=20 * 100, 
        is_random_min_frame=False,
    ):
    """
    Generate cut points for audio segmentation based on VAD info.
    
    This function analyzes voice activity detection data and creates optimal
    segments for audio processing with appropriate silence padding.
    
    Args:
        vad_info: List of boolean values indicating speech/non-speech
        min_active_frame: Minimum frames for speech activity (default: 20)
        sil_frame: Silence padding frames at segment edges (default: 50)
        sil_mid_frame: Maximum silence frames within segments (default: 100)
        cut_min_frame: Minimum total frames per segment (default: 800)
        cut_max_frame: Maximum total frames per segment (default: 2000)
        is_random_min_frame: Whether to randomize min segment length (default: False)
        
    Returns:
        List of segment information for audio cutting
    """
    # Initialize current minimum frame length
    curr_min_frame = cut_min_frame
    
    # Clean up very short speech segments (likely noise)
    last_active_frame = 0
    is_last_active = False
    for i, is_curr_active in enumerate(vad_info):
        if is_curr_active and not is_last_active:
            # Start of potential speech segment
            last_active_frame = i
        elif not is_curr_active and is_last_active and i - last_active_frame <= min_active_frame:
            # If speech segment was too short, mark as non-speech (noise)
            for j in range(last_active_frame, i):
                vad_info[j] = False
        is_last_active = is_curr_active

    # Build activation info by grouping consecutive frames with same activity
    start_pos = 0
    end_pos = 0  # This will be updated in the loop
    duration = 0
    is_active = vad_info[0]
    activate_info = []
    
    for pos, vi in enumerate(vad_info):
        if is_active == vi:
            # Continue current segment
            duration += 1
            end_pos = pos  # Update end position as we go
        else:
            # End of segment, save info
            activate_info.append(ActivateInfo(is_active, duration, start_pos, pos-1))
            
            # Start new segment
            is_active = vi
            start_pos = pos
            duration = 1
            
    # Add final segment
    activate_info.append(ActivateInfo(is_active, duration, start_pos, end_pos))
    
    # Create segments based on activation info
    segment_info = []
    curr_segment = []
    curr_segment_duration = 0
    max_active_block = len(activate_info)
    
    # Process each activation block
    for i in range(max_active_block):
        curr_ai = activate_info[i]
        
        if curr_ai.active:  # Processing speech segment
            # If starting a new segment
            if curr_segment_duration == 0:
                if i == 0:
                    # Add padding silence at beginning of file
                    curr_segment.append(SegmentInfo("pad", sil_frame))
                else:
                    # Use part of preceding silence block as padding
                    sil_seg = activate_info[i-1]
                    raw_sil_duration = min(sil_frame, sil_seg.duration // 2)
                    end_pos = sil_seg.end_pos
                    curr_segment = get_sil_segments(
                        ActivateInfo(
                            True, 
                            duration=raw_sil_duration,
                            start_pos=sil_seg.end_pos-raw_sil_duration, 
                            end_pos=sil_seg.end_pos
                        ),
                        sil_frame=sil_frame,
                        attach_pos="head"
                    )
                curr_segment_duration += sil_frame
                
            # Calculate new duration if we add current speech segment
            next_duration = curr_segment_duration + curr_ai.duration
            curr_ai_seg = SegmentInfo(start_pos=curr_ai.start_pos, end_pos=curr_ai.end_pos)
            
            # Check if adding this segment would exceed maximum length
            if next_duration > cut_max_frame:
                # Segment would be too long
                if curr_ai.duration > curr_segment_duration:
                    # Current speech is longer than accumulated - start new segment with it
                    new_segment = get_sil_segments(activate_info[i-1], sil_frame, "head")
                    new_segment.append(curr_ai_seg)
                    
                    # Add trailing silence
                    if i < max_active_block - 1:
                        new_segment.extend(get_sil_segments(activate_info[i+1], sil_frame, "tail"))
                    else:
                        new_segment.append(SegmentInfo(type="pad", duration=sil_frame))
                        
                    segment_info.append(merge_segment(new_segment))
                    
                    # Possibly randomize next segment length
                    if is_random_min_frame:
                        curr_min_frame = random_frame(cut_min_frame, cut_max_frame)
                        
                    # Reset segment tracking
                    curr_segment = []
                    curr_segment_duration = 0
                else:
                    # Add current segment if it's substantial
                    if curr_segment_duration > 10 * 100:  # > 1 second
                        segment_info.append(merge_segment(curr_segment))
                        if is_random_min_frame:
                            curr_min_frame = random_frame(cut_min_frame, cut_max_frame)
                            
                    # Start new segment with current speech
                    curr_segment = get_sil_segments(activate_info[i-1], sil_frame, "head")
                    curr_segment.append(curr_ai_seg)
                    curr_segment_duration = sil_frame + curr_ai.duration
            elif next_duration > curr_min_frame:
                # Segment is long enough to finalize
                curr_segment.append(curr_ai_seg)
                
                # Add trailing silence
                if i < max_active_block - 1:
                    curr_segment.extend(get_sil_segments(activate_info[i+1], sil_frame, "tail"))
                else:
                    curr_segment.append(SegmentInfo(type="pad", duration=sil_frame))
                    
                # Add completed segment to results
                segment_info.append(merge_segment(curr_segment))
                
                # Possibly randomize next segment length
                if is_random_min_frame:
                    curr_min_frame = random_frame(cut_min_frame, cut_max_frame)
                    
                # Reset segment tracking
                curr_segment = []
                curr_segment_duration = 0
            else:
                # Add speech but continue building segment
                curr_segment.append(curr_ai_seg)
                curr_segment_duration += curr_ai.duration
        else:  # Processing silence segment
            if curr_segment_duration == 0:
                # Starting a new segment with silence
                raw_sil_duration = min(sil_frame, curr_ai.duration // 2)
                end_pos = curr_ai.end_pos
                curr_segment = get_sil_segments(
                    ActivateInfo(
                        True, 
                        duration=raw_sil_duration,
                        start_pos=curr_ai.end_pos-raw_sil_duration, 
                        end_pos=curr_ai.end_pos
                    ),
                    sil_frame=sil_frame,
                    attach_pos="head"
                )
                curr_segment_duration += sil_frame 
            else:
                # Adding silence to existing segment
                if curr_ai.duration > sil_mid_frame:
                    # Long silence - end current segment and start new one
                    curr_segment.extend(get_sil_segments(curr_ai, sil_frame, "tail"))
                    segment_info.append(merge_segment(curr_segment))
                    
                    if is_random_min_frame:
                        curr_min_frame = random_frame(cut_min_frame, cut_max_frame)
                        
                    curr_segment = []
                    curr_segment_duration = 0
                else:
                    # Short silence - include in current segment
                    curr_segment.extend(get_sil_segments(curr_ai, sil_mid_frame+1, attach_pos="mid"))
                    curr_segment_duration += min(sil_mid_frame, curr_ai.duration)
                    
    # Handle any remaining segment
    if len(curr_segment) > 3 and curr_segment_duration > 7 * 100:  # > 0.7 seconds and substantial
        if activate_info[-1].active:
            # Add trailing silence if needed
            curr_segment.append(SegmentInfo(type="pad", duration=sil_frame))
        segment_info.append(merge_segment(curr_segment))
        
    return segment_info


def ms_to_mm_ss_xx(milliseconds):
    """
    Convert milliseconds to formatted time string [MM:SS.XX].
    
    Args:
        milliseconds: Time in milliseconds
        
    Returns:
        Formatted string in [MM:SS.XX] format
    """
    minutes = milliseconds // 60000
    seconds = (milliseconds % 60000) // 1000
    milliseconds_part = (milliseconds % 1000) // 10
    
    return f"[{minutes:02d}:{seconds:02d}.{milliseconds_part:02d}]"


def process_timestamp(timestamp, cut_min_ms):
    """
    Process timestamp and filter by minimum duration.
    
    Args:
        timestamp: Timestamp string in format "start:end" (milliseconds)
        cut_min_ms: Minimum duration in milliseconds to keep
        
    Returns:
        Formatted timestamp string or None if duration is too short
    """
    start, end = map(int, timestamp.split(":"))
    duration = end - start
    
    # Only return timestamps for segments meeting minimum duration
    if duration >= cut_min_ms:
        return ms_to_mm_ss_xx(start)
    else:
        return None


def cut_points_storage_generator(raw_vad_info, cut_points: list, cut_min_ms=600) -> tuple:
    """
    Generate VAD content string and timestamp text from cut points.
    
    Args:
        raw_vad_info: List of boolean VAD results
        cut_points: List of segment cut points
        cut_min_ms: Minimum segment duration in milliseconds (default: 600ms)
        
    Returns:
        Tuple of (raw_vad_content, timestamps_text)
    """
    # Create string representation of VAD info (1=speech, 0=silence)
    raw_vad_content = " ".join(["1" if i else "0" for i in raw_vad_info])
    
    # Process cut points to generate timestamps
    content = []
    for cut_point in cut_points:
        time_ranges = []
        for s in cut_point:
            s_str = str(s)
            # Only process actual timestamps (not padding indicators)
            if not s_str.startswith('['):
                result = process_timestamp(s_str, cut_min_ms)
                if result is not None:
                    time_ranges.append(result)
        
        # Add first timestamp from each cut point
        if time_ranges:
            content.append(time_ranges[0]) 
            
    return raw_vad_content, "\n".join(content)


def wavs_generator(raw_wav: numpy.array, cut_points: list, filename: str, sr: int, frame_duration: int) -> list:
    """
    Generate wave files from the segmented audio.
    
    Args:
        raw_wav: Raw audio waveform as numpy array
        cut_points: List of segment cut points
        filename: Base filename for generated WAV files
        sr: Sample rate in Hz
        frame_duration: Frame duration in milliseconds
        
    Returns:
        List of tuples (audio_data, filename) for each segment
    """
    wavs = []
    
    for idx, cp in enumerate(cut_points):
        # Concatenate all segments (including padding) for this cut point
        clip = numpy.concatenate(
            [s.get_wav_seg(raw_wav, sr, frame_duration) for s in cp],
            axis=0
        )
        
        # Generate filename with index and duration information
        output_filename = f"{filename}_{idx}_{int(clip.shape[0]/sr*1000)}.wav"
        
        wavs.append((clip, output_filename))
        
    return wavs
