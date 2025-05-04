# Audio Processing Pipeline (SEP) and Voice Activity Detection (VAD)

## Overview

Soundsation is a comprehensive pipeline for audio processing, featuring high-quality source separation, voice activity detection (VAD), and automatic speech recognition (ASR). The system implements state-of-the-art techniques for isolating and processing audio components from mixed signals.

## Key Components

- **Source Separation**: Advanced ensemble-based approach using Demucs architecture with three models for minimal signal interference
- **Voice Activity Detection**: Identifies speech segments within audio recordings
- **Automatic Speech Recognition**: Converts detected speech to text

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Sufficient disk space for model checkpoints
- Hugging Face account (for downloading pretrained models)

## Setup Instructions

### 1. Source Separation Setup

```bash
# Create and activate the environment
conda create -n soundsation python=3.10
conda activate soundsation

# Navigate to the source separation directory
cd sep
pip install -r requirements.txt

# Login to Hugging Face
huggingface-cli login

# Download separation model checkpoints
python download_sep_ckpt.py
```

### 2. VAD and ASR Setup

```bash
# Ensure you're in the soundsation environment
conda activate soundsation

# Navigate to the ASR directory
cd asr

# Login to Hugging Face (if not already logged in)
huggingface-cli login

# Download ASR model checkpoints
python download_asr_ckpt.py
```

## Usage

### Source Separation

The separation module uses an ensemble of three models to extract clean audio sources from mixed audio.

```bash
# Running on a single node
cd sep
bash run.multigpu.sh mtgjamendo false 0 1 8 2
```

Parameters explained:

- `mtgjamendo`: Dataset name
- `false`: Flag for specific processing mode
- `0 1`: GPU IDs to use
- `8`: Batch size
- `2`: Number of workers per GPU

For multi-node processing using Slurm:

```bash
sbatch run.slurm.seperate.sh
```

### Voice Activity Detection and ASR

The VAD and ASR pipeline identifies speech segments and transcribes them.

```bash
cd asr
bash run_vad_asr.sh
```

For detailed parameter options, check the script contents:

```bash
cat run_vad_asr.sh
```

## Technical Details

### Source Separation

The separation module is based on the Demucs architecture, which combines:

- Time-domain convolutional networks
- LSTM layers for temporal modeling
- Hybrid time-frequency domain processing
- Wiener filtering for post-processing

The ensemble approach combines predictions from multiple models for more robust source separation.

### VAD and ASR Pipeline

The voice activity detection system identifies speech regions in audio before passing them to the automatic speech recognition module, improving transcription accuracy and efficiency.

## Output Files

After processing, you can expect the following outputs:

- Separated audio tracks (for source separation)
- Detected speech segments (for VAD)
- Text transcriptions (for ASR)

## Troubleshooting

- **CUDA out of memory errors**: Try reducing batch size or processing audio in smaller segments
- **Missing model files**: Ensure Hugging Face authentication succeeded and download scripts completed properly
- **Processing errors**: Check that input audio matches expected format (sampling rate, channels)
