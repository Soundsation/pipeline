# Final Year Individual Dissertation

## Overview

This thesis introduces Soundsation, a novel diffusion-based framework for high-fidelity 
long-form text-to-music and lyrics-to-song synthesis that addresses fundamental challenges in generative music modeling. 
Despite significant advances in generative AI, existing music generation systems struggle 
with maintaining rhythmic coherence, structural organization, 
and alignment between textual prompts and musical attributes, particularly over extended durations. 
These limitations severely constrain the practical utility and creative potential of current systems.
we overcome these limitations through a rigorous latent diffusion approach operating in a carefully designed representation space. 
The architecture comprises two key components: a Variational Autoencoder (VAE) that compresses high-dimensional audio into compact latent representations, 
and a Diffusion Transformer that generates songs through iterative denoising. 
This design enables the generation of full-length stereo musical compositions (up to 5m 10s) at 44.1kHz 
sampling rate while maintaining high musicality and lyrical intelligibility.
The framework introduces several innovative elements, including: 

(1) a hierarchical VAE representation scheme that captures musical information at multiple temporal resolutions, 
(2) a novel sentence-level alignment mechanism that addresses the unique challenges of discontinuous vocal content in songs, 
(3) an LSTM-based style processing component that enhances global coherence, and 
(4) a comprehensive three-stream preprocessing pipeline that handles lyrics, style conditioning, and waveform processing simultaneously. 

These innovations collectively enable precise alignment between textual descriptions and generated musical attributes.

## Key Features

- Text-to-music generation with LRC (lyric) file support
- Multiple conditioning options (text prompts or reference audio)
- Multi-language text processing and segmentation
- Advanced audio source separation capabilities
- Configurable inference parameters for customized output
- Support for various music styles and attributes

## Repository Structure (Tree Visualization)

```
soundsation-pipeline/
├── LangSegment/
│   ├── utils/
│   │   └── num.py (8.5 KB)
│   └── LangSegment.py (53.3 KB)
├── config/
│   ├── accelerate_config.yaml (359 B)
│   ├── config.json (210 B)
│   └── default.ini (1.3 KB)
├── data/
│   └── preprocessing/
│       ├── sep/
│       │   ├── demucs/
│       │   │   ├── __main__.py (9.5 KB)
│       │   │   ├── apply.py (11.0 KB)
│       │   │   ├── demucs.py (17.3 KB)
│       │   │   ├── filtering.py (13.6 KB)
│       │   │   ├── hdemucs.py (30.4 KB)
│       │   │   ├── model.py (7.3 KB)
│       │   │   ├── model_v2.py (7.2 KB)
│       │   │   ├── pretrained.py (5.4 KB)
│       │   │   ├── repo.py (4.4 KB)
│       │   │   ├── spec.py (1.3 KB)
│       │   │   ├── states.py (4.2 KB)
│       │   │   ├── tasnet.py (14.5 KB)
│       │   │   ├── tasnet_v2.py (14.7 KB)
│       │   │   ├── transformer.py (27.2 KB)
│       │   │   └── utils.py (16.4 KB)
│       │   ├── README.txt (0 B)
│       │   ├── __version__.py (132 B)
│       │   ├── download_sep_ckpt.py (217 B)
│       │   ├── main.py (61.1 KB)
│       │   ├── requirements.txt (724 B)
│       │   ├── run_sep.sh (3.0 KB)
│       │   └── separate.py (56.2 KB)
│       ├── vad/
│       │   ├── run_vad.sh (366 B)
│       │   ├── vad_tool.py (15.9 KB)
│       │   └── vad_webrtcvad.py (4.4 KB)
│       ├── .gitignore (158 B)
│       ├── README.txt (891 B)
│       └── run_vad.sh (513 B)
├── dataset/
│   ├── latent/
│   │   ├── 2626046476.pt (1.2 MB)
│   │   ├── 28528706.pt (2.0 MB)
│   │   ├── 547512549.pt (1017.4 KB)
│   │   └── 549162241.pt (915.9 KB)
│   ├── lrc/
│   │   ├── 2626046476.pt (10.4 KB)
│   │   ├── 28528706.pt (5.1 KB)
│   │   ├── 547512549.pt (10.7 KB)
│   │   └── 549162241.pt (7.8 KB)
│   ├── style/
│   │   ├── 2626046476.pt (3.2 KB)
│   │   ├── 28528706.pt (3.2 KB)
│   │   ├── 547512549.pt (3.2 KB)
│   │   └── 549162241.pt (3.2 KB)
│   ├── dataset.py (6.1 KB)
│   └── train.scp (360 B)
├── inference/
│   ├── sampling/
│   │   └── vocal.npy (2.1 KB)
│   ├── infer.py (4.4 KB)
│   └── infer_utils.py (8.9 KB)
├── model/
│   ├── cfm.py (9.4 KB)
│   ├── dit.py (6.7 KB)
│   ├── modules.py (19.8 KB)
│   ├── trainer.py (11.9 KB)
│   └── utils.py (6.3 KB)
├── scripts/
│   ├── infer_prompt_ref.bat (216 B)
│   ├── infer_prompt_ref.sh (476 B)
│   ├── infer_wav_ref.bat (203 B)
│   ├── infer_wav_ref.sh (462 B)
│   └── train.sh (534 B)
├── train/
│   └── train.py (1.7 KB)
├── .gitignore (31 B)
├── README.txt (438 B)
└── requirements.txt (186 B)
```

## Getting Started

### Prerequisites

- Python 3.9.12
- PyTorch 2.1.0
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Soundsation/pipeline.git
cd soundsation-pipeline
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### Inference with Text Prompt

```bash
python infer/infer.py --lrc-path infer/example/eg_en.lrc --ref-prompt "classical genres, hopeful mood, piano." --audio-length 95 --repo_id Soundsation/base --output-dir infer/example/output_en --chunked
```

#### Inference with Audio Reference

```bash
python infer/infer.py --lrc-path infer/example/eg_en.lrc --ref-audio-path infer/example/eg_cn.wav --audio-length 95 --repo_id Soundsation/base --output-dir infer/example/output_en --chunked
```

## Technical Approach

Soundsation employs a two-stage architecture similar to approaches like Melodist, 
separating the music generation process into distinct components:

1. **Text Understanding**: Transforms text descriptions into meaningful conditioning signals
2. **Audio Synthesis**: Generates musical output with appropriate harmony, instrumentation, and mood

The system leverages techniques from diffusion models to progressively denoise 
audio representations and create coherent musical structures that align with the input prompts.

## Data Processing

Soundsation relies on sophisticated data preprocessing to ensure high-quality output. The pipeline includes modules for audio source separation, language segmentation, and feature extraction.

## Model Configuration

The base model architecture is defined in `config/config.json`:

```json
{
  "model_type": "soundsation",
  "model": {
    "dim": 2048,
    "depth": 16,
    "heads": 32,
    "ff_mult": 4,
    "text_dim": 512,
    "conv_layers": 4,
    "mel_dim": 64,
    "text_num_embeds": 363
  }
}
```

Additional parameters can be customized in `config/default.ini`.

## License

This code is provided for research and educational purposes. Please see the LICENSE.txt file for more details.

---

# Data Preprocessing

## Overview

The data preprocessing pipeline for Soundsation includes sophisticated tools for audio source separation (SEP),
voice activity detection (VAD), automatic speech recognition (ASR), and multi-language text processing. 
These components ensure high-quality training data and enable effective inference.

## Audio Source Separation (SEP)

The separation module uses an ensemble approach similar to established audio diffusion models to extract clean audio sources from mixed audio.

### Setup

```bash
conda create -n soundsation python=3.9.12
conda activate soundsation
cd data/preprocessing/sep
pip install -r requirements.txt

# Login to HuggingFace
huggingface-cli login

# Download source separation checkpoints
python download_sep_ckpt.py
```

### Running Separation

```bash
# Single node processing
cd data/preprocessing/sep
bash run.multigpu.sh mtgjamendo false 0 1 8 2

# Or using Slurm for multi-node processing
sbatch run.slurm.seperate.sh
```

## Voice Activity Detection and ASR Pipeline

The VAD and ASR pipeline identifies speech segments and transcribes them, essential for creating aligned text-audio pairs.

### Setup

```bash
conda activate soundsation
cd data/preprocessing/asr

# Login to HuggingFace
huggingface-cli login

# Download ASR checkpoints
python download_asr_ckpt.py
```

### Running VAD and ASR

```bash
bash run_vad_asr.sh
```

## Language Segmentation (LangSegment)

The LangSegment module provides powerful multi-language detection and segmentation capabilities:

- Automatically identifies and segments text by language (Chinese, English, Japanese, Korean, etc.)
- Supports up to 97 different languages with customizable filters
- Handles mixed-language content elegantly
- Includes SSML support for enhanced text-to-speech processing

### Example Usage

```python
from LangSegment.LangSegment import getTexts, getCounts

# Configure language filters (prioritized order)
from LangSegment.LangSegment import setfilters
setfilters(["en"])

text = "Hello，world! This is from Soundsation!"
segments = getTexts(text)

# Get language statistics
lang_stats = getCounts()
main_lang, count = lang_stats[0]  # Get predominant language
```

## Dataset Construction

Similar to approaches described in recent research, Soundsation uses a combination of cleaned audio samples and generated pseudo captions to create training pairs. This approach addresses the data scarcity challenges common in text-to-music generation systems.

For more information on the audio separation architecture and implementation details, see the comments in the relevant source files.
