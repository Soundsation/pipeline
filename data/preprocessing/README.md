# Data Preprocessing Guide

---

## Overview
This guide provides instructions on how to set up the environment and run the data preprocessing pipeline for **Soundsation**. 
The preprocessing involves tasks like tokenizing lyrics, converting text to phonemes, and preparing datasets for training.

## Prerequisites
Before running the preprocessing pipeline, ensure you have the following:
1. Python Environment:
   - Python 3.12 or higher.
   - Conda or virtualenv for managing dependencies.
2. Dependencies:
   - Required Python libraries (listed in `requirements.txt`).
   - External tools like `espeak-ng` for phoneme generation.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Soundsation/pipeline.git
cd pipeline
```

### 2. Install Dependencies
#### Using Conda:
```bash
conda create -n soundsation-pipeline python=3.12
conda activate soundsation-pipeline
pip install -r requirements.txt
```

#### Using Virtualenv:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate     # For Windows
pip install -r requirements.txt
```

### 3. Install `espeak-ng`
#### Linux (Debian-based):
```bash
sudo apt-get install espeak-ng
```
#### MacOS:
```bash
brew install espeak-ng
```
#### Windows:
- Download the `.msi` installer from [Espeak-ng Releases](https://github.com/espeak-ng/espeak-ng/releases).
- Set environment variables:
  - `PHONEMIZER_ESPEAK_LIBRARY`: Path to `libespeak-ng.dll`.
  - `PHONEMIZER_ESPEAK_PATH`: Path to Espeak installation directory.

## Pipeline Overview

### Files Involved in Preprocessing
1. Dataset Preparation:
   - `dataset/dataset.py`: Handles loading, tokenization, and alignment of audio and lyrics.
2. Grapheme-to-Phoneme Conversion (G2P):
   - `g2p/g2p_generation.py`: Main script for generating phonemes.
   - Language-specific files (e.g., `g2p/g2p/english.py`, `g2p/g2p/french.py`, etc.) handle text normalization and phoneme generation.
3. Utility Files:
   - `g2p/utils/g2p.py`: Provides backend integration for phonemization.
   - `g2p/utils/front_utils.py`: Generates polyphonic lexicons for Mandarin Chinese.

## Running the Preprocessing Pipeline

### Step 1: Tokenize Lyrics (G2P)
Run the Grapheme-to-Phoneme conversion script to process lyrics into phonemes:
```bash
python g2p/g2p_generation.py --input_dir /path/to/lyrics --output_dir /path/to/phonemes --language en
```
Replace `en` with the appropriate language code (`zh`, `fr`, `de`, etc.).

### Step 2: Prepare Dataset
Use the dataset preparation script to combine processed lyrics and audio features into training-ready batches:
```bash
python dataset/dataset.py --file_path /path/to/train.scp --output_dir /path/to/final_dataset
```
Ensure that the file paths in your `.scp` file point to valid audio and lyric files.

## Example Workflow
1. Place raw audio files in `/data/audio/`.
2. Place lyrics files in `/data/lyrics/`.
3. Run G2P conversion:
   ```bash
   python g2p/g2p_generation.py --input_dir /data/lyrics --output_dir /processed_data/phonemes --language en
   ```
4. Run dataset preparation:
   ```bash
   python dataset/dataset.py --file_path /processed_data/train.scp --output_dir /processed_data/final_dataset
   ```

## Contact
For issues or questions, please contact us via [GitHub Issues](https://github.com/Soundsation/pipeline/issues).
