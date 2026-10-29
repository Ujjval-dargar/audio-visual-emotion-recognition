# Audio–Visual Emotion Recognition

A machine-learning course project exploring emotion recognition from paired facial images and speech recordings in the CREMA-D dataset.

## Overview

The notebook combines visual HOG and intensity-histogram features with speech MFCC features to classify six emotions: anger, disgust, fear, happiness, neutral, and sadness. It also includes image and audio exploration, signal transformations, and a NumPy implementation of principal component analysis (PCA).

## Implemented components

- Image preprocessing, grayscale conversion, histograms, local binary patterns, Sobel edges, and HOG descriptors.
- Audio resampling, zero-crossing rate, spectrograms, Mel/log-Mel spectrograms, constant-Q transforms, and MFCC features.
- PCA using covariance eigendecomposition.
- Early fusion of paired audio and image feature vectors.
- A notebook baseline using linear regression on one-hot emotion labels.
- A standalone experimental pipeline for image-only, audio-only, and fused features, with logistic regression, SVM, random forest, and k-nearest-neighbor models.

## Tech stack

Python, NumPy, pandas, SciPy, scikit-learn, OpenCV, Librosa, Matplotlib, and Seaborn.

## Setup

Python 3.11 or 3.12 is recommended as a starting environment; dependencies are not version-locked and a clean full run has not yet been verified for this repository.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is not included. Supply the course-prepared CREMA-D image/audio export using this layout:

```text
data/
  emotion/
    dataset_metadata.csv
    images/<filename>.jpeg
    audio/<filename>.wav
```

Metadata columns: `filename`, `actor_id`, `sentence_text`, `emotion_text`, `emotion_level_text`. The filename field links each image and audio recording. The raw upstream dataset may require preparation to match this layout.

The notebook also contains optional format-comparison exercises requiring the original course `data/audiocomp/` and `data/imagecomp/` folders.

```bash
jupyter notebook emotion_recognition.ipynb
# Or run the extended experimental pipeline:
python multimodal_emotion_recognition.py
```

## Evaluation status and next steps

This repository preserves coursework and an experimental extension. Notebook outputs are cleared; no new benchmark results are claimed.

- The notebook PCA transformation needs to reuse the training mean for new samples; the standalone implementation already does so.
- Current splits are sample-level. Actor-disjoint train/validation/test splits are planned to evaluate unseen-person generalization.
- The standalone script currently compares and selects models using test scores. Model selection should move to validation data before reporting final test performance.
- Planned reporting includes macro-F1, per-class recall, and modality ablations on one fixed held-out set.

## Files

- `emotion_recognition.ipynb`: original coursework code and exercises, with saved outputs removed.
- `multimodal_emotion_recognition.py`: extended experimental pipeline.
- `requirements.txt`: required Python packages.
