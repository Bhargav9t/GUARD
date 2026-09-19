# GUARD Dataset Directory Structure

This directory holds the raw and preprocessed images for training, cross-dataset validation, and degradation sweeps.

## Expected Directory Layout:

```
data/
├── raw/
│   ├── real/               # Authentic, unmanipulated face and scene images
│   │   ├── image_001.jpg
│   │   └── ...
│   └── fake/               # Deepfakes / AI-generated (Diffusion, GAN) images
│       ├── image_001.jpg
│       └── ...
├── processed/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
└── benchmarks/             # External evaluation datasets
    ├── FaceForensics++/
    ├── Celeb-DF-v2/
    ├── DFDC/
    └── DiffusionForensics/
```
