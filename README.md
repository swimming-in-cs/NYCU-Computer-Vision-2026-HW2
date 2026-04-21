# NYCU Computer Vision 2026 HW2 — Digit Detection

Student ID: 111550005

## Introduction

This repository contains the implementation of Digit Detection using Deformable DETR with ResNet-50 backbone for NYCU Computer Vision 2026 HW2.

## Environment Setup

```bash
pip install torch torchvision
pip install pycocotools scipy albumentations tqdm
```

Or on Google Colab:

```bash
!pip install pycocotools scipy albumentations tqdm
```

## Usage

### Training

Paste the contents of `train.py` into a Google Colab cell and run.

The script will automatically resume from the latest checkpoint if one exists.

### Inference

After training, `pred.json` will be generated automatically. To run inference manually:

```python
best_ck = torch.load("path/to/best.pth", map_location=device)
model.load_state_dict(best_ck["model_state"])
run_inference(model, cfg, device)
```

Then zip and submit:

```bash
zip submission.zip pred.json
```

## Performance Snapshot

| Split | mAP@[0.50:0.95] |
|-------|----------------|
| Validation | 0.4342 |
| Public Test (CodaBench) | 0.35 |
