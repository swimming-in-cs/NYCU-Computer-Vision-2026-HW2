# NYCU Computer Vision 2026 HW2 — Digit Detection

Student ID: 111550005

Name: 林均豪

## Introduction

This project implements a digit detection model using Deformable DETR
with ResNet-50 as the backbone. The model is trained end-to-end with
Hungarian matching for set-based loss computation and multi-scale
deformable attention for efficient feature aggregation. Per-class NMS
is applied during inference to preserve overlapping digits of different
classes.

## Environment Setup
```bash
pip install torch torchvision
pip install pycocotools scipy albumentations tqdm
```

## Usage

**Training & Prediction:**

```bash
python train.py
```

```bash
# The script automatically resumes from last.pth if a checkpoint exists
# pred.json is generated automatically after training
```

**Data structure:**
```
nycu-hw2-data/
├── train/
├── valid/
├── test/
├── train.json
└── valid.json
```


## Performance Snapshot

| Epoch | Val mAP@[0.50:0.95] |
|-------|---------------------|
| 0     | 0.311               |
| 3     | 0.391               |
| 9     | 0.411               |
| 12    | 0.434               |

| Split | Score |
|-------|-------|
| Validation mAP | 0.4342 |
| Public Test (CodaBench) | 0.35 |

![Leaderboard Score](images/score.png)
