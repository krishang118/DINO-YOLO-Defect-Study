# Weakly-Supervised Industrial Defect Detection - Exploring DINO + YOLO for Annotation-Free Quality Control

An empirical study demonstrating both the promise and fundamental limitations of attention-based pseudo-labeling for industrial defect detection.

## Overview

The Challenge: In real-world manufacturing, obtaining precise defect annotations (bounding boxes, pixel masks) is expensive, time-consuming, and impractical for constantly evolving defect types.
The Question: Can self-supervised Vision Transformers (DINO) generate training labels for real-time object detectors (YOLO) without any manual annotation?
The Answer: Partially. This project demonstrates where the approach works, where it fails, and, most importantly, *why*.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Self-Supervised Feature Learning                  │
│  ┌──────────┐      ┌────────┐      ┌──────────────┐       │
│  │  Image   │ ───> │  DINO  │ ───> │  Attention   │       │
│  │ (224×224)│      │  ViT   │      │     Maps     │       │
│  └──────────┘      └────────┘      └──────────────┘       │
│                                           │                  │
└───────────────────────────────────────────┼─────────────────┘
                                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Pseudo-Label Generation                           │
│  ┌──────────────┐      ┌────────────┐      ┌────────────┐ │
│  │  Attention   │ ───> │ Multi-Scale│ ───> │  Pseudo    │ │
│  │   Hierarchy  │      │ Threshold  │      │  Bboxes    │ │
│  └──────────────┘      └────────────┘      └────────────┘ │
│  (layers -1,-2,-3)      (95th %ile)         (NMS @ 0.5)   │
└─────────────────────────────────────────────┼─────────────┘
                                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: Real-Time Detector Training                       │
│  ┌────────────┐      ┌────────┐      ┌──────────────┐     │
│  │  Pseudo    │ ───> │ YOLOv8 │ ───> │   Trained    │     │
│  │  Labels    │      │(50 ep) │      │   Detector   │     │
│  └────────────┘      └────────┘      └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: Production Inference + AI Reasoning               │
│  ┌────────┐   ┌────────┐   ┌───────────┐   ┌──────────┐  │
│  │  New   │──>│ YOLO   │──>│ Defect    │──>│ Decision │  │
│  │ Image  │   │Detect  │   │ Reasoner  │   │ Engine   │  │
│  └────────┘   └────────┘   └───────────┘   └──────────┘  │
│                                                             │
│  Decision Types: AUTO_ACCEPT | AUTO_REJECT | HUMAN_REVIEW │
└─────────────────────────────────────────────────────────────┘
```

Key Innovation: Signal-based AI reasoning analyzes detected regions using:
- Spatial features (compactness, edge strength, aspect ratio)
- Frequency analysis (FFT-based texture repetition)
- Semantic features (DINO layer activation patterns)
- Multi-dimensional confidence vectors

## Results Summary

Tested on 15 categories from MVTec Anomaly Detection dataset.

### Universal Finding: 100% Human Review Across All Categories
**All 15 categories** exhibited the same behavior:
- 0% False Accept Rate - Safe failure mode
- 100% Human Review - No automation achieved
- 0% False Rejects - No over-flagging

### Training Success Spectrum

| Category | Pseudo-Labels | mAP50 | Training | Result |
|----------|---------------|-------|----------|---------|
| **screw** | 83 | 87.5% | Excellent | Giant boxes |
| **pill** | 98 | 34.6% | Good | Giant boxes |
| **metal_nut** | 65 | 27.5% | Good | Giant boxes |
| **zipper** | 83 | 28.0% | Good | Giant boxes |
| **cable** | 64 | 15.7% | Weak | Giant boxes |
| **capsule** | 76 | 10.9% | Weak | Giant boxes |
| **toothbrush** | 21 | 9.3% | Weak | Giant boxes |
| **bottle** | 44 | 6.7% | Failed | Giant boxes |
| **hazelnut** | 49 | 3.4% | Failed | Giant boxes |
| **grid** | 39 | 3.3% | Failed | Giant boxes |
| **leather** | 64 | 3.2% | Failed | Giant boxes |
| **carpet** | 62 | 1.4% | Failed | Giant boxes |
| **tile** | 58 | 1.4% | Failed | Giant boxes |
| **wood** | 42 | 7.3% | Failed | Giant boxes |
| **transistor** | 28 | 4.2% | Failed | Giant boxes |

Key Observation: Even when YOLO achieved high mAP (87.5% for screw), the learned detections consisted of giant boxes covering 70-99% of images rather than tight defect localizations.

## Root Cause Analysis

### Why Did Everything Default to Human Review?

#### 1. DINO Attention is Semantic, Not Localized
- DINO highlights "interesting regions" but doesn't provide tight boundaries
- Attention spreads across entire objects, not just defect pixels
- 95th percentile thresholding creates very large boxes

**Example (Screw):**
```
Detection: Box[198, 145, 636, 735] on 1024×1024 image
→ Covers 45% of image area
→ Confidence: 0.017 (1.7%)
```

#### 2. Pseudo-Labels Teach YOLO the Wrong Pattern
With 40-83 training examples of "giant boxes = defects":
- YOLO learns: "Draw large boxes covering most of the image"
- High mAP (87.5%) means "boxes overlap with pseudo-labels"
- Doesn't mean "boxes accurately localize defects"

#### 3. Decision Engine Correctly Rejects Giant Boxes
```python
# In InspectionDecisionEngine
if box_area > (image_area * 0.8):
```
- Boxes covering >80% of image are obviously suspicious
- System correctly flags them as uncertain
- Result: 100% deferred to human review

#### 4. Low Confidence Compounds the Issue
- YOLO outputs 80-300 detections per image
- Top confidence: 0.01-0.20 (1-20%)
- AI Reasoner marks as "low trust"
- Decision: HUMAN_REVIEW

## What This Study Successfully Demonstrates

### 1. Safe Failure Modes
- 0% false accepts across all 15 categories
- System prefers uncertainty over risky automation
- Production-viable as conservative pre-screening filter

### 2. Complete Pipeline Implementation
- Self-supervised feature extraction (DINO)
- Automated pseudo-label generation
- Real-time detector training (YOLO)
- Signal-based AI reasoning (no LLMs)
- Production decision logic

### 3. Rigorous Evaluation
- Proper train/test split (70/30)
- No data leakage
- Held-out evaluation samples
- Comprehensive 15-category testing

### 4. Honest Technical Analysis
- Identified fundamental limitations
- Root cause diagnosis
- Clear understanding of when methods work/don't work

## Final Key Learnings & Insights

1. Self-supervised models are semantic, not geometric
   - DINO excels at "what" but struggles with "where exactly"
   - Attention maps identify salient regions, not precise boundaries

2. Weakly-supervised != Unsupervised
   - Still requires quality pseudo-labels
   - Garbage pseudo-labels → garbage detector

3. mAP can be misleading
   - 87.5% mAP doesn't guarantee useful detections
   - Must evaluate actual localization quality, not just overlap

## How To Run

1. Make sure you have Python 3.8+ installed.
2. Clone this repository on your local machine.
3. Install the required dependencies:
```bash
pip install torch torchvision opencv-python ultralytics matplotlib tqdm Pillow
```
4. Download the [MVTec AD dataset](https://www.kaggle.com/datasets/ipythonx/mvtec-ad), and extract it inside the main project directory as folder `MVTecAD`.
5. Open and run the cells of the `DINOYOLO.ipynb` Jupyter Notebook for the specified category.


DINO-YOLO-Defect-Study
A technical study of attention-based pseudo-labeling for industrial defect detection using DINO for self-supervised attention and YOLO for weakly-supervised training.
