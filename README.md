# Fine-Grained Classification of Deer Species using Deep Learning

## 🔬 Project Repository
[DL-Project GitHub](https://github.com/saadi54humayun/DL-Project)

---

## Overview

This project addresses the **fine-grained classification of four visually similar deer species** using camera trap imagery. It combines modern deep learning techniques with a multi-stage processing pipeline to tackle common challenges in ecological monitoring: high intra- and inter-class visual similarity, data imbalance, and noisy real-world image data.

The work was conducted as a course project and contributes by designing, evaluating, and improving a pipeline for robust wildlife classification using pretrained CNN backbones, CLIP-based features, and ensemble learning strategies.

---

##  Problem Statement

Camera trap data used in wildlife monitoring presents several challenges:
- Large volume of unlabeled and irrelevant images
- Difficult lighting, occlusion, and variable animal poses
- Visually similar species with subtle differences

Our primary goal is to build an automated pipeline to:
- Detect deer in camera trap images
- Accurately classify each of four species
- Handle class imbalance
- Ensure robustness under realistic noisy conditions

---

##  Target Species

The classifier distinguishes between the following four deer species from the Missouri Camera Trap dataset:
1. White-tailed Deer
2. Mule Deer
3. Elk
4. Moose

---

##  Dataset

### Source
- [Missouri Camera Trap Dataset (LILA BC)](https://lila.science/)

### Characteristics
- Real-world images captured via camera traps
- High visual similarity between species
- Imbalanced class distribution
- Presence of background noise, occlusion, and lighting variation

---

##  Methodology

We implemented a three-stage approach to build and improve the classification pipeline.

###  Stage 1: Baseline – ResNet50 Fine-tuning
- **Detection:** Used MegaDetector (YOLO-based) to crop bounding boxes around detected animals.
- **Classifier:** Fine-tuned ResNet50 pretrained on ImageNet.
- **Training:** Standard cross-entropy loss.
- **Result:** Achieved 92% test accuracy.

---

###  Stage 2: CLIP + Stacking Ensemble
- **Feature Extraction:** Used OpenAI’s CLIP model to extract high-dimensional semantic image embeddings.
- **Ensemble Classifier:** Combined:
  - Multi-Layer Perceptron (MLP)
  - XGBoost
- **Loss Function:** cross-entropy loss for class imbalance
- **Result:** Achieved improved accuracy of 95%.

---

###  Stage 3: Robustness Testing
- **Objective:** Evaluate resilience to real-world conditions like noise, blur, and brightness changes.
- **Methadology:** added Gaussian noise to the dataset to see against the 
- **Loss Function:** Switched to Focal Loss to counter class imbalance.
- **Comparison Models:** Evaluated against Random Forest and Histogram Gradient Boosting
- **Findings:** CLIP-based ensemble maintained high performance under moderate perturbations.
- **Result:** Achieved improved accuracy of 95% and better recall for minority classes.

---

##  Experiments & Results

| Model/Method                      | Accuracy | Notes                                           |
|----------------------------------|----------|-------------------------------------------------|
| ResNet50 Fine-tuned              | 92%      | Strong baseline after cropping via detection    |
| CLIP + MLP + XGBoost (Ensemble)  | **95%**  | Improved accuracy and generalization            |
| CLIP + Random Forest             | ~93%     | Competitive but less effective than XGBoost     |
| Focal Loss (vs Cross-Entropy)    | Improved Recall/Precision | Especially beneficial for minority classes |
| Noise Testing                    | ~90%     | Maintained high accuracy on noisy images        |

---

##  Challenges

- **Fine-Grained Similarity:** Overcome using CLIP’s semantic features.
- **Imbalanced Dataset:** Mitigated with Focal Loss and ensemble modeling.
- **Real-World Noise:** Tackled through detection preprocessing and robustness testing.
- **Overfitting Risk:** Reduced via feature reuse and model stacking.

---

##  Accomplishments

- Designed a full pipeline from detection to classification
- Improved baseline accuracy from 92% → 95% using CLIP + ensemble
- Validated robustness under synthetic noise conditions
- Contributed to scalable, automated biodiversity monitoring solutions

---

## Technologies Used

- Python
- PyTorch
- TensorFlow
- CLIP (OpenAI)
- MegaDetector (YOLOv5)
- XGBoost
- scikit-learn
- Matplotlib, Seaborn

---


