# DL-Project
Fine-Grained Classification of Deer Species using Deep Learning

Overview
This project addresses the fine-grained classification of four visually similar deer species using camera trap imagery. It combines modern deep learning techniques with a multi-stage processing pipeline to tackle common challenges in ecological monitoring: high intra- and inter-class visual similarity, data imbalance, and noisy real-world image data. The work was conducted as a course project and makes contributions in designing, evaluating, and improving a pipeline for robust wildlife classification, using both pretrained CNN backbones and CLIP-based feature representations, culminating in an ensemble learning strategy that achieves state-of-the-art results for this niche yet critical task.

Problem Statement
Traditional wildlife monitoring through camera traps produces large volumes of image data, much of which includes:
    Blank or irrelevant images
    Animals under difficult lighting, pose, or occlusion
    Visually similar species with minute distinguishing features

The primary goal of this project is to automate the fine-grained classification of four deer species from camera trap images using deep learning. Specifically, we aim to:
    Filter and focus on images containing deer via detection
    Accurately classify each deer species using visual features
    Address class imbalance
    Ensure robustness to real-world noise and variability

Target Species
The classification pipeline distinguishes between the following four deer species from the Missouri Camera Trap dataset:
  Red Brocket Deer 
  White Tailed Deer 
  Red Deer 
  Roe Deer 
  
Dataset
Source:
Missouri Camera Trap dataset hosted on LILA BC (Labeled Information Library of Alexandria: Biology and Conservation)
Characteristics:
  Camera trap images in varying lighting, angles, and resolutions
  High visual similarity across species (fine-grained classification challenge)
  Significant class imbalance (some species underrepresented)
  Presence of noise and irrelevant background in raw images

Methodology
  Our pipeline was developed in three stages, with each stage progressively improving classification accuracy and robustness.

Stage 1: Baseline Model – ResNet50 Fine-tuning
  Detection: Used MegaDetector (YOLO-based) to crop images around detected animals, reducing irrelevant background and empty frames.
  Model: ResNet50 pre-trained on ImageNet
  Training: Fine-tuned on cropped deer images using cross-entropy loss
  Result: Achieved a strong baseline test accuracy of 92%

Stage 2: CLIP Feature Extraction + Stacking Ensemble
  Feature Extraction: Used OpenAI’s CLIP (Contrastive Language-Image Pre-training) to generate high-quality embeddings of cropped deer images
  Classifier: Built a stacking ensemble using:
  Multi-Layer Perceptron (MLP)
  XGBoost
  Loss Function: cross-entropy loss for multi-class classification
  Result: Achieved improved test accuracy of 95%

Stage 3: Robustness Testing with Noisy Inputs
  Evaluation: Assessed the robustness of the final model using synthetic noise augmentations (Added Gausian Noise)
  Goal: Ensure model reliability in realistic field conditions where data is often imperfect
  Loss Function: Switched to Focal Loss to reduce bias toward majority classes and enhance minority class precision
  Comparison Models: Evaluated against Random Forest and Histogram Gradient Boosting
  Findings: The final ensemble classifier maintained good performance under moderate noise, demonstrating generalizability.

Experiments & Results

| Model/Method                      | Accuracy | Notes                                           |
|----------------------------------|----------|-------------------------------------------------|
| ResNet50 Fine-tuned              | 92%      | Strong baseline after cropping via detection    |
| CLIP + MLP + XGBoost (Ensemble)  | **95%**  | Improved accuracy and generalization            |
| CLIP + Random Forest             | ~93%     | Competitive but less effective than XGBoost     |
| Focal Loss (vs Cross-Entropy)    | ↑ Recall/Precision | Especially beneficial for minority classes |
| Noise Testing                    | ~90%     | Maintained high accuracy on noisy images        |

Challenges Encountered
High Intra- and Inter-class Visual Similarity
    Tackled via CLIP embeddings capturing nuanced semantics

Class Imbalance
    Handled through Focal Loss and ensemble training with sampling

Noisy, Unbalanced, Low-light, and Occluded Images
    Resolved partially through preprocessing and robustness testing

Generalization Beyond Training Data
    Validated with data augmentation and multiple classifier baselines

Accomplishments
    Successfully developed a multi-stage deep learning pipeline for fine-grained species classification
    Demonstrated effectiveness of CLIP-based representations for wildlife tasks
    Integrated ensemble learning to enhance model generalization and robustness
    Contributed to the automation of biodiversity analysis with scalable and accurate solutions

Technologies Used
    Python, PyTorch, TensorFlow, scikit-learn, XGBoost
    ResNet50
    MegaDetector (YOLO-based) for detection
    CLIP (OpenAI) for image embeddings
    Matplotlib, Seaborn for visualizations


Way Forward
    Integrate self-supervised pretraining for better generalization on small datasets
    Expand to multi-location or multi-season datasets
    Include real-time inference deployment for edge camera trap devices
    Further tackle background clutter with segmentation-based cropping

