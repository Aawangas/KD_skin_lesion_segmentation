# Skin Lesion Segmentation with Knowledge Distillation

This project implements a lightweight student model for skin lesion segmentation using Knowledge Distillation (KD). The framework uses a powerful ResNet-101 based DeepLabV3+ model as a teacher to guide a smaller ResNet-18 based DeepLabV3+ student model.

## Features

- **Knowledge Distillation**: Intermediate feature distillation and logit distillation.
- **Configurable**: Easy-to-use YAML configuration for all parameters.
- **Robust Data Augmentation**: Including mirror padding, random affine transforms, and color jittering.

## Project Structure

```text
.
├── configs/            # Configuration files (YAML)
├── models/             # Model architectures (Teacher, Student, DeepLabV3+)
├── utils/              # Utility functions (Data loading, Loss, Training)
├── distill.py          # Main script for distillation and evaluation
├── train_teacher.py    # Script for training the teacher model
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/wanganbang/KD_skin_lesion_segmentation.git
   cd KD_skin_lesion_segmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Configuration

Edit `configs/config.yaml` to set your data paths, model parameters, and training settings.

### 2. Training the Teacher Model

To train the teacher model from scratch:
```bash
python train_teacher.py --backbone resnet
```

### 3. Knowledge Distillation

To train the student model using knowledge distillation:
```bash
python distill.py --mode distill --teacher_weights models/model_files/teacher1.pth
```

### 4. Training Raw Student (Baseline)

To train the student model without distillation:
```bash
python distill.py --mode raw
```

### 5. Evaluation

To evaluate a trained student model:
```bash
python distill.py --mode test --student_weights checkpoints/student_distill.pth
```

## Acknowledgments

- This project was developed as part of a UROP project on Deep Learning for Skin Lesion Segmentation.
- The DeepLabV3+ implementation is inspired by various open-source PyTorch implementations.
