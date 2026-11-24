
# About the Project

This repository contains a framework-agnostic pipeline designed for training keypoint detection models. It covers the entire end-to-end workflow, including data augmentation, model training, and a dedicated script for running predictions (inference) on new images.

While the core architecture is designed to be framework-agnostic and can be extended to support other deep learning libraries, the current implementation provided in this repository uses **PyTorch**.

# Dataset

This project uses the **Cervical Spine X-ray Atlas (CSXA) V3.0** dataset.

Yu Ran, "Cervical Spine X-ray Atlas (CSXA) V3.0." Science Data Bank, May 15, 2024. [Online]. Available: [https://doi.org/10.57760/sciencedb.15391](https://doi.org/10.57760/sciencedb.15391). [Accessed: Apr. 08, 2026].

Changes were made to the original dataset:

- Multiple JSON annotation files were combined into a single summary CSV file named `atlas_verteba_full.csv`.
- The dataset was split into five folds (`fold_0` to `fold_4`), each containing training and test portions.
- The fold files contain image-label pairs with the original JSON annotations.

In this project, the folds are used to select images, while the summary CSV file provides their corresponding annotations.

**Repository Usage Note:**

The repository contains only a small number of sample images, along with generated `val.csv` and `test.csv` files. To use the full dataset, download all images, place them in the `images/` directory, and update the configuration to use the desired dataset fold.

# Setup and Installation

## Prerequisites

- Python 3.12+ (tested with Python 3.12.5 and 3.12.10)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AdelaSvitilova/KeypointDetection.git
   cd KeypointDetection
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

# Usage

## 1. Training the Model

To start the training process, execute the training script. The repository includes a small sample dataset, allowing you to run the script out of the box and quickly verify that the pipeline works correctly.

For further experiments or training on the full dataset, you can adjust the hyperparameters and other settings in the configuration files.

```bash
python train.py
```

## 2. Running Predictions (Inference)

To run predictions on new images, you can either use a model trained in the previous step or use one of the pre-trained models generated during this thesis to avoid training from scratch.

**Step 1: Download a pre-trained model**

Download the entire pre-trained experiment folder from [Google Drive](https://drive.google.com/drive/folders/1qH2ezr4zkesicUub4X2DCb9tuE-ZjEmQ?usp=sharing).

**Step 2: Prepare the directory**

Extract the downloaded folder and place it directly into the `results/` directory. The directory structure should look like this:

```text
results/
└── experiment_name/
```

**Step 3: Update the configuration**

Before running the prediction script, specify which experiment to use. Open `configs/base_config.yaml` and set the `experiment_name` variable to match the exact name of the downloaded folder.

**Step 4: Run the prediction script**

Once the experiment name is configured, execute:

```bash
python full_predict.py
```

# Project Structure

- `configs/` - YAML files containing experiment configurations.
- `data/` - Directory for datasets.
- `results/` - Stores experiment outputs, including trained models, checkpoints, TensorBoard training logs, and detailed prediction analysis.
- `src/` - Core source code, including datasets, augmentations, losses, metrics, models, trainers, and utilities.
- `train.py` - Main script for model training.
- `full_predict.py` - Main script for running predictions and generating analysis.

# Acknowledgements

This project uses and adapts code from the following open-source implementations:

## Stacked Hourglass

The Stacked Hourglass model is based on the PyTorch implementation by [Chris Rockwell](https://github.com/princeton-vl/pytorch_stacked_hourglass), which builds upon the following works:

- **Stacked Hourglass Networks for Human Pose Estimation**  
  Alejandro Newell, Kaiyu Yang, and Jia Deng. ECCV, 2016.  
  [Original implementation](https://github.com/princeton-vl/pose-hg-train)

- **Associative Embedding: End-to-end Learning for Joint Detection and Grouping**  
  Alejandro Newell, Zhiao Huang, and Jia Deng. NeurIPS, 2017.  
  [Original implementation](https://github.com/princeton-vl/pose-ae-train)

The original PyTorch implementation is distributed under the **BSD 3-Clause License**. Modifications and adaptations have been made for this project. A copy of the original license is included in the `licenses/` directory.

## HRNet

The HRNet model is based on the [HRNet PyTorch implementation](https://github.com/HRNet/HRNet-Human-Pose-Estimation), which builds upon the following works:

- **Deep High-Resolution Representation Learning for Human Pose Estimation**  
  Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. CVPR, 2019.

- **Simple Baselines for Human Pose Estimation and Tracking**  
  Bin Xiao, Haiping Wu, and Yichen Wei. ECCV, 2018.

The original implementation is distributed under the **MIT License**. Modifications and adaptations have been made for this project. A copy of the original license is included in the `licenses/` directory.

## Cascaded Pyramid Network (CPN)

The CPN model is based on the [PyTorch re-implementation by GengDavid](https://github.com/GengDavid/pytorch-cpn), which implements the method described in:

- **Cascaded Pyramid Network for Multi-Person Pose Estimation**  
  Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang Zhang, Gang Yu, and Jian Sun. CVPR, 2018.  
  [Paper](https://arxiv.org/abs/1711.07319)

The original PyTorch implementation is distributed under the **GNU General Public License v3.0 (GPL-3.0)**. Modifications and adaptations have been made for this project. A copy of the original license is included in the `licenses/` directory.
