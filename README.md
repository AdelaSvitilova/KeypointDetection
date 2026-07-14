## About the Project

This repository contains a framework-agnostic pipeline designed for training keypoint detection models. It covers the entire end-to-end workflow, including data augmentation, model training, and a dedicated script for running predictions (inference) on new images.

While the core architecture is designed to be framework-agnostic and is ready to be extended with other deep learning libraries, the current active implementation provided in this repository uses **PyTorch**.

## Dataset

This project uses the Cervical Spine X-ray Atlas (CSXA) V3.0 dataset.

Yu Ran, “Cervical Spine X-ray Atlas (CSXA) V3.0.” Science Data Bank, May 15, 2024 [Online]. Available: [ScienceDB (DOI: 10.57760/sciencedb.15391)](https://doi.org/10.57760/sciencedb.15391). [Accessed: Apr. 08, 2026]

Changes were made to the original dataset.
The dataset was modified: multiple JSON files were combined into a single summary CSV file called atlas_verteba_full.csv.

The data was then split into 5 folds (fold_0 to fold_4), each consisting of a training and test portion. These fold files contain only image-label pairs with the original JSON annotations.

For this project, the folds are used to select images, while the summary CSV provides their annotations.

Repository Usage Note:
The repository contains only a few test images, along with generated val.csv and test.csv files. To use the full dataset, download all images, place them in the images folder, and update the config to use the desired images—specifically, set it to load a particular fold.

## Setup and Installation

**Prerequisites:**
* Python 3.12+ (Tested on Python 3.12.5 and 3.12.10)

**Installation:**
1. Clone the repository:
```bash
   git clone https://github.com/AdelaSvitilova/KeypointDetection.git
   cd KeypointDetection
```
2. Install the required dependencies:
```bash
   pip install -r requirements.txt
```

## Usage

### 1. Training the Model

To start the training loop, simply execute the training script. The repository includes a small sample dataset, allowing you to run this script out-of-the-box to quickly verify its functionality. 

If you wish to run further experiments, change hyperparameters, or train on the full dataset, you can adjust the settings in the configuration files.

```bash
   python train.py
```

### 2. Running Predictions (Inference)
To run predictions on new images, you can either use a model you trained yourself in the previous step, or use the pre-trained models generated during this thesis to avoid training from scratch.

**Step 1: Download the pre-trained model**
Download the entire pre-trained experiment folder from here: [Google Drive](https://drive.google.com/drive/folders/1qH2ezr4zkesicUub4X2DCb9tuE-ZjEmQ?usp=sharing)

**Step 2: Prepare the directory**
Extract the downloaded folder and place it directly into the `results/` directory. The structure should look like this:
`results/experiment_name/`

**Step 3: Update configuration**
Before running the prediction script, you need to specify which experiment to use. Open `configs/base_config.yaml` and change the `experiment_name` variable to match the exact name of the downloaded folder.

**Step 4: Run the prediction**
Once the experiment name is set, execute the prediction script:

```bash
   python full_predict.py
```

## Project Structure
* `configs/` - YAML files with experiment configurations.
* `data/` - Directory for datasets.
* `results/` - Stores all experiment outputs, including trained model, checkpoints, TensorBoard training logs, and detailed analysis from predictions.
* `src/` - Core source code (datsets, augmentations, losses, metrics, models, trainers, utils).
* `train.py` - Main script for training the model.
* `full_predict.py` - Main script for running predictions and analysis.


## Acknowledgements / Third-Party Code

This project uses the following third-party implementations:

### Stacked Hourglass Networks (PyTorch implementation)  
https://github.com/princeton-vl/pytorch_stacked_hourglass  

The above implementation is based on the following original works:

Stacked Hourglass Networks for Human Pose Estimation  
Alejandro Newell, Kaiyu Yang, and Jia Deng  
ECCV 2016  
https://github.com/princeton-vl/pose-hg-train  

Associative Embedding: End-to-end Learning for Joint Detection and Grouping  
Alejandro Newell, Zhiao Huang, and Jia Deng  
NeurIPS 2017  
https://github.com/princeton-vl/pose-ae-train  

Original PyTorch implementation by Chris Rockwell.  

Licensed under the BSD 3-Clause License.  

Modifications have been made in this repository.

### High-Resolution Network (HRNet) for Human Pose Estimation  
https://github.com/HRNet/HRNet-Human-Pose-Estimation  

The above implementation is based on the following works:

Deep High-Resolution Representation Learning for Human Pose Estimation  
Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang  
CVPR 2019  

Simple Baselines for Human Pose Estimation and Tracking  
Bin Xiao, Haiping Wu, and Yichen Wei  
ECCV 2018  

Licensed under the MIT License.  

Modifications have been made in this repository.