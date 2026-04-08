## Dataset

This project uses the Cervical Spine X-ray Atlas (CSXA) V3.0 dataset.

Yu Ran, “Cervical Spine X-ray Atlas (CSXA) V3.0.” Science Data Bank, May 15, 2024 [Online]. Available: https://doi.org/10.57760/sciencedb.15391. [Accessed: Apr. 08, 2026]

Changes were made to the original dataset.
The dataset was modified: multiple JSON files were combined into a single summary CSV file called atlas_verteba_full.csv.

The data was then split into 5 folds (fold_0 to fold_4), each consisting of a training and test portion. These fold files contain only image-label pairs with the original JSON annotations.

For this project, the folds are used to select images, while the summary CSV provides their annotations.

Repository Usage Note:
The repository contains only a few test images, along with generated val.csv and test.csv files. To use the full dataset, download all images, place them in the images folder, and update the config to use the desired images—specifically, set it to load a particular fold.

## Acknowledgements / Third-Party Code

This project is based on the PyTorch implementation of Stacked Hourglass Networks:

PyTorch implementation:
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

The original code is licensed under the BSD 3-Clause License.
Modifications have been made in this repository.