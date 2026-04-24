# Solar Panel Defect Classification System
ASCL: A Spatial-Channel Self-Attention and Multi-Scale Feature Fusion Network for Solar Panel Defect Classification
# Project Overview 
This project aims to construct an efficient solar photovoltaic panel defect classification model utilizing deep learning technology. By adopting an enhanced FasterViT network architecture, combined with the SCSA (Spatial-Channel Self-Attention) module and the LFF multi-scale feature fusion module, it achieves automatic multi-label recognition of common defects (cracks, poor contact, interconnection faults, corrosion) on photovoltaic panels, providing a high-precision and lightweight solution for real-time quality inspection of photovoltaic panels.
# Project Structure
```
SolarPanelDefect/
├──README.md               #Project description document
├── main.py                # Model definition
└── dataset/
    ├── train/
    ├── val/
    └── test/
```
# Core Technologies 
__1.Improved FaterVIT__：Optimize the hyperparameters and initialization strategy to enhance training stability.
__2.Learning rate adjustment：__：
Using the cosine annealing strategy to dynamically adjust the learning rate enables the model to converge finely with smaller steps in the later stages of training
__3.Gradient Clipping__: 
Widen the gradient clipping threshold to 5.0, to prevent gradient explosion while retaining more effective gradient information
# Recommended Environment
Python 3.10.19  
PyTorch == 2.9.1
# Dataset acquisition and structure
The dataset should be organized in the following structure:
```
 dataset/
    ├── train/
    │ ├── images/          # training set images
    │ └── labels.csv       # training set labels
    ├── val/
    │ ├── images/          # val set images
    │ └── labels.csv       # val set labels
    └── test/
    ├── images/            # test set images
    └── labels.csv         # val set labels
```
Each category folder contains images of solar panel defects corresponding to that category.
# Model Training 
The project uses an integrated script `main.py` to handle the entire workflow, including training, validation, and final testing.
```
# Training Parameter Description
During the training process, model weights will be automatically saved. After training, the model performance will be evaluated on the test set.
| Initial learning rate | Epoch| Batch size | 
|:------|:----:|-------:|
|0.001  | 100  | 32   |  
# Performance Evaluation
The script executes an end-to-end workflow, automatically evaluating the model on the test set and outputting key metrics (e.g., accuracy) immediately after training concludes.
# References and contact information
The paper is in the submission stage and will update the BiBTeX citation format after its official publication. Currently, it can be temporarily cited:
```
@article{tssc_pea_disease,  
  title={ASCL: A Spatial-Channel Self-Attention and Multi-Scale Feature Fusion Network for Solar Panel Defect Classification},  
  author={[Author's name, to be added when published]},  
  journal={[Journal name, to be supplemented after acceptance]},  
  year={2026},  
  note={Manuscript submitted for publication}  
}  
```
# Contact Information
If you encounter code running issues or academic exchange needs, please contact:  
Email:liuchenshuohuuc@yeah.net  
GitHub Issue：Submit an issue directly in this warehouse
