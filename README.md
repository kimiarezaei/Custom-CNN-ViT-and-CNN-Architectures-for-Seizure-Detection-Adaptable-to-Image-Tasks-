# Custom CNN-ViT & CNN for seizure detection using ECG Spectrograms (Reusable for Image Classification)

This repository contains **my custom-designed CNN-ViT and CNN architectures** for detecting seizures from ECG spectrograms.  
While designed for ECG data, the models are **fully adaptable for other image classification tasks**, making them reusable beyond this project.

---

## Process

1. Customized dataset is build for pytorch
2. A  patient independent 5-fold cross validation is conducted in pytorch to train the models using train and validation dataset and then test it on the unseen test set 
4. Different measures such as confusion matrix, AUC, Accuracy, MCC, F1 Score were used to evaluate the models performance

---
CNN-ViT architecture is illustrated below:

![Model Architecture](cnnvit.png)


CNN architecture is illustrated below:

![Model Architecture](CNNmodel_seizure.png)

---

This method is detailed in the following paper, where it is compared with other techniques. If you find this code useful, please consider citing the paper.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kimiarezaei/Custom-CNN-ViT-and-CNN-Architectures-for-Seizure-Detection-Adaptable-to-Image-Tasks-.git
cd Custom-CNN-ViT-and-CNN-Architectures-for-Seizure-Detection-Adaptable-to-Image-Tasks-
