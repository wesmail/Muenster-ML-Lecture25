# Lecture-5: Deep Convolutional Neural Networks

This lecture introduces and explores the use of **1D and 2D convolutional neural networks (CNNs)** in practical machine learning tasks, including image classification and time series forecasting. We also discuss the concept of **pooling**, **data augmentation**, and **transfer learning** using pretrained models.

The lecture is organized into **four Jupyter notebooks**, each focused on a key topic.


## Notebooks Overview

### 1. **Convolution Explained + Manual Implementation** (`Convolution_Introduction.ipynb`)
- What is convolution (mathematical vs deep learning definition)?
- Difference between convolution and cross-correlation
- How flipping works in convolution
- Manual implementation of 1D convolution (using loops and vectorized operations)
- Optional: including stride, padding
- PyTorch equivalent using `nn.Conv1d`

### 2️. **MNIST Hello World + Importance of Augmentation** (`MNIST.ipynb`)
- Load and visualize the MNIST digit dataset
- Build a simple CNN using `nn.Sequential`
- Train on MNIST and achieve high accuracy
- Apply **rotations to test data** to show how accuracy drops
- Discuss and demonstrate the importance of **data augmentation**
- Explain how **pooling layers improve robustness**

### 3. **Transfer Learning with ResNet18** (`Transfer_Learning.ipynb`)
- What is transfer learning and why is it useful?
- Introduction to ResNet and its role in image classification
- Adapt pretrained `resnet18` to classify MNIST digits
- Fine-tune the model and evaluate performance
- Discussion of the **ImageNet-trained model's structure** and why it has 1000 outputs


### 4. **Time Series Forecasting with 1D Convolution** (`Time_Series_Analysis.ipynb`)
- Generate a clean sine wave dataset
- Use sliding windows to build training samples
- Train a **1D CNN** to predict the next value in the sequence
- Evaluate the model and **plot predictions vs true values**
- Plot **residuals** to visualize prediction error


Feel free to extend these notebooks to real-world datasets such as EEG/ECG signals, seismic data, or even audio classification!