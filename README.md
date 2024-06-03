# Generic_ANN
Generic ANN implementation, supports spreadsheet-like and features detection of input and output feature types

# Overview
This is a library that provides a clean setup for training and running an Artificial Neural Network (ANN).
It features inputting from CSV file, as well as JSON and Dict through functions. Output of pretrained models will be decoded based on original training data as needed.
NN Optimizations include gradient descent and batching normalization. Also uses Optuna lib to find the best hyperparameters so you dont have to know anything about AI or Machine Learning to make use of this software. 

## Installation
For PyTorch installation, please follow the guide for Cuda 12.1 at https://pytorch.org/get-started/locally/.

For the other packages, use:
```
pip install -r requirements.txt
```
