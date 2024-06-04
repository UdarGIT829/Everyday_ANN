# Generic_ANN
Generic ANN implementation, supports spreadsheet-like and features detection of input and output feature types

# Overview
This is a library that provides a clean setup for training and running an Artificial Neural Network (ANN) on user provided data.

It features inputting from CSV file, as well as JSON and Dict through functions. Output of pretrained models will be decoded based on original training data as needed.
NN Optimizations include gradient descent and batching normalization. Also uses Optuna lib to find the best hyperparameters so you dont have to know anything about AI or Machine Learning to make use of this software. 

## Installation
For PyTorch installation, please follow the guide for Cuda 12.1 at https://pytorch.org/get-started/locally/.

For the other packages, use:
```
pip install -r requirements.txt
```

## Usage
See `main.py` for inspiration for function usage

Generally, aim to use a csv file as an input for the training data and for predictions, although a JSON string or dictionary with the correct column headers will suffice. 

## Planned Features

As a library the code is in a functional state, but more can be done compatibility wise to enable the following features:
- FastAPI server setup
- Basic WebUI for API server
- Electron App/ C# GUI Wrapper
- Stream output for Optuna trials
- CI/CD using GitHub Actions and Docker
- Screenshots
- Support for SQL and MongoDB
- Support for online dataset loaders like Kaggle and HuggingFace 