# Generic_ANN
[![Everyday ANN](https://github.com/UdarGIT829/Everyday_ANN/actions/workflows/Everyday-ANN.yml/badge.svg)](https://github.com/UdarGIT829/Everyday_ANN/actions/workflows/Everyday-ANN.yml)

Generic ANN implementation, supports spreadsheet-like and features detection of input and output feature types

# Overview
This is a library that provides a clean setup for training and running an Artificial Neural Network (ANN) on user provided data.

It features inputting from CSV file, as well as JSON and Dict through functions. Output of pretrained models will be decoded based on original training data as needed.
NN Optimizations include gradient descent and batching normalization. Also uses Optuna lib to find the best hyperparameters so you dont have to know anything about AI or Machine Learning to make use of this software. 

## Installation

For all the data handling packages, use:
```
pip install -r requirements.txt
```

If not using CUDA, you should instead install with:
```
pip install -r requirements_NOCUDA.txt
```
For CUDA instructions, see below.

### CUDA
For PyTorch installation, please follow the guide for Cuda 12.1 at https://pytorch.org/get-started/locally/.
The command I use is: 
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage
See `main.py` for inspiration for usage as a library

Generally, aim to use a csv file as an input for the training data and for predictions, although a JSON string or dictionary with the correct column headers will suffice. 

### Server Usage

To run the server, use the following command:
```
uvicorn app:app --reload
```

## Roadmap
✔️ = Complete | 🔄 = In Progress | 📝 = Planning

Library Functionality:
- ✔️ Support for JSON
- ✔️ Support for CSV
- ✔️ CI/CD using GitHub Actions and Docker
- 🔄 Screenshots
- 📝 Use some kind of GitHub Project Management to show this Roadmap
- 📝 Support for SQL and MongoDB
- 📝 Support for online dataset loaders like Kaggle and HuggingFace 

Server Functionality:
- ✔️ FastAPI server setup
- 📝 Basic WebUI for API server
- 📝 Electron App/ C# GUI Wrapper
- 📝 Stream output for Optuna trials
- 🔄 Screenshots

## References
The sample dataset is from Kaggle at `https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset`