import os
import pytest
from easydict import EasyDict
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Network import train_nn, save_nn, load_nn, run_model
from DataImporter import DataImporter
from main import train_model, run_saved_model

DEFAULT_trial_details = EasyDict({
    "hidden_layer_sizes": {"low": 4, "high": 128},
    "num_epochs": {"low": 50, "high": 1000},
    "learning_rate": {"low": 1e-5, "high": 1e-1},
    "batch_size": {"low": 16, "high": 128},
    "activation": ['ReLU', 'Tanh', 'LeakyReLU', 'ELU', 'GELU', 'Swish']
})

@pytest.fixture
def data_path():
    return 'Crop_Recommendation.csv'  # Adjust to your actual file path if necessary

@pytest.fixture
def model_path():
    return 'test_best_model.pkl'

@pytest.fixture
def trial_details():
    return EasyDict(
        {"num_epochs": {"high": 400}}
    )

def test_train_model(model_path, data_path, trial_details):
    train_model(
        modelPath=model_path,
        data_path=data_path,
        trial_details=trial_details,
        n_trials=1
    )
    assert os.path.exists(model_path)

def test_run_saved_model_csv(model_path, data_path):
    input_data = "Testing_Sample.csv"
    prediction = run_saved_model(
        modelPath=model_path,
        input_data=input_data,
        data_path=data_path
    )
    assert prediction is not None
    assert isinstance(prediction, dict)
    assert isinstance(prediction['predictions'],list)
    assert isinstance(prediction['file'],str)

def test_run_saved_model_dict(model_path, data_path):
    input_data = {
        'Nitrogen': [90],
        'Phosphorus': [42],
        'Potassium': [43],
        'Temperature': [20.87974371],
        'Humidity': [82.00274423],
        'pH_Value': [6.502985292],
        'Rainfall': [202.9355362],
        'Crop': ['-']
    }
    prediction = run_saved_model(
        modelPath=model_path,
        input_data=input_data,
        data_path=data_path
    )
    assert prediction is not None
    assert isinstance(prediction, dict)
    assert isinstance(prediction['predictions'],list)

def test_run_saved_model_json(model_path, data_path):
    input_data = '''
    {
        "Nitrogen": [90],
        "Phosphorus": [42],
        "Potassium": [43],
        "Temperature": [20.87974371],
        "Humidity": [82.00274423],
        "pH_Value": [6.502985292],
        "Rainfall": [202.9355362],
        "Crop": ["-"]
    }
    '''
    prediction = run_saved_model(
        modelPath=model_path,
        input_data=input_data,
        data_path=data_path
    )
    assert prediction is not None
    assert isinstance(prediction, dict)
    assert isinstance(prediction['predictions'],list)
