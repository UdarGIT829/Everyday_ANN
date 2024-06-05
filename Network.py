import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

import optuna
from functools import partial
from easydict import EasyDict
import pickle

from DataImporter import DataImporter

class FlexibleANN(nn.Module):
    """Description:
    Flexible Artificial Neural Network (ANN) for Wide Usability

    This class implements a flexible artificial neural network using PyTorch. It supports various activation functions,
    hidden layer sizes, and batch normalization. The network can be trained, saved, loaded, and used for predictions
    with ease.

    Attributes:
        network (nn.Sequential): The neural network model.
        column_headers (list): List of column headers for the input features.

    Methods:
        forward(x): Performs a forward pass through the network.
        run(prediction_input_data, data_path): Runs the model to make predictions on the given input data.
    
    Usage Instructions:
    
    1. **Training the Model:**
        - Use the `train_nn` function to train the model with your data and desired hyperparameters.
        - Example:
            ```python
            trial_details = EasyDict({
                "hidden_layer_sizes": {"low": 4, "high": 128},
                "num_epochs": {"low": 50, "high": 1000},
                "learning_rate": {"low": 1e-5, "high": 1e-1},
                "batch_size": {"low": 16, "high": 128},
                "activation": ['ReLU', 'Tanh', 'LeakyReLU', 'ELU', 'GELU', 'Swish']
            })
            best_model = train_nn('path/to/data.csv', trial_details, n_trials=100)
            save_nn(best_model, 'best_model.pkl')
            ```

    2. **Loading the Model:**
        - Load a previously saved model using the `load_nn` function.
        - Example:
            ```python
            model = load_nn('best_model.pkl')
            ```

    3. **Making Predictions:**
        - Use the `run` method of the loaded model to make predictions on new data.
        - Example:
            ```python
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
            data_path = 'path/to/data.csv'
            prediction = model.run(input_data, data_path)
            print(f"Predicted Value: {prediction}")
            ```
    
    4. **Running from CSV:**
        - The `run` method can also accept a path to a CSV file as input.
        - Example:
            ```python
            prediction = model.run('path/to/input_data.csv', 'path/to/data.csv')
            print(f"Predicted Value: {prediction}")
            ```
    """
    def __init__(self, input_size, hidden_layer_sizes, output_size, activation_function):
        super(FlexibleANN, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Add batch normalization
            layers.append(activation_function)
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        layers.append(nn.Softmax(dim=1))
        self.network = nn.Sequential(*layers)

        self.column_headers = []
    
    def forward(self, x):
        return self.network(x)
    
    def run(self, prediction_input_data, data_path):
        """Description:
        Run the model to make predictions.

        Args:
            prediction_input_data (str, dict, or JSON string): The input data for making predictions. This can be a path to a CSV file, a dictionary, or a JSON string.
            data_path (str): The path to the data file used for training the model.

        Returns:
            list: The predicted labels for the input data.
            dict: If the input data is a CSV file, a dictionary containing the path to the output file with predictions is returned.
        """
        # Prepare the scaler and data importer for the input validation
        data_importer = DataImporter(data_path)

        input_features, _ = data_importer.get_encoded_data()
        scaler = StandardScaler()
        scaler.fit(input_features.values)  # Fit scaler on the full dataset

        # Create a DataFrame from the prediction_input
        prediction_input = DataImporter(prediction_input_data)
        input_data, _ = prediction_input.get_encoded_data()

        # Ensure all required features are present
        missing_features = [feature for feature in input_features.columns if feature not in input_data.columns]
        if missing_features:
            raise ValueError(f'Missing features: {missing_features}')

        # Standardize the input data
        input_data_scaled = scaler.transform(input_data)

        # Convert to PyTorch tensor
        input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

        # Run the model
        self.eval()
        with torch.no_grad():
            model_output = self(input_tensor)
            _, predicted = torch.max(model_output.data, 1)

        output = []
        # Decode the predicted output
        if len(predicted) == 1:
            predicted = [predicted]

        for iterPrediction in predicted:
            predicted_label = iterPrediction.item()
            reverse_mapping = {v: k for k, v in data_importer.feature_encoders[data_importer.data.columns[-1]].mapping.items()}
            predicted_label_decoded = reverse_mapping[predicted_label]
            output.append(predicted_label_decoded)

        return output

def get_activation_function(name):
    activations = {
        'ReLU': nn.ReLU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'LeakyReLU': nn.LeakyReLU(),
        'ELU': nn.ELU(),
        'GELU': nn.GELU(),
        'Swish': nn.SiLU()  # Swish is implemented as SiLU in PyTorch
    }
    return activations[name]

def train_nn(data_path, trial_details, n_trials=100):

    def objective(trial, data_path, trial_details):
        # Hyperparameters to tune
        hidden_layer_sizes = [trial.suggest_int(f'n_units_l{i}', trial_details.hidden_layer_sizes.low, trial_details.hidden_layer_sizes.high) 
                              for i in range(trial.suggest_int('n_layers', 1, 3))]
        num_epochs = trial.suggest_int('num_epochs', trial_details.num_epochs.low, trial_details.num_epochs.high)
        learning_rate = trial.suggest_float('learning_rate', trial_details.learning_rate.low, trial_details.learning_rate.high, log=True)
        batch_size = trial.suggest_int('batch_size', trial_details.batch_size.low, trial_details.batch_size.high)
        activation_name = trial.suggest_categorical('activation', trial_details.activation)
        activation_function = get_activation_function(activation_name)

        # Load and prepare the data
        data_importer = DataImporter(data_path)
        input_features, output_feature = data_importer.get_encoded_data()

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            input_features.values, 
            output_feature.values, 
            test_size=0.2, 
            random_state=42
        )

        # Standardize the input features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Create DataLoader for mini-batching
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize the neural network, loss function, and optimizer
        input_size = X_train.shape[1]
        output_size = len(data_importer.feature_encoders[data_importer.data.columns[-1]].mapping)
        model = FlexibleANN(input_size, hidden_layer_sizes, output_size, activation_function)
        model.column_headers = data_importer.column_names
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training the model with backpropagation
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()  # Clear the gradients
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the weights
                epoch_loss += loss.item()

        # Testing the model
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        
        # Store the model in the user attributes of the trial
        trial.set_user_attr('model', model)

        return accuracy

    # Create and run the Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(partial(objective, data_path=data_path, trial_details=trial_details), n_trials=n_trials)

    # Get the best model
    trial = study.best_trial
    best_model = trial.user_attrs['model']
    return best_model


def save_nn(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_nn(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def run_model(model, prediction_input_data, data_path):

    output = model.run(prediction_input_data, data_path)

    output_file = None
    # If input_data_source is a CSV file, write the predictions back to a new CSV file
    if isinstance(prediction_input_data, str) and prediction_input_data.endswith('.csv'):
        input_df = pd.read_csv(prediction_input_data)
        output_column = input_df.columns[-1]
        input_df[output_column] = output
        output_file = prediction_input_data.replace('.csv', '_predicted.csv')
        input_df.to_csv(output_file, index=False)
        print(f"Predictions written to {output_file}")

    result = {"predictions": output}
    
    if output_file != None:
        result["file"] = output_file

    return result

DEFAULT_trial_details = EasyDict({
    "hidden_layer_sizes": {"low": 4, "high": 128},
    "num_epochs": {"low": 50, "high": 1000},
    "learning_rate": {"low": 1e-5, "high": 1e-1},
    "batch_size": {"low": 16, "high": 128},
    "activation": ['ReLU', 'Tanh', 'LeakyReLU', 'ELU', 'GELU', 'Swish']
})

def merge_trial_details(defaults, overrides):
    merged = EasyDict(defaults.copy())
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged

def train_model(modelPath, data_path, trial_details: EasyDict = EasyDict(), n_trials=5):
    # Merge trial details
    trial_details = merge_trial_details(DEFAULT_trial_details, trial_details)

    # Train the model
    best_model = train_nn(data_path, trial_details, n_trials)

    # Save the model
    save_nn(best_model, modelPath)

def run_saved_model(modelPath, input_data, data_path: str):
    # Load the model
    model = load_nn(modelPath)

    # Run the model
    predicted_label = run_model(model, input_data, data_path)
    return predicted_label
