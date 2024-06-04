from Network import train_nn, save_nn, load_nn, run_model
from DataImporter import DataImporter
from sklearn.preprocessing import StandardScaler
from easydict import EasyDict

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

if __name__ == "__main__":
    modelPath = 'best_model.pkl'

    trial_details = EasyDict(
        {"num_epochs": {"high": 400}}
    )

    # https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
    data_path = 'Crop_Recommendation.csv'

    train_model(
        modelPath=modelPath,
        data_path=data_path,
        trial_details=trial_details,
        n_trials=1
    )

    print()
    
    input_data = "Testing_Sample.csv"

    prediction = run_saved_model(
        modelPath=modelPath,
        input_data=input_data,
        data_path=data_path
    )

    print(f"CSV Predicted Value: {prediction}.")
    print()

    input_data = {
        'Nitrogen': [90],
        'Phosphorus': [42],
        'Potassium': [43],
        'Temperature': [20.87974371],
        'Humidity': [82.00274423],
        'pH_Value': [6.502985292],
        'Rainfall': [202.9355362],
        'Crop':[]
    }
    input_data['Crop'] = "-"*len(input_data[list(input_data.keys())[0]])

    prediction = run_saved_model(
        modelPath=modelPath,
        input_data=input_data,
        data_path=data_path
    )

    print(f"Dict Predicted Value: {prediction}.")
    print()

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
        modelPath=modelPath,
        input_data=input_data,
        data_path=data_path
    )

    print(f"JSON string Predicted Value: {prediction}.")
