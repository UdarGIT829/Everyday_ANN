from Network import train_model, run_saved_model
from DataImporter import DataImporter
from sklearn.preprocessing import StandardScaler
from easydict import EasyDict


if __name__ == "__main__":
    modelPath = 'best_model.pkl'

    trial_details = EasyDict(
        {"num_epochs": {"high": 400}}
    )

    # https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
    data_path = 'data\Crop_Recommendation.csv'

    train_model(
        modelPath=modelPath,
        data_path=data_path,
        trial_details=trial_details,
        n_trials=5
    )

    # print()
    
    # input_data = "data\Testing_Sample.csv"

    # prediction = run_saved_model(
    #     modelPath=modelPath,
    #     input_data=input_data,
    #     data_path=data_path
    # )

    # print(f"CSV Predicted Value: {prediction}.")
    # print()

    # input_data = {
    #     'Nitrogen': [90],
    #     'Phosphorus': [42],
    #     'Potassium': [43],
    #     'Temperature': [20.87974371],
    #     'Humidity': [82.00274423],
    #     'pH_Value': [6.502985292],
    #     'Rainfall': [202.9355362],
    #     'Crop':[]
    # }
    # input_data['Crop'] = "-"*len(input_data[list(input_data.keys())[0]])

    # prediction = run_saved_model(
    #     modelPath=modelPath,
    #     input_data=input_data,
    #     data_path=data_path
    # )

    # print(f"Dict Predicted Value: {prediction}.")
    # print()

    # input_data = '''
    # {
    #     "Nitrogen": [90],
    #     "Phosphorus": [42],
    #     "Potassium": [43],
    #     "Temperature": [20.87974371],
    #     "Humidity": [82.00274423],
    #     "pH_Value": [6.502985292],
    #     "Rainfall": [202.9355362],
    #     "Crop": ["-"]
    # }
    # '''

    # prediction = run_saved_model(
    #     modelPath=modelPath,
    #     input_data=input_data,
    #     data_path=data_path
    # )

    # print(f"JSON string Predicted Value: {prediction}.")
